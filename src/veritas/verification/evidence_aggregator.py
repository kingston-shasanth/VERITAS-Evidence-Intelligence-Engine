"""
Evidence Aggregator for VERITAS-X.
Combines multiple evidence passages, discounts duplicate or near-duplicate snippets,
filters uninformative passages, and resolves stance consensus with explicit cross-passage
stance-disparity detection to separate genuine evidentiary conflict from false claims.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

from veritas.data.schema import CanonicalVeracity


@dataclass
class AggregatedEvidenceVerdict:
    """Consensus synthesis output from multiple evidence snippets."""
    canonical_verdict: CanonicalVeracity
    confidence: float
    entailment_mass: float
    contradiction_mass: float
    conflict_ratio: float
    total_informative_mass: float
    raw_logits: List[float]  # [True, False, Conflicting, Insufficient]
    unique_passages_used: int
    duplicates_filtered: int
    irrelevant_passages_filtered: int
    evidence_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    has_cross_passage_disparity: bool = False
    disparity_details: Dict[str, Any] = field(default_factory=dict)


class EvidenceAggregator:
    """
    Synthesizes multi-premise NLI predictions into a single, calibrated veracity verdict.
    Features:
    - Jaccard lexical deduplication for near-duplicate suppression.
    - Explicit cross-passage stance-disparity detection requiring distinct supporting
      and refuting passages.
    - Joint-premise suppression when stance disparity is detected.
    - Full hyperparameter configurability for validation-only tuning.
    """

    def __init__(
        self,
        jaccard_dedup_threshold: float = 0.75,
        neutral_threshold: float = 0.70,
        min_informative_mass: float = 0.20,
        conflict_threshold: float = 0.40,
        disparity_entail_threshold: float = 0.40,
        disparity_contra_threshold: float = 0.40,
        joint_weight_default: float = 1.0,
        joint_weight_disparity: float = 0.2,
    ):
        """
        Threshold Rationale & Configurable Bounds:
        - jaccard_dedup_threshold (0.75): Filters republished or syndicated copies.
        - neutral_threshold (0.70): Filters passages where NLI neutral stance dominates.
        - min_informative_mass (0.20): Minimum mass required to adjudicate beyond INSUFFICIENT.
        - conflict_threshold (0.40): Balance ratio 2*min(E,C)/(E+C) needed to declare conflict.
        - disparity_entail_threshold (0.40): Minimum entailment for an individual passage to qualify
          as an active supporter (above 3-class uniform random baseline 0.33).
        - disparity_contra_threshold (0.40): Minimum contradiction for an individual passage to qualify
          as an active refuter (above 3-class uniform random baseline 0.33).
        - joint_weight_default (1.0): Full weight of joint premise when passages agree.
        - joint_weight_disparity (0.2): De-weighted joint multiplier when distinct passages conflict,
          preventing joint premise contradiction bias from smothering the supporting passage.
        """
        self.jaccard_dedup_threshold = float(jaccard_dedup_threshold)
        self.neutral_threshold = float(neutral_threshold)
        self.min_informative_mass = float(min_informative_mass)
        self.conflict_threshold = float(conflict_threshold)
        self.disparity_entail_threshold = float(disparity_entail_threshold)
        self.disparity_contra_threshold = float(disparity_contra_threshold)
        self.joint_weight_default = float(joint_weight_default)
        self.joint_weight_disparity = float(joint_weight_disparity)

    @staticmethod
    def _compute_jaccard(text_a: str, text_b: str) -> float:
        set_a = set(text_a.lower().split())
        set_b = set(text_b.lower().split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

    def detect_cross_passage_disparity(
        self,
        candidate_passages: List[Dict[str, Any]],
        nli_predictions: List[Dict[str, Any]],
        filtered_indices: List[int],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detects whether at least two distinct, non-duplicate passages take opposing stances.
        Ensures the same passage CANNOT satisfy both sides.
        """
        supporting_indices: List[int] = []
        refuting_indices: List[int] = []

        for idx in filtered_indices:
            cand = candidate_passages[idx]
            if cand.get("is_joint", False):
                continue
            if not cand.get("is_relevant", True):
                continue
            if not cand.get("is_informative", True):
                continue
            if not cand.get("is_claim_aligned", True):
                continue
            if cand.get("is_figurative", False):
                continue
            if not cand.get("is_decision_eligible", True):
                continue
            pred = nli_predictions[idx]
            e_prob = float(pred.get("entailment", 0.0))
            c_prob = float(pred.get("contradiction", 0.0))
            n_prob = float(pred.get("neutral", 0.0))

            # Active supporter: strong entailment, dominant over contradiction AND neutral
            if e_prob >= self.disparity_entail_threshold and e_prob > c_prob and e_prob > n_prob and n_prob < self.neutral_threshold:
                supporting_indices.append(idx)

            # Active refuter: strong contradiction, dominant over entailment AND neutral.
            # Require genuine claim-relation alignment before a passage can create a contradiction or conflict.
            if c_prob >= self.disparity_contra_threshold and c_prob > e_prob and c_prob > n_prob and n_prob < self.neutral_threshold:
                refuting_indices.append(idx)

        # Cross-passage disparity requires distinct indices: s_idx != r_idx
        for s_idx in supporting_indices:
            for r_idx in refuting_indices:
                if s_idx != r_idx:
                    s_cov = candidate_passages[s_idx].get("claim_coverage", "full")
                    r_cov = candidate_passages[r_idx].get("claim_coverage", "full")
                    # A partial-coverage supporter cannot trigger cross-passage disparity against a full-coverage refuter
                    if s_cov == "partial" and r_cov == "full":
                        continue
                    return True, {
                        "has_disparity": True,
                        "supporter_idx": s_idx,
                        "refuter_idx": r_idx,
                        "supporter_doc_id": candidate_passages[s_idx].get("doc_id"),
                        "refuter_doc_id": candidate_passages[r_idx].get("doc_id"),
                        "supporter_entailment": float(nli_predictions[s_idx].get("entailment", 0.0)),
                        "refuter_contradiction": float(nli_predictions[r_idx].get("contradiction", 0.0)),
                        "all_supporting_indices": supporting_indices,
                        "all_refuting_indices": refuting_indices,
                    }

        return False, {
            "has_disparity": False,
            "all_supporting_indices": supporting_indices,
            "all_refuting_indices": refuting_indices,
        }

    def aggregate(
        self,
        candidate_passages: List[Dict[str, Any]],
        nli_predictions: List[Dict[str, Any]],
        joint_nli_prediction: Optional[Dict[str, Any]] = None,
    ) -> AggregatedEvidenceVerdict:
        """
        Synthesizes NLI predictions over retrieved passages into a unified verdict.
        """
        if not candidate_passages or not nli_predictions:
            # Zero evidence available -> Insufficient Evidence
            return AggregatedEvidenceVerdict(
                canonical_verdict=CanonicalVeracity.INSUFFICIENT_EVIDENCE,
                confidence=0.95,
                entailment_mass=0.0,
                contradiction_mass=0.0,
                conflict_ratio=0.0,
                total_informative_mass=0.0,
                raw_logits=[-3.0, -3.0, -3.0, 3.0],
                unique_passages_used=0,
                duplicates_filtered=0,
                irrelevant_passages_filtered=0,
                evidence_breakdown=[],
                has_cross_passage_disparity=False,
                disparity_details={"reason": "empty_evidence"},
            )

        # 1. Deduplicate passages (near-duplicate text discounting)
        filtered_indices: List[int] = []
        duplicate_indices: List[int] = []
        duplicates_count = 0

        for i, cand in enumerate(candidate_passages):
            # Do not deduplicate joint premise
            if cand.get("is_joint", False):
                filtered_indices.append(i)
                continue

            t_curr = cand.get("text", "")
            is_dup = False
            for prev_idx in filtered_indices:
                if candidate_passages[prev_idx].get("is_joint", False):
                    continue
                t_prev = candidate_passages[prev_idx].get("text", "")
                if self._compute_jaccard(t_curr, t_prev) >= self.jaccard_dedup_threshold:
                    is_dup = True
                    duplicates_count += 1
                    duplicate_indices.append(i)
                    break
            if not is_dup:
                filtered_indices.append(i)

        # 2. Cross-Passage Stance Disparity Detection across distinct individual passages
        has_disparity, disparity_details = self.detect_cross_passage_disparity(
            candidate_passages=candidate_passages,
            nli_predictions=nli_predictions,
            filtered_indices=filtered_indices,
        )

        # 3. Accumulate weighted stance mass across informative passages
        entailment_mass = 0.0
        contradiction_mass = 0.0
        irrelevant_count = 0
        evidence_breakdown: List[Dict[str, Any]] = []

        has_subject_coverage = any(
            candidate_passages[i].get("is_relevant", False)
            for i in filtered_indices
            if not candidate_passages[i].get("is_joint", False)
        )
        has_predicate_coverage = any(
            candidate_passages[i].get("is_claim_aligned", False)
            for i in filtered_indices
            if not candidate_passages[i].get("is_joint", False)
        )
        # Pre-compute whether joint-premise reasoning is safe and justified:
        # Require either at least two distinct individually decision-eligible passages,
        # or genuine multi-hop complementary coverage where distinct passages cover different components.
        eligible_non_joint_indices = [
            i for i in filtered_indices
            if not candidate_passages[i].get("is_joint", False)
            and candidate_passages[i].get("is_relevant", True)
            and candidate_passages[i].get("is_informative", True)
            and candidate_passages[i].get("is_claim_aligned", True)
            and not candidate_passages[i].get("is_figurative", False)
            and candidate_passages[i].get("is_decision_eligible", True)
        ]

        # Check if joint premise should be considered:
        # A joint premise is eligible if either:
        # (1) There are >= 2 individually decision-eligible passages, OR
        # (2) There is genuine complementary multi-hop coverage: at least one informative passage covers
        #     the subject/entity AND a distinct informative passage covers the predicate/action/relation.
        # Off-topic noise passages (which cover neither subject nor predicate) must NEVER activate joint aggregation.
        informative_non_joint = [
            i for i in filtered_indices
            if not candidate_passages[i].get("is_joint", False)
            and candidate_passages[i].get("is_informative", True)
            and not candidate_passages[i].get("is_figurative", False)
        ]
        subj_coverage_indices = {
            i for i in informative_non_joint
            if candidate_passages[i].get("is_relevant", False)
            or "Subject / Entity" in candidate_passages[i].get("claim_components_covered", [])
        }
        pred_coverage_indices = {
            i for i in informative_non_joint
            if candidate_passages[i].get("is_claim_aligned", False)
            or any(
                c in candidate_passages[i].get("claim_components_covered", [])
                for c in ("Action / Status", "Attribute / Property", "Temporal / Date", "Venue / Location")
            )
        }
        distinct_multihop = bool(
            subj_coverage_indices
            and pred_coverage_indices
            and len(subj_coverage_indices | pred_coverage_indices) >= 2
            and (subj_coverage_indices - pred_coverage_indices or pred_coverage_indices - subj_coverage_indices)
        )
        can_use_joint = bool(joint_nli_prediction) and (
            len(eligible_non_joint_indices) >= 2 or distinct_multihop
        )

        rank = 1
        for idx in filtered_indices:
            cand = candidate_passages[idx]
            pred = nli_predictions[idx]

            c_prob = float(pred.get("contradiction", 0.0))
            e_prob = float(pred.get("entailment", 0.0))
            n_prob = float(pred.get("neutral", 0.0))

            is_joint = bool(cand.get("is_joint", False))
            is_rel = bool(cand.get("is_relevant", True))
            is_inf = bool(cand.get("is_informative", True))
            is_aligned = bool(cand.get("is_claim_aligned", True))
            is_fig = bool(cand.get("is_figurative", False))
            is_eligible = bool(is_rel and is_inf and is_aligned and not is_fig)

            if is_joint:
                weight = 0.0 if not can_use_joint else (self.joint_weight_disparity if has_disparity else self.joint_weight_default)
            elif not is_eligible:
                irrelevant_count += 1
                weight = 0.0  # Zero weight for ineligible (off-topic, uninformative, or unaligned) passages
            elif n_prob >= self.neutral_threshold or max(e_prob, c_prob) < 0.25:
                irrelevant_count += 1
                weight = 0.0  # Zero weight for dominant neutral passages
            else:
                # Rank decay weight: Rank 1 has higher weight than lower ranks
                rank_decay = 1.0 / (1.0 + 0.2 * (rank - 1))
                # Confidence scaling: high neutral diminishes evidence weight
                weight = rank_decay * (1.0 - n_prob)
                rank += 1

            entailment_mass += weight * e_prob
            contradiction_mass += weight * c_prob

            if is_fig or not is_rel:
                item_stance = "neutral"
            elif is_eligible:
                if e_prob >= c_prob and (e_prob >= n_prob or e_prob >= self.disparity_entail_threshold):
                    item_stance = "support"
                elif c_prob > e_prob and (c_prob >= n_prob or c_prob >= self.disparity_contra_threshold):
                    item_stance = "refute"
                else:
                    item_stance = "neutral"
            else:
                item_stance = "neutral"

            evidence_breakdown.append({
                "doc_id": cand.get("doc_id"),
                "text": cand.get("text"),
                "weight": float(weight),
                "stance": item_stance,
                "entailment_prob": float(e_prob),
                "contradiction_prob": float(c_prob),
                "neutral_prob": float(n_prob),
                "source_url": cand.get("source_url"),
                "source_type": cand.get("source_type", "local"),
                "source_title": cand.get("source_title"),
                "published_date": cand.get("published_date"),
                "is_joint": is_joint,
                "is_relevant": is_rel,
                "is_informative": is_inf,
                "is_claim_aligned": is_aligned,
                "is_decision_eligible": is_eligible,
                "is_duplicate": False,
                "is_figurative": bool(cand.get("is_figurative", False)),
                "claim_coverage": cand.get("claim_coverage", "full"),
                "claim_components_covered": cand.get("claim_components_covered", []),
                "is_active_supporter": idx in disparity_details.get("all_supporting_indices", []),
                "is_active_refuter": idx in disparity_details.get("all_refuting_indices", []),
            })

        # Append discounted duplicate passages with zero weight
        for dup_idx in duplicate_indices:
            cand = candidate_passages[dup_idx]
            pred = nli_predictions[dup_idx] if dup_idx < len(nli_predictions) else {}
            evidence_breakdown.append({
                "doc_id": cand.get("doc_id"),
                "text": cand.get("text"),
                "weight": 0.0,
                "stance": "neutral",
                "entailment_prob": float(pred.get("entailment", 0.0)),
                "contradiction_prob": float(pred.get("contradiction", 0.0)),
                "neutral_prob": float(pred.get("neutral", 1.0)),
                "source_url": cand.get("source_url"),
                "source_type": cand.get("source_type", "local"),
                "source_title": cand.get("source_title"),
                "published_date": cand.get("published_date"),
                "is_joint": False,
                "is_relevant": bool(cand.get("is_relevant", True)),
                "is_informative": bool(cand.get("is_informative", True)),
                "is_claim_aligned": bool(cand.get("is_claim_aligned", True)),
                "is_decision_eligible": False,
                "is_duplicate": True,
                "is_figurative": bool(cand.get("is_figurative", False)),
                "claim_coverage": cand.get("claim_coverage", "full"),
                "claim_components_covered": cand.get("claim_components_covered", []),
                "is_active_supporter": False,
                "is_active_refuter": False,
            })

        # Incorporate joint_nli_prediction only if safe joint eligibility is confirmed
        # and not already processed as an is_joint candidate item
        has_joint_in_items = any(candidate_passages[i].get("is_joint", False) for i in filtered_indices)
        if joint_nli_prediction is not None and can_use_joint and not has_joint_in_items:
            j_weight = self.joint_weight_disparity if has_disparity else self.joint_weight_default
            j_e = float(joint_nli_prediction.get("entailment", 0.0))
            j_c = float(joint_nli_prediction.get("contradiction", 0.0))
            entailment_mass += j_weight * j_e
            contradiction_mass += j_weight * j_c

        total_mass = entailment_mass + contradiction_mass
        conflict_ratio = (
            (2.0 * min(entailment_mass, contradiction_mass)) / (total_mass + 1e-6)
            if total_mass > 0 else 0.0
        )

        # 4. Decision Logic & 4-Class Logits Construction
        # Class mapping: [VERIFIED_TRUE, VERIFIED_FALSE, CONFLICTING, INSUFFICIENT]
        z_true = (entailment_mass - contradiction_mass) * 3.0
        z_false = (contradiction_mass - entailment_mass) * 3.0
        if has_disparity:
            z_conf = (conflict_ratio * total_mass - 0.25) * 4.0
        else:
            # Strongly penalize conflicting class when no genuine cross-passage disparity exists
            z_conf = -5.0
        z_insuf = (self.min_informative_mass - total_mass) * 5.0

        raw_logits = [float(z_true), float(z_false), float(z_conf), float(z_insuf)]

        # Softmax for normalized decision probabilities
        exp_z = np.exp(raw_logits - np.max(raw_logits))
        probs = exp_z / np.sum(exp_z)

        # Categorical assignment
        if total_mass < self.min_informative_mass:
            verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE
            confidence = float(probs[3])
        elif (
            has_disparity
            and entailment_mass >= 0.15
            and contradiction_mass >= 0.15
            and conflict_ratio >= self.conflict_threshold
        ):
            verdict = CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED
            confidence = float(probs[2])
        elif entailment_mass > contradiction_mass:
            verdict = CanonicalVeracity.VERIFIED_TRUE
            confidence = float(probs[0])
        else:
            verdict = CanonicalVeracity.VERIFIED_FALSE
            confidence = float(probs[1])

        return AggregatedEvidenceVerdict(
            canonical_verdict=verdict,
            confidence=confidence,
            entailment_mass=float(entailment_mass),
            contradiction_mass=float(contradiction_mass),
            conflict_ratio=float(conflict_ratio),
            total_informative_mass=float(total_mass),
            raw_logits=raw_logits,
            unique_passages_used=len([i for i in filtered_indices if not candidate_passages[i].get("is_joint")]),
            duplicates_filtered=duplicates_count,
            irrelevant_passages_filtered=irrelevant_count,
            evidence_breakdown=evidence_breakdown,
            has_cross_passage_disparity=has_disparity,
            disparity_details=disparity_details,
        )

    # Method alias for callers expecting aggregate_evidence
    aggregate_evidence = aggregate
