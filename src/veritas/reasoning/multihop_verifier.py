"""
Multi-Hop Claim Verifier for VERITAS-X 2.0.
Performs targeted subclaim retrieval, deduplication of cross-claim evidence mass,
independent subclaim NLI adjudication, and transparent compound parent-claim synthesis.
"""
from typing import List, Dict, Any, Optional, Tuple
from veritas.data.schema import CanonicalVeracity
from veritas.reasoning.types import SubClaim, SubClaimStatus
from veritas.verification.engine import VerificationEngine, ForensicVerdict
from veritas.verification.nli_model import CrossEncoderNLI
from veritas.verification.evidence_aggregator import EvidenceAggregator
from veritas.verification.calibrator import TemperatureCalibrator, SelectiveAbstainer
from veritas.retrieval.hybrid_retriever import HybridRetriever


class MultiHopVerifier:
    """
    Coordinates multi-hop verification across decomposed propositions.
    Maintains clean provenance and prevents artificial evidence mass inflation.
    """

    def __init__(
        self,
        engine: VerificationEngine,
        retriever: Optional[HybridRetriever] = None,
        calibrator: Optional[TemperatureCalibrator] = None,
        abstainer: Optional[SelectiveAbstainer] = None,
    ):
        self.engine = engine
        self.retriever = retriever or getattr(engine, "retriever", None)
        self.calibrator = calibrator or getattr(engine, "calibrator", None)
        self.abstainer = abstainer or getattr(engine, "abstainer", None)

    def verify_compound_claim(
        self,
        claim_text: str,
        subclaims: List[SubClaim],
        top_k_per_subclaim: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes multi-hop verification for each subclaim and synthesizes parent verdict.
        """
        if not subclaims:
            return {
                "verdict": CanonicalVeracity.INSUFFICIENT_EVIDENCE.value,
                "confidence": 0.0,
                "subclaims": [],
                "evidence": [],
                "explanation": "No valid propositions to evaluate.",
                "coverage_ratio": 0.0,
            }

        # 1. Targeted retrieval per subclaim with cross-subclaim deduplication
        all_evidence_map: Dict[str, Dict[str, Any]] = {}
        subclaim_to_ev_ids: Dict[str, List[str]] = {}
        ev_counter = 1

        for sc in subclaims:
            sc_ev_ids: List[str] = []
            if self.retriever:
                retrieved = self.retriever.search(sc.text, top_k=top_k_per_subclaim)
                for doc in retrieved:
                    doc_text = doc.get("text", "").strip()
                    doc_id = doc.get("doc_id") or f"doc_{hash(doc_text) & 0xffffffff}"

                    # Find existing or assign new E-XX identifier
                    existing_ev = next((e for e in all_evidence_map.values() if e.get("doc_id") == doc_id or e.get("text") == doc_text), None)
                    if existing_ev:
                        ev_id = existing_ev["id"]
                    else:
                        ev_id = f"E-{ev_counter:02d}"
                        ev_counter += 1
                        all_evidence_map[ev_id] = {
                            "id": ev_id,
                            "doc_id": doc_id,
                            "text": doc_text,
                            "source": doc.get("source_url") or doc.get("source_dataset") or "Local Evidence Corpus",
                            "score": float(doc.get("score", 0.0)),
                            "metadata": doc.get("metadata", {}),
                        }
                    sc_ev_ids.append(ev_id)

            subclaim_to_ev_ids[sc.subclaim_id] = sc_ev_ids
            sc.evidence_ids = sc_ev_ids

        # 2. Evaluate NLI for each subclaim against its retrieved passages
        evidence_evaluations: Dict[str, Dict[str, float]] = {}

        for sc in subclaims:
            ev_list = [all_evidence_map[eid] for eid in sc.evidence_ids if eid in all_evidence_map]
            premises = [e["text"] for e in ev_list]

            if not premises:
                sc.status = SubClaimStatus.UNRESOLVED
                sc.verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE.value
                sc.confidence = 0.25
                sc.explanation = "No relevant evidence retrieved for this atomic proposition."
                continue

            # Run engine NLI on this subclaim
            sub_verdict = self.engine.verify_claim(sc.text, premises)

            # Store per-passage NLI probabilities
            for b in sub_verdict.evidence_breakdown:
                if b.get("is_joint", False):
                    continue
                doc_text = b.get("text", "")
                matched_e = next((e for e in ev_list if e["text"] == doc_text), None)
                if matched_e:
                    evidence_evaluations[matched_e["id"]] = {
                        "entailment_prob": float(b.get("entailment_prob", 0.0)),
                        "contradiction_prob": float(b.get("contradiction_prob", 0.0)),
                        "neutral_prob": float(b.get("neutral_prob", 0.0)),
                    }

            # Map subclaim status
            sc.verdict = sub_verdict.verdict.value if hasattr(sub_verdict.verdict, "value") else str(sub_verdict.verdict)
            sc.confidence = float(sub_verdict.confidence)
            sc.probabilities = {
                k: round(float(v), 4) for k, v in (sub_verdict.class_probabilities or {}).items()
            }

            # Tally stances
            sup_c = sum(1 for b in sub_verdict.evidence_breakdown if not b.get("is_joint") and b.get("entailment_prob", 0.0) > 0.45)
            ref_c = sum(1 for b in sub_verdict.evidence_breakdown if not b.get("is_joint") and b.get("contradiction_prob", 0.0) > 0.45)
            neut_c = len(premises) - (sup_c + ref_c)
            sc.stance_distribution = {"support": sup_c, "refute": ref_c, "neutral": neut_c}

            if sc.verdict == CanonicalVeracity.VERIFIED_TRUE.value and sup_c > 0:
                sc.status = SubClaimStatus.SUPPORTED
                sc.explanation = f"Supported by {sup_c} documentary passage(s)."
            elif sc.verdict == CanonicalVeracity.VERIFIED_FALSE.value and ref_c > 0:
                sc.status = SubClaimStatus.REFUTED
                sc.explanation = f"Refuted by {ref_c} contradictory passage(s)."
            elif sc.verdict == CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value:
                sc.status = SubClaimStatus.CONFLICTING
                sc.explanation = f"Contradictory evidence: {sup_c} supporting vs {ref_c} refuting."
            else:
                sc.status = SubClaimStatus.UNRESOLVED
                sc.explanation = "Insufficient informative evidence to establish or refute this subclaim."

        # 3. Enrich evidence items with NLI probabilities
        formatted_evidence: List[Dict[str, Any]] = []
        for eid, edata in all_evidence_map.items():
            nli_probs = evidence_evaluations.get(eid, {"entailment_prob": 0.0, "contradiction_prob": 0.0, "neutral_prob": 1.0})
            e_p = nli_probs["entailment_prob"]
            c_p = nli_probs["contradiction_prob"]
            n_p = nli_probs["neutral_prob"]

            if e_p >= c_p and e_p >= n_p:
                stance = "support"
                score = e_p
            elif c_p >= e_p and c_p >= n_p:
                stance = "refute"
                score = c_p
            else:
                stance = "neutral"
                score = n_p

            is_inf = not (n_p >= 0.70 or max(e_p, c_p) < 0.25)
            formatted_evidence.append({
                "id": eid,
                "doc_id": edata.get("doc_id"),
                "source": edata.get("source"),
                "text": edata.get("text"),
                "stance": stance,
                "score": round(score, 4),
                "entailment_prob": round(e_p, 4),
                "contradiction_prob": round(c_p, 4),
                "neutral_prob": round(n_p, 4),
                "is_informative": is_inf,
            })

        # 4. Compound Claim Adjudication Synthesis
        supported_scs = [sc for sc in subclaims if sc.status == SubClaimStatus.SUPPORTED]
        refuted_scs = [sc for sc in subclaims if sc.status == SubClaimStatus.REFUTED]
        conflicting_scs = [sc for sc in subclaims if sc.status == SubClaimStatus.CONFLICTING]
        unresolved_scs = [sc for sc in subclaims if sc.status == SubClaimStatus.UNRESOLVED]

        total_sc = len(subclaims)
        coverage_ratio = (len(supported_scs) + len(refuted_scs) + len(conflicting_scs)) / total_sc

        # Synthesize compound parent verdict
        if len(supported_scs) == total_sc:
            parent_verdict = CanonicalVeracity.VERIFIED_TRUE.value
            parent_confidence = sum(sc.confidence for sc in subclaims) / total_sc
            explanation = f"All {total_sc} atomic propositions are supported by retrieved documentary evidence."
        elif len(refuted_scs) > 0:
            parent_verdict = CanonicalVeracity.VERIFIED_FALSE.value
            parent_confidence = max(sc.confidence for sc in refuted_scs)
            ref_texts = [f"'{sc.text}'" for sc in refuted_scs]
            explanation = f"Compound claim is refuted: {len(refuted_scs)} atomic proposition(s) contradicted ({', '.join(ref_texts)})."
        elif len(conflicting_scs) > 0 or (len(supported_scs) > 0 and len(refuted_scs) > 0):
            parent_verdict = CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value
            parent_confidence = 0.65
            explanation = f"Compound claim contains conflicting findings: {len(supported_scs)} supported, {len(refuted_scs)} refuted, {len(conflicting_scs)} internally conflicting."
        elif len(supported_scs) > 0 and len(unresolved_scs) > 0:
            # Partially supported but incomplete
            parent_verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE.value
            parent_confidence = 0.40
            explanation = f"Partially supported ({len(supported_scs)}/{total_sc} propositions), but {len(unresolved_scs)} proposition(s) lack documentary evidence to substantiate the full compound claim."
        else:
            parent_verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE.value
            parent_confidence = 0.25
            explanation = f"Insufficient evidentiary grounding across all {total_sc} subclaims."

        return {
            "verdict": parent_verdict,
            "confidence": round(parent_confidence, 4),
            "subclaims": [sc.to_dict() for sc in subclaims],
            "evidence": formatted_evidence,
            "explanation": explanation,
            "coverage_ratio": round(coverage_ratio, 3),
            "supported_count": len(supported_scs),
            "refuted_count": len(refuted_scs),
            "unresolved_count": len(unresolved_scs),
        }
