"""
Retrieval Evaluation Engine for VERITAS-X.
Computes Hit@K, Recall@K, and Mean Reciprocal Rank (MRR) across languages and systems.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional
import numpy as np

from .evidence_corpus import EvidenceCorpus


@dataclass
class RetrievalMetrics:
    """Stores quantitative evaluation metrics for a retrieval model."""
    model_name: str
    evaluated_claims_count: int
    unverifiable_skipped_count: int
    hit_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    language_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Retrieval Evaluation: {self.model_name} ===",
            f"Evaluated Verifiable Claims : {self.evaluated_claims_count}",
            f"Skipped (No Ground Truth)    : {self.unverifiable_skipped_count}",
            f"Mean Reciprocal Rank (MRR)   : {self.mrr:.4f}",
            "--- Top-K Metrics ---"
        ]
        for k in sorted(self.hit_at_k.keys()):
            hit = self.hit_at_k[k]
            rec = self.recall_at_k.get(k, 0.0)
            lines.append(f"  K = {k:2d} | Hit@{k:2d}: {hit * 100:6.2f}% | Recall@{k:2d}: {rec * 100:6.2f}%")

        if self.language_breakdown:
            lines.append("--- Language Breakdown (Hit@5) ---")
            for lang, metrics in sorted(self.language_breakdown.items()):
                h5 = metrics.get("hit_5", 0.0)
                mrr_l = metrics.get("mrr", 0.0)
                n = int(metrics.get("count", 0))
                lines.append(f"  {lang:8s} (N={n:3d}) | Hit@5: {h5 * 100:6.2f}% | MRR: {mrr_l:.4f}")

        return "\n".join(lines)


class RetrievalEvaluator:
    """
    Evaluates any retriever (BM25, Dense, or Hybrid) against ground-truth evidence.
    """

    def __init__(self, corpus: EvidenceCorpus):
        self.corpus = corpus

    def evaluate(
        self,
        claims: List[Any],
        retriever: Any,
        k_values: Optional[List[int]] = None,
        max_k: int = 10,
    ) -> RetrievalMetrics:
        """
        Runs retrieval evaluation over candidate claims with ground-truth evidence in the corpus.
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        model_name = getattr(retriever, "__class__", type(retriever)).__name__
        hits_acc = {k: 0 for k in k_values}
        recall_acc = {k: 0.0 for k in k_values}
        rr_acc: List[float] = []

        lang_records: Dict[str, Dict[str, Any]] = {}
        eval_count = 0
        skipped_count = 0

        for claim in claims:
            if isinstance(claim, dict):
                cid = claim.get("claim_id")
                ctext = claim.get("claim_text")
                clang = claim.get("language", "en")
            else:
                cid = getattr(claim, "claim_id", None)
                ctext = getattr(claim, "claim_text", None)
                clang = getattr(claim, "language", "en")

            # Obtain ground-truth doc IDs present in our local evidence corpus
            gt_doc_ids: Set[str] = set(self.corpus.get_ground_truth_doc_ids(cid))

            # If the claim has no evidence text indexed in our local corpus (e.g., raw FEVER pointers or NEI),
            # it cannot be evaluated for text retrieval against this corpus.
            if not gt_doc_ids:
                skipped_count += 1
                continue

            eval_count += 1
            if clang not in lang_records:
                lang_records[clang] = {"count": 0, "hit_5": 0, "rr": []}
            lang_records[clang]["count"] += 1

            # Retrieve top candidates
            results = retriever.search(ctext, top_k=max_k)
            retrieved_ids = [
                (r.doc_id if hasattr(r, "doc_id") else r["doc_id"])
                for r in results
            ]

            # 1. MRR Calculation
            first_rank = None
            for idx, r_id in enumerate(retrieved_ids, start=1):
                if r_id in gt_doc_ids:
                    first_rank = idx
                    break

            reciprocal_rank = (1.0 / first_rank) if first_rank is not None else 0.0
            rr_acc.append(reciprocal_rank)
            lang_records[clang]["rr"].append(reciprocal_rank)

            # 2. Hit@K and Recall@K Calculation
            for k in k_values:
                top_k_ids = set(retrieved_ids[:k])
                intersection = top_k_ids.intersection(gt_doc_ids)

                if len(intersection) > 0:
                    hits_acc[k] += 1
                    if k == 5:
                        lang_records[clang]["hit_5"] += 1

                recall_val = len(intersection) / len(gt_doc_ids)
                recall_acc[k] += recall_val

        # Aggregate averages
        if eval_count == 0:
            return RetrievalMetrics(
                model_name=model_name,
                evaluated_claims_count=0,
                unverifiable_skipped_count=skipped_count,
            )

        final_hit = {k: hits_acc[k] / eval_count for k in k_values}
        final_recall = {k: recall_acc[k] / eval_count for k in k_values}
        final_mrr = float(np.mean(rr_acc)) if rr_acc else 0.0

        final_lang_breakdown: Dict[str, Dict[str, float]] = {}
        for lang, dat in lang_records.items():
            l_count = dat["count"]
            if l_count > 0:
                final_lang_breakdown[lang] = {
                    "count": float(l_count),
                    "hit_5": dat["hit_5"] / l_count,
                    "mrr": float(np.mean(dat["rr"])) if dat["rr"] else 0.0,
                }

        return RetrievalMetrics(
            model_name=model_name,
            evaluated_claims_count=eval_count,
            unverifiable_skipped_count=skipped_count,
            hit_at_k=final_hit,
            recall_at_k=final_recall,
            mrr=final_mrr,
            language_breakdown=final_lang_breakdown,
        )
