

"""
Hybrid Evidence Retriever for VERITAS-X.
Fuses sparse BM25 and dense multilingual semantic embeddings using Reciprocal Rank Fusion (RRF).
Reference: Cormack, Clarke, & Büttcher (SIGIR 2009).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever


@dataclass
class RetrievalResult:
    """Individual evidence result with multi-system provenance and fusion metrics."""
    doc_id: str
    rrf_score: float
    rank: int
    text: str
    source_dataset: str
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    source_url: Optional[str] = None
    claim_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "rrf_score": self.rrf_score,
            "rank": self.rank,
            "text": self.text,
            "source_dataset": self.source_dataset,
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "source_url": self.source_url,
            "claim_id": self.claim_id,
            "metadata": self.metadata,
        }


class HybridRetriever:
    """
    Orchestrates hybrid sparse-dense evidence retrieval using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: Optional[DenseRetriever] = None,
        rrf_k: int = 60,
        weight_sparse: float = 1.0,
        weight_dense: float = 1.0,
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.weight_sparse = weight_sparse
        self.weight_dense = weight_dense

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
    ) -> List[RetrievalResult]:
        """
        Executes dual sparse and dense queries and merges candidates via RRF.
        Formula:
            Score(d) = (w_sparse / (k + rank_sparse(d))) + (w_dense / (k + rank_dense(d)))
        """
        if not query or not query.strip():
            return []

        # Retrieve candidate pools from available modalities
        sparse_candidates = self.bm25.search(query, top_k=candidate_k) if self.bm25 is not None else []
        dense_candidates = self.dense.search(query, top_k=candidate_k) if self.dense is not None else []

        doc_fusion: Dict[str, Dict[str, Any]] = {}

        # 1. Accumulate sparse BM25 scores
        for item in sparse_candidates:
            did = item["doc_id"]
            rank = item["rank"]
            score_contrib = self.weight_sparse / (self.rrf_k + rank)

            if did not in doc_fusion:
                doc_fusion[did] = {
                    "doc_id": did,
                    "rrf_score": 0.0,
                    "bm25_rank": rank,
                    "dense_rank": None,
                    "bm25_score": item["score"],
                    "dense_score": None,
                    "text": item["text"],
                    "source_dataset": item["source_dataset"],
                    "source_url": item["source_url"],
                    "claim_id": item["claim_id"],
                    "metadata": item["metadata"],
                }
            doc_fusion[did]["rrf_score"] += score_contrib

        # 2. Accumulate dense semantic scores
        for item in dense_candidates:
            did = item["doc_id"]
            rank = item["rank"]
            score_contrib = self.weight_dense / (self.rrf_k + rank)

            if did not in doc_fusion:
                doc_fusion[did] = {
                    "doc_id": did,
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "dense_rank": rank,
                    "bm25_score": None,
                    "dense_score": item["score"],
                    "text": item["text"],
                    "source_dataset": item["source_dataset"],
                    "source_url": item["source_url"],
                    "claim_id": item["claim_id"],
                    "metadata": item["metadata"],
                }
            else:
                doc_fusion[did]["dense_rank"] = rank
                doc_fusion[did]["dense_score"] = item["score"]

            doc_fusion[did]["rrf_score"] += score_contrib

        # Sort all fused candidates descending by RRF score with deterministic doc_id tie-breaking
        sorted_docs = sorted(doc_fusion.values(), key=lambda x: (-x["rrf_score"], str(x["doc_id"])))

        results: List[RetrievalResult] = []
        for final_rank, doc in enumerate(sorted_docs[:top_k], start=1):
            results.append(
                RetrievalResult(
                    doc_id=doc["doc_id"],
                    rrf_score=doc["rrf_score"],
                    rank=final_rank,
                    text=doc["text"],
                    source_dataset=doc["source_dataset"],
                    bm25_rank=doc["bm25_rank"],
                    dense_rank=doc["dense_rank"],
                    bm25_score=doc["bm25_score"],
                    dense_score=doc["dense_score"],
                    source_url=doc["source_url"],
                    claim_id=doc["claim_id"],
                    metadata=doc["metadata"],
                )
            )

        return results
