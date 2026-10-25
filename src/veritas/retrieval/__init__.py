"""
Retrieval module for VERITAS-X: sparse, dense, and hybrid forensic evidence retrieval.
"""
from .tokenizer import MultilingualTokenizer
from .evidence_corpus import EvidenceCorpus, EvidenceDocument
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .hybrid_retriever import HybridRetriever, RetrievalResult
from .evaluation import RetrievalEvaluator, RetrievalMetrics
from .tavily_provider import (
    TavilySearchProvider,
    TavilyUsageTracker,
    load_dotenv,
    deduplicate_passages,
    MONTHLY_SEARCH_CREDIT_LIMIT,
)

__all__ = [
    "MultilingualTokenizer",
    "EvidenceCorpus",
    "EvidenceDocument",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "RetrievalEvaluator",
    "RetrievalMetrics",
    "TavilySearchProvider",
    "TavilyUsageTracker",
    "load_dotenv",
    "deduplicate_passages",
    "MONTHLY_SEARCH_CREDIT_LIMIT",
]
