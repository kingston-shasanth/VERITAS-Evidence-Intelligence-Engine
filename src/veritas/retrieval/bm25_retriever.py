"""
BM25 Keyword Search Retriever for VERITAS-X.
Implements Okapi BM25 ranking over multilingual tokenized text for exact matching
of proper names, dates, numerical values, and domain-specific terminology.
"""
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from .tokenizer import MultilingualTokenizer
from .evidence_corpus import EvidenceCorpus, EvidenceDocument


class BM25Retriever:
    """
    Sparse keyword retriever using BM25Okapi with multilingual subword/entity tokenization.
    """

    def __init__(self, corpus: EvidenceCorpus):
        self.corpus = corpus
        self.doc_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._build_index()

    def _build_index(self):
        """Indexes all non-empty documents in the EvidenceCorpus."""
        self.doc_ids = []
        self.tokenized_corpus = []

        for doc in self.corpus.get_all_documents():
            tokens = MultilingualTokenizer.tokenize(doc.text)
            self.doc_ids.append(doc.doc_id)
            self.tokenized_corpus.append(tokens)

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the top-K matching documents for a query string.
        Returns a list of dicts:
            {'doc_id': str, 'score': float, 'rank': int, 'text': str, 'metadata': dict}
        """
        if not self.bm25 or not query or not query.strip():
            return []

        query_tokens = MultilingualTokenizer.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices with positive relevance
        # Sort indices descending by score
        indexed_scores = [(idx, float(score)) for idx, score in enumerate(scores) if score > 0.0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(indexed_scores[:top_k], start=1):
            doc_id = self.doc_ids[idx]
            doc = self.corpus.get_document(doc_id)
            if doc is None:
                continue

            results.append({
                "doc_id": doc_id,
                "score": score,
                "rank": rank,
                "text": doc.text,
                "source_dataset": doc.source_dataset,
                "source_url": doc.source_url,
                "claim_id": doc.claim_id,
                "metadata": doc.metadata,
            })

        return results
