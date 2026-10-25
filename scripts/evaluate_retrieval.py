"""
Evaluation Benchmark Runner for VERITAS-X Retrieval Engine.
Evaluates BM25, Multilingual Dense E5 + FAISS, and Hybrid RRF across all verifiable claims.
"""
import os
import sys
import time
import json

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from veritas.data.schema import CanonicalClaim
from veritas.retrieval.evidence_corpus import EvidenceCorpus
from veritas.retrieval.bm25_retriever import BM25Retriever
from veritas.retrieval.dense_retriever import DenseRetriever
from veritas.retrieval.hybrid_retriever import HybridRetriever
from veritas.retrieval.evaluation import RetrievalEvaluator

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(WORKSPACE_ROOT, "data", "processed")


def main():
    canonical_path = os.path.join(PROCESSED_DIR, "canonical_claims.jsonl")
    corpus_path = os.path.join(PROCESSED_DIR, "evidence_corpus.jsonl")

    print("[*] Loading canonical claims from:", canonical_path)
    claims = []
    with open(canonical_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                claims.append(CanonicalClaim.from_dict(json.loads(line)))
    print(f"    Loaded {len(claims)} canonical claims.")

    print("\n[*] Building local reproducible EvidenceCorpus...")
    corpus = EvidenceCorpus.from_canonical_claims(claims)
    corpus_size = len(corpus)
    print(f"    Indexed {corpus_size} unique verified evidence passages.")
    corpus.save_to_jsonl(corpus_path)
    print(f"    Saved corpus to: {corpus_path}")

    # Initialize Evaluator
    evaluator = RetrievalEvaluator(corpus)

    # 1. BM25 Evaluation
    print("\n" + "="*60)
    print("[*] Initializing BM25 Sparse Retriever...")
    t0 = time.time()
    bm25 = BM25Retriever(corpus)
    bm25_index_time = time.time() - t0
    print(f"    BM25 indexing completed in {bm25_index_time:.3f}s")

    t0 = time.time()
    bm25_metrics = evaluator.evaluate(claims, bm25, k_values=[1, 3, 5, 10])
    bm25_eval_time = time.time() - t0
    print(bm25_metrics.summary())
    print(f"    Evaluation elapsed time: {bm25_eval_time:.2f}s ({bm25_eval_time / bm25_metrics.evaluated_claims_count * 1000:.1f}ms / query)")

    # 2. Dense E5 + FAISS Evaluation
    print("\n" + "="*60)
    print("[*] Initializing Multilingual Dense Retriever (multilingual-e5-small + FAISS)...")
    t0 = time.time()
    dense = DenseRetriever(corpus, device="cpu")
    dense_index_time = time.time() - t0
    print(f"    Dense indexing ({corpus_size} passages) completed in {dense_index_time:.3f}s")

    t0 = time.time()
    dense_metrics = evaluator.evaluate(claims, dense, k_values=[1, 3, 5, 10])
    dense_eval_time = time.time() - t0
    print(dense_metrics.summary())
    print(f"    Evaluation elapsed time: {dense_eval_time:.2f}s ({dense_eval_time / dense_metrics.evaluated_claims_count * 1000:.1f}ms / query)")

    # 3. Hybrid RRF Evaluation
    print("\n" + "="*60)
    print("[*] Initializing Hybrid Retriever (RRF k=60)...")
    hybrid = HybridRetriever(bm25, dense, rrf_k=60, weight_sparse=1.0, weight_dense=1.0)
    t0 = time.time()
    hybrid_metrics = evaluator.evaluate(claims, hybrid, k_values=[1, 3, 5, 10])
    hybrid_eval_time = time.time() - t0
    print(hybrid_metrics.summary())
    print(f"    Evaluation elapsed time: {hybrid_eval_time:.2f}s ({hybrid_eval_time / hybrid_metrics.evaluated_claims_count * 1000:.1f}ms / query)")

    print("\n" + "="*60)
    print("=== SUMMARY COMPARISON TABLE ===")
    print(f"{'Retriever':<20} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'Hit@10':<8} | {'MRR':<8}")
    print("-" * 72)
    for name, m in [("BM25 (Sparse)", bm25_metrics), ("Dense E5 (FAISS)", dense_metrics), ("Hybrid RRF", hybrid_metrics)]:
        print(f"{name:<20} | {m.hit_at_k.get(1,0)*100:6.2f}% | {m.hit_at_k.get(3,0)*100:6.2f}% | {m.hit_at_k.get(5,0)*100:6.2f}% | {m.hit_at_k.get(10,0)*100:6.2f}% | {m.mrr:.4f}")


if __name__ == "__main__":
    main()
