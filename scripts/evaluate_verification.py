"""
Verification Benchmarking Runner for VERITAS-X Phase 3.
Evaluates end-to-end evidence reasoning, temperature calibration, and selective abstention
on leakage-free validation and test splits.
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
from veritas.verification.nli_model import CrossEncoderNLI
from veritas.verification.engine import VerificationEngine
from veritas.verification.evaluation import VeracityEvaluator

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

    print("\n[*] Loading local reproducible EvidenceCorpus from:", corpus_path)
    corpus = EvidenceCorpus.load_from_jsonl(corpus_path)
    print(f"    Loaded {len(corpus)} verified evidence passages.")

    print("\n[*] Initializing Hybrid Retrieval Engine (BM25 + Dense E5 + RRF)...")
    t0 = time.time()
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, device="cpu")
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)
    print(f"    Retrieval subsystem ready in {time.time() - t0:.2f}s")

    print("\n[*] Initializing CrossEncoder NLI Engine (DeBERTa-v3)...")
    t0 = time.time()
    nli = CrossEncoderNLI(device="cpu")
    print(f"    NLI model loaded in {time.time() - t0:.2f}s")

    engine = VerificationEngine(nli_model=nli, retriever=hybrid)

    # Filter claims to those evaluated against our local evidence index
    # (AVeriTeC real-world claims + Multilingual diagnostic claims)
    eval_claims = [c for c in claims if c.source_dataset.value in ("AVERITEC", "MULTILINGUAL_DIAGNOSTIC")]
    print(f"\n[*] Filtering to verifiable evidentiary claims: {len(eval_claims)} claims.")

    print("\n[*] Creating leakage-resistant stratified validation (50%) and test (50%) splits...")
    val_split, test_split = VeracityEvaluator.create_stratified_splits(eval_claims, val_ratio=0.5, random_seed=42)
    print(f"    Validation claims (Calibration Split) : {len(val_split)}")
    print(f"    Test claims (Unseen Evaluation Split)  : {len(test_split)}")

    evaluator = VeracityEvaluator()

    # Step 1: Fit temperature calibration strictly on validation split
    print("\n" + "="*70)
    print("[*] Fitting Platt Temperature Scaling strictly on Validation split...")
    t0 = time.time()
    calibrator = evaluator.fit_calibrator(engine, val_split)
    print(f"    Calibration optimization finished in {time.time() - t0:.2f}s")
    print(f"    Optimal Learned Temperature (T): {calibrator.temperature:.4f}")

    # Attach learned calibrator to engine
    engine.calibrator = calibrator

    # Step 2: Evaluate strictly on unseen test split
    print("\n" + "="*70)
    print("[*] Executing full forensic verification on unseen Test split...")
    t0 = time.time()
    report = evaluator.evaluate(engine, test_split)
    eval_time = time.time() - t0
    print(f"    Test evaluation finished in {eval_time:.2f}s ({eval_time / len(test_split) * 1000:.1f}ms / claim)")

    print("\n" + report.summary())


if __name__ == "__main__":
    main()
