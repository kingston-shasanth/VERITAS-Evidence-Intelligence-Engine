"""
Phase 3.2.3: Validation-Only Hyperparameter Tuning for VERITAS-X Evidence Aggregator.

Optimizes cross-passage stance disparity and aggregation thresholds strictly on the
fixed validation split (N=255) to maximize 4-class Validation Macro-F1.

Guarantees:
- The held-out test split (N=256) is never loaded, inspected, scored, or tuned against.
- Benchmark result caches (validation_model_comparison.json, test_model_evaluation.json)
  remain strictly untouched.
- NLI premise inference is performed once on validation claims and cached to disk/memory,
  allowing fast and deterministic parameter exploration.
- Abstention coverage is explicitly measured; configurations with coverage < 85%
  are flagged as disqualified to prevent artificial precision inflation.
"""
import os
import sys
import json
import time
import argparse
import itertools
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from veritas.data.schema import CanonicalClaim, CanonicalVeracity
from veritas.verification.evaluation import VeracityEvaluator
from veritas.verification.evidence_aggregator import EvidenceAggregator
from veritas.verification.nli_model import CrossEncoderNLI, resolve_device

CLASS_LABELS = [
    CanonicalVeracity.VERIFIED_TRUE.value,
    CanonicalVeracity.VERIFIED_FALSE.value,
    CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value,
    CanonicalVeracity.INSUFFICIENT_EVIDENCE.value,
]

DEFAULT_GRID = {
    "disparity_entail_threshold": [0.35, 0.40, 0.45, 0.50],
    "disparity_contra_threshold": [0.35, 0.40, 0.45, 0.50],
    "conflict_threshold": [0.30, 0.35, 0.40, 0.45],
    "min_informative_mass": [0.10, 0.15, 0.20, 0.25],
    "joint_weight_disparity": [0.0, 0.2, 0.4],
    "joint_weight_default": [1.0, 1.2],
}


def load_validation_split_only(canonical_path: str) -> List[CanonicalClaim]:
    """
    Loads canonical claims and extracts STRICTLY the validation split (N=255).
    The held-out test split is deliberately deleted immediately to ensure zero leakage.
    """
    claims: List[CanonicalClaim] = []
    with open(canonical_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                claims.append(CanonicalClaim.from_dict(json.loads(line)))

    eval_claims = [c for c in claims if c.source_dataset.value in ("AVERITEC", "MULTILINGUAL_DIAGNOSTIC")]
    val_split, test_split = VeracityEvaluator.create_stratified_splits(eval_claims, val_ratio=0.5, random_seed=42)

    # Strictly isolate validation records; purge test_split from memory
    del test_split
    if len(val_split) != 255:
        raise ValueError(f"Expected validation split size of 255 claims, got {len(val_split)}")

    return val_split


def extract_or_cache_val_predictions(
    val_split: List[CanonicalClaim],
    nli_model: Optional[CrossEncoderNLI] = None,
    cache_path: Optional[str] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    """
    Computes or retrieves cached NLI predictions for all validation claims.
    Caches:
    - candidate_passages: Individual premises formatted as 'Question: {Q} Answer: {A}'
    - nli_predictions: Model scores for individual premises
    - joint_nli_prediction: Model score for concatenated joint premise (if >1 premise)
    """
    if cache_path and os.path.exists(cache_path) and not force_recompute:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, list) and len(cached) == len(val_split):
                # Verify alignment of claim IDs
                if all(cached[i]["claim_id"] == val_split[i].claim_id for i in range(len(val_split))):
                    print(f"[*] Loaded {len(cached)} cached validation NLI predictions from: {cache_path}")
                    return cached
        except Exception as e:
            print(f"[!] Could not load cache from {cache_path} ({e}). Recomputing...")

    if nli_model is None:
        raise ValueError("NLI model must be provided when cache is missing or recomputed.")

    print(f"[*] Computing NLI premise scores on {len(val_split)} validation claims using {nli_model.model_name}...")
    t0 = time.time()
    records: List[Dict[str, Any]] = []

    for idx, claim in enumerate(val_split, 1):
        premises: List[str] = []
        for ev in claim.evidence:
            ans = (ev.text or "").strip()
            q = (ev.metadata.get("question") or "").strip()
            if not ans or ans.lower() in ("no answer could be found.", "no answer could be found"):
                continue
            premise = f"Question: {q} Answer: {ans}" if q else ans
            premises.append(premise)

        if not premises:
            records.append({
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "true_label": claim.canonical_label.value,
                "candidate_passages": [],
                "nli_predictions": [],
                "joint_nli_prediction": None,
            })
            continue

        candidate_passages = [
            {"doc_id": f"{claim.claim_id}_ev_{i}", "text": p, "is_joint": False}
            for i, p in enumerate(premises)
        ]
        pairs = [(p, claim.claim_text) for p in premises]
        individual_preds = nli_model.predict_batch(pairs)

        joint_pred = None
        if len(premises) > 1:
            joint_premise = " ".join(premises)
            joint_res = nli_model.predict_batch([(joint_premise, claim.claim_text)])
            joint_pred = joint_res[0] if joint_res else None

        records.append({
            "claim_id": claim.claim_id,
            "claim_text": claim.claim_text,
            "true_label": claim.canonical_label.value,
            "candidate_passages": candidate_passages,
            "nli_predictions": individual_preds,
            "joint_nli_prediction": joint_pred,
        })

        if idx % 50 == 0 or idx == len(val_split):
            elapsed = time.time() - t0
            print(f"    Claim {idx}/{len(val_split)} processed ({elapsed:.1f}s, {elapsed/idx:.3f}s/claim)")

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"[*] Validation NLI predictions successfully cached to: {cache_path}")

    return records


def evaluate_aggregator_configuration(
    aggregator_params: Dict[str, Any],
    cached_records: List[Dict[str, Any]],
    min_coverage_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Evaluates a specific aggregator parameter set over the precomputed validation cache.
    Executes in pure Python/NumPy arithmetic in < 1 ms per configuration.
    """
    aggregator = EvidenceAggregator(**aggregator_params)
    y_true: List[str] = []
    y_pred: List[str] = []
    disparity_flags: List[bool] = []

    for item in cached_records:
        y_true.append(item["true_label"])
        verdict = aggregator.aggregate(
            candidate_passages=item["candidate_passages"],
            nli_predictions=item["nli_predictions"],
            joint_nli_prediction=item["joint_nli_prediction"],
        )
        y_pred.append(verdict.canonical_verdict.value)
        disparity_flags.append(verdict.has_cross_passage_disparity)

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    macro_p = float(np.mean(prec))
    macro_r = float(np.mean(rec))
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist()

    # Measure prediction coverage: fraction of claims where verdict != INSUFFICIENT_EVIDENCE
    # Note: INSUFFICIENT_EVIDENCE is a canonical domain class, distinguished from selective abstention.
    total_claims = len(y_pred)
    non_insufficient_count = sum(1 for p in y_pred if p != CanonicalVeracity.INSUFFICIENT_EVIDENCE.value)
    coverage = float(non_insufficient_count / total_claims)

    per_class: Dict[str, Any] = {}
    for i, cname in enumerate(CLASS_LABELS):
        per_class[cname] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }

    return {
        "parameters": aggregator_params,
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "coverage": coverage,
        "is_low_coverage": coverage < min_coverage_threshold,
        "disparity_trigger_count": int(sum(disparity_flags)),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def compute_majority_baseline(y_true: List[str]) -> Dict[str, Any]:
    """Computes majority-class baseline performance on validation split."""
    maj_class = Counter(y_true).most_common(1)[0][0]
    maj_pred = [maj_class] * len(y_true)
    acc = float(accuracy_score(y_true, maj_pred))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, maj_pred, labels=CLASS_LABELS, zero_division=0
    )
    return {
        "majority_class": maj_class,
        "accuracy": acc,
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "coverage": 1.0,
    }


def run_grid_search(
    cached_records: List[Dict[str, Any]],
    grid: Dict[str, List[Any]],
    min_coverage_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Performs grid search across parameter space, enforcing coverage constraints.
    """
    param_names = list(grid.keys())
    param_values = list(grid.values())
    all_combinations = [dict(zip(param_names, v)) for v in itertools.product(*param_values)]

    total_combs = len(all_combinations)
    print(f"[*] Starting parameter grid search over {total_combs} combinations...")
    t_start = time.time()

    results: List[Dict[str, Any]] = []
    disqualified_low_coverage = 0

    for idx, params in enumerate(all_combinations, 1):
        res = evaluate_aggregator_configuration(
            params, cached_records, min_coverage_threshold=min_coverage_threshold
        )
        results.append(res)
        if res["is_low_coverage"]:
            disqualified_low_coverage += 1

    elapsed = time.time() - t_start
    print(f"[*] Evaluated {total_combs} combinations in {elapsed:.2f}s ({elapsed/total_combs*1000:.3f} ms/eval).")
    print(f"[*] Configurations meeting coverage >= {min_coverage_threshold*100:.0f}%: {total_combs - disqualified_low_coverage} / {total_combs}")

    # Valid candidates meeting minimum coverage
    valid_candidates = [r for r in results if not r["is_low_coverage"]]
    if not valid_candidates:
        print("[!] Warning: Zero configurations met the minimum coverage constraint! Relaxing to all.")
        valid_candidates = results

    # Sort candidates by:
    # 1. Macro-F1 (primary)
    # 2. Conflicting F1 (tie-breaker for forensic sensitivity)
    # 3. Accuracy (tie-breaker)
    def sort_key(r: Dict[str, Any]):
        conf_f1 = r["per_class"][CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value]["f1"]
        return (r["macro_f1"], conf_f1, r["accuracy"])

    valid_candidates.sort(key=sort_key, reverse=True)
    optimal_config = valid_candidates[0]

    # Evaluate default configuration for comparison
    default_params = {
        "disparity_entail_threshold": 0.40,
        "disparity_contra_threshold": 0.40,
        "conflict_threshold": 0.40,
        "min_informative_mass": 0.20,
        "joint_weight_default": 1.0,
        "joint_weight_disparity": 0.2,
    }
    default_eval = evaluate_aggregator_configuration(default_params, cached_records)

    y_val_true = [r["true_label"] for r in cached_records]
    majority_baseline = compute_majority_baseline(y_val_true)

    return {
        "total_combinations_tested": total_combs,
        "disqualified_low_coverage_count": disqualified_low_coverage,
        "search_time_sec": elapsed,
        "majority_class_baseline": majority_baseline,
        "default_configuration": default_eval,
        "selected_optimal_configuration": optimal_config,
        "top_5_configurations": valid_candidates[:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3.2.3: Validation-Only Threshold Tuning for VERITAS-X Evidence Aggregator."
    )
    parser.add_argument(
        "--canonical-path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "canonical_claims.jsonl"),
        help="Path to canonical claims JSONL file.",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", ".val_nli_predictions_cache.json"),
        help="Path to cached validation NLI predictions.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "threshold_tuning_results.json"),
        help="Destination path for tuning results JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device for initial NLI caching (auto-detects CUDA).",
    )
    parser.add_argument(
        "--force-recompute-cache",
        action="store_true",
        help="Force recomputation of validation NLI scores even if cache exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("PHASE 3.2.3: VALIDATION-ONLY HYPERPARAMETER TUNING")
    print("=" * 75)

    # 1. Load STRICTLY validation split (N=255)
    val_split = load_validation_split_only(args.canonical_path)
    print(f"[*] Validation Split Size: {len(val_split)} claims (Held-out test split excluded)")

    # 2. Extract or load cached NLI predictions
    cache_exists = os.path.exists(args.cache_file) and not args.force_recompute_cache
    nli_model = None
    if not cache_exists:
        device = resolve_device(args.device)
        print(f"[*] Initializing NLI model on device: {device}")
        nli_model = CrossEncoderNLI(device=device)

    cached_records = extract_or_cache_val_predictions(
        val_split=val_split,
        nli_model=nli_model,
        cache_path=args.cache_file,
        force_recompute=args.force_recompute_cache,
    )

    # 3. Run Grid Search
    search_summary = run_grid_search(cached_records, DEFAULT_GRID, min_coverage_threshold=0.85)

    # 4. Save results artifact
    output_data = {
        "dataset_split": "validation_only",
        "split_size": len(val_split),
        "evaluation_timestamp": time.time(),
        "model_used": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "tuning_objective": "maximize_validation_macro_f1_with_min_coverage_85",
        "parameter_grid": DEFAULT_GRID,
        "search_summary": search_summary,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"[*] Threshold tuning results saved to: {args.output_file}")

    # 5. Print Summary Comparison
    opt = search_summary["selected_optimal_configuration"]
    default = search_summary["default_configuration"]
    base = search_summary["majority_class_baseline"]

    print("\n" + "=" * 75)
    print("VALIDATION TUNING PERFORMANCE COMPARISON (N=255)")
    print("=" * 75)
    print(f"{'Metric':<25} | {'Majority Baseline':<18} | {'Default (P3.2.2)':<18} | {'Tuned Optimal':<18}")
    print("-" * 85)
    print(f"{'Macro-F1':<25} | {base['macro_f1']*100:6.2f}%            | {default['macro_f1']*100:6.2f}%            | {opt['macro_f1']*100:6.2f}%")
    print(f"{'Macro-Precision':<25} | {base['macro_precision']*100:6.2f}%            | {default['macro_precision']*100:6.2f}%            | {opt['macro_precision']*100:6.2f}%")
    print(f"{'Macro-Recall':<25} | {base['macro_recall']*100:6.2f}%            | {default['macro_recall']*100:6.2f}%            | {opt['macro_recall']*100:6.2f}%")
    print(f"{'Accuracy':<25} | {base['accuracy']*100:6.2f}%            | {default['accuracy']*100:6.2f}%            | {opt['accuracy']*100:6.2f}%")
    print(f"{'Coverage':<25} | {base['coverage']*100:6.2f}%            | {default['coverage']*100:6.2f}%            | {opt['coverage']*100:6.2f}%")

    print("\n--- Selected Optimal Parameters ---")
    for k, v in opt["parameters"].items():
        print(f"  {k:<30s}: {v}")

    print("\n--- Per-Class Performance (Tuned Optimal Configuration) ---")
    for cname, m in opt["per_class"].items():
        print(f"  {cname:<28s}: P={m['precision']*100:5.1f}% | R={m['recall']*100:5.1f}% | F1={m['f1']*100:5.1f}% (sup={m['support']})")

    print("\n--- 4x4 Confusion Matrix (Tuned Optimal Configuration) ---")
    print(f"Labels: {CLASS_LABELS}")
    for row in opt["confusion_matrix"]:
        print(f"  {row}")


if __name__ == "__main__":
    main()
