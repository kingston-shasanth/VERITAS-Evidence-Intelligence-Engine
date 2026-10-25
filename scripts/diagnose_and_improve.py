"""
Phase 3.1: Comprehensive Verification Model Diagnosis and Improvement.

Systematically audits:
1. Contextualization: Raw snippet vs. Question-Contextualized Evidence.
2. NLI Model Candidates (evaluated strictly on fixed Validation Split, N=255):
   - cross-encoder/nli-deberta-v3-xsmall
   - cross-encoder/nli-distilroberta-base
   - cross-encoder/nli-deberta-v3-base
   - MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
3. Aggregation strategy (Individual passage aggregation vs. Joint premise scoring).
4. Threshold tuning on validation data.
5. Final unbiased evaluation on unchanged held-out test set (N=256).

Safety & Resumption Features:
- Atomic per-model JSON checkpointing.
- Startup cache verification and skipping of completed models.
- Interrupted/incomplete model result validation (never treated as completed).
- CLI filtering to benchmark one selected model at a time.
- GPU detection (torch.cuda.is_available()) with CPU fallback.
- Granular progress and timing logging.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from veritas.data.schema import CanonicalClaim, CanonicalVeracity
from veritas.verification.evaluation import VeracityEvaluator
from veritas.verification.calibrator import TemperatureCalibrator

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "huggingface")
)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CLASS_LABELS = [
    CanonicalVeracity.VERIFIED_TRUE.value,
    CanonicalVeracity.VERIFIED_FALSE.value,
    CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value,
    CanonicalVeracity.INSUFFICIENT_EVIDENCE.value,
]

CANDIDATE_MODELS = [
    "cross-encoder/nli-deberta-v3-xsmall",
    "cross-encoder/nli-distilroberta-base",
    "cross-encoder/nli-deberta-v3-base",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
]


def get_device(requested_device: Optional[str] = None) -> str:
    """
    Selects compute device. Honors override if given; otherwise checks CUDA availability with CPU fallback.
    """
    if requested_device:
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_completed_result(model_data: Any) -> bool:
    """
    Verifies that a cached model result is complete, valid, and not from an interrupted run.
    """
    if not isinstance(model_data, dict):
        return False
    if model_data.get("status") != "completed":
        return False
    raw = model_data.get("raw")
    ctx = model_data.get("contextualized")
    if not isinstance(raw, dict) or not isinstance(ctx, dict):
        return False
    required_keys = ["accuracy", "macro_f1", "per_class", "confusion_matrix"]
    if not all(k in raw for k in required_keys) or not all(k in ctx for k in required_keys):
        return False
    return True


def load_comparison_results(filepath: str) -> Dict[str, Any]:
    """
    Safely loads comparison results from disk if the file exists and contains valid JSON.
    """
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as e:
        print(f"[!] Warning: Could not read existing results from {filepath} ({e}). Starting fresh.")
    return {}


def save_comparison_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Atomically writes comparison results to disk via temporary file and rename.
    """
    abs_path = os.path.abspath(filepath)
    target_dir = os.path.dirname(abs_path)
    os.makedirs(target_dir, exist_ok=True)
    temp_path = os.path.join(target_dir, f".tmp_{os.path.basename(abs_path)}")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, abs_path)


class RobustNLI:
    """
    Robust NLI loader supporting dynamic id2label mapping across any model checkpoint.
    """
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 256):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model.to(device)
        self.model.eval()

        # Dynamic label mapping
        label2id = {str(k).lower(): v for k, v in self.model.config.label2id.items()}
        self.idx_contra = label2id.get("contradiction", 0)
        self.idx_entail = label2id.get("entailment", 1)
        self.idx_neutral = label2id.get("neutral", 2)

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        if not pairs:
            return []
        premises = [p if p.strip() else "empty" for p, _ in pairs]
        hypotheses = [h if h.strip() else "empty" for _, h in pairs]

        inputs = self.tokenizer(
            premises,
            hypotheses,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        results = []
        for idx in range(len(pairs)):
            results.append({
                "contradiction": float(probs[idx][self.idx_contra]),
                "entailment": float(probs[idx][self.idx_entail]),
                "neutral": float(probs[idx][self.idx_neutral]),
            })
        return results


def build_evidence_premises(claim: CanonicalClaim, use_context: bool = True) -> List[str]:
    """
    Extracts evidence premises for a claim.
    If use_context=True, formats as 'Question: {Q} Answer: {A}'.
    """
    premises = []
    for ev in claim.evidence:
        ans = (ev.text or "").strip()
        q = (ev.metadata.get("question") or "").strip()
        if not ans or ans.lower() in ("no answer could be found.", "no answer could be found"):
            continue
        if use_context and q:
            premises.append(f"Question: {q} Answer: {ans}")
        else:
            premises.append(ans)
    return premises


def predict_claim_veracity(
    claim: CanonicalClaim,
    nli: Any,
    use_context: bool = True,
    use_joint: bool = True,
    conflict_threshold: float = 0.35,
    min_mass: float = 0.15,
) -> Tuple[str, List[float]]:
    """
    Predicts veracity for a claim given its evidence using NLI reasoning.
    Returns (predicted_label, raw_logits).
    """
    premises = build_evidence_premises(claim, use_context=use_context)
    if not premises:
        # Zero evidence -> Insufficient
        return CanonicalVeracity.INSUFFICIENT_EVIDENCE.value, [-3.0, -3.0, -3.0, 3.0]

    pairs = [(p, claim.claim_text) for p in premises]

    # If use_joint, also add the concatenated joint premise
    if use_joint and len(premises) > 1:
        joint_p = " ".join(premises)
        pairs.append((joint_p, claim.claim_text))

    nli_results = nli.predict_batch(pairs)

    entail_mass = 0.0
    contra_mass = 0.0

    # Weight individual premises
    num_individual = len(premises)
    for i in range(num_individual):
        r = nli_results[i]
        # Ignore highly neutral snippets
        if r["neutral"] > 0.85:
            continue
        entail_mass += r["entailment"]
        contra_mass += r["contradiction"]

    # Weight joint premise if present
    if use_joint and len(premises) > 1:
        joint_res = nli_results[-1]
        entail_mass += 1.5 * joint_res["entailment"]
        contra_mass += 1.5 * joint_res["contradiction"]

    total_mass = entail_mass + contra_mass
    conflict_ratio = (
        (2.0 * min(entail_mass, contra_mass)) / (total_mass + 1e-6)
        if total_mass > 0 else 0.0
    )

    # 4-class logits: [TRUE, FALSE, CONFLICTING, INSUFFICIENT]
    z_true = (entail_mass - contra_mass) * 2.5
    z_false = (contra_mass - entail_mass) * 2.5
    z_conf = (conflict_ratio * total_mass - 0.25) * 3.0
    z_insuf = (min_mass - total_mass) * 4.0

    raw_logits = [z_true, z_false, z_conf, z_insuf]

    if total_mass < min_mass:
        pred_label = CanonicalVeracity.INSUFFICIENT_EVIDENCE.value
    elif (
        entail_mass >= 0.15
        and contra_mass >= 0.15
        and conflict_ratio >= conflict_threshold
    ):
        pred_label = CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value
    elif entail_mass > contra_mass:
        pred_label = CanonicalVeracity.VERIFIED_TRUE.value
    else:
        pred_label = CanonicalVeracity.VERIFIED_FALSE.value

    return pred_label, raw_logits


def evaluate_split(
    claims: List[CanonicalClaim],
    nli: Any,
    use_context: bool = True,
    use_joint: bool = True,
    conflict_threshold: float = 0.35,
    min_mass: float = 0.15,
    pass_name: str = "",
    log_interval: int = 50,
) -> Dict[str, Any]:
    y_true = []
    y_pred = []
    all_logits = []
    total = len(claims)
    t_start = time.time()

    for idx, c in enumerate(claims, 1):
        pred, logits = predict_claim_veracity(
            c, nli, use_context=use_context, use_joint=use_joint,
            conflict_threshold=conflict_threshold, min_mass=min_mass,
        )
        y_true.append(c.canonical_label.value)
        y_pred.append(pred)
        all_logits.append(logits)

        if log_interval > 0 and (idx % log_interval == 0 or idx == total):
            elapsed = time.time() - t_start
            tag = f"[{pass_name}] " if pass_name else ""
            print(f"      {tag}Claim {idx}/{total} processed ({elapsed:.1f}s elapsed, {elapsed/idx:.3f}s/claim)")

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    per_class = {}
    for i, l in enumerate(CLASS_LABELS):
        per_class[l] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }

    return {
        "accuracy": float(acc),
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "logits": all_logits,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    default_out = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "processed", "validation_model_comparison.json"
    )
    parser = argparse.ArgumentParser(
        description="Phase 3.1: Verification Model Diagnosis, Benchmarking, and Evaluation."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional: Run only a specific model (name or substring match). E.g. 'cross-encoder/nli-deberta-v3-xsmall'.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=default_out,
        help="Path to JSON file where comparison results are cached and saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force compute device ('cpu', 'cuda', etc.). If omitted, auto-detects CUDA.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation of models even if completed results already exist in cache.",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate model strictly on the untouched held-out test split (N=256) exactly once.",
    )
    return parser.parse_args(args)


def main(
    cli_args: Optional[List[str]] = None,
    nli_factory: Optional[Callable[[str, str], Any]] = None,
    claims_override: Optional[List[CanonicalClaim]] = None,
) -> Dict[str, Any]:
    args = parse_args(cli_args)

    device = get_device(args.device)
    print("=" * 75)
    print("PHASE 3.1: VERIFICATION MODEL DIAGNOSIS AND MODEL SELECTION")
    print("=" * 75)
    print(f"[*] Compute device selected: {device} (CUDA available: {torch.cuda.is_available()})")
    print(f"[*] Results cache target:    {os.path.abspath(args.output_file)}")

    if nli_factory is None:
        nli_factory = lambda m, dev: RobustNLI(m, device=dev)

    # 1. Load claims and establish validation split
    if claims_override is not None:
        claims = claims_override
    else:
        canonical_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "canonical_claims.jsonl")
        claims = []
        with open(canonical_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    claims.append(CanonicalClaim.from_dict(json.loads(line)))

    eval_claims = [c for c in claims if c.source_dataset.value in ("AVERITEC", "MULTILINGUAL_DIAGNOSTIC")]
    val_split, test_split = VeracityEvaluator.create_stratified_splits(eval_claims, val_ratio=0.5, random_seed=42)

    default_val_out = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "processed", "validation_model_comparison.json"
    )
    if args.eval_test:
        eval_split = test_split
        split_label = f"Held-Out Test Split (N={len(test_split)})"
        out_file = (
            args.output_file
            if args.output_file != default_val_out
            else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "test_model_evaluation.json")
        )
    else:
        eval_split = val_split
        split_label = f"Validation Split (N={len(val_split)})"
        out_file = args.output_file

    print(f"Target Evaluation Split: {split_label}")
    print(f"Results target file:     {os.path.abspath(out_file)}")

    # 2. Majority-Class Baseline on Validation & Test
    val_y = [c.canonical_label.value for c in val_split]
    test_y = [c.canonical_label.value for c in test_split]
    maj_class = Counter(val_y).most_common(1)[0][0]

    val_maj_pred = [maj_class] * len(val_y)
    test_maj_pred = [maj_class] * len(test_y)

    val_maj_f1 = np.mean(precision_recall_fscore_support(val_y, val_maj_pred, labels=CLASS_LABELS, zero_division=0)[2])
    test_maj_f1 = np.mean(precision_recall_fscore_support(test_y, test_maj_pred, labels=CLASS_LABELS, zero_division=0)[2])
    print(f"\n[Baseline 1] Majority Class ({maj_class}):")
    print(f"  Validation Acc: {accuracy_score(val_y, val_maj_pred)*100:.2f}% | Macro-F1: {val_maj_f1*100:.2f}%")
    print(f"  Test Acc:       {accuracy_score(test_y, test_maj_pred)*100:.2f}% | Macro-F1: {test_maj_f1*100:.2f}%")

    # 3. Model Filtering
    if args.model:
        matched = [m for m in CANDIDATE_MODELS if args.model.lower() in m.lower() or args.model == m]
        if matched:
            models_to_run = matched
        else:
            models_to_run = [args.model]
        print(f"\n[*] Filter applied: Running only selected model(s): {models_to_run}")
    else:
        models_to_run = list(CANDIDATE_MODELS)
        print(f"\n[*] Evaluating full candidate model suite: {models_to_run}")

    # 4. Load existing results
    val_results = load_comparison_results(out_file)
    if val_results:
        completed_cached = [m for m, res in val_results.items() if is_completed_result(res)]
        print(f"[*] Found existing cache with {len(completed_cached)} completed models: {completed_cached}")

    # 5. Model Evaluation Loop
    for idx, m_name in enumerate(models_to_run, 1):
        print(f"\n" + "-" * 75)
        print(f"--- [Model {idx}/{len(models_to_run)}] {m_name} ---")

        # Skip if already completed and not forced
        if not args.force and m_name in val_results and is_completed_result(val_results[m_name]):
            cached_res = val_results[m_name]
            ctx_acc = cached_res.get("contextualized", {}).get("accuracy", 0.0) * 100
            ctx_f1 = cached_res.get("contextualized", {}).get("macro_f1", 0.0) * 100
            print(f"[*] Skipping {m_name}: valid completed results already cached.")
            print(f"    Cached Contextualized Acc: {ctx_acc:.2f}% | Macro-F1: {ctx_f1:.2f}%")
            continue

        if m_name in val_results and not is_completed_result(val_results[m_name]):
            print(f"[!] Previous cached entry for {m_name} was incomplete or corrupted. Re-evaluating...")

        t_model_start = time.time()
        try:
            print(f"    Loading model on device '{device}'...")
            t0 = time.time()
            nli = nli_factory(m_name, device)
            load_time = time.time() - t0
            print(f"    Loaded in {load_time:.2f}s")

            # Pass 1: Raw answers alone
            print(f"    Pass 1/2: Raw Answers Alone (N={len(eval_split)} claims)...")
            t1 = time.time()
            res_raw = evaluate_split(
                eval_split, nli, use_context=False, use_joint=False, pass_name="Pass 1/2 Raw"
            )
            t_raw = time.time() - t1
            print(f"    Pass 1/2 Complete: Acc={res_raw['accuracy']*100:.2f}%, Macro-F1={res_raw['macro_f1']*100:.2f}% ({t_raw:.1f}s)")
            for l in CLASS_LABELS:
                p = res_raw["per_class"][l]
                print(f"      {l:28s}: P={p['precision']*100:5.1f}% | R={p['recall']*100:5.1f}% | F1={p['f1']*100:5.1f}%")

            # Pass 2: Contextualized Q+A + Joint
            print(f"    Pass 2/2: Contextualized Q+A + Joint (N={len(eval_split)} claims)...")
            t2 = time.time()
            res_ctx = evaluate_split(
                eval_split, nli, use_context=True, use_joint=True, pass_name="Pass 2/2 Ctx+Joint"
            )
            t_ctx = time.time() - t2
            print(f"    Pass 2/2 Complete: Acc={res_ctx['accuracy']*100:.2f}%, Macro-F1={res_ctx['macro_f1']*100:.2f}% ({t_ctx:.1f}s)")
            for l in CLASS_LABELS:
                p = res_ctx["per_class"][l]
                print(f"      {l:28s}: P={p['precision']*100:5.1f}% | R={p['recall']*100:5.1f}% | F1={p['f1']*100:5.1f}%")

            # Store completed result entry only after both passes finish successfully
            val_results[m_name] = {
                "status": "completed",
                "model_name": m_name,
                "split": "test" if args.eval_test else "validation",
                "device": device,
                "completed_at_timestamp": time.time(),
                "load_time_sec": load_time,
                "raw_time_sec": t_raw,
                "contextualized_time_sec": t_ctx,
                "total_time_sec": time.time() - t_model_start,
                "raw": res_raw,
                "contextualized": res_ctx,
            }

            # Atomic save immediately after each completed model
            save_comparison_results(val_results, out_file)
            print(f"[*] Checkpoint saved atomically for {m_name} to {out_file}")

        except Exception as e:
            print(f"[!] FAILED evaluation for {m_name}: {e}")
            # Ensure incomplete or errored result is never committed as completed
            if m_name in val_results and not is_completed_result(val_results[m_name]):
                val_results.pop(m_name, None)

    print("\n" + "=" * 75)
    print(f"[*] Evaluation run finished. Results saved in: {out_file}")
    print("=" * 75)
    return val_results


if __name__ == "__main__":
    main()
