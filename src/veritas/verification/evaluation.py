"""
Veracity Evaluation & Diagnostic Suite for VERITAS-X.
Computes Macro-F1, Per-Class Precision/Recall, 4x4 Confusion Matrices,
ECE calibration diagnostics, and Coverage-Risk abstention curves on leakage-free splits.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from veritas.data.schema import CanonicalClaim, CanonicalVeracity
from .engine import VerificationEngine, ForensicVerdict
from .calibrator import TemperatureCalibrator


@dataclass
class VeracityEvaluationReport:
    """Full quantitative performance assessment on the unseen test split."""
    total_evaluated: int
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    class_names: List[str]
    uncalibrated_ece: float
    calibrated_ece: float
    abstention_rate: float
    coverage_risk_curve: List[Dict[str, float]] = field(default_factory=list)
    language_breakdown: Dict[str, Dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== VERITAS-X Veracity Evaluation Report (Unseen Test Split) ===",
            f"Total Test Claims Evaluated : {self.total_evaluated}",
            f"Macro-F1 Score              : {self.macro_f1 * 100:.2f}%",
            f"Macro-Precision             : {self.macro_precision * 100:.2f}%",
            f"Macro-Recall                : {self.macro_recall * 100:.2f}%",
            f"Abstention Rate             : {self.abstention_rate * 100:.2f}%",
            f"Calibration Error (ECE)     : {self.uncalibrated_ece * 100:.2f}% -> {self.calibrated_ece * 100:.2f}%",
            "",
            "--- Per-Class Performance ---",
            f"{'Class Label':<30} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Support':<7}",
            "-" * 65,
        ]
        for cname in self.class_names:
            metrics = self.per_class_metrics.get(cname, {})
            p = metrics.get("precision", 0.0) * 100
            r = metrics.get("recall", 0.0) * 100
            f1 = metrics.get("f1", 0.0) * 100
            sup = int(metrics.get("support", 0))
            lines.append(f"{cname:<30} | {p:5.1f}% | {r:5.1f}% | {f1:5.1f}% | {sup:<7d}")

        lines.extend([
            "",
            "--- 4x4 Confusion Matrix ---",
            f"{'Ground Truth \\ Predicted':<28} | " + " | ".join(f"{c[:10]:<10}" for c in self.class_names),
            "-" * 78,
        ])
        for idx, row in enumerate(self.confusion_matrix):
            c_true = self.class_names[idx]
            row_str = " | ".join(f"{val:<10d}" for val in row)
            lines.append(f"{c_true:<28} | {row_str}")

        if self.coverage_risk_curve:
            lines.extend([
                "",
                "--- Selective Abstention (Coverage vs. Risk) ---",
                f"{'Coverage':<12} | {'Selective Risk (Error)':<22} | {'Confidence Threshold':<20}",
                "-" * 60,
            ])
            for pt in self.coverage_risk_curve:
                cov = pt["coverage"] * 100
                risk = pt["risk"] * 100
                th = pt["threshold"]
                lines.append(f"{cov:5.1f}%       | {risk:5.1f}%                | >= {th:.2f}")

        if self.language_breakdown:
            lines.extend([
                "",
                "--- Language Breakdown ---",
                f"{'Language':<10} | {'Evaluated':<10} | {'Accuracy':<10} | {'Macro-F1':<10}",
                "-" * 45,
            ])
            for lang, dat in sorted(self.language_breakdown.items()):
                cnt = int(dat.get("count", 0))
                acc = dat.get("accuracy", 0.0) * 100
                f1_l = dat.get("macro_f1", 0.0) * 100
                lines.append(f"{lang:<10} | {cnt:<10d} | {acc:5.1f}%    | {f1_l:5.1f}%")

        return "\n".join(lines)


class VeracityEvaluator:
    """
    Evaluates verification performance across strictly partitioned validation and test splits.
    """

    CLASS_LABELS = [
        CanonicalVeracity.VERIFIED_TRUE.value,
        CanonicalVeracity.VERIFIED_FALSE.value,
        CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value,
        CanonicalVeracity.INSUFFICIENT_EVIDENCE.value,
    ]

    @classmethod
    def create_stratified_splits(
        cls,
        claims: List[CanonicalClaim],
        val_ratio: float = 0.5,
        random_seed: int = 42,
    ) -> Tuple[List[CanonicalClaim], List[CanonicalClaim]]:
        """
        Stratifies claims by canonical_label into independent validation and test sets.
        """
        rng = np.random.RandomState(random_seed)
        by_class: Dict[str, List[CanonicalClaim]] = {}

        for c in claims:
            lbl = c.canonical_label.value
            if lbl not in by_class:
                by_class[lbl] = []
            by_class[lbl].append(c)

        val_split: List[CanonicalClaim] = []
        test_split: List[CanonicalClaim] = []

        for lbl, group in by_class.items():
            shuffled = list(group)
            rng.shuffle(shuffled)
            split_idx = int(len(shuffled) * val_ratio)
            val_split.extend(shuffled[:split_idx])
            test_split.extend(shuffled[split_idx:])

        # Shuffle splits to remove class ordering
        rng.shuffle(val_split)
        rng.shuffle(test_split)

        return val_split, test_split

    def fit_calibrator(
        self,
        engine: VerificationEngine,
        val_claims: List[CanonicalClaim],
    ) -> TemperatureCalibrator:
        """
        Runs the engine over the validation split, gathers logits, and fits temperature scaling.
        Never touches the test split.
        """
        val_logits: List[List[float]] = []
        val_targets: List[int] = []

        for claim in val_claims:
            verdict = engine.verify_claim(
                claim_text=claim.claim_text,
                claim_date=claim.claim_date,
                claim_id=claim.claim_id,
            )
            # Find class index
            true_lbl = claim.canonical_label.value
            if true_lbl in self.CLASS_LABELS:
                target_idx = self.CLASS_LABELS.index(true_lbl)
                # Map probabilities to logits
                probs = [verdict.class_probabilities[c] for c in self.CLASS_LABELS]
                eps = 1e-12
                # Invert softmax to log-odds
                logits = [float(np.log(max(p, eps))) for p in probs]
                val_logits.append(logits)
                val_targets.append(target_idx)

        val_logits_np = np.array(val_logits, dtype=np.float32)
        val_targets_np = np.array(val_targets, dtype=int)

        calibrator = TemperatureCalibrator()
        calibrator.fit(val_logits_np, val_targets_np)
        return calibrator

    def evaluate(
        self,
        engine: VerificationEngine,
        test_claims: List[CanonicalClaim],
    ) -> VeracityEvaluationReport:
        """
        Evaluates the fully calibrated engine strictly on unseen test claims.
        """
        y_true: List[str] = []
        y_pred: List[str] = []
        confidences: List[float] = []
        probs_all: List[List[float]] = []
        abstained_count = 0

        lang_data: Dict[str, Dict[str, List[Any]]] = {}

        for claim in test_claims:
            lang = claim.language or "en"
            if lang not in lang_data:
                lang_data[lang] = {"true": [], "pred": []}

            verdict: ForensicVerdict = engine.verify_claim(
                claim_text=claim.claim_text,
                claim_date=claim.claim_date,
                claim_id=claim.claim_id,
            )

            true_str = claim.canonical_label.value
            pred_str = verdict.verdict.value

            y_true.append(true_str)
            y_pred.append(pred_str)
            confidences.append(verdict.confidence)
            probs_all.append([verdict.class_probabilities[c] for c in self.CLASS_LABELS])

            lang_data[lang]["true"].append(true_str)
            lang_data[lang]["pred"].append(pred_str)

            if verdict.was_abstained:
                abstained_count += 1

        # Macro and Per-Class PRF
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=self.CLASS_LABELS, zero_division=0
        )
        macro_p = float(np.mean(prec))
        macro_r = float(np.mean(rec))
        macro_f = float(np.mean(f1))

        per_class: Dict[str, Dict[str, float]] = {}
        for i, cname in enumerate(self.CLASS_LABELS):
            per_class[cname] = {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": float(sup[i]),
            }

        cm = confusion_matrix(y_true, y_pred, labels=self.CLASS_LABELS).tolist()

        # Calibration metrics on test split
        probs_np = np.array(probs_all, dtype=np.float32)
        true_indices = np.array([self.CLASS_LABELS.index(yt) for yt in y_true], dtype=int)
        calib_ece, _, _, _ = TemperatureCalibrator.compute_ece(probs_np, true_indices)

        # Uncalibrated baseline ECE on test split (temperature=1.0)
        uncalib_ece, _, _, _ = TemperatureCalibrator.compute_ece(probs_np ** engine.calibrator.temperature, true_indices)

        # Coverage-Risk Trade-off Curve
        cov_risk: List[Dict[str, float]] = []
        conf_arr = np.array(confidences)
        correct_arr = (np.array(y_true) == np.array(y_pred)).astype(float)

        for th in [0.25, 0.40, 0.50, 0.60, 0.75]:
            covered_mask = conf_arr >= th
            cov = float(np.mean(covered_mask))
            if np.sum(covered_mask) > 0:
                risk = float(1.0 - np.mean(correct_arr[covered_mask]))
            else:
                risk = 0.0
            cov_risk.append({"threshold": th, "coverage": cov, "risk": risk})

        # Language breakdown
        lang_breakdown: Dict[str, Dict[str, float]] = {}
        for l_code, l_dict in lang_data.items():
            l_true = l_dict["true"]
            l_pred = l_dict["pred"]
            l_acc = float(np.mean(np.array(l_true) == np.array(l_pred)))
            _, _, l_f1s, _ = precision_recall_fscore_support(
                l_true, l_pred, labels=self.CLASS_LABELS, zero_division=0
            )
            lang_breakdown[l_code] = {
                "count": float(len(l_true)),
                "accuracy": l_acc,
                "macro_f1": float(np.mean(l_f1s)),
            }

        return VeracityEvaluationReport(
            total_evaluated=len(test_claims),
            macro_f1=macro_f,
            macro_precision=macro_p,
            macro_recall=macro_r,
            per_class_metrics=per_class,
            confusion_matrix=cm,
            class_names=self.CLASS_LABELS,
            uncalibrated_ece=uncalib_ece,
            calibrated_ece=calib_ece,
            abstention_rate=float(abstained_count / len(test_claims)),
            coverage_risk_curve=cov_risk,
            language_breakdown=lang_breakdown,
        )
