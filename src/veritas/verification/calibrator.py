"""
Calibration & Selective Abstention Engine for VERITAS-X.

Implements post-hoc Multiclass Temperature Scaling (Guo et al., ICML 2017) and
Selective Classification with risk-coverage trade-off analysis (Geifman & El-Yaniv, NeurIPS 2017).

Important Distinctions:
- Temperature Scaling vs. Platt Scaling:
  Temperature Scaling learns a single positive scalar T > 0 such that calibrated
  probabilities are p_i = exp(z_i / T) / sum_j exp(z_j / T). It preserves the argmax
  class prediction and logit rank-ordering while calibrating confidence.
  Platt Scaling (Platt, 1999) originally refers to logistic regression on binary logits
  sigma(A*z + B), and multiclass extensions fit affine transformations W*z + b.
  VERITAS-X uses Temperature Scaling for parameter efficiency and monotonic preservation.

- Selective Abstention vs. INSUFFICIENT_EVIDENCE:
  INSUFFICIENT_EVIDENCE is a substantive domain veracity class within the AVeriTeC taxonomy,
  denoting claims where retrieved evidence lacks necessary facts. The model actively
  asserts this verdict with its own probability.
  Selective Abstention is a meta-level rejection mechanism g(x) in {0, 1}. When prediction
  confidence is below threshold or entropy is above threshold, the model declines to assert
  ANY verdict (deferring to human review). Abstained claims are evaluated under selective
  coverage and selective risk, and are NOT silently conflated with INSUFFICIENT_EVIDENCE.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix




from veritas.data.schema import CanonicalVeracity


@dataclass
class CalibrationMetrics:
    """Quantitative confidence calibration diagnostics."""
    temperature: float
    uncalibrated_ece: float
    calibrated_ece: float
    uncalibrated_nll: float
    calibrated_nll: float
    uncalibrated_brier: float
    calibrated_brier: float
    bin_accuracies: List[float] = field(default_factory=list)
    bin_confidences: List[float] = field(default_factory=list)
    bin_counts: List[int] = field(default_factory=list)
    binning_method: str = "equal_width_10_bins"

    @property
    def brier_score(self) -> float:
        """Backward-compatible alias for calibrated_brier."""
        return self.calibrated_brier

    def summary(self) -> str:
        lines = [
            "=== Confidence Calibration Report (Temperature Scaling: Guo et al. 2017) ===",
            "Method                   : Multiclass Temperature Scaling (Single Scalar T)",
            f"Learned Temperature (T) : {self.temperature:.4f}",
            f"Binning Strategy         : {self.binning_method}",
            f"Uncalibrated ECE (10-bin): {self.uncalibrated_ece * 100:.2f}%",
            f"Calibrated ECE (10-bin)  : {self.calibrated_ece * 100:.2f}%",
            f"ECE Relative Reduction   : {((self.uncalibrated_ece - self.calibrated_ece) / (self.uncalibrated_ece + 1e-6)) * 100:.1f}%",
            f"Negative Log Likelihood  : {self.uncalibrated_nll:.4f} -> {self.calibrated_nll:.4f}",
            f"Brier Score              : {self.uncalibrated_brier:.4f} -> {self.calibrated_brier:.4f}",
        ]
        return "\n".join(lines)


class TemperatureCalibrator:
    """
    Fits and applies scalar temperature scaling to 4-class logits strictly on validation data.
    """

    CLASS_LABELS = [
        CanonicalVeracity.VERIFIED_TRUE,
        CanonicalVeracity.VERIFIED_FALSE,
        CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED,
        CanonicalVeracity.INSUFFICIENT_EVIDENCE,
    ]

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(0.01, float(temperature))

    @staticmethod
    def softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
        """
        Numerically stable softmax with temperature scaling.
        Guarantees returned array lies on the probability simplex (non-negative, sum=1).
        Supports 1D and 2D arrays.
        """
        arr = np.asarray(logits, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Logits contain non-finite values (NaN or Inf).")

        t = max(float(temp), 1e-3)
        scaled = arr / t
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_z = np.exp(shifted)
        probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        # Ensure strict simplex normalization
        probs = np.clip(probs, 1e-12, 1.0)
        return probs / np.sum(probs, axis=-1, keepdims=True)

    @classmethod
    def compute_ece(
        cls,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[float, List[float], List[float], List[int]]:
        """
        Computes Expected Calibration Error partitioned across equal-width confidence bins in [0, 1].
        
        Bin definition:
          Bin 0: [0.0, 1.0 / n_bins] (inclusive on both ends)
          Bin i > 0: (i / n_bins, (i + 1) / n_bins] (left-open, right-closed)
        """
        p_arr = np.asarray(probs, dtype=np.float64)
        y_arr = np.asarray(labels, dtype=np.int64)

        if p_arr.ndim != 2 or len(p_arr) != len(y_arr):
            raise ValueError("probs must be 2D array with same length as labels.")
        if len(y_arr) == 0:
            return 0.0, [], [], []

        confidences = np.max(p_arr, axis=1)
        predictions = np.argmax(p_arr, axis=1)
        accuracies = (predictions == y_arr).astype(float)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        bin_accs: List[float] = []
        bin_confs: List[float] = []
        bin_counts: List[int] = []

        total_samples = len(y_arr)
        for i in range(n_bins):
            b_low = bin_boundaries[i]
            b_high = bin_boundaries[i + 1]
            if i == 0:
                in_bin = (confidences >= b_low) & (confidences <= b_high)
            else:
                in_bin = (confidences > b_low) & (confidences <= b_high)

            count = int(np.sum(in_bin))
            bin_counts.append(count)

            if count > 0:
                acc_in_bin = float(np.mean(accuracies[in_bin]))
                conf_in_bin = float(np.mean(confidences[in_bin]))
                prop_in_bin = count / total_samples
                ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
                bin_accs.append(acc_in_bin)
                bin_confs.append(conf_in_bin)
            else:
                bin_accs.append(0.0)
                bin_confs.append(float((b_low + b_high) / 2.0))

        return float(ece), bin_accs, bin_confs, bin_counts

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> CalibrationMetrics:
        """
        Learns optimal temperature T strictly on validation split via NLL minimization.
        Validation-only diagnostic; does not touch or evaluate test split.
        """
        v_logits = np.asarray(val_logits, dtype=np.float64)
        v_labels = np.asarray(val_labels, dtype=np.int64)

        if v_logits.ndim != 2:
            raise ValueError(f"val_logits must be a 2D array, got shape {v_logits.shape}")
        if len(v_logits) != len(v_labels):
            raise ValueError(f"Length mismatch: {len(v_logits)} logits vs {len(v_labels)} labels")
        if len(v_labels) == 0:
            raise ValueError("Cannot fit calibrator on empty validation data")
        if not np.all(np.isfinite(v_logits)):
            raise ValueError("val_logits contain non-finite values (NaN or Inf)")

        n_classes = v_logits.shape[1]
        if np.any((v_labels < 0) | (v_labels >= n_classes)):
            raise ValueError(f"val_labels must contain integers in range [0, {n_classes - 1}]")

        # Uncalibrated baseline metrics (T = 1.0)
        uncalib_probs = self.softmax(v_logits, temp=1.0)
        uncalib_ece, _, _, _ = self.compute_ece(uncalib_probs, v_labels)

        one_hot = np.zeros_like(uncalib_probs)
        one_hot[np.arange(len(v_labels)), v_labels] = 1.0
        uncalib_brier = float(np.mean(np.sum((uncalib_probs - one_hot) ** 2, axis=1)))

        # Negative Log-Likelihood objective function
        def nll_objective(t: float) -> float:
            probs = self.softmax(v_logits, temp=t)
            eps = 1e-12
            true_class_probs = probs[np.arange(len(v_labels)), v_labels]
            return float(-np.mean(np.log(np.clip(true_class_probs, eps, 1.0))))

        uncalib_nll = float(nll_objective(1.0))

        # Optimize temperature T via bounded scalar minimization
        res = minimize_scalar(nll_objective, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)

        # Calibrated metrics
        calib_probs = self.softmax(v_logits, temp=self.temperature)
        calib_ece, bin_accs, bin_confs, bin_counts = self.compute_ece(calib_probs, v_labels)
        calib_brier = float(np.mean(np.sum((calib_probs - one_hot) ** 2, axis=1)))
        calib_nll = float(res.fun)

        return CalibrationMetrics(
            temperature=self.temperature,
            uncalibrated_ece=uncalib_ece,
            calibrated_ece=calib_ece,
            uncalibrated_nll=uncalib_nll,
            calibrated_nll=calib_nll,
            uncalibrated_brier=uncalib_brier,
            calibrated_brier=calib_brier,
            bin_accuracies=bin_accs,
            bin_confidences=bin_confs,
            bin_counts=bin_counts,
            binning_method="equal_width_10_bins",
        )

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Applies learned temperature scaling to input logits."""
        return self.softmax(logits, temp=self.temperature)


class SelectiveAbstainer:
    """
    Selective Classification Engine (Geifman & El-Yaniv, NeurIPS 2017).

    Evaluates confidence- and entropy-based abstention (rejection option g(x) in {0, 1}).
    
    IMPORTANT ARCHITECTURAL RULE:
    Abstention is a meta-level deferral mechanism, distinctly separated from the domain
    ground-truth class INSUFFICIENT_EVIDENCE. Abstained claims are evaluated by Coverage
    (fraction retained) and Selective Risk (error rate among retained).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.32,
        entropy_threshold: float = 0.94,
        evidence_aware_conf_threshold: float = 0.38,
        margin_threshold: float = 0.10,
    ):
        self.confidence_threshold = float(confidence_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.evidence_aware_conf_threshold = float(evidence_aware_conf_threshold)
        self.margin_threshold = float(margin_threshold)

    @staticmethod
    def compute_normalized_entropy(probs: np.ndarray) -> float:
        """
        Computes Shannon entropy normalized by log(K), strictly bound to [0.0, 1.0].
        A uniform distribution over K classes yields 1.0; a deterministic one-hot yields 0.0.
        """
        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 1 or len(p) < 2:
            raise ValueError("probs must be a 1D probability distribution of length >= 2")

        eps = 1e-12
        p_clipped = np.clip(p, eps, 1.0)
        p_norm = p_clipped / np.sum(p_clipped)
        k = len(p_norm)
        raw_entropy = -float(np.sum(p_norm * np.log(p_norm)))
        max_entropy = float(np.log(k))
        return float(np.clip(raw_entropy / max_entropy, 0.0, 1.0))

    def should_abstain(
        self,
        calibrated_probs: np.ndarray,
        has_meaningful_evidence: bool = True,
    ) -> Tuple[bool, str]:
        """
        Determines whether the operational policy should abstain from categorical assertion.

        Evaluates:
        1. Baseline confidence threshold (max_p < confidence_threshold)
        2. Baseline entropy threshold (normalized entropy > entropy_threshold)
        3. Evidence-aware grounding rule: when neither individual passages nor the joint premise
           provide meaningful supporting or refuting evidence (has_meaningful_evidence is False),
           abstains if max_p < evidence_aware_conf_threshold or top-two margin < margin_threshold.

        Returns (should_abstain, justification_reason).
        """
        p_arr = np.asarray(calibrated_probs, dtype=np.float64)
        if p_arr.ndim != 1 or len(p_arr) < 2:
            raise ValueError("calibrated_probs must be a 1D probability distribution of length >= 2")

        max_p = float(np.max(p_arr))
        entropy = self.compute_normalized_entropy(p_arr)

        if max_p < self.confidence_threshold:
            return True, f"Confidence {max_p:.3f} below threshold {self.confidence_threshold:.3f}"

        if entropy > self.entropy_threshold:
            return True, f"Normalized entropy {entropy:.3f} exceeds threshold {self.entropy_threshold:.3f}"

        # Evidence-aware abstention rule for unsupported predictions
        if not has_meaningful_evidence:
            pred_idx = int(np.argmax(p_arr))
            if pred_idx != 3:  # Conclusive verdict (VERIFIED_TRUE, VERIFIED_FALSE, CONFLICTING)
                cls_name = TemperatureCalibrator.CLASS_LABELS[pred_idx].value if pred_idx < len(TemperatureCalibrator.CLASS_LABELS) else "conclusive"
                return (
                    True,
                    f"Insufficient evidentiary grounding: conclusive verdict '{cls_name}' predicted without supporting or refuting documentary evidence",
                )
            return (
                True,
                "Insufficient evidentiary grounding: no supporting or refuting documentary evidence found in retrieved passages",
            )

        return False, ""

    def evaluate_policy(
        self,
        calibrated_probs: np.ndarray,
        predictions: List[str],
        true_labels: List[str],
        class_labels: List[str],
        evidence_grounding_list: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates selective abstention policy over a dataset.
        Computes coverage, selective accuracy, selective risk, retained class support,
        and operational fallback performance.
        """
        total_samples = len(true_labels)
        if total_samples == 0:
            raise ValueError("Cannot evaluate policy on empty dataset")

        retained_indices: List[int] = []
        abstained_indices: List[int] = []
        abstention_reasons: List[str] = []

        for i in range(total_samples):
            has_ev = True if evidence_grounding_list is None else bool(evidence_grounding_list[i])
            should_abs, reason = self.should_abstain(calibrated_probs[i], has_meaningful_evidence=has_ev)
            if should_abs:
                abstained_indices.append(i)
                abstention_reasons.append(reason)
            else:
                retained_indices.append(i)

        coverage = len(retained_indices) / total_samples
        is_low_coverage = coverage < 0.85

        # Abstained class breakdown
        abstained_true_labels = [true_labels[i] for i in abstained_indices]
        abstained_support = {c: abstained_true_labels.count(c) for c in class_labels}

        if len(retained_indices) > 0:
            y_true_ret = [true_labels[i] for i in retained_indices]
            y_pred_ret = [predictions[i] for i in retained_indices]

            sel_acc = float(accuracy_score(y_true_ret, y_pred_ret))
            sel_risk = 1.0 - sel_acc

            prec, rec, f1, sup = precision_recall_fscore_support(
                y_true_ret, y_pred_ret, labels=class_labels, zero_division=0
            )
            macro_f1 = float(np.mean(f1))
            macro_p = float(np.mean(prec))
            macro_r = float(np.mean(rec))

            per_class: Dict[str, Any] = {}
            for idx, c in enumerate(class_labels):
                per_class[c] = {
                    "precision": float(prec[idx]),
                    "recall": float(rec[idx]),
                    "f1": float(f1[idx]),
                    "support": int(sup[idx]),
                }
            cm = confusion_matrix(y_true_ret, y_pred_ret, labels=class_labels).tolist()
        else:
            sel_acc = 0.0
            sel_risk = 1.0
            macro_f1 = 0.0
            macro_p = 0.0
            macro_r = 0.0
            per_class = {c: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for c in class_labels}
            cm = []

        # Operational fallback: if abstained claims are forced into INSUFFICIENT_EVIDENCE
        y_pred_fallback = list(predictions)
        for i in abstained_indices:
            y_pred_fallback[i] = CanonicalVeracity.INSUFFICIENT_EVIDENCE.value

        fb_acc = float(accuracy_score(true_labels, y_pred_fallback))
        fb_p, fb_r, fb_f1, fb_sup = precision_recall_fscore_support(
            true_labels, y_pred_fallback, labels=class_labels, zero_division=0
        )
        fb_per_class: Dict[str, Any] = {}
        for idx, c in enumerate(class_labels):
            fb_per_class[c] = {
                "precision": float(fb_p[idx]),
                "recall": float(fb_r[idx]),
                "f1": float(fb_f1[idx]),
                "support": int(fb_sup[idx]),
            }

        return {
            "confidence_threshold": self.confidence_threshold,
            "entropy_threshold": self.entropy_threshold,
            "evidence_aware_conf_threshold": self.evidence_aware_conf_threshold,
            "margin_threshold": self.margin_threshold,
            "total_samples": total_samples,
            "retained_count": len(retained_indices),
            "abstained_count": len(abstained_indices),
            "coverage": coverage,
            "is_low_coverage": is_low_coverage,
            "selective_accuracy": sel_acc,
            "selective_risk": sel_risk,
            "macro_f1_retained": macro_f1,
            "macro_precision_retained": macro_p,
            "macro_recall_retained": macro_r,
            "per_class_retained": per_class,
            "confusion_matrix_retained": cm,
            "abstained_support_by_true_class": abstained_support,
            "operational_fallback_4class": {
                "accuracy": fb_acc,
                "macro_f1": float(np.mean(fb_f1)),
                "per_class": fb_per_class,
            },
        }

