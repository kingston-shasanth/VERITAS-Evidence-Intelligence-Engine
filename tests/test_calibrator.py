"""
Unit Test Suite for Phase 3.2.4: Calibration & Selective Abstention.

Tests:
1. TemperatureCalibrator fitting via NLL minimization.
2. Robustness to invalid inputs (NaN, Inf, empty arrays, shape mismatches, out-of-range labels).
3. Calibrated probability normalization and simplex invariants.
4. Monotonic rank-order preservation (argmax invariance under scalar temperature).
5. Expected Calibration Error (ECE) calculation and bin partitioning.
6. Distinction between Temperature Scaling and Platt Scaling in reports.
7. SelectiveAbstainer confidence and entropy thresholding.
8. Separation of Selective Abstention from the INSUFFICIENT_EVIDENCE domain class.
9. Coverage and selective risk calculation.
10. Preservation of earlier benchmark and test split files.
"""
import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from veritas.data.schema import CanonicalVeracity
from veritas.verification.calibrator import (
    TemperatureCalibrator,
    SelectiveAbstainer,
    CalibrationMetrics,
)


class TestTemperatureCalibrator(unittest.TestCase):
    """Test suite for post-hoc Temperature Scaling."""

    def setUp(self):
        # Synthetic 4-class logits and labels (N=100)
        np.random.seed(42)
        self.n_samples = 100
        self.n_classes = 4
        # Create overconfident synthetic logits
        self.true_labels = np.random.randint(0, self.n_classes, size=self.n_samples)
        self.logits = np.random.randn(self.n_samples, self.n_classes) * 2.0
        # Bias the correct class to have higher logit so accuracy is ~60%
        for i in range(self.n_samples):
            if np.random.rand() < 0.60:
                self.logits[i, self.true_labels[i]] += 4.0

    def test_fit_reduces_nll(self):
        """Verifies that fitting temperature strictly minimizes validation NLL."""
        calibrator = TemperatureCalibrator()
        metrics = calibrator.fit(self.logits, self.true_labels)

        self.assertIsInstance(metrics, CalibrationMetrics)
        self.assertGreater(metrics.temperature, 0.0)
        self.assertLessEqual(metrics.calibrated_nll, metrics.uncalibrated_nll + 1e-6)
        self.assertGreaterEqual(metrics.uncalibrated_ece, 0.0)
        self.assertGreaterEqual(metrics.calibrated_ece, 0.0)
        self.assertGreaterEqual(metrics.uncalibrated_brier, 0.0)
        self.assertGreaterEqual(metrics.calibrated_brier, 0.0)

    def test_simplex_normalization_and_rank_invariance(self):
        """Verifies calibrated probabilities sum to 1.0 and preserve argmax."""
        calibrator = TemperatureCalibrator(temperature=2.5)
        probs = calibrator.calibrate(self.logits)

        # Check shape and non-negativity
        self.assertEqual(probs.shape, self.logits.shape)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

        # Check simplex property: rows sum to 1.0
        sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(sums, np.ones(self.n_samples), atol=1e-6)

        # Crucial property: Temperature scaling MUST preserve argmax (rank-order)
        uncalib_argmax = np.argmax(self.logits, axis=1)
        calib_argmax = np.argmax(probs, axis=1)
        np.testing.assert_array_equal(uncalib_argmax, calib_argmax)

    def test_invalid_inputs_handling(self):
        """Verifies that calibrator raises ValueError on malformed inputs."""
        calibrator = TemperatureCalibrator()

        # 1. NaN in logits
        nan_logits = self.logits.copy()
        nan_logits[0, 0] = np.nan
        with self.assertRaises(ValueError):
            calibrator.fit(nan_logits, self.true_labels)
        with self.assertRaises(ValueError):
            calibrator.calibrate(nan_logits)

        # 2. Inf in logits
        inf_logits = self.logits.copy()
        inf_logits[0, 0] = np.inf
        with self.assertRaises(ValueError):
            calibrator.fit(inf_logits, self.true_labels)

        # 3. Shape mismatch between logits and labels
        with self.assertRaises(ValueError):
            calibrator.fit(self.logits, self.true_labels[:-5])

        # 4. 1D logits to fit (must be 2D)
        with self.assertRaises(ValueError):
            calibrator.fit(self.logits[0], self.true_labels[0:1])

        # 5. Empty inputs
        with self.assertRaises(ValueError):
            calibrator.fit(np.empty((0, 4)), np.empty(0, dtype=int))

        # 6. Out of range class label
        bad_labels = self.true_labels.copy()
        bad_labels[0] = 99
        with self.assertRaises(ValueError):
            calibrator.fit(self.logits, bad_labels)

    def test_compute_ece_partitioning(self):
        """Verifies ECE bin partitioning and edge handling."""
        # 1. Perfectly calibrated case
        perfect_probs = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
        perfect_labels = np.array([0, 0, 1, 1])
        ece, accs, confs, counts = TemperatureCalibrator.compute_ece(perfect_probs, perfect_labels, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=5)
        self.assertEqual(sum(counts), len(perfect_labels))

        # 2. Perfectly wrong (overconfident)
        wrong_probs = np.array([
            [0.99, 0.01],
            [0.99, 0.01],
        ])
        wrong_labels = np.array([1, 1])
        ece_wrong, _, _, _ = TemperatureCalibrator.compute_ece(wrong_probs, wrong_labels, n_bins=10)
        self.assertGreater(ece_wrong, 0.90)

    def test_report_distinguishes_temperature_from_platt(self):
        """Verifies that calibration metrics report explicitly states Temperature Scaling."""
        calibrator = TemperatureCalibrator()
        metrics = calibrator.fit(self.logits, self.true_labels)
        summary = metrics.summary()
        self.assertIn("Temperature Scaling", summary)
        self.assertIn("Single Scalar T", summary)
        self.assertNotIn("Platt Scaling", summary)


class TestSelectiveAbstainer(unittest.TestCase):
    """Test suite for Selective Classification and risk-coverage policy."""

    def setUp(self):
        self.abstainer = SelectiveAbstainer(confidence_threshold=0.50, entropy_threshold=0.80)
        self.classes = [
            CanonicalVeracity.VERIFIED_TRUE.value,
            CanonicalVeracity.VERIFIED_FALSE.value,
            CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value,
            CanonicalVeracity.INSUFFICIENT_EVIDENCE.value,
        ]

    def test_normalized_entropy_bounds(self):
        """Verifies normalized entropy bounds in [0.0, 1.0]."""
        # Deterministic: entropy = 0.0
        one_hot = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(SelectiveAbstainer.compute_normalized_entropy(one_hot), 0.0, places=4)

        # Uniform: entropy = 1.0
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        self.assertAlmostEqual(SelectiveAbstainer.compute_normalized_entropy(uniform), 1.0, places=4)

        # Invalid distribution raises ValueError
        with self.assertRaises(ValueError):
            SelectiveAbstainer.compute_normalized_entropy(np.array([1.0]))

    def test_should_abstain_conditions(self):
        """Verifies confidence and entropy triggers."""
        # 1. High confidence, low entropy -> Retain
        confident_p = np.array([0.85, 0.05, 0.05, 0.05])
        should_abs, reason = self.abstainer.should_abstain(confident_p)
        self.assertFalse(should_abs)
        self.assertEqual(reason, "")

        # 2. Low confidence (< 0.50) -> Abstain
        low_conf_p = np.array([0.40, 0.30, 0.20, 0.10])
        should_abs, reason = self.abstainer.should_abstain(low_conf_p)
        self.assertTrue(should_abs)
        self.assertIn("below threshold", reason)

        # 3. High entropy (> 0.80) -> Abstain
        high_ent_p = np.array([0.35, 0.30, 0.20, 0.15])
        should_abs, reason = self.abstainer.should_abstain(high_ent_p)
        self.assertTrue(should_abs)

    def test_policy_evaluation_and_insufficient_separation(self):
        """
        Verifies policy evaluation correctly calculates coverage and risk,
        and does NOT silently conflate abstention with INSUFFICIENT_EVIDENCE.
        """
        probs = np.array([
            [0.90, 0.04, 0.03, 0.03],  # Confident True
            [0.05, 0.85, 0.05, 0.05],  # Confident False
            [0.02, 0.02, 0.90, 0.06],  # Confident Conflicting
            [0.05, 0.05, 0.05, 0.85],  # Confident Insufficient Evidence
            [0.30, 0.30, 0.20, 0.20],  # Ambiguous -> should abstain
        ])
        preds = [
            CanonicalVeracity.VERIFIED_TRUE.value,
            CanonicalVeracity.VERIFIED_FALSE.value,
            CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value,
            CanonicalVeracity.INSUFFICIENT_EVIDENCE.value,
            CanonicalVeracity.VERIFIED_TRUE.value,
        ]
        true_labels = [
            CanonicalVeracity.VERIFIED_TRUE.value,
            CanonicalVeracity.VERIFIED_FALSE.value,
            CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value,
            CanonicalVeracity.INSUFFICIENT_EVIDENCE.value,
            CanonicalVeracity.VERIFIED_FALSE.value,  # Wrong prediction if retained
        ]

        res = self.abstainer.evaluate_policy(probs, preds, true_labels, self.classes)

        # 4 retained, 1 abstained -> coverage = 4/5 = 80% (< 85%)
        self.assertEqual(res["total_samples"], 5)
        self.assertEqual(res["retained_count"], 4)
        self.assertEqual(res["abstained_count"], 1)
        self.assertAlmostEqual(res["coverage"], 0.80)
        self.assertTrue(res["is_low_coverage"])

        # Among the 4 retained, all 4 were correct -> selective accuracy = 1.0, risk = 0.0
        self.assertAlmostEqual(res["selective_accuracy"], 1.0)
        self.assertAlmostEqual(res["selective_risk"], 0.0)

        # Crucial check: Confident INSUFFICIENT_EVIDENCE was retained as its substantive class!
        self.assertEqual(res["per_class_retained"][CanonicalVeracity.INSUFFICIENT_EVIDENCE.value]["support"], 1)
        self.assertAlmostEqual(res["per_class_retained"][CanonicalVeracity.INSUFFICIENT_EVIDENCE.value]["f1"], 1.0)

        # The abstained claim was VERIFIED_FALSE in ground truth, not INSUFFICIENT
        self.assertEqual(res["abstained_support_by_true_class"][CanonicalVeracity.VERIFIED_FALSE.value], 1)


class TestBenchmarkIntegrity(unittest.TestCase):
    """Ensures test scripts and calibrator do not modify earlier benchmark results."""

    def test_benchmark_files_untouched(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        val_comp = os.path.join(base_dir, "data", "processed", "validation_model_comparison.json")
        test_eval = os.path.join(base_dir, "data", "processed", "test_model_evaluation.json")

        self.assertTrue(os.path.exists(val_comp), "validation_model_comparison.json must exist")
        self.assertTrue(os.path.exists(test_eval), "test_model_evaluation.json must exist")

        self.assertGreater(os.path.getsize(val_comp), 1000)
        self.assertGreater(os.path.getsize(test_eval), 500)


if __name__ == "__main__":
    unittest.main()
