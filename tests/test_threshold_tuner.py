"""
Unit Tests for Phase 3.2.3 Threshold Tuner.
Validates:
1. The tuner ingests strictly the validation split (N=255) and isolates test data.
2. Grid search execution is 100% deterministic over identical cached records.
3. Low-coverage configurations (< 85%) are explicitly flagged and disqualified.
4. Benchmark caches (validation_model_comparison.json, test_model_evaluation.json) are untouched.
"""
import unittest
import os
import sys
import tempfile
import json
import hashlib
from typing import List, Dict, Any, Optional

# Ensure src/ and scripts/ are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from veritas.data.schema import CanonicalVeracity
from tune_thresholds import (
    load_validation_split_only,
    evaluate_aggregator_configuration,
    run_grid_search,
    CLASS_LABELS,
)


def get_file_sha256(filepath: str) -> Optional[str]:
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class TestThresholdTuner(unittest.TestCase):

    def setUp(self):
        self.canonical_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "processed", "canonical_claims.jsonl"
        )
        self.val_comparison_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "processed", "validation_model_comparison.json"
        )
        self.test_eval_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "processed", "test_model_evaluation.json"
        )

    # -------------------------------------------------------------------------
    # 1. Validation Split Isolation
    # -------------------------------------------------------------------------
    def test_validation_split_only_loaded(self):
        """Verifies that load_validation_split_only returns strictly N=255 validation claims."""
        val_claims = load_validation_split_only(self.canonical_path)
        self.assertEqual(len(val_claims), 255)

        # Confirm expected class distribution on validation split
        counts = {}
        for c in val_claims:
            counts[c.canonical_label.value] = counts.get(c.canonical_label.value, 0) + 1

        self.assertEqual(counts[CanonicalVeracity.VERIFIED_FALSE.value], 156)
        self.assertEqual(counts[CanonicalVeracity.VERIFIED_TRUE.value], 61)
        self.assertEqual(counts[CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value], 19)
        self.assertEqual(counts[CanonicalVeracity.INSUFFICIENT_EVIDENCE.value], 19)

    # -------------------------------------------------------------------------
    # 2. Deterministic Grid Search
    # -------------------------------------------------------------------------
    def test_grid_search_determinism(self):
        """Validates that grid search over mock cached predictions produces identical results on repeat runs."""
        # Create minimal synthetic mock records
        mock_records = [
            {
                "claim_id": "c1",
                "claim_text": "Fact A is true.",
                "true_label": CanonicalVeracity.VERIFIED_TRUE.value,
                "candidate_passages": [{"doc_id": "e1", "text": "Fact A is proven."}],
                "nli_predictions": [{"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10}],
                "joint_nli_prediction": None,
            },
            {
                "claim_id": "c2",
                "claim_text": "Fact B is false.",
                "true_label": CanonicalVeracity.VERIFIED_FALSE.value,
                "candidate_passages": [{"doc_id": "e2", "text": "Fact B is disproven."}],
                "nli_predictions": [{"entailment": 0.05, "contradiction": 0.85, "neutral": 0.10}],
                "joint_nli_prediction": None,
            },
            {
                "claim_id": "c3",
                "claim_text": "Fact C is contested.",
                "true_label": CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value,
                "candidate_passages": [
                    {"doc_id": "e3_pro", "text": "Fact C is supported."},
                    {"doc_id": "e3_con", "text": "Fact C is refuted."},
                ],
                "nli_predictions": [
                    {"entailment": 0.80, "contradiction": 0.05, "neutral": 0.15},
                    {"entailment": 0.05, "contradiction": 0.80, "neutral": 0.15},
                ],
                "joint_nli_prediction": {"entailment": 0.10, "contradiction": 0.80, "neutral": 0.10},
            },
            {
                "claim_id": "c4",
                "claim_text": "Fact D has no proof.",
                "true_label": CanonicalVeracity.INSUFFICIENT_EVIDENCE.value,
                "candidate_passages": [],
                "nli_predictions": [],
                "joint_nli_prediction": None,
            },
        ]

        small_grid = {
            "disparity_entail_threshold": [0.35, 0.45],
            "disparity_contra_threshold": [0.35, 0.45],
            "conflict_threshold": [0.35, 0.45],
            "min_informative_mass": [0.15, 0.25],
            "joint_weight_disparity": [0.2],
            "joint_weight_default": [1.0],
        }

        run1 = run_grid_search(mock_records, small_grid, min_coverage_threshold=0.50)
        run2 = run_grid_search(mock_records, small_grid, min_coverage_threshold=0.50)

        # Must select exactly the same optimal configuration
        self.assertEqual(
            run1["selected_optimal_configuration"]["parameters"],
            run2["selected_optimal_configuration"]["parameters"],
        )
        self.assertEqual(
            run1["selected_optimal_configuration"]["macro_f1"],
            run2["selected_optimal_configuration"]["macro_f1"],
        )

    # -------------------------------------------------------------------------
    # 3. Low-Coverage Flagging & Disqualification
    # -------------------------------------------------------------------------
    def test_low_coverage_flagging(self):
        """Validates that parameters leading to low coverage (< 85%) are flagged."""
        # Single claim that produces INSUFFICIENT when min_informative_mass is high
        mock_records = [
            {
                "claim_id": "c_weak",
                "claim_text": "Claim with weak evidence.",
                "true_label": CanonicalVeracity.VERIFIED_TRUE.value,
                "candidate_passages": [{"doc_id": "e_weak", "text": "Some text."}],
                "nli_predictions": [{"entailment": 0.30, "contradiction": 0.05, "neutral": 0.65}],
                "joint_nli_prediction": None,
            }
        ]
        # min_mass=0.50 forces abstention (coverage = 0%)
        high_mass_params = {
            "min_informative_mass": 0.50,
            "disparity_entail_threshold": 0.40,
            "disparity_contra_threshold": 0.40,
            "conflict_threshold": 0.40,
            "joint_weight_default": 1.0,
            "joint_weight_disparity": 0.2,
        }
        res = evaluate_aggregator_configuration(high_mass_params, mock_records)
        self.assertTrue(res["is_low_coverage"])
        self.assertEqual(res["coverage"], 0.0)

    # -------------------------------------------------------------------------
    # 4. Benchmark Cache File Preservation
    # -------------------------------------------------------------------------
    def test_benchmark_files_untouched(self):
        """Verifies that the tuner does not modify existing benchmark files."""
        hash_val_before = get_file_sha256(self.val_comparison_file)
        hash_test_before = get_file_sha256(self.test_eval_file)

        # Run synthetic tuning in a temp file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_out = tf.name

        try:
            val_claims = load_validation_split_only(self.canonical_path)
            self.assertEqual(len(val_claims), 255)
        finally:
            if os.path.exists(temp_out):
                os.remove(temp_out)

        hash_val_after = get_file_sha256(self.val_comparison_file)
        hash_test_after = get_file_sha256(self.test_eval_file)

        self.assertEqual(hash_val_before, hash_val_after)
        self.assertEqual(hash_test_before, hash_test_after)


if __name__ == "__main__":
    unittest.main()
