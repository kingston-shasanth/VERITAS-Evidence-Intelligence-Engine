"""
Lightweight Unit Tests for Phase 3.1 Benchmark Safety and Resumption.

Tests:
1. is_completed_result validation (complete vs incomplete vs corrupted).
2. Atomic saving and safe loading of comparison results.
3. Command-line argument parsing and model filtering.
4. Resumption and skipping of completed models with mocked NLI inference.
5. Recovery and re-evaluation when encountering incomplete/interrupted cached entries.

NOTE: No real models are loaded or downloaded in this test suite.
"""
import unittest
import os
import sys
import tempfile
import json
from typing import List, Tuple, Dict

# Ensure src and scripts are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from veritas.data.schema import CanonicalClaim, CanonicalVeracity, DatasetSource, EvidenceItem
from diagnose_and_improve import (
    is_completed_result,
    load_comparison_results,
    save_comparison_results,
    parse_args,
    get_device,
    main as diagnose_main,
    CANDIDATE_MODELS,
)


class MockNLI:
    """Mock NLI engine that returns deterministic probabilities without loading weights."""
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        results = []
        for p, h in pairs:
            results.append({
                "contradiction": 0.10,
                "entailment": 0.80,
                "neutral": 0.10,
            })
        return results


def make_mock_claims() -> List[CanonicalClaim]:
    """Creates a balanced minimal list of CanonicalClaim objects for testing."""
    claims = []
    classes = [
        CanonicalVeracity.VERIFIED_TRUE,
        CanonicalVeracity.VERIFIED_FALSE,
        CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED,
        CanonicalVeracity.INSUFFICIENT_EVIDENCE,
    ]
    # 2 of each class to allow 50% stratified splitting
    for idx, label in enumerate(classes * 2):
        ev = EvidenceItem(
            source_id=f"doc_{idx}",
            text=f"Mock evidence text supporting or refuting fact {idx}.",
            metadata={"question": f"Is claim {idx} valid?"},
        )
        claims.append(
            CanonicalClaim(
                claim_id=f"mock_claim_{idx}",
                claim_text=f"Mock claim statement number {idx}.",
                raw_label=label.value,
                canonical_label=label,
                source_dataset=DatasetSource.AVERITEC,
                evidence=[ev],
                metadata={"justification": f"Mock justification for claim {idx}"},
            )
        )
    return claims


class TestDiagnoseCheckpointing(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_file = os.path.join(self.temp_dir.name, "test_comparison.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Validation of Completed Results
    # -------------------------------------------------------------------------
    def test_is_completed_result_valid(self):
        valid_entry = {
            "status": "completed",
            "model_name": "test-model",
            "raw": {
                "accuracy": 0.75,
                "macro_f1": 0.70,
                "per_class": {},
                "confusion_matrix": [],
            },
            "contextualized": {
                "accuracy": 0.80,
                "macro_f1": 0.76,
                "per_class": {},
                "confusion_matrix": [],
            },
        }
        self.assertTrue(is_completed_result(valid_entry))

    def test_is_completed_result_incomplete(self):
        # Missing status
        entry1 = {"raw": {}, "contextualized": {}}
        self.assertFalse(is_completed_result(entry1))

        # Status not completed (e.g., interrupted mid-run)
        entry2 = {"status": "in_progress", "raw": {"accuracy": 0.5, "macro_f1": 0.5, "per_class": {}, "confusion_matrix": []}, "contextualized": {}}
        self.assertFalse(is_completed_result(entry2))

        # Missing contextualized pass
        entry3 = {
            "status": "completed",
            "raw": {"accuracy": 0.5, "macro_f1": 0.5, "per_class": {}, "confusion_matrix": []},
            "contextualized": None,
        }
        self.assertFalse(is_completed_result(entry3))

        # Missing required metric keys
        entry4 = {
            "status": "completed",
            "raw": {"accuracy": 0.5},
            "contextualized": {"accuracy": 0.6},
        }
        self.assertFalse(is_completed_result(entry4))

        # Non-dict input
        self.assertFalse(is_completed_result("corrupted_string"))
        self.assertFalse(is_completed_result(None))

    # -------------------------------------------------------------------------
    # 2. Atomic Saving and Safe Loading
    # -------------------------------------------------------------------------
    def test_atomic_save_and_load(self):
        data = {"model_a": {"status": "completed", "score": 1.0}}
        save_comparison_results(data, self.output_file)

        self.assertTrue(os.path.exists(self.output_file))
        loaded = load_comparison_results(self.output_file)
        self.assertEqual(loaded, data)

        # Ensure no leftover temporary files in directory
        temp_files = [f for f in os.listdir(self.temp_dir.name) if f.startswith(".tmp_")]
        self.assertEqual(len(temp_files), 0)

    def test_load_corrupted_file_handles_gracefully(self):
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("{corrupted_json: incomplete...")

        # Should log warning and return empty dict without crashing
        loaded = load_comparison_results(self.output_file)
        self.assertEqual(loaded, {})

    # -------------------------------------------------------------------------
    # 3. CLI Argument Parsing and Device Selection
    # -------------------------------------------------------------------------
    def test_cli_argument_parsing(self):
        args = parse_args(["--model", "deberta-v3-xsmall", "--device", "cpu", "--force"])
        self.assertEqual(args.model, "deberta-v3-xsmall")
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.force)

    def test_device_fallback(self):
        # Explicit override
        self.assertEqual(get_device("cpu"), "cpu")
        self.assertEqual(get_device("cuda:0"), "cuda:0")

    # -------------------------------------------------------------------------
    # 4. Mocked Execution: Checkpointing and Resuming
    # -------------------------------------------------------------------------
    def test_checkpoint_saving_and_skipping(self):
        claims = make_mock_claims()
        load_count = 0

        def mock_factory(name, dev):
            nonlocal load_count
            load_count += 1
            return MockNLI(name, dev)

        target_model = "cross-encoder/nli-deberta-v3-xsmall"

        # First run: evaluate selected model
        res1 = diagnose_main(
            cli_args=["--output-file", self.output_file, "--model", target_model, "--device", "cpu"],
            nli_factory=mock_factory,
            claims_override=claims,
        )

        self.assertEqual(load_count, 1)
        self.assertIn(target_model, res1)
        self.assertTrue(is_completed_result(res1[target_model]))

        # Verify saved to disk
        disk_res = load_comparison_results(self.output_file)
        self.assertIn(target_model, disk_res)
        self.assertEqual(disk_res[target_model]["status"], "completed")

        # Second run: should detect existing completed result and SKIP mock loading
        res2 = diagnose_main(
            cli_args=["--output-file", self.output_file, "--model", target_model, "--device", "cpu"],
            nli_factory=mock_factory,
            claims_override=claims,
        )

        # load_count should still be 1 because it skipped loading
        self.assertEqual(load_count, 1)
        self.assertIn(target_model, res2)

    # -------------------------------------------------------------------------
    # 5. Incomplete Results Are Re-evaluated
    # -------------------------------------------------------------------------
    def test_incomplete_result_is_reevaluated(self):
        claims = make_mock_claims()
        target_model = "cross-encoder/nli-distilroberta-base"

        # Pre-seed file with an interrupted/incomplete result
        incomplete_data = {
            target_model: {
                "status": "interrupted",
                "model_name": target_model,
                "raw": {"accuracy": 0.5},  # incomplete
            }
        }
        save_comparison_results(incomplete_data, self.output_file)

        load_count = 0
        def mock_factory(name, dev):
            nonlocal load_count
            load_count += 1
            return MockNLI(name, dev)

        # Run main: it must detect incomplete state and re-evaluate
        res = diagnose_main(
            cli_args=["--output-file", self.output_file, "--model", target_model, "--device", "cpu"],
            nli_factory=mock_factory,
            claims_override=claims,
        )

        self.assertEqual(load_count, 1)
        self.assertEqual(res[target_model]["status"], "completed")
        self.assertTrue(is_completed_result(res[target_model]))


if __name__ == "__main__":
    unittest.main()
