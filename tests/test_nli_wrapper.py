"""
Unit Tests for CrossEncoderNLI wrapper (Step 3.2.1):
Tests dynamic label resolution, device selection/fallback, prediction semantics,
and edge-case handling using fully mocked components (zero model download/overhead).
"""
import unittest
import os
import sys
from types import SimpleNamespace
from typing import List, Dict, Any, Tuple
import torch

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from veritas.verification.nli_model import (
    CrossEncoderNLI,
    resolve_device,
    resolve_nli_label_mapping,
    DEFAULT_MODEL,
    LEGACY_MODEL,
)


class MockTokenizer:
    """Mock HuggingFace tokenizer producing deterministic tensors."""
    def __call__(self, premises: List[str], hypotheses: List[str], **kwargs):
        batch_size = len(premises)
        return {
            "input_ids": torch.zeros((batch_size, 16), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 16), dtype=torch.long),
        }


class MockClassificationModel:
    """Mock HuggingFace AutoModelForSequenceClassification with custom config and logits."""
    def __init__(self, config: Any, logits_per_item: List[List[float]]):
        self.config = config
        self.logits_per_item = logits_per_item
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device
        return self

    def eval(self):
        return self

    def __call__(self, **inputs):
        batch_size = inputs["input_ids"].shape[0]
        # Tile or slice logits to match batch size
        if len(self.logits_per_item) == 1:
            raw_logits = self.logits_per_item * batch_size
        else:
            raw_logits = self.logits_per_item[:batch_size]
        logits_tensor = torch.tensor(raw_logits, dtype=torch.float32, device=self.device)
        return SimpleNamespace(logits=logits_tensor)


class TestNLIWrapper(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. Label Mapping Resolution Tests
    # -------------------------------------------------------------------------
    def test_resolve_moritzlaurer_mapping(self):
        """Validates correct resolution for MoritzLaurer label indices: {E:0, N:1, C:2}."""
        config = SimpleNamespace(label2id={"contradiction": 2, "entailment": 0, "neutral": 1})
        c_idx, e_idx, n_idx = resolve_nli_label_mapping(config, "MoritzLaurer-Test")
        self.assertEqual(e_idx, 0)
        self.assertEqual(n_idx, 1)
        self.assertEqual(c_idx, 2)

    def test_resolve_legacy_deberta_mapping(self):
        """Validates correct resolution for legacy cross-encoder indices: {C:0, E:1, N:2}."""
        config = SimpleNamespace(label2id={"contradiction": 0, "entailment": 1, "neutral": 2})
        c_idx, e_idx, n_idx = resolve_nli_label_mapping(config, "Legacy-DeBERTa-Test")
        self.assertEqual(c_idx, 0)
        self.assertEqual(e_idx, 1)
        self.assertEqual(n_idx, 2)

    def test_resolve_from_id2label_dict(self):
        """Validates resolution when model defines id2label instead of label2id."""
        config = SimpleNamespace(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})
        c_idx, e_idx, n_idx = resolve_nli_label_mapping(config, "id2label-Test")
        self.assertEqual(e_idx, 0)
        self.assertEqual(n_idx, 1)
        self.assertEqual(c_idx, 2)

    def test_resolve_case_insensitive_and_whitespace(self):
        """Validates case-insensitivity and whitespace resilience."""
        config = SimpleNamespace(label2id={"  CONTRADICTION  ": 1, " ENTAILMENT": 2, "Neutral": 0})
        c_idx, e_idx, n_idx = resolve_nli_label_mapping(config, "Case-Test")
        self.assertEqual(n_idx, 0)
        self.assertEqual(c_idx, 1)
        self.assertEqual(e_idx, 2)

    def test_rejects_generic_labels(self):
        """Rejects generic unmapped labels such as LABEL_0, LABEL_1, LABEL_2 with a ValueError."""
        config = SimpleNamespace(label2id={"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2})
        with self.assertRaises(ValueError) as ctx:
            resolve_nli_label_mapping(config, "Generic-Model")
        self.assertIn("generic classification labels", str(ctx.exception))

    def test_rejects_missing_mapping(self):
        """Rejects configs that lack both label2id and id2label."""
        config = SimpleNamespace()
        with self.assertRaises(ValueError) as ctx:
            resolve_nli_label_mapping(config, "Empty-Config")
        self.assertIn("neither 'label2id' nor 'id2label'", str(ctx.exception))

    def test_rejects_incomplete_stances(self):
        """Rejects configs that only provide a subset of stances."""
        config = SimpleNamespace(label2id={"entailment": 0, "neutral": 1})
        with self.assertRaises(ValueError) as ctx:
            resolve_nli_label_mapping(config, "Incomplete-Config")
        self.assertIn("Missing stance(s): ['contradiction']", str(ctx.exception))

    def test_rejects_duplicate_indices(self):
        """Rejects configs where multiple stances point to the same index."""
        config = SimpleNamespace(label2id={"entailment": 0, "neutral": 0, "contradiction": 1})
        with self.assertRaises(ValueError) as ctx:
            resolve_nli_label_mapping(config, "Duplicate-Indices")
        self.assertIn("non-unique stance label indices", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 2. Device Selection & Fallback Tests
    # -------------------------------------------------------------------------
    def test_resolve_device_explicit(self):
        """Honors explicit device selection."""
        self.assertEqual(resolve_device("cpu"), "cpu")
        self.assertEqual(resolve_device("CPU"), "cpu")
        self.assertEqual(resolve_device("cuda:0"), "cuda:0")

    def test_resolve_device_auto_detection(self):
        """Auto-detects CUDA if available, falls back to CPU otherwise."""
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(resolve_device(None), expected)

    # -------------------------------------------------------------------------
    # 3. Prediction Semantics with Mocked MoritzLaurer Configuration
    # -------------------------------------------------------------------------
    def test_moritzlaurer_prediction_semantics(self):
        """
        Validates that when MoritzLaurer id2label {0:E, 1:N, 2:C} is loaded,
        logits with highest value at index 0 produce ENTAILMENT, NOT CONTRADICTION.
        """
        config = SimpleNamespace(label2id={"entailment": 0, "neutral": 1, "contradiction": 2})
        tokenizer = MockTokenizer()

        # Logits: High entailment (index 0), low neutral, very low contradiction
        logits = [[5.0, 0.0, -5.0]]
        model = MockClassificationModel(config=config, logits_per_item=logits)

        nli = CrossEncoderNLI(
            model_name="Mock-MoritzLaurer",
            device="cpu",
            tokenizer=tokenizer,
            model=model,
        )

        res = nli.predict_pair("Water freezes at 0C.", "Water becomes ice at 0C.")
        self.assertEqual(res["predicted_label"], "ENTAILMENT")
        self.assertGreater(res["entailment"], 0.95)
        self.assertLess(res["contradiction"], 0.01)

        # Logits: High contradiction (index 2)
        model.logits_per_item = [[-5.0, 0.0, 5.0]]
        res_contra = nli.predict_pair("Water freezes at 0C.", "Water boils at 0C.")
        self.assertEqual(res_contra["predicted_label"], "CONTRADICTION")
        self.assertGreater(res_contra["contradiction"], 0.95)
        self.assertLess(res_contra["entailment"], 0.01)

    # -------------------------------------------------------------------------
    # 4. Legacy Model Backward Compatibility
    # -------------------------------------------------------------------------
    def test_legacy_deberta_prediction_semantics(self):
        """
        Validates that legacy DeBERTa id2label {0:C, 1:E, 2:N} continues to work properly.
        """
        config = SimpleNamespace(label2id={"contradiction": 0, "entailment": 1, "neutral": 2})
        tokenizer = MockTokenizer()

        # Logits: High contradiction at index 0
        logits = [[5.0, -5.0, 0.0]]
        model = MockClassificationModel(config=config, logits_per_item=logits)

        nli = CrossEncoderNLI(
            model_name="Mock-LegacyDeBERTa",
            device="cpu",
            tokenizer=tokenizer,
            model=model,
        )

        res = nli.predict_pair("The Earth is flat.", "The Earth is round.")
        self.assertEqual(res["predicted_label"], "CONTRADICTION")
        self.assertGreater(res["contradiction"], 0.95)
        self.assertLess(res["entailment"], 0.01)

    # -------------------------------------------------------------------------
    # 5. Empty / Whitespace Premise Handling
    # -------------------------------------------------------------------------
    def test_empty_premise_handling(self):
        """Validates that empty or whitespace premises safely return neutral fallback."""
        config = SimpleNamespace(label2id={"entailment": 0, "neutral": 1, "contradiction": 2})
        tokenizer = MockTokenizer()
        model = MockClassificationModel(config=config, logits_per_item=[[0.0, 0.0, 0.0]])

        nli = CrossEncoderNLI(
            model_name="Mock-EmptyPremise",
            device="cpu",
            tokenizer=tokenizer,
            model=model,
        )

        res = nli.predict_pair("", "Some claim hypothesis")
        self.assertEqual(res["predicted_label"], "NEUTRAL")
        self.assertEqual(res["neutral"], 1.0)
        self.assertEqual(res["contradiction"], 0.0)
        self.assertEqual(res["entailment"], 0.0)


if __name__ == "__main__":
    unittest.main()
