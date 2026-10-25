"""
Regression tests for NLI polarity, snippet hierarchy, and reasoning correctness.
Validates that:
1. Snippets with introductory teaser questions or myth headers do not invert refutations into entailment.
2. Legitimate debunking passages produce contradiction stances.
3. Debunked claims aggregated across evidence yield VERIFIED_FALSE rather than CONFLICTING_OR_CHERRYPICKED.
"""
import unittest
import os
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from veritas.data.schema import CanonicalVeracity
from veritas.verification.nli_model import CrossEncoderNLI
from veritas.verification.evidence_aggregator import EvidenceAggregator


class TestNLIPolarityRegressions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize the CrossEncoderNLI model once for regression tests."""
        cls.nli = CrossEncoderNLI(device="cpu")
        cls.aggregator = EvidenceAggregator()

    def test_inquiry_teaser_does_not_invert_contradiction(self):
        """
        Premise starting with a teaser question and myth heading followed by
        explicit refutation must be classified as CONTRADICTION, not ENTAILMENT.
        """
        premise = (
            "Do humans really use just 10% of their brains? "
            "### Myth: Humans Only Use 10% of Their Brains [...] "
            "The myth that humans only use 10% of their brains is FALSE. "
            "Neuroimaging shows humans use virtually 100% of their brain throughout the day."
        )
        hypothesis = "Humans use only 10% of their brains."

        pred = self.nli.predict_pair(premise, hypothesis)
        self.assertEqual(pred["predicted_label"], "CONTRADICTION",
                         f"Expected CONTRADICTION but got {pred['predicted_label']} with scores {pred}")
        self.assertGreater(pred["contradiction"], 0.70)
        self.assertLess(pred["entailment"], 0.20)

    def test_direct_factual_entailment_preserved(self):
        """Direct factual statement supporting hypothesis produces entailment."""
        premise = "The Eiffel Tower is located in Paris, France."
        hypothesis = "The Eiffel Tower is in Paris."

        pred = self.nli.predict_pair(premise, hypothesis)
        self.assertEqual(pred["predicted_label"], "ENTAILMENT")
        self.assertGreater(pred["entailment"], 0.75)

    def test_direct_factual_contradiction_preserved(self):
        """Direct factual statement refuting hypothesis produces contradiction."""
        premise = "The capital of Australia is Canberra, not Sydney or Melbourne."
        hypothesis = "Sydney is the capital of Australia."

        pred = self.nli.predict_pair(premise, hypothesis)
        self.assertEqual(pred["predicted_label"], "CONTRADICTION")
        self.assertGreater(pred["contradiction"], 0.80)

    def test_brain_myth_end_to_end_aggregation(self):
        """
        The 10% brain myth claim verified against debunking evidence must yield
        VERIFIED_FALSE, not CONFLICTING_OR_CHERRYPICKED.
        """
        hypothesis = "Humans use only 10% of their brains."
        passages = [
            {
                "doc_id": "steam_ahead",
                "text": "Do humans really use just 10% of their brains? [...] The myth that humans only use 10% of their brains is FALSE. Neurological research confirms we use virtually all parts of the brain.",
            },
            {
                "doc_id": "mit_mcgovern",
                "text": "A common myth is that we only use 10% of our brains. This is entirely untrue. Even in sleep, all areas of the brain show activity.",
            },
        ]

        pairs = [(p["text"], hypothesis) for p in passages]
        preds = self.nli.predict_batch(pairs)

        verdict = self.aggregator.aggregate(passages, preds)

        self.assertEqual(verdict.canonical_verdict, CanonicalVeracity.VERIFIED_FALSE,
                         f"Expected VERIFIED_FALSE, got {verdict.canonical_verdict}")
        self.assertFalse(verdict.has_cross_passage_disparity)
        self.assertGreater(verdict.contradiction_mass, 0.70)


if __name__ == "__main__":
    unittest.main()
