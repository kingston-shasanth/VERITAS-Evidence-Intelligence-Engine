"""
Unit Tests for EvidenceAggregator & Cross-Passage Stance Disparity (Step 3.2.2).
Validates:
1. One passage supporting and another contradicting the claim (genuine disparity -> CONFLICTING).
2. Multiple passages all supporting the claim (unanimous support -> VERIFIED_TRUE).
3. Multiple passages all contradicting the claim (unanimous refutation -> VERIFIED_FALSE).
4. Weak opposing noise that must not trigger conflict (noise suppression -> TRUE/FALSE).
5. A single passage with mixed scores that must not count as cross-passage conflict (single source != conflict).
6. Duplicate passages discounted by Jaccard filter.
7. Empty evidence returns INSUFFICIENT_EVIDENCE.
8. Joint-premise de-weighting under stance disparity.
"""
import unittest
import os
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from veritas.data.schema import CanonicalVeracity
from veritas.verification.evidence_aggregator import EvidenceAggregator, AggregatedEvidenceVerdict


class TestEvidenceAggregator(unittest.TestCase):

    def setUp(self):
        self.aggregator = EvidenceAggregator(
            jaccard_dedup_threshold=0.75,
            neutral_threshold=0.70,
            min_informative_mass=0.20,
            conflict_threshold=0.40,
            disparity_entail_threshold=0.40,
            disparity_contra_threshold=0.40,
            joint_weight_default=1.0,
            joint_weight_disparity=0.2,
        )

    # -------------------------------------------------------------------------
    # 1. Genuine Cross-Passage Stance Disparity (Conflict Detection)
    # -------------------------------------------------------------------------
    def test_genuine_cross_passage_conflict(self):
        """Passage 1 strongly entails, Passage 2 strongly contradicts -> CONFLICTING."""
        passages = [
            {"doc_id": "doc_pro", "text": "Official records confirm the subsidy was granted in full."},
            {"doc_id": "doc_con", "text": "An investigative report revealed the subsidy was denied and revoked."},
        ]
        nli_preds = [
            {"entailment": 0.88, "contradiction": 0.04, "neutral": 0.08},
            {"entailment": 0.03, "contradiction": 0.89, "neutral": 0.08},
        ]

        verdict = self.aggregator.aggregate(passages, nli_preds)

        self.assertTrue(verdict.has_cross_passage_disparity)
        self.assertEqual(verdict.canonical_verdict, CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED)
        self.assertGreaterEqual(verdict.conflict_ratio, 0.40)
        self.assertEqual(verdict.disparity_details["supporter_doc_id"], "doc_pro")
        self.assertEqual(verdict.disparity_details["refuter_doc_id"], "doc_con")

    # -------------------------------------------------------------------------
    # 2. Unanimous Supporting Passages
    # -------------------------------------------------------------------------
    def test_multiple_passages_all_supporting(self):
        """All distinct passages support the claim -> VERIFIED_TRUE (no conflict)."""
        passages = [
            {"doc_id": "doc_sup_1", "text": "NASA satellite measurements show the global sea level rose by 0.3 cm."},
            {"doc_id": "doc_sup_2", "text": "Independent oceanographic buoys recorded an identical sea level increase."},
        ]
        nli_preds = [
            {"entailment": 0.85, "contradiction": 0.03, "neutral": 0.12},
            {"entailment": 0.82, "contradiction": 0.02, "neutral": 0.16},
        ]

        verdict = self.aggregator.aggregate(passages, nli_preds)

        self.assertFalse(verdict.has_cross_passage_disparity)
        self.assertEqual(verdict.canonical_verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertGreater(verdict.entailment_mass, verdict.contradiction_mass)

    # -------------------------------------------------------------------------
    # 3. Multiple Passages All Contradicting
    # -------------------------------------------------------------------------
    def test_multiple_passages_all_contradicting(self):
        """All distinct passages contradict the claim -> VERIFIED_FALSE (no conflict)."""
        passages = [
            {"doc_id": "doc_ref_1", "text": "Geologists confirmed the volcano did not erupt on Sunday."},
            {"doc_id": "doc_ref_2", "text": "Seismic activity remained dormant throughout the entire weekend."},
        ]
        nli_preds = [
            {"entailment": 0.02, "contradiction": 0.90, "neutral": 0.08},
            {"entailment": 0.03, "contradiction": 0.87, "neutral": 0.10},
        ]

        verdict = self.aggregator.aggregate(passages, nli_preds)

        self.assertFalse(verdict.has_cross_passage_disparity)
        self.assertEqual(verdict.canonical_verdict, CanonicalVeracity.VERIFIED_FALSE)
        self.assertGreater(verdict.contradiction_mass, verdict.entailment_mass)

    # -------------------------------------------------------------------------
    # 4. Weak Opposing Noise Must NOT Trigger Conflict
    # -------------------------------------------------------------------------
    def test_weak_opposing_noise_suppression(self):
        """
        Passage 1 is a strong supporter, but Passage 2 has minor contradiction noise (0.18 < 0.40)
        from background text. This must NOT activate cross-passage disparity.
        """
        passages = [
            {"doc_id": "doc_strong_e", "text": "The candidate graduated with honors from Oxford University in 2012."},
            {"doc_id": "doc_noisy_bg", "text": "Some commentators previously doubted candidate qualifications."},
        ]
        nli_preds = [
            {"entailment": 0.86, "contradiction": 0.04, "neutral": 0.10},
            {"entailment": 0.08, "contradiction": 0.18, "neutral": 0.74},  # c=0.18 is below disparity_contra_threshold (0.40)
        ]

        verdict = self.aggregator.aggregate(passages, nli_preds)

        self.assertFalse(verdict.has_cross_passage_disparity)
        self.assertEqual(verdict.canonical_verdict, CanonicalVeracity.VERIFIED_TRUE)

    # -------------------------------------------------------------------------
    # 5. Single Passage with Mixed Scores Must NOT Count as Cross-Passage Conflict
    # -------------------------------------------------------------------------
    def test_single_passage_mixed_scores_not_cross_passage_conflict(self):
        """
        A single passage with ambiguous/split scores (e.g. 0.45 E and 0.45 C)
        cannot satisfy cross-passage disparity because i != j is impossible with N=1.
        """
        passages = [
            {"doc_id": "single_doc", "text": "The bill passed the house but died in the senate committee."}
        ]
        nli_preds = [
            {"entailment": 0.45, "contradiction": 0.45, "neutral": 0.10}
        ]

        verdict = self.aggregator.aggregate(passages, nli_preds)

        self.assertFalse(verdict.has_cross_passage_disparity)
        self.assertNotEqual(verdict.canonical_verdict, CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED)

    # -------------------------------------------------------------------------
    # 6. Duplicate Passage Suppression via Jaccard
    # -------------------------------------------------------------------------
    def test_duplicate_passages_discounted(self):
        """Near-identical text copies must be deduplicated."""
        text = "The treaty was signed in Paris on December 12 2015 by all member states."
        passages = [
            {"doc_id": "copy_1", "text": text},
            {"doc_id": "copy_2", "text": text},  # Exact duplicate
        ]
        nli_preds = [
            {"entailment": 0.90, "contradiction": 0.02, "neutral": 0.08},
            {"entailment": 0.90, "contradiction": 0.02, "neutral": 0.08},
        ]

        verdict = self.aggregator.aggregate(passages, nli_preds)

        self.assertEqual(verdict.duplicates_filtered, 1)
        self.assertEqual(verdict.unique_passages_used, 1)

    # -------------------------------------------------------------------------
    # 7. Empty Evidence Handling
    # -------------------------------------------------------------------------
    def test_empty_evidence_insufficient(self):
        """Empty input lists must return INSUFFICIENT_EVIDENCE with confidence 0.95."""
        verdict = self.aggregator.aggregate([], [])
        self.assertEqual(verdict.canonical_verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertEqual(verdict.confidence, 0.95)
        self.assertFalse(verdict.has_cross_passage_disparity)

    # -------------------------------------------------------------------------
    # 8. Joint-Premise De-weighting Under Stance Disparity
    # -------------------------------------------------------------------------
    def test_joint_premise_deweighting_under_disparity(self):
        """
        When two distinct passages conflict, a concatenated joint premise biased towards
        contradiction must be de-weighted so it does not overwhelm the supporting passage.
        """
        passages = [
            {"doc_id": "p_support", "text": "The company reported 10 million in revenue for Q3."},
            {"doc_id": "p_refute", "text": "Auditors found the 10 million revenue was fabricated and actual revenue was zero."},
        ]
        individual_preds = [
            {"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10},
            {"entailment": 0.05, "contradiction": 0.85, "neutral": 0.10},
        ]
        # Joint premise strongly contradiction-biased
        joint_pred = {"entailment": 0.05, "contradiction": 0.90, "neutral": 0.05}

        # Case A: With disparity de-weighting (joint_weight_disparity=0.2)
        agg_with_protection = EvidenceAggregator(joint_weight_disparity=0.2)
        verdict_prot = agg_with_protection.aggregate(passages, individual_preds, joint_nli_prediction=joint_pred)
        self.assertTrue(verdict_prot.has_cross_passage_disparity)
        self.assertEqual(verdict_prot.canonical_verdict, CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED)

        # Case B: Without disparity de-weighting (joint_weight_disparity=1.5 - unmitigated)
        agg_no_protection = EvidenceAggregator(joint_weight_disparity=1.5)
        verdict_noprot = agg_no_protection.aggregate(passages, individual_preds, joint_nli_prediction=joint_pred)
        # Without protection, contradiction mass overwhelmingly suppresses the conflict
        self.assertGreater(verdict_noprot.contradiction_mass, verdict_noprot.entailment_mass * 2.0)

    # -------------------------------------------------------------------------
    # 9. Synthetic Regression: All-Neutral or Irrelevant Passages
    # -------------------------------------------------------------------------
    def test_all_neutral_or_irrelevant_evidence_returns_insufficient(self):
        """
        When all candidate passages are neutral or irrelevant (like the Sun/Earth core query),
        the system must return INSUFFICIENT_EVIDENCE even if the concatenated joint premise
        exhibits artifact contradiction. It must NEVER invent VERIFIED_FALSE.
        """
        passages = [
            {"doc_id": "doc_noise_1", "text": "As president, Biden will lead the world to address climate change."},
            {"doc_id": "doc_noise_2", "text": "Collins quote misrepresents praise for political figures."},
            {"doc_id": "doc_noise_3", "text": "Xinjiang is an autonomous region of China."},
            {"doc_id": "doc_noise_4", "text": "Iran maintains an enriched uranium stockpile."},
            {"doc_id": "doc_noise_5", "text": "Particles from smoke over Oregon turn skies orange."},
        ]
        individual_preds = [
            {"entailment": 0.05, "contradiction": 0.01, "neutral": 0.94},
            {"entailment": 0.02, "contradiction": 0.01, "neutral": 0.97},
            {"entailment": 0.04, "contradiction": 0.01, "neutral": 0.95},
            {"entailment": 0.03, "contradiction": 0.01, "neutral": 0.96},
            {"entailment": 0.01, "contradiction": 0.01, "neutral": 0.98},
        ]
        verdict = self.aggregator.aggregate(passages, individual_preds)

        self.assertEqual(verdict.canonical_verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        # INSUFFICIENT_EVIDENCE logit must strictly exceed true/false logits
        self.assertGreater(verdict.raw_logits[3], verdict.raw_logits[0])
        self.assertGreater(verdict.raw_logits[3], verdict.raw_logits[1])
        self.assertFalse(verdict.has_cross_passage_disparity)

    # -------------------------------------------------------------------------
    # 10. Low Entailment Must NOT Be Treated as Meaningful Refutation
    # -------------------------------------------------------------------------
    def test_low_entailment_not_treated_as_refutation(self):
        """
        A passage with low entailment score (e.g. e=0.03, neu=0.96, c=0.01) must NOT
        be counted as refuting evidence. It must result in INSUFFICIENT_EVIDENCE.
        """
        passages = [
            {"doc_id": "doc_unrelated", "text": "A recipe for chocolate cake requires flour and cocoa."}
        ]
        individual_preds = [
            {"entailment": 0.03, "contradiction": 0.01, "neutral": 0.96}
        ]
        verdict = self.aggregator.aggregate(passages, individual_preds)

        self.assertEqual(verdict.canonical_verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertLess(verdict.contradiction_mass, 0.05)

    # -------------------------------------------------------------------------
    # 11. Genuine Supporting Evidence Fixture
    # -------------------------------------------------------------------------
    def test_genuine_supporting_evidence_fixture(self):
        """Passage directly entailing the proposition yields VERIFIED_TRUE."""
        passages = [
            {"doc_id": "doc_solar_core", "text": "The Sun's core temperature reaches approximately 15 million Kelvin."}
        ]
        individual_preds = [
            {"entailment": 0.92, "contradiction": 0.02, "neutral": 0.06}
        ]
        verdict = self.aggregator.aggregate(passages, individual_preds)

        self.assertEqual(verdict.canonical_verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertGreater(verdict.entailment_mass, 0.5)

    # -------------------------------------------------------------------------
    # 12. Genuine Refuting Evidence Fixture
    # -------------------------------------------------------------------------
    def test_genuine_refuting_evidence_fixture(self):
        """Passage directly contradicting the proposition yields VERIFIED_FALSE."""
        passages = [
            {"doc_id": "doc_contra", "text": "Measurements prove that Earth's core is hotter than the Sun's core."}
        ]
        individual_preds = [
            {"entailment": 0.02, "contradiction": 0.93, "neutral": 0.05}
        ]
        verdict = self.aggregator.aggregate(passages, individual_preds)

        self.assertEqual(verdict.canonical_verdict, CanonicalVeracity.VERIFIED_FALSE)
        self.assertGreater(verdict.contradiction_mass, 0.5)


if __name__ == "__main__":
    unittest.main()
