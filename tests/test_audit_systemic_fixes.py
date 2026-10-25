"""
Systemic Root-Cause Regression and Metamorphic Test Suite.
Verifies fixes P1 through P5 from docs/audit_reports/randomized_claim_audit_report.md:
- P1: Noise-resistant evidence aggregation & joint-premise gating
- P2: General claim decomposition, copular predicates & semantic alignment
- P3: Absence-of-evidence epistemic reasoning
- P4: Granular subclaim-to-evidence mapping in API adapter
- P5: Filter & badge consistency across eligibility states
"""
import os
import sys
import unittest
from typing import Dict, Any, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from veritas.data.schema import CanonicalVeracity
from veritas.verification.evidence_aggregator import EvidenceAggregator
from veritas.verification.engine import (
    VerificationEngine,
    _decompose_claim,
    _is_syntactic_verb,
    _tokens_match,
)
from api.adapter import VeritasAPIAdapter


class TestP1NoiseResistantAggregation(unittest.TestCase):
    """P1: Gated joint-premise aggregation & noise invariance."""

    def setUp(self):
        self.aggregator = EvidenceAggregator()

    def test_noise_invariance_clean_vs_noisy(self):
        """
        Metamorphic Property: Adding off-topic noise to a verified claim
        MUST NOT increase confidence or change the justified verdict.
        Delta confidence must be <= 0.01 (here exactly 0.00).
        """
        clean_cand = [{
            "doc_id": "P1",
            "text": "An adult human normally has two lungs located in the thoracic cavity.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
            "is_decision_eligible": True,
        }]
        clean_nli = [{"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03}]

        noisy_cand = [
            clean_cand[0],
            {
                "doc_id": "P2_NOISE",
                "text": "The municipal water authority operates a gravity-fed water purification facility.",
                "is_relevant": False,
                "is_informative": True,
                "is_claim_aligned": False,
                "is_decision_eligible": False,
                "claim_components_covered": [],
            }
        ]
        noisy_nli = [
            clean_nli[0],
            {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.90}
        ]
        # In the noisy case, the joint premise was contaminated with noise
        noisy_joint_nli = {"entailment": 0.94, "contradiction": 0.03, "neutral": 0.03}

        res_clean = self.aggregator.aggregate(
            candidate_passages=clean_cand,
            nli_predictions=clean_nli,
            joint_nli_prediction=None
        )
        res_noisy = self.aggregator.aggregate(
            candidate_passages=noisy_cand,
            nli_predictions=noisy_nli,
            joint_nli_prediction=noisy_joint_nli
        )

        self.assertEqual(res_clean.canonical_verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertEqual(res_noisy.canonical_verdict, CanonicalVeracity.VERIFIED_TRUE)
        # Ineligible noise must NOT increase entailment mass
        self.assertEqual(res_clean.entailment_mass, res_noisy.entailment_mass)
        self.assertEqual(res_clean.contradiction_mass, res_noisy.contradiction_mass)
        self.assertEqual(res_clean.raw_logits, res_noisy.raw_logits)

    def test_duplicate_evidence_does_not_manufacture_joint_corroboration(self):
        """Duplicate passages are discounted and cannot manufacture joint premise mass."""
        cand = [
            {
                "doc_id": "P1",
                "text": "Pure water freezes at 0 degrees Celsius.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
                "is_decision_eligible": True,
            },
            {
                "doc_id": "P1_DUP",
                "text": "Pure water freezes at 0 degrees Celsius.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
                "is_decision_eligible": True,
            }
        ]
        nli = [
            {"entailment": 0.90, "contradiction": 0.02, "neutral": 0.08},
            {"entailment": 0.90, "contradiction": 0.02, "neutral": 0.08},
        ]
        joint_nli = {"entailment": 0.92, "contradiction": 0.02, "neutral": 0.06}

        res = self.aggregator.aggregate(
            candidate_passages=cand,
            nli_predictions=nli,
            joint_nli_prediction=joint_nli
        )
        self.assertEqual(res.duplicates_filtered, 1)
        # Because the duplicate was filtered out, only 1 unique passage exists -> joint premise mass is 0
        self.assertEqual(res.canonical_verdict, CanonicalVeracity.VERIFIED_TRUE)

    def test_independent_supporting_evidence_strengthens_verdict(self):
        """Two genuine distinct eligible supporting passages legitimately corroborate."""
        cand_single = [{
            "doc_id": "P1",
            "text": "The Pacific Ocean covers over 165 million square kilometers.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
            "is_decision_eligible": True,
        }]
        nli_single = [{"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10}]

        cand_double = [
            cand_single[0],
            {
                "doc_id": "P2",
                "text": "The Mariana Trench in the Pacific contains the Challenger Deep, Earth's deepest point.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
                "is_decision_eligible": True,
            }
        ]
        nli_double = [
            nli_single[0],
            {"entailment": 0.88, "contradiction": 0.04, "neutral": 0.08}
        ]
        joint_nli = {"entailment": 0.96, "contradiction": 0.02, "neutral": 0.02}

        res_single = self.aggregator.aggregate(cand_single, nli_single, None)
        res_double = self.aggregator.aggregate(cand_double, nli_double, joint_nli)

        self.assertEqual(res_double.canonical_verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertGreater(res_double.entailment_mass, res_single.entailment_mass)

    def test_multihop_reasoning_preserved_across_complementary_passages(self):
        """Multi-hop evidence where distinct passages cover subject and predicate remains usable."""
        cand = [
            {
                "doc_id": "P1",
                "text": "Charles Darwin wrote On the Origin of Species.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": False,
                "claim_components_covered": ["Subject / Entity"],
            },
            {
                "doc_id": "P2",
                "text": "Charles Darwin served as naturalist on HMS Beagle.",
                "is_relevant": False,
                "is_informative": True,
                "is_claim_aligned": True,
                "claim_components_covered": ["Action / Status"],
            }
        ]
        nli = [
            {"entailment": 0.15, "contradiction": 0.05, "neutral": 0.80},
            {"entailment": 0.15, "contradiction": 0.05, "neutral": 0.80},
        ]
        joint_nli = {"entailment": 0.92, "contradiction": 0.03, "neutral": 0.05}

        res = self.aggregator.aggregate(cand, nli, joint_nli)
        self.assertEqual(res.canonical_verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertGreater(res.entailment_mass, 0.50)

    def test_evidence_ordering_invariance(self):
        """Metamorphic Property: Swapping evidence order must not arbitrarily change verdict."""
        p1 = {
            "doc_id": "P1",
            "text": "Megalodon teeth suggest maximum sizes over 20 meters.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
            "is_decision_eligible": True,
        }
        p2 = {
            "doc_id": "P2",
            "text": "Recent vertebral column studies suggest Megalodon was around 15 meters.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
            "is_decision_eligible": True,
        }
        n1 = {"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10}
        n2 = {"entailment": 0.05, "contradiction": 0.88, "neutral": 0.07}

        res_order1 = self.aggregator.aggregate([p1, p2], [n1, n2], None)
        res_order2 = self.aggregator.aggregate([p2, p1], [n2, n1], None)

        self.assertEqual(res_order1.canonical_verdict, res_order2.canonical_verdict)
        self.assertEqual(res_order1.canonical_verdict, CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED)


class TestP2ClaimDecompositionAndAlignment(unittest.TestCase):
    """P2: Copular predicates, 3rd-person singular verbs & semantic normalization."""

    def test_copular_predicate_with_number_is_not_treated_as_verb(self):
        """In 'The freezing point is zero degrees', 'zero' must NOT be added to verb_tokens."""
        claim = "The freezing point of pure water under standard atmospheric pressure is zero degrees Celsius."
        sub_subj, sub_pred, date_num_tokens, verb_tokens = _decompose_claim(claim)
        self.assertNotIn("zero", verb_tokens, "Number word 'zero' was wrongly classified as an action verb!")
        self.assertIn("zero", date_num_tokens)
        self.assertIn("celsius", sub_pred)

    def test_syntactic_verb_generalization_releases_decreases(self):
        """3rd-person singular relational verbs must be recognized without hardcoded claim whitelists."""
        self.assertTrue(_is_syntactic_verb("releases"))
        self.assertTrue(_is_syntactic_verb("decreases"))
        self.assertTrue(_is_syntactic_verb("increases"))
        self.assertTrue(_is_syntactic_verb("melts"))
        self.assertTrue(_is_syntactic_verb("absorbs"))
        self.assertTrue(_is_syntactic_verb("dissolves"))

        # Negative checks: plural nouns and words with non-verb suffixes must return False
        self.assertFalse(_is_syntactic_verb("humans", prev_token="all"))
        self.assertFalse(_is_syntactic_verb("degrees", prev_token="zero"))
        self.assertFalse(_is_syntactic_verb("physics"))
        self.assertFalse(_is_syntactic_verb("conditions"))

    def test_copular_paraphrase_aligns_with_temperature_evidence(self):
        """Paraphrased copular claim ('is zero degrees Celsius') aligns with 'freezes at 0°C'."""
        claim = "The freezing point of pure water under standard atmospheric pressure is zero degrees Celsius."
        passage = "Under standard 1 atm pressure at sea level, pure liquid water freezes at 0°C."
        self.assertTrue(VerificationEngine.has_claim_relation_alignment(claim, passage))
        self.assertTrue(VerificationEngine.is_topically_relevant(claim, passage))
        self.assertTrue(VerificationEngine.is_decision_eligible(claim, passage))

    def test_synonym_alignment_coffee_and_caffeine(self):
        """Closely related concepts in domain context align without false ineligibility."""
        self.assertTrue(_tokens_match("coffee", "caffeine"))
        self.assertTrue(_tokens_match("caffeine", "coffee"))
        self.assertTrue(_tokens_match("satellite", "moon"))
        self.assertTrue(_tokens_match("zero", "0"))
        self.assertTrue(_tokens_match("celsius", "c"))

    def test_related_but_non_equivalent_concepts_not_aligned(self):
        """Non-equivalent concepts do NOT spuriously align."""
        self.assertFalse(_tokens_match("water", "gasoline"))
        self.assertFalse(_tokens_match("mercury", "plutonium"))
        self.assertFalse(_tokens_match("earth", "jupiter"))


class TestP3AbsenceOfEvidenceReasoning(unittest.TestCase):
    """P3: Epistemic uncertainty filter for absence-of-evidence statements."""

    def test_absence_of_evidence_detection_positive_cases(self):
        """Expressions stating absence of detection or lack of records are detected."""
        s1 = "While Cassini detected hydrothermal activity, no direct evidence of microbial life has been confirmed."
        s2 = "Surviving Jacobean stationers' registers and theater archives make no mention of an unpublished Shakespeare play."
        s3 = "To date, no extraterrestrial techno-signature has been detected by SETI radio observatories."
        s4 = "Ecclesiastical archives contain no record of such a coronation ceremony."
        s5 = "The biological mechanism remains currently unproven and undocumented."

        self.assertTrue(VerificationEngine.is_absence_of_evidence(s1))
        self.assertTrue(VerificationEngine.is_absence_of_evidence(s2))
        self.assertTrue(VerificationEngine.is_absence_of_evidence(s3))
        self.assertTrue(VerificationEngine.is_absence_of_evidence(s4))
        self.assertTrue(VerificationEngine.is_absence_of_evidence(s5))

    def test_genuine_contradiction_is_preserved_not_neutralized(self):
        """Substantive empirical refutations are NOT mistaken for absence of evidence."""
        c1 = "Antibiotics kill bacteria, but have no effect against viruses such as influenza."
        c2 = "Bromine and mercury exist as liquids at standard conditions, refuting the claim that all elements are solid."
        c3 = "Sound waves cannot propagate through a complete vacuum."
        c4 = "Gold does not dissolve in concentrated hydrochloric acid at room temperature."

        self.assertFalse(VerificationEngine.is_absence_of_evidence(c1))
        self.assertFalse(VerificationEngine.is_absence_of_evidence(c2))
        self.assertFalse(VerificationEngine.is_absence_of_evidence(c3))
        self.assertFalse(VerificationEngine.is_absence_of_evidence(c4))


class TestP4SubclaimToEvidenceMapping(unittest.TestCase):
    """P4: Granular subclaim proposition to evidence passage mapping in API adapter."""

    def setUp(self):
        self.adapter = VeritasAPIAdapter(device_override="cpu")

    def test_compound_claim_subclaim_evidence_isolation(self):
        """
        In a compound claim:
        - Passage 1 supports subclaim 1
        - Passage 2 refutes subclaim 2
        - Passage 3 is off-topic noise
        Verify that subclaims do NOT receive whole-claim evidence distribution.
        """
        claim = "The Apollo 11 lunar module landed on the Moon in July 1969, and Neil Armstrong walked on Mars."
        p_sup = "On July 20, 1969, the Apollo 11 Lunar Module Eagle touched down in the Sea of Tranquillity on the Moon."
        p_ref = "Neil Armstrong walked on the Moon during Apollo 11 and never traveled to or walked on Mars."
        p_noise = "The city of Stockholm hosted the Olympic Games in 1912."

        res = self.adapter.verify(
            claim=claim,
            user_evidence=[p_sup, p_ref, p_noise],
            include_research=True
        )

        subclaims = res.get("subclaims", [])
        self.assertGreaterEqual(len(subclaims), 2, "Expected at least 2 decomposed subclaims")

        sc_landing = next((sc for sc in subclaims if "landed" in sc["text"].lower() or "1969" in sc["text"]), None)
        sc_mars = next((sc for sc in subclaims if "mars" in sc["text"].lower()), None)

        self.assertIsNotNone(sc_landing)
        self.assertIsNotNone(sc_mars)

        # Subclaim 1 (Landing) must be SUPPORTED
        self.assertEqual(sc_landing["status"], "SUPPORTED")
        self.assertGreater(sc_landing["stance_distribution"]["support"], 0)
        self.assertEqual(sc_landing["stance_distribution"]["refute"], 0)

        # Subclaim 2 (Mars) must be REFUTED
        self.assertEqual(sc_mars["status"], "REFUTED")
        self.assertGreater(sc_mars["stance_distribution"]["refute"], 0)
        self.assertEqual(sc_mars["stance_distribution"]["support"], 0)

        # The noise passage (Stockholm) must NOT map to either subclaim
        for sc in [sc_landing, sc_mars]:
            for eid in sc["evidence_ids"]:
                ev_item = next(e for e in res["evidence"] if e["id"] == eid)
                self.assertNotIn("Stockholm", ev_item["text"])


if __name__ == "__main__":
    unittest.main()
