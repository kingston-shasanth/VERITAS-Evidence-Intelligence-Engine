"""
Phase 1: Core Verification Integrity & NLP Regression Test Suite.
Validates all data-flow, spelling normalization, linguistics, verdict-evidence consistency,
and multi-hop reasoning requirements defined in Phase 1.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

# Ensure repo root and src/ are on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from veritas.data.schema import CanonicalVeracity
from veritas.verification.engine import VerificationEngine, ForensicVerdict
from veritas.verification.evidence_aggregator import EvidenceAggregator
from veritas.verification.calibrator import TemperatureCalibrator, SelectiveAbstainer
from veritas.nlp.spelling import SpellingNormalizer
from veritas.nlp.linguistics import LinguisticAnalyzer
from veritas.nlp.claim_extractor import ClaimExtractor
from api.adapter import VeritasAPIAdapter


class TestPhase1NormalizationAndNLPRegressions(unittest.TestCase):
    """
    Validates Section 2 and Section 3: Conservative spelling normalization,
    language safety, and normalization-to-verification data flow.
    """

    def setUp(self):
        self.normalizer = SpellingNormalizer()
        self.analyzer = LinguisticAnalyzer()
        self.extractor = ClaimExtractor()

    def test_ordinary_correctable_typo_accepted(self):
        """Accepted typo correction is properly identified and normalized."""
        res = self.normalizer.normalize("Pur watur frezes at 0 C")
        self.assertEqual(res.normalized_text, "Pure water freezes at 0 C")
        self.assertTrue(res.meaning_preservation_passed)
        self.assertGreater(len(res.corrections), 0)

    def test_sentence_initial_titles_preserved(self):
        """Sentence-initial titles like 'Dr.' must not be altered into lowercase or ordinary words."""
        res = self.normalizer.normalize("Dr. Smith treated the patient.")
        self.assertTrue(res.normalized_text.startswith("Dr. Smith"))

    def test_ordinals_preserved(self):
        """Ordinals like 45th must not be mutated into dictionary words."""
        res = self.normalizer.normalize("Donald Trump was the 45th president.")
        self.assertIn("45th", res.normalized_text)

    def test_scientific_units_and_notation_preserved(self):
        """Scientific units and exponential notations must remain intact."""
        units_test = "Light travels at 300,000 km/s, moon is 3.84e5 km away, water freezes at 0 °C and speed is 100 km/h."
        res = self.normalizer.normalize(units_test)
        self.assertIn("300,000 km/s", res.normalized_text)
        self.assertIn("3.84e5 km", res.normalized_text)
        self.assertIn("0 °C", res.normalized_text)
        self.assertIn("100 km/h", res.normalized_text)

        metric_test = "Dosage was 500 mg and mass was 25 kg with 10 cm and 5 mm tolerance."
        res_m = self.normalizer.normalize(metric_test)
        self.assertIn("500 mg", res_m.normalized_text)
        self.assertIn("25 kg", res_m.normalized_text)
        self.assertIn("10 cm", res_m.normalized_text)
        self.assertIn("5 mm", res_m.normalized_text)

    def test_negation_typo_safety(self):
        """Typo 'noot' must not mutate into affirmative word 'noon'."""
        res = self.normalizer.normalize("Water noot freeze at 100 C")
        self.assertNotIn("noon", res.normalized_text.lower())

    def test_non_english_text_safe_bypass(self):
        """Non-English text must not be corrupted by English dictionary corrections."""
        french_text = "L'eau gèle à 0 degré Celsius."
        res = self.normalizer.normalize(french_text)
        self.assertEqual(res.normalized_text, french_text)
        self.assertTrue(any("non-english" in w.lower() for w in res.warnings))

    def test_gerund_subject_claim_extraction(self):
        """Rule-based parser extracts gerund subjects correctly."""
        claim = self.extractor.extract("Smoking causes lung cancer.")
        self.assertEqual(claim.subject.lower(), "smoking")
        self.assertIn("cause", claim.predicate.lower() if claim.predicate else claim.predicate_lemma)
        self.assertIn("lung cancer", claim.object.lower())

    def test_subordinate_clause_negation_isolation(self):
        """Negation in subordinate clause must not attach to the main matrix verb."""
        result = self.analyzer.analyze("Scientists confirmed that vaccines do not cause autism.")
        sent = result.sentences[0]
        self.assertTrue(sent.clauses)
        sub_clause = next((c for c in sent.clauses if c.clause_type == "subordinate"), None)
        self.assertIsNotNone(sub_clause)
        self.assertTrue(sub_clause.is_negated)
        root_clause = next((c for c in sent.clauses if c.clause_type == "root"), None)
        self.assertIsNotNone(root_clause)
        self.assertFalse(root_clause.is_negated)

    def test_adapter_normalization_flow_distinguishes_trace_values(self):
        """
        Adapter correctly distinguishes original input, sanitized claim,
        normalized claim, retrieval query, and NLI hypothesis.
        """
        adapter = VeritasAPIAdapter(device_override="cpu")
        res = adapter.verify(
            claim="Pur water frezes at zero degrees.",
            user_evidence=["Pure water freezes at zero degrees Celsius."],
            include_processing_trace=True,
        )
        self.assertEqual(res["original_claim"], "Pur water frezes at zero degrees.")
        self.assertIn("Pure", res["normalized_claim"])
        self.assertEqual(res["normalization_status"], "accepted")
        self.assertEqual(res["retrieval_query_used"], res["normalized_claim"])
        self.assertEqual(res["nli_hypothesis_used"], res["sanitized_claim"])


class TestPhase1VerdictEvidenceConsistencyRegressions(unittest.TestCase):
    """
    Validates Section 5: Verdict-evidence consistency invariants,
    mandatory grounding, and conservative abstention.
    """

    def setUp(self):
        self.calibrator = TemperatureCalibrator(temperature=4.1039)
        self.abstainer = SelectiveAbstainer()

    def test_frozen_calibration_temperature_preserved(self):
        """Calibrator temperature T = 4.1039 is strictly frozen."""
        self.assertAlmostEqual(self.calibrator.temperature, 4.1039, places=4)

    def test_zero_eligible_supporting_passages_cannot_produce_verified_true(self):
        """Zero eligible supporting passages must NEVER result in VERIFIED_TRUE."""
        mock_nli = MagicMock()
        mock_nli.predict_batch.return_value = [
            {"predicted_label": "NEUTRAL", "entailment": 0.10, "contradiction": 0.05, "neutral": 0.85},
            {"predicted_label": "NEUTRAL", "entailment": 0.12, "contradiction": 0.08, "neutral": 0.80},
        ]
        engine = VerificationEngine(nli_model=mock_nli, calibrator=self.calibrator, abstainer=self.abstainer)
        passages = [
            {"doc_id": "p1", "text": "Water is a transparent chemical substance."},
            {"doc_id": "p2", "text": "Rivers and oceans are composed primarily of water."},
        ]
        verdict = engine.verify_claim("Pure water freezes at 0 degrees Celsius.", candidate_passages=passages)
        self.assertNotEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertTrue(verdict.was_abstained)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)

    def test_zero_eligible_refuting_passages_cannot_produce_verified_false(self):
        """Zero eligible refuting passages must NEVER result in VERIFIED_FALSE."""
        mock_nli = MagicMock()
        mock_nli.predict_batch.return_value = [
            {"predicted_label": "NEUTRAL", "entailment": 0.05, "contradiction": 0.10, "neutral": 0.85},
            {"predicted_label": "NEUTRAL", "entailment": 0.08, "contradiction": 0.12, "neutral": 0.80},
        ]
        engine = VerificationEngine(nli_model=mock_nli, calibrator=self.calibrator, abstainer=self.abstainer)
        passages = [
            {"doc_id": "p1", "text": "Water is a chemical substance found on Earth."},
            {"doc_id": "p2", "text": "Atmospheric conditions affect liquids."},
        ]
        verdict = engine.verify_claim("Water boils at 20 degrees Celsius.", candidate_passages=passages)
        self.assertNotEqual(verdict.verdict, CanonicalVeracity.VERIFIED_FALSE)
        self.assertTrue(verdict.was_abstained)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)

    def test_unsupported_relationship_cannot_be_verified_true(self):
        """Neutral biographies and general clinical trial pages cannot verify a specific role relationship."""
        mock_nli = MagicMock()
        mock_nli.predict_batch.return_value = [
            {"predicted_label": "NEUTRAL", "entailment": 0.10, "contradiction": 0.05, "neutral": 0.85},
            {"predicted_label": "NEUTRAL", "entailment": 0.05, "contradiction": 0.05, "neutral": 0.90},
        ]
        engine = VerificationEngine(nli_model=mock_nli, calibrator=self.calibrator, abstainer=self.abstainer)
        passages = [
            {"doc_id": "p1", "text": "Dr. Alvarez is an oncologist who graduated from Harvard University."},
            {"doc_id": "p2", "text": "Clinical trials are biomedical research studies performed in people to evaluate medical treatments."},
        ]
        verdict = engine.verify_claim("Dr. Alvarez oversaw the clinical trial.", candidate_passages=passages)
        self.assertNotEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)

    def test_internally_contradictory_claim_not_mislabeled_as_documentary_conflict(self):
        """An internally contradictory claim is not mislabeled as external documentary conflict."""
        mock_nli = MagicMock()
        mock_nli.predict_batch.return_value = [
            {"predicted_label": "NEUTRAL", "entailment": 0.20, "contradiction": 0.20, "neutral": 0.60},
        ]
        engine = VerificationEngine(nli_model=mock_nli, calibrator=self.calibrator, abstainer=self.abstainer)
        passages = [
            {"doc_id": "p1", "text": "The Earth is the third planet from the Sun and the only astronomical object known to harbor life."},
        ]
        claim = "The Earth is a planet, and the Earth is not a planet."
        verdict = engine.verify_claim(claim, candidate_passages=passages)
        self.assertNotEqual(verdict.verdict, CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED)

    def test_normal_compound_claim_independently_supported(self):
        """A compound claim with independently supported clauses evaluates to VERIFIED_TRUE."""
        mock_nli = MagicMock()
        def mock_predict(pairs):
            results = []
            for prem, hyp in pairs:
                hl = hyp.lower()
                pl = prem.lower()
                if "james webb" in hl and "james webb" in pl:
                    results.append({"predicted_label": "ENTAILMENT", "entailment": 0.95, "contradiction": 0.02, "neutral": 0.03})
                elif "jupiter" in hl and "jupiter" in pl:
                    results.append({"predicted_label": "ENTAILMENT", "entailment": 0.95, "contradiction": 0.02, "neutral": 0.03})
                elif "james webb" in pl and "jupiter" in pl:
                    results.append({"predicted_label": "ENTAILMENT", "entailment": 0.95, "contradiction": 0.02, "neutral": 0.03})
                else:
                    results.append({"predicted_label": "NEUTRAL", "entailment": 0.05, "contradiction": 0.05, "neutral": 0.90})
            return results

        mock_nli.predict_batch.side_effect = mock_predict
        engine = VerificationEngine(nli_model=mock_nli, calibrator=self.calibrator, abstainer=self.abstainer)
        passages = [
            {"doc_id": "p1", "text": "The James Webb Space Telescope launched on December 25, 2021."},
            {"doc_id": "p2", "text": "Jupiter is the largest planet in our solar system."},
        ]
        claim = "The James Webb Space Telescope launched in December 2021, and Jupiter is the largest planet."
        verdict = engine.verify_claim(claim, candidate_passages=passages)
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)

    def test_pressure_condition_mismatch_not_verified_true(self):
        """Water boiling claim under standard pressure is not verified true by altitude/variable pressure passages."""
        mock_nli = MagicMock()
        mock_nli.predict_batch.return_value = [
            {"predicted_label": "NEUTRAL", "entailment": 0.15, "contradiction": 0.15, "neutral": 0.70},
        ]
        engine = VerificationEngine(nli_model=mock_nli, calibrator=self.calibrator, abstainer=self.abstainer)
        passages = [
            {"doc_id": "p1", "text": "At high altitudes where atmospheric pressure is low, water boils below 100 degrees Celsius."},
        ]
        claim = "Water does not boil at 100 degrees Celsius at standard atmospheric pressure."
        verdict = engine.verify_claim(claim, candidate_passages=passages)
        self.assertNotEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)


class TestPhase1MultiHopBridgeRegressions(unittest.TestCase):
    """
    Validates Section 6: Multi-hop reasoning with valid vs invalid bridge discrimination,
    role-to-name identity linking, and preservation of Axiom 6 safety guarantees.
    """

    def setUp(self):
        self.calibrator = TemperatureCalibrator(temperature=4.1039)
        self.abstainer = SelectiveAbstainer()

    def test_role_to_name_valid_bridge_verified_true(self):
        """A valid bridge linking a person by role in one passage and by name in another verifies true."""
        mock_nli = MagicMock()
        def mock_predict(pairs):
            results = []
            for prem, hyp in pairs:
                if "Dr. Alvarez was elected director" in prem and "director of the institute authorized" in prem:
                    results.append({"predicted_label": "ENTAILMENT", "entailment": 0.92, "contradiction": 0.02, "neutral": 0.06})
                else:
                    results.append({"predicted_label": "NEUTRAL", "entailment": 0.15, "contradiction": 0.05, "neutral": 0.80})
            return results

        mock_nli.predict_batch.side_effect = mock_predict
        engine = VerificationEngine(nli_model=mock_nli, calibrator=self.calibrator, abstainer=self.abstainer)
        passages = [
            {"doc_id": "h1", "text": "Dr. Alvarez was elected director of the institute in September 2021."},
            {"doc_id": "h2", "text": "The director of the institute authorized and oversaw the clinical trial."},
        ]
        verdict = engine.verify_claim("Dr. Alvarez oversaw the clinical trial.", candidate_passages=passages)
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertFalse(verdict.was_abstained)
        self.assertTrue(verdict.has_meaningful_evidence)

    def test_role_to_name_broken_bridge_rejected(self):
        """If the identity link between role and person is missing, joint reasoning must reject the bridge."""
        mock_nli = MagicMock()
        mock_nli.predict_batch.return_value = [
            {"predicted_label": "NEUTRAL", "entailment": 0.10, "contradiction": 0.05, "neutral": 0.85},
            {"predicted_label": "NEUTRAL", "entailment": 0.10, "contradiction": 0.05, "neutral": 0.85},
            {"predicted_label": "NEUTRAL", "entailment": 0.15, "contradiction": 0.05, "neutral": 0.80},
        ]
        engine = VerificationEngine(nli_model=mock_nli, calibrator=self.calibrator, abstainer=self.abstainer)
        passages = [
            {"doc_id": "h1", "text": "Dr. Alvarez attended the annual medical research symposium."},
            {"doc_id": "h2", "text": "The director of the institute authorized and oversaw the clinical trial."},
        ]
        verdict = engine.verify_claim("Dr. Alvarez oversaw the clinical trial.", candidate_passages=passages)
        self.assertNotEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)


if __name__ == "__main__":
    unittest.main()
