"""
Unit & Integration Test Suite for VERITAS-X Phase 3:
Verification, Multi-Evidence Consensus, Temporal Integrity, and Uncertainty Calibration.
"""
import unittest
import os
import sys

# Ensure src/ and repo root are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

from veritas.data.schema import CanonicalVeracity
from veritas.verification.nli_model import CrossEncoderNLI
from veritas.verification.evidence_aggregator import EvidenceAggregator
from veritas.verification.temporal_analyzer import TemporalAnalyzer, TemporalStatus
from veritas.verification.calibrator import TemperatureCalibrator, SelectiveAbstainer
from veritas.verification.engine import VerificationEngine, ForensicVerdict


class TestVerificationEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load NLI model once for all test methods
        cls.nli = CrossEncoderNLI(device="cpu")
        cls.aggregator = EvidenceAggregator()
        cls.temporal = TemporalAnalyzer()
        cls.calibrator = TemperatureCalibrator(temperature=2.0)
        cls.abstainer = SelectiveAbstainer(confidence_threshold=0.40, entropy_threshold=0.85)
        cls.engine = VerificationEngine(
            nli_model=cls.nli,
            calibrator=cls.calibrator,
            abstainer=cls.abstainer,
        )

    # ---------------------------------------------------------
    # 1. Direct Contradiction Test
    # ---------------------------------------------------------
    def test_contradictory_evidence(self):
        claim = "The Earth is a flat disc supported on the back of giant elephants."
        evidence = [{
            "doc_id": "test_ev_01",
            "text": "Scientific geodesy and orbital satellite photography prove that the Earth is an oblate spheroid, not flat.",
            "source_url": "https://nasa.gov",
        }]
        verdict = self.engine.verify_claim(claim_text=claim, candidate_passages=evidence)
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_FALSE)
        self.assertGreater(verdict.class_probabilities[CanonicalVeracity.VERIFIED_FALSE.value], 0.40)

    # ---------------------------------------------------------
    # 2. Direct Entailment Test
    # ---------------------------------------------------------
    def test_entailing_evidence(self):
        claim = "Pure water freezes at 0 degrees Celsius at sea level."
        evidence = [{
            "doc_id": "test_ev_02",
            "text": "Under standard atmospheric pressure at sea level, the freezing point of pure water is exactly 0 degrees Celsius.",
            "source_url": "https://britannica.com",
        }]
        verdict = self.engine.verify_claim(claim_text=claim, candidate_passages=evidence)
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertGreater(verdict.class_probabilities[CanonicalVeracity.VERIFIED_TRUE.value], 0.40)

    # ---------------------------------------------------------
    # 3. Conflicting Evidence Detection
    # ---------------------------------------------------------
    def test_conflicting_evidence_detection(self):
        claim = "The local tax reform was beneficial to the community."
        evidence = [
            {
                "doc_id": "ev_pos",
                "text": "A comprehensive economic review proved that the local tax reform was highly beneficial to the community.",
            },
            {
                "doc_id": "ev_neg",
                "text": "An independent audit proved that the local tax reform was not beneficial and caused severe harm to the community.",
            }
        ]
        verdict = self.engine.verify_claim(claim_text=claim, candidate_passages=evidence)
        # Should detect conflict between positive and negative evidence
        self.assertEqual(verdict.verdict, CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED)
        self.assertGreater(verdict.class_probabilities[CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value], 0.40)

    # ---------------------------------------------------------
    # 4. Missing Evidence Abstention
    # ---------------------------------------------------------
    def test_missing_evidence_abstention(self):
        claim = "Alien spacecraft landed in the central plaza yesterday."
        verdict = self.engine.verify_claim(claim_text=claim, candidate_passages=[])
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertEqual(verdict.evidence_used_count, 0)
        self.assertEqual(verdict.class_probabilities[CanonicalVeracity.INSUFFICIENT_EVIDENCE.value], 1.0)

    # ---------------------------------------------------------
    # 4b. Irrelevant Evidence Regression Tests
    # ---------------------------------------------------------
    def test_irrelevant_evidence_abstention_reported_claim(self):
        """
        Regression test: Claim 'this veritas x is better than chatgpt' should NOT
        be adjudicated VERIFIED_FALSE when retrieved evidence contains unrelated
        rhetorical negation ('No, In Collins quote...').
        Must return INSUFFICIENT_EVIDENCE with was_abstained=True and has_meaningful_evidence=False.
        """
        claim = "this veritas x is better than chatgpt"
        evidence = [
            {
                "doc_id": "doc_averitec_230_1",
                "text": "No, In Collins quote, 'it\'s' has been replace by 'he\'s' which changes the meaning - wrongly implying that the National leader said the current US president was better than all those before him.",
                "source_url": "https://example.com/factcheck",
            },
            {
                "doc_id": "doc_averitec_320_0",
                "text": "The claim appeared in a video showing Moynihan claiming pandemic face masks are ineffective.",
                "source_url": "https://example.com/video",
            },
        ]
        verdict = self.engine.verify_claim(claim_text=claim, candidate_passages=evidence)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertTrue(verdict.was_abstained)
        self.assertFalse(verdict.has_meaningful_evidence)
        self.assertGreater(verdict.irrelevant_passages_filtered, 0)
        self.assertIn("Insufficient evidentiary grounding", verdict.abstention_reason)

    def test_irrelevant_evidence_abstention_general_mechanism(self):
        """
        Ensures the general topical relevance failure mechanism protects any arbitrary claim
        from false conclusive verdicts when retrieved passages lack topical overlap.
        """
        claim = "Quantum microchips operate at ambient room temperature in standard smartphones."
        evidence = [{
            "doc_id": "doc_unrelated_negation",
            "text": "No, the senator denied that the new tax policy was better than the previous economic proposal.",
            "source_url": "https://example.com/politics",
        }]
        verdict = self.engine.verify_claim(claim_text=claim, candidate_passages=evidence)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertTrue(verdict.was_abstained)
        self.assertFalse(verdict.has_meaningful_evidence)
        self.assertGreater(verdict.irrelevant_passages_filtered, 0)

    def test_is_topically_relevant_filter(self):
        """Tests the VerificationEngine.is_topically_relevant static method."""
        claim = "this veritas x is better than chatgpt"
        unrelated = "No, In Collins quote, 'it\'s' has been replace by 'he\'s' which changes the meaning."
        related = "Benchmarks indicate that Veritas X outperforms ChatGPT on automated fact-checking tasks."
        self.assertFalse(VerificationEngine.is_topically_relevant(claim, unrelated))
        self.assertTrue(VerificationEngine.is_topically_relevant(claim, related))
        self.assertFalse(VerificationEngine.is_topically_relevant("", related))
        self.assertFalse(VerificationEngine.is_topically_relevant(claim, ""))

    def test_strong_support_plus_unrelated_contradictory_distractor_not_conflicting(self):
        """
        Regression Test: Strong supporting evidence paired with an unrelated,
        highly contradictory distractor passage must NOT produce a false CONFLICTING_OR_CHERRYPICKED verdict.
        """
        claim = "The New York Post endorsed Donald Trump for re-election."
        evidence = [
            {
                "doc_id": "ev_support",
                "text": "The New York Post officially endorses President Donald Trump for re-election in 2020.",
                "source_url": "https://nypost.com/endorsement",
            },
            {
                "doc_id": "ev_distractor",
                "text": "The Washington Post published an article about the history of the postal service.",
                "source_url": "https://example.com/post",
            }
        ]
        verdict = self.engine.verify_claim(claim_text=claim, candidate_passages=evidence)
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertFalse(verdict.was_abstained)

    def test_neutral_dominant_evidence_abstains_to_insufficient(self):
        """
        Regression Test: Neutral-dominant evidence that fails to establish the claim predicate
        must evaluate to has_meaningful_evidence=False and abstain to INSUFFICIENT_EVIDENCE.
        """
        claim = "Amy Coney Barrett was confirmed as US Supreme Court Justice on October 26, 2020"
        evidence = [
            {
                "doc_id": "ev_neutral_dominant",
                "text": "Amy Coney Barrett fought against health care expansion and faced questioning at her nomination hearings.",
                "source_url": "https://example.com/news",
            }
        ]
        verdict = self.engine.verify_claim(claim_text=claim, candidate_passages=evidence)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertTrue(verdict.was_abstained)
        self.assertFalse(verdict.has_meaningful_evidence)

    # ---------------------------------------------------------
    # 5. Temporal Consistency & Anachronism Warnings
    # ---------------------------------------------------------
    def test_conflicting_dates_temporal_warning(self):
        claim_date = "2024-06-01"
        old_evidence = [{
            "doc_id": "ev_old",
            "text": "The Prime Minister announced a new trade treaty.",
            "source_url": "https://web.archive.org/web/20150115000000/https://news.org/treaty",
            "metadata": {"date": "2015-01-15"}
        }]
        rep = self.temporal.analyze(claim_date, old_evidence)
        self.assertEqual(rep.status, TemporalStatus.ANACHRONISM_WARNING)
        self.assertIsNotNone(rep.warning_message)
        self.assertIn("Anachronism Warning", rep.warning_message)
        self.assertGreater(abs(rep.delta_days), 3000)
        self.assertGreater(rep.confidence_penalty, 0.0)

    def test_aligned_dates_no_warning(self):
        claim_date = "2023-08-25"
        recent_evidence = [{
            "doc_id": "ev_aligned",
            "text": "ISRO launched the rocket on August 23, 2023.",
            "metadata": {"date": "2023-08-23"}
        }]
        rep = self.temporal.analyze(claim_date, recent_evidence)
        self.assertEqual(rep.status, TemporalStatus.ALIGNED)
        self.assertIsNone(rep.warning_message)
        self.assertEqual(rep.confidence_penalty, 0.0)

    # ---------------------------------------------------------
    # 6. Low-Confidence & High-Entropy Selective Abstention
    # ---------------------------------------------------------
    def test_low_confidence_abstention(self):
        abstainer = SelectiveAbstainer(confidence_threshold=0.60, entropy_threshold=0.75)
        # Ambiguous distribution: [0.25, 0.25, 0.25, 0.25]
        ambiguous_probs = [0.25, 0.25, 0.25, 0.25]
        should_abstain, reason = abstainer.should_abstain(ambiguous_probs)
        self.assertTrue(should_abstain)
        self.assertIn("below threshold", reason)

        # High certainty distribution: [0.90, 0.04, 0.03, 0.03]
        confident_probs = [0.90, 0.04, 0.03, 0.03]
        should_abstain, _ = abstainer.should_abstain(confident_probs)
        self.assertFalse(should_abstain)

    # ---------------------------------------------------------
    # 7. Multilingual Inputs
    # ---------------------------------------------------------
    def test_multilingual_inputs(self):
        # Hindi premise-hypothesis verification
        claim_hi = "विश्व स्वास्थ्य संगठन ने भारत को पोलियो मुक्त घोषित किया।"
        evidence_hi = [{
            "doc_id": "hi_ev",
            "text": "विश्व स्वास्थ्य संगठन ने 2014 में भारत को आधिकारिक रूप से पोलियो मुक्त घोषित किया था।",
        }]
        verdict = self.engine.verify_claim(claim_text=claim_hi, candidate_passages=evidence_hi)
        self.assertIsInstance(verdict.verdict, CanonicalVeracity)
        self.assertTrue(len(verdict.evidence_breakdown) > 0)


class MockNLI:
    """Deterministic synthetic fixture for NLI predictions."""

    def __init__(self, pred_map=None, default_pred=None):
        self.pred_map = pred_map or {}
        self.default_pred = default_pred or {"entailment": 0.01, "contradiction": 0.01, "neutral": 0.98}

    def predict_batch(self, pairs):
        results = []
        for premise, hypothesis in pairs:
            premise_str = str(premise).strip()
            if premise_str in self.pred_map:
                results.append(dict(self.pred_map[premise_str]))
            else:
                results.append(dict(self.default_pred))
        return results


class TestEvidenceAwareCalibratedAbstention(unittest.TestCase):
    """
    Deterministic regression test suite for Option C: Evidence-Aware Calibrated Abstention.
    Uses synthetic fixtures to verify all 8 required behaviors without live model variance.
    """

    def setUp(self):
        self.calibrator = TemperatureCalibrator(temperature=4.1039)
        self.abstainer = SelectiveAbstainer(
            confidence_threshold=0.32,
            entropy_threshold=0.94,
            evidence_aware_conf_threshold=0.38,
            margin_threshold=0.10,
        )

    # -------------------------------------------------------------------------
    # 1. Apollo-like claims with all-neutral evidence cannot be definitive false
    # -------------------------------------------------------------------------
    def test_apollo_neutral_evidence_cannot_be_definitive_false(self):
        from api.adapter import FROZEN_AGGREGATOR_PARAMS

        p1 = "Apollo 11 launched on July 16, 1969 with Armstrong, Aldrin, and Collins."
        p2 = "The Apollo lunar module landed at the Sea of Tranquility on the Moon."
        p3 = "Astronauts collected lunar core samples and deployed scientific instruments."
        joint = f"{p1} {p2} {p3}"

        mock_nli = MockNLI(
            pred_map={
                p1: {"entailment": 0.005, "contradiction": 0.015, "neutral": 0.980},
                p2: {"entailment": 0.004, "contradiction": 0.018, "neutral": 0.978},
                p3: {"entailment": 0.005, "contradiction": 0.016, "neutral": 0.979},
                joint: {"entailment": 0.025, "contradiction": 0.145, "neutral": 0.830},
            }
        )
        engine = VerificationEngine(
            nli_model=mock_nli,
            calibrator=self.calibrator,
            abstainer=self.abstainer,
        )
        engine.aggregator = EvidenceAggregator(**FROZEN_AGGREGATOR_PARAMS)
        claim = "Apollo 11 astronauts placed a nuclear bomb on the moon."
        candidate_passages = [
            {"doc_id": "doc1", "text": p1},
            {"doc_id": "doc2", "text": p2},
            {"doc_id": "doc3", "text": p3},
        ]

        verdict = engine.verify_claim(claim, candidate_passages=candidate_passages)

        # Must be abstained, retaining the underlying prediction as non-final
        self.assertTrue(verdict.was_abstained)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertEqual(verdict.raw_prediction, CanonicalVeracity.VERIFIED_FALSE.value)
        self.assertFalse(verdict.has_meaningful_evidence)
        self.assertIn("Insufficient evidentiary grounding", verdict.abstention_reason)

    # -------------------------------------------------------------------------
    # 2. Clearly supporting evidence can still support VERIFIED_TRUE
    # -------------------------------------------------------------------------
    def test_clearly_supporting_evidence_supports_true(self):
        p_sup = "Clinical trials confirmed the vaccine provides 95% protection against infection."
        mock_nli = MockNLI(
            pred_map={
                p_sup: {"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04},
            }
        )
        engine = VerificationEngine(
            nli_model=mock_nli,
            calibrator=self.calibrator,
            abstainer=self.abstainer,
        )
        claim = "The vaccine is effective against infection."
        verdict = engine.verify_claim(claim, candidate_passages=[{"doc_id": "doc_sup", "text": p_sup}])

        self.assertFalse(verdict.was_abstained)
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertEqual(verdict.raw_prediction, CanonicalVeracity.VERIFIED_TRUE.value)
        self.assertTrue(verdict.has_meaningful_evidence)

    # -------------------------------------------------------------------------
    # 3. Clearly refuting evidence can still support VERIFIED_FALSE
    # -------------------------------------------------------------------------
    def test_clearly_refuting_evidence_supports_false(self):
        p_ref = "Official financial audits confirmed the agency had zero debt and a multi-billion dollar surplus."
        mock_nli = MockNLI(
            pred_map={
                p_ref: {"entailment": 0.01, "contradiction": 0.95, "neutral": 0.04},
            }
        )
        engine = VerificationEngine(
            nli_model=mock_nli,
            calibrator=self.calibrator,
            abstainer=self.abstainer,
        )
        claim = "The agency went bankrupt due to severe unpaid debt."
        verdict = engine.verify_claim(claim, candidate_passages=[{"doc_id": "doc_ref", "text": p_ref}])

        self.assertFalse(verdict.was_abstained)
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_FALSE)
        self.assertEqual(verdict.raw_prediction, CanonicalVeracity.VERIFIED_FALSE.value)
        self.assertTrue(verdict.has_meaningful_evidence)

    # -------------------------------------------------------------------------
    # 4. Genuine opposing evidence triggers conflict detection
    # -------------------------------------------------------------------------
    def test_genuine_opposing_evidence_triggers_conflict(self):
        p_pro = "The new policy increased regional employment by twelve percent according to state data."
        p_con = "The new policy reduced regional employment and caused massive layoffs according to census audits."
        joint = f"{p_pro} {p_con}"
        mock_nli = MockNLI(
            pred_map={
                p_pro: {"entailment": 0.88, "contradiction": 0.02, "neutral": 0.10},
                p_con: {"entailment": 0.02, "contradiction": 0.88, "neutral": 0.10},
                joint: {"entailment": 0.40, "contradiction": 0.40, "neutral": 0.20},
            }
        )
        engine = VerificationEngine(
            nli_model=mock_nli,
            calibrator=self.calibrator,
            abstainer=self.abstainer,
        )
        claim = "The new policy boosted regional employment."
        candidate_passages = [
            {"doc_id": "doc_pro", "text": p_pro},
            {"doc_id": "doc_con", "text": p_con},
        ]
        verdict = engine.verify_claim(claim, candidate_passages=candidate_passages)

        self.assertFalse(verdict.was_abstained)
        self.assertEqual(verdict.verdict, CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED)
        self.assertTrue(verdict.has_meaningful_evidence)

    # -------------------------------------------------------------------------
    # 5. Multi-hop claims with meaningful joint evidence are not abstained
    # -------------------------------------------------------------------------
    def test_multihop_claims_with_meaningful_joint_evidence_retained(self):
        p1 = "Dr. Alvarez was elected director of the institute in September 2021."
        p2 = "The director of the institute authorized and oversaw the clinical trial."
        joint = f"{p1} {p2}"
        mock_nli = MockNLI(
            pred_map={
                p1: {"entailment": 0.22, "contradiction": 0.03, "neutral": 0.75},
                p2: {"entailment": 0.22, "contradiction": 0.03, "neutral": 0.75},
                joint: {"entailment": 0.82, "contradiction": 0.03, "neutral": 0.15},
            }
        )
        engine = VerificationEngine(
            nli_model=mock_nli,
            calibrator=self.calibrator,
            abstainer=self.abstainer,
        )
        claim = "Dr. Alvarez oversaw the clinical trial."
        candidate_passages = [
            {"doc_id": "hop1", "text": p1},
            {"doc_id": "hop2", "text": p2},
        ]
        verdict = engine.verify_claim(claim, candidate_passages=candidate_passages)

        self.assertTrue(verdict.has_meaningful_evidence)
        self.assertFalse(verdict.was_abstained)
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)

    # -------------------------------------------------------------------------
    # 6. Flat probability distributions trigger the intended policy
    # -------------------------------------------------------------------------
    def test_flat_probability_distribution_triggers_abstention(self):
        # Strict uniform distribution
        flat_probs = np.array([0.25, 0.25, 0.25, 0.25])
        should_abs, reason = self.abstainer.should_abstain(flat_probs)
        self.assertTrue(should_abs)
        self.assertIn("below threshold", reason)

        # Ambiguous distribution: confidence 0.360 in [0.32, 0.38] and margin 0.04 < 0.10 without documentary evidence
        ambiguous_probs = np.array([0.36, 0.32, 0.27, 0.05])
        should_abs_no_ev, reason_no_ev = self.abstainer.should_abstain(ambiguous_probs, has_meaningful_evidence=False)
        self.assertTrue(should_abs_no_ev)
        self.assertIn("Insufficient evidentiary grounding", reason_no_ev)

        # The same distribution with meaningful evidence passes through
        should_abs_with_ev, _ = self.abstainer.should_abstain(ambiguous_probs, has_meaningful_evidence=True)
        self.assertFalse(should_abs_with_ev)

    # -------------------------------------------------------------------------
    # 7. Frontend accurately distinguishes abstained result from confident verdict
    # -------------------------------------------------------------------------
    def test_frontend_payload_distinguishes_abstained_from_confident(self):
        from api.adapter import VeritasAPIAdapter
        adapter = VeritasAPIAdapter(lazy_init=True)

        # Abstained report
        abstained_verdict = ForensicVerdict(
            claim_id="VX-TEST-ABS",
            claim_text="Unsupported rumor claim",
            verdict=CanonicalVeracity.INSUFFICIENT_EVIDENCE,
            confidence=0.3510,
            uncalibrated_confidence=0.5638,
            was_abstained=True,
            abstention_reason="Insufficient evidentiary grounding: confidence 0.351 below 0.380 without supporting or refuting documentary evidence",
            class_probabilities={"VERIFIED_TRUE": 0.28, "VERIFIED_FALSE": 0.35, "CONFLICTING_OR_CHERRYPICKED": 0.10, "INSUFFICIENT_EVIDENCE": 0.27},
            temporal_report=TemporalAnalyzer().analyze(None, []),
            evidence_used_count=5,
            duplicates_filtered=0,
            irrelevant_passages_filtered=5,
            evidence_breakdown=[],
            raw_prediction=CanonicalVeracity.VERIFIED_FALSE.value,
            has_meaningful_evidence=False,
        )
        report_abs = adapter._format_report(abstained_verdict, t_infer_start=0.0, init_delta_ms=0.0, t_prep_ms=0.0, t_ret_ms=0.0, t_eng_ms=0.0)

        self.assertTrue(report_abs["abstained"])
        self.assertEqual(report_abs["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report_abs["raw_verdict"], "VERIFIED_FALSE")
        self.assertFalse(report_abs["has_meaningful_evidence"])
        self.assertIn("Abstained — Insufficient Evidentiary Grounding", report_abs["reason"])

        # Confident report
        confident_verdict = ForensicVerdict(
            claim_id="VX-TEST-CONF",
            claim_text="Confirmed scientific claim",
            verdict=CanonicalVeracity.VERIFIED_TRUE,
            confidence=0.8800,
            uncalibrated_confidence=0.9500,
            was_abstained=False,
            abstention_reason="",
            class_probabilities={"VERIFIED_TRUE": 0.88, "VERIFIED_FALSE": 0.04, "CONFLICTING_OR_CHERRYPICKED": 0.04, "INSUFFICIENT_EVIDENCE": 0.04},
            temporal_report=TemporalAnalyzer().analyze(None, []),
            evidence_used_count=1,
            duplicates_filtered=0,
            irrelevant_passages_filtered=0,
            evidence_breakdown=[{"doc_id": "d1", "text": "proof", "stance": "support", "entailment_prob": 0.95, "contradiction_prob": 0.01, "neutral_prob": 0.04, "is_joint": False}],
            raw_prediction=CanonicalVeracity.VERIFIED_TRUE.value,
            has_meaningful_evidence=True,
        )
        report_conf = adapter._format_report(confident_verdict, t_infer_start=0.0, init_delta_ms=0.0, t_prep_ms=0.0, t_ret_ms=0.0, t_eng_ms=0.0)

        self.assertFalse(report_conf["abstained"])
        self.assertEqual(report_conf["verdict"], "VERIFIED_TRUE")
        self.assertIn("supports the claim", report_conf["reason"])

    # -------------------------------------------------------------------------
    # 8. API response fields and calibration outputs remain consistent
    # -------------------------------------------------------------------------
    def test_api_response_fields_and_calibration_simplex(self):
        from api.adapter import VeritasAPIAdapter
        adapter = VeritasAPIAdapter(lazy_init=True)

        verdict = ForensicVerdict(
            claim_id="VX-CHECK",
            claim_text="Consistency check claim",
            verdict=CanonicalVeracity.VERIFIED_FALSE,
            confidence=0.75,
            uncalibrated_confidence=0.82,
            was_abstained=False,
            abstention_reason="",
            class_probabilities={"VERIFIED_TRUE": 0.10, "VERIFIED_FALSE": 0.75, "CONFLICTING_OR_CHERRYPICKED": 0.05, "INSUFFICIENT_EVIDENCE": 0.10},
            temporal_report=TemporalAnalyzer().analyze(None, []),
            evidence_used_count=1,
            duplicates_filtered=0,
            irrelevant_passages_filtered=0,
            evidence_breakdown=[],
            raw_prediction=CanonicalVeracity.VERIFIED_FALSE.value,
            has_meaningful_evidence=True,
        )
        report = adapter._format_report(verdict, t_infer_start=0.0, init_delta_ms=0.0, t_prep_ms=0.0, t_ret_ms=0.0, t_eng_ms=0.0)

        required_keys = {
            "claim_id", "claim", "verdict", "confidence", "probabilities",
            "has_cross_passage_disparity", "abstained", "reason", "evidence",
            "device", "latency_ms"
        }
        self.assertTrue(required_keys.issubset(report.keys()))

        # Check calibrated probabilities sum to 1.0 (simplex guarantee)
        prob_sum = sum(report["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)
        for p in report["probabilities"].values():
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

        # Check that abstained reports include raw_verdict and grounding fields
        abstained_verdict = ForensicVerdict(
            claim_id="VX-CHECK-ABS",
            claim_text="Consistency check claim",
            verdict=CanonicalVeracity.INSUFFICIENT_EVIDENCE,
            confidence=0.35,
            uncalibrated_confidence=0.55,
            was_abstained=True,
            abstention_reason="Insufficient evidentiary grounding: confidence 0.350 below 0.380",
            class_probabilities={"VERIFIED_TRUE": 0.28, "VERIFIED_FALSE": 0.35, "CONFLICTING_OR_CHERRYPICKED": 0.10, "INSUFFICIENT_EVIDENCE": 0.27},
            temporal_report=TemporalAnalyzer().analyze(None, []),
            evidence_used_count=1,
            duplicates_filtered=0,
            irrelevant_passages_filtered=0,
            evidence_breakdown=[],
            raw_prediction=CanonicalVeracity.VERIFIED_FALSE.value,
            has_meaningful_evidence=False,
        )
        report_abs = adapter._format_report(abstained_verdict)
        self.assertIn("raw_verdict", report_abs)
        self.assertIn("has_meaningful_evidence", report_abs)
        self.assertIn("abstention_reason", report_abs)


if __name__ == "__main__":
    unittest.main()
