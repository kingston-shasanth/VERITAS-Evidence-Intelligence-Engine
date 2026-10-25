"""
Comprehensive Regression Test Suite for VERITAS-X Systemic Root-Cause Fixes.
Addresses:
- P1: Noise-Resistant Evidence Aggregation & Metamorphic Invariance (META-02)
- P2: General Claim Decomposition & Semantic Copular/Paraphrase Alignment (META-04, EXP-13)
- P3: Epistemic Absence-of-Evidence Reasoning (AUDIT-09, AUDIT-11)
- P4: Fine-Grained Subclaim-to-Evidence Mapping & Provenance Isolation
- Multi-Hop Composition & Non-Equivalent Negative Checks
"""
import os
import sys
import unittest
from typing import List, Dict, Any
from unittest.mock import MagicMock

# Ensure src/ and repo root are on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from veritas.data.schema import CanonicalVeracity
from veritas.verification.engine import (
    VerificationEngine,
    _decompose_claim,
    _is_syntactic_verb,
    _tokens_match,
)
from veritas.verification.evidence_aggregator import (
    EvidenceAggregator,
    AggregatedEvidenceVerdict,
)
from api.adapter import VeritasAPIAdapter
from veritas.reasoning.types import SubClaimStatus


class TestP1NoiseResistantAggregation(unittest.TestCase):
    """Validates that EvidenceAggregator strictly enforces noise invariance."""

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

    def test_metamorphic_noise_invariance_single_supporter(self):
        """
        META-02 Root Cause Regression:
        Adding off-topic/ineligible evidence MUST NOT increase confidence or change verdict.
        Target: |Conf(Clean + Noise) - Conf(Clean)| <= 0.01
        """
        clean_passage = {
            "doc_id": "clean_01",
            "text": "Humans typically have two lungs located in the thoracic cavity.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
            "is_decision_eligible": True,
            "is_figurative": False,
        }
        clean_pred = {"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03}

        # 1. Evaluate clean evidence alone
        clean_verdict = self.aggregator.aggregate([clean_passage], [clean_pred])
        self.assertEqual(clean_verdict.canonical_verdict, CanonicalVeracity.VERIFIED_TRUE)
        base_confidence = clean_verdict.confidence

        # 2. Add an irrelevant/off-topic noise passage
        noise_passage = {
            "doc_id": "noise_01",
            "text": "The municipal water treatment plant was constructed in 1994 to serve 50,000 residents.",
            "is_relevant": False,
            "is_informative": True,
            "is_claim_aligned": False,
            "is_decision_eligible": False,
            "is_figurative": False,
        }
        noise_pred = {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.90}

        # Candidate joint prediction that DeBERTa would produce when concatenating clean + noise
        noisy_joint_pred = {"entailment": 0.94, "contradiction": 0.02, "neutral": 0.04}

        noisy_verdict = self.aggregator.aggregate(
            [clean_passage, noise_passage],
            [clean_pred, noise_pred],
            joint_nli_prediction=noisy_joint_pred,
        )

        self.assertEqual(noisy_verdict.canonical_verdict, CanonicalVeracity.VERIFIED_TRUE)
        delta_conf = abs(noisy_verdict.confidence - base_confidence)
        self.assertLessEqual(
            delta_conf,
            0.01,
            f"Noise inflation detected: base conf={base_confidence:.4f}, noisy conf={noisy_verdict.confidence:.4f}, delta={delta_conf:.4f}",
        )
        self.assertAlmostEqual(noisy_verdict.entailment_mass, clean_verdict.entailment_mass, places=4)

    def test_duplicate_evidence_does_not_manufacture_corroboration(self):
        """Duplicating near-identical text must be suppressed by Jaccard deduplication."""
        p1 = {
            "doc_id": "p1",
            "text": "The speed of light in vacuum is approximately 300,000 kilometers per second.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
            "is_decision_eligible": True,
        }
        p2 = {
            "doc_id": "p2",
            "text": "The speed of light in vacuum is approximately 300,000 kilometers per second.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
            "is_decision_eligible": True,
        }
        preds = [
            {"entailment": 0.90, "contradiction": 0.02, "neutral": 0.08},
            {"entailment": 0.90, "contradiction": 0.02, "neutral": 0.08},
        ]
        verdict = self.aggregator.aggregate([p1, p2], preds)
        self.assertEqual(verdict.duplicates_filtered, 1)
        self.assertEqual(verdict.unique_passages_used, 1)

    def test_independent_corroboration_strengthens_verdict(self):
        """Two genuine, distinct, eligible supporting passages must accumulate mass legitimately."""
        p1 = {
            "doc_id": "source_a",
            "text": "Seismic buoy networks observed seafloor displacement along the trench.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
            "is_decision_eligible": True,
        }
        p2 = {
            "doc_id": "source_b",
            "text": "Satellite altimetry independently recorded corresponding wave elevation spikes.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
            "is_decision_eligible": True,
        }
        preds = [
            {"entailment": 0.85, "contradiction": 0.03, "neutral": 0.12},
            {"entailment": 0.88, "contradiction": 0.02, "neutral": 0.10},
        ]

        single_v = self.aggregator.aggregate([p1], [preds[0]])
        multi_v = self.aggregator.aggregate([p1, p2], preds)

        self.assertGreater(multi_v.entailment_mass, single_v.entailment_mass)
        self.assertGreater(multi_v.confidence, single_v.confidence)

    def test_evidence_order_invariance_on_identical_stances(self):
        """Permuting the presentation order of passages must not alter the categorical veracity verdict."""
        p1 = {"doc_id": "a", "text": "Passage Alpha", "is_relevant": True, "is_informative": True, "is_claim_aligned": True, "is_decision_eligible": True}
        p2 = {"doc_id": "b", "text": "Passage Beta", "is_relevant": True, "is_informative": True, "is_claim_aligned": True, "is_decision_eligible": True}
        pred_a = {"entailment": 0.80, "contradiction": 0.05, "neutral": 0.15}
        pred_b = {"entailment": 0.80, "contradiction": 0.05, "neutral": 0.15}

        v_forward = self.aggregator.aggregate([p1, p2], [pred_a, pred_b])
        v_reversed = self.aggregator.aggregate([p2, p1], [pred_b, pred_a])

        self.assertEqual(v_forward.canonical_verdict, v_reversed.canonical_verdict)
        self.assertAlmostEqual(v_forward.confidence, v_reversed.confidence, places=3)


class TestP2ClaimDecompositionAndAlignment(unittest.TestCase):
    """Validates copular predicate handling, number words, and semantic alignment."""

    def test_copular_predicate_with_number_is_not_treated_as_verb(self):
        """
        META-04 Root Cause Regression:
        In 'The freezing point of water is zero degrees Celsius', 'zero' must NOT be in verb_tokens.
        """
        claim = "The freezing point of water is zero degrees Celsius."
        sub_subj, sub_pred, date_num, verb_tokens = _decompose_claim(claim)

        self.assertNotIn("zero", verb_tokens, "'zero' must not be classified as a syntactic verb")
        self.assertIn("zero", date_num, "'zero' must be recognized in quantitative date_num tokens")
        self.assertTrue("celsius" in sub_pred or "degrees" in sub_pred)

    def test_copular_number_passage_aligns_correctly(self):
        """Passage stating '0 °C' or '0 degrees' must align with 'zero degrees Celsius'."""
        claim = "The freezing point of water is zero degrees Celsius."
        passage = "At standard atmospheric pressure, the freezing point of water is 0 degrees Celsius."

        is_aligned = VerificationEngine.has_claim_relation_alignment(claim, passage)
        is_relevant = VerificationEngine.is_topically_relevant(claim, passage)
        is_eligible = VerificationEngine.is_decision_eligible(claim, passage)

        self.assertTrue(is_relevant, "Passage must be topically relevant")
        self.assertTrue(is_aligned, "Copular quantitative attribute must align")
        self.assertTrue(is_eligible, "Passage must be decision eligible")

    def test_paraphrase_preserves_alignment(self):
        """Paraphrased claim variants must maintain consistent alignment."""
        c1 = "The freezing point of water is zero degrees Celsius."
        c2 = "Water freezes at 0 degrees Celsius at sea level."
        passage = "Pure water freezes at 0 degrees Celsius under standard atmospheric pressure."

        self.assertTrue(VerificationEngine.has_claim_relation_alignment(c1, passage))
        self.assertTrue(VerificationEngine.has_claim_relation_alignment(c2, passage))

    def test_semantic_similarity_does_not_establish_entailment_negative(self):
        """Non-equivalent concepts (e.g. Megalodon size vs Blue Whale) must not align as supporting."""
        claim = "The prehistoric Megalodon reached a maximum body length exceeding 40 meters."
        counter_passage = "The largest marine animal is the blue whale, which reaches 30 meters in length."

        # Blue whale passage lacks Megalodon subject alignment
        self.assertFalse(VerificationEngine.is_topically_relevant(claim, counter_passage))

    def test_complex_medical_claim_decomposition(self):
        """AUDIT-07 Regression: 'Moderate daily coffee consumption decreases all-cause cardiovascular mortality.'"""
        claim = "Moderate daily coffee consumption decreases all-cause cardiovascular mortality."
        sub_subj, sub_pred, date_num, verb_tokens = _decompose_claim(claim)

        self.assertIn("coffee", sub_subj, "Subject must contain 'coffee'")
        self.assertIn("decreases", verb_tokens, "Verb must be 'decreases'")
        self.assertIn("mortality", sub_pred, "Predicate must contain 'mortality'")


class TestP3AbsenceOfEvidenceReasoning(unittest.TestCase):
    """Validates epistemic abstention on absence-of-evidence passages."""

    def test_absence_of_evidence_detection(self):
        """Asserts that lack-of-evidence expressions are recognized as epistemic uncertainty."""
        p1 = "To date, no direct evidence of microbial life has been detected in the plumes of Enceladus."
        p2 = "Surviving historical records make no mention of any private manuscript by Shakespeare."
        p3 = "It remains scientifically unproven whether string theory describes quantum gravity."

        self.assertTrue(VerificationEngine.is_absence_of_evidence(p1))
        self.assertTrue(VerificationEngine.is_absence_of_evidence(p2))
        self.assertTrue(VerificationEngine.is_absence_of_evidence(p3))

    def test_genuine_refutations_are_not_flagged_as_absence(self):
        """Direct counterevidence and physical impossibilities must NOT be marked as absence."""
        p1 = "An adult human skeleton normally consists of 206 distinct bones, not 450."
        p2 = "The Moon produces no light of its own through fusion; it strictly reflects sunlight."
        p3 = "Sound waves cannot propagate through a vacuum because mechanical waves require a medium."

        self.assertFalse(VerificationEngine.is_absence_of_evidence(p1))
        self.assertFalse(VerificationEngine.is_absence_of_evidence(p2))
        self.assertFalse(VerificationEngine.is_absence_of_evidence(p3))

    def test_absence_of_evidence_does_not_force_false_verdict(self):
        """When evidence merely states no proof was found, engine must abstain to INSUFFICIENT_EVIDENCE."""
        mock_nli = MagicMock()
        # Raw cross-encoder wrongly predicted 0.95 contradiction on 'no evidence found'
        mock_nli.predict_batch.return_value = [
            {"entailment": 0.02, "contradiction": 0.95, "neutral": 0.03}
        ]

        engine = VerificationEngine(retriever=None, nli_model=mock_nli)
        claim = "Extraterrestrial microbial life currently exists in the subsurface ocean of Enceladus."
        passages = [
            {
                "doc_id": "nasa_01",
                "text": "To date, no direct evidence of microbial life has been detected in the plumes or ocean of Enceladus.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            }
        ]

        verdict = engine.verify_claim(claim, candidate_passages=passages)
        self.assertEqual(
            verdict.verdict,
            CanonicalVeracity.INSUFFICIENT_EVIDENCE,
            f"Absence of evidence must abstain to INSUFFICIENT_EVIDENCE, got {verdict.verdict.value}",
        )


class TestP4SubclaimToEvidenceMapping(unittest.TestCase):
    """Validates that API Adapter maps evidence to constituent subclaims without broadcasting."""

    def test_compound_claim_evidence_isolation(self):
        """
        Compound claim: 'A and B'.
        Passage 1 supports A.
        Passage 2 refutes B.
        Passage 3 is irrelevant noise.
        Subclaim A must only have E1. Subclaim B must only have E2. E3 must be mapped to neither.
        """
        mock_engine = MagicMock()
        mock_engine.device = "cpu"
        mock_verdict = MagicMock()
        mock_verdict.claim_id = "TEST-CID"
        mock_verdict.verdict = CanonicalVeracity.VERIFIED_FALSE
        mock_verdict.confidence = 0.88
        mock_verdict.was_abstained = False
        mock_verdict.abstention_reason = ""
        mock_verdict.has_meaningful_evidence = True
        mock_verdict.duplicates_filtered = 0
        mock_verdict.irrelevant_passages_filtered = 1
        mock_verdict.temporal_report = MagicMock()
        mock_verdict.temporal_report.status.value = "CONSISTENT"
        mock_verdict.temporal_report.warning_message = None
        mock_verdict.temporal_report.delta_days = 0

        # Formatted evidence returned from verification breakdown
        mock_verdict.evidence_breakdown = [
            {
                "doc_id": "doc_a",
                "text": "The 1912 Olympic Games were held in Stockholm.",
                "weight": 1.0,
                "stance": "support",
                "entailment_prob": 0.92,
                "contradiction_prob": 0.02,
                "neutral_prob": 0.06,
                "is_joint": False,
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
                "is_decision_eligible": True,
                "claim_coverage": "partial",
                "claim_components_covered": ["Subject / Entity", "Venue / Location"],
            },
            {
                "doc_id": "doc_b",
                "text": "Electronic timing was first introduced in 1968, not at the 1912 Games.",
                "weight": 0.83,
                "stance": "refute",
                "entailment_prob": 0.03,
                "contradiction_prob": 0.91,
                "neutral_prob": 0.06,
                "is_joint": False,
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
                "is_decision_eligible": True,
                "claim_coverage": "partial",
                "claim_components_covered": ["Asserted Value / Quantity"],
            },
            {
                "doc_id": "doc_c_noise",
                "text": "Stockholm municipal water regulations were revised in 1980.",
                "weight": 0.0,
                "stance": "neutral",
                "entailment_prob": 0.05,
                "contradiction_prob": 0.05,
                "neutral_prob": 0.90,
                "is_joint": False,
                "is_relevant": False,
                "is_informative": True,
                "is_claim_aligned": False,
                "is_decision_eligible": False,
                "claim_coverage": "none",
                "claim_components_covered": [],
            },
        ]
        mock_engine.verify_claim.return_value = mock_verdict
        mock_engine.nli.predict_batch.return_value = [
            {"predicted_label": "ENTAILMENT", "entailment": 0.94, "contradiction": 0.02, "neutral": 0.04},
            {"predicted_label": "CONTRADICTION", "entailment": 0.02, "contradiction": 0.95, "neutral": 0.03},
        ]

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        claim = "The 1912 Olympic Games were held in Stockholm, and electronic timing was featured."

        result = adapter.verify(
            claim=claim,
            include_research=True,
            user_evidence=[
                "The 1912 Olympic Games were held in Stockholm.",
                "Electronic timing was first introduced in 1968, not at the 1912 Games.",
                "Stockholm municipal water regulations were revised in 1980.",
            ],
        )

        subclaims = result.get("subclaims", [])
        self.assertGreaterEqual(len(subclaims), 2, "Compound claim must decompose into at least 2 subclaims")

        sc1 = subclaims[0]
        sc2 = subclaims[1]

        # Verify that each subclaim is assigned ONLY its relevant evidence passage
        # and NEVER the irrelevant noise passage
        for sc in subclaims:
            self.assertNotIn("E-03", sc["evidence_ids"], "Irrelevant passage E-03 must never be mapped to subclaims")

        # SC1 (Stockholm venue) should be supported by E-01 only
        self.assertEqual(sc1["evidence_ids"], ["E-01"])
        self.assertEqual(sc1["status"], SubClaimStatus.SUPPORTED.value)
        self.assertEqual(sc1["stance_distribution"]["support"], 1)
        self.assertEqual(sc1["stance_distribution"]["refute"], 0)

        # SC2 (Electronic timing) should be contradicted by E-02 only
        self.assertEqual(sc2["evidence_ids"], ["E-02"])
        self.assertEqual(sc2["status"], SubClaimStatus.REFUTED.value)
        self.assertEqual(sc2["stance_distribution"]["refute"], 1)
        self.assertEqual(sc2["stance_distribution"]["support"], 0)


class TestGeneralizedEpistemicAbsenceOfEvidence(unittest.TestCase):
    """Validates generalized absence-of-evidence detection across diverse phrasing while preserving affirmative refutations."""

    def test_absence_with_intervening_adjectives_and_containment_verbs(self):
        phrases = [
            "Historical archives contain no surviving record or documented mention of the alleged decree.",
            "Excavations at the site yielded no verifiable physical evidence of any extraterrestrial impact.",
            "Clinical trials uncovered no observable evidence of adverse toxicity in the cohort.",
            "Modern astronomical surveys produced no detectable trace of the theorized companion star.",
            "Comprehensive investigations found no credible documentary evidence linking the minister to the contract.",
            "Biographical research shows no surviving personal correspondence mentioning the event.",
        ]
        for p in phrases:
            self.assertTrue(
                VerificationEngine.is_absence_of_evidence(p),
                f"Failed to identify absence-of-evidence in: '{p}'"
            )

    def test_explicit_refutations_preserved_not_neutralized(self):
        affirmative_refutations = [
            "The autopsy confirmed that the victim died of cyanide poisoning rather than natural causes.",
            "Geological core drilling conclusively disproved the impact hypothesis by finding terrestrial quartz.",
            "Radar telemetry proved the aircraft was stationed in Madrid at the time of the flight.",
            "Chemical assays verified the presence of synthetic arsenic in the blood samples.",
            "Eyewitness testimony confirmed the suspect was in custody during the robbery.",
        ]
        for p in affirmative_refutations:
            self.assertFalse(
                VerificationEngine.is_absence_of_evidence(p),
                f"Affirmative refutation was incorrectly flagged as absence of evidence: '{p}'"
            )


class TestSyntacticIrregularVerbsAndYearTokens(unittest.TestCase):
    """Validates irregular verb forms, year-to-word component matching, and non-blocking syntactic parsing."""

    def test_irregular_verbs_recognized(self):
        verbs = ["fought", "sought", "taught", "caught", "brought", "thought", "spent", "found", "built"]
        for v in verbs:
            self.assertTrue(_is_syntactic_verb(v), f"Expected irregular verb '{v}' to be recognized")

    def test_verb_following_four_digit_year(self):
        # 4-digit years in titles/names must NOT be treated as count quantifiers blocking subsequent verbs
        self.assertTrue(_is_syntactic_verb("fought", prev_token="1984"))
        self.assertTrue(_is_syntactic_verb("constructed", prev_token="1889"))
        self.assertTrue(_is_syntactic_verb("declared", prev_token="1776"))

    def test_year_to_word_token_matching(self):
        # 1984 -> nineteen, eighty, four
        self.assertTrue(_tokens_match("1984", "nineteen"))
        self.assertTrue(_tokens_match("1984", "eighty"))
        self.assertTrue(_tokens_match("1984", "four"))

        # 1889 -> eighteen, eighty, nine
        self.assertTrue(_tokens_match("1889", "eighteen"))
        self.assertTrue(_tokens_match("1889", "eighty"))
        self.assertTrue(_tokens_match("1889", "nine"))

        # 1776 -> seventeen, seventy, six
        self.assertTrue(_tokens_match("1776", "seventeen"))
        self.assertTrue(_tokens_match("1776", "seventy"))
        self.assertTrue(_tokens_match("1776", "six"))

        # Negative checks: mismatching years
        self.assertFalse(_tokens_match("1984", "seventeen"))
        self.assertFalse(_tokens_match("1889", "nineteen"))
        self.assertFalse(_tokens_match("1776", "eighty"))

    def test_decomposition_with_year_in_referring_expression(self):
        sub_subj, sub_pred, date_num_tokens, verb_tokens = _decompose_claim("The author of 1984 fought in the Spanish Civil War.")
        # Subject should contain '1984' and 'author'
        self.assertIn("1984", sub_subj)
        self.assertIn("author", sub_subj)
        # Predicate should contain 'fought' and 'civil'
        self.assertIn("fought", verb_tokens)
        self.assertIn("civil", sub_pred)


class TestMultiHopReasoningInvariance(unittest.TestCase):
    """Validates multi-hop bridge integrity and rejection of invalid bridges."""

    def test_valid_bridge_versus_invalid_bridge_discrimination(self):
        mock_nli = MagicMock()
        engine = VerificationEngine(nli_model=mock_nli)

        # Hop 1: George Orwell wrote 1984
        # Hop 2: George Orwell fought in the Spanish Civil War
        # Joint: The author of 1984 fought in the Spanish Civil War
        claim = "The author of 1984 fought in the Spanish Civil War."

        def mock_predict(pairs):
            results = []
            for prem, hyp in pairs:
                if "Hemingway" in prem:
                    results.append({"predicted_label": "CONTRADICTION", "entailment": 0.02, "contradiction": 0.96, "neutral": 0.02})
                elif "George Orwell" in prem and "1984" in prem and "Spanish Civil War" in prem:
                    results.append({"predicted_label": "ENTAILMENT", "entailment": 0.98, "contradiction": 0.01, "neutral": 0.01})
                else:
                    # Individual hops alone do not entail the compound relation
                    results.append({"predicted_label": "NEUTRAL", "entailment": 0.05, "contradiction": 0.05, "neutral": 0.90})
            return results

        mock_nli.predict_batch.side_effect = mock_predict

        # Case A: Valid bridge entity (George Orwell bridges Hop 1 and Hop 2)
        passages = [
            {"doc_id": "hop_01", "text": "George Orwell is the English author who wrote the dystopian novel 1984."},
            {"doc_id": "hop_02", "text": "George Orwell fought in the Spanish Civil War as part of the POUM militia."}
        ]
        res_valid = engine.verify_claim(claim, candidate_passages=passages)
        self.assertEqual(res_valid.verdict, CanonicalVeracity.VERIFIED_TRUE)

        # Case B: Invalid bridge entity (swapped entity: Hemingway does not bridge author of 1984)
        invalid_passages = [
            {"doc_id": "hop_01", "text": "George Orwell is the English author who wrote the dystopian novel 1984."},
            {"doc_id": "hop_02", "text": "Ernest Hemingway reported on and fought in the Spanish Civil War."}
        ]
        res_invalid = engine.verify_claim(claim, candidate_passages=invalid_passages)
        self.assertNotEqual(res_invalid.verdict, CanonicalVeracity.VERIFIED_TRUE)


class TestCausalAndCompoundConjunctionRegressions(unittest.TestCase):
    """
    Deterministic regressions for the four demonstrated reliability defect classes:
    1. Chocolate / Nobel / Intelligence (Association != Causation, Compound Conjunction Abstention)
    2. Hospital / Death-Rate Causal Claim (Observational/Confounded != Falsity, Abstain)
    3. Great Wall Compound Claim (Truth-Functional Conjunction: Partial -> Abstain; Refuted -> Verified False)
    4. Lighter / Lung-Cancer Claim (Confounded Observational Evidence -> Abstain)
    5. Smoking / Lung-Cancer Claim (Explicit Randomized/Causal Proof -> Verified True)
    """

    def test_chocolate_nobel_causal_compound_abstention(self):
        """Evidence of correlation between chocolate and Nobel laureates must not validate causation."""
        mock_nli = MagicMock()
        def mock_predict(pairs):
            results = []
            for prem, hyp in pairs:
                hl = hyp.lower()
                pl = prem.lower()
                if "correlat" in hl and "correlat" in pl:
                    results.append({"predicted_label": "ENTAILMENT", "entailment": 0.96, "contradiction": 0.02, "neutral": 0.02})
                elif "cause" in hl:
                    # Raw cross-encoder has high spurious entailment on token overlap
                    results.append({"predicted_label": "ENTAILMENT", "entailment": 0.92, "contradiction": 0.03, "neutral": 0.05})
                else:
                    results.append({"predicted_label": "NEUTRAL", "entailment": 0.05, "contradiction": 0.05, "neutral": 0.90})
            return results

        mock_nli.predict_batch.side_effect = mock_predict
        engine = VerificationEngine(nli_model=mock_nli)

        claim = "Chocolate consumption correlates with Nobel laureates and causes higher intelligence."
        passages = [{
            "doc_id": "nejm_01",
            "text": "A 2012 study in the New England Journal of Medicine observed a strong correlation between per capita chocolate consumption and the number of Nobel laureates per capita in various nations.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
        }]

        verdict = engine.verify_claim(claim, candidate_passages=passages)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertTrue(verdict.was_abstained)
        self.assertIn("lacks documentary evidence", verdict.abstention_reason)

    def test_hospital_admission_mortality_causal_abstention(self):
        """Observational correlation or lack of evidence must not be treated as proof of falsity."""
        mock_nli = MagicMock()
        mock_nli.predict_batch.return_value = [
            # Raw cross-encoder might predict contradiction or spurious entailment
            {"predicted_label": "CONTRADICTION", "entailment": 0.05, "contradiction": 0.92, "neutral": 0.03}
        ]
        engine = VerificationEngine(nli_model=mock_nli)

        claim = "Higher hospital admission rates cause increased patient mortality."
        passages = [{
            "doc_id": "hosp_01",
            "text": "Observational data from regional health networks indicates that hospital admission rates correlate with increased patient mortality, because patients admitted to hospitals are significantly more ill. No clinical trial demonstrates that hospital admission causes mortality.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
        }]

        verdict = engine.verify_claim(claim, candidate_passages=passages)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertTrue(verdict.was_abstained)
        self.assertNotEqual(verdict.verdict, CanonicalVeracity.VERIFIED_FALSE)

    def test_great_wall_compound_conjunction_partial_evidence(self):
        """Partial evidence supporting only Ming Dynasty construction must NOT validate visibility from the Moon."""
        mock_nli = MagicMock()
        def mock_predict(pairs):
            results = []
            for prem, hyp in pairs:
                hl = hyp.lower()
                pl = prem.lower()
                if "ming dynasty" in hl and "ming dynasty" in pl:
                    results.append({"predicted_label": "ENTAILMENT", "entailment": 0.97, "contradiction": 0.01, "neutral": 0.02})
                elif "moon" in hl:
                    # Unevidenced in Ming Dynasty passage
                    results.append({"predicted_label": "NEUTRAL", "entailment": 0.05, "contradiction": 0.05, "neutral": 0.90})
                else:
                    results.append({"predicted_label": "ENTAILMENT", "entailment": 0.90, "contradiction": 0.05, "neutral": 0.05})
            return results

        mock_nli.predict_batch.side_effect = mock_predict
        engine = VerificationEngine(nli_model=mock_nli)

        claim = "The Great Wall of China is visible from the Moon with the naked eye, and it was constructed under the Ming Dynasty."
        passages = [{
            "doc_id": "gw_01",
            "text": "The existing Great Wall of China was extensively constructed and fortified during the Ming Dynasty (1368-1644).",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
        }]

        verdict = engine.verify_claim(claim, candidate_passages=passages)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertTrue(verdict.was_abstained)
        self.assertIn("lacks documentary evidence", verdict.abstention_reason)

    def test_great_wall_compound_conjunction_refuted_constituent(self):
        """Refutation of lunar visibility must falsify the entire compound conjunction."""
        mock_nli = MagicMock()
        def mock_predict(pairs):
            results = []
            for prem, hyp in pairs:
                hl = hyp.lower()
                pl = prem.lower()
                if "moon" in hl and "not visible" in pl:
                    results.append({"predicted_label": "CONTRADICTION", "entailment": 0.01, "contradiction": 0.98, "neutral": 0.01})
                elif "ming dynasty" in hl and "ming dynasty" in pl:
                    results.append({"predicted_label": "ENTAILMENT", "entailment": 0.97, "contradiction": 0.01, "neutral": 0.02})
                else:
                    results.append({"predicted_label": "NEUTRAL", "entailment": 0.05, "contradiction": 0.05, "neutral": 0.90})
            return results

        mock_nli.predict_batch.side_effect = mock_predict
        engine = VerificationEngine(nli_model=mock_nli)

        claim = "The Great Wall of China is visible from the Moon with the naked eye, and it was constructed under the Ming Dynasty."
        passages = [
            {
                "doc_id": "gw_refute",
                "text": "Apollo astronauts confirmed that the Great Wall of China is not visible from the Moon with the naked eye.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "gw_support",
                "text": "The Great Wall of China was constructed under the Ming Dynasty.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            }
        ]

        verdict = engine.verify_claim(claim, candidate_passages=passages)
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_FALSE)
        self.assertFalse(verdict.was_abstained)
        self.assertIn("is refuted by documentary evidence", verdict.abstention_reason)

    def test_lighter_lung_cancer_confounded_abstention(self):
        """Confounded observational evidence warrants honest abstention."""
        mock_nli = MagicMock()
        mock_nli.predict_batch.return_value = [
            {"predicted_label": "ENTAILMENT", "entailment": 0.94, "contradiction": 0.03, "neutral": 0.03}
        ]
        engine = VerificationEngine(nli_model=mock_nli)

        claim = "Carrying matches or a lighter causes lung cancer."
        passages = [{
            "doc_id": "light_01",
            "text": "Epidemiological studies observe an association between carrying lighters or matches and lung cancer incidence; however, this correlation is heavily confounded by cigarette smoking behavior.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
        }]

        verdict = engine.verify_claim(claim, candidate_passages=passages)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertTrue(verdict.was_abstained)

    def test_genuine_causality_with_explicit_trial_proof_verified_true(self):
        """Explicit randomized controlled trial and causal proof validates causal proposition."""
        mock_nli = MagicMock()
        mock_nli.predict_batch.return_value = [
            {"predicted_label": "ENTAILMENT", "entailment": 0.98, "contradiction": 0.01, "neutral": 0.01}
        ]
        engine = VerificationEngine(nli_model=mock_nli)

        claim = "Cigarette smoking causes lung cancer."
        passages = [{
            "doc_id": "smok_01",
            "text": "Extensive randomized controlled trials and causal epidemiological research conclusively proven that cigarette smoking causes lung cancer through DNA-damaging carcinogens.",
            "is_relevant": True,
            "is_informative": True,
            "is_claim_aligned": True,
        }]

        verdict = engine.verify_claim(claim, candidate_passages=passages)
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertFalse(verdict.was_abstained)


class TestEvidenceConsistencyAndNeutralInvariance(unittest.TestCase):
    """
    Deterministic regressions for verdict-evidence consistency and neutral invariance:
    1. Electricity / Economic Growth defect class: 5 neutral passages must NOT yield VERIFIED_FALSE.
    2. Insufficient evidence cannot become VERIFIED_FALSE solely from neutral passages.
    3. Constituent-level refutation properly justifies VERIFIED_FALSE with documented evidence path.
    4. Adapter explanation consistency: reason must match final decision and eligible evidence.
    """

    def test_electricity_economic_growth_neutral_passages_not_verified_false(self):
        """
        Reproduced defect test:
        Electricity/economic-growth claim evaluated against 5 neutral candidate passages
        must produce INSUFFICIENT_EVIDENCE, not VERIFIED_FALSE, and explanation must not claim contradiction.
        """
        mock_nli = MagicMock()
        # All 5 passages are neutral context with slight non-zero noise
        mock_nli.predict_batch.return_value = [
            {"predicted_label": "NEUTRAL", "entailment": 0.12, "contradiction": 0.28, "neutral": 0.60},
            {"predicted_label": "NEUTRAL", "entailment": 0.10, "contradiction": 0.26, "neutral": 0.64},
            {"predicted_label": "NEUTRAL", "entailment": 0.14, "contradiction": 0.27, "neutral": 0.59},
            {"predicted_label": "NEUTRAL", "entailment": 0.09, "contradiction": 0.25, "neutral": 0.66},
            {"predicted_label": "NEUTRAL", "entailment": 0.11, "contradiction": 0.29, "neutral": 0.60},
            # Joint premise prediction
            {"predicted_label": "NEUTRAL", "entailment": 0.10, "contradiction": 0.42, "neutral": 0.48},
        ]
        engine = VerificationEngine(nli_model=mock_nli)

        claim = "Electricity generation and consumption cause economic growth."
        passages = [
            {"doc_id": "doc_el_01", "text": "Electricity generation from coal increased by 43 per cent in 2019 according to ministry data.", "is_relevant": True, "is_informative": True, "is_claim_aligned": True},
            {"doc_id": "doc_el_02", "text": "Economic growth remained steady throughout the previous fiscal quarter across OECD countries.", "is_relevant": True, "is_informative": True, "is_claim_aligned": True},
            {"doc_id": "doc_el_03", "text": "Energy prices fluctuated in New Zealand following regulatory changes in the gas market.", "is_relevant": True, "is_informative": True, "is_claim_aligned": True},
            {"doc_id": "doc_el_04", "text": "The power grid requires continuous maintenance to prevent blackouts during peak winter hours.", "is_relevant": True, "is_informative": True, "is_claim_aligned": True},
            {"doc_id": "doc_el_05", "text": "A report on infrastructure spending was submitted to the parliamentary committee on energy.", "is_relevant": True, "is_informative": True, "is_claim_aligned": True},
        ]

        verdict = engine.verify_claim(claim, candidate_passages=passages)

        # Must NOT be VERIFIED_FALSE
        self.assertNotEqual(verdict.verdict, CanonicalVeracity.VERIFIED_FALSE)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertTrue(verdict.was_abstained)

        # Zero refuting passages in evidence breakdown
        ref_count = len([e for e in verdict.evidence_breakdown if e.get("stance") == "refute" and e.get("is_decision_eligible", False)])
        sup_count = len([e for e in verdict.evidence_breakdown if e.get("stance") == "support" and e.get("is_decision_eligible", False)])
        self.assertEqual(ref_count, 0)
        self.assertEqual(sup_count, 0)

        # Explanation must not claim evidence contradicts
        self.assertNotIn("directly contradicts", verdict.abstention_reason.lower())
        self.assertIn("insufficient", verdict.abstention_reason.lower())

    def test_insufficient_evidence_cannot_become_verified_false_solely_from_neutral_passages(self):
        """
        Verification pipeline must not assign VERIFIED_FALSE when candidate passages are neutral,
        ensuring insufficient evidence cannot become VERIFIED_FALSE solely from neutral passages.
        """
        mock_nli = MagicMock()
        # Predictions where neutral dominates over individual contradiction, but contradiction > entailment
        mock_nli.predict_batch.return_value = [
            {"predicted_label": "NEUTRAL", "entailment": 0.12, "contradiction": 0.28, "neutral": 0.60}
            for _ in range(6)
        ]
        engine = VerificationEngine(nli_model=mock_nli)
        claim = "Global mineral extraction increased steadily during the twentieth century."
        passages = [
            {"doc_id": f"p_{i}", "text": f"General background geological text discussing mining {i}.", "is_relevant": True, "is_informative": True, "is_claim_aligned": True}
            for i in range(5)
        ]

        verdict = engine.verify_claim(claim, candidate_passages=passages)
        self.assertEqual(verdict.verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertNotEqual(verdict.verdict, CanonicalVeracity.VERIFIED_FALSE)
        self.assertTrue(verdict.was_abstained)
        self.assertTrue(bool(verdict.abstention_reason))

    def test_constituent_level_refutation_justifies_verified_false_with_documented_evidence_path(self):
        """
        When a constituent proposition of a compound claim is explicitly refuted by documentary evidence,
        VERIFIED_FALSE IS justified, and the exact evidence path must be documented.
        """
        mock_nli = MagicMock()
        def mock_predict(pairs):
            results = []
            for prem, hyp in pairs:
                hl = hyp.lower()
                pl = prem.lower()
                if "never stepped on the moon" in hl and "walked on the lunar surface" in pl:
                    results.append({"predicted_label": "CONTRADICTION", "entailment": 0.01, "contradiction": 0.98, "neutral": 0.01})
                elif "born in ohio" in hl and "born in wapakoneta" in pl:
                    results.append({"predicted_label": "ENTAILMENT", "entailment": 0.95, "contradiction": 0.02, "neutral": 0.03})
                else:
                    results.append({"predicted_label": "NEUTRAL", "entailment": 0.10, "contradiction": 0.10, "neutral": 0.80})
            return results

        mock_nli.predict_batch.side_effect = mock_predict
        engine = VerificationEngine(nli_model=mock_nli)

        claim = "Neil Armstrong was born in Ohio, and he never stepped on the Moon."
        passages = [
            {
                "doc_id": "p_nasa",
                "text": "Historical telemetry and video transmission confirm Neil Armstrong walked on the lunar surface on July 20, 1969.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            },
            {
                "doc_id": "p_bio",
                "text": "Neil Armstrong was born in Wapakoneta, Ohio, on August 5, 1930.",
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
            }
        ]

        verdict = engine.verify_claim(claim, candidate_passages=passages)
        # Here VERIFIED_FALSE is legitimately justified by constituent refutation
        self.assertEqual(verdict.verdict, CanonicalVeracity.VERIFIED_FALSE)
        self.assertFalse(verdict.was_abstained)
        self.assertIn("refuted by documentary evidence", verdict.abstention_reason)

    def test_api_adapter_payload_consistency_for_neutral_evidence(self):
        """
        API adapter response payload must have mutually consistent verdict, evidence counts,
        and reason strings when candidate passages are neutral.
        """
        from api.adapter import VeritasAPIAdapter
        adapter = VeritasAPIAdapter(device_override="cpu", lazy_init=True)

        # Mock engine verify_claim to return INSUFFICIENT_EVIDENCE with neutral evidence breakdown
        mock_engine = MagicMock()
        mock_verdict = MagicMock()
        mock_verdict.verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE
        mock_verdict.confidence = 0.328
        mock_verdict.was_abstained = True
        mock_verdict.abstention_reason = "Insufficient evidentiary grounding: no supporting or refuting documentary evidence."
        mock_verdict.class_probabilities = {
            "VERIFIED_TRUE": 0.20,
            "VERIFIED_FALSE": 0.24,
            "CONFLICTING_OR_CHERRYPICKED": 0.01,
            "INSUFFICIENT_EVIDENCE": 0.55,
        }
        mock_verdict.evidence_breakdown = [
            {
                "doc_id": f"doc_{i}",
                "text": f"Background energy text {i}",
                "stance": "neutral",
                "entailment_prob": 0.10,
                "contradiction_prob": 0.20,
                "neutral_prob": 0.70,
                "is_relevant": True,
                "is_informative": True,
                "is_claim_aligned": True,
                "is_decision_eligible": False,
                "is_joint": False,
                "is_figurative": False,
                "claim_coverage": "full",
                "claim_components_covered": [],
            }
            for i in range(5)
        ]
        mock_verdict.unique_passages_used = 5
        mock_verdict.duplicates_filtered = 0
        mock_engine.verify_claim.return_value = mock_verdict
        adapter._engine = mock_engine

        res = adapter.verify("Electricity generation and consumption cause economic growth.", mode="local")

        self.assertEqual(res["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(res["abstained"])
        self.assertEqual(res["evidence_counts"]["eligible_supporting"], 0)
        self.assertEqual(res["evidence_counts"]["eligible_refuting"], 0)
        self.assertEqual(res["evidence_counts"]["relevant_neutral"], 5)
        self.assertIsInstance(res["reason"], str)
        self.assertNotIn("directly contradicts", res["reason"])
        self.assertIn("Insufficient", res["reason"])


if __name__ == "__main__":
    unittest.main()


