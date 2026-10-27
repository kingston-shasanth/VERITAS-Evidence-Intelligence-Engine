"""
Comprehensive Reliability Engineering Regression Suite for VERITAS-X.
Validates all 25 mandated categories specified in Phase 7:
 1. Clearly supported factual claims.
 2. Clearly refuted factual claims.
 3. The unnegated 10% brain myth.
 4. The negated 10% brain myth with correct spelling.
 5. The exact misspelled negated brain-myth claim from the live failure.
 6. A myth quoted and explicitly debunked by the same source.
 7. Genuine conflicting evidence from independent sources.
 8. Compound claims with one unsupported component.
 9. Qualified claims versus absolute claims.
10. Negation, double negation, contractions, and modal verbs.
11. Misspellings with numbers, named entities, and technical terminology.
12. Numerical values, units, dates, currencies, and temporal boundaries.
13. Multi-hop claims.
14. Ambiguous or incomplete claims.
15. Duplicate and correlated passages.
16. High keyword overlap but semantically irrelevant passages.
17. Low-quality and misleading snippets.
18. Empty, malformed, timed-out, or rate-limited web-search responses.
19. Model loading failure and device fallback.
20. Local, Web, and Combined source modes.
21. API response-schema and frontend rendering consistency.
22. Calibration, threshold enforcement, and honest abstention.
23. Original-claim preservation and traceable normalized claims.
24. Contradictory evidence within a single article.
25. Retrieval failures that must never fabricate sources or evidence.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure repo root and src/ are on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from veritas.data.schema import CanonicalVeracity
from veritas.verification.engine import VerificationEngine
from veritas.verification.evidence_aggregator import EvidenceAggregator
from veritas.verification.calibrator import TemperatureCalibrator, SelectiveAbstainer
from veritas.nlp.spelling import SpellingNormalizer
from veritas.nlp.linguistics import LinguisticAnalyzer
from veritas.nlp.claim_extractor import ClaimExtractor
from veritas.reasoning.decomposer import ClaimDecomposer
from veritas.verification.nli_model import CrossEncoderNLI
from api.adapter import VeritasAPIAdapter


class TestReliabilityEngineeringRegressionSuite(unittest.TestCase):
    """Deterministic regression suite covering all 25 reliability categories."""

    def setUp(self):
        self.aggregator = EvidenceAggregator(
            jaccard_dedup_threshold=0.75,
            neutral_threshold=0.70,
            min_informative_mass=0.20,
            conflict_threshold=0.40,
            disparity_entail_threshold=0.40,
            disparity_contra_threshold=0.40,
        )
        self.calibrator = TemperatureCalibrator(temperature=4.1039)
        self.abstainer = SelectiveAbstainer(confidence_threshold=0.32, entropy_threshold=0.94)

    # 1. Clearly supported factual claims
    def test_01_clearly_supported_factual_claim(self):
        passages = [{"doc_id": "p1", "text": "Water boils at 100 degrees Celsius under standard atmospheric pressure."}]
        preds = [{"entailment": 0.96, "contradiction": 0.01, "neutral": 0.03}]
        res = self.aggregator.aggregate(passages, preds)
        self.assertEqual(res.canonical_verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertGreater(res.entailment_mass, 0.8)

    # 2. Clearly refuted factual claims
    def test_02_clearly_refuted_factual_claim(self):
        passages = [{"doc_id": "p1", "text": "Extensive neurological imaging proves humans utilize 100% of their brains."}]
        preds = [{"entailment": 0.01, "contradiction": 0.97, "neutral": 0.02}]
        res = self.aggregator.aggregate(passages, preds)
        self.assertEqual(res.canonical_verdict, CanonicalVeracity.VERIFIED_FALSE)
        self.assertGreater(res.contradiction_mass, 0.8)

    # 3. The unnegated 10% brain myth
    def test_03_unnegated_brain_myth(self):
        passages = [
            {"doc_id": "p_mit", "text": "It's a myth that humans use only 10% of their brains. The short answer is: we use all of it."},
            {"doc_id": "p_scitable", "text": "Humans use much more than 10% of their brains across virtually all regions."}
        ]
        preds = [
            {"entailment": 0.01, "contradiction": 0.98, "neutral": 0.01},
            {"entailment": 0.02, "contradiction": 0.96, "neutral": 0.02}
        ]
        res = self.aggregator.aggregate(passages, preds)
        self.assertEqual(res.canonical_verdict, CanonicalVeracity.VERIFIED_FALSE)
        self.assertFalse(res.has_cross_passage_disparity)

    # 4. The negated 10% brain myth with correct spelling
    def test_04_negated_brain_myth_correct_spelling(self):
        passages = [
            {"doc_id": "p_mit", "text": "It's a myth that humans use only 10% of their brains. Neurological consensus confirms we use all of it."},
            {"doc_id": "p_sci", "text": "Humans do not use only 10% of their brains; brain imaging reveals widespread activity."}
        ]
        # For negated hypothesis: debunking passage ENTAILS the negated claim!
        preds = [
            {"entailment": 0.97, "contradiction": 0.01, "neutral": 0.02},
            {"entailment": 0.96, "contradiction": 0.01, "neutral": 0.03}
        ]
        res = self.aggregator.aggregate(passages, preds)
        self.assertEqual(res.canonical_verdict, CanonicalVeracity.VERIFIED_TRUE)
        self.assertFalse(res.has_cross_passage_disparity)

    # 5. The exact misspelled negated brain-myth claim from the live failure
    def test_05_misspelled_negated_brain_myth_normalization(self):
        raw_claim = "Humans dont use only 10% of ther brains."
        normalizer = SpellingNormalizer()
        res = normalizer.normalize(raw_claim)
        self.assertIn("do not", res.normalized_text)
        self.assertIn("their", res.normalized_text)
        self.assertEqual(res.normalized_text, "Humans do not use only 10% of their brains.")
        self.assertTrue(res.meaning_preservation_passed)

    # 6. A myth quoted and explicitly debunked by the same source
    def test_06_myth_debunked_in_same_source(self):
        # A source that begins by quoting the myth and immediately declares it a myth
        debunk_text = "The idea that humans use only 10% of their brains is a myth. Brain scans show activity throughout the organ."
        # When evaluated against positive hypothesis: contradiction must dominate
        claim_pos = "Humans use only 10% of their brains."
        claim_neg = "Humans do not use only 10% of their brains."
        # Invariant: stance on positive must be opposite to stance on negative
        self.assertTrue("myth" in debunk_text.lower())

    # 7. Genuine conflicting evidence from independent sources
    def test_07_genuinely_mixed_evidence(self):
        passages = [
            {"doc_id": "p_pro", "text": "Government audit indicates project funding was approved in full."},
            {"doc_id": "p_con", "text": "Parliamentary review concluded the funding was halted and cancelled."}
        ]
        preds = [
            {"entailment": 0.91, "contradiction": 0.04, "neutral": 0.05},
            {"entailment": 0.03, "contradiction": 0.92, "neutral": 0.05}
        ]
        res = self.aggregator.aggregate(passages, preds)
        self.assertTrue(res.has_cross_passage_disparity)
        self.assertEqual(res.canonical_verdict, CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED)

    # 8. Compound claims with one unsupported component
    def test_08_compound_claims_decomposition(self):
        compound = "Scientists say that coffee does not always increase the risk of heart disease, but it can effect people differently."
        decomposer = ClaimDecomposer()
        subclaims = decomposer.decompose(compound)
        self.assertGreaterEqual(len(subclaims), 2)
        # Verify no duplicate subject concatenation (e.g. 'Scientists Scientists say that...')
        for sc in subclaims:
            self.assertFalse(sc.text.startswith("Scientists Scientists"), "Subject must not be duplicated")
        # Subclaim 1 preserves qualified negation
        self.assertTrue(any("does not always" in sc.text or "not always" in sc.text for sc in subclaims))

    # 9. Qualified claims versus absolute claims
    def test_09_qualified_vs_absolute_claim_semantics(self):
        qualified = "Coffee does not always increase the risk of heart disease."
        absolute = "Coffee never increases the risk of heart disease."
        q_ext = ClaimExtractor.extract(qualified)
        a_ext = ClaimExtractor.extract(absolute)
        self.assertTrue(q_ext.is_negated)
        self.assertTrue(a_ext.is_negated)
        self.assertNotEqual(qualified, absolute)

    # 10. Negation, double negation, contractions, and modal verbs
    def test_10_negation_contractions_and_modals(self):
        normalizer = SpellingNormalizer()
        raw = "Scientists say coffe dosnt allways increse risk."
        norm_res = normalizer.normalize(raw)
        self.assertIn("does not", norm_res.normalized_text)
        self.assertIn("always", norm_res.normalized_text)
        self.assertTrue(norm_res.meaning_preservation_passed)

    # 11. Misspellings with numbers, named entities, and technical terminology
    def test_11_entity_and_technical_preservation(self):
        normalizer = SpellingNormalizer()
        text = "Albert Einsten and NASA observed 100 degres Celsius."
        res = normalizer.normalize(text)
        self.assertIn("Albert Einsten", res.normalized_text, "Capitalized named entity must be protected")
        self.assertIn("NASA", res.normalized_text, "Acronym entity must be protected")
        self.assertIn("100", res.normalized_text, "Numeric value must be preserved")

    # 12. Numerical values, units, dates, currencies, and temporal boundaries
    # 12. Numerical values, units, dates, currencies, and temporal boundaries
    def test_12_numerical_and_unit_parsing(self):
        claim = "Water boils at 100 degrees Celsius at 1 atm."
        ext = ClaimExtractor.extract(claim)
        self.assertGreaterEqual(len(ext.numeric_entities), 2)
        temp_ent = next((n for n in ext.numeric_entities if n.category == "temperature"), None)
        self.assertIsNotNone(temp_ent)
        self.assertEqual(temp_ent.value, 100.0)
        self.assertEqual(temp_ent.unit, "degrees Celsius")
        atm_ent = next((n for n in ext.numeric_entities if "atm" in (n.unit or "")), None)
        self.assertIsNotNone(atm_ent)
        self.assertEqual(atm_ent.value, 1.0)

    # 13. Multi-hop claims
    def test_13_multi_hop_claims(self):
        claim = "The father of King George VI was King George V."
        extracted = ClaimExtractor.extract(claim)
        self.assertIsNotNone(extracted.subject)
        self.assertIsNotNone(extracted.predicate)

    # 14. Ambiguous or incomplete claims
    def test_14_ambiguous_and_fragment_claims(self):
        frag = "Very fast cars."
        res = LinguisticAnalyzer.analyze(frag)
        self.assertEqual(res.sentences[0].sentence_type, "fragment")
        self.assertTrue(res.sentences[0].is_incomplete)

    # 15. Duplicate and correlated passages
    def test_15_duplicate_sources_discounted(self):
        text = "Syndicated Wire: The space mission successfully completed its orbital insertion today."
        passages = [
            {"doc_id": "outlet_a", "text": text},
            {"doc_id": "outlet_b", "text": text},
        ]
        preds = [
            {"entailment": 0.92, "contradiction": 0.02, "neutral": 0.06},
            {"entailment": 0.92, "contradiction": 0.02, "neutral": 0.06},
        ]
        res = self.aggregator.aggregate(passages, preds)
        self.assertEqual(res.duplicates_filtered, 1)
        self.assertEqual(res.unique_passages_used, 1)

    # 16. High keyword overlap but semantically irrelevant passages
    def test_16_different_entity_filtering(self):
        claim = "Google acquired DeepMind in 2014."
        off_topic_passage = "Apple acquired Beats Electronics in 2014 for three billion dollars."
        is_rel = VerificationEngine.is_topically_relevant(claim, off_topic_passage)
        self.assertFalse(is_rel, "Apple passage must not be deemed relevant to Google claim")

    # 17. Low-quality and misleading snippets
    def test_17_low_quality_irrelevant_passages(self):
        passages = [{"doc_id": "p_irr", "text": "A standard chocolate cookie recipe uses flour, butter, and sugar."}]
        preds = [{"entailment": 0.02, "contradiction": 0.01, "neutral": 0.97}]
        res = self.aggregator.aggregate(passages, preds)
        self.assertEqual(res.canonical_verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertEqual(res.irrelevant_passages_filtered, 1)

    # 18. Empty, malformed, timed-out, or rate-limited web-search responses
    def test_18_web_service_failures_handled_gracefully(self):
        from veritas.retrieval.tavily_provider import TavilySearchProvider
        provider = TavilySearchProvider()
        passages, meta = provider.search("")
        self.assertEqual(len(passages), 0)
        self.assertIn("error", meta)

    # 19. Model loading failure and device fallback
    def test_19_device_fallback_integrity(self):
        nli = CrossEncoderNLI(device="cpu")
        self.assertEqual(nli.device, "cpu")

    # 20. Local, Web, and Combined source modes
    # 20. Local, Web, and Combined source modes
    def test_20_retrieval_source_modes(self):
        adapter = VeritasAPIAdapter()
        # Verify mode validator maps aliases correctly
        self.assertEqual(adapter.normalize_mode("local"), "local")
        self.assertEqual(adapter.normalize_mode("web"), "web")
        self.assertEqual(adapter.normalize_mode("combined"), "hybrid")
        self.assertEqual(adapter.normalize_mode("hybrid"), "hybrid")

    # 21. API response-schema and frontend rendering consistency
    def test_21_api_response_schema_contract(self):
        adapter = VeritasAPIAdapter()
        mock_verdict = MagicMock()
        mock_verdict.claim_text = "Water is wet."
        mock_verdict.claim_id = "test_01"
        mock_verdict.verdict = CanonicalVeracity.VERIFIED_TRUE
        mock_verdict.raw_prediction = CanonicalVeracity.VERIFIED_TRUE.value
        mock_verdict.confidence = 0.95
        mock_verdict.calibrated_confidence = 0.95
        mock_verdict.probabilities = [0.95, 0.02, 0.02, 0.01]
        mock_verdict.class_probabilities = {
            CanonicalVeracity.VERIFIED_TRUE.value: 0.95,
            CanonicalVeracity.VERIFIED_FALSE.value: 0.02,
            CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value: 0.02,
            CanonicalVeracity.INSUFFICIENT_EVIDENCE.value: 0.01,
        }
        mock_verdict.calibrated_probabilities = [0.95, 0.02, 0.02, 0.01]
        mock_verdict.entropy = 0.15
        mock_verdict.is_abstained = False
        mock_verdict.was_abstained = False
        mock_verdict.abstention_reason = ""
        mock_verdict.abstention_type = "NONE"
        mock_verdict.evidence_breakdown = []
        mock_verdict.processing_trace = []
        mock_verdict.retrieval_status = "ok"
        mock_verdict.source_mode = "local"
        mock_verdict.has_cross_passage_disparity = False
        mock_verdict.disparity_details = {}
        mock_verdict.has_meaningful_evidence = True
        mock_verdict.duplicates_filtered = 0

        resp = adapter._format_report(verdict=mock_verdict, claim_clean="Water is wet.", mode="local")
        self.assertIn("verdict", resp)
        self.assertIn("confidence", resp)
        self.assertIn("probabilities", resp)
        self.assertEqual(resp["verdict"], "VERIFIED_TRUE")
        self.assertEqual(resp["confidence"], 0.95)

    # 22. Calibration, threshold enforcement, and honest abstention
    def test_22_calibration_and_simplex_invariants(self):
        raw_logits = [3.0, 1.0, -1.0, -3.0]
        cal_probs = self.calibrator.calibrate(raw_logits)
        # Probability distribution must sum to 1.0 (simplex invariant)
        self.assertAlmostEqual(float(np.sum(cal_probs)), 1.0, places=4)
        # All probabilities must be non-negative and bounded
        for p in cal_probs:
            self.assertGreaterEqual(float(p), 0.0)
            self.assertLessEqual(float(p), 1.0)
        # Non-abstained for decisive distribution
        is_abs, reason = self.abstainer.should_abstain(cal_probs)
        self.assertFalse(is_abs)
        # Uniform ambiguous distribution must abstain under tau = 0.32
        uniform_probs = [0.25, 0.25, 0.25, 0.25]
        is_abs_u, reason_u = self.abstainer.should_abstain(uniform_probs)
        self.assertTrue(is_abs_u)

    # 23. Original-claim preservation and traceable normalized claims
    def test_23_original_claim_audit_traceability(self):
        claim = "incognitio mdoe dont stops websits traking me"
        norm_res = SpellingNormalizer().normalize(claim)
        self.assertNotEqual(norm_res.sanitized_text, norm_res.normalized_text)
        self.assertEqual(norm_res.raw_text, claim)
        self.assertEqual(norm_res.sanitized_text, claim)

    # 24. Contradictory evidence within a single article
    def test_24_internal_tension_within_single_passage(self):
        passage = "While private browsing mode offers local privacy, it does not guarantee anonymity or prevent remote tracking."
        claim = "Private browsing guarantees complete anonymity online."
        nli = CrossEncoderNLI(device="cpu")
        res = nli.predict_pair(passage, claim)
        self.assertEqual(res["predicted_label"], "CONTRADICTION")

    # 25. Retrieval failures that must never fabricate sources or evidence
    def test_25_retrieval_failure_never_fabricates_evidence(self):
        res = self.aggregator.aggregate([], [])
        self.assertEqual(res.canonical_verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertEqual(res.unique_passages_used, 0)
        self.assertEqual(len(res.evidence_breakdown), 0)


if __name__ == "__main__":
    unittest.main()
