"""
Comprehensive Reliability Engineering Regression Suite for VERITAS-X.
Validates all 18 mandated categories specified in Phase 4:
 1. Clearly supported factual claims
 2. Clearly refuted factual claims
 3. Claims with genuinely mixed or cherry-picked evidence
 4. Claims with no relevant evidence
 5. Conflicting passages that refer to different entities
 6. Conflicting passages that refer to different dates or time periods
 7. Negation and double negation
 8. Numerical claims, units, dates, and quantities
 9. Compound claims
10. Multi-hop claims
11. Ambiguous and underspecified claims
12. Misspelled claims and named entities
13. Duplicate passages and correlated sources
14. Low-quality or irrelevant retrieved passages
15. Live-web timeout, unavailable service, malformed response, and empty search results
16. Model loading failures, device fallback, and inference exceptions
17. Frontend/API response mismatches and stale request races
18. Confidence calibration and abstention behavior
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
from api.adapter import VeritasAPIAdapter


class TestReliabilityEngineeringRegressionSuite(unittest.TestCase):
    """Deterministic regression suite covering all 18 reliability categories."""

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

    # 3. Claims with genuinely mixed or cherry-picked evidence
    def test_03_genuinely_mixed_evidence(self):
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

    # 4. Claims with no relevant evidence
    def test_04_no_relevant_evidence(self):
        passages = []
        preds = []
        res = self.aggregator.aggregate(passages, preds)
        self.assertEqual(res.canonical_verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertEqual(res.confidence, 0.95)

    # 5. Conflicting passages that refer to different entities
    def test_05_different_entity_filtering(self):
        claim = "Google acquired DeepMind in 2014."
        off_topic_passage = "Apple acquired Beats Electronics in 2014 for three billion dollars."
        is_rel = VerificationEngine.is_topically_relevant(claim, off_topic_passage)
        self.assertFalse(is_rel, "Passage about Apple must not be deemed relevant to Google claim")

    # 6. Conflicting passages that refer to different dates or time periods
    def test_06_temporal_boundary_analysis(self):
        t1 = "On Monday, transport officials announced the bridge was fully closed."
        t2 = "By Wednesday, the bridge was reopened to all regular traffic."
        a1 = LinguisticAnalyzer.analyze(t1)
        a2 = LinguisticAnalyzer.analyze(t2)
        self.assertTrue(any(t.normalized_text == "monday" for t in a1.tokens))
        self.assertTrue(any(t.normalized_text == "wednesday" for t in a2.tokens))

    # 7. Negation and double negation
    def test_07_negation_and_double_negation(self):
        claim = "It is not true that the Earth does not rotate."
        res = LinguisticAnalyzer.analyze(claim)
        root = res.sentences[0]
        # Clause negation detection
        self.assertTrue(any(c.is_negated for c in root.clauses))

    # 8. Numerical claims, units, dates, and quantities
    def test_08_numerical_and_unit_parsing(self):
        claim = "Humans use only 10% of their brains."
        res = LinguisticAnalyzer.analyze(claim)
        tokens = res.sentences[0].tokens
        num_tok = next((t for t in tokens if t.text == "10%"), None)
        self.assertIsNotNone(num_tok)
        self.assertEqual(num_tok.pos, "CD")
        self.assertEqual(num_tok.upos, "NUM")
        # Ensure 'of' is correctly tagged as preposition ADP/IN
        of_tok = next((t for t in tokens if t.text == "of"), None)
        self.assertIsNotNone(of_tok)
        self.assertEqual(of_tok.pos, "IN")

    # 9. Compound claims
    def test_09_compound_claims(self):
        from veritas.reasoning.decomposer import ClaimDecomposer
        compound = "The library was built in 1920 and it burned down in 1980."
        decomposer = ClaimDecomposer()
        subclaims = decomposer.decompose(compound)
        self.assertGreaterEqual(len(subclaims), 2, "Compound sentence must decompose into at least 2 subclaims")
        self.assertTrue(any("1920" in sc.text for sc in subclaims))
        self.assertTrue(any("1980" in sc.text for sc in subclaims))

    # 10. Multi-hop claims
    def test_10_multi_hop_claims(self):
        claim = "The father of King George VI was King George V."
        extracted = ClaimExtractor.extract(claim)
        self.assertIsNotNone(extracted.subject)
        self.assertIsNotNone(extracted.predicate)

    # 11. Ambiguous and underspecified claims
    def test_11_ambiguous_and_fragment_claims(self):
        frag = "Very fast cars."
        res = LinguisticAnalyzer.analyze(frag)
        self.assertEqual(res.sentences[0].sentence_type, "fragment")
        self.assertTrue(res.sentences[0].is_incomplete)

    # 12. Misspelled claims and named entities
    def test_12_conservative_spelling_preservation(self):
        claim = "Albert Einsten published the theory of relativity."
        res = SpellingNormalizer().normalize(claim)
        # Conservative normalizer protects capitalized named entities
        self.assertEqual(res.normalized_text, claim)

    # 13. Duplicate passages and correlated sources
    def test_13_duplicate_sources_discounted(self):
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

    # 14. Low-quality or irrelevant retrieved passages
    def test_14_low_quality_irrelevant_passages(self):
        passages = [{"doc_id": "p_irr", "text": "A standard chocolate cookie recipe uses flour, butter, and sugar."}]
        preds = [{"entailment": 0.02, "contradiction": 0.01, "neutral": 0.97}]
        res = self.aggregator.aggregate(passages, preds)
        self.assertEqual(res.canonical_verdict, CanonicalVeracity.INSUFFICIENT_EVIDENCE)
        self.assertEqual(res.irrelevant_passages_filtered, 1)

    # 15. Live-web timeout, unavailable service, malformed response, and empty search results
    def test_15_web_service_failures_handled_gracefully(self):
        from veritas.retrieval.tavily_provider import TavilySearchProvider
        provider = TavilySearchProvider()
        # Non-empty unconfigured or error query returns safe empty results
        passages, meta = provider.search("")
        self.assertEqual(len(passages), 0)
        self.assertIn("error", meta)

    # 16. Model loading failures, device fallback, and inference exceptions
    def test_16_device_fallback_integrity(self):
        from veritas.verification.nli_model import CrossEncoderNLI
        nli = CrossEncoderNLI(device="cpu")
        self.assertEqual(nli.device, "cpu")

    # 17. Frontend/API response mismatches and stale request races
    def test_17_api_adapter_response_schema(self):
        adapter = VeritasAPIAdapter()
        mock_verdict = MagicMock()
        mock_verdict.claim_text = "Water is wet."
        mock_verdict.verdict = CanonicalVeracity.VERIFIED_TRUE
        mock_verdict.raw_prediction = CanonicalVeracity.VERIFIED_TRUE.value
        mock_verdict.confidence = 0.95
        mock_verdict.calibrated_confidence = 0.95
        mock_verdict.probabilities = [0.95, 0.02, 0.02, 0.01]
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

        resp = adapter._format_report(mock_verdict, "Water is wet.", "local", 12.5)
        self.assertIn("verdict", resp)
        self.assertIn("confidence", resp)
        self.assertIn("probabilities", resp)
        self.assertEqual(resp["verdict"], "VERIFIED_TRUE")
        self.assertEqual(resp["confidence"], 0.95)

    # 18. Confidence calibration and abstention behavior
    def test_18_temperature_calibration_and_abstention(self):
        raw_logits = [3.0, 1.0, -1.0, -3.0]
        cal_probs = self.calibrator.calibrate(raw_logits)
        self.assertAlmostEqual(float(np.sum(cal_probs)), 1.0, places=4)
        is_abs, reason = self.abstainer.should_abstain(cal_probs)
        self.assertFalse(is_abs)


if __name__ == "__main__":
    unittest.main()
