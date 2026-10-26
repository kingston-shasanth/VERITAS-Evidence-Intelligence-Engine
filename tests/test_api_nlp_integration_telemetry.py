"""
Integration and Telemetry Verification Suite for VERITAS-X Phase 4A.
Tests live integration of Phase 2 (sanitization, spelling normalization)
and Phase 3 (linguistics, structured claim extraction, meaning guardrails)
into VeritasAPIAdapter and VeritasRequestHandler HTTP server.

Covers all 22 required evaluation areas:
1. Correctly spelled input.
2. Simple spelling mistakes.
3. Multiple typos and grammar errors in one sentence.
4. Informal wording and contractions.
5. Negation preservation.
6. Quantifier preservation.
7. Modality and attribution.
8. Names, dates, numbers, percentages, and units.
9. Causal statements.
10. Conditional and compound claims.
11. Questions versus assertions.
12. Ambiguous corrections, including 'sever' versus 'seven'.
13. Ambiguous pronouns and incomplete sentences.
14. Empty and excessively long inputs.
15. NLP initialization and runtime failures.
16. Retrieval query selection and fallback.
17. NLI hypothesis selection (strictly canonical).
18. Original-claim preservation.
19. Ordered trace sequences and real measured durations.
20. Explicit representation of skipped, uncertain, fallback, and failed stages.
21. Backward compatibility with existing API clients.
22. Existing verdict and evidence-aggregation behavior.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from api.adapter import VeritasAPIAdapter, generate_claim_id
from tests.test_api_adapter import make_dummy_verdict
from veritas.data.schema import CanonicalVeracity
from veritas.nlp import (
    PROCESSING_TRACE_VERSION,
    ProcessingStage,
    StageStatus,
    WordCorrectionEvent,
    TraceEvent,
    ProcessingTraceCollector,
    GuardrailDecision,
)


class TestLiveNLPPipelineIntegrationAndTelemetry(unittest.TestCase):
    """Exhaustive test suite for Phase 4A Live NLP & Telemetry Integration."""

    def setUp(self):
        self.mock_engine = MagicMock()
        self.mock_engine.verify_claim.return_value = make_dummy_verdict(
            claim="Standard scientific claim",
            verdict=CanonicalVeracity.VERIFIED_TRUE,
            confidence=0.88,
        )
        self.adapter = VeritasAPIAdapter(engine=self.mock_engine, lazy_init=True)

    # --------------------------------------------------------------------------
    # 1. Correctly spelled input
    # --------------------------------------------------------------------------
    def test_01_correctly_spelled_input(self):
        claim = "Pure water freezes at 0 degrees Celsius."
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        self.assertEqual(res["original_claim"], claim)
        self.assertEqual(res["sanitized_claim"], claim)
        self.assertEqual(res["normalized_claim"], claim)
        self.assertEqual(res["normalization_status"], "accepted")
        self.assertEqual(len(res["normalization_corrections"]), 0)
        self.assertEqual(res["retrieval_query_used"], claim)
        self.assertEqual(res["nli_hypothesis_used"], claim)

        stages = [e["stage"] for e in res["processing_trace"]]
        self.assertEqual(len(stages), 11)
        self.assertEqual(stages[0], ProcessingStage.INPUT_RECEIVED.value)
        self.assertEqual(stages[-1], ProcessingStage.FINAL_RESULT.value)

    # --------------------------------------------------------------------------
    # 2. Simple spelling mistakes
    # --------------------------------------------------------------------------
    def test_02_simple_spelling_mistakes(self):
        claim = "Pur water frezes at zero degrees."
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        self.assertEqual(res["original_claim"], claim)
        self.assertIn("Pure", res["normalized_claim"])
        self.assertIn("freezes", res["normalized_claim"])
        self.assertEqual(res["normalization_status"], "accepted")
        self.assertGreater(len(res["normalization_corrections"]), 0)
        # Retrieval uses normalized query to maximize recall
        self.assertIn("Pure", res["retrieval_query_used"])
        # NLI hypothesis STRICTLY preserves original sanitized claim
        self.assertEqual(res["nli_hypothesis_used"], res["sanitized_claim"])

    # --------------------------------------------------------------------------
    # 3. Multiple typos and grammar errors in one sentence
    # --------------------------------------------------------------------------
    def test_03_multiple_typos_and_grammar_errors(self):
        claim = "The vacine dosent caus autsm in childrn."
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        self.assertEqual(res["original_claim"], claim)
        self.assertIn("vaccine", res["normalized_claim"].lower())
        self.assertIn("autism", res["normalized_claim"].lower())
        self.assertIn("children", res["normalized_claim"].lower())
        self.assertGreater(len(res["normalization_corrections"]), 2)

        # Confirm word-level correction structure for UI animations
        spelling_stage = [e for e in res["processing_trace"] if e["stage"] == ProcessingStage.SPELLING_NORMALIZATION.value][0]
        word_events = spelling_stage["details"]["word_correction_events"]
        self.assertGreater(len(word_events), 0)
        for w in word_events:
            self.assertIn("original", w)
            self.assertIn("replacement", w)
            self.assertIn("start_char", w)
            self.assertIn("end_char", w)
            self.assertIn("status", w)

    # --------------------------------------------------------------------------
    # 4. Informal wording and contractions
    # --------------------------------------------------------------------------
    def test_04_informal_wording_and_contractions(self):
        claim = "Govt doesn't hide info w/o public review."
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        self.assertIn("does not", res["normalized_claim"].lower())
        self.assertIn("without", res["normalized_claim"].lower())
        self.assertIn("government", res["normalized_claim"].lower())

    # --------------------------------------------------------------------------
    # 5. Negation preservation
    # --------------------------------------------------------------------------
    def test_05_negation_preservation(self):
        claim = "Vaccines do not cause autism."
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        # Meaning preservation guardrail must verify negation
        guard_stage = [e for e in res["processing_trace"] if e["stage"] == ProcessingStage.MEANING_PRESERVATION.value][0]
        self.assertTrue(guard_stage["details"]["invariants_passed"].get("negation_and_polarity", False))
        self.assertTrue(res["structured_claim"]["is_negated"])
        self.assertIn("not", res["structured_claim"]["negation_tokens"])

    # --------------------------------------------------------------------------
    # 6. Quantifier preservation
    # --------------------------------------------------------------------------
    def test_06_quantifier_preservation(self):
        claim = "All humans require oxygen to survive."
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        self.assertTrue(len(res["structured_claim"]["quantifiers"]) > 0)
        q = res["structured_claim"]["quantifiers"][0]
        self.assertEqual(q["text"].lower(), "all")
        self.assertEqual(q["type"], "UNIVERSAL")

    # --------------------------------------------------------------------------
    # 7. Modality and attribution
    # --------------------------------------------------------------------------
    def test_07_modality_and_attribution(self):
        claim = "According to leaked reports, officials allegedly concealed the data."
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        self.assertEqual(res["structured_claim"]["modality"], "ALLEGED")
        self.assertIn("allegedly", res["structured_claim"]["modality_markers"])

    # --------------------------------------------------------------------------
    # 8. Names, dates, numbers, percentages, and units
    # --------------------------------------------------------------------------
    def test_08_names_dates_numbers_percentages_units(self):
        claim = "NASA launched Perseverance on July 30, 2020 at 12000 km/h with 99.8% precision."
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        sc = res["structured_claim"]
        self.assertTrue(any("2020" in t["raw_text"] for t in sc["temporal_entities"]))
        self.assertTrue(any("99.8" in n["raw_text"] for n in sc["numeric_entities"]))
        self.assertTrue(any("NASA" in name for name in sc["named_entities"]))

    # --------------------------------------------------------------------------
    # 9. Causal statements
    # --------------------------------------------------------------------------
    def test_09_causal_statements(self):
        claim = "Cigarette smoking causes coronary heart disease."
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        sc = res["structured_claim"]
        self.assertTrue(sc["is_causal"])
        self.assertEqual(sc["causal_marker"], "causes")

    # --------------------------------------------------------------------------
    # 10. Conditional and compound claims
    # --------------------------------------------------------------------------
    def test_10_conditional_and_compound_claims(self):
        claim = "If pressure increases, then temperature rises, and volume decreases."
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        sc = res["structured_claim"]
        self.assertTrue(sc["is_conditional"])
        self.assertTrue(sc["is_compound"])
        self.assertGreater(len(sc["sub_propositions"]), 1)

    # --------------------------------------------------------------------------
    # 11. Questions versus assertions
    # --------------------------------------------------------------------------
    def test_11_questions_versus_assertions(self):
        claim = "Does pure water freeze at 0 degrees Celsius?"
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        self.assertTrue(res["structured_claim"]["is_question"])
        self.assertEqual(res["structured_claim"]["sentence_type"], "question")

    # --------------------------------------------------------------------------
    # 12. Ambiguous corrections: 'sever' versus 'seven'
    # --------------------------------------------------------------------------
    def test_12_ambiguous_corrections_sever_vs_seven(self):
        claim = "The pacient had sever syndroms"
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        # Ambiguous candidate must flag normalization as uncertain or reverted
        self.assertEqual(res["normalization_status"], "uncertain")
        # Retrieval falls back conservatively to sanitized text
        self.assertEqual(res["retrieval_query_used"], res["sanitized_claim"])
        # Canonical NLI hypothesis strictly preserves original claim
        self.assertEqual(res["nli_hypothesis_used"], res["sanitized_claim"])
        self.assertNotIn("seven", res["normalized_claim"])  # Must NOT hallucinate a number

    # --------------------------------------------------------------------------
    # 13. Ambiguous pronouns and incomplete sentences
    # --------------------------------------------------------------------------
    def test_13_ambiguous_pronouns_and_incomplete_sentences(self):
        claim = "It was strongly denied by them."
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        sc = res["structured_claim"]
        self.assertTrue(len(sc["unresolved_references"]) > 0)
        self.assertIn("It", sc["unresolved_references"])

    # --------------------------------------------------------------------------
    # 14. Empty and excessively long inputs
    # --------------------------------------------------------------------------
    def test_14_empty_and_excessively_long_inputs(self):
        with self.assertRaises(ValueError):
            self.adapter.verify(claim="")

        with self.assertRaises(ValueError):
            self.adapter.verify(claim="   \n\t  ")

        long_claim = "The patient exhibited symptoms. " * 150
        res = self.adapter.verify(claim=long_claim, include_processing_trace=True)
        self.assertEqual(res["original_claim"], long_claim)
        self.assertIsInstance(res["processing_trace"], list)
        self.assertEqual(len(res["processing_trace"]), 11)

    # --------------------------------------------------------------------------
    # 15. NLP initialization and runtime failures
    # --------------------------------------------------------------------------
    def test_15_nlp_initialization_and_runtime_failures(self):
        with patch.object(self.adapter._normalizer, "normalize", side_effect=RuntimeError("Dictionary corruption")):
            res = self.adapter.verify(claim="Test system resiliency under fault", include_processing_trace=True)

            self.assertEqual(res["normalization_status"], "failed")
            self.assertEqual(res["retrieval_query_used"], res["sanitized_claim"])
            self.assertEqual(res["nli_hypothesis_used"], res["sanitized_claim"])

            norm_stage = [e for e in res["processing_trace"] if e["stage"] == ProcessingStage.SPELLING_NORMALIZATION.value][0]
            self.assertEqual(norm_stage["status"], StageStatus.FAILED.value)

    # --------------------------------------------------------------------------
    # 16. Retrieval query selection and fallback
    # --------------------------------------------------------------------------
    def test_16_retrieval_query_selection_and_fallback(self):
        # Case A: Accepted normalization uses normalized query
        clean_typo = "Pur watur frezes at 0 C"
        res_a = self.adapter.verify(claim=clean_typo, include_processing_trace=True)
        if res_a["normalization_status"] == "accepted":
            self.assertEqual(res_a["retrieval_query_used"], res_a["normalized_claim"])

        # Case B: Reverted or uncertain normalization falls back to sanitized
        with patch.object(self.adapter._guardrails, "evaluate", return_value=MagicMock(
            is_safe=False,
            decision=GuardrailDecision.REVERT_TO_ORIGINAL,
            violations=["Polarity inversion detected"],
            warnings=[],
            invariants_passed={},
            rationale="Safety fallback",
        )):
            res_b = self.adapter.verify(claim="Vaccines do not work", include_processing_trace=True)
            self.assertEqual(res_b["normalization_status"], "reverted")
            self.assertEqual(res_b["retrieval_query_used"], res_b["sanitized_claim"])

    # --------------------------------------------------------------------------
    # 17. NLI hypothesis selection (strictly canonical)
    # --------------------------------------------------------------------------
    def test_17_nli_hypothesis_selection_strictly_original(self):
        claim_with_typo = "The vacine dosent caus autsm"
        res = self.adapter.verify(claim=claim_with_typo, include_processing_trace=True)

        # Retrieval can use normalized
        self.assertNotEqual(res["retrieval_query_used"], claim_with_typo)
        # But NLI hypothesis is strictly sanitized original claim
        self.assertEqual(res["nli_hypothesis_used"], res["sanitized_claim"])

        # Verify engine was actually called with the canonical hypothesis
        self.mock_engine.verify_claim.assert_called()
        _, kwargs = self.mock_engine.verify_claim.call_args
        self.assertEqual(kwargs["claim_text"], res["sanitized_claim"])

    # --------------------------------------------------------------------------
    # 18. Original-claim preservation
    # --------------------------------------------------------------------------
    def test_18_original_claim_preservation(self):
        raw = "  “Unsanitized   typography” — test \t\n  "
        res = self.adapter.verify(claim=raw, include_processing_trace=True)

        self.assertEqual(res["original_claim"], raw)
        self.assertNotEqual(res["sanitized_claim"], raw)
        self.assertEqual(res["sanitized_claim"], '"Unsanitized typography" - test')

    # --------------------------------------------------------------------------
    # 19. Ordered trace sequences and real measured durations
    # --------------------------------------------------------------------------
    def test_19_ordered_trace_sequences_and_real_durations(self):
        claim = "Atmospheric carbon dioxide concentrations are rising."
        res = self.adapter.verify(claim=claim, include_processing_trace=True)

        trace = res["processing_trace"]
        self.assertEqual(len(trace), 11)

        expected_order = [
            ProcessingStage.INPUT_RECEIVED.value,
            ProcessingStage.SANITIZATION.value,
            ProcessingStage.SPELLING_NORMALIZATION.value,
            ProcessingStage.MEANING_PRESERVATION.value,
            ProcessingStage.TOKENIZATION.value,
            ProcessingStage.LINGUISTIC_ANALYSIS.value,
            ProcessingStage.CLAIM_EXTRACTION.value,
            ProcessingStage.RETRIEVAL.value,
            ProcessingStage.EVIDENCE_PROCESSING.value,
            ProcessingStage.NLI_VERIFICATION.value,
            ProcessingStage.FINAL_RESULT.value,
        ]

        actual_order = [evt["stage"] for evt in trace]
        self.assertEqual(actual_order, expected_order)

        for i, evt in enumerate(trace):
            self.assertEqual(evt["sequence"], i + 1)
            self.assertGreaterEqual(evt["duration_ms"], 0.0)
            self.assertGreaterEqual(evt["started_at_ms"], 0.0)
            self.assertIn("summary", evt)
            self.assertIn("details", evt)

    # --------------------------------------------------------------------------
    # 20. Representation of skipped, uncertain, fallback, and failed stages
    # --------------------------------------------------------------------------
    def test_20_representation_of_skipped_uncertain_fallback_failed_stages(self):
        # 1. Skipped: When candidate_passages are supplied directly
        fixtures = [{"doc_id": "f1", "text": "Evidence fixture text"}]
        res = self.adapter.verify(claim="Test claim", candidate_passages=fixtures, include_processing_trace=True)
        ret_stage = [e for e in res["processing_trace"] if e["stage"] == ProcessingStage.RETRIEVAL.value][0]
        self.assertEqual(ret_stage["status"], StageStatus.SKIPPED.value)

        # 2. Uncertain: When verdict is abstained
        self.mock_engine.verify_claim.return_value = make_dummy_verdict(was_abstained=True)
        res_abst = self.adapter.verify(claim="Unverifiable rumor", include_processing_trace=True)
        final_stage = [e for e in res_abst["processing_trace"] if e["stage"] == ProcessingStage.FINAL_RESULT.value][0]
        self.assertEqual(final_stage["status"], StageStatus.UNCERTAIN.value)

    # --------------------------------------------------------------------------
    # 21. Backward compatibility with existing API clients
    # --------------------------------------------------------------------------
    def test_21_backward_compatibility_existing_api_clients(self):
        # Legacy call with default parameters
        res = self.adapter.verify(claim="Pure water freezes at 0 degrees Celsius.")

        expected_contract_keys = {
            "claim_id",
            "claim",
            "verdict",
            "confidence",
            "probabilities",
            "has_cross_passage_disparity",
            "abstained",
            "reason",
            "evidence",
            "device",
            "latency_ms",
            "mode",
            "retrieval_status",
            "evidence_counts",
        }
        self.assertEqual(set(res.keys()), expected_contract_keys)
        self.assertNotIn("processing_trace", res)

    # --------------------------------------------------------------------------
    # 22. Existing verdict and evidence-aggregation behavior
    # --------------------------------------------------------------------------
    def test_22_verdict_and_evidence_aggregation_preservation(self):
        res = self.adapter.verify(claim="Earth orbits the Sun", include_processing_trace=True)

        # Temperature calibration and probabilities structure must be frozen
        self.assertEqual(res["verdict"], "VERIFIED_TRUE")
        self.assertEqual(set(res["probabilities"].keys()), {
            "VERIFIED_TRUE",
            "VERIFIED_FALSE",
            "CONFLICTING_OR_CHERRYPICKED",
            "INSUFFICIENT_EVIDENCE",
        })
        self.assertIn("has_cross_passage_disparity", res)
        self.assertIn("evidence_counts", res)


if __name__ == "__main__":
    unittest.main()
