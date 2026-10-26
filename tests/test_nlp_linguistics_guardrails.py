"""
Unit Test Suite for VERITAS-X Phase 3: Linguistic Analysis, Claim Structure & Meaning-Preservation Guardrails.

Test Categories:
1. Simple declarative sentences.
2. Questions and incomplete sentences.
3. Subject-verb-object (SVO) extraction.
4. Passive voice and agent extraction.
5. Negation and nested negation scoping.
6. Quantifiers: all, some, none, not all, always, sometimes.
7. Modal terms: may, might, must, allegedly.
8. Dates, numbers, percentages, and units.
9. Compound claims and causal relationships.
10. Conditionals and contrasts.
11. Ambiguous pronouns and missing antecedents.
12. Informal wording after Phase 2 normalization.
13. Meaning-changing spelling corrections (guardrail REVERT_TO_ORIGINAL).
14. Unknown or unparseable sentence structures (uncertainty flagging).
15. Invariance across multiple calls and deterministic execution.
"""
import os
import sys
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from veritas.nlp.linguistics import (
    LinguisticAnalyzer,
    AnalyzedSentence,
    LinguisticToken,
)
from veritas.nlp.claim_extractor import (
    ClaimExtractor,
    StructuredClaim,
    ModalityLevel,
    QuantifierType,
)
from veritas.nlp.guardrails import (
    MeaningPreservationGuardrails,
    GuardrailResult,
    GuardrailDecision,
)
from veritas.nlp.spelling import SpellingNormalizer


class TestNLPLinguisticsAndGuardrails(unittest.TestCase):
    """Production test suite for linguistic analysis, claim extraction, and meaning guardrails."""

    @classmethod
    def setUpClass(cls):
        cls.normalizer = SpellingNormalizer()
        cls.analyzer = LinguisticAnalyzer()
        cls.extractor = ClaimExtractor()
        cls.guardrails = MeaningPreservationGuardrails()

    # --------------------------------------------------------------------------
    # 1. Simple Declarative Sentences
    # --------------------------------------------------------------------------
    def test_01_simple_declarative_sentences(self):
        """Verifies simple declarative sentence segmentation and structural parsing."""
        claim = "Pure water freezes at 0 degrees Celsius under standard atmospheric pressure."
        res = self.extractor.extract(claim)

        self.assertEqual(res.sentence_type, "assertion")
        self.assertFalse(res.is_question)
        self.assertFalse(res.is_incomplete)
        self.assertFalse(res.is_negated)
        self.assertIn("water", res.subject.lower())
        self.assertIn("freeze", res.predicate.lower() if "freeze" in res.predicate.lower() else res.predicate_lemma)
        self.assertEqual(res.modality, ModalityLevel.CERTAIN)

    # --------------------------------------------------------------------------
    # 2. Questions and Incomplete Sentences
    # --------------------------------------------------------------------------
    def test_02_questions_and_incomplete_sentences(self):
        """Differentiates inquiries and incomplete fragments from assertions."""
        # Standard question with question mark
        q1 = "Does incognito mode hide your IP address from websites?"
        res_q1 = self.extractor.extract(q1)
        self.assertTrue(res_q1.is_question)
        self.assertEqual(res_q1.sentence_type, "question")

        # Inverted auxiliary question without question mark
        q2 = "Is pure water safe to drink"
        res_q2 = self.extractor.extract(q2)
        self.assertTrue(res_q2.is_question)

        # Incomplete fragment lacking finite verb
        frag = "At 0 degrees Celsius without pressure."
        res_frag = self.extractor.extract(frag)
        self.assertTrue(res_frag.is_incomplete)
        self.assertEqual(res_frag.sentence_type, "fragment")
        self.assertIsNone(res_frag.predicate)

    # --------------------------------------------------------------------------
    # 3. Subject-Verb-Object (SVO) Extraction
    # --------------------------------------------------------------------------
    def test_03_svo_extraction(self):
        """Validates correct identification of Subject, Predicate, and Object."""
        claim = "The New York Post endorsed Donald Trump's bid for re-election."
        res = self.extractor.extract(claim)

        self.assertIsNotNone(res.subject)
        self.assertIn("New York Post", res.subject)
        self.assertEqual(res.predicate_lemma, "endorse")
        self.assertIsNotNone(res.object)
        self.assertIn("Donald Trump", res.object)

    # --------------------------------------------------------------------------
    # 4. Passive Voice and Agent Extraction
    # --------------------------------------------------------------------------
    def test_04_passive_voice_and_agent(self):
        """Verifies passive voice detection and logical subject (agent) extraction."""
        claim = "Amy Coney Barrett was confirmed to the Supreme Court by the US Senate."
        res = self.extractor.extract(claim)

        self.assertTrue(res.is_passive)
        self.assertIn("Amy Coney Barrett", res.subject)
        self.assertIn("confirmed", res.predicate)
        self.assertIsNotNone(res.logical_subject)
        self.assertIn("Senate", res.logical_subject)

    # --------------------------------------------------------------------------
    # 5. Negation and Nested Negation Scoping
    # --------------------------------------------------------------------------
    def test_05_negation_and_scoping(self):
        """Verifies detection of affirmative negation and its scope."""
        claim = "Using incognito mode does not prevent websites from tracking user activity."
        res = self.extractor.extract(claim)

        self.assertTrue(res.is_negated)
        self.assertIn("not", res.negation_tokens)
        self.assertEqual(res.negation_scope, "predicate")

        # Nested / double negation check: "not impossible"
        double_neg = "It is not impossible for water to supercool below 0 degrees."
        res_dn = self.extractor.extract(double_neg)
        self.assertTrue(res_dn.is_negated)

    # --------------------------------------------------------------------------
    # 6. Quantifiers (Universal, Existential, Negative, Partial Negation)
    # --------------------------------------------------------------------------
    def test_06_quantifiers_classification(self):
        """Categorizes universal, existential, negative, and partial quantifiers."""
        # Universal
        c_all = "All swans are white."
        res_all = self.extractor.extract(c_all)
        self.assertTrue(any(q.qtype == QuantifierType.UNIVERSAL for q in res_all.quantifiers))

        # Existential
        c_some = "Some birds cannot fly."
        res_some = self.extractor.extract(c_some)
        self.assertTrue(any(q.qtype == QuantifierType.EXISTENTIAL for q in res_some.quantifiers))

        # Negative
        c_none = "None of the patients exhibited symptoms."
        res_none = self.extractor.extract(c_none)
        self.assertTrue(any(q.qtype == QuantifierType.NEGATIVE for q in res_none.quantifiers))

        # Partial Negation ("not all")
        c_not_all = "Not all scientific studies are replicable."
        res_not_all = self.extractor.extract(c_not_all)
        self.assertTrue(any(q.qtype == QuantifierType.PARTIAL_NEGATION for q in res_not_all.quantifiers))

        # Frequency
        c_always = "Pure water always boils at 100°C under 1 atm."
        res_always = self.extractor.extract(c_always)
        self.assertTrue(any(q.qtype == QuantifierType.FREQUENCY for q in res_always.quantifiers))

    # --------------------------------------------------------------------------
    # 7. Modal Terms (Certainty vs Possibility vs Allegation)
    # --------------------------------------------------------------------------
    def test_07_modality_and_hedging(self):
        """Extracts epistemic modality across certainty levels."""
        # Alleged
        c_alleged = "The suspect allegedly transferred funds to an overseas bank."
        res_alleged = self.extractor.extract(c_alleged)
        self.assertEqual(res_alleged.modality, ModalityLevel.ALLEGED)
        self.assertIn("allegedly", res_alleged.modality_markers)

        # Possible (may / might)
        c_possible = "Drinking cold water might lower body temperature temporarily."
        res_possible = self.extractor.extract(c_possible)
        self.assertEqual(res_possible.modality, ModalityLevel.POSSIBLE)

        # Probable (should / likely)
        c_probable = "The economy will likely grow next quarter."
        res_probable = self.extractor.extract(c_probable)
        self.assertEqual(res_probable.modality, ModalityLevel.PROBABLE)

        # Certain (direct assertion)
        c_certain = "The Earth orbits the Sun."
        res_certain = self.extractor.extract(c_certain)
        self.assertEqual(res_certain.modality, ModalityLevel.CERTAIN)

    # --------------------------------------------------------------------------
    # 8. Dates, Numbers, Percentages, and Units
    # --------------------------------------------------------------------------
    def test_08_dates_numbers_percentages_units(self):
        """Extracts structured entities for dates, numbers, percentages, and units."""
        claim = (
            "On October 26, 2020, water froze at 0°C under 1 atm while 10% of "
            "participants observed the event."
        )
        res = self.extractor.extract(claim)

        # Dates
        self.assertTrue(any("October 26, 2020" in d.raw_text for d in res.temporal_entities))

        # Numbers and Units
        units = [n.unit for n in res.numeric_entities if n.unit]
        self.assertTrue(any("°C" in u for u in units))
        self.assertTrue(any("atm" in u for u in units))

        # Percentage
        pcts = [n for n in res.numeric_entities if n.category == "percentage"]
        self.assertEqual(len(pcts), 1)
        self.assertEqual(pcts[0].value, 10.0)

    # --------------------------------------------------------------------------
    # 9. Compound Claims and Causal Relationships
    # --------------------------------------------------------------------------
    def test_09_compound_claims_and_causality(self):
        """Extracts sub-propositions and explicit causal links."""
        claim_causal = "Smoking causes lung cancer."
        res_causal = self.extractor.extract(claim_causal)
        self.assertTrue(res_causal.is_causal)
        self.assertEqual(res_causal.causal_marker.lower(), "causes")
        self.assertIn("Smoking", res_causal.cause_span)
        self.assertIn("lung cancer", res_causal.effect_span)

        claim_compound = (
            "The bridge was closed for repairs on Monday, and the transport authority "
            "announced that it reopened on Tuesday."
        )
        res_compound = self.extractor.extract(claim_compound)
        self.assertTrue(res_compound.is_compound)
        self.assertGreaterEqual(len(res_compound.sub_propositions), 2)

    # --------------------------------------------------------------------------
    # 10. Conditionals and Contrasts
    # --------------------------------------------------------------------------
    def test_10_conditionals_and_contrasts(self):
        """Extracts antecedent and consequent in conditional propositions."""
        claim = "If atmospheric pressure decreases, then the boiling point drops."
        res = self.extractor.extract(claim)

        self.assertTrue(res.is_conditional)
        self.assertIsNotNone(res.condition_clause)
        self.assertIn("pressure decreases", res.condition_clause.lower())
        self.assertIsNotNone(res.consequence_clause)
        self.assertIn("boiling point drops", res.consequence_clause.lower())

    # --------------------------------------------------------------------------
    # 11. Ambiguous Pronouns and Missing Antecedents
    # --------------------------------------------------------------------------
    def test_11_ambiguous_pronouns(self):
        """Identifies dangling pronouns lacking prior antecedents in sentence."""
        claim = "It reopened on Tuesday without explanation."
        res = self.extractor.extract(claim)
        self.assertIn("It", res.unresolved_references)

    # --------------------------------------------------------------------------
    # 12. Informal Wording After Phase 2 Normalization
    # --------------------------------------------------------------------------
    def test_12_informal_wording_after_phase2_normalization(self):
        """Tests that normalized informal text parses cleanly into structured claims."""
        noisy = "incognito mode doesnt hide ur ip adress from websites"
        norm_res = self.normalizer.normalize(noisy)
        # Verify Phase 2 normalized text
        self.assertIn("does not", norm_res.normalized_text)
        self.assertIn("your", norm_res.normalized_text)
        self.assertIn("IP address", norm_res.normalized_text)

        # Extract structured claim from normalized result
        structured = self.extractor.extract(norm_res.normalized_text)
        self.assertTrue(structured.is_negated)
        self.assertIn("incognito mode", structured.subject)
        self.assertIn("hide", structured.predicate)
        self.assertEqual(structured.sentence_type, "assertion")

    # --------------------------------------------------------------------------
    # 13. Meaning-Changing Corrections Trigger Guardrail Revert
    # --------------------------------------------------------------------------
    def test_13_meaning_changing_corrections_trigger_revert(self):
        """
        Simulates dangerous meaning-changing modifications and confirms
        that the guardrail flags violations and enforces REVERT_TO_ORIGINAL.
        """
        # Case A: Negation dropped
        orig_neg = "Smoking does not cause lung cancer."
        tampered_pos = "Smoking does cause lung cancer."
        guard_res_neg = self.guardrails.evaluate(orig_neg, tampered_pos)
        self.assertFalse(guard_res_neg.is_safe)
        self.assertEqual(guard_res_neg.decision, GuardrailDecision.REVERT_TO_ORIGINAL)
        self.assertEqual(guard_res_neg.recommended_text, orig_neg)
        self.assertFalse(guard_res_neg.invariants_passed[MeaningPreservationGuardrails.INV_NEGATION_POLARITY])

        # Case B: Quantifier shift (all -> some)
        orig_all = "All swans are white."
        tampered_some = "Some swans are white."
        guard_res_quant = self.guardrails.evaluate(orig_all, tampered_some)
        self.assertFalse(guard_res_quant.is_safe)
        self.assertEqual(guard_res_quant.decision, GuardrailDecision.REVERT_TO_ORIGINAL)
        self.assertFalse(guard_res_quant.invariants_passed[MeaningPreservationGuardrails.INV_QUANTIFIERS])

        # Case C: Numeric value modified (0°C -> 10°C)
        orig_temp = "Pure water freezes at 0 degrees Celsius."
        tampered_temp = "Pure water freezes at 10 degrees Celsius."
        guard_res_temp = self.guardrails.evaluate(orig_temp, tampered_temp)
        self.assertFalse(guard_res_temp.is_safe)
        self.assertEqual(guard_res_temp.decision, GuardrailDecision.REVERT_TO_ORIGINAL)
        self.assertFalse(guard_res_temp.invariants_passed[MeaningPreservationGuardrails.INV_ENTITIES_NUMBERS])

        # Case D: Safe normalization passes cleanly
        raw_noisy = "Using incognitio mdoe prvents websits from traking user activty"
        norm_safe = self.normalizer.normalize(raw_noisy)
        guard_safe = self.guardrails.evaluate(raw_noisy, norm_safe)
        self.assertTrue(guard_safe.is_safe)
        self.assertEqual(guard_safe.decision, GuardrailDecision.USE_NORMALIZED)

    # --------------------------------------------------------------------------
    # 14. Unknown or Unparseable Structures Flag Uncertainty
    # --------------------------------------------------------------------------
    def test_14_unknown_structures_flag_uncertainty(self):
        """Ensures missing subjects or unparseable text return fragment flags without fabricating parses."""
        unparseable = "&&& ### @@@"
        res = self.extractor.extract(unparseable)
        self.assertTrue(res.is_incomplete)
        self.assertIsNone(res.subject)
        self.assertIsNone(res.predicate)

        single_word = "Whatever."
        res_single = self.extractor.extract(single_word)
        self.assertTrue(res_single.is_incomplete)

    # --------------------------------------------------------------------------
    # 15. Invariance Across Repeated Calls & Deterministic Execution
    # --------------------------------------------------------------------------
    def test_15_deterministic_execution(self):
        """Guarantees 100% deterministic output across repeated calls."""
        claim = "Amy Coney Barrett was confirmed to the Supreme Court on October 26, 2020."
        res1 = self.extractor.extract(claim).to_dict()
        res2 = self.extractor.extract(claim).to_dict()
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    unittest.main()
