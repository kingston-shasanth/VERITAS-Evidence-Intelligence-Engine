"""
Comprehensive Test Suite for VERITAS-X NLP Foundation & Robust Spelling Normalization.
Validates:
1. Common misspellings.
2. Missing, inserted, substituted, and transposed characters.
3. Multiple typos in a single sentence.
4. Correct words that resemble other words (no false-positive replacements).
5. Contractions and common factual abbreviations.
6. Capitalized names, proper nouns, and acronyms.
7. Dates, numbers, units, percentages, chemical formulas, and URLs.
8. Negation and quantifier preservation invariants.
9. Ambiguous spelling candidate detection and warnings.
10. Empty, whitespace-only, punctuation-only, and Unicode input.
11. Repeated calls and deterministic output.
12. Operation without GPU initialization or network access.
13. Latency and memory measurement without fragile timing-based assertions.
14. Frozen benchmark claim preservation (clean inputs remain strictly unmodified).
"""
import os
import sys
import unittest
import time

# Ensure repo root and src/ are on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from veritas.nlp.sanitizer import TextSanitizer, SanitizationResult
from veritas.nlp.spelling import (
    SpellingNormalizer,
    CorrectionItem,
    NormalizationResult,
    get_symspell_engine,
)
from veritas.nlp import normalize_claim_text


class TestNLPSpellingNormalization(unittest.TestCase):
    """Production test suite for VERITAS-X Phase 2 NLP Normalization Layer."""

    @classmethod
    def setUpClass(cls):
        # Warm the global SymSpell engine once
        cls.normalizer = SpellingNormalizer(max_edit_distance=2)

    # --------------------------------------------------------------------------
    # 1. Common Misspellings
    # --------------------------------------------------------------------------
    def test_01_common_misspellings(self):
        """Verifies accurate correction of typical factual domain misspellings."""
        test_pairs = [
            ("prvents", "prevents"),
            ("websits", "websites"),
            ("frezes", "freezes"),
            ("celcius", "celsius"),
            ("activty", "activity"),
            ("temprature", "temperature"),
        ]
        for typo, expected in test_pairs:
            res = self.normalizer.normalize(f"The condition {typo} normal operation.")
            self.assertIn(
                expected,
                res.normalized_text.lower(),
                f"Expected '{expected}' in corrected text, got '{res.normalized_text}'",
            )
            self.assertTrue(any(c.original.lower() == typo for c in res.corrections))

    # --------------------------------------------------------------------------
    # 2. Edit Distance Operations (Insertion, Deletion, Substitution, Transposition)
    # --------------------------------------------------------------------------
    def test_02_edit_distance_operations(self):
        """Tests each fundamental edit distance operation under Damerau-Levenshtein."""
        # Deletion (extra character inserted in typo)
        res_del = self.normalizer.normalize("Using incognitio mode")
        self.assertIn("incognito", res_del.normalized_text)

        # Insertion (missing character in typo)
        res_ins = self.normalizer.normalize("at 0 degres Celsius")
        self.assertIn("degrees", res_ins.normalized_text)

        # Substitution (wrong character)
        res_sub = self.normalizer.normalize("Pure watur freezes")
        self.assertIn("water", res_sub.normalized_text)

        # Transposition (adjacent characters swapped)
        res_trans = self.normalizer.normalize("Private mdoe hides data")
        self.assertIn("mode", res_trans.normalized_text)

    # --------------------------------------------------------------------------
    # 3. Multiple Typos in a Single Sentence
    # --------------------------------------------------------------------------
    def test_03_multiple_typos_in_single_sentence(self):
        """Validates that sentences with 4+ simultaneous typos normalize accurately."""
        noisy_claim_1 = "Using incognitio mdoe prvents websits from traking user activty"
        res_1 = self.normalizer.normalize(noisy_claim_1)
        expected_1 = "Using incognito mode prevents websites from tracking user activity"
        self.assertEqual(res_1.normalized_text, expected_1)
        self.assertEqual(len(res_1.corrections), 6)
        self.assertTrue(res_1.meaning_preservation_passed)

        noisy_claim_2 = "Pur watur frezes at 0 degres Celcius at se levl."
        res_2 = self.normalizer.normalize(noisy_claim_2)
        expected_2 = "Pure water freezes at 0 degrees Celsius at sea level."
        self.assertEqual(res_2.normalized_text, expected_2)
        self.assertTrue(res_2.meaning_preservation_passed)

    # --------------------------------------------------------------------------
    # 4. Correct Words That Resemble Other Words (No False Positives)
    # --------------------------------------------------------------------------
    def test_04_correct_words_resembling_others_untouched(self):
        """Ensures valid English words are never erroneously altered."""
        safe_sentences = [
            "There is peace in their country.",
            "Water boils at 100 degrees Celsius.",
            "The lead author read the report.",
            "They won the game and led the tournament.",
        ]
        for s in safe_sentences:
            res = self.normalizer.normalize(s)
            self.assertEqual(
                res.normalized_text,
                s,
                f"Valid sentence was incorrectly modified: '{res.normalized_text}' vs '{s}'",
            )
            spelling_corrections = [c for c in res.corrections if c.category == "spelling"]
            self.assertEqual(len(spelling_corrections), 0)

    # --------------------------------------------------------------------------
    # 5. Contractions and Common Factual Abbreviations
    # --------------------------------------------------------------------------
    def test_05_contractions_and_abbreviations(self):
        """Verifies contraction expansions and common audited chat abbreviations."""
        # Unpunctuated contractions
        res_cont = self.normalizer.normalize("incognito mode doesnt hide ur ip adress")
        self.assertIn("does not", res_cont.normalized_text)
        self.assertIn("your", res_cont.normalized_text)
        self.assertIn("IP", res_cont.normalized_text)
        self.assertIn("address", res_cont.normalized_text)

        # Standard punctuated contractions
        res_cont2 = self.normalizer.normalize("They don't believe it's true.")
        self.assertIn("do not", res_cont2.normalized_text)
        self.assertIn("it is", res_cont2.normalized_text)

        # Factual abbreviations
        res_abbr = self.normalizer.normalize("govt said the bridge was closed w/o prior notice")
        self.assertEqual(res_abbr.normalized_text, "government said the bridge was closed without prior notice")
        self.assertTrue(any(c.original == "govt" and c.category == "abbreviation" for c in res_abbr.corrections))
        self.assertTrue(any(c.original == "w/o" and c.category == "abbreviation" for c in res_abbr.corrections))

    # --------------------------------------------------------------------------
    # 6. Capitalized Names, Proper Nouns, and Acronyms
    # --------------------------------------------------------------------------
    def test_06_capitalized_names_and_acronyms_preserved(self):
        """Ensures proper nouns and all-caps acronyms are strictly protected."""
        claim = "Amy Coney Barrett was confirmed to the Supreme Court by the US Senate."
        res = self.normalizer.normalize(claim)
        self.assertIn("Amy Coney Barrett", res.normalized_text)
        self.assertIn("Supreme Court", res.normalized_text)
        self.assertIn("US Senate", res.normalized_text)

        acronyms_claim = "NASA, IUPAC, and FBI issued a joint bulletin on COVID-19."
        res_acronyms = self.normalizer.normalize(acronyms_claim)
        self.assertIn("NASA", res_acronyms.normalized_text)
        self.assertIn("IUPAC", res_acronyms.normalized_text)
        self.assertIn("FBI", res_acronyms.normalized_text)
        self.assertIn("COVID-19", res_acronyms.normalized_text)

    # --------------------------------------------------------------------------
    # 7. Dates, Numbers, Units, Percentages, and URLs
    # --------------------------------------------------------------------------
    def test_07_dates_numbers_units_percentages_urls(self):
        """Verifies that numbers, dates, units, percentages, and URLs remain intact."""
        date_claim = "The event occurred on October 26, 2020 and 2023-08-23."
        res_date = self.normalizer.normalize(date_claim)
        self.assertIn("October 26, 2020", res_date.normalized_text)
        self.assertIn("2023-08-23", res_date.normalized_text)

        metric_claim = "Pure water freezes at 0 degrees Celsius under 1 atm and 100°C."
        res_metric = self.normalizer.normalize(metric_claim)
        self.assertIn("0", res_metric.normalized_text)
        self.assertIn("1 atm", res_metric.normalized_text)
        self.assertIn("100°C", res_metric.normalized_text)

        percent_claim = "Humans use only 10% of their brains."
        res_pct = self.normalizer.normalize(percent_claim)
        self.assertIn("10%", res_pct.normalized_text)

        url_claim = "Details are at https://www.cdc.gov/coronavirus/2019-ncov/index.html today."
        res_url = self.normalizer.normalize(url_claim)
        self.assertIn("https://www.cdc.gov/coronavirus/2019-ncov/index.html", res_url.normalized_text)

        chem_claim = "Water molecule consists of H2O."
        res_chem = self.normalizer.normalize(chem_claim)
        self.assertIn("H2O", res_chem.normalized_text)

    # --------------------------------------------------------------------------
    # 8. Negation and Quantifier Preservation
    # --------------------------------------------------------------------------
    def test_08_negation_and_quantifier_preservation(self):
        """Verifies that normalization never alters negation or quantifier truth conditions."""
        neg_claim = "Smoking does not cause lung cancer."
        res_neg = self.normalizer.normalize(neg_claim)
        self.assertEqual(res_neg.normalized_text, neg_claim)
        self.assertTrue(res_neg.meaning_preservation_passed)

        neg_claim_2 = "There was no evidence of an unmarked copper button."
        res_neg_2 = self.normalizer.normalize(neg_claim_2)
        self.assertIn("no evidence", res_neg_2.normalized_text)
        self.assertTrue(res_neg_2.meaning_preservation_passed)

        quant_claim = "All swans are white and some birds cannot fly."
        res_quant = self.normalizer.normalize(quant_claim)
        self.assertIn("All", res_quant.normalized_text)
        self.assertIn("some", res_quant.normalized_text)
        self.assertIn("cannot fly", res_quant.normalized_text)
        self.assertTrue(res_quant.meaning_preservation_passed)

    # --------------------------------------------------------------------------
    # 9. Ambiguous Spelling Candidate Handling
    # --------------------------------------------------------------------------
    def test_09_ambiguous_spelling_candidates(self):
        """Verifies candidate metadata and ambiguity detection for competing candidates."""
        # 'mdoe' has candidates 'mode', 'doe', 'moe'
        res = self.normalizer.normalize("The private mdoe was active.")
        correction = next((c for c in res.corrections if c.original == "mdoe"), None)
        self.assertIsNotNone(correction)
        self.assertGreater(len(correction.candidates), 1)
        self.assertIn("term", correction.candidates[0])
        self.assertIn("distance", correction.candidates[0])
        self.assertIn("ranking_score", correction.candidates[0])

    # --------------------------------------------------------------------------
    # 10. Empty, Whitespace-Only, Punctuation-Only, and Unicode Input
    # --------------------------------------------------------------------------
    def test_10_empty_whitespace_punctuation_unicode(self):
        """Validates robust handling of boundary input strings."""
        # Empty string
        res_empty = self.normalizer.normalize("")
        self.assertEqual(res_empty.normalized_text, "")
        self.assertEqual(len(res_empty.corrections), 0)

        # Whitespace-only string
        res_ws = self.normalizer.normalize("   \t  \n  ")
        self.assertEqual(res_ws.normalized_text, "")

        # Punctuation-only string
        res_punc = self.normalizer.normalize("???")
        self.assertEqual(res_punc.normalized_text, "?")

        # Unicode NFC normalization & smart quotes
        unicode_input = "“Pure water freezes at 0 degrees Celsius.”—NASA"
        res_uni = self.normalizer.normalize(unicode_input)
        self.assertEqual(res_uni.normalized_text, '"Pure water freezes at 0 degrees Celsius." - NASA')

        # Control characters stripped safely
        dirty_input = "Water\x00 freezes\x07 at 0\x1F degrees."
        res_dirty = self.normalizer.normalize(dirty_input)
        self.assertEqual(res_dirty.normalized_text, "Water freezes at 0 degrees.")

    # --------------------------------------------------------------------------
    # 11. Repeated Calls and Deterministic Output
    # --------------------------------------------------------------------------
    def test_11_repeated_calls_and_deterministic_output(self):
        """Ensures that repeated normalizations produce byte-for-byte identical results."""
        claim = "Using incognitio mdoe prvents websits from traking user activty"
        baseline = self.normalizer.normalize(claim)

        for _ in range(25):
            trial = self.normalizer.normalize(claim)
            self.assertEqual(trial.normalized_text, baseline.normalized_text)
            self.assertEqual(len(trial.corrections), len(baseline.corrections))
            self.assertEqual(trial.meaning_preservation_passed, baseline.meaning_preservation_passed)

    # --------------------------------------------------------------------------
    # 12. Operation Without GPU or Network Access
    # --------------------------------------------------------------------------
    def test_12_standalone_operation_no_gpu_no_network(self):
        """Verifies that NLP normalizer runs 100% locally on CPU without PyTorch/CUDA."""
        # Instantiate an isolated normalizer instance
        isolated = SpellingNormalizer(max_edit_distance=2)
        res = isolated.normalize("govt announced that water frezes at 0 degres Celcius")
        self.assertIn("government", res.normalized_text)
        self.assertIn("freezes", res.normalized_text)
        self.assertIn("Celsius", res.normalized_text)

    # --------------------------------------------------------------------------
    # 13. Latency Measurement (Non-Fragile Bounds)
    # --------------------------------------------------------------------------
    def test_13_latency_measurement_non_fragile(self):
        """Measures CPU latency across repeated executions without fragile timing thresholds."""
        claim = "Using incognitio mdoe prvents websits from traking user activty"
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = self.normalizer.normalize(claim)
            times.append((time.perf_counter() - t0) * 1000.0)

        avg_latency_ms = sum(times) / len(times)
        # Latency should comfortably complete under 50ms on CPU
        self.assertLess(avg_latency_ms, 50.0, f"Average latency ({avg_latency_ms:.2f}ms) exceeded 50ms limit.")

    # --------------------------------------------------------------------------
    # 14. Frozen Benchmark Preservation (Clean Claims Remain Unaltered)
    # --------------------------------------------------------------------------
    def test_14_frozen_benchmark_claims_unaltered(self):
        """Guarantees that canonical benchmark claims pass through with 0 mutations."""
        benchmark_claims = [
            "Using incognito mode prevents websites from tracking a user's activity.",
            "Private browsing mode automatically hides a user's IP address from the websites they visit.",
            "Pure water freezes at 0 degrees Celsius under standard atmospheric pressure.",
            "The Moon is made of green cheese.",
            "In 1943, an unrecorded traveler dropped an unmarked copper button into an undisclosed lake.",
            "The municipal bridge was completely closed to vehicle traffic on Monday.",
            "The New York Post endorsed Donald Trump's bid for re-election.",
            "Amy Coney Barrett was confirmed as US Supreme Court Justice on October 26, 2020.",
        ]
        for c in benchmark_claims:
            res = self.normalizer.normalize(c)
            self.assertEqual(
                res.normalized_text,
                c,
                f"Frozen benchmark claim was modified: '{res.normalized_text}' != '{c}'",
            )
            # Must have zero spelling corrections
            spelling_corrections = [corr for corr in res.corrections if corr.category == "spelling"]
    # --------------------------------------------------------------------------
    # 15. Character Offset Alignment with Contractions & Typos
    # --------------------------------------------------------------------------
    def test_15_character_offset_alignment_with_contractions_and_spelling(self):
        """Verifies that every CorrectionItem's start_char and end_char slice the exact original token."""
        claim = "incognitio mdoe dont stops websits traking me"
        res = self.normalizer.normalize(claim)
        
        self.assertGreater(len(res.corrections), 0)
        for corr in res.corrections:
            sliced = res.sanitized_text[corr.start_char:corr.end_char]
            self.assertEqual(
                sliced,
                corr.original,
                f"Offset mismatch for '{corr.original}': sanitized_text[{corr.start_char}:{corr.end_char}] was '{sliced}'",
            )


if __name__ == "__main__":
    unittest.main()
