"""
Comprehensive Data Foundation Test Suite for VERITAS-X.
Covers schema conformance, dataset loaders, preprocessing, deduplication,
missing value handling, and label distribution integrity.
"""
import unittest
import os
import json
import tempfile
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from veritas.data.schema import (
    CanonicalClaim,
    CanonicalVeracity,
    DatasetSource,
    EvidenceItem,
    EvidenceType,
    ValidationReport,
)
from veritas.data.preprocessor import TextPreprocessor
from veritas.data.fever_loader import FEVERLoader
from veritas.data.averitec_loader import AVeriTeCLoader
from veritas.data.multilingual_loader import MultilingualLoader
from veritas.data.pipeline import UnifiedDataPipeline

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(WORKSPACE_ROOT, "data", "raw")


class TestDataFoundation(unittest.TestCase):

    def setUp(self):
        self.fever_path = os.path.join(RAW_DATA_DIR, "fever", "sample_dev.jsonl")
        self.averitec_path = os.path.join(RAW_DATA_DIR, "averitec", "sample_dev.json")
        self.multilingual_path = os.path.join(RAW_DATA_DIR, "multilingual", "diagnostic_multilingual.json")

    # ---------------------------------------------------------
    # 1. Preprocessing & Linguistic Analysis Tests
    # ---------------------------------------------------------
    def test_unicode_normalization_nfc(self):
        # Test combined Devanagari characters: 'क़' can be represented as single char (U+0958) or decomposed (U+0915 + U+093C)
        decomposed = "\u0915\u093C\u093F\u0924\u093E\u092C" # किताब / क़िताब
        normalized = TextPreprocessor.normalize_text(decomposed)
        self.assertEqual(normalized, unicodedata_nfc := unicodedata_normalize("NFC", decomposed))

    def test_whitespace_and_control_char_cleaning(self):
        dirty = "  Claim   text \t with  multiple \n  spaces \x00 and null char   "
        clean = TextPreprocessor.normalize_text(dirty)
        self.assertEqual(clean, "Claim text with multiple spaces and null char")

    def test_script_and_language_detection(self):
        # Hindi
        script, lang = TextPreprocessor.detect_script_and_language("भारत एक महान देश है।")
        self.assertEqual(script, "Devanagari")
        self.assertEqual(lang, "hi")

        # Bengali
        script, lang = TextPreprocessor.detect_script_and_language("রবীন্দ্রনাথ ঠাকুর সাহিত্যের নোবেল পুরস্কার পেয়েছিলেন।")
        self.assertEqual(script, "Bengali")
        self.assertEqual(lang, "bn")

        # English
        script, lang = TextPreprocessor.detect_script_and_language("The United Nations was established in 1945.")
        self.assertEqual(script, "Latin")
        self.assertEqual(lang, "en")

        # Hinglish Code-Mixed
        script, lang = TextPreprocessor.detect_script_and_language("Yeh news bilkul fake hai aur sarkar ne aisi koi scheme shuru nahi ki.")
        self.assertEqual(script, "Latin-CodeMixed")
        self.assertEqual(lang, "hi-en")

    def test_date_parser_formats(self):
        self.assertEqual(TextPreprocessor.parse_date_to_iso("31-10-2020"), "2020-10-31")
        self.assertEqual(TextPreprocessor.parse_date_to_iso("2023-08-23"), "2023-08-23")
        self.assertEqual(TextPreprocessor.parse_date_to_iso("23/08/2023"), "2023-08-23")
        self.assertIsNone(TextPreprocessor.parse_date_to_iso("Unknown date"))
        self.assertIsNone(TextPreprocessor.parse_date_to_iso(None))

    # ---------------------------------------------------------
    # 2. FEVER Loader & Integrity Tests
    # ---------------------------------------------------------
    def test_fever_loader_actual_file(self):
        self.assertTrue(os.path.exists(self.fever_path), f"Missing {self.fever_path}")
        loader = FEVERLoader(self.fever_path)
        claims, report = loader.load_and_validate()

        self.assertEqual(report.total_records, 600)
        self.assertEqual(report.valid_records, 600)
        self.assertEqual(report.invalid_records, 0)
        self.assertTrue(report.is_healthy())

        # Assert balanced distribution in sample
        self.assertEqual(report.label_distribution["SUPPORTS"], 200)
        self.assertEqual(report.label_distribution["REFUTES"], 200)
        self.assertEqual(report.label_distribution["NOT ENOUGH INFO"], 200)

        # Assert canonical label mapping
        self.assertEqual(report.canonical_distribution[CanonicalVeracity.VERIFIED_TRUE.value], 200)
        self.assertEqual(report.canonical_distribution[CanonicalVeracity.VERIFIED_FALSE.value], 200)
        self.assertEqual(report.canonical_distribution[CanonicalVeracity.INSUFFICIENT_EVIDENCE.value], 200)

        # Assert sample record properties
        sample = claims[0]
        self.assertTrue(sample.claim_id.startswith("fever_"))
        self.assertEqual(sample.source_dataset, DatasetSource.FEVER)
        self.assertIsInstance(sample.claim_text, str)
        self.assertTrue(len(sample.claim_text) > 5)

    def test_fever_loader_corrupt_record_handling(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write('{"id": 1, "claim": "Valid claim", "label": "SUPPORTS", "evidence": []}\n')
            f.write('corrupt json line\n')
            f.write('{"id": 2, "claim": "", "label": "REFUTES"}\n') # Missing claim text
            f.write('{"id": 3, "claim": "Unknown label claim", "label": "SOME_UNKNOWN_LABEL"}\n')
            tmp_path = f.name

        try:
            loader = FEVERLoader(tmp_path)
            claims, report = loader.load_and_validate()
            self.assertEqual(report.total_records, 4)
            self.assertEqual(report.valid_records, 1)
            self.assertEqual(report.invalid_records, 3)
            self.assertEqual(len(claims), 1)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ---------------------------------------------------------
    # 3. AVeriTeC Loader & Integrity Tests
    # ---------------------------------------------------------
    def test_averitec_loader_actual_file(self):
        self.assertTrue(os.path.exists(self.averitec_path), f"Missing {self.averitec_path}")
        loader = AVeriTeCLoader(self.averitec_path)
        claims, report = loader.load_and_validate()

        self.assertEqual(report.total_records, 500)
        self.assertEqual(report.valid_records, 491)
        self.assertEqual(report.duplicates_detected, 9)
        self.assertEqual(report.invalid_records, 0)
        self.assertTrue(report.is_healthy())

        # Assert 4-class distribution in AVeriTeC dev set
        self.assertIn("Refuted", report.label_distribution)
        self.assertIn("Supported", report.label_distribution)
        self.assertIn("Conflicting Evidence/Cherrypicking", report.label_distribution)
        self.assertIn("Not Enough Evidence", report.label_distribution)

        # Assert canonical harmonization
        self.assertIn(CanonicalVeracity.VERIFIED_TRUE.value, report.canonical_distribution)
        self.assertIn(CanonicalVeracity.VERIFIED_FALSE.value, report.canonical_distribution)
        self.assertIn(CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value, report.canonical_distribution)
        self.assertIn(CanonicalVeracity.INSUFFICIENT_EVIDENCE.value, report.canonical_distribution)

        # Check evidence extraction from questions/answers
        claims_with_evidence = [c for c in claims if len(c.evidence) > 0]
        self.assertGreater(len(claims_with_evidence), 400)
        sample_ev = claims_with_evidence[0].evidence[0]
        self.assertIsNotNone(sample_ev.evidence_type)
        self.assertIn("question", sample_ev.metadata)

    def test_averitec_malformed_handling(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write(json.dumps([
                {"claim": "Valid claim", "label": "Supported", "questions": []},
                {"claim": "Missing label claim"}, # No label
                "invalid string instead of dict",
            ]))
            tmp_path = f.name

        try:
            loader = AVeriTeCLoader(tmp_path)
            claims, report = loader.load_and_validate()
            self.assertEqual(report.total_records, 3)
            self.assertEqual(report.valid_records, 1)
            self.assertEqual(report.invalid_records, 2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ---------------------------------------------------------
    # 4. Multilingual Benchmark Loader Tests
    # ---------------------------------------------------------
    def test_multilingual_loader_actual_file(self):
        self.assertTrue(os.path.exists(self.multilingual_path), f"Missing {self.multilingual_path}")
        loader = MultilingualLoader(self.multilingual_path)
        claims, report = loader.load_and_validate()

        self.assertEqual(report.total_records, 20)
        self.assertEqual(report.valid_records, 20)
        self.assertEqual(report.invalid_records, 0)
        self.assertTrue(report.is_healthy())

        # Assert language coverage
        self.assertIn("hi", report.language_distribution)
        self.assertIn("bn", report.language_distribution)
        self.assertIn("hi-en", report.language_distribution)
        self.assertIn("en", report.language_distribution)

        self.assertEqual(report.language_distribution["hi"], 5)
        self.assertEqual(report.language_distribution["bn"], 5)
        self.assertEqual(report.language_distribution["hi-en"], 5)
        self.assertEqual(report.language_distribution["en"], 5)

    # ---------------------------------------------------------
    # 5. Deduplication & Deterministic Hashing Tests
    # ---------------------------------------------------------
    def test_deduplication(self):
        c1 = CanonicalClaim(
            claim_id="test_1",
            claim_text="The Earth orbits the Sun.",
            source_dataset=DatasetSource.FEVER,
            raw_label="SUPPORTS",
            canonical_label=CanonicalVeracity.VERIFIED_TRUE,
        )
        c2 = CanonicalClaim(
            claim_id="test_2",
            claim_text="  the earth   orbits the sun.  ", # Whitespace & case variance
            source_dataset=DatasetSource.FEVER,
            raw_label="SUPPORTS",
            canonical_label=CanonicalVeracity.VERIFIED_TRUE,
        )
        self.assertEqual(c1.compute_hash(), c2.compute_hash())

    # ---------------------------------------------------------
    # 6. Unified Pipeline Integration & Export Tests
    # ---------------------------------------------------------
    def test_unified_pipeline_integration_and_export(self):
        pipeline = UnifiedDataPipeline()
        
        # Load all datasets
        pipeline.load_fever(self.fever_path)
        pipeline.load_averitec(self.averitec_path)
        pipeline.load_multilingual(self.multilingual_path)

        combined = pipeline.get_combined_dataset()
        # 600 (FEVER unique) + 491 (AVeriTeC unique, 9 duplicates filtered) + 20 (Multilingual) = 1111 records
        self.assertEqual(len(combined), 1111)

        global_report = pipeline.generate_global_report()
        self.assertEqual(global_report.valid_records, 1111)
        self.assertEqual(global_report.duplicates_detected, 0)

        # Test export and roundtrip reload
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            tmp_export = f.name

        try:
            exported_count = pipeline.export_canonical_jsonl(tmp_export)
            self.assertEqual(exported_count, 1111)

            # Read back first line and verify deserialization
            with open(tmp_export, "r", encoding="utf-8") as in_f:
                first_line = in_f.readline()
                obj_dict = json.loads(first_line)
                reloaded_claim = CanonicalClaim.from_dict(obj_dict)
                self.assertIsInstance(reloaded_claim, CanonicalClaim)
                self.assertEqual(reloaded_claim.claim_id, combined[0].claim_id)
        finally:
            if os.path.exists(tmp_export):
                os.remove(tmp_export)


def unicodedata_normalize(form, text):
    import unicodedata
    return unicodedata.normalize(form, text)


if __name__ == "__main__":
    unittest.main()
