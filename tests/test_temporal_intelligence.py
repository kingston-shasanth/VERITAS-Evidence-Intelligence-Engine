import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from unittest import TestCase
from veritas.temporal.temporal_analyzer import TemporalAnalyzer
from veritas.reasoning.types import TemporalScope, TemporalStatus


class TestTemporalIntelligence(TestCase):

    def setUp(self):
        self.analyzer = TemporalAnalyzer()

    def test_extract_from_wayback_url(self):
        url = "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-imac/"
        meta = self.analyzer.extract_metadata(source_url=url)

        self.assertEqual(meta.publication_date, "2020-11-29")
        self.assertEqual(meta.date_provenance, "archive_url")
        self.assertTrue(meta.is_verified)
        self.assertGreaterEqual(meta.date_confidence, 0.90)

    def test_extract_from_news_url_path(self):
        url = "https://www.aljazeera.com/news/2020/10/26/french-president-comments"
        meta = self.analyzer.extract_metadata(source_url=url)

        self.assertEqual(meta.publication_date, "2020-10-26")
        self.assertEqual(meta.date_provenance, "news_path")
        self.assertTrue(meta.is_verified)

    def test_extract_from_text_month_day_year(self):
        text = "On October 30, 2020, officials released an official clarification."
        meta = self.analyzer.extract_metadata(text=text)

        self.assertEqual(meta.event_date, "2020-10-30")
        self.assertEqual(meta.date_provenance, "text_extraction")
        self.assertFalse(meta.is_verified)

    def test_extract_from_text_historical_month_year(self):
        text = "Apollo 11 landed on the lunar surface in July 1969."
        meta = self.analyzer.extract_metadata(text=text)

        self.assertEqual(meta.event_date, "1969-07-01")
        self.assertEqual(meta.date_provenance, "text_extraction")

    def test_missing_date_never_uses_system_time(self):
        text = "Photosynthetic organisms convert carbon dioxide into organic compounds."
        meta = self.analyzer.extract_metadata(text=text, source_url="https://example.org/biology")

        self.assertIsNone(meta.publication_date)
        self.assertIsNone(meta.event_date)
        self.assertEqual(meta.date_provenance, "unknown")
        self.assertEqual(meta.date_confidence, 0.0)

    def test_temporal_claim_scope_classification(self):
        # Current claim
        c1 = "The highway currently remains closed to vehicular traffic."
        a1 = self.analyzer.analyze_claim(c1, [])
        self.assertEqual(a1["temporal_scope"], TemporalScope.CURRENT.value)

        # Historical claim
        c2 = "The atomic bomb was dropped on Hiroshima in August 1945."
        a2 = self.analyzer.analyze_claim(c2, [])
        self.assertEqual(a2["temporal_scope"], TemporalScope.HISTORICAL.value)

    def test_timeline_chronological_ordering_and_shift(self):
        claim = "The bridge is currently open to traffic."
        evidence = [
            {
                "id": "E-02",
                "source": "https://web.archive.org/web/20230510120000/https://news.org/reopening",
                "text": "The bridge reopened on May 10, 2023 after safety certifications.",
                "stance": "support",
            },
            {
                "id": "E-01",
                "source": "https://web.archive.org/web/20230501090000/https://news.org/closure",
                "text": "The bridge was closed on May 1, 2023 due to cable fractures.",
                "stance": "refute",
            },
        ]
        res = self.analyzer.analyze_claim(claim, evidence)

        events = res["timeline_events"]
        self.assertEqual(len(events), 2)
        # Must be chronologically sorted: 2023-05-01 before 2023-05-10
        self.assertEqual(events[0]["date"], "2023-05-01")
        self.assertEqual(events[1]["date"], "2023-05-10")
        self.assertEqual(res["temporal_status"], TemporalStatus.OUTDATED_SUPERSEDED.value)

    def test_undated_evidence_reports_unresolved_honestly(self):
        claim = "Dark energy comprises roughly 68 percent of the universe."
        evidence = [
            {"id": "E-01", "source": "Cosmology Journal", "text": "Undated physics paragraph.", "stance": "support"}
        ]
        res = self.analyzer.analyze_claim(claim, evidence)
        self.assertEqual(res["temporal_status"], TemporalStatus.UNRESOLVED.value)
        self.assertIn("Temporal status unresolved", res["explanation"])
