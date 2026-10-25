"""
VERITAS-X Phase 6: End-to-End Quality Assurance & Acceptance Test Suite.
Validates:
1. Supporting, refuting, neutral, conflicting, and insufficient-evidence fixtures.
2. Rationale-to-evidence stance alignment (no false refutation claims).
3. Source provenance and document identifiers preservation.
4. Input validation (empty query, malformed JSON).
5. Sanitized error handling (zero stack trace leakage).
6. Accessibility and reduced-motion stylesheet integrity.
"""
import os
import sys
import json
import unittest
from unittest.mock import MagicMock

# Ensure repo root and src/ are on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from veritas.data.schema import CanonicalVeracity
from veritas.verification.engine import ForensicVerdict
from veritas.verification.temporal_analyzer import TemporalReport, TemporalStatus
from veritas.verification.evidence_aggregator import AggregatedEvidenceVerdict
from api.adapter import VeritasAPIAdapter
from api.server import create_server
import urllib.request
import urllib.error
import threading


def create_mock_verdict(
    verdict: CanonicalVeracity,
    confidence: float,
    evidence_items: list,
    was_abstained: bool = False,
    abstention_reason: str = None,
) -> ForensicVerdict:
    probs = {
        CanonicalVeracity.VERIFIED_TRUE.value: 0.1,
        CanonicalVeracity.VERIFIED_FALSE.value: 0.1,
        CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value: 0.1,
        CanonicalVeracity.INSUFFICIENT_EVIDENCE.value: 0.1,
    }
    probs[verdict.value] = confidence
    # Normalize remaining
    rem = (1.0 - confidence) / 3.0
    for k in probs:
        if k != verdict.value:
            probs[k] = rem

    return ForensicVerdict(
        claim_id="QA-TEST-001",
        claim_text="QA Synthetic Claim",
        verdict=verdict,
        confidence=confidence,
        uncalibrated_confidence=confidence,
        was_abstained=was_abstained,
        abstention_reason=abstention_reason,
        class_probabilities=probs,
        temporal_report=TemporalReport(
            status=TemporalStatus.ALIGNED,
            delta_days=0,
            claim_date=None,
            evidence_date=None,
            warning_message=None,
            confidence_penalty=0.0,
        ),
        evidence_used_count=len(evidence_items),
        duplicates_filtered=0,
        irrelevant_passages_filtered=0,
        evidence_breakdown=evidence_items,
    )


class TestPhase6EndToEndQA(unittest.TestCase):
    """Phase 6 QA Acceptance tests against controlled synthetic fixtures."""

    def test_supporting_case_stance_and_rationale_consistency(self):
        """Case 1: Supporting evidence yields VERIFIED_TRUE and accurate count rationale."""
        mock_engine = MagicMock()
        ev_items = [
            {
                "doc_id": "doc_sup_01",
                "text": "Confirmed empirical data supporting claim.",
                "entailment_prob": 0.94,
                "contradiction_prob": 0.01,
                "neutral_prob": 0.05,
                "source_url": "https://doi.org/10.1000/182",
                "is_joint": False,
            }
        ]
        mock_engine.verify_claim.return_value = create_mock_verdict(
            verdict=CanonicalVeracity.VERIFIED_TRUE,
            confidence=0.88,
            evidence_items=ev_items,
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Verified scientific thesis.")

        self.assertEqual(res["verdict"], "VERIFIED_TRUE")
        self.assertAlmostEqual(res["confidence"], 0.88, places=2)
        self.assertEqual(len(res["evidence"]), 1)
        self.assertEqual(res["evidence"][0]["stance"], "support")
        self.assertEqual(res["evidence"][0]["id"], "E-01")
        self.assertEqual(res["evidence"][0]["source"], "https://doi.org/10.1000/182")
        self.assertTrue(res["evidence"][0]["is_informative"])
        self.assertIn("Local evidence supports the claim with 1 supporting passage(s)", res["reason"])

    def test_refuting_case_stance_and_rationale_consistency(self):
        """Case 2: Refuting evidence yields VERIFIED_FALSE and accurate count rationale."""
        mock_engine = MagicMock()
        ev_items = [
            {
                "doc_id": "doc_ref_01",
                "text": "Direct documentary contradiction debunking claim.",
                "entailment_prob": 0.02,
                "contradiction_prob": 0.95,
                "neutral_prob": 0.03,
                "source_url": "https://archives.gov/doc1",
                "is_joint": False,
            }
        ]
        mock_engine.verify_claim.return_value = create_mock_verdict(
            verdict=CanonicalVeracity.VERIFIED_FALSE,
            confidence=0.91,
            evidence_items=ev_items,
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Debunked historical assertion.")

        self.assertEqual(res["verdict"], "VERIFIED_FALSE")
        self.assertAlmostEqual(res["confidence"], 0.91, places=2)
        self.assertEqual(res["evidence"][0]["stance"], "refute")
        self.assertEqual(res["evidence"][0]["source"], "https://archives.gov/doc1")
        self.assertIn("Local evidence refutes the claim with 1 refuting passage(s)", res["reason"])

    def test_neutral_case_does_not_invent_refutation(self):
        """Case 3: All neutral passages MUST NOT be reported as refuting evidence."""
        mock_engine = MagicMock()
        ev_items = [
            {
                "doc_id": f"doc_neut_{i}",
                "text": f"Tangential uninformative passage {i}.",
                "entailment_prob": 0.04,
                "contradiction_prob": 0.05,
                "neutral_prob": 0.91,
                "source_url": f"https://neutral-archive.org/{i}",
                "is_joint": False,
            }
            for i in range(5)
        ]
        mock_engine.verify_claim.return_value = create_mock_verdict(
            verdict=CanonicalVeracity.VERIFIED_FALSE,
            confidence=0.35,
            evidence_items=ev_items,
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Claim with zero documentary refutation.")

        self.assertEqual(res["verdict"], "VERIFIED_FALSE")
        for ev in res["evidence"]:
            self.assertEqual(ev["stance"], "neutral")
            self.assertFalse(ev["is_informative"])

        # Crucial Phase 1 / Phase 6 check: Rationale must NOT claim local evidence refutes
        self.assertNotIn("refutes the claim with", res["reason"])
        self.assertIn("no individual retrieved passage directly refutes the proposition", res["reason"])
        self.assertIn("5 neutral/uninformative passage(s)", res["reason"])

    def test_conflicting_case_disparity_detection(self):
        """Case 4: Multi-source opposing evidence triggers disparity flag and rationale."""
        mock_engine = MagicMock()
        mock_aggregator = MagicMock()
        mock_aggregator.last_verdict = AggregatedEvidenceVerdict(
            canonical_verdict=CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED,
            confidence=0.49,
            entailment_mass=0.5,
            contradiction_mass=0.5,
            conflict_ratio=1.0,
            total_informative_mass=1.0,
            raw_logits=[0.0, 0.0, 1.5, -2.0],
            unique_passages_used=2,
            duplicates_filtered=0,
            irrelevant_passages_filtered=0,
            has_cross_passage_disparity=True,
        )
        mock_engine.aggregator = mock_aggregator

        ev_items = [
            {
                "doc_id": "doc_pro",
                "text": "Evidence arguing in favor.",
                "entailment_prob": 0.88,
                "contradiction_prob": 0.02,
                "neutral_prob": 0.10,
                "source_url": "https://source-pro.org",
                "is_joint": False,
            },
            {
                "doc_id": "doc_con",
                "text": "Evidence arguing against.",
                "entailment_prob": 0.03,
                "contradiction_prob": 0.89,
                "neutral_prob": 0.08,
                "source_url": "https://source-con.org",
                "is_joint": False,
            },
        ]
        mock_engine.verify_claim.return_value = create_mock_verdict(
            verdict=CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED,
            confidence=0.49,
            evidence_items=ev_items,
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        adapter._aggregator = mock_aggregator
        res = adapter.verify(claim="Disputed factual statement.")

        self.assertEqual(res["verdict"], "CONFLICTING_OR_CHERRYPICKED")
        self.assertTrue(res["has_cross_passage_disparity"])
        self.assertIn("Cross-passage stance disparity detected", res["reason"])
        self.assertIn("1 supporting passage(s) and 1 refuting passage(s)", res["reason"])

    def test_insufficient_evidence_and_abstention(self):
        """Case 5: Low confidence trigger selective abstention and clear justification."""
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = create_mock_verdict(
            verdict=CanonicalVeracity.INSUFFICIENT_EVIDENCE,
            confidence=0.25,
            evidence_items=[],
            was_abstained=True,
            abstention_reason="Confidence 0.25 < threshold 0.32",
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Obscure unindexed assertion.")

        self.assertEqual(res["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertTrue(res["abstained"])
        self.assertIn("Abstained from definitive verdict", res["reason"])

    def test_reduced_motion_css_accessibility_present(self):
        """Case 6: Stylesheet includes standard prefers-reduced-motion media query."""
        css_path = os.path.join(REPO_ROOT, "ui", "veritas-x-advanced-ui", "src", "styles.css")
        self.assertTrue(os.path.exists(css_path), f"styles.css not found at {css_path}")

        with open(css_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("@media (prefers-reduced-motion: reduce)", content)
        self.assertIn("animation-duration", content)
        self.assertIn("transition-duration", content)


class TestPhase6ServerSecurityAndValidation(unittest.TestCase):
    """Server-level security, error sanitization, and input rejection tests."""

    @classmethod
    def setUpClass(cls):
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = create_mock_verdict(
            verdict=CanonicalVeracity.VERIFIED_TRUE,
            confidence=0.85,
            evidence_items=[],
        )
        cls.adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        cls.server = create_server(host="127.0.0.1", port=0, adapter=cls.adapter)
        cls.host, cls.port = cls.server.server_address
        cls.base_url = f"http://{cls.host}:{cls.port}"

        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_empty_query_returns_http_400(self):
        url = f"{self.base_url}/api/verify"
        payload = json.dumps({"claim": "   "}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)
        err = json.loads(ctx.exception.read().decode("utf-8"))
        self.assertIn("error", err)

    def test_malformed_json_returns_http_400(self):
        url = f"{self.base_url}/api/verify"
        req = urllib.request.Request(
            url,
            data=b"INVALID_NOT_JSON",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_internal_exception_sanitization_zero_leak(self):
        broken_engine = MagicMock()
        broken_engine.verify_claim.side_effect = RuntimeError("CRITICAL_INTERNAL_DB_PASSWORD_LEAK")
        broken_adapter = VeritasAPIAdapter(engine=broken_engine, lazy_init=True)
        broken_server = create_server(host="127.0.0.1", port=0, adapter=broken_adapter)
        b_host, b_port = broken_server.server_address
        b_thread = threading.Thread(target=broken_server.serve_forever, daemon=True)
        b_thread.start()

        try:
            url = f"http://{b_host}:{b_port}/api/verify"
            payload = json.dumps({"claim": "Exploit attempt"}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 500)
            data = json.loads(ctx.exception.read().decode("utf-8"))
            self.assertIn("error", data)
            self.assertNotIn("CRITICAL_INTERNAL_DB_PASSWORD_LEAK", data["error"])
            self.assertNotIn("Traceback", data["error"])
        finally:
            broken_server.shutdown()
            broken_server.server_close()


if __name__ == "__main__":
    unittest.main()
