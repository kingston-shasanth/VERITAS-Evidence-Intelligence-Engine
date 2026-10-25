"""
Unit tests for VERITAS-X Live API Adapter and HTTP Server.
Uses mocked verification engine to ensure tests are fast, self-contained,
and do NOT download models or run held-out benchmark evaluations.
"""
import os
import sys
import json
import urllib.request
import urllib.error
import threading
from unittest import TestCase, mock
from unittest.mock import MagicMock, patch

# Ensure repo root and src/ are on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from veritas.data.schema import CanonicalVeracity
from veritas.verification.engine import ForensicVerdict
from veritas.verification.temporal_analyzer import TemporalReport, TemporalStatus
from veritas.verification.evidence_aggregator import AggregatedEvidenceVerdict

from api.adapter import VeritasAPIAdapter, ObservableEvidenceAggregator
from api.server import create_server


def make_dummy_verdict(
    claim: str = "Test claim",
    verdict: CanonicalVeracity = CanonicalVeracity.VERIFIED_TRUE,
    confidence: float = 0.82,
    was_abstained: bool = False,
    abstention_reason: str = "Confidence above threshold",
    probabilities: dict = None,
    evidence_breakdown: list = None,
) -> ForensicVerdict:
    """Creates a mock ForensicVerdict for deterministic testing."""
    probs = probabilities or {
        CanonicalVeracity.VERIFIED_TRUE.value: 0.82,
        CanonicalVeracity.VERIFIED_FALSE.value: 0.08,
        CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value: 0.04,
        CanonicalVeracity.INSUFFICIENT_EVIDENCE.value: 0.06,
    }
    breakdown = evidence_breakdown or [
        {
            "doc_id": "test_doc_01",
            "text": "Confirmed evidence passage for testing.",
            "weight": 1.0,
            "entailment_prob": 0.88,
            "contradiction_prob": 0.02,
            "neutral_prob": 0.10,
            "source_url": "https://example.org/source1",
            "is_joint": False,
            "is_active_supporter": True,
            "is_active_refuter": False,
        }
    ]
    return ForensicVerdict(
        claim_id="VX-TEST01",
        claim_text=claim,
        verdict=verdict,
        confidence=confidence,
        uncalibrated_confidence=0.85,
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
        evidence_used_count=len(breakdown),
        duplicates_filtered=0,
        irrelevant_passages_filtered=0,
        evidence_breakdown=breakdown,
    )


class TestVeritasAPIAdapter(TestCase):
    """Tests for VeritasAPIAdapter mapping logic and contract compliance."""

    def test_verify_returns_exact_contract_schema(self):
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = make_dummy_verdict()

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Test scientific hypothesis.")

        # Check top-level contract keys
        expected_keys = {
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
        self.assertEqual(set(res.keys()), expected_keys)
        self.assertEqual(res["claim"], "Test scientific hypothesis.")
        self.assertEqual(res["verdict"], "VERIFIED_TRUE")
        self.assertAlmostEqual(res["confidence"], 0.82, places=3)
        self.assertFalse(res["abstained"])
        self.assertIsInstance(res["has_cross_passage_disparity"], bool)
        self.assertIsInstance(res["reason"], str)

        # Check 4-class probabilities
        probs = res["probabilities"]
        expected_prob_keys = {
            "VERIFIED_TRUE",
            "VERIFIED_FALSE",
            "CONFLICTING_OR_CHERRYPICKED",
            "INSUFFICIENT_EVIDENCE",
        }
        self.assertEqual(set(probs.keys()), expected_prob_keys)

        # Check evidence schema
        self.assertEqual(len(res["evidence"]), 1)
        ev = res["evidence"][0]
        self.assertEqual(ev["id"], "E-01")
        self.assertEqual(ev["source"], "https://example.org/source1")
        self.assertEqual(ev["stance"], "support")
        self.assertAlmostEqual(ev["score"], 0.88, places=3)

    def test_verify_empty_claim_raises_value_error(self):
        mock_engine = MagicMock()
        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        with self.assertRaises(ValueError):
            adapter.verify("")
        with self.assertRaises(ValueError):
            adapter.verify("   \n\t  ")

    def test_verify_user_evidence_forwarding(self):
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = make_dummy_verdict()
        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)

        user_snippets = ["Passage 1 text", "Passage 2 text"]
        adapter.verify(claim="Test claim", user_evidence=user_snippets)

        mock_engine.verify_claim.assert_called_once()
        _, kwargs = mock_engine.verify_claim.call_args
        cand = kwargs.get("candidate_passages")
        self.assertIsNotNone(cand)
        self.assertEqual(len(cand), 2)
        self.assertEqual(cand[0]["text"], "Passage 1 text")
        self.assertEqual(cand[1]["text"], "Passage 2 text")

    def test_verify_handles_abstention(self):
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = make_dummy_verdict(
            verdict=CanonicalVeracity.INSUFFICIENT_EVIDENCE,
            confidence=0.28,
            was_abstained=True,
            abstention_reason="Confidence 0.28 < threshold 0.32",
        )
        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Unverifiable rumor")

        self.assertTrue(res["abstained"])
        self.assertIn("Abstained", res["reason"])

    def test_verify_detects_cross_passage_disparity(self):
        mock_engine = MagicMock()
        mock_aggregator = MagicMock()
        mock_aggregator.last_verdict = AggregatedEvidenceVerdict(
            canonical_verdict=CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED,
            confidence=0.48,
            entailment_mass=0.45,
            contradiction_mass=0.55,
            conflict_ratio=0.85,
            total_informative_mass=1.0,
            raw_logits=[0.0, 0.0, 1.2, -2.0],
            unique_passages_used=2,
            duplicates_filtered=0,
            irrelevant_passages_filtered=0,
            has_cross_passage_disparity=True,
        )
        mock_engine.aggregator = mock_aggregator
        mock_engine.verify_claim.return_value = make_dummy_verdict(
            verdict=CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED,
            confidence=0.48,
            was_abstained=False,
            evidence_breakdown=[
                {
                    "doc_id": "doc_a",
                    "text": "Supports claim.",
                    "entailment_prob": 0.85,
                    "contradiction_prob": 0.05,
                    "neutral_prob": 0.10,
                    "source_url": "https://source-a.org",
                    "is_joint": False,
                    "is_active_supporter": True,
                    "is_active_refuter": False,
                },
                {
                    "doc_id": "doc_b",
                    "text": "Refutes claim.",
                    "entailment_prob": 0.05,
                    "contradiction_prob": 0.85,
                    "neutral_prob": 0.10,
                    "source_url": "https://source-b.org",
                    "is_joint": False,
                    "is_active_supporter": False,
                    "is_active_refuter": True,
                },
            ],
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        adapter._aggregator = mock_aggregator
        res = adapter.verify(claim="Disputed public policy event")

        self.assertTrue(res["has_cross_passage_disparity"])
        self.assertEqual(res["verdict"], "CONFLICTING_OR_CHERRYPICKED")
        self.assertEqual(len(res["evidence"]), 2)
        self.assertEqual(res["evidence"][0]["stance"], "support")
        self.assertEqual(res["evidence"][1]["stance"], "refute")


class TestHTTPServerIntegration(TestCase):
    """Tests HTTP server handling, CORS headers, validation, and error sanitization."""

    @classmethod
    def setUpClass(cls):
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = make_dummy_verdict(claim="Server test claim")
        cls.mock_adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)

        # Bind to port 0 for an available dynamic ephemeral port
        cls.server = create_server(host="127.0.0.1", port=0, adapter=cls.mock_adapter)
        cls.host, cls.port = cls.server.server_address
        cls.base_url = f"http://{cls.host}:{cls.port}"

        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_health_check_endpoint(self):
        url = f"{self.base_url}/api/health"
        req = urllib.request.Request(url, headers={"Origin": "http://127.0.0.1:5173"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "http://127.0.0.1:5173")
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "ok")
            self.assertEqual(data.get("version"), "Phase 3.2.5")

    def test_cors_preflight_allowed_origin(self):
        url = f"{self.base_url}/api/verify"
        req = urllib.request.Request(url, method="OPTIONS", headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        })
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 204)
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "http://127.0.0.1:5173")
            self.assertIn("POST", resp.headers.get("Access-Control-Allow-Methods", ""))

    def test_cors_preflight_unauthorized_origin_rejected(self):
        url = f"{self.base_url}/api/verify"
        req = urllib.request.Request(url, method="OPTIONS", headers={
            "Origin": "http://untrusted-external-site.com",
            "Access-Control-Request-Method": "POST",
        })
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 403)

    def test_post_verify_success(self):
        url = f"{self.base_url}/api/verify"
        payload = json.dumps({"claim": "Server test claim"}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Origin": "http://127.0.0.1:5173",
            },
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "http://127.0.0.1:5173")
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("verdict"), "VERIFIED_TRUE")
            self.assertIn("probabilities", data)

    def test_post_verify_empty_claim_returns_400(self):
        url = f"{self.base_url}/api/verify"
        payload = json.dumps({"claim": "   "}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)
        err = json.loads(ctx.exception.read().decode("utf-8"))
        self.assertIn("error", err)

    def test_post_verify_sanitizes_internal_exceptions(self):
        # Configure mock engine to raise an unhandled internal exception
        broken_engine = MagicMock()
        broken_engine.verify_claim.side_effect = RuntimeError("Internal private system failure message")
        broken_adapter = VeritasAPIAdapter(engine=broken_engine, lazy_init=True)

        broken_server = create_server(host="127.0.0.1", port=0, adapter=broken_adapter)
        b_host, b_port = broken_server.server_address
        b_thread = threading.Thread(target=broken_server.serve_forever, daemon=True)
        b_thread.start()

        try:
            url = f"http://{b_host}:{b_port}/api/verify"
            payload = json.dumps({"claim": "Trigger failure"}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 500)
            data = json.loads(ctx.exception.read().decode("utf-8"))
            self.assertIn("error", data)
            # Ensure no stack trace or internal private details leaked to browser
            self.assertNotIn("Internal private system failure message", data["error"])
            self.assertNotIn("Traceback", data["error"])
        finally:
            broken_server.shutdown()
            broken_server.server_close()


class TestStanceMappingAndVerdictRegression(TestCase):
    """
    Regression tests verifying stance mapping, 4-class verdict interpretation,
    and honest behavior on real scientific premises like water freezing.
    """

    def test_evidence_stance_mapping_support(self):
        """Passage with dominant entailment probability must map to 'support'."""
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = make_dummy_verdict(
            evidence_breakdown=[{
                "doc_id": "doc_sup",
                "text": "Entailing passage text",
                "entailment_prob": 0.89,
                "contradiction_prob": 0.02,
                "neutral_prob": 0.09,
                "source_url": "https://science.org/reference",
                "is_joint": False,
            }]
        )
        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Test hypothesis")
        self.assertEqual(len(res["evidence"]), 1)
        self.assertEqual(res["evidence"][0]["stance"], "support")
        self.assertAlmostEqual(res["evidence"][0]["score"], 0.89, places=2)

    def test_evidence_stance_mapping_refute(self):
        """Passage with dominant contradiction probability must map to 'refute'."""
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = make_dummy_verdict(
            verdict=CanonicalVeracity.VERIFIED_FALSE,
            evidence_breakdown=[{
                "doc_id": "doc_ref",
                "text": "Contradicting passage text",
                "entailment_prob": 0.03,
                "contradiction_prob": 0.91,
                "neutral_prob": 0.06,
                "source_url": "https://factcheck.org/ref",
                "is_joint": False,
            }]
        )
        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Test hypothesis")
        self.assertEqual(res["evidence"][0]["stance"], "refute")
        self.assertAlmostEqual(res["evidence"][0]["score"], 0.91, places=2)

    def test_evidence_stance_mapping_neutral(self):
        """Passage with dominant neutral probability must map to 'neutral'."""
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = make_dummy_verdict(
            verdict=CanonicalVeracity.INSUFFICIENT_EVIDENCE,
            evidence_breakdown=[{
                "doc_id": "doc_neut",
                "text": "Uninformative or tangential passage text",
                "entailment_prob": 0.08,
                "contradiction_prob": 0.11,
                "neutral_prob": 0.81,
                "source_url": "https://tangential.org/note",
                "is_joint": False,
            }]
        )
        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Test hypothesis")
        self.assertEqual(res["evidence"][0]["stance"], "neutral")
        self.assertAlmostEqual(res["evidence"][0]["score"], 0.81, places=2)

    def test_all_four_verdict_classes_serialization(self):
        """Validates that all four canonical verdict classes serialize correctly."""
        classes = [
            (CanonicalVeracity.VERIFIED_TRUE, "VERIFIED_TRUE"),
            (CanonicalVeracity.VERIFIED_FALSE, "VERIFIED_FALSE"),
            (CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED, "CONFLICTING_OR_CHERRYPICKED"),
            (CanonicalVeracity.INSUFFICIENT_EVIDENCE, "INSUFFICIENT_EVIDENCE"),
        ]
        for enum_val, str_val in classes:
            mock_engine = MagicMock()
            mock_engine.verify_claim.return_value = make_dummy_verdict(verdict=enum_val)
            adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
            res = adapter.verify(claim="Sample claim")
            self.assertEqual(res["verdict"], str_val)
            self.assertIn(str_val, res["probabilities"])
            self.assertIsInstance(res["reason"], str)
            self.assertTrue(len(res["reason"]) > 0)

    def test_water_freezing_pipeline_trace_neutral_insufficient_explanation(self):
        """
        Regression trace for the user-supplied example:
        Claim: 'Pure water freezes at 0 degrees Celsius at sea level.'
        Supplied passage: 'At standard atmospheric pressure, pure water freezes at 0 degrees Celsius.'

        Tests the pipeline trace:
        1. When NLI evaluates this (premise=passage, hypothesis=claim), DeBERTa without
           world-knowledge connecting 1 atm to sea level assigns NEUTRAL (~0.8003).
        2. Neutral mass exceeds threshold (0.70), discounting informative mass below min_mass (0.10).
        3. Aggregator produces INSUFFICIENT_EVIDENCE.
        4. Post-temperature calibration (T=4.1039) yields confidence ~0.3244 (32.4%).
        5. The pipeline honestly preserves this result without hardcoding or fabrication.
        """
        mock_engine = MagicMock()
        # Simulated genuine output from frozen pipeline
        mock_engine.verify_claim.return_value = ForensicVerdict(
            claim_id="VX-2FDACD",
            claim_text="Pure water freezes at 0 degrees Celsius at sea level.",
            verdict=CanonicalVeracity.INSUFFICIENT_EVIDENCE,
            confidence=0.3244,
            uncalibrated_confidence=0.3244,
            was_abstained=False,
            abstention_reason="Insufficient informative evidence mass retrieved to adjudicate this claim.",
            class_probabilities={
                CanonicalVeracity.VERIFIED_TRUE.value: 0.2938,
                CanonicalVeracity.VERIFIED_FALSE.value: 0.2948,
                CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value: 0.0870,
                CanonicalVeracity.INSUFFICIENT_EVIDENCE.value: 0.3244,
            },
            temporal_report=TemporalReport(
                status=TemporalStatus.ALIGNED,
                delta_days=0,
                claim_date=None,
                evidence_date=None,
                warning_message=None,
                confidence_penalty=0.0,
            ),
            evidence_used_count=1,
            duplicates_filtered=0,
            irrelevant_passages_filtered=1,
            evidence_breakdown=[{
                "doc_id": "user_evidence_01",
                "text": "At standard atmospheric pressure, pure water freezes at 0 degrees Celsius.",
                "weight": 0.1,
                "entailment_prob": 0.0885,
                "contradiction_prob": 0.1114,
                "neutral_prob": 0.8003,
                "source_url": "User-supplied evidence passage",
                "is_joint": False,
                "is_active_supporter": False,
                "is_active_refuter": False,
            }],
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(
            claim="Pure water freezes at 0 degrees Celsius at sea level.",
            user_evidence=["At standard atmospheric pressure, pure water freezes at 0 degrees Celsius."],
        )

        self.assertEqual(res["verdict"], "INSUFFICIENT_EVIDENCE")
        self.assertAlmostEqual(res["confidence"], 0.3244, places=3)
        self.assertEqual(res["evidence"][0]["stance"], "neutral")
        self.assertAlmostEqual(res["evidence"][0]["score"], 0.8003, places=3)
        self.assertAlmostEqual(res["probabilities"]["INSUFFICIENT_EVIDENCE"], 0.3244, places=3)
        self.assertIn("Insufficient informative evidence mass", res["reason"])


class TestEvidenceIntegrityAndRationale(TestCase):
    """
    Deterministic regression tests ensuring evidence integrity, stance consistency,
    and truthful rationale generation. Prevents contradictions between evidence cards
    and report summaries.
    """

    def test_neutral_passages_rationale_does_not_claim_refutation(self):
        """
        Regression Test 1: When all passages are neutral, the rationale MUST NOT
        claim that local evidence refutes or supports the claim across N passages.
        """
        mock_engine = MagicMock()
        neutral_breakdown = [
            {
                "doc_id": f"doc_{i}",
                "text": f"Unrelated text passage {i}",
                "weight": 0.1,
                "entailment_prob": 0.01,
                "contradiction_prob": 0.02,
                "neutral_prob": 0.97,
                "source_url": f"https://example.org/{i}",
                "is_joint": False,
                "is_active_supporter": False,
                "is_active_refuter": False,
            }
            for i in range(5)
        ]
        mock_engine.verify_claim.return_value = make_dummy_verdict(
            claim="A thermonuclear bomb was detonated on the Moon.",
            verdict=CanonicalVeracity.VERIFIED_FALSE,
            confidence=0.351,
            evidence_breakdown=neutral_breakdown,
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="A thermonuclear bomb was detonated on the Moon.")

        # Assert all 5 passages are classified as neutral
        self.assertEqual(len(res["evidence"]), 5)
        for ev in res["evidence"]:
            self.assertEqual(ev["stance"], "neutral")
            self.assertFalse(ev["is_informative"])

        # The rationale must NOT claim that evidence refutes the claim across passages
        self.assertNotIn("refutes the claim with", res["reason"])
        self.assertNotIn("across 5 passage", res["reason"])
        self.assertIn("no individual retrieved passage directly refutes the proposition", res["reason"])
        self.assertIn("5 neutral/uninformative passage(s)", res["reason"])

    def test_supporting_passages_rationale_matches_stance(self):
        """
        Regression Test 2: When supporting evidence is present, rationale accurately states supporting count.
        """
        mock_engine = MagicMock()
        supporting_breakdown = [
            {
                "doc_id": "doc_sup_1",
                "text": "Valid scientific law text.",
                "weight": 1.0,
                "entailment_prob": 0.92,
                "contradiction_prob": 0.02,
                "neutral_prob": 0.06,
                "source_url": "https://science.org",
                "is_joint": False,
                "is_active_supporter": True,
                "is_active_refuter": False,
            }
        ]
        mock_engine.verify_claim.return_value = make_dummy_verdict(
            claim="Pure water freezes at 0C.",
            verdict=CanonicalVeracity.VERIFIED_TRUE,
            confidence=0.88,
            evidence_breakdown=supporting_breakdown,
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Pure water freezes at 0C.")

        self.assertEqual(res["verdict"], "VERIFIED_TRUE")
        self.assertEqual(res["evidence"][0]["stance"], "support")
        self.assertTrue(res["evidence"][0]["is_informative"])
        self.assertIn("Local evidence supports the claim with 1 supporting passage(s)", res["reason"])

    def test_refuting_passages_rationale_matches_stance(self):
        """
        Regression Test 3: When refuting evidence is present, rationale accurately states refuting count.
        """
        mock_engine = MagicMock()
        refuting_breakdown = [
            {
                "doc_id": "doc_ref_1",
                "text": "Direct counter-evidence debunking claim.",
                "weight": 1.0,
                "entailment_prob": 0.03,
                "contradiction_prob": 0.94,
                "neutral_prob": 0.03,
                "source_url": "https://debunk.org",
                "is_joint": False,
                "is_active_supporter": False,
                "is_active_refuter": True,
            }
        ]
        mock_engine.verify_claim.return_value = make_dummy_verdict(
            claim="The Moon produces its own thermonuclear light.",
            verdict=CanonicalVeracity.VERIFIED_FALSE,
            confidence=0.89,
            evidence_breakdown=refuting_breakdown,
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="The Moon produces its own thermonuclear light.")

        self.assertEqual(res["verdict"], "VERIFIED_FALSE")
        self.assertEqual(res["evidence"][0]["stance"], "refute")
        self.assertTrue(res["evidence"][0]["is_informative"])
        self.assertIn("Local evidence refutes the claim with 1 refuting passage(s)", res["reason"])

    def test_conflicting_passages_rationale_matches_disparity(self):
        """
        Regression Test 4: Conflicting evidence reporting accurately reflects opposing stances data.
        """
        mock_engine = MagicMock()
        mock_aggregator = MagicMock()
        mock_aggregator.last_verdict = AggregatedEvidenceVerdict(
            canonical_verdict=CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED,
            confidence=0.72,
            entailment_mass=0.45,
            contradiction_mass=0.55,
            conflict_ratio=0.90,
            total_informative_mass=1.0,
            raw_logits=[-1.0, -1.0, 3.0, -2.0],
            unique_passages_used=2,
            duplicates_filtered=0,
            irrelevant_passages_filtered=0,
            has_cross_passage_disparity=True,
        )
        mock_engine.verify_claim.return_value = make_dummy_verdict(
            claim="Bridge is closed / open",
            verdict=CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED,
            confidence=0.72,
            evidence_breakdown=[
                {
                    "doc_id": "p1",
                    "text": "Bridge open",
                    "entailment_prob": 0.88,
                    "contradiction_prob": 0.02,
                    "neutral_prob": 0.10,
                    "is_joint": False,
                    "is_active_supporter": True,
                    "is_active_refuter": False,
                },
                {
                    "doc_id": "p2",
                    "text": "Bridge closed",
                    "entailment_prob": 0.02,
                    "contradiction_prob": 0.91,
                    "neutral_prob": 0.07,
                    "is_joint": False,
                    "is_active_supporter": False,
                    "is_active_refuter": True,
                },
            ],
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        adapter._aggregator = mock_aggregator
        res = adapter.verify(claim="Bridge is closed / open")

        self.assertTrue(res["has_cross_passage_disparity"])
        self.assertEqual(res["verdict"], "CONFLICTING_OR_CHERRYPICKED")
        self.assertIn("1 supporting passage(s) and 1 refuting passage(s)", res["reason"])

    def test_abstention_rationale_surfaces_selective_deferral(self):
        """
        Regression Test 6: Low-confidence prediction and abstention status are clearly surfaced.
        """
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = make_dummy_verdict(
            claim="Borderline ambiguous claim",
            verdict=CanonicalVeracity.VERIFIED_FALSE,
            confidence=0.29,
            was_abstained=True,
            abstention_reason="Confidence below 0.320 threshold",
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Borderline ambiguous claim")

        self.assertTrue(res["abstained"])
        self.assertIn("Abstained from definitive verdict", res["reason"])
        self.assertIn("Confidence below 0.320 threshold", res["reason"])

    def test_server_latency_present_and_positive(self):
        """
        Regression Test 8: Accurate request latency is measured and surfaced.
        """
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = make_dummy_verdict()

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Latency timing test claim")

        self.assertIn("latency_ms", res)
        self.assertIsInstance(res["latency_ms"], float)
        self.assertGreaterEqual(res["latency_ms"], 0.0)

    def test_verify_mode_local(self):
        """Verifies local mode correctly invokes local retriever and sets mode=local."""
        mock_engine = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_engine.retriever = mock_retriever
        mock_engine.verify_claim.return_value = make_dummy_verdict()

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Test claim for local mode", mode="local")

        self.assertEqual(res["mode"], "local")
        self.assertEqual(res["retrieval_status"]["mode"], "local")
        self.assertEqual(res["retrieval_status"]["provider"], "local_corpus")
        mock_retriever.search.assert_called_once()

    def test_verify_mode_web_unconfigured_policy(self):
        """Verifies web mode returns unconfigured status when TAVILY_API_KEY is not set."""
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = make_dummy_verdict(was_abstained=True)

        with patch.dict(os.environ, {}, clear=True):
            adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
            adapter.tavily_provider.api_key = ""
            res = adapter.verify(claim="Test claim for web mode", mode="web")

            self.assertEqual(res["mode"], "web")
            self.assertEqual(res["retrieval_status"]["status"], "unconfigured")
            self.assertIn("TAVILY_API_KEY", res["retrieval_status"]["error"])

    def test_verify_mode_web_with_mocked_results(self):
        """Verifies web mode correctly routes results from Tavily to verify_claim."""
        mock_engine = MagicMock()
        mock_engine.verify_claim.return_value = make_dummy_verdict(
            evidence_breakdown=[{
                "doc_id": "web_12345678",
                "text": "Live web citation passage.",
                "source_url": "https://example.com/live-news",
                "source_title": "Live News",
                "source_type": "web",
                "entailment_prob": 0.85,
                "contradiction_prob": 0.05,
                "neutral_prob": 0.10,
                "is_relevant": True,
                "is_claim_aligned": True,
                "is_active_supporter": True,
                "is_active_refuter": False,
            }]
        )

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        adapter.tavily_provider.search = MagicMock(return_value=([
            {
                "doc_id": "web_12345678",
                "text": "Live web citation passage.",
                "source_url": "https://example.com/live-news",
                "source_title": "Live News",
                "source_type": "web",
                "score": 0.95,
            }
        ], {
            "mode": "web",
            "provider": "tavily",
            "status": "success",
            "results_count": 1,
            "error": None,
        }))

        res = adapter.verify(claim="Test claim with web evidence", mode="web")
        self.assertEqual(res["mode"], "web")
        self.assertEqual(res["retrieval_status"]["status"], "success")
        self.assertEqual(len(res["evidence"]), 1)
        self.assertEqual(res["evidence"][0]["source_type"], "web")
        self.assertEqual(res["evidence"][0]["source_url"], "https://example.com/live-news")

    def test_verify_mode_hybrid(self):
        """Verifies hybrid mode combines local and web evidence and deduplicates."""
        mock_engine = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_engine.retriever = mock_retriever
        mock_engine.verify_claim.return_value = make_dummy_verdict()

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        adapter.tavily_provider.search = MagicMock(return_value=([], {
            "mode": "web",
            "provider": "tavily",
            "status": "no_results",
            "results_count": 0,
            "error": None,
        }))

        res = adapter.verify(claim="Test claim for hybrid mode", mode="hybrid")
        self.assertEqual(res["mode"], "hybrid")
        self.assertEqual(res["retrieval_status"]["mode"], "hybrid")
        self.assertEqual(res["retrieval_status"]["provider"], "hybrid_local_and_web")

    def test_verify_mode_hybrid_web_unavailable_falls_back_to_local_only(self):
        """Verifies hybrid mode sets is_local_only: True with clear notice when web is unconfigured."""
        mock_engine = MagicMock()
        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "doc_id": "loc_1",
            "text": "Local evidence text.",
            "source_dataset": "Local Corpus",
        }
        mock_retriever.search.return_value = [mock_doc]
        mock_engine.retriever = mock_retriever
        mock_engine.verify_claim.return_value = make_dummy_verdict()

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        adapter.tavily_provider.search = MagicMock(return_value=([], {
            "mode": "web",
            "provider": "tavily",
            "status": "unconfigured",
            "results_count": 0,
            "error": "TAVILY_API_KEY is not configured in D:\\VERITAS-X\\.env or environment.",
        }))

        res = adapter.verify(claim="Test claim for hybrid fallback", mode="hybrid")
        self.assertEqual(res["mode"], "hybrid")
        self.assertTrue(res["retrieval_status"]["is_local_only"])
        self.assertIsNotNone(res["retrieval_status"]["notice"])
        self.assertIn("Live Web search was unavailable", res["retrieval_status"]["notice"])
        self.assertEqual(res["retrieval_status"]["local_count"], 1)
        self.assertEqual(res["retrieval_status"]["web_count"], 0)

    def test_verify_mode_hybrid_web_limit_reached_falls_back_to_local_only(self):
        """Verifies hybrid mode sets is_local_only: True with clear notice when limit is reached."""
        mock_engine = MagicMock()
        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "doc_id": "loc_2",
            "text": "Local corpus verified passage.",
            "source_dataset": "Local Corpus",
        }
        mock_retriever.search.return_value = [mock_doc]
        mock_engine.retriever = mock_retriever
        mock_engine.verify_claim.return_value = make_dummy_verdict()

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        adapter.tavily_provider.search = MagicMock(return_value=([], {
            "mode": "web",
            "provider": "tavily",
            "status": "limit_reached",
            "results_count": 0,
            "error": "Monthly free-tier safety limit of 800 search credits reached.",
        }))

        res = adapter.verify(claim="Test claim for limit reached fallback", mode="hybrid")
        self.assertEqual(res["mode"], "hybrid")
        self.assertTrue(res["retrieval_status"]["is_local_only"])
        self.assertIsNotNone(res["retrieval_status"]["notice"])
        self.assertIn("Monthly free-tier safety limit", res["retrieval_status"]["notice"])

    def test_local_mode_displays_only_local_corpus_provenance(self):
        """Verify that local mode strictly returns source_type='local' even with archived URLs."""
        mock_engine = MagicMock()
        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "doc_id": "loc_archived_99",
            "text": "Local corpus article passage with archived web reference.",
            "source_dataset": "AVeriTeC",
            "source_url": "https://archived.independent.co.uk/news/123",
        }
        mock_retriever.search.return_value = [mock_doc]
        mock_engine.retriever = mock_retriever

        dummy_verdict = make_dummy_verdict(
            evidence_breakdown=[{
                "doc_id": "loc_archived_99",
                "text": "Local corpus article passage with archived web reference.",
                "source_dataset": "AVeriTeC",
                "source_url": "https://archived.independent.co.uk/news/123",
                "source_type": "local",
                "entailment_prob": 0.85,
                "contradiction_prob": 0.05,
                "neutral_prob": 0.10,
                "is_joint": False,
                "is_active_supporter": True,
            }]
        )
        mock_engine.verify_claim.return_value = dummy_verdict

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        res = adapter.verify(claim="Claim tested against local corpus", mode="local")

        self.assertEqual(res["mode"], "local")
        self.assertEqual(res["retrieval_status"]["mode"], "local")
        self.assertEqual(res["retrieval_status"]["provider"], "local_corpus")
        self.assertEqual(res["retrieval_status"]["local_count"], 1)
        self.assertEqual(res["retrieval_status"]["web_count"], 0)
        self.assertEqual(len(res["evidence"]), 1)
        # Provenance must be strictly local
        self.assertEqual(res["evidence"][0]["source_type"], "local")

    def test_hybrid_mode_identifies_local_and_web_evidence_individually(self):
        """Verify that hybrid mode maintains individual provenance: local vs web."""
        mock_engine = MagicMock()
        mock_retriever = MagicMock()
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "doc_id": "loc_doc_01",
            "text": "Local archive evidence.",
            "source_dataset": "Local Corpus",
        }
        mock_retriever.search.return_value = [mock_doc]
        mock_engine.retriever = mock_retriever

        dummy_verdict = make_dummy_verdict(
            evidence_breakdown=[
                {
                    "doc_id": "loc_doc_01",
                    "text": "Local archive evidence.",
                    "source_type": "local",
                    "source_title": "Local Corpus",
                    "entailment_prob": 0.90,
                    "contradiction_prob": 0.02,
                    "neutral_prob": 0.08,
                    "is_joint": False,
                    "is_active_supporter": True,
                },
                {
                    "doc_id": "web_a1b2c3d4",
                    "text": "Live web citation evidence.",
                    "source_type": "web",
                    "source_title": "Live News Daily",
                    "source_url": "https://livenews.com/story",
                    "entailment_prob": 0.88,
                    "contradiction_prob": 0.04,
                    "neutral_prob": 0.08,
                    "is_joint": False,
                    "is_active_supporter": True,
                },
            ]
        )
        mock_engine.verify_claim.return_value = dummy_verdict

        adapter = VeritasAPIAdapter(engine=mock_engine, lazy_init=True)
        adapter.tavily_provider.search = MagicMock(return_value=([
            {
                "doc_id": "web_a1b2c3d4",
                "text": "Live web citation evidence.",
                "source_url": "https://livenews.com/story",
                "source_title": "Live News Daily",
                "source_type": "web",
                "score": 0.92,
            }
        ], {
            "mode": "web",
            "provider": "tavily",
            "status": "success",
            "results_count": 1,
            "error": None,
        }))

        res = adapter.verify(claim="Claim tested in hybrid mode", mode="hybrid")

        self.assertEqual(res["mode"], "hybrid")
        self.assertEqual(res["retrieval_status"]["local_count"], 1)
        self.assertEqual(res["retrieval_status"]["web_count"], 1)
        self.assertEqual(len(res["evidence"]), 2)

        ev_local = [e for e in res["evidence"] if e["doc_id"] == "loc_doc_01"][0]
        ev_web = [e for e in res["evidence"] if e["doc_id"] == "web_a1b2c3d4"][0]

        self.assertEqual(ev_local["source_type"], "local")
        self.assertEqual(ev_web["source_type"], "web")
        self.assertEqual(ev_web["source_url"], "https://livenews.com/story")


if __name__ == "__main__":
    import unittest
    unittest.main()

