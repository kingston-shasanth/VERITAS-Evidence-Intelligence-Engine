"""
Live HTTP Smoke & Integration Test Suite for VERITAS-X.
Validates end-to-end HTTP request/response lifecycles, CORS negotiation,
schema adherence, input sanitation, and error sanitization.
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error
from unittest import TestCase
from unittest.mock import MagicMock

# Ensure repo root and src/ are on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from veritas.data.schema import CanonicalVeracity
from api.adapter import VeritasAPIAdapter
from api.server import create_server
from tests.test_api_adapter import make_dummy_verdict


class TestLiveSmokeIntegration(TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure mock engine to avoid heavy GPU download during CI smoke tests
        cls.mock_engine = MagicMock()
        cls.mock_engine.device = "cpu"
        cls.mock_engine.verify_claim.return_value = make_dummy_verdict(
            claim="Smoke test claim",
            verdict=CanonicalVeracity.VERIFIED_TRUE,
            confidence=0.88,
        )

        cls.adapter = VeritasAPIAdapter(engine=cls.mock_engine, lazy_init=True)
        # Port 0 assigns an available ephemeral port
        cls.server = create_server(host="127.0.0.1", port=0, adapter=cls.adapter)
        cls.host, cls.port = cls.server.server_address
        cls.base_url = f"http://{cls.host}:{cls.port}"

        import threading
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_health_check_endpoint(self):
        req = urllib.request.Request(f"{self.base_url}/api/health")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "ok")
            self.assertEqual(data.get("version"), "Phase 3.2.5")
            self.assertIn("device", data)

    def test_verify_smoke_success(self):
        payload = json.dumps({
            "claim": "Water boils at 100 degrees Celsius at sea level.",
            "evidence": ["At 1 atm sea level pressure, the boiling point of pure water is 100 C."],
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/verify",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Origin": "http://127.0.0.1:5173",
            },
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            # Verify CORS header
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "http://127.0.0.1:5173")
            data = json.loads(resp.read().decode("utf-8"))

            required_fields = [
                "claim_id", "claim", "verdict", "confidence",
                "probabilities", "has_cross_passage_disparity",
                "abstained", "reason", "evidence", "device",
            ]
            for f in required_fields:
                self.assertIn(f, data, f"Missing required response field: {f}")

            # Verify 4-class probabilities
            probs = data["probabilities"]
            for label in [
                "VERIFIED_TRUE", "VERIFIED_FALSE",
                "CONFLICTING_OR_CHERRYPICKED", "INSUFFICIENT_EVIDENCE",
            ]:
                self.assertIn(label, probs)
            total_prob = sum(probs.values())
            self.assertAlmostEqual(total_prob, 1.0, places=2)

    def test_verify_smoke_rejects_empty_claim(self):
        payload = json.dumps({"claim": "   "}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/verify",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_cors_preflight_negotiation(self):
        req = urllib.request.Request(
            f"{self.base_url}/api/verify",
            headers={"Origin": "http://127.0.0.1:5173"},
            method="OPTIONS",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 204)
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "http://127.0.0.1:5173")
            self.assertIn("POST", resp.headers.get("Access-Control-Allow-Methods", ""))
