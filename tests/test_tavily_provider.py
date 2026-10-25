"""
Unit tests for Tavily Live Web Evidence Provider with Strict Free-Tier Protection.
All tests use deterministic mocks and isolated usage trackers to ensure fast execution,
zero external network dependency, no consumption of API quotas, and zero test interference.
"""
import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from veritas.retrieval.tavily_provider import (
    TavilySearchProvider,
    TavilyUsageTracker,
    load_dotenv,
    MONTHLY_SEARCH_CREDIT_LIMIT,
    deduplicate_passages,
    _sanitize_text,
)


class TestLoadDotenv(unittest.TestCase):
    """Tests for safe, pure-Python .env loading."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.env_file = os.path.join(self.test_dir.name, ".env")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_load_dotenv_parses_cleanly(self):
        content = (
            "# This is a comment\n"
            "\n"
            "TAVILY_API_KEY=\"tvly-mock-secret-key-999\"\n"
            "SINGLE_QUOTED='single_value'\n"
            "UNQUOTED=raw_value\n"
            "EMPTY_VAL=\n"
            "SPACED_KEY = spaced_value\n"
        )
        with open(self.env_file, "w", encoding="utf-8") as f:
            f.write(content)

        # Clear any preexisting values in environ
        for k in ["TAVILY_API_KEY", "SINGLE_QUOTED", "UNQUOTED", "SPACED_KEY"]:
            os.environ.pop(k, None)

        loaded = load_dotenv(self.env_file)
        self.assertTrue(loaded)
        self.assertEqual(os.environ.get("TAVILY_API_KEY"), "tvly-mock-secret-key-999")
        self.assertEqual(os.environ.get("SINGLE_QUOTED"), "single_value")
        self.assertEqual(os.environ.get("UNQUOTED"), "raw_value")
        self.assertEqual(os.environ.get("SPACED_KEY"), "spaced_value")

    def test_load_dotenv_does_not_overwrite_existing(self):
        os.environ["EXISTING_ENV_VAR"] = "original_value"
        with open(self.env_file, "w", encoding="utf-8") as f:
            f.write("EXISTING_ENV_VAR=should_not_override\n")

        load_dotenv(self.env_file)
        self.assertEqual(os.environ.get("EXISTING_ENV_VAR"), "original_value")

    def test_load_dotenv_handles_missing_file(self):
        missing_path = os.path.join(self.test_dir.name, "missing.env")
        loaded = load_dotenv(missing_path)
        self.assertFalse(loaded)


class TestTavilyUsageTracker(unittest.TestCase):
    """Tests for application-side search credit accounting and free-tier safety limits."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.usage_file = os.path.join(self.test_dir.name, "tavily_usage.json")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_initialization_defaults(self):
        tracker = TavilyUsageTracker(usage_file=self.usage_file, monthly_limit=800)
        status = tracker.get_status()

        self.assertEqual(status["credits_used"], 0)
        self.assertEqual(status["monthly_limit"], 800)
        self.assertEqual(status["remaining_credits"], 800)
        self.assertFalse(status["is_billing_disabled"])
        self.assertIsNone(status["billing_error_reason"])
        self.assertIsNone(status["last_request_time"])

    def test_reserve_credit_workflow(self):
        tracker = TavilyUsageTracker(usage_file=self.usage_file, monthly_limit=800)
        ok, used, err = tracker.reserve_credit()

        self.assertTrue(ok)
        self.assertEqual(used, 1)
        self.assertIsNone(err)

        status = tracker.get_status()
        self.assertEqual(status["credits_used"], 1)
        self.assertEqual(status["remaining_credits"], 799)
        self.assertIsNotNone(status["last_request_time"])

        # Second reservation
        ok2, used2, _ = tracker.reserve_credit()
        self.assertTrue(ok2)
        self.assertEqual(used2, 2)
        self.assertEqual(tracker.get_status()["remaining_credits"], 798)

    def test_monthly_ceiling_enforcement(self):
        tracker = TavilyUsageTracker(usage_file=self.usage_file, monthly_limit=3)

        for i in range(3):
            can_req, _ = tracker.can_request()
            self.assertTrue(can_req)
            ok, used, _ = tracker.reserve_credit()
            self.assertTrue(ok)
            self.assertEqual(used, i + 1)

        # 4th request must be blocked
        can_req, reason = tracker.can_request()
        self.assertFalse(can_req)
        self.assertIn("safety limit of 3 search credits reached", reason)

        ok, used, err = tracker.reserve_credit()
        self.assertFalse(ok)
        self.assertEqual(used, 3)
        self.assertIn("safety limit of 3 search credits reached", err)

    def test_monthly_rollover(self):
        # Pre-seed tracker file with usage from a previous month
        with open(self.usage_file, "w", encoding="utf-8") as f:
            json.dump({
                "current_month": "2026-08",
                "credits_used": 795,
                "monthly_limit": 800,
                "is_billing_disabled": False,
                "billing_error_reason": None,
                "history": {},
            }, f)

        tracker = TavilyUsageTracker(usage_file=self.usage_file, monthly_limit=800)
        status = tracker.get_status()

        # Should roll over to current month, resetting count
        self.assertEqual(status["credits_used"], 0)
        self.assertEqual(status["remaining_credits"], 800)

        # Verify previous month was archived in history
        with open(self.usage_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertIn("2026-08", saved["history"])
        self.assertEqual(saved["history"]["2026-08"]["credits_used"], 795)

    def test_record_billing_error(self):
        tracker = TavilyUsageTracker(usage_file=self.usage_file)
        tracker.record_billing_error("HTTP 402: Payment required")

        can_req, reason = tracker.can_request()
        self.assertFalse(can_req)
        self.assertIn("Tavily online retrieval is paused", reason)

        ok, used, err = tracker.reserve_credit()
        self.assertFalse(ok)
        self.assertIn("Tavily online retrieval is paused", err)

        status = tracker.get_status()
        self.assertTrue(status["is_billing_disabled"])
        self.assertEqual(status["billing_error_reason"], "HTTP 402: Payment required")


class TestTavilySearchProvider(unittest.TestCase):
    """Tests for Tavily Live Web Search Provider using isolated usage tracker and deterministic mocks."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.usage_file = os.path.join(self.test_dir.name, "tracker.json")
        self.tracker = TavilyUsageTracker(usage_file=self.usage_file, monthly_limit=800)

        self.mock_results = {
            "query": "The New York Post endorsed Donald Trump for re-election.",
            "results": [
                {
                    "title": "New York Post Editorial Endorsement",
                    "url": "https://nypost.com/2020/10/26/endorsement-donald-trump/",
                    "content": "The New York Post officially endorses President Donald J. Trump for re-election in 2020.",
                    "score": 0.985,
                    "published_date": "2020-10-26T12:00:00Z",
                },
                {
                    "title": "Quad City Times Article",
                    "url": "https://qctimes.com/news/local/article.html",
                    "content": "Although behind a pay wall, a headline in the Quad City Times dated 09/09/2020 discusses election updates.",
                    "score": 0.820,
                    "published_date": "2020-09-09T08:00:00Z",
                },
            ],
            "response_time": 0.32,
        }

    def tearDown(self):
        self.test_dir.cleanup()

    # 1. Successful Retrieval with Mocked Response
    @patch("urllib.request.urlopen")
    def test_successful_retrieval(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(self.mock_results).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        provider = TavilySearchProvider(api_key="tvly-mock-test-key-12345", usage_tracker=self.tracker)
        passages, status = provider.search("The New York Post endorsed Donald Trump for re-election.")

        self.assertEqual(status["status"], "success")
        self.assertIsNone(status["error"])
        self.assertEqual(status["results_count"], 2)
        self.assertEqual(len(passages), 2)
        self.assertEqual(status["credits_used"], 1)
        self.assertEqual(status["remaining_credits"], 799)

        # Provenance preservation
        first = passages[0]
        self.assertEqual(first["source_type"], "web")
        self.assertEqual(first["source_dataset"], "Live Web (Tavily)")
        self.assertEqual(first["source_title"], "New York Post Editorial Endorsement")
        self.assertEqual(first["source_url"], "https://nypost.com/2020/10/26/endorsement-donald-trump/")
        self.assertEqual(first["published_date"], "2020-10-26T12:00:00Z")
        self.assertIn("officially endorses President Donald J. Trump", first["text"])
        self.assertTrue(first["doc_id"].startswith("web_"))

    # 2. Strict Basic Search Payload Verification
    @patch("urllib.request.urlopen")
    def test_payload_strictly_basic_search(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps(self.mock_results).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        provider = TavilySearchProvider(api_key="tvly-mock-key", usage_tracker=self.tracker)
        provider.search("Verify payload", top_k=10)

        mock_urlopen.assert_called_once()
        sent_req = mock_urlopen.call_args[0][0]
        payload = json.loads(sent_req.data.decode("utf-8"))

        self.assertEqual(payload["search_depth"], "basic")  # Free 1-credit basic search only
        self.assertFalse(payload["include_answer"])
        self.assertFalse(payload["include_raw_content"])
        self.assertFalse(payload["include_images"])
        self.assertLessEqual(payload["max_results"], 5)  # Enforced ceiling

    # 3. Monthly Limit Reached Blocks Outgoing Requests
    @patch("urllib.request.urlopen")
    def test_monthly_limit_reached_blocks_requests(self, mock_urlopen):
        self.tracker.monthly_limit = 1
        self.tracker.reserve_credit()  # Consume the 1 available credit

        provider = TavilySearchProvider(api_key="tvly-mock-key", usage_tracker=self.tracker)
        passages, status = provider.search("Another claim")

        self.assertEqual(status["status"], "limit_reached")
        self.assertIn("safety limit", status["error"])
        self.assertEqual(len(passages), 0)
        mock_urlopen.assert_not_called()  # Must NOT make any network call

    # 4. Unconfigured API Key Policy
    @patch.dict(os.environ, {}, clear=True)
    def test_unconfigured_api_key(self):
        provider = TavilySearchProvider(api_key=None, usage_tracker=self.tracker)
        self.assertFalse(provider.is_configured)

        passages, status = provider.search("Any claim text")
        self.assertEqual(status["status"], "unconfigured")
        self.assertIn("TAVILY_API_KEY", status["error"])
        self.assertEqual(len(passages), 0)

    # 5. Empty Query Handling
    def test_empty_query(self):
        provider = TavilySearchProvider(api_key="tvly-mock-key", usage_tracker=self.tracker)
        passages, status = provider.search("   ")
        self.assertEqual(status["status"], "empty_query")
        self.assertEqual(len(passages), 0)

    # 6. Network Timeout Handling (Zero Retries)
    @patch("urllib.request.urlopen")
    def test_network_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection timed out")

        provider = TavilySearchProvider(api_key="tvly-mock-key", usage_tracker=self.tracker)
        passages, status = provider.search("Test claim")

        self.assertEqual(status["status"], "timeout")
        self.assertIn("timed out", status["error"].lower())
        self.assertEqual(len(passages), 0)
        self.assertEqual(mock_urlopen.call_count, 1)  # Strict zero-retry policy

    # 7. Network Connection Failure Handling (Zero Retries)
    @patch("urllib.request.urlopen")
    def test_network_failure(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Name or service not known")

        provider = TavilySearchProvider(api_key="tvly-mock-key", usage_tracker=self.tracker)
        passages, status = provider.search("Test claim")

        self.assertEqual(status["status"], "network_error")
        self.assertIn("Network connection failed", status["error"])
        self.assertEqual(len(passages), 0)
        self.assertEqual(mock_urlopen.call_count, 1)  # Strict zero-retry policy

    # 8. HTTP 402 Payment Required Handling
    @patch("urllib.request.urlopen")
    def test_http_402_payment_required_disables_retrieval(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.tavily.com/search",
            code=402,
            msg="Payment Required",
            hdrs={},
            fp=None,
        )
        provider = TavilySearchProvider(api_key="tvly-mock-key", usage_tracker=self.tracker)
        passages, status = provider.search("Test claim")

        self.assertEqual(status["status"], "billing_restricted")
        self.assertIn("HTTP 402", status["error"])
        self.assertTrue(self.tracker.get_status()["is_billing_disabled"])
        self.assertEqual(mock_urlopen.call_count, 1)

    # 9. HTTP 429 Quota Exceeded Handling
    @patch("urllib.request.urlopen")
    def test_http_429_quota_exceeded_disables_retrieval(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.tavily.com/search",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=None,
        )
        provider = TavilySearchProvider(api_key="tvly-mock-key", usage_tracker=self.tracker)
        passages, status = provider.search("Test claim")

        self.assertEqual(status["status"], "quota_exceeded")
        self.assertIn("quota or rate limit reached", status["error"].lower())
        self.assertTrue(self.tracker.get_status()["is_billing_disabled"])
        self.assertEqual(mock_urlopen.call_count, 1)

    # 10. HTTP 401 Unauthorized Handling
    @patch("urllib.request.urlopen")
    def test_http_401_unauthorized(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.tavily.com/search",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )
        provider = TavilySearchProvider(api_key="tvly-mock-key", usage_tracker=self.tracker)
        passages, status = provider.search("Test claim")

        self.assertEqual(status["status"], "unauthorized")
        self.assertIn("unauthorized or invalid", status["error"].lower())

    # 11. Malformed JSON Response
    @patch("urllib.request.urlopen")
    def test_malformed_json_response(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b"<html>Bad Gateway</html>"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        provider = TavilySearchProvider(api_key="tvly-mock-key", usage_tracker=self.tracker)
        passages, status = provider.search("Test claim")

        self.assertEqual(status["status"], "malformed_response")
        self.assertEqual(len(passages), 0)

    # 12. Empty Search Results
    @patch("urllib.request.urlopen")
    def test_empty_results_handling(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"query": "obscure query", "results": []}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        provider = TavilySearchProvider(api_key="tvly-mock-key", usage_tracker=self.tracker)
        passages, status = provider.search("obscure query")

        self.assertEqual(status["status"], "no_results")
        self.assertIsNone(status["error"])
        self.assertEqual(len(passages), 0)

    # 13. Deduplication Logic
    def test_deduplicate_passages(self):
        raw_passages = [
            {"doc_id": "1", "source_url": "https://example.com/page1", "text": "Unique passage 1."},
            {"doc_id": "2", "source_url": "https://example.com/page1?ref=twitter", "text": "Duplicate URL passage."},
            {"doc_id": "3", "source_url": "https://example.org/page2", "text": "Unique passage 1."},  # Duplicate text
            {"doc_id": "4", "source_url": "https://example.org/page3", "text": "Unique passage 3."},
        ]
        deduped = deduplicate_passages(raw_passages)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["doc_id"], "1")
        self.assertEqual(deduped[1]["doc_id"], "4")


if __name__ == "__main__":
    unittest.main()
