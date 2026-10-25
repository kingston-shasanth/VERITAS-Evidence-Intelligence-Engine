"""
Tavily Live Web Evidence Provider for VERITAS-X with Strict Free-Only Protection.
Official Tavily Search API client for retrieving real-time web citations,
normalizing them into verified evidence schema, deduplicating passages,
and strictly enforcing a persistent monthly limit of 800 search credits to guarantee
zero accidental paid charges or account upgrades.

Security: TAVILY_API_KEY is read strictly from backend environment variables or D:\\VERITAS-X\\.env.
Never logs, serializes, or exposes the API key to frontend clients or console logs.
"""
import os
import re
import json
import time
import hashlib
import logging
import threading
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("veritas.tavily_provider")

TAVILY_API_URL = "https://api.tavily.com/search"
DEFAULT_TIMEOUT_SEC = 8.0
DEFAULT_MAX_RESULTS = 5
MONTHLY_SEARCH_CREDIT_LIMIT = 800  # Strict free-tier protection ceiling (free quota: 1000/mo)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_USAGE_FILE = os.path.join(REPO_ROOT, "data", "tavily_usage.json")


def load_dotenv(dotenv_path: Optional[str] = None) -> bool:
    """
    Safely loads environment variables from a .env file without external dependencies.
    Never prints, logs, or returns sensitive values.
    """
    if dotenv_path is None:
        candidates = [
            os.path.join("D:\\VERITAS-X", ".env"),
            os.path.join(REPO_ROOT, ".env"),
            os.path.abspath(".env"),
        ]
        for c in candidates:
            if os.path.exists(c):
                dotenv_path = c
                break

    if not dotenv_path or not os.path.exists(dotenv_path):
        return False

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("\"'").strip()
                    if key and (key not in os.environ or not os.environ[key].strip()):
                        os.environ[key] = val
        return True
    except Exception as e:
        logger.debug("Failed reading .env file: %s", e)
        return False


# Attempt module-level environment loading once upon initial import
load_dotenv()


def _sanitize_text(text: str) -> str:
    """Removes excessive whitespace and unprintable characters while preserving paragraph breaks."""
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def deduplicate_passages(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates a list of candidate passages based on canonical URL and text hash.
    Preserves original order and highest confidence/relevance metadata.
    """
    seen_urls = set()
    seen_hashes = set()
    unique_passages = []

    for p in passages:
        url = (p.get("source_url") or "").strip().lower()
        url_clean = re.sub(r"[?#].*$", "", url).rstrip("/")

        text = _sanitize_text(p.get("text", ""))
        if not text:
            continue

        text_hash = hashlib.sha256(text.lower().encode("utf-8")).hexdigest()[:16]

        if url_clean and url_clean in seen_urls:
            continue
        if text_hash in seen_hashes:
            continue

        if url_clean:
            seen_urls.add(url_clean)
        seen_hashes.add(text_hash)
        unique_passages.append(p)

    return unique_passages


class TavilyUsageTracker:
    """
    Persistent application-side search credit tracker.
    Enforces a strict 800 monthly limit (200 credit buffer below the 1,000 free quota)
    and permanently disables online search upon quota, payment, or billing errors.
    Accounts for every outgoing request before dispatching it.
    Resets usage ONLY at the start of a new calendar month.
    """

    def __init__(
        self,
        usage_file: str = DEFAULT_USAGE_FILE,
        monthly_limit: int = MONTHLY_SEARCH_CREDIT_LIMIT,
    ):
        self.usage_file = os.path.abspath(usage_file)
        self.monthly_limit = int(monthly_limit)
        self._lock = threading.Lock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
        except Exception:
            pass

    def _get_current_month(self) -> str:
        return time.strftime("%Y-%m")

    def _load_data_unlocked(self) -> Dict[str, Any]:
        cur_month = self._get_current_month()
        data: Dict[str, Any] = {}

        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning("Could not read usage file %s (%s). Re-initializing safely.", self.usage_file, e)

        saved_month = data.get("current_month")
        if saved_month != cur_month:
            # Rollover to new calendar month: archive old usage and reset counter
            history = data.get("history", {})
            if saved_month:
                history[saved_month] = {
                    "credits_used": data.get("credits_used", 0),
                    "monthly_limit": data.get("monthly_limit", self.monthly_limit),
                }
            data = {
                "current_month": cur_month,
                "credits_used": 0,
                "monthly_limit": self.monthly_limit,
                "is_billing_disabled": False,
                "billing_error_reason": None,
                "last_request_time": None,
                "history": history,
            }
            try:
                with open(self.usage_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass
        else:
            # Maintain current month fields
            data.setdefault("credits_used", 0)
            data.setdefault("monthly_limit", self.monthly_limit)
            data.setdefault("is_billing_disabled", False)
            data.setdefault("billing_error_reason", None)
            data.setdefault("last_request_time", None)

        return data

    def can_request(self) -> Tuple[bool, Optional[str]]:
        """
        Determines whether a new basic search request is permitted.
        Returns (is_allowed: bool, reason: Optional[str]).
        """
        with self._lock:
            data = self._load_data_unlocked()
            if data.get("is_billing_disabled"):
                reason = data.get("billing_error_reason") or "Online search disabled due to provider quota or billing constraint."
                return False, f"Tavily online retrieval is paused: {reason}"

            used = int(data.get("credits_used", 0))
            limit = int(data.get("monthly_limit", self.monthly_limit))
            if used >= limit:
                return False, (
                    f"Monthly free-tier safety limit of {limit} search credits reached "
                    f"({used}/{limit} used in {data.get('current_month')}). "
                    f"Online search is stopped to protect your account. Resets next calendar month."
                )
            return True, None

    def reserve_credit(self) -> Tuple[bool, int, Optional[str]]:
        """
        Accounts for one outgoing Basic Search request BEFORE sending it over the network.
        Returns (success: bool, credits_used: int, error_message: Optional[str]).
        """
        with self._lock:
            data = self._load_data_unlocked()
            if data.get("is_billing_disabled"):
                reason = data.get("billing_error_reason") or "Online retrieval disabled due to billing constraint."
                return False, int(data.get("credits_used", 0)), f"Tavily online retrieval is paused: {reason}"

            used = int(data.get("credits_used", 0))
            limit = int(data.get("monthly_limit", self.monthly_limit))
            if used >= limit:
                return False, used, (
                    f"Monthly free-tier safety limit of {limit} search credits reached "
                    f"({used}/{limit} used in {data.get('current_month')}). "
                    f"Online search is stopped to protect your account. Resets next calendar month."
                )

            # Increment count prior to request
            data["credits_used"] = used + 1
            data["last_request_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                with open(self.usage_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error("Failed saving Tavily credit usage to %s: %s", self.usage_file, e)

            return True, data["credits_used"], None

    def rollback_credit(self) -> int:
        """
        Reverts a reserved credit if the request failed at the local network/DNS layer
        before reaching the provider server.
        """
        with self._lock:
            data = self._load_data_unlocked()
            used = max(0, int(data.get("credits_used", 0)) - 1)
            data["credits_used"] = used
            try:
                with open(self.usage_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error("Failed saving rolled-back Tavily credit usage to %s: %s", self.usage_file, e)
            return used

    def record_billing_error(self, reason: str) -> None:
        """
        Disables online search if provider returns quota, payment, or billing errors.
        Prevents retrying or incurring accidental paid charges.
        """
        with self._lock:
            data = self._load_data_unlocked()
            data["is_billing_disabled"] = True
            data["billing_error_reason"] = str(reason)
            try:
                with open(self.usage_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        """Returns safe, telemetry-only credit usage metrics (no API keys)."""
        with self._lock:
            data = self._load_data_unlocked()
            used = int(data.get("credits_used", 0))
            limit = int(data.get("monthly_limit", self.monthly_limit))
            return {
                "current_month": data.get("current_month"),
                "credits_used": used,
                "monthly_limit": limit,
                "remaining_credits": max(0, limit - used),
                "is_billing_disabled": bool(data.get("is_billing_disabled")),
                "billing_error_reason": data.get("billing_error_reason"),
                "last_request_time": data.get("last_request_time"),
            }


# Default shared usage tracker instance
_GLOBAL_TRACKER: Optional[TavilyUsageTracker] = None
_GLOBAL_TRACKER_LOCK = threading.Lock()


def get_default_usage_tracker() -> TavilyUsageTracker:
    global _GLOBAL_TRACKER
    with _GLOBAL_TRACKER_LOCK:
        if _GLOBAL_TRACKER is None:
            _GLOBAL_TRACKER = TavilyUsageTracker()
        return _GLOBAL_TRACKER


class TavilySearchProvider:
    """
    Live web retrieval provider using the official Tavily Search API.
    Strictly restricted to Basic Search (1 credit/request) with a persistent 800-credit
    monthly cap, zero retries, no automatic upgrades, and zero key exposure.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        max_results: int = DEFAULT_MAX_RESULTS,
        usage_tracker: Optional[TavilyUsageTracker] = None,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("TAVILY_API_KEY", "").strip()
        self.timeout = timeout
        self.max_results = max_results
        self.tracker = usage_tracker or get_default_usage_tracker()

    def _get_active_key(self) -> str:
        key = self.api_key if self.api_key else os.getenv("TAVILY_API_KEY", "").strip()
        if key and key not in ("replace_with_your_key", "your_new_key_here") and not key.startswith("your_"):
            return key
        return ""

    @property
    def is_configured(self) -> bool:
        """Returns True if a valid, non-placeholder API key is available in .env or environment."""
        return bool(self._get_active_key())

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes a single live Basic Search via Tavily under free-only limits:
          1. Validates query.
          2. Verifies backend credential presence.
          3. Checks monthly credit ceiling (800 credits).
          4. Accounts for credit BEFORE making the network request.
          5. Sends exactly one Basic Search request (no retries, no advanced crawling).
          6. Handles billing/quota errors without retrying.
          7. Returns candidate passages and detailed retrieval telemetry.
        """
        clean_query = (query or "").strip()
        usage_info = self.tracker.get_status()

        if not clean_query:
            return [], {
                "provider": "tavily",
                "mode": "web",
                "status": "empty_query",
                "error": "Query string is empty.",
                "results_count": 0,
                "latency_ms": 0.0,
                "credits_used": usage_info["credits_used"],
                "monthly_limit": usage_info["monthly_limit"],
                "remaining_credits": usage_info["remaining_credits"],
            }

        active_key = self._get_active_key()
        if not active_key:
            return [], {
                "provider": "tavily",
                "mode": "web",
                "status": "unconfigured",
                "error": "TAVILY_API_KEY is not configured in D:\\VERITAS-X\\.env or environment. Configure your key to enable Live Web verification.",
                "results_count": 0,
                "latency_ms": 0.0,
                "credits_used": usage_info["credits_used"],
                "monthly_limit": usage_info["monthly_limit"],
                "remaining_credits": usage_info["remaining_credits"],
            }

        # Check application-side monthly safety limit & billing disable status
        can_proceed, block_reason = self.tracker.can_request()
        if not can_proceed:
            return [], {
                "provider": "tavily",
                "mode": "web",
                "status": "limit_reached" if "safety limit" in (block_reason or "") else "billing_restricted",
                "error": block_reason,
                "results_count": 0,
                "latency_ms": 0.0,
                "credits_used": usage_info["credits_used"],
                "monthly_limit": usage_info["monthly_limit"],
                "remaining_credits": usage_info["remaining_credits"],
            }

        # Reserve 1 search credit BEFORE dispatching request
        reserved, used_credits, reserve_err = self.tracker.reserve_credit()
        if not reserved:
            return [], {
                "provider": "tavily",
                "mode": "web",
                "status": "limit_reached",
                "error": reserve_err or "Failed reserving search credit limit.",
                "results_count": 0,
                "latency_ms": 0.0,
                "credits_used": used_credits,
                "monthly_limit": usage_info["monthly_limit"],
                "remaining_credits": max(0, usage_info["monthly_limit"] - used_credits),
            }

        # Parameters strictly restricted to Basic Search (1 credit cost only)
        k = min(top_k or self.max_results, 5)
        payload = {
            "api_key": active_key,
            "query": clean_query,
            "search_depth": "basic",  # STRICTLY Basic Search (1 credit only)
            "max_results": k,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }

        t0 = time.perf_counter()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "VERITAS-X/2.0-WebRetriever-Basic",
        }

        # Execute single request — NO automatic retries
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(TAVILY_API_URL, data=req_data, headers=headers, method="POST")

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                status_code = response.status
                if status_code != 200:
                    return [], {
                        "provider": "tavily",
                        "mode": "web",
                        "status": "http_error",
                        "error": f"Tavily returned unexpected status code {status_code}.",
                        "results_count": 0,
                        "latency_ms": latency_ms,
                        "credits_used": used_credits,
                        "monthly_limit": usage_info["monthly_limit"],
                        "remaining_credits": max(0, usage_info["monthly_limit"] - used_credits),
                    }

                raw_bytes = response.read()
                data = json.loads(raw_bytes.decode("utf-8"))

        except urllib.error.HTTPError as he:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            code = he.code

            # Check for quota or payment/billing restrictions
            if code == 402:
                err_msg = "Provider returned HTTP 402 Payment Required. Online retrieval disabled to protect free tier."
                self.tracker.record_billing_error(err_msg)
                status_str = "billing_restricted"
            elif code == 429:
                err_msg = "Provider monthly quota or rate limit reached (HTTP 429). Online retrieval paused without retries."
                self.tracker.record_billing_error(err_msg)
                status_str = "quota_exceeded"
            elif code in (401, 403):
                err_msg = "Tavily API key is unauthorized or invalid."
                status_str = "unauthorized"
            else:
                err_msg = f"Tavily HTTP error {code}: {he.reason}"
                status_str = "http_error"

            logger.warning("Tavily search HTTP response [%s]: %s", code, err_msg)
            return [], {
                "provider": "tavily",
                "mode": "web",
                "status": status_str,
                "error": err_msg,
                "results_count": 0,
                "latency_ms": latency_ms,
                "credits_used": used_credits,
                "monthly_limit": usage_info["monthly_limit"],
                "remaining_credits": max(0, usage_info["monthly_limit"] - used_credits),
            }

        except urllib.error.URLError as ue:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            reason = str(ue.reason)
            is_timeout = "timed out" in reason.lower()
            if "getaddrinfo" in reason.lower() or "name or service not known" in reason.lower():
                err_msg = f"Network connection failed: DNS resolution failed for api.tavily.com ({reason}). Check local internet connectivity and DNS configuration."
            elif is_timeout:
                err_msg = "Tavily request timed out."
            else:
                err_msg = f"Network connection failed: {reason}"
            logger.warning("Tavily search network failure: %s", err_msg)
            # Revert reserved credit since provider was never contacted
            reverted_credits = self.tracker.rollback_credit()
            return [], {
                "provider": "tavily",
                "mode": "web",
                "status": "timeout" if is_timeout else "network_error",
                "error": err_msg,
                "results_count": 0,
                "latency_ms": latency_ms,
                "credits_used": reverted_credits,
                "monthly_limit": usage_info["monthly_limit"],
                "remaining_credits": max(0, usage_info["monthly_limit"] - reverted_credits),
            }

        except json.JSONDecodeError as je:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            logger.warning("Tavily response malformed JSON: %s", je)
            return [], {
                "provider": "tavily",
                "mode": "web",
                "status": "malformed_response",
                "error": "Failed to parse JSON response from search provider.",
                "results_count": 0,
                "latency_ms": latency_ms,
                "credits_used": used_credits,
                "monthly_limit": usage_info["monthly_limit"],
                "remaining_credits": max(0, usage_info["monthly_limit"] - used_credits),
            }

        except Exception as ex:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            logger.error("Unexpected error during Tavily retrieval: %s", ex)
            return [], {
                "provider": "tavily",
                "mode": "web",
                "status": "error",
                "error": "Internal retrieval error occurred during online search.",
                "results_count": 0,
                "latency_ms": latency_ms,
                "credits_used": used_credits,
                "monthly_limit": usage_info["monthly_limit"],
                "remaining_credits": max(0, usage_info["monthly_limit"] - used_credits),
            }

        # Parse and normalize Tavily results
        results = data.get("results")
        if not isinstance(results, list) or len(results) == 0:
            return [], {
                "provider": "tavily",
                "mode": "web",
                "status": "no_results",
                "error": None,
                "results_count": 0,
                "latency_ms": latency_ms,
                "query": clean_query,
                "credits_used": used_credits,
                "monthly_limit": usage_info["monthly_limit"],
                "remaining_credits": max(0, usage_info["monthly_limit"] - used_credits),
            }

        passages: List[Dict[str, Any]] = []
        for i, item in enumerate(results):
            if not isinstance(item, dict):
                continue
            text = _sanitize_text(item.get("content", ""))
            if not text:
                continue

            url = (item.get("url") or "").strip()
            title = _sanitize_text(item.get("title", "")) or "Web Source"
            score = float(item.get("score") or round(1.0 / (i + 1), 4))
            pub_date = item.get("published_date") or None

            id_seed = url if url else text[:64]
            doc_id = f"web_{hashlib.sha256(id_seed.encode('utf-8')).hexdigest()[:8]}"

            passages.append({
                "doc_id": doc_id,
                "text": text,
                "source_url": url or "Live Web Citation",
                "source_title": title,
                "source_dataset": "Live Web (Tavily)",
                "source_type": "web",
                "published_date": pub_date,
                "score": score,
                "metadata": {
                    "provider": "tavily",
                    "raw_score": score,
                    "published_date": pub_date,
                    "rank": i + 1,
                    "search_depth": "basic",
                },
            })

        deduped = deduplicate_passages(passages)

        return deduped, {
            "provider": "tavily",
            "mode": "web",
            "status": "success",
            "error": None,
            "results_count": len(deduped),
            "raw_count": len(results),
            "latency_ms": latency_ms,
            "query": clean_query,
            "credits_used": used_credits,
            "monthly_limit": usage_info["monthly_limit"],
            "remaining_credits": max(0, usage_info["monthly_limit"] - used_credits),
        }
