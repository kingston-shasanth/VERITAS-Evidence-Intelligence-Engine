"""
VERITAS-X Verification HTTP Server.
Serves POST /api/verify and GET /api/health over standard Python http.server
with strict CORS enforcement, input validation, and clean error sanitization.
"""
import os
import sys
import json
import logging
import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional, Set

# Ensure repo root and src/ are on sys.path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from veritas.retrieval.tavily_provider import load_dotenv
load_dotenv()

from api.adapter import VeritasAPIAdapter
from veritas.verification.nli_model import resolve_device

logger = logging.getLogger("veritas_api")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

ALLOWED_ORIGINS: Set[str] = {
    "http://127.0.0.1:5173",
    "http://localhost:5173",
}


class VeritasRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler serving the VERITAS-X verification endpoint.
    Strictly isolates internal exception tracebacks from API clients.
    """

    adapter: VeritasAPIAdapter = None  # Injected on server startup

    def get_cors_origin(self) -> Optional[str]:
        origin = self.headers.get("Origin")
        if origin and origin in ALLOWED_ORIGINS:
            return origin
        # If no Origin header was sent (e.g. curl / direct test), allow development origin
        if not origin:
            return "http://127.0.0.1:5173"
        return None

    def _set_cors_headers(self, origin: Optional[str]) -> None:
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.send_header("Vary", "Origin")

    def send_json(self, status_code: int, data: dict, origin: Optional[str] = None) -> None:
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._set_cors_headers(origin)
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        """Handles CORS preflight requests."""
        origin = self.get_cors_origin()
        if origin:
            self.send_response(204)
            self._set_cors_headers(origin)
            self.send_header("Access-Control-Max-Age", "86400")
            self.end_headers()
        else:
            self.send_response(403)
            self.end_headers()

    def do_GET(self) -> None:
        """Handles health check and route discovery."""
        origin = self.get_cors_origin()
        if self.path in ("/api/health", "/api/health/"):
            device_str = "cpu"
            try:
                device_override = getattr(self.adapter, "device_override", None) if self.adapter else None
                device_str = resolve_device(device_override)
            except Exception:
                device_str = "unknown"

            is_ready = bool(self.adapter and getattr(self.adapter, "is_model_ready", False))
            is_loading = bool(self.adapter and getattr(self.adapter, "is_model_loading", False))
            init_latency = getattr(self.adapter, "last_init_latency_ms", None) if self.adapter else None

            self.send_json(200, {
                "status": "ok",
                "service": "VERITAS-X Evidence Verification API",
                "version": "Phase 3.2.5",
                "device": device_str,
                "model_ready": is_ready,
                "model_loading": is_loading,
                "warmup_state": "ready" if is_ready else ("initializing" if is_loading else "pending"),
                "init_latency_ms": init_latency,
            }, origin)
        else:
            self.send_json(404, {"error": f"Endpoint '{self.path}' not found."}, origin)

    def do_POST(self) -> None:
        """Handles POST /api/verify with strict payload validation."""
        origin = self.get_cors_origin()

        if self.path not in ("/api/verify", "/api/verify/"):
            self.send_json(404, {"error": f"Endpoint '{self.path}' not found."}, origin)
            return

        # Read Content-Length
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            self.send_json(400, {"error": "Invalid Content-Length header."}, origin)
            return

        if content_length <= 0:
            self.send_json(400, {"error": "Request body cannot be empty."}, origin)
            return

        # Read and parse body
        try:
            raw_body = self.rfile.read(content_length).decode("utf-8")
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            self.send_json(400, {"error": "Invalid JSON format in request body."}, origin)
            return
        except UnicodeDecodeError:
            self.send_json(400, {"error": "Request body must be valid UTF-8 text."}, origin)
            return

        if not isinstance(body, dict):
            self.send_json(400, {"error": "Request JSON must be an object."}, origin)
            return

        # Validate claim field
        claim = body.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            self.send_json(400, {
                "error": "Missing or invalid 'claim' field. A non-empty string is required."
            }, origin)
            return

        # Validate optional evidence list
        evidence = body.get("evidence")
        if evidence is None:
            evidence = body.get("user_evidence", [])
        if evidence is not None and not isinstance(evidence, list):
            self.send_json(400, {
                "error": "Optional 'evidence' field must be a list of strings if provided."
            }, origin)
            return

        claim_id = body.get("claim_id")
        if claim_id is not None and not isinstance(claim_id, str):
            self.send_json(400, {
                "error": "Optional 'claim_id' field must be a string."
            }, origin)
            return

        mode = body.get("mode") or body.get("evidence_mode") or "local"
        if not isinstance(mode, str):
            mode = "local"
        mode = mode.lower().strip()

        # Execute verification via thin adapter
        try:
            include_research = bool(body.get("include_research", True))
            include_processing_trace = bool(body.get("include_processing_trace", True))
            result = self.adapter.verify(
                claim=claim.strip(),
                user_evidence=evidence if evidence else None,
                claim_id=claim_id.strip() if claim_id else None,
                mode=mode,
                include_timings=True,
                include_research=include_research,
                include_processing_trace=include_processing_trace,
            )
            self.send_json(200, result, origin)
        except ValueError as ve:
            logger.warning("Validation error during verification: %s", ve)
            self.send_json(400, {"error": str(ve)}, origin)
        except Exception as exc:
            # Do not expose internal stack traces to client browsers
            logger.error("Unhandled verification exception: %s", exc, exc_info=True)
            self.send_json(500, {
                "error": "Internal verification error occurred. Please try again."
            }, origin)

    def log_message(self, format, *args):
        """Clean custom request logging."""
        logger.info("%s - - [%s] %s" % (self.client_address[0], self.log_date_time_string(), format % args))


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    adapter: Optional[VeritasAPIAdapter] = None,
    lazy_init: bool = False,
    background_warmup: bool = False,
) -> ThreadingHTTPServer:
    if adapter is None:
        logger.info("Initializing VeritasAPIAdapter with frozen Phase 3.2.5 configuration...")
        # If background warmup is active, do not block socket binding on synchronous init
        adapter = VeritasAPIAdapter(lazy_init=(lazy_init or background_warmup))

    class DedicatedRequestHandler(VeritasRequestHandler):
        pass

    DedicatedRequestHandler.adapter = adapter
    server = ThreadingHTTPServer((host, port), DedicatedRequestHandler)
    logger.info("Server configured on http://%s:%d (CORS allowed: %s)", host, port, ALLOWED_ORIGINS)

    if background_warmup and not adapter.is_model_ready:
        logger.info("Triggering non-blocking background model warm-up...")
        adapter.warmup(background=True)

    return server


def run_server(
    host: str = "127.0.0.1",
    port: int = 8011,
    lazy_init: bool = False,
    background_warmup: bool = False,
) -> None:
    """Starts the verification HTTP server."""
    server = create_server(host=host, port=port, lazy_init=lazy_init, background_warmup=background_warmup)
    print(f"\n=======================================================")
    print(f"  VERITAS-X Live Verification API")
    print(f"  Endpoint: http://{host}:{port}/api/verify")
    print(f"  Health:   http://{host}:{port}/api/health")
    print(f"  Frozen:   Phase 3.2.5 Configuration Active")
    print(f"  Warmup:   {'Deferred (Lazy)' if lazy_init else ('Background' if background_warmup else 'Synchronous')}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user interrupt.")
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start VERITAS-X Verification API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8011, help="Port (default: 8011)")
    parser.add_argument("--lazy", action="store_true", help="Lazy load model on first verify request")
    parser.add_argument("--warmup", action="store_true", help="Synchronously warm up models before accepting requests")
    parser.add_argument("--background-warmup", action="store_true", help="Bind port immediately and warm models in background")
    args = parser.parse_args()

    # Determine execution warmup mode
    lazy = args.lazy
    bg_warmup = args.background_warmup or (not args.lazy and not args.warmup)

    run_server(host=args.host, port=args.port, lazy_init=lazy, background_warmup=bg_warmup)
