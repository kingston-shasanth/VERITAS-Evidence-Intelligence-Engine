"""
VERITAS-X Thin Live API Adapter.
Bridges frontend requests to the frozen Phase 3.2.5 verification engine
without modifying core verification components or held-out evaluation artifacts.
"""
import os
import sys
import time
import hashlib
import logging
import threading
from typing import List, Dict, Any, Optional

logger = logging.getLogger("veritas.api_adapter")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from veritas.data.schema import CanonicalVeracity
from veritas.verification.engine import (
    VerificationEngine,
    ForensicVerdict,
    _match_token_set,
    _TOPICAL_STOP_WORDS,
)
from veritas.retrieval.tokenizer import MultilingualTokenizer
from veritas.verification.nli_model import CrossEncoderNLI, resolve_device
from veritas.verification.evidence_aggregator import EvidenceAggregator, AggregatedEvidenceVerdict
from veritas.verification.calibrator import TemperatureCalibrator, SelectiveAbstainer
from veritas.retrieval.evidence_corpus import EvidenceCorpus
from veritas.retrieval.bm25_retriever import BM25Retriever
from veritas.retrieval.dense_retriever import DenseRetriever
from veritas.retrieval.hybrid_retriever import HybridRetriever
from veritas.reasoning.decomposer import ClaimDecomposer
from veritas.reasoning.types import SubClaim, SubClaimStatus
from veritas.temporal.temporal_analyzer import TemporalAnalyzer
from veritas.graph.graph_builder import ReasoningGraphBuilder
from veritas.retrieval.tavily_provider import TavilySearchProvider, deduplicate_passages, load_dotenv


FROZEN_AGGREGATOR_PARAMS: Dict[str, float] = {
    "disparity_entail_threshold": 0.35,
    "disparity_contra_threshold": 0.50,
    "conflict_threshold": 0.45,
    "min_informative_mass": 0.10,
    "joint_weight_disparity": 0.40,
    "joint_weight_default": 1.20,
}

FROZEN_CALIBRATION_TEMPERATURE: float = 4.1039
FROZEN_ABSTENTION_CONF_THRESHOLD: float = 0.32
FROZEN_ABSTENTION_ENT_THRESHOLD: float = 0.94
FROZEN_EVIDENCE_AWARE_CONF_THRESHOLD: float = 0.38
FROZEN_MARGIN_THRESHOLD: float = 0.10

DEFAULT_CORPUS_PATH = os.path.join(_REPO_ROOT, "data", "processed", "evidence_corpus.jsonl")


class ObservableEvidenceAggregator(EvidenceAggregator):
    """
    Subclasses the frozen EvidenceAggregator to record the last aggregated consensus verdict.
    This safely surfaces the `has_cross_passage_disparity` flag to API consumers without
    altering `src/veritas/verification/engine.py`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_verdict: Optional[AggregatedEvidenceVerdict] = None

    def aggregate(self, *args, **kwargs) -> AggregatedEvidenceVerdict:
        verdict = super().aggregate(*args, **kwargs)
        self.last_verdict = verdict
        return verdict


def generate_claim_id(claim_text: str) -> str:
    """Produces a deterministic, professional claim ID for UI tracking."""
    digest = hashlib.sha256(claim_text.strip().encode("utf-8")).hexdigest()[:6].upper()
    return f"VX-{digest}"


class VeritasAPIAdapter:
    """
    Adapter adhering to the schema specified in `backend_contract.md`.
    Loads local hybrid retrieval and frozen Phase 3.2.5 parameters.
    """

    def __init__(
        self,
        engine: Optional[VerificationEngine] = None,
        corpus_path: Optional[str] = None,
        device_override: Optional[str] = None,
        lazy_init: bool = False,
    ):
        self.corpus_path = corpus_path or DEFAULT_CORPUS_PATH
        self.device_override = device_override
        self._engine = engine
        self._aggregator: Optional[ObservableEvidenceAggregator] = None
        self._init_lock = threading.Lock()
        self._is_loading: bool = False
        self._is_ready: bool = bool(self._engine is not None)
        self.last_init_latency_ms: Optional[float] = None

        # VERITAS-X 2.0 Reasoning, Temporal, and Graph Services
        self._decomposer = ClaimDecomposer()
        self._temporal_analyzer = TemporalAnalyzer()
        self._graph_builder = ReasoningGraphBuilder()
        self.tavily_provider = TavilySearchProvider()

        if not lazy_init and self._engine is None:
            self._init_engine()

    @property
    def is_model_ready(self) -> bool:
        """Returns True if the verification engine is fully loaded in memory."""
        return self._engine is not None and self._is_ready

    @property
    def is_model_loading(self) -> bool:
        """Returns True if background warming or initialization is actively underway."""
        return self._is_loading

    def warmup(self, background: bool = False) -> None:
        """Pre-warms the models and retrieval index so the first request is instant."""
        if self.is_model_ready:
            return

        def _worker():
            if self._engine is None:
                self._init_engine()
            # Fast warm-up pass to compile CUDA graphs / kernel caches
            try:
                if self._engine and hasattr(self._engine, "nli"):
                    _ = self._engine.nli.predict_batch([("Warmup evidence text", "Warmup claim")])
            except Exception:
                pass

        if background:
            t = threading.Thread(target=_worker, daemon=True, name="VeritasWarmupThread")
            t.start()
        else:
            _worker()

    def _init_engine(self) -> None:
        """Instantiates the real local retrieval and frozen verification pipeline."""
        with self._init_lock:
            if self._engine is not None:
                return

            self._is_loading = True
            t_init_start = time.perf_counter()
            try:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                device = resolve_device(self.device_override)

                # 1. Load frozen NLI Cross-Encoder
                nli_model = CrossEncoderNLI(
                    model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                    device=device,
                )

                if device == "cuda":
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                # 2. Load local reproducible corpus & hybrid retriever
                hybrid_retriever = None
                if os.path.exists(self.corpus_path):
                    corpus = EvidenceCorpus.load_from_jsonl(self.corpus_path)
                    bm25 = BM25Retriever(corpus)
                    try:
                        # Dense retrieval query encoding runs cleanly and deterministically on CPU,
                        # leaving 100% of GPU VRAM dedicated to the DeBERTa-v3 cross-encoder
                        dense = DenseRetriever(corpus, device="cpu")
                    except Exception as e:
                        logger.warning("DenseRetriever init failed (%s)", e)
                        dense = None
                    hybrid_retriever = HybridRetriever(bm25, dense, rrf_k=60)

                # 3. Instantiate frozen calibrator & selective abstainer
                calibrator = TemperatureCalibrator(temperature=FROZEN_CALIBRATION_TEMPERATURE)
                abstainer = SelectiveAbstainer(
                    confidence_threshold=FROZEN_ABSTENTION_CONF_THRESHOLD,
                    entropy_threshold=FROZEN_ABSTENTION_ENT_THRESHOLD,
                    evidence_aware_conf_threshold=FROZEN_EVIDENCE_AWARE_CONF_THRESHOLD,
                    margin_threshold=FROZEN_MARGIN_THRESHOLD,
                )

                # 4. Instantiate observable aggregator with frozen tuned hyper-parameters
                self._aggregator = ObservableEvidenceAggregator(**FROZEN_AGGREGATOR_PARAMS)

                # 5. Wire into VerificationEngine
                self._engine = VerificationEngine(
                    nli_model=nli_model,
                    retriever=hybrid_retriever,
                    calibrator=calibrator,
                    abstainer=abstainer,
                    device=device,
                )
                self._engine.aggregator = self._aggregator
                self.last_init_latency_ms = round((time.perf_counter() - t_init_start) * 1000.0, 1)
                self._is_ready = True
            finally:
                self._is_loading = False

    @property
    def engine(self) -> VerificationEngine:
        if self._engine is None:
            self._init_engine()
        return self._engine

    def verify(
        self,
        claim: str,
        user_evidence: Optional[List[str]] = None,
        claim_id: Optional[str] = None,
        mode: str = "local",
        include_timings: bool = False,
        include_research: bool = False,
        candidate_passages: Optional[List[Dict[str, Any]]] = None,
        include_pipeline_trace: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes verification across requested evidence mode ('local', 'web', 'hybrid')
        and returns exact payload documented in `backend_contract.md`.
        Instruments per-stage monotonic timings without modifying model logic.
        """
        # Ensure engine is loaded and record cold init latency separately if deferred
        init_start = time.perf_counter()
        active_engine = self.engine
        init_delta_ms = round((time.perf_counter() - init_start) * 1000.0, 1)

        t_infer_start = time.perf_counter()

        # Step A: Claim Preprocessing & Validation
        t_prep_0 = time.perf_counter()
        claim_clean = (claim or "").strip()
        if not claim_clean:
            raise ValueError("Claim text cannot be empty.")
        cid = claim_id or generate_claim_id(claim_clean)
        t_prep_ms = round((time.perf_counter() - t_prep_0) * 1000.0, 3)

        # Normalize mode
        normalized_mode = str(mode or "local").lower().strip()
        if normalized_mode in ("web", "online"):
            normalized_mode = "web"
        elif normalized_mode not in ("local", "web", "hybrid"):
            normalized_mode = "local"

        # Convert user-supplied evidence strings to candidate passage format if provided
        retrieval_status: Dict[str, Any] = {
            "mode": normalized_mode,
            "provider": "unknown",
            "status": "pending",
            "results_count": 0,
            "error": None,
        }

        if candidate_passages is not None:
            retrieval_status = {
                "mode": normalized_mode,
                "provider": "cached_or_fixture",
                "status": "success",
                "results_count": len(candidate_passages),
                "error": None,
            }
        elif user_evidence:
            candidate_passages = []
            for i, item in enumerate(user_evidence):
                if isinstance(item, dict):
                    cand = dict(item)
                    if "doc_id" not in cand:
                        cand["doc_id"] = f"user_evidence_{i+1:02d}"
                    if "source_type" not in cand:
                        cand["source_type"] = "user"
                    candidate_passages.append(cand)
                elif str(item).strip():
                    candidate_passages.append({
                        "doc_id": f"user_evidence_{i+1:02d}",
                        "text": str(item).strip(),
                        "source_url": "User-supplied evidence passage",
                        "source_title": "User Submission",
                        "source_dataset": "User Submission",
                        "source_type": "user",
                    })
            if candidate_passages:
                normalized_mode = "user_supplied"
                retrieval_status = {
                    "mode": "user_supplied",
                    "provider": "user",
                    "status": "success",
                    "results_count": len(candidate_passages),
                    "error": None,
                }

        # Step B: Evidence Retrieval (Local / Web / Hybrid)
        t_ret_0 = time.perf_counter()
        if candidate_passages is None:
            if normalized_mode == "web":
                web_passages, web_status = self.tavily_provider.search(claim_clean, top_k=5)
                candidate_passages = web_passages
                status_code = web_status.get("status", "success")
                retrieval_status = {
                    **web_status,
                    "mode": "web",
                    "status": status_code,
                    "count": web_status.get("results_count", len(web_passages)),
                    "results_count": web_status.get("results_count", len(web_passages)),
                    "local_count": 0,
                    "web_count": len(web_passages),
                    "time_ms": web_status.get("latency_ms", 0),
                }
            elif normalized_mode == "hybrid":
                local_passages: List[Dict[str, Any]] = []
                if active_engine.retriever is not None:
                    local_results = active_engine.retriever.search(claim_clean, top_k=5)
                    local_passages = [r.to_dict() for r in local_results]
                    for p in local_passages:
                        p["source_type"] = "local"
                        p["source_title"] = p.get("source_dataset", "Local Corpus")

                web_passages, web_status = self.tavily_provider.search(claim_clean, top_k=5)
                combined = deduplicate_passages(local_passages + web_passages)
                candidate_passages = combined

                is_local_only = len(web_passages) == 0 and len(local_passages) > 0
                web_err = web_status.get("error") if (not web_passages and web_status.get("status") in ("unconfigured", "limit_reached", "billing_restricted", "quota_exceeded", "http_error", "timeout", "network_error")) else None
                notice_str = None
                if is_local_only and web_err:
                    notice_str = f"Live Web search was unavailable ({web_err}). Adjudicated strictly using verified local corpus evidence only."

                retrieval_status = {
                    "mode": "hybrid",
                    "provider": "hybrid_local_and_web",
                    "status": "success" if combined else "no_results",
                    "local_count": len(local_passages),
                    "web_count": len(web_passages),
                    "results_count": len(combined),
                    "is_local_only": is_local_only,
                    "web_status": web_status,
                    "error": web_err,
                    "notice": notice_str,
                }
            else:
                # Default: Local corpus retrieval
                local_passages = []
                if active_engine.retriever is not None:
                    local_results = active_engine.retriever.search(claim_clean, top_k=5)
                    local_passages = [r.to_dict() for r in local_results]
                    for p in local_passages:
                        p["source_type"] = "local"
                        p["source_title"] = p.get("source_dataset", "Local Corpus")
                candidate_passages = local_passages
                retrieval_status = {
                    "mode": "local",
                    "provider": "local_corpus",
                    "status": "success" if candidate_passages else "no_results",
                    "results_count": len(candidate_passages),
                    "local_count": len(candidate_passages),
                    "web_count": 0,
                    "error": None,
                }

        t_ret_ms = round((time.perf_counter() - t_ret_0) * 1000.0, 2)
        if retrieval_status is not None:
            if "time_ms" not in retrieval_status or not retrieval_status["time_ms"]:
                retrieval_status["time_ms"] = t_ret_ms

        # Step C: NLI Forward Pass & Engine Stance Adjudication
        t_eng_0 = time.perf_counter()
        verdict: ForensicVerdict = active_engine.verify_claim(
            claim_text=claim_clean,
            candidate_passages=candidate_passages,
            claim_id=cid,
            top_k=5,
        )
        t_eng_ms = round((time.perf_counter() - t_eng_0) * 1000.0, 2)

        return self._format_report(
            verdict=verdict,
            claim_clean=claim_clean,
            cid=cid,
            mode=normalized_mode,
            retrieval_status=retrieval_status,
            t_infer_start=t_infer_start,
            init_delta_ms=init_delta_ms,
            t_prep_ms=t_prep_ms,
            t_ret_ms=t_ret_ms,
            t_eng_ms=t_eng_ms,
            include_timings=include_timings,
            include_research=include_research,
            include_pipeline_trace=include_pipeline_trace,
        )

    def _format_report(
        self,
        verdict: ForensicVerdict,
        claim_clean: str = "",
        cid: str = "",
        mode: str = "local",
        retrieval_status: Optional[Dict[str, Any]] = None,
        t_infer_start: float = 0.0,
        init_delta_ms: float = 0.0,
        t_prep_ms: float = 0.0,
        t_ret_ms: float = 0.0,
        t_eng_ms: float = 0.0,
        include_timings: bool = True,
        include_research: bool = False,
        include_pipeline_trace: bool = False,
    ) -> Dict[str, Any]:
        """Formats a ForensicVerdict into the client API payload contract."""
        claim_clean = claim_clean or getattr(verdict, "claim_text", "")
        cid = cid or getattr(verdict, "claim_id", "") or generate_claim_id(claim_clean)

        # Extract cross-passage disparity state
        disparity_state = False
        if self._aggregator is not None and self._aggregator.last_verdict is not None:
            disparity_state = bool(self._aggregator.last_verdict.has_cross_passage_disparity)
        elif self._engine and hasattr(self._engine, "aggregator") and hasattr(self._engine.aggregator, "last_verdict"):
            lv = getattr(self._engine.aggregator, "last_verdict")
            if lv is not None:
                disparity_state = bool(getattr(lv, "has_cross_passage_disparity", False))
        else:
            has_sup = any(item.get("is_active_supporter") for item in verdict.evidence_breakdown)
            has_ref = any(item.get("is_active_refuter") for item in verdict.evidence_breakdown)
            disparity_state = bool(has_sup and has_ref)

        # Map canonical probabilities
        raw_probs = verdict.class_probabilities or {}
        probabilities = {
            CanonicalVeracity.VERIFIED_TRUE.value: round(float(raw_probs.get(CanonicalVeracity.VERIFIED_TRUE.value, 0.0)), 4),
            CanonicalVeracity.VERIFIED_FALSE.value: round(float(raw_probs.get(CanonicalVeracity.VERIFIED_FALSE.value, 0.0)), 4),
            CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value: round(float(raw_probs.get(CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value, 0.0)), 4),
            CanonicalVeracity.INSUFFICIENT_EVIDENCE.value: round(float(raw_probs.get(CanonicalVeracity.INSUFFICIENT_EVIDENCE.value, 0.0)), 4),
        }

        # Normalize verdict label
        verdict_str = verdict.verdict.value if hasattr(verdict.verdict, "value") else str(verdict.verdict)

        # Map evidence items to backend contract
        formatted_evidence: List[Dict[str, Any]] = []
        item_counter = 1
        for item in verdict.evidence_breakdown:
            if item.get("is_joint", False):
                continue  # Skip synthetic joint premise

            e_prob = float(item.get("entailment_prob", 0.0))
            c_prob = float(item.get("contradiction_prob", 0.0))
            n_prob = float(item.get("neutral_prob", 0.0))

            is_rel = bool(item.get("is_relevant", True))
            is_aligned = bool(item.get("is_claim_aligned", True))
            text_inf = bool(item.get("is_informative", True))
            is_fig = bool(item.get("is_figurative", False))
            cov_level = str(item.get("claim_coverage", "full"))
            cov_labels = list(item.get("claim_components_covered", []))

            is_informative = bool(is_rel and is_aligned and text_inf and not is_fig and not (n_prob >= 0.70 or max(e_prob, c_prob) < 0.25))
            is_eligible = bool(item.get("is_decision_eligible", bool(is_rel and is_informative and is_aligned)) and not is_fig)
            raw_url = item.get("source_url") or ""

            if item.get("stance") in ("support", "refute", "neutral"):
                stance = item["stance"]
                score = e_prob if stance == "support" else (c_prob if stance == "refute" else (0.0 if not is_rel else n_prob))
            elif is_fig:
                stance = "neutral"
                score = n_prob
            elif is_informative:
                if e_prob >= c_prob and (e_prob >= n_prob or e_prob >= 0.45):
                    stance = "support"
                    score = e_prob
                elif c_prob > e_prob and c_prob >= 0.50 and c_prob >= n_prob:
                    stance = "refute"
                    score = c_prob
                else:
                    stance = "neutral"
                    score = n_prob
            elif not is_rel:
                stance = "neutral"
                score = 0.0  # Off-topic/irrelevant evidence gets zero relevance score
            else:
                stance = "neutral"
                score = n_prob if n_prob >= max(e_prob, c_prob) else max(e_prob, c_prob)

            # Determine provenance strictly based on mode and passage metadata:
            if mode == "local":
                source_type = "local"
                source_title = item.get("source_title") or item.get("source_dataset") or "Local Evidence Corpus"
                source = raw_url if str(raw_url).startswith("http") else source_title
            elif mode == "web":
                source_type = "web"
                source_title = item.get("source_title") or (raw_url if str(raw_url).startswith("http") else "Live Web Citation")
                source = raw_url if str(raw_url).startswith("http") else source_title
            elif mode == "user_supplied":
                source_type = "user_supplied"
                source_title = item.get("source_title") or "Supplied Fixture Premise"
                source = source_title
            else:  # hybrid
                raw_st = item.get("source_type")
                if raw_st in ("web", "local", "user_supplied", "user"):
                    source_type = "user_supplied" if raw_st == "user" else raw_st
                elif str(item.get("doc_id", "")).startswith("web_"):
                    source_type = "web"
                else:
                    source_type = "local"

                if source_type == "web":
                    source_title = item.get("source_title") or (raw_url if str(raw_url).startswith("http") else "Live Web Citation")
                    source = raw_url if str(raw_url).startswith("http") else source_title
                else:
                    source_title = item.get("source_title") or item.get("source_dataset") or "Local Evidence Corpus"
                    source = raw_url if str(raw_url).startswith("http") else source_title

            # Grounded explanation per passage
            if not is_rel:
                explanation = "Passage does not discuss the subject of this claim and was excluded from verdict determination."
            elif is_fig:
                explanation = "Passage defines the figurative expression or idiom, rather than providing empirical scientific evidence for the literal proposition."
            elif not text_inf:
                explanation = "Passage provides general context, but lacks sufficient complete information to confirm or deny the statement."
            elif not is_aligned:
                explanation = "Passage mentions related topics, but does not address the specific attribute or relation asserted."
            elif stance == "support" and cov_level == "partial":
                explanation = f"Passage partially supports the statement ({', '.join(cov_labels) if cov_labels else 'partial components'}), but does not independently establish all components of the claim."
            elif stance == "support":
                explanation = "Passage directly confirms the factual assertion made in the statement."
            elif stance == "refute":
                explanation = "Passage directly contradicts the statement by documenting conflicting facts."
            else:
                explanation = "Passage discusses related topics without confirming or refuting the specific assertion."

            # A passage contributes decisively to the complete claim only if eligible, active, and having full coverage
            contributed_decisive = bool(
                is_eligible
                and (item.get("is_active_supporter") or item.get("is_active_refuter"))
                and (cov_level == "full" or not cov_labels or len(cov_labels) <= 1)
            )

            formatted_evidence.append({
                "id": f"E-{item_counter:02d}",
                "source": str(source),
                "source_title": str(source_title),
                "source_url": raw_url if raw_url.startswith("http") else None,
                "source_type": source_type,
                "source_label": "Live Web" if source_type == "web" else ("Supplied Fixture" if source_type == "user_supplied" else "Local Corpus"),
                "published_date": item.get("published_date"),
                "text": str(item.get("text", "")),
                "stance": stance,
                "score": round(score, 4),
                "entailment_prob": round(e_prob, 4),
                "contradiction_prob": round(c_prob, 4),
                "neutral_prob": round(n_prob, 4),
                "is_informative": is_informative,
                "is_relevant": is_rel,
                "relevance_status": "relevant" if is_rel else "unrelated",
                "is_claim_aligned": is_aligned,
                "is_decision_eligible": is_eligible,
                "is_figurative": is_fig,
                "claim_coverage": cov_level,
                "claim_components_covered": cov_labels,
                "contributed_to_decision": contributed_decisive,
                "explanation": explanation,
                "doc_id": item.get("doc_id"),
            })
            item_counter += 1

        # Count actual stance breakdown across non-joint informative passages
        total_ev = len(formatted_evidence)
        sup_count = len([e for e in formatted_evidence if e["stance"] in ("support", "entailment") and e.get("is_decision_eligible", False)])
        ref_count = len([e for e in formatted_evidence if e["stance"] in ("refute", "contradiction") and e.get("is_decision_eligible", False)])
        neut_count = len([e for e in formatted_evidence if e["stance"] == "neutral" and e.get("is_relevant", True)])
        unrelated_count = len([e for e in formatted_evidence if not e.get("is_relevant", True)])
        uninformative_count = len([e for e in formatted_evidence if e.get("is_relevant", True) and not e.get("is_informative", True)])
        partial_count = len([e for e in formatted_evidence if e.get("claim_coverage") == "partial" and e.get("is_relevant", True)])
        conf_pct = verdict.confidence * 100.0

        evidence_counts = {
            "total": total_ev,
            "eligible_supporting": sup_count,
            "eligible_refuting": ref_count,
            "relevant_neutral": neut_count,
            "relevant_partial": partial_count,
            "off_topic": unrelated_count,
            "duplicates": getattr(verdict, "duplicates_filtered", 0),
        }

        raw_pred = getattr(verdict, "raw_prediction", None) or verdict_str
        has_ev = getattr(verdict, "has_meaningful_evidence", True)

        # Context-aware evidence description for reason string
        if mode == "web":
            src_label = "Live web"
        elif mode == "hybrid":
            src_label = "Hybrid (local & web)"
        elif mode == "user_supplied":
            src_label = "Supplied"
        else:
            src_label = "Local"

        # Build concise, evidence-consistent factual reason
        ret_status_code = (retrieval_status or {}).get("status")
        ret_error = (retrieval_status or {}).get("error")

        if mode == "web" and ret_status_code == "unconfigured":
            reason = "Live web verification is unconfigured: TAVILY_API_KEY environment variable is not configured. Configure your API key in D:\\VERITAS-X\\.env or select Local Corpus mode."
        elif mode == "web" and ret_status_code in ("limit_reached", "quota_exceeded"):
            reason = f"Live web retrieval stopped: {ret_error or 'Monthly free-tier safety limit reached'}. Select Local Corpus mode to verify offline."
        elif mode == "web" and ret_status_code == "billing_restricted":
            reason = f"Live web retrieval paused: {ret_error or 'Provider billing protection active'}. Safe offline verification remains functional in Local Corpus mode."
        elif mode == "web" and ret_status_code in ("timeout", "network_error", "http_error", "unauthorized"):
            reason = f"Live web retrieval failed: {ret_error or 'Network request to search provider could not be completed'}. No online evidence could be evaluated."
        elif mode == "web" and (ret_status_code == "no_results" or total_ev == 0):
            reason = "Live web search returned zero relevant passages for this claim. No factual conclusion was established."
        elif mode == "hybrid" and (retrieval_status or {}).get("is_local_only"):
            notice = (retrieval_status or {}).get("notice")
            if notice:
                reason = notice
            elif total_ev == 0:
                reason = "No relevant evidence passages were located in local or web retrieval. No factual conclusion was established."
            else:
                reason = f"Hybrid verification fell back to local corpus evidence ({sup_count} supporting, {ref_count} refuting, {neut_count} neutral). Calibrated confidence {conf_pct:.1f}%."
        elif total_ev == 0 and not verdict.was_abstained:
            reason = f"No relevant evidence passages were located in the {src_label.lower()} corpus for this claim. No factual conclusion was established."
        elif verdict.was_abstained:
            abst_desc = getattr(verdict, "abstention_reason", "")
            if unrelated_count == total_ev and total_ev > 0:
                reason = f"No relevant evidence was found in the {src_label.lower()} corpus ({unrelated_count} off-topic passage(s) excluded). No factual conclusion was established."
            elif not has_ev:
                reason = f"Abstained — Insufficient Evidentiary Grounding ({abst_desc}). Retrieved passages lack supporting or refuting evidence for the asserted proposition."
            else:
                reason = f"Abstained from definitive verdict ({abst_desc}). Coverage policy deferred this claim for human review (calibrated confidence {conf_pct:.1f}%)."
        elif verdict_str == CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value:
            if sup_count > 0 and ref_count > 0:
                reason = f"Cross-passage stance disparity detected: {sup_count} supporting passage(s) and {ref_count} refuting passage(s) present contradictory documentary evidence (calibrated confidence {conf_pct:.1f}%)."
            else:
                reason = f"Cross-passage stance disparity detected: opposing evidence positions were documented across retrieved passages (calibrated confidence {conf_pct:.1f}%)."
        elif verdict_str == CanonicalVeracity.INSUFFICIENT_EVIDENCE.value:
            if unrelated_count == total_ev and total_ev > 0:
                reason = f"No relevant evidence was found in the {src_label.lower()} corpus ({unrelated_count} off-topic passage(s) excluded). No factual conclusion was established."
            else:
                reason = verdict.abstention_reason or f"Insufficient informative evidence mass retrieved to adjudicate this claim ({neut_count} of {total_ev} passage(s) neutral or uninformative)."
        elif verdict_str == CanonicalVeracity.VERIFIED_TRUE.value:
            if sup_count > 0:
                reason = f"{src_label} evidence supports the claim with {sup_count} supporting passage(s) (calibrated confidence {conf_pct:.1f}%)."
            else:
                reason = f"Plurality probability assigned to VERIFIED_TRUE with low calibrated confidence ({conf_pct:.1f}%), but no individual retrieved passage directly entails the proposition ({neut_count} neutral/uninformative passage(s))."
        elif verdict_str == CanonicalVeracity.VERIFIED_FALSE.value:
            if ref_count > 0:
                reason = f"{src_label} evidence refutes the claim with {ref_count} refuting passage(s) (calibrated confidence {conf_pct:.1f}%)."
            elif getattr(verdict, "abstention_reason", "") and "Constituent proposition" in getattr(verdict, "abstention_reason", ""):
                reason = f"{verdict.abstention_reason} (calibrated confidence {conf_pct:.1f}%)."
            else:
                reason = f"Insufficient informative evidence retrieved: no individual retrieved passage directly refutes the proposition ({neut_count} neutral/uninformative passage(s))."
        else:
            reason = "Adjudication completed by VERITAS-X verification pipeline."

        # If engine synthesized a canonical reason and evidence counts directly from the 5-stage pipeline,
        # adopt them as the canonical single-source-of-truth unless a web retrieval provider error intervened:
        if isinstance(getattr(verdict, "evidence_counts", None), dict):
            evidence_counts = verdict.evidence_counts

        if isinstance(getattr(verdict, "canonical_reason", None), str) and getattr(verdict, "canonical_reason", None) and (mode != "web" or ret_status_code not in ("unconfigured", "limit_reached", "quota_exceeded", "billing_restricted", "timeout", "network_error", "http_error", "unauthorized")):
            reason = verdict.canonical_reason

        latency_ms = round((time.perf_counter() - t_infer_start) * 1000.0, 1) if t_infer_start > 0 else 0.0
        timing_breakdown = {
            "init_ms": init_delta_ms,
            "claim_prep_ms": t_prep_ms,
            "retrieval_ms": t_ret_ms,
            "engine_inference_ms": t_eng_ms,
            "total_ms": latency_ms,
        }
        self.last_timing_ms = timing_breakdown

        response_payload = {
            "claim_id": cid,
            "claim": claim_clean,
            "verdict": verdict_str,
            "confidence": round(float(verdict.confidence), 4),
            "probabilities": probabilities,
            "has_cross_passage_disparity": disparity_state,
            "abstained": bool(verdict.was_abstained),
            "reason": reason,
            "evidence": formatted_evidence,
            "device": str(getattr(self._engine, "device", "cpu")) if self._engine else "cpu",
            "latency_ms": latency_ms,
            "mode": mode,
            "retrieval_status": retrieval_status or {
                "mode": mode,
                "provider": "local_corpus" if mode == "local" else "tavily",
                "status": "success",
                "results_count": len(formatted_evidence),
                "error": None,
            },
            "evidence_counts": evidence_counts,
        }

        if verdict.was_abstained or include_research:
            response_payload["raw_verdict"] = raw_pred
            response_payload["abstention_reason"] = getattr(verdict, "abstention_reason", "")
            response_payload["has_meaningful_evidence"] = has_ev

        if include_research or include_pipeline_trace:
            response_payload["pipeline_trace"] = getattr(verdict, "pipeline_trace", {})

        if include_timings:
            response_payload["timing_ms"] = timing_breakdown
            response_payload["model_ready"] = True

        if include_research:
            try:
                subclaims_objs = self._decomposer.decompose(claim_clean)

                # Precompute token sets for each subclaim
                sc_token_maps = []
                for sc in subclaims_objs:
                    tokens = {
                        t.lower() for t in MultilingualTokenizer.tokenize(sc.text)
                        if t.lower() not in _TOPICAL_STOP_WORDS and len(t) > 1
                    }
                    sc_token_maps.append(tokens)

                # Map each eligible evidence item to the subclaim(s) it best addresses
                mapped_evidence_by_sc: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(subclaims_objs))}

                if len(subclaims_objs) == 1:
                    mapped_evidence_by_sc[0] = [
                        e for e in formatted_evidence
                        if e.get("is_relevant") is not False and e.get("is_decision_eligible") is not False
                    ]
                else:
                    for e in formatted_evidence:
                        if e.get("is_relevant") is False or e.get("is_decision_eligible") is False:
                            continue
                        e_tokens = {t.lower() for t in MultilingualTokenizer.tokenize(e.get("text", ""))}

                        scores = []
                        for sc_idx, sc_tokens in enumerate(sc_token_maps):
                            overlap = _match_token_set(sc_tokens, e_tokens)
                            req_overlap = 2 if len(sc_tokens) >= 3 else 1
                            if len(overlap) >= req_overlap:
                                score = len(overlap) / max(1, len(sc_tokens))
                                if score >= 0.25 or len(overlap) >= 3:
                                    scores.append((sc_idx, score))

                        if scores:
                            max_score = max(s[1] for s in scores)
                            for sc_idx, score in scores:
                                # Assign to best matching subclaim (or co-matching if within 15% margin)
                                if score >= max_score * 0.85:
                                    mapped_evidence_by_sc[sc_idx].append(e)

                # If compound claim with multiple subclaims, independently assess NLI stance of mapped evidence towards each constituent proposition
                sc_stance_map: Dict[int, Dict[str, List[Dict[str, Any]]]] = {
                    i: {"support": [], "refute": [], "neutral": []}
                    for i in range(len(subclaims_objs))
                }
                if len(subclaims_objs) > 1 and hasattr(self, "_engine") and self._engine and hasattr(self._engine, "nli"):
                    sc_eval_pairs = []
                    sc_eval_meta = []
                    for sc_idx, sc in enumerate(subclaims_objs):
                        for e in mapped_evidence_by_sc[sc_idx]:
                            sc_eval_pairs.append((e.get("text", ""), sc.text))
                            sc_eval_meta.append((sc_idx, e))

                    if sc_eval_pairs:
                        try:
                            sc_preds = self._engine.nli.predict_batch(sc_eval_pairs)
                            for (sc_idx, e), pred in zip(sc_eval_meta, sc_preds):
                                if isinstance(pred, dict):
                                    pred = VerificationEngine.adjust_causal_nli_prediction(subclaims_objs[sc_idx].text, e.get("text", ""), pred)
                                e_entry = dict(e)
                                p_label = pred.get("predicted_label", "").upper()
                                ep = float(pred.get("entailment", 0.0))
                                cp = float(pred.get("contradiction", 0.0))
                                np_ = float(pred.get("neutral", 0.0))
                                if p_label == "ENTAILMENT" or (ep > cp and ep >= np_ and ep >= 0.35):
                                    e_entry["stance"] = "support"
                                    e_entry["score"] = ep
                                    sc_stance_map[sc_idx]["support"].append(e_entry)
                                elif p_label == "CONTRADICTION" or (cp > ep and cp >= np_ and cp >= 0.35):
                                    e_entry["stance"] = "refute"
                                    e_entry["score"] = cp
                                    sc_stance_map[sc_idx]["refute"].append(e_entry)
                                else:
                                    e_entry["stance"] = "neutral"
                                    e_entry["score"] = np_
                                    sc_stance_map[sc_idx]["neutral"].append(e_entry)
                        except Exception as nli_err:
                            logger.warning("Subclaim NLI evaluation fallback: %s", nli_err)
                            sc_stance_map = None
                else:
                    sc_stance_map = None

                for sc_idx, sc in enumerate(subclaims_objs):
                    sc_ev = mapped_evidence_by_sc[sc_idx]
                    sc.evidence_ids = [e["id"] for e in sc_ev]
                    if sc_stance_map and sc_idx in sc_stance_map:
                        sup_e = sc_stance_map[sc_idx]["support"]
                        ref_e = sc_stance_map[sc_idx]["refute"]
                        neu_e = sc_stance_map[sc_idx]["neutral"]
                    else:
                        sup_e = [e for e in sc_ev if e.get("stance") in ("support", "entailment")]
                        ref_e = [e for e in sc_ev if e.get("stance") in ("refute", "contradiction")]
                        neu_e = [e for e in sc_ev if e not in sup_e and e not in ref_e]

                    sc.stance_distribution = {
                        "support": len(sup_e),
                        "refute": len(ref_e),
                        "neutral": len(neu_e),
                    }
                    if len(sup_e) > 0 and len(ref_e) > 0:
                        sc.status = SubClaimStatus.CONFLICTING
                        sc.verdict = CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value
                        sc.confidence = round(float(verdict.confidence), 4)
                        sc.explanation = f"Conflicting evidence: {len(sup_e)} supporting and {len(ref_e)} refuting passage(s) directly address this constituent."
                    elif len(ref_e) > 0:
                        sc.status = SubClaimStatus.REFUTED
                        sc.verdict = CanonicalVeracity.VERIFIED_FALSE.value
                        sc.confidence = round(max((float(e.get("score", 0.8)) for e in ref_e), default=0.8), 4)
                        sc.explanation = f"Refuted by {len(ref_e)} evidence passage(s) addressing this constituent."
                    elif len(sup_e) > 0:
                        sc.status = SubClaimStatus.SUPPORTED
                        sc.verdict = CanonicalVeracity.VERIFIED_TRUE.value
                        sc.confidence = round(max((float(e.get("score", 0.8)) for e in sup_e), default=0.8), 4)
                        sc.explanation = f"Supported by {len(sup_e)} evidence passage(s) addressing this constituent."
                    else:
                        sc.status = SubClaimStatus.UNRESOLVED
                        sc.verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE.value
                        sc.confidence = 0.0
                        sc.explanation = "No eligible documentary evidence directly addresses this constituent proposition."
                    sc.probabilities = probabilities

                temporal_analysis = self._temporal_analyzer.analyze_claim(claim_clean, formatted_evidence)

                graph = self._graph_builder.build_graph(
                    claim_id=cid,
                    claim_text=claim_clean,
                    verdict=verdict_str,
                    confidence=round(float(verdict.confidence), 4),
                    subclaims=subclaims_objs,
                    evidence_items=formatted_evidence,
                    temporal_events=temporal_analysis.get("timeline_events", []),
                )

                sorted_probs = sorted(probabilities.values(), reverse=True)
                top1 = sorted_probs[0] if len(sorted_probs) > 0 else 0.0
                top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
                prob_margin = round(top1 - top2, 4)
                import math
                entropy = round(-sum(p * math.log(max(p, 1e-12)) for p in sorted_probs if p > 0.0), 4)
                cov_ratio = round(
                    sum(1 for sc in subclaims_objs if sc.status != SubClaimStatus.UNRESOLVED) / max(1, len(subclaims_objs)),
                    3,
                )

                response_payload["subclaims"] = [sc.to_dict() for sc in subclaims_objs]
                response_payload["temporal"] = temporal_analysis
                response_payload["graph"] = graph.to_dict()
                response_payload["uncertainty"] = {
                    "is_abstained": bool(verdict.was_abstained),
                    "abstention_reason": getattr(verdict, "abstention_reason", "") if verdict.was_abstained else "CONFIDENCE_SUFFICIENT",
                    "has_meaningful_evidence": has_ev,
                    "coverage_ratio": cov_ratio,
                    "evidence_coverage_ratio": cov_ratio,
                    "calibrated_confidence": round(float(verdict.confidence), 4),
                    "top_margin": prob_margin,
                    "top2_margin": prob_margin,
                    "shannon_entropy": entropy,
                    "entropy": entropy,
                    "subclaim_count": len(subclaims_objs),
                    "unresolved_subclaim_count": sum(1 for sc in subclaims_objs if sc.status == SubClaimStatus.UNRESOLVED),
                    "temporal_status": temporal_analysis.get("temporal_status", "UNRESOLVED"),
                }
            except Exception as exc:
                logger.warning("Error generating research platform extensions: %s", exc)

        return response_payload
