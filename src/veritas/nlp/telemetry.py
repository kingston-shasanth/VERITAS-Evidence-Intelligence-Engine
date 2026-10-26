"""
Animation-Ready Telemetry and Processing Trace Module for VERITAS-X.

Provides an ordered, truthful execution trace tracking all verification stages:
1. input_received
2. sanitization
3. spelling_normalization
4. meaning_preservation
5. tokenization
6. linguistic_analysis
7. claim_extraction
8. retrieval
9. evidence_processing
10. nli_verification
11. final_result

Design Principles:
- Captures genuine execution-time events (actual monotonic start/end times, actual durations).
- Distinguishes completed, skipped, fallback, uncertain, and failed states.
- Exposes word-level correction records and linguistic annotations for future cinematic UI animations.
- Never fabricates intermediate scores, fake progress timers, or simulated stages.
"""
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import List, Dict, Any, Optional


PROCESSING_TRACE_VERSION: int = 1


class ProcessingStage(str, Enum):
    INPUT_RECEIVED = "input_received"
    SANITIZATION = "sanitization"
    SPELLING_NORMALIZATION = "spelling_normalization"
    MEANING_PRESERVATION = "meaning_preservation"
    TOKENIZATION = "tokenization"
    LINGUISTIC_ANALYSIS = "linguistic_analysis"
    CLAIM_EXTRACTION = "claim_extraction"
    RETRIEVAL = "retrieval"
    EVIDENCE_PROCESSING = "evidence_processing"
    NLI_VERIFICATION = "nli_verification"
    FINAL_RESULT = "final_result"


class StageStatus(str, Enum):
    COMPLETED = "completed"
    RUNNING = "running"
    PENDING = "pending"
    SKIPPED = "skipped"
    FALLBACK = "fallback"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


@dataclass
class WordCorrectionEvent:
    """Detailed record of an individual word-level correction for animation."""
    original: str
    replacement: str
    status: str                       # "accepted", "rejected", "uncertain"
    category: str                     # "spelling", "contraction", "abbreviation"
    confidence: float                 # Heuristic ranking score from engine
    edit_distance: int
    start_char: int
    end_char: int
    reason: Optional[str] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "replacement": self.replacement,
            "status": self.status,
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "edit_distance": self.edit_distance,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "reason": self.reason,
            "candidates": self.candidates,
        }


@dataclass
class TraceEvent:
    """Ordered event record in the processing trace."""
    event_id: str
    sequence: int
    stage: ProcessingStage
    status: StageStatus
    started_at_ms: float
    duration_ms: float
    summary: str
    input_ref: Optional[str] = None
    output_ref: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "stage": self.stage.value,
            "status": self.status.value,
            "started_at_ms": round(self.started_at_ms, 2),
            "duration_ms": round(self.duration_ms, 2),
            "summary": self.summary,
            "input_ref": self.input_ref,
            "output_ref": self.output_ref,
            "details": self.details,
        }


class ProcessingTraceCollector:
    """
    Collects real-time execution telemetry across all verification stages.
    Thread-safe per-request collector tracking monotonic elapsed time.
    """

    def __init__(self):
        self._t_start = time.perf_counter()
        self._events: List[TraceEvent] = []
        self._sequence_counter = 1
        self._active_stage_start: Optional[float] = None
        self._active_stage: Optional[ProcessingStage] = None

    @property
    def elapsed_ms(self) -> float:
        """Elapsed milliseconds since trace initialization."""
        return (time.perf_counter() - self._t_start) * 1000.0

    def start_stage(self, stage: ProcessingStage) -> None:
        """Marks the start of an active execution stage."""
        self._active_stage = stage
        self._active_stage_start = time.perf_counter()

    def record_stage(
        self,
        stage: ProcessingStage,
        status: StageStatus,
        summary: str,
        duration_ms: Optional[float] = None,
        input_ref: Optional[str] = None,
        output_ref: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> TraceEvent:
        """
        Records a completed, fallback, uncertain, skipped, or failed stage.
        """
        now = time.perf_counter()
        if duration_ms is None:
            if self._active_stage == stage and self._active_stage_start is not None:
                duration_ms = (now - self._active_stage_start) * 1000.0
            else:
                duration_ms = 0.0

        started_at = (self._active_stage_start - self._t_start) * 1000.0 if self._active_stage_start else self.elapsed_ms

        event = TraceEvent(
            event_id=f"evt_{self._sequence_counter:02d}_{stage.value}",
            sequence=self._sequence_counter,
            stage=stage,
            status=status,
            started_at_ms=max(0.0, started_at),
            duration_ms=max(0.0, duration_ms),
            summary=summary,
            input_ref=input_ref,
            output_ref=output_ref,
            details=details or {},
        )
        self._events.append(event)
        self._sequence_counter += 1

        # Reset active stage marker
        self._active_stage = None
        self._active_stage_start = None
        return event

    def get_trace(self) -> List[Dict[str, Any]]:
        """Returns the complete, ordered list of event dictionaries."""
        return [evt.to_dict() for evt in self._events]
