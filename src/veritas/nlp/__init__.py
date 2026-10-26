"""
VERITAS-X Natural Language Processing (NLP) Package.
Provides safe text sanitization, context-aware spelling normalization,
contraction expansion, linguistic analysis, structured claim extraction,
and meaning-preservation guardrails for truth-seeking verification.
"""
from typing import Any

from .sanitizer import (
    TextSanitizer,
    SanitizationResult,
    SanitizationTransformation,
)
from .spelling import (
    SpellingNormalizer,
    CorrectionItem,
    NormalizationResult,
    get_symspell_engine,
)
from .linguistics import (
    LinguisticAnalyzer,
    LinguisticAnalysisResult,
    AnalyzedSentence,
    LinguisticToken,
    Clause,
    Morphology,
)
from .claim_extractor import (
    ClaimExtractor,
    StructuredClaim,
    ModalityLevel,
    QuantifierType,
    QuantifierItem,
    NumericEntity,
    TemporalEntity,
    SubProposition,
)
from .guardrails import (
    MeaningPreservationGuardrails,
    GuardrailResult,
    GuardrailDecision,
)
from .telemetry import (
    PROCESSING_TRACE_VERSION,
    ProcessingStage,
    StageStatus,
    WordCorrectionEvent,
    TraceEvent,
    ProcessingTraceCollector,
)

_DEFAULT_NORMALIZER = None


def normalize_claim_text(text: str, max_edit_distance: int = 2) -> NormalizationResult:
    """
    Convenience function for normalizing claim text using the default singleton normalizer.
    Guarantees thread-safe, sub-millisecond execution.
    """
    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = SpellingNormalizer(max_edit_distance=max_edit_distance)
    return _DEFAULT_NORMALIZER.normalize(text)


def extract_structured_claim(text: str) -> StructuredClaim:
    """Convenience helper for extracting a structured proposition representation."""
    return ClaimExtractor.extract(text)


def verify_meaning_preservation(original_text: str, normalized_text_or_res: Any) -> GuardrailResult:
    """Convenience helper for evaluating semantic invariants between pre- and post-normalization claims."""
    return MeaningPreservationGuardrails.evaluate(original_text, normalized_text_or_res)


__all__ = [
    # Sanitization
    "TextSanitizer",
    "SanitizationResult",
    "SanitizationTransformation",
    # Spelling & Typo Correction
    "SpellingNormalizer",
    "CorrectionItem",
    "NormalizationResult",
    "get_symspell_engine",
    "normalize_claim_text",
    # Linguistics
    "LinguisticAnalyzer",
    "LinguisticAnalysisResult",
    "AnalyzedSentence",
    "LinguisticToken",
    "Clause",
    "Morphology",
    # Claim Structure Extraction
    "ClaimExtractor",
    "StructuredClaim",
    "ModalityLevel",
    "QuantifierType",
    "QuantifierItem",
    "NumericEntity",
    "TemporalEntity",
    "SubProposition",
    "extract_structured_claim",
    # Meaning-Preservation Guardrails
    "MeaningPreservationGuardrails",
    "GuardrailResult",
    "GuardrailDecision",
    "verify_meaning_preservation",
    # Animation-Ready Telemetry
    "PROCESSING_TRACE_VERSION",
    "ProcessingStage",
    "StageStatus",
    "WordCorrectionEvent",
    "TraceEvent",
    "ProcessingTraceCollector",
]
