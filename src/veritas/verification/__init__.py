"""
Verification and uncertainty module for VERITAS-X.
Provides NLI reasoning, multi-evidence aggregation, temporal consistency checks,
temperature calibration, and selective abstention.
"""
from .nli_model import CrossEncoderNLI
from .evidence_aggregator import EvidenceAggregator, AggregatedEvidenceVerdict
from .temporal_analyzer import TemporalAnalyzer, TemporalReport, TemporalStatus
from .calibrator import TemperatureCalibrator, SelectiveAbstainer, CalibrationMetrics
from .engine import VerificationEngine, ForensicVerdict
from .evaluation import VeracityEvaluator, VeracityEvaluationReport

__all__ = [
    "CrossEncoderNLI",
    "EvidenceAggregator",
    "AggregatedEvidenceVerdict",
    "TemporalAnalyzer",
    "TemporalReport",
    "TemporalStatus",
    "TemperatureCalibrator",
    "SelectiveAbstainer",
    "CalibrationMetrics",
    "VerificationEngine",
    "ForensicVerdict",
    "VeracityEvaluator",
    "VeracityEvaluationReport",
]
