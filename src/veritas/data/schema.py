"""
Unified Data Schema for VERITAS-X
Provides canonical representations, enums, dataclasses, and validation reports.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import hashlib
import json


class CanonicalVeracity(str, Enum):
    """
    Unified forensic label topology for VERITAS-X.
    Preserves semantic distinctions while reconciling diverse benchmark schemas.
    """
    VERIFIED_TRUE = "VERIFIED_TRUE"
    VERIFIED_FALSE = "VERIFIED_FALSE"
    CONFLICTING_OR_CHERRYPICKED = "CONFLICTING_OR_CHERRYPICKED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class DatasetSource(str, Enum):
    """Origin benchmark dataset."""
    FEVER = "FEVER"
    AVERITEC = "AVERITEC"
    INDIC_CLAIM_BUSTER = "INDIC_CLAIM_BUSTER"
    MULTILINGUAL_DIAGNOSTIC = "MULTILINGUAL_DIAGNOSTIC"


class EvidenceType(str, Enum):
    """Categorization of evidence format."""
    WIKIPEDIA_SENTENCE = "WIKIPEDIA_SENTENCE"
    WEB_EXTRACTIVE = "WEB_EXTRACTIVE"
    WEB_ABSTRACTIVE = "WEB_ABSTRACTIVE"
    NEWS_PASSAGE = "NEWS_PASSAGE"
    UNKNOWN = "UNKNOWN"


@dataclass
class EvidenceItem:
    """Individual piece of supporting/refuting context."""
    text: Optional[str] = None
    source_url: Optional[str] = None
    source_id: Optional[str] = None
    evidence_type: EvidenceType = EvidenceType.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source_url": self.source_url,
            "source_id": self.source_id,
            "evidence_type": self.evidence_type.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceItem":
        ev_type = EvidenceType(data.get("evidence_type", EvidenceType.UNKNOWN.value))
        return cls(
            text=data.get("text"),
            source_url=data.get("source_url"),
            source_id=data.get("source_id"),
            evidence_type=ev_type,
            metadata=data.get("metadata", {}),
        )


@dataclass
class CanonicalClaim:
    """
    Standardized claim object across all ingested benchmarks.
    Guarantees strict schema uniformity across disparate data formats.
    """
    claim_id: str
    claim_text: str
    source_dataset: DatasetSource
    raw_label: str
    canonical_label: CanonicalVeracity
    language: str = "en"
    claim_date: Optional[str] = None
    speaker: Optional[str] = None
    evidence: List[EvidenceItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Deterministic SHA-256 fingerprint of the normalized claim text."""
        norm_text = " ".join(self.claim_text.strip().lower().split())
        return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "source_dataset": self.source_dataset.value,
            "raw_label": self.raw_label,
            "canonical_label": self.canonical_label.value,
            "language": self.language,
            "claim_date": self.claim_date,
            "speaker": self.speaker,
            "evidence": [e.to_dict() for e in self.evidence],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalClaim":
        return cls(
            claim_id=data["claim_id"],
            claim_text=data["claim_text"],
            source_dataset=DatasetSource(data["source_dataset"]),
            raw_label=data["raw_label"],
            canonical_label=CanonicalVeracity(data["canonical_label"]),
            language=data.get("language", "en"),
            claim_date=data.get("claim_date"),
            speaker=data.get("speaker"),
            evidence=[EvidenceItem.from_dict(e) for e in data.get("evidence", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ValidationReport:
    """Summary of dataset integrity, schema checks, and label distributions."""
    dataset_name: str
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    duplicates_detected: int = 0
    missing_critical_fields: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    canonical_distribution: Dict[str, int] = field(default_factory=dict)
    language_distribution: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """Returns True if there are valid records and no critical structural failures."""
        return self.valid_records > 0 and self.invalid_records == 0

    def summary(self) -> str:
        lines = [
            f"=== Validation Report for {self.dataset_name} ===",
            f"Total Records Processed : {self.total_records}",
            f"Valid Canonical Records  : {self.valid_records}",
            f"Invalid / Malformed      : {self.invalid_records}",
            f"Duplicates Detected      : {self.duplicates_detected}",
            f"Missing Critical Fields  : {self.missing_critical_fields}",
            f"Raw Label Counts         : {dict(self.label_distribution)}",
            f"Canonical Label Counts   : {dict(self.canonical_distribution)}",
            f"Language Breakdown       : {dict(self.language_distribution)}",
        ]
        if self.errors:
            lines.append(f"Sample Errors (first 5) : {self.errors[:5]}")
        return "\n".join(lines)
