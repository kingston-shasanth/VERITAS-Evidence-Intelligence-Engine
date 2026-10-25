"""
VERITAS-X 2.0 Unified Reasoning, Temporal, and Graph Data Types.
Defines typed data structures for subclaims, multi-hop verification,
temporal provenance, and the interactive evidence relationship graph.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class SubClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    CONFLICTING = "CONFLICTING"
    UNRESOLVED = "UNRESOLVED"


class TemporalScope(str, Enum):
    HISTORICAL = "HISTORICAL"
    CURRENT = "CURRENT"
    POINT_IN_TIME = "POINT_IN_TIME"
    TIME_SPAN = "TIME_SPAN"
    UNSPECIFIED = "UNSPECIFIED"


class TemporalStatus(str, Enum):
    CONSISTENT = "CONSISTENT"
    OUTDATED_SUPERSEDED = "OUTDATED_SUPERSEDED"
    ANACHRONISTIC = "ANACHRONISTIC"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class TemporalMetadata:
    """Provenance-backed temporal metadata for claims and evidence."""
    publication_date: Optional[str] = None          # YYYY-MM-DD or None (Never current system time!)
    event_date: Optional[str] = None                # Referenced event date or period
    date_provenance: str = "unknown"                # 'archive_url', 'news_path', 'text_extraction', 'metadata', 'unknown'
    date_confidence: float = 0.0                    # 0.0 to 1.0
    is_verified: bool = False
    raw_date_str: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "publication_date": self.publication_date,
            "event_date": self.event_date,
            "date_provenance": self.date_provenance,
            "date_confidence": round(self.date_confidence, 3),
            "is_verified": self.is_verified,
            "raw_date_str": self.raw_date_str,
        }


@dataclass
class SubClaim:
    """An atomic proposition extracted from a compound claim."""
    subclaim_id: str
    text: str
    span: Tuple[int, int] = (0, 0)                  # (start_char, end_char) in original claim
    dependencies: List[str] = field(default_factory=list)  # IDs of prerequisite subclaims
    relation_to_parent: str = "conjunct"            # 'conjunct', 'adversative', 'causal', 'temporal', 'clause'
    evidence_ids: List[str] = field(default_factory=list)
    verdict: Optional[str] = None                   # Canonical verdict for this subclaim
    confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    stance_distribution: Dict[str, int] = field(default_factory=lambda: {"support": 0, "refute": 0, "neutral": 0})
    status: SubClaimStatus = SubClaimStatus.UNRESOLVED
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subclaim_id": self.subclaim_id,
            "text": self.text,
            "span": list(self.span),
            "dependencies": self.dependencies,
            "relation_to_parent": self.relation_to_parent,
            "evidence_ids": self.evidence_ids,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "probabilities": self.probabilities,
            "stance_distribution": self.stance_distribution,
            "status": self.status.value,
            "explanation": self.explanation,
        }


@dataclass
class GraphNode:
    """A node in the evidence relationship graph."""
    id: str                                         # e.g. 'claim-root', 'sc-01', 'e-01', 'src-01'
    type: str                                       # 'claim', 'subclaim', 'passage', 'source'
    label: str
    text: str
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    stance: Optional[str] = None                    # 'support', 'refute', 'neutral'
    score: Optional[float] = None
    date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "text": self.text,
            "metadata": self.metadata,
        }
        if self.verdict is not None:
            d["verdict"] = self.verdict
        if self.confidence is not None:
            d["confidence"] = round(self.confidence, 4)
        if self.stance is not None:
            d["stance"] = self.stance
        if self.score is not None:
            d["score"] = round(self.score, 4)
        if self.date is not None:
            d["date"] = self.date
        return d


@dataclass
class GraphEdge:
    """A directed edge in the evidence relationship graph."""
    source: str
    target: str
    relation: str                                   # 'decomposes_to', 'retrieved_for', 'supports', 'refutes', 'neutral', 'published_by', 'depends_on', 'precedes'
    weight: float = 1.0
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": round(self.weight, 4),
            "label": self.label or self.relation,
            "metadata": self.metadata,
        }


@dataclass
class ReasoningGraph:
    """Complete graph topology of claim reasoning and evidence provenance."""
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "summary": self.summary,
        }
