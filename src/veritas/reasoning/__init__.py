"""
VERITAS-X Reasoning and Multi-Hop Verification Package.
"""
from veritas.reasoning.types import (
    SubClaim,
    SubClaimStatus,
    TemporalScope,
    TemporalStatus,
    TemporalMetadata,
    GraphNode,
    GraphEdge,
    ReasoningGraph,
)
from veritas.reasoning.decomposer import ClaimDecomposer
from veritas.reasoning.multihop_verifier import MultiHopVerifier

__all__ = [
    "SubClaim",
    "SubClaimStatus",
    "TemporalScope",
    "TemporalStatus",
    "TemporalMetadata",
    "GraphNode",
    "GraphEdge",
    "ReasoningGraph",
    "ClaimDecomposer",
    "MultiHopVerifier",
]
