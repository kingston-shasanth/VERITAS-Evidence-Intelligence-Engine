"""
Data module for VERITAS-X: loaders, schemas, and preprocessing pipelines.
"""
from .schema import (
    CanonicalVeracity,
    EvidenceItem,
    CanonicalClaim,
    DatasetSource,
    ValidationReport,
)
from .preprocessor import TextPreprocessor
from .fever_loader import FEVERLoader
from .averitec_loader import AVeriTeCLoader
from .multilingual_loader import MultilingualLoader
from .pipeline import UnifiedDataPipeline

__all__ = [
    "CanonicalVeracity",
    "EvidenceItem",
    "CanonicalClaim",
    "DatasetSource",
    "ValidationReport",
    "TextPreprocessor",
    "FEVERLoader",
    "AVeriTeCLoader",
    "MultilingualLoader",
    "UnifiedDataPipeline",
]
