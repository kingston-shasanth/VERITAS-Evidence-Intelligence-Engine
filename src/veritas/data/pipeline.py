"""
Unified Data Pipeline for VERITAS-X.
Coordinates loading, schema harmonization, validation, deduplication, and export.
"""
from typing import List, Dict, Optional, Tuple
import os
import json
from .schema import (
    CanonicalClaim,
    ValidationReport,
    DatasetSource,
)
from .fever_loader import FEVERLoader
from .averitec_loader import AVeriTeCLoader
from .multilingual_loader import MultilingualLoader


class UnifiedDataPipeline:
    """
    High-level orchestrator for multi-source dataset ingestion and validation.
    """

    def __init__(self):
        self.registry: Dict[str, List[CanonicalClaim]] = {}
        self.reports: Dict[str, ValidationReport] = {}

    def load_fever(self, filepath: str) -> Tuple[List[CanonicalClaim], ValidationReport]:
        """Loads and validates a FEVER JSONL file."""
        loader = FEVERLoader(filepath)
        claims, report = loader.load_and_validate()
        self.registry["FEVER"] = claims
        self.reports["FEVER"] = report
        return claims, report

    def load_averitec(self, filepath: str) -> Tuple[List[CanonicalClaim], ValidationReport]:
        """Loads and validates an AVeriTeC JSON file."""
        loader = AVeriTeCLoader(filepath)
        claims, report = loader.load_and_validate()
        self.registry["AVERITEC"] = claims
        self.reports["AVERITEC"] = report
        return claims, report

    def load_multilingual(
        self, filepath: str, source_tag: DatasetSource = DatasetSource.MULTILINGUAL_DIAGNOSTIC
    ) -> Tuple[List[CanonicalClaim], ValidationReport]:
        """Loads and validates a multilingual claim JSON file."""
        loader = MultilingualLoader(filepath, source_tag=source_tag)
        claims, report = loader.load_and_validate()
        self.registry[source_tag.value] = claims
        self.reports[source_tag.value] = report
        return claims, report

    def get_combined_dataset(self) -> List[CanonicalClaim]:
        """Combines all currently loaded datasets into a single unified list."""
        combined: List[CanonicalClaim] = []
        for claims in self.registry.values():
            combined.extend(claims)
        return combined

    def generate_global_report(self) -> ValidationReport:
        """Aggregates all individual dataset validation reports into a global summary."""
        global_report = ValidationReport(dataset_name="VERITAS-X Global Unified")
        seen_hashes = set()

        for d_name, claims in self.registry.items():
            for claim in claims:
                global_report.total_records += 1
                chash = claim.compute_hash()
                if chash in seen_hashes:
                    global_report.duplicates_detected += 1
                    continue
                seen_hashes.add(chash)

                global_report.valid_records += 1
                global_report.canonical_distribution[claim.canonical_label.value] = (
                    global_report.canonical_distribution.get(claim.canonical_label.value, 0) + 1
                )
                global_report.language_distribution[claim.language] = (
                    global_report.language_distribution.get(claim.language, 0) + 1
                )

        return global_report

    def export_canonical_jsonl(self, output_path: str) -> int:
        """
        Exports all valid, unified claims across registered datasets to a JSONL file.
        Returns the total number of exported records.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        combined = self.get_combined_dataset()
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for claim in combined:
                line = json.dumps(claim.to_dict(), ensure_ascii=False)
                f.write(line + "\n")
                count += 1
        return count
