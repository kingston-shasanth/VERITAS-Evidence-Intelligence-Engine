"""
Dedicated loader and parser for FEVER (Fact Extraction and VERification) datasets.
Paper: Thorne et al., NAACL 2018 (https://aclanthology.org/N18-1074/)
"""
import json
from typing import Iterator, List, Tuple
from .schema import (
    CanonicalClaim,
    CanonicalVeracity,
    DatasetSource,
    EvidenceItem,
    EvidenceType,
    ValidationReport,
)
from .preprocessor import TextPreprocessor


class FEVERLoader:
    """
    Parses and harmonizes official FEVER JSONL splits.
    """

    LABEL_MAP = {
        "SUPPORTS": CanonicalVeracity.VERIFIED_TRUE,
        "REFUTES": CanonicalVeracity.VERIFIED_FALSE,
        "NOT ENOUGH INFO": CanonicalVeracity.INSUFFICIENT_EVIDENCE,
    }

    def __init__(self, filepath: str):
        self.filepath = filepath

    def stream_canonical(self) -> Iterator[Tuple[CanonicalClaim, bool, str]]:
        """
        Yields (CanonicalClaim, is_valid, error_msg) for each line in the JSONL file.
        """
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception as e:
                    yield (None, False, f"Line {line_no}: JSON decode error: {str(e)}")
                    continue

                # Schema checks
                claim_id = str(record.get("id", ""))
                claim_text = record.get("claim", "")
                raw_label = record.get("label", "")

                if not claim_id or not claim_text or not raw_label:
                    yield (None, False, f"Line {line_no}: Missing required fields (id, claim, or label)")
                    continue

                if raw_label not in self.LABEL_MAP:
                    yield (None, False, f"Line {line_no}: Unknown FEVER label '{raw_label}'")
                    continue

                normalized_claim = TextPreprocessor.normalize_text(claim_text)
                canonical_label = self.LABEL_MAP[raw_label]

                # Parse FEVER evidence structure
                # Format: list of evidence sets -> [[annotation_id, evidence_id, wiki_title, sentence_id]]
                evidence_items: List[EvidenceItem] = []
                raw_evidence_sets = record.get("evidence", [])
                
                for ev_set in raw_evidence_sets:
                    for ev_tuple in ev_set:
                        if isinstance(ev_tuple, (list, tuple)) and len(ev_tuple) >= 4:
                            _, _, wiki_title, sentence_id = ev_tuple[:4]
                            if wiki_title is not None and sentence_id is not None:
                                evidence_items.append(
                                    EvidenceItem(
                                        text=None, # FEVER claims store pointers; text retrieved from wiki corpus
                                        source_id=f"{wiki_title}#s_{sentence_id}",
                                        source_url=f"https://en.wikipedia.org/wiki/{wiki_title}",
                                        evidence_type=EvidenceType.WIKIPEDIA_SENTENCE,
                                        metadata={
                                            "wiki_title": wiki_title,
                                            "sentence_id": sentence_id,
                                        },
                                    )
                                )

                claim_obj = CanonicalClaim(
                    claim_id=f"fever_{claim_id}",
                    claim_text=normalized_claim,
                    source_dataset=DatasetSource.FEVER,
                    raw_label=raw_label,
                    canonical_label=canonical_label,
                    language="en",
                    evidence=evidence_items,
                    metadata={
                        "verifiable": record.get("verifiable"),
                        "raw_fever_id": record.get("id"),
                    },
                )
                yield (claim_obj, True, "")

    def load_and_validate(self) -> Tuple[List[CanonicalClaim], ValidationReport]:
        """
        Loads all records, enforces deduplication, and generates a detailed ValidationReport.
        """
        claims: List[CanonicalClaim] = []
        report = ValidationReport(dataset_name="FEVER")
        seen_hashes = set()

        for claim, is_valid, err in self.stream_canonical():
            report.total_records += 1
            if not is_valid:
                report.invalid_records += 1
                report.errors.append(err)
                continue

            # Deduplication by text hash
            chash = claim.compute_hash()
            if chash in seen_hashes:
                report.duplicates_detected += 1
                continue
            seen_hashes.add(chash)

            # Record stats
            report.valid_records += 1
            report.label_distribution[claim.raw_label] = report.label_distribution.get(claim.raw_label, 0) + 1
            report.canonical_distribution[claim.canonical_label.value] = (
                report.canonical_distribution.get(claim.canonical_label.value, 0) + 1
            )
            report.language_distribution[claim.language] = report.language_distribution.get(claim.language, 0) + 1
            claims.append(claim)

        return claims, report
