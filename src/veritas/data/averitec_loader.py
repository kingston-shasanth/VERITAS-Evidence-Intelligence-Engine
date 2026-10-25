"""
Dedicated loader and parser for AVeriTeC (Automated Verification of Textual Claims).
Paper: Schlichtkrull et al., NeurIPS 2023 (https://openreview.net/forum?id=7hJg6b1Q0q)
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


class AVeriTeCLoader:
    """
    Parses and harmonizes official AVeriTeC JSON splits.
    """

    LABEL_MAP = {
        "Supported": CanonicalVeracity.VERIFIED_TRUE,
        "Refuted": CanonicalVeracity.VERIFIED_FALSE,
        "Conflicting Evidence/Cherrypicking": CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED,
        "Not Enough Evidence": CanonicalVeracity.INSUFFICIENT_EVIDENCE,
    }

    def __init__(self, filepath: str):
        self.filepath = filepath

    def stream_canonical(self) -> Iterator[Tuple[CanonicalClaim, bool, str]]:
        """
        Yields (CanonicalClaim, is_valid, error_msg) for each record in the AVeriTeC JSON array.
        """
        with open(self.filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                yield (None, False, f"Failed to load AVeriTeC JSON file: {str(e)}")
                return

        if not isinstance(data, list):
            yield (None, False, f"Expected root JSON list in AVeriTeC, got {type(data)}")
            return

        for idx, record in enumerate(data):
            if not isinstance(record, dict):
                yield (None, False, f"Record {idx}: Not a valid JSON object")
                continue

            claim_text = record.get("claim", "")
            raw_label = record.get("label", "")

            if not claim_text or not raw_label:
                yield (None, False, f"Record {idx}: Missing claim text or label")
                continue

            if raw_label not in self.LABEL_MAP:
                yield (None, False, f"Record {idx}: Unknown AVeriTeC label '{raw_label}'")
                continue

            normalized_claim = TextPreprocessor.normalize_text(claim_text)
            canonical_label = self.LABEL_MAP[raw_label]
            iso_date = TextPreprocessor.parse_date_to_iso(record.get("claim_date"))

            # Extract Evidence from intermediate questions and answers
            evidence_items: List[EvidenceItem] = []
            questions = record.get("questions", [])
            for q_idx, q_obj in enumerate(questions):
                q_text = q_obj.get("question", "")
                for a_idx, ans in enumerate(q_obj.get("answers", [])):
                    ans_text = ans.get("answer")
                    source_url = ans.get("source_url") or ans.get("cached_source_url")
                    ans_type = (ans.get("answer_type") or "").lower()

                    ev_type = EvidenceType.WEB_EXTRACTIVE
                    if "abstractive" in ans_type:
                        ev_type = EvidenceType.WEB_ABSTRACTIVE

                    evidence_items.append(
                        EvidenceItem(
                            text=ans_text,
                            source_url=source_url,
                            source_id=f"q{q_idx}_a{a_idx}",
                            evidence_type=ev_type,
                            metadata={
                                "question": q_text,
                                "answer_type": ans.get("answer_type"),
                                "source_medium": ans.get("source_medium"),
                                "cached_url": ans.get("cached_source_url"),
                            },
                        )
                    )

            claim_obj = CanonicalClaim(
                claim_id=f"averitec_{idx}",
                claim_text=normalized_claim,
                source_dataset=DatasetSource.AVERITEC,
                raw_label=raw_label,
                canonical_label=canonical_label,
                language="en",
                claim_date=iso_date,
                speaker=record.get("speaker"),
                evidence=evidence_items,
                metadata={
                    "justification": record.get("justification"),
                    "fact_checking_article": record.get("fact_checking_article"),
                    "reporting_source": record.get("reporting_source"),
                    "location_ISO_code": record.get("location_ISO_code"),
                    "claim_types": record.get("claim_types", []),
                },
            )
            yield (claim_obj, True, "")

    def load_and_validate(self) -> Tuple[List[CanonicalClaim], ValidationReport]:
        """
        Loads all records, enforces deduplication, and generates a detailed ValidationReport.
        """
        claims: List[CanonicalClaim] = []
        report = ValidationReport(dataset_name="AVeriTeC")
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
