"""
Dedicated loader and parser for Multilingual and Cross-Lingual Claim datasets
(e.g., IndicClaimBuster format and diagnostic multilingual splits).
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


class MultilingualLoader:
    """
    Parses and harmonizes multilingual claims across Devanagari, Bengali, and Code-Mixed text.
    """

    LABEL_MAP = {
        "SUPPORTS": CanonicalVeracity.VERIFIED_TRUE,
        "SUPPORTED": CanonicalVeracity.VERIFIED_TRUE,
        "TRUE": CanonicalVeracity.VERIFIED_TRUE,
        "REFUTES": CanonicalVeracity.VERIFIED_FALSE,
        "REFUTED": CanonicalVeracity.VERIFIED_FALSE,
        "FALSE": CanonicalVeracity.VERIFIED_FALSE,
        "NOT ENOUGH INFO": CanonicalVeracity.INSUFFICIENT_EVIDENCE,
        "NEI": CanonicalVeracity.INSUFFICIENT_EVIDENCE,
        "UNVERIFIED": CanonicalVeracity.INSUFFICIENT_EVIDENCE,
        "CONFLICTING": CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED,
    }

    def __init__(self, filepath: str, source_tag: DatasetSource = DatasetSource.MULTILINGUAL_DIAGNOSTIC):
        self.filepath = filepath
        self.source_tag = source_tag

    def stream_canonical(self) -> Iterator[Tuple[CanonicalClaim, bool, str]]:
        with open(self.filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                yield (None, False, f"Failed to parse JSON file {self.filepath}: {str(e)}")
                return

        if not isinstance(data, list):
            yield (None, False, f"Expected list of claims in {self.filepath}, got {type(data)}")
            return

        for idx, record in enumerate(data):
            if not isinstance(record, dict):
                yield (None, False, f"Record {idx}: Not a valid JSON object")
                continue

            claim_text = record.get("claim", "")
            raw_label = (record.get("label") or "").upper().strip()

            if not claim_text or not raw_label:
                yield (None, False, f"Record {idx}: Missing claim or label")
                continue

            if raw_label not in self.LABEL_MAP:
                yield (None, False, f"Record {idx}: Unknown label '{raw_label}'")
                continue

            normalized_claim = TextPreprocessor.normalize_text(claim_text)
            canonical_label = self.LABEL_MAP[raw_label]

            # Detect or verify script & language
            script, detected_lang = TextPreprocessor.detect_script_and_language(normalized_claim)
            annotated_lang = record.get("language")
            # If annotated is provided and not unknown, use it; otherwise use detected
            final_lang = annotated_lang if annotated_lang else detected_lang

            # Extract evidence
            evidence_items: List[EvidenceItem] = []
            raw_evidence = record.get("evidence", [])
            if isinstance(raw_evidence, list):
                for e_idx, ev in enumerate(raw_evidence):
                    if isinstance(ev, str):
                        evidence_items.append(
                            EvidenceItem(
                                text=TextPreprocessor.normalize_text(ev),
                                source_id=f"ev_{e_idx}",
                                evidence_type=EvidenceType.NEWS_PASSAGE,
                            )
                        )
                    elif isinstance(ev, dict):
                        evidence_items.append(
                            EvidenceItem(
                                text=TextPreprocessor.normalize_text(ev.get("text")),
                                source_url=ev.get("source_url"),
                                source_id=ev.get("source_id", f"ev_{e_idx}"),
                                evidence_type=EvidenceType.NEWS_PASSAGE,
                                metadata=ev.get("metadata", {}),
                            )
                        )

            claim_id = record.get("id", f"{self.source_tag.value.lower()}_{idx}")
            claim_obj = CanonicalClaim(
                claim_id=str(claim_id),
                claim_text=normalized_claim,
                source_dataset=self.source_tag,
                raw_label=record.get("label"),
                canonical_label=canonical_label,
                language=final_lang,
                claim_date=record.get("claim_date"),
                speaker=record.get("speaker"),
                evidence=evidence_items,
                metadata={
                    "domain": record.get("domain", "general"),
                    "detected_script": script,
                    "annotated_lang": annotated_lang,
                },
            )
            yield (claim_obj, True, "")

    def load_and_validate(self) -> Tuple[List[CanonicalClaim], ValidationReport]:
        claims: List[CanonicalClaim] = []
        report = ValidationReport(dataset_name=self.source_tag.value)
        seen_hashes = set()

        for claim, is_valid, err in self.stream_canonical():
            report.total_records += 1
            if not is_valid:
                report.invalid_records += 1
                report.errors.append(err)
                continue

            chash = claim.compute_hash()
            if chash in seen_hashes:
                report.duplicates_detected += 1
                continue
            seen_hashes.add(chash)

            report.valid_records += 1
            report.label_distribution[claim.raw_label] = report.label_distribution.get(claim.raw_label, 0) + 1
            report.canonical_distribution[claim.canonical_label.value] = (
                report.canonical_distribution.get(claim.canonical_label.value, 0) + 1
            )
            report.language_distribution[claim.language] = report.language_distribution.get(claim.language, 0) + 1
            claims.append(claim)

        return claims, report
