"""
Evidence Corpus Store for VERITAS-X.
Manages a local, self-contained, reproducible collection of evidence passages.
Guarantees that only verified evidence text (not references or URLs) is indexed.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator
import json
import os
import hashlib


@dataclass
class EvidenceDocument:
    """Represents an indexed evidence passage in the local corpus."""
    doc_id: str
    text: str
    source_dataset: str
    source_url: Optional[str] = None
    claim_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        norm = " ".join(self.text.strip().lower().split())
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source_dataset": self.source_dataset,
            "source_url": self.source_url,
            "claim_id": self.claim_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceDocument":
        return cls(
            doc_id=data["doc_id"],
            text=data["text"],
            source_dataset=data["source_dataset"],
            source_url=data.get("source_url"),
            claim_id=data.get("claim_id"),
            metadata=data.get("metadata", {}),
        )


class EvidenceCorpus:
    """
    Stores and indexes evidence passages for reproducible sparse & dense retrieval.
    """

    def __init__(self):
        self.documents: Dict[str, EvidenceDocument] = {}
        self.claim_to_doc_ids: Dict[str, List[str]] = {}

    def add_document(self, doc: EvidenceDocument) -> bool:
        """
        Adds a document if text is valid and non-empty.
        Returns True if added, False if skipped (empty or duplicate).
        """
        if not doc.text or not doc.text.strip():
            return False

        self.documents[doc.doc_id] = doc
        if doc.claim_id:
            if doc.claim_id not in self.claim_to_doc_ids:
                self.claim_to_doc_ids[doc.claim_id] = []
            self.claim_to_doc_ids[doc.claim_id].append(doc.doc_id)
        return True

    def get_document(self, doc_id: str) -> Optional[EvidenceDocument]:
        return self.documents.get(doc_id)

    def get_ground_truth_doc_ids(self, claim_id: str) -> List[str]:
        return self.claim_to_doc_ids.get(claim_id, [])

    def get_all_documents(self) -> List[EvidenceDocument]:
        return list(self.documents.values())

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[EvidenceDocument]:
        return iter(self.documents.values())

    @classmethod
    def from_canonical_claims(cls, claims: List[Any]) -> "EvidenceCorpus":
        """
        Extracts verified evidence texts from a list of CanonicalClaim objects.
        Explicitly rejects items that only contain URLs or IDs without textual context.
        """
        corpus = cls()
        skipped_no_text = 0
        added_count = 0
        seen_hashes = set()

        for claim in claims:
            if isinstance(claim, dict):
                cid = claim.get("claim_id")
                source_ds = claim.get("source_dataset")
                evidence_items = claim.get("evidence", [])
            else:
                cid = getattr(claim, "claim_id", None)
                source_ds = getattr(claim, "source_dataset", None)
                evidence_items = getattr(claim, "evidence", [])

            if hasattr(source_ds, "value"):
                source_ds = source_ds.value

            for e_idx, ev in enumerate(evidence_items):
                if isinstance(ev, dict):
                    ev_text = ev.get("text")
                    ev_url = ev.get("source_url")
                    ev_meta = ev.get("metadata", {})
                else:
                    ev_text = getattr(ev, "text", None)
                    ev_url = getattr(ev, "source_url", None)
                    ev_meta = getattr(ev, "metadata", {})

                # Strict validation: Require non-empty text string
                if not ev_text or not isinstance(ev_text, str) or not ev_text.strip():
                    skipped_no_text += 1
                    continue

                clean_text = ev_text.strip()
                thash = hashlib.sha256(clean_text.lower().encode("utf-8")).hexdigest()
                if thash in seen_hashes:
                    continue
                seen_hashes.add(thash)

                doc = EvidenceDocument(
                    doc_id=f"doc_{cid}_{e_idx}",
                    text=clean_text,
                    source_dataset=str(source_ds),
                    source_url=ev_url,
                    claim_id=str(cid),
                    metadata=ev_meta,
                )
                corpus.add_document(doc)
                added_count += 1

        return corpus

    def save_to_jsonl(self, filepath: str) -> int:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        count = 0
        with open(filepath, "w", encoding="utf-8") as f:
            for doc in self.documents.values():
                line = json.dumps(doc.to_dict(), ensure_ascii=False)
                f.write(line + "\n")
                count += 1
        return count

    @classmethod
    def load_from_jsonl(cls, filepath: str) -> "EvidenceCorpus":
        corpus = cls()
        corpus.filepath = os.path.abspath(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                doc = EvidenceDocument.from_dict(data)
                corpus.add_document(doc)
        return corpus
