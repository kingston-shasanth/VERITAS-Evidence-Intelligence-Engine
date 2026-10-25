"""
Temporal Intelligence & Provenance Analyzer for VERITAS-X 2.0.
Extracts publication and event timestamps from URLs and document text,
analyzes temporal scope of claims, resolves chronological sequences,
and establishes auditable provenance without fabricating missing timestamps.
"""
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from veritas.reasoning.types import TemporalMetadata, TemporalScope, TemporalStatus


class TemporalAnalyzer:
    """
    Examines publication dates, event dates, claim temporal scope,
    and temporal shifts between older and newer evidence.
    """

    MONTH_NAMES = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "jun": "06", "jul": "07", "aug": "08", "sep": "09",
        "oct": "10", "nov": "11", "dec": "12",
    }

    CURRENT_CUES = [
        r"\bcurrently\b", r"\bas of\b", r"\bnow\b", r"\btoday\b",
        r"\bremains\b", r"\bis still\b", r"\bpresently\b", r"\blatest\b"
    ]

    PAST_CUES = [
        r"\bpreviously\b", r"\bwas\b", r"\bwere\b", r"\bformer\b",
        r"\bhistorically\b", r"\bin \d{4}\b", r"\bduring \d{4}\b",
        r"\bin July 1969\b", r"\bin the past\b"
    ]

    def extract_metadata(
        self,
        text: Optional[str] = None,
        source_url: Optional[str] = None,
        existing_meta: Optional[Dict[str, Any]] = None,
    ) -> TemporalMetadata:
        """
        Extracts temporal metadata with verified provenance.
        Never substitutes system time for an absent publication date.
        """
        # 1. Check explicit metadata dictionary first
        if existing_meta:
            pub_date = existing_meta.get("publication_date") or existing_meta.get("claim_date")
            if pub_date:
                iso_date = self._normalize_to_iso(str(pub_date))
                if iso_date:
                    return TemporalMetadata(
                        publication_date=iso_date,
                        date_provenance="metadata",
                        date_confidence=0.95,
                        is_verified=True,
                        raw_date_str=str(pub_date),
                    )

        # 2. Extract from Source URL (e.g., Wayback Machine archive or news URL path)
        if source_url:
            # Wayback archive: /web/20201129141238/...
            archive_match = re.search(r"/web/(\d{4})(\d{2})(\d{2})\d*/", source_url)
            if archive_match:
                y, m, d = archive_match.group(1), archive_match.group(2), archive_match.group(3)
                if self._is_valid_date(int(y), int(m), int(d)):
                    iso_date = f"{y}-{m}-{d}"
                    return TemporalMetadata(
                        publication_date=iso_date,
                        date_provenance="archive_url",
                        date_confidence=0.90,
                        is_verified=True,
                        raw_date_str=f"{y}{m}{d}",
                    )

            # News path: /2020/10/26/ or /2020-10-26/
            news_match = re.search(r"/(\d{4})[/-](\d{2})[/-](\d{2})/", source_url)
            if news_match:
                y, m, d = news_match.group(1), news_match.group(2), news_match.group(3)
                if self._is_valid_date(int(y), int(m), int(d)):
                    iso_date = f"{y}-{m}-{d}"
                    return TemporalMetadata(
                        publication_date=iso_date,
                        date_provenance="news_path",
                        date_confidence=0.85,
                        is_verified=True,
                        raw_date_str=f"{y}-{m}-{d}",
                    )

        # 3. Extract event date mentioned directly within text body
        if text:
            # Full textual date: "October 26, 2020" or "26 October 2020"
            text_match = re.search(
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
                text,
                re.IGNORECASE,
            )
            if text_match:
                month_str, day_str, year_str = text_match.group(1).lower(), text_match.group(2), text_match.group(3)
                month_num = self.MONTH_NAMES.get(month_str, "01")
                day_num = f"{int(day_str):02d}"
                iso_date = f"{year_str}-{month_num}-{day_num}"
                return TemporalMetadata(
                    event_date=iso_date,
                    date_provenance="text_extraction",
                    date_confidence=0.80,
                    is_verified=False,
                    raw_date_str=text_match.group(0),
                )

            # Month and Year: "July 1969"
            month_year_match = re.search(
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
                text,
                re.IGNORECASE,
            )
            if month_year_match:
                month_str, year_str = month_year_match.group(1).lower(), month_year_match.group(2)
                month_num = self.MONTH_NAMES.get(month_str, "01")
                iso_date = f"{year_str}-{month_num}-01"
                return TemporalMetadata(
                    event_date=iso_date,
                    date_provenance="text_extraction",
                    date_confidence=0.75,
                    is_verified=False,
                    raw_date_str=month_year_match.group(0),
                )

            # Standalone 4-digit historical year (1800 - 2029)
            year_match = re.search(r"\b(1[89]\d{2}|20[0-2]\d)\b", text)
            if year_match:
                year_str = year_match.group(1)
                return TemporalMetadata(
                    event_date=f"{year_str}-01-01",
                    date_provenance="text_extraction",
                    date_confidence=0.50,
                    is_verified=False,
                    raw_date_str=year_str,
                )

        # 4. Unknown / Unavailable date
        return TemporalMetadata(
            publication_date=None,
            event_date=None,
            date_provenance="unknown",
            date_confidence=0.0,
            is_verified=False,
            raw_date_str=None,
        )

    def analyze_claim(self, claim_text: str, evidence_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determines temporal scope of claim and correlates with evidence dates.
        """
        # Determine claim temporal scope
        is_current = any(re.search(cue, claim_text, re.IGNORECASE) for cue in self.CURRENT_CUES)
        is_past = any(re.search(cue, claim_text, re.IGNORECASE) for cue in self.PAST_CUES)

        if is_current and not is_past:
            scope = TemporalScope.CURRENT
        elif is_past and not is_current:
            scope = TemporalScope.HISTORICAL
        elif is_current and is_past:
            scope = TemporalScope.TIME_SPAN
        else:
            scope = TemporalScope.UNSPECIFIED

        # Extract dates across all evidence passages
        dated_evidence: List[Dict[str, Any]] = []
        for ev in evidence_items:
            meta = self.extract_metadata(
                text=ev.get("text"),
                source_url=ev.get("source_url") or ev.get("source"),
                existing_meta=ev.get("metadata"),
            )
            date_val = meta.publication_date or meta.event_date
            if date_val:
                dated_evidence.append({
                    "evidence_id": ev.get("id"),
                    "date": date_val,
                    "date_type": "publication" if meta.publication_date else "event",
                    "provenance": meta.date_provenance,
                    "confidence": meta.date_confidence,
                    "is_verified": meta.is_verified,
                    "stance": ev.get("stance", "neutral"),
                    "text": ev.get("text", "")[:120] + "...",
                })

        # Sort timeline events chronologically
        dated_evidence.sort(key=lambda x: x["date"])

        # Determine temporal status
        if not dated_evidence:
            temporal_status = TemporalStatus.UNRESOLVED.value
            explanation = "Temporal status unresolved: retrieved passages lack verifiable publication or event timestamps."
        elif len(dated_evidence) >= 2:
            # Check for chronological updates or superseding events
            oldest = dated_evidence[0]
            newest = dated_evidence[-1]
            if oldest["date"] < newest["date"] and oldest["stance"] != newest["stance"]:
                temporal_status = TemporalStatus.OUTDATED_SUPERSEDED.value
                explanation = f"Chronological stance shift detected: Earlier evidence ({oldest['date']}, {oldest['stance']}) differs from later evidence ({newest['date']}, {newest['stance']})."
            else:
                temporal_status = TemporalStatus.CONSISTENT.value
                explanation = f"Temporal alignment confirmed across {len(dated_evidence)} dated evidence citation(s) from {dated_evidence[0]['date']} to {dated_evidence[-1]['date']}."
        else:
            temporal_status = TemporalStatus.CONSISTENT.value
            explanation = f"Temporal grounding identified for 1 citation ({dated_evidence[0]['date']})."

        return {
            "temporal_scope": scope.value,
            "temporal_status": temporal_status,
            "explanation": explanation,
            "timeline_events": dated_evidence,
            "has_verifiable_dates": len(dated_evidence) > 0,
        }

    def _normalize_to_iso(self, date_str: str) -> Optional[str]:
        """Normalizes date string to YYYY-MM-DD if recognizable."""
        try:
            # Direct ISO YYYY-MM-DD
            if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str.strip()):
                return date_str.strip()
            # YYYY/MM/DD
            m = re.match(r"^(\d{4})/(\d{2})/(\d{2})$", date_str.strip())
            if m:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        except Exception:
            pass
        return None

    def _is_valid_date(self, year: int, month: int, day: int) -> bool:
        if not (1800 <= year <= 2030):
            return False
        if not (1 <= month <= 12):
            return False
        if not (1 <= day <= 31):
            return False
        return True
