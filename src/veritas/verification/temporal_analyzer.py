"""
Temporal Consistency & Anachronism Analyzer for VERITAS-X.
Analyzes timestamps between claim events and evidence publication dates.
Treats temporal discrepancies as forensic warnings, not automatic proof of falsity.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import re

from veritas.data.preprocessor import TextPreprocessor


class TemporalStatus(str, Enum):
    """Categorical classification of temporal alignment."""
    ALIGNED = "ALIGNED"
    ACCEPTABLE_DRIFT = "ACCEPTABLE_DRIFT"
    ANACHRONISM_WARNING = "ANACHRONISM_WARNING"
    POST_DATED_WARNING = "POST_DATED_WARNING"
    UNDATED = "UNDATED"


@dataclass
class TemporalReport:
    """Forensic report detailing temporal alignment and warnings."""
    status: TemporalStatus
    delta_days: Optional[int]
    claim_date: Optional[str]
    evidence_date: Optional[str]
    warning_message: Optional[str]
    confidence_penalty: float # Small penalty factor [0.0, 0.10] softening overconfidence


class TemporalAnalyzer:
    """
    Evaluates temporal discrepancies between claims and cited evidence.
    """

    WAYBACK_DATE_REGEX = re.compile(r"/web/(\d{4})(\d{2})(\d{2})")
    ISO_DATE_REGEX = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

    @classmethod
    def _parse_iso(cls, date_str: Optional[str]) -> Optional[date]:
        if not date_str:
            return None
        iso_str = TextPreprocessor.parse_date_to_iso(date_str)
        if not iso_str:
            return None
        try:
            return datetime.strptime(iso_str, "%Y-%m-%d").date()
        except Exception:
            return None

    @classmethod
    def _extract_evidence_date(cls, evidence_item: Dict[str, Any]) -> Optional[date]:
        """
        Attempts to extract publication date from metadata, explicit fields,
        or archived Wayback Machine URLs (e.g. /web/20201130144023/).
        """
        # 1. Check explicit metadata
        meta = evidence_item.get("metadata", {})
        if "date" in meta and meta["date"]:
            d = cls._parse_iso(meta["date"])
            if d:
                return d

        # 2. Check source_url / cached_url for Wayback Machine pattern
        urls = [evidence_item.get("source_url"), meta.get("cached_url")]
        for u in urls:
            if not u:
                continue
            match = cls.WAYBACK_DATE_REGEX.search(u)
            if match:
                y, m, day = match.groups()
                try:
                    return date(int(y), int(m), int(day))
                except Exception:
                    pass

        # 3. Check ISO pattern in text
        text = evidence_item.get("text", "")
        if text:
            m = cls.ISO_DATE_REGEX.search(text)
            if m:
                y, m, day = m.groups()
                try:
                    return date(int(y), int(m), int(day))
                except Exception:
                    pass

        return None

    def analyze(
        self,
        claim_date_str: Optional[str],
        evidence_items: List[Dict[str, Any]],
    ) -> TemporalReport:
        """
        Computes temporal latency and flags potential anachronisms.
        """
        claim_d = self._parse_iso(claim_date_str)
        if not claim_d:
            return TemporalReport(
                status=TemporalStatus.UNDATED,
                delta_days=None,
                claim_date=None,
                evidence_date=None,
                warning_message=None,
                confidence_penalty=0.0,
            )

        # Find earliest and latest evidence dates
        ev_dates: List[date] = []
        for ev in evidence_items:
            ed = self._extract_evidence_date(ev)
            if ed:
                ev_dates.append(ed)

        if not ev_dates:
            return TemporalReport(
                status=TemporalStatus.UNDATED,
                delta_days=None,
                claim_date=claim_d.isoformat(),
                evidence_date=None,
                warning_message=None,
                confidence_penalty=0.0,
            )

        # Compare against closest evidence date
        closest_ed = min(ev_dates, key=lambda d: abs((d - claim_d).days))
        delta = (closest_ed - claim_d).days
        abs_delta = abs(delta)

        # Categorize
        if abs_delta <= 180: # Within 6 months
            return TemporalReport(
                status=TemporalStatus.ALIGNED,
                delta_days=delta,
                claim_date=claim_d.isoformat(),
                evidence_date=closest_ed.isoformat(),
                warning_message=None,
                confidence_penalty=0.0,
            )

        elif abs_delta <= 365: # Within 1 year
            return TemporalReport(
                status=TemporalStatus.ACCEPTABLE_DRIFT,
                delta_days=delta,
                claim_date=claim_d.isoformat(),
                evidence_date=closest_ed.isoformat(),
                warning_message=f"Moderate temporal gap ({abs_delta} days) between claim and evidence.",
                confidence_penalty=0.02,
            )

        elif delta < -365: # Evidence is much older than the claim
            years = abs_delta / 365.25
            return TemporalReport(
                status=TemporalStatus.ANACHRONISM_WARNING,
                delta_days=delta,
                claim_date=claim_d.isoformat(),
                evidence_date=closest_ed.isoformat(),
                warning_message=(
                    f"Temporal Anachronism Warning: Evidence was published {years:.1f} years BEFORE "
                    f"the asserted claim date. Facts may have changed in the interim."
                ),
                confidence_penalty=0.08,
            )

        else: # Evidence was published much later than the claim
            years = abs_delta / 365.25
            return TemporalReport(
                status=TemporalStatus.POST_DATED_WARNING,
                delta_days=delta,
                claim_date=claim_d.isoformat(),
                evidence_date=closest_ed.isoformat(),
                warning_message=(
                    f"Post-Dated Warning: Evidence was published {years:.1f} years AFTER the alleged event date."
                ),
                confidence_penalty=0.05,
            )
