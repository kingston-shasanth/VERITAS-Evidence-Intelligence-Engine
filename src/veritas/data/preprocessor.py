"""
Text preprocessing and linguistic sanitization engine for VERITAS-X.
Handles Unicode normalization, multilingual script classification, and date parsing.
"""
import unicodedata
import re
from typing import Tuple, Optional


class TextPreprocessor:
    """
    Sanitizes, normalizes, and classifies multilingual textual inputs.
    """

    # Unicode Character Ranges
    DEVANAGARI_RANGE = (0x0900, 0x097F)
    BENGALI_RANGE = (0x0980, 0x09FF)
    LATIN_RANGE_1 = (0x0041, 0x007A) # Standard Latin ASCII
    LATIN_RANGE_2 = (0x00C0, 0x024F) # Latin-1 Supplement & Extended

    @classmethod
    def normalize_text(cls, text: Optional[str]) -> str:
        """
        Applies Unicode Canonical Decomposition followed by Canonical Composition (NFC).
        Essential for Devanagari and Bengali conjuncts and vowel marks (matras).
        Cleans control characters and trims excess whitespace.
        """
        if text is None:
            return ""
        
        # 1. Unicode NFC normalization
        normalized = unicodedata.normalize("NFC", text)
        
        # 2. Remove non-printable / control characters except standard whitespace
        normalized = "".join(
            ch for ch in normalized 
            if unicodedata.category(ch)[0] != "C" or ch in "\n\t "
        )
        
        # 3. Collapse multiple whitespace and strip edges
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @classmethod
    def detect_script_and_language(cls, text: str) -> Tuple[str, str]:
        """
        Detects primary script and inferred language tag via character frequency.
        Returns (script_name, iso_language_code).
        Supported:
            - 'en' (English / Latin)
            - 'hi' (Hindi / Devanagari)
            - 'bn' (Bengali / Bangla)
            - 'hi-en' (Code-Mixed Hinglish)
        """
        clean_text = cls.normalize_text(text)
        if not clean_text:
            return "unknown", "und"

        devanagari_count = 0
        bengali_count = 0
        latin_count = 0
        total_letters = 0

        for ch in clean_text:
            code = ord(ch)
            if unicodedata.category(ch).startswith("L"): # Letter
                total_letters += 1
                if cls.DEVANAGARI_RANGE[0] <= code <= cls.DEVANAGARI_RANGE[1]:
                    devanagari_count += 1
                elif cls.BENGALI_RANGE[0] <= code <= cls.BENGALI_RANGE[1]:
                    bengali_count += 1
                elif (cls.LATIN_RANGE_1[0] <= code <= cls.LATIN_RANGE_1[1]) or \
                     (cls.LATIN_RANGE_2[0] <= code <= cls.LATIN_RANGE_2[1]):
                    latin_count += 1

        if total_letters == 0:
            return "symbolic", "und"

        deva_ratio = devanagari_count / total_letters
        beng_ratio = bengali_count / total_letters
        lat_ratio = latin_count / total_letters

        # Dominant script decision logic
        if beng_ratio > 0.4:
            return "Bengali", "bn"
        if deva_ratio > 0.4:
            return "Devanagari", "hi"
        
        # Heuristic for Code-Mixed Hinglish:
        # Latin script text with common Hinglish phonetic marker words
        if lat_ratio > 0.6:
            hinglish_markers = {
                "hai", "nahi", "aur", "mein", "yeh", "kya", "tha", "raha", 
                "apne", "wale", "karein", "bhi", "par", "hoga", "shuru", "desh",
                "police", "sarkar", "yojana", "modi", "khabar", "sach", "jhooth"
            }
            words = set(re.findall(r"\b[a-zA-Z]{2,}\b", clean_text.lower()))
            matches = words.intersection(hinglish_markers)
            if len(matches) >= 2 or (len(words) > 0 and len(matches) / len(words) >= 0.2):
                return "Latin-CodeMixed", "hi-en"
            return "Latin", "en"

        return "Mixed", "und"

    @classmethod
    def parse_date_to_iso(cls, date_str: Optional[str]) -> Optional[str]:
        """
        Converts non-standard date strings (e.g., '31-10-2020', '2020/10/31')
        into ISO 8601 YYYY-MM-DD. Returns None if unparseable.
        """
        if not date_str:
            return None
        date_str = date_str.strip()
        
        # Match DD-MM-YYYY or DD/MM/YYYY
        m1 = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", date_str)
        if m1:
            d, m, y = m1.groups()
            return f"{y}-{int(m):02d}-{int(d):02d}"

        # Match YYYY-MM-DD or YYYY/MM/DD
        m2 = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", date_str)
        if m2:
            y, m, d = m2.groups()
            return f"{y}-{int(m):02d}-{int(d):02d}"

        return None
