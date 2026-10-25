"""
Multilingual Tokenizer for VERITAS-X Retrieval.
Preserves Unicode scripts (Devanagari, Bengali, Latin), exact dates, percentages,
and numerical forensic markers without losing linguistic or factual fidelity.
"""
import re
import unicodedata
from typing import List, Optional


class MultilingualTokenizer:
    """
    Tokenizes text for BM25 and sparse retrieval across English, Hindi, Bengali,
    and Code-Mixed Hinglish.
    """

    # Comprehensive regex capturing:
    # 1. Dates: YYYY-MM-DD, DD-MM-YYYY, YYYY/MM/DD, DD/MM/YYYY
    # 2. Percentages & Currencies: 10%, $500, ₹2000
    # 3. Numeric tokens: 3.14, 2024, 100,000
    # 4. Devanagari words: [\u0900-\u097F]+
    # 5. Bengali words: [\u0980-\u09FF]+
    # 6. Latin alphanumeric words with internal hyphens/apostrophes: e.g. COVID-19, don't
    TOKEN_PATTERN = re.compile(
        r"(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4})|"           # Dates
        r"(?:[₹$€£]\s*\d+(?:,\d+)*(?:\.\d+)?)|"         # Currencies
        r"(?:\d+(?:\.\d+)?%)|"                           # Percentages
        r"(?:\d+(?:,\d+)*(?:\.\d+)?)|"                  # Standard numbers
        r"(?:[\u0900-\u097F]+)|"                        # Devanagari words
        r"(?:[\u0980-\u09FF]+)|"                        # Bengali words
        r"(?:[a-zA-Z0-9]+(?:[-_'][a-zA-Z0-9]+)*)",      # Latin & Alphanumeric tokens
        re.UNICODE
    )

    @classmethod
    def tokenize(cls, text: Optional[str]) -> List[str]:
        """
        Tokenizes text into a normalized list of sparse retrieval tokens.
        Preserves Indic scripts in native NFC form; lowercases Latin tokens.
        """
        if not text:
            return []

        # 1. Canonical Unicode NFC decomposition & composition
        normalized = unicodedata.normalize("NFC", text)

        # 2. Extract structured tokens
        raw_tokens = cls.TOKEN_PATTERN.findall(normalized)

        tokens: List[str] = []
        for t in raw_tokens:
            t = t.strip()
            if not t:
                continue
            
            # If Latin script, lowercase for case-insensitive matching
            # Indic characters (Devanagari / Bengali) have no case concept
            first_char = t[0]
            if ("a" <= first_char <= "z") or ("A" <= first_char <= "Z") or first_char in "$₹€£":
                tokens.append(t.lower())
            else:
                tokens.append(t)

        return tokens
