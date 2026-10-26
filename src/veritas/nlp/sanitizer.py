"""
Text sanitization and safe Unicode normalization module for VERITAS-X.
Preserves raw user input immutably while performing safe Unicode normalization (NFC),
typographic variant unification, whitespace cleaning, and conservative repeated-character reduction.

Architectural Design Rationale:
- Uses Canonical Decomposition + Canonical Composition (NFC, unicodedata.normalize("NFC"))
  rather than Compatibility Decomposition (NFKC). NFKC is lossy for mathematical superscripts
  (e.g., x², m³), fractions (½), chemical notation, and formatting ligatures. NFC guarantees
  visual and semantic character equivalence without altering numbers, formulas, or units.
- Preserves all negation tokens ('not', 'never', 'no', 'without', 'cannot'), numbers, dates,
  units, acronyms, and question marks. Never changes a question into an assertion.
"""
from dataclasses import dataclass, field
import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class SanitizationTransformation:
    """Auditable record of an individual sanitization transformation."""
    rule: str
    original: str
    replacement: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "original": self.original,
            "replacement": self.replacement,
            "description": self.description,
        }


@dataclass
class SanitizationResult:
    """Immutable result envelope for text sanitization."""
    raw_text: str
    sanitized_text: str
    transformations: List[SanitizationTransformation] = field(default_factory=list)
    is_question: bool = False
    has_negation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "sanitized_text": self.sanitized_text,
            "transformations": [t.to_dict() for t in self.transformations],
            "is_question": self.is_question,
            "has_negation": self.has_negation,
        }


class TextSanitizer:
    """
    Production text sanitizer for VERITAS-X.
    Performs deterministic, non-lossy linguistic sanitization while tracking audit records.
    """

    # Typographic quote mapping: converts curly and angled quotes to ASCII equivalents
    TYPOGRAPHIC_QUOTES = {
        # Double quotes
        '\u201C': '"',  # Left double quotation mark
        '\u201D': '"',  # Right double quotation mark
        '\u201E': '"',  # Double low-9 quotation mark
        '\u201F': '"',  # Double high-reversed-9 quotation mark
        '\u00AB': '"',  # Left-pointing double angle quotation mark
        '\u00BB': '"',  # Right-pointing double angle quotation mark
        '\u300C': '"',  # Left corner bracket (CJK)
        '\u300D': '"',  # Right corner bracket (CJK)
        # Single quotes / apostrophes
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201A': "'",  # Single low-9 quotation mark
        '\u201B': "'",  # Single high-reversed-9 quotation mark
        '\u2032': "'",  # Prime
        '\u00B4': "'",  # Acute accent
        '`': "'",       # Grave accent / backtick used as apostrophe
    }

    # Typographic dashes and hyphens
    TYPOGRAPHIC_DASHES = {
        '\u2014': ' - ',  # Em-dash -> space-padded hyphen
        '\u2015': ' - ',  # Horizontal bar
        '\u2013': '-',    # En-dash -> standard hyphen
        '\u2212': '-',    # Minus sign -> standard hyphen
        '\u2010': '-',    # Hyphen
        '\u2011': '-',    # Non-breaking hyphen
    }

    # Non-standard spaces to standard ASCII space (0x20)
    TYPOGRAPHIC_SPACES = {
        '\u00A0': ' ',  # Non-breaking space
        '\u2002': ' ',  # En space
        '\u2003': ' ',  # Em space
        '\u2004': ' ',  # Three-per-em space
        '\u2005': ' ',  # Four-per-em space
        '\u2006': ' ',  # Six-per-em space
        '\u2007': ' ',  # Figure space
        '\u2008': ' ',  # Punctuation space
        '\u2009': ' ',  # Thin space
        '\u200A': ' ',  # Hair space
        '\u202F': ' ',  # Narrow no-break space
        '\u205F': ' ',  # Medium mathematical space
        '\u3000': ' ',  # Ideographic space
    }

    # Zero-width characters to strip completely
    ZERO_WIDTH_CHARS = {
        '\u200B',  # Zero width space
        '\u200C',  # Zero width non-joiner
        '\u200D',  # Zero width joiner
        '\u2060',  # Word joiner
        '\uFEFF',  # Zero width non-breaking space / Byte Order Mark
        '\u00AD',  # Soft hyphen
    }

    # Negation pattern for invariant verification
    NEGATION_PATTERN = re.compile(
        r"\b(?:not|never|no|neither|nor|none|without|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|wouldn't|shouldn't|couldn't)\b",
        re.IGNORECASE
    )

    # URL pattern to protect URLs during character and whitespace normalization
    URL_PATTERN = re.compile(
        r"https?://(?:[a-zA-Z0-9$-_@.&+!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        re.IGNORECASE
    )

    @classmethod
    def sanitize(cls, text: Optional[str]) -> SanitizationResult:
        """
        Sanitizes input text safely while recording all transformations.
        Guarantees that raw_text is preserved immutably.
        """
        raw = text if text is not None else ""
        if not raw:
            return SanitizationResult(
                raw_text=raw,
                sanitized_text="",
                transformations=[],
                is_question=False,
                has_negation=False,
            )

        transformations: List[SanitizationTransformation] = []
        current = raw

        # Step 1: Detect raw question and negation properties before transformation
        is_question = cls._detect_question_intent(raw)
        has_negation = bool(cls.NEGATION_PATTERN.search(raw))

        # Step 2: Unicode NFC normalization (Canonical Decomposition + Composition)
        nfc_normalized = unicodedata.normalize("NFC", current)
        if nfc_normalized != current:
            transformations.append(SanitizationTransformation(
                rule="unicode_nfc",
                original=current,
                replacement=nfc_normalized,
                description="Applied Unicode NFC normalization to compose combining characters"
            ))
            current = nfc_normalized

        # Step 3: Strip zero-width and invisible control characters (preserve standard whitespace)
        cleaned_chars = []
        stripped_count = 0
        for ch in current:
            if ch in cls.ZERO_WIDTH_CHARS:
                stripped_count += 1
                continue
            cat = unicodedata.category(ch)
            # Remove Cc (control), Cf (format), Cs (surrogate), Co (private use), Cn (unassigned)
            # but PRESERVE standard newline '\n' and tab '\t'
            if cat.startswith("C") and ch not in ("\n", "\t", "\r"):
                stripped_count += 1
                continue
            cleaned_chars.append(ch)

        if stripped_count > 0:
            current_without_invisibles = "".join(cleaned_chars)
            transformations.append(SanitizationTransformation(
                rule="strip_invisible_control_chars",
                original=f"[{stripped_count} invisible/control characters]",
                replacement="",
                description=f"Removed {stripped_count} non-printable zero-width or control characters"
            ))
            current = current_without_invisibles
        else:
            current = "".join(cleaned_chars)

        # Step 4: Typographic quote & space & dash normalization
        mapped_chars = []
        quote_changes = 0
        space_changes = 0
        dash_changes = 0

        for ch in current:
            if ch in cls.TYPOGRAPHIC_QUOTES:
                mapped_chars.append(cls.TYPOGRAPHIC_QUOTES[ch])
                quote_changes += 1
            elif ch in cls.TYPOGRAPHIC_SPACES:
                mapped_chars.append(cls.TYPOGRAPHIC_SPACES[ch])
                space_changes += 1
            elif ch in cls.TYPOGRAPHIC_DASHES:
                mapped_chars.append(cls.TYPOGRAPHIC_DASHES[ch])
                dash_changes += 1
            else:
                mapped_chars.append(ch)

        current = "".join(mapped_chars)
        if quote_changes > 0:
            transformations.append(SanitizationTransformation(
                rule="typographic_quotes",
                original=f"[{quote_changes} curly/typographic quotes]",
                replacement="ASCII quotes (' or \")",
                description=f"Standardized {quote_changes} curly or smart quotation marks to ASCII"
            ))
        if space_changes > 0:
            transformations.append(SanitizationTransformation(
                rule="typographic_spaces",
                original=f"[{space_changes} non-standard spaces]",
                replacement="ASCII space",
                description=f"Unified {space_changes} non-breaking or typographic spaces to standard ASCII space"
            ))
        if dash_changes > 0:
            transformations.append(SanitizationTransformation(
                rule="typographic_dashes",
                original=f"[{dash_changes} typographic dashes]",
                replacement="Standard hyphens (-)",
                description=f"Standardized {dash_changes} em-dashes, en-dashes, or minus signs"
            ))

        # Step 5: Conservative repeated punctuation reduction (preserves single ? and !)
        prev_punc = current
        # Collapse multiple ? into single ?
        current = re.sub(r"\?{2,}", "?", current)
        # Collapse multiple ! into single !
        current = re.sub(r"\!{2,}", "!", current)
        # Collapse 4+ consecutive dots into 3-dot ellipsis
        current = re.sub(r"\.{4,}", "...", current)
        # Handle mixed ?! or !? conservatively (preserve question intent)
        current = re.sub(r"\?+\!+|\!+\?+", "?!", current)

        if current != prev_punc:
            transformations.append(SanitizationTransformation(
                rule="collapse_repeated_punctuation",
                original="Multiple repeated punctuation (e.g. ??? or !!!)",
                replacement="Standard punctuation mark",
                description="Normalized excessive repeated punctuation while preserving question/exclamation intent"
            ))

        # Step 6: Conservative repeated character reduction in words (3+ consecutive -> 2)
        # e.g., 'soooo' -> 'soo', 'cooool' -> 'cool', 'reallllly' -> 'really'
        # Reducing to 2 characters allows edit-distance-1/2 spelling correction to resolve
        # standard dictionary forms without corrupting valid double letters ('good', 'agree', 'committee')
        prev_repeat = current
        # Protect URLs from character reduction
        urls = cls.URL_PATTERN.findall(current)
        if not urls:
            current = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", current)
            if current != prev_repeat:
                transformations.append(SanitizationTransformation(
                    rule="reduce_repeated_characters",
                    original="3+ consecutive identical letters",
                    replacement="2 letters (e.g., 'soooo' -> 'soo')",
                    description="Conservatively reduced 3+ repeated characters to double letter for spelling lookup"
                ))

        # Step 7: Whitespace normalization (collapse multi-spaces, preserve single clean paragraph/line)
        prev_ws = current
        # Convert carriage returns
        current = current.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple horizontal spaces and tabs into a single space
        current = re.sub(r"[ \t]+", " ", current)
        # Collapse multiple blank lines into double newline
        current = re.sub(r"\n\s*\n+", "\n\n", current)
        # Trim leading and trailing whitespace
        current = current.strip()

        if current != prev_ws:
            transformations.append(SanitizationTransformation(
                rule="collapse_whitespace",
                original="Excess horizontal spaces / trailing whitespace",
                replacement="Single space",
                description="Collapsed multiple horizontal spaces and trimmed boundaries"
            ))

        # Final Verification Invariants:
        # 1. Question intent must not be lost
        if is_question and not current.endswith("?") and "?" in raw:
            # If the original raw input had a question mark at the end or near the end, restore it
            if not current.endswith("?"):
                current = current.rstrip(".! ") + "?"
                transformations.append(SanitizationTransformation(
                    rule="restore_question_mark",
                    original="Missing terminal punctuation",
                    replacement="?",
                    description="Restored terminal question mark to preserve inquiry intent"
                ))

        return SanitizationResult(
            raw_text=raw,
            sanitized_text=current,
            transformations=transformations,
            is_question=is_question,
            has_negation=has_negation,
        )

    @staticmethod
    def _detect_question_intent(text: str) -> bool:
        """
        Determines whether the input expresses an inquiry or question rather than a factual assertion.
        Checks for terminal question marks, leading interrogative words, or auxiliary inversion.
        """
        clean = text.strip()
        if not clean:
            return False
        if clean.endswith("?"):
            return True

        # Check for leading question patterns:
        # "Is it true that...", "Does X cause Y", "Can X do Y", "Why did...", "What is..."
        leading_interrogative = re.match(
            r"^(?:is|are|was|were|do|does|did|can|could|would|should|will|why|what|when|where|who|whom|how)\b",
            clean,
            re.IGNORECASE
        )
        return bool(leading_interrogative and ("?" in clean or len(clean.split()) <= 15))
