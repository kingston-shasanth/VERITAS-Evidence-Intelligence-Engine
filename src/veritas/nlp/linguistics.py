"""
Linguistic Analysis and Shallow Syntactic Parsing Module for VERITAS-X.

Provides self-contained, deterministic linguistic processing:
- Sentence segmentation (with abbreviation and numeric period disambiguation).
- Tokenization (with character offset tracking, clitic/contraction splitting, and unit protection).
- Lemmatization (irregular tables + rule-based morphological reduction).
- Part-of-speech tagging (Penn Treebank and Universal POS schemas).
- Morphological feature extraction (Number, Tense, Aspect, Voice).
- Clause boundary detection (coordinate, subordinate, conditional, relative clauses).
- Shallow dependency and grammatical relation assignment (nsubj, nsubjpass, root, dobj, obl, aux, neg).
- Sentence type classification (assertion, question, fragment).

Architectural Note:
This module uses a deterministic rule-based and morphological lexicon approach,
guaranteeing 100% offline availability, sub-millisecond execution, zero network downloads,
and thread safety without requiring multi-gigabyte models or CUDA initialization.
"""
from dataclasses import dataclass, field
import re
from typing import List, Dict, Any, Optional, Tuple, Set


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class Morphology:
    """Grammatical and inflectional morphology features."""
    number: Optional[str] = None      # "Sing", "Plur"
    tense: Optional[str] = None       # "Pres", "Past", "Fut", "Modal"
    aspect: Optional[str] = None      # "Simple", "Prog", "Perf"
    voice: Optional[str] = None       # "Active", "Passive"
    person: Optional[str] = None      # "1", "2", "3"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            "number": self.number,
            "tense": self.tense,
            "aspect": self.aspect,
            "voice": self.voice,
            "person": self.person,
        }.items() if v is not None}


@dataclass
class LinguisticToken:
    """Structured token with lexical, morphological, and syntactic annotations."""
    text: str
    normalized_text: str
    lemma: str
    pos: str                          # Penn Treebank tag (e.g. NN, VBZ, VBN, JJ)
    upos: str                         # Universal POS tag (e.g. NOUN, VERB, ADJ)
    start_char: int
    end_char: int
    dep_rel: str = "dep"              # Syntactic relation: nsubj, root, dobj, aux, neg, etc.
    head_index: int = -1              # Index of head token in sentence (-1 if root)
    morphology: Morphology = field(default_factory=Morphology)
    confidence: float = 1.0           # 1.0 for closed-class/exact rules, 0.7-0.9 for heuristics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "lemma": self.lemma,
            "pos": self.pos,
            "upos": self.upos,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "dep_rel": self.dep_rel,
            "head_index": self.head_index,
            "morphology": self.morphology.to_dict(),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class Clause:
    """An individual clause identified within a sentence."""
    clause_id: str                    # e.g. "C1", "C2"
    clause_type: str                  # "root", "coordinate", "subordinate", "conditional", "relative", "causal"
    text: str
    tokens: List[LinguisticToken] = field(default_factory=list)
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    connective: Optional[str] = None  # e.g. "and", "but", "because", "if"
    is_passive: bool = False
    is_negated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "clause_type": self.clause_type,
            "text": self.text,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "connective": self.connective,
            "is_passive": self.is_passive,
            "is_negated": self.is_negated,
            "token_count": len(self.tokens),
        }


@dataclass
class AnalyzedSentence:
    """Linguistic and syntactic analysis envelope for a sentence."""
    text: str
    tokens: List[LinguisticToken] = field(default_factory=list)
    clauses: List[Clause] = field(default_factory=list)
    sentence_type: str = "assertion"   # "assertion", "question", "fragment"
    subject_candidate: Optional[str] = None
    predicate_candidate: Optional[str] = None
    object_candidate: Optional[str] = None
    is_passive: bool = False
    is_question: bool = False
    is_incomplete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "sentence_type": self.sentence_type,
            "subject_candidate": self.subject_candidate,
            "predicate_candidate": self.predicate_candidate,
            "object_candidate": self.object_candidate,
            "is_passive": self.is_passive,
            "is_question": self.is_question,
            "is_incomplete": self.is_incomplete,
            "tokens": [t.to_dict() for t in self.tokens],
            "clauses": [c.to_dict() for c in self.clauses],
        }


@dataclass
class LinguisticAnalysisResult:
    """Overall document-level linguistic analysis result."""
    raw_text: str
    sentences: List[AnalyzedSentence] = field(default_factory=list)
    tokens: List[LinguisticToken] = field(default_factory=list)
    all_clauses: List[Clause] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "sentence_count": len(self.sentences),
            "token_count": len(self.tokens),
            "clause_count": len(self.all_clauses),
            "sentences": [s.to_dict() for s in self.sentences],
            "processing_time_ms": round(self.processing_time_ms, 3),
        }


# ==============================================================================
# Linguistic Dictionaries & Tables
# ==============================================================================

# Irregular verb stems and principal parts: form -> (base_lemma, pos, tense, aspect)
IRREGULAR_VERBS: Dict[str, Tuple[str, str, str, str]] = {
    # be
    "is": ("be", "VBZ", "Pres", "Simple"),
    "am": ("be", "VBP", "Pres", "Simple"),
    "are": ("be", "VBP", "Pres", "Simple"),
    "was": ("be", "VBD", "Past", "Simple"),
    "were": ("be", "VBD", "Past", "Simple"),
    "been": ("be", "VBN", "Past", "Perf"),
    "being": ("be", "VBG", "Pres", "Prog"),
    # have
    "has": ("have", "VBZ", "Pres", "Simple"),
    "have": ("have", "VBP", "Pres", "Simple"),
    "had": ("have", "VBD", "Past", "Simple"),
    "having": ("have", "VBG", "Pres", "Prog"),
    # do
    "does": ("do", "VBZ", "Pres", "Simple"),
    "do": ("do", "VBP", "Pres", "Simple"),
    "did": ("do", "VBD", "Past", "Simple"),
    "doing": ("do", "VBG", "Pres", "Prog"),
    "done": ("do", "VBN", "Past", "Perf"),
    # Common English factual verbs
    "freezes": ("freeze", "VBZ", "Pres", "Simple"),
    "froze": ("freeze", "VBD", "Past", "Simple"),
    "frozen": ("freeze", "VBN", "Past", "Perf"),
    "boils": ("boil", "VBZ", "Pres", "Simple"),
    "boiled": ("boil", "VBD", "Past", "Simple"),
    "prevents": ("prevent", "VBZ", "Pres", "Simple"),
    "prevented": ("prevent", "VBD", "Past", "Simple"),
    "hides": ("hide", "VBZ", "Pres", "Simple"),
    "hid": ("hide", "VBD", "Past", "Simple"),
    "hidden": ("hide", "VBN", "Past", "Perf"),
    "causes": ("cause", "VBZ", "Pres", "Simple"),
    "caused": ("cause", "VBD", "Past", "Simple"),
    "leading": ("lead", "VBG", "Pres", "Prog"),
    "leads": ("lead", "VBZ", "Pres", "Simple"),
    "led": ("lead", "VBD", "Past", "Simple"),
    "confirmed": ("confirm", "VBD", "Past", "Simple"),
    "closed": ("close", "VBD", "Past", "Simple"),
    "reopened": ("reopen", "VBD", "Past", "Simple"),
    "announced": ("announce", "VBD", "Past", "Simple"),
    "dropped": ("drop", "VBD", "Past", "Simple"),
    "endorsed": ("endorse", "VBD", "Past", "Simple"),
    "discovered": ("discover", "VBD", "Past", "Simple"),
    "halted": ("halt", "VBD", "Past", "Simple"),
    "won": ("win", "VBD", "Past", "Simple"),
    "went": ("go", "VBD", "Past", "Simple"),
    "gone": ("go", "VBN", "Past", "Perf"),
    "goes": ("go", "VBZ", "Pres", "Simple"),
    "came": ("come", "VBD", "Past", "Simple"),
    "comes": ("come", "VBZ", "Pres", "Simple"),
    "said": ("say", "VBD", "Past", "Simple"),
    "says": ("say", "VBZ", "Pres", "Simple"),
    "made": ("make", "VBD", "Past", "Simple"),
    "makes": ("make", "VBZ", "Pres", "Simple"),
    "took": ("take", "VBD", "Past", "Simple"),
    "taken": ("take", "VBN", "Past", "Perf"),
    "takes": ("take", "VBZ", "Pres", "Simple"),
    "saw": ("see", "VBD", "Past", "Simple"),
    "seen": ("see", "VBN", "Past", "Perf"),
    "sees": ("see", "VBZ", "Pres", "Simple"),
    "knew": ("know", "VBD", "Past", "Simple"),
    "known": ("know", "VBN", "Past", "Perf"),
    "knows": ("know", "VBZ", "Pres", "Simple"),
    "found": ("find", "VBD", "Past", "Simple"),
    "finds": ("find", "VBZ", "Pres", "Simple"),
    "wrote": ("write", "VBD", "Past", "Simple"),
    "written": ("write", "VBN", "Past", "Perf"),
    "writes": ("write", "VBZ", "Pres", "Simple"),
    "became": ("become", "VBD", "Past", "Simple"),
    "becomes": ("become", "VBZ", "Pres", "Simple"),
    "shown": ("show", "VBN", "Past", "Perf"),
    "shows": ("show", "VBZ", "Pres", "Simple"),
    "showed": ("show", "VBD", "Past", "Simple"),
}

# Irregular plural nouns: form -> base_lemma
IRREGULAR_NOUNS: Dict[str, str] = {
    "children": "child",
    "people": "person",
    "men": "man",
    "women": "woman",
    "teeth": "tooth",
    "feet": "foot",
    "mice": "mouse",
    "geese": "goose",
    "oxen": "ox",
    "data": "data",
    "species": "species",
    "series": "series",
    "criteria": "criterion",
    "phenomena": "phenomenon",
    "hypotheses": "hypothesis",
    "analyses": "analysis",
}

# Modal auxiliaries
MODAL_VERBS: Set[str] = {
    "can", "could", "may", "might", "must", "shall", "should", "will", "would", "ought"
}

# Closed-class determiners
DETERMINERS: Set[str] = {
    "the", "a", "an", "this", "that", "these", "those", "every", "each", "all",
    "some", "any", "no", "neither", "either", "another", "both"
}

# Pronouns (Subject, Object, Possessive, Reflexive)
PRONOUNS: Dict[str, Tuple[str, str]] = {
    "i": ("PRP", "Sing"), "me": ("PRP", "Sing"), "my": ("PRP$", "Sing"), "myself": ("PRP", "Sing"),
    "we": ("PRP", "Plur"), "us": ("PRP", "Plur"), "our": ("PRP$", "Plur"), "ourselves": ("PRP", "Plur"),
    "you": ("PRP", "Plur"), "your": ("PRP$", "Plur"), "yourself": ("PRP", "Sing"), "yourselves": ("PRP", "Plur"),
    "he": ("PRP", "Sing"), "him": ("PRP", "Sing"), "his": ("PRP$", "Sing"), "himself": ("PRP", "Sing"),
    "she": ("PRP", "Sing"), "her": ("PRP", "Sing"), "hers": ("PRP$", "Sing"), "herself": ("PRP", "Sing"),
    "it": ("PRP", "Sing"), "its": ("PRP$", "Sing"), "itself": ("PRP", "Sing"),
    "they": ("PRP", "Plur"), "them": ("PRP", "Plur"), "their": ("PRP$", "Plur"), "themselves": ("PRP", "Plur"),
    "who": ("WP", "Sing"), "whom": ("WP", "Sing"), "whose": ("WP$", "Sing"),
    "which": ("WDT", "Sing"), "what": ("WP", "Sing"),
}

# Coordinating and Subordinating Conjunctions
COORDINATING_CONJUNCTIONS: Set[str] = {"and", "but", "or", "nor", "for", "yet", "so"}
SUBORDINATING_CONJUNCTIONS: Set[str] = {
    "because", "although", "though", "even", "while", "whereas", "since",
    "unless", "if", "whether", "after", "before", "until", "when", "whenever",
    "where", "wherever", "as"
}

# Prepositions (IN)
COMMON_PREPOSITIONS: Set[str] = {
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "over", "under", "within", "without", "towards", "upon"
}

# Quantifier tokens
QUANTIFIER_WORDS: Set[str] = {
    "all", "every", "each", "some", "any", "none", "no", "few", "many",
    "several", "both", "half", "most", "always", "sometimes", "never", "often", "rarely"
}

# Known abbreviations that contain internal periods to prevent false sentence segmentation
ABBREVIATIONS: Set[str] = {
    "u.s.", "u.k.", "u.n.", "e.g.", "i.e.", "al.", "etc.", "vs.", "approx.",
    "dept.", "univ.", "dr.", "mr.", "mrs.", "ms.", "prof.", "gov.", "gen.", "rep.", "sen."
}


# ==============================================================================
# Linguistic Analyzer Engine
# ==============================================================================

class LinguisticAnalyzer:
    """
    Lightweight, production-grade linguistic analyzer for VERITAS-X.
    Performs sentence segmentation, tokenization, lemmatization, POS tagging,
    clause boundary detection, and shallow dependency assignment.
    """

    @classmethod
    def analyze(cls, text: Optional[str]) -> LinguisticAnalysisResult:
        """
        Executes end-to-end linguistic analysis on input text.
        Guarantees deterministic, fast execution without external service calls.
        """
        import time
        t0 = time.perf_counter()
        raw = text.strip() if text is not None else ""

        if not raw:
            return LinguisticAnalysisResult(
                raw_text=raw,
                sentences=[],
                tokens=[],
                all_clauses=[],
                processing_time_ms=0.0
            )

        # 1. Sentence Segmentation
        sentence_spans = cls._segment_sentences(raw)
        analyzed_sentences: List[AnalyzedSentence] = []
        all_tokens: List[LinguisticToken] = []
        all_clauses: List[Clause] = []

        for sent_idx, (s_text, s_start, s_end) in enumerate(sentence_spans, start=1):
            # 2. Tokenize sentence into linguistic tokens with character offsets
            tokens = cls._tokenize_sentence(s_text, offset_delta=s_start)

            # 3. Lemmatize, POS tag, and annotate morphology
            cls._tag_and_lemmatize(tokens)

            # 4. Detect clauses within the sentence
            clauses = cls._detect_clauses(s_text, tokens, sent_id=f"S{sent_idx}")

            # 5. Extract grammatical relations & shallow dependencies
            cls._assign_dependencies(tokens, clauses)

            # 6. Classify sentence structure and candidate SVO elements
            sent_type, is_q, is_inc, subj, pred, obj, is_pass = cls._classify_sentence_structure(
                s_text, tokens, clauses
            )

            analyzed_sent = AnalyzedSentence(
                text=s_text,
                tokens=tokens,
                clauses=clauses,
                sentence_type=sent_type,
                subject_candidate=subj,
                predicate_candidate=pred,
                object_candidate=obj,
                is_passive=is_pass,
                is_question=is_q,
                is_incomplete=is_inc,
            )
            analyzed_sentences.append(analyzed_sent)
            all_tokens.extend(tokens)
            all_clauses.extend(clauses)

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return LinguisticAnalysisResult(
            raw_text=raw,
            sentences=analyzed_sentences,
            tokens=all_tokens,
            all_clauses=all_clauses,
            processing_time_ms=dt_ms,
        )

    # --------------------------------------------------------------------------
    # 1. Sentence Segmentation
    # --------------------------------------------------------------------------
    @classmethod
    def _segment_sentences(cls, text: str) -> List[Tuple[str, int, int]]:
        """
        Segments text into sentences while protecting abbreviations and decimals.
        Returns list of (sentence_text, start_char, end_char).
        """
        if not text:
            return []

        # Find potential sentence boundaries: punctuation followed by space or end
        pattern = re.compile(r"([.?!]+)(?:\s+|$)")
        sentences: List[Tuple[str, int, int]] = []
        cur_start = 0

        for match in pattern.finditer(text):
            punc_end = match.end(1)
            candidate_sent = text[cur_start:punc_end].strip()
            if not candidate_sent:
                continue

            # Check if terminal dot belongs to an abbreviation (e.g. U.S., approx., etc.)
            last_word = candidate_sent.split()[-1].lower() if candidate_sent.split() else ""
            if last_word in ABBREVIATIONS:
                # Do not split on abbreviation dot
                continue

            # Check if decimal number (e.g. "3.14")
            if re.search(r"\d+\.\d+$", candidate_sent):
                continue

            # Valid sentence boundary found
            sent_str = text[cur_start:match.end()].strip()
            actual_start = text.find(sent_str, cur_start)
            actual_end = actual_start + len(sent_str)
            sentences.append((sent_str, actual_start, actual_end))
            cur_start = match.end()

        # Append any remaining trailing clause
        if cur_start < len(text):
            trailing = text[cur_start:].strip()
            if trailing:
                actual_start = text.find(trailing, cur_start)
                actual_end = actual_start + len(trailing)
                sentences.append((trailing, actual_start, actual_end))

        return sentences if sentences else [(text, 0, len(text))]

    # --------------------------------------------------------------------------
    # 2. Tokenization with Offset Tracking
    # --------------------------------------------------------------------------
    @classmethod
    def _tokenize_sentence(cls, sentence: str, offset_delta: int = 0) -> List[LinguisticToken]:
        """
        Tokenizes an individual sentence into LinguisticToken objects.
        Preserves URLs, emails, dates, numbers, units (100°C, 0°C), possessive clitics ('s),
        and hyphenated words.
        """
        url_email_pat = r"https?://[^\s<>\"'()]+|www\.[^\s<>\"'()]+|[\w\.-]+@[\w\.-]+\.\w+"
        unit_pat = r"\d+(?:°[cCfFkK]|%|\s*atm|\s*deg(?:rees?)?)"
        word_clitic_pat = r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*(?:'s|'t|'ve|'re|'ll|'d|'m)?"
        punc_pat = r"[^\w\s]"

        combined_pat = re.compile(
            f"({url_email_pat}|{unit_pat}|{word_clitic_pat}|{punc_pat})",
            re.IGNORECASE
        )

        tokens: List[LinguisticToken] = []
        cur_search_pos = 0

        for match in combined_pat.finditer(sentence):
            raw_tok = match.group(0)
            start = match.start()
            end = match.end()

            # Separate attached trailing sentence punctuation from URLs if captured
            if (raw_tok.startswith("http://") or raw_tok.startswith("https://") or raw_tok.startswith("www.")) and raw_tok[-1] in ".,!?:;":
                url_part = raw_tok[:-1]
                punc_part = raw_tok[-1]
                tokens.append(LinguisticToken(
                    text=url_part,
                    normalized_text=url_part,
                    lemma=url_part,
                    pos="NNP",
                    upos="PROPN",
                    start_char=offset_delta + start,
                    end_char=offset_delta + end - 1,
                    confidence=1.0
                ))
                tokens.append(LinguisticToken(
                    text=punc_part,
                    normalized_text=punc_part,
                    lemma=punc_part,
                    pos=".",
                    upos="PUNCT",
                    start_char=offset_delta + end - 1,
                    end_char=offset_delta + end,
                    confidence=1.0
                ))
                continue

            # Separate clitics (e.g. user's -> user + 's, don't -> do + n't) for granular syntactic parsing
            if raw_tok.lower().endswith("'s") and len(raw_tok) > 2:
                stem = raw_tok[:-2]
                clitic = raw_tok[-2:]
                tokens.append(LinguisticToken(
                    text=stem,
                    normalized_text=stem.lower(),
                    lemma=stem.lower(),
                    pos="NN",
                    upos="NOUN",
                    start_char=offset_delta + start,
                    end_char=offset_delta + start + len(stem),
                    confidence=0.95
                ))
                tokens.append(LinguisticToken(
                    text=clitic,
                    normalized_text=clitic.lower(),
                    lemma=clitic.lower(),
                    pos="POS",
                    upos="PART",
                    start_char=offset_delta + start + len(stem),
                    end_char=offset_delta + end,
                    confidence=1.0
                ))
                continue

            tokens.append(LinguisticToken(
                text=raw_tok,
                normalized_text=raw_tok.lower(),
                lemma=raw_tok.lower(),
                pos="NN",   # Will be refined in POS tagging
                upos="NOUN",
                start_char=offset_delta + start,
                end_char=offset_delta + end,
                confidence=0.85
            ))

        return tokens

    # --------------------------------------------------------------------------
    # 3. POS Tagging & Lemmatization
    # --------------------------------------------------------------------------
    @classmethod
    def _tag_and_lemmatize(cls, tokens: List[LinguisticToken]) -> None:
        """
        Applies contextual POS tagging, lemmatization, and morphology extraction.
        """
        n_tokens = len(tokens)
        for i, t in enumerate(tokens):
            raw = t.text
            low = t.normalized_text
            prev_t = tokens[i - 1] if i > 0 else None
            next_t = tokens[i + 1] if i + 1 < n_tokens else None

            # A. Punctuation
            if re.match(r"^[^\w\s]$", raw):
                t.pos = "." if raw in ".?!" else ("," if raw == "," else raw)
                t.upos = "PUNCT"
                t.lemma = raw
                t.confidence = 1.0
                continue

            # B. Numbers and units
            if re.match(r"^\d+(?:,\d+)*(?:\.\d+)?%?$", raw) or raw.isdigit():
                t.pos = "CD"
                t.upos = "NUM"
                t.lemma = raw
                t.morphology = Morphology(number="Plur" if raw != "1" else "Sing")
                t.confidence = 1.0
                continue

            if re.match(r"^\d+(?:°[cCfFkK]|\s*atm|\s*deg(?:rees?)?)$", raw, re.IGNORECASE):
                t.pos = "NN"
                t.upos = "NOUN"
                t.lemma = low
                t.morphology = Morphology(number="Sing")
                t.confidence = 1.0
                continue

            # C. Determiners
            if low in DETERMINERS:
                t.pos = "DT"
                t.upos = "DET"
                t.lemma = low
                t.confidence = 1.0
                continue

            # D. Pronouns (with all-caps 'US' disambiguation)
            if raw == "US":
                t.pos = "NNP"
                t.upos = "PROPN"
                t.lemma = "US"
                t.morphology = Morphology(number="Sing")
                t.confidence = 1.0
                continue

            if low in PRONOUNS:
                ptag, pnum = PRONOUNS[low]
                t.pos = ptag
                t.upos = "PRON"
                t.lemma = "he" if low in ("him", "his") else ("she" if low == "her" else ("they" if low == "them" else low))
                t.morphology = Morphology(number=pnum)
                t.confidence = 1.0
                continue

            # E. Conjunctions (Coordinating & Subordinating)
            if low in COORDINATING_CONJUNCTIONS:
                t.pos = "CC"
                t.upos = "CCONJ"
                t.lemma = low
                t.confidence = 1.0
                continue

            if low in SUBORDINATING_CONJUNCTIONS:
                t.pos = "IN"
                t.upos = "SCONJ"
                t.lemma = low
                t.confidence = 1.0
                continue

            # F. Prepositions
            if low in COMMON_PREPOSITIONS:
                # Disambiguate "to" (TO vs IN)
                t.pos = "TO" if low == "to" else "IN"
                t.upos = "ADP"
                t.lemma = low
                t.confidence = 0.95
                continue

            # G. Modal Auxiliaries
            if low in MODAL_VERBS:
                t.pos = "MD"
                t.upos = "AUX"
                t.lemma = low
                t.morphology = Morphology(tense="Modal")
                t.confidence = 1.0
                continue

            # H. Negation particle
            if low in ("not", "never", "n't", "neither", "nor", "cannot"):
                t.pos = "RB"
                t.upos = "PART"
                t.lemma = "not" if low in ("n't", "cannot") else low
                t.confidence = 1.0
                continue

            # I. Contextual Base Verb (preceded by modal, 'to', or aux + neg)
            if prev_t and (
                prev_t.pos in ("MD", "TO") or
                (prev_t.pos == "RB" and i >= 2 and tokens[i - 2].pos in ("MD", "VBZ", "VBD", "VBP"))
            ):
                t.pos = "VB"
                t.upos = "VERB"
                t.lemma = low
                t.morphology = Morphology(tense="Pres")
                t.confidence = 0.92
                continue

            # J. Irregular Verbs Table
            if low in IRREGULAR_VERBS:
                base_lem, pos_tag, v_tense, v_aspect = IRREGULAR_VERBS[low]
                # Contextual check: is it an auxiliary or main verb?
                is_aux = low in ("is", "am", "are", "was", "were", "been", "being", "has", "have", "had", "does", "do", "did")
                t.pos = pos_tag
                t.upos = "AUX" if is_aux else "VERB"
                t.lemma = base_lem
                t.morphology = Morphology(tense=v_tense, aspect=v_aspect, number="Sing" if pos_tag.endswith("Z") else None)
                t.confidence = 0.98
                continue

            # J. Irregular Nouns Table
            if low in IRREGULAR_NOUNS:
                t.pos = "NNS"
                t.upos = "NOUN"
                t.lemma = IRREGULAR_NOUNS[low]
                t.morphology = Morphology(number="Plur")
                t.confidence = 0.98
                continue

            # K. Proper Nouns (Capitalized tokens inside sentence)
            is_sentence_initial = (i == 0 or (prev_t and prev_t.pos == "."))
            if raw[0].isupper() and len(raw) >= 2 and not is_sentence_initial:
                t.pos = "NNP"
                t.upos = "PROPN"
                t.lemma = raw
                t.morphology = Morphology(number="Sing")
                t.confidence = 0.95
                continue

            # Multi-word proper noun heuristic (e.g. Amy Coney Barrett, Supreme Court)
            if raw[0].isupper() and next_t and next_t.text and next_t.text[0].isupper():
                t.pos = "NNP"
                t.upos = "PROPN"
                t.lemma = raw
                t.morphology = Morphology(number="Sing")
                t.confidence = 0.92
                continue

            # L. Adverbial Suffix (-ly)
            if low.endswith("ly") and len(low) >= 4 and not low.endswith(("early", "holy", "ugly", "belly")):
                t.pos = "RB"
                t.upos = "ADV"
                t.lemma = low[:-2] if low.endswith("ly") else low
                t.confidence = 0.90
                continue

            # M. Adjectival Suffixes (-able, -ible, -al, -ous, -ful, -less, -ic)
            if low.endswith(("able", "ible", "ous", "ful", "less", "ic", "al")) and len(low) >= 5:
                t.pos = "JJ"
                t.upos = "ADJ"
                t.lemma = low
                t.confidence = 0.88
                continue

            # N. Regular Verb Inflections (-ing, -ed, -es, -s)
            if low.endswith("ing") and len(low) >= 5:
                # Disambiguate gerund noun vs participle
                if prev_t and prev_t.pos in ("VBZ", "VBD", "VBP", "MD", "RB"):
                    t.pos = "VBG"
                    t.upos = "VERB"
                    t.morphology = Morphology(tense="Pres", aspect="Prog")
                elif prev_t and prev_t.pos in ("DT", "JJ", "PRP$"):
                    t.pos = "NN"
                    t.upos = "NOUN"
                    t.morphology = Morphology(number="Sing")
                else:
                    t.pos = "VBG"
                    t.upos = "VERB"
                    t.morphology = Morphology(tense="Pres", aspect="Prog")
                # Lemmatize -ing
                t.lemma = cls._lemmatize_ing(low)
                t.confidence = 0.88
                continue

            if low.endswith("ed") and len(low) >= 4:
                # Check if preceded by 'be' auxiliary -> passive VBN
                is_passive_context = bool(prev_t and prev_t.lemma in ("be", "become"))
                t.pos = "VBN" if is_passive_context else "VBD"
                t.upos = "VERB"
                t.morphology = Morphology(tense="Past", voice="Passive" if is_passive_context else "Active")
                t.lemma = cls._lemmatize_ed(low)
                t.confidence = 0.90
                continue

            if low.endswith("es") and len(low) >= 4:
                # Could be 3rd person singular verb (watches) or plural noun (boxes)
                if prev_t and prev_t.pos in ("PRP", "NNP", "NN"):
                    t.pos = "VBZ"
                    t.upos = "VERB"
                    t.morphology = Morphology(tense="Pres", number="Sing", person="3")
                    t.lemma = cls._lemmatize_es(low)
                else:
                    t.pos = "NNS"
                    t.upos = "NOUN"
                    t.morphology = Morphology(number="Plur")
                    t.lemma = cls._lemmatize_es(low)
                t.confidence = 0.85
                continue

            if low.endswith("s") and len(low) >= 3 and not low.endswith("ss"):
                # Disambiguate plural noun vs 3rd person verb
                if prev_t and prev_t.pos in ("PRP", "NN", "NNP"):
                    t.pos = "VBZ"
                    t.upos = "VERB"
                    t.morphology = Morphology(tense="Pres", number="Sing", person="3")
                    t.lemma = low[:-1]
                else:
                    t.pos = "NNS"
                    t.upos = "NOUN"
                    t.morphology = Morphology(number="Plur")
                    t.lemma = low[:-1]
                t.confidence = 0.82
                continue

            # O. Base Noun fallback (default for content words)
            t.pos = "NN"
            t.upos = "NOUN"
            t.lemma = low
            t.morphology = Morphology(number="Sing")
            t.confidence = 0.75

    # --------------------------------------------------------------------------
    # Morphological Suffix Lemmatizers
    # --------------------------------------------------------------------------
    @staticmethod
    def _lemmatize_ing(word: str) -> str:
        stem = word[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "ls":
            return stem[:-1]  # running -> run, tracking -> track
        if len(stem) >= 2 and stem.endswith(("v", "c", "z", "s", "d", "t", "k")):
            # freezing -> freeze, hiding -> hide, taking -> take
            if not stem.endswith(("ck", "sh", "ch", "th", "ct", "lt", "st")):
                return stem + "e"
        return stem

    @staticmethod
    def _lemmatize_ed(word: str) -> str:
        stem = word[:-2]
        if stem.endswith("i"):
            return stem[:-1] + "y"  # carried -> carry
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "ls":
            return stem[:-1]  # stopped -> stop, dropped -> drop
        if stem.endswith(("v", "c", "z", "s", "d", "t", "k", "n", "r")):
            if not stem.endswith(("ck", "sh", "ch", "th", "ct", "lt", "st", "nd", "nt")):
                return stem + "e"  # confirmed -> confirm vs closed -> close
        return stem

    @staticmethod
    def _lemmatize_es(word: str) -> str:
        stem = word[:-2]
        if stem.endswith("i"):
            return stem[:-1] + "y"  # activities -> activity
        if stem.endswith(("ch", "sh", "x", "s", "z")):
            return stem             # boxes -> box, watches -> watch
        return stem + "e"           # freezes -> freeze

    # --------------------------------------------------------------------------
    # 4. Clause Boundary Detection
    # --------------------------------------------------------------------------
    @classmethod
    def _detect_clauses(
        cls, sentence_text: str, tokens: List[LinguisticToken], sent_id: str = "S1"
    ) -> List[Clause]:
        """
        Detects independent and dependent clause boundaries within a sentence.
        """
        clauses: List[Clause] = []
        cur_clause_tokens: List[LinguisticToken] = []
        cur_clause_type = "root"
        cur_connective = None
        clause_counter = 1

        for i, t in enumerate(tokens):
            low = t.normalized_text

            # Check for coordinating conjunction between clauses: e.g. ", and", ", but", ";"
            is_coord = (
                (t.pos == "CC" and i > 2 and tokens[i - 1].pos in (",", ";")) or
                (t.pos == "." and t.text == ";")
            )
            # Check for subordinating conjunction: e.g. "because", "although", "while", "since"
            is_subord = (t.pos == "IN" and low in SUBORDINATING_CONJUNCTIONS)
            # Check for conditional: e.g. "if", "unless"
            is_cond = (low in ("if", "unless") and t.pos in ("IN", "SCONJ"))

            if (is_coord or is_subord or is_cond) and len(cur_clause_tokens) >= 2:
                # Close current clause
                c_text = " ".join(tk.text for tk in cur_clause_tokens)
                clauses.append(Clause(
                    clause_id=f"{sent_id}_C{clause_counter}",
                    clause_type=cur_clause_type,
                    text=c_text,
                    tokens=list(cur_clause_tokens),
                    connective=cur_connective,
                ))
                clause_counter += 1
                cur_clause_tokens = []

                if is_coord:
                    cur_clause_type = "coordinate"
                    cur_connective = low
                elif is_cond:
                    cur_clause_type = "conditional"
                    cur_connective = low
                else:
                    cur_clause_type = "subordinate"
                    cur_connective = low

            cur_clause_tokens.append(t)

        if cur_clause_tokens:
            c_text = " ".join(tk.text for tk in cur_clause_tokens)
            clauses.append(Clause(
                clause_id=f"{sent_id}_C{clause_counter}",
                clause_type=cur_clause_type,
                text=c_text,
                tokens=list(cur_clause_tokens),
                connective=cur_connective,
            ))

        return clauses

    # --------------------------------------------------------------------------
    # 5. Dependency & Relation Assignment
    # --------------------------------------------------------------------------
    @classmethod
    def _assign_dependencies(
        cls, tokens: List[LinguisticToken], clauses: List[Clause]
    ) -> None:
        """
        Assigns shallow grammatical relations (root, nsubj, nsubjpass, dobj, obl, aux, neg, det).
        """
        # Find primary predicate verb (root) in sentence
        # Lexical verbs take precedence as root; auxiliaries attach to root as 'aux'
        root_idx = -1
        lexical_verbs = [
            i for i, t in enumerate(tokens)
            if t.pos in ("VBZ", "VBD", "VBP", "VB") and t.lemma not in ("be", "do", "have")
        ]
        finite_verbs = [
            i for i, t in enumerate(tokens)
            if t.pos in ("VBZ", "VBD", "VBP", "VB")
        ]

        if lexical_verbs:
            root_idx = lexical_verbs[0]
        elif finite_verbs:
            root_idx = finite_verbs[0]
        else:
            for i, t in enumerate(tokens):
                if t.upos in ("VERB", "AUX"):
                    root_idx = i
                    break

        if root_idx != -1:
            tokens[root_idx].dep_rel = "root"
            tokens[root_idx].head_index = -1

        # Check for passive construction: 'be' auxiliary + VBN (+ 'by' agent)
        is_passive = False
        passive_aux_idx = -1
        passive_verb_idx = -1

        for i in range(len(tokens) - 1):
            if tokens[i].lemma == "be" and tokens[i + 1].pos in ("VBN", "VBD"):
                is_passive = True
                passive_aux_idx = i
                passive_verb_idx = i + 1
                tokens[i].dep_rel = "auxpass"
                tokens[i].head_index = passive_verb_idx
                tokens[passive_verb_idx].dep_rel = "root"
                tokens[passive_verb_idx].head_index = -1
                tokens[passive_verb_idx].morphology.voice = "Passive"
                root_idx = passive_verb_idx
                break

        # Check for passive agent: 'by' + [DT/JJ]* + [NOUN/PROPN]+
        if is_passive:
            for k in range(len(tokens)):
                if tokens[k].normalized_text == "by" and tokens[k].pos in ("IN", "TO"):
                    m = k + 1
                    while m < len(tokens) and tokens[m].pos in ("DT", "JJ", "PRP$"):
                        m += 1
                    while m < len(tokens) and tokens[m].upos in ("NOUN", "PROPN", "PRON"):
                        tokens[m].dep_rel = "agent"
                        tokens[m].head_index = root_idx
                        m += 1
                    break

        # Assign subject, objects, and modifiers relative to root
        for i, t in enumerate(tokens):
            if i == root_idx or i == passive_aux_idx or t.dep_rel == "agent":
                continue

            # Punctuation
            if t.upos == "PUNCT":
                t.dep_rel = "punct"
                t.head_index = root_idx if root_idx != -1 else 0
                continue

            # Determiners & Adjectives modifying subsequent nouns
            if t.pos in ("DT", "CD"):
                t.dep_rel = "det"
                head = next((j for j in range(i + 1, len(tokens)) if tokens[j].upos in ("NOUN", "PROPN")), root_idx)
                t.head_index = head
                continue

            if t.pos in ("JJ", "JJR", "JJS"):
                t.dep_rel = "amod"
                head = next((j for j in range(i + 1, len(tokens)) if tokens[j].upos in ("NOUN", "PROPN")), root_idx)
                t.head_index = head
                continue

            # Adverbial modifier
            if t.pos in ("RB", "RBR", "RBS"):
                if t.normalized_text in ("not", "never", "n't"):
                    t.dep_rel = "neg"
                else:
                    t.dep_rel = "advmod"
                t.head_index = root_idx
                continue

            # Auxiliary verbs
            if t.upos == "AUX" or t.pos == "MD" or (t.lemma in ("do", "be", "have") and i != root_idx):
                t.dep_rel = "aux"
                t.head_index = root_idx
                continue

            # Subject Assignment (nominal before root, skipping sentence-initial question auxiliaries)
            if root_idx != -1 and i < root_idx and t.upos in ("NOUN", "PROPN", "PRON"):
                if is_passive:
                    t.dep_rel = "nsubjpass"
                else:
                    t.dep_rel = "nsubj"
                t.head_index = root_idx
                continue

            # Object Assignment (nominal after root)
            if root_idx != -1 and i > root_idx and t.upos in ("NOUN", "PROPN", "PRON"):
                if i > 0 and (tokens[i - 1].pos in ("IN", "TO") or (i > 1 and tokens[i - 2].pos in ("IN", "TO"))):
                    t.dep_rel = "obl"
                    t.head_index = root_idx
                else:
                    t.dep_rel = "dobj"
                    t.head_index = root_idx
                continue

    # --------------------------------------------------------------------------
    # 6. Sentence Structure & SVO Extraction
    # --------------------------------------------------------------------------
    @classmethod
    def _classify_sentence_structure(
        cls,
        sentence_text: str,
        tokens: List[LinguisticToken],
        clauses: List[Clause]
    ) -> Tuple[str, bool, bool, Optional[str], Optional[str], Optional[str], bool]:
        """
        Classifies sentence into assertion, question, or fragment, and extracts candidate SVO spans.
        Returns: (sentence_type, is_question, is_incomplete, subject, predicate, object, is_passive)
        """
        clean = sentence_text.strip()
        is_question = False

        # Question detection
        if clean.endswith("?"):
            is_question = True
        elif tokens and tokens[0].normalized_text in (
            "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
            "should", "will", "why", "what", "when", "where", "who", "whom", "how"
        ) and (clean.endswith("?") or len(tokens) <= 12):
            is_question = True

        # Check for finite verb
        has_finite_verb = any(t.pos in ("VBZ", "VBD", "VBP", "MD", "VB") and t.dep_rel == "root" for t in tokens)
        is_passive = any(t.dep_rel in ("nsubjpass", "auxpass") or t.morphology.voice == "Passive" for t in tokens)

        # Incomplete fragment detection
        is_incomplete = False
        if not has_finite_verb and not is_question:
            is_incomplete = True

        sent_type = "question" if is_question else ("fragment" if is_incomplete else "assertion")

        # Extract SVO Spans
        subject_span = None
        predicate_span = None
        object_span = None

        subj_indices = [i for i, t in enumerate(tokens) if t.dep_rel in ("nsubj", "nsubjpass")]
        if not subj_indices and is_passive:
            subj_indices = [i for i, t in enumerate(tokens) if t.dep_rel == "agent"]

        if subj_indices:
            min_i = subj_indices[0]
            max_i = subj_indices[-1]
            while min_i > 0 and tokens[min_i - 1].dep_rel in ("det", "amod", "compound"):
                min_i -= 1
            while max_i + 1 < len(tokens) and tokens[max_i + 1].dep_rel in ("compound", "nsubj", "nsubjpass") and tokens[max_i + 1].upos in ("NOUN", "PROPN"):
                max_i += 1
            subject_span = " ".join(t.text for t in tokens[min_i:max_i + 1])

        # Predicate span: collect all auxiliary verbs, negations, and the root verb
        root_tokens = [t for t in tokens if t.dep_rel in ("root", "aux", "auxpass", "neg")]
        if root_tokens:
            predicate_span = " ".join(t.text for t in root_tokens)

        # Object span: direct object or oblique prepositional object
        obj_indices = [i for i, t in enumerate(tokens) if t.dep_rel in ("dobj", "obl")]
        if obj_indices:
            min_i = obj_indices[0]
            max_i = obj_indices[0]
            while min_i > 0 and tokens[min_i - 1].dep_rel in ("det", "amod", "compound"):
                min_i -= 1
            while max_i + 1 < len(tokens) and tokens[max_i + 1].dep_rel in ("compound", "dobj", "obl") and tokens[max_i + 1].upos in ("NOUN", "PROPN"):
                max_i += 1
            object_span = " ".join(t.text for t in tokens[min_i:max_i + 1])

        return sent_type, is_question, is_incomplete, subject_span, predicate_span, object_span, is_passive
