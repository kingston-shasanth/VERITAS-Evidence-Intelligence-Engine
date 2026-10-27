"""
Verification Engine for VERITAS-X.
Orchestrates hybrid retrieval, cross-encoder NLI, multi-evidence consensus aggregation,
temporal anachronism checks, temperature calibration, and selective abstention.
"""
from dataclasses import dataclass, field
import math
import re
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np

from veritas.data.schema import CanonicalVeracity
from veritas.retrieval.hybrid_retriever import HybridRetriever
from veritas.retrieval.tokenizer import MultilingualTokenizer
from .nli_model import (
    CrossEncoderNLI,
    _is_local_storage_scope_mismatch,
    _is_ip_hiding_scope_mismatch,
    _is_generalized_scope_mismatch,
)
from .evidence_aggregator import EvidenceAggregator, AggregatedEvidenceVerdict
from .temporal_analyzer import TemporalAnalyzer, TemporalReport, TemporalStatus
from .calibrator import TemperatureCalibrator, SelectiveAbstainer

_CAUSAL_CLAIM_REGEX = re.compile(
    r"\b(?:causes?|causing|caused|leads? to|leading to|induces?|inducing|induced|brings? about|resulting in|results in|proves? that .*? causes?)\b",
    re.IGNORECASE,
)

_ASSOCIATION_OR_CORRELATION_REGEX = re.compile(
    r"\b(?:correlation|correlates?|correlated|correlating|association|associated|linked|link|relationship between|observational study|observational data|epidemiological association)\b",
    re.IGNORECASE,
)

_CONFOUNDING_OR_NON_CAUSAL_REGEX = re.compile(
    r"\b(?:confound\w*|spurious|reverse causation|lurking variable|third variable|correlation does not (?:imply|prove|mean|equate to) causation|not (?:shown|proven|established|found) to cause|no (?:causal|direct causal) (?:link|relationship|evidence|proof))\b",
    re.IGNORECASE,
)

_EXPLICIT_CAUSAL_PROOF_REGEX = re.compile(
    r"\b(?:randomized controlled trial|rct|double-blind|causal mechanism established|proven to cause|demonstrated causality|causally established|conclusively causes|causation was demonstrated)\b",
    re.IGNORECASE,
)

_TOPICAL_STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'could',
    'did', 'do', 'does', 'doing', 'don', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had',
    'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i',
    'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no',
    'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out',
    'over', 'own', 's', 'same', 'she', 'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their',
    'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who',
    'whom', 'why', 'will', 'with', 'would', 'better', 'worse', 'good', 'bad', 'best', 'worst',
}

_BRIDGE_STOP_WORDS = {
    'the', 'a', 'an', 'is', 'was', 'were', 'are', 'been', 'be', 'being',
    'in', 'on', 'at', 'of', 'to', 'for', 'and', 'or', 'as', 'by', 'with',
    'that', 'this', 'these', 'those', 'from', 'it', 'its', 'they', 'them',
    'their', 'who', 'whom', 'whose', 'which', 'part', 'also', 'has', 'had',
    'have', 'having', 'said', 'stated', 'reported', 'one', 'two', 'first',
    'second', 'new', 'time', 'year', 'years', 'day', 'days', 'after', 'before',
    'over', 'under', 'between', 'both', 'all', 'some', 'any', 'not', 'no'
}

_AUX_OR_MODAL_REGEX = re.compile(
    r"\b(is|was|are|were|been|being|has|have|had|will|would|shall|should|can|could|may|might|must|did|does|do|cannot|can't|won't|wouldn't|shouldn't|couldn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|don't|didn't)\b",
    re.IGNORECASE
)

_ADVERB_PREDICATE_REGEX = re.compile(
    r"\b(never|always|rarely|seldom|hardly|scarcely|normally|typically|generally|often|frequently|no longer|officially|reportedly|supposedly|allegedly|automatically|directly|completely|partially|consistently|primarily|solely)\b",
    re.IGNORECASE
)

# Deterministic set of closed-class nouns that end in -s, -ise, or -ics and must NEVER be treated as verbs
_KNOWN_NON_VERB_NOUNS = {
    # -ics scientific / academic nouns
    "physics", "economics", "mathematics", "statistics", "linguistics", "optics",
    "acoustics", "mechanics", "thermodynamics", "electronics", "genetics", "robotics", "ethics",
    # Biological / anatomical / medical nouns
    "lungs", "bones", "cells", "veins", "arteries", "organs", "muscles", "nerves", "tissues",
    "limbs", "teeth", "ribs", "digits", "genes", "diabetes", "measles", "rabies", "syphilis",
    # Physical / astronomical / chemical nouns
    "atoms", "molecules", "elements", "compounds", "particles", "electrons", "protons", "neutrons",
    "planets", "stars", "galaxies", "satellites", "telescopes", "asteroids", "comets", "meteors",
    # General plural nouns and nouns ending in -s, -ise, -ize
    "humans", "people", "adults", "children", "animals", "mammals", "birds", "plants", "trees",
    "games", "olympics", "awards", "medals", "prizes", "prize", "records", "events", "years", "days",
    "hours", "minutes", "seconds", "degrees", "celsius", "fahrenheit", "kelvin", "meters",
    "kilometers", "miles", "feet", "inches", "grams", "kilograms", "pounds", "tons", "liters",
    "species", "series", "corps", "headquarters", "crossroads", "barracks", "enterprise", "exercise",
    "franchise", "compromise", "premise", "promise", "surprise",
    "gas", "gases", "data", "status", "basis", "axis", "news", "levels", "results", "effects",
    "details", "conditions", "sources", "countries", "cities", "caffeine",
}

# English irregular verb stems and inflected forms
_IRREGULAR_VERB_FORMS = {
    "held", "hold", "holds", "won", "win", "wins", "wrote", "write", "writes", "written",
    "fell", "fall", "falls", "fallen", "ran", "run", "runs", "sent", "send", "sends",
    "spoke", "speak", "speaks", "spoken", "grew", "grow", "grows", "grown",
    "led", "lead", "leads", "shot", "shoot", "shoots", "stood", "stand", "stands",
    "met", "meet", "meets", "kept", "keep", "keeps", "paid", "pay", "pays",
    "hit", "hits", "cut", "cuts", "put", "puts", "flew", "fly", "flies", "flown",
    "ate", "eat", "eats", "eaten", "drank", "drink", "drinks", "drunk",
    "drove", "drive", "drives", "driven", "knew", "know", "knows", "known",
    "came", "come", "comes", "went", "go", "goes", "gone", "rose", "rise", "rises", "risen",
    "began", "begin", "begins", "begun", "took", "take", "takes", "taken",
    "gave", "give", "gives", "given", "saw", "see", "sees", "seen", "told", "tell", "tells",
    "found", "find", "finds", "died", "die", "dies", "built", "build", "builds",
    "broke", "break", "breaks", "broken", "sold", "sell", "sells", "bought", "buy", "buys",
    "cost", "costs", "read", "reads", "lost", "lose", "loses", "became", "become", "becomes",
    "fight", "fights", "fought", "seek", "seeks", "sought",
    "teach", "teaches", "taught", "catch", "catches", "caught",
    "bring", "brings", "brought", "think", "thinks", "thought",
    "spend", "spends", "spent",
    "born", "situated", "founded", "known",
}

_DETERMINERS_AND_MODIFIERS = {
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "all", "each", "every", "some", "any", "no", "both", "few", "many", "several",
}

_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
}

_PERIODIC_TABLE_SYMBOLS = {
    "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al", "si", "p", "s", "cl", "ar",
    "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn", "ga", "ge", "as", "se", "br", "kr",
    "rb", "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn", "sb", "te", "i", "xe",
    "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb", "lu",
    "hf", "ta", "w", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi", "po", "at", "rn", "fr", "ra", "ac",
    "th", "pa", "u", "np", "pu", "am", "cm", "bk", "cf", "es", "fm", "md", "no", "lr", "rf", "db", "sg", "bh",
    "hs", "mt", "ds", "rg", "cn", "nh", "fl", "mc", "lv", "ts", "og"
}

_COMMON_ENGLISH_WORDS_2LETTER = {
    "dr", "mr", "mrs", "ms", "prof", "st", "mt", "he", "it", "we", "me", "us", "my", "no", "so", "to",
    "in", "on", "at", "by", "if", "as", "of", "or", "an", "is", "be", "do", "go", "up", "am"
}

_DATE_NUM_PATTERN = re.compile(
    r"\b("
    r"\d{1,6}|"
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|trillion|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"january|february|march|april|may|june|july|august|september|october|november|december"
    r")\b",
    re.IGNORECASE
)


def _is_syntactic_verb(token: str, prev_token: str = "", next_token: str = "", is_capitalized: bool = False, in_title_phrase: bool = False) -> bool:
    """
    Syntactic & morphological validator for arbitrary English verbs.
    Guarantees proper nouns, plural nouns, calendar months, and words in noun phrases
    are NEVER mistaken for verbs without relying on claim-specific whitelists.
    """
    tl = token.lower()

    # Priority: Closed-class auxiliaries & copulas or verified irregular verbs
    if _AUX_OR_MODAL_REGEX.fullmatch(tl):
        return True

    if tl in _IRREGULAR_VERB_FORMS:
        if not (prev_token.lower() in _DETERMINERS_AND_MODIFIERS and tl.endswith("s")):
            return True

    # Guard 1: Determiner / count quantifier guard (preceded by article/determiner/count numeral -> head of NP, not verb)
    # 4-digit years (e.g. 1984, 1889) in titles or date adverbials are not count determiners.
    is_count_num = (prev_token.isdigit() and not re.match(r'^(?:1[789]\d{2}|20\d{2})$', prev_token)) or (
        prev_token.lower() in _NUMBER_WORDS and not re.match(r'^(?:nineteen|twenty)\b', prev_token.lower())
    )
    if prev_token.lower() in _DETERMINERS_AND_MODIFIERS or is_count_num:
        return False

    # Guard 2: Proper noun guard (capitalized words inside multi-word titles/named entities)
    if is_capitalized and in_title_phrase:
        return False

    # Guard 3: Known non-verb noun / academic domain guard
    if tl in _KNOWN_NON_VERB_NOUNS or tl.endswith("ics"):
        return False

    # Guard 4: Calendar month guard
    if tl in {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}:
        return False

    # Rule 5: Closed-class auxiliaries & copulas
    if _AUX_OR_MODAL_REGEX.fullmatch(tl):
        return True

    # Rule 6: Irregular verbs
    if tl in _IRREGULAR_VERB_FORMS:
        return True

    # Rule 7: Regular past/participle morphology: ends with -ed (length >= 4)
    if tl.endswith("ed") and len(tl) >= 4:
        if tl not in {"united", "limited", "associated", "advanced", "detailed", "red", "bed", "led"}:
            return True

    # Rule 8: Derivational verb suffixes (-ize, -ized, -ise, -ised, -ify, -ified)
    if re.search(r"(?:iz|is|if)[ye]d?$", tl) and len(tl) >= 5 and tl not in _KNOWN_NON_VERB_NOUNS:
        return True

    # Rule 9: 3rd-person singular present (-s, -es, -ies) where root is a plausible verb
    if tl.endswith(("s", "es", "ies")) and len(tl) >= 4:
        if tl.endswith(("ous", "less", "ness", "tion", "tions", "sion", "sions", "ment", "ments", "ship", "ships", "itis", "sis", "ics")):
            return False
        if tl in {
            "strikes", "causes", "boils", "freezes", "orbits", "absorbs", "emits", "burns",
            "carries", "produces", "prevents", "requires", "contains", "protects",
            "hides", "masks", "conceals", "blocks", "stops", "shields",
            "destroys", "generates", "affects", "proves", "serves", "belongs", "originates",
            "defines", "means", "states", "claims", "disproves", "debunks", "refutes",
            "supports", "contradicts", "exceeds", "equals", "surpasses", "reaches", "occurs",
            "releases", "decreases", "increases", "melts", "expands", "contracts",
            "converts", "transforms", "leads", "forms", "yields", "supplies", "provides",
            "receives", "transmits", "regulates", "controls", "inhibits", "stimulates",
            "allows", "enables", "shows", "indicates", "suggests", "reduces", "maintains",
            "expresses", "interacts", "binds", "dissolves", "combines", "reacts", "features",
        }:
            return True
        root = tl[:-3] + "y" if tl.endswith("ies") else (tl[:-2] if tl.endswith("es") and tl not in {"eyes"} else tl[:-1])
        if root in _IRREGULAR_VERB_FORMS or (len(root) >= 3 and root.endswith(("ize", "ise", "ify"))):
            return True

    return False


_NORM_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "first": "1st", "second": "2nd", "third": "3rd",
    "celsius": "c", "centigrade": "c", "fahrenheit": "f",
    "degrees": "deg", "degree": "deg", "percent": "%"
}

_CONCEPT_EQUIVALENTS = {
    "coffee": {"caffeine"},
    "caffeine": {"coffee"},
    "moon": {"satellite", "lunar"},
    "lunar": {"moon"},
    "satellite": {"moon"},
    "mortality": {"death", "fatalities"},
    "death": {"mortality"},
}

_YEAR_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
_YEAR_TEENS = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"}
_YEAR_UNITS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def _year_to_word_components(year_str: str) -> Set[str]:
    """Generates standard English spoken word components for 4-digit years (e.g. 1984 -> {nineteen, eighty, four})."""
    if not (year_str.isdigit() and len(year_str) == 4):
        return set()
    y = int(year_str)
    c, d = y // 100, y % 100
    words = set()
    if c in _YEAR_TEENS:
        words.add(_YEAR_TEENS[c])
    elif c in _YEAR_TENS:
        words.add(_YEAR_TENS[c])
    if d in _YEAR_TEENS:
        words.add(_YEAR_TEENS[d])
    elif d >= 20:
        words.add(_YEAR_TENS[d // 10])
        if d % 10 in _YEAR_UNITS:
            words.add(_YEAR_UNITS[d % 10])
    elif d in _YEAR_UNITS:
        words.add(_YEAR_UNITS[d])
    return words


def _tokens_match(t1: str, t2: str) -> bool:
    t1, t2 = t1.lower(), t2.lower()
    if t1 == t2:
        return True
    n1 = _NORM_MAP.get(t1, t1)
    n2 = _NORM_MAP.get(t2, t2)
    if n1 == n2:
        return True
    if (t1 in _NORM_MAP and _NORM_MAP[t1] == t2) or (t2 in _NORM_MAP and _NORM_MAP[t2] == t1):
        return True
    if t2 in _CONCEPT_EQUIVALENTS.get(t1, set()) or t1 in _CONCEPT_EQUIVALENTS.get(t2, set()):
        return True
    # 4-digit year to English word components equivalence (e.g. 1984 and nineteen/eighty/four)
    if t1.isdigit() and len(t1) == 4 and t2 in _year_to_word_components(t1):
        return True
    if t2.isdigit() and len(t2) == 4 and t1 in _year_to_word_components(t2):
        return True
    l1, l2 = len(t1), len(t2)
    if l1 >= 4 and l2 >= 4:
        if t1 in t2 or t2 in t1:
            return True
        prefix_len = min(5, min(l1, l2))
        if prefix_len >= 4 and t1[:prefix_len] == t2[:prefix_len] and abs(l1 - l2) <= 4:
            return True
    return False


def _match_token_set(target_set: Set[str], text_tokens: Set[str]) -> Set[str]:
    matched = set()
    for target in target_set:
        for t in text_tokens:
            if _tokens_match(target, t):
                matched.add(target)
                break
    return matched


def _decompose_claim(claim_text: str) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    Decomposes a factual claim into core subject tokens, predicate/relation tokens,
    date/number tokens, and main verb tokens using robust syntax and morphology.
    """
    if not claim_text:
        return set(), set(), set(), set()

    clean_claim = claim_text.strip()

    # Strategy 1: Auxiliary / Modal boundary (e.g. 'was launched', 'can kill', 'does not cause')
    aux_matches = list(_AUX_OR_MODAL_REGEX.finditer(clean_claim))
    chosen_match = None
    if aux_matches:
        for m in aux_matches:
            prefix = clean_claim[:m.start()].strip()
            if prefix and any(c.isalnum() for c in prefix):
                word = m.group(0)
                if word.lower() == "may":
                    suffix = clean_claim[m.end():].strip()
                    if re.search(r'\b(in|during|since|until|by|of|on)\s*$', prefix, re.IGNORECASE) or re.match(r'^\d{1,4}\b', suffix):
                        continue
                chosen_match = m
                break

    # Strategy 2: Adverb of frequency/negation/modality preceding predicate (e.g. 'never strikes')
    if chosen_match is None:
        adv_matches = list(_ADVERB_PREDICATE_REGEX.finditer(clean_claim))
        for m in adv_matches:
            prefix = clean_claim[:m.start()].strip()
            if prefix and any(c.isalnum() for c in prefix):
                chosen_match = m
                break

    # Strategy 3: General morphological and syntactic verb detection across tokens
    if chosen_match is None:
        words = clean_claim.split()
        char_pos = 0
        for i, w in enumerate(words):
            word_clean = re.sub(r'^[^\w]+|[^\w]+$', '', w)
            pos = clean_claim.find(w, char_pos)
            char_pos = pos + len(w)

            if i == 0 and len(words) > 1:
                continue

            prev_w = re.sub(r'^[^\w]+|[^\w]+$', '', words[i-1]) if i > 0 else ""
            next_w = re.sub(r'^[^\w]+|[^\w]+$', '', words[i+1]) if i+1 < len(words) else ""

            is_cap = w[0].isupper() if w else False
            in_title = is_cap and ((i > 0 and words[i-1][0].isupper()) or (i+1 < len(words) and words[i+1][0].isupper()))

            if _is_syntactic_verb(word_clean, prev_token=prev_w, next_token=next_w, is_capitalized=is_cap, in_title_phrase=in_title):
                prefix = clean_claim[:pos].strip()
                if prefix and any(c.isalnum() for c in prefix):
                    class MatchMock:
                        def __init__(self, s, e, g):
                            self._s = s
                            self._e = e
                            self._g = g
                        def start(self): return self._s
                        def end(self): return self._e
                        def group(self, n=0): return self._g
                    chosen_match = MatchMock(pos, pos + len(w), w)
                    break

    if chosen_match is not None:
        subj_text = clean_claim[:chosen_match.start()].strip()
        verb_part = chosen_match.group(0).strip()
        pred_text = clean_claim[chosen_match.start():].strip()
    else:
        words = clean_claim.split()
        if len(words) >= 4:
            subj_text = " ".join(words[:2])
            verb_part = words[2]
            pred_text = " ".join(words[2:])
        else:
            subj_text = clean_claim
            verb_part = ""
            pred_text = ""

    s_tokens = MultilingualTokenizer.tokenize(subj_text)
    sub_subj = {t.lower() for t in s_tokens if t.lower() not in _TOPICAL_STOP_WORDS and (len(t) > 1 or t.isalnum())}

    p_tokens = MultilingualTokenizer.tokenize(pred_text)
    sub_pred = {t.lower() for t in p_tokens if t.lower() not in _TOPICAL_STOP_WORDS and (len(t) > 1 or t.isalnum())}

    all_tokens = MultilingualTokenizer.tokenize(clean_claim)
    date_num_tokens = {t.lower() for t in all_tokens if _DATE_NUM_PATTERN.match(t)}

    verb_tokens = {t.lower() for t in MultilingualTokenizer.tokenize(verb_part) if t.lower() not in _TOPICAL_STOP_WORDS and t.lower() not in _KNOWN_NON_VERB_NOUNS and not t.lower().endswith("ics")}

    # Also check if verb_part is an auxiliary followed by a main verb in pred_text (e.g. "do not kill", "was completed")
    aux_vp_match = re.search(r'\b(?:do|does|did|can|could|will|would|should|must|may|might|cannot|was|were|is|are|has|have|had)\s+(?:not\s+)?([a-z]+)\b', pred_text, re.IGNORECASE)
    if aux_vp_match:
        cand_v = aux_vp_match.group(1).lower()
        if (
            cand_v not in _TOPICAL_STOP_WORDS
            and cand_v not in _KNOWN_NON_VERB_NOUNS
            and not cand_v.endswith("ics")
            and cand_v not in _NUMBER_WORDS
            and not cand_v.isdigit()
            and not _DATE_NUM_PATTERN.match(cand_v)
            and _is_syntactic_verb(cand_v)
        ):
            verb_tokens.add(cand_v)

    for t in p_tokens:
        tl = t.lower()
        if tl in _KNOWN_NON_VERB_NOUNS or tl.endswith('ics') or tl in _NUMBER_WORDS:
            continue
        if tl not in _TOPICAL_STOP_WORDS and (tl.endswith(('ed', 'ing', 'ize', 'ized', 'ise', 'ised', 'ify', 'ified')) or tl in _IRREGULAR_VERB_FORMS or _is_syntactic_verb(tl)):
            verb_tokens.add(tl)
            if len(verb_tokens) >= 3:
                break

    return sub_subj, sub_pred, date_num_tokens, verb_tokens



@dataclass
class ForensicVerdict:
    """Comprehensive forensic adjudication for a verified claim."""
    claim_id: Optional[str]
    claim_text: str
    verdict: CanonicalVeracity
    confidence: float
    uncalibrated_confidence: float
    was_abstained: bool
    abstention_reason: str
    class_probabilities: Dict[str, float]
    temporal_report: TemporalReport
    evidence_used_count: int
    duplicates_filtered: int
    irrelevant_passages_filtered: int
    evidence_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    raw_prediction: Optional[str] = None
    has_meaningful_evidence: bool = True
    claim_components: List[Dict[str, Any]] = field(default_factory=list)
    canonical_reason: Optional[str] = None
    evidence_counts: Dict[str, int] = field(default_factory=dict)
    pipeline_trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "uncalibrated_confidence": self.uncalibrated_confidence,
            "was_abstained": self.was_abstained,
            "abstention_reason": self.abstention_reason,
            "class_probabilities": self.class_probabilities,
            "temporal_status": self.temporal_report.status.value,
            "temporal_warning": self.temporal_report.warning_message,
            "temporal_delta_days": self.temporal_report.delta_days,
            "evidence_used_count": self.evidence_used_count,
            "duplicates_filtered": self.duplicates_filtered,
            "irrelevant_passages_filtered": self.irrelevant_passages_filtered,
            "evidence_breakdown": self.evidence_breakdown,
            "raw_prediction": self.raw_prediction,
            "has_meaningful_evidence": self.has_meaningful_evidence,
            "claim_components": self.claim_components,
            "canonical_reason": self.canonical_reason,
            "evidence_counts": self.evidence_counts,
            "pipeline_trace": self.pipeline_trace,
        }


class VerificationEngine:
    """
    End-to-end evidence verification pipeline.
    """

    LABEL_NAMES = [
        CanonicalVeracity.VERIFIED_TRUE.value,
        CanonicalVeracity.VERIFIED_FALSE.value,
        CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value,
        CanonicalVeracity.INSUFFICIENT_EVIDENCE.value,
    ]

    def __init__(
        self,
        nli_model: Optional[CrossEncoderNLI] = None,
        retriever: Optional[HybridRetriever] = None,
        calibrator: Optional[TemperatureCalibrator] = None,
        abstainer: Optional[SelectiveAbstainer] = None,
        device: Optional[str] = None,
    ):
        self.device = device
        self.nli = nli_model or CrossEncoderNLI(device=device)
        self.retriever = retriever
        self.aggregator = EvidenceAggregator()
        self.temporal_analyzer = TemporalAnalyzer()
        self.calibrator = calibrator or TemperatureCalibrator(temperature=1.0)
        self.abstainer = abstainer or SelectiveAbstainer(confidence_threshold=0.40, entropy_threshold=0.88)
        from veritas.reasoning.decomposer import ClaimDecomposer
        self.decomposer = ClaimDecomposer()

    @staticmethod
    def adjust_causal_nli_prediction(claim_text: str, passage_text: str, nli_pred: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjusts NLI predictions where the proposition asserts causation, but the documentary
        evidence merely asserts correlation, observational association, or confounding.
        Raw cross-encoders (e.g. DeBERTa) frequently assign high entailment probability to
        observational studies due to token overlap. This function enforces the epistemological
        principle that correlation/association does not entail causation.
        """
        if not claim_text or not passage_text:
            return nli_pred
        asserts_causation = bool(_CAUSAL_CLAIM_REGEX.search(claim_text))
        if not asserts_causation:
            return nli_pred

        has_confounding = bool(_CONFOUNDING_OR_NON_CAUSAL_REGEX.search(passage_text))
        has_association = bool(_ASSOCIATION_OR_CORRELATION_REGEX.search(passage_text))
        has_explicit_proof = bool(_EXPLICIT_CAUSAL_PROOF_REGEX.search(passage_text))

        # If passage only documents association/correlation without explicit causal proof,
        # or documents confounding/non-causal spuriousness:
        if (has_confounding or has_association) and not has_explicit_proof:
            e_prob = float(nli_pred.get("entailment", 0.0))
            c_prob = float(nli_pred.get("contradiction", 0.0))
            # If passage explicitly refutes the causal link (e.g. "carrying a lighter does not cause lung cancer"),
            # retain refutation/contradiction
            if c_prob > e_prob and c_prob >= 0.40:
                return nli_pred
            # Otherwise, observational correlation cannot establish causation -> normalize to NEUTRAL
            if e_prob > float(nli_pred.get("neutral", 0.0)):
                return {
                    "contradiction": 0.0,
                    "entailment": 0.0,
                    "neutral": 1.0,
                    "logits": [-5.0, -5.0, 5.0],
                    "predicted_label": "NEUTRAL",
                }
        return nli_pred

    @staticmethod
    def adjust_scope_mismatch_nli_prediction(claim_text: str, passage_text: str, nli_pred: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjusts NLI predictions where the proposition asserts that an action or mode prevents
        remote/network observation or third-party tracking (e.g. by websites, trackers, ISPs),
        but the documentary evidence merely describes client-side local data persistence (e.g.
        clearing or not saving cookies, browsing history, or form data on the local device).
        Raw cross-encoders (e.g. DeBERTa) frequently assign high entailment probability to
        local storage descriptions due to token overlap ('incognito', 'cookies', 'history', 'saved').
        This function enforces the epistemological rule that client-side local storage deletion
        does not establish prevention of remote/network tracking.
        """
        if not claim_text or not passage_text:
            return nli_pred

        if not _is_generalized_scope_mismatch(claim_text, passage_text):
            return nli_pred

        c_prob = float(nli_pred.get("contradiction", 0.0))
        e_prob = float(nli_pred.get("entailment", 0.0))
        # If passage explicitly refutes the claim, retain refutation
        if c_prob > e_prob and c_prob >= 0.40:
            return nli_pred

        if e_prob > float(nli_pred.get("neutral", 0.0)):
            return {
                "contradiction": 0.0,
                "entailment": 0.0,
                "neutral": 1.0,
                "logits": [-5.0, -5.0, 5.0],
                "predicted_label": "NEUTRAL",
            }
        return nli_pred

    @staticmethod
    def is_passage_informative(passage_text: str, claim_text: str = "") -> Tuple[bool, str]:
        """
        Evaluates whether a passage contains sufficient substantive context and completeness
        to evaluate a factual proposition. Excludes truncated fragments, isolated dates or numbers,
        unexplained pronouns, dictionary entries without assertions, and unanswered rhetorical questions.
        """
        if not passage_text or not passage_text.strip():
            return False, "empty_passage"

        clean_text = passage_text.strip()
        words = clean_text.split()
        if len(words) < 6 or len(clean_text) < 25:
            return False, "insufficient_length_less_than_6_words"

        # Pure placeholder or ellipsis
        if re.fullmatch(r"[\s.\[\]()#*_-]+", clean_text):
            return False, "placeholder_text"

        # Truncated trailing fragment (ending in dangling preposition, conjunction, or article)
        if len(words) < 14 and re.search(r"\b(and|or|because|although|whereas|such as|including|with|which|that|the|a|an)\s*[.….]*$", clean_text, re.IGNORECASE):
            return False, "truncated_dangling_fragment"

        # Isolated dates/numbers/metadata without descriptive clause
        non_stop_alnum = [w for w in words if w.lower() not in _TOPICAL_STOP_WORDS and re.search(r'\w', w)]
        if len(non_stop_alnum) <= 2 and all(_DATE_NUM_PATTERN.match(w) or w.startswith("http") or w.isdigit() for w in non_stop_alnum):
            return False, "isolated_datum_without_context"

        # Dictionary / glossary / pronunciation headers (e.g. "lightning /'lait.niŋ/ noun" or "word (noun):")
        has_pos = bool(re.search(r"\b(?:noun|verb|adjective|adverb)\b", clean_text, re.IGNORECASE))
        has_phonetic = bool(re.search(r"/[^/]+/", clean_text))
        if (has_pos or has_phonetic) and len(words) < 10:
            if re.match(r"^[\w\s'-]+(?:\s*/[^/]+/|\s*\(.*?\))?\s*(?:noun|verb|adjective|adverb)?\s*[:.-]?\s*$", clean_text, re.IGNORECASE):
                return False, "lexical_entry_header"
        if re.match(r"^[\w\s'-]{1,30}:\s*$", clean_text):
            return False, "lexical_entry_header"

        # Pure rhetorical question without answer
        if clean_text.endswith("?") and "?" not in clean_text[:-1] and ("." not in clean_text and "!" not in clean_text):
            return False, "unanswered_rhetorical_question"

        # Unresolved anaphoric pronoun as sole subject when claim has explicit named entity
        if re.match(r"^(it|he|she|they|this|that)\s+(was|is|did|had|has|were|are)\b", clean_text, re.IGNORECASE) and len(words) < 12:
            if claim_text:
                sub_subj, _, _, _ = _decompose_claim(claim_text)
                p_tokens = {t.lower() for t in MultilingualTokenizer.tokenize(clean_text)}
                if not _match_token_set(sub_subj, p_tokens):
                    return False, "unresolved_anaphoric_pronoun"

        return True, "informative"

    @staticmethod
    def is_figurative_or_lexical_definition(text: str) -> bool:
        """
        Detects whether a passage is a dictionary entry, glossary entry, or contextual article
        explaining the metaphorical or figurative meaning of an idiom, proverb, or colloquial saying
        (e.g., 'lightning never strikes twice is an idiom used to say...'), rather than asserting
        an empirical or scientific fact.
        """
        if not text:
            return False
        clean = text.strip()
        if re.search(r"\b(idiom|proverb|saying|adage|figure of speech|metaphor|figurative(?:ly)?|colloquial(?:ism)?)\b", clean, re.IGNORECASE):
            if re.search(r"\b(used to (?:say|mean|express|describe|indicate)|expression meaning|means? that|definition of|phrase meaning|popular myth)\b", clean, re.IGNORECASE):
                return True
            if re.search(r"^['\"“]?[\w\s'-]+['\"”]?\s*:\s*(?:idiom|proverb|saying|noun|expression)\b", clean, re.IGNORECASE):
                return True
        if re.search(r"^definition of\s+['\"“]?[\w\s'-]+['\"”]?", clean, re.IGNORECASE):
            return True
        return False

    @staticmethod
    def is_absence_of_evidence(text: str) -> bool:
        """
        Detects statements expressing epistemic uncertainty or lack of documentary evidence/detection
        (e.g., 'no evidence has been found', 'records make no mention', 'it remains unconfirmed whether')
        rather than definitive factual contradictions or positive empirical refutations.
        In fact-checking, absence of evidence is epistemically neutral (uncertainty), not proof of falsity.
        """
        if not text:
            return False
        clean = text.strip()
        # Pattern 1: No / little / insufficient / scant / lack of [adjectives] evidence / proof / records / traces / trials ...
        # followed by (has / was / is [ever / yet] found / detected / discovered / confirmed / observed / reported / exists / survives)
        # or followed by (shows / demonstrates / proves / indicates / establishes)
        if re.search(
            r"\b(?:no|little|insufficient|scant|lack of|without(?: any)?)\s+"
            r"(?:[\w'-]+\s+){0,4}"
            r"(?:evidence|proof|documentation|confirmation|trace|traces|indication|indications|record|records|data)\b.*?"
            r"\b(?:has|have|had|is|was|were|exists?|survives?)\b.*?"
            r"\b(?:found|detected|confirmed|discovered|observed|reported|documented|substantiated|verified|survive|survives|exist|exists)\b",
            clean,
            re.IGNORECASE,
        ) or re.search(
            r"\b(?:no|little|insufficient|scant|lack of|without(?: any)?)\s+"
            r"(?:[\w'-]+\s+){0,4}"
            r"(?:evidence|proof|documentation|confirmation|trace|traces|indication|indications|record|records|data|trial|trials|study|studies)\b.*?"
            r"\b(?:shows?|showed|demonstrates?|demonstrated|proves?|proved|indicates?|indicated|supports?|supported|finds?|found|established|establishes)\b",
            clean,
            re.IGNORECASE,
        ) or re.search(
            r"\b(?:there\s+is|there\s+was|there\s+has\s+been)\s+no\s+(?:[\w'-]+\s+){0,4}(?:evidence|proof|indication|data|record)\b",
            clean,
            re.IGNORECASE,
        ):
            return True

        # Pattern 2: Verb of containment/discovery + no [adjectives] mention / record / evidence of
        # (e.g. 'contains no surviving record or documented mention of', 'yielded no evidence of', 'found no trace of', 'makes no mention of')
        if re.search(
            r"\b(?:makes?|contains?|bears?|provides?|yields?|yielded|found|shows?|showed|detected|observed|uncovered?|reveals?|revealed|discovered|produced|includes?)\s+"
            r"no\s+(?:[\w'-]+\s+){0,4}"
            r"(?:mention|record|records|documentation|trace|traces|indication|indications|evidence|reference|proof|correspondence|letters?|writings?|papers?)\b.*?"
            r"\b(?:of|concerning|regarding|about|to|mentioning)\b",
            clean,
            re.IGNORECASE,
        ):
            return True

        # Pattern 3: Has not been found / discovered / detected / confirmed / proven / substantiated
        if re.search(
            r"\b(?:has|have|had|is|was|were)\s+(?:not\s+yet|never|not|scarcely)\s+(?:(?:ever|conclusively|definitively|reliably)\s+)?"
            r"(?:been\s+)?(?:found|discovered|detected|confirmed|verified|proven|observed|documented|substantiated)\b",
            clean,
            re.IGNORECASE,
        ) or re.search(
            r"\bno\s+[\w\s'-]{1,40}?\s+(?:has|have|had|was|were)\s+(?:ever\s+)?been\s+(?:found|discovered|detected|confirmed|verified|proven|observed|documented|substantiated)\b",
            clean,
            re.IGNORECASE,
        ):
            return True

        # Pattern 4: Remains / is unproven / unconfirmed / unknown / undocumented / undetected
        if re.search(
            r"\b(?:remains?|is|are|were)\s+(?:scientifically\s+|currently\s+|presently\s+|historically\s+)?"
            r"(?:unproven|unconfirmed|unknown|undetected|unsubstantiated|undocumented)\b",
            clean,
            re.IGNORECASE,
        ):
            return True

        # Pattern 5: No record / records exist or survives
        if re.search(
            r"\bno\s+(?:[\w'-]+\s+){0,3}"
            r"(?:record|records|mention|document|documents|manuscript|manuscripts|entry|entries|evidence)\s+"
            r"(?:has been|exists?|was found|were found|survives?|can be found)\b",
            clean,
            re.IGNORECASE,
        ):
            return True

        return False

    @staticmethod
    def is_topically_relevant(claim_text: str, passage_text: str) -> bool:
        """
        Determines whether a passage actually concerns the subject and relation expressed in the claim.
        A passage mentioning the same person, organization, place, date, or broad topic is not
        necessarily relevant evidence for the proposition. Requires substantive subject alignment.
        """
        if not claim_text or not passage_text:
            return False
        sub_subj, sub_pred, date_num_tokens, verb_tokens = _decompose_claim(claim_text)
        p_tokens = {t.lower() for t in MultilingualTokenizer.tokenize(passage_text)}

        all_claim = sub_subj | sub_pred
        m_all = _match_token_set(all_claim, p_tokens)
        if len(m_all) < (2 if len(all_claim) >= 2 else 1):
            return False

        if sub_subj:
            m_subj = _match_token_set(sub_subj, p_tokens)
            if len(m_subj) == 0:
                return False
            if len(sub_subj) >= 3 and len(m_subj) < 2:
                # Long subjects require either >= 2 subject tokens or 1 subject token + at least 1 substantive
                # attribute token from the predicate (excluding dates/temporal tokens, numbers, and verbs)
                attr_tokens = {
                    t for t in sub_pred
                    if t not in date_num_tokens
                    and t not in verb_tokens
                    and t not in _TOPICAL_STOP_WORDS
                    and len(t) > 2
                }
                m_attr = _match_token_set(attr_tokens, p_tokens) if attr_tokens else set()
                if not m_attr:
                    return False

        return True

    @staticmethod
    def has_claim_relation_alignment(claim_text: str, passage_text: str) -> bool:
        """
        Determines whether a passage has genuine relation or attribute alignment with the claim.
        Crucial relevance rule: Relevance must not require the passage to repeat the claim's
        exact predicate or asserted value (e.g. stating 'two lungs' is aligned with 'three lungs').
        Evaluated independently of subject topical relevance to properly support multi-hop reasoning.
        """
        if not claim_text or not passage_text:
            return False

        sub_subj, sub_pred, date_num_tokens, verb_tokens = _decompose_claim(claim_text)
        p_tokens = {t.lower() for t in MultilingualTokenizer.tokenize(passage_text)}

        # Attribute tokens in predicate: substantive nouns/properties being asserted or quantified
        attribute_tokens = {
            t for t in sub_pred
            if t not in date_num_tokens
            and t not in verb_tokens
            and t not in _TOPICAL_STOP_WORDS
            and t not in _NUMBER_WORDS
            and not t.isdigit()
            and len(t) > 2
        }

        m_attr = _match_token_set(attribute_tokens, p_tokens) if attribute_tokens else set()
        m_verb = _match_token_set(verb_tokens, p_tokens) if verb_tokens else set()

        # Temporal date tokens (4-digit year or calendar month)
        temporal_tokens = {
            t for t in date_num_tokens
            if re.match(r"^(1[789]\d{2}|20\d{2})$", t) or t in {
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december"
            }
        }
        m_temp = _match_token_set(temporal_tokens, p_tokens) if temporal_tokens else set()

        # Quantitative number tokens in predicate (excluding subject entity designation numbers like '11' in Apollo 11)
        pred_nums = date_num_tokens & sub_pred
        qty_tokens = pred_nums - temporal_tokens
        p_has_any_num = any(t in _NUMBER_WORDS or t.isdigit() for t in p_tokens)

        # 1. If claim asserts a specific date (e.g. 1921 Nobel Prize):
        # Passage must match the date OR (action verb + at least one substantive attribute) OR (at least 2 attributes)
        if temporal_tokens:
            if m_temp:
                return True
            if m_verb and m_attr:
                return True
            if len(m_attr) >= 2:
                return True
            return False

        # 2. If claim asserts a quantified attribute (e.g. 'three lungs', 'zero degrees Celsius'):
        # Passage stating an alternative quantity ('two lungs') or matching quantity + property
        m_qty = _match_token_set(qty_tokens, p_tokens) if qty_tokens else set()
        if qty_tokens and attribute_tokens:
            if m_attr and (p_has_any_num or m_verb):
                return True
            if m_qty and m_attr:
                return True
            return False

        # 3. Attribute match
        if m_attr:
            return True

        # 4. Action verb match
        if m_verb:
            return True

        # 5. General predicate overlap fallback (must match non-number tokens)
        if sub_pred:
            non_num_pred = {t for t in sub_pred if t not in date_num_tokens and t not in _NUMBER_WORDS and not t.isdigit()}
            if non_num_pred:
                m_pred = _match_token_set(non_num_pred, p_tokens)
                if len(m_pred) >= 1:
                    return True
            else:
                m_pred = _match_token_set(sub_pred, p_tokens)
                if len(m_pred) >= 1:
                    return True

        return False

    @staticmethod
    def decompose_claim_components(claim_text: str) -> List[Dict[str, Any]]:
        """
        Decomposes a compound or atomic claim into constituent material components:
        - Subject/Entity: The primary subject phrase.
        - Venue/Location: Specific location or venue constraints (e.g., 'in Athens', 'on the Moon').
        - Temporal/Date: Specific dates, years, or timeframes (e.g., 'in 1896', 'in December 2021').
        - Action/Relation: The core predicate or relation asserted.
        - Attribute/Property: Quantified attributes or properties (e.g., 'three lungs', '0 degrees Celsius').
        """
        if not claim_text:
            return []

        clean = claim_text.strip()
        components: List[Dict[str, Any]] = []

        sub_subj, sub_pred, date_num_tokens, verb_tokens = _decompose_claim(clean)

        if sub_subj:
            components.append({
                "type": "subject",
                "label": "Subject / Entity",
                "tokens": set(sub_subj),
            })

        # Attribution / Source component
        attr_source_match = re.search(r"\b(?:according to|reported by|sources claim(?: that)?|it is claimed that)\s+([^,]+)", clean, re.IGNORECASE)
        if attr_source_match:
            components.append({
                "type": "attribution",
                "label": "Attribution / Source",
                "tokens": {t.lower() for t in MultilingualTokenizer.tokenize(attr_source_match.group(1)) if t.lower() not in _TOPICAL_STOP_WORDS},
            })

        # Condition / Qualifier component
        cond_match = re.search(r"\b(?:at sea level|in vacuum|under standard\w*|normally|typically|generally)\b", clean, re.IGNORECASE)
        if cond_match:
            components.append({
                "type": "condition",
                "label": "Condition / Qualifier",
                "tokens": {t.lower() for t in MultilingualTokenizer.tokenize(cond_match.group(0)) if t.lower() not in _TOPICAL_STOP_WORDS},
            })

        year_match = re.search(r"\b(1[789]\d{2}|20\d{2})\b", clean)
        month_match = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            clean,
            re.IGNORECASE,
        )
        temporal_tokens = set()
        if year_match:
            temporal_tokens.add(year_match.group(0).lower())
        if month_match:
            temporal_tokens.add(month_match.group(0).lower())
        if temporal_tokens:
            components.append({
                "type": "temporal",
                "label": "Time / Date",
                "tokens": temporal_tokens,
            })

        loc_match = re.search(r"\b(?:in|at|on)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b", clean)
        if loc_match:
            loc_word = loc_match.group(1).lower()
            if loc_word not in {
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december"
            }:
                components.append({
                    "type": "location",
                    "label": "Venue / Location",
                    "tokens": {loc_word},
                })

        if verb_tokens:
            components.append({
                "type": "action",
                "label": "Action / Status",
                "tokens": set(verb_tokens),
            })

        # Asserted Value / Quantity
        raw_values = set(date_num_tokens) - temporal_tokens
        chem_match = re.findall(r"\b[A-Z][a-z]?\b", clean)
        for cm in chem_match:
            cml = cm.lower()
            if (
                cml in _PERIODIC_TABLE_SYMBOLS
                and cml not in _COMMON_ENGLISH_WORDS_2LETTER
                and cml not in _TOPICAL_STOP_WORDS
                and cm not in {"I", "A"}
            ):
                raw_values.add(cml)
        if raw_values:
            components.append({
                "type": "value",
                "label": "Asserted Value / Quantity",
                "tokens": set(raw_values),
            })

        attr_tokens = {
            t for t in sub_pred
            if t not in date_num_tokens and t not in verb_tokens and t not in _TOPICAL_STOP_WORDS and len(t) > 2
        }
        if attr_tokens and not any(c["type"] == "location" and attr_tokens.issubset(c["tokens"]) for c in components):
            components.append({
                "type": "attribute",
                "label": "Attribute / Property",
                "tokens": attr_tokens,
            })

        return components

    @staticmethod
    def assess_claim_coverage(components: List[Dict[str, Any]], passage_text: str) -> Tuple[str, List[str]]:
        """
        Determines which material claim components are addressed by the passage.
        Returns:
            (coverage_level: 'full' | 'partial' | 'none', covered_component_labels: List[str])
        """
        if not components or not passage_text:
            return "full", []

        p_tokens = {t.lower() for t in MultilingualTokenizer.tokenize(passage_text)}
        covered_labels: List[str] = []

        for comp in components:
            c_tokens = comp["tokens"]
            matches = _match_token_set(c_tokens, p_tokens)
            req = 2 if len(c_tokens) >= 3 else 1
            if len(matches) >= req:
                covered_labels.append(comp["label"])

        total = len(components)
        covered_count = len(covered_labels)

        if total <= 1:
            coverage = "full" if covered_count > 0 else "none"
        elif covered_count == total:
            coverage = "full"
        elif covered_count > 0:
            coverage = "partial"
        else:
            coverage = "none"

        return coverage, covered_labels

    @staticmethod
    def is_decision_eligible(claim_text: str, passage_text: str) -> bool:
        """
        Determines whether a passage is sufficiently relevant, informative, and relation-aligned
        to contribute decisive weight to the final verdict.
        Excludes off-topic context, uninformative fragments, unaligned relations, and figurative/idiom definitions.
        """
        if not VerificationEngine.is_topically_relevant(claim_text, passage_text):
            return False
        is_inf, _ = VerificationEngine.is_passage_informative(passage_text, claim_text)
        if not is_inf:
            return False
        if VerificationEngine.is_figurative_or_lexical_definition(passage_text):
            return False
        if not VerificationEngine.has_claim_relation_alignment(claim_text, passage_text):
            return False
        return True

    @staticmethod
    def detect_valid_bridge(
        claim_text: str,
        p1: Dict[str, Any],
        p2: Dict[str, Any],
    ) -> Tuple[bool, Set[str]]:
        """
        Evaluates candidate bridge relationships between two passages.
        Permits joint reasoning only when the passages establish a defensible chain of entities and relations.
        Requirements:
        1. Both passages are informative and non-figurative.
        2. One passage covers the subject/entity (or subject component), and the other covers the predicate/relation/property.
        3. The passages share at least one substantive, non-stopword entity/role/noun bridge token (length > 2)
           that establishes an identity or relational link between them.
        4. Rejects invalid bridges (e.g. swapped entities like Hemingway for Orwell, or unrelated passages).
        """
        if not claim_text:
            return False, set()
        t1 = (p1.get("text") or "").strip()
        t2 = (p2.get("text") or "").strip()
        if not t1 or not t2:
            return False, set()
        if p1.get("is_figurative", False) or p2.get("is_figurative", False):
            return False, set()
        if not p1.get("is_informative", True) or not p2.get("is_informative", True):
            return False, set()

        p1_subj = bool(p1.get("is_relevant", False) or "Subject / Entity" in p1.get("claim_components_covered", []))
        p2_subj = bool(p2.get("is_relevant", False) or "Subject / Entity" in p2.get("claim_components_covered", []))
        p1_pred = bool(p1.get("is_claim_aligned", False) or any(
            c in p1.get("claim_components_covered", [])
            for c in ("Action / Status", "Attribute / Property", "Temporal / Date", "Venue / Location", "Condition / Qualifier")
        ))
        p2_pred = bool(p2.get("is_claim_aligned", False) or any(
            c in p2.get("claim_components_covered", [])
            for c in ("Action / Status", "Attribute / Property", "Temporal / Date", "Venue / Location", "Condition / Qualifier")
        ))

        has_complementary = (p1_subj and p2_pred) or (p2_subj and p1_pred)
        if not has_complementary:
            return False, set()

        tokens1 = {w.lower() for w in MultilingualTokenizer.tokenize(t1)}
        tokens2 = {w.lower() for w in MultilingualTokenizer.tokenize(t2)}

        shared = {
            w for w in (tokens1 & tokens2)
            if len(w) > 2
            and w not in _BRIDGE_STOP_WORDS
            and w not in _TOPICAL_STOP_WORDS
            and w not in _COMMON_ENGLISH_WORDS_2LETTER
            and not w.isdigit()
        }
        if not shared:
            return False, set()

        return True, shared

    @staticmethod
    def check_evidence_grounding(
        evidence_breakdown: List[Dict[str, Any]],
        joint_nli_prediction: Optional[Dict[str, Any]] = None,
        neutral_threshold: float = 0.70,
        min_informative_prob: float = 0.25,
    ) -> bool:
        """
        Determines whether the evaluated passages or the joint premise provide
        meaningful, non-neutral documentary signal. Neutral topical passages,
        uninformative fragments, and unrelated refuters must not count as meaningful evidence.
        Supports multi-hop evidence collections where subject and predicate are documented across passages.
        """
        for item in evidence_breakdown:
            if item.get("is_joint", False):
                continue
            if not item.get("is_relevant", True):
                continue
            if not item.get("is_informative", True):
                continue
            if not item.get("is_claim_aligned", True):
                continue
            e_prob = float(item.get("entailment_prob", 0.0))
            c_prob = float(item.get("contradiction_prob", 0.0))
            n_prob = float(item.get("neutral_prob", 0.0))
            max_signal = max(e_prob, c_prob)
            if max_signal > n_prob and n_prob < neutral_threshold and max_signal >= min_informative_prob:
                return True

        has_any_eligible = any(
            item.get("is_relevant", False) and item.get("is_informative", False) and item.get("is_claim_aligned", False)
            for item in evidence_breakdown
            if not item.get("is_joint", False)
        )
        has_subject_coverage = any(
            item.get("is_relevant", False)
            for item in evidence_breakdown
            if not item.get("is_joint", False)
        )
        has_predicate_coverage = any(
            item.get("is_claim_aligned", False)
            for item in evidence_breakdown
            if not item.get("is_joint", False)
        )
        has_valid_bridge = any(
            item.get("has_valid_bridge", False)
            for item in evidence_breakdown
            if not item.get("is_joint", False)
        )
        has_joint_grounding = has_any_eligible or (has_subject_coverage and has_predicate_coverage) or has_valid_bridge

        if has_joint_grounding and joint_nli_prediction is not None:
            j_e = float(joint_nli_prediction.get("entailment", 0.0))
            j_c = float(joint_nli_prediction.get("contradiction", 0.0))
            j_n = float(joint_nli_prediction.get("neutral", 0.0))
            max_j = max(j_e, j_c)
            if max_j > j_n and j_n < neutral_threshold and max_j >= min_informative_prob:
                return True

        return False

    def verify_claim(
        self,
        claim_text: str,
        claim_date: Optional[str] = None,
        candidate_passages: Optional[List[Dict[str, Any]]] = None,
        claim_id: Optional[str] = None,
        top_k: int = 5,
    ) -> ForensicVerdict:
        """
        Adjudicates a claim through retrieval, NLI premise scoring, temporal analysis,
        temperature calibration, and selective abstention.
        """
        # 1. Retrieve evidence if not explicitly passed
        if candidate_passages is None:
            if self.retriever is not None:
                results = self.retriever.search(claim_text, top_k=top_k)
                candidate_passages = [r.to_dict() for r in results]
            else:
                candidate_passages = []

        # If zero candidate passages exist -> immediate Insufficient Evidence
        if not candidate_passages:
            temporal_rep = self.temporal_analyzer.analyze(claim_date, [])
            return ForensicVerdict(
                claim_id=claim_id,
                claim_text=claim_text,
                verdict=CanonicalVeracity.INSUFFICIENT_EVIDENCE,
                confidence=1.0,
                uncalibrated_confidence=1.0,
                was_abstained=True,
                abstention_reason="No relevant evidence passages were found to evaluate this claim.",
                class_probabilities={
                    CanonicalVeracity.VERIFIED_TRUE.value: 0.0,
                    CanonicalVeracity.VERIFIED_FALSE.value: 0.0,
                    CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED.value: 0.0,
                    CanonicalVeracity.INSUFFICIENT_EVIDENCE.value: 1.0,
                },
                temporal_report=temporal_rep,
                evidence_used_count=0,
                duplicates_filtered=0,
                irrelevant_passages_filtered=0,
                evidence_breakdown=[],
                raw_prediction=CanonicalVeracity.INSUFFICIENT_EVIDENCE.value,
                has_meaningful_evidence=False,
            )

        # Extract structured claim components for compound claim coverage assessment
        claim_components = self.decompose_claim_components(claim_text)

        # Stage 1: Evidence Eligibility Assessment
        candidate_passages = self.stage1_assess_eligibility(claim_text, candidate_passages, claim_components)
        pipeline_trace = {
            "execution_order": [
                "stage1_assess_eligibility",
                "stage2_classify_stance",
                "stage3_aggregate_consensus",
                "stage4_resolve_verdict_policy",
                "stage5_synthesize_canonical_verdict",
            ],
            "stage1_eligibility": {
                "total_candidates": len(candidate_passages),
                "eligible_count": sum(1 for p in candidate_passages if p.get("is_decision_eligible")),
                "off_topic_count": sum(1 for p in candidate_passages if not p.get("is_relevant")),
                "uninformative_count": sum(1 for p in candidate_passages if not p.get("is_informative")),
                "unaligned_count": sum(1 for p in candidate_passages if not p.get("is_claim_aligned")),
                "scope_mismatch_count": sum(1 for p in candidate_passages if p.get("is_scope_mismatch")),
            },
        }

        # Multi-evidence complementary joint premise formulation
        joint_premise = None
        if len(candidate_passages) > 1:
            seen_texts = set()
            unique_premises = []
            for p in candidate_passages:
                t = (p.get("text") or "").strip()
                if (
                    t
                    and t not in seen_texts
                    and not p.get("is_figurative", False)
                    and p.get("is_informative", True)
                    and (p.get("is_relevant", True) or p.get("is_claim_aligned", True) or p.get("claim_components_covered"))
                ):
                    seen_texts.add(t)
                    unique_premises.append(t)
            if len(unique_premises) > 1:
                joint_premise = " ".join(unique_premises)

        # Stage 2: Stance Classification & Semantic Polarity
        nli_predictions, joint_pred = self.stage2_classify_stance(claim_text, candidate_passages, joint_premise)
        pipeline_trace["stage2_stance"] = {
            "support_count": sum(1 for p in candidate_passages if p.get("stance") == "support"),
            "refute_count": sum(1 for p in candidate_passages if p.get("stance") == "refute"),
            "neutral_count": sum(1 for p in candidate_passages if p.get("stance") == "neutral"),
            "joint_evaluated": joint_pred is not None,
        }

        # Stage 3: Evidence Consensus Aggregation
        agg_verdict = self.stage3_aggregate_consensus(candidate_passages, nli_predictions, joint_pred)
        pipeline_trace["stage3_consensus"] = {
            "unique_passages_used": agg_verdict.unique_passages_used,
            "duplicates_filtered": agg_verdict.duplicates_filtered,
            "has_cross_passage_disparity": agg_verdict.has_cross_passage_disparity,
            "raw_verdict": agg_verdict.canonical_verdict.value,
            "raw_confidence": round(agg_verdict.confidence, 4),
        }

        # Temporal Consistency Analysis
        temporal_report = self.temporal_analyzer.analyze(claim_date, candidate_passages)

        # Stage 4: Verdict Policy Resolution
        policy_resolution = self.stage4_resolve_verdict_policy(
            claim_text=claim_text,
            agg_verdict=agg_verdict,
            candidate_passages=candidate_passages,
            nli_predictions=nli_predictions,
            joint_pred=joint_pred,
            claim_components=claim_components,
            joint_premise=joint_premise,
        )
        pol_verd = policy_resolution["verdict"]
        pipeline_trace["stage4_verdict_policy"] = {
            "triggered_axiom": policy_resolution["triggered_axiom"],
            "policy_verdict": pol_verd.value if hasattr(pol_verd, "value") else str(pol_verd),
            "was_abstained": policy_resolution["was_abstained"],
            "abstention_reason": policy_resolution["abstention_reason"],
        }

        # Stage 5: Confidence Calibration, Selective Abstention & Canonical Synthesis
        raw_logits_np = np.array([agg_verdict.raw_logits], dtype=np.float32)
        return self.stage5_synthesize_canonical_verdict(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_date=claim_date,
            candidate_passages=candidate_passages,
            agg_verdict=agg_verdict,
            policy_resolution=policy_resolution,
            temporal_report=temporal_report,
            claim_components=claim_components,
            joint_pred=joint_pred,
            raw_logits_np=raw_logits_np,
            pipeline_trace=pipeline_trace,
        )

    def stage1_assess_eligibility(
        self,
        claim_text: str,
        candidate_passages: List[Dict[str, Any]],
        claim_components: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Evidence Eligibility Assessment.
        Evaluates candidate passages for topical subject relevance, informativeness,
        claim-relation alignment, figurative idiom check, and generalized threat-boundary/scope mismatch.
        Tags every candidate with explicit eligibility metadata and ineligibility reasons.
        """
        for p in candidate_passages:
            t = (p.get("text") or "").strip()
            if "is_figurative" not in p:
                p["is_figurative"] = self.is_figurative_or_lexical_definition(t)
            if "is_relevant" not in p:
                p["is_relevant"] = self.is_topically_relevant(claim_text, t)
            if "is_informative" not in p:
                is_inf, inf_reason = self.is_passage_informative(t, claim_text)
                p["is_informative"] = is_inf
                p["informative_reason"] = inf_reason
            if "is_claim_aligned" not in p:
                p["is_claim_aligned"] = self.has_claim_relation_alignment(claim_text, t)
            if "claim_coverage" not in p or "claim_components_covered" not in p:
                cov_level, cov_labels = self.assess_claim_coverage(claim_components, t)
                p["claim_coverage"] = cov_level
                p["claim_components_covered"] = cov_labels
            if "is_scope_mismatch" not in p:
                p["is_scope_mismatch"] = _is_generalized_scope_mismatch(claim_text, t)

            # Ineligibility reason determination
            ineligibility_reason = None
            if not p["is_relevant"]:
                ineligibility_reason = "off_topic"
            elif not p["is_informative"]:
                ineligibility_reason = p.get("informative_reason") or "uninformative_fragment"
            elif p.get("is_figurative", False):
                ineligibility_reason = "figurative_or_lexical_definition"
            elif not p["is_claim_aligned"]:
                ineligibility_reason = "unaligned_claim_relation"
            elif p.get("is_scope_mismatch", False):
                ineligibility_reason = "scope_boundary_mismatch"

            p["ineligibility_reason"] = ineligibility_reason

            if "is_decision_eligible" not in p:
                p["is_decision_eligible"] = bool(
                    p["is_relevant"]
                    and p["is_informative"]
                    and p["is_claim_aligned"]
                    and not p.get("is_figurative", False)
                )

        # Pairwise evaluation for defensible multi-hop bridge relationships
        for i in range(len(candidate_passages)):
            for j in range(i + 1, len(candidate_passages)):
                is_br, br_tokens = self.detect_valid_bridge(claim_text, candidate_passages[i], candidate_passages[j])
                if is_br:
                    candidate_passages[i]["has_valid_bridge"] = True
                    candidate_passages[j]["has_valid_bridge"] = True
                    candidate_passages[i].setdefault("bridge_tokens", []).extend(list(br_tokens))
                    candidate_passages[j].setdefault("bridge_tokens", []).extend(list(br_tokens))

        return candidate_passages

    def stage2_classify_stance(
        self,
        claim_text: str,
        candidate_passages: List[Dict[str, Any]],
        joint_premise: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Stage 2: Stance Classification & Semantic Polarity.
        Executes NLI batch predictions, applies causal adjustment, generalized threat-boundary
        scope mismatch normalization, and epistemic absence-of-evidence normalization.
        Sets canonical stance ('support', 'refute', 'neutral') on each candidate.
        """
        pairs = [(p.get("text", ""), claim_text) for p in candidate_passages]
        if joint_premise:
            pairs.append((joint_premise, claim_text))

        all_predictions = self.nli.predict_batch(pairs)

        if joint_premise and len(all_predictions) > len(candidate_passages):
            nli_predictions = list(all_predictions[:-1])
            joint_pred = all_predictions[-1]
        else:
            nli_predictions = list(all_predictions[:len(candidate_passages)])
            joint_pred = None

        while len(nli_predictions) < len(candidate_passages):
            nli_predictions.append({
                "contradiction": 0.0,
                "entailment": 0.0,
                "neutral": 1.0,
                "logits": [-5.0, -5.0, 5.0],
                "predicted_label": "NEUTRAL",
            })

        # Apply causal adjustment & scope mismatch adjustment
        for i, p in enumerate(candidate_passages):
            pt = p.get("text", "")
            nli_predictions[i] = self.adjust_causal_nli_prediction(claim_text, pt, nli_predictions[i])
            nli_predictions[i] = self.adjust_scope_mismatch_nli_prediction(claim_text, pt, nli_predictions[i])

        if joint_pred is not None and joint_premise:
            joint_pred = self.adjust_causal_nli_prediction(claim_text, joint_premise, joint_pred)
            joint_pred = self.adjust_scope_mismatch_nli_prediction(claim_text, joint_premise, joint_pred)

        claim_asserts_absence = bool(re.search(r"\b(?:no evidence|not found|unproven|unconfirmed|never)\b", claim_text, re.IGNORECASE))
        for i, p in enumerate(candidate_passages):
            pt = p.get("text", "")
            is_fig = bool(p.get("is_figurative", False))
            is_absence = self.is_absence_of_evidence(pt)
            if is_absence:
                p["is_absence_of_evidence"] = True

            if is_fig:
                nli_predictions[i] = {
                    "contradiction": 0.0,
                    "entailment": 0.0,
                    "neutral": 1.0,
                    "logits": [-5.0, -5.0, 5.0],
                    "predicted_label": "NEUTRAL",
                }
            elif is_absence and not claim_asserts_absence:
                orig_pred = nli_predictions[i]
                if float(orig_pred.get("contradiction", 0.0)) > float(orig_pred.get("entailment", 0.0)):
                    nli_predictions[i] = {
                        "contradiction": 0.0,
                        "entailment": 0.0,
                        "neutral": 1.0,
                        "logits": [-5.0, -5.0, 5.0],
                        "predicted_label": "NEUTRAL",
                    }

            pred = nli_predictions[i]
            ep = float(pred.get("entailment", 0.0))
            cp = float(pred.get("contradiction", 0.0))
            np_ = float(pred.get("neutral", 0.0))

            # Canonical stance assignment
            if p.get("is_scope_mismatch"):
                if cp > ep and cp >= 0.40 and cp >= np_:
                    stance = "refute"
                else:
                    stance = "neutral"
            elif ep >= cp and (ep >= np_ or ep >= 0.45):
                stance = "support"
            elif cp > ep and cp >= 0.40 and cp >= np_:
                stance = "refute"
            else:
                stance = "neutral"

            p["stance"] = stance
            p["entailment_prob"] = ep
            p["contradiction_prob"] = cp
            p["neutral_prob"] = np_

        return nli_predictions, joint_pred

    def stage3_aggregate_consensus(
        self,
        candidate_passages: List[Dict[str, Any]],
        nli_predictions: List[Dict[str, Any]],
        joint_nli_prediction: Optional[Dict[str, Any]] = None,
    ) -> AggregatedEvidenceVerdict:
        """
        Stage 3: Evidence Consensus Aggregation.
        Executes multi-evidence consensus aggregation, deduplication, and cross-passage
        stance disparity detection.
        """
        return self.aggregator.aggregate(
            candidate_passages=candidate_passages,
            nli_predictions=nli_predictions,
            joint_nli_prediction=joint_nli_prediction,
        )

    def stage4_resolve_verdict_policy(
        self,
        claim_text: str,
        agg_verdict: AggregatedEvidenceVerdict,
        candidate_passages: List[Dict[str, Any]],
        nli_predictions: List[Dict[str, Any]],
        joint_pred: Optional[Dict[str, Any]],
        claim_components: List[Dict[str, Any]],
        joint_premise: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stage 4: Verdict Policy Resolution.
        Applies explicit decision axioms:
        - Axiom 1: Constituent Refutation (refuted subclaim -> VERIFIED_FALSE)
        - Axiom 2: Constituent Conflict (conflicting subclaim -> CONFLICTING)
        - Axiom 3: Compound Completeness (all subclaims supported -> VERIFIED_TRUE)
        - Axiom 4: Constituent Insufficiency (unresolved subclaim -> INSUFFICIENT)
        - Axiom 5: Stance Disparity / Cross-passage conflict -> CONFLICTING
        - Axiom 6: Affirmative Grounding Guard (VERIFIED_TRUE requires >= 1 eligible supporter or joint support)
        - Axiom 7: Negative Refutation Guard (VERIFIED_FALSE requires >= 1 eligible refuter or constituent refutation)
        - Axiom 8: Essential Component Coverage (missing essential components -> INSUFFICIENT)
        - Axiom 9: Default Consensus Policy
        """
        subclaims = self.decomposer.decompose(claim_text)
        triggered_axiom = "Axiom 9: Default Consensus Policy"
        final_verdict = agg_verdict.canonical_verdict
        was_abstained = False
        abstain_reason = ""
        subclaim_evals: List[Dict[str, Any]] = []

        if len(subclaims) > 1:
            subclaim_pairs = []
            subclaim_pairs_meta = []

            for sc in subclaims:
                sc_text = sc.text
                sc_passages = []
                for p in candidate_passages:
                    pt = p.get("text", "")
                    if not pt or p.get("is_figurative", False) or not p.get("is_informative", True):
                        continue
                    if self.is_topically_relevant(sc_text, pt) or self.has_claim_relation_alignment(sc_text, pt):
                        sc_passages.append(p)

                if not sc_passages:
                    subclaim_evals.append({"subclaim": sc, "status": "unresolved"})
                else:
                    for p in sc_passages:
                        subclaim_pairs.append((p.get("text", ""), sc_text))
                        subclaim_pairs_meta.append((sc, p))

            if subclaim_pairs:
                try:
                    sc_preds = self.nli.predict_batch(subclaim_pairs)
                    if not isinstance(sc_preds, list) or len(sc_preds) != len(subclaim_pairs):
                        sc_preds = []
                        for sc, p in subclaim_pairs_meta:
                            idx = candidate_passages.index(p) if p in candidate_passages else 0
                            sc_preds.append(nli_predictions[idx])
                except Exception:
                    sc_preds = []
                    for sc, p in subclaim_pairs_meta:
                        idx = candidate_passages.index(p) if p in candidate_passages else 0
                        sc_preds.append(nli_predictions[idx])

                sc_group: Dict[str, Dict[str, bool]] = {sc.subclaim_id: {"has_support": False, "has_refute": False} for sc in subclaims}
                for (sc, p), pred in zip(subclaim_pairs_meta, sc_preds):
                    pt = p.get("text", "")
                    pred = self.adjust_causal_nli_prediction(sc.text, pt, pred)
                    if self.is_absence_of_evidence(pt) and not bool(re.search(r"\b(?:no evidence|not found|never)\b", sc.text, re.IGNORECASE)):
                        if float(pred.get("contradiction", 0.0)) > float(pred.get("entailment", 0.0)):
                            pred = {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0, "predicted_label": "NEUTRAL"}

                    ep = float(pred.get("entailment", 0.0))
                    cp = float(pred.get("contradiction", 0.0))
                    np_ = float(pred.get("neutral", 0.0))

                    if ep > cp and ep >= np_ and ep >= 0.35:
                        sc_group[sc.subclaim_id]["has_support"] = True
                    elif cp > ep and cp >= np_ and cp >= 0.35:
                        sc_group[sc.subclaim_id]["has_refute"] = True

                for sc in subclaims:
                    if sc.subclaim_id in sc_group:
                        grp = sc_group[sc.subclaim_id]
                        if grp["has_support"] and grp["has_refute"]:
                            subclaim_evals.append({"subclaim": sc, "status": "conflicting"})
                        elif grp["has_refute"]:
                            subclaim_evals.append({"subclaim": sc, "status": "refuted"})
                        elif grp["has_support"]:
                            subclaim_evals.append({"subclaim": sc, "status": "supported"})
                        else:
                            subclaim_evals.append({"subclaim": sc, "status": "unresolved"})

            refuted_scs = [e for e in subclaim_evals if e["status"] == "refuted"]
            conflicting_scs = [e for e in subclaim_evals if e["status"] == "conflicting"]
            unresolved_scs = [e for e in subclaim_evals if e["status"] == "unresolved"]
            supported_scs = [e for e in subclaim_evals if e["status"] == "supported"]

            if refuted_scs:
                final_verdict = CanonicalVeracity.VERIFIED_FALSE
                was_abstained = False
                abstain_reason = f"Constituent proposition '{refuted_scs[0]['subclaim'].text}' is refuted by documentary evidence."
                triggered_axiom = "Axiom 1: Constituent Refutation"
            elif conflicting_scs:
                final_verdict = CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED
                was_abstained = False
                abstain_reason = f"Constituent proposition '{conflicting_scs[0]['subclaim'].text}' has conflicting documentary evidence."
                triggered_axiom = "Axiom 2: Constituent Stance Conflict"
            elif len(supported_scs) == len(subclaims):
                final_verdict = CanonicalVeracity.VERIFIED_TRUE
                was_abstained = False
                triggered_axiom = "Axiom 3: Compound Completeness Supported"
            elif unresolved_scs:
                final_verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE
                was_abstained = True
                abstain_reason = f"Constituent proposition '{unresolved_scs[0]['subclaim'].text}' lacks documentary evidence. Partial evidence for other components does not validate the complete compound claim."
                triggered_axiom = "Axiom 4: Constituent Insufficiency"
            else:
                final_verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE
                was_abstained = True
                abstain_reason = "Constituent propositions lack documentary evidence in retrieved passages."
                triggered_axiom = "Axiom 4: Constituent Insufficiency"

        elif agg_verdict.canonical_verdict == CanonicalVeracity.VERIFIED_TRUE and len(claim_components) >= 2:
            supported_component_labels = set()
            for item in agg_verdict.evidence_breakdown:
                if item.get("is_decision_eligible") and item.get("stance") == "support":
                    supported_component_labels.update(item.get("claim_components_covered", []))
                elif (
                    item.get("is_informative", True)
                    and not item.get("is_figurative", False)
                    and item.get("stance") != "refute"
                    and (item.get("is_relevant", True) or item.get("claim_coverage") in ("full", "partial"))
                ):
                    supported_component_labels.update(item.get("claim_components_covered", []))

            if joint_pred is not None and float(joint_pred.get("entailment", 0.0)) > float(joint_pred.get("contradiction", 0.0)):
                if joint_premise:
                    _, j_cov_labels = self.assess_claim_coverage(claim_components, joint_premise)
                    supported_component_labels.update(j_cov_labels)

            essential_types = {"temporal", "location", "value"}
            missing_essential = [
                comp["label"] for comp in claim_components
                if comp.get("type") in essential_types and comp["label"] not in supported_component_labels
            ]
            if missing_essential:
                final_verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE
                was_abstained = True
                abstain_reason = f"Essential claim components remain unresolved: missing documentary evidence for {', '.join(missing_essential)}."
                triggered_axiom = "Axiom 8: Essential Component Missing"
            else:
                final_verdict = CanonicalVeracity.VERIFIED_TRUE
                was_abstained = False
                triggered_axiom = "Axiom 9: Consensus Verified True"
        else:
            final_verdict = agg_verdict.canonical_verdict
            was_abstained = (final_verdict == CanonicalVeracity.INSUFFICIENT_EVIDENCE)
            triggered_axiom = f"Axiom 9: Default Consensus ({agg_verdict.canonical_verdict.value})"

        # Affirmative Grounding Guard
        if final_verdict == CanonicalVeracity.VERIFIED_TRUE:
            has_eligible_support = any(
                item.get("stance") == "support"
                and item.get("is_decision_eligible", False)
                and not item.get("is_joint", False)
                for item in agg_verdict.evidence_breakdown
            )
            has_valid_bridge = any(
                p.get("has_valid_bridge", False) for p in candidate_passages
            )
            has_complementary_coverage = (
                (
                    len([p for p in candidate_passages if p.get("is_decision_eligible", False)]) >= 2
                    and any(p.get("claim_coverage") == "partial" for p in candidate_passages if p.get("is_decision_eligible", False))
                )
                or has_valid_bridge
            )
            has_joint_support = (
                joint_pred is not None
                and has_complementary_coverage
                and float(joint_pred.get("entailment", 0.0)) > float(joint_pred.get("contradiction", 0.0))
                and float(joint_pred.get("entailment", 0.0)) >= 0.50
            )
            if not has_eligible_support and not has_joint_support:
                final_verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE
                was_abstained = True
                neut_count = len([
                    e for e in agg_verdict.evidence_breakdown
                    if e.get("stance") == "neutral" and not e.get("is_joint", False)
                ])
                abstain_reason = (
                    f"Insufficient affirmative evidence retrieved: available passages do not contain documentary support for this proposition ({neut_count} neutral passage(s))."
                )
                triggered_axiom = "Axiom 6: Affirmative Grounding Guard"

        # Refutation Grounding Guard
        if final_verdict == CanonicalVeracity.VERIFIED_FALSE:
            has_eligible_contradiction = any(
                item.get("stance") == "refute"
                and item.get("is_decision_eligible", False)
                and not item.get("is_joint", False)
                for item in agg_verdict.evidence_breakdown
            )
            has_constituent_refutation = bool([e for e in subclaim_evals if e["status"] == "refuted"])

            if not has_eligible_contradiction and not has_constituent_refutation:
                final_verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE
                was_abstained = True
                neut_count = len([
                    e for e in agg_verdict.evidence_breakdown
                    if e.get("stance") == "neutral" and not e.get("is_joint", False)
                ])
                abstain_reason = (
                    f"Insufficient informative evidence retrieved: available passages do not contain documentary refutation or support for this proposition ({neut_count} neutral passage(s))."
                )
                triggered_axiom = "Axiom 7: Negative Refutation Guard"
            elif has_constituent_refutation and not abstain_reason:
                refuted_item = [e for e in subclaim_evals if e["status"] == "refuted"][0]
                abstain_reason = f"Constituent proposition '{refuted_item['subclaim'].text}' is refuted by documentary evidence."

        if final_verdict == CanonicalVeracity.INSUFFICIENT_EVIDENCE:
            was_abstained = True
            if not abstain_reason:
                neut_count = len([
                    e for e in agg_verdict.evidence_breakdown
                    if e.get("stance") == "neutral" and not e.get("is_joint", False)
                ])
                abstain_reason = (
                    f"Insufficient informative evidence retrieved: available passages do not contain documentary refutation or support for this proposition ({neut_count} neutral passage(s))."
                )

        return {
            "verdict": final_verdict,
            "was_abstained": was_abstained,
            "abstention_reason": abstain_reason,
            "triggered_axiom": triggered_axiom,
            "subclaim_evals": subclaim_evals,
        }

    def stage5_synthesize_canonical_verdict(
        self,
        claim_id: Optional[str],
        claim_text: str,
        claim_date: Optional[str],
        candidate_passages: List[Dict[str, Any]],
        agg_verdict: AggregatedEvidenceVerdict,
        policy_resolution: Dict[str, Any],
        temporal_report: TemporalReport,
        claim_components: List[Dict[str, Any]],
        joint_pred: Optional[Dict[str, Any]],
        raw_logits_np: np.ndarray,
        pipeline_trace: Dict[str, Any],
    ) -> ForensicVerdict:
        """
        Stage 5: Confidence Calibration, Selective Abstention & Canonical Synthesis.
        Applies temperature calibration, evaluates selective abstention, constructs canonical
        evidence counts and deterministic reasoning, and assembles the ForensicVerdict.
        """
        calibrated_probs = self.calibrator.calibrate(raw_logits_np)[0]
        prob_dict = {
            self.LABEL_NAMES[i]: float(calibrated_probs[i])
            for i in range(len(self.LABEL_NAMES))
        }

        has_meaningful_ev = self.check_evidence_grounding(
            agg_verdict.evidence_breakdown, joint_pred
        )
        should_abstain, abst_reason = self.abstainer.should_abstain(
            calibrated_probs, has_meaningful_evidence=has_meaningful_ev
        )

        final_verdict = policy_resolution["verdict"]
        was_abstained = policy_resolution["was_abstained"]
        final_abstain_reason = policy_resolution["abstention_reason"]

        if should_abstain:
            final_verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE
            was_abstained = True
            if not final_abstain_reason or "Constituent proposition" not in final_abstain_reason:
                if abst_reason:
                    final_abstain_reason = abst_reason
        elif final_verdict == CanonicalVeracity.INSUFFICIENT_EVIDENCE:
            was_abstained = True
            if not final_abstain_reason:
                final_abstain_reason = "Insufficient informative evidence mass retrieved to adjudicate this claim."

        uncalib_conf = agg_verdict.confidence
        final_conf = float(np.max(calibrated_probs))

        if temporal_report.confidence_penalty > 0:
            final_conf = max(0.1, final_conf - temporal_report.confidence_penalty)

        # Build canonical evidence counts
        total_ev = len(candidate_passages)
        sup_count = len([p for p in candidate_passages if p.get("is_decision_eligible", False) and p.get("stance") == "support"])
        ref_count = len([p for p in candidate_passages if p.get("is_decision_eligible", False) and p.get("stance") == "refute"])
        neut_count = len([p for p in candidate_passages if p.get("is_relevant", True) and p.get("stance") == "neutral"])
        partial_count = len([p for p in candidate_passages if p.get("is_relevant", True) and p.get("claim_coverage") == "partial"])
        off_topic_count = len([p for p in candidate_passages if not p.get("is_relevant", True)])

        evidence_counts = {
            "total": total_ev,
            "eligible_supporting": sup_count,
            "eligible_refuting": ref_count,
            "relevant_neutral": neut_count,
            "relevant_partial": partial_count,
            "off_topic": off_topic_count,
            "duplicates": agg_verdict.duplicates_filtered,
        }

        conf_pct = final_conf * 100.0

        # Section 5: Final Verdict–Evidence Consistency Validator
        has_valid_bridge = any(p.get("has_valid_bridge", False) for p in candidate_passages)
        has_decisive_joint = (
            has_valid_bridge
            and joint_pred is not None
            and float(joint_pred.get("entailment", 0.0)) >= 0.50
            and float(joint_pred.get("entailment", 0.0)) > float(joint_pred.get("contradiction", 0.0))
        )
        has_constituent_refutation = bool([e for e in policy_resolution.get("subclaim_evals", []) if e.get("status") == "refuted"])

        if final_verdict == CanonicalVeracity.VERIFIED_TRUE:
            if sup_count == 0 and not has_decisive_joint:
                final_verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE
                was_abstained = True
                final_abstain_reason = "Consistency violation: VERIFIED_TRUE requires eligible supporting documentary evidence or a verified interlocking bridge."
        elif final_verdict == CanonicalVeracity.VERIFIED_FALSE:
            if ref_count == 0 and not has_constituent_refutation:
                final_verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE
                was_abstained = True
                final_abstain_reason = "Consistency violation: VERIFIED_FALSE requires eligible refuting documentary evidence or refuted constituent proposition."
        elif final_verdict == CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED:
            if sup_count == 0 or ref_count == 0:
                has_subclaim_conflict = bool([e for e in policy_resolution.get("subclaim_evals", []) if e.get("status") == "conflicting"])
                has_any_opposing = (
                    any(p.get("stance") == "support" for p in candidate_passages)
                    and any(p.get("stance") == "refute" for p in candidate_passages)
                )
                if not has_subclaim_conflict and not has_any_opposing:
                    final_verdict = CanonicalVeracity.INSUFFICIENT_EVIDENCE
                    was_abstained = True
                    final_abstain_reason = "Consistency violation: CONFLICTING requires documented opposing stances across eligible evidence."

        # Construct canonical reasoning
        if final_verdict == CanonicalVeracity.CONFLICTING_OR_CHERRYPICKED:
            if sup_count > 0 and ref_count > 0:
                canonical_reason = f"Cross-passage stance disparity detected: {sup_count} supporting passage(s) and {ref_count} refuting passage(s) present contradictory documentary evidence (calibrated confidence {conf_pct:.1f}%)."
            else:
                canonical_reason = f"Cross-passage stance disparity detected: opposing evidence positions were documented across retrieved passages (calibrated confidence {conf_pct:.1f}%)."
        elif final_verdict == CanonicalVeracity.INSUFFICIENT_EVIDENCE:
            if off_topic_count == total_ev and total_ev > 0:
                canonical_reason = f"No relevant evidence was found in the corpus ({off_topic_count} off-topic passage(s) excluded). No factual conclusion was established."
            elif not has_meaningful_ev:
                canonical_reason = f"Abstained — Insufficient Evidentiary Grounding ({final_abstain_reason}). Retrieved passages lack supporting or refuting evidence for the asserted proposition."
            else:
                canonical_reason = final_abstain_reason or f"Insufficient informative evidence mass retrieved to adjudicate this claim ({neut_count} of {total_ev} passage(s) neutral or uninformative)."
        elif final_verdict == CanonicalVeracity.VERIFIED_TRUE:
            if sup_count > 0:
                canonical_reason = f"Evidence supports the claim with {sup_count} supporting passage(s) (calibrated confidence {conf_pct:.1f}%)."
            elif has_decisive_joint:
                bridge_count = len([p for p in candidate_passages if p.get("has_valid_bridge")])
                canonical_reason = f"Evidence jointly supports the claim via {bridge_count} interlocking bridge passages (calibrated confidence {conf_pct:.1f}%)."
            else:
                canonical_reason = f"Evidence supports the claim (calibrated confidence {conf_pct:.1f}%)."
        elif final_verdict == CanonicalVeracity.VERIFIED_FALSE:
            if ref_count > 0:
                canonical_reason = f"Evidence refutes the claim with {ref_count} refuting passage(s) (calibrated confidence {conf_pct:.1f}%)."
            elif final_abstain_reason and "Constituent proposition" in final_abstain_reason:
                canonical_reason = f"{final_abstain_reason} (calibrated confidence {conf_pct:.1f}%)."
            else:
                canonical_reason = f"Evidence refutes the claim (calibrated confidence {conf_pct:.1f}%)."
        else:
            canonical_reason = "Adjudication completed by VERITAS-X verification pipeline."

        # Finalize pipeline trace
        pipeline_trace["stage5_synthesis"] = {
            "final_verdict": final_verdict.value,
            "calibrated_confidence": round(final_conf, 4),
            "was_abstained": was_abstained,
            "canonical_reason": canonical_reason,
            "evidence_counts": evidence_counts,
        }

        return ForensicVerdict(
            claim_id=claim_id,
            claim_text=claim_text,
            verdict=final_verdict,
            confidence=final_conf,
            uncalibrated_confidence=uncalib_conf,
            was_abstained=was_abstained,
            abstention_reason=final_abstain_reason,
            class_probabilities=prob_dict,
            temporal_report=temporal_report,
            evidence_used_count=agg_verdict.unique_passages_used,
            duplicates_filtered=agg_verdict.duplicates_filtered,
            irrelevant_passages_filtered=agg_verdict.irrelevant_passages_filtered,
            evidence_breakdown=agg_verdict.evidence_breakdown,
            raw_prediction=agg_verdict.canonical_verdict.value,
            has_meaningful_evidence=has_meaningful_ev,
            claim_components=claim_components,
            canonical_reason=canonical_reason,
            evidence_counts=evidence_counts,
            pipeline_trace=pipeline_trace,
        )
