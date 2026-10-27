"""
Structured Claim Extraction Module for VERITAS-X.

Extracts structured propositions from user claims and fact-checking statements:
- Grammatical and logical Subject, Predicate, and Object.
- Negation detection and explicit scoping (predicate vs universal vs existential).
- Quantifier classification (universal: 'all', 'every'; existential: 'some', 'few'; negative: 'none', 'no'; frequency: 'always', 'never').
- Epistemic modality and certainty scale (CERTAIN, PROBABLE, POSSIBLE, DOUBTFUL, ALLEGED).
- Temporal expressions (dates, years, days of the week).
- Numeric measurements, percentages, and scientific units.
- Named entities (proper nouns and recognized multi-token entities).
- Conditional propositions (antecedent / condition -> consequent).
- Causal propositions (cause -> effect relationships).
- Question versus assertion intent.
- Compound proposition tree preserving coordinating, subordinating, and causal relations.
- Unresolved anaphora / dangling pronouns flagging.
"""
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import List, Dict, Any, Optional, Set, Tuple

from .linguistics import (
    LinguisticAnalyzer,
    LinguisticAnalysisResult,
    AnalyzedSentence,
    LinguisticToken,
    Clause
)


class ModalityLevel(str, Enum):
    CERTAIN = "CERTAIN"          # Asserted as definite fact: "is", "will", "confirmed"
    PROBABLE = "PROBABLE"        # High likelihood: "probably", "likely", "should"
    POSSIBLE = "POSSIBLE"        # Epistemic possibility: "may", "might", "could", "possibly"
    ALLEGED = "ALLEGED"          # Attributed / unverified claim: "allegedly", "reportedly", "claimed"
    DOUBTFUL = "DOUBTFUL"        # Skeptical / uncertain: "unlikely", "doubtful", "alleged"


class QuantifierType(str, Enum):
    UNIVERSAL = "UNIVERSAL"      # "all", "every", "each"
    EXISTENTIAL = "EXISTENTIAL"  # "some", "any", "few", "many", "several"
    NEGATIVE = "NEGATIVE"        # "none", "no", "neither"
    PARTIAL_NEGATION = "PARTIAL" # "not all"
    FREQUENCY = "FREQUENCY"      # "always", "sometimes", "never", "often", "rarely"


@dataclass
class QuantifierItem:
    text: str
    qtype: QuantifierType
    target_scope: str            # "subject", "predicate", "object", "temporal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.qtype.value,
            "target_scope": self.target_scope,
        }


@dataclass
class NumericEntity:
    raw_text: str
    value: Optional[float]
    unit: Optional[str]
    category: str                # "number", "percentage", "temperature", "measurement", "currency"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "value": self.value,
            "unit": self.unit,
            "category": self.category,
        }


@dataclass
class TemporalEntity:
    raw_text: str
    temporal_type: str           # "date", "year", "day_of_week", "relative"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "temporal_type": self.temporal_type,
        }


@dataclass
class SubProposition:
    """An individual proposition extracted from a compound sentence."""
    prop_id: str
    text: str
    relation_to_parent: str      # "root", "conjunction", "adversative", "contrast", "cause", "condition", "attribution"
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    is_negated: bool = False
    is_passive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prop_id": self.prop_id,
            "text": self.text,
            "relation_to_parent": self.relation_to_parent,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "is_negated": self.is_negated,
            "is_passive": self.is_passive,
        }


@dataclass
class StructuredClaim:
    """
    Complete, typed structured claim representation for verification and guardrails.
    """
    raw_text: str
    sentence_type: str                        # "assertion", "question", "fragment"
    is_question: bool = False
    is_incomplete: bool = False
    is_negated: bool = False
    negation_tokens: List[str] = field(default_factory=list)
    negation_scope: str = "none"              # "predicate", "subject", "entire_proposition", "none"
    
    # Core Predicate-Argument Structure
    subject: Optional[str] = None
    subject_lemma: Optional[str] = None
    predicate: Optional[str] = None
    predicate_lemma: Optional[str] = None
    object: Optional[str] = None
    object_lemma: Optional[str] = None
    is_passive: bool = False
    logical_subject: Optional[str] = None     # Agent in passive ("...by the US Senate")

    # Epistemic & Semantic Annotations
    modality: ModalityLevel = ModalityLevel.CERTAIN
    modality_markers: List[str] = field(default_factory=list)
    quantifiers: List[QuantifierItem] = field(default_factory=list)
    named_entities: List[str] = field(default_factory=list)
    numeric_entities: List[NumericEntity] = field(default_factory=list)
    temporal_entities: List[TemporalEntity] = field(default_factory=list)

    # Complex Logical Relations
    is_conditional: bool = False
    condition_clause: Optional[str] = None
    consequence_clause: Optional[str] = None
    is_causal: bool = False
    causal_marker: Optional[str] = None
    cause_span: Optional[str] = None
    effect_span: Optional[str] = None
    
    # Compound Structure
    is_compound: bool = False
    sub_propositions: List[SubProposition] = field(default_factory=list)
    unresolved_references: List[str] = field(default_factory=list)

    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "sentence_type": self.sentence_type,
            "is_question": self.is_question,
            "is_incomplete": self.is_incomplete,
            "is_negated": self.is_negated,
            "negation_tokens": self.negation_tokens,
            "negation_scope": self.negation_scope,
            "subject": self.subject,
            "subject_lemma": self.subject_lemma,
            "predicate": self.predicate,
            "predicate_lemma": self.predicate_lemma,
            "object": self.object,
            "object_lemma": self.object_lemma,
            "is_passive": self.is_passive,
            "logical_subject": self.logical_subject,
            "modality": self.modality.value,
            "modality_markers": self.modality_markers,
            "quantifiers": [q.to_dict() for q in self.quantifiers],
            "named_entities": self.named_entities,
            "numeric_entities": [n.to_dict() for n in self.numeric_entities],
            "temporal_entities": [t.to_dict() for t in self.temporal_entities],
            "is_conditional": self.is_conditional,
            "condition_clause": self.condition_clause,
            "consequence_clause": self.consequence_clause,
            "is_causal": self.is_causal,
            "causal_marker": self.causal_marker,
            "cause_span": self.cause_span,
            "effect_span": self.effect_span,
            "is_compound": self.is_compound,
            "sub_propositions": [p.to_dict() for p in self.sub_propositions],
            "unresolved_references": self.unresolved_references,
            "confidence": round(self.confidence, 3),
        }


# ==============================================================================
# Claim Extractor Engine
# ==============================================================================

class ClaimExtractor:
    """
    Extracts structured propositions, logical operators, and semantic boundaries from claims.
    """

    # Causal markers regex
    CAUSAL_MARKER_REGEX = re.compile(
        r"\b(causes?|causing|caused|leads? to|leading to|induces?|inducing|induced|results? in|resulting in|due to|because of)\b",
        re.IGNORECASE
    )

    # Attribution / reporting verbs
    REPORTING_VERB_REGEX = re.compile(
        r"\b(announced that|reported that|claimed that|stated that|confirmed that|said that|revealed that)\b",
        re.IGNORECASE
    )

    # Date regexes
    DATE_PATTERNS = [
        (re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s+\d{4})?\b", re.IGNORECASE), "date"),
        (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "date"),
        (re.compile(r"\b(?:in|during|since|before|after)\s+(\d{4})\b", re.IGNORECASE), "year"),
        (re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE), "day_of_week"),
    ]

    # Numeric & Unit patterns
    NUMERIC_PATTERNS = [
        (re.compile(r"\b([+-]?\d+(?:\.\d+)?)\s*(°[cCfFkK]|degrees?\s+(?:Celsius|Fahrenheit|Centigrade|Kelvin))\b", re.IGNORECASE), "temperature"),
        (re.compile(r"\b(\d+(?:\.\d+)?)\s*%(?!\w)"), "percentage"),
        (re.compile(r"\b(\d+(?:,\d+)*(?:\.\d+)?)\s*(atm|bar|pascal|kpa|psi)\b", re.IGNORECASE), "measurement"),
        (re.compile(r"\b(\d+(?:,\d+)*(?:\.\d+)?)\s*(meters?|kilometers?|miles?|feet|inches|grams?|kilograms?|liters?)\b", re.IGNORECASE), "measurement"),
        (re.compile(r"\b([$₹€£]\s*\d+(?:,\d+)*(?:\.\d+)?)\b"), "currency"),
        (re.compile(r"\b(\d+(?:,\d+)*(?:\.\d+)?)\b"), "number"),
    ]

    @classmethod
    def extract(cls, text: Optional[str]) -> StructuredClaim:
        """
        Extracts structured claim representation using linguistic analysis and semantic parsing.
        """
        raw = text.strip() if text is not None else ""
        if not raw:
            return StructuredClaim(
                raw_text="",
                sentence_type="fragment",
                is_incomplete=True,
                confidence=1.0,
            )

        # 1. Perform underlying linguistic analysis
        analysis_res = LinguisticAnalyzer.analyze(raw)
        if not analysis_res.sentences:
            return StructuredClaim(
                raw_text=raw,
                sentence_type="fragment",
                is_incomplete=True,
                confidence=1.0,
            )

        sent = analysis_res.sentences[0]
        tokens = sent.tokens

        # 2. Extract Polarity, Negation, and Scope
        is_negated, neg_tokens, neg_scope = cls._extract_negation(tokens, raw)

        # 3. Extract Quantifiers
        quantifiers = cls._extract_quantifiers(tokens, raw)

        # 4. Extract Epistemic Modality
        modality, modal_markers = cls._extract_modality(tokens)

        # 5. Extract Entities, Numbers, and Temporal Markers
        named_entities = cls._extract_named_entities(tokens)
        numeric_entities = cls._extract_numeric_entities(raw)
        temporal_entities = cls._extract_temporal_entities(raw)

        # 6. Extract Conditional Relationships
        is_cond, cond_clause, cons_clause = cls._extract_conditionals(raw, tokens, sent.clauses)

        # 7. Extract Causal Relationships
        is_causal, causal_marker, cause_span, effect_span = cls._extract_causality(raw, tokens)

        # 8. Check for Unresolved References (dangling pronouns)
        unresolved_refs = cls._extract_unresolved_references(tokens)

        # 9. Extract Compound Sub-Propositions
        is_compound, sub_props = cls._extract_compound_propositions(raw, sent.clauses)

        # 10. Extract Logical Subject in Passives
        logical_subject = None
        if sent.is_passive:
            # Check for prepositional 'by [agent]'
            for i, t in enumerate(tokens):
                if t.dep_rel == "agent":
                    # Collect agent noun phrase
                    agent_tokens = [tokens[j].text for j in range(i, len(tokens)) if tokens[j].upos in ("NOUN", "PROPN", "DET", "ADJ")]
                    if agent_tokens:
                        logical_subject = " ".join(agent_tokens)
                    break

        return StructuredClaim(
            raw_text=raw,
            sentence_type=sent.sentence_type,
            is_question=sent.is_question,
            is_incomplete=sent.is_incomplete,
            is_negated=is_negated,
            negation_tokens=neg_tokens,
            negation_scope=neg_scope,
            subject=sent.subject_candidate,
            subject_lemma=cls._get_lemma_for_span(tokens, sent.subject_candidate),
            predicate=sent.predicate_candidate,
            predicate_lemma=cls._get_lemma_for_span(tokens, sent.predicate_candidate),
            object=sent.object_candidate,
            object_lemma=cls._get_lemma_for_span(tokens, sent.object_candidate),
            is_passive=sent.is_passive,
            logical_subject=logical_subject,
            modality=modality,
            modality_markers=modal_markers,
            quantifiers=quantifiers,
            named_entities=named_entities,
            numeric_entities=numeric_entities,
            temporal_entities=temporal_entities,
            is_conditional=is_cond,
            condition_clause=cond_clause,
            consequence_clause=cons_clause,
            is_causal=is_causal,
            causal_marker=causal_marker,
            cause_span=cause_span,
            effect_span=effect_span,
            is_compound=is_compound,
            sub_propositions=sub_props,
            unresolved_references=unresolved_refs,
            confidence=0.95 if not sent.is_incomplete else 0.70,
        )

    # --------------------------------------------------------------------------
    # Extraction Helpers
    # --------------------------------------------------------------------------
    @classmethod
    def _extract_negation(
        cls, tokens: List[LinguisticToken], raw_text: str
    ) -> Tuple[bool, List[str], str]:
        """Detects presence, tokens, and logical scope of negation."""
        neg_tokens = []
        for t in tokens:
            if t.dep_rel == "neg" or t.normalized_text in ("not", "never", "n't", "neither", "nor", "cannot", "without"):
                neg_tokens.append(t.text)

        if not neg_tokens:
            # Check raw regex fallback (e.g. w/o, unpunctuated contractions, typos)
            m = re.findall(
                r"\b(w/o|no|none|dont|doesnt|dosent|didnt|cant|wont|isnt|arent|wasnt|werent|hasnt|havent|hadnt|wouldnt|shouldnt|couldnt|noot)\b|n't\b",
                raw_text,
                re.IGNORECASE,
            )
            if m:
                neg_tokens.extend(m)

        if not neg_tokens:
            return False, [], "none"

        # Determine scope:
        low = raw_text.lower()
        if "not all" in low:
            scope = "existential"  # "not all" = some are not
        elif re.search(r"\b(?:no|none|neither)\b", low):
            scope = "universal"    # "no X does Y" = for all X, not Y
        elif any(
            t.lower() in (
                "not", "n't", "never", "cannot", "dont", "doesnt", "dosent", "didnt",
                "cant", "wont", "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
                "hadnt", "wouldnt", "shouldnt", "couldnt", "noot"
            )
            for t in neg_tokens
        ):
            scope = "predicate"    # "does not prevent" = negates the predicate
        else:
            scope = "entire_proposition"

        return True, neg_tokens, scope

    @classmethod
    def _extract_quantifiers(
        cls, tokens: List[LinguisticToken], raw_text: str
    ) -> List[QuantifierItem]:
        """Extracts universal, existential, negative, and frequency quantifiers."""
        items: List[QuantifierItem] = []
        low = raw_text.lower()

        # Check multi-word quantifiers first: e.g. "not all"
        if "not all" in low:
            items.append(QuantifierItem(text="not all", qtype=QuantifierType.PARTIAL_NEGATION, target_scope="subject"))

        for t in tokens:
            w = t.normalized_text
            if w in ("all", "every", "each") and "not all" not in low:
                scope = "subject" if t.head_index <= len(tokens) // 2 else "object"
                items.append(QuantifierItem(text=t.text, qtype=QuantifierType.UNIVERSAL, target_scope=scope))
            elif w in ("some", "any", "few", "many", "several", "most"):
                scope = "subject" if t.head_index <= len(tokens) // 2 else "object"
                items.append(QuantifierItem(text=t.text, qtype=QuantifierType.EXISTENTIAL, target_scope=scope))
            elif w in ("none", "no", "neither"):
                items.append(QuantifierItem(text=t.text, qtype=QuantifierType.NEGATIVE, target_scope="subject"))
            elif w in ("always", "never", "sometimes", "often", "rarely", "seldom"):
                items.append(QuantifierItem(text=t.text, qtype=QuantifierType.FREQUENCY, target_scope="temporal"))

        return items

    @classmethod
    def _extract_modality(
        cls, tokens: List[LinguisticToken]
    ) -> Tuple[ModalityLevel, List[str]]:
        """Extracts modal auxiliaries and epistemic adverbs into certainty ratings."""
        markers: List[str] = []
        level = ModalityLevel.CERTAIN

        for t in tokens:
            low = t.normalized_text
            if low in ("allegedly", "reportedly", "supposedly", "purportedly", "claimed"):
                markers.append(t.text)
                level = ModalityLevel.ALLEGED
            elif low in ("might", "may", "could", "possibly", "potential"):
                markers.append(t.text)
                if level != ModalityLevel.ALLEGED:
                    level = ModalityLevel.POSSIBLE
            elif low in ("probably", "likely", "should", "expected"):
                markers.append(t.text)
                if level not in (ModalityLevel.ALLEGED, ModalityLevel.POSSIBLE):
                    level = ModalityLevel.PROBABLE
            elif low in ("unlikely", "doubtful", "improbable"):
                markers.append(t.text)
                level = ModalityLevel.DOUBTFUL

        return level, markers

    @classmethod
    def _extract_named_entities(cls, tokens: List[LinguisticToken]) -> List[str]:
        """Extracts capitalized proper noun phrases and acronyms."""
        entities: List[str] = []
        cur_entity_tokens: List[str] = []

        for i, t in enumerate(tokens):
            is_initial = (i == 0)
            is_acronym = (t.text.isupper() and len(t.text) >= 2)
            is_propn = (t.upos == "PROPN" or t.pos == "NNP" or is_acronym)
            # Protect proper nouns not at sentence start, uppercase acronyms, or multi-word capitalized titles
            if is_propn and (not is_initial or is_acronym or (i + 1 < len(tokens) and tokens[i + 1].pos == "NNP")):
                cur_entity_tokens.append(t.text)
            else:
                if cur_entity_tokens:
                    ent_str = " ".join(cur_entity_tokens)
                    if ent_str not in entities:
                        entities.append(ent_str)
                    cur_entity_tokens = []

        if cur_entity_tokens:
            ent_str = " ".join(cur_entity_tokens)
            if ent_str not in entities:
                entities.append(ent_str)

        return entities

    @classmethod
    def _extract_numeric_entities(cls, text: str) -> List[NumericEntity]:
        """Extracts numbers, percentages, temperatures, and units."""
        entities: List[NumericEntity] = []
        seen_spans: Set[str] = set()

        for pattern, cat in cls.NUMERIC_PATTERNS:
            for m in pattern.finditer(text):
                full = m.group(0).strip()
                if full in seen_spans:
                    continue
                seen_spans.add(full)

                # Parse float value if possible
                val_str = m.group(1).replace(",", "").replace("$", "").replace("₹", "").replace("€", "").replace("£", "")
                try:
                    num_val = float(val_str)
                except ValueError:
                    num_val = None

                unit = m.group(2) if len(m.groups()) >= 2 else None
                entities.append(NumericEntity(
                    raw_text=full,
                    value=num_val,
                    unit=unit,
                    category=cat
                ))

        return entities

    @classmethod
    def _extract_temporal_entities(cls, text: str) -> List[TemporalEntity]:
        """Extracts dates, years, and days of week."""
        entities: List[TemporalEntity] = []
        seen: Set[str] = set()

        for pattern, ttype in cls.DATE_PATTERNS:
            for m in pattern.finditer(text):
                raw_m = m.group(0).strip()
                if raw_m in seen:
                    continue
                seen.add(raw_m)
                entities.append(TemporalEntity(
                    raw_text=raw_m,
                    temporal_type=ttype
                ))

        return entities

    @classmethod
    def _extract_conditionals(
        cls, raw_text: str, tokens: List[LinguisticToken], clauses: List[Clause]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Identifies conditional relations (if X then Y / unless X)."""
        m = re.match(r"^\s*if\s+(.*?)\s*,?\s*then\s+(.*)$", raw_text, re.IGNORECASE)
        if m:
            return True, m.group(1).strip(), m.group(2).strip()

        m2 = re.match(r"^\s*if\s+(.*?)\s*,\s*(.*)$", raw_text, re.IGNORECASE)
        if m2:
            return True, m2.group(1).strip(), m2.group(2).strip()

        # Check for conditional clause
        for c in clauses:
            if c.clause_type == "conditional":
                return True, c.text, None

        return False, None, None

    @classmethod
    def _extract_causality(
        cls, raw_text: str, tokens: List[LinguisticToken]
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """Identifies causal assertions (cause -> effect)."""
        m = cls.CAUSAL_MARKER_REGEX.search(raw_text)
        if m:
            marker = m.group(0).strip()
            start, end = m.start(), m.end()
            cause = raw_text[:start].strip()
            effect = raw_text[end:].strip()
            if cause and effect:
                return True, marker, cause, effect

        return False, None, None, None

    @classmethod
    def _extract_unresolved_references(cls, tokens: List[LinguisticToken]) -> List[str]:
        """Identifies dangling or ambiguous third-person pronouns lacking explicit antecedent."""
        unresolved = []
        pronouns_to_check = {"it", "its", "they", "them", "their", "this", "that", "these", "those", "he", "him", "she", "her"}
        for i, t in enumerate(tokens):
            if t.normalized_text in pronouns_to_check and t.upos == "PRON":
                # If pronoun occurs at start of sentence or without earlier nominal in same clause
                has_prior_noun = any(tokens[j].upos in ("NOUN", "PROPN") for j in range(i))
                if not has_prior_noun or i == 0:
                    unresolved.append(t.text)

        return unresolved

    @classmethod
    def _extract_compound_propositions(
        cls, raw_text: str, clauses: List[Clause]
    ) -> Tuple[bool, List[SubProposition]]:
        """Extracts structured sub-propositions while preserving coordinating and causal relationships."""
        if len(clauses) <= 1:
            return False, []

        sub_props: List[SubProposition] = []
        for idx, c in enumerate(clauses, start=1):
            sub_props.append(SubProposition(
                prop_id=f"P{idx}",
                text=c.text,
                relation_to_parent=c.clause_type,
                subject=c.subject,
                predicate=c.predicate,
                object=c.object,
                is_negated=c.is_negated,
                is_passive=c.is_passive,
            ))

        return True, sub_props

    @staticmethod
    def _get_lemma_for_span(tokens: List[LinguisticToken], span: Optional[str]) -> Optional[str]:
        """Finds the base lemma sequence for a text span."""
        if not span:
            return None
        span_words = span.lower().split()
        lemmas = []
        for w in span_words:
            matched_tok = next((t for t in tokens if t.normalized_text == w), None)
            lemmas.append(matched_tok.lemma if matched_tok else w)
        return " ".join(lemmas)
