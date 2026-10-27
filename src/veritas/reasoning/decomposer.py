"""
Claim Decomposition Component for VERITAS-X 2.0.
Extracts atomic propositions, traceable text spans, and reasoning dependencies
from complex and compound claims locally and deterministically.
Does NOT rely on external LLM APIs.
"""
import re
from typing import List, Tuple, Dict, Any, Optional
from veritas.reasoning.types import SubClaim, SubClaimStatus


class ClaimDecomposer:
    """
    Decomposes compound sentences into atomic, testable subclaims.
    Identifies syntactic coordinators, reporting clauses, causal clauses,
    and resolves coreferences for targeted downstream retrieval.
    """

    REPORTING_VERBS = [
        "announced that", "reported that", "claimed that", "stated that",
        "confirmed that", "declared that", "alleged that", "said that",
        "revealed that", "admitted that"
    ]

    COORDINATORS = [
        (r"\s*,\s*and\s+|\s+and\s+(?=(?:the|a|an|it|they|he|she|this|that|[a-z]+(?:ed|es|s))\s+)", "conjunct"),
        (r"\s*;\s*and\s+", "conjunct"),
        (r"\s*;\s*", "conjunct"),
        (r"\s*,\s*but\s+|\s+but\s+(?=(?:the|a|an|it|they|he|she|this|that)\s+)", "adversative"),
        (r"\s*,\s*yet\s+", "adversative"),
        (r"\s*,\s*while\s+", "contrast"),
        (r"\s*,\s*whereas\s+", "contrast"),
        (r"\s*,\s*although\s+", "concession"),
        (r"\s*,\s*after\s+which\s+", "temporal"),
        (r"\s*,\s*subsequently\s+", "temporal"),
    ]

    def decompose(self, claim_text: str) -> List[SubClaim]:
        """
        Decomposes a claim into atomic subclaims.
        If atomic or simple, returns a single subclaim.
        """
        raw_text = claim_text.strip()
        if not raw_text:
            return []

        # 1. Attempt rule-based and syntactic multi-clause decomposition
        clauses = self._extract_propositions(raw_text)

        # 2. If no multi-clause pattern matches or only 1 clause found, return single subclaim
        if len(clauses) <= 1:
            return [
                SubClaim(
                    subclaim_id="SC-01",
                    text=raw_text,
                    span=(0, len(raw_text)),
                    dependencies=[],
                    relation_to_parent="root",
                    status=SubClaimStatus.UNRESOLVED,
                )
            ]

        # 3. Build structured SubClaim objects with dependencies
        subclaims: List[SubClaim] = []
        subject_cache = self._extract_subject(clauses[0][0])

        for idx, (prop_text, span, relation, deps) in enumerate(clauses, start=1):
            sc_id = f"SC-{idx:02d}"
            
            # Resolve pronoun coreference if needed for standalone retrieval
            resolved_text = self._resolve_anaphora(prop_text, subject_cache)

            # If this clause introduces a new explicit subject, update subject_cache
            curr_subj = self._extract_subject(prop_text)
            if curr_subj and not self._is_pronoun(curr_subj):
                subject_cache = curr_subj

            subclaims.append(
                SubClaim(
                    subclaim_id=sc_id,
                    text=resolved_text,
                    span=span,
                    dependencies=deps,
                    relation_to_parent=relation,
                    status=SubClaimStatus.UNRESOLVED,
                )
            )

        return subclaims

    def _extract_propositions(self, text: str) -> List[Tuple[str, Tuple[int, int], str, List[str]]]:
        """
        Breaks text hierarchically into atomic propositions with spans, relations, and dependency IDs.
        """
        # Step 1: Split on coordinating conjunctions (', and ', ', but ', '; ', etc.)
        for regex, relation in self.COORDINATORS:
            parts = re.split(regex, text, flags=re.IGNORECASE)
            if len(parts) > 1:
                sub_propositions: List[Tuple[str, Tuple[int, int], str, List[str]]] = []
                cur_pos = 0
                for part in parts:
                    clean_part = part.strip().rstrip(".?! ")
                    if not clean_part:
                        continue
                    start_idx = text.find(clean_part, cur_pos)
                    end_idx = start_idx + len(clean_part) if start_idx != -1 else cur_pos + len(clean_part)
                    cur_pos = end_idx

                    # Recursively check this conjunct for attribution or causal modifiers
                    inner_props = self._extract_clause_propositions(clean_part, (start_idx, end_idx), relation)
                    sub_propositions.extend(inner_props)

                if len(sub_propositions) > 1:
                    return sub_propositions

        # Step 2: Check for single-clause attribution or causal modifiers
        return self._extract_clause_propositions(text, (0, len(text)), "root")

    def _extract_clause_propositions(
        self, clause_text: str, span: Tuple[int, int], parent_rel: str
    ) -> List[Tuple[str, Tuple[int, int], str, List[str]]]:
        """
        Extracts attribution and causal modifiers from an individual clause.
        """
        props: List[Tuple[str, Tuple[int, int], str, List[str]]] = []

        # Check for causal split first (e.g. "closed for repairs on Monday")
        causal_split = self._split_causal_clause(clause_text)
        if causal_split:
            for c_text, c_rel in causal_split:
                deps = ["SC-01"] if c_rel != "primary_event" else []
                props.append((c_text, span, c_rel, deps))
            return props

        # Check for reporting verbs: "X announced that Y"
        for rep_verb in self.REPORTING_VERBS:
            pattern = re.compile(rf"^(.*?)\b({rep_verb})\b\s*(.*)$", re.IGNORECASE)
            match = pattern.match(clause_text)
            if match:
                prefix = match.group(1).strip()
                verb = match.group(2).strip()
                suffix = match.group(3).strip()

                if prefix and suffix:
                    # Attribution proposition
                    p1_text = f"{prefix} {verb} {suffix}"
                    props.append((p1_text, span, "attribution", []))

                    # Complement event proposition
                    p2_text = suffix[0].upper() + suffix[1:] if len(suffix) > 1 else suffix.upper()
                    p2_text = p2_text.rstrip(".?! ")
                    props.append((p2_text, span, "complement", ["SC-01"]))
                    return props

        return [(clause_text, span, parent_rel, [])]

    def _split_causal_clause(self, clause: str) -> Optional[List[Tuple[str, str]]]:
        """
        Splits clauses with internal causal/purpose modifiers.
        e.g. "The bridge was closed for repairs on Monday" ->
        [("The bridge was closed on Monday", "root"), ("The closure was due to repairs", "causal")]
        """
        m = re.match(r"^(.*?)\s+for\s+([a-zA-Z\s]+?)\s+(on\s+[a-zA-Z]+|in\s+\d{4}|during\s+[a-zA-Z\d\s]+)$", clause, re.IGNORECASE)
        if m:
            main_event = m.group(1).strip()
            reason = m.group(2).strip()
            temporal = m.group(3).strip()

            p1 = f"{main_event} {temporal}"
            p2 = f"The action was due to {reason}"
            # Refine noun form if known: "closed" -> "closure"
            if "was closed" in main_event.lower():
                p2 = f"The closure was due to {reason}"
            elif "was delayed" in main_event.lower():
                p2 = f"The delay was due to {reason}"

            return [(p1, "primary_event"), (p2, "causal_reason")]

        return None

    def _extract_subject(self, text: str) -> Optional[str]:
        """Extracts the leading noun phrase subject from a clause, unwrapping reporting frames."""
        for rep in ("say that", "said that", "state that", "stated that", "report that", "reported that", "claim that", "claimed that", "show that", "showed that", "found that"):
            pattern = re.compile(rf"\b{rep}\b", re.IGNORECASE)
            m = pattern.search(text)
            if m:
                after = text[m.end():].strip()
                sub = self._extract_subject(after)
                if sub:
                    return sub

        tokens = text.strip().split()
        if not tokens:
            return None
        auxiliaries = {"was", "were", "is", "are", "has", "had", "will", "would", "can", "could", "should", "does", "do", "did", "cannot"}
        subj_tokens = []
        for t in tokens:
            if t.lower() in auxiliaries:
                break
            subj_tokens.append(t)
        if subj_tokens and len(subj_tokens) <= 6:
            return " ".join(subj_tokens)
        if tokens[0].lower() in ("the", "a", "an", "this", "that"):
            if len(tokens) >= 2:
                return f"{tokens[0]} {tokens[1]}"
        return tokens[0]

    def _is_pronoun(self, text: str) -> bool:
        return text.strip().lower() in ("it", "he", "she", "they", "this", "that", "its", "their")

    def _resolve_anaphora(self, prop_text: str, antecedent_subject: Optional[str]) -> str:
        """
        Substitutes leading pronouns with the antecedent subject for self-contained retrieval.
        e.g. "it reopened on Tuesday" -> "The bridge reopened on Tuesday"
        """
        if not antecedent_subject or self._is_pronoun(antecedent_subject):
            return prop_text

        # If prop_text already begins with the antecedent subject, never prepend or duplicate
        if prop_text.lower().startswith(antecedent_subject.lower()):
            return prop_text

        words = prop_text.split()
        if not words:
            return prop_text

        first_word = words[0].lower()
        if first_word in ("it", "this", "that", "he", "she"):
            # Capitalize antecedent subject
            subj = antecedent_subject[0].upper() + antecedent_subject[1:]
            return f"{subj} {' '.join(words[1:])}"

        # If clause begins with an elided verb predicate (e.g. "reopened on Tuesday" or "causes higher intelligence")
        if first_word.endswith(("ed", "es")) or (
            first_word.endswith("s")
            and not first_word.endswith("ss")
            and len(words) >= 2
            and first_word not in ("this", "is", "as", "yes", "always", "scientists", "officials", "experts")
        ) or first_word in ("was", "were", "is", "are", "became", "has", "had"):
            subj = antecedent_subject[0].upper() + antecedent_subject[1:]
            return f"{subj} {prop_text}"

        return prop_text
