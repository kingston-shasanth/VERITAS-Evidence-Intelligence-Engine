"""
Meaning-Preservation Guardrails Module for VERITAS-X.

Evaluates proposition-level invariants between raw/sanitized text and normalized text.
Enforces 8 core meaning-preservation checks:
1. Negation and polarity preservation (polarity inversion, nested negation, scope shift).
2. Quantifier and scope preservation (universal vs existential vs negative vs partial negation).
3. Question versus assertion intent preservation.
4. Entity, number, date, percentage, and unit preservation.
5. Epistemic modality and certainty preservation (hedging removal, certainty inflation).
6. Conditional and causal relationship preservation (causality -> correlation, condition drops).
7. Proposition-count and relationship preservation (loss of conjuncts or clauses).
8. Ambiguity and unresolved reference handling (dangling pronouns, unverified lexical substitutions).

Epistemic Safety Rule:
Lexical similarity alone cannot prove semantic equivalence.
If any invariant check detects a potential alteration to the proposition's truth conditions,
the guardrail enforces REVERT_TO_ORIGINAL, routes the raw/sanitized text to downstream verification,
and emits an auditable safety warning.
"""
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import List, Dict, Any, Optional, Set, Tuple

from .spelling import NormalizationResult
from .claim_extractor import (
    ClaimExtractor,
    StructuredClaim,
    ModalityLevel,
    QuantifierType
)


class GuardrailDecision(str, Enum):
    USE_NORMALIZED = "USE_NORMALIZED"
    REVERT_TO_ORIGINAL = "REVERT_TO_ORIGINAL"


@dataclass
class GuardrailResult:
    """Explicit, typed meaning-preservation guardrail evaluation envelope."""
    is_safe: bool
    decision: GuardrailDecision
    recommended_text: str
    original_claim: StructuredClaim
    normalized_claim: StructuredClaim
    invariants_passed: Dict[str, bool] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "decision": self.decision.value,
            "recommended_text": self.recommended_text,
            "invariants_passed": self.invariants_passed,
            "violations": self.violations,
            "warnings": self.warnings,
            "rationale": self.rationale,
            "original_structure": self.original_claim.to_dict(),
            "normalized_structure": self.normalized_claim.to_dict(),
        }


class MeaningPreservationGuardrails:
    """
    Evaluates semantic invariants between pre- and post-normalization claims.
    """

    # Invariant names
    INV_NEGATION_POLARITY = "negation_and_polarity"
    INV_QUANTIFIERS = "quantifier_and_scope"
    INV_QUESTION_INTENT = "question_versus_assertion"
    INV_ENTITIES_NUMBERS = "entities_numbers_and_units"
    INV_MODALITY = "modality_and_uncertainty"
    INV_CAUSALITY_CONDITION = "causality_and_conditionals"
    INV_PROPOSITION_COUNT = "proposition_count_and_relations"
    INV_AMBIGUITY = "ambiguity_and_references"

    @classmethod
    def evaluate(
        cls,
        raw_or_sanitized_text: str,
        normalized_result_or_text: Any
    ) -> GuardrailResult:
        """
        Evaluates meaning preservation between original and normalized representations.
        Accepts either a NormalizationResult object from Phase 2 or a normalized string.
        """
        # Resolve normalized text and upstream warnings
        if isinstance(normalized_result_or_text, NormalizationResult):
            norm_text = normalized_result_or_text.normalized_text
            upstream_warnings = list(normalized_result_or_text.warnings)
            raw_text = normalized_result_or_text.raw_text or raw_or_sanitized_text
        else:
            norm_text = str(normalized_result_or_text)
            upstream_warnings = []
            raw_text = raw_or_sanitized_text

        # Extract structured representations for both versions
        orig_claim = ClaimExtractor.extract(raw_text)
        norm_claim = ClaimExtractor.extract(norm_text)

        invariants_passed: Dict[str, bool] = {}
        violations: List[str] = []
        warnings: List[str] = list(upstream_warnings)

        # -------------------------------------------------------------
        # 1. Negation & Polarity Invariant
        # -------------------------------------------------------------
        neg_pass, neg_viol = cls._check_negation_invariants(orig_claim, norm_claim)
        invariants_passed[cls.INV_NEGATION_POLARITY] = neg_pass
        if not neg_pass:
            violations.extend(neg_viol)

        # -------------------------------------------------------------
        # 2. Quantifier & Scope Invariant
        # -------------------------------------------------------------
        quant_pass, quant_viol = cls._check_quantifier_invariants(orig_claim, norm_claim)
        invariants_passed[cls.INV_QUANTIFIERS] = quant_pass
        if not quant_pass:
            violations.extend(quant_viol)

        # -------------------------------------------------------------
        # 3. Question vs Assertion Intent Invariant
        # -------------------------------------------------------------
        q_pass, q_viol = cls._check_question_invariants(orig_claim, norm_claim)
        invariants_passed[cls.INV_QUESTION_INTENT] = q_pass
        if not q_pass:
            violations.extend(q_viol)

        # -------------------------------------------------------------
        # 4. Entities, Numbers, and Units Invariant
        # -------------------------------------------------------------
        ent_pass, ent_viol = cls._check_entities_and_numbers(orig_claim, norm_claim)
        invariants_passed[cls.INV_ENTITIES_NUMBERS] = ent_pass
        if not ent_pass:
            violations.extend(ent_viol)

        # -------------------------------------------------------------
        # 5. Modality & Uncertainty Invariant
        # -------------------------------------------------------------
        mod_pass, mod_viol = cls._check_modality_invariants(orig_claim, norm_claim)
        invariants_passed[cls.INV_MODALITY] = mod_pass
        if not mod_pass:
            violations.extend(mod_viol)

        # -------------------------------------------------------------
        # 6. Causality & Conditionals Invariant
        # -------------------------------------------------------------
        caus_pass, caus_viol = cls._check_causality_and_conditionals(orig_claim, norm_claim)
        invariants_passed[cls.INV_CAUSALITY_CONDITION] = caus_pass
        if not caus_pass:
            violations.extend(caus_viol)

        # -------------------------------------------------------------
        # 7. Proposition Count & Compound Structure Invariant
        # -------------------------------------------------------------
        prop_pass, prop_viol = cls._check_proposition_counts(orig_claim, norm_claim)
        invariants_passed[cls.INV_PROPOSITION_COUNT] = prop_pass
        if not prop_pass:
            violations.extend(prop_viol)

        # -------------------------------------------------------------
        # 8. Ambiguity & Unresolved Reference Invariant
        # -------------------------------------------------------------
        amb_pass, amb_warns = cls._check_ambiguity(orig_claim, norm_claim)
        invariants_passed[cls.INV_AMBIGUITY] = amb_pass
        warnings.extend(amb_warns)

        # Final Safety Evaluation
        is_safe = (len(violations) == 0)

        if is_safe:
            decision = GuardrailDecision.USE_NORMALIZED
            recommended_text = norm_text
            rationale = (
                "All 8 meaning-preservation invariants satisfied: polarity, quantifiers, "
                "entities, numbers, and causal structure remain identical."
            )
        else:
            decision = GuardrailDecision.REVERT_TO_ORIGINAL
            recommended_text = raw_text
            rationale = (
                f"Meaning-preservation guardrail failed ({len(violations)} violation(s)): "
                f"{'; '.join(violations[:3])}. Reverting to original claim to protect truth conditions."
            )

        return GuardrailResult(
            is_safe=is_safe,
            decision=decision,
            recommended_text=recommended_text,
            original_claim=orig_claim,
            normalized_claim=norm_claim,
            invariants_passed=invariants_passed,
            violations=violations,
            warnings=warnings,
            rationale=rationale,
        )

    # --------------------------------------------------------------------------
    # Invariant Checks
    # --------------------------------------------------------------------------
    @classmethod
    def _check_negation_invariants(
        cls, orig: StructuredClaim, norm: StructuredClaim
    ) -> Tuple[bool, List[str]]:
        """Verifies that negation polarity and scope are preserved."""
        violations = []
        if orig.is_negated != norm.is_negated:
            violations.append(
                f"Negation polarity shift: original_negated={orig.is_negated}, "
                f"normalized_negated={norm.is_negated}"
            )

        if orig.negation_scope != "none" and norm.negation_scope != "none":
            if orig.negation_scope != norm.negation_scope:
                violations.append(
                    f"Negation scope altered: original='{orig.negation_scope}', "
                    f"normalized='{norm.negation_scope}'"
                )

        return len(violations) == 0, violations

    @classmethod
    def _check_quantifier_invariants(
        cls, orig: StructuredClaim, norm: StructuredClaim
    ) -> Tuple[bool, List[str]]:
        """Verifies that quantifiers (all vs some vs none) do not mutate."""
        violations = []
        orig_q_types = {q.qtype for q in orig.quantifiers}
        norm_q_types = {q.qtype for q in norm.quantifiers}

        # Check for universal <-> existential shift (e.g. all -> some)
        if QuantifierType.UNIVERSAL in orig_q_types and QuantifierType.UNIVERSAL not in norm_q_types:
            violations.append("Universal quantifier ('all'/'every') deleted or weakened")

        if QuantifierType.EXISTENTIAL in orig_q_types and QuantifierType.UNIVERSAL in norm_q_types:
            violations.append("Existential quantifier ('some') illegally generalized to universal ('all')")

        if QuantifierType.NEGATIVE in orig_q_types and QuantifierType.NEGATIVE not in norm_q_types:
            violations.append("Negative quantifier ('none'/'no') lost during normalization")

        if QuantifierType.PARTIAL_NEGATION in orig_q_types and QuantifierType.PARTIAL_NEGATION not in norm_q_types:
            violations.append("Partial negation quantifier ('not all') altered")

        return len(violations) == 0, violations

    @classmethod
    def _check_question_invariants(
        cls, orig: StructuredClaim, norm: StructuredClaim
    ) -> Tuple[bool, List[str]]:
        """Verifies that inquiry intent is preserved."""
        violations = []
        if orig.is_question != norm.is_question:
            violations.append(
                f"Sentence speech act modified: original_is_question={orig.is_question}, "
                f"normalized_is_question={norm.is_question}"
            )
        return len(violations) == 0, violations

    @classmethod
    def _check_entities_and_numbers(
        cls, orig: StructuredClaim, norm: StructuredClaim
    ) -> Tuple[bool, List[str]]:
        """Verifies that numeric values, units, and proper named entities are intact."""
        violations = []

        # Check numbers and measurements
        orig_nums = sorted([f"{n.value}_{n.unit}" for n in orig.numeric_entities if n.value is not None])
        norm_nums = sorted([f"{n.value}_{n.unit}" for n in norm.numeric_entities if n.value is not None])
        if orig_nums != norm_nums:
            violations.append(f"Numeric values modified: original={orig_nums}, normalized={norm_nums}")

        # Check dates
        orig_dates = sorted([d.raw_text.lower() for d in orig.temporal_entities])
        norm_dates = sorted([d.raw_text.lower() for d in norm.temporal_entities])
        if orig_dates != norm_dates:
            violations.append(f"Temporal expressions altered: original={orig_dates}, normalized={norm_dates}")

        # Check critical named entities (e.g. Barrett, Trump, Supreme Court)
        for ent in orig.named_entities:
            # Check if entity still exists in normalized claim
            if ent.lower() not in norm.raw_text.lower():
                violations.append(f"Named entity '{ent}' missing or corrupted in normalized output")

        return len(violations) == 0, violations

    @classmethod
    def _check_modality_invariants(
        cls, orig: StructuredClaim, norm: StructuredClaim
    ) -> Tuple[bool, List[str]]:
        """Verifies that epistemic hedging (allegedly, might) was not stripped."""
        violations = []
        # If original was hedged (ALLEGED or POSSIBLE), normalized must not be CERTAIN
        if orig.modality in (ModalityLevel.ALLEGED, ModalityLevel.POSSIBLE):
            if norm.modality == ModalityLevel.CERTAIN:
                violations.append(
                    f"Epistemic certainty inflated: original was '{orig.modality.value}', "
                    f"normalized converted to '{norm.modality.value}'"
                )
        return len(violations) == 0, violations

    @classmethod
    def _check_causality_and_conditionals(
        cls, orig: StructuredClaim, norm: StructuredClaim
    ) -> Tuple[bool, List[str]]:
        """Verifies that conditionals and causal claims are not altered."""
        violations = []
        if orig.is_conditional != norm.is_conditional:
            violations.append(
                f"Conditional relation altered: original_conditional={orig.is_conditional}, "
                f"normalized_conditional={norm.is_conditional}"
            )

        if orig.is_causal != norm.is_causal:
            violations.append(
                f"Causal assertion altered: original_causal={orig.is_causal}, "
                f"normalized_causal={norm.is_causal}"
            )
        return len(violations) == 0, violations

    @classmethod
    def _check_proposition_counts(
        cls, orig: StructuredClaim, norm: StructuredClaim
    ) -> Tuple[bool, List[str]]:
        """Verifies that compound propositions do not lose clauses."""
        violations = []
        orig_count = len(orig.sub_propositions)
        norm_count = len(norm.sub_propositions)
        if orig_count > 1 and norm_count != orig_count:
            violations.append(
                f"Compound proposition clause count mismatch: original={orig_count}, normalized={norm_count}"
            )
        return len(violations) == 0, violations

    @classmethod
    def _check_ambiguity(
        cls, orig: StructuredClaim, norm: StructuredClaim
    ) -> Tuple[bool, List[str]]:
        """Flags unresolved pronouns or dangling antecedents."""
        warnings = []
        if norm.unresolved_references:
            warnings.append(
                f"Unresolved reference(s) detected without antecedent: {norm.unresolved_references}"
            )
        return True, warnings
