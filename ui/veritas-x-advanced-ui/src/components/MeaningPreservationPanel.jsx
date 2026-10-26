import React from 'react';
import { Icon } from './Icons.jsx';

/**
 * The 8 core semantic invariants verified by Phase 3 meaning preservation guardrails.
 */
const INVARIANTS_DEF = [
  { id: 'negation_and_polarity', label: 'Negation & Polarity Invariance', desc: 'Guarantees negative assertions do not invert to positive (and vice versa).' },
  { id: 'quantifier_and_scope', label: 'Quantifier & Scope Invariance', desc: 'Preserves universal ("all"), existential ("some"), and negative ("none") scopes.' },
  { id: 'question_versus_assertion', label: 'Question vs Assertion Intent', desc: 'Ensures inquiry statements do not transform into assertive claims.' },
  { id: 'entities_numbers_and_units', label: 'Entities, Numbers & Units', desc: 'Guarantees dates, numbers, percentages, and scientific units are never altered.' },
  { id: 'modality_and_uncertainty', label: 'Modality & Uncertainty Scale', desc: 'Prevents certainty inflation (e.g. "allegedly" -> "definitely").' },
  { id: 'causality_and_conditionals', label: 'Causality & Conditional Logic', desc: 'Preserves cause-and-effect and "if-then" implication structures.' },
  { id: 'proposition_count_and_relations', label: 'Proposition Count Invariance', desc: 'Ensures compound clauses and conjuncts are fully retained.' },
  { id: 'ambiguity_and_references', label: 'Ambiguity & Reference Handling', desc: 'Flags dangling anaphora and ambiguous multi-candidate terms.' },
];

/**
 * Meaning Preservation Checkpoint Component
 * Visually confirms semantic invariance before evidence retrieval.
 */
export function MeaningPreservationPanel({
  originalClaim = '',
  normalizedClaim = '',
  guardrailDetails = {},
  normalizationStatus = 'accepted',
}) {
  const isSafe = guardrailDetails.is_safe !== false && normalizationStatus === 'accepted';
  const decision = guardrailDetails.decision || (isSafe ? 'USE_NORMALIZED' : 'REVERT_TO_ORIGINAL');
  const violations = guardrailDetails.violations || [];
  const warnings = guardrailDetails.warnings || [];
  const invariantsPassed = guardrailDetails.invariants_passed || {};

  return (
    <div className="meaning-preservation-workspace" role="region" aria-label="Meaning Preservation Guardrail Checkpoint">
      <div className="guardrail-hero-banner">
        <div className="guardrail-hero-left">
          <div className="guardrail-icon-glow">
            <Icon name="shield" size={24} />
          </div>
          <div>
            <span className="cinematic-eyebrow">
              <span className="eyebrow-glow-dot" />
              SEMANTIC INVARIANCE GATEWAY
            </span>
            <h3 className="cinematic-scene-title">
              Meaning-Preservation <span>Checkpoint.</span>
            </h3>
            <p className="cinematic-scene-desc">
              Improving text readability is strictly prohibited from altering the factual proposition being verified.
            </p>
          </div>
        </div>

        <div className="guardrail-decision-card">
          <span className="decision-label">GUARDRAIL VERDICT</span>
          <strong className={`decision-badge ${isSafe ? 'decision-approved' : 'decision-reverted'}`}>
            {decision}
          </strong>
          <span className="decision-subtext">
            {isSafe ? 'Semantic invariants confirmed' : 'Original claim preserved for safety'}
          </span>
        </div>
      </div>

      {/* Side-by-Side Proposition Comparison */}
      <div className="comparison-cards-grid">
        <div className="comparison-card original-card">
          <div className="comparison-card-top">
            <span className="comparison-card-tag">CANONICAL USER CLAIM</span>
            <span className="provenance-tag">Exact User Submission</span>
          </div>
          <p className="comparison-claim-text">{originalClaim}</p>
          <div className="comparison-foot">
            <span className="foot-status-dot green-dot" />
            <span>Always preserved as the truthful NLI verification hypothesis</span>
          </div>
        </div>

        <div className={`comparison-card candidate-card ${isSafe ? 'card-approved' : 'card-fallback'}`}>
          <div className="comparison-card-top">
            <span className="comparison-card-tag">NORMALIZED CANDIDATE</span>
            <span className={`provenance-tag ${isSafe ? 'tag-approved' : 'tag-fallback'}`}>
              {isSafe ? 'Approved for Retrieval' : 'Rejected by Guardrails'}
            </span>
          </div>
          <p className="comparison-claim-text">{normalizedClaim}</p>
          <div className="comparison-foot">
            <span className={`foot-status-dot ${isSafe ? 'blue-dot' : 'amber-dot'}`} />
            <span>
              {isSafe
                ? 'Safe for BM25 and dense retrieval keyword expansion'
                : 'Fell back to original text to prevent semantic drift'}
            </span>
          </div>
        </div>
      </div>

      {/* Violations / Warnings Banner if Present */}
      {(violations.length > 0 || warnings.length > 0) && (
        <div className="guardrail-violations-banner" role="alert">
          <div className="violations-header">
            <Icon name="alert" size={16} />
            <strong>Guardrail Audit Log:</strong>
          </div>
          <ul className="violations-list">
            {violations.map((v, i) => (
              <li key={`v-${i}`} className="violation-item">
                <span className="violation-dot" />
                <span>{v}</span>
              </li>
            ))}
            {warnings.map((w, i) => (
              <li key={`w-${i}`} className="warning-item">
                <span className="warning-dot" />
                <span>{w}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 8 Invariants Grid */}
      <div className="invariants-checklist-section">
        <span className="invariants-section-title">Verified Proposition Invariants:</span>
        <div className="invariants-grid">
          {INVARIANTS_DEF.map((inv) => {
            const hasExplicitPass = invariantsPassed[inv.id] !== undefined ? invariantsPassed[inv.id] : isSafe;
            const isViolated = violations.some((v) => v.toLowerCase().includes(inv.id.replace(/_/g, ' ')));
            const passed = hasExplicitPass && !isViolated;

            return (
              <div key={inv.id} className={`invariant-card ${passed ? 'inv-passed' : 'inv-violated'}`}>
                <div className="inv-top-row">
                  <span className="inv-status-icon">
                    <Icon name={passed ? 'check' : 'close'} size={12} />
                  </span>
                  <strong className="inv-title">{inv.label}</strong>
                </div>
                <p className="inv-desc">{inv.desc}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default MeaningPreservationPanel;
