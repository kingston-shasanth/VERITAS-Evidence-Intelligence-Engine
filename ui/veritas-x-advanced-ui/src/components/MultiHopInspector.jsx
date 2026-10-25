import React from 'react';
import { Icon } from './Icons.jsx';

/**
 * VERITAS-X 2.0 Multi-Hop Claim Decomposition Inspector
 * Visualizes atomic propositions, reasoning dependencies, subclaim veracity,
 * and transparent compound claim adjudication.
 */
function MultiHopInspectorComponent({
  claim,
  subclaims = [],
  evidence = [],
  onSelectEvidence = null,
  onNavigateToDossier = null,
}) {
  if (!subclaims || subclaims.length === 0) {
    return (
      <div className="multihop-empty-container">
        <Icon name="git-merge" size={24} />
        <p>No decomposed subclaims available for this proposition.</p>
      </div>
    );
  }

  const supportedCount = subclaims.filter((sc) => sc.status === 'SUPPORTED').length;
  const refutedCount = subclaims.filter((sc) => sc.status === 'REFUTED').length;
  const unresolvedCount = subclaims.filter((sc) => sc.status === 'UNRESOLVED').length;
  const totalCount = subclaims.length;
  const coveragePct = Math.round(((supportedCount + refutedCount) / totalCount) * 100);

  return (
    <div className="multihop-inspector-container" role="region" aria-label="Multi-Hop Claim Decomposition Inspector">
      {/* Header and Summary Telemetry */}
      <div className="multihop-header-bar">
        <div>
          <span className="multihop-subhead">STEP-BY-STEP CLAIM BREAKDOWN</span>
          <h4 className="multihop-title">Statement Components & Dependencies</h4>
          <p style={{ margin: '0.25rem 0 0', fontSize: '0.84rem', color: 'var(--text-secondary, #9ca3af)' }}>
            Breaks complex statements into individual parts to verify each component against the evidence.
          </p>
        </div>
        <div className="multihop-coverage-stats">
          <div className="coverage-badge">
            <span className="badge-label">EVIDENCE COVERAGE</span>
            <strong className="badge-value">{coveragePct}%</strong>
          </div>
          <div className="sc-tally-row">
            <span className="tally-pill pill-sup">{supportedCount} Supported</span>
            <span className="tally-pill pill-ref">{refutedCount} Contradicted</span>
            <span className="tally-pill pill-unres">{unresolvedCount} Unresolved</span>
          </div>
        </div>
      </div>

      {/* Proposition Cards */}
      <div className="subclaims-grid">
        {subclaims.map((sc, idx) => {
          const rawStatus = String(sc.status || 'UNRESOLVED').toUpperCase();
          const displayStatus = rawStatus === 'SUPPORTED' ? 'SUPPORTED' : rawStatus === 'REFUTED' ? 'CONTRADICTED' : 'UNRESOLVED';
          const hasDeps = Array.isArray(sc.dependencies) && sc.dependencies.length > 0;

          return (
            <div key={sc.subclaim_id || idx} className={`subclaim-card status-card-${rawStatus.toLowerCase()}`}>
              {/* Card Header */}
              <div className="sc-card-top">
                <div className="sc-id-group">
                  <span className="sc-chip">{sc.subclaim_id}</span>
                  <span className="sc-relation-chip">
                    <code>{sc.relation_to_parent === 'conjunct' ? 'Combined Part' : sc.relation_to_parent === 'condition' ? 'Condition' : sc.relation_to_parent || 'Part'}</code>
                  </span>
                </div>
                <span className={`sc-status-pill pill-${rawStatus.toLowerCase()}`}>
                  {displayStatus}
                </span>
              </div>

              {/* Proposition Text */}
              <p className="sc-text">"{sc.text}"</p>

              {/* Dependencies Alert if present */}
              {hasDeps && (
                <div className="sc-dependencies-callout">
                  <Icon name="corner-down-right" size={13} />
                  <span>
                    Prerequisite dependency: <strong>{sc.dependencies.join(', ')}</strong>
                  </span>
                </div>
              )}

              {/* Stance Tally & Confidence */}
              <div className="sc-metrics-row">
                <div className="sc-stance-counts">
                  <span className="count-dot dot-sup">{sc.stance_distribution?.support || 0} supports</span>
                  <span className="count-dot dot-ref">{sc.stance_distribution?.refute || 0} contradicts</span>
                  <span className="count-dot dot-neu">{sc.stance_distribution?.neutral || 0} neutral</span>
                </div>
                <div className="sc-confidence">
                  <span>Confidence: </span>
                  <strong>{Math.round(Number(sc.confidence || 0) * 100)}%</strong>
                </div>
              </div>

              {/* Rationale explanation */}
              {sc.explanation && (
                <div className="sc-explanation">
                  <p>{sc.explanation}</p>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Dynamic Breakdown Interpretation Card */}
      <div className="dynamic-interpretation-box" style={{
        marginTop: '1.25rem',
        padding: '0.85rem 1rem',
        borderRadius: 'var(--radius-sm, 6px)',
        background: 'var(--bg-tertiary, #1f2937)',
        border: '1px solid var(--border-default, #374151)',
        fontSize: '0.86rem',
        lineHeight: 1.5,
        color: 'var(--text-secondary, #d1d5db)',
      }}>
        <strong style={{ color: 'var(--text-primary, #f9fafb)', display: 'block', marginBottom: '0.25rem' }}>
          Dynamic Step-by-Step Interpretation:
        </strong>
        {refutedCount > 0
          ? `${refutedCount} of ${totalCount} statement component(s) are contradicted by evidence. If a core component is refuted, the overall statement cannot be accepted as true.`
          : supportedCount === totalCount
          ? `All ${totalCount} parts of the statement are verified by evidence. The combined claim is fully supported.`
          : `${supportedCount} part(s) supported and ${unresolvedCount} part(s) could not be resolved from available evidence.`
        }
      </div>
    </div>
  );
}

export const MultiHopInspector = React.memo(MultiHopInspectorComponent);
export default MultiHopInspector;
