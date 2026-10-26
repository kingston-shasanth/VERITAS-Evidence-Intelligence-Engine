import React, { useState } from 'react';
import { Icon } from './Icons.jsx';

/**
 * Modality badge definitions with color styling and semantic explanation.
 */
const MODALITY_CONFIG = {
  certain: { label: 'CERTAIN', color: 'modality-certain', desc: 'Definite unconditioned assertion' },
  probable: { label: 'PROBABLE', color: 'modality-probable', desc: 'High likelihood modal marker (e.g. likely, probably)' },
  possible: { label: 'POSSIBLE', color: 'modality-possible', desc: 'Feasible possibility marker (e.g. may, might, could)' },
  alleged: { label: 'ALLEGED', color: 'modality-alleged', desc: 'Attributed hearsay or unverified assertion (e.g. allegedly, reportedly)' },
  doubtful: { label: 'DOUBTFUL', color: 'modality-doubtful', desc: 'Skeptical or questioned epistemic status' },
};

/**
 * ClaimStructureView Component
 * Stage 7: Proposition structure decomposition showing extracted semantic elements,
 * modality, negation boundaries, named & numeric entities, and compound propositions.
 */
export function ClaimStructureView({
  structuredClaim = {},
  claimExtractionDetails = {},
  sanitizedClaim = '',
}) {
  const [viewMode, setViewMode] = useState('VISUAL'); // 'VISUAL' | 'COMPACT_JSON'

  // Extract structured claim properties from either structuredClaim or claimExtractionDetails
  const sc = structuredClaim.subject !== undefined
    ? structuredClaim
    : (claimExtractionDetails.structured_claim || {});

  const subject = sc.subject || claimExtractionDetails.subject || null;
  const predicate = sc.predicate || claimExtractionDetails.predicate || null;
  const object = sc.object || claimExtractionDetails.object || null;
  const modalityKey = (sc.modality || claimExtractionDetails.modality || 'certain').toLowerCase();
  const modalityMeta = MODALITY_CONFIG[modalityKey] || MODALITY_CONFIG.certain;

  const isNegated = Boolean(sc.is_negated ?? claimExtractionDetails.has_negation);
  const negationScope = sc.negation_scope || claimExtractionDetails.negation_scope || null;
  const isConditional = Boolean(sc.is_conditional ?? claimExtractionDetails.is_conditional);
  const isCausal = Boolean(sc.is_causal ?? claimExtractionDetails.is_causal);

  const namedEntities = Array.isArray(sc.named_entities) ? sc.named_entities : [];
  const numericEntities = Array.isArray(sc.numeric_entities) ? sc.numeric_entities : [];
  const quantifiers = Array.isArray(sc.quantifiers) ? sc.quantifiers : [];
  const subPropositions = Array.isArray(sc.sub_propositions) ? sc.sub_propositions : [];

  return (
    <div className="claim-structure-workspace" role="region" aria-label="Structured Proposition Workspace">
      {/* Stage Header */}
      <div className="claim-structure-header">
        <div className="claim-header-meta">
          <span className="cinematic-eyebrow">
            <span className="eyebrow-glow-dot" />
            STAGE 07 &bull; STRUCTURED CLAIM EXTRACTION
          </span>
          <h3 className="cinematic-scene-title">
            Propositional <span>Architecture.</span>
          </h3>
          <p className="cinematic-scene-desc">
            Formal extraction of propositional truth conditions, epistemic modality, negation scope, entities, and numeric quantities.
          </p>
        </div>

        {/* View Mode Toggle */}
        <div className="claim-view-toggle">
          <button
            type="button"
            className={`toggle-tab ${viewMode === 'VISUAL' ? 'active' : ''}`}
            onClick={() => setViewMode('VISUAL')}
          >
            Visual Model
          </button>
          <button
            type="button"
            className={`toggle-tab ${viewMode === 'COMPACT_JSON' ? 'active' : ''}`}
            onClick={() => setViewMode('COMPACT_JSON')}
          >
            Compact Schema
          </button>
        </div>
      </div>

      {viewMode === 'COMPACT_JSON' ? (
        <div className="compact-json-container">
          <pre className="compact-json-code">
            {JSON.stringify(
              {
                canonical_claim: sanitizedClaim,
                structured_claim: {
                  subject,
                  predicate,
                  object,
                  modality: modalityKey,
                  is_negated: isNegated,
                  negation_scope: negationScope,
                  is_conditional: isConditional,
                  is_causal: isCausal,
                  named_entities: namedEntities,
                  numeric_entities: numericEntities,
                  quantifiers: quantifiers,
                  sub_propositions: subPropositions,
                },
              },
              null,
              2
            )}
          </pre>
        </div>
      ) : (
        <div className="visual-proposition-model">
          {/* Proposition Core Node */}
          <div className="proposition-card-main">
            <div className="proposition-top-badges">
              <span className={`modality-pill ${modalityMeta.color}`}>
                <span className="modality-dot" />
                Modality: <strong>{modalityMeta.label}</strong>
              </span>

              {isNegated && (
                <span className="negation-active-pill">
                  <Icon name="xCircle" size={14} />
                  NEGATED TRUTH CONDITION
                </span>
              )}

              {isConditional && (
                <span className="logic-pill">
                  <Icon name="gitCommit" size={14} />
                  CONDITIONAL (IF-THEN)
                </span>
              )}

              {isCausal && (
                <span className="logic-pill">
                  <Icon name="rotate" size={14} />
                  CAUSAL RELATION
                </span>
              )}
            </div>

            {/* S-P-O Proposition Elements */}
            <div className="proposition-elements-row">
              <div className="prop-element-box prop-subj">
                <span className="element-kind">SUBJECT ARGUMENT</span>
                <strong className="element-text">{subject || 'None extracted'}</strong>
              </div>

              <div className="prop-relation-vector">
                <span className="vector-tag">ASSERTION</span>
                <div className="vector-divider">&rarr;</div>
              </div>

              <div className="prop-element-box prop-pred">
                <span className="element-kind">PREDICATE RELATION</span>
                <strong className="element-text">{predicate || 'None extracted'}</strong>
              </div>

              <div className="prop-relation-vector">
                <span className="vector-tag">TARGET</span>
                <div className="vector-divider">&rarr;</div>
              </div>

              <div className="prop-element-box prop-obj">
                <span className="element-kind">OBJECT / COMPLEMENT</span>
                <strong className="element-text">{object || 'Intransitive / Elliptical'}</strong>
              </div>
            </div>

            {/* Negation Scope Callout if present */}
            {isNegated && (
              <div className="negation-scope-banner">
                <Icon name="shieldAlert" size={16} />
                <div>
                  <strong>Explicit Negation Invariant:</strong>
                  {negationScope ? (
                    <span> Scope restricted to clause: <code>"{negationScope}"</code></span>
                  ) : (
                    <span> Complete sentence proposition is negated.</span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Extracted Semantic Entities Grid */}
          <div className="semantic-entities-grid">
            {/* Named Entities */}
            <div className="entity-panel">
              <div className="entity-panel-head">
                <Icon name="sparkles" size={16} />
                <span>NAMED ENTITIES ({namedEntities.length})</span>
              </div>
              <div className="entity-panel-body">
                {namedEntities.length === 0 ? (
                  <span className="entity-empty-note">No capitalized named entities detected.</span>
                ) : (
                  <div className="entity-pills-wrap">
                    {namedEntities.map((ent, i) => (
                      <span key={i} className="named-entity-pill">
                        {ent}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Numeric Entities & Units */}
            <div className="entity-panel">
              <div className="entity-panel-head">
                <Icon name="database" size={16} />
                <span>NUMERIC QUANTITIES ({numericEntities.length})</span>
              </div>
              <div className="entity-panel-body">
                {numericEntities.length === 0 ? (
                  <span className="entity-empty-note">No numbers, percentages, or units identified.</span>
                ) : (
                  <div className="entity-pills-wrap">
                    {numericEntities.map((num, i) => (
                      <span key={i} className="numeric-entity-pill">
                        <strong>{num.numeric_value ?? num.raw_value}</strong>
                        {num.unit && <span className="unit-label"> {num.unit}</span>}
                        {num.entity_type && <span className="type-label"> ({num.entity_type})</span>}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Quantifiers */}
            <div className="entity-panel">
              <div className="entity-panel-head">
                <Icon name="layers" size={16} />
                <span>QUANTIFIER SCOPE ({quantifiers.length})</span>
              </div>
              <div className="entity-panel-body">
                {quantifiers.length === 0 ? (
                  <span className="entity-empty-note">Default existential / specific scope.</span>
                ) : (
                  <div className="entity-pills-wrap">
                    {quantifiers.map((q, i) => (
                      <span key={i} className="quantifier-pill">
                        <code>{q.text}</code> &ndash; {q.quantifier_type || 'scope'}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Sub-Propositions for Compound Claims */}
          {subPropositions.length > 1 && (
            <div className="sub-propositions-section">
              <h4 className="subprop-title">COMPOUND CONSTITUENT PROPOSITIONS ({subPropositions.length})</h4>
              <p className="subprop-desc">
                Compound claim decomposed into atomic verifiable assertions:
              </p>
              <div className="subprop-cards-grid">
                {subPropositions.map((prop, i) => (
                  <div key={i} className="subprop-card">
                    <span className="subprop-number">P{i + 1}</span>
                    <strong className="subprop-text">"{prop}"</strong>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ClaimStructureView;
