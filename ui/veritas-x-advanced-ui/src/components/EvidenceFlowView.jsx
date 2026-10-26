import React, { useState } from 'react';
import { Icon } from './Icons.jsx';

/**
 * Stance badge styling definitions based on real backend annotations.
 */
const STANCE_BADGES = {
  support: { label: 'SUPPORTS', color: 'badge-support', icon: 'checkCircle' },
  entailment: { label: 'SUPPORTS', color: 'badge-support', icon: 'checkCircle' },
  refute: { label: 'REFUTES', color: 'badge-refute', icon: 'xCircle' },
  contradiction: { label: 'REFUTES', color: 'badge-refute', icon: 'xCircle' },
  neutral: { label: 'NEUTRAL', color: 'badge-neutral', icon: 'info' },
  unrelated: { label: 'OFF-TOPIC', color: 'badge-unrelated', icon: 'alertCircle' },
};

/**
 * EvidenceFlowView Component
 * Stage 8 & 9: Evidence retrieval and candidate processing visualization.
 * Highlights the exact query variant used, multi-system retrieval indices,
 * and passes candidate passages to cross-encoder preparation.
 */
export function EvidenceFlowView({
  retrievalDetails = {},
  evidence = [],
  retrievalStatus = {},
}) {
  const [selectedPassageId, setSelectedPassageId] = useState(null);

  const queryUsed = retrievalDetails.retrieval_query || '';
  const queryVariant = retrievalDetails.query_variant || 'sanitized_fallback';
  const mode = retrievalDetails.retrieval_mode || retrievalStatus.mode || 'local';
  const provider = retrievalDetails.provider || retrievalStatus.provider || 'local_corpus';
  const systemsRun = Array.isArray(retrievalDetails.systems_run) ? retrievalDetails.systems_run : ['bm25'];
  const resultsCount = retrievalDetails.results_count ?? evidence.length;
  const durationMs = retrievalDetails.duration_ms ?? retrievalStatus.time_ms ?? 0;
  const retrievalError = retrievalDetails.error || retrievalStatus.error;

  const safeEvidence = Array.isArray(evidence) ? evidence : [];

  return (
    <div className="evidence-flow-workspace" role="region" aria-label="Evidence Retrieval and Ingestion Workspace">
      {/* Stage Header */}
      <div className="evidence-flow-header">
        <div className="evidence-header-meta">
          <span className="cinematic-eyebrow">
            <span className="eyebrow-glow-dot" />
            STAGE 08 &bull; MULTI-INDEX EVIDENCE RETRIEVAL
          </span>
          <h3 className="cinematic-scene-title">
            Corpus Search <span>& Passage Ingestion.</span>
          </h3>
          <p className="cinematic-scene-desc">
            Evidence discovery across lexical, dense neural, and real-time live web indexes using conservative query formulation.
          </p>
        </div>

        <div className="retrieval-query-variant-badge">
          <span className="variant-label">QUERY VARIANT:</span>
          <strong className={`variant-tag ${queryVariant === 'normalized' ? 'variant-norm' : 'variant-fallback'}`}>
            {queryVariant === 'normalized' ? 'NORMALIZED QUERY' : 'SANITIZED ORIGINAL FALLBACK'}
          </strong>
        </div>
      </div>

      {/* Query Formulation Card */}
      <div className="query-inspection-card">
        <div className="query-card-top">
          <span className="query-subhead">EXECUTED SEARCH QUERY</span>
          <div className="query-telemetry-strip">
            <span className="query-tele-item">Mode: <strong>{mode.toUpperCase()}</strong></span>
            <span className="query-tele-item">Provider: <strong>{provider}</strong></span>
            <span className="query-tele-item">Retrieved: <strong>{resultsCount} passages</strong></span>
            <span className="query-tele-item">Latency: <strong>{Number(durationMs).toFixed(1)}ms</strong></span>
          </div>
        </div>

        <div className="query-string-display">
          <Icon name="search" size={18} />
          <code className="query-string-code">"{queryUsed}"</code>
        </div>

        {queryVariant === 'sanitized_fallback' && (
          <div className="query-fallback-alert">
            <Icon name="shieldAlert" size={15} />
            <span>
              <strong>Safe Query Fallback:</strong> The system safely used the sanitized original claim rather than normalized spelling to prevent any semantic distortion of retrieval targets.
            </span>
          </div>
        )}

        {retrievalError && (
          <div className="query-error-alert">
            <Icon name="alertCircle" size={15} />
            <span><strong>Retrieval Notice:</strong> {retrievalError}</span>
          </div>
        )}
      </div>

      {/* Retrieval Engines Multi-Index Pipeline */}
      <div className="retrieval-systems-row">
        <span className="systems-row-title">DISPATCHED RETRIEVAL PIPELINE:</span>
        <div className="systems-tags-wrap">
          {systemsRun.map((sys, idx) => (
            <div key={idx} className="system-tech-badge">
              <span className="system-dot" />
              <span>
                {sys === 'bm25'
                  ? 'BM25 Lexical (Inverted Index)'
                  : sys === 'dense'
                  ? 'Dense Semantic Embedding'
                  : sys === 'tavily_search'
                  ? 'Tavily Neural Web Search'
                  : sys}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Retrieved Evidence Stream */}
      <div className="retrieved-passages-section">
        <div className="passages-section-header">
          <h4 className="passages-title">RETRIEVED DOCUMENTARY PASSAGES ({safeEvidence.length})</h4>
          <span className="passages-subtitle">Ordered by hybrid retrieval score and relevance eligibility</span>
        </div>

        {safeEvidence.length === 0 ? (
          <div className="no-evidence-callout">
            <Icon name="database" size={24} />
            <strong>No Passages Retrieved</strong>
            <p>The search query did not yield any documentary passages from the active corpus.</p>
          </div>
        ) : (
          <div className="passages-grid">
            {safeEvidence.map((ev, i) => {
              const isSelected = selectedPassageId === (ev.passage_id || i);
              const stanceKey = (ev.stance || 'neutral').toLowerCase();
              const badgeMeta = STANCE_BADGES[stanceKey] || STANCE_BADGES.neutral;
              const isOffTopic = ev.is_relevant === false;

              return (
                <div
                  key={ev.passage_id || i}
                  className={`evidence-card-cinematic ${isSelected ? 'card-expanded' : ''} ${isOffTopic ? 'card-offtopic' : ''}`}
                  onClick={() => setSelectedPassageId(isSelected ? null : (ev.passage_id || i))}
                >
                  <div className="evidence-card-header">
                    <div className="evidence-meta-left">
                      <span className="evidence-rank">#{i + 1}</span>
                      <span className="evidence-source-title">{ev.source_title || ev.source || 'Corpus Source'}</span>
                      <span className="evidence-type-chip">{ev.source_type || 'local'}</span>
                    </div>

                    <div className="evidence-meta-right">
                      <span className={`stance-badge ${badgeMeta.color}`}>
                        <Icon name={badgeMeta.icon} size={13} />
                        {badgeMeta.label}
                      </span>
                    </div>
                  </div>

                  <p className="evidence-text-snippet">
                    "{ev.text}"
                  </p>

                  <div className="evidence-card-footer">
                    <div className="evidence-metrics">
                      {ev.score !== undefined && (
                        <span className="metric-tag">
                          Score: <strong>{(Number(ev.score) * 100).toFixed(0)}%</strong>
                        </span>
                      )}
                      {ev.claim_coverage && (
                        <span className="metric-tag">
                          Coverage: <strong>{ev.claim_coverage}</strong>
                        </span>
                      )}
                      {isOffTopic && (
                        <span className="metric-tag tag-excluded">
                          Excluded from Verdict (Off-Topic)
                        </span>
                      )}
                    </div>

                    {ev.explanation && (
                      <div className="evidence-rationale">
                        <Icon name="info" size={13} />
                        <span>{ev.explanation}</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default EvidenceFlowView;
