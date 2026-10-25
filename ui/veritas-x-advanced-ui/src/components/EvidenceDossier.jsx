import React, { useState, useMemo } from 'react';
import { Icon } from './Icons';
import { EvidenceGraph } from './EvidenceGraph';

function EvidenceDossierComponent({
  evidence = [],
  claim = '',
  verdict = '',
  confidence = 0,
  mode = 'local',
  retrievalStatus = null,
  isFixture = false,
  onComparePassages,
}) {
  const [viewMode, setViewMode] = useState('cards'); // 'cards' | 'graph' | 'compare'
  const [stanceFilter, setStanceFilter] = useState('ALL'); // 'ALL' | 'support' | 'refute' | 'neutral'
  const [sortBy, setSortBy] = useState('score'); // 'score' | 'id' | 'stance'
  const [expandedIds, setExpandedIds] = useState({});
  const [selectedForCompare, setSelectedForCompare] = useState([]);
  const [copiedId, setCopiedId] = useState(null);

  const toggleExpand = (id) => {
    setExpandedIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleCompareSelect = (id) => {
    setSelectedForCompare((prev) => {
      if (prev.includes(id)) {
        return prev.filter((item) => item !== id);
      }
      if (prev.length >= 2) {
        return [prev[1], id]; // Shift to keep at most 2
      }
      return [...prev, id];
    });
  };

  const copyCitation = (item) => {
    const citation = `[Citation: ${item.id}] Source: ${item.source}\nStance: ${item.stance.toUpperCase()} (Score: ${(item.score * 100).toFixed(1)}%)\nText: "${item.text}"`;
    navigator.clipboard.writeText(citation);
    setCopiedId(item.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const isPassageSupporting = (e) =>
    ['support', 'entailment'].includes(String(e.stance || 'neutral').toLowerCase()) &&
    e.is_informative !== false &&
    e.is_relevant !== false &&
    e.is_claim_aligned !== false &&
    e.is_figurative !== true &&
    e.is_decision_eligible !== false;

  const isPassageRefuting = (e) =>
    ['refute', 'contradiction'].includes(String(e.stance || 'neutral').toLowerCase()) &&
    e.is_informative !== false &&
    e.is_relevant !== false &&
    e.is_claim_aligned !== false &&
    e.is_figurative !== true &&
    e.is_decision_eligible !== false;

  const filteredAndSortedEvidence = useMemo(() => {
    let list = [...evidence];

    // Filter
    if (stanceFilter !== 'ALL') {
      list = list.filter((item) => {
        if (stanceFilter === 'support') return isPassageSupporting(item);
        if (stanceFilter === 'refute') return isPassageRefuting(item);
        if (stanceFilter === 'neutral') return !isPassageSupporting(item) && !isPassageRefuting(item);
        return true;
      });
    }

    // Sort
    list.sort((a, b) => {
      if (sortBy === 'score') {
        return (b.score || 0) - (a.score || 0);
      }
      if (sortBy === 'id') {
        return String(a.id).localeCompare(String(b.id));
      }
      if (sortBy === 'stance') {
        const rank = { support: 1, entailment: 1, refute: 2, contradiction: 2, neutral: 3 };
        return (rank[a.stance] || 4) - (rank[b.stance] || 4);
      }
      return 0;
    });

    return list;
  }, [evidence, stanceFilter, sortBy]);

  // Counts aligned with decision criteria
  const supCount = evidence.filter(isPassageSupporting).length;
  const refCount = evidence.filter(isPassageRefuting).length;
  const neutCount = evidence.length - (supCount + refCount);

  const comparedItems = evidence.filter((e) => selectedForCompare.includes(e.id));

  return (
    <div className="evidence-dossier-root" role="region" aria-label="Evidence Dossier & Docket">
      {(mode === 'user_supplied' || isFixture) && (
        <div
          className="fixture-evidence-notice-banner"
          style={{
            background: 'var(--primary-soft, #EEF2FF)',
            border: '1px solid var(--primary-border, #C7D2FE)',
            borderRadius: 'var(--radius-sm, 6px)',
            padding: '0.75rem 1rem',
            marginBottom: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.65rem',
            color: 'var(--text-primary)',
            fontSize: '0.82rem',
          }}
        >
          <Icon name="files" size={16} />
          <span>
            <strong>Supplied Fixture Evidence:</strong> This result was evaluated against candidate premise passages supplied directly with the claim rather than retrieved corpus evidence.
          </span>
        </div>
      )}

      {/* Dossier Controls Toolbar */}
      <div className="dossier-toolbar">
        <div className="toolbar-left">
          <div className="dossier-title-group">
            <h3 className="dossier-heading">Evidence Dossier & Provenance Docket</h3>
            <span className="dossier-count-badge">
              {evidence.length} Retrieved Passage{evidence.length !== 1 ? 's' : ''}
            </span>
          </div>

          {/* Stance Filters */}
          <div className="stance-filter-tabs" role="tablist" aria-label="Filter evidence by stance">
            <button
              type="button"
              className={`filter-tab ${stanceFilter === 'ALL' ? 'tab-active' : ''}`}
              onClick={() => setStanceFilter('ALL')}
            >
              All ({evidence.length})
            </button>
            <button
              type="button"
              className={`filter-tab tab-support ${stanceFilter === 'support' ? 'tab-active' : ''}`}
              onClick={() => setStanceFilter('support')}
            >
              Supporting ({supCount})
            </button>
            <button
              type="button"
              className={`filter-tab tab-refute ${stanceFilter === 'refute' ? 'tab-active' : ''}`}
              onClick={() => setStanceFilter('refute')}
            >
              Refuting ({refCount})
            </button>
            <button
              type="button"
              className={`filter-tab tab-neutral ${stanceFilter === 'neutral' ? 'tab-active' : ''}`}
              onClick={() => setStanceFilter('neutral')}
            >
              Neutral ({neutCount})
            </button>
          </div>
        </div>

        <div className="toolbar-right">
          {/* Sorting Control */}
          <div className="dossier-sort-group">
            <Icon name="sort" size={15} />
            <label htmlFor="dossier-sort-select" className="sr-only">Sort Evidence By</label>
            <select
              id="dossier-sort-select"
              className="dossier-select"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="score">Sort: Highest Score</option>
              <option value="stance">Sort: Stance (Support → Refute)</option>
              <option value="id">Sort: Passage ID</option>
            </select>
          </div>

          {/* View Mode Toggle: Cards vs Topology Graph vs Comparison */}
          <div className="view-mode-group">
            <button
              type="button"
              className={`view-btn ${viewMode === 'cards' ? 'view-btn-active' : ''}`}
              onClick={() => setViewMode('cards')}
              title="Card Docket View"
            >
              <Icon name="files" size={15} />
              <span>Dossier</span>
            </button>
            <button
              type="button"
              className={`view-btn ${viewMode === 'graph' ? 'view-btn-active' : ''}`}
              onClick={() => setViewMode('graph')}
              title="Interactive Stance Topology Graph"
            >
              <Icon name="radar" size={15} />
              <span>Topology</span>
            </button>
            <button
              type="button"
              className={`view-btn ${viewMode === 'compare' ? 'view-btn-active' : ''}`}
              onClick={() => setViewMode('compare')}
              title="Side-by-Side Comparison"
            >
              <Icon name="compare" size={15} />
              <span>Compare ({selectedForCompare.length}/2)</span>
            </button>
          </div>
        </div>
      </div>

      {/* Comparison Drawer Notice */}
      {selectedForCompare.length > 0 && viewMode !== 'compare' && (
        <div className="compare-floating-banner">
          <div className="banner-text">
            <strong>{selectedForCompare.length} passage{selectedForCompare.length > 1 ? 's' : ''} selected</strong> for side-by-side comparison ({selectedForCompare.join(', ')}).
          </div>
          <div className="banner-actions">
            <button
              type="button"
              className="banner-compare-btn"
              onClick={() => setViewMode('compare')}
            >
              Launch Side-by-Side Comparison →
            </button>
            <button
              type="button"
              className="banner-clear-btn"
              onClick={() => setSelectedForCompare([])}
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {/* VIEW MODE 1: EVIDENCE CARDS */}
      {viewMode === 'cards' && (
        <div className="evidence-cards-list">
          {filteredAndSortedEvidence.length === 0 ? (
            <div className="empty-dossier-notice">
              <Icon name="info" size={24} />
              <p>No evidence passages match the selected filter criteria.</p>
            </div>
          ) : (
            filteredAndSortedEvidence.map((item, idx) => {
              const isExpanded = Boolean(expandedIds[item.id]);
              const isSelectedCompare = selectedForCompare.includes(item.id);
              const stanceKey = String(item.stance || 'neutral').toLowerCase();
              const isSupport = isPassageSupporting(item);
              const isRefute = isPassageRefuting(item);
              const isUnrelated = item.is_relevant === false;
              const isIneligible = item.is_decision_eligible === false;

              const stanceBadgeLabel = isUnrelated
                ? 'OFF-TOPIC / UNRELATED'
                : isSupport
                ? 'SUPPORTS CLAIM'
                : isRefute
                ? 'CONTRADICTS CLAIM'
                : isIneligible
                ? 'CONTEXT ONLY (EXCLUDED)'
                : 'NEUTRAL CONTEXT';

              const stanceClass = isUnrelated
                ? 'stance-neutral is-unrelated-card'
                : isSupport
                ? 'stance-support'
                : isRefute
                ? 'stance-refute'
                : 'stance-neutral';

              const fullText = item.text || 'No text snippet available.';
              const isLong = fullText.length > 240;
              const displayText = isExpanded || !isLong ? fullText : `${fullText.slice(0, 240)}...`;

              const isWebUrl = typeof item.source === 'string' && (item.source.startsWith('http://') || item.source.startsWith('https://'));

              return (
                <article
                  key={item.id}
                  className={`evidence-card ${stanceClass} ${isSelectedCompare ? 'card-selected-compare' : ''}`}
                  id={`evidence-${item.id}`}
                >
                  <div className="card-header-row">
                    <div className="header-meta-left">
                      <span className="passage-id-badge">{item.id}</span>
                      <span className={`stance-badge ${stanceClass}`}>{stanceBadgeLabel}</span>
                      <span
                        className="stance-badge"
                        style={{
                          background:
                            item.source_type === 'web'
                              ? '#eff6ff'
                              : item.source_type === 'user_supplied' || item.source_type === 'user'
                              ? '#fdf4ff'
                              : 'var(--bg-surface-elevated)',
                          color:
                            item.source_type === 'web'
                              ? '#1d4ed8'
                              : item.source_type === 'user_supplied' || item.source_type === 'user'
                              ? '#86198f'
                              : 'var(--text-secondary)',
                          borderColor:
                            item.source_type === 'web'
                              ? '#bfdbfe'
                              : item.source_type === 'user_supplied' || item.source_type === 'user'
                              ? '#f0abfc'
                              : 'var(--border-default)',
                          fontSize: '0.7rem',
                        }}
                      >
                        {item.source_type === 'web'
                          ? 'LIVE WEB'
                          : item.source_type === 'user_supplied' || item.source_type === 'user'
                          ? 'SUPPLIED EVIDENCE'
                          : 'LOCAL CORPUS'}
                      </span>
                      {item.source_type === 'web' && (
                        <span
                          className="stance-badge"
                          style={{ background: '#f8fafc', color: 'var(--text-tertiary)', borderColor: 'var(--border-default)', fontSize: '0.68rem' }}
                          title="Search index snippet excerpt from live web retrieval"
                        >
                          ONLINE CITATION
                        </span>
                      )}
                      {item.is_claim_aligned === false && (
                        <span
                          className="stance-badge"
                          style={{ background: '#fef2f2', color: '#dc2626', borderColor: '#fecaca', fontSize: '0.7rem' }}
                          title="Mentions the same general topic, but does not establish or prove the specific claim"
                        >
                          Mentions topic but doesn't prove claim
                        </span>
                      )}
                      {item.contributed_to_decision && !isUnrelated && !isIneligible && (
                        <span
                          className="stance-badge"
                          style={{ background: 'var(--teal-soft)', color: 'var(--teal-500)', borderColor: 'var(--teal-border)', fontSize: '0.7rem' }}
                          title="This premise contributed decisively to the final consensus verdict"
                        >
                          DECISIVE
                        </span>
                      )}
                      {item.is_informative === false && (
                        <span className="uninformative-pill" title="NLI neutral probability exceeds 70% or entailment/contradiction mass < 25%">
                          Low Informative Signal
                        </span>
                      )}
                    </div>

                    <div className="header-meta-right">
                      {/* Checkbox for side-by-side comparison */}
                      <button
                        type="button"
                        className={`compare-select-btn ${isSelectedCompare ? 'is-selected' : ''}`}
                        onClick={() => toggleCompareSelect(item.id)}
                        aria-pressed={isSelectedCompare}
                        title={isSelectedCompare ? 'Deselect from comparison' : 'Select for side-by-side comparison (max 2)'}
                      >
                        <Icon name="compare" size={13} />
                        <span>{isSelectedCompare ? 'Comparing' : 'Compare'}</span>
                      </button>

                      {/* Copy Citation Button */}
                      <button
                        type="button"
                        className="card-action-icon-btn"
                        onClick={() => copyCitation(item)}
                        title="Copy formatted citation"
                      >
                        <Icon name={copiedId === item.id ? 'check' : 'copy'} size={15} />
                        {copiedId === item.id && <span className="copy-tooltip">Copied!</span>}
                      </button>
                    </div>
                  </div>

                  {/* Evidence Text Passage */}
                  <div className="card-passage-body">
                    <p className="passage-text">"{displayText}"</p>
                    {isLong && (
                      <button
                        type="button"
                        className="expand-text-toggle"
                        onClick={() => toggleExpand(item.id)}
                      >
                        {isExpanded ? (
                          <><span>Collapse Passage</span> <Icon name="chevronUp" size={13} /></>
                        ) : (
                          <><span>Read Full Passage ({fullText.length} chars)</span> <Icon name="chevronDown" size={13} /></>
                        )}
                      </button>
                    )}
                  </div>

                  {/* Provenance & NLI Breakdown Footer */}
                  <div className="card-footer-grid">
                    <div className="provenance-col">
                      <span className="footer-label">SOURCE PROVENANCE:</span>
                      {item.source_url || isWebUrl ? (
                        <a
                          href={item.source_url || item.source}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="source-url-link"
                          title={item.source_url || item.source}
                        >
                          <Icon name="external" size={13} />
                          <span className="source-link-text">{item.source_title || item.source || item.source_url}</span>
                        </a>
                      ) : (
                        <span className="source-plain-text">{item.source_title || item.source || 'Verified Local Corpus'}</span>
                      )}
                      {item.published_date && (
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem', fontSize: '0.72rem', color: 'var(--text-muted)', marginLeft: '0.5rem' }}>
                          <Icon name="clock" size={11} />
                          <span>{item.published_date}</span>
                        </span>
                      )}
                    </div>

                    <div className="retrieval-method-col">
                      <span className="footer-label">RETRIEVAL METHOD:</span>
                      <span className="retrieval-method-tag">
                        {item.source_type === 'web'
                          ? 'Tavily Search API (Basic)'
                          : item.source_type === 'user_supplied' || item.source_type === 'user'
                          ? 'Direct User Input'
                          : 'Hybrid RRF (k=60) • BM25 + Dense E5'}
                      </span>
                    </div>

                    <div className="nli-probabilities-col">
                      <span className="footer-label">NLI STANCE PROBABILITIES:</span>
                      <div className="nli-prob-bars">
                        <div className="prob-item" title={`P(Entailment): ${(Number(item.entailment_prob ?? (isSupport ? item.score : 0)) * 100).toFixed(1)}%`}>
                          <span className="prob-code">E:</span>
                          <span className="prob-pct">
                            {item.entailment_prob != null ? `${(item.entailment_prob * 100).toFixed(0)}%` : isSupport ? `${(item.score * 100).toFixed(0)}%` : '—'}
                          </span>
                        </div>
                        <div className="prob-item" title={`P(Contradiction): ${(Number(item.contradiction_prob ?? (isRefute ? item.score : 0)) * 100).toFixed(1)}%`}>
                          <span className="prob-code">C:</span>
                          <span className="prob-pct">
                            {item.contradiction_prob != null ? `${(item.contradiction_prob * 100).toFixed(0)}%` : isRefute ? `${(item.score * 100).toFixed(0)}%` : '—'}
                          </span>
                        </div>
                        <div className="prob-item" title={`P(Neutral): ${(Number(item.neutral_prob ?? (isNeutral ? item.score : 0)) * 100).toFixed(1)}%`}>
                          <span className="prob-code">N:</span>
                          <span className="prob-pct">
                            {item.neutral_prob != null ? `${(item.neutral_prob * 100).toFixed(0)}%` : isNeutral ? `${(item.score * 100).toFixed(0)}%` : '—'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </article>
              );
            })
          )}
        </div>
      )}

      {/* VIEW MODE 2: EVIDENCE RELATIONSHIP GRAPH */}
      {viewMode === 'graph' && (
        <div className="evidence-graph-section">
          <EvidenceGraph
            claim={claim}
            verdict={verdict}
            confidence={confidence}
            evidence={evidence}
          />
        </div>
      )}

      {/* VIEW MODE 3: SIDE-BY-SIDE COMPARISON */}
      {viewMode === 'compare' && (
        <div className="comparison-mode-view">
          <div className="compare-header-box">
            <h4>Side-by-Side Passage Comparison</h4>
            <p className="compare-instructions">
              {comparedItems.length === 2
                ? 'Comparing the 2 selected evidence passages below across documentary source, textual context, and cross-encoder stance.'
                : 'Please select 2 evidence passages using the "Compare" button on the cards to compare them side by side.'}
            </p>
          </div>

          {comparedItems.length < 2 ? (
            <div className="comparison-insufficient-prompt">
              <Icon name="compare" size={32} />
              <p>You have selected {comparedItems.length} of 2 passages.</p>
              <button
                type="button"
                className="return-docket-btn"
                onClick={() => setViewMode('cards')}
              >
                ← Return to Card Docket to Select Passages
              </button>
            </div>
          ) : (
            <div className="comparison-dual-grid">
              {comparedItems.map((item, idx) => (
                <div key={item.id} className="compare-card-column">
                  <div className="compare-col-header">
                    <span className="compare-slot-tag">DOCUMENT {idx + 1}</span>
                    <span className="compare-id-tag">{item.id}</span>
                    <span className={`stance-badge stance-${String(item.stance || 'neutral').toLowerCase()}`}>
                      {String(item.stance || 'neutral').toUpperCase()}
                    </span>
                  </div>

                  <div className="compare-body">
                    <h5 className="compare-subheading">Passage Content</h5>
                    <blockquote className="compare-quote">"{item.text}"</blockquote>

                    <h5 className="compare-subheading">Source Provenance</h5>
                    <div className="compare-source-box">
                      <code>{item.source}</code>
                    </div>

                    <h5 className="compare-subheading">Stance Probabilities</h5>
                    <div className="compare-nli-stats">
                      <div className="compare-nli-row">
                        <span>Entailment P(E):</span>
                        <strong>{item.entailment_prob != null ? `${(item.entailment_prob * 100).toFixed(1)}%` : '—'}</strong>
                      </div>
                      <div className="compare-nli-row">
                        <span>Contradiction P(C):</span>
                        <strong>{item.contradiction_prob != null ? `${(item.contradiction_prob * 100).toFixed(1)}%` : '—'}</strong>
                      </div>
                      <div className="compare-nli-row">
                        <span>Neutral P(N):</span>
                        <strong>{item.neutral_prob != null ? `${(item.neutral_prob * 100).toFixed(1)}%` : '—'}</strong>
                      </div>
                      <div className="compare-nli-row">
                        <span>Informative Signal:</span>
                        <strong>{item.is_informative ? 'HIGH' : 'LOW'}</strong>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export const EvidenceDossier = React.memo(EvidenceDossierComponent);
export default EvidenceDossier;
