import React, { useState } from 'react';
import { Icon } from './Icons';
import { VoiceNarrator } from './VoiceNarrator';
import { EvidenceGraph } from './EvidenceGraph';
import { TimelineView } from './TimelineView';
import { MultiHopInspector } from './MultiHopInspector';
import { PipelineVisualizer } from './PipelineVisualizer';

const LABELS = ['VERIFIED_TRUE', 'VERIFIED_FALSE', 'CONFLICTING_OR_CHERRYPICKED', 'INSUFFICIENT_EVIDENCE'];

const LABEL_DISPLAY = {
  VERIFIED_TRUE: 'Verified True',
  VERIFIED_FALSE: 'Verified False',
  CONFLICTING_OR_CHERRYPICKED: 'Conflicting Evidence',
  INSUFFICIENT_EVIDENCE: 'Insufficient Evidence',
};

const LABEL_CLASS = {
  VERIFIED_TRUE: 'true',
  VERIFIED_FALSE: 'false',
  CONFLICTING_OR_CHERRYPICKED: 'conflict',
  INSUFFICIENT_EVIDENCE: 'insufficient',
};

const VERDICT_EXPLANATIONS = {
  VERIFIED_TRUE: 'The available evidence supports the claim.',
  VERIFIED_FALSE: 'The available evidence contradicts the claim.',
  CONFLICTING_OR_CHERRYPICKED: 'Relevant sources support different conclusions.',
  INSUFFICIENT_EVIDENCE: 'The available evidence is not strong enough to decide reliably.',
};

function formatPct(num) {
  return `${(Number(num || 0) * 100).toFixed(1)}%`;
}

function friendlyVerdict(value) {
  return LABEL_DISPLAY[value] || String(value || 'Pending Review').replaceAll('_', ' ');
}

function ForensicReportComponent({ result, onInvestigateRelated, roundTripMs }) {
  const [copiedId, setCopiedId] = useState(null);
  const [expandedPassages, setExpandedPassages] = useState({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [stanceFilter, setStanceFilter] = useState('ALL');

  if (!result) return null;

  const evidenceList = result.evidence || [];

  const isPassageSupporting = (e) =>
    ['support', 'entailment'].includes(String(e.stance || '').toLowerCase()) &&
    e.is_informative !== false &&
    e.is_relevant !== false &&
    e.is_claim_aligned !== false &&
    e.is_figurative !== true &&
    e.is_decision_eligible !== false;

  const isPassageRefuting = (e) =>
    ['refute', 'contradiction'].includes(String(e.stance || '').toLowerCase()) &&
    e.is_informative !== false &&
    e.is_relevant !== false &&
    e.is_claim_aligned !== false &&
    e.is_figurative !== true &&
    e.is_decision_eligible !== false;

  const relevantEvidence = evidenceList.filter((e) => e.is_relevant !== false);
  const irrelevantEvidence = evidenceList.filter((e) => e.is_relevant === false);

  const supCount = evidenceList.filter(isPassageSupporting).length;
  const refCount = evidenceList.filter(isPassageRefuting).length;
  const neutCount = relevantEvidence.length - (supCount + refCount);
  const unrelatedCount = irrelevantEvidence.length;

  const toggleExpand = (id) => {
    setExpandedPassages((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Concise plain-language rationale grounded in evidence
  const generateConciseReason = () => {
    // Grounded explanation directly from the verification pipeline decision and evidence
    if (result.reason) {
      return result.reason;
    }
    if (result.summary) {
      return result.summary;
    }

    const verdict = result.abstained ? 'INSUFFICIENT_EVIDENCE' : result.verdict;
    const isAbstained = result.abstained;
    const hasDisparity = result.disparity;
    const totalEv = evidenceList.length;
    const modeLabel =
      result.mode === 'web'
        ? 'Live Web'
        : result.mode === 'hybrid'
        ? 'Combined (Local + Web)'
        : result.mode === 'user_supplied'
        ? 'Supplied Evidence'
        : 'Local Corpus';

    if (totalEv === 0) {
      return `No matching evidence passages were found in ${modeLabel.toLowerCase()}. The system abstains from making an unsupported determination.`;
    }

    if (verdict === 'VERIFIED_TRUE' && supCount > 0) {
      return `Available evidence consistently supports the claim (${supCount} supporting passage${supCount > 1 ? 's' : ''}) without contradiction.`;
    }

    if (verdict === 'VERIFIED_FALSE') {
      if (refCount > 0) {
        return `Available evidence directly contradicts the statement (${refCount} refuting passage${refCount > 1 ? 's' : ''}).`;
      }
      return `Retrieved evidence mentions related topics but does not establish or disprove the statement. Credible documentation is not sufficient to confirm or deny the claim.`;
    }

    if (verdict === 'CONFLICTING_OR_CHERRYPICKED' || hasDisparity || (supCount > 0 && refCount > 0)) {
      return `Disagreement detected: ${supCount} passage${supCount > 1 ? 's support' : ' supports'} while ${refCount} passage${refCount > 1 ? 's refute' : ' refutes'} the statement. Relevant sources disagree.`;
    }

    if (isAbstained || verdict === 'INSUFFICIENT_EVIDENCE' || (supCount === 0 && refCount === 0)) {
      return `Retrieved evidence mentions related topics but does not establish or disprove the statement. Credible documentation is not sufficient to confirm or deny the claim.`;
    }

    return result.summary || 'Adjudication completed based on multi-source evidence evaluation.';
  };

  const conciseReason = generateConciseReason();

  const filteredEvidence = evidenceList.filter((e) => {
    if (stanceFilter === 'ALL') return true;
    if (stanceFilter === 'SUPPORT') return isPassageSupporting(e);
    if (stanceFilter === 'REFUTE') return isPassageRefuting(e);
    if (stanceFilter === 'NEUTRAL') return !isPassageSupporting(e) && !isPassageRefuting(e) && e.is_relevant !== false;
    if (stanceFilter === 'UNRELATED') return e.is_relevant === false;
    return true;
  });

  // Sort evidence:
  // 1. Decisive passages first
  // 2. Substantive relevant evidence before off-topic background passages
  // 3. Higher relevance score first
  const displayEvidence = [...filteredEvidence].sort((a, b) => {
    if (a.contributed_to_decision && !b.contributed_to_decision) return -1;
    if (!a.contributed_to_decision && b.contributed_to_decision) return 1;

    const aRel = a.is_relevant !== false;
    const bRel = b.is_relevant !== false;
    if (aRel && !bRel) return -1;
    if (!aRel && bRel) return 1;

    return (b.score || 0) - (a.score || 0);
  });

  const verdictKey = result.abstained ? 'INSUFFICIENT_EVIDENCE' : result.verdict;
  const verdictClass = LABEL_CLASS[verdictKey] || 'insufficient';

  return (
    <section className="clean-results-section" id="forensic-report-results" aria-label="Verification Results">
      {/* 1. What is the Verdict? */}
      <div className={`clean-verdict-card verdict-${verdictClass}`}>
        <div className="verdict-icon-badge">
          <Icon
            name={
              verdictKey === 'VERIFIED_TRUE'
                ? 'check'
                : verdictKey === 'VERIFIED_FALSE'
                ? 'close'
                : verdictKey === 'CONFLICTING_OR_CHERRYPICKED'
                ? 'nodes'
                : 'alert'
            }
            size={24}
          />
        </div>

        <div className="verdict-primary-info">
          <span className="verdict-kicker">VERDICT</span>
          <h2 className="verdict-title">{friendlyVerdict(verdictKey)}</h2>
          <p className="verdict-subtitle">
            {VERDICT_EXPLANATIONS[verdictKey] || 'Adjudication based on available evidence.'}
          </p>

          {/* Explicit Abstention Clarification Notice */}
          {verdictKey === 'INSUFFICIENT_EVIDENCE' && (
            <div className="clean-abstention-notice">
              <Icon name="help" size={14} />
              <span>
                <strong>Clarification:</strong> Insufficient evidence does not mean the claim is false. It means credible documentation was not found to either prove or disprove it.
              </span>
            </div>
          )}

          <div className="verdict-claim-display">
            "{result.claim}"
          </div>
        </div>

        <div className="verdict-confidence-box">
          <span className="conf-title">Confidence</span>
          <div className="conf-pct">{formatPct(result.confidence)}</div>
          <div className="conf-track">
            <div
              className={`conf-bar bar-${verdictClass}`}
              style={{ width: `${Math.max(6, Math.min(100, Number(result.confidence || 0) * 100))}%` }}
            />
          </div>
          <p className="conf-note">
            Model assessment based on available evidence, not an absolute guarantee.
          </p>
        </div>
      </div>

      {/* 2. Why Did the System Reach This Verdict? */}
      <div className="clean-reason-card">
        <div className="reason-header">
          <Icon name="files" size={16} />
          <h3>Why did the system reach this verdict?</h3>
        </div>
        <p className="reason-text">
          {conciseReason}
        </p>
        <div className="reason-metadata-row">
          <span>
            <strong>Evidence evaluated:</strong> {relevantEvidence.length} relevant passage{relevantEvidence.length === 1 ? '' : 's'} (
            {supCount} supporting, {refCount} contradicting, {neutCount} neutral)
            {unrelatedCount > 0 ? ` • ${unrelatedCount} off-topic background excluded` : ''}
          </span>
          <span>&bull;</span>
          <span>
            <strong>Source mode:</strong>{' '}
            {result.mode === 'web'
              ? 'Live Web'
              : result.mode === 'hybrid'
              ? 'Combined (Local + Web)'
              : result.mode === 'user_supplied'
              ? 'Supplied Evidence'
              : 'Local Corpus'}
          </span>
        </div>
      </div>

      {/* 3. What Evidence Was Used? */}
      <div className="clean-evidence-section">
        <div className="evidence-header-row">
          <div>
            <h3 className="evidence-section-title">Evidence Used in Verification</h3>
            <span className="evidence-section-count">
              Showing {displayEvidence.length} of {evidenceList.length} passage{evidenceList.length === 1 ? '' : 's'}
              {unrelatedCount > 0 ? ` (${relevantEvidence.length} relevant, ${unrelatedCount} off-topic)` : ''}
            </span>
          </div>

          {/* Stance Filter Pills */}
          {evidenceList.length > 1 && (
            <div className="clean-filter-group">
              {[
                { id: 'ALL', label: 'All' },
                { id: 'SUPPORT', label: `Supports (${supCount})` },
                { id: 'REFUTE', label: `Contradicts (${refCount})` },
                { id: 'NEUTRAL', label: `Neutral (${neutCount})` },
                ...(unrelatedCount > 0 ? [{ id: 'UNRELATED', label: `Off-topic / Unrelated (${unrelatedCount})` }] : []),
              ].map((f) => (
                <button
                  key={f.id}
                  type="button"
                  className={`filter-btn ${stanceFilter === f.id ? 'active' : ''}`}
                  onClick={() => setStanceFilter(f.id)}
                >
                  {f.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {relevantEvidence.length === 0 && unrelatedCount > 0 && stanceFilter === 'ALL' && (
          <div className="clean-all-unrelated-alert" style={{
            background: '#f8fafc',
            border: '1px solid #e2e8f0',
            borderLeft: '4px solid #64748b',
            borderRadius: '6px',
            padding: '0.85rem 1.1rem',
            marginBottom: '1rem',
            color: '#334155'
          }}>
            <strong style={{ fontSize: '0.9rem' }}>No Topically Relevant Evidence Found in Local Corpus</strong>
            <p style={{ margin: '0.25rem 0 0', fontSize: '0.85rem', color: '#64748b' }}>
              All {unrelatedCount} retrieved passages are off-topic (do not discuss the asserted subject) and were excluded from verdict determination.
            </p>
          </div>
        )}

        {displayEvidence.length === 0 ? (
          <div className="clean-empty-evidence">
            <p>No evidence passages match the selected filter.</p>
          </div>
        ) : (
          <div className="clean-evidence-list">
            {displayEvidence.map((item, idx) => {
              const isRelevant = item.is_relevant !== false;
              const stanceKey = String(item.stance || 'neutral').toLowerCase();
              const isSupport = isRelevant && (stanceKey === 'support' || stanceKey === 'entailment');
              const isRefute = isRelevant && (stanceKey === 'refute' || stanceKey === 'contradiction');
              const isUnrelated = !isRelevant;
              const isFirstUnrelated = isUnrelated && (idx === 0 || displayEvidence[idx - 1].is_relevant !== false);
              const isExpanded = Boolean(expandedPassages[item.id || idx]);
              const fullText = item.text || '';
              const shouldTruncate = fullText.length > 220;
              const displayText = shouldTruncate && !isExpanded ? `${fullText.slice(0, 220)}...` : fullText;

              return (
                <React.Fragment key={item.id || idx}>
                  {isFirstUnrelated && relevantEvidence.length > 0 && stanceFilter === 'ALL' && (
                    <div className="clean-unrelated-divider" style={{
                      margin: '1.5rem 0 1rem',
                      padding: '0.75rem 1rem',
                      background: '#f8fafc',
                      border: '1px dashed #cbd5e1',
                      borderRadius: '6px',
                      color: '#64748b'
                    }}>
                      <div style={{ fontWeight: 700, fontSize: '0.85rem', color: '#475569' }}>
                        Off-topic Context Passages ({unrelatedCount})
                      </div>
                      <div style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '0.2rem' }}>
                        Retrieved during search but do not discuss this claim's subject; excluded from verdict determination.
                      </div>
                    </div>
                  )}
                  <article
                  key={item.id || idx}
                  className={`clean-evidence-card ${isUnrelated ? 'stance-border-neutral is-unrelated-card' : `stance-border-${isSupport ? 'support' : isRefute ? 'refute' : 'neutral'}`}`}
                >
                  <div className="card-top-row">
                    <div className="card-meta-left">
                      {/* Stance Pill */}
                      {isUnrelated ? (
                        <span className="clean-stance-badge badge-unrelated" title="Retrieved during search but off-topic; excluded from verdict determination">
                          Off-topic / Unrelated
                        </span>
                      ) : (
                        <span className={`clean-stance-badge badge-${isSupport ? 'support' : isRefute ? 'refute' : 'neutral'}`}>
                          {isSupport ? 'Supports Claim' : isRefute ? 'Contradicts Claim' : 'Neutral Context'}
                        </span>
                      )}

                      {/* Source Type Badge */}
                      <span className="clean-source-type-badge">
                        {item.source_type === 'web'
                          ? 'Live Web'
                          : item.source_type === 'user_supplied' || item.source_type === 'user'
                          ? 'Supplied Evidence'
                          : 'Local Corpus'}
                      </span>

                      {/* Specific Topic / Claim Alignment Tag */}
                      {isUnrelated ? (
                        <span className="clean-warning-badge" title="Retrieved from local corpus but does not discuss this claim's subject">
                          Off-topic / Unrelated to Claim
                        </span>
                      ) : item.is_claim_aligned === false ? (
                        <span className="clean-warning-badge" title="Mentions the same general topic, but does not establish or prove the specific claim">
                          Mentions topic but doesn't prove claim
                        </span>
                      ) : null}

                      {/* Partial coverage badge */}
                      {!isUnrelated && isSupport && item.claim_coverage === 'partial' && (
                        <span className="clean-warning-badge" title={`Partially supports the compound statement: addresses ${item.claim_components_covered?.join(', ') || 'partial components'}`}>
                          Partial Support
                        </span>
                      )}

                      {/* Decisive badge if engine marked it and has full coverage */}
                      {item.contributed_to_decision && !isUnrelated && item.claim_coverage !== 'partial' && (
                        <span className="clean-decisive-badge" title="Contributed decisively to the final consensus verdict">
                          Decisive
                        </span>
                      )}
                    </div>

                    <div className="card-meta-right">
                      {/* Copy snippet button */}
                      <button
                        type="button"
                        className="clean-copy-btn"
                        onClick={() => copyToClipboard(fullText, item.id || idx)}
                        title="Copy passage text"
                      >
                        <Icon name={copiedId === (item.id || idx) ? 'check' : 'copy'} size={13} />
                        <span>{copiedId === (item.id || idx) ? 'Copied' : 'Copy'}</span>
                      </button>
                    </div>
                  </div>

                  {/* Source Title & Provenance Link */}
                  <div className="clean-source-link-row">
                    {item.source_url ? (
                      <a
                        href={item.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="clean-source-anchor"
                        title={item.source_url}
                      >
                        <Icon name="external" size={13} />
                        <span>{item.source_title || item.source || item.source_url}</span>
                      </a>
                    ) : (
                      <span className="clean-source-text">
                        <Icon name="files" size={13} />
                        <span>Source: {item.source_title || item.source || 'Verified Reference'}</span>
                      </span>
                    )}

                    {item.published_date && (
                      <span className="clean-source-date">
                        <Icon name="clock" size={12} />
                        {item.published_date}
                      </span>
                    )}
                  </div>

                  {/* Passage Text */}
                  <blockquote className="clean-passage-quote">
                    "{displayText}"
                  </blockquote>

                  {shouldTruncate && (
                    <button
                      type="button"
                      className="clean-expand-btn"
                      onClick={() => toggleExpand(item.id || idx)}
                    >
                      {isExpanded ? 'Show less' : 'Read full passage'}
                    </button>
                  )}

                  {/* 1-Sentence Plain-English Relevance Explanation */}
                  <div className="clean-why-matters-box">
                    <strong>Why this matters: </strong>
                    {item.explanation || (
                      item.is_informative === false
                        ? 'Passage provides general context, but lacks specific details to confirm or deny the statement.'
                        : item.is_claim_aligned === false
                        ? 'Passage mentions related subjects, but does not address the specific claim assertion.'
                        : isSupport
                        ? 'Passage directly confirms the factual assertion made in the statement.'
                        : isRefute
                        ? 'Passage directly contradicts the statement by stating conflicting facts.'
                        : 'Passage discusses related topics without taking a definitive stance.'
                    )}
                  </div>
                </article>
              </React.Fragment>
            );
            })}
          </div>
        )}
      </div>

      {/* 4. Collapsible Advanced Diagnostics & Visualizations */}
      <div className="clean-advanced-accordion">
        <button
          type="button"
          className="advanced-accordion-trigger"
          onClick={() => setShowAdvanced(!showAdvanced)}
          aria-expanded={showAdvanced}
        >
          <div className="trigger-left">
            <Icon name="sliders" size={16} />
            <strong>Advanced Diagnostics &amp; Evidence Visualizations</strong>
            <span className="trigger-sub">Charts, probability spectrum, audio narration &amp; telemetry</span>
          </div>
          <span className="trigger-icon">{showAdvanced ? 'Hide ▲' : 'Show ▼'}</span>
        </button>

        {showAdvanced && (
          <div className="advanced-accordion-body">
            {/* Charts Grid */}
            <div className="diagnostics-charts-grid">
              {/* How System Weighed Possibilities */}
              <div className="diagnostic-chart-card">
                <div className="chart-card-head">
                  <Icon name="chart" size={16} />
                  <div>
                    <h4>Possibility Distribution</h4>
                    <p>Model confidence across all four potential verdicts</p>
                  </div>
                </div>

                <div className="spectrum-bars-list">
                  {LABELS.map((labelKey) => {
                    const prob = Number(result.probs?.[labelKey] ?? 0);
                    const isTop = labelKey === verdictKey;

                    return (
                      <div key={labelKey} className={`spectrum-row ${isTop ? 'row-top-verdict' : ''}`}>
                        <div className="spectrum-label-cell">
                          <span className={`spectrum-dot dot-${LABEL_CLASS[labelKey]}`} />
                          <span className="spectrum-label-text">{LABEL_DISPLAY[labelKey]}</span>
                        </div>
                        <div className="spectrum-track-cell">
                          <div
                            className={`spectrum-fill fill-${LABEL_CLASS[labelKey]}`}
                            style={{ width: `${Math.max(1, prob * 100)}%` }}
                          />
                        </div>
                        <span className="spectrum-val-cell">{formatPct(prob)}</span>
                      </div>
                    );
                  })}
                </div>

                <div className="chart-interpretation-box">
                  <strong>Interpretation: </strong>
                  The system identified highest confidence for <strong>{friendlyVerdict(verdictKey)}</strong> ({formatPct(result.confidence)}).
                </div>
              </div>

              {/* Evidence Agreement & Disagreement */}
              <div className="diagnostic-chart-card">
                <div className="chart-card-head">
                  <Icon name="nodes" size={16} />
                  <div>
                    <h4>Evidence Agreement &amp; Disagreement</h4>
                    <p>Checks whether sources agree or take opposing stances</p>
                  </div>
                </div>

                <div className="stance-count-summary">
                  <div className="stance-count-item">
                    <span className="dot-support" />
                    <span>Supporting:</span>
                    <strong>{supCount}</strong>
                  </div>
                  <div className="stance-count-item">
                    <span className="dot-refute" />
                    <span>Contradicting:</span>
                    <strong>{refCount}</strong>
                  </div>
                  <div className="stance-count-item">
                    <span className="dot-neutral" />
                    <span>Neutral:</span>
                    <strong>{neutCount}</strong>
                  </div>
                </div>

                <div className="chart-interpretation-box">
                  <strong>Interpretation: </strong>
                  {result.disparity
                    ? 'Disagreement detected: Supporting and refuting sources take contradictory positions on this claim.'
                    : supCount > 0 && refCount === 0
                    ? 'Consensus: All active informative sources corroborate the claim without contradiction.'
                    : refCount > 0 && supCount === 0
                    ? 'Consistent refutation: Active sources contradict the statement.'
                    : 'Neutral: Passages do not take definitive opposing stances.'}
                </div>
              </div>
            </div>

            {/* Accessible Audio Briefing */}
            <div className="diagnostic-audio-card" style={{ marginTop: '1rem' }}>
              <VoiceNarrator result={result} active={true} />
            </div>

            {/* Pipeline Trace & System Specs */}
            <div className="diagnostic-telemetry-card" style={{ marginTop: '1rem' }}>
              <div className="telemetry-compact-grid">
                <div>
                  <span className="t-label">Model Checkpoint:</span>
                  <code className="t-val">DeBERTa-v3-base-mnli-fever-anli</code>
                </div>
                <div>
                  <span className="t-label">Retrieval Engine:</span>
                  <span className="t-val">BM25 + Multilingual E5 Dense RRF</span>
                </div>
                <div>
                  <span className="t-label">Inference Device:</span>
                  <span className="t-val">{(result.raw?.device || 'CUDA / Local').toUpperCase()}</span>
                </div>
                <div>
                  <span className="t-label">Server Latency:</span>
                  <span className="t-val">{result.server_latency_ms ? `${result.server_latency_ms} ms` : 'Local'}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

export const ForensicReport = React.memo(ForensicReportComponent);
export default ForensicReport;
