import React from 'react';
import { Icon } from './Icons.jsx';

/**
 * Verdict styling definitions with canonical color tokens and icons.
 */
const VERDICT_CONFIG = {
  VERIFIED_TRUE: {
    label: 'VERIFIED TRUE',
    badgeClass: 'verdict-true',
    icon: 'checkCircle',
    headline: 'Empirical Documentary Evidence Confirms Proposition',
  },
  VERIFIED_FALSE: {
    label: 'VERIFIED FALSE',
    badgeClass: 'verdict-false',
    icon: 'xCircle',
    headline: 'Documentary Evidence Directly Refutes Proposition',
  },
  CONFLICTING_OR_CHERRYPICKED: {
    label: 'CONFLICTING EVIDENCE',
    badgeClass: 'verdict-conflict',
    icon: 'alertTriangle',
    headline: 'Cross-Passage Stance Disparity: Irreconcilable Contradictions',
  },
  INSUFFICIENT_EVIDENCE: {
    label: 'INSUFFICIENT EVIDENCE',
    badgeClass: 'verdict-insufficient',
    icon: 'shieldAlert',
    headline: 'Evidence Inconclusive or Lacks Required Factual Coverage',
  },
};

/**
 * VerificationReveal Component
 * Stage 10 & 11: NLI Cross-Encoder Hypothesis Evaluation & Calibrated Verdict Reveal.
 * Highlights the authentic hypothesis evaluated by DeBERTa-v3, temperature calibration,
 * cross-passage disparity checks, and the final veracity verdict.
 */
export function VerificationReveal({
  nliDetails = {},
  resultDetails = {},
  canonicalReason = '',
  retrievalQueryUsed = '',
}) {
  const hypothesisUsed = nliDetails.hypothesis_used || '';
  const hypothesisSource = nliDetails.hypothesis_source || 'canonical_sanitized_user_claim';
  const modelName = nliDetails.model || 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli';
  const device = nliDetails.device || 'cpu';
  const temperature = nliDetails.calibration_temperature ?? 4.1039;
  const inferenceMs = nliDetails.inference_ms || 0;

  const verdictKey = resultDetails.verdict || 'INSUFFICIENT_EVIDENCE';
  const verdictMeta = VERDICT_CONFIG[verdictKey] || VERDICT_CONFIG.INSUFFICIENT_EVIDENCE;
  const confidence = Number(resultDetails.confidence || 0);
  const confidencePct = (confidence * 100).toFixed(1);

  const probabilities = resultDetails.probabilities || {};
  const isAbstained = Boolean(resultDetails.abstained);
  const abstentionReason = resultDetails.abstention_reason;
  const hasDisparity = Boolean(resultDetails.has_cross_passage_disparity);
  const evidenceCounts = resultDetails.evidence_counts || {};

  return (
    <div className="verification-reveal-workspace" role="region" aria-label="NLI Verification and Adjudication Workspace">
      {/* Stage Header */}
      <div className="verification-header">
        <div className="verification-header-meta">
          <span className="cinematic-eyebrow">
            <span className="eyebrow-glow-dot" />
            STAGE 10 &amp; 11 &bull; NLI VERIFICATION &amp; ADJUDICATION
          </span>
          <h3 className="cinematic-scene-title">
            DeBERTa-v3 <span>Cross-Encoder Adjudication.</span>
          </h3>
          <p className="cinematic-scene-desc">
            Direct token cross-attention comparing retrieved documentary evidence against the canonical claim hypothesis.
          </p>
        </div>

        <div className="calibration-status-chip">
          <Icon name="activity" size={14} />
          <span>Platt Temp: <strong>T={Number(temperature).toFixed(4)}</strong></span>
        </div>
      </div>

      {/* Hypothesis Distinction Banner */}
      <div className="hypothesis-reassurance-card">
        <div className="hypo-card-top">
          <div className="hypo-meta-left">
            <span className="hypo-badge">CANONICAL NLI HYPOTHESIS</span>
            <span className="hypo-source-note">Source: {hypothesisSource}</span>
          </div>
          <div className="hypo-meta-right">
            <span>Evaluated Model: <code>{modelName.split('/').pop()}</code> &bull; {device.toUpperCase()}</span>
          </div>
        </div>

        <div className="hypo-text-display">
          <Icon name="check" size={18} />
          <strong className="hypo-claim-text">"{hypothesisUsed}"</strong>
        </div>

        <div className="hypo-comparison-row">
          <div className="hypo-comp-col">
            <span className="comp-label">RETRIEVAL SEARCH TARGET:</span>
            <span className="comp-value">"{retrievalQueryUsed}"</span>
          </div>
          <div className="hypo-comp-arrow">&ne;</div>
          <div className="hypo-comp-col">
            <span className="comp-label">VERIFIED PROPOSITION HYPOTHESIS:</span>
            <span className="comp-value">"{hypothesisUsed}"</span>
          </div>
        </div>

        <p className="hypo-guarantee-note">
          <Icon name="shield" size={13} />
          <strong>Guaranteed Alignment:</strong> The NLI cross-encoder adjudicates the exact factual proposition submitted by the user. Text normalization was strictly restricted to retrieval indexing.
        </p>
      </div>

      {/* Final Verdict Banner */}
      <div className={`verdict-hero-card ${verdictMeta.badgeClass}`}>
        <div className="verdict-hero-glow" />

        <div className="verdict-hero-content">
          <div className="verdict-badge-row">
            <div className="verdict-main-badge">
              <Icon name={verdictMeta.icon} size={24} />
              <span>{verdictMeta.label}</span>
            </div>

            <div className="verdict-confidence-badge">
              <span className="conf-label">CALIBRATED CONFIDENCE</span>
              <span className="conf-value">{confidencePct}%</span>
            </div>

            {isAbstained && (
              <div className="verdict-abstained-badge">
                <Icon name="shieldAlert" size={14} />
                ABSTAINED (UNCERTAINTY SAFETY)
              </div>
            )}

            {hasDisparity && (
              <div className="verdict-disparity-badge">
                <Icon name="alertTriangle" size={14} />
                CROSS-PASSAGE DISPARITY
              </div>
            )}
          </div>

          <h4 className="verdict-headline">{verdictMeta.headline}</h4>

          <p className="verdict-canonical-reason">
            {canonicalReason || 'Adjudication completed by VERITAS-X verification pipeline.'}
          </p>

          {isAbstained && abstentionReason && (
            <div className="abstention-callout">
              <Icon name="info" size={14} />
              <span><strong>Abstention Guardrail:</strong> {abstentionReason}</span>
            </div>
          )}
        </div>
      </div>

      {/* Probability Distribution & Evidence Tally Grid */}
      <div className="verification-breakdown-grid">
        {/* Class Probabilities */}
        <div className="breakdown-card">
          <div className="breakdown-head">
            <Icon name="pieChart" size={16} />
            <span>CALIBRATED PROBABILITY DISTRIBUTION</span>
          </div>
          <div className="prob-bars-list">
            {[
              { key: 'VERIFIED_TRUE', label: 'Verified True', val: probabilities.VERIFIED_TRUE ?? 0, color: 'bar-true' },
              { key: 'VERIFIED_FALSE', label: 'Verified False', val: probabilities.VERIFIED_FALSE ?? 0, color: 'bar-false' },
              { key: 'CONFLICTING_OR_CHERRYPICKED', label: 'Conflicting / Disparity', val: probabilities.CONFLICTING_OR_CHERRYPICKED ?? 0, color: 'bar-conflict' },
              { key: 'INSUFFICIENT_EVIDENCE', label: 'Insufficient Evidence', val: probabilities.INSUFFICIENT_EVIDENCE ?? 0, color: 'bar-insufficient' },
            ].map((p) => {
              const pct = (p.val * 100).toFixed(1);
              return (
                <div key={p.key} className="prob-bar-row">
                  <div className="prob-bar-labels">
                    <span className="prob-label">{p.label}</span>
                    <strong className="prob-percent">{pct}%</strong>
                  </div>
                  <div className="prob-track">
                    <div className={`prob-fill ${p.color}`} style={{ width: `${pct}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Evidence Breakdown Tally */}
        <div className="breakdown-card">
          <div className="breakdown-head">
            <Icon name="database" size={16} />
            <span>EVIDENCE ADJUDICATION TALLY</span>
          </div>
          <div className="tally-metrics-grid">
            <div className="tally-box">
              <span className="tally-num num-sup">{evidenceCounts.eligible_supporting ?? 0}</span>
              <span className="tally-label">Eligible Supporting</span>
            </div>
            <div className="tally-box">
              <span className="tally-num num-ref">{evidenceCounts.eligible_refuting ?? 0}</span>
              <span className="tally-label">Eligible Refuting</span>
            </div>
            <div className="tally-box">
              <span className="tally-num num-neu">{evidenceCounts.relevant_neutral ?? 0}</span>
              <span className="tally-label">Relevant Neutral</span>
            </div>
            <div className="tally-box">
              <span className="tally-num num-off">{evidenceCounts.off_topic ?? 0}</span>
              <span className="tally-label">Off-Topic (Excluded)</span>
            </div>
          </div>
          <div className="tally-meta-footer">
            <span>Inference Latency: <strong>{Number(inferenceMs).toFixed(1)}ms</strong></span>
            <span>Duplicate Passages Deduplicated: <strong>{evidenceCounts.duplicates ?? 0}</strong></span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VerificationReveal;
