import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Icon } from './Icons.jsx';
import { WordCorrectionAnimation } from './WordCorrectionAnimation.jsx';
import { MeaningPreservationPanel } from './MeaningPreservationPanel.jsx';
import { TokenInspector } from './TokenInspector.jsx';
import { SyntaxRelationshipView } from './SyntaxRelationshipView.jsx';
import { ClaimStructureView } from './ClaimStructureView.jsx';
import { StageDetailsModal } from './StageDetailsModal.jsx';

import { PHASES } from '../lib/stages.js';


const VERDICT_DISPLAY = {
  VERIFIED_TRUE: { label: 'Verified True', cssClass: 'phase-verdict-true', icon: 'checkCircle' },
  VERIFIED_FALSE: { label: 'Verified False', cssClass: 'phase-verdict-false', icon: 'xCircle' },
  CONFLICTING_OR_CHERRYPICKED: { label: 'Conflicting Evidence', cssClass: 'phase-verdict-conflict', icon: 'alertTriangle' },
  INSUFFICIENT_EVIDENCE: { label: 'Insufficient Evidence', cssClass: 'phase-verdict-insufficient', icon: 'shieldAlert' },
};

/**
 * NLPVisualization — Phase 2 Rewrite
 *
 * - Shows completed results immediately (no forced autoplay).
 * - Groups 11 internal stages into 6 clear user-facing phases.
 * - Provides an optional "Show How It Worked" walkthrough.
 * - Preserves full 11-stage trace in Technical Details.
 * - Never fabricates stages, results, or live-processing events.
 */
export function NLPVisualization({ result = {}, onClose }) {
  const trace = useMemo(
    () => (Array.isArray(result.processing_trace) ? result.processing_trace : []),
    [result.processing_trace]
  );

  const prefersReducedMotion =
    typeof window !== 'undefined' &&
    window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Phase navigation — default to final phase (verdict) so user sees result immediately
  const [activePhaseIndex, setActivePhaseIndex] = useState(PHASES.length - 1);
  // Walkthrough playback state — NOT playing by default
  const [isWalkthroughPlaying, setIsWalkthroughPlaying] = useState(false);
  const [walkthroughSpeed, setWalkthroughSpeed] = useState(1);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  const timerRef = useRef(null);
  // Track result identity to reset on new investigations
  const resultIdRef = useRef(null);

  // Reset state when result changes (new investigation)
  useEffect(() => {
    const newId = result?.id || result?.claim_id || result?.claim || null;
    if (newId !== resultIdRef.current) {
      resultIdRef.current = newId;
      setActivePhaseIndex(PHASES.length - 1);
      setIsWalkthroughPlaying(false);
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    }
  }, [result]);

  // Walkthrough auto-advance
  useEffect(() => {
    if (!isWalkthroughPlaying || PHASES.length <= 1) {
      if (timerRef.current) clearTimeout(timerRef.current);
      return;
    }
    if (activePhaseIndex >= PHASES.length - 1) {
      setIsWalkthroughPlaying(false);
      return;
    }
    const intervalMs = 2400 / walkthroughSpeed;
    timerRef.current = setTimeout(() => {
      setActivePhaseIndex((prev) => {
        if (prev < PHASES.length - 1) return prev + 1;
        setIsWalkthroughPlaying(false);
        return prev;
      });
    }, intervalMs);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isWalkthroughPlaying, activePhaseIndex, walkthroughSpeed]);

  // Walkthrough controls
  const handleStartWalkthrough = useCallback(() => {
    setActivePhaseIndex(0);
    setIsWalkthroughPlaying(!prefersReducedMotion);
  }, [prefersReducedMotion]);

  const handlePause = useCallback(() => setIsWalkthroughPlaying(false), []);

  const handleResume = useCallback(() => {
    if (activePhaseIndex >= PHASES.length - 1) {
      setActivePhaseIndex(0);
    }
    setIsWalkthroughPlaying(true);
  }, [activePhaseIndex]);

  const handleNext = useCallback(() => {
    setIsWalkthroughPlaying(false);
    setActivePhaseIndex((prev) => Math.min(prev + 1, PHASES.length - 1));
  }, []);

  const handlePrev = useCallback(() => {
    setIsWalkthroughPlaying(false);
    setActivePhaseIndex((prev) => Math.max(prev - 1, 0));
  }, []);

  const handleSkip = useCallback(() => {
    setIsWalkthroughPlaying(false);
    setActivePhaseIndex(PHASES.length - 1);
  }, []);

  const handleReplay = useCallback(() => {
    setActivePhaseIndex(0);
    setIsWalkthroughPlaying(!prefersReducedMotion);
  }, [prefersReducedMotion]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e) => {
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        setIsWalkthroughPlaying((p) => !p);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        handleNext();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        handlePrev();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        handleReplay();
      } else if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        handleSkip();
      } else if (e.key === 'Escape') {
        if (showTechnicalDetails) setShowTechnicalDetails(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleNext, handlePrev, handleReplay, handleSkip, showTechnicalDetails]);

  // Extract data from result
  const rawOriginalClaim = result.original_claim || result.claim || '';
  const sanitizedClaim = result.sanitized_claim || result.claim || '';
  const normalizedClaim = result.normalized_claim || sanitizedClaim;
  const normalizationStatus = result.normalization_status || 'accepted';
  const normCorrections = Array.isArray(result.normalization_corrections)
    ? result.normalization_corrections
    : [];
  const normWarnings = Array.isArray(result.normalization_warnings)
    ? result.normalization_warnings
    : [];
  const structuredClaim = result.structured_claim || {};
  const retrievalQueryUsed = result.retrieval_query_used || sanitizedClaim;
  const nliHypothesisUsed = result.nli_hypothesis_used || sanitizedClaim;
  const evidenceList = result.evidence || [];
  const evidenceCounts = result.evidence_counts || {};

  // Active phase
  const activePhase = PHASES[activePhaseIndex];

  // Get trace stages for a phase
  const getPhaseTraceStages = (phase) =>
    phase.stageIds
      .map((sid) => trace.find((t) => t.stage === sid))
      .filter(Boolean);

  // Get phase status from its trace stages
  const getPhaseStatus = (phase) => {
    const stages = getPhaseTraceStages(phase);
    if (stages.length === 0) return 'unavailable';
    if (stages.some((s) => s.status === 'failed')) return 'failed';
    if (stages.some((s) => s.status === 'uncertain' || s.status === 'fallback')) return 'attention';
    if (stages.every((s) => s.status === 'completed')) return 'completed';
    return 'completed'; // All backend stages are completed in the response
  };

  // Get active trace details for current phase
  const activeTraceStages = getPhaseTraceStages(activePhase);
  const activeDetails = activeTraceStages.length > 0
    ? activeTraceStages.reduce((acc, s) => ({ ...acc, ...(s.details || {}) }), {})
    : {};

  // Verdict data
  const verdictKey = result.abstained ? 'INSUFFICIENT_EVIDENCE' : (result.verdict || 'INSUFFICIENT_EVIDENCE');
  const verdictMeta = VERDICT_DISPLAY[verdictKey] || VERDICT_DISPLAY.INSUFFICIENT_EVIDENCE;
  const confidence = Number(result.confidence || 0);
  const confidencePct = (confidence * 100).toFixed(1);

  // ======================== PHASE CONTENT RENDERERS ========================

  const renderClaimPhase = () => (
    <div className="phase-content phase-claim">
      <div className="phase-answer-block">
        <p className="phase-question">{activePhase.question}</p>
        <blockquote className="claim-display-quote">"{rawOriginalClaim}"</blockquote>
        <div className="claim-metadata-row">
          <span>{rawOriginalClaim.length} characters</span>
          <span>{rawOriginalClaim.split(/\s+/).filter(Boolean).length} words</span>
        </div>
      </div>
    </div>
  );

  const renderCleanupPhase = () => {
    const hasCorrections = normCorrections.length > 0;
    const isUncertain = normalizationStatus === 'uncertain';
    const isReverted = normalizationStatus === 'reverted';
    const isAccepted = normalizationStatus === 'accepted';
    const claimChanged = rawOriginalClaim !== normalizedClaim;

    // Determine which spelling stage data to use
    const spellingStage = trace.find((s) => s.stage === 'spelling_normalization');
    const meaningStage = trace.find((s) => s.stage === 'meaning_preservation');
    const spellingDetails = spellingStage?.details || {};
    const meaningDetails = meaningStage?.details || {};

    return (
      <div className="phase-content phase-cleanup">
        <p className="phase-question">{activePhase.question}</p>

        {!hasCorrections && !claimChanged && (
          <div className="phase-info-card phase-info-neutral">
            <Icon name="check" size={16} />
            <div>
              <strong>No corrections needed</strong>
              <p>The submitted text was clear and well-formed. No spelling or formatting changes were made.</p>
            </div>
          </div>
        )}

        {hasCorrections && (
          <WordCorrectionAnimation
            originalClaim={rawOriginalClaim || sanitizedClaim}
            normalizedClaim={normalizedClaim}
            corrections={spellingDetails.corrections || normCorrections}
            wordEvents={spellingDetails.word_correction_events || []}
            warnings={spellingDetails.warnings || normWarnings}
            hasAmbiguousCorrections={spellingDetails.has_ambiguous_corrections}
            normalizationStatus={normalizationStatus}
          />
        )}

        {isUncertain && (
          <div className="phase-info-card phase-info-warning">
            <Icon name="alertCircle" size={16} />
            <div>
              <strong>Uncertain corrections</strong>
              <p>Some corrections may change the meaning. The original wording was kept as a safety fallback for verification.</p>
            </div>
          </div>
        )}

        {isReverted && (
          <div className="phase-info-card phase-info-warning">
            <Icon name="alertTriangle" size={16} />
            <div>
              <strong>Corrections reverted</strong>
              <p>The meaning-preservation check determined that corrections could alter the claim's intent. The original text was used for verification.</p>
            </div>
          </div>
        )}

        {isAccepted && claimChanged && (
          <div className="phase-info-card phase-info-success">
            <Icon name="checkCircle" size={16} />
            <div>
              <strong>Corrections accepted</strong>
              <p>Meaning-preservation checks confirmed the corrections preserve the original intent.</p>
            </div>
          </div>
        )}

        {claimChanged && (
          <div className="cleanup-comparison">
            <div className="cleanup-comparison-item">
              <span className="cleanup-label">Original</span>
              <span className="cleanup-text">"{rawOriginalClaim}"</span>
            </div>
            <div className="cleanup-comparison-item">
              <span className="cleanup-label">After cleanup</span>
              <span className="cleanup-text cleanup-text-normalized">"{normalizedClaim}"</span>
            </div>
          </div>
        )}

        {/* Retrieval query vs NLI hypothesis distinction */}
        {claimChanged && isAccepted && (
          <div className="cleanup-query-distinction">
            <div className="distinction-item">
              <span className="distinction-label">Evidence search used</span>
              <span className="distinction-value">"{retrievalQueryUsed}"</span>
            </div>
            <div className="distinction-item">
              <span className="distinction-label">Verification tested</span>
              <span className="distinction-value">"{nliHypothesisUsed}"</span>
              <span className="distinction-note">Original wording preserved for accuracy</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderAnalysisPhase = () => {
    const tokStage = trace.find((s) => s.stage === 'tokenization');
    const lingStage = trace.find((s) => s.stage === 'linguistic_analysis');
    const extStage = trace.find((s) => s.stage === 'claim_extraction');
    const tokDetails = tokStage?.details || {};
    const lingDetails = lingStage?.details || {};
    const extDetails = extStage?.details || {};
    const hasTokens = tokDetails.tokens && tokDetails.tokens.length > 0;
    const hasSVO = lingDetails.subject_candidate || lingDetails.predicate_candidate;
    const hasClaim = structuredClaim && (structuredClaim.subject || structuredClaim.predicate);

    if (!hasTokens && !hasSVO && !hasClaim) {
      return (
        <div className="phase-content phase-analysis">
          <p className="phase-question">{activePhase.question}</p>
          <div className="phase-info-card phase-info-neutral">
            <Icon name="alertCircle" size={16} />
            <div>
              <strong>Could not confidently identify the sentence structure</strong>
              <p>The linguistic analyzer was unable to determine a clear subject, predicate, or object for this input. This does not prevent verification — evidence matching proceeds based on the full claim text.</p>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="phase-content phase-analysis">
        <p className="phase-question">{activePhase.question}</p>

        {hasClaim && (
          <div className="analysis-structure-summary">
            {structuredClaim.subject && (
              <div className="structure-item">
                <span className="structure-label">Subject</span>
                <span className="structure-value">{structuredClaim.subject}</span>
              </div>
            )}
            {structuredClaim.predicate && (
              <div className="structure-item">
                <span className="structure-label">Action</span>
                <span className="structure-value">
                  {structuredClaim.is_negated && <span className="negation-marker">not </span>}
                  {structuredClaim.predicate}
                </span>
              </div>
            )}
            {structuredClaim.object && (
              <div className="structure-item">
                <span className="structure-label">Object</span>
                <span className="structure-value">{structuredClaim.object}</span>
              </div>
            )}
          </div>
        )}

        {/* Key attributes */}
        <div className="analysis-attributes">
          {structuredClaim.is_negated && (
            <span className="attr-chip attr-negation">
              <Icon name="minus" size={12} /> Negated
              {structuredClaim.negation_scope ? ` (${structuredClaim.negation_scope.length > 30 ? 'scoped' : structuredClaim.negation_scope})` : ''}
            </span>
          )}
          {structuredClaim.is_conditional && (
            <span className="attr-chip attr-condition"><Icon name="git-merge" size={12} /> Conditional</span>
          )}
          {structuredClaim.is_causal && (
            <span className="attr-chip attr-causal"><Icon name="activity" size={12} /> Causal</span>
          )}
          {structuredClaim.modality && structuredClaim.modality !== 'certain' && (
            <span className="attr-chip attr-modality"><Icon name="help" size={12} /> {structuredClaim.modality}</span>
          )}
          {lingDetails.sentence_type && (
            <span className="attr-chip attr-type"><Icon name="edit" size={12} /> {lingDetails.sentence_type}</span>
          )}
          {structuredClaim.named_entities && structuredClaim.named_entities.length > 0 && (
            <span className="attr-chip attr-entity"><Icon name="user" size={12} /> {structuredClaim.named_entities.join(', ')}</span>
          )}
          {structuredClaim.numeric_entities && structuredClaim.numeric_entities.length > 0 && (
            <span className="attr-chip attr-numeric"><Icon name="hash" size={12} /> Contains numbers</span>
          )}
        </div>
      </div>
    );
  };

  const renderSearchPhase = () => {
    const retStage = trace.find((s) => s.stage === 'retrieval');
    const retDetails = retStage?.details || {};
    const retStatus = result.retrieval_status || {};
    const mode = result.mode || 'local';
    const passageCount = evidenceList.length;

    const modeLabel =
      mode === 'web' ? 'Live Web' :
      mode === 'hybrid' ? 'Local + Web' :
      'Local Corpus';

    return (
      <div className="phase-content phase-search">
        <p className="phase-question">{activePhase.question}</p>

        <div className="search-summary-card">
          <div className="search-stat">
            <span className="search-stat-value">{passageCount}</span>
            <span className="search-stat-label">passage{passageCount !== 1 ? 's' : ''} found</span>
          </div>
          <div className="search-stat">
            <span className="search-stat-value">{modeLabel}</span>
            <span className="search-stat-label">source mode</span>
          </div>
          {retDetails.duration_ms && (
            <div className="search-stat">
              <span className="search-stat-value">{Number(retDetails.duration_ms).toFixed(0)}ms</span>
              <span className="search-stat-label">search time</span>
            </div>
          )}
        </div>

        <div className="search-query-box">
          <span className="search-query-label">Search query used:</span>
          <span className="search-query-text">"{retDetails.retrieval_query || retrievalQueryUsed}"</span>
          {retDetails.query_variant === 'normalized' && rawOriginalClaim !== normalizedClaim && (
            <span className="search-query-note">Spelling-corrected version used for better search results</span>
          )}
        </div>

        {passageCount === 0 && (
          <div className="phase-info-card phase-info-warning">
            <Icon name="alertCircle" size={16} />
            <div>
              <strong>No matching evidence found</strong>
              <p>The search did not find any passages relevant to this claim. The verdict reflects insufficient evidence, not a determination of truth or falsity.</p>
            </div>
          </div>
        )}

        {retStatus.error && (
          <div className="phase-info-card phase-info-error">
            <Icon name="xCircle" size={16} />
            <div>
              <strong>Search error</strong>
              <p>{retStatus.error}</p>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderReviewPhase = () => {
    const supCount = evidenceCounts.eligible_supporting ?? 0;
    const refCount = evidenceCounts.eligible_refuting ?? 0;
    const neutCount = evidenceCounts.relevant_neutral ?? 0;
    const offTopicCount = evidenceCounts.off_topic ?? 0;
    const total = evidenceList.length;

    return (
      <div className="phase-content phase-review">
        <p className="phase-question">{activePhase.question}</p>

        {total === 0 ? (
          <div className="phase-info-card phase-info-neutral">
            <Icon name="files" size={16} />
            <div>
              <strong>No evidence to review</strong>
              <p>No passages were retrieved for this claim.</p>
            </div>
          </div>
        ) : (
          <>
            <div className="review-tally">
              {supCount > 0 && (
                <div className="tally-item tally-support">
                  <span className="tally-count">{supCount}</span>
                  <span className="tally-label">Supporting</span>
                </div>
              )}
              {refCount > 0 && (
                <div className="tally-item tally-refute">
                  <span className="tally-count">{refCount}</span>
                  <span className="tally-label">Contradicting</span>
                </div>
              )}
              {neutCount > 0 && (
                <div className="tally-item tally-neutral">
                  <span className="tally-count">{neutCount}</span>
                  <span className="tally-label">Neutral</span>
                </div>
              )}
              {offTopicCount > 0 && (
                <div className="tally-item tally-offtopic">
                  <span className="tally-count">{offTopicCount}</span>
                  <span className="tally-label">Off-topic (excluded)</span>
                </div>
              )}
            </div>

            {/* Top evidence preview */}
            <div className="review-evidence-preview">
              {evidenceList
                .filter((e) => e.is_relevant !== false)
                .slice(0, 3)
                .map((item, idx) => {
                  const stanceKey = String(item.stance || 'neutral').toLowerCase();
                  const isSupport = stanceKey === 'support' || stanceKey === 'entailment';
                  const isRefute = stanceKey === 'refute' || stanceKey === 'contradiction';
                  const stanceClass = isSupport ? 'support' : isRefute ? 'refute' : 'neutral';
                  const stanceLabel = isSupport ? 'Supports' : isRefute ? 'Contradicts' : 'Neutral';
                  const text = item.text || '';
                  const displayText = text.length > 160 ? text.slice(0, 160) + '…' : text;

                  return (
                    <div key={item.id || idx} className={`review-evidence-card review-stance-${stanceClass}`}>
                      <div className="review-card-header">
                        <span className={`review-stance-badge badge-${stanceClass}`}>{stanceLabel}</span>
                        <span className="review-source">{item.source_title || item.source || 'Evidence'}</span>
                      </div>
                      <p className="review-passage-text">"{displayText}"</p>
                      {item.explanation && (
                        <p className="review-explanation">{item.explanation}</p>
                      )}
                    </div>
                  );
                })}
              {evidenceList.filter((e) => e.is_relevant !== false).length > 3 && (
                <p className="review-more-note">
                  + {evidenceList.filter((e) => e.is_relevant !== false).length - 3} more passage(s) in the full report below
                </p>
              )}
            </div>
          </>
        )}
      </div>
    );
  };

  const renderVerdictPhase = () => {
    const reason = result.reason || result.summary || 'Verification completed.';
    const isAbstained = Boolean(result.abstained);
    const hasDisparity = Boolean(result.disparity || result.has_cross_passage_disparity);

    return (
      <div className="phase-content phase-verdict">
        <p className="phase-question">{activePhase.question}</p>

        <div className={`verdict-result-card ${verdictMeta.cssClass}`}>
          <div className="verdict-result-icon">
            <Icon name={verdictMeta.icon} size={28} />
          </div>
          <div className="verdict-result-main">
            <h3 className="verdict-result-label">{verdictMeta.label}</h3>
            <div className="verdict-confidence-row">
              <span className="verdict-confidence-pct">{confidencePct}% confidence</span>
              <div className="verdict-confidence-track">
                <div
                  className="verdict-confidence-fill"
                  style={{ width: `${Math.max(4, Math.min(100, confidence * 100))}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        <p className="verdict-reason-text">{reason}</p>

        {isAbstained && (
          <div className="phase-info-card phase-info-warning">
            <Icon name="shieldAlert" size={16} />
            <div>
              <strong>Confidence too low for a definitive verdict</strong>
              <p>{result.abstention_reason || 'The system could not reach sufficient confidence to assert a verdict.'}</p>
            </div>
          </div>
        )}

        {hasDisparity && (
          <div className="phase-info-card phase-info-warning">
            <Icon name="alertTriangle" size={16} />
            <div>
              <strong>Evidence disagreement detected</strong>
              <p>Some passages support while others contradict this claim. The verdict reflects this conflict.</p>
            </div>
          </div>
        )}

        <p className="verdict-disclaimer">
          This assessment is based on available evidence under calibrated confidence scoring. It represents evidence alignment, not an absolute guarantee.
        </p>
      </div>
    );
  };

  // Phase content dispatcher
  const renderPhaseContent = () => {
    if (trace.length === 0 && activePhase.id !== 'verdict') {
      return (
        <div className="phase-content phase-unavailable">
          <div className="phase-info-card phase-info-neutral">
            <Icon name="alertCircle" size={16} />
            <div>
              <strong>Processing details unavailable</strong>
              <p>The server response did not include a processing trace. The verdict and evidence are still valid.</p>
            </div>
          </div>
        </div>
      );
    }

    switch (activePhase.id) {
      case 'claim': return renderClaimPhase();
      case 'cleanup': return renderCleanupPhase();
      case 'analysis': return renderAnalysisPhase();
      case 'search': return renderSearchPhase();
      case 'review': return renderReviewPhase();
      case 'verdict': return renderVerdictPhase();
      default: return renderVerdictPhase();
    }
  };

  // ======================== RENDER ========================

  return (
    <div className="nlp-visualization-container" role="region" aria-label="Claim Verification Workflow">
      {/* Header */}
      <div className="nlp-header-section">
        <div className="nlp-header-top">
          <div className="nlp-header-titles">
            <h3 className="nlp-panel-title">How It Was Verified</h3>
            <p className="nlp-panel-subtitle">
              {isWalkthroughPlaying
                ? 'Walking through the verification steps…'
                : 'Explore each step of the verification process.'}
            </p>
          </div>
          <div className="nlp-header-actions">
            <span className={`nlp-status-badge ${isWalkthroughPlaying ? 'status-replaying' : 'status-completed'} strip-status-pill`}>
              {isWalkthroughPlaying ? 'Replaying' : 'Completed'}
            </span>
            {onClose && (
              <button type="button" className="nlp-close-btn workspace-close-btn" onClick={onClose} aria-label="Close">
                <Icon name="x" size={15} />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* 6-Phase Compact Stepper */}
      <nav className="phase-stepper" aria-label="Verification Steps">
        <div className="phase-stepper-track">
          {PHASES.map((phase, idx) => {
            const status = getPhaseStatus(phase);
            const isActive = idx === activePhaseIndex;
            const isPast = idx < activePhaseIndex;

            let stateClass = 'phase-pending';
            if (isActive) stateClass = 'phase-active';
            else if (status === 'completed' || isPast) stateClass = 'phase-completed';
            if (status === 'attention') stateClass += ' phase-attention';
            if (status === 'failed') stateClass += ' phase-failed';
            if (status === 'unavailable' && trace.length === 0) stateClass += ' phase-unavailable';

            return (
              <button
                key={phase.id}
                type="button"
                className={`phase-step ${stateClass}`}
                onClick={() => {
                  setIsWalkthroughPlaying(false);
                  setActivePhaseIndex(idx);
                }}
                aria-current={isActive ? 'step' : undefined}
                title={phase.label}
              >
                <span className="phase-step-icon">
                  {status === 'completed' || (isPast && status !== 'failed') ? (
                    <Icon name="checkCircle" size={14} />
                  ) : status === 'attention' ? (
                    <Icon name="alertCircle" size={14} />
                  ) : status === 'failed' ? (
                    <Icon name="xCircle" size={14} />
                  ) : (
                    <Icon name={phase.icon} size={14} />
                  )}
                </span>
                <span className="phase-step-label">{phase.label}</span>
              </button>
            );
          })}
        </div>
      </nav>

      {/* Phase Content Canvas */}
      <div className="phase-canvas" key={activePhase.id}>
        {renderPhaseContent()}
      </div>

      {/* Walkthrough Controls */}
      <div className="walkthrough-controls" role="toolbar" aria-label="Walkthrough Controls">
        <div className="walkthrough-left">
          <span className="walkthrough-position">
            Step <strong>{activePhaseIndex + 1}</strong> of {PHASES.length}
            <span className="walkthrough-separator">·</span>
            <strong>{activePhase.label}</strong>
          </span>
        </div>

        <div className="walkthrough-buttons">
          <button type="button" className="wt-btn" onClick={handleReplay} title="Replay from start (R)" aria-label="Replay">
            <Icon name="rotate" size={13} />
          </button>
          <button type="button" className="wt-btn" onClick={handlePrev} disabled={activePhaseIndex === 0} title="Previous (←)" aria-label="Previous step">
            <Icon name="chevronLeft" size={13} />
          </button>
          <button
            type="button"
            className={`wt-btn wt-btn-primary ${isWalkthroughPlaying ? 'is-playing' : ''}`}
            onClick={isWalkthroughPlaying ? handlePause : handleResume}
            title={isWalkthroughPlaying ? 'Pause (Space)' : 'Play walkthrough (Space)'}
            aria-label={isWalkthroughPlaying ? 'Pause walkthrough' : 'Start walkthrough'}
          >
            <Icon name={isWalkthroughPlaying ? 'pause' : 'play'} size={14} />
            <span>{isWalkthroughPlaying ? 'Pause' : 'Play'}</span>
          </button>
          <button type="button" className="wt-btn" onClick={handleNext} disabled={activePhaseIndex >= PHASES.length - 1} title="Next (→)" aria-label="Next step">
            <Icon name="chevronRight" size={13} />
          </button>
          <button type="button" className="wt-btn" onClick={handleSkip} disabled={activePhaseIndex >= PHASES.length - 1} title="Skip to verdict (S)" aria-label="Skip to verdict">
            <Icon name="fastForward" size={13} />
            <span className="wt-btn-label-optional">Skip</span>
          </button>
        </div>

        <div className="walkthrough-right">
          {/* Speed selector */}
          <div className="wt-speed" title="Walkthrough speed">
            {[1, 1.5, 2].map((s) => (
              <button
                key={s}
                type="button"
                className={`wt-speed-btn ${walkthroughSpeed === s ? 'active' : ''}`}
                onClick={() => setWalkthroughSpeed(s)}
                aria-label={`Speed ${s}x`}
              >
                {s}×
              </button>
            ))}
          </div>

          {/* Technical details toggle */}
          <button
            type="button"
            className={`wt-btn wt-inspect-btn ${showTechnicalDetails ? 'active' : ''}`}
            onClick={() => setShowTechnicalDetails(true)}
            title="View full technical processing trace"
            aria-label="View technical details"
          >
            <Icon name="terminal" size={13} />
            <span>Technical Details</span>
          </button>
        </div>
      </div>

      {/* Technical Details Modal — Full 11-Stage Trace */}
      <StageDetailsModal
        isOpen={showTechnicalDetails}
        onClose={() => setShowTechnicalDetails(false)}
        activeStageData={activeTraceStages[0] || null}
        allStages={trace}
      />
    </div>
  );
}

export default NLPVisualization;
