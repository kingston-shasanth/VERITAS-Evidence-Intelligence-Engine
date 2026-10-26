import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Icon } from './Icons.jsx';
import { ProcessingTimeline } from './ProcessingTimeline.jsx';
import { PlaybackControls } from './PlaybackControls.jsx';
import { WordCorrectionAnimation } from './WordCorrectionAnimation.jsx';
import { MeaningPreservationPanel } from './MeaningPreservationPanel.jsx';
import { TokenInspector } from './TokenInspector.jsx';
import { SyntaxRelationshipView } from './SyntaxRelationshipView.jsx';
import { ClaimStructureView } from './ClaimStructureView.jsx';
import { EvidenceFlowView } from './EvidenceFlowView.jsx';
import { VerificationReveal } from './VerificationReveal.jsx';
import { StageDetailsModal } from './StageDetailsModal.jsx';

/**
 * Top-Level Integrated NLP Understanding Visualization Component
 * Calm, clean, light-theme interface following the signature VERITAS-X design system.
 * Orchestrates sequential playback across the 11 pipeline stages.
 */
export function NLPVisualization({
  result = {},
  onClose,
}) {
  const trace = Array.isArray(result.processing_trace) ? result.processing_trace : [];

  // Determine user reduced-motion preference
  const prefersReducedMotion = typeof window !== 'undefined'
    && window.matchMedia
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Playback state
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(!prefersReducedMotion && trace.length > 0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [isInspectorOpen, setIsInspectorOpen] = useState(false);

  const timerRef = useRef(null);

  // Active stage object
  const activeStage = trace[currentStageIndex] || trace[0] || null;
  const activeStageName = activeStage ? activeStage.stage : 'input_received';

  // Base stage duration in ms for auto-playback (1.8s per stage at 1x)
  const baseStageDurationMs = 1800;

  // Playback advancement
  useEffect(() => {
    if (!isPlaying || trace.length <= 1) {
      if (timerRef.current) clearTimeout(timerRef.current);
      return;
    }

    if (currentStageIndex >= trace.length - 1) {
      setIsPlaying(false);
      return;
    }

    const interval = baseStageDurationMs / playbackSpeed;
    timerRef.current = setTimeout(() => {
      setCurrentStageIndex((prev) => {
        if (prev < trace.length - 1) {
          return prev + 1;
        } else {
          setIsPlaying(false);
          return prev;
        }
      });
    }, interval);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isPlaying, currentStageIndex, trace.length, playbackSpeed]);

  // Handlers for controls
  const handlePlay = useCallback(() => {
    if (currentStageIndex >= trace.length - 1) {
      setCurrentStageIndex(0);
    }
    setIsPlaying(true);
  }, [currentStageIndex, trace.length]);

  const handlePause = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const handleNext = useCallback(() => {
    setIsPlaying(false);
    if (currentStageIndex < trace.length - 1) {
      setCurrentStageIndex((prev) => prev + 1);
    }
  }, [currentStageIndex, trace.length]);

  const handlePrev = useCallback(() => {
    setIsPlaying(false);
    if (currentStageIndex > 0) {
      setCurrentStageIndex((prev) => prev - 1);
    }
  }, [currentStageIndex]);

  const handleReplay = useCallback(() => {
    setCurrentStageIndex(0);
    setIsPlaying(!prefersReducedMotion);
  }, [prefersReducedMotion]);

  const handleSkip = useCallback(() => {
    setIsPlaying(false);
    setCurrentStageIndex(trace.length - 1);
  }, [trace.length]);

  const handleSelectStage = useCallback((index) => {
    setIsPlaying(false);
    setCurrentStageIndex(index);
  }, []);

  // Keyboard navigation shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't trigger if typing in an input
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;

      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        setIsPlaying((p) => !p);
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
        if (isInspectorOpen) {
          setIsInspectorOpen(false);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleNext, handlePrev, handleReplay, handleSkip, isInspectorOpen]);

  // Extract core properties from result
  const rawOriginalClaim = result.original_claim || result.claim || '';
  const sanitizedClaim = result.sanitized_claim || result.claim || '';
  const normalizedClaim = result.normalized_claim || sanitizedClaim;
  const normalizationStatus = result.normalization_status || 'accepted';
  const normCorrections = Array.isArray(result.normalization_corrections) ? result.normalization_corrections : [];
  const normWarnings = Array.isArray(result.normalization_warnings) ? result.normalization_warnings : [];
  const structuredClaim = result.structured_claim || {};
  const retrievalQueryUsed = result.retrieval_query_used || sanitizedClaim;
  const nliHypothesisUsed = result.nli_hypothesis_used || sanitizedClaim;

  // Active stage details
  const activeDetails = activeStage ? (activeStage.details || {}) : {};

  // Status badge calculation
  let statusBadgeText = 'Completed';
  let statusBadgeKey = 'completed';
  if (isPlaying) {
    statusBadgeText = 'Processing';
    statusBadgeKey = 'running';
  } else if (normalizationStatus === 'uncertain') {
    statusBadgeText = 'Needs review';
    statusBadgeKey = 'uncertain';
  } else if (normalizationStatus === 'reverted') {
    statusBadgeText = 'Reverted';
    statusBadgeKey = 'reverted';
  }

  // Render the active scene canvas
  const renderStageScene = () => {
    if (!activeStage) {
      return (
        <div className="empty-trace-callout">
          <Icon name="alertCircle" size={20} />
          <div>
            <strong>Processing Trace Unavailable</strong>
            <p>The verification response did not include a structured pipeline processing trace.</p>
          </div>
        </div>
      );
    }

    switch (activeStageName) {
      case 'input_received':
        return (
          <div className="scene-input-received">
            <div className="stage-scene-header">
              <span className="stage-step-kicker">Stage 1 &bull; Input</span>
              <h4 className="stage-main-title cinematic-scene-title">
                Raw Input Ingestion
              </h4>
              <p className="stage-desc cinematic-scene-desc">
                The exact, unvarnished user string is preserved immutably with zero silent mutations.
              </p>
            </div>

            <div className="clean-content-box">
              <div className="box-meta-strip">
                <span className="box-kicker">Submitted String</span>
                <div className="intake-metadata-strip">
                  <span>Length: <strong>{activeDetails.char_length || rawOriginalClaim.length} chars</strong></span>
                  <span>Words: <strong>{activeDetails.raw_word_count || rawOriginalClaim.split(/\s+/).filter(Boolean).length}</strong></span>
                  <span>Encoding: <strong>UTF-8 Canonical</strong></span>
                </div>
              </div>
              <blockquote className="raw-intake-text">"{rawOriginalClaim}"</blockquote>
            </div>
          </div>
        );

      case 'sanitization':
        return (
          <div className="scene-sanitization">
            <div className="stage-scene-header">
              <span className="stage-step-kicker">Stage 2 &bull; Sanitization</span>
              <h4 className="stage-main-title cinematic-scene-title">
                Unicode &amp; Whitespace Sanitization
              </h4>
              <p className="stage-desc cinematic-scene-desc">
                Removal of invisible control characters, zero-width spaces, and duplicate whitespace while preserving intentional punctuation and casing.
              </p>
            </div>

            <div className="clean-content-box">
              <div className="san-card-top">
                <span className="box-kicker">Sanitized Canonical Claim</span>
                <div className="san-intent-pills">
                  {activeDetails.is_question && <span className="intent-pill pill-question">Question Invariant</span>}
                  {activeDetails.has_negation && <span className="intent-pill pill-negation">Negation Operator</span>}
                </div>
              </div>
              <blockquote className="sanitized-text">"{sanitizedClaim}"</blockquote>
            </div>

            {activeDetails.transformations && activeDetails.transformations.length > 0 && (
              <div className="transformations-list">
                <span className="trans-list-title">Applied Rules ({activeDetails.transformations.length}):</span>
                <div className="trans-chips-wrap">
                  {activeDetails.transformations.map((t, idx) => (
                    <span key={idx} className="trans-chip">
                      <Icon name="check" size={12} />
                      {t.name || t.description || 'Sanitization rule'}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 'spelling_normalization':
        return (
          <WordCorrectionAnimation
            originalClaim={rawOriginalClaim || sanitizedClaim}
            normalizedClaim={normalizedClaim}
            corrections={activeDetails.corrections || normCorrections}
            wordEvents={activeDetails.word_correction_events || []}
            warnings={activeDetails.warnings || normWarnings}
            hasAmbiguousCorrections={activeDetails.has_ambiguous_corrections}
            normalizationStatus={normalizationStatus}
          />
        );

      case 'meaning_preservation':
        return (
          <MeaningPreservationPanel
            originalClaim={sanitizedClaim}
            normalizedClaim={normalizedClaim}
            guardrailDetails={activeDetails}
            normalizationStatus={normalizationStatus}
          />
        );

      case 'tokenization':
        return (
          <TokenInspector
            tokens={activeDetails.tokens || []}
            tokenCount={activeDetails.token_count || (activeDetails.tokens || []).length}
            sanitizedClaim={sanitizedClaim}
          />
        );

      case 'linguistic_analysis':
        return (
          <SyntaxRelationshipView
            linguisticDetails={activeDetails}
            sanitizedClaim={sanitizedClaim}
          />
        );

      case 'claim_extraction':
        return (
          <ClaimStructureView
            structuredClaim={structuredClaim}
            claimExtractionDetails={activeDetails}
            sanitizedClaim={sanitizedClaim}
          />
        );

      case 'retrieval':
      case 'evidence_processing':
        return (
          <EvidenceFlowView
            retrievalDetails={activeDetails}
            evidence={result.evidence || []}
            retrievalStatus={result.retrieval_status || {}}
          />
        );

      case 'nli_verification':
      case 'final_result':
        return (
          <VerificationReveal
            nliDetails={activeStageName === 'nli_verification' ? activeDetails : (trace.find(s => s.stage === 'nli_verification')?.details || {})}
            resultDetails={result}
            canonicalReason={result.reason || ''}
            retrievalQueryUsed={retrievalQueryUsed}
          />
        );

      default:
        return (
          <div className="scene-generic">
            <h4>Stage: {activeStageName}</h4>
            <p>{activeStage.summary}</p>
          </div>
        );
    }
  };

  return (
    <div className="nlp-visualization-container" role="region" aria-label="NLP Understanding Pipeline">
      {/* Compact Header: Natural, Integrated Design */}
      <div className="nlp-header-section">
        <div className="nlp-header-top">
          <div className="nlp-header-titles">
            <h3 className="nlp-panel-title">NLP Processing</h3>
            <p className="nlp-panel-subtitle">See how VERITAS-X interprets your claim.</p>
          </div>

          <div className="nlp-header-actions">
            <span className={`nlp-status-badge status-${statusBadgeKey} strip-status-pill`}>
              {statusBadgeText}
            </span>
            {onClose && (
              <button
                type="button"
                className="nlp-close-btn workspace-close-btn"
                onClick={onClose}
                aria-label="Close NLP Visualizer"
              >
                <Icon name="x" size={15} />
              </button>
            )}
          </div>
        </div>

        {/* Submitted Claim Container */}
        <div className="nlp-claim-container workspace-claim-strip">
          <span className="nlp-claim-label strip-title">Submitted Claim:</span>
          <span className="strip-claim-text nlp-claim-text">"{rawOriginalClaim}"</span>
          {rawOriginalClaim !== normalizedClaim && normalizationStatus === 'accepted' && (
            <span className="nlp-claim-normalized-preview strip-normalized-preview">
              &bull; Normalized for retrieval: <strong>"{normalizedClaim}"</strong>
            </span>
          )}
        </div>
      </div>

      {/* Sequential Processing Timeline (11 Stages) */}
      <ProcessingTimeline
        stages={trace}
        currentStageIndex={currentStageIndex}
        onSelectStage={handleSelectStage}
      />

      {/* Central Active Stage Canvas */}
      <div className="nlp-scene-canvas" key={activeStageName}>
        {renderStageScene()}
      </div>

      {/* Sequential Playback Controls */}
      <PlaybackControls
        isPlaying={isPlaying}
        onPlay={handlePlay}
        onPause={handlePause}
        onNext={handleNext}
        onPrev={handlePrev}
        onReplay={handleReplay}
        onSkip={handleSkip}
        currentStageIndex={currentStageIndex}
        totalStages={trace.length || 11}
        playbackSpeed={playbackSpeed}
        onChangeSpeed={setPlaybackSpeed}
        serverLatencyMs={result.latency_ms || (activeStage ? activeStage.duration_ms : 0)}
        onToggleDetails={() => setIsInspectorOpen(true)}
        showDetails={isInspectorOpen}
        activeStageName={activeStage ? activeStage.stage : 'input_received'}
        stageStatus={activeStage ? activeStage.status : 'completed'}
      />

      {/* Telemetry Inspector Modal */}
      <StageDetailsModal
        isOpen={isInspectorOpen}
        onClose={() => setIsInspectorOpen(false)}
        activeStageData={activeStage}
        allStages={trace}
      />
    </div>
  );
}

export default NLPVisualization;
