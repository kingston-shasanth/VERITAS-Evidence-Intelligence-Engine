import React, { useState, useEffect } from 'react';
import { Icon } from './Icons.jsx';

/**
 * Word-by-Word Spelling Transformation Animation Component
 * Visualizes individual lexical transformations, character-level offsets, candidate rankings,
 * and ambiguity warnings based strictly on real backend telemetry.
 */
export function WordCorrectionAnimation({
  originalClaim = '',
  normalizedClaim = '',
  corrections = [],
  warnings = [],
  normalizationStatus = 'accepted',
  isPlaying = false,
  playbackSpeed = 1,
}) {
  const [selectedWordIndex, setSelectedWordIndex] = useState(0);
  const [activeStep, setActiveStep] = useState(0);

  // Auto-advance through corrections during playback
  useEffect(() => {
    if (!isPlaying || corrections.length <= 1) return;

    const intervalTime = Math.max(800, 2000 / playbackSpeed);
    const interval = setInterval(() => {
      setSelectedWordIndex((prev) => (prev + 1) % corrections.length);
    }, intervalTime);

    return () => clearInterval(interval);
  }, [isPlaying, corrections.length, playbackSpeed]);

  const activeCorrection = corrections[selectedWordIndex] || null;

  // Split original text into segments with token highlights
  const renderedTokens = [];
  if (corrections.length === 0) {
    renderedTokens.push({
      text: originalClaim,
      isCorrection: false,
    });
  } else {
    // Sort corrections by start_char
    const sorted = [...corrections].sort((a, b) => (a.start_char || 0) - (b.start_char || 0));
    let lastIndex = 0;

    sorted.forEach((corr, i) => {
      let start = Math.max(0, corr.start_char || 0);
      let end = Math.min(originalClaim.length, corr.end_char || start + (corr.original || '').length);

      // Verify that originalClaim[start:end] actually matches corr.original
      if (originalClaim.slice(start, end) !== corr.original) {
        const found = originalClaim.indexOf(corr.original, lastIndex);
        if (found !== -1) {
          start = found;
          end = found + corr.original.length;
        }
      }

      if (start > lastIndex) {
        renderedTokens.push({
          text: originalClaim.slice(lastIndex, start),
          isCorrection: false,
        });
      }

      renderedTokens.push({
        text: originalClaim.slice(start, end) || corr.original,
        isCorrection: true,
        correctionIndex: i,
        correction: corr,
        isSelected: i === selectedWordIndex,
      });

      lastIndex = Math.max(lastIndex, end);
    });

    if (lastIndex < originalClaim.length) {
      renderedTokens.push({
        text: originalClaim.slice(lastIndex),
        isCorrection: false,
      });
    }
  }

  const isAmbiguous = normalizationStatus === 'uncertain' || corrections.some((c) => c.ambiguous);
  const isReverted = normalizationStatus === 'reverted';

  return (
    <div className="word-correction-workspace" role="region" aria-label="Spelling Normalization Analysis">
      {/* Top Banner: Status & Overview */}
      <div className="correction-header-row">
        <div>
          <span className="cinematic-eyebrow">
            <span className="eyebrow-glow-dot" />
            TOKEN-LEVEL NORMALIZATION ENGINE
          </span>
          <h3 className="cinematic-scene-title">
            Spelling &amp; Noisy-Text <span>Correction.</span>
          </h3>
          <p className="cinematic-scene-desc">
            Evaluating typographical errors using Damerau-Levenshtein distance (d &le; 2) and domain dictionaries.
          </p>
        </div>

        <div className="normalization-status-badge-group">
          <span className={`normalization-pill ${normalizationStatus}`}>
            {normalizationStatus.toUpperCase()}
          </span>
          <span className="corrections-count-pill">
            <strong>{corrections.length}</strong> Proposed Change{corrections.length === 1 ? '' : 's'}
          </span>
        </div>
      </div>

      {/* Main Claim Token Transformation Canvas */}
      <div className="token-canvas-panel">
        <div className="canvas-header-strip">
          <span className="canvas-tag">ORIGINAL SUBMITTED TEXT</span>
          <span className="canvas-hint">Click any highlighted token to inspect candidate rankings</span>
        </div>

        <div className="interactive-claim-stream">
          {renderedTokens.map((tok, idx) => {
            if (!tok.isCorrection) {
              return <span key={`plain-${idx}`} className="stream-plain-text">{tok.text}</span>;
            }

            const c = tok.correction;
            const statusClass = c.ambiguous ? 'tok-uncertain' : (c.applied ? 'tok-accepted' : 'tok-rejected');

            return (
              <button
                key={`tok-${idx}`}
                type="button"
                className={`stream-token-chip ${statusClass} ${tok.isSelected ? 'is-focused-token' : ''}`}
                onClick={() => setSelectedWordIndex(tok.correctionIndex)}
                title={`Click to inspect: "${c.original}" -> "${c.replacement}"`}
              >
                <span className="token-original-str">{c.original}</span>
                <span className="token-morph-arrow">&rarr;</span>
                <span className="token-replacement-str">{c.replacement}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Ambiguity Warning Callout if Present */}
      {isAmbiguous && (
        <div className="cinematic-alert-callout alert-amber" role="alert">
          <Icon name="alert" size={16} />
          <div>
            <strong>Ambiguous Lexical Interpretation Detected:</strong>
            <p>
              Multiple plausible candidate corrections were identified with close heuristic scores.
              To guarantee truthful verification, VERITAS-X marks this input as <strong>uncertain</strong> and falls back conservatively to user-submitted wording.
            </p>
            {warnings.length > 0 && (
              <ul className="alert-details-list">
                {warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}

      {/* Reversion Notice if Guardrails rejected */}
      {isReverted && (
        <div className="cinematic-alert-callout alert-violet" role="alert">
          <Icon name="shield" size={16} />
          <div>
            <strong>Normalization Reverted to Original Claim:</strong>
            <p>
              Meaning-preservation guardrails determined that candidate replacements could alter the proposition's truth conditions. Downstream retrieval and NLI verification were safely routed to the original text.
            </p>
          </div>
        </div>
      )}

      {/* Active Correction Detail Inspector */}
      {activeCorrection && (
        <div className="active-correction-card">
          <div className="correction-card-header">
            <div className="correction-title-meta">
              <span className="correction-index-tag">CHANGE {selectedWordIndex + 1} OF {corrections.length}</span>
              <strong className="correction-pair-title">
                "{activeCorrection.original}" <span className="title-arrow">&rarr;</span> "{activeCorrection.replacement}"
              </strong>
            </div>

            <div className="correction-meta-chips">
              <span className="meta-chip">
                Category: <strong>{activeCorrection.category || 'spelling'}</strong>
              </span>
              <span className="meta-chip">
                Edit Distance: <strong>{activeCorrection.edit_distance ?? 1}</strong>
              </span>
              {activeCorrection.confidence !== undefined && (
                <span className="meta-chip">
                  Confidence: <strong>{Math.round(activeCorrection.confidence * 100)}%</strong>
                </span>
              )}
              <span className={`status-tag ${activeCorrection.ambiguous ? 'tag-amber' : (activeCorrection.applied ? 'tag-green' : 'tag-red')}`}>
                {activeCorrection.ambiguous ? 'Uncertain Ranking' : (activeCorrection.applied ? 'Accepted' : 'Rejected')}
              </span>
            </div>
          </div>

          {/* Candidate Rankings List if available */}
          {activeCorrection.candidates && activeCorrection.candidates.length > 0 && (
            <div className="candidates-list-section">
              <span className="candidates-headline">Evaluated Dictionary Candidates:</span>
              <div className="candidates-grid">
                {activeCorrection.candidates.map((cand, ci) => (
                  <div key={ci} className={`candidate-pill ${cand.term === activeCorrection.replacement ? 'candidate-selected' : ''}`}>
                    <strong className="cand-term">{cand.term}</strong>
                    <span className="cand-meta">
                      d={cand.distance} &bull; freq={cand.count ? (cand.count > 1e6 ? `${(cand.count / 1e6).toFixed(1)}M` : cand.count) : 'domain'}
                      {cand.ranking_score ? ` • score ${cand.ranking_score.toFixed(1)}` : ''}
                    </span>
                    {cand.term === activeCorrection.replacement && (
                      <span className="selected-badge">Selected</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeCorrection.reason && (
            <div className="correction-reason-note">
              <Icon name="info" size={13} />
              <span>{activeCorrection.reason}</span>
            </div>
          )}
        </div>
      )}

      {/* Post-Normalization Output Result */}
      <div className="normalized-output-preview">
        <div className="output-header">
          <span className="output-tag">NORMALIZED RESULT FOR RETRIEVAL</span>
          <span className="output-badge">
            {normalizationStatus === 'accepted' ? 'Approved for BM25/Dense Search' : 'Conservative Fallback Active'}
          </span>
        </div>
        <div className="normalized-claim-text">
          {normalizationStatus === 'accepted' ? normalizedClaim : originalClaim}
        </div>
      </div>
    </div>
  );
}

export default WordCorrectionAnimation;
