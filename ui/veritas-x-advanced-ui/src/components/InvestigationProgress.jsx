import React, { useState, useEffect } from 'react';
import { Icon } from './Icons';

/**
 * Clean, readable progress state during claim verification.
 * Follows the 5 natural verification stages in plain English without ML jargon:
 * 1. Preparing the claim
 * 2. Finding relevant evidence
 * 3. Comparing evidence with the claim
 * 4. Evaluating agreement and disagreement
 * 5. Preparing the result
 */
export function InvestigationProgress({ active = true, mode = 'live', evidenceMode = 'local', elapsedSeconds = 0 }) {
  // Determine active stage based on elapsed time for honest, smooth progress
  const currentStageIndex = elapsedSeconds < 0.5 ? 0 : elapsedSeconds < 1.4 ? 1 : elapsedSeconds < 2.5 ? 2 : elapsedSeconds < 3.8 ? 3 : 4;

  const stages = [
    {
      num: '1',
      title: 'Preparing the claim',
      desc: 'Checking statement format and normalizing text',
    },
    {
      num: '2',
      title: 'Finding relevant evidence',
      desc: evidenceMode === 'web'
        ? 'Searching online sources for matching documentation'
        : evidenceMode === 'hybrid'
        ? 'Searching local collection and online sources'
        : 'Searching local knowledge collection offline',
    },
    {
      num: '3',
      title: 'Comparing evidence with the claim',
      desc: 'Evaluating whether each passage supports, contradicts, or provides context',
    },
    {
      num: '4',
      title: 'Evaluating agreement and disagreement',
      desc: 'Checking for consensus, conflicting reports, or insufficient information',
    },
    {
      num: '5',
      title: 'Preparing the result',
      desc: 'Compiling the final verdict, confidence, and source citations',
    },
  ];

  return (
    <div className="investigation-progress-container" role="status" aria-live="polite">
      <div className="progress-header-row">
        <div className="progress-header-left">
          <div className="progress-spinner-ring" />
          <div>
            <h4 className="progress-title">Verifying Claim</h4>
            <p className="progress-subtitle">
              Source: <strong>{evidenceMode === 'web' ? 'Live Web' : evidenceMode === 'hybrid' ? 'Hybrid (Local + Web)' : 'Local Corpus'}</strong> &bull; Please wait while evidence is retrieved and evaluated
            </p>
          </div>
        </div>

        <div className="progress-timer-chip">
          <Icon name="clock" size={13} />
          <span>Elapsed: <strong>{Number(elapsedSeconds || 0).toFixed(1)}s</strong></span>
        </div>
      </div>

      <div className="progress-bar-track">
        <div
          className="progress-bar-fill"
          style={{ width: `${Math.min(95, Math.max(15, (currentStageIndex + 1) * 20))}%` }}
        />
      </div>

      <div className="progress-stages-grid">
        {stages.map((st, idx) => {
          const isDone = idx < currentStageIndex;
          const isCurrent = idx === currentStageIndex;
          const isPending = idx > currentStageIndex;

          return (
            <div
              key={st.num}
              className={`progress-stage-card ${isCurrent ? 'stage-current' : isDone ? 'stage-done' : 'stage-pending'}`}
            >
              <div className="stage-top-indicator">
                <span className="stage-step-circle">
                  {isDone ? <Icon name="check" size={12} /> : st.num}
                </span>
                <span className="stage-status-label">
                  {isDone ? 'Completed' : isCurrent ? 'In progress...' : 'Queued'}
                </span>
              </div>
              <strong className="stage-title">{st.title}</strong>
              <p className="stage-desc">{st.desc}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default InvestigationProgress;
