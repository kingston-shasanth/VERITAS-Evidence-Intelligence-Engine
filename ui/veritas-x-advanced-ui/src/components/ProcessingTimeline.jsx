import React from 'react';
import { Icon } from './Icons.jsx';

/**
 * Standardized 11-stage configuration with clean, intuitive stage names.
 * Follows VERITAS-X design guidelines for simple, human-readable stepper titles.
 */
export const STAGE_CONFIG = [
  { id: 'input_received', num: '1', short: 'Input', label: 'Input Received', desc: 'Raw claim registered & checksummed' },
  { id: 'sanitization', num: '2', short: 'Sanitization', label: 'Text Sanitization', desc: 'Unicode & whitespace normalization' },
  { id: 'spelling_normalization', num: '3', short: 'Spelling', label: 'Spelling Normalization', desc: 'Context-aware typo correction' },
  { id: 'meaning_preservation', num: '4', short: 'Guardrails', label: 'Meaning Guardrails', desc: '8 semantic invariant checks' },
  { id: 'tokenization', num: '5', short: 'Tokenization', label: 'Morphological Tokens', desc: 'Lexical tokens & boundary offsets' },
  { id: 'linguistic_analysis', num: '6', short: 'Syntax', label: 'Linguistic Analysis', desc: 'Sentence type, SVO & clause parsing' },
  { id: 'claim_extraction', num: '7', short: 'Claim Structure', label: 'Claim Extraction', desc: 'Modality, negation scope & entities' },
  { id: 'retrieval', num: '8', short: 'Retrieval', label: 'Evidence Retrieval', desc: 'BM25 & dense corpus lookup' },
  { id: 'evidence_processing', num: '9', short: 'Evidence', label: 'Evidence Processing', desc: 'Stance adjudication & filtering' },
  { id: 'nli_verification', num: '10', short: 'NLI Verification', label: 'NLI Cross-Encoder', desc: 'DeBERTa-v3 hypothesis scoring' },
  { id: 'final_result', num: '11', short: 'Verdict', label: 'Final Result', desc: 'Calibrated consensus & disparity check' },
];

/**
 * Sequential Processing Timeline
 * Clean, modern horizontal stepper with clear completed, current, and pending states.
 * Supports smooth horizontal touch scrolling on mobile without overflow or clipping.
 */
export function ProcessingTimeline({
  stages = [],
  traceEvents = [],
  currentStageIndex = 0,
  currentIndex = 0,
  onSelectStage,
}) {
  const events = Array.isArray(stages) && stages.length > 0 ? stages : (Array.isArray(traceEvents) ? traceEvents : []);
  const activeIdx = currentStageIndex !== undefined ? currentStageIndex : (currentIndex || 0);

  return (
    <nav className="processing-timeline-container" aria-label="Processing Stages Stepper">
      <div className="timeline-nodes-track">
        {STAGE_CONFIG.map((cfg, idx) => {
          const isDone = idx < activeIdx;
          const isCurrent = idx === activeIdx;

          // Match with actual trace event if available
          const traceEvt = events.find((e) => e.stage === cfg.id) || events[idx];
          const status = traceEvt ? traceEvt.status : (isDone ? 'completed' : isCurrent ? 'running' : 'pending');
          const durationMs = traceEvt && traceEvt.duration_ms !== undefined ? traceEvt.duration_ms : null;

          const isUncertain = status === 'uncertain';
          const isFailed = status === 'failed';
          const isFallback = status === 'fallback';

          let stateClass = 'node-pending';
          if (isCurrent) stateClass = 'node-active';
          else if (isDone || status === 'completed') stateClass = 'node-completed';
          if (isUncertain) stateClass += ' node-uncertain';
          if (isFailed) stateClass += ' node-failed';
          if (isFallback) stateClass += ' node-fallback';

          return (
            <button
              key={cfg.id}
              type="button"
              className={`timeline-stage-node ${stateClass}`}
              onClick={() => onSelectStage && onSelectStage(idx)}
              aria-current={isCurrent ? 'step' : undefined}
              title={`${cfg.num}. ${cfg.label} (${status}) ${durationMs !== null ? `— ${durationMs.toFixed(1)}ms` : ''}`}
            >
              <div className="node-top-row">
                <span className="node-num">{cfg.num}</span>
                <span className="node-status-icon">
                  {isFailed ? (
                    <Icon name="xCircle" size={13} />
                  ) : isUncertain ? (
                    <Icon name="alertCircle" size={13} />
                  ) : isDone || status === 'completed' ? (
                    <Icon name="checkCircle" size={13} />
                  ) : isCurrent ? (
                    <span className="node-active-dot" />
                  ) : (
                    <span className="node-pending-dot" />
                  )}
                </span>
              </div>

              <span className="node-label">{cfg.short}</span>

              <span className="node-time">
                {durationMs !== null && durationMs > 0
                  ? (durationMs < 1 ? '<1ms' : `${Number(durationMs).toFixed(1)}ms`)
                  : isDone ? '0.0ms' : 'queued'}
              </span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}

export default ProcessingTimeline;
