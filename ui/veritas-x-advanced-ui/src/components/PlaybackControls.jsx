import React from 'react';
import { Icon } from './Icons.jsx';

/**
 * Clean, Integrated Playback Controls Toolbar
 * Styled naturally according to the VERITAS-X light design system.
 * Provides accessible controls: Play, Pause, Next Step, Previous Step, Replay, Skip, Speed, and Raw Data toggle.
 */
export function PlaybackControls({
  currentStageIndex,
  currentStage,
  totalStages = 11,
  isPlaying = false,
  playbackSpeed = 1,
  onPlay,
  onPause,
  onNext,
  onPrev,
  onReplay,
  onSkip,
  onChangeSpeed,
  onSpeedChange,
  onToggleDetails,
  onInspectData,
  showDetails = false,
  activeStageName = '',
  stageStatus = 'completed',
  serverLatencyMs,
  backendLatencyMs,
}) {
  const actualStageIndex = currentStageIndex !== undefined ? currentStageIndex : (currentStage !== undefined ? currentStage : 0);
  const actualChangeSpeed = onChangeSpeed || onSpeedChange || (() => {});
  const actualToggleDetails = onToggleDetails || onInspectData || (() => {});
  const actualLatency = serverLatencyMs !== undefined ? serverLatencyMs : backendLatencyMs;

  const isFirst = actualStageIndex === 0;
  const isLast = actualStageIndex >= totalStages - 1;

  const statusColors = {
    completed: 'status-completed',
    running: 'status-running',
    uncertain: 'status-uncertain',
    fallback: 'status-fallback',
    skipped: 'status-skipped',
    failed: 'status-failed',
  };

  const statusClass = statusColors[stageStatus] || 'status-completed';

  return (
    <div className="playback-controls-container" role="toolbar" aria-label="Pipeline Playback Controls">
      <div className="controls-left">
        <div className="stage-counter-badge">
          <span className="stage-step-tag">
            Stage <strong>{actualStageIndex + 1}</strong> of {totalStages}
          </span>
          {activeStageName && (
            <>
              <span className="stage-separator">&bull;</span>
              <strong className="stage-name-label">{activeStageName.replace(/_/g, ' ')}</strong>
            </>
          )}
          <span className={`status-pill ${statusClass}`}>{stageStatus || 'completed'}</span>
        </div>
      </div>

      <div className="controls-button-group">
        {/* Replay */}
        <button
          type="button"
          className="playback-btn"
          onClick={onReplay}
          title="Replay from Beginning (R)"
          aria-label="Replay timeline from beginning"
        >
          <Icon name="rotate" size={14} />
        </button>

        {/* Previous Step */}
        <button
          type="button"
          className="playback-btn"
          onClick={onPrev}
          disabled={isFirst}
          title="Previous Step (Left Arrow)"
          aria-label="Go to previous processing stage"
        >
          <Icon name="chevronLeft" size={14} />
        </button>

        {/* Play / Pause Toggle */}
        <button
          type="button"
          className={`playback-btn playback-btn-primary ${isPlaying ? 'is-playing' : ''}`}
          onClick={isPlaying ? onPause : onPlay}
          title={isPlaying ? 'Pause Animation (Space)' : 'Play Animation (Space)'}
          aria-label={isPlaying ? 'Pause animation playback' : 'Start animation playback'}
        >
          <Icon name={isPlaying ? 'pause' : 'play'} size={15} />
          <span>{isPlaying ? 'Pause' : 'Play'}</span>
        </button>

        {/* Next Step */}
        <button
          type="button"
          className="playback-btn"
          onClick={onNext}
          disabled={isLast}
          title="Next Step (Right Arrow)"
          aria-label="Go to next processing stage"
        >
          <Icon name="chevronRight" size={14} />
        </button>

        {/* Skip to Final Result */}
        <button
          type="button"
          className="playback-btn skip-btn"
          onClick={onSkip}
          disabled={isLast}
          title="Skip Animation to Final Verdict (S)"
          aria-label="Skip animation to final verdict"
        >
          <Icon name="fastForward" size={14} />
          <span className="skip-label">Skip</span>
        </button>
      </div>

      <div className="controls-meta-group">
        {/* Speed Selector */}
        <div className="speed-selector-wrap" title="Animation playback pacing">
          <span className="speed-label">Speed:</span>
          {[1, 1.5, 2].map((s) => (
            <button
              key={s}
              type="button"
              className={`speed-btn ${playbackSpeed === s ? 'active' : ''}`}
              onClick={() => actualChangeSpeed(s)}
              aria-label={`Set playback speed to ${s}x`}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* Latency Telemetry Chip */}
        {actualLatency !== null && actualLatency !== undefined && (
          <div className="latency-metric-chip" title="Actual backend execution time">
            <Icon name="zap" size={12} />
            <span>Backend: <strong>{Number(actualLatency).toFixed(1)}ms</strong></span>
          </div>
        )}

        {/* Raw Telemetry Inspector Toggle */}
        <button
          type="button"
          className={`inspect-data-btn ${showDetails ? 'active' : ''}`}
          onClick={actualToggleDetails}
          title="Inspect raw backend event telemetry"
          aria-label="Inspect raw event JSON"
        >
          <Icon name="terminal" size={13} />
          <span>Inspect Data</span>
        </button>
      </div>
    </div>
  );
}

export default PlaybackControls;
