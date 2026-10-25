import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Icon } from './Icons';
import {
  buildNarrationScript,
  buildNarrationSegments,
  SpeechBriefingController,
} from '../lib/narrator.js';

export { buildNarrationScript, buildNarrationSegments };

/**
 * Accessible Voice Briefing & Live Captions Component.
 * Provides synchronized sentence-level captions, transcript expansion,
 * rate and voice selection, and graceful no-audio fallback.
 */
export function VoiceNarrator({ result, active = true }) {
  const [supported, setSupported] = useState(false);
  const [playbackState, setPlaybackState] = useState('idle'); // 'idle' | 'speaking' | 'paused' | 'stopped' | 'completed'
  const [currentSegmentIndex, setCurrentSegmentIndex] = useState(-1);
  const [showFullTranscript, setShowFullTranscript] = useState(false);
  const [speechRate, setSpeechRate] = useState(1.0);
  const [availableVoices, setAvailableVoices] = useState([]);
  const [selectedVoiceURI, setSelectedVoiceURI] = useState('');
  const [copiedTranscript, setCopiedTranscript] = useState(false);

  const controllerRef = useRef(null);

  const segments = useMemo(() => {
    return buildNarrationSegments(result);
  }, [result]);

  const fullScript = useMemo(() => {
    return segments.map((s) => s.text).join(' ');
  }, [segments]);

  // Initialize Speech Controller and discover voices
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const synth = window.speechSynthesis;
    const Utterance = window.SpeechSynthesisUtterance;

    if (!synth || !Utterance) {
      setSupported(false);
      return;
    }

    setSupported(true);

    const controller = new SpeechBriefingController({
      speechSynthesis: synth,
      UtteranceClass: Utterance,
      onStateChange: (state, meta) => {
        setPlaybackState(state);
        if (state === 'idle' || state === 'stopped') {
          setCurrentSegmentIndex(-1);
        }
      },
      onSegmentChange: (index) => {
        setCurrentSegmentIndex(index);
      },
      onError: (err) => {
        setPlaybackState('stopped');
      },
    });

    controllerRef.current = controller;

    // Load available browser voices
    const updateVoices = () => {
      try {
        const voices = synth.getVoices() || [];
        // Filter for English or language matching document
        const enVoices = voices.filter((v) => v.lang.startsWith('en') || v.lang.startsWith('en-'));
        const displayVoices = enVoices.length > 0 ? enVoices : voices.slice(0, 8);
        setAvailableVoices(displayVoices);

        // Pick high-quality default if available
        if (!selectedVoiceURI && displayVoices.length > 0) {
          const natural = displayVoices.find((v) =>
            v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('David') || v.name.includes('Samantha')
          );
          const chosen = natural || displayVoices[0];
          setSelectedVoiceURI(chosen.voiceURI);
          controller.setVoice(chosen);
        }
      } catch {
        // Safe noop
      }
    };

    updateVoices();
    if (typeof synth.onvoiceschanged !== 'undefined') {
      synth.onvoiceschanged = updateVoices;
    }

    return () => {
      controller.stop();
      if (typeof synth.onvoiceschanged !== 'undefined') {
        synth.onvoiceschanged = null;
      }
    };
  }, []);

  // Sync segments whenever result changes
  useEffect(() => {
    if (controllerRef.current) {
      controllerRef.current.loadSegments(segments);
      setPlaybackState('idle');
      setCurrentSegmentIndex(-1);
    }
  }, [segments]);

  // Handle voice selection change
  const handleVoiceChange = (e) => {
    const uri = e.target.value;
    setSelectedVoiceURI(uri);
    const chosen = availableVoices.find((v) => v.voiceURI === uri);
    if (controllerRef.current && chosen) {
      controllerRef.current.setVoice(chosen);
    }
  };

  // Handle speech rate change
  const handleRateChange = (newRate) => {
    const rateVal = Number(newRate);
    setSpeechRate(rateVal);
    if (controllerRef.current) {
      controllerRef.current.setRate(rateVal);
    }
  };

  const handlePlay = () => {
    if (!controllerRef.current) return;
    controllerRef.current.play();
  };

  const handlePause = () => {
    if (!controllerRef.current) return;
    controllerRef.current.pause();
  };

  const handleResume = () => {
    if (!controllerRef.current) return;
    controllerRef.current.resume();
  };

  const handleStop = () => {
    if (!controllerRef.current) return;
    controllerRef.current.stop();
  };

  const handleReplay = () => {
    if (!controllerRef.current) return;
    controllerRef.current.replay();
  };

  const handleJumpToSegment = (index) => {
    if (!controllerRef.current) return;
    controllerRef.current.stop();
    controllerRef.current.currentIndex = index;
    controllerRef.current.isCancelled = false;
    controllerRef.current.speakCurrentSegment();
  };

  const copyTranscriptText = () => {
    navigator.clipboard.writeText(fullScript);
    setCopiedTranscript(true);
    setTimeout(() => setCopiedTranscript(false), 2000);
  };

  if (!result || !active) return null;

  const currentSegment = segments[currentSegmentIndex] || segments[0];
  const isPlaying = playbackState === 'speaking';
  const isPaused = playbackState === 'paused';
  const isCompleted = playbackState === 'completed';

  const progressPct = segments.length > 0 && currentSegmentIndex >= 0
    ? Math.round(((currentSegmentIndex + 1) / segments.length) * 100)
    : 0;

  return (
    <div
      className={`voice-narrator-root ${isPlaying ? 'is-narrating' : ''}`}
      role="region"
      aria-label="Forensic Voice Briefing and Captions"
    >
      {/* Header Bar */}
      <div className="narrator-header-bar">
        <div className="narrator-title-group">
          <div className="audio-anim-icon" aria-hidden="true">
            <span className={`bar-wave ${isPlaying ? 'wave-active' : ''}`} style={{ height: '14px' }} />
            <span className={`bar-wave ${isPlaying ? 'wave-active' : ''}`} style={{ height: '20px' }} />
            <span className={`bar-wave ${isPlaying ? 'wave-active' : ''}`} style={{ height: '11px' }} />
            <span className={`bar-wave ${isPlaying ? 'wave-active' : ''}`} style={{ height: '17px' }} />
          </div>
          <div>
            <h4 className="narrator-main-title">Forensic Audio Briefing &amp; Live Captions</h4>
            <span className="narrator-subtitle">
              {supported
                ? isPlaying
                  ? `Speaking: Utterance ${currentSegmentIndex + 1} of ${segments.length}`
                  : isPaused
                  ? 'Narration paused'
                  : isCompleted
                  ? 'Briefing concluded'
                  : 'Web Speech API native synthesis • 100% client-side'
                : 'Browser Speech API unavailable • Transcript mode active'}
            </span>
          </div>
        </div>

        {/* Playback Status Pill */}
        <div className="narrator-status-pill">
          {isPlaying && <span className="status-dot dot-live" />}
          {isPaused && <span className="status-dot dot-paused" />}
          <span className="status-pill-text">
            {isPlaying ? 'SPEAKING' : isPaused ? 'PAUSED' : isCompleted ? 'CONCLUDED' : 'STANDBY'}
          </span>
        </div>
      </div>

      {/* Main Playback & Configuration Controls */}
      <div className="narrator-controls-toolbar">
        {/* Play / Pause / Resume / Stop / Replay */}
        <div className="controls-button-cluster">
          {supported ? (
            <>
              {!isPlaying && !isPaused ? (
                <button
                  type="button"
                  className="narrator-action-btn btn-play"
                  onClick={handlePlay}
                  id="voice-read-verdict-btn"
                >
                  <Icon name="zap" size={14} />
                  <span>{isCompleted ? 'Replay Briefing' : 'Read Verdict'}</span>
                </button>
              ) : isPaused ? (
                <button
                  type="button"
                  className="narrator-action-btn btn-resume"
                  onClick={handleResume}
                >
                  <Icon name="arrow" size={14} />
                  <span>Resume</span>
                </button>
              ) : (
                <button
                  type="button"
                  className="narrator-action-btn btn-pause"
                  onClick={handlePause}
                >
                  <span className="pause-icon" />
                  <span>Pause</span>
                </button>
              )}

              {(isPlaying || isPaused) && (
                <button
                  type="button"
                  className="narrator-action-btn btn-stop"
                  onClick={handleStop}
                >
                  <span className="stop-icon" />
                  <span>Stop</span>
                </button>
              )}

              {isCompleted && (
                <button
                  type="button"
                  className="narrator-action-btn btn-replay"
                  onClick={handleReplay}
                >
                  <Icon name="refresh" size={14} />
                  <span>Replay</span>
                </button>
              )}
            </>
          ) : (
            <span className="no-audio-chip">Audio Not Available in Browser</span>
          )}
        </div>

        {/* Rate and Voice Adjusters */}
        {supported && (
          <div className="controls-settings-cluster">
            {/* Speed Rate Control */}
            <div className="setting-item-group">
              <span className="setting-label">SPEED:</span>
              <div className="rate-chips-group">
                {[0.75, 1.0, 1.25, 1.5].map((r) => (
                  <button
                    key={r}
                    type="button"
                    className={`rate-btn ${speechRate === r ? 'rate-btn-active' : ''}`}
                    onClick={() => handleRateChange(r)}
                  >
                    {r}×
                  </button>
                ))}
              </div>
            </div>

            {/* Voice Dropdown */}
            {availableVoices.length > 0 && (
              <div className="setting-item-group voice-select-wrap">
                <span className="setting-label">VOICE:</span>
                <select
                  className="voice-dropdown"
                  value={selectedVoiceURI}
                  onChange={handleVoiceChange}
                  aria-label="Select Narration Voice"
                >
                  {availableVoices.map((v) => (
                    <option key={v.voiceURI} value={v.voiceURI}>
                      {v.name} ({v.lang})
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        )}

        {/* Transcript Toggle & Copy Actions */}
        <div className="controls-actions-cluster">
          <button
            type="button"
            className="transcript-toggle-btn"
            onClick={() => setShowFullTranscript(!showFullTranscript)}
            aria-expanded={showFullTranscript}
          >
            <Icon name="files" size={14} />
            <span>{showFullTranscript ? 'Hide Transcript' : 'Full Transcript'}</span>
            <Icon name={showFullTranscript ? 'chevronUp' : 'chevronDown'} size={12} />
          </button>

          <button
            type="button"
            className="copy-transcript-btn"
            onClick={copyTranscriptText}
            title="Copy plain-text narration script"
          >
            <Icon name={copiedTranscript ? 'check' : 'copy'} size={14} />
            <span>{copiedTranscript ? 'Copied' : 'Copy Script'}</span>
          </button>
        </div>
      </div>

      {/* Synchronized Live Caption Display */}
      <div
        className={`live-caption-monitor ${isPlaying ? 'caption-speaking' : ''}`}
        aria-live="polite"
        role="status"
      >
        <div className="caption-monitor-head">
          <div className="caption-meta-left">
            <span className="caption-live-indicator">LIVE CAPTIONS</span>
            {currentSegment && (
              <span className="caption-segment-type">{currentSegment.label}</span>
            )}
          </div>
          {currentSegmentIndex >= 0 && (
            <span className="caption-counter-badge">
              UTTERANCE {currentSegmentIndex + 1} OF {segments.length} ({progressPct}%)
            </span>
          )}
        </div>

        {/* Highlighted Live Sentence */}
        <div className="caption-text-body">
          <p className="caption-current-sentence">
            {currentSegmentIndex >= 0
              ? currentSegment?.text
              : 'Press "Read Verdict" to begin accessible voice narration. Live synchronized captions will track here sentence-by-sentence.'}
          </p>
        </div>

        {/* Sentence Progress Indicator */}
        {isPlaying && (
          <div className="caption-progress-track">
            <div
              className="caption-progress-fill"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        )}
      </div>

      {/* Expandable Full Transcript Drawer */}
      {showFullTranscript && (
        <div className="transcript-drawer-panel">
          <div className="drawer-header">
            <h5>Complete Forensic Narration Transcript</h5>
            <span className="drawer-tip">Click any sentence to jump narration to that segment:</span>
          </div>

          <div className="transcript-segments-list">
            {segments.map((seg, idx) => {
              const isCurrent = currentSegmentIndex === idx;

              return (
                <div
                  key={seg.id}
                  className={`transcript-segment-row ${isCurrent ? 'segment-active-highlight' : ''}`}
                  onClick={() => handleJumpToSegment(idx)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') handleJumpToSegment(idx);
                  }}
                  title="Click to play from this sentence"
                >
                  <div className="segment-index-tag">
                    {isCurrent ? <span className="active-dot-pulse" /> : <span>0{idx + 1}</span>}
                  </div>
                  <div className="segment-content-col">
                    <span className="segment-tag-label">{seg.label}</span>
                    <p className="segment-text-body">{seg.text}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default VoiceNarrator;
