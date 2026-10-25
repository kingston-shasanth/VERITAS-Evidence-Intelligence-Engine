/**
 * VERITAS-X Forensic Voice Briefing & Caption Utility
 * Constructs authentic, plain-language forensic narration briefings
 * derived strictly from real model outputs and evidence citations.
 * Never invents facts or states unreturned verdicts.
 */

/**
 * Builds the complete plain-text narration script.
 */
export function buildNarrationScript(result) {
  if (!result) return '';
  const segments = buildNarrationSegments(result);
  return segments.map((s) => s.text).join(' ');
}

/**
 * Breaks the forensic verdict and evidence into typed, sequentially narratable segments.
 * Each segment corresponds to a discrete utterance with live caption metadata.
 */
export function buildNarrationSegments(result) {
  if (!result) return [];
  const claim = (result.claim || 'The submitted proposition').trim();
  const verdict = result.verdict ? result.verdict.replaceAll('_', ' ').toLowerCase() : 'undetermined';
  const confPct = Math.round(Number(result.confidence || 0) * 100);
  const evidenceList = result.evidence || [];
  const isPassageSupporting = (e) => ['support', 'entailment'].includes(String(e.stance).toLowerCase()) && e.is_informative !== false && e.is_claim_aligned !== false;
  const isPassageRefuting = (e) => ['refute', 'contradiction'].includes(String(e.stance).toLowerCase()) && e.is_informative !== false && e.is_claim_aligned !== false;

  const supCount = evidenceList.filter(isPassageSupporting).length;
  const refCount = evidenceList.filter(isPassageRefuting).length;
  const neutCount = evidenceList.length - (supCount + refCount);

  const segments = [
    {
      id: 'seg-claim',
      type: 'claim',
      label: 'Proposition Under Review',
      text: `Forensic briefing for claim: "${claim}".`,
    },
    {
      id: 'seg-verdict',
      type: 'verdict',
      label: result.abstained ? 'Operational Policy' : 'Adjudicated Consensus',
      text: result.abstained
        ? `The operational coverage policy abstained from asserting a definitive verdict due to insufficient evidentiary grounding.`
        : `The adjudicated verdict is ${verdict}, with calibrated confidence of ${confPct} percent.`,
    },
  ];

  if (result.abstained) {
    segments.push({
      id: 'seg-abstained',
      type: 'abstention',
      label: 'Operational Abstention',
      text: 'Notice: The selective coverage gate abstained from asserting a definitive verdict due to insufficient evidentiary grounding. Human adjudication is recommended.',
    });
  }

  if (evidenceList.length > 0) {
    segments.push({
      id: 'seg-evidence-tally',
      type: 'evidence',
      label: 'Corpus Analysis',
      text: `The system examined ${evidenceList.length} evidence passages: ${supCount} supporting, ${refCount} refuting, and ${neutCount} neutral.`,
    });

    if (evidenceList[0]?.text) {
      const topSnippet = evidenceList[0].text.slice(0, 160).trim();
      segments.push({
        id: 'seg-top-citation',
        type: 'citation',
        label: 'Primary Evidence Excerpt',
        text: `Primary evidence cited: "${topSnippet}".`,
      });
    }
  } else {
    segments.push({
      id: 'seg-no-evidence',
      type: 'evidence',
      label: 'Corpus Coverage',
      text: 'No informative evidence passages were returned from the corpus for this proposition.',
    });
  }

  if (result.disparity) {
    segments.push({
      id: 'seg-disparity',
      type: 'disparity',
      label: 'Stance Disparity Warning',
      text: 'Caution: A cross-passage stance disparity was detected between opposing documents.',
    });
  }

  segments.push({
    id: 'seg-disclaimer',
    type: 'disclaimer',
    label: 'Epistemic Limitation',
    text: 'Remember that model confidence reflects evidence alignment, not absolute ground truth. Please examine the source documents directly.',
  });

  return segments;
}

/**
 * Controller class coordinating sequential SpeechSynthesisUtterance playback,
 * live caption synchronization, rate adjustment, and error boundaries.
 */
export class SpeechBriefingController {
  constructor({
    speechSynthesis = typeof window !== 'undefined' ? window.speechSynthesis : null,
    UtteranceClass = typeof window !== 'undefined' ? window.SpeechSynthesisUtterance : null,
    onStateChange = () => {},
    onSegmentChange = () => {},
    onError = () => {},
  } = {}) {
    this.synth = speechSynthesis;
    this.UtteranceClass = UtteranceClass;
    this.onStateChange = onStateChange;
    this.onSegmentChange = onSegmentChange;
    this.onError = onError;

    this.segments = [];
    this.currentIndex = -1;
    this.state = 'idle'; // 'idle' | 'speaking' | 'paused' | 'stopped' | 'completed'
    this.rate = 1.0;
    this.pitch = 1.0;
    this.voice = null;
    this.isCancelled = false;
    this.activeUtterance = null;
  }

  isSupported() {
    return Boolean(this.synth && this.UtteranceClass);
  }

  loadSegments(segments) {
    this.stop();
    this.segments = segments || [];
    this.currentIndex = -1;
    this.setState('idle');
  }

  setState(newState) {
    this.state = newState;
    this.onStateChange(this.state, {
      currentIndex: this.currentIndex,
      currentSegment: this.segments[this.currentIndex] || null,
      totalSegments: this.segments.length,
    });
  }

  setRate(rate) {
    this.rate = Math.max(0.5, Math.min(2.0, Number(rate) || 1.0));
  }

  setPitch(pitch) {
    this.pitch = Math.max(0.5, Math.min(1.5, Number(pitch) || 1.0));
  }

  setVoice(voice) {
    this.voice = voice;
  }

  play() {
    if (!this.isSupported() || this.segments.length === 0) return;

    if (this.state === 'paused') {
      this.resume();
      return;
    }

    this.stop();
    this.isCancelled = false;
    this.currentIndex = 0;
    this.speakCurrentSegment();
  }

  resume() {
    if (!this.isSupported()) return;
    if (this.state === 'paused') {
      try {
        this.synth.resume();
      } catch {
        // Fallback if browser failed resume
      }
      this.setState('speaking');
    }
  }

  pause() {
    if (!this.isSupported() || this.state !== 'speaking') return;
    try {
      this.synth.pause();
    } catch {
      // Fallback
    }
    this.setState('paused');
  }

  stop() {
    this.isCancelled = true;
    if (this.synth) {
      try {
        this.synth.cancel();
      } catch {
        // Safe noop
      }
    }
    this.activeUtterance = null;
    this.currentIndex = -1;
    this.setState('stopped');
  }

  replay() {
    this.play();
  }

  speakCurrentSegment() {
    if (this.isCancelled || this.currentIndex < 0 || this.currentIndex >= this.segments.length) {
      if (!this.isCancelled && this.currentIndex >= this.segments.length) {
        this.setState('completed');
      }
      return;
    }

    const segment = this.segments[this.currentIndex];
    this.onSegmentChange(this.currentIndex, segment);
    this.setState('speaking');

    try {
      const utterance = new this.UtteranceClass(segment.text);
      utterance.rate = this.rate;
      utterance.pitch = this.pitch;
      if (this.voice) utterance.voice = this.voice;

      utterance.onend = () => {
        if (this.isCancelled) return;
        this.currentIndex += 1;
        if (this.currentIndex < this.segments.length) {
          this.speakCurrentSegment();
        } else {
          this.setState('completed');
        }
      };

      utterance.onerror = (event) => {
        if (this.isCancelled) return;
        // If error was triggered by user cancel/interruption, do not emit as fatal
        if (event.error === 'canceled' || event.error === 'interrupted') return;
        this.onError(event.error || 'Speech synthesis error');
        this.setState('stopped');
      };

      this.activeUtterance = utterance;
      this.synth.speak(utterance);
    } catch (err) {
      this.onError(err.message);
      this.setState('stopped');
    }
  }
}

export default {
  buildNarrationScript,
  buildNarrationSegments,
  SpeechBriefingController,
};
