import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeResponse } from '../src/lib/api.js';
import {
  buildNarrationScript,
  buildNarrationSegments,
  SpeechBriefingController,
} from '../src/lib/narrator.js';

describe('VERITAS-X UI Response Normalization & Contract Tests', () => {
  test('normalizeResponse correctly handles 4-class probabilities and contract fields', () => {
    const raw = {
      claim_id: 'VX-TEST01',
      claim: 'Test scientific hypothesis',
      verdict: 'VERIFIED_TRUE',
      confidence: 0.8245,
      probabilities: {
        VERIFIED_TRUE: 0.8245,
        VERIFIED_FALSE: 0.0812,
        CONFLICTING_OR_CHERRYPICKED: 0.0411,
        INSUFFICIENT_EVIDENCE: 0.0532,
      },
      has_cross_passage_disparity: false,
      abstained: false,
      reason: 'Local evidence supports the proposition.',
      evidence: [
        {
          id: 'E-01',
          source: 'https://science.org/journal',
          text: 'Confirmed documentary evidence snippet.',
          stance: 'support',
          score: 0.892,
        },
      ],
      device: 'cuda',
    };

    const normalized = normalizeResponse(raw, 'Fallback claim');
    assert.equal(normalized.id, 'VX-TEST01');
    assert.equal(normalized.claim, 'Test scientific hypothesis');
    assert.equal(normalized.verdict, 'VERIFIED_TRUE');
    assert.equal(normalized.confidence, 0.8245);
    assert.equal(normalized.disparity, false);
    assert.equal(normalized.abstained, false);
    assert.equal(normalized.evidence.length, 1);
    assert.equal(normalized.evidence[0].stance, 'support');
    assert.equal(normalized.evidence[0].score, 0.892);
    assert.equal(normalized.raw.device, 'cuda');
  });

  test('normalizeResponse throws an honest error on empty/invalid input', () => {
    assert.throws(() => normalizeResponse(null, 'test'), /empty or invalid/);
    assert.throws(() => normalizeResponse(undefined, 'test'), /empty or invalid/);
    assert.throws(() => normalizeResponse('not-an-object', 'test'), /empty or invalid/);
  });

  test('normalizeResponse correctly normalizes all 4 verdict classes', () => {
    const verdicts = [
      'VERIFIED_TRUE',
      'VERIFIED_FALSE',
      'CONFLICTING_OR_CHERRYPICKED',
      'INSUFFICIENT_EVIDENCE',
    ];

    for (const v of verdicts) {
      const res = normalizeResponse({ verdict: v, probabilities: { [v]: 0.9 } }, 'claim');
      assert.equal(res.verdict, v);
      assert.equal(res.confidence, 0.9);
    }
  });

  test('normalizeResponse handles water freezing example with Insufficient Evidence', () => {
    const rawWaterResult = {
      claim_id: 'VX-2FDACD',
      claim: 'Pure water freezes at 0 degrees Celsius at sea level.',
      verdict: 'INSUFFICIENT_EVIDENCE',
      confidence: 0.3244,
      probabilities: {
        VERIFIED_TRUE: 0.2938,
        VERIFIED_FALSE: 0.2948,
        CONFLICTING_OR_CHERRYPICKED: 0.087,
        INSUFFICIENT_EVIDENCE: 0.3244,
      },
      has_cross_passage_disparity: false,
      abstained: false,
      reason: 'Insufficient informative evidence mass retrieved to adjudicate this claim.',
      evidence: [
        {
          id: 'E-01',
          source: 'User-supplied evidence passage',
          text: 'At standard atmospheric pressure, pure water freezes at 0 degrees Celsius.',
          stance: 'neutral',
          score: 0.8003,
        },
      ],
      device: 'cuda',
    };

    const norm = normalizeResponse(rawWaterResult, rawWaterResult.claim);
    assert.equal(norm.verdict, 'INSUFFICIENT_EVIDENCE');
    assert.equal(norm.confidence, 0.3244);
    assert.equal(norm.evidence[0].stance, 'neutral');
    assert.equal(norm.evidence[0].score, 0.8003);
    assert.equal(norm.probs.INSUFFICIENT_EVIDENCE, 0.3244);
  });

  test('normalizeResponse maps mode, retrieval_status, and enriched web evidence attributes', () => {
    const rawWebResult = {
      claim_id: 'VX-WEB-99',
      claim: 'Target web claim',
      verdict: 'VERIFIED_TRUE',
      confidence: 0.88,
      mode: 'web',
      retrieval_status: { status: 'ok', count: 2, time_ms: 120 },
      evidence: [
        {
          id: 'W-01',
          source: 'https://example.com/news',
          source_type: 'web',
          source_title: 'Example News Source',
          source_url: 'https://example.com/news',
          published_date: '2026-05-01',
          text: 'Documented fact affirming the claim.',
          stance: 'support',
          score: 0.88,
          is_claim_aligned: true,
          contributed_to_decision: true,
        },
      ],
    };

    const norm = normalizeResponse(rawWebResult, rawWebResult.claim);
    assert.equal(norm.mode, 'web');
    assert.deepEqual(norm.retrieval_status, { status: 'ok', count: 2, time_ms: 120 });
    assert.equal(norm.evidence[0].source_type, 'web');
    assert.equal(norm.evidence[0].source_title, 'Example News Source');
    assert.equal(norm.evidence[0].source_url, 'https://example.com/news');
    assert.equal(norm.evidence[0].published_date, '2026-05-01');
    assert.equal(norm.evidence[0].is_claim_aligned, true);
    assert.equal(norm.evidence[0].contributed_to_decision, true);
  });

  test('normalizeResponse retains is_relevant: false and relevance_status', () => {
    const rawResultWithUnrelated = {
      claim: "NASA's James Webb Space Telescope launched in December 2021",
      verdict: 'INSUFFICIENT_EVIDENCE',
      confidence: 0.95,
      mode: 'local',
      probabilities: {
        VERIFIED_TRUE: 0.0,
        VERIFIED_FALSE: 0.0,
        CONFLICTING_OR_CHERRYPICKED: 0.0,
        INSUFFICIENT_EVIDENCE: 1.0,
      },
      has_cross_passage_disparity: false,
      abstained: true,
      reason: 'No relevant evidence passages were found to evaluate this claim.',
      evidence: [
        {
          id: 'E-01',
          text: 'The municipal water authority increased residential tax rates in December 2021.',
          stance: 'neutral',
          score: 0.0,
          is_informative: false,
          is_relevant: false,
          relevance_status: 'unrelated',
          is_claim_aligned: false,
          contributed_to_decision: false,
        },
      ],
    };

    const norm = normalizeResponse(rawResultWithUnrelated);
    assert.equal(norm.evidence[0].is_relevant, false);
    assert.equal(norm.evidence[0].relevance_status, 'unrelated');
    assert.equal(norm.evidence[0].score, 0.0);
    assert.equal(norm.evidence[0].contributed_to_decision, false);
  });
});

describe('VERITAS-X Native Voice Narrator Script Builder Tests', () => {
  test('buildNarrationScript includes claim, verdict, and confidence accurately', () => {
    const result = {
      id: 'VX-123',
      claim: 'Pure water freezes at 0 degrees Celsius at sea level.',
      verdict: 'VERIFIED_TRUE',
      confidence: 0.91,
      evidence: [
        {
          id: 'E-01',
          text: 'At standard atmospheric pressure at sea level, water freezes at 0C.',
          stance: 'support',
        },
      ],
      disparity: false,
      abstained: false,
    };

    const script = buildNarrationScript(result);
    assert.ok(script.includes('Pure water freezes at 0 degrees Celsius at sea level.'));
    assert.ok(script.includes('verified true'));
    assert.ok(script.includes('91 percent'));
    assert.ok(script.includes('1 supporting'));
    assert.ok(script.includes('model confidence reflects evidence alignment'));
  });

  test('buildNarrationScript dispatches abstention and disparity warnings honestly', () => {
    const result = {
      id: 'VX-999',
      claim: 'Disputed claim with conflicting evidence',
      verdict: 'CONFLICTING_OR_CHERRYPICKED',
      confidence: 0.46,
      evidence: [
        { id: 'E-01', text: 'Supports claim', stance: 'support' },
        { id: 'E-02', text: 'Refutes claim', stance: 'refute' },
      ],
      disparity: true,
      abstained: false,
    };

    const script = buildNarrationScript(result);
    assert.ok(script.includes('cross-passage stance disparity was detected'));
    assert.ok(script.includes('conflicting or cherrypicked'));
    assert.ok(script.includes('1 supporting, 1 refuting'));
  });

  test('buildNarrationScript handles selective abstention notice', () => {
    const result = {
      id: 'VX-ABS',
      claim: 'Unverifiable proposition',
      verdict: 'INSUFFICIENT_EVIDENCE',
      confidence: 0.28,
      evidence: [],
      disparity: false,
      abstained: true,
    };

    const script = buildNarrationScript(result);
    assert.ok(script.includes('selective coverage gate abstained'));
    assert.ok(script.includes('Human adjudication is recommended'));
  });

  test('buildNarrationScript for all-neutral passages reports 0 supporting, 0 refuting accurately', () => {
    const result = {
      id: 'VX-APOLLO',
      claim: 'A thermonuclear bomb was detonated on the Moon during Apollo 11.',
      verdict: 'VERIFIED_FALSE',
      confidence: 0.35,
      evidence: [
        { id: 'E-01', text: 'Pope blessed Apollo 11.', stance: 'neutral' },
        { id: 'E-02', text: 'Buzz Aldrin communion.', stance: 'neutral' },
      ],
      disparity: false,
      abstained: false,
    };

    const script = buildNarrationScript(result);
    assert.ok(script.includes('0 supporting, 0 refuting, and 2 neutral'));
    assert.ok(script.includes('35 percent'));
  });

  test('normalizeResponse maps is_informative flag and server_latency_ms accurately', () => {
    const raw = {
      claim_id: 'VX-LAT',
      claim: 'Test latency claim',
      verdict: 'VERIFIED_FALSE',
      confidence: 0.35,
      latency_ms: 142.5,
      probabilities: { VERIFIED_FALSE: 0.35 },
      evidence: [
        { id: 'E-01', text: 'Unrelated snippet', stance: 'neutral', is_informative: false, score: 0.98 },
        { id: 'E-02', text: 'Refuting snippet', stance: 'refute', is_informative: true, score: 0.88 },
      ],
    };

    const norm = normalizeResponse(raw, raw.claim);
    assert.equal(norm.server_latency_ms, 142.5);
    assert.equal(norm.evidence[0].is_informative, false);
    assert.equal(norm.evidence[1].is_informative, true);
  });
});

describe('VERITAS-X Phase 4 Accessible Voice Narrator & Speech Controller Tests', () => {
  // Deterministic Mock SpeechSynthesis classes for Node.js test environment
  class MockUtterance {
    constructor(text) {
      this.text = text;
      this.rate = 1.0;
      this.pitch = 1.0;
      this.voice = null;
      this.onstart = null;
      this.onend = null;
      this.onerror = null;
    }
  }

  class MockSpeechSynthesis {
    constructor() {
      this.speaking = false;
      this.paused = false;
      this.queue = [];
      this.callLog = [];
    }

    getVoices() {
      return [
        { name: 'Natural English (US)', lang: 'en-US', voiceURI: 'en-us-natural' },
        { name: 'Google UK English Male', lang: 'en-GB', voiceURI: 'en-gb-male' },
      ];
    }

    speak(utterance) {
      this.callLog.push({ action: 'speak', text: utterance.text });
      this.speaking = true;
      this.paused = false;
      this.queue.push(utterance);
    }

    pause() {
      this.callLog.push({ action: 'pause' });
      this.paused = true;
    }

    resume() {
      this.callLog.push({ action: 'resume' });
      this.paused = false;
      this.speaking = true;
    }

    cancel() {
      this.callLog.push({ action: 'cancel' });
      this.speaking = false;
      this.paused = false;
      this.queue = [];
    }
  }

  const sampleResult = {
    id: 'VX-VOICE-01',
    claim: 'Pure water freezes at 0 degrees Celsius under standard atmospheric pressure.',
    verdict: 'VERIFIED_TRUE',
    confidence: 0.92,
    abstained: false,
    disparity: false,
    evidence: [
      { id: 'E-01', text: 'Water freezes at 0 degrees Celsius at 1 atm.', stance: 'support' },
      { id: 'E-02', text: 'Standard ice point is 273.15 Kelvin.', stance: 'support' },
    ],
  };

  test('buildNarrationSegments decomposes result into typed, valid segments', () => {
    const segments = buildNarrationSegments(sampleResult);
    assert.ok(Array.isArray(segments));
    assert.ok(segments.length >= 4);

    const claimSeg = segments.find((s) => s.type === 'claim');
    assert.ok(claimSeg);
    assert.ok(claimSeg.text.includes('Pure water freezes at 0 degrees Celsius'));

    const verdictSeg = segments.find((s) => s.type === 'verdict');
    assert.ok(verdictSeg);
    assert.ok(verdictSeg.text.includes('verified true'));
    assert.ok(verdictSeg.text.includes('92 percent'));

    const tallySeg = segments.find((s) => s.id === 'seg-evidence-tally');
    assert.ok(tallySeg);
    assert.ok(tallySeg.text.includes('2 evidence passages: 2 supporting, 0 refuting, and 0 neutral'));

    const disclaimerSeg = segments.find((s) => s.type === 'disclaimer');
    assert.ok(disclaimerSeg);
    assert.ok(disclaimerSeg.text.includes('model confidence reflects evidence alignment'));
  });

  test('buildNarrationSegments handles operational abstention segment', () => {
    const abstainedResult = {
      ...sampleResult,
      abstained: true,
      confidence: 0.25,
    };
    const segments = buildNarrationSegments(abstainedResult);
    const absSeg = segments.find((s) => s.type === 'abstention');
    assert.ok(absSeg);
    assert.ok(absSeg.text.includes('selective coverage gate abstained'));
  });

  test('SpeechBriefingController initializes and reports support correctly', () => {
    const synth = new MockSpeechSynthesis();
    const controller = new SpeechBriefingController({
      speechSynthesis: synth,
      UtteranceClass: MockUtterance,
    });

    assert.equal(controller.isSupported(), true);

    const unsupported = new SpeechBriefingController({
      speechSynthesis: null,
      UtteranceClass: null,
    });
    assert.equal(unsupported.isSupported(), false);
  });

  test('SpeechBriefingController transitions play -> pause -> resume -> stop', () => {
    const synth = new MockSpeechSynthesis();
    const stateHistory = [];

    const controller = new SpeechBriefingController({
      speechSynthesis: synth,
      UtteranceClass: MockUtterance,
      onStateChange: (state) => stateHistory.push(state),
    });

    const segments = buildNarrationSegments(sampleResult);
    controller.loadSegments(segments);

    assert.equal(controller.state, 'idle');

    // Play
    controller.play();
    assert.equal(controller.state, 'speaking');
    assert.equal(synth.callLog.some((c) => c.action === 'speak'), true);

    // Pause
    controller.pause();
    assert.equal(controller.state, 'paused');
    assert.equal(synth.callLog.some((c) => c.action === 'pause'), true);

    // Resume
    controller.resume();
    assert.equal(controller.state, 'speaking');
    assert.equal(synth.callLog.some((c) => c.action === 'resume'), true);

    // Stop
    controller.stop();
    assert.equal(controller.state, 'stopped');
    assert.equal(synth.callLog.some((c) => c.action === 'cancel'), true);
  });

  test('SpeechBriefingController advances segments sequentially until completed', () => {
    const synth = new MockSpeechSynthesis();
    let completed = false;

    const controller = new SpeechBriefingController({
      speechSynthesis: synth,
      UtteranceClass: MockUtterance,
      onStateChange: (state) => {
        if (state === 'completed') completed = true;
      },
    });

    const segments = buildNarrationSegments(sampleResult);
    controller.loadSegments(segments);
    controller.play();

    assert.equal(controller.currentIndex, 0);

    // Simulate speech-synthesis onend for each utterance in queue
    while (controller.currentIndex < segments.length && !completed) {
      const active = controller.activeUtterance;
      assert.ok(active);
      active.onend();
    }

    assert.equal(completed, true);
    assert.equal(controller.state, 'completed');
  });

  test('SpeechBriefingController safely clamps rate and pitch', () => {
    const controller = new SpeechBriefingController();
    controller.setRate(10);
    assert.equal(controller.rate, 2.0); // Clamped max

    controller.setRate(0.1);
    assert.equal(controller.rate, 0.5); // Clamped min

    controller.setRate(1.25);
    assert.equal(controller.rate, 1.25);
  });

  test('SpeechBriefingController handles rapid repeated interactions without race errors', () => {
    const synth = new MockSpeechSynthesis();
    const controller = new SpeechBriefingController({
      speechSynthesis: synth,
      UtteranceClass: MockUtterance,
    });

    const segments = buildNarrationSegments(sampleResult);
    controller.loadSegments(segments);

    // Rapid spam simulation
    assert.doesNotThrow(() => {
      controller.play();
      controller.pause();
      controller.play();
      controller.stop();
      controller.replay();
      controller.stop();
      controller.stop();
    });

    assert.equal(controller.state, 'stopped');
  });
});

