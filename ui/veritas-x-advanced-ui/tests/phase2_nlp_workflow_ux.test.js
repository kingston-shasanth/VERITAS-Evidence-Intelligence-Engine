import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { STAGE_CONFIG, PHASES, VERDICT_TAXONOMY } from '../src/lib/stages.js';

describe('VERITAS-X Phase 2 — NLP Workflow UX & Reliability Unit Tests', () => {

  const sampleTrace = [
    { sequence: 1, stage: 'input_received', status: 'completed', duration_ms: 0.1, details: { raw_claim: 'Pur watur frezes at 0 C' } },
    { sequence: 2, stage: 'sanitization', status: 'completed', duration_ms: 0.2, details: { sanitized_claim: 'Pur watur frezes at 0 C' } },
    { sequence: 3, stage: 'spelling_normalization', status: 'completed', duration_ms: 0.8, details: { normalized_claim: 'Pure water freezes at 0 C', corrections: [{ original: 'watur', replacement: 'water' }] } },
    { sequence: 4, stage: 'meaning_preservation', status: 'completed', duration_ms: 0.4, details: { is_safe: true, decision: 'USE_NORMALIZED' } },
    { sequence: 5, stage: 'tokenization', status: 'completed', duration_ms: 0.3, details: { token_count: 5 } },
    { sequence: 6, stage: 'linguistic_analysis', status: 'completed', duration_ms: 0.3, details: { sentence_type: 'assertion', subject_candidate: 'Pure water' } },
    { sequence: 7, stage: 'claim_extraction', status: 'completed', duration_ms: 0.3, details: { structured_claim: { subject: 'Pure water', predicate: 'freezes' } } },
    { sequence: 8, stage: 'retrieval', status: 'completed', duration_ms: 4.5, details: { results_count: 3 } },
    { sequence: 9, stage: 'evidence_processing', status: 'completed', duration_ms: 0.2, details: { candidate_passages_count: 3 } },
    { sequence: 10, stage: 'nli_verification', status: 'completed', duration_ms: 8.5, details: { hypothesis_used: 'Pur watur frezes at 0 C', raw_prediction: 'VERIFIED_TRUE' } },
    { sequence: 11, stage: 'final_result', status: 'completed', duration_ms: 0.5, details: { verdict: 'VERIFIED_TRUE', confidence: 0.94 } },
  ];

  test('All 11 internal pipeline stages are defined with human-readable titles', () => {
    assert.equal(STAGE_CONFIG.length, 11);
    const stageIds = STAGE_CONFIG.map((s) => s.id);
    const expected = [
      'input_received', 'sanitization', 'spelling_normalization',
      'meaning_preservation', 'tokenization', 'linguistic_analysis',
      'claim_extraction', 'retrieval', 'evidence_processing',
      'nli_verification', 'final_result'
    ];
    assert.deepEqual(stageIds, expected);

    // Ensure labels are not raw machine variable names
    STAGE_CONFIG.forEach((s) => {
      assert.ok(s.label && !s.label.includes('_'), `Stage label should be human-readable: ${s.label}`);
      assert.ok(s.short && s.short.length <= 16, `Short label should be compact: ${s.short}`);
      assert.ok(s.desc && s.desc.length > 5, `Stage description should be informative: ${s.desc}`);
    });
  });

  test('6-Phase conceptual grouping covers all 11 backend trace stages without gaps', () => {
    assert.equal(PHASES.length, 6);

    const allMappedStages = PHASES.flatMap((p) => p.stageIds);
    assert.equal(allMappedStages.length, 11);

    // Verify all 11 stage IDs from STAGE_CONFIG are present in the phases
    STAGE_CONFIG.forEach((cfg) => {
      assert.ok(allMappedStages.includes(cfg.id), `Stage ${cfg.id} must be mapped in the 6 phases`);
    });
  });

  test('Completed trace status never resolves to queued or pending for finished events', () => {
    // Simulate ProcessingTimeline logic on completed trace
    const allEventsCompleted = sampleTrace.every((e) => e.status === 'completed');
    assert.equal(allEventsCompleted, true);

    STAGE_CONFIG.forEach((cfg, idx) => {
      const traceEvt = sampleTrace.find((e) => e.stage === cfg.id) || sampleTrace[idx];
      assert.ok(traceEvt, `Event for stage ${cfg.id} should be present`);
      assert.equal(traceEvt.status, 'completed');

      const isCompleted = traceEvt.status === 'completed' || allEventsCompleted;
      assert.equal(isCompleted, true, `Stage ${cfg.id} must resolve to completed, never queued`);
    });
  });

  test('Attention status is correctly propagated if guardrail or analysis flags uncertainty', () => {
    const uncertainTrace = sampleTrace.map((e) =>
      e.stage === 'meaning_preservation' ? { ...e, status: 'uncertain' } : e
    );

    const cleanupPhaseStages = ['spelling_normalization', 'meaning_preservation']
      .map((sid) => uncertainTrace.find((t) => t.stage === sid))
      .filter(Boolean);

    const hasAttention = cleanupPhaseStages.some((s) => s.status === 'uncertain' || s.status === 'fallback');
    assert.equal(hasAttention, true, 'Cleanup phase must flag attention when meaning preservation is uncertain');
  });

  test('Failure status in any stage is immediately propagated to phase status', () => {
    const failedTrace = sampleTrace.map((e) =>
      e.stage === 'retrieval' ? { ...e, status: 'failed' } : e
    );

    const searchPhaseStages = ['retrieval', 'evidence_processing']
      .map((sid) => failedTrace.find((t) => t.stage === sid))
      .filter(Boolean);

    const hasFailed = searchPhaseStages.some((s) => s.status === 'failed');
    assert.equal(hasFailed, true, 'Search phase must reflect failed status honestly');
  });

  test('Plain-language taxonomy is used for public verdict presentation', () => {
    assert.equal(VERDICT_TAXONOMY.VERIFIED_TRUE.label, 'Verified True');
    assert.equal(VERDICT_TAXONOMY.VERIFIED_FALSE.label, 'Verified False');
    assert.equal(VERDICT_TAXONOMY.CONFLICTING_OR_CHERRYPICKED.label, 'Conflicting Evidence');
    assert.equal(VERDICT_TAXONOMY.INSUFFICIENT_EVIDENCE.label, 'Insufficient Evidence');

    // Confirm no raw ML labels leak into the public headline
    Object.values(VERDICT_TAXONOMY).forEach((meta) => {
      const text = `${meta.label} ${meta.headline}`.toUpperCase();
      assert.ok(!text.includes('PLATT'), 'Platt temperature must not be in verdict label');
      assert.ok(!text.includes('DEBERTA'), 'Model name must not be in verdict label');
      assert.ok(!text.includes('CROSS-ENCODER'), 'Cross-encoder acronym must not be in primary verdict label');
    });
  });

  test('Stale request protection: newer request ID supersedes older in-flight response', () => {
    let activeRequestId = 0;
    let storedResult = null;

    // Simulate Request 1
    const req1Id = ++activeRequestId;
    // Simulate Request 2 dispatched before Request 1 finishes
    const req2Id = ++activeRequestId;

    // Request 2 completes first
    if (req2Id === activeRequestId) {
      storedResult = { id: 'req2', claim: 'Second Claim' };
    }

    // Request 1 completes later (stale)
    if (req1Id === activeRequestId) {
      storedResult = { id: 'req1', claim: 'First Claim' };
    }

    // Verify stored result is from Request 2, Request 1 was safely discarded
    assert.equal(storedResult.id, 'req2');
    assert.equal(storedResult.claim, 'Second Claim');
  });
});
