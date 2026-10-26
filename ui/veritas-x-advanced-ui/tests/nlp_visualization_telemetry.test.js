import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeResponse } from '../src/lib/api.js';

describe('VERITAS-X Phase 4B — Cinematic NLP Visualization & Telemetry Tests', () => {

  const sampleBackendPayload = {
    claim_id: 'claim_20261026_test',
    claim: 'incognitio mdoe dont stops websits traking me',
    original_claim: 'incognitio mdoe dont stops websits traking me',
    sanitized_claim: 'incognitio mdoe dont stops websits traking me',
    normalized_claim: 'incognito mode does not stop websites tracking me',
    normalization_status: 'accepted',
    normalization_corrections: [
      { original: 'incognitio', replacement: 'incognito', category: 'typo', confidence: 0.96, start_char: 0, end_char: 10, edit_distance: 1 },
      { original: 'mdoe', replacement: 'mode', category: 'typo', confidence: 0.98, start_char: 11, end_char: 15, edit_distance: 1 },
      { original: 'dont', replacement: 'does not', category: 'contraction', confidence: 0.99, start_char: 16, end_char: 20, edit_distance: 1 },
      { original: 'stops', replacement: 'stop', category: 'agreement', confidence: 0.94, start_char: 21, end_char: 26, edit_distance: 1 },
      { original: 'websits', replacement: 'websites', category: 'typo', confidence: 0.97, start_char: 27, end_char: 34, edit_distance: 1 },
      { original: 'traking', replacement: 'tracking', category: 'typo', confidence: 0.98, start_char: 35, end_char: 42, edit_distance: 1 },
    ],
    normalization_warnings: [],
    structured_claim: {
      subject: 'incognito mode',
      predicate: 'stop',
      object: 'websites tracking me',
      modality: 'certain',
      is_negated: true,
      negation_scope: 'dont stops websits traking me',
      is_conditional: false,
      is_causal: false,
      named_entities: [],
      numeric_entities: [],
      quantifiers: [],
      sub_propositions: [],
    },
    retrieval_query_used: 'incognito mode does not stop websites tracking me',
    nli_hypothesis_used: 'incognitio mdoe dont stops websits traking me',
    processing_trace_version: '1.0.0',
    processing_trace: [
      { sequence: 1, stage: 'input_received', status: 'completed', duration_ms: 0.1, input_ref: 'incognitio mdoe dont stops websits traking me', output_ref: 'incognitio mdoe dont stops websits traking me', details: { raw_claim: 'incognitio mdoe dont stops websits traking me', char_length: 45, raw_word_count: 6 } },
      { sequence: 2, stage: 'sanitization', status: 'completed', duration_ms: 0.2, input_ref: 'incognitio mdoe dont stops websits traking me', output_ref: 'incognitio mdoe dont stops websits traking me', details: { sanitized_claim: 'incognitio mdoe dont stops websits traking me', transformations: [], is_question: false, has_negation: true } },
      { sequence: 3, stage: 'spelling_normalization', status: 'completed', duration_ms: 0.8, input_ref: 'incognitio mdoe dont stops websits traking me', output_ref: 'incognito mode does not stop websites tracking me', details: { normalized_claim: 'incognito mode does not stop websites tracking me', corrections: [{ original: 'mdoe', replacement: 'mode' }], word_correction_events: [{ original: 'mdoe', replacement: 'mode', status: 'accepted', confidence: 0.98, start_char: 11, end_char: 15 }] } },
      { sequence: 4, stage: 'meaning_preservation', status: 'completed', duration_ms: 0.4, input_ref: 'incognito mode does not stop websites tracking me', output_ref: 'incognito mode does not stop websites tracking me', details: { is_safe: true, decision: 'USE_NORMALIZED', invariants_passed: { negation_and_polarity: true, quantifier_and_scope: true } } },
      { sequence: 5, stage: 'tokenization', status: 'completed', duration_ms: 0.3, input_ref: 'incognitio mdoe dont stops websits traking me', output_ref: '6 tokens', details: { token_count: 6, tokens: [{ text: 'incognitio', lemma: 'incognitio', pos: 'NN', upos: 'NOUN', start_char: 0, end_char: 10 }] } },
      { sequence: 6, stage: 'linguistic_analysis', status: 'completed', duration_ms: 0.3, input_ref: 'incognitio mdoe dont stops websits traking me', output_ref: 'type:assertion', details: { sentence_type: 'assertion', is_question: false, is_passive: false, subject_candidate: 'incognito mode', predicate_candidate: 'stop', object_candidate: 'websites tracking me', clauses: [], parser_engine: 'shallow_syntactic_rule_parser' } },
      { sequence: 7, stage: 'claim_extraction', status: 'completed', duration_ms: 0.3, input_ref: 'incognitio mdoe dont stops websits traking me', output_ref: 'Modality:certain', details: { structured_claim: { subject: 'incognito mode', predicate: 'stop' }, has_negation: true, modality: 'certain' } },
      { sequence: 8, stage: 'retrieval', status: 'completed', duration_ms: 4.5, input_ref: 'incognito mode does not stop websites tracking me', output_ref: '2 passages', details: { retrieval_query: 'incognito mode does not stop websites tracking me', query_variant: 'normalized', normalization_status: 'accepted', retrieval_mode: 'local', provider: 'local_corpus', systems_run: ['bm25', 'dense'], results_count: 2 } },
      { sequence: 9, stage: 'evidence_processing', status: 'completed', duration_ms: 0.2, input_ref: '2 candidates', output_ref: '2 ready', details: { candidate_passages_count: 2, retrieval_mode: 'local' } },
      { sequence: 10, stage: 'nli_verification', status: 'completed', duration_ms: 8.5, input_ref: "hypothesis: 'incognitio mdoe dont stops websits traking me'", output_ref: 'raw_verdict: VERIFIED_TRUE', details: { hypothesis_used: 'incognitio mdoe dont stops websits traking me', hypothesis_source: 'canonical_sanitized_user_claim', model: 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli', device: 'cpu', calibration_temperature: 4.1039, candidate_count: 2, inference_ms: 8.5, raw_prediction: 'VERIFIED_TRUE' } },
      { sequence: 11, stage: 'final_result', status: 'completed', duration_ms: 0.5, input_ref: 'raw_prediction:VERIFIED_TRUE', output_ref: 'VERIFIED_TRUE', details: { verdict: 'VERIFIED_TRUE', confidence: 0.92, probabilities: { VERIFIED_TRUE: 0.92, VERIFIED_FALSE: 0.03, CONFLICTING_OR_CHERRYPICKED: 0.01, INSUFFICIENT_EVIDENCE: 0.04 }, abstained: false } }
    ],
    verdict: 'VERIFIED_TRUE',
    confidence: 0.92,
    probabilities: { VERIFIED_TRUE: 0.92, VERIFIED_FALSE: 0.03, CONFLICTING_OR_CHERRYPICKED: 0.01, INSUFFICIENT_EVIDENCE: 0.04 },
    has_cross_passage_disparity: false,
    abstained: false,
    reason: 'Local evidence supports the claim with 2 supporting passage(s) (calibrated confidence 92.0%).',
    evidence: [
      {
        passage_id: 'ev_01',
        text: 'Private browsing and incognito mode prevent local browsing history and cookies from persisting after a session ends, but do not prevent external websites or network providers from tracking the user.',
        source: 'Browser Privacy Whitepaper',
        source_title: 'Browser Privacy Whitepaper',
        source_type: 'local',
        stance: 'support',
        score: 0.93,
        is_relevant: true,
        is_decision_eligible: true,
        claim_coverage: 'full',
      }
    ],
    retrieval_status: { mode: 'local', provider: 'local_corpus', status: 'success', results_count: 1 },
    timing_ms: { init_ms: 0, claim_prep_ms: 2.1, retrieval_ms: 4.5, engine_inference_ms: 8.5, total_ms: 15.1 },
    latency_ms: 15.1,
    mode: 'local',
  };

  test('normalizeResponse maps all Phase 4A NLP fields accurately into client contract', () => {
    const normalized = normalizeResponse(sampleBackendPayload, 'incognitio mdoe dont stops websits traking me');

    assert.equal(normalized.original_claim, 'incognitio mdoe dont stops websits traking me');
    assert.equal(normalized.sanitized_claim, 'incognitio mdoe dont stops websits traking me');
    assert.equal(normalized.normalized_claim, 'incognito mode does not stop websites tracking me');
    assert.equal(normalized.normalization_status, 'accepted');
    assert.equal(normalized.normalization_corrections.length, 6);
    assert.equal(normalized.retrieval_query_used, 'incognito mode does not stop websites tracking me');
    assert.equal(normalized.nli_hypothesis_used, 'incognitio mdoe dont stops websits traking me');
    assert.equal(normalized.processing_trace_version, '1.0.0');
    assert.equal(normalized.processing_trace.length, 11);
  });

  test('Processing trace contains 11 ordered stages with accurate stage identifiers', () => {
    const normalized = normalizeResponse(sampleBackendPayload);
    const trace = normalized.processing_trace;

    const expectedStages = [
      'input_received',
      'sanitization',
      'spelling_normalization',
      'meaning_preservation',
      'tokenization',
      'linguistic_analysis',
      'claim_extraction',
      'retrieval',
      'evidence_processing',
      'nli_verification',
      'final_result',
    ];

    assert.equal(trace.length, 11);
    trace.forEach((stageItem, idx) => {
      assert.equal(stageItem.sequence, idx + 1);
      assert.equal(stageItem.stage, expectedStages[idx]);
      assert.ok(typeof stageItem.duration_ms === 'number');
      assert.ok(stageItem.details !== undefined);
    });
  });

  test('Retrieval search query and canonical NLI hypothesis are strictly distinguished', () => {
    const normalized = normalizeResponse(sampleBackendPayload);

    // Retrieval query uses the normalized query to improve search recall
    assert.equal(normalized.retrieval_query_used, 'incognito mode does not stop websites tracking me');

    // NLI hypothesis strictly preserves original user proposition for cross-encoder truth verification
    assert.equal(normalized.nli_hypothesis_used, 'incognitio mdoe dont stops websits traking me');

    // NLI verification stage records canonical user claim as hypothesis source
    const nliStage = normalized.processing_trace.find((s) => s.stage === 'nli_verification');
    assert.ok(nliStage);
    assert.equal(nliStage.details.hypothesis_source, 'canonical_sanitized_user_claim');
    assert.equal(nliStage.details.hypothesis_used, 'incognitio mdoe dont stops websits traking me');
    assert.equal(nliStage.details.calibration_temperature, 4.1039);
  });

  test('Word correction events preserve character offsets and category metadata', () => {
    const normalized = normalizeResponse(sampleBackendPayload);
    const corrections = normalized.normalization_corrections;

    assert.equal(corrections[0].original, 'incognitio');
    assert.equal(corrections[0].replacement, 'incognito');
    assert.equal(corrections[0].start_char, 0);
    assert.equal(corrections[0].end_char, 10);
    assert.equal(corrections[0].category, 'typo');

    assert.equal(corrections[1].original, 'mdoe');
    assert.equal(corrections[1].replacement, 'mode');
    assert.equal(corrections[1].start_char, 11);
    assert.equal(corrections[1].end_char, 15);
  });

  test('Meaning preservation checkpoint records invariant gatekeeper verdict', () => {
    const normalized = normalizeResponse(sampleBackendPayload);
    const guardStage = normalized.processing_trace.find((s) => s.stage === 'meaning_preservation');

    assert.ok(guardStage);
    assert.equal(guardStage.details.is_safe, true);
    assert.equal(guardStage.details.decision, 'USE_NORMALIZED');
    assert.equal(guardStage.details.invariants_passed.negation_and_polarity, true);
    assert.equal(guardStage.details.invariants_passed.quantifier_and_scope, true);
  });

  test('Tokenization stage records token boundary offsets, lemmas, and POS tags', () => {
    const normalized = normalizeResponse(sampleBackendPayload);
    const tokStage = normalized.processing_trace.find((s) => s.stage === 'tokenization');

    assert.ok(tokStage);
    assert.equal(tokStage.details.token_count, 6);
    assert.ok(Array.isArray(tokStage.details.tokens));
    const firstTok = tokStage.details.tokens[0];
    assert.equal(firstTok.text, 'incognitio');
    assert.equal(firstTok.start_char, 0);
    assert.equal(firstTok.end_char, 10);
    assert.equal(firstTok.upos, 'NOUN');
  });

  test('Linguistic analysis reports rule parser engine and SVO arguments', () => {
    const normalized = normalizeResponse(sampleBackendPayload);
    const lingStage = normalized.processing_trace.find((s) => s.stage === 'linguistic_analysis');

    assert.ok(lingStage);
    assert.equal(lingStage.details.parser_engine, 'shallow_syntactic_rule_parser');
    assert.equal(lingStage.details.sentence_type, 'assertion');
    assert.equal(lingStage.details.is_question, false);
    assert.equal(lingStage.details.subject_candidate, 'incognito mode');
    assert.equal(lingStage.details.predicate_candidate, 'stop');
  });

  test('Claim extraction structured proposition preserves negation scope and modality', () => {
    const normalized = normalizeResponse(sampleBackendPayload);
    const struct = normalized.structured_claim;

    assert.equal(struct.subject, 'incognito mode');
    assert.equal(struct.predicate, 'stop');
    assert.equal(struct.is_negated, true);
    assert.equal(struct.modality, 'certain');
    assert.equal(struct.negation_scope, 'dont stops websits traking me');
  });

  test('Missing or empty processing trace is handled gracefully with empty array', () => {
    const payloadWithoutTrace = {
      claim: 'A simple claim without trace',
      verdict: 'VERIFIED_TRUE',
      confidence: 0.88,
    };

    const normalized = normalizeResponse(payloadWithoutTrace);
    assert.ok(Array.isArray(normalized.processing_trace));
    assert.equal(normalized.processing_trace.length, 0);
    assert.equal(normalized.original_claim, 'A simple claim without trace');
    assert.equal(normalized.sanitized_claim, 'A simple claim without trace');
    assert.equal(normalized.normalization_status, 'accepted');
  });

});
