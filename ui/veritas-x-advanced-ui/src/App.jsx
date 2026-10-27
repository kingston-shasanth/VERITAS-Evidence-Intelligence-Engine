import React, { useMemo, useState, useEffect, useRef } from 'react';
import { verifyClaim, apiBase, normalizeResponse } from './lib/api.js';
import { Icon } from './components/Icons.jsx';
import { VeritasWordmark, VeritasSymbol } from './components/VeritasLogo.jsx';
import { CommandCenterHero } from './components/CommandCenterHero.jsx';
import { PipelineVisualizer } from './components/PipelineVisualizer.jsx';
import { InvestigationProgress } from './components/InvestigationProgress.jsx';
import { ForensicReport } from './components/ForensicReport.jsx';
import { EvidenceDossier } from './components/EvidenceDossier.jsx';
import { EvidenceGraph } from './components/EvidenceGraph.jsx';
import { TimelineView } from './components/TimelineView.jsx';
import { MultiHopInspector } from './components/MultiHopInspector.jsx';
import { ModelPerformanceLab } from './components/ModelPerformanceLab.jsx';
import { SystemTelemetry } from './components/SystemTelemetry.jsx';
import { NLPVisualization } from './components/NLPVisualization.jsx';

const LABELS = ['VERIFIED_TRUE', 'VERIFIED_FALSE', 'CONFLICTING_OR_CHERRYPICKED', 'INSUFFICIENT_EVIDENCE'];

const LABEL_DISPLAY = {
  VERIFIED_TRUE: 'Verified True',
  VERIFIED_FALSE: 'Verified False',
  CONFLICTING_OR_CHERRYPICKED: 'Conflicting Evidence',
  INSUFFICIENT_EVIDENCE: 'Insufficient Evidence',
};

const LABEL_CLASS = {
  VERIFIED_TRUE: 'true',
  VERIFIED_FALSE: 'false',
  CONFLICTING_OR_CHERRYPICKED: 'conflict',
  INSUFFICIENT_EVIDENCE: 'insufficient',
};

const BENCHMARK_BARS = [
  { label: 'Verified True', value: 58.9, color: 'var(--verdict-true)' },
  { label: 'Verified False', value: 71.8, color: 'var(--verdict-false)' },
  { label: 'Conflicting', value: 16.0, color: 'var(--verdict-conflict)' },
  { label: 'Insufficient', value: 31.5, color: 'var(--verdict-insufficient)' },
];

const FIXTURE_CASES = [
  {
    id: 'VX-DEMO-014',
    claim: 'Pure water freezes at 0 degrees Celsius at sea level.',
    verdict: 'VERIFIED_TRUE',
    confidence: 0.91,
    probs: { VERIFIED_TRUE: 0.91, VERIFIED_FALSE: 0.03, CONFLICTING_OR_CHERRYPICKED: 0.01, INSUFFICIENT_EVIDENCE: 0.05 },
    summary: 'The provided passages consistently support the claim under standard atmospheric pressure.',
    disparity: false,
    abstained: false,
    subclaims: [
      { subclaim_id: 'SC-01', text: 'Pure water freezes at 0 degrees Celsius', status: 'SUPPORTED', confidence: 0.94, stance_distribution: { support: 2, refute: 0, neutral: 0 }, relation_to_parent: 'conjunct', explanation: 'Entailed by standard chemical thermodynamics.' },
      { subclaim_id: 'SC-02', text: 'The freezing point applies at sea level under 1 atm', status: 'SUPPORTED', confidence: 0.88, dependencies: ['SC-01'], stance_distribution: { support: 1, refute: 0, neutral: 0 }, relation_to_parent: 'condition', explanation: 'Standard atmospheric pressure condition verified.' },
    ],
    temporal: {
      temporal_scope: 'CURRENT',
      temporal_status: 'CONSISTENT',
      explanation: 'Physical laws and IUPAC standard chemical constants are time-invariant.',
      timeline_events: [
        { evidence_id: 'E-01', date: '1954-01-01', date_type: 'publication', stance: 'support', text: 'A water molecule consists of two hydrogen atoms bonded to one oxygen atom (H₂O). Under standard pressure at sea level, it freezes at 0°C.', provenance: 'Chemistry Reference Manual' },
        { evidence_id: 'E-02', date: '1975-06-15', date_type: 'publication', stance: 'support', text: 'The Celsius temperature scale defines zero degrees as the melting and freezing point of water at 1 atm.', provenance: 'Physical Science Handbook' },
      ],
    },
    uncertainty: {
      shannon_entropy: 0.32,
      top_margin: 0.86,
      coverage_ratio: 1.0,
      is_abstained: false,
    },
    evidence: [
      { id: 'E-01', source: 'Chemistry Reference Manual', stance: 'support', score: 0.94, entailment_prob: 0.94, contradiction_prob: 0.02, neutral_prob: 0.04, is_informative: true, text: 'A water molecule consists of two hydrogen atoms bonded to one oxygen atom (H₂O). Under standard pressure at sea level, it freezes at 0°C.' },
      { id: 'E-02', source: 'Physical Science Handbook', stance: 'support', score: 0.88, entailment_prob: 0.88, contradiction_prob: 0.03, neutral_prob: 0.09, is_informative: true, text: 'The Celsius temperature scale defines zero degrees as the melting and freezing point of water at 1 atm.' },
    ],
  },
  {
    id: 'VX-DEMO-027',
    claim: 'The Moon is a star that produces its own light through nuclear fusion.',
    verdict: 'VERIFIED_FALSE',
    confidence: 0.86,
    probs: { VERIFIED_TRUE: 0.04, VERIFIED_FALSE: 0.86, CONFLICTING_OR_CHERRYPICKED: 0.03, INSUFFICIENT_EVIDENCE: 0.07 },
    summary: 'The demonstration evidence directly contradicts the claim: the Moon is an oblate planetary satellite.',
    disparity: false,
    abstained: false,
    subclaims: [
      { subclaim_id: 'SC-01', text: 'The Moon is a star', status: 'REFUTED', confidence: 0.92, stance_distribution: { support: 0, refute: 2, neutral: 0 }, relation_to_parent: 'conjunct', explanation: 'Directly contradicted by astronomical classification.' },
      { subclaim_id: 'SC-02', text: 'The Moon produces light through nuclear fusion', status: 'REFUTED', confidence: 0.86, dependencies: ['SC-01'], stance_distribution: { support: 0, refute: 2, neutral: 0 }, relation_to_parent: 'causal', explanation: 'Contradicted: the Moon reflects solar illumination and lacks core fusion mass.' },
    ],
    temporal: {
      temporal_scope: 'HISTORICAL',
      temporal_status: 'CONSISTENT',
      explanation: 'Astrophysical planetary classification is established and current.',
      timeline_events: [
        { evidence_id: 'E-03', date: '1969-07-20', date_type: 'event', stance: 'refute', text: 'The Moon is a natural rocky satellite that reflects solar illumination; it lacks sufficient mass for nuclear fusion.', provenance: 'Astronomy Primer' },
        { evidence_id: 'E-04', date: '1984-11-10', date_type: 'publication', stance: 'refute', text: 'Stars generate light through sustained thermonuclear reactions in their cores, whereas the Moon is cold rock.', provenance: 'Space Physics Compendium' },
      ],
    },
    uncertainty: {
      shannon_entropy: 0.44,
      top_margin: 0.79,
      coverage_ratio: 1.0,
      is_abstained: false,
    },
    evidence: [
      { id: 'E-03', source: 'Astronomy Primer', stance: 'refute', score: 0.92, entailment_prob: 0.02, contradiction_prob: 0.92, neutral_prob: 0.06, is_informative: true, text: 'The Moon is a natural rocky satellite that reflects solar illumination; it lacks sufficient mass for nuclear fusion.' },
      { id: 'E-04', source: 'Space Physics Compendium', stance: 'refute', score: 0.86, entailment_prob: 0.03, contradiction_prob: 0.86, neutral_prob: 0.11, is_informative: true, text: 'Stars generate light through sustained thermonuclear reactions in their cores, whereas the Moon is cold rock.' },
    ],
  },
  {
    id: 'VX-DEMO-038',
    claim: 'The municipal bridge was completely closed to vehicle traffic on Monday.',
    verdict: 'CONFLICTING_OR_CHERRYPICKED',
    confidence: 0.46,
    probs: { VERIFIED_TRUE: 0.32, VERIFIED_FALSE: 0.22, CONFLICTING_OR_CHERRYPICKED: 0.46, INSUFFICIENT_EVIDENCE: 0.0 },
    summary: 'Cross-passage stance disparity detected: two distinct municipal bulletins present mutually contradictory statements.',
    disparity: true,
    abstained: false,
    subclaims: [
      { subclaim_id: 'SC-01', text: 'The municipal bridge was completely closed to vehicle traffic', status: 'UNRESOLVED', confidence: 0.46, stance_distribution: { support: 1, refute: 1, neutral: 0 }, relation_to_parent: 'conjunct', explanation: 'Disparity between Public Works bulletin and Transport Authority dispatch.' },
      { subclaim_id: 'SC-02', text: 'The closure was on Monday', status: 'SUPPORTED', confidence: 0.82, stance_distribution: { support: 2, refute: 0, neutral: 0 }, relation_to_parent: 'temporal_anchor', explanation: 'Both bulletins agree that events referenced Monday.' },
    ],
    temporal: {
      temporal_scope: 'TIME_SPAN',
      temporal_status: 'OUTDATED_SUPERSEDED',
      explanation: 'Later Metropolitan Transport Authority notice supersedes earlier emergency dispatch.',
      timeline_events: [
        { evidence_id: 'E-05', date: '2024-03-04T07:00:00Z', date_type: 'publication', stance: 'support', text: 'The municipal public works bulletin stated the bridge was closed to all traffic on Monday for emergency span repairs.', provenance: 'Public Works Department Dispatch' },
        { evidence_id: 'E-06', date: '2024-03-04T09:30:00Z', date_type: 'publication', stance: 'refute', text: 'The metropolitan transport authority stated that the bridge remained fully open to commercial and private vehicles on Monday.', provenance: 'Metropolitan Transport Authority Notice' },
      ],
    },
    uncertainty: {
      shannon_entropy: 1.15,
      top_margin: 0.14,
      coverage_ratio: 1.0,
      is_abstained: false,
    },
    evidence: [
      { id: 'E-05', source: 'Public Works Department Dispatch', stance: 'support', score: 0.84, entailment_prob: 0.84, contradiction_prob: 0.06, neutral_prob: 0.10, is_informative: true, text: 'The municipal public works bulletin stated the bridge was closed to all traffic on Monday for emergency span repairs.' },
      { id: 'E-06', source: 'Metropolitan Transport Authority Notice', stance: 'refute', score: 0.79, entailment_prob: 0.05, contradiction_prob: 0.79, neutral_prob: 0.16, is_informative: true, text: 'The metropolitan transport authority stated that the bridge remained fully open to commercial and private vehicles on Monday.' },
    ],
  },
  {
    id: 'VX-DEMO-049',
    claim: 'Extraterrestrial spacecraft landed in the central city plaza yesterday evening.',
    verdict: 'INSUFFICIENT_EVIDENCE',
    confidence: 0.39,
    probs: { VERIFIED_TRUE: 0.18, VERIFIED_FALSE: 0.16, CONFLICTING_OR_CHERRYPICKED: 0.12, INSUFFICIENT_EVIDENCE: 0.54 },
    summary: 'Available evidence is insufficient to verify the claim. The selective coverage policy deferred this case.',
    disparity: false,
    abstained: true,
    subclaims: [
      { subclaim_id: 'SC-01', text: 'Extraterrestrial spacecraft landed in the central city plaza', status: 'UNRESOLVED', confidence: 0.39, stance_distribution: { support: 0, refute: 0, neutral: 1 }, relation_to_parent: 'conjunct', explanation: 'No informative documentary evidence found in registries.' },
    ],
    temporal: {
      temporal_scope: 'UNSPECIFIED',
      temporal_status: 'UNRESOLVED',
      explanation: 'Temporal status unresolved: evidence passages lack verifiable timestamps.',
      timeline_events: [],
    },
    uncertainty: {
      shannon_entropy: 1.34,
      top_margin: 0.08,
      coverage_ratio: 0.0,
      is_abstained: true,
      abstention_reason: 'ABSTAINED — INSUFFICIENT EVIDENTIARY GROUNDING',
    },
    evidence: [
      { id: 'E-07', source: 'Municipal Incident Registry', stance: 'neutral', score: 0.22, entailment_prob: 0.08, contradiction_prob: 0.14, neutral_prob: 0.78, is_informative: false, text: 'No municipal records, aviation telemetry, or verified journalistic reports document unconfirmed aerial landings in the city plaza.' },
    ],
  },
  {
    id: 'VX-DEMO-055',
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
    structured_claim: {
      subject: 'incognito mode',
      predicate: 'stop',
      object: 'websites tracking me',
      modality: 'certain',
      is_negated: true,
      negation_scope: 'dont stops websits traking me',
      named_entities: [],
      numeric_entities: [],
    },
    retrieval_query_used: 'incognito mode does not stop websites tracking me',
    nli_hypothesis_used: 'incognitio mdoe dont stops websits traking me',
    verdict: 'VERIFIED_TRUE',
    confidence: 0.92,
    probs: { VERIFIED_TRUE: 0.92, VERIFIED_FALSE: 0.03, CONFLICTING_OR_CHERRYPICKED: 0.01, INSUFFICIENT_EVIDENCE: 0.04 },
    summary: 'Private browsing and incognito mode prevent local browsing history and cookies from persisting after a session ends, but do not prevent external websites or network providers from tracking user IP addresses or fingerprinting.',
    disparity: false,
    abstained: false,
    subclaims: [
      { subclaim_id: 'SC-01', text: 'Incognito mode does not prevent external websites from tracking users', status: 'SUPPORTED', confidence: 0.93, stance_distribution: { support: 2, refute: 0, neutral: 0 }, relation_to_parent: 'conjunct', explanation: 'Directly confirmed by browser security specifications.' },
    ],
    temporal: {
      temporal_scope: 'CURRENT',
      temporal_status: 'CONSISTENT',
      explanation: 'Modern browser network architecture is standard and current.',
      timeline_events: [
        { evidence_id: 'E-10', date: '2023-01-15', date_type: 'publication', stance: 'support', text: 'Private browsing and incognito mode prevent local browsing history and cookies from persisting after a session ends, but do not prevent external websites or network providers from tracking the user via IP addresses or browser fingerprinting.', provenance: 'Browser Privacy Whitepaper' },
        { evidence_id: 'E-11', date: '2023-08-20', date_type: 'publication', stance: 'support', text: 'Incognito mode is designed strictly for local device privacy; it does not cloak user activity from web servers, analytics trackers, or ISPs.', provenance: 'Electronic Privacy Information Center' },
      ],
    },
    uncertainty: {
      shannon_entropy: 0.31,
      top_margin: 0.88,
      coverage_ratio: 1.0,
      is_abstained: false,
    },
    evidence: [
      { id: 'E-10', source: 'Browser Privacy Whitepaper', stance: 'support', score: 0.93, entailment_prob: 0.93, contradiction_prob: 0.02, neutral_prob: 0.05, is_informative: true, text: 'Private browsing and incognito mode prevent local browsing history and cookies from persisting after a session ends, but do not prevent external websites or network providers from tracking the user via IP addresses or browser fingerprinting.' },
      { id: 'E-11', source: 'Electronic Privacy Information Center', stance: 'support', score: 0.89, entailment_prob: 0.89, contradiction_prob: 0.03, neutral_prob: 0.08, is_informative: true, text: 'Incognito mode is designed strictly for local device privacy; it does not cloak user activity from web servers, analytics trackers, or ISPs.' },
    ],
  },
];

function buildFixtureTrace(c) {
  const original = c.original_claim || c.claim || '';
  const sanitized = c.sanitized_claim || original;
  const normalized = c.normalized_claim || sanitized;
  const normStatus = c.normalization_status || 'accepted';
  const corrections = c.normalization_corrections || [];
  const struct = c.structured_claim || {
    subject: c.subclaims?.[0]?.text?.split(' ')?.[0] || 'Subject',
    predicate: 'asserts',
    object: c.subclaims?.[0]?.text || c.claim,
    modality: 'certain',
    is_negated: false,
    named_entities: [],
    numeric_entities: [],
  };

  const words = sanitized.split(/\s+/).filter(Boolean);

  return [
    {
      sequence: 1,
      stage: 'input_received',
      status: 'completed',
      summary: `Received input claim of ${original.length} character(s).`,
      duration_ms: 0.1,
      input_ref: original,
      output_ref: original,
      details: { raw_claim: original, char_length: original.length, raw_word_count: words.length },
    },
    {
      sequence: 2,
      stage: 'sanitization',
      status: 'completed',
      summary: 'Sanitized input text: normalized Unicode, collapsed whitespace.',
      duration_ms: 0.2,
      input_ref: original,
      output_ref: sanitized,
      details: {
        sanitized_claim: sanitized,
        transformations: [{ name: 'whitespace_collapse', description: 'Normalized redundant spacing' }],
        is_question: sanitized.trim().endsWith('?'),
        has_negation: Boolean(struct.is_negated),
      },
    },
    {
      sequence: 3,
      stage: 'spelling_normalization',
      status: normStatus === 'accepted' ? 'completed' : normStatus === 'uncertain' ? 'uncertain' : 'fallback',
      summary: `Spelling normalization identified ${corrections.length} word correction(s).`,
      duration_ms: 0.8,
      input_ref: sanitized,
      output_ref: normalized,
      details: {
        normalized_claim: normalized,
        corrections: corrections,
        word_correction_events: corrections.map((cr) => ({
          original: cr.original,
          replacement: cr.replacement,
          status: cr.status || 'accepted',
          category: cr.category || 'typo',
          confidence: cr.confidence || 0.95,
          edit_distance: cr.edit_distance || 1,
          start_char: cr.start_char || 0,
          end_char: cr.end_char || cr.original.length,
          reason: cr.reason || null,
          candidates: cr.candidates || [cr.replacement],
        })),
        warnings: c.normalization_warnings || [],
        has_ambiguous_corrections: normStatus === 'uncertain',
      },
    },
    {
      sequence: 4,
      stage: 'meaning_preservation',
      status: normStatus === 'accepted' ? 'completed' : 'fallback',
      summary: normStatus === 'accepted'
        ? 'Meaning preservation guardrails verified: all 8 semantic invariants preserved.'
        : 'Meaning preservation guardrails enforced safety fallback.',
      duration_ms: 0.4,
      input_ref: normalized,
      output_ref: normStatus === 'accepted' ? normalized : sanitized,
      details: {
        is_safe: normStatus === 'accepted',
        decision: normStatus === 'accepted' ? 'USE_NORMALIZED' : 'REVERT_TO_ORIGINAL',
        invariants_passed: {
          negation_and_polarity: true,
          quantifier_and_scope: true,
          question_versus_assertion: true,
          entities_numbers_and_units: true,
          modality_and_uncertainty: true,
          causality_and_conditionals: true,
          proposition_count_and_relations: true,
          ambiguity_and_references: normStatus !== 'uncertain',
        },
        violations: normStatus === 'reverted' ? ['Potential entity modification detected'] : [],
        warnings: normStatus === 'uncertain' ? ['Ambiguous correction candidate ranking'] : [],
        rationale: 'Guardrail evaluated meaning invariance between original and proposed normalized claim.',
      },
    },
    {
      sequence: 5,
      stage: 'tokenization',
      status: 'completed',
      summary: `Tokenized canonical claim into ${words.length} token(s) with boundary offsets.`,
      duration_ms: 0.3,
      input_ref: sanitized,
      output_ref: `${words.length} tokens`,
      details: {
        token_count: words.length,
        tokens: words.map((w, idx) => {
          const start = sanitized.indexOf(w);
          const end = start >= 0 ? start + w.length : (idx * 6) + w.length;
          const upos = idx === 0 ? 'NOUN' : idx === 1 ? 'VERB' : idx === 2 ? 'PART' : 'NOUN';
          return {
            text: w,
            lemma: w.toLowerCase().replace(/s$/, ''),
            pos: upos === 'VERB' ? 'VBZ' : upos === 'PART' ? 'RB' : 'NN',
            upos: upos,
            start_char: start >= 0 ? start : idx * 6,
            end_char: end,
          };
        }),
      },
    },
    {
      sequence: 6,
      stage: 'linguistic_analysis',
      status: 'completed',
      summary: 'Linguistic analysis: type=assertion, 1 clause(s), passive=false.',
      duration_ms: 0.3,
      input_ref: sanitized,
      output_ref: 'type:assertion',
      details: {
        sentence_type: 'assertion',
        is_question: sanitized.trim().endsWith('?'),
        is_passive: false,
        subject_candidate: struct.subject,
        predicate_candidate: struct.predicate,
        object_candidate: struct.object,
        clauses: [
          {
            text: sanitized,
            clause_type: 'independent',
            subject: struct.subject,
            predicate: struct.predicate,
            object: struct.object,
            is_negated: struct.is_negated,
            modality: struct.modality || 'certain',
          },
        ],
        parser_engine: 'shallow_syntactic_rule_parser',
      },
    },
    {
      sequence: 7,
      stage: 'claim_extraction',
      status: 'completed',
      summary: `Extracted claim structure: S='${struct.subject}', P='${struct.predicate}', O='${struct.object}'.`,
      duration_ms: 0.3,
      input_ref: sanitized,
      output_ref: `Modality:${struct.modality || 'certain'}`,
      details: {
        structured_claim: struct,
        has_negation: struct.is_negated,
        negation_scope: struct.negation_scope || null,
        modality: struct.modality || 'certain',
        is_conditional: false,
        is_causal: false,
        quantifiers: [],
        named_entities: struct.named_entities || [],
        numeric_entities: struct.numeric_entities || [],
      },
    },
    {
      sequence: 8,
      stage: 'retrieval',
      status: 'completed',
      summary: `Retrieved ${c.evidence?.length || 2} evidence passage(s) via local_corpus.`,
      duration_ms: 4.2,
      input_ref: normStatus === 'accepted' ? normalized : sanitized,
      output_ref: `${c.evidence?.length || 2} passages`,
      details: {
        retrieval_query: normStatus === 'accepted' ? normalized : sanitized,
        query_variant: normStatus === 'accepted' ? 'normalized' : 'sanitized_fallback',
        normalization_status: normStatus,
        retrieval_mode: 'local',
        provider: 'local_corpus',
        systems_run: ['bm25', 'dense'],
        results_count: c.evidence?.length || 2,
        duration_ms: 4.2,
        status: 'success',
        error: null,
      },
    },
    {
      sequence: 9,
      stage: 'evidence_processing',
      status: 'completed',
      summary: `Prepared ${c.evidence?.length || 2} candidate passage(s) for cross-encoder inference.`,
      duration_ms: 0.2,
      input_ref: `${c.evidence?.length || 2} candidates`,
      output_ref: `${c.evidence?.length || 2} ready`,
      details: { candidate_passages_count: c.evidence?.length || 2, retrieval_mode: 'local' },
    },
    {
      sequence: 10,
      stage: 'nli_verification',
      status: 'completed',
      summary: `NLI cross-encoder evaluated hypothesis against ${c.evidence?.length || 2} evidence passage(s).`,
      duration_ms: 8.5,
      input_ref: `hypothesis: '${sanitized}'`,
      output_ref: `raw_verdict: ${c.verdict}`,
      details: {
        hypothesis_used: sanitized,
        hypothesis_source: 'canonical_sanitized_user_claim',
        model: 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
        device: 'cpu',
        calibration_temperature: 4.1039,
        candidate_count: c.evidence?.length || 2,
        inference_ms: 8.5,
        raw_prediction: c.verdict,
      },
    },
    {
      sequence: 11,
      stage: 'final_result',
      status: 'completed',
      summary: `Final adjudication complete: ${c.verdict} (Confidence: ${(c.confidence * 100).toFixed(1)}%).`,
      duration_ms: 0.5,
      input_ref: `raw_prediction:${c.verdict}`,
      output_ref: c.verdict,
      details: {
        verdict: c.verdict,
        confidence: c.confidence,
        probabilities: c.probs,
        abstained: Boolean(c.abstained),
        abstention_reason: c.uncertainty?.abstention_reason || '',
        has_cross_passage_disparity: Boolean(c.disparity),
        evidence_counts: {
          total: c.evidence?.length || 2,
          eligible_supporting: c.evidence?.filter((e) => e.stance === 'support').length || 0,
          eligible_refuting: c.evidence?.filter((e) => e.stance === 'refute').length || 0,
          relevant_neutral: c.evidence?.filter((e) => e.stance === 'neutral').length || 0,
          off_topic: 0,
          duplicates: 0,
        },
      },
    },
  ];
}

function formatPct(num) {
  return `${(Number(num || 0) * 100).toFixed(1)}%`;
}

function friendlyVerdict(value) {
  return LABEL_DISPLAY[value] || String(value || 'Pending Review').replaceAll('_', ' ');
}

export default function App() {
  const [activePage, setActivePage] = useState('investigate');
  const [mode, setMode] = useState(import.meta.env.VITE_DEMO_MODE === 'true' ? 'demo' : 'live');
  const [evidenceMode, setEvidenceMode] = useState('local'); // 'local' | 'web' | 'hybrid'
  const [claim, setClaim] = useState('');
  const [evidenceText, setEvidenceText] = useState('');
  const [result, setResult] = useState(null);
  const [roundTripMs, setRoundTripMs] = useState(null);
  const [selectedFixture, setSelectedFixture] = useState(FIXTURE_CASES[0]);
  const [busy, setBusy] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState('');
  const [showEvidenceInput, setShowEvidenceInput] = useState(false);
  const [showBenchmarkFixtures, setShowBenchmarkFixtures] = useState(false);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [toast, setToast] = useState('');
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [history, setHistory] = useState([]);
  const [apiHealth, setApiHealth] = useState({ online: false, device: 'checking...', modelReady: false, modelLoading: false, warmupState: 'pending' });

  const progressRef = useRef(null);
  const resultsRef = useRef(null);
  const errorRef = useRef(null);
  const abortControllerRef = useRef(null);
  const requestIdRef = useRef(0);

  function handleNewInvestigation() {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    requestIdRef.current += 1;
    setClaim('');
    setEvidenceText('');
    setResult(null);
    setError('');
    setBusy(false);
    navigate('investigate');
  }

  function scrollToTarget(ref) {
    if (!ref || !ref.current) return;
    const prefersReduced = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    ref.current.scrollIntoView({
      behavior: prefersReduced ? 'auto' : 'smooth',
      block: 'start',
    });
  }

  // Elapsed timer during analysis
  useEffect(() => {
    let timer;
    if (busy) {
      setElapsedSeconds(0);
      timer = setInterval(() => {
        setElapsedSeconds((prev) => +(prev + 0.1).toFixed(1));
      }, 100);
    } else {
      setElapsedSeconds(0);
    }
    return () => clearInterval(timer);
  }, [busy]);

  // Periodically check live API health
  useEffect(() => {
    let active = true;
    const checkHealth = async () => {
      try {
        const res = await fetch(`${apiBase}/api/health`);
        if (res.ok) {
          const data = await res.json();
          if (active) {
            setApiHealth({
              online: true,
              device: data.device || 'active',
              modelReady: Boolean(data.model_ready),
              modelLoading: Boolean(data.model_loading),
              warmupState: data.warmup_state || 'ready',
              initLatencyMs: data.init_latency_ms,
            });
          }
        } else {
          if (active) setApiHealth({ online: false, device: 'offline', modelReady: false, modelLoading: false, warmupState: 'offline' });
        }
      } catch {
        if (active) setApiHealth({ online: false, device: 'offline', modelReady: false, modelLoading: false, warmupState: 'offline' });
      }
    };

    checkHealth();
    const timer = setInterval(checkHealth, 8000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const navItems = [
    {
      group: 'Core Verification',
      items: [
        { id: 'investigate', label: 'Investigate', icon: 'scan', shortcut: '01' },
        { id: 'evidence', label: 'Evidence', icon: 'files', shortcut: '02' },
        { id: 'graph', label: 'Graphs', icon: 'nodes', shortcut: '03' },
        { id: 'timeline', label: 'History & Timeline', icon: 'clock', shortcut: '04' },
      ],
    },
    {
      group: 'Advanced Tools & System',
      items: [
        { id: 'multihop', label: 'Multi-Hop Reasoning', icon: 'git-merge', shortcut: '05' },
        { id: 'overview', label: 'Command Center', icon: 'grid', shortcut: '06' },
        { id: 'evaluation', label: 'System & Evaluation', icon: 'chart', shortcut: '07' },
        { id: 'system', label: 'System Telemetry', icon: 'sliders', shortcut: '08' },
      ],
    },
  ];

  function navigate(page) {
    setActivePage(page);
    setMobileNavOpen(false);
    setError('');
  }

  function showToast(message) {
    setToast(message);
    setTimeout(() => setToast(''), 2600);
  }

  function copySessionReport() {
    if (history.length === 0) {
      showToast('No investigations recorded in this session.');
      return;
    }
    const report = `# VERITAS-X Session Evidence Report\nDate: ${new Date().toISOString()}\nTotal Inquiries: ${history.length}\n\n` +
      history.map((h, i) => `### ${i + 1}. [${h.id}] ${h.claim}\n- Verdict: ${h.verdict}\n- Mode: ${h.mode}\n- Time: ${h.time}\n`).join('\n');
    navigator.clipboard.writeText(report);
    showToast('Forensic session report copied to clipboard.');
  }

  function openFixture(fixture, autoSubmit = false) {
    if (!fixture) return;
    setSelectedFixture(fixture);
    setClaim(fixture.claim || '');
    const premiseTexts = (fixture.evidence || []).map((e) => e.text).filter(Boolean).join('\n');
    setEvidenceText(premiseTexts);
    setShowEvidenceInput(true);
    setResult(null); // CRITICAL: Clear previous stale investigation result so old verdict does not remain
    setError('');
    setActivePage('investigate');
    showToast(`Loaded ${fixture.id || 'case'}: "${(fixture.claim || '').slice(0, 38)}..." (${fixture.evidence?.length || 0} premise passages).`);

    if (autoSubmit) {
      setTimeout(() => {
        runInvestigation(fixture.claim, premiseTexts);
      }, 50);
    } else {
      setTimeout(() => {
        const target = progressRef.current || resultsRef.current;
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    }
  }

  async function runInvestigation(overrideClaim, overrideEvidence) {
    const value = (overrideClaim !== undefined ? overrideClaim : claim).trim();
    const currentEvidence = overrideEvidence !== undefined ? overrideEvidence : evidenceText;

    if (!value) {
      setError('Please enter a statement you want to fact-check before verifying.');
      setResult(null);
      setTimeout(() => scrollToTarget(errorRef), 50);
      return;
    }

    // Abort previous in-flight request if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    const currentRequestId = ++requestIdRef.current;
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setError('');
    setBusy(true);
    setResult(null); // CRITICAL: Clear old result immediately so previous verdict is never left displayed
    setRoundTripMs(null);

    const startTime = performance.now();

    setTimeout(() => {
      scrollToTarget(progressRef);
    }, 60);

    try {
      if (mode === 'demo') {
        const lower = value.toLowerCase();
        let fixture = selectedFixture;
        if (lower.includes('incognito') || lower.includes('incognitio') || lower.includes('mdoe') || lower.includes('traking') || lower.includes('websits')) fixture = FIXTURE_CASES[4];
        else if (lower.includes('moon') || lower.includes('star')) fixture = FIXTURE_CASES[1];
        else if (lower.includes('bridge') || lower.includes('traffic') || lower.includes('closed')) fixture = FIXTURE_CASES[2];
        else if (lower.includes('alien') || lower.includes('spacecraft') || lower.includes('plaza')) fixture = FIXTURE_CASES[3];
        else if (lower.includes('water') || lower.includes('freeze') || lower.includes('celsius')) fixture = FIXTURE_CASES[0];

        await new Promise((r) => setTimeout(r, 600));

        // Check if this request was superseded while waiting
        if (currentRequestId !== requestIdRef.current || controller.signal.aborted) {
          return;
        }

        const elapsed = Math.round(performance.now() - startTime);
        const demonstration = {
          ...fixture,
          claim: value,
          original_claim: fixture.original_claim || value,
          sanitized_claim: fixture.sanitized_claim || value,
          normalized_claim: fixture.normalized_claim || value,
          normalization_status: fixture.normalization_status || 'accepted',
          normalization_corrections: fixture.normalization_corrections || [],
          normalization_warnings: fixture.normalization_warnings || [],
          structured_claim: fixture.structured_claim || {},
          retrieval_query_used: fixture.retrieval_query_used || fixture.normalized_claim || value,
          nli_hypothesis_used: fixture.nli_hypothesis_used || fixture.sanitized_claim || value,
          processing_trace: fixture.processing_trace || buildFixtureTrace(fixture),
          fixture: true,
        };
        setRoundTripMs(elapsed);
        setResult(demonstration);
        setHistory((old) => [
          { id: demonstration.id, claim: value, verdict: demonstration.verdict, time: 'Just now', mode: 'Illustrative Demo', raw: demonstration },
          ...old,
        ].slice(0, 16));

        setTimeout(() => {
          scrollToTarget(resultsRef);
        }, 120);
      } else {
        const providedEvidence = currentEvidence.split('\n').map((s) => s.trim()).filter(Boolean);
        const tFetchStart = performance.now();
        const raw = await verifyClaim(value, providedEvidence, evidenceMode, controller.signal);
        const tFetchEnd = performance.now();
        const clientApiWait = Math.round(tFetchEnd - tFetchStart);

        // Check if this request was superseded while fetching
        if (currentRequestId !== requestIdRef.current || controller.signal.aborted) {
          return;
        }

        const normalized = normalizeResponse(raw, value);
        if (!normalized.verdict) {
          throw new Error('API returned malformed payload without verdict key.');
        }

        const elapsed = Math.round(tFetchEnd - startTime);
        normalized.round_trip_ms = elapsed;
        normalized.client_api_wait_ms = clientApiWait;

        setRoundTripMs(elapsed);
        setResult(normalized);
        setHistory((old) => [
          { id: normalized.id, claim: value, verdict: normalized.verdict, time: 'Just now', mode: `Live (${(normalized.mode || evidenceMode).toUpperCase()})`, raw: normalized },
          ...old,
        ].slice(0, 16));

        // Measure DOM paint duration accurately
        requestAnimationFrame(() => {
          const tRenderEnd = performance.now();
          const renderMs = Math.max(1, Math.round(tRenderEnd - tFetchEnd));
          setResult((cur) => (cur ? { ...cur, client_render_ms: renderMs } : cur));
        });

        setTimeout(() => {
          scrollToTarget(resultsRef);
        }, 120);
      }
    } catch (e) {
      if (e.name === 'AbortError' || controller.signal.aborted) {
        // Superseded by newer request, silently exit
        return;
      }
      setResult(null); // CRITICAL: Ensure previous verdict is never left displayed on error
      setError(`${e.message}. Live engine connection failed. Please ensure the backend server is running on http://127.0.0.1:8011.`);
      setTimeout(() => {
        scrollToTarget(errorRef);
      }, 80);
    } finally {
      if (currentRequestId === requestIdRef.current) {
        setBusy(false);
      }
    }
  }

  return (
    <div className="app-shell">
      {/* Main Forensic Sidebar */}
      <aside className={`sidebar ${mobileNavOpen ? 'sidebar-open' : ''}`}>
        <div className="brand-header">
          <VeritasWordmark size="md" theme="indigo" tagline="Explainable AI Fact Verification" />
          <button
            className="mobile-close"
            onClick={() => setMobileNavOpen(false)}
            aria-label="Close Navigation"
          >
            <Icon name="close" size={18} />
          </button>
        </div>

        {/* New Investigation Quick Action */}
        <div style={{ padding: '0.85rem 1rem 0.4rem' }}>
          <button
            className="primary-action-btn"
            style={{ width: '100%', justifyContent: 'center', gap: '0.5rem', padding: '0.65rem 1rem' }}
            onClick={handleNewInvestigation}
            id="sidebar-new-claim-btn"
          >
            <Icon name="plus" size={15} />
            <span>New Investigation</span>
          </button>
        </div>

        {/* Sidebar Nav */}
        <nav className="sidebar-nav">
          {navItems.map((group) => (
            <div className="nav-group" key={group.group}>
              <div className="nav-heading">{group.group}</div>
              {group.items.map((item) => (
                <button
                  key={item.id}
                  className={`nav-btn ${activePage === item.id ? 'active' : ''}`}
                  onClick={() => navigate(item.id)}
                  id={`nav-link-${item.id}`}
                >
                  <Icon name={item.icon} size={16} />
                  <span>{item.label}</span>
                  <span className="nav-shortcut">{item.shortcut}</span>
                </button>
              ))}
            </div>
          ))}

          {/* Real Session Inquiries History */}
          {history.length > 0 && (
            <div className="nav-group" style={{ marginTop: '0.75rem' }}>
              <div className="nav-heading">Recent Inquiries ({history.length})</div>
              {history.slice(0, 5).map((h) => (
                <button
                  key={h.id}
                  className="nav-btn session-history-btn"
                  onClick={() => {
                    setClaim(h.claim);
                    if (h.raw) setResult(h.raw);
                    navigate('investigate');
                  }}
                  title={h.claim}
                  style={{ fontSize: '0.78rem', padding: '0.45rem 0.75rem', display: 'flex', alignItems: 'center', gap: '0.45rem', width: '100%' }}
                >
                  <span className={`verdict-dot dot-${LABEL_CLASS[h.verdict] || 'neutral'}`} style={{ width: 7, height: 7, borderRadius: '50%', flexShrink: 0 }} />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{h.claim}</span>
                </button>
              ))}
            </div>
          )}
        </nav>

        {/* Engine Status Card at Bottom */}
        <div className="sidebar-bottom">
          <div className="engine-status-box">
            <div className="status-box-head">
              <span className={`live-pulse-dot ${apiHealth.online ? (apiHealth.modelReady ? 'dot-online' : 'dot-warming') : 'dot-offline'}`} />
              <span className="box-title">{apiHealth.online ? (apiHealth.modelReady ? 'Engine Ready' : 'Model Warming') : 'Engine Standby'}</span>
              <span className={`status-pill ${mode === 'demo' ? 'pill-demo' : 'pill-live'}`}>
                {mode === 'demo' ? 'DEMO' : 'LIVE'}
              </span>
            </div>
            <div className="engine-model-title">DeBERTa-v3 &bull; MNLI/FEVER</div>
            <div className="engine-meta-line">
              <span>Device: {apiHealth.online ? apiHealth.device.toUpperCase() : 'STANDBY'}</span>
              <span>{apiHealth.modelReady ? 'WARMED' : (apiHealth.online ? 'WARMING...' : 'OFFLINE')}</span>
            </div>
          </div>

          <div className="sidebar-foot-links">
            <button className="foot-link-btn" onClick={() => navigate('system')}>
              <Icon name="help" size={14} />
              <span>System Specs</span>
            </button>
            <div className="foot-version">BUILD 2026.10</div>
          </div>
        </div>
      </aside>

      {mobileNavOpen && <div className="scrim" onClick={() => setMobileNavOpen(false)} />}

      {/* Main Content Area */}
      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="mobile-menu" onClick={() => setMobileNavOpen(true)} aria-label="Open Navigation">
              <Icon name="menu" size={20} />
            </button>
            <div className="breadcrumb-trail">
              <span className="crumb-root" onClick={() => navigate('investigate')} style={{ cursor: 'pointer' }} title="Return to Claim Investigator">VERITAS-X</span>
              <span className="crumb-slash">/</span>
              <span className="crumb-current">
                {({
                  investigate: 'Investigate',
                  evidence: 'Evidence Dossier',
                  graph: 'Graphs',
                  timeline: 'History & Timeline',
                  multihop: 'Multi-Hop Reasoning',
                  overview: 'Command Center',
                  evaluation: 'System & Evaluation',
                  system: 'System Telemetry',
                })[activePage]}
              </span>
            </div>
          </div>

          <div className="topbar-right">
            {/* Live API / Demo Mode Selector */}
            <div className="mode-segmented-control">
              <button
                className={`seg-btn ${mode === 'demo' ? 'seg-active' : ''}`}
                onClick={() => { setMode('demo'); setError(''); }}
                id="mode-demo-btn"
                title="Use illustrative demo fixtures without calling the model"
              >
                <Icon name="eye" size={13} />
                <span>Demo Mode</span>
              </button>
              <button
                className={`seg-btn ${mode === 'live' ? 'seg-active' : ''}`}
                onClick={() => { setMode('live'); setResult(null); setError(''); }}
                id="mode-live-btn"
                title="Connect to the local live verification API on port 8011"
              >
                <Icon name="zap" size={13} />
                <span>Live API</span>
              </button>
            </div>

            {/* API Health Chip */}
            <div className="api-health-chip" title={`Endpoint: ${apiBase}/api/verify (State: ${apiHealth.warmupState || 'ready'})`}>
              <span className={`pulse-health ${apiHealth.online ? (apiHealth.modelReady ? 'health-ok' : 'health-warming') : 'health-warn'}`} />
              <span className="health-label">
                {!apiHealth.online ? 'API OFFLINE' : (apiHealth.modelReady ? 'ENGINE READY' : 'MODEL WARMING')}
              </span>
            </div>
          </div>
        </header>

        {/* Demo Mode Notice Banner */}
        {mode === 'demo' && (
          <div className="demo-notice-strip">
            <div className="notice-icon"><Icon name="alert" size={16} /></div>
            <div className="notice-text">
              <strong>Demonstration Mode Active:</strong> Output relies on illustrative fixtures. These are not real model predictions or benchmark measurements.
            </div>
            <button className="notice-switch-btn" onClick={() => { setMode('live'); setError(''); }}>
              Connect Live Engine <Icon name="arrow" size={13} />
            </button>
          </div>
        )}

        {/* Page Content Router */}
        <div className="page-content-wrapper">
          {/* ================= PAGE 1: COMMAND CENTER ================= */}
          {activePage === 'overview' && (
            <div className="page-section overview-page">
              <div className="page-title-row">
                <div>
                  <div className="section-eyebrow">
                    <span className="eyebrow-accent" />
                    FORENSIC INTELLIGENCE COMMAND
                  </div>
                  <h1 className="page-headline">Command <span>Center.</span></h1>
                  <p className="page-subline">Real-time claim surveillance, dual-modal evidence retrieval, and uncertainty calibration.</p>
                </div>

                <div className="page-title-actions">
                  <button className="secondary-action-btn" onClick={() => navigate('investigate')}>
                    ← Back to Claim Investigator
                  </button>
                  <button className="primary-action-btn" onClick={() => { setClaim(''); setResult(null); navigate('investigate'); }}>
                    <Icon name="plus" size={16} />
                    <span>New Investigation</span>
                  </button>
                </div>
              </div>

              {/* Command Center Hero Briefing (Replaced empty canvas) */}
              <CommandCenterHero
                onStartInvestigation={() => { setClaim(''); setResult(null); navigate('investigate'); }}
                onQuickLoadClaim={(demoClaim) => {
                  setClaim(demoClaim);
                  navigate('investigate');
                }}
                apiHealth={apiHealth}
                mode={mode}
                historyCount={history.length}
              />

              {/* Audited Forensic Snapshot Header */}
              <div className="metrics-section-title">
                <div className="title-with-pill">
                  <h3>Audited Forensic Snapshot</h3>
                  <span className="phase-pill">PHASE 2 VALIDATION AUDIT (N=255)</span>
                </div>
                <button className="subtle-link-btn" onClick={() => navigate('evaluation')}>
                  <span>Full Performance Lab</span>
                  <Icon name="arrow" size={13} />
                </button>
              </div>

              {/* KPI Quad Grid */}
              <div className="hud-metric-grid">
                <article className="hud-metric-card">
                  <div className="card-top">
                    <span className="metric-label">VALIDATION ACCURACY</span>
                    <span className="card-icon"><Icon name="shield" size={16} /></span>
                  </div>
                  <div className="card-main-val">59.22%</div>
                  <div className="card-footer">
                    <span className="val-detail">151 of 255 claims exact match</span>
                    <span className="val-delta positive">Macro-P: 49.91%</span>
                  </div>
                </article>

                <article className="hud-metric-card">
                  <div className="card-top">
                    <span className="metric-label">MACRO-F1</span>
                    <span className="card-icon"><Icon name="chart" size={16} /></span>
                  </div>
                  <div className="card-main-val">44.46%</div>
                  <div className="card-footer">
                    <span className="val-detail">Balanced across 4 classes</span>
                    <span className="val-delta positive">Macro-R: 50.92%</span>
                  </div>
                </article>

                <article className="hud-metric-card">
                  <div className="card-top">
                    <span className="metric-label">SELECTIVE ACCURACY</span>
                    <span className="card-icon"><Icon name="scan" size={16} /></span>
                  </div>
                  <div className="card-main-val text-teal">61.57%</div>
                  <div className="card-footer">
                    <span className="val-detail">Coverage: 94.90% (242 retained)</span>
                    <span className="val-delta positive">+2.35 pp vs full split</span>
                  </div>
                </article>

                <article className="hud-metric-card">
                  <div className="card-top">
                    <span className="metric-label">CALIBRATED ECE</span>
                    <span className="card-icon"><Icon name="sliders" size={16} /></span>
                  </div>
                  <div className="card-main-val text-amber">11.83%</div>
                  <div className="card-footer">
                    <span className="val-detail">Post-scaling with T=4.1039</span>
                    <span className="val-delta positive">-7.38 pp error reduction</span>
                  </div>
                </article>
              </div>

              {/* Lower Section: Class F1 + Recent Investigations */}
              <div className="overview-duo-columns">
                <div className="hud-panel class-performance-panel">
                  <div className="hud-panel-head">
                    <div>
                      <h4>Class-Wise F1 Breakdown</h4>
                      <p>Authorized validation split &bull; N=255</p>
                    </div>
                    <span className="hud-tag-box">VALIDATED</span>
                  </div>

                  <div className="benchmark-bars-stack">
                    {BENCHMARK_BARS.map((bar) => (
                      <div className="benchmark-bar-row" key={bar.label}>
                        <span className="bar-class-name">{bar.label}</span>
                        <div className="bar-track">
                          <div className="bar-fill" style={{ width: `${bar.value}%`, background: bar.color }} />
                        </div>
                        <span className="bar-val-text">{bar.value.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>

                  <div className="forensic-limitation-note">
                    <Icon name="alert" size={16} />
                    <div>
                      <strong>Audited System Limitation:</strong> Conflicting-evidence recall is 10.53% (2 of 19 validation claims identified). Conservative thresholds prioritize precision to prevent false-alarm conflict declarations in web discourse.
                    </div>
                  </div>
                </div>

                <div className="hud-panel activity-stream-panel">
                  <div className="hud-panel-head">
                    <div>
                      <h4>Recent Investigation Stream</h4>
                      <p>Session activity &bull; {history.length} record(s)</p>
                    </div>
                    {history.length > 0 && (
                      <button className="subtle-clear-btn" onClick={() => { setHistory([]); showToast('Session logs cleared.'); }}>
                        Clear
                      </button>
                    )}
                  </div>

                  {history.length === 0 ? (
                    <div className="empty-stream-box">
                      <div className="empty-stream-icon"><Icon name="database" size={24} /></div>
                      <h5>No Inquiries Executed Yet</h5>
                      <p>Claims verified in this session will populate this real-time forensic activity ledger.</p>
                      <button className="btn-secondary-sm" onClick={() => navigate('investigate')}>
                        Open Investigator <Icon name="arrow" size={13} />
                      </button>
                    </div>
                  ) : (
                    <div className="history-stream-list">
                      {history.slice(0, 5).map((item, i) => (
                        <div
                          key={`${item.id}-${i}`}
                          className="history-stream-row"
                          onClick={() => {
                            setClaim(item.claim);
                            setResult(item.raw || null);
                            navigate('investigate');
                          }}
                        >
                          <span className="stream-index">0{i + 1}</span>
                          <div className="stream-body">
                            <strong className="stream-claim-text">{item.claim}</strong>
                            <div className="stream-meta">
                              <span>{item.id}</span>
                              <span>&bull;</span>
                              <span>{item.mode}</span>
                              <span>&bull;</span>
                              <span>{item.time}</span>
                            </div>
                          </div>
                          <span className={`verdict-pill verdict-${LABEL_CLASS[item.verdict] || 'neutral'}`}>
                            {friendlyVerdict(item.verdict)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ================= PAGE 2: INVESTIGATE CLAIM ================= */}
          {activePage === 'investigate' && (
            <div className="page-section investigate-page">
              {/* Compact Clean Hero Heading */}
              <div className="claim-hero-section">
                <h1 className="claim-hero-title">What statement do you want to check?</h1>
                <p className="claim-hero-desc">
                  Enter any factual claim to verify it against authentic documentary evidence.
                </p>
              </div>

              {/* Clean Single Composer Card */}
              <div className="composer-container-card">
                {/* 3-Way Verification Source Selector */}
                <div className="evidence-mode-selector-bar">
                  <span className="mode-selector-label">VERIFICATION SOURCE:</span>
                  <div className="mode-pills-row">
                    <button
                      type="button"
                      className={`mode-pill-tab ${evidenceMode === 'local' ? 'active' : ''}`}
                      onClick={() => setEvidenceMode('local')}
                      id="mode-select-local"
                      title="Uses evidence available in the local collection. Works offline."
                    >
                      <Icon name="database" size={14} />
                      <span className="pill-text-title">Local</span>
                      <span className="pill-badge">Offline</span>
                    </button>

                    <button
                      type="button"
                      className={`mode-pill-tab ${evidenceMode === 'web' ? 'active' : ''}`}
                      onClick={() => setEvidenceMode('web')}
                      id="mode-select-web"
                      title="Searches online sources when the service is configured and reachable."
                    >
                      <Icon name="globe" size={14} />
                      <span className="pill-text-title">Web</span>
                      <span className="pill-badge">Online</span>
                    </button>

                    <button
                      type="button"
                      className={`mode-pill-tab ${evidenceMode === 'hybrid' ? 'active' : ''}`}
                      onClick={() => setEvidenceMode('hybrid')}
                      id="mode-select-hybrid"
                      title="Uses local and online evidence when both are available."
                    >
                      <Icon name="nodes" size={14} />
                      <span className="pill-text-title">Combined</span>
                      <span className="pill-badge">Local + Web</span>
                    </button>
                  </div>
                </div>

                {/* Light Mode 1-Sentence Source Note */}
                <div className="clean-mode-note">
                  <Icon name={evidenceMode === 'local' ? 'database' : evidenceMode === 'web' ? 'globe' : 'nodes'} size={13} />
                  <span>
                    <strong>{evidenceMode === 'local' ? 'Local: ' : evidenceMode === 'web' ? 'Web: ' : 'Combined: '}</strong>
                    {evidenceMode === 'local' && 'Uses evidence available in the local collection. Works offline.'}
                    {evidenceMode === 'web' && 'Searches online sources when the service is configured and reachable.'}
                    {evidenceMode === 'hybrid' && 'Uses local and online evidence when both are available.'}
                  </span>
                </div>

                {/* Compact Example Section for the 4 Reference Fixtures */}
                <div className="compact-examples-bar">
                  <span className="examples-bar-label">Try an example:</span>
                  <div className="examples-pills-row">
                    {FIXTURE_CASES.map((item, idx) => (
                      <div key={item.id} className="example-fixture-pill-group">
                        <button
                          type="button"
                          className={`example-pill-btn ${selectedFixture.id === item.id ? 'active' : ''}`}
                          onClick={() => openFixture(item, false)}
                          id={`example-fixture-pill-${idx + 1}`}
                          title={`Load fixture: "${item.claim}"`}
                        >
                          <span className={`example-pill-dot dot-${LABEL_CLASS[item.verdict] || 'neutral'}`} />
                          <span className="example-pill-title">{friendlyVerdict(item.verdict)}</span>
                        </button>
                        <button
                          type="button"
                          className="example-pill-quick-verify"
                          onClick={() => openFixture(item, true)}
                          id={`fixture-verify-case-0${idx + 1}`}
                          title={`Instantly verify "${item.claim}" against supplied premises`}
                        >
                          <Icon name="zap" size={11} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Main Textarea Input */}
                <div className="composer-input-wrapper">
                  <textarea
                    id="claim-textarea"
                    className="clean-claim-composer"
                    value={claim}
                    onChange={(e) => setClaim(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        runInvestigation();
                      }
                    }}
                    placeholder="Humans use only 10% of their brains."
                    rows={3}
                    maxLength={2000}
                  />

                  <div className="composer-footer-row">
                    <div className="composer-footer-left">
                      <button
                        type="button"
                        className="composer-secondary-btn"
                        onClick={() => setShowEvidenceInput(!showEvidenceInput)}
                      >
                        <Icon name="files" size={14} />
                        <span>{showEvidenceInput ? 'Hide Supplied Premises' : '+ Add Custom Premise'}</span>
                      </button>
                      {(claim || evidenceText) && (
                        <button
                          type="button"
                          className="composer-clear-link"
                          onClick={() => { setClaim(''); setEvidenceText(''); setError(''); }}
                        >
                          Clear
                        </button>
                      )}
                      <span className="composer-char-limit">Press Enter or click Verify Claim</span>
                    </div>

                    <button
                      className="composer-submit-btn"
                      onClick={() => runInvestigation()}
                      disabled={busy || !claim.trim()}
                      id="run-investigation-btn"
                    >
                      {busy ? (
                        <>
                          <span className="action-spinner" />
                          <span>Verifying ({evidenceMode.toUpperCase()})...</span>
                        </>
                      ) : (
                        <>
                          <Icon name="scan" size={15} />
                          <span>Verify Claim</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Optional User Evidence Accordion */}
                {showEvidenceInput && (
                  <div className="custom-premise-container">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                      <label className="field-title" htmlFor="evidence-textarea">
                        USER-SUPPLIED PASSAGES (ONE PER LINE)
                      </label>
                      <span style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)' }}>
                        {evidenceText.split('\n').filter(s => s.trim()).length} passage(s) loaded
                      </span>
                    </div>
                    <textarea
                      id="evidence-textarea"
                      className="evidence-input-area"
                      value={evidenceText}
                      onChange={(e) => setEvidenceText(e.target.value)}
                      placeholder="Paste verifiable source text snippet(s), one passage per line..."
                      rows={3}
                    />
                    <p className="field-helper-text">
                      When provided, the engine verifies directly against these candidate premises. If left blank, the retriever searches according to the selected mode ({evidenceMode.toUpperCase()}).
                    </p>
                  </div>
                )}

                {error && (
                  <div ref={errorRef} className="investigation-error-banner" role="alert">
                    <Icon name="alert" size={18} />
                    <div className="error-body-content">
                      <strong>Verification Request Failed:</strong>
                      <p>{error}</p>
                      <button
                        type="button"
                        className="retry-investigation-btn"
                        onClick={() => runInvestigation()}
                      >
                        <Icon name="scan" size={14} />
                        <span>Retry Verification</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Loading Feedback: 5-Stage Plain-English Progress during active analysis */}
              {busy && (
                <div ref={progressRef} className="investigation-active-progress" style={{ marginTop: '1.5rem' }}>
                  <InvestigationProgress
                    active={busy}
                    mode={mode}
                    evidenceMode={evidenceMode}
                    elapsedSeconds={elapsedSeconds}
                  />
                </div>
              )}

              {/* Forensic Investigation Report: Rendered IMMEDIATELY when results are ready */}
              <div ref={resultsRef}>
                {result && !busy && (
                  <>
                    {/* Cinematic Live NLP Understanding Interface (Phase 4B) */}
                    <NLPVisualization
                      result={result}
                    />

                    <ForensicReport
                      result={result}
                      roundTripMs={roundTripMs}
                      onInvestigateRelated={(newClaim) => {
                        setClaim(newClaim);
                        scrollToTarget(progressRef);
                      }}
                    />
                  </>
                )}
              </div>
            </div>
          )}

          {/* ================= PAGE 3: EVIDENCE DOCKET ================= */}
          {activePage === 'evidence' && (
            <div className="page-section evidence-page">
              <div className="page-title-row">
                <div>
                  <div className="section-eyebrow">
                    <span className="eyebrow-accent" />
                    INTELLIGENCE ARCHIVE &amp; DOSSIER
                  </div>
                  <h1 className="page-headline">Evidence <span>Docket.</span></h1>
                  <p className="page-subline">Browse, filter, and compare documentary evidence and citations recorded across active investigations.</p>
                </div>

                <div className="page-title-actions">
                  <button className="secondary-action-btn" onClick={() => navigate('investigate')}>
                    ← Back to Claim Investigator
                  </button>
                  <button className="secondary-action-btn" onClick={copySessionReport}>
                    <Icon name="copy" size={15} />
                    <span>Export Session Report</span>
                  </button>
                  <button className="primary-action-btn" onClick={() => navigate('investigate')}>
                    <Icon name="plus" size={15} />
                    <span>Investigate New Claim</span>
                  </button>
                </div>
              </div>

              {/* Active Result Evidence Dossier */}
              {result?.evidence && result.evidence.length > 0 ? (
                <EvidenceDossier
                  evidence={result.evidence}
                  claim={result.claim}
                  verdict={result.verdict}
                  confidence={result.confidence}
                  mode={result.mode}
                  retrievalStatus={result.retrieval_status}
                  isFixture={Boolean(result.fixture || result.mode === 'user_supplied')}
                />
              ) : (
                <div className="hud-panel empty-docket-card">
                  <div className="empty-stream-box">
                    <div className="empty-stream-icon"><Icon name="files" size={32} /></div>
                    <h4>No Active Evidence Dossier Loaded</h4>
                    <p>Run a verification or load a benchmark fixture to examine passages, stances, and side-by-side comparisons.</p>
                    <button className="primary-action-btn" onClick={() => navigate('investigate')}>
                      <Icon name="scan" size={15} />
                      <span>Open Claim Investigator</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ================= PAGE: EVIDENCE GRAPH ================= */}
          {activePage === 'graph' && (
            <div className="page-section graph-page">
              <div className="page-title-row">
                <div>
                  <div className="section-eyebrow">
                    <span className="eyebrow-accent" />
                    REASONING TOPOLOGY &amp; GRAPH
                  </div>
                  <h1 className="page-headline">Evidence <span>Graph.</span></h1>
                  <p className="page-subline">Interactive multi-tier relationship network linking hypotheses, decomposed subclaims, citations, and publishers.</p>
                </div>

                <div className="page-title-actions">
                  <button className="secondary-action-btn" onClick={() => navigate('investigate')}>
                    ← Back to Claim Investigator
                  </button>
                  <button className="secondary-action-btn" onClick={() => navigate('evidence')}>
                    <Icon name="files" size={15} />
                    <span>Evidence Dossier</span>
                  </button>
                </div>
              </div>

              {result ? (
                <EvidenceGraph
                  claim={result.claim}
                  claimId={result.id}
                  verdict={result.verdict}
                  confidence={result.confidence}
                  subclaims={result.subclaims}
                  evidence={result.evidence}
                  graph={result.graph}
                  onNavigateToDossier={() => navigate('evidence')}
                />
              ) : (
                <div className="hud-panel empty-docket-card">
                  <div className="empty-stream-box">
                    <div className="empty-stream-icon"><Icon name="nodes" size={32} /></div>
                    <h4>No Active Evidence Graph Loaded</h4>
                    <p>Run a verification inquiry or load a benchmark fixture to examine the interactive consensus topology.</p>
                    <button className="primary-action-btn" onClick={() => openFixture(FIXTURE_CASES[0], true)}>
                      <Icon name="scan" size={15} />
                      <span>Load Benchmark Case &amp; Visualize Graph</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ================= PAGE: TEMPORAL EVIDENCE TIMELINE ================= */}
          {activePage === 'timeline' && (
            <div className="page-section timeline-page">
              <div className="page-title-row">
                <div>
                  <div className="section-eyebrow">
                    <span className="eyebrow-accent" />
                    TEMPORAL PROVENANCE &amp; ARCHIVE
                  </div>
                  <h1 className="page-headline">Temporal <span>Timeline.</span></h1>
                  <p className="page-subline">Time-aware chronological reconstruction detecting publication dates, archived snapshots, and outdated evidence.</p>
                </div>

                <div className="page-title-actions">
                  <button className="secondary-action-btn" onClick={() => navigate('investigate')}>
                    ← Back to Claim Investigator
                  </button>
                  <button className="secondary-action-btn" onClick={() => navigate('evidence')}>
                    <Icon name="files" size={15} />
                    <span>Evidence Docket</span>
                  </button>
                </div>
              </div>

              {result ? (
                <TimelineView
                  claim={result.claim}
                  temporal={result.temporal}
                  evidence={result.evidence}
                  onNavigateToDossier={() => navigate('evidence')}
                />
              ) : (
                <div className="hud-panel empty-docket-card">
                  <div className="empty-stream-box">
                    <div className="empty-stream-icon"><Icon name="clock" size={32} /></div>
                    <h4>No Active Timeline Data Loaded</h4>
                    <p>Run a claim verification to extract publication and event dates from cited sources.</p>
                    <button className="primary-action-btn" onClick={() => openFixture(FIXTURE_CASES[2], true)}>
                      <Icon name="scan" size={15} />
                      <span>Load Municipal Bridge Case &amp; Trace Timeline</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ================= PAGE: MULTI-HOP DECOMPOSITION ================= */}
          {activePage === 'multihop' && (
            <div className="page-section multihop-page">
              <div className="page-title-row">
                <div>
                  <div className="section-eyebrow">
                    <span className="eyebrow-accent" />
                    ATOMIC PROPOSITION DECOMPOSITION
                  </div>
                  <h1 className="page-headline">Multi-Hop <span>Reasoning.</span></h1>
                  <p className="page-subline">Deconstruct complex conjunctions, causal claims, and attribution clauses into atomic verifiable subclaims.</p>
                </div>

                <div className="page-title-actions">
                  <button className="secondary-action-btn" onClick={() => navigate('investigate')}>
                    ← Back to Claim Investigator
                  </button>
                  <button className="secondary-action-btn" onClick={() => navigate('graph')}>
                    <Icon name="nodes" size={15} />
                    <span>View Graph</span>
                  </button>
                </div>
              </div>

              {result ? (
                <MultiHopInspector
                  claim={result.claim}
                  subclaims={result.subclaims}
                  evidence={result.evidence}
                  onNavigateToDossier={() => navigate('evidence')}
                />
              ) : (
                <div className="hud-panel empty-docket-card">
                  <div className="empty-stream-box">
                    <div className="empty-stream-icon"><Icon name="git-merge" size={32} /></div>
                    <h4>No Subclaim Decomposition Loaded</h4>
                    <p>Deconstruct compound claims with conjunctions or clauses into atomic verifiable propositions.</p>
                    <button className="primary-action-btn" onClick={() => openFixture(FIXTURE_CASES[0], true)}>
                      <Icon name="scan" size={15} />
                      <span>Load Sample Case &amp; Decompose</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ================= PAGE 4: MODEL PERFORMANCE LAB ================= */}
          {activePage === 'evaluation' && (
            <div className="page-section evaluation-page">
              <div className="page-title-row" style={{ marginBottom: '1.25rem' }}>
                <div>
                  <div className="section-eyebrow">
                    <span className="eyebrow-accent" />
                    SCIENTIFIC EVALUATION &amp; BENCHMARK LAB
                  </div>
                  <h1 className="page-headline">Model Performance <span>Lab.</span></h1>
                  <p className="page-subline">Defensible accuracy, macro-F1, selective abstention, and calibration benchmarks.</p>
                </div>
                <div className="page-title-actions">
                  <button className="secondary-action-btn" onClick={() => navigate('investigate')}>
                    ← Back to Claim Investigator
                  </button>
                </div>
              </div>
              <ModelPerformanceLab />
            </div>
          )}

          {/* ================= PAGE 5: SYSTEM TELEMETRY ================= */}
          {activePage === 'system' && (
            <div className="page-section system-page">
              <div className="page-title-row" style={{ marginBottom: '1.25rem' }}>
                <div>
                  <div className="section-eyebrow">
                    <span className="eyebrow-accent" />
                    RUNTIME INFRASTRUCTURE &amp; TELEMETRY
                  </div>
                  <h1 className="page-headline">System <span>Telemetry.</span></h1>
                  <p className="page-subline">Real-time health, device status, API latency, retrieval index parameters, and model configuration.</p>
                </div>
                <div className="page-title-actions">
                  <button className="secondary-action-btn" onClick={() => navigate('investigate')}>
                    ← Back to Claim Investigator
                  </button>
                </div>
              </div>
              <SystemTelemetry
                apiHealth={apiHealth}
                lastResult={result}
                mode={mode}
              />
            </div>
          )}
        </div>
      </main>

      {/* Floating Toast Notification */}
      {toast && (
        <div className="app-toast-alert" role="status">
          <Icon name="check" size={16} />
          <span>{toast}</span>
        </div>
      )}
    </div>
  );
}
