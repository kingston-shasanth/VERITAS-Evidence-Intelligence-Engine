/**
 * VERITAS-X Stage, Phase, and Taxonomy Constants
 * Shared between UI presentation components and automated verification tests.
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

export const PHASES = [
  {
    id: 'claim',
    label: 'Your Claim',
    icon: 'edit',
    stageIds: ['input_received', 'sanitization'],
    question: 'What was submitted?',
  },
  {
    id: 'cleanup',
    label: 'Text Cleanup',
    icon: 'sparkles',
    stageIds: ['spelling_normalization', 'meaning_preservation'],
    question: 'Were any corrections applied?',
  },
  {
    id: 'analysis',
    label: 'Language Analysis',
    icon: 'layers',
    stageIds: ['tokenization', 'linguistic_analysis', 'claim_extraction'],
    question: 'How was the claim understood?',
  },
  {
    id: 'search',
    label: 'Evidence Search',
    icon: 'search',
    stageIds: ['retrieval', 'evidence_processing'],
    question: 'What evidence was found?',
  },
  {
    id: 'review',
    label: 'Evidence Review',
    icon: 'files',
    stageIds: ['nli_verification'],
    question: 'How does the evidence relate to the claim?',
  },
  {
    id: 'verdict',
    label: 'Final Verdict',
    icon: 'shield',
    stageIds: ['final_result'],
    question: 'What is the conclusion?',
  },
];

export const VERDICT_TAXONOMY = {
  VERIFIED_TRUE: {
    label: 'Verified True',
    headline: 'Documentary evidence confirms this statement is accurate.',
    badgeClass: 'verdict-true',
    icon: 'checkCircle',
  },
  VERIFIED_FALSE: {
    label: 'Verified False',
    headline: 'Documentary evidence directly contradicts this statement.',
    badgeClass: 'verdict-false',
    icon: 'xCircle',
  },
  CONFLICTING_OR_CHERRYPICKED: {
    label: 'Conflicting Evidence',
    headline: 'Sources contain conflicting evidence from different viewpoints.',
    badgeClass: 'verdict-conflict',
    icon: 'alertTriangle',
  },
  INSUFFICIENT_EVIDENCE: {
    label: 'Insufficient Evidence',
    headline: 'Not enough reliable evidence was found to verify this statement.',
    badgeClass: 'verdict-insufficient',
    icon: 'shieldAlert',
  },
};
