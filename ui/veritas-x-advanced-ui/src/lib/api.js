const API_BASE = (
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL) ||
  (typeof process !== 'undefined' && process.env?.VITE_API_BASE_URL) ||
  'http://127.0.0.1:8011'
).replace(/\/$/, '');

/**
 * Normalizes backend responses into consistent forensic dossier data structures.
 */
export function normalizeResponse(raw, claimText) {
  if (!raw || typeof raw !== 'object') throw new Error('The API response was empty or invalid.');
  const verdict = raw.verdict || raw.predicted_verdict || raw.label;
  const rawVerdict = raw.raw_verdict || null;
  const probabilities = raw.probabilities || raw.calibrated_probabilities || raw.class_probabilities;
  return {
    id: raw.claim_id || `VX-${Date.now().toString().slice(-6)}`,
    claim: raw.claim || claimText,
    verdict,
    raw_verdict: rawVerdict,
    confidence: raw.confidence ?? raw.max_probability ?? (probabilities ? Math.max(...Object.values(probabilities).map(Number)) : 0),
    probs: probabilities || {},
    summary: raw.reason || raw.explanation || 'Verification completed by forensic engine.',
    disparity: Boolean(raw.has_cross_passage_disparity ?? raw.disparity_detected),
    abstained: Boolean(raw.abstained ?? raw.is_abstained),
    abstention_reason: raw.abstention_reason || '',
    has_meaningful_evidence: Boolean(raw.has_meaningful_evidence ?? true),
    server_latency_ms: raw.latency_ms ?? null,
    timing_ms: raw.timing_ms ?? null,
    model_ready: raw.model_ready ?? true,
    subclaims: Array.isArray(raw.subclaims) ? raw.subclaims : [],
    temporal: raw.temporal && typeof raw.temporal === 'object' ? raw.temporal : null,
    graph: raw.graph && typeof raw.graph === 'object' ? raw.graph : null,
    uncertainty: raw.uncertainty && typeof raw.uncertainty === 'object' ? raw.uncertainty : null,
    mode: raw.mode || 'local',
    retrieval_status: raw.retrieval_status || null,
    evidence_counts: raw.evidence_counts || null,
    // Phase 4A Live NLP & Telemetry Fields
    original_claim: raw.original_claim || raw.claim || claimText,
    sanitized_claim: raw.sanitized_claim || raw.claim || claimText,
    normalized_claim: raw.normalized_claim || raw.claim || claimText,
    normalization_status: raw.normalization_status || 'accepted',
    normalization_corrections: Array.isArray(raw.normalization_corrections) ? raw.normalization_corrections : [],
    normalization_warnings: Array.isArray(raw.normalization_warnings) ? raw.normalization_warnings : [],
    structured_claim: raw.structured_claim && typeof raw.structured_claim === 'object' ? raw.structured_claim : null,
    retrieval_query_used: raw.retrieval_query_used || raw.claim || claimText,
    nli_hypothesis_used: raw.nli_hypothesis_used || raw.sanitized_claim || raw.claim || claimText,
    processing_trace: Array.isArray(raw.processing_trace) ? raw.processing_trace : [],
    processing_trace_version: raw.processing_trace_version || 1,
    evidence: (raw.evidence || raw.retrieved_evidence || []).map((item, i) => {
      // Determine provenance strictly adhering to raw.mode and item.source_type
      let itemSourceType = 'local';
      if (raw.mode === 'local') {
        itemSourceType = 'local';
      } else if (raw.mode === 'web') {
        itemSourceType = 'web';
      } else if (raw.mode === 'user_supplied') {
        itemSourceType = 'user_supplied';
      } else {
        // hybrid or unspecified mode: inspect item provenance
        if (item.source_type) {
          itemSourceType = item.source_type === 'user' ? 'user_supplied' : item.source_type;
        } else if (item.doc_id && String(item.doc_id).startsWith('web_')) {
          itemSourceType = 'web';
        } else {
          itemSourceType = 'local';
        }
      }

      return {
        id: item.id || item.evidence_id || `E-${String(i + 1).padStart(2, '0')}`,
        source: item.source || item.source_title || (itemSourceType === 'web' ? (item.source_url || item.url || 'Live Web Citation') : 'Local Evidence Corpus'),
        source_type: itemSourceType,
        source_title: item.source_title || item.title || item.source || (itemSourceType === 'web' ? 'Web Source' : 'Local Evidence Corpus'),
        source_url: item.source_url || item.url || '',
        published_date: item.published_date || item.date || '',
        stance: item.stance || item.predicted_stance || 'neutral',
        score: item.score ?? item.relevance_score ?? item.confidence ?? 0,
        text: item.text || item.passage || item.content || '',
        entailment_prob: item.entailment_prob,
        contradiction_prob: item.contradiction_prob,
        neutral_prob: item.neutral_prob,
        is_informative: item.is_informative !== undefined ? Boolean(item.is_informative) : true,
        is_relevant: item.is_relevant !== undefined ? Boolean(item.is_relevant) : true,
        relevance_status: item.relevance_status || (item.is_relevant === false ? 'unrelated' : 'relevant'),
        is_claim_aligned: item.is_claim_aligned !== undefined ? Boolean(item.is_claim_aligned) : true,
        is_decision_eligible: item.is_decision_eligible !== undefined ? Boolean(item.is_decision_eligible) : (Boolean(item.is_relevant ?? true) && Boolean(item.is_informative ?? true) && Boolean(item.is_claim_aligned ?? true)),
        is_figurative: Boolean(item.is_figurative),
        claim_coverage: item.claim_coverage || 'full',
        claim_components_covered: Array.isArray(item.claim_components_covered) ? item.claim_components_covered : [],
        contributed_to_decision: Boolean(item.contributed_to_decision),
        explanation: item.explanation || '',
      };
    }),
    raw,
  };
}

/**
 * Adapter contract for the VERITAS-X Python API.
 * The existing engine's precise result schema must be mapped in the backend adapter.
 * Expected response keys are documented in backend_contract.md.
 */
export async function verifyClaim(claim, evidence = [], mode = 'local', signal = null) {
  const fetchOptions = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      claim,
      evidence,
      mode,
      include_research: true,
      include_processing_trace: true,
    }),
  };
  if (signal) {
    fetchOptions.signal = signal;
  }
  const response = await fetch(`${API_BASE}/api/verify`, fetchOptions);
  if (!response.ok) {
    const message = await response.text().catch(() => 'Unknown API error');
    throw new Error(`Verification API returned ${response.status}: ${message.slice(0, 220)}`);
  }
  return response.json();
}

export const apiBase = API_BASE;

