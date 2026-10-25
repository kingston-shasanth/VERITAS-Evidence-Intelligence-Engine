import React, { useState } from 'react';
import { Icon } from './Icons';

export const PIPELINE_STAGES = [
  {
    id: 'intake',
    step: '01',
    name: 'Claim Intake',
    shortName: 'Intake',
    category: 'Preprocessing',
    spec: 'Unicode normalization, proposition extraction, validation',
    details: 'Validates non-empty input, sanitizes whitespace, and generates deterministic claim ID hash.',
    file: 'api/adapter.py',
    metricsKey: 'claim_id',
  },
  {
    id: 'bm25',
    step: '02',
    name: 'BM25 Sparse Retrieval',
    shortName: 'BM25',
    category: 'Retrieval',
    spec: 'Inverted lexical index, score > 0.0 threshold filtering',
    details: 'Searches 1,047 local corpus passages. Strict positive score gating guarantees zero lexical matches never receive phantom rank contributions.',
    file: 'src/veritas/retrieval/bm25_retriever.py',
    metricsKey: 'bm25_hits',
  },
  {
    id: 'dense',
    step: '03',
    name: 'Dense Semantic Retrieval',
    shortName: 'Dense E5',
    category: 'Retrieval',
    spec: 'intfloat/multilingual-e5-small + FAISS IndexFlatIP',
    details: 'Encodes proposition into 384-dimensional unit vector and performs inner product search independent of lexical overlap.',
    file: 'src/veritas/retrieval/dense_retriever.py',
    metricsKey: 'dense_hits',
  },
  {
    id: 'rrf',
    step: '04',
    name: 'Reciprocal Rank Fusion',
    shortName: 'Hybrid RRF',
    category: 'Retrieval Fusion',
    spec: 'RRF(d) = Σ 1/(60 + r_m(d)) + deterministic doc_id tie-breaking',
    details: 'Fuses sparse and dense candidate pools. Discards duplicates and applies secondary doc_id ordering to guarantee reproducible rankings.',
    file: 'src/veritas/retrieval/hybrid_retriever.py',
    metricsKey: 'retrieved_count',
  },
  {
    id: 'nli',
    step: '05',
    name: 'Pairwise Cross-Encoder',
    shortName: 'DeBERTa NLI',
    category: 'Inference',
    spec: 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (512 tokens max)',
    details: 'Evaluates contextualized premises ("Question: Q Answer: A") against the claim hypothesis, outputting raw P(Entailment), P(Contradiction), P(Neutral).',
    file: 'src/veritas/verification/nli_model.py',
    metricsKey: 'nli_eval',
  },
  {
    id: 'aggregation',
    step: '06',
    name: 'Evidence Mass Aggregation',
    shortName: 'Aggregation',
    category: 'Reasoning',
    spec: 'Evidence masses M_E, M_C, M_N + Disparity check (e_i≥0.35, c_j≥0.50)',
    details: 'Computes directional evidence mass. Flags cross-passage stance disparity between opposing sources to detect cherrypicked propositions.',
    file: 'src/veritas/aggregation/evidence_aggregator.py',
    metricsKey: 'disparity',
  },
  {
    id: 'calibration',
    step: '07',
    name: 'Calibration & Selective Policy',
    shortName: 'Calibration',
    category: 'Decision Theory',
    spec: 'Temperature scaling (T=4.1039) + Selective gate (P_max≥0.40, H≤1.35)',
    details: 'Calibrates 4-class multi-class logits to minimize ECE (reduced from 19.21% to 11.83%). Safely abstains (g(x)=0) on high-entropy or low-confidence claims.',
    file: 'src/veritas/calibration/temperature_scaling.py',
    metricsKey: 'confidence',
  },
  {
    id: 'report',
    step: '08',
    name: 'Forensic Dossier Dispatch',
    shortName: 'Dossier',
    category: 'Dispatch',
    spec: 'Evidence-grounded rationale, source citations & audit trail',
    details: 'Generates factual rationales strictly aligned with non-neutral evidence counts. Preserves full URL provenance and latency metrics.',
    file: 'api/adapter.py',
    metricsKey: 'verdict',
  },
];

export function PipelineVisualizer({ isAnalyzing = false, result = null, elapsedSeconds = 0 }) {
  const [selectedStageId, setSelectedStageId] = useState('rrf');

  const selectedStage = PIPELINE_STAGES.find((s) => s.id === selectedStageId) || PIPELINE_STAGES[3];

  return (
    <div className="pipeline-visualizer-card" role="region" aria-label="Forensic Architecture Pipeline">
      <div className="pipeline-card-header">
        <div className="header-left">
          <span className="pipeline-badge-kicker">VERITAS-X CORE PIPELINE</span>
          <h3 className="pipeline-title">8-Stage Forensic Inference Flow</h3>
        </div>
        <div className="header-right">
          {isAnalyzing ? (
            <div className="pipeline-status-pill analyzing">
              <span className="pulse-indicator" />
              <span>Inference In-Flight ({elapsedSeconds.toFixed(1)}s)</span>
            </div>
          ) : result ? (
            <div className="pipeline-status-pill resolved">
              <Icon name="check" size={14} />
              <span>Dossier Resolved ({result.server_latency_ms ? `${result.server_latency_ms}ms` : 'Complete'})</span>
            </div>
          ) : (
            <div className="pipeline-status-pill standby">
              <Icon name="radar" size={14} />
              <span>Engine Ready</span>
            </div>
          )}
        </div>
      </div>

      {/* Honest indeterminate loading indicator when in-flight */}
      {isAnalyzing && (
        <div className="indeterminate-pipeline-bar" role="progressbar" aria-valuetext="Inference in-flight">
          <div className="indeterminate-runner" />
        </div>
      )}

      {/* The 8 stages horizontal/responsive track */}
      <div className="pipeline-stages-track">
        {PIPELINE_STAGES.map((stage, idx) => {
          const isSelected = stage.id === selectedStageId;
          const isFinal = idx === PIPELINE_STAGES.length - 1;

          return (
            <React.Fragment key={stage.id}>
              <button
                type="button"
                className={`stage-node ${isSelected ? 'stage-node-active' : ''} ${result ? 'stage-node-completed' : ''}`}
                onClick={() => setSelectedStageId(stage.id)}
                aria-pressed={isSelected}
              >
                <div className="node-num-pill">{stage.step}</div>
                <div className="node-text-col">
                  <span className="node-category">{stage.category}</span>
                  <strong className="node-name">{stage.shortName}</strong>
                </div>
              </button>

              {!isFinal && (
                <div className={`stage-connector ${isAnalyzing ? 'connector-pulsing' : ''}`}>
                  <span className="connector-arrow">→</span>
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Selected Stage Detail Inspector */}
      <div className="stage-detail-inspector">
        <div className="inspector-grid">
          <div className="inspector-main">
            <div className="inspector-tag-row">
              <span className="stage-step-tag">STAGE {selectedStage.step} OF 08</span>
              <span className="stage-cat-tag">{selectedStage.category}</span>
              <span className="stage-file-tag">{selectedStage.file}</span>
            </div>
            <h4 className="inspector-heading">{selectedStage.name}</h4>
            <p className="inspector-desc">{selectedStage.details}</p>
            <div className="inspector-spec-box">
              <span className="spec-label">MATHEMATICAL SPECIFICATION:</span>
              <code className="spec-code">{selectedStage.spec}</code>
            </div>
          </div>

          <div className="inspector-telemetry">
            <span className="telemetry-label">ACTIVE PIPELINE CONTEXT</span>
            {result ? (
              <div className="telemetry-values">
                <div className="telemetry-row">
                  <span>Verdict Status</span>
                  <strong className="mono-val">{result.verdict}</strong>
                </div>
                <div className="telemetry-row">
                  <span>Passages Examined</span>
                  <strong className="mono-val">{result.evidence?.length || 0}</strong>
                </div>
                <div className="telemetry-row">
                  <span>Disparity State</span>
                  <strong className="mono-val">{result.disparity ? 'DETECTED' : 'CONCORDANT'}</strong>
                </div>
                <div className="telemetry-row">
                  <span>Calibrated Confidence</span>
                  <strong className="mono-val">{Math.round((result.confidence || 0) * 100)}%</strong>
                </div>
              </div>
            ) : (
              <div className="telemetry-standby-hint">
                <Icon name="info" size={16} />
                <span>Submit a claim or run a demonstration case to inspect live telemetry through this stage.</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default PipelineVisualizer;
