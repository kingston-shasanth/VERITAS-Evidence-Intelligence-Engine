import React, { useState } from 'react';
import { Icon } from './Icons';
import { ConfusionMatrix } from './ConfusionMatrix';

function ModelPerformanceLabComponent() {
  const [activeTab, setActiveTab] = useState('metrics'); // 'metrics' | 'matrix' | 'calibration' | 'retrieval' | 'errors'
  const [selectedSplit, setSelectedSplit] = useState('validation'); // 'validation' | 'frozen_test'

  const VALIDATION_METRICS = {
    splitName: 'Authorized Validation Split',
    sampleCount: 255,
    accuracy: 59.22,
    macroF1: 44.46,
    macroPrecision: 49.91,
    macroRecall: 50.92,
    coverage: 94.90,
    selectiveAccuracy: 61.57,
    selectiveMacroF1: 46.01,
    selectiveRisk: 38.43,
    classes: [
      { key: 'VERIFIED_TRUE', name: 'Verified True', support: 61, precision: 64.71, recall: 54.10, f1: 58.93, color: '#10b981' },
      { key: 'VERIFIED_FALSE', name: 'Verified False', support: 156, precision: 79.69, recall: 65.38, f1: 71.83, color: '#f43f5e' },
      { key: 'CONFLICTING_OR_CHERRYPICKED', name: 'Conflicting', support: 19, precision: 33.33, recall: 10.53, f1: 16.00, color: '#a855f7' },
      { key: 'INSUFFICIENT_EVIDENCE', name: 'Insufficient Evid.', support: 19, precision: 20.00, recall: 73.68, f1: 31.46, color: '#f59e0b' },
    ],
  };

  const FROZEN_TEST_METRICS = {
    splitName: 'Frozen Phase 3.2.5 Held-Out Test Set (FROZEN BENCHMARK)',
    sampleCount: 256,
    accuracy: 60.16,
    macroF1: 43.37,
    macroPrecision: 47.81,
    macroRecall: 49.07,
    coverage: 94.53,
    selectiveAccuracy: 62.40,
    selectiveMacroF1: 44.89,
    selectiveRisk: 37.60,
    classes: [
      { key: 'VERIFIED_TRUE', name: 'Verified True', support: 61, precision: 60.00, recall: 49.18, f1: 54.05, color: '#10b981' },
      { key: 'VERIFIED_FALSE', name: 'Verified False', support: 157, precision: 82.95, recall: 68.15, f1: 74.83, color: '#f43f5e' },
      { key: 'CONFLICTING_OR_CHERRYPICKED', name: 'Conflicting', support: 19, precision: 28.57, recall: 10.53, f1: 15.38, color: '#a855f7' },
      { key: 'INSUFFICIENT_EVIDENCE', name: 'Insufficient Evid.', support: 19, precision: 19.70, recall: 68.42, f1: 30.59, color: '#f59e0b' },
    ],
  };

  const currentDataset = selectedSplit === 'validation' ? VALIDATION_METRICS : FROZEN_TEST_METRICS;

  const RETRIEVAL_DATA = [
    { method: 'BM25 (Sparse Lexical)', hit1: 44.32, hit3: 58.52, hit5: 63.32, hit10: 71.18, mrr: 0.5278, latency: '2.7 ms' },
    { method: 'Dense E5 (Multilingual FAISS)', hit1: 51.53, hit3: 65.07, hit5: 72.27, hit10: 81.00, mrr: 0.6032, latency: '38.4 ms' },
    { method: 'Hybrid RRF (k=60 + Tie-break)', hit1: 52.62, hit3: 67.03, hit5: 73.80, hit10: 79.69, mrr: 0.6139, latency: '43.4 ms', isBest: true },
  ];

  return (
    <div className="performance-lab-card" role="region" aria-label="Model Performance Lab">
      {/* Header and Controls */}
      <div className="lab-header-row">
        <div>
          <span className="lab-kicker">SCIENTIFIC EVALUATION & AUDIT</span>
          <h3 className="lab-title">Model Performance & Calibration Lab</h3>
          <p className="lab-subtitle">
            Defensible, reproducible ML evaluation metrics derived strictly from authorized evaluation splits.
          </p>
        </div>

        {/* Dataset Split Selector */}
        <div className="split-selector-control">
          <label className="split-label">DATASET BENCHMARK:</label>
          <div className="split-toggle-group">
            <button
              type="button"
              className={`split-btn ${selectedSplit === 'validation' ? 'split-btn-active' : ''}`}
              onClick={() => setSelectedSplit('validation')}
            >
              Validation Split (N=255)
            </button>
            <button
              type="button"
              className={`split-btn ${selectedSplit === 'frozen_test' ? 'split-btn-active' : ''}`}
              onClick={() => setSelectedSplit('frozen_test')}
            >
              Frozen Test Benchmark (N=256)
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="lab-nav-tabs" role="tablist">
        <button
          type="button"
          className={`lab-tab ${activeTab === 'metrics' ? 'lab-tab-active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          <Icon name="chart" size={15} />
          <span>Core Metrics & Per-Class F1</span>
        </button>
        <button
          type="button"
          className={`lab-tab ${activeTab === 'matrix' ? 'lab-tab-active' : ''}`}
          onClick={() => setActiveTab('matrix')}
        >
          <Icon name="grid" size={15} />
          <span>4×4 Confusion Matrix</span>
        </button>
        <button
          type="button"
          className={`lab-tab ${activeTab === 'calibration' ? 'lab-tab-active' : ''}`}
          onClick={() => setActiveTab('calibration')}
        >
          <Icon name="sliders" size={15} />
          <span>Calibration & Selective Risk</span>
        </button>
        <button
          type="button"
          className={`lab-tab ${activeTab === 'retrieval' ? 'lab-tab-active' : ''}`}
          onClick={() => setActiveTab('retrieval')}
        >
          <Icon name="files" size={15} />
          <span>Retrieval Benchmark (RRF)</span>
        </button>
        <button
          type="button"
          className={`lab-tab ${activeTab === 'errors' ? 'lab-tab-active' : ''}`}
          onClick={() => setActiveTab('errors')}
        >
          <Icon name="alert" size={15} />
          <span>Error & Failure Analysis</span>
        </button>
      </div>

      {/* TAB 1: CORE METRICS & PER-CLASS TABLE */}
      {activeTab === 'metrics' && (
        <div className="lab-panel">
          <div className="kpi-quad-grid">
            <div className="kpi-card">
              <span className="kpi-label">OVERALL ACCURACY</span>
              <div className="kpi-value-row">
                <span className="kpi-big-num">{currentDataset.accuracy}%</span>
                <span className="kpi-subtext">N={currentDataset.sampleCount}</span>
              </div>
              <p className="kpi-desc">Overall percentage of benchmark claims where the system's verdict matched human ground truth.</p>
            </div>

            <div className="kpi-card">
              <span className="kpi-label">MACRO-AVERAGED F1</span>
              <div className="kpi-value-row">
                <span className="kpi-big-num">{currentDataset.macroF1}%</span>
                <span className="kpi-subtext">Balanced</span>
              </div>
              <p className="kpi-desc">Balanced benchmark score weighing all four verdict categories equally, so rare classes are not ignored.</p>
            </div>

            <div className="kpi-card">
              <span className="kpi-label">SELECTIVE ACCURACY</span>
              <div className="kpi-value-row">
                <span className="kpi-big-num text-teal">{currentDataset.selectiveAccuracy}%</span>
                <span className="kpi-subtext">Cov: {currentDataset.coverage}%</span>
              </div>
              <p className="kpi-desc">Accuracy when the system answers confident claims, prudently abstaining when evidence is ambiguous.</p>
            </div>

            <div className="kpi-card">
              <span className="kpi-label">CALIBRATED ERROR (ECE)</span>
              <div className="kpi-value-row">
                <span className="kpi-big-num text-amber">11.83%</span>
                <span className="kpi-subtext">T=4.1039</span>
              </div>
              <p className="kpi-desc">Measures probability reliability—ensuring that an 80% confidence level corresponds to an 80% true rate.</p>
            </div>
          </div>

          {/* Per Class Breakdown Table */}
          <div className="lab-table-container">
            <h4 className="table-caption">Per-Class Verification Performance ({currentDataset.splitName})</h4>
            <table className="lab-data-table">
              <thead>
                <tr>
                  <th>Verdict Class</th>
                  <th>Support (N)</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1 Score</th>
                  <th>Relative Performance Bar</th>
                </tr>
              </thead>
              <tbody>
                {currentDataset.classes.map((c) => (
                  <tr key={c.key}>
                    <td>
                      <span className="class-pill-indicator" style={{ borderLeftColor: c.color }}>
                        {c.name}
                      </span>
                    </td>
                    <td className="mono-cell">{c.support}</td>
                    <td className="mono-cell">{c.precision.toFixed(1)}%</td>
                    <td className="mono-cell">{c.recall.toFixed(1)}%</td>
                    <td className="mono-cell"><strong>{c.f1.toFixed(1)}%</strong></td>
                    <td className="bar-cell">
                      <div className="table-progress-track">
                        <div
                          className="table-progress-fill"
                          style={{ width: `${c.f1}%`, backgroundColor: c.color }}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 2: CONFUSION MATRIX */}
      {activeTab === 'matrix' && (
        <div className="lab-panel">
          <ConfusionMatrix />
        </div>
      )}

      {/* TAB 3: CALIBRATION & SELECTIVE RISK */}
      {activeTab === 'calibration' && (
        <div className="lab-panel">
          <div className="calibration-overview-grid">
            <div className="calibration-stat-card">
              <h4 className="card-mini-title">Temperature Calibration Effect</h4>
              <p className="card-mini-desc">
                Raw cross-encoder multi-class logits suffer from empirical overconfidence. Post-hoc temperature scaling (T=4.1039) fits validation predictions to align model confidence with empirical frequency.
              </p>
              <div className="calib-metric-pairs">
                <div className="pair-row">
                  <span>Expected Calibration Error (ECE)</span>
                  <div className="pair-vals">
                    <span className="val-old">19.21%</span> → <strong className="val-new text-teal">11.83%</strong>
                  </div>
                </div>
                <div className="pair-row">
                  <span>Negative Log-Likelihood (NLL)</span>
                  <div className="pair-vals">
                    <span className="val-old">1.1524</span> → <strong className="val-new text-teal">0.9627</strong>
                  </div>
                </div>
                <div className="pair-row">
                  <span>Brier Score (Multi-Class)</span>
                  <div className="pair-vals">
                    <span className="val-old">0.5482</span> → <strong className="val-new text-teal">0.4988</strong>
                  </div>
                </div>
              </div>
            </div>

            <div className="calibration-stat-card">
              <h4 className="card-mini-title">Selective Abstention Policy</h4>
              <p className="card-mini-desc">
                The selective predictor issues a decision g(x) = 1 only when calibrated confidence P_max ≥ 0.40 and normalized Shannon entropy H(P) ≤ 1.35.
              </p>
              <div className="risk-coverage-stats">
                <div className="risk-row">
                  <span>Coverage Rate (g(x) = 1):</span>
                  <strong>94.90% (242 / 255 claims retained)</strong>
                </div>
                <div className="risk-row">
                  <span>Operational Abstentions (g(x) = 0):</span>
                  <strong>5.10% (13 claims deferred to humans)</strong>
                </div>
                <div className="risk-row">
                  <span>Selective Risk (Error on Retained):</span>
                  <strong className="text-teal">38.43% (vs. 40.78% full error)</strong>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: RETRIEVAL BENCHMARK */}
      {activeTab === 'retrieval' && (
        <div className="lab-panel">
          <div className="retrieval-summary-box">
            <h4 className="table-caption">Multi-Modal Retrieval Evaluation (N=458 claims, 1,047 passages)</h4>
            <p className="table-subtitle">
              Evaluated using <code>scripts/evaluate_retrieval.py</code> on canonical claim-passage alignment.
            </p>
            <table className="lab-data-table">
              <thead>
                <tr>
                  <th>Retrieval Method</th>
                  <th>Hit@1</th>
                  <th>Hit@3</th>
                  <th>Hit@5</th>
                  <th>Hit@10</th>
                  <th>MRR</th>
                  <th>Mean Latency</th>
                </tr>
              </thead>
              <tbody>
                {RETRIEVAL_DATA.map((r) => (
                  <tr key={r.method} className={r.isBest ? 'highlight-row' : ''}>
                    <td>
                      <strong>{r.method}</strong>
                      {r.isBest && <span className="best-tag">OPTIMAL MRR</span>}
                    </td>
                    <td className="mono-cell">{r.hit1}%</td>
                    <td className="mono-cell">{r.hit3}%</td>
                    <td className="mono-cell">{r.hit5}%</td>
                    <td className="mono-cell">{r.hit10}%</td>
                    <td className="mono-cell"><strong>{r.mrr.toFixed(4)}</strong></td>
                    <td className="mono-cell">{r.latency}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 5: ERROR ANALYSIS & UNRESOLVED CHALLENGES */}
      {activeTab === 'errors' && (
        <div className="lab-panel">
          <div className="error-analysis-grid">
            <div className="error-card">
              <span className="error-type-tag">CHALLENGE 01</span>
              <h4>Subtle Stance Disparity in Web Discourse</h4>
              <p>
                Conflicting claims (19 cases in validation) had a recall of 10.53%. Real-world cherrypicking rarely features two clean, symmetric 50/50 opposing paragraphs in the top-5 passages; one perspective often dominates retrieval, skewing the verdict toward <code>VERIFIED_FALSE</code>.
              </p>
            </div>

            <div className="error-card">
              <span className="error-type-tag">CHALLENGE 02</span>
              <h4>Multi-Hop Evidence Synthesis</h4>
              <p>
                When a claim requires combining 2 separate facts (e.g. date of birth + legislation enactment date), individual Q&A passages test neutral in isolation. Concatenation in joint-premise cross-encoding mitigates this, but cross-passage context limits remain bounded by 512 tokens.
              </p>
            </div>

            <div className="error-card">
              <span className="error-type-tag">CHALLENGE 03</span>
              <h4>Non-Entailing vs. Refuting Evidence Distinctions</h4>
              <p>
                The absence of evidence is not refutation. The system routes claims with zero active evidence mass to <code>INSUFFICIENT_EVIDENCE</code> rather than assuming falsehood.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export const ModelPerformanceLab = React.memo(ModelPerformanceLabComponent);
export default ModelPerformanceLab;
