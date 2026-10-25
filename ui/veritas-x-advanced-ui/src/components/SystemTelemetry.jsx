import React, { useState, useEffect } from 'react';
import { Icon } from './Icons';
import { apiBase } from '../lib/api';

export function SystemTelemetry({ apiHealth, lastResult, mode }) {
  const [telemetry, setTelemetry] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    const fetchHealth = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${apiBase}/api/health`);
        if (res.ok) {
          const data = await res.json();
          if (active) setTelemetry(data);
        }
      } catch {
        if (active) setTelemetry({ status: 'offline' });
      } finally {
        if (active) setLoading(false);
      }
    };

    fetchHealth();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="telemetry-dashboard-card" role="region" aria-label="System Telemetry & Architecture">
      <div className="telemetry-head">
        <div>
          <span className="telemetry-kicker">ENGINE TELEMETRY & HARDWARE RUNTIME</span>
          <h3 className="telemetry-title">System Infrastructure & Reproducibility Specs</h3>
        </div>
        <div className="telemetry-status-badge">
          <span className={`status-orb ${apiHealth.online ? 'orb-online' : 'orb-offline'}`} />
          <span>{apiHealth.online ? 'ENGINE ONLINE' : 'ENGINE OFFLINE'}</span>
        </div>
      </div>

      <div className="telemetry-sections-grid">
        {/* Runtime State */}
        <div className="telemetry-box">
          <h4 className="telemetry-box-title">
            <Icon name="sliders" size={16} />
            <span>Compute & Process Runtime</span>
          </h4>
          <div className="telemetry-kv-list">
            <div className="telemetry-kv-row">
              <span className="kv-key">Backend Service</span>
              <strong className="kv-val">{telemetry?.service || 'VERITAS-X Verification API'}</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Engine Version</span>
              <strong className="kv-val">{telemetry?.version || 'Phase 3.2.5'}</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Inference Device</span>
              <strong className="kv-val text-teal">
                {(telemetry?.device || apiHealth?.device || 'cpu').toUpperCase()}
              </strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Execution Mode</span>
              <strong className="kv-val">{mode === 'live' ? 'Live API (port 8011)' : 'Demo Fixture Mode'}</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Engine Model Readiness</span>
              <strong className="kv-val text-teal">
                {telemetry?.model_ready || apiHealth?.modelReady
                  ? 'Ready (Warmed in RAM/VRAM)'
                  : (telemetry?.model_loading || apiHealth?.modelLoading ? 'Initializing Weights...' : 'Pending First Use')}
              </strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Client Round-Trip Latency</span>
              <strong className="kv-val">
                {lastResult?.round_trip_ms ? `${lastResult.round_trip_ms} ms (browser)` : 'Awaiting claim'}
              </strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Server Inference Latency</span>
              <strong className="kv-val">
                {lastResult?.server_latency_ms ? `${lastResult.server_latency_ms} ms (monotonic)` : 'Awaiting claim'}
              </strong>
            </div>
          </div>
        </div>

        {/* NLI & Modeling Architecture */}
        <div className="telemetry-box">
          <h4 className="telemetry-box-title">
            <Icon name="nodes" size={16} />
            <span>NLI & Representation Models</span>
          </h4>
          <div className="telemetry-kv-list">
            <div className="telemetry-kv-row">
              <span className="kv-key">Cross-Encoder Checkpoint</span>
              <strong className="kv-val">MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Max Premise Token Limit</span>
              <strong className="kv-val">512 tokens</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Dense Embedding Model</span>
              <strong className="kv-val">intfloat/multilingual-e5-small (384-dim)</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">FAISS Index Type</span>
              <strong className="kv-val">IndexFlatIP (Cosine normalized)</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Sparse Lexical Index</span>
              <strong className="kv-val">BM25Okapi (k1=1.5, b=0.75, score&gt;0 gate)</strong>
            </div>
          </div>
        </div>

        {/* Calibration & Aggregation Parameters */}
        <div className="telemetry-box">
          <h4 className="telemetry-box-title">
            <Icon name="shield" size={16} />
            <span>Frozen Calibration & Aggregation</span>
          </h4>
          <div className="telemetry-kv-list">
            <div className="telemetry-kv-row">
              <span className="kv-key">Calibration Temperature (T)</span>
              <strong className="kv-val text-teal">4.1039 (Validation Fitted)</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Confidence Abstention Threshold</span>
              <strong className="kv-val">τ_abstain = 0.40</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Entropy Abstention Threshold</span>
              <strong className="kv-val">τ_entropy = 1.35</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Disparity Entailment Threshold</span>
              <strong className="kv-val">e_i ≥ 0.35</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Disparity Contradiction Threshold</span>
              <strong className="kv-val">c_j ≥ 0.50</strong>
            </div>
          </div>
        </div>

        {/* Corpus & Reproducibility */}
        <div className="telemetry-box">
          <h4 className="telemetry-box-title">
            <Icon name="database" size={16} />
            <span>Corpus Provenance & Reproducibility</span>
          </h4>
          <div className="telemetry-kv-list">
            <div className="telemetry-kv-row">
              <span className="kv-key">Local Corpus Size</span>
              <strong className="kv-val">1,047 Canonical Evidence Passages</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Primary Evidence Source</span>
              <strong className="kv-val">AVeriTeC Multi-Hop Fact Checking Dataset</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Hybrid RRF Constant (k)</span>
              <strong className="kv-val">k = 60</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Tie-Breaking Protocol</span>
              <strong className="kv-val">Deterministic (RRF_Score desc, Doc_ID asc)</strong>
            </div>
            <div className="telemetry-kv-row">
              <span className="kv-key">Per-Stage Backend Latency</span>
              <strong className="kv-val">
                {lastResult?.timing_ms ? (
                  <span>
                    Retrieval: <strong>{lastResult.timing_ms.retrieval_ms ?? '0'}ms</strong> &bull; NLI Forward: <strong>{lastResult.timing_ms.engine_inference_ms ?? '0'}ms</strong>
                  </span>
                ) : (
                  'Instrumented (Monotonic breakdown active)'
                )}
              </strong>
            </div>
            {lastResult?.client_api_wait_ms != null && (
              <div className="telemetry-kv-row">
                <span className="kv-key">Browser API Waiting Time</span>
                <strong className="kv-val">{lastResult.client_api_wait_ms} ms</strong>
              </div>
            )}
            {lastResult?.client_render_ms != null && (
              <div className="telemetry-kv-row">
                <span className="kv-key">Frontend Render Duration</span>
                <strong className="kv-val">{lastResult.client_render_ms} ms</strong>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default React.memo(SystemTelemetry);
