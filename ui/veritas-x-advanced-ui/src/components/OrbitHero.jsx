import React, { useEffect, useRef, useState } from 'react';

/**
 * High-performance, animated forensic evidence-orbit visualization.
 * Renders an interactive scientific instrumentation display connecting:
 * CLAIM (01) -> EVIDENCE STREAM (02) -> ADJUDICATED VERDICT (03).
 */
export function OrbitHero({ onStartInvestigation, totalInvestigations = 0, activeMode = 'demo' }) {
  const canvasRef = useRef(null);
  const [activeTelemetry, setActiveTelemetry] = useState('EVIDENCE');
  const [hoveredNode, setHoveredNode] = useState(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId;
    let angle = 0;
    const particles = Array.from({ length: 42 }, () => ({
      orbitRadius: 70 + Math.random() * 110,
      orbitSpeed: (0.004 + Math.random() * 0.008) * (Math.random() > 0.4 ? 1 : -1),
      currentAngle: Math.random() * Math.PI * 2,
      size: 1 + Math.random() * 2.2,
      alpha: 0.2 + Math.random() * 0.7,
      color: Math.random() > 0.6 ? '#00f5d4' : Math.random() > 0.3 ? '#8b5cf6' : '#38bdf8',
    }));

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    };

    resize();
    window.addEventListener('resize', resize);

    const render = () => {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      const cx = width / 2;
      const cy = height / 2;

      ctx.clearRect(0, 0, width, height);

      // 1. Concentric orbital tracks
      const rings = [65, 115, 160];
      rings.forEach((r, idx) => {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = idx === 1 ? 'rgba(0, 245, 212, 0.16)' : 'rgba(56, 189, 248, 0.09)';
        ctx.lineWidth = idx === 1 ? 1.5 : 1;
        if (idx === 1) ctx.setLineDash([4, 6]);
        else ctx.setLineDash([]);
        ctx.stroke();
      });

      // 2. Radar sweep segment
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      const sweepGrad = ctx.createRadialGradient(0, 0, 10, 0, 0, 165);
      sweepGrad.addColorStop(0, 'rgba(0, 245, 212, 0.25)');
      sweepGrad.addColorStop(0.5, 'rgba(0, 245, 212, 0.05)');
      sweepGrad.addColorStop(1, 'transparent');
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.arc(0, 0, 165, -0.35, 0.35);
      ctx.closePath();
      ctx.fillStyle = sweepGrad;
      ctx.fill();
      ctx.restore();

      // 3. Flowing particles
      particles.forEach((p) => {
        p.currentAngle += p.orbitSpeed;
        const px = cx + Math.cos(p.currentAngle) * p.orbitRadius;
        const py = cy + Math.sin(p.currentAngle) * p.orbitRadius;

        ctx.beginPath();
        ctx.arc(px, py, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.alpha;
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 6;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1.0;
      });

      // 4. Energy flow line connecting (Claim -> Evidence -> Verdict)
      const claimX = cx + Math.cos(angle * 0.7 - 2.1) * 115;
      const claimY = cy + Math.sin(angle * 0.7 - 2.1) * 115;
      const verdictX = cx + Math.cos(angle * 0.7 + 1.2) * 115;
      const verdictY = cy + Math.sin(angle * 0.7 + 1.2) * 115;

      ctx.beginPath();
      ctx.moveTo(claimX, claimY);
      ctx.quadraticCurveTo(cx, cy, verdictX, verdictY);
      ctx.strokeStyle = 'rgba(0, 245, 212, 0.28)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      angle += 0.012;
      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <section className="orbit-hero-panel">
      {/* Background Matrix Grid */}
      <div className="orbit-grid-matrix" />
      <div className="orbit-glow-radial" />

      {/* Left Copy & Actions */}
      <div className="orbit-hero-content">
        <div className="orbit-telemetry-badge">
          <span className="live-indicator-dot" />
          <span>FORENSIC SURVEILLANCE &bull; PHASE 3.2.5 CALIBRATED</span>
          <span className="telemetry-pill">{activeMode.toUpperCase()}</span>
        </div>

        <h1 className="orbit-hero-title">
          Autonomous <em>Evidence Forensics</em> &amp; Cross-Stance Verification.
        </h1>

        <p className="orbit-hero-desc">
          Rigorous dual-retrieval claim adjudication. Inspects multi-source consensus, isolates cherry-picked stance disparities, and scales confidence via empirical temperature calibration.
        </p>

        <div className="orbit-hero-actions">
          <button className="orbit-primary-btn" onClick={onStartInvestigation} id="hero-start-investigation-btn">
            <span className="btn-icon-pulse">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
            </span>
            <span>Launch Investigation</span>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="5" y1="12" x2="19" y2="12" />
              <polyline points="13 6 19 12 13 18" />
            </svg>
          </button>

          <div className="orbit-stat-chip">
            <span className="stat-label">AUDITED ACCURACY</span>
            <strong className="stat-val">59.38%</strong>
            <span className="stat-sub">Held-out N=256</span>
          </div>

          <div className="orbit-stat-chip">
            <span className="stat-label">SELECTIVE COVERAGE</span>
            <strong className="stat-val">94.53%</strong>
            <span className="stat-sub">14 claims deferred</span>
          </div>
        </div>

        {/* Live System State Footer */}
        <div className="orbit-status-strip">
          <div className="status-item">
            <span className="dot-green" />
            <span>NLI Engine: DeBERTa-v3</span>
          </div>
          <div className="status-sep">&bull;</div>
          <div className="status-item">
            <span className="dot-cyan" />
            <span>Retrieval: Hybrid RRF (BM25 + E5)</span>
          </div>
          <div className="status-sep">&bull;</div>
          <div className="status-item">
            <span className="dot-violet" />
            <span>Calibrator: T=4.1039</span>
          </div>
        </div>
      </div>

      {/* Right Canvas Orbit Instrumentation */}
      <div className="orbit-canvas-wrapper">
        <canvas ref={canvasRef} className="orbit-canvas" />

        {/* HUD Center Core */}
        <div className="orbit-core-node" title="VERITAS-X Adjudication Core">
          <div className="core-inner-ring" />
          <div className="core-brand-icon">VX</div>
          <span className="core-pulse-beacon" />
        </div>

        {/* Key Orbital Nodes */}
        <div
          className={`orbit-node node-claim ${hoveredNode === 'CLAIM' ? 'hovered' : ''}`}
          onMouseEnter={() => { setHoveredNode('CLAIM'); setActiveTelemetry('CLAIM'); }}
          onMouseLeave={() => setHoveredNode(null)}
        >
          <div className="node-pip" />
          <div className="node-tooltip">
            <strong>01 &bull; CLAIM HYPOTHESIS</strong>
            <span>Input proposition parsed into semantic embedding space</span>
          </div>
        </div>

        <div
          className={`orbit-node node-evidence ${hoveredNode === 'EVIDENCE' ? 'hovered' : ''}`}
          onMouseEnter={() => { setHoveredNode('EVIDENCE'); setActiveTelemetry('EVIDENCE'); }}
          onMouseLeave={() => setHoveredNode(null)}
        >
          <div className="node-pip pip-evidence" />
          <div className="node-tooltip">
            <strong>02 &bull; EVIDENCE DOCKET</strong>
            <span>Multi-passage RRF consensus &amp; stance disparity check</span>
          </div>
        </div>

        <div
          className={`orbit-node node-verdict ${hoveredNode === 'VERDICT' ? 'hovered' : ''}`}
          onMouseEnter={() => { setHoveredNode('VERDICT'); setActiveTelemetry('VERDICT'); }}
          onMouseLeave={() => setHoveredNode(null)}
        >
          <div className="node-pip pip-verdict" />
          <div className="node-tooltip">
            <strong>03 &bull; ADJUDICATED VERDICT</strong>
            <span>Calibrated confidence &amp; selective abstention gate</span>
          </div>
        </div>

        {/* Mini HUD readout box */}
        <div className="orbit-hud-readout">
          <div className="hud-line">
            <span className="hud-tag">STREAM</span>
            <span className="hud-value">{activeTelemetry} VECTOR</span>
          </div>
          <div className="hud-line">
            <span className="hud-tag">STATUS</span>
            <span className="hud-status-active">MONITORING</span>
          </div>
          <div className="hud-coords">
            <span>RRF-K: 60</span>
            <span>&theta;_C: 0.32</span>
            <span>&theta;_H: 0.94</span>
          </div>
        </div>
      </div>
    </section>
  );
}
