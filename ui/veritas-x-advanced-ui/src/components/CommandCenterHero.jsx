import React from 'react';
import { Icon } from './Icons';
import { VeritasSymbol } from './VeritasLogo';

export function CommandCenterHero({
  onStartInvestigation,
  onQuickLoadClaim,
  apiHealth,
  mode,
  historyCount = 0,
}) {
  const QUICK_DEMO_CLAIMS = [
    {
      title: 'Trump Endorsement',
      claim: "The New York Post endorsed Donald Trump's bid for re-election.",
      label: 'VERIFIED_TRUE',
      badgeClass: 'verdict-true',
    },
    {
      title: 'Barrett Confirmation Date',
      claim: "Amy Coney Barrett was confirmed as US Supreme Court Justice on October 26, 2020.",
      label: 'INSUFFICIENT',
      badgeClass: 'verdict-insufficient',
    },
    {
      title: 'Subjective Comparison',
      claim: "this veritas x is better than chatgpt",
      label: 'ABSTAINED',
      badgeClass: 'verdict-abstained',
    },
    {
      title: 'Water Freezing Point',
      claim: "Pure water freezes at 0 degrees Celsius at sea level.",
      label: 'VERIFIED_TRUE',
      badgeClass: 'verdict-true',
    },
    {
      title: 'Moon Core Fusion',
      claim: "The Moon is a star that produces its own light through nuclear fusion.",
      label: 'VERIFIED_FALSE',
      badgeClass: 'verdict-false',
    },
  ];

  return (
    <div className="command-hero-root" role="region" aria-label="Forensic Mission Briefing">
      <div className="command-hero-grid">
        {/* Main Mission Briefing Column */}
        <div className="command-hero-briefing">
          <div className="hero-kicker-row">
            <span className="hero-kicker-tag">Automated Fact Verification</span>
            <span className="hero-version-pill">Veritas-X 2.0</span>
          </div>

          <h2 className="hero-headline">
            Fact-Check Statements with Transparent Evidence
          </h2>

          <p className="hero-editorial-copy">
            VERITAS-X verifies factual statements by comparing them directly against authoritative documentary sources. Instead of generating unverified text or guesses, the system retrieves relevant evidence, tests whether cited passages support or contradict your statement, detects conflicting sources, and explains every verdict with traceable citations.
          </p>

          <div className="hero-core-capabilities-row">
            <div className="capability-chip">
              <span className="chip-bullet" />
              <span>Multi-Hop Claim Decomposition</span>
            </div>
            <div className="capability-chip">
              <span className="chip-bullet" />
              <span>Interactive Evidence Graph</span>
            </div>
            <div className="capability-chip">
              <span className="chip-bullet" />
              <span>Time-Aware Provenance Timeline</span>
            </div>
            <div className="capability-chip">
              <span className="chip-bullet" />
              <span>Evidence-Aware Calibrated Abstention</span>
            </div>
          </div>

          <div className="hero-actions-row">
            <button
              type="button"
              className="hero-primary-btn"
              onClick={onStartInvestigation}
            >
              <Icon name="scan" size={17} />
              <span>Open Investigation Workspace</span>
              <span className="btn-arrow">→</span>
            </button>
          </div>
        </div>

        {/* Live Operational Status Card */}
        <div className="command-hero-status-column">
          <div className="live-status-card">
            <div className="status-card-header">
              <div className="card-header-left">
                <VeritasSymbol size={28} />
                <span className="status-card-title">Runtime Telemetry</span>
              </div>
              <span className={`status-live-badge ${apiHealth?.online ? (apiHealth?.modelReady ? 'badge-online' : 'badge-warming') : 'badge-offline'}`}>
                {apiHealth?.online ? (apiHealth?.modelReady ? 'Engine Ready' : 'Model Warming') : 'API Standby'}
              </span>
            </div>

            <div className="status-card-metrics">
              <div className="card-metric-row">
                <span className="metric-label">Local Evidence Corpus</span>
                <strong className="metric-value">1,047 Canonical Passages</strong>
              </div>
              <div className="card-metric-row">
                <span className="metric-label">Inference Compute Device</span>
                <strong className="metric-value" style={{ color: 'var(--primary)' }}>
                  {(apiHealth?.device || 'cpu').toUpperCase()}
                </strong>
              </div>
              <div className="card-metric-row">
                <span className="metric-label">Verification Cross-Encoder</span>
                <strong className="metric-value">DeBERTa-v3 Base MNLI/FEVER</strong>
              </div>
              <div className="card-metric-row">
                <span className="metric-label">Temperature Calibrator</span>
                <strong className="metric-value">T = 4.1039 (ECE: 11.83%)</strong>
              </div>
              <div className="card-metric-row">
                <span className="metric-label">Session Investigations</span>
                <strong className="metric-value">{historyCount} Recorded</strong>
              </div>
            </div>

            {/* Benchmark Quick Launch */}
            <div className="quick-launch-section">
              <span className="quick-launch-heading">Benchmark Demonstration Propositions</span>
              <div className="quick-launch-grid">
                {QUICK_DEMO_CLAIMS.map((item) => (
                  <button
                    key={item.title}
                    type="button"
                    className="quick-claim-item"
                    onClick={() => onQuickLoadClaim(item.claim)}
                  >
                    <div className="quick-claim-top">
                      <span className="quick-claim-title">{item.title}</span>
                      <span className={`quick-verdict-pill ${item.badgeClass}`}>{item.label}</span>
                    </div>
                    <p className="quick-claim-text">"{item.claim}"</p>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CommandCenterHero;
