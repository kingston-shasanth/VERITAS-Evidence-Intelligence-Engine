import React, { useState, useMemo } from 'react';
import { Icon } from './Icons.jsx';

/**
 * VERITAS-X 2.0 Provenance-Aware Evidence Timeline
 * Renders chronological events with verified date provenance.
 * Never invents timestamps or assumes system time for missing dates.
 */
function TimelineViewComponent({
  claim,
  temporal = null,
  evidence = [],
  onSelectEvidence = null,
  onNavigateToDossier = null,
}) {
  const [filterStance, setFilterStance] = useState('all');

  const events = useMemo(() => {
    if (temporal && Array.isArray(temporal.timeline_events)) {
      return temporal.timeline_events;
    }
    return [];
  }, [temporal]);

  const filteredEvents = useMemo(() => {
    if (filterStance === 'all') return events;
    return events.filter((ev) => {
      const st = String(ev.stance || '').toLowerCase();
      if (filterStance === 'support') return ['support', 'entailment'].includes(st);
      if (filterStance === 'refute') return ['refute', 'contradiction'].includes(st);
      if (filterStance === 'neutral') return st === 'neutral';
      return true;
    });
  }, [events, filterStance]);

  const temporalStatus = temporal?.temporal_status || 'UNRESOLVED';
  const temporalScope = temporal?.temporal_scope || 'UNSPECIFIED';
  const explanation = temporal?.explanation || 'Temporal status unresolved: evidence passages lack verifiable timestamps.';

  return (
    <div className="timeline-view-container" role="region" aria-label="Evidence Timeline and Provenance">
      {/* Header Bar */}
      <div className="timeline-header-bar">
        <div>
          <div className="timeline-badge-row">
            <span className="timeline-subhead">CHRONOLOGICAL PROVENANCE</span>
            <span className={`status-pill status-${temporalStatus.toLowerCase()}`}>
              {temporalStatus.replace('_', ' ')}
            </span>
            <span className="scope-pill">SCOPE: {temporalScope}</span>
          </div>
          <h4 className="timeline-title">Time-Aware Evidence & Event Sequence</h4>
        </div>

        {/* Filters */}
        <div className="timeline-filters">
          <div className="filter-pill-group">
            {['all', 'support', 'refute', 'neutral'].map((st) => (
              <button
                key={st}
                type="button"
                className={`timeline-pill-btn ${filterStance === st ? 'active' : ''}`}
                onClick={() => setFilterStance(st)}
              >
                {st.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Forensic Temporal Rationale Banner */}
      <div className={`timeline-rationale-banner banner-${temporalStatus.toLowerCase()}`}>
        <Icon name={temporalStatus === 'CONSISTENT' ? 'check' : temporalStatus === 'OUTDATED_SUPERSEDED' ? 'clock' : 'alert'} size={16} />
        <p className="timeline-rationale-text">{explanation}</p>
      </div>

      {/* Timeline Sequence */}
      {filteredEvents.length > 0 ? (
        <>
          <div className="timeline-track-wrapper">
            <div className="timeline-vertical-spine" />
            <div className="timeline-event-list">
              {filteredEvents.map((ev, idx) => {
                const stance = String(ev.stance || 'neutral').toLowerCase();
                const stanceLabel = stance === 'support' ? 'SUPPORTS' : stance === 'refute' ? 'CONTRADICTS' : 'NEUTRAL';
                return (
                  <div key={idx} className={`timeline-card-node node-${stance}`}>
                    {/* Marker Dot on Spine */}
                    <div className={`timeline-dot dot-${stance}`} />

                    {/* Card Body */}
                    <div className="timeline-card">
                      <div className="timeline-card-header">
                        <span className="timeline-date-chip">
                          <Icon name="clock" size={12} /> {ev.date}
                        </span>
                        <span className={`timeline-type-chip type-${ev.date_type}`}>
                          {ev.date_type === 'publication' ? 'PUBLISHED' : 'EVENT DATE'}
                        </span>
                        <span className={`timeline-stance-badge badge-${stance}`}>
                          {stanceLabel}
                        </span>
                        <span className="timeline-provenance-tag">
                          Source: <code>{ev.provenance || 'unknown'}</code>
                        </span>
                      </div>

                      <p className="timeline-passage-excerpt">"{ev.text}"</p>

                      <div className="timeline-card-footer">
                        <span className="evidence-ref-id">Citation: {ev.evidence_id}</span>
                        <button
                          type="button"
                          className="btn-inspect-timeline-evidence"
                          onClick={() => {
                            const matched = evidence.find((e) => e.id === ev.evidence_id);
                            if (matched && onSelectEvidence) onSelectEvidence(matched);
                            if (onNavigateToDossier) onNavigateToDossier();
                          }}
                        >
                          <Icon name="external" size={12} /> Inspect in Dossier
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Dynamic Timeline Interpretation Card */}
          <div className="dynamic-interpretation-box" style={{
            marginTop: '1.25rem',
            padding: '0.85rem 1rem',
            borderRadius: 'var(--radius-sm, 6px)',
            background: 'var(--bg-tertiary, #1f2937)',
            border: '1px solid var(--border-default, #374151)',
            fontSize: '0.86rem',
            lineHeight: 1.5,
            color: 'var(--text-secondary, #d1d5db)',
          }}>
            <strong style={{ color: 'var(--text-primary, #f9fafb)', display: 'block', marginBottom: '0.25rem' }}>
              Dynamic Timeline Interpretation:
            </strong>
            {temporalStatus === 'CONSISTENT'
              ? `All ${filteredEvents.length} dated evidence passages agree chronologically. The facts have remained stable across the documented timeframe.`
              : temporalStatus === 'OUTDATED_SUPERSEDED'
              ? `Evidence spans different points in time. Newer reports contradict or update older disclosures, showing that circumstances changed over the recorded dates.`
              : `${filteredEvents.length} dated event(s) recorded. Compare the dates above to see how reports developed chronologically.`
            }
          </div>
        </>
      ) : (
        <div className="timeline-empty-state">
          <Icon name="clock" size={28} />
          <h5>No Specific Dates Found in Passages</h5>
          <p>
            The retrieved evidence passages do not mention specific publication dates or event timestamps.
            VERITAS-X never fabricates timestamps or guesses dates when they are not explicitly present in the source text.
          </p>
        </div>
      )}
    </div>
  );
}

export const TimelineView = React.memo(TimelineViewComponent);
export default TimelineView;
