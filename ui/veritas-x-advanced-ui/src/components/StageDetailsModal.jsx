import React, { useState } from 'react';
import { Icon } from './Icons.jsx';

/**
 * StageDetailsModal Component
 * Full-fidelity telemetry inspector allowing users to view the raw JSON event
 * for the active stage or the complete trace array.
 */
export function StageDetailsModal({
  isOpen = false,
  onClose,
  activeStageData = null,
  allStages = [],
}) {
  const [copied, setCopied] = useState(false);
  const [inspectMode, setInspectMode] = useState('CURRENT_STAGE'); // 'CURRENT_STAGE' | 'FULL_TRACE'

  if (!isOpen) return null;

  const displayData = inspectMode === 'CURRENT_STAGE' ? (activeStageData || {}) : allStages;
  const jsonString = JSON.stringify(displayData, null, 2);

  const handleCopy = () => {
    try {
      navigator.clipboard.writeText(jsonString);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.warn('Clipboard write failed:', e);
    }
  };

  return (
    <div className="telemetry-modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div className="telemetry-modal-container">
        {/* Modal Header */}
        <div className="telemetry-modal-header">
          <div className="modal-title-wrap">
            <Icon name="code" size={20} />
            <div>
              <h3 id="modal-title" className="modal-main-title">Telemetry &amp; Execution Trace Inspector</h3>
              <p className="modal-subtitle">
                Authentic backend JSON payload recorded by Phase 4A telemetry collector
              </p>
            </div>
          </div>

          <div className="modal-actions-right">
            {/* View Mode Toggle */}
            <div className="modal-toggle-tabs">
              <button
                type="button"
                className={`modal-tab ${inspectMode === 'CURRENT_STAGE' ? 'active' : ''}`}
                onClick={() => setInspectMode('CURRENT_STAGE')}
              >
                Active Stage {activeStageData?.sequence ? `#${activeStageData.sequence}` : ''}
              </button>
              <button
                type="button"
                className={`modal-tab ${inspectMode === 'FULL_TRACE' ? 'active' : ''}`}
                onClick={() => setInspectMode('FULL_TRACE')}
              >
                Full Trace ({allStages.length} stages)
              </button>
            </div>

            {/* Copy Button */}
            <button
              type="button"
              className="modal-copy-btn"
              onClick={handleCopy}
              title="Copy JSON to clipboard"
            >
              <Icon name={copied ? 'checkCircle' : 'copy'} size={15} />
              <span>{copied ? 'Copied' : 'Copy JSON'}</span>
            </button>

            {/* Close Button */}
            <button
              type="button"
              className="modal-close-btn"
              onClick={onClose}
              aria-label="Close Inspector"
            >
              <Icon name="x" size={20} />
            </button>
          </div>
        </div>

        {/* Modal Body: Code Display */}
        <div className="telemetry-modal-body">
          <pre className="telemetry-json-viewer">
            <code>{jsonString}</code>
          </pre>
        </div>

        {/* Modal Footer */}
        <div className="telemetry-modal-footer">
          <span className="telemetry-footer-note">
            Trace format: VERITAS-X Phase 4A Structured Telemetry Protocol v1.0.0
          </span>
          <button type="button" className="telemetry-done-btn" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

export default StageDetailsModal;
