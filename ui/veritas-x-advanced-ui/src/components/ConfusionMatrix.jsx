import React, { useState } from 'react';

const MATRIX_LABELS = [
  { key: 'TRUE', label: 'True', color: '#00f5d4' },
  { key: 'FALSE', label: 'False', color: '#f43f5e' },
  { key: 'CONFLICT', label: 'Conflict', color: '#a78bfa' },
  { key: 'INSUF', label: 'Insufficient', color: '#fbbf24' },
];

const AUDITED_MATRIX = [
  [30, 12, 2, 17],   // Row 0: True (N=61)
  [17, 107, 2, 31],  // Row 1: False (N=157)
  [3, 9, 2, 5],      // Row 2: Conflicting (N=19)
  [2, 3, 1, 13],     // Row 3: Insufficient (N=19)
];

const ROW_TOTALS = [61, 157, 19, 19];
const COL_TOTALS = [52, 131, 7, 66];

function ConfusionMatrixComponent() {
  const [hoveredCell, setHoveredCell] = useState(null);

  const maxVal = 107; // For heatmap intensity scaling

  return (
    <div className="confusion-matrix-wrapper">
      <div className="matrix-head-row">
        <div>
          <span className="matrix-sub">AUDITED ONE-SHOT TEST SET</span>
          <h4 className="matrix-heading">4 &times; 4 Confusion Matrix (N=256)</h4>
        </div>
        <div className="matrix-stats-pills">
          <span className="pill-metric">Diagonal Correct: <strong>152 / 256</strong> (59.38%)</span>
          <span className="pill-metric">Errors: <strong>104 / 256</strong> (40.62%)</span>
        </div>
      </div>

      <div className="matrix-grid-layout">
        {/* Y Axis Label */}
        <div className="matrix-y-label">
          <span>ACTUAL GROUND TRUTH &rarr;</span>
        </div>

        {/* The Matrix Body */}
        <div className="matrix-table-container">
          {/* Top Column Headers */}
          <div className="matrix-row matrix-col-headers">
            <div className="matrix-corner-cell" />
            {MATRIX_LABELS.map((col) => (
              <div key={`col-${col.key}`} className="matrix-col-header" style={{ color: col.color }}>
                {col.label}
              </div>
            ))}
            <div className="matrix-col-header matrix-total-header">Total</div>
          </div>

          {/* Matrix Data Rows */}
          {AUDITED_MATRIX.map((row, rIdx) => {
            const rowLabel = MATRIX_LABELS[rIdx];
            const rowTotal = ROW_TOTALS[rIdx];

            return (
              <div key={`row-${rowLabel.key}`} className="matrix-row">
                <div className="matrix-row-header" style={{ color: rowLabel.color }}>
                  {rowLabel.label}
                </div>

                {row.map((val, cIdx) => {
                  const isDiagonal = rIdx === cIdx;
                  const intensity = Math.min(1, val / maxVal);
                  const isHovered = hoveredCell?.r === rIdx && hoveredCell?.c === cIdx;

                  let bgColor = 'rgba(255, 255, 255, 0.03)';
                  let textColor = '#94a3b8';
                  let border = '1px solid rgba(148, 163, 184, 0.1)';

                  if (isDiagonal) {
                    bgColor = `rgba(0, 245, 212, ${0.12 + intensity * 0.45})`;
                    textColor = '#ffffff';
                    border = '1px solid rgba(0, 245, 212, 0.35)';
                  } else if (val > 0) {
                    bgColor = `rgba(244, 63, 94, ${0.05 + intensity * 0.25})`;
                    textColor = '#fda4af';
                  }

                  return (
                    <div
                      key={`cell-${rIdx}-${cIdx}`}
                      className={`matrix-cell ${isDiagonal ? 'cell-diagonal' : ''} ${isHovered ? 'cell-hovered' : ''}`}
                      style={{ background: bgColor, color: textColor, border }}
                      onMouseEnter={() => setHoveredCell({ r: rIdx, c: cIdx, val })}
                      onMouseLeave={() => setHoveredCell(null)}
                    >
                      <strong className="cell-num">{val}</strong>
                      <span className="cell-pct">{((val / rowTotal) * 100).toFixed(0)}%</span>
                    </div>
                  );
                })}

                <div className="matrix-cell matrix-total-cell">
                  <strong>{rowTotal}</strong>
                </div>
              </div>
            );
          })}

          {/* Bottom Totals Row */}
          <div className="matrix-row matrix-totals-row">
            <div className="matrix-row-header matrix-total-header">Predicted</div>
            {COL_TOTALS.map((total, idx) => (
              <div key={`col-total-${idx}`} className="matrix-cell matrix-total-cell">
                <strong>{total}</strong>
              </div>
            ))}
            <div className="matrix-cell matrix-total-cell matrix-grand-total">
              <strong>256</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Dynamic Hover Tooltip / Insight Bar */}
      <div className="matrix-telemetry-foot">
        {hoveredCell ? (
          <div className="matrix-hover-info">
            <span className="telemetry-tag">CELL INSPECTION</span>
            <span>
              Actual: <strong>{MATRIX_LABELS[hoveredCell.r].label}</strong> &rarr; Predicted:{' '}
              <strong>{MATRIX_LABELS[hoveredCell.c].label}</strong>: <strong>{hoveredCell.val} claims</strong> (
              {((hoveredCell.val / ROW_TOTALS[hoveredCell.r]) * 100).toFixed(1)}% of actual{' '}
              {MATRIX_LABELS[hoveredCell.r].label} claims)
            </span>
          </div>
        ) : (
          <div className="matrix-hover-info info-default">
            <span className="telemetry-tag">KEY FINDING</span>
            <span>
              Diagonal represents correct test classifications (N=152). 107 false claims correctly identified (68.2% recall). Conflicting recall is 10.53% (2/19).
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export const ConfusionMatrix = React.memo(ConfusionMatrixComponent);
export default ConfusionMatrix;
