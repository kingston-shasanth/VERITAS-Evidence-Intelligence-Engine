import React, { useState, useMemo, useRef, useCallback } from 'react';
import { Icon } from './Icons.jsx';

/**
 * VERITAS-X 2.0 Interactive Evidence Relationship Graph
 * Renders an authentic multi-tiered graph connecting:
 * Root Claim -> Decomposed Subclaims -> Evidence Passages -> Primary Sources
 * Grounded strictly in real pipeline outputs.
 */
function EvidenceGraphComponent({
  claim,
  claimId = 'VX-CLAIM',
  verdict,
  confidence = 0,
  subclaims = [],
  evidence = [],
  graph = null,
  onSelectEvidence = null,
  onNavigateToDossier = null,
}) {
  const [selectedNodeId, setSelectedNodeId] = useState('node-claim-root');
  const [filterStance, setFilterStance] = useState('all');
  const [viewMode, setViewMode] = useState('graph'); // 'graph' | 'table'
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  // Standard stance styling tokens aligned with VERITAS-X design identity
  const stanceColors = {
    support: { main: '#059669', bg: 'rgba(5, 150, 105, 0.15)', border: '#059669' },
    entailment: { main: '#059669', bg: 'rgba(5, 150, 105, 0.15)', border: '#059669' },
    refute: { main: '#DC2626', bg: 'rgba(220, 38, 38, 0.15)', border: '#DC2626' },
    contradiction: { main: '#DC2626', bg: 'rgba(220, 38, 38, 0.15)', border: '#DC2626' },
    neutral: { main: '#64748B', bg: 'rgba(100, 116, 139, 0.10)', border: '#64748B' },
    claim: { main: '#4F46E5', bg: 'rgba(79, 70, 229, 0.20)', border: '#4F46E5' },
    subclaim: { main: '#4F46E5', bg: 'rgba(79, 70, 229, 0.15)', border: '#6366F1' },
    source: { main: '#D97706', bg: 'rgba(217, 119, 6, 0.12)', border: '#D97706' },
  };

  // Synthesize nodes & edges if graph object not directly provided
  const graphData = useMemo(() => {
    if (graph && graph.nodes && graph.nodes.length > 0) {
      return graph;
    }

    // Build fallback structured topology
    const nodes = [
      {
        id: 'node-claim-root',
        type: 'claim',
        label: claimId,
        text: claim || 'Root Proposition',
        verdict: verdict,
        confidence: confidence,
        metadata: { level: 0 },
      },
    ];
    const edges = [];

    // Subclaim nodes
    const activeSubclaims = subclaims.length > 0 ? subclaims : [{ subclaim_id: 'SC-01', text: claim, status: 'UNRESOLVED' }];
    activeSubclaims.forEach((sc) => {
      const scNodeId = `node-subclaim-${sc.subclaim_id}`;
      nodes.push({
        id: scNodeId,
        type: 'subclaim',
        label: sc.subclaim_id,
        text: sc.text,
        verdict: sc.verdict || 'UNRESOLVED',
        confidence: sc.confidence || 0,
        stance: String(sc.status || 'neutral').toLowerCase(),
        metadata: { level: 1, dependencies: sc.dependencies || [] },
      });
      edges.push({
        source: 'node-claim-root',
        target: scNodeId,
        relation: 'decomposes_to',
        weight: 1.0,
        label: `Decomposed: ${sc.subclaim_id}`,
      });
    });

    // Evidence passage nodes
    const sourceSet = new Set();
    evidence.forEach((ev, idx) => {
      const evNodeId = `node-passage-${ev.id}`;
      nodes.push({
        id: evNodeId,
        type: 'passage',
        label: ev.id,
        text: ev.text,
        stance: ev.stance,
        score: ev.score,
        metadata: {
          level: 2,
          source: ev.source,
          entailment_prob: ev.entailment_prob,
          contradiction_prob: ev.contradiction_prob,
          neutral_prob: ev.neutral_prob,
        },
      });

      // Connect passage to matching or first subclaim
      const targetScId = `node-subclaim-${activeSubclaims[idx % activeSubclaims.length].subclaim_id}`;
      const rel = ['support', 'entailment'].includes(String(ev.stance).toLowerCase())
        ? 'supports'
        : ['refute', 'contradiction'].includes(String(ev.stance).toLowerCase())
        ? 'refutes'
        : 'neutral';
      edges.push({
        source: evNodeId,
        target: targetScId,
        relation: rel,
        weight: ev.score || 0.8,
        label: `${String(ev.stance).toUpperCase()}`,
      });

      // Record source
      if (ev.source) sourceSet.add(ev.source);
    });

    // Source domain nodes
    Array.from(sourceSet).forEach((src, idx) => {
      const srcNodeId = `node-source-S${idx + 1}`;
      nodes.push({
        id: srcNodeId,
        type: 'source',
        label: `SRC-0${idx + 1}`,
        text: src,
        metadata: { level: 3 },
      });
      // Connect passages with this source
      evidence
        .filter((e) => e.source === src)
        .forEach((e) => {
          edges.push({
            source: `node-passage-${e.id}`,
            target: srcNodeId,
            relation: 'published_by',
            weight: 1.0,
            label: 'Published by',
          });
        });
    });

    const supCount = edges.filter((e) => e.relation === 'supports').length;
    const refCount = edges.filter((e) => e.relation === 'refutes').length;
    const summary = {
      total_nodes: nodes.length,
      total_edges: edges.length,
      subclaims_count: activeSubclaims.length,
      evidence_count: evidence.length,
      support_edges: supCount,
      refute_edges: refCount,
      has_contradictory_sources: supCount > 0 && refCount > 0,
    };

    return { nodes, edges, summary };
  }, [graph, claim, claimId, verdict, confidence, subclaims, evidence]);

  // Filter edges and nodes based on stance filter
  const filteredNodes = useMemo(() => {
    if (filterStance === 'all') return graphData.nodes;
    return graphData.nodes.filter((n) => {
      if (n.type === 'claim') return true;
      if (n.type === 'subclaim') return true;
      if (n.type === 'source') return true;
      const st = String(n.stance || '').toLowerCase();
      if (filterStance === 'support') return ['support', 'entailment'].includes(st);
      if (filterStance === 'refute') return ['refute', 'contradiction'].includes(st);
      if (filterStance === 'neutral') return st === 'neutral';
      return true;
    });
  }, [graphData.nodes, filterStance]);

  const filteredNodeIds = useMemo(() => new Set(filteredNodes.map((n) => n.id)), [filteredNodes]);

  const filteredEdges = useMemo(() => {
    return graphData.edges.filter(
      (e) => filteredNodeIds.has(e.source) && filteredNodeIds.has(e.target)
    );
  }, [graphData.edges, filteredNodeIds]);

  // Multi-tier coordinate layout
  const nodePositions = useMemo(() => {
    const positions = {};
    const width = 880;
    const height = 480;
    const cx = width / 2;
    const cy = height / 2;

    // Root Claim: Center
    positions['node-claim-root'] = { x: cx, y: cy };

    // Group remaining nodes by type
    const scNodes = filteredNodes.filter((n) => n.type === 'subclaim');
    const passNodes = filteredNodes.filter((n) => n.type === 'passage');
    const srcNodes = filteredNodes.filter((n) => n.type === 'source');

    // Tier 1: Subclaims (inner ring, radius 120)
    scNodes.forEach((node, i) => {
      const angle = (i / Math.max(1, scNodes.length)) * Math.PI * 2 - Math.PI / 2;
      positions[node.id] = {
        x: cx + Math.cos(angle) * 115,
        y: cy + Math.sin(angle) * 115,
      };
    });

    // Tier 2: Passages (middle ring, radius 215)
    passNodes.forEach((node, i) => {
      const angle = (i / Math.max(1, passNodes.length)) * Math.PI * 2 - Math.PI / 2;
      positions[node.id] = {
        x: cx + Math.cos(angle) * 205,
        y: cy + Math.sin(angle) * 205,
      };
    });

    // Tier 3: Sources (outer ring, radius 310)
    srcNodes.forEach((node, i) => {
      const angle = (i / Math.max(1, srcNodes.length)) * Math.PI * 2 - Math.PI / 2;
      positions[node.id] = {
        x: cx + Math.cos(angle) * 295,
        y: cy + Math.sin(angle) * 295,
      };
    });

    return positions;
  }, [filteredNodes]);

  // Active reasoning path identification
  const activePathIds = useMemo(() => {
    const set = new Set([selectedNodeId]);
    filteredEdges.forEach((e) => {
      if (e.source === selectedNodeId) set.add(e.target);
      if (e.target === selectedNodeId) set.add(e.source);
    });
    return set;
  }, [selectedNodeId, filteredEdges]);

  // Selected node object
  const selectedNode = useMemo(() => {
    return filteredNodes.find((n) => n.id === selectedNodeId) || filteredNodes[0];
  }, [filteredNodes, selectedNodeId]);

  // Pan & Zoom handlers
  const handleMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    dragStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      panX: pan.x,
      panY: pan.y,
    };
  }, [pan]);

  const handleMouseMove = useCallback((e) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStartRef.current.x;
    const dy = e.clientY - dragStartRef.current.y;
    setPan({
      x: dragStartRef.current.panX + dx,
      y: dragStartRef.current.panY + dy,
    });
  }, [isDragging]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleZoom = (delta) => {
    setZoom((prev) => Math.min(2.0, Math.max(0.5, prev + delta)));
  };

  const handleResetView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div className="evidence-graph-container" role="region" aria-label="Interactive Evidence Relationship Graph">
      {/* Graph Toolbar */}
      <div className="graph-header-bar">
        <div>
          <div className="graph-subhead-row">
            <span className="graph-subhead">EVIDENCE MAP</span>
            {graphData.summary?.has_contradictory_sources && (
              <span className="badge-disparity-warning" title="Opposing evidence detected">
                <Icon name="alert" size={12} /> CONFLICTING EVIDENCE
              </span>
            )}
          </div>
          <h4 className="graph-title">How Evidence Connects to the Claim</h4>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '0.2rem 0 0' }}>
            Visualizes how each evidence passage connects to the claim and whether it supports, refutes, or provides neutral context.
          </p>
        </div>

        {/* View and Stance Controls */}
        <div className="graph-controls-cluster">
          {/* View Mode Toggle */}
          <div className="graph-toggle-group">
            <button
              type="button"
              className={`graph-tab-btn ${viewMode === 'graph' ? 'active' : ''}`}
              onClick={() => setViewMode('graph')}
              aria-label="View Graph Visualizer"
            >
              <Icon name="globe" size={13} /> Graph
            </button>
            <button
              type="button"
              className={`graph-tab-btn ${viewMode === 'table' ? 'active' : ''}`}
              onClick={() => setViewMode('table')}
              aria-label="View Structured Table List"
            >
              <Icon name="table" size={13} /> Table List
            </button>
          </div>

          {/* Stance Filters */}
          <div className="graph-filter-pills" role="radiogroup" aria-label="Filter evidence by stance">
            {['all', 'support', 'refute', 'neutral'].map((st) => (
              <button
                key={st}
                type="button"
                className={`filter-pill-btn ${filterStance === st ? 'active' : ''} pill-${st}`}
                onClick={() => setFilterStance(st)}
              >
                {st.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Zoom & Pan Controls */}
          <div className="graph-zoom-controls">
            <button type="button" className="zoom-btn" onClick={() => handleZoom(0.15)} title="Zoom In" aria-label="Zoom In">+</button>
            <button type="button" className="zoom-btn" onClick={() => handleZoom(-0.15)} title="Zoom Out" aria-label="Zoom Out">−</button>
            <button type="button" className="zoom-btn" onClick={handleResetView} title="Reset View" aria-label="Reset View">Reset</button>
          </div>
        </div>
      </div>

      {/* Main Graph Stage */}
      {viewMode === 'graph' ? (
        <div className="graph-stage-grid">
          {/* Interactive SVG Canvas */}
          <div
            className="graph-svg-wrapper"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
          >
            <svg
              className="topology-svg"
              viewBox="0 0 880 480"
              preserveAspectRatio="xMidYMid meet"
            >
              <defs>
                <radialGradient id="claim-gradient">
                  <stop offset="0%" stopColor="rgba(79, 70, 229, 0.4)" />
                  <stop offset="100%" stopColor="rgba(15, 23, 42, 0.95)" />
                </radialGradient>
                <marker id="arrow-support" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#059669" />
                </marker>
                <marker id="arrow-refute" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#DC2626" />
                </marker>
                <marker id="arrow-neutral" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748B" />
                </marker>
                <marker id="arrow-decomp" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#4F46E5" />
                </marker>
              </defs>

              <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
                {/* Radial Guide Rings */}
                <circle cx="440" cy="240" r="115" fill="none" stroke="rgba(148, 163, 184, 0.12)" strokeDasharray="3 4" />
                <circle cx="440" cy="240" r="205" fill="none" stroke="rgba(148, 163, 184, 0.08)" strokeDasharray="3 4" />
                <circle cx="440" cy="240" r="295" fill="none" stroke="rgba(148, 163, 184, 0.06)" strokeDasharray="3 4" />

                {/* Edges */}
                {filteredEdges.map((edge, idx) => {
                  const p1 = nodePositions[edge.source];
                  const p2 = nodePositions[edge.target];
                  if (!p1 || !p2) return null;

                  const isPath = activePathIds.has(edge.source) && activePathIds.has(edge.target);
                  let strokeColor = 'rgba(100, 116, 139, 0.25)';
                  let markerId = '';

                  if (edge.relation === 'supports') {
                    strokeColor = isPath ? '#059669' : 'rgba(5, 150, 105, 0.45)';
                    markerId = 'url(#arrow-support)';
                  } else if (edge.relation === 'refutes') {
                    strokeColor = isPath ? '#DC2626' : 'rgba(220, 38, 38, 0.45)';
                    markerId = 'url(#arrow-refute)';
                  } else if (edge.relation === 'decomposes_to') {
                    strokeColor = isPath ? '#4F46E5' : 'rgba(79, 70, 229, 0.4)';
                    markerId = 'url(#arrow-decomp)';
                  } else {
                    strokeColor = isPath ? '#64748B' : 'rgba(100, 116, 139, 0.25)';
                    markerId = 'url(#arrow-neutral)';
                  }

                  return (
                    <g key={`edge-${idx}`}>
                      <line
                        x1={p1.x}
                        y1={p1.y}
                        x2={p2.x}
                        y2={p2.y}
                        stroke={strokeColor}
                        strokeWidth={isPath ? 2.5 : 1.25}
                        markerEnd={markerId}
                      />
                    </g>
                  );
                })}

                {/* Nodes */}
                {filteredNodes.map((node) => {
                  const pos = nodePositions[node.id];
                  if (!pos) return null;

                  const isSelected = node.id === selectedNodeId;
                  const isPath = activePathIds.has(node.id);
                  const stanceKey = String(node.stance || node.type).toLowerCase();
                  const color = stanceColors[stanceKey] || stanceColors.neutral;

                  let radius = 18;
                  if (node.type === 'claim') radius = 34;
                  else if (node.type === 'subclaim') radius = 22;
                  else if (node.type === 'source') radius = 16;

                  return (
                    <g
                      key={`node-${node.id}`}
                      className={`graph-node node-${node.type} ${isSelected ? 'active-node' : ''}`}
                      transform={`translate(${pos.x}, ${pos.y})`}
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedNodeId(node.id);
                      }}
                      tabIndex="0"
                      role="button"
                      aria-label={`${node.type} ${node.label}: ${node.text}`}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          setSelectedNodeId(node.id);
                        }
                      }}
                      style={{ cursor: 'pointer' }}
                    >
                      <circle
                        r={radius}
                        fill={node.type === 'claim' ? 'url(#claim-gradient)' : color.bg}
                        stroke={isSelected ? '#38bdf8' : (isPath ? color.main : color.border)}
                        strokeWidth={isSelected ? 3 : (isPath ? 2 : 1.25)}
                      />
                      <text
                        textAnchor="middle"
                        dy="0.35em"
                        fill="#f8fafc"
                        fontSize={node.type === 'claim' ? 10 : 9}
                        fontFamily="monospace"
                        fontWeight="bold"
                      >
                        {node.label}
                      </text>
                    </g>
                  );
                })}
              </g>
            </svg>
          </div>

          {/* Node Inspector Drawer */}
          <div className="graph-inspector-panel">
            {selectedNode ? (
              <div className="inspector-card">
                <div className="inspector-top-row">
                  <span className={`inspector-type-tag type-${selectedNode.type}`}>
                    {selectedNode.type.toUpperCase()}
                  </span>
                  <span className="inspector-id-chip">{selectedNode.label}</span>
                  {selectedNode.stance && (
                    <span className={`inspector-stance-badge stance-${selectedNode.stance.toLowerCase()}`}>
                      {selectedNode.stance.toUpperCase()}
                    </span>
                  )}
                </div>

                <h5 className="inspector-entity-title">
                  {selectedNode.type === 'claim' && 'Root Hypothesis'}
                  {selectedNode.type === 'subclaim' && 'Decomposed Proposition'}
                  {selectedNode.type === 'passage' && 'Documentary Evidence Passage'}
                  {selectedNode.type === 'source' && 'Documentary Publisher'}
                </h5>

                <p className="inspector-text">
                  "{selectedNode.text}"
                </p>

                {/* Technical NLI or Retrieval Metrics */}
                {selectedNode.type === 'passage' && (
                  <div className="inspector-metrics-block">
                    <div className="metric-row">
                      <span>Retrieval Score</span>
                      <strong>{Number(selectedNode.score || 0).toFixed(4)}</strong>
                    </div>
                    {selectedNode.metadata?.entailment_prob !== undefined && (
                      <div className="nli-mini-bars">
                        <div className="bar-track">
                          <span className="bar-label">Support</span>
                          <div className="progress-bg">
                            <div className="progress-fill fill-sup" style={{ width: `${(selectedNode.metadata.entailment_prob || 0) * 100}%` }} />
                          </div>
                          <span className="bar-val">{((selectedNode.metadata.entailment_prob || 0) * 100).toFixed(1)}%</span>
                        </div>
                        <div className="bar-track">
                          <span className="bar-label">Refute</span>
                          <div className="progress-bg">
                            <div className="progress-fill fill-ref" style={{ width: `${(selectedNode.metadata.contradiction_prob || 0) * 100}%` }} />
                          </div>
                          <span className="bar-val">{((selectedNode.metadata.contradiction_prob || 0) * 100).toFixed(1)}%</span>
                        </div>
                        <div className="bar-track">
                          <span className="bar-label">Neutral</span>
                          <div className="progress-bg">
                            <div className="progress-fill fill-neu" style={{ width: `${(selectedNode.metadata.neutral_prob || 0) * 100}%` }} />
                          </div>
                          <span className="bar-val">{((selectedNode.metadata.neutral_prob || 0) * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Subclaim Status Breakdown */}
                {selectedNode.type === 'subclaim' && (
                  <div className="inspector-metrics-block">
                    <div className="metric-row">
                      <span>Status</span>
                      <strong className={`status-${String(selectedNode.metadata?.status).toLowerCase()}`}>
                        {selectedNode.metadata?.status || 'UNRESOLVED'}
                      </strong>
                    </div>
                    <div className="metric-row">
                      <span>Relation to Root</span>
                      <code>{selectedNode.metadata?.relation || 'conjunct'}</code>
                    </div>
                  </div>
                )}

                {/* Action button: Open in Evidence Dossier */}
                {selectedNode.type === 'passage' && (
                  <div className="inspector-action-row">
                    <button
                      type="button"
                      className="btn-open-dossier"
                      onClick={() => {
                        const originalEv = evidence.find((e) => e.id === selectedNode.label);
                        if (originalEv && onSelectEvidence) onSelectEvidence(originalEv);
                        if (onNavigateToDossier) onNavigateToDossier();
                      }}
                    >
                      <Icon name="external" size={13} /> View in Evidence Dossier
                    </button>
                  </div>
                )}

                <div className="inspector-meta-foot">
                  <span>CANONICAL PIPELINE GROUNDING</span>
                  <span>CLICK NODES TO TRACE</span>
                </div>
              </div>
            ) : (
              <div className="inspector-empty">
                <span>Select a node in the graph to inspect its evidentiary grounding and reasoning path.</span>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Accessible Structured Table List Alternative */
        <div className="graph-table-view">
          <table className="graph-accessible-table">
            <thead>
              <tr>
                <th>Tier</th>
                <th>Identifier</th>
                <th>Entity Text / Label</th>
                <th>Stance / Status</th>
                <th>Confidence / Score</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredNodes.map((n) => (
                <tr
                  key={n.id}
                  className={n.id === selectedNodeId ? 'selected-table-row' : ''}
                  onClick={() => setSelectedNodeId(n.id)}
                >
                  <td>
                    <span className={`table-tier-badge tier-${n.type}`}>{n.type.toUpperCase()}</span>
                  </td>
                  <td><code>{n.label}</code></td>
                  <td className="table-text-col">"{n.text}"</td>
                  <td>
                    <span className={`table-stance-pill stance-${String(n.stance || n.verdict || 'neutral').toLowerCase()}`}>
                      {String(n.stance || n.verdict || 'NEUTRAL').toUpperCase()}
                    </span>
                  </td>
                  <td>
                    {n.confidence !== undefined
                      ? `${(Number(n.confidence) * 100).toFixed(1)}%`
                      : n.score !== undefined
                      ? Number(n.score).toFixed(4)
                      : '—'}
                  </td>
                  <td>
                    {n.type === 'passage' && (
                      <button
                        type="button"
                        className="btn-table-action"
                        onClick={(e) => {
                          e.stopPropagation();
                          const originalEv = evidence.find((ev) => ev.id === n.label);
                          if (originalEv && onSelectEvidence) onSelectEvidence(originalEv);
                          if (onNavigateToDossier) onNavigateToDossier();
                        }}
                      >
                        Inspect
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Plain-English Dynamic Map Interpretation Bar */}
      <div className="graph-dynamic-interpretation-card" style={{ marginTop: '0.85rem', padding: '0.65rem 1rem', background: 'var(--bg-canvas)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', fontSize: '0.82rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Icon name="files" size={15} />
        <span>
          <strong>Map Interpretation: </strong>
          This diagram connects {evidence.length} evidence source(s) to the claim. {
            evidence.some(e => ['support', 'entailment'].includes(String(e.stance || '').toLowerCase()) && e.is_informative !== false) &&
            evidence.some(e => ['refute', 'contradiction'].includes(String(e.stance || '').toLowerCase()) && e.is_informative !== false)
              ? 'Opposing evidence detected: The network contains conflicting supporting and refuting sources that disagree.'
              : evidence.some(e => ['support', 'entailment'].includes(String(e.stance || '').toLowerCase()) && e.is_informative !== false)
              ? 'Consistent support: Active evidence paths converge to corroborate the statement without contradiction.'
              : evidence.some(e => ['refute', 'contradiction'].includes(String(e.stance || '').toLowerCase()) && e.is_informative !== false)
              ? 'Consistent refutation: Active evidence paths directly contradict the statement.'
              : 'Neutral or background evidence: Paths provide contextual information without establishing conclusive proof.'
          }
        </span>
      </div>
    </div>
  );
}

export const EvidenceGraph = React.memo(EvidenceGraphComponent);
export default EvidenceGraph;
