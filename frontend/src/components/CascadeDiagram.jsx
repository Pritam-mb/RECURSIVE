import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import useStore from '../store/useStore';
import { nodeKey, setCachedPosition } from '../utils/cascadeGraph';
import '../styles/analysis.css';

// Severity → state class. Colour is only used to encode state.
const SEVERITY_CLASS = {
  CRITICAL: 'an-sev-critical',
  WARNING: 'an-sev-warning',
  WATCH: 'an-sev-watch',
  NOMINAL: 'an-sev-nominal',
  CASCADE: 'an-sev-cascade',
};
const LEGEND = ['CRITICAL', 'WARNING', 'WATCH', 'NOMINAL'];

const EDGE_CLASS = {
  active: 'an-edge--active',
  cascade: 'an-edge--cascade',
  resolved: 'an-edge--resolved',
};

const EMPTY_GRAPH = { nodes: [], edges: [] };

// Force-layout constants (same model as the previous canvas simulation)
const REPULSION = 2500;
const SPRING_K = 0.04;
const DAMPING = 0.85;
const CENTER_GRAVITY = 0.02;
const SETTLE_VELOCITY = 0.02;

const PAD_X = 48; // room for node labels
const LABEL_CHAR_W = 6; // ~10px mono glyph width
const LABEL_H = 12;
const SEVERITY_RANK = { CRITICAL: 3, WARNING: 2, WATCH: 1 };

function shortLabel(text) {
  if (!text) return '';
  return text.length > 12 ? `${text.slice(0, 11)}…` : text;
}

/**
 * Pick which node names to draw: most severe first, skipping any label whose
 * box would overlap one already placed or another node's dot. Hovering a node
 * still shows its full name in the tooltip.
 */
function pickLabelledNodes(nodes) {
  const placed = [];
  const keep = new Set();
  const ordered = [...nodes].sort(
    (a, b) => (SEVERITY_RANK[b.severity] ?? 0) - (SEVERITY_RANK[a.severity] ?? 0),
  );
  for (const n of ordered) {
    const w = shortLabel(n.label).length * LABEL_CHAR_W;
    const box = { x: n.x - (w / 2), y: n.y + n.r + 2, w, h: LABEL_H };
    const hitsLabel = placed.some((p) => box.x < p.x + p.w && box.x + box.w > p.x && box.y < p.y + p.h && box.y + box.h > p.y);
    const hitsDot = nodes.some((m) => m !== n
      && m.x + m.r > box.x && m.x - m.r < box.x + box.w
      && m.y + m.r > box.y && m.y - m.r < box.y + box.h);
    if (hitsLabel || hitsDot) continue;
    placed.push(box);
    keep.add(nodeKey(n.id));
  }
  return keep;
}
const PAD_Y = 28;

const nodeRadius = (cpi) => 3.5 + Math.min(Math.max(Number(cpi) || 0, 0), 10) * 0.45;

/**
 * Runs the force simulation synchronously until it settles (or hits the
 * iteration cap) and returns centre-relative positions keyed by nodeKey.
 * Called once per graph identity — no per-frame animation loop.
 */
function computeLayout(graph) {
  const nodes = graph.nodes.map((n) => ({ id: n.id, x: n.x ?? 0, y: n.y ?? 0, vx: 0, vy: 0 }));
  const byKey = new Map(nodes.map((n) => [nodeKey(n.id), n]));
  const links = [];
  for (const e of graph.edges) {
    const src = byKey.get(nodeKey(e.source));
    const tgt = byKey.get(nodeKey(e.target));
    if (src && tgt && src !== tgt) links.push({ src, tgt, p: Number(e.p_collision) || 0 });
  }

  const maxIter = nodes.length > 120 ? 150 : 400;
  for (let iter = 0; iter < maxIter; iter++) {
    for (const n of nodes) {
      n.vx += -n.x * CENTER_GRAVITY;
      n.vy += -n.y * CENTER_GRAVITY;
      for (const m of nodes) {
        if (m === n) continue;
        const dx = n.x - m.x;
        const dy = n.y - m.y;
        const force = REPULSION / Math.max(dx * dx + dy * dy, 1);
        n.vx += dx * force;
        n.vy += dy * force;
      }
    }
    for (const { src, tgt, p } of links) {
      const dx = tgt.x - src.x;
      const dy = tgt.y - src.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const force = (dist - (100 + (1 - p) * 40)) * SPRING_K;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      src.vx += fx; src.vy += fy;
      tgt.vx -= fx; tgt.vy -= fy;
    }
    let maxV = 0;
    for (const n of nodes) {
      n.vx *= DAMPING;
      n.vy *= DAMPING;
      n.x += n.vx;
      n.y += n.vy;
      maxV = Math.max(maxV, Math.abs(n.vx), Math.abs(n.vy));
    }
    if (maxV < SETTLE_VELOCITY) break;
  }

  const positions = new Map();
  for (const n of nodes) {
    // Remember centre-relative position so rebuilt graphs / remounts keep it
    setCachedPosition(n.id, n.x, n.y);
    positions.set(nodeKey(n.id), { x: n.x, y: n.y });
  }
  return positions;
}

const formatDistance = (km) => (km < 1 ? `${(km * 1000).toFixed(0)} m` : `${km.toFixed(1)} km`);

const edgeLabel = (e) => {
  const parts = [];
  if (e.miss_distance > 0) parts.push(formatDistance(e.miss_distance));
  if (e.tca_hours != null && e.tca_hours > 0) {
    parts.push(e.tca_hours < 1 ? `T-${(e.tca_hours * 60).toFixed(0)}m` : `T-${e.tca_hours.toFixed(1)}h`);
  } else if (e.tca_utc) {
    parts.push(`${e.tca_utc.slice(11, 16)}Z`);
  }
  return parts.join(' · ');
};

export default function CascadeDiagram({ graph = EMPTY_GRAPH }) {
  const bodyRef = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [tooltip, setTooltip] = useState(null);
  const setSelectedSatId = useStore((s) => s.setSelectedSatelliteId);
  const setSelectedAlertId = useStore((s) => s.setSelectedAlertId);
  const selectedSatId = useStore((s) => s.selectedSatelliteId);
  const selectedAlertId = useStore((s) => s.selectedAlertId);

  // Track container size (rounded, so sub-pixel jitter doesn't re-render)
  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return undefined;
    const measure = () => {
      const w = Math.round(el.clientWidth);
      const h = Math.round(el.clientHeight);
      setSize((prev) => (prev.w === w && prev.h === h ? prev : { w, h }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Layout is memoised by graph identity; resize only re-projects.
  const positions = useMemo(() => computeLayout(graph), [graph]);

  const view = useMemo(() => {
    const { w, h } = size;
    if (!w || !h || graph.nodes.length === 0) return null;

    let minX = Infinity; let maxX = -Infinity; let minY = Infinity; let maxY = -Infinity;
    for (const p of positions.values()) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    }
    const spanX = Math.max(maxX - minX, 1);
    const spanY = Math.max(maxY - minY, 1);
    const scale = Math.min(3, Math.max(w - PAD_X * 2, 1) / spanX, Math.max(h - PAD_Y * 2, 1) / spanY);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;

    const project = (p) => ({ x: w / 2 + (p.x - cx) * scale, y: h / 2 + (p.y - cy) * scale });

    const nodes = graph.nodes.map((n) => {
      const pos = positions.get(nodeKey(n.id)) ?? { x: 0, y: 0 };
      return { ...n, ...project(pos), r: nodeRadius(n.cpi) };
    });
    const byKey = new Map(nodes.map((n) => [nodeKey(n.id), n]));
    const edges = [];
    for (const e of graph.edges) {
      const src = byKey.get(nodeKey(e.source));
      const tgt = byKey.get(nodeKey(e.target));
      if (src && tgt) edges.push({ e, src, tgt, label: edgeLabel(e) });
    }
    return { nodes, edges, labelled: pickLabelledNodes(nodes) };
  }, [graph, positions, size]);

  const showTooltip = useCallback((evt, n) => {
    const rect = bodyRef.current?.getBoundingClientRect();
    if (!rect) return;
    setTooltip({
      x: evt.clientX - rect.left + 12,
      y: evt.clientY - rect.top - 8,
      label: n.label,
      agency: n.agency,
      cpi: n.cpi,
      severity: n.severity,
    });
  }, []);

  const hasGraph = graph.nodes.length > 0;

  return (
    <section className="ui-panel an-panel an-cascade">
      <header className="ui-panel-header">
        <span className="ui-label an-title">Cascade Risk Graph</span>
        <div className="an-header-meta">
          <div className="an-legend">
            {LEGEND.map((k) => (
              <span className="an-legend-item" key={k}>
                <span className={`an-legend-swatch ${SEVERITY_CLASS[k]}`} />
                {k}
              </span>
            ))}
          </div>
          {hasGraph && (
            <span className="an-count">
              {graph.nodes.length}N / {graph.edges.length}E
            </span>
          )}
        </div>
      </header>

      <div
        className="an-graph-body"
        ref={bodyRef}
        onMouseLeave={() => setTooltip(null)}
      >
        {view && (
          <svg
            className="an-graph-svg"
            width={size.w}
            height={size.h}
            viewBox={`0 0 ${size.w} ${size.h}`}
            role="img"
            aria-label="Cascade risk graph"
          >
            <g>
              {view.edges.map(({ e, src, tgt }) => (
                <g
                  className="an-edge-group"
                  key={e.id}
                  onClick={() => setSelectedAlertId(e.id)}
                >
                  <line className="an-edge-hit" x1={src.x} y1={src.y} x2={tgt.x} y2={tgt.y} />
                  <line
                    className={`an-edge ${EDGE_CLASS[e.status] ?? EDGE_CLASS.active}${selectedAlertId === e.id ? ' is-selected' : ''}`}
                    x1={src.x}
                    y1={src.y}
                    x2={tgt.x}
                    y2={tgt.y}
                  />
                </g>
              ))}
            </g>
            <g>
              {view.edges.map(({ e, src, tgt, label }) => (label && selectedAlertId === e.id ? (
                <text
                  key={`l-${e.id}`}
                  className="an-edge-label"
                  x={(src.x + tgt.x) / 2}
                  y={(src.y + tgt.y) / 2 - 4}
                  textAnchor="middle"
                >
                  {label}
                </text>
              ) : null))}
            </g>
            <g>
              {view.nodes.map((n) => {
                const sevClass = SEVERITY_CLASS[n.severity] ?? 'an-sev-unknown';
                const selected = selectedSatId != null && nodeKey(selectedSatId) === nodeKey(n.id);
                const label = selected || view.labelled.has(nodeKey(n.id)) ? shortLabel(n.label) : '';
                return (
                  <g
                    key={nodeKey(n.id)}
                    className={`an-node ${sevClass}`}
                    onClick={() => setSelectedSatId(n.id)}
                    onMouseMove={(evt) => showTooltip(evt, n)}
                    onMouseLeave={() => setTooltip(null)}
                  >
                    <circle cx={n.x} cy={n.y} r={n.r + 5} fill="transparent" />
                    {selected && <circle className="an-node-ring" cx={n.x} cy={n.y} r={n.r + 3} />}
                    <circle
                      className="an-node-dot"
                      cx={n.x}
                      cy={n.y}
                      r={n.r}
                      fill="currentColor"
                      stroke="currentColor"
                    />
                    {label && (
                      <text className="an-node-label" x={n.x} y={n.y + n.r + 11} textAnchor="middle">
                        {label}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          </svg>
        )}

        {!hasGraph && (
          <div className="an-empty">
            <span className="ui-label">No active cascade scenarios</span>
            <span className="an-empty-sub">Load the cascade demo to see cascade resolution</span>
          </div>
        )}

        {tooltip && (
          <div className="an-tooltip" style={{ left: Math.max(4, Math.min(tooltip.x, size.w - 170)), top: Math.max(4, tooltip.y) }}>
            <div className="an-tooltip-name">{tooltip.label}</div>
            <span className="an-tooltip-k">CPI</span>
            <span className="an-tooltip-v">{(tooltip.cpi ?? 0).toFixed(1)}</span>
            <span className="an-tooltip-k">State</span>
            <span className={`an-tooltip-v ${SEVERITY_CLASS[tooltip.severity] ?? 'an-sev-unknown'}`}>{tooltip.severity}</span>
            {tooltip.agency && (
              <>
                <span className="an-tooltip-k">Agency</span>
                <span className="an-tooltip-v">{tooltip.agency}</span>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
