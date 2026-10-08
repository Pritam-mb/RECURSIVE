import { useEffect, useRef, useState, useCallback } from 'react';
import useStore from '../store/useStore';
import { nodeKey, setCachedPosition } from '../utils/cascadeGraph';

const NODE_COLORS = {
  CRITICAL: '#ef4444',
  WARNING: '#eab308',
  WATCH: '#4a90d9',
  NOMINAL: '#22c55e',
  CASCADE: '#f97316',
};

const EDGE_COLORS = {
  active: '#ef4444',
  cascade: '#f97316',
  resolved: '#22c55e',
};

const REPULSION = 2500;
const SPRING_K = 0.04;
const DAMPING = 0.85;
const CENTER_GRAVITY = 0.02;

export default function CascadeDiagram({ graph = { nodes: [], edges: [] } }) {
  const canvasRef = useRef(null);
  const simRef = useRef({ nodes: [], edges: [], animId: null });
  const [tooltip, setTooltip] = useState(null);
  const setSelectedSatId = useStore((s) => s.setSelectedSatelliteId);
  const setSelectedAlertId = useStore((s) => s.setSelectedAlertId);

  // Sync graph to sim
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = canvas.offsetWidth || 400;
    const H = canvas.offsetHeight || 280;

    // Merge new nodes/edges, preserving positions
    const nodeMap = new Map(simRef.current.nodes.map((n) => [nodeKey(n.id), n]));

    // New nodes start at the graph's deterministic position (cached sim
    // position or NORAD-seeded offset, both centre-relative) — no Math.random().
    simRef.current.nodes = graph.nodes.map((n) => {
      const existing = nodeMap.get(nodeKey(n.id));
      return existing
        ? { ...n, x: existing.x, y: existing.y, vx: existing.vx, vy: existing.vy }
        : { ...n, x: W / 2 + (n.x ?? 0), y: H / 2 + (n.y ?? 0), vx: 0, vy: 0 };
    });

    simRef.current.edges = graph.edges;
  }, [graph]);

  // Force simulation + rendering loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    let frame = 0;

    const tick = () => {
      simRef.current.animId = requestAnimationFrame(tick);
      frame++;

      const { nodes, edges } = simRef.current;
      const W = canvas.width;
      const H = canvas.height;

      if (nodes.length === 0) {
        ctx.clearRect(0, 0, W, H);
        return;
      }

      // ── Force simulation ───────────────────────────────────────────────────
      for (const n of nodes) {
        // Center gravity
        n.vx += (W / 2 - n.x) * CENTER_GRAVITY;
        n.vy += (H / 2 - n.y) * CENTER_GRAVITY;

        // Repulsion between nodes
        for (const m of nodes) {
          if (m === n) continue;
          const dx = n.x - m.x;
          const dy = n.y - m.y;
          const dist2 = Math.max(dx * dx + dy * dy, 1);
          const force = REPULSION / dist2;
          n.vx += dx * force;
          n.vy += dy * force;
        }
      }

      // Spring attraction along edges
      for (const e of edges) {
        const src = nodes.find((n) => n.id === e.source);
        const tgt = nodes.find((n) => n.id === e.target);
        if (!src || !tgt) continue;
        const dx = tgt.x - src.x;
        const dy = tgt.y - src.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const ideal = 100 + (1 - e.p_collision) * 40;
        const force = (dist - ideal) * SPRING_K;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        src.vx += fx; src.vy += fy;
        tgt.vx -= fx; tgt.vy -= fy;
      }

      // Integrate positions
      for (const n of nodes) {
        n.vx *= DAMPING;
        n.vy *= DAMPING;
        n.x = Math.max(20, Math.min(W - 20, n.x + n.vx));
        n.y = Math.max(20, Math.min(H - 20, n.y + n.vy));
        // Remember centre-relative position so rebuilt graphs / remounts keep it
        setCachedPosition(n.id, n.x - W / 2, n.y - H / 2);
      }

      // ── Draw ───────────────────────────────────────────────────────────────
      ctx.clearRect(0, 0, W, H);

      // Draw edges
      for (const e of edges) {
        const src = nodes.find((n) => n.id === e.source);
        const tgt = nodes.find((n) => n.id === e.target);
        if (!src || !tgt) continue;

        const color = EDGE_COLORS[e.status] ?? '#ef4444';
        const thickness = Math.max(1, Math.min(6, e.p_collision * 1e4 + 1));

        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);
        ctx.strokeStyle = color;
        ctx.globalAlpha = e.status === 'resolved' ? 0.5 : 0.85;
        ctx.lineWidth = thickness;

        // Pulse cascade edges
        if (e.status === 'cascade') {
          ctx.globalAlpha = 0.5 + 0.35 * Math.sin(frame * 0.08);
        }

        ctx.stroke();

        // Draw distance + TCA labels on the edge midpoint
        const midX = (src.x + tgt.x) / 2;
        const midY = (src.y + tgt.y) / 2;

        // Distance label
        if (e.miss_distance > 0) {
          const distStr = e.miss_distance < 1.0
            ? `${(e.miss_distance * 1000).toFixed(0)}m`
            : `${e.miss_distance.toFixed(1)}km`;

          ctx.save();
          ctx.font = 'bold 8px JetBrains Mono, monospace';
          const textWidth = ctx.measureText(distStr).width;
          // pill background
          const ph = 12, pr = 3;
          ctx.fillStyle = 'rgba(6, 13, 20, 0.92)';
          ctx.beginPath();
          ctx.roundRect
            ? ctx.roundRect(midX - textWidth / 2 - pr, midY - ph / 2, textWidth + pr * 2, ph, pr)
            : ctx.fillRect(midX - textWidth / 2 - pr, midY - ph / 2, textWidth + pr * 2, ph);
          ctx.fill();
          ctx.strokeStyle = EDGE_COLORS[e.status] ?? '#ef4444';
          ctx.lineWidth = 0.5;
          ctx.stroke();

          ctx.fillStyle = e.status === 'active' ? '#fca5a5'
            : e.status === 'cascade' ? '#fdba74'
            : '#86efac';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(distStr, midX, midY);
          ctx.restore();
        }

        // TCA label (above distance)
        if (e.tca_hours != null && e.tca_hours > 0) {
          const tcaStr = e.tca_hours < 1
            ? `${(e.tca_hours * 60).toFixed(0)}min`
            : `${e.tca_hours.toFixed(1)}h`;
          ctx.save();
          ctx.font = '7px JetBrains Mono, monospace';
          const tw2 = ctx.measureText(tcaStr).width;
          ctx.fillStyle = 'rgba(6,13,20,0.85)';
          ctx.fillRect(midX - tw2 / 2 - 2, midY - 22, tw2 + 4, 10);
          ctx.fillStyle = 'rgba(251, 191, 36, 0.9)';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(`⏱ ${tcaStr}`, midX, midY - 17);
          ctx.restore();
        } else if (e.tca_utc) {
          const tcaStr = `${e.tca_utc.slice(11, 16)} UTC`;
          ctx.save();
          ctx.font = '7px JetBrains Mono, monospace';
          const tw2 = ctx.measureText(tcaStr).width;
          ctx.fillStyle = 'rgba(6,13,20,0.85)';
          ctx.fillRect(midX - tw2 / 2 - 2, midY - 22, tw2 + 4, 10);
          ctx.fillStyle = 'rgba(251, 191, 36, 0.9)';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(`⏱ ${tcaStr}`, midX, midY - 17);
          ctx.restore();
        }

        ctx.globalAlpha = 1;
      }

      // Draw nodes
      for (const n of nodes) {
        const r = Math.max(8, Math.min(24, (n.cpi ?? 0) * 2.5));
        const color = NODE_COLORS[n.severity] ?? '#6b7280';

        // Glow
        ctx.beginPath();
        ctx.arc(n.x, n.y, r + 3, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.15;
        ctx.fill();
        ctx.globalAlpha = 1;

        // Node circle
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fillStyle = color + '33';
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();

        // Label
        ctx.fillStyle = '#d1d5db';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const label = n.label?.length > 10 ? n.label.slice(0, 9) + '…' : (n.label ?? '');
        ctx.fillText(label, n.x, n.y + r + 8);
      }
    };

    simRef.current.animId = requestAnimationFrame(tick);

    return () => {
      if (simRef.current.animId) cancelAnimationFrame(simRef.current.animId);
      ro.disconnect();
    };
  }, []);

  // Click handler
  const handleCanvasClick = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const { nodes, edges } = simRef.current;

    // Check node click
    for (const n of nodes) {
      const r = Math.max(8, Math.min(24, (n.cpi ?? 0) * 2.5));
      const dx = mx - n.x;
      const dy = my - n.y;
      if (dx * dx + dy * dy <= r * r) {
        setSelectedSatId(n.id);
        return;
      }
    }

    // Check edge click
    for (const e of edges) {
      const src = nodes.find((n) => n.id === e.source);
      const tgt = nodes.find((n) => n.id === e.target);
      if (!src || !tgt) continue;
      const dx = tgt.x - src.x;
      const dy = tgt.y - src.y;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;
      const t = Math.max(0, Math.min(1, ((mx - src.x) * dx + (my - src.y) * dy) / (len * len)));
      const nearX = src.x + t * dx;
      const nearY = src.y + t * dy;
      if ((mx - nearX) ** 2 + (my - nearY) ** 2 <= 36) {
        setSelectedAlertId(e.id);
        return;
      }
    }
  }, [setSelectedSatId, setSelectedAlertId]);

  // Hover handler for tooltip
  const handleMouseMove = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const { nodes } = simRef.current;

    for (const n of nodes) {
      const r = Math.max(8, Math.min(24, (n.cpi ?? 0) * 2.5));
      const dx = mx - n.x;
      const dy = my - n.y;
      if (dx * dx + dy * dy <= (r + 4) ** 2) {
        setTooltip({ x: mx + 12, y: my - 8, label: n.label, cpi: n.cpi, severity: n.severity });
        return;
      }
    }
    setTooltip(null);
  }, []);

  const hasGraph = graph.nodes.length > 0;

  return (
    <div className="cascade-diagram">
      <canvas
        ref={canvasRef}
        className="cascade-canvas"
        onClick={handleCanvasClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setTooltip(null)}
      />

      {!hasGraph && (
        <div className="cascade-empty">
          No active cascade scenarios<br />
          Load the cascade demo to see cascade resolution
        </div>
      )}

      {/* Legend */}
      {hasGraph && (
        <div className="cascade-legend">
          {Object.entries(NODE_COLORS).slice(0, 4).map(([k, c]) => (
            <div className="cascade-legend-item" key={k}>
              <div className="cascade-legend-dot" style={{ background: c }} />
              {k}
            </div>
          ))}
        </div>
      )}

      {/* Tooltip */}
      {tooltip && (
        <div
          className="cascade-tooltip"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          {tooltip.label} · CPI {(tooltip.cpi ?? 0).toFixed(1)} · {tooltip.severity}
        </div>
      )}
    </div>
  );
}
