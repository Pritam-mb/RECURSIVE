import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  SEV_CLASS, SEV_RANK, edgeLabel, fmtDist, fmtPc, hotspotRing, nodeRadius, packedLayout, shortName, starPath,
} from './cxGraph';
import '../../styles/cascade.css';

const EDGE_CLASS = {
  screening: 'cx-edge--screening',
  debris: 'cx-edge--debris',
  'event-parent': 'cx-edge--parent',
  cascade: 'cx-edge--cascade',
};
const EDGE_SEV = { CRITICAL: 'cx-edge--critical', WARNING: 'cx-edge--warning' };
const LABEL_CHAR_W = 6;

function pickLabelled(nodes, always) {
  const placed = [];
  const keep = new Set();
  const ordered = [...nodes].sort((a, b) => (always.has(b.id) - always.has(a.id))
    || (SEV_RANK[b.severity] ?? 0) - (SEV_RANK[a.severity] ?? 0));
  for (const n of ordered) {
    const w = shortName(n.name).length * LABEL_CHAR_W;
    const box = { x: n.x - w / 2, y: n.y + n.r + 2, w, h: 11 };
    const hit = placed.some((p) => box.x < p.x + p.w && box.x + box.w > p.x && box.y < p.y + p.h && box.y + box.h > p.y)
      || nodes.some((m) => m !== n && m.x + m.r > box.x && m.x - m.r < box.x + box.w && m.y + m.r > box.y && m.y - m.r < box.y + box.h);
    if (hit && !always.has(n.id)) continue;
    placed.push(box);
    keep.add(n.id);
  }
  return keep;
}

/**
 * Inline-SVG cascade graph shared by the panel (static fit) and the explorer
 * (pan by drag, zoom by wheel). Positions are layout only.
 *
 * props: model {nodes, edges, hotspots}, interactive, selection {type,id},
 *        onSelect(sel), highlight: [{source, target, name, miss_km, pc, unsafe}],
 *        edgeLabels: 'auto' | 'all', hotspotLabel(h) -> string
 */
export default function CxGraphView({
  model, interactive = false, selection = null, onSelect, highlight = null, edgeLabels = 'auto',
  hotspotLabel, ariaLabel = 'Cascade risk graph',
}) {
  const bodyRef = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState({ k: 1, tx: 0, ty: 0 });
  const [hover, setHover] = useState(null);
  const drag = useRef(null);

  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return undefined;
    const measure = () => {
      const w = Math.round(el.clientWidth);
      const h = Math.round(el.clientHeight);
      setSize((p) => (p.w === w && p.h === h ? p : { w, h }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const aspect = size.w && size.h ? size.w / size.h : 1.6;
  const layout = useMemo(
    () => packedLayout(model.nodes, model.edges, Math.round(aspect * 4) / 4),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [model.nodes, model.edges, Math.round(aspect * 4)],
  );

  // reset the view whenever the layout changes
  useEffect(() => { setView({ k: 1, tx: 0, ty: 0 }); }, [layout]);

  const fit = useMemo(() => {
    const { w, h } = size;
    if (!w || !h) return null;
    const pad = 16;
    const s = Math.min(1.6, (w - pad * 2) / layout.width, (h - pad * 2) / layout.height);
    return { s, ox: (w - layout.width * s) / 2, oy: (h - layout.height * s) / 2 };
  }, [size, layout]);

  const scene = useMemo(() => {
    if (!fit) return null;
    const proj = (p) => ({
      x: (fit.ox + p.x * fit.s) * view.k + view.tx,
      y: (fit.oy + p.y * fit.s) * view.k + view.ty,
    });
    const nodes = model.nodes.map((n) => {
      const p = layout.pos.get(n.id) ?? { x: 0, y: 0 };
      return { ...n, ...proj(p), r: nodeRadius(n) };
    });
    const by = new Map(nodes.map((n) => [n.id, n]));
    const edges = model.edges
      .map((e) => ({ e, a: by.get(e.source), b: by.get(e.target) }))
      .filter((x) => x.a && x.b);
    const rings = model.hotspots
      .map((h) => ({ h, ring: hotspotRing(h, (id) => by.get(id)) }))
      .filter((x) => x.ring);
    // option-effect highlight: secondaries may not be in the graph → ghost nodes around the mover
    const ghosts = [];
    const hiEdges = [];
    (highlight ?? []).forEach((s, i) => {
      const a = by.get(String(s.source));
      if (!a) return;
      let b = by.get(String(s.target));
      if (!b) {
        const ang = -Math.PI / 2 + (i * 2 * Math.PI) / Math.max((highlight ?? []).length, 1) + 0.4;
        b = { id: `ghost:${s.target}`, x: a.x + Math.cos(ang) * 70, y: a.y + Math.sin(ang) * 70, r: 4, name: s.name, ghost: true };
        ghosts.push(b);
      }
      hiEdges.push({ s, a, b });
    });
    const sel = selection?.type === 'node' ? selection.id : null;
    const always = new Set([sel, hover?.type === 'node' ? hover.id : null].filter(Boolean));
    for (const { a, b } of hiEdges) { always.add(a.id); always.add(b.id); }
    return { nodes, edges, rings, ghosts, hiEdges, labelled: pickLabelled(nodes, always), by };
  }, [fit, layout, model, view, highlight, selection, hover]);

  // ── pan / zoom (explorer only) ────────────────────────────────────────
  const onWheel = useCallback((evt) => {
    if (!interactive) return;
    evt.preventDefault();
    const rect = bodyRef.current.getBoundingClientRect();
    const mx = evt.clientX - rect.left;
    const my = evt.clientY - rect.top;
    setView((v) => {
      const k = Math.min(8, Math.max(0.3, v.k * Math.exp(-evt.deltaY * 0.0015)));
      const f = k / v.k;
      return { k, tx: mx - (mx - v.tx) * f, ty: my - (my - v.ty) * f };
    });
  }, [interactive]);

  useEffect(() => {
    const el = bodyRef.current;
    if (!el || !interactive) return undefined;
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [onWheel, interactive]);

  const onPointerDown = (evt) => {
    if (!interactive || evt.button !== 0) return;
    drag.current = { x: evt.clientX, y: evt.clientY, tx: view.tx, ty: view.ty, moved: false };
  };
  const onPointerMove = (evt) => {
    if (!drag.current) return;
    const dx = evt.clientX - drag.current.x;
    const dy = evt.clientY - drag.current.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
    if (drag.current.moved) setView((v) => ({ ...v, tx: drag.current.tx + dx, ty: drag.current.ty + dy }));
  };
  const endDrag = () => { setTimeout(() => { drag.current = null; }, 0); };
  const click = (sel) => (evt) => {
    evt.stopPropagation();
    if (drag.current?.moved) return;
    onSelect?.(sel);
  };

  const hoverAt = (evt, item) => {
    const rect = bodyRef.current?.getBoundingClientRect();
    if (!rect) return;
    setHover({ ...item, x: evt.clientX - rect.left, y: evt.clientY - rect.top });
  };

  const showEdgeLabels = edgeLabels === 'all' || (scene && scene.edges.length <= (interactive ? 90 : 26)) || view.k > 1.6;
  const isSel = (type, id) => selection?.type === type && String(selection.id) === String(id);
  const hoverNode = hover?.type === 'node' ? scene?.by.get(hover.id) : null;
  const hoverEdge = hover?.type === 'edge' ? scene?.edges.find((x) => x.e.id === hover.id)?.e : null;

  return (
    <div
      className={`cx-graph${interactive ? ' cx-graph--interactive' : ''}`}
      ref={bodyRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerLeave={() => { endDrag(); setHover(null); }}
    >
      {scene && (
        <svg width={size.w} height={size.h} viewBox={`0 0 ${size.w} ${size.h}`} role="img" aria-label={ariaLabel}>
          <g>
            {scene.rings.map(({ h, ring }) => (
              <g
                key={h.id}
                className={`cx-ring ${SEV_CLASS[h.severity] ?? ''}${isSel('hotspot', h.id) ? ' is-selected' : ''}`}
                onClick={click({ type: 'hotspot', id: h.id })}
                onMouseMove={(evt) => hoverAt(evt, { type: 'hotspot', id: h.id, h })}
                onMouseLeave={() => setHover(null)}
              >
                <circle className="cx-ring-fill" cx={ring.cx} cy={ring.cy} r={ring.r} />
                <circle className="cx-ring-line" cx={ring.cx} cy={ring.cy} r={ring.r} />
                <text className="cx-ring-label" x={ring.cx} y={ring.cy - ring.r - 4} textAnchor="middle">
                  {hotspotLabel ? hotspotLabel(h) : `HS${h.index + 1}`}
                </text>
              </g>
            ))}
          </g>
          <g>
            {scene.edges.map(({ e, a, b }) => (
              <g
                key={e.id}
                className="cx-edge-group"
                onClick={click({ type: 'edge', id: e.id, edge: e })}
                onMouseMove={(evt) => hoverAt(evt, { type: 'edge', id: e.id })}
                onMouseLeave={() => setHover(null)}
              >
                <line className="cx-edge-hit" x1={a.x} y1={a.y} x2={b.x} y2={b.y} />
                <line
                  className={`cx-edge ${EDGE_CLASS[e.kind] ?? ''} ${EDGE_SEV[e.severity] ?? ''}${isSel('edge', e.id) ? ' is-selected' : ''}`}
                  x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                />
              </g>
            ))}
          </g>
          <g className="cx-hi">
            {scene.hiEdges.map(({ s, a, b }) => (
              <g key={`hi-${s.target}`}>
                <line className={`cx-hi-edge${s.unsafe ? ' is-unsafe' : ''}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} />
                <text className="cx-hi-label" x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 4} textAnchor="middle">
                  {`${fmtDist(s.miss_km)} · Pc ${fmtPc(s.pc)}`}
                </text>
              </g>
            ))}
            {scene.ghosts.map((g) => (
              <g key={g.id}>
                <circle className="cx-ghost" cx={g.x} cy={g.y} r={5} />
                <text className="cx-node-label" x={g.x} y={g.y + 15} textAnchor="middle">{shortName(g.name)}</text>
              </g>
            ))}
          </g>
          {showEdgeLabels && (
            <g pointerEvents="none">
              {scene.edges.map(({ e, a, b }) => {
                const label = edgeLabel(e);
                if (!label) return null;
                return (
                  <text key={`l-${e.id}`} className="cx-edge-label" x={(a.x + b.x) / 2} y={(a.y + b.y) / 2 - 3} textAnchor="middle">
                    {label}
                  </text>
                );
              })}
            </g>
          )}
          <g>
            {scene.nodes.map((n) => (
              <g
                key={n.id}
                className={`cx-node ${SEV_CLASS[n.severity] ?? 'cx-sev-unknown'}${isSel('node', n.id) ? ' is-selected' : ''}`}
                onClick={click({ type: 'node', id: n.id, node: n })}
                onMouseMove={(evt) => hoverAt(evt, { type: 'node', id: n.id })}
                onMouseLeave={() => setHover(null)}
              >
                <circle cx={n.x} cy={n.y} r={n.r + 5} fill="transparent" />
                {isSel('node', n.id) && <circle className="cx-node-ring" cx={n.x} cy={n.y} r={n.r + 3.5} />}
                {n.kind === 'event'
                  ? <path className="cx-node-star" d={starPath(n.x, n.y, n.r + 2)} />
                  : <circle className="cx-node-dot" cx={n.x} cy={n.y} r={n.r} />}
              </g>
            ))}
          </g>
          <g pointerEvents="none">
            {scene.nodes.map((n) => (scene.labelled.has(n.id) ? (
              <text key={`n-${n.id}`} className="cx-node-label" x={n.x} y={n.y + n.r + 11} textAnchor="middle">
                {shortName(n.name)}
              </text>
            ) : null))}
          </g>
        </svg>
      )}

      {hover && (hoverNode || hoverEdge || hover.h) && (
        <div className="cx-tip" style={{ left: Math.min(hover.x + 12, size.w - 200), top: Math.max(4, hover.y - 10) }}>
          {hoverNode && (
            <>
              <div className="cx-tip-name">{hoverNode.name}</div>
              <span className="cx-tip-k">{hoverNode.kind === 'event' ? 'Collision event' : (hoverNode.object_type || 'Object')}</span>
              <span className="cx-tip-v">{hoverNode.kind === 'event' ? `${hoverNode.fragment_count ?? hoverNode.event?.fragment_count ?? '—'} frag` : (hoverNode.agency || '—')}</span>
              {hoverNode.kind !== 'event' && (
                <>
                  <span className="cx-tip-k">P(hit)</span>
                  <span className="cx-tip-v">{fmtPc(hoverNode.probability)}</span>
                  <span className="cx-tip-k">State</span>
                  <span className={`cx-tip-v ${SEV_CLASS[hoverNode.severity] ?? ''}`}>{hoverNode.severity ?? '—'}</span>
                </>
              )}
            </>
          )}
          {hoverEdge && (
            <>
              <div className="cx-tip-name">
                {(scene.by.get(hoverEdge.source)?.name ?? hoverEdge.source)} ↔ {(scene.by.get(hoverEdge.target)?.name ?? hoverEdge.target)}
              </div>
              <span className="cx-tip-k">{hoverEdge.kind === 'event-parent' ? 'Breakup miss' : 'Miss dist'}</span>
              <span className="cx-tip-v">{fmtDist(hoverEdge.miss_distance_km)}</span>
              <span className="cx-tip-k">Pc</span>
              <span className="cx-tip-v">{fmtPc(hoverEdge.pc)}</span>
              {hoverEdge.fragments > 1 && (
                <>
                  <span className="cx-tip-k">Fragments</span>
                  <span className="cx-tip-v">{hoverEdge.fragments}</span>
                </>
              )}
              <span className="cx-tip-k">TCA</span>
              <span className="cx-tip-v">{hoverEdge.tca_utc ? `${String(hoverEdge.tca_utc).slice(11, 19)}Z` : '—'}</span>
            </>
          )}
          {hover.h && (
            <>
              <div className="cx-tip-name">{`Hotspot ${hover.h.index + 1}`}</div>
              <span className="cx-tip-k">Radius</span>
              <span className="cx-tip-v">{fmtDist(hover.h.radius_km)}</span>
              <span className="cx-tip-k">Members</span>
              <span className="cx-tip-v">{hover.h.members.length}</span>
              <span className="cx-tip-k">Score</span>
              <span className="cx-tip-v">{fmtPc(hover.h.score)}</span>
            </>
          )}
        </div>
      )}
    </div>
  );
}
