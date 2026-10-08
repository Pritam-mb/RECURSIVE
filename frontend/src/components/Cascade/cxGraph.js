/**
 * cxGraph.js — shared graph model + layout for the Cascade Risk Graph panel
 * and the Cascade & Hotspot Explorer.
 *
 * Common shape (both the store graph and /api/cascade/explorer are adapted to it):
 *   node { id, kind: 'object'|'event', name, severity, probability, agency, object_type, ... }
 *   edge { id, source, target, kind: 'screening'|'debris'|'event-parent'|'cascade',
 *          miss_distance_km, pc, tca_utc, severity, fragments, alert_id }
 *   hotspot { id, members: [nodeId], radius_km, severity, label }
 *
 * Layout: each connected component gets its own small force layout, then the
 * components are shelf-packed (largest first). Positions are layout only — no
 * data is encoded in them.
 */
import { seededUnit } from '../../utils/cascadeGraph';

export const key = (id) => String(id);

const finite = (v) => {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

export function fmtDist(km) {
  const v = finite(km);
  if (v == null) return '—';
  if (v < 1) return `${(v * 1000).toFixed(0)} m`;
  return v < 10 ? `${v.toFixed(2)} km` : `${v.toFixed(1)} km`;
}

export function fmtPc(p) {
  const v = finite(p);
  if (v == null) return '—';
  return v === 0 ? '0' : v.toExponential(1);
}

export function fmtCountdown(minutes) {
  const v = finite(minutes);
  if (v == null) return '—';
  const sign = v >= 0 ? 'T−' : 'T+';
  const s = Math.abs(Math.round(v * 60));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return `${sign}${h > 0 ? `${h}:` : ''}${pad(m)}:${pad(sec)}`;
}

export const SEV_CLASS = {
  CRITICAL: 'cx-sev-critical',
  WARNING: 'cx-sev-warning',
  WATCH: 'cx-sev-watch',
  NOMINAL: 'cx-sev-nominal',
  EVENT: 'cx-sev-event',
};
export const SEV_RANK = { EVENT: 4, CRITICAL: 3, WARNING: 2, WATCH: 1 };

/** Adapt the store graph (utils/cascadeGraph.buildCascadeGraph) + raw store hotspots. */
export function fromStoreGraph(graph, rawHotspots = []) {
  const kindMap = { conjunction: 'screening', debris: 'debris', breakup: 'event-parent', cascade: 'cascade' };
  const nodes = (graph?.nodes ?? []).map((n) => ({
    id: key(n.id),
    rawId: n.id,
    kind: n.kind === 'event' ? 'event' : 'object',
    name: n.label,
    severity: n.severity,
    probability: n.probability,
    agency: n.agency,
    object_type: n.object_type,
    cpi: n.cpi,
    depth: n.cascade_depth,
    event: n.event,
  }));
  const edges = (graph?.edges ?? []).map((e) => ({
    id: key(e.id),
    source: key(e.source),
    target: key(e.target),
    kind: kindMap[e.kind] ?? e.kind,
    miss_distance_km: finite(e.miss_distance),
    pc: finite(e.p_collision),
    tca_utc: e.tca_utc,
    tca_hours: e.tca_hours,
    severity: e.severity,
    fragments: e.count ?? null,
    alert_id: e.alert_id ?? e.id,
  }));
  const hotspots = (rawHotspots ?? []).map((h, i) => ({
    id: `hotspot:${i}`,
    index: i,
    members: (h.affected_satellites ?? []).map((m) => key(m.id)),
    radius_km: finite(h.zone_radius_km),
    severity: h.severity,
    score: finite(h.hotspot_score),
    tca_utc: h.tca_utc,
  }));
  return { nodes, edges, hotspots };
}

/** Adapt /api/cascade/explorer. */
export function fromExplorer(payload) {
  const all = payload?.graph?.nodes ?? [];
  const nodes = all.filter((n) => n.kind !== 'hotspot').map((n) => ({ ...n, id: key(n.id) }));
  const edges = (payload?.graph?.edges ?? []).map((e) => ({ ...e, id: key(e.id), source: key(e.source), target: key(e.target) }));
  const hotspots = (payload?.hotspots ?? []).map((h, i) => ({
    id: h.id,
    index: i,
    members: (h.area?.members ?? []).map((m) => key(m.id)),
    radius_km: h.area?.radius_km ?? null,
    severity: h.severity,
    score: h.score,
    tca_utc: h.area?.tca_utc,
  }));
  return { nodes, edges, hotspots };
}

/** Connected components → Map nodeId → componentIndex. */
export function components(nodes, edges) {
  const parent = new Map(nodes.map((n) => [n.id, n.id]));
  const find = (x) => {
    let r = x;
    while (parent.get(r) !== r) r = parent.get(r);
    let c = x;
    while (parent.get(c) !== r) { const nx = parent.get(c); parent.set(c, r); c = nx; }
    return r;
  };
  for (const e of edges) {
    if (!parent.has(e.source) || !parent.has(e.target)) continue;
    const a = find(e.source); const b = find(e.target);
    if (a !== b) parent.set(a, b);
  }
  const ids = new Map();
  const comp = new Map();
  for (const n of nodes) {
    const r = find(n.id);
    if (!ids.has(r)) ids.set(r, ids.size);
    comp.set(n.id, ids.get(r));
  }
  return comp;
}

/**
 * Focus subset: the CRITICAL/WARNING links, collision events with their parent
 * and fragment links, the objects on them, every member of a hotspot that
 * touches them, and any link between two kept objects (for context distances).
 * Returns { nodes, edges, hotspots, hidden }.
 */
export function focusSubset(model, showAll) {
  const { nodes, edges, hotspots } = model;
  if (showAll) return { ...model, hidden: 0 };
  const hot = new Set(['CRITICAL', 'WARNING']);
  const strong = new Set(['event-parent', 'debris', 'cascade']);
  const keep = new Set();
  for (const n of nodes) if (n.kind === 'event' || hot.has(n.severity)) keep.add(n.id);
  for (const e of edges) {
    if (hot.has(e.severity) || strong.has(e.kind)) { keep.add(e.source); keep.add(e.target); }
  }
  const hs = hotspots.filter((h) => h.members.some((m) => keep.has(m)));
  for (const h of hs) for (const m of h.members) keep.add(m);
  const known = new Set(nodes.map((n) => n.id));
  return {
    nodes: nodes.filter((n) => keep.has(n.id)),
    edges: edges.filter((e) => keep.has(e.source) && keep.has(e.target)),
    hotspots: hs.map((h) => ({ ...h, members: h.members.filter((m) => known.has(m)) })),
    hidden: nodes.length - [...keep].filter((id) => known.has(id)).length,
  };
}

function forceComponent(cnodes, cedges) {
  const pts = cnodes.map((n) => {
    const s = seededUnit(n.id);
    return { id: n.id, x: s.u * 60 * Math.sqrt(cnodes.length), y: s.v * 60 * Math.sqrt(cnodes.length), vx: 0, vy: 0 };
  });
  if (pts.length === 1) { pts[0].x = 0; pts[0].y = 0; return pts; }
  const by = new Map(pts.map((p) => [p.id, p]));
  const links = cedges.map((e) => [by.get(e.source), by.get(e.target)]).filter(([a, b]) => a && b && a !== b);
  const iters = pts.length > 80 ? 160 : 300;
  const L = 70;
  for (let it = 0; it < iters; it++) {
    for (const a of pts) {
      a.vx -= a.x * 0.01; a.vy -= a.y * 0.01;
      for (const b of pts) {
        if (a === b) continue;
        const dx = a.x - b.x; const dy = a.y - b.y;
        const d2 = Math.max(dx * dx + dy * dy, 25);
        const f = 2200 / d2;
        a.vx += dx * f / Math.sqrt(d2); a.vy += dy * f / Math.sqrt(d2);
      }
    }
    for (const [a, b] of links) {
      const dx = b.x - a.x; const dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (d - L) * 0.06;
      a.vx += (dx / d) * f; a.vy += (dy / d) * f;
      b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
    }
    let maxV = 0;
    for (const p of pts) {
      p.vx *= 0.82; p.vy *= 0.82;
      p.x += p.vx; p.y += p.vy;
      maxV = Math.max(maxV, Math.abs(p.vx), Math.abs(p.vy));
    }
    if (maxV < 0.02) break;
  }
  return pts;
}

/**
 * Packed per-component layout. Returns { pos: Map id → {x,y}, width, height }
 * in abstract units (callers fit it to their viewport).
 */
export function packedLayout(nodes, edges, aspect = 1.6) {
  const comp = components(nodes, edges);
  const groups = new Map();
  for (const n of nodes) {
    const c = comp.get(n.id);
    if (!groups.has(c)) groups.set(c, { nodes: [], edges: [] });
    groups.get(c).nodes.push(n);
  }
  for (const e of edges) {
    const c = comp.get(e.source);
    if (c != null && comp.get(e.target) === c) groups.get(c).edges.push(e);
  }
  const boxes = [];
  for (const g of groups.values()) {
    const pts = forceComponent(g.nodes, g.edges);
    let minX = Infinity; let maxX = -Infinity; let minY = Infinity; let maxY = -Infinity;
    for (const p of pts) { minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x); minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y); }
    const pad = 46;
    boxes.push({ pts, minX, minY, w: maxX - minX + pad * 2, h: maxY - minY + pad * 2, pad, n: pts.length });
  }
  boxes.sort((a, b) => b.n - a.n || b.w * b.h - a.w * a.h);
  const area = boxes.reduce((s, b) => s + b.w * b.h, 0);
  const rowW = Math.max(Math.sqrt(area * aspect), boxes[0]?.w ?? 0);
  const pos = new Map();
  let x = 0; let y = 0; let rowH = 0; let width = 0;
  for (const b of boxes) {
    if (x > 0 && x + b.w > rowW) { x = 0; y += rowH; rowH = 0; }
    for (const p of b.pts) pos.set(p.id, { x: x + b.pad + (p.x - b.minX), y: y + b.pad + (p.y - b.minY) });
    x += b.w; rowH = Math.max(rowH, b.h); width = Math.max(width, x);
  }
  return { pos, width: Math.max(width, 1), height: Math.max(y + rowH, 1) };
}

/** Ring around a hotspot's member nodes (in screen space). */
export function hotspotRing(h, project) {
  const pts = h.members.map(project).filter(Boolean);
  if (!pts.length) return null;
  const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
  const r = Math.max(...pts.map((p) => Math.hypot(p.x - cx, p.y - cy))) + 14;
  return { cx, cy, r };
}

/** 5-point star path (collision events). */
export function starPath(cx, cy, R, r = R * 0.45) {
  let d = '';
  for (let i = 0; i < 10; i++) {
    const rad = i % 2 === 0 ? R : r;
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    d += `${i === 0 ? 'M' : 'L'}${(cx + rad * Math.cos(a)).toFixed(1)},${(cy + rad * Math.sin(a)).toFixed(1)}`;
  }
  return `${d}Z`;
}

export const nodeRadius = (n) => {
  if (n.kind === 'event') return 8;
  const p = finite(n.probability);
  if (p == null || p <= 0) return 4;
  // radius grows one step per decade above 1e-9 (P(hit) is encoded in size)
  return 4 + Math.min(Math.max(Math.log10(p) + 9, 0), 6) * 0.8;
};

export function edgeLabel(e) {
  if (e.kind === 'event-parent') return e.miss_distance_km != null ? `parent · ${fmtDist(e.miss_distance_km)}` : 'parent';
  const parts = [];
  if (e.fragments > 1) parts.push(`${e.fragments} frag`);
  if (e.miss_distance_km != null) parts.push(fmtDist(e.miss_distance_km));
  return parts.join(' · ');
}

export function shortName(text, n = 14) {
  if (!text) return '';
  return text.length > n ? `${text.slice(0, n - 1)}…` : text;
}
