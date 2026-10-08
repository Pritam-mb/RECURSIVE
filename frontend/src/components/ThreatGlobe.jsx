import { useRef, useEffect, useMemo } from 'react';
import useTestMode from '../hooks/useTestMode';
import useStore from '../store/useStore';
import { severityLabel } from '../utils/severity';
import {
  cloudCentroid, cloudFragments, cloudLabel, cloudRadius,
} from '../utils/debrisCloud';
import {
  CORRECTION_DECAY_S,
  MAX_EXTRAPOLATION_S,
  correctionFor,
  estimateSimRate,
  extrapolate,
  observeActivity,
} from '../utils/motion';
import { onFlyTo, playheadNow } from './Impact/impactClock';

// Mirrors the design tokens in index.css (canvas can't read CSS vars cheaply).
const PALETTE = {
  void: '#030508',
  sphere: '#070a0e',
  line: '#1a212b',
  lineStrong: '#27303c',
  text: '#c3cbd5',
  dim: '#6b7685',
  bright: '#eef2f6',
  accent: '#4c8dff',
  info: '#7cc4ff',
  nominal: '#3ccf7a',
  caution: '#f0b429',
  warning: '#ff5a4f',
};
const MONO_9 = "500 9px 'IBM Plex Mono', ui-monospace, Consolas, monospace";
const CAPS_10 = "600 10px 'Barlow Semi Condensed', 'Barlow', 'Segoe UI', sans-serif";

const TWO_PI = Math.PI * 2;
const DEG = Math.PI / 180;
const FRAME_INTERVAL_MS = 1000 / 30;
const MAX_DPR = 1.5;
const ROTATION_DEG_PER_S = 4.8;
const MAX_CONJUNCTION_LABELS = 3;
const EMPTY = [];
// Longest pair link drawn (km). Beyond this the chord leaves the surface.
const MAX_LINK_KM = 1500;
// Objects at or below this radius are inside the Earth (bad state / a
// centroid between diverging streams) and are never drawn.
const EARTH_RADIUS_KM = 6378.137;
// A debris ring bigger than this would cover a large part of the disc.
const MAX_RING_KM = 500;
const aboveSurface = (p) => !!p && Math.hypot(p.x, p.y, p.z) > EARTH_RADIUS_KM;
const RAD2DEG = 180 / Math.PI;

// ── Hand navigation ──────────────────────────────────────────────────────
const DEFAULT_LON = 80;
const MAX_TILT_DEG = 70;
const ZOOM_MIN = 0.6;
const ZOOM_MAX = 2.2;
const DRAG_THRESHOLD_PX = 4;
const INERTIA_DECAY_PER_S = 2.4; // velocity *= exp(-k·dt): gentle glide
const MAX_SPIN_DEG_PER_S = 240;
const AUTO_RESUME_MS = 6000;
const HINT_TEXT = 'drag to rotate · scroll to zoom · double-click to reset';

// ── Layers (shared store, CONTRACT3) — safe defaults before the store has them
const DEFAULT_LAYERS = {
  satellites: true,
  selectedOrbit: true,
  conjunctionLines: true,
  labels: true,
  hotspots: true,
  parentTracks: true,
  fragments: true,
  fragmentTrails: false,
  debrisEnvelope: true,
  threatened: true,
};
const layerOn = (layers, key) => (layers && typeof layers[key] === 'boolean' ? layers[key] : DEFAULT_LAYERS[key]);
const MAX_HOTSPOTS = 24;

// ── Impact replay ────────────────────────────────────────────────────────
// Fragment colour ramps (caution amber family, dark → light = small → large).
const FRAG_RAMP = ['#8a6418', '#c48d1f', '#f0b429', '#ffe3a0'];
const FRAG_PARENT = ['#f0b429', '#ff8a3d', '#c48d1f'];
const SIZE_EDGES_M = [0.1, 0.3, 1.0];
const DV_EDGES_MS = [50, 150, 300];
const TRAIL_SAMPLES = 8;
const FLASH_MS = 2000;
const MAX_THREAT_CALLOUTS = 3;
const SIZE_LEGEND = ['<0.1 M', '0.1–0.3 M', '0.3–1 M', '≥1 M'];
const DV_LEGEND = ['<50 M/S', '50–150', '150–300', '≥300 M/S'];

// Simplified coastline data as lat/lon polylines (very compressed)
// Each sub-array is a connected polyline [[lat,lon],...]
const COAST_LINES = [
  // Africa
  [[37.3,9.5],[36.9,11.0],[22.2,37.1],[11.8,44.9],[1.7,41.6],[-4.7,39.8],[-10.8,40.5],[-26.0,32.9],[-34.8,20.0],[-33.9,18.3],[-29.9,16.7],[-15.8,11.9],[-5.5,5.3],[4.0,2.4],[5.1,1.2],[6.0,2.9],[5.0,5.0],[4.2,7.0],[5.5,11.2],[12.0,15.0],[19.1,12.3],[21.9,23.1],[23.9,32.9],[26.7,33.5],[31.1,32.1],[37.3,9.5]],
  // Europe
  [[71.2,25.8],[69.7,30.0],[65.0,25.5],[60.4,22.0],[56.0,21.0],[54.4,18.5],[54.5,10.0],[57.7,8.0],[58.0,5.4],[55.7,5.0],[51.4,2.6],[47.8,-4.5],[43.3,-8.7],[37.0,-9.0],[36.0,-5.4],[37.5,-0.6],[40.0,0.5],[42.0,3.3],[43.5,7.4],[43.9,15.0],[45.7,13.6],[45.0,14.8],[44.5,14.5],[41.9,12.5],[37.9,15.6],[38.0,15.0],[37.3,15.0],[36.8,11.1],[37.3,9.5]],
  // North America (simplified)
  [[71.4,-156],[70.5,-149],[67.5,-143],[60.4,-145],[59.7,-151],[58.5,-137],[55.5,-133],[49.0,-124],[37.5,-122],[36.6,-121],[34.4,-120],[32.6,-117],[30.0,-110],[25.5,-97],[22.9,-97],[19.6,-87],[15.9,-85],[10.9,-83],[8.0,-77],[8.9,-79],[9.5,-79],[9.4,-82],[9.5,-83],[8.2,-76],[9.6,-75],[11.0,-74],[12.5,-71],[16.0,-61],[20.0,-72],[22.0,-78],[24.0,-81],[25.8,-80],[30.6,-81],[35.0,-75],[38.9,-74],[41.2,-70],[42.0,-69],[44.5,-66],[47.0,-53],[51.2,-55],[53.9,-57],[58.5,-67],[62.0,-70],[66.0,-64],[70.0,-51],[71.0,-52],[71.0,-69],[71.4,-156]],
  // South America (simplified)
  [[11.0,-74],[10.5,-62],[10.6,-61],[10.0,-62],[8.8,-60],[5.2,-52],[4.0,-51],[2.0,-50],[0,-50],[-5,-35],[-8,-35],[-13,-39],[-23,-43],[-23,-44],[-25,-48],[-33,-52],[-34,-53],[-35,-57],[-38,-62],[-42,-65],[-44,-66],[-51,-69],[-55,-65],[-55,-70],[-53,-73],[-50,-75],[-43,-73],[-35,-72],[-25,-70],[-18,-70],[-16,-72],[-18,-70],[-16,-75],[-2,-80],[0,-78],[5,-77],[10,-75],[11,-74]],
  // Asia (simplified)
  [[71.2,25.8],[72.0,52.0],[73.0,68.0],[71.0,87.0],[72.0,105.0],[70.0,131.0],[65.0,141.0],[60.0,163.0],[59.0,164.0],[53.0,159.0],[47.0,142.0],[43.0,132.0],[38.0,121.0],[32.0,122.0],[26.0,120.0],[22.0,114.0],[18.0,110.0],[10.0,104.0],[1.0,104.0],[-5.0,105.0],[-8.0,115.0],[-8.0,124.0],[1.0,131.0],[5.0,126.0],[14.0,120.0],[18.0,122.0],[25.0,122.0],[22.0,114.0],[18.0,110.0],[10.0,104.0],[0.0,103.9],[1.0,103.9],[1.0,104.0],[3.0,103.0],[6.0,102.0],[13.0,100.0],[16.0,98.0],[18.0,92.0],[20.0,86.0],[16.0,81.0],[10.0,79.0],[8.0,77.0],[8.0,76.0],[9.0,78.0],[13.0,80.0],[20.0,87.0],[22.0,91.0],[24.0,90.0],[23.0,89.0],[21.0,88.0],[21.0,86.0],[20.0,86.0],[15.0,74.0],[15.0,73.0],[18.0,73.0],[22.0,70.0],[23.0,68.0],[25.0,67.0],[24.0,63.0],[23.0,58.0],[22.0,59.0],[20.0,58.0],[12.0,44.0],[11.8,44.9],[12.0,45.0],[11.5,43.0],[12.5,43.0],[15.0,42.0],[22.0,37.0],[26.0,33.5],[30.0,33.0],[31.0,32.0],[32.0,34.5],[37.3,35.0],[36.5,36.0],[37.0,37.0],[41.0,36.0],[41.0,30.0],[41.5,28.0],[41.0,29.0],[37.0,27.0],[36.5,28.0],[37.5,26.5],[38.0,26.0],[36.5,22.0],[37.0,22.0],[38.0,21.5],[37.5,22.0],[36.5,22.0],[35.0,24.0],[35.0,25.0],[36.0,28.0],[37.0,27.0],[41.0,29.0],[41.0,28.0],[42.0,28.5],[43.0,28.0],[43.5,28.5],[43.0,30.0],[41.0,30.0],[41.0,36.0],[42.0,41.0],[43.0,41.0],[43.0,40.0],[43.0,51.0],[47.0,53.0],[48.0,59.0],[51.0,60.0],[53.0,59.0],[55.0,60.0],[58.0,62.0],[62.0,60.0],[64.0,40.0],[66.0,33.0],[68.0,31.0],[71.2,25.8]],
  // Australia
  [[-14,130],[-13,136],[-12,136],[-12,135],[-14,130],[-15,129],[-16,123],[-22,114],[-31,115],[-35,117],[-35,118],[-38,140],[-39,144],[-37,147],[-37,150],[-33,152],[-28,153],[-24,152],[-22,150],[-19,147],[-18,147],[-17,146],[-16,145],[-14,144],[-11,143],[-12,142],[-12,136],[-12,132],[-14,130]],
];

// ── Precomputed geometry ──────────────────────────────────────────────────
// Everything is stored as unit vectors so per-frame projection is a few
// multiplies (no trig, no allocation).

function latLonToUnit(lat, lon, out, i) {
  const phi = lat * DEG;
  const lam = lon * DEG;
  const c = Math.cos(phi);
  out[i] = c * Math.cos(lam);
  out[i + 1] = c * Math.sin(lam);
  out[i + 2] = Math.sin(phi);
}

function toUnitPolyline(points) {
  const arr = new Float32Array(points.length * 3);
  points.forEach(([lat, lon], k) => latLonToUnit(lat, lon, arr, k * 3));
  return arr;
}

const COAST_UNIT = COAST_LINES.map(toUnitPolyline);

const GRID_UNIT = (() => {
  const lines = [];
  for (let lat = -60; lat <= 60; lat += 30) {
    const pts = [];
    for (let lon = -180; lon <= 180; lon += 4) pts.push([lat, lon]);
    lines.push(toUnitPolyline(pts));
  }
  for (let lon = 0; lon < 360; lon += 30) {
    const pts = [];
    for (let lat = -90; lat <= 90; lat += 4) pts.push([lat, lon]);
    lines.push(toUnitPolyline(pts));
  }
  return lines;
})();

/**
 * Orthographic view state for one frame. The view is a rotation about the
 * polar axis by the view longitude V, then a tilt T about the screen's
 * horizontal axis (T > 0 = looking down from the north). With
 *   d0 = ux·cosV + uy·sinV   (depth before tilt)
 * a unit vector (ux, uy, uz) projects to
 *   sx = cx + (uy·cosV − ux·sinV)·R
 *   sy = cy − (uz·cosT − d0·sinT)·R
 *   depth = d0·cosT + uz·sinT   (≥ 0 → near hemisphere)
 */
const view = { cx: 0, cy: 0, R: 1, cosV: 1, sinV: 0, cosT: 1, sinT: 0 };

function strokeUnitPolylines(ctx, lines) {
  const { cx, cy, R, cosV, sinV, cosT, sinT } = view;
  ctx.beginPath();
  for (const arr of lines) {
    let penDown = false;
    for (let i = 0; i < arr.length; i += 3) {
      const ux = arr[i];
      const uy = arr[i + 1];
      const uz = arr[i + 2];
      const d0 = (ux * cosV) + (uy * sinV);
      if ((d0 * cosT) + (uz * sinT) < 0) { penDown = false; continue; }
      const sx = cx + (((uy * cosV) - (ux * sinV)) * R);
      const sy = cy - (((uz * cosT) - (d0 * sinT)) * R);
      if (penDown) ctx.lineTo(sx, sy);
      else { ctx.moveTo(sx, sy); penDown = true; }
    }
  }
  ctx.stroke();
}

// Objects are drawn at a (log-compressed) altitude above the disc rather
// than flattened onto it, so an orbit is a ring around the Earth — like the
// 3D globe — instead of a track painted across the disc. GEO lands at ~1.3 R.
const ALT_GAIN = 0.3 / Math.log(1 + (35786 / 500));
const displayScale = (r) => 1 + (ALT_GAIN * Math.log(1 + (Math.max(r - 6371, 0) / 500)));

/**
 * Projects an ECI km vector (≈ earth-fixed for display) into `out`.
 * Hidden only when it is inside the Earth or behind the disc (far hemisphere
 * AND inside the limb) — the canvas has no depth buffer, so this is the
 * occlusion test.
 */
function projectEci(x, y, z, out) {
  const r = Math.sqrt((x * x) + (y * y) + (z * z));
  if (!(r > EARTH_RADIUS_KM)) { out.visible = false; return out; }
  const ux = x / r;
  const uy = y / r;
  const uz = z / r;
  const { cx, cy, R, cosV, sinV, cosT, sinT } = view;
  const k = R * displayScale(r);
  const d0 = (ux * cosV) + (uy * sinV);
  const px = ((uy * cosV) - (ux * sinV)) * k;
  const py = ((uz * cosT) - (d0 * sinT)) * k;
  out.sx = cx + px;
  out.sy = cy - py;
  out.visible = (d0 * cosT) + (uz * sinT) >= 0 || Math.hypot(px, py) > R + 1;
  return out;
}

function threatColor(cpi) {
  if (cpi >= 8) return PALETTE.warning;
  if (cpi >= 5) return PALETTE.caution;
  return null;
}

// Colour level from the backend severity tier (CRITICAL ≥ 8, WARNING ≥ 5 on
// the threatColor scale); the tier, not a CPI cut, decides the colour.
const TIER_LEVEL = { CRITICAL: 9, WARNING: 6, WATCH: 0 };
const alertLevel = (alert) => TIER_LEVEL[severityLabel(alert)] ?? 0;

function formatMiss(km) {
  if (km == null || !(km > 0)) return '';
  return km < 1 ? `${(km * 1000).toFixed(0)} M` : `${km.toFixed(1)} KM`;
}

function formatTca(alert) {
  if (alert.tca_utc) return `TCA ${alert.tca_utc.slice(11, 16)}Z`;
  const h = Number(alert.tca_hours ?? 0);
  return h > 0 ? `T-${h.toFixed(1)}H` : '';
}

/** Greedy label placement: reserves and returns true if the box is free. */
function reserve(placed, x, y, w, h) {
  for (let i = 0; i < placed.length; i += 4) {
    if (x < placed[i] + placed[i + 2] && x + w > placed[i]
      && y < placed[i + 1] + placed[i + 3] && y + h > placed[i + 1]) return false;
  }
  placed.push(x, y, w, h);
  return true;
}

/** Solid dark tag with a 1px hairline border; text vertically centred. */
function drawTag(ctx, x, y, w, h, text, color) {
  ctx.fillStyle = PALETTE.void;
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = PALETTE.lineStrong;
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  ctx.fillStyle = color;
  ctx.fillText(text, x + 5, y + (h / 2) + 0.5);
}

/**
 * Callout: places a tag just outside the globe limb, radially out from its
 * anchor, nudging around the limb to avoid other tags, and joins it to the
 * anchor with a thin leader line. Tags never sit on the sphere itself, so the
 * disc stays clean. Returns false when no free slot was found.
 */
const CALLOUT_GAP = 14;
const CALLOUT_NUDGES = [0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.42, -0.42, 0.56, -0.56];
function drawCallout(ctx, anchor, text, color, W, H) {
  const { cx, cy, R } = view;
  const w = ctx.measureText(text).width + 10;
  const h = 14;
  const base = Math.atan2(anchor.sy - cy, anchor.sx - cx);
  // Anchors sit at altitude, possibly beyond the limb: start outside them.
  const ring = Math.max(R, Math.hypot(anchor.sx - cx, anchor.sy - cy)) + CALLOUT_GAP;
  for (const nudge of CALLOUT_NUDGES) {
    const a = base + nudge;
    const cos = Math.cos(a);
    const ex = cx + (cos * ring);
    const ey = cy + (Math.sin(a) * ring);
    const x = Math.min(Math.max(cos >= 0 ? ex : ex - w, 2), W - w - 2);
    const y = Math.min(Math.max(ey - (h / 2), 2), H - h - 2);
    // Reject slots where clamping pushed the tag back onto the disc.
    const nx = Math.min(Math.max(cx, x), x + w);
    const ny = Math.min(Math.max(cy, y), y + h);
    if (Math.hypot(nx - cx, ny - cy) < R + 3) continue;
    if (!reserve(placedBoxes, x, y, w, h)) continue;

    const attachX = cos >= 0 ? x : x + w;
    const attachY = y + (h / 2);
    ctx.strokeStyle = PALETTE.lineStrong;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(anchor.sx, anchor.sy);
    ctx.lineTo(ex, ey);
    ctx.lineTo(attachX, attachY);
    ctx.stroke();
    ctx.fillStyle = PALETTE.lineStrong;
    ctx.beginPath();
    ctx.arc(anchor.sx, anchor.sy, 1.5, 0, TWO_PI);
    ctx.fill();
    drawTag(ctx, x, y, w, h, text, color);
    return true;
  }
  return false;
}

// ── Impact replay preparation (once per payload, never per frame) ─────────
const finiteOr = (v, d) => {
  const n = Number(v);
  return v != null && Number.isFinite(n) ? n : d;
};
function bucketOf(v, edges) {
  const n = Number(v);
  if (v == null || !Number.isFinite(n)) return 0; // unknown → first bin (same as the 3D globe)
  let b = 0;
  while (b < edges.length && n >= edges[b]) b += 1;
  return b;
}
function readXyz(p, out, k) {
  if (!Array.isArray(p) || p.length < 3) return false;
  const x = Number(p[0]); const y = Number(p[1]); const z = Number(p[2]);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return false;
  out[k] = x; out[k + 1] = y; out[k + 2] = z;
  return true;
}

/**
 * Flattens a replay payload (CONTRACT3) into typed arrays so the render loop
 * only does index arithmetic. Positions are the backend's propagated samples;
 * between samples the globe interpolates linearly (labelled on canvas).
 */
function prepareReplay(replay) {
  const tArr = replay?.t_rel_s;
  if (!Array.isArray(tArr) || tArr.length === 0) return null;
  const nS = tArr.length;
  const t = Float64Array.from(tArr, (v) => finiteOr(v, 0));
  const frags = replay.fragments || {};
  const fRows = Array.isArray(frags.positions) ? frags.positions : [];
  const firstRow = fRows.find((row) => Array.isArray(row) && row.length > 0);
  const nF = Math.min(400, Array.isArray(frags.ids) ? frags.ids.length : (firstRow?.length ?? 0));
  const fragPos = new Float32Array(nS * nF * 3);
  const fragOk = new Uint8Array(nS * nF);
  for (let si = 0; si < nS && si < fRows.length; si += 1) {
    const row = fRows[si];
    if (!Array.isArray(row)) continue;
    for (let f = 0; f < nF; f += 1) {
      const k = (si * nF) + f;
      if (readXyz(row[f], fragPos, k * 3)) fragOk[k] = 1;
    }
  }

  const parentsObj = replay.parents || {};
  const declared = Array.isArray(replay.parent_ids) ? replay.parent_ids.map(String) : [];
  const parentKeys = declared.filter((id) => parentsObj[id]);
  for (const id of Object.keys(parentsObj)) if (!parentKeys.includes(id)) parentKeys.push(id);
  const parents = parentKeys.slice(0, 2).map((id, idx) => {
    const src = parentsObj[id] || {};
    const pos = new Float32Array(nS * 3);
    const ok = new Uint8Array(nS);
    const rows = Array.isArray(src.positions) ? src.positions : [];
    for (let si = 0; si < nS && si < rows.length; si += 1) if (readXyz(rows[si], pos, si * 3)) ok[si] = 1;
    return {
      id: Number(id), name: src.name || `#${id}`, color: idx === 0 ? PALETTE.accent : PALETTE.info,
      pos, ok, cur: { x: 0, y: 0, z: 0, ok: false }, proj: { sx: 0, sy: 0, visible: false },
    };
  });

  const parentOf = Array.isArray(frags.parent_of) ? frags.parent_of : [];
  const bucket = { parent: new Uint8Array(nF), size: new Uint8Array(nF), dv: new Uint8Array(nF) };
  for (let f = 0; f < nF; f += 1) {
    const po = Number(parentOf[f]);
    bucket.parent[f] = parents[0] && po === parents[0].id ? 0 : parents[1] && po === parents[1].id ? 1 : 2;
    bucket.size[f] = bucketOf(frags.size_m?.[f], SIZE_EDGES_M);
    bucket.dv[f] = bucketOf(frags.dv_ms?.[f], DV_EDGES_MS);
  }

  const env = Array.isArray(replay.envelope) ? replay.envelope : [];
  const envC = new Float32Array(nS * 3);
  const envR = new Float32Array(nS);
  const envOk = new Uint8Array(nS);
  for (let si = 0; si < nS && si < env.length; si += 1) {
    const e = env[si];
    const r90 = Number(e?.p90_km);
    if (e && readXyz(e.centroid, envC, si * 3) && Number.isFinite(r90) && r90 > 0) {
      envR[si] = r90;
      envOk[si] = 1;
    }
  }

  // Collision point: backend value if given, else the parents' mean at t≈0.
  const collision = new Float32Array(3);
  let hasCollision = readXyz(replay.collision_point_eci_km, collision, 0);
  if (!hasCollision && parents.length > 0) {
    let best = 0;
    for (let si = 1; si < nS; si += 1) if (Math.abs(t[si]) < Math.abs(t[best])) best = si;
    let n = 0;
    for (const par of parents) {
      if (!par.ok[best]) continue;
      collision[0] += par.pos[best * 3]; collision[1] += par.pos[(best * 3) + 1]; collision[2] += par.pos[(best * 3) + 2];
      n += 1;
    }
    if (n > 0) { collision[0] /= n; collision[1] /= n; collision[2] /= n; hasCollision = true; }
  }

  return {
    nS, nF, t, fragPos, fragOk, parents, bucket, envC, envR, envOk,
    collision, hasCollision,
    stepS: finiteOr(replay.step_s, nS > 1 ? t[1] - t[0] : 0),
    threatened: Array.isArray(replay.threatened) ? replay.threatened : EMPTY,
    // Per-frame scratch (reused).
    i0: 0, i1: 0, a: 0,
    cur: new Float32Array(nF * 3),
    curOk: new Uint8Array(nF),
    curSx: new Float32Array(nF),
    curSy: new Float32Array(nF),
    curVis: new Uint8Array(nF),
    alive: 0,
  };
}

/** Locates the bracketing samples for tRel (binary search, no allocation). */
function locateSample(prep, tRel) {
  const { t, nS } = prep;
  if (!(tRel > t[0])) { prep.i0 = 0; prep.i1 = 0; prep.a = 0; return; }
  if (tRel >= t[nS - 1]) { prep.i0 = nS - 1; prep.i1 = nS - 1; prep.a = 0; return; }
  let lo = 0;
  let hi = nS - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= tRel) lo = mid; else hi = mid;
  }
  prep.i0 = lo;
  prep.i1 = hi;
  prep.a = (tRel - t[lo]) / (t[hi] - t[lo] || 1);
}

/** Linear interpolation of fragments + parents between real samples. */
function interpolateReplay(prep) {
  const { nF, fragPos, fragOk, cur, curOk, i0, i1, a } = prep;
  const b = 1 - a;
  let alive = 0;
  for (let f = 0; f < nF; f += 1) {
    const k0 = (i0 * nF) + f;
    const k1 = (i1 * nF) + f;
    // Strict: both bracketing samples must exist (no fragment before breakup
    // or after decay), except exactly on a sample.
    const ok = fragOk[k0] && (a === 0 || fragOk[k1]);
    curOk[f] = ok ? 1 : 0;
    if (!ok) continue;
    const o = f * 3;
    const p0 = k0 * 3;
    const p1 = k1 * 3;
    cur[o] = (fragPos[p0] * b) + (fragPos[p1] * a);
    cur[o + 1] = (fragPos[p0 + 1] * b) + (fragPos[p1 + 1] * a);
    cur[o + 2] = (fragPos[p0 + 2] * b) + (fragPos[p1 + 2] * a);
    alive += 1;
  }
  prep.alive = alive;
  for (const par of prep.parents) {
    const ok = par.ok[i0] && (a === 0 || par.ok[i1]);
    par.cur.ok = !!ok;
    if (!ok) continue;
    const p0 = i0 * 3;
    const p1 = i1 * 3;
    par.cur.x = (par.pos[p0] * b) + (par.pos[p1] * a);
    par.cur.y = (par.pos[p0 + 1] * b) + (par.pos[p1 + 1] * a);
    par.cur.z = (par.pos[p0 + 2] * b) + (par.pos[p1 + 2] * a);
  }
}

function formatTRel(s) {
  const sign = s < 0 ? '−' : '+';
  const v = Math.abs(Math.round(s));
  const h = Math.floor(v / 3600);
  const m = Math.floor((v % 3600) / 60);
  const sec = v % 60;
  const mm = String(m).padStart(2, '0');
  const ss = String(sec).padStart(2, '0');
  return h > 0 ? `T${sign}${h}:${mm}:${ss}` : `T${sign}${mm}:${ss}`;
}

const testPosA = { sx: 0, sy: 0, visible: false };
const testPosB = { sx: 0, sy: 0, visible: false };
const scratchProj = { sx: 0, sy: 0, visible: false };
const fragAnchor = { sx: 0, sy: 0, visible: true };
const midpoint = { sx: 0, sy: 0 };
const placedBoxes = [];

export default function ThreatGlobe({ alerts = EMPTY, satellites = EMPTY, selectedSatId }) {
  const canvasRef = useRef(null);
  const testActive = useTestMode((s) => s.testActive);
  const testSatellites = useTestMode((s) => s.testSatellites);
  const computed = useTestMode((s) => s.computed);
  const debrisClouds = useStore((s) => s.debrisClouds);
  const snapshotTimestamp = useStore((s) => s.snapshotTimestamp);
  const selectedOrbit = useStore((s) => s.selectedOrbit);

  // Alert-derived lookups, rebuilt only when alerts change (not per frame).
  const threat = useMemo(() => {
    const cpiById = new Map();
    const bump = (id, cpi) => {
      if (id == null) return;
      const key = Number(id);
      if ((cpiById.get(key) ?? -1) < cpi) cpiById.set(key, cpi);
    };
    for (const a of alerts) {
      const level = alertLevel(a);
      bump(a.sat1?.id, level);
      bump(a.sat2?.id, level);
    }
    const ranked = [...alerts].sort((a, b) => (alertLevel(b) - alertLevel(a))
      || (Number(b.cpi_score ?? 0) - Number(a.cpi_score ?? 0)));
    return { cpiById, ranked };
  }, [alerts]);

  // Latest inputs for the render loop, so new snapshots don't restart it.
  const propsRef = useRef(null);
  propsRef.current = { threat, alertCount: alerts.length, selectedSatId, testActive, testSatellites, computed, debrisClouds, selectedOrbit };

  // Per-satellite dead-reckoning state (see utils/motion.js).
  const motionRef = useRef({ sats: new Map(), lastSimMs: null, lastWallMs: 0, simRate: 1 });

  useEffect(() => {
    const motion = motionRef.current;
    const nowMs = performance.now();
    const parsed = snapshotTimestamp ? Date.parse(snapshotTimestamp) : Number.NaN;
    const simMs = Number.isFinite(parsed) ? parsed : null;
    // Same or slightly older snapshot (REST/WS race): sync membership only.
    const stale = simMs != null && motion.lastSimMs != null
      && simMs <= motion.lastSimMs && simMs > motion.lastSimMs - 10000;
    if (!stale) {
      if (simMs != null) {
        motion.simRate = estimateSimRate(motion.lastSimMs, motion.lastWallMs, simMs, nowMs, motion.simRate);
        motion.lastSimMs = simMs;
      }
      motion.lastWallMs = nowMs;
    }

    const stamp = (motion.stamp = (motion.stamp || 0) + 1);
    for (const sat of satellites) {
      if (!sat.position) continue;
      const id = Number(sat.norad_id);
      let entry = motion.sats.get(id);
      if (!entry) {
        const { x, y, z } = sat.position;
        entry = {
          name: sat.name,
          base: sat.position,
          velocity: sat.velocity ?? null,
          correction: null,
          corrBuf: { x: 0, y: 0, z: 0 },
          render: { x, y, z },
          proj: { sx: 0, sy: 0, visible: false },
          stamp,
        };
        motion.sats.set(id, entry);
        continue;
      }
      entry.stamp = stamp;
      entry.name = sat.name;
      if (stale) continue;
      entry.correction = correctionFor(entry.render, sat.position, entry.corrBuf);
      entry.base = sat.position;
      entry.velocity = sat.velocity ?? null;
    }
    for (const [id, entry] of motion.sats) {
      if (entry.stamp !== stamp) motion.sats.delete(id);
    }
  }, [satellites, snapshotTimestamp]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d', { alpha: true });
    const size = { w: 0, h: 0, dpr: 1 };
    let rafId = 0;
    let running = false;
    let lastFrameMs = 0;
    let lastTickMs = 0;

    // View navigation state (hand-driven + auto-rotation + inertia).
    const nav = {
      lon: DEFAULT_LON, lat: 0, zoom: 1,
      vLon: 0, vLat: 0,
      lastInteractMs: -Infinity,
      autoBlend: 1,
      resetting: false,
      // Fly-to target (impact console "fly to impact"): eased in the loop.
      flying: false, flyLon: 0, flyLat: 0,
    };
    // Pointer state (declared before the loop reads it).
    const pointers = new Map();
    const drag = { active: false, moved: false, startX: 0, startY: 0, lastX: 0, lastY: 0, lastT: 0 };
    const pinch = { active: false, dist: 1, zoom: 1 };
    // Impact replay state carried between frames.
    let prep = null;
    let prepFor = null;
    let prevTRel = Number.NaN;
    let flashStartMs = -Infinity;

    // Backing store sized only on resize, DPR capped.
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      size.dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
      size.w = rect.width;
      size.h = rect.height;
      const bw = Math.max(1, Math.round(size.w * size.dpr));
      const bh = Math.max(1, Math.round(size.h * size.dpr));
      if (canvas.width !== bw) canvas.width = bw;
      if (canvas.height !== bh) canvas.height = bh;
    };
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();

    const drawCompass = (W) => {
      // Tiny orientation indicator: equator ellipse squashed by the tilt,
      // the north pole's position, and the view-centre longitude / tilt.
      const r = 12;
      const x = W - 24;
      const y = 24;
      ctx.lineWidth = 1;
      ctx.strokeStyle = PALETTE.lineStrong;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, TWO_PI);
      ctx.stroke();
      ctx.strokeStyle = PALETTE.dim;
      ctx.beginPath();
      ctx.ellipse(x, y, r, Math.max(0.5, r * Math.abs(view.sinT)), 0, 0, TWO_PI);
      ctx.stroke();
      ctx.fillStyle = PALETTE.info;
      ctx.beginPath();
      ctx.arc(x, y - (r * view.cosT), view.sinT >= 0 ? 2 : 1.25, 0, TWO_PI);
      ctx.fill();
      ctx.font = MONO_9;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = PALETTE.dim;
      const lon = ((((nav.lon + 180) % 360) + 360) % 360) - 180;
      const lonTxt = `${Math.abs(lon).toFixed(0).padStart(3, '0')}°${lon >= 0 ? 'E' : 'W'}`;
      const tiltTxt = `${nav.lat >= 0 ? '+' : '−'}${Math.abs(nav.lat).toFixed(0)}°`;
      ctx.fillText(lonTxt, x - r - 6, y - 5);
      ctx.fillText(`TILT ${tiltTxt}`, x - r - 6, y + 6);
      if (nav.zoom < 0.99 || nav.zoom > 1.01) ctx.fillText(`×${nav.zoom.toFixed(2)}`, x - r - 6, y + 17);
      ctx.textAlign = 'left';
    };

    const draw = (nowMs) => {
      const W = size.w;
      const H = size.h;
      if (W < 2 || H < 2) return;
      const {
        threat: { cpiById, ranked },
        alertCount, selectedSatId: selectedId, testActive: tActive,
        testSatellites: tSats, computed: comp, debrisClouds: clouds, selectedOrbit: orbit,
      } = propsRef.current;
      const motion = motionRef.current;
      // Shared store keys (CONTRACT3) read per frame — no React re-render.
      const st = useStore.getState();
      const layers = st.layers;
      const impact = st.impact;
      const colorMode = st.fragmentColorMode === 'size' || st.fragmentColorMode === 'dv' ? st.fragmentColorMode : 'parent';
      const showSats = layerOn(layers, 'satellites');
      const showOrbit = layerOn(layers, 'selectedOrbit');
      const showLinks = layerOn(layers, 'conjunctionLines');
      const showLabels = layerOn(layers, 'labels');
      const showHotspots = layerOn(layers, 'hotspots');
      const showFrags = layerOn(layers, 'fragments');
      const showEnvelope = layerOn(layers, 'debrisEnvelope');
      const showThreatened = layerOn(layers, 'threatened');
      const showParents = layerOn(layers, 'parentTracks');
      const showTrails = layerOn(layers, 'fragmentTrails');

      const replay = impact?.replay ?? null;
      if (replay !== prepFor) {
        prepFor = replay;
        prep = replay ? prepareReplay(replay) : null;
        prevTRel = Number.NaN;
        flashStartMs = -Infinity;
      }
      // While playing, the store tRelS is only published at ≤ 10 Hz: use the
      // shared smooth playhead (Impact/impactClock), clamped to the window.
      let tRel = finiteOr(impact?.tRelS, 0);
      if (prep && impact?.playing) {
        const smooth = playheadNow();
        if (Number.isFinite(smooth)) tRel = Math.max(prep.t[0], Math.min(prep.t[prep.nS - 1], smooth));
      }
      if (prep) {
        if (prevTRel < 0 && tRel >= 0) flashStartMs = nowMs; // crossed the collision
        prevTRel = tRel;
        locateSample(prep, tRel);
        interpolateReplay(prep);
      }

      view.cx = W / 2;
      view.cy = H / 2;
      // Zoom scales the disc; callouts never sit on it (drawCallout rejects
      // on-disc slots), so when zoomed in they simply drop out.
      view.R = Math.max(20, Math.min(W, H) * 0.40 * nav.zoom);
      view.cosV = Math.cos(nav.lon * DEG);
      view.sinV = Math.sin(nav.lon * DEG);
      view.cosT = Math.cos(nav.lat * DEG);
      view.sinT = Math.sin(nav.lat * DEG);
      const { cx, cy, R } = view;

      placedBoxes.length = 0;
      ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';

      // ── Sphere, graticule, coastlines ──────────────────────────────────
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, TWO_PI);
      ctx.fillStyle = PALETTE.sphere;
      ctx.fill();

      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, TWO_PI);
      ctx.clip();

      ctx.lineWidth = 1;
      ctx.strokeStyle = PALETTE.line;
      strokeUnitPolylines(ctx, GRID_UNIT);
      ctx.strokeStyle = PALETTE.lineStrong;
      strokeUnitPolylines(ctx, COAST_UNIT);
      // Orbital objects sit above the surface and may extend past the limb.
      ctx.restore();

      // ── Conjunction hotspots (screened TCA locations) ──────────────────
      const hotspots = st.hotspots;
      if (showHotspots && Array.isArray(hotspots) && hotspots.length > 0) {
        ctx.strokeStyle = PALETTE.warning;
        ctx.fillStyle = PALETTE.warning;
        ctx.lineWidth = 1;
        const n = Math.min(hotspots.length, MAX_HOTSPOTS);
        for (let i = 0; i < n; i += 1) {
          const pos = hotspots[i]?.position;
          if (!aboveSurface(pos)) continue;
          const p = projectEci(pos.x, pos.y, pos.z, scratchProj);
          if (!p.visible) continue;
          const zone = Number(hotspots[i].zone_radius_km);
          const rr = Math.max(4, Math.min(Number.isFinite(zone) ? zone * (R / 6371.0) : 4, 18));
          ctx.globalAlpha = 0.45;
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, rr, 0, TWO_PI);
          ctx.stroke();
          ctx.globalAlpha = 0.9;
          ctx.beginPath();
          ctx.moveTo(p.sx, p.sy - 2.5);
          ctx.lineTo(p.sx + 2.5, p.sy);
          ctx.lineTo(p.sx, p.sy + 2.5);
          ctx.lineTo(p.sx - 2.5, p.sy);
          ctx.closePath();
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }

      // ── Debris clouds (live): fragments + dashed envelope ring ─────────
      // While an impact replay is loaded, the replay owns the debris layers.
      if (!prep && clouds && clouds.length > 0 && (showFrags || showEnvelope || showLabels)) {
        ctx.font = MONO_9;
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = PALETTE.caution;
        const breathe = 0.45 + (0.15 * Math.sin(nowMs / 1000 * (TWO_PI / 6)));
        const seenLabels = new Set();
        for (const cloud of clouds) {
          const c = cloudCentroid(cloud);
          if (!c) continue;
          const radius = cloudRadius(cloud);
          // Real fragments, near side only (the sphere hides the far side).
          const frags = cloudFragments(cloud, 300);
          let hasAnchor = false;
          let nearestD2 = Infinity;
          ctx.setLineDash([]);
          ctx.fillStyle = PALETTE.caution;
          ctx.globalAlpha = 0.75;
          ctx.beginPath();
          for (const f of frags) {
            if (!aboveSurface(f)) continue;
            const fp = projectEci(f.x, f.y, f.z, scratchProj);
            if (!fp.visible) continue;
            if (showFrags) ctx.rect(fp.sx - 0.75, fp.sy - 0.75, 1.5, 1.5);
            const d2 = ((f.x - c.x) ** 2) + ((f.y - c.y) ** 2) + ((f.z - c.z) ** 2);
            if (d2 < nearestD2) { nearestD2 = d2; hasAnchor = true; fragAnchor.sx = fp.sx; fragAnchor.sy = fp.sy; }
          }
          ctx.fill();
          ctx.setLineDash([3, 3]);
          // Diverging streams leave the centroid in empty space or inside the
          // Earth with a radius of thousands of km: a ring there would cover
          // the whole disc. Ring only a compact cloud whose centroid is in orbit.
          const compact = aboveSurface(c) && (!radius || radius.km <= MAX_RING_KM);
          let p;
          if (compact) {
            p = projectEci(c.x, c.y, c.z, scratchProj);
            if (!p.visible) continue;
          } else if (hasAnchor) {
            p = fragAnchor;
          } else {
            continue;
          }
          // Percentile radius of the real fragment spread; a dot when absent.
          const radiusPx = compact && radius ? Math.max(3, radius.km * (R / 6371.0)) : 3;
          if (showEnvelope) {
            ctx.strokeStyle = PALETTE.caution;
            ctx.globalAlpha = breathe;
            ctx.beginPath();
            ctx.arc(p.sx, p.sy, radiusPx, 0, TWO_PI);
            ctx.stroke();
          }
          ctx.globalAlpha = 1;
          ctx.setLineDash([]);
          const text = cloudLabel(cloud);
          if (!showLabels || seenLabels.has(text)) { ctx.setLineDash([3, 3]); continue; }
          seenLabels.add(text);
          const tw = ctx.measureText(text).width + 10;
          const bx = p.sx - (tw / 2);
          const by = p.sy - radiusPx - 16;
          if (reserve(placedBoxes, bx, by, tw, 13)) drawTag(ctx, bx, by, tw, 13, text, PALETTE.caution);
          ctx.setLineDash([3, 3]);
        }
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }

      // ── Test-mode trajectories ─────────────────────────────────────────
      if (tActive && comp?.trajectoryA && comp?.trajectoryB) {
        ctx.lineWidth = 1;
        for (let k = 0; k < 2; k += 1) {
          const traj = k === 0 ? comp.trajectoryA : comp.trajectoryB;
          ctx.strokeStyle = k === 0 ? PALETTE.accent : PALETTE.info;
          ctx.globalAlpha = 0.6;
          ctx.beginPath();
          let penDown = false;
          for (const pt of traj) {
            const p = projectEci(pt[1], pt[2], pt[3], scratchProj);
            if (!p.visible) { penDown = false; continue; }
            if (penDown) ctx.lineTo(p.sx, p.sy);
            else { ctx.moveTo(p.sx, p.sy); penDown = true; }
          }
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      // ── Advance + project every satellite (no allocation) ──────────────
      const wallSeconds = Math.min((performance.now() - motion.lastWallMs) / 1000, MAX_EXTRAPOLATION_S);
      const simSeconds = wallSeconds * (motion.simRate || 1);
      const correctionWeight = Math.exp(-wallSeconds / CORRECTION_DECAY_S);
      for (const entry of motion.sats.values()) {
        const r = extrapolate(entry.base, entry.velocity, simSeconds, entry.correction, correctionWeight, entry.render);
        projectEci(r.x, r.y, r.z, entry.proj);
      }

      const testA = tActive && tSats?.a && tSats?.b ? tSats.a : null;
      const testB = testA ? tSats.b : null;
      const testIdA = testA ? Number(testA.norad_id) : null;
      const testIdB = testB ? Number(testB.norad_id) : null;
      if (testA) {
        const pa = testA.position || {};
        const pb = testB.position || {};
        projectEci(pa.x ?? 0, pa.y ?? 0, pa.z ?? 0, testPosA);
        projectEci(pb.x ?? 0, pb.y ?? 0, pb.z ?? 0, testPosB);
      }
      const posOf = (id) => {
        if (id === testIdA) return testPosA;
        if (id === testIdB) return testPosB;
        return motion.sats.get(id)?.proj ?? null;
      };
      const cpiOf = (id) => ((id === testIdA || id === testIdB) ? 10 : (cpiById.get(id) ?? 0));

      // ── Selected satellite's orbit: near side only, hidden behind the globe ─
      if (showOrbit && orbit && orbit.length > 1 && selectedId != null) {
        ctx.strokeStyle = PALETTE.accent;
        ctx.globalAlpha = 0.75;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        let penDown = false;
        for (const sample of orbit) {
          const pos = sample.position;
          if (!pos) { penDown = false; continue; }
          const p = projectEci(pos.x, pos.y, pos.z, scratchProj);
          if (!p.visible) { penDown = false; continue; }
          if (penDown) ctx.lineTo(p.sx, p.sy);
          else { ctx.moveTo(p.sx, p.sy); penDown = true; }
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1;
      }

      // Nominal satellites: one batched path of tiny dim dots.
      if (showSats) {
        ctx.fillStyle = PALETTE.dim;
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        for (const [id, entry] of motion.sats) {
          const p = entry.proj;
          if (!p.visible || threatColor(cpiOf(id))) continue;
          ctx.rect(p.sx - 0.75, p.sy - 0.75, 1.5, 1.5);
        }
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      // ── Impact replay: parents, fragments, envelope, flash ─────────────
      if (prep) {
        const { i0, nF, cur, curOk, curSx, curSy, curVis, bucket } = prep;
        // Parent tracks (full propagated window) + moving dots.
        if (showParents) {
          for (const par of prep.parents) {
            ctx.strokeStyle = par.color;
            ctx.lineWidth = 1.25;
            for (let pass = 0; pass < 2; pass += 1) {
              // pass 0: pre-collision (solid); pass 1: post-collision (dashed, faint)
              ctx.setLineDash(pass === 0 ? [] : [2, 4]);
              ctx.globalAlpha = pass === 0 ? 0.6 : 0.25;
              ctx.beginPath();
              let penDown = false;
              for (let si = 0; si < prep.nS; si += 1) {
                const ts = prep.t[si];
                if ((pass === 0 && ts > 0) || (pass === 1 && ts < 0) || !par.ok[si]) { penDown = false; continue; }
                const o = si * 3;
                const p = projectEci(par.pos[o], par.pos[o + 1], par.pos[o + 2], scratchProj);
                if (!p.visible) { penDown = false; continue; }
                if (penDown) ctx.lineTo(p.sx, p.sy);
                else { ctx.moveTo(p.sx, p.sy); penDown = true; }
              }
              ctx.stroke();
            }
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;
            par.proj.visible = false;
            if (!par.cur.ok) continue;
            projectEci(par.cur.x, par.cur.y, par.cur.z, par.proj);
            if (!par.proj.visible) continue;
            ctx.fillStyle = par.color;
            ctx.strokeStyle = par.color;
            ctx.beginPath();
            ctx.arc(par.proj.sx, par.proj.sy, 3, 0, TWO_PI);
            if (tRel < 0) {
              ctx.fill();
            } else {
              // After breakup the intact parent no longer exists: hollow marker.
              ctx.globalAlpha = 0.45;
              ctx.stroke();
              ctx.globalAlpha = 1;
            }
          }
          ctx.lineWidth = 1;
        }

        // Project current fragment positions once (reused by trails/threat lines).
        for (let f = 0; f < nF; f += 1) {
          curVis[f] = 0;
          if (!curOk[f]) continue;
          const o = f * 3;
          const p = projectEci(cur[o], cur[o + 1], cur[o + 2], scratchProj);
          if (!p.visible) continue;
          curVis[f] = 1;
          curSx[f] = p.sx;
          curSy[f] = p.sy;
        }
        const buckets = colorMode === 'parent' ? bucket.parent : colorMode === 'size' ? bucket.size : bucket.dv;
        const colors = colorMode === 'parent' ? FRAG_PARENT : FRAG_RAMP;

        // Short trails: the last TRAIL_SAMPLES real samples → current position.
        if (showTrails && showFrags) {
          const first = Math.max(0, i0 - TRAIL_SAMPLES);
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.35;
          for (let b = 0; b < colors.length; b += 1) {
            ctx.strokeStyle = colors[b];
            ctx.beginPath();
            for (let f = 0; f < nF; f += 1) {
              if (buckets[f] !== b || !curVis[f]) continue;
              let penDown = false;
              for (let si = first; si <= i0; si += 1) {
                const k = (si * nF) + f;
                if (!prep.fragOk[k]) { penDown = false; continue; }
                const o = k * 3;
                const p = projectEci(prep.fragPos[o], prep.fragPos[o + 1], prep.fragPos[o + 2], scratchProj);
                if (!p.visible) { penDown = false; continue; }
                if (penDown) ctx.lineTo(p.sx, p.sy);
                else { ctx.moveTo(p.sx, p.sy); penDown = true; }
              }
              if (penDown) ctx.lineTo(curSx[f], curSy[f]);
            }
            ctx.stroke();
          }
          ctx.globalAlpha = 1;
        }

        // Debris envelope: p90 ring around the interpolated centroid.
        if (showEnvelope && tRel >= 0) {
          const { i1, a, envOk, envC, envR } = prep;
          if (envOk[i0] && (a === 0 || envOk[i1])) {
            const b0 = 1 - a;
            const o0 = i0 * 3;
            const o1 = i1 * 3;
            const ex = (envC[o0] * b0) + (envC[o1] * a);
            const ey = (envC[o0 + 1] * b0) + (envC[o1 + 1] * a);
            const ez = (envC[o0 + 2] * b0) + (envC[o1 + 2] * a);
            const rKm = (envR[i0] * b0) + (envR[i1] * a);
            const p = projectEci(ex, ey, ez, scratchProj);
            // A p90 spread of thousands of km (diverged streams) is not a ring.
            if (p.visible && rKm <= MAX_RING_KM * 4) {
              ctx.setLineDash([3, 3]);
              ctx.strokeStyle = PALETTE.caution;
              ctx.globalAlpha = 0.55;
              ctx.beginPath();
              ctx.arc(p.sx, p.sy, Math.max(4, rKm * (R / 6371.0)), 0, TWO_PI);
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.globalAlpha = 1;
            }
          }
        }

        // Fragments: one batched path per colour bucket.
        if (showFrags) {
          ctx.globalAlpha = 0.9;
          for (let b = 0; b < colors.length; b += 1) {
            ctx.fillStyle = colors[b];
            ctx.beginPath();
            for (let f = 0; f < nF; f += 1) {
              if (buckets[f] !== b || !curVis[f]) continue;
              ctx.rect(curSx[f] - 1, curSy[f] - 1, 2, 2);
            }
            ctx.fill();
          }
          ctx.globalAlpha = 1;
        }

        // Collision marker (after breakup) and the flash / shockwave.
        if (prep.hasCollision && (showFrags || showParents)) {
          const c = prep.collision;
          const p = projectEci(c[0], c[1], c[2], scratchProj);
          if (p.visible) {
            const age = nowMs - flashStartMs;
            if (age >= 0 && age < FLASH_MS) {
              const k = age / FLASH_MS;
              const ease = 1 - ((1 - k) ** 3);
              const fade = 1 - k;
              const scale = Math.max(0.6, R / 300);
              const rCore = (10 + (ease * 34)) * scale;
              const g = ctx.createRadialGradient(p.sx, p.sy, 0, p.sx, p.sy, rCore);
              g.addColorStop(0, `rgba(255,248,235,${(0.95 * fade).toFixed(3)})`);
              g.addColorStop(0.35, `rgba(255,214,140,${(0.7 * fade).toFixed(3)})`);
              g.addColorStop(1, 'rgba(240,180,41,0)');
              ctx.fillStyle = g;
              ctx.beginPath();
              ctx.arc(p.sx, p.sy, rCore, 0, TWO_PI);
              ctx.fill();
              ctx.strokeStyle = PALETTE.caution;
              ctx.lineWidth = (2 * fade) + 0.5;
              ctx.globalAlpha = fade;
              ctx.beginPath();
              ctx.arc(p.sx, p.sy, (6 + (ease * 90)) * scale, 0, TWO_PI);
              ctx.stroke();
              ctx.globalAlpha = fade * 0.5;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.arc(p.sx, p.sy, (4 + (ease * 55)) * scale, 0, TWO_PI);
              ctx.stroke();
              ctx.globalAlpha = 1;
            }
            if (tRel >= 0) {
              ctx.strokeStyle = PALETTE.bright;
              ctx.globalAlpha = 0.8;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(p.sx - 3, p.sy - 3); ctx.lineTo(p.sx + 3, p.sy + 3);
              ctx.moveTo(p.sx + 3, p.sy - 3); ctx.lineTo(p.sx - 3, p.sy + 3);
              ctx.stroke();
              ctx.globalAlpha = 1;
            }
          }
        }
      }

      // ── Conjunction lines (+ a pill for the top few) ───────────────────
      ctx.font = MONO_9;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      let pillCount = 0;
      const pills = [];
      // Alerts are future encounters, so the two objects can be on opposite
      // sides of the Earth right now. This canvas has no depth test: a chord
      // between distant objects would be drawn straight across the disc. Only
      // link a pair while it is genuinely close (short chord ≈ on-surface).
      const eciOf = (id) => {
        if (id === testIdA) return testA.position;
        if (id === testIdB) return testB.position;
        return motion.sats.get(id)?.render ?? null;
      };
      for (const alert of showLinks ? ranked : EMPTY) {
        const isDebris = alert.source === 'debris';
        const idA = Number(alert.sat1?.id);
        const idB = Number(alert.sat2?.id);
        let pA = posOf(idA);
        let pB = posOf(idB);
        let rA = eciOf(idA);
        let rB = eciOf(idB);
        if (isDebris && (!pA || !pB)) {
          // The fragment has no TLE: anchor at its own reported state.
          const fr = alert.fragment_state?.r_km;
          const anchor = Array.isArray(fr) && fr.length === 3
            ? { x: Number(fr[0]), y: Number(fr[1]), z: Number(fr[2]) } : null;
          if (anchor && !aboveSurface(anchor)) continue;
          const proj = anchor ? projectEci(anchor.x, anchor.y, anchor.z, { sx: 0, sy: 0, visible: false }) : null;
          if (!pA) { pA = proj; rA = anchor; } else { pB = proj; rB = anchor; }
        }
        if (!pA || !pB || !pA.visible || !pB.visible || !rA || !rB) continue;
        const sepKm = Math.hypot((rA.x ?? 0) - (rB.x ?? 0), (rA.y ?? 0) - (rB.y ?? 0), (rA.z ?? 0) - (rB.z ?? 0));
        if (!(sepKm <= MAX_LINK_KM)) continue;
        const isTop = pillCount < MAX_CONJUNCTION_LABELS;
        const color = threatColor(alertLevel(alert));
        ctx.setLineDash(isDebris ? [1, 3] : [4, 3]);
        ctx.strokeStyle = isDebris ? PALETTE.caution : isTop && color ? color : PALETTE.lineStrong;
        ctx.globalAlpha = isTop ? 0.7 : 0.9;
        ctx.beginPath();
        ctx.moveTo(pA.sx, pA.sy);
        ctx.lineTo(pB.sx, pB.sy);
        ctx.stroke();
        if (!isTop) continue;
        pillCount += 1;
        pills.push(alert, pA, pB);
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      // ── Threatened + selected satellites ───────────────────────────────
      const drawThreatDot = (id, p) => {
        if (!p || !p.visible) return;
        const color = showSats ? threatColor(cpiOf(id)) : null;
        if (color) {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, cpiOf(id) >= 8 ? 2.5 : 2, 0, TWO_PI);
          ctx.fill();
        }
        if (id === selectedId) {
          if (!color) {
            ctx.fillStyle = PALETTE.accent;
            ctx.beginPath();
            ctx.arc(p.sx, p.sy, 2, 0, TWO_PI);
            ctx.fill();
          }
          ctx.strokeStyle = PALETTE.accent;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, 5, 0, TWO_PI);
          ctx.stroke();
        }
      };
      // Caution first, warning on top.
      for (let pass = 0; pass < 2; pass += 1) {
        for (const [id, cpi] of cpiById) {
          if ((pass === 0) === (cpi >= 8)) continue;
          if (id === testIdA || id === testIdB) continue;
          drawThreatDot(id, posOf(id));
        }
      }
      if (testA) {
        drawThreatDot(testIdA, testPosA);
        drawThreatDot(testIdB, testPosB);
      }
      if (selectedId != null && !cpiById.has(Number(selectedId))) {
        const sid = Number(selectedId);
        drawThreatDot(sid, posOf(sid));
      }

      // ── Limb ───────────────────────────────────────────────────────────
      ctx.beginPath();
      ctx.arc(cx, cy, R + 0.5, 0, TWO_PI);
      ctx.strokeStyle = PALETTE.lineStrong;
      ctx.lineWidth = 1;
      ctx.stroke();

      // ── Impact: satellites threatened by this event's fragments ────────
      // Red pulse ring + dashed line to the nearest live (interpolated) fragment.
      ctx.font = MONO_9;
      ctx.textBaseline = 'middle';
      if (prep && showThreatened && prep.threatened.length > 0) {
        const phase = (nowMs % 1400) / 1400;
        const { nF, cur, curOk, curVis, curSx, curSy } = prep;
        let threatCallouts = 0;
        for (const th of prep.threatened) {
          const entry = motion.sats.get(Number(th?.sat_id));
          if (!entry || !entry.proj.visible) continue;
          const p = entry.proj;
          const r = entry.render;
          let best = -1;
          let bestD2 = Infinity;
          for (let f = 0; f < nF; f += 1) {
            if (!curOk[f]) continue;
            const o = f * 3;
            const dx = cur[o] - r.x;
            const dy = cur[o + 1] - r.y;
            const dz = cur[o + 2] - r.z;
            const d2 = (dx * dx) + (dy * dy) + (dz * dz);
            if (d2 < bestD2) { bestD2 = d2; best = f; }
          }
          if (best >= 0 && curVis[best]) {
            ctx.strokeStyle = PALETTE.warning;
            ctx.globalAlpha = 0.6;
            ctx.setLineDash([2, 3]);
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(p.sx, p.sy);
            ctx.lineTo(curSx[best], curSy[best]);
            ctx.stroke();
            ctx.setLineDash([]);
          }
          ctx.globalAlpha = 1;
          ctx.fillStyle = PALETTE.warning;
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, 2.75, 0, TWO_PI);
          ctx.fill();
          ctx.strokeStyle = PALETTE.warning;
          ctx.lineWidth = 1.25;
          ctx.globalAlpha = 1 - phase;
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, 4 + (phase * 9), 0, TWO_PI);
          ctx.stroke();
          ctx.globalAlpha = 1;
          ctx.lineWidth = 1;
          if (showLabels && threatCallouts < MAX_THREAT_CALLOUTS) {
            const miss = formatMiss(Number(th.miss_km));
            const name = String(th.name ?? entry.name ?? `#${th.sat_id}`).slice(0, 14).toUpperCase();
            if (drawCallout(ctx, p, miss ? `${name}  ${miss}` : name, PALETTE.warning, W, H)) threatCallouts += 1;
          }
        }
      }

      // ── Callout labels, outside the disc (near-side objects only) ──────
      if (showLabels) {
        if (prep && showParents) {
          for (const par of prep.parents) {
            if (par.cur.ok && par.proj.visible) {
              drawCallout(ctx, par.proj, String(par.name).slice(0, 16).toUpperCase(), par.color, W, H);
            }
          }
        }
        for (let i = 0; i < pills.length; i += 3) {
          const alert = pills[i];
          const pA = pills[i + 1];
          const pB = pills[i + 2];
          const missKm = alert.miss_distance_km == null ? null : Number(alert.miss_distance_km);
          const miss = formatMiss(Number.isFinite(missKm) ? missKm : null);
          const tca = formatTca(alert);
          const core = miss && tca ? `${miss}  ${tca}` : (miss || tca);
          if (!core) continue;
          const text = alert.source === 'debris' ? `DEB  ${core}` : core;
          midpoint.sx = (pA.sx + pB.sx) / 2;
          midpoint.sy = (pA.sy + pB.sy) / 2;
          drawCallout(ctx, midpoint, text, threatColor(alertLevel(alert)) || PALETTE.text, W, H);
        }
        if (selectedId != null) {
          const sid = Number(selectedId);
          const p = posOf(sid);
          if (p && p.visible) {
            const name = sid === testIdA ? testA.name : sid === testIdB ? testB.name : motion.sats.get(sid)?.name;
            drawCallout(ctx, p, (name ?? `#${sid}`).slice(0, 18).toUpperCase(), PALETTE.bright, W, H);
          }
        }
      }

      ctx.font = CAPS_10;
      if ('letterSpacing' in ctx) ctx.letterSpacing = '1.4px';
      ctx.fillStyle = PALETTE.dim;
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(`${cpiById.size} AT RISK`, 10, H - 24);
      ctx.fillText(`${alertCount} CONJUNCTIONS`, 10, H - 10);

      // Impact replay readout + colour legend (top-left, under the panel label).
      if (prep) {
        ctx.fillStyle = tRel >= 0 ? PALETTE.caution : PALETTE.text;
        ctx.fillText(`IMPACT REPLAY  ${formatTRel(tRel)}`, 10, 44);
        if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
        ctx.font = MONO_9;
        ctx.fillStyle = PALETTE.dim;
        ctx.fillText(`${prep.alive}/${prep.nF} FRAGMENTS · ${colorMode.toUpperCase()} COLOUR`, 10, 58);
        ctx.fillText(`LINEAR INTERP. BETWEEN ${Math.round(prep.stepS)} S PROPAGATED SAMPLES`, 10, 70);
        const cols = colorMode === 'parent' ? FRAG_PARENT : FRAG_RAMP;
        const n = colorMode === 'parent' ? Math.min(2, prep.parents.length || 2) : FRAG_RAMP.length;
        let lx = 10;
        for (let i = 0; i < n; i += 1) {
          const txt = colorMode === 'parent'
            ? `${String(prep.parents[i]?.name ?? (i === 0 ? 'A' : 'B')).slice(0, 12).toUpperCase()} FRAG`
            : (colorMode === 'size' ? SIZE_LEGEND : DV_LEGEND)[i];
          ctx.fillStyle = cols[i];
          ctx.fillRect(lx, 79, 6, 6);
          ctx.fillStyle = PALETTE.dim;
          ctx.fillText(txt, lx + 9, 85);
          lx += ctx.measureText(txt).width + 20;
        }
      }
      if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';

      // Interaction hint (bottom-right, dim) + orientation indicator.
      ctx.font = MONO_9;
      ctx.textAlign = 'right';
      ctx.fillStyle = PALETTE.dim;
      ctx.globalAlpha = 0.6;
      ctx.fillText(HINT_TEXT, W - 10, H - 10);
      ctx.globalAlpha = 1;
      ctx.textAlign = 'left';
      drawCompass(W);
    };

    const tick = (nowMs) => {
      rafId = 0;
      if (!running) return;
      rafId = requestAnimationFrame(tick);
      if (nowMs - lastFrameMs < FRAME_INTERVAL_MS - 1) return; // ~30 fps cap
      lastFrameMs = nowMs;
      // Time-based motion, independent of frame rate and pauses.
      const dt = Math.min((nowMs - lastTickMs) / 1000, 0.1);
      lastTickMs = nowMs;
      if (!drag.active) {
        // Inertia glide after a fling (gentle exponential decay).
        nav.lon += nav.vLon * dt;
        nav.lat = Math.max(-MAX_TILT_DEG, Math.min(MAX_TILT_DEG, nav.lat + (nav.vLat * dt)));
        const decay = Math.exp(-INERTIA_DECAY_PER_S * dt);
        nav.vLon *= decay;
        nav.vLat *= decay;
        if (Math.abs(nav.vLon) < 0.05) nav.vLon = 0;
        if (Math.abs(nav.vLat) < 0.05) nav.vLat = 0;
      }
      if (nav.flying) {
        const k = 1 - Math.exp(-5 * dt);
        const dLon = ((((nav.flyLon - nav.lon) % 360) + 540) % 360) - 180; // shortest way round
        nav.lon += dLon * k;
        nav.lat += (nav.flyLat - nav.lat) * k;
        nav.lastInteractMs = nowMs; // hold the view on the target a while
        nav.autoBlend = 0;
        if (Math.abs(dLon) < 0.1 && Math.abs(nav.flyLat - nav.lat) < 0.1) nav.flying = false;
      }
      if (nav.resetting) {
        const k = 1 - Math.exp(-8 * dt);
        nav.lat += (0 - nav.lat) * k;
        nav.zoom += (1 - nav.zoom) * k;
        if (Math.abs(nav.lat) < 0.05 && Math.abs(nav.zoom - 1) < 0.002) {
          nav.lat = 0;
          nav.zoom = 1;
          nav.resetting = false;
        }
      }
      // Auto-rotation pauses while interacting and eases back in after idle.
      const target = nowMs - nav.lastInteractMs > AUTO_RESUME_MS ? 1 : 0;
      nav.autoBlend += (target - nav.autoBlend) * Math.min(1, dt * 1.2);
      nav.lon = (nav.lon + (ROTATION_DEG_PER_S * nav.autoBlend * dt)) % 360;
      draw(nowMs);
    };

    const start = () => {
      if (running) return;
      running = true;
      lastTickMs = performance.now();
      lastFrameMs = 0;
      rafId = requestAnimationFrame(tick);
    };
    const stop = () => {
      running = false;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = 0;
    };

    // ── Pointer interaction: drag rotate, pinch/wheel zoom, click select ──
    const markInteract = () => {
      nav.lastInteractMs = performance.now();
      nav.autoBlend = 0;
      nav.resetting = false;
      nav.flying = false;
    };
    const localX = (evt) => evt.clientX - canvas.getBoundingClientRect().left;
    const localY = (evt) => evt.clientY - canvas.getBoundingClientRect().top;
    const pinchDist = () => {
      const it = pointers.values();
      const a = it.next().value;
      const b = it.next().value;
      return Math.max(1, Math.hypot(a.x - b.x, a.y - b.y));
    };
    const selectAt = (x, y) => {
      if (!layerOn(useStore.getState().layers, 'satellites')) return;
      let bestId = null;
      let bestD2 = 10 * 10; // px pick radius
      for (const [id, entry] of motionRef.current.sats) {
        const p = entry.proj;
        if (!p.visible) continue;
        const d2 = ((p.sx - x) ** 2) + ((p.sy - y) ** 2);
        if (d2 < bestD2) { bestD2 = d2; bestId = id; }
      }
      if (bestId != null) useStore.getState().setSelectedSatelliteId?.(bestId);
    };
    const onPointerDown = (evt) => {
      if (evt.pointerType === 'mouse' && evt.button !== 0) return;
      const x = localX(evt);
      const y = localY(evt);
      pointers.set(evt.pointerId, { x, y });
      try { canvas.setPointerCapture(evt.pointerId); } catch { /* capture is best-effort */ }
      if (pointers.size === 1) {
        drag.active = true;
        drag.moved = false;
        drag.startX = x; drag.startY = y;
        drag.lastX = x; drag.lastY = y;
        drag.lastT = performance.now();
        nav.vLon = 0;
        nav.vLat = 0;
        markInteract();
      } else if (pointers.size === 2) {
        pinch.active = true;
        pinch.dist = pinchDist();
        pinch.zoom = nav.zoom;
        drag.moved = true; // a pinch is never a click
      }
    };
    const onPointerMove = (evt) => {
      const pt = pointers.get(evt.pointerId);
      if (!pt) return;
      const x = localX(evt);
      const y = localY(evt);
      pt.x = x; pt.y = y;
      if (pinch.active && pointers.size >= 2) {
        nav.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, pinch.zoom * (pinchDist() / pinch.dist)));
        markInteract();
        return;
      }
      if (!drag.active) return;
      if (!drag.moved) {
        if (Math.hypot(x - drag.startX, y - drag.startY) <= DRAG_THRESHOLD_PX) return;
        drag.moved = true;
        canvas.style.cursor = 'grabbing';
      }
      const now = performance.now();
      const dtS = Math.max((now - drag.lastT) / 1000, 1 / 240);
      const R = Math.max(view.R, 1);
      // The surface under the cursor follows it: right drag → globe turns right.
      const dLon = -((x - drag.lastX) / R) * RAD2DEG;
      const dLat = ((y - drag.lastY) / R) * RAD2DEG;
      nav.lon += dLon;
      nav.lat = Math.max(-MAX_TILT_DEG, Math.min(MAX_TILT_DEG, nav.lat + dLat));
      const clampV = (v) => Math.max(-MAX_SPIN_DEG_PER_S, Math.min(MAX_SPIN_DEG_PER_S, v));
      nav.vLon = clampV((0.6 * (dLon / dtS)) + (0.4 * nav.vLon));
      nav.vLat = clampV((0.6 * (dLat / dtS)) + (0.4 * nav.vLat));
      drag.lastX = x; drag.lastY = y; drag.lastT = now;
      markInteract();
    };
    const endPointer = (evt, isClick) => {
      if (!pointers.has(evt.pointerId)) return;
      pointers.delete(evt.pointerId);
      try { canvas.releasePointerCapture(evt.pointerId); } catch { /* already released */ }
      if (pointers.size < 2) pinch.active = false;
      if (pointers.size > 0) return;
      if (drag.active) {
        if (isClick && !drag.moved) selectAt(localX(evt), localY(evt));
        // Held still before release → no fling.
        if (performance.now() - drag.lastT > 80) { nav.vLon = 0; nav.vLat = 0; }
        markInteract();
      }
      drag.active = false;
      canvas.style.cursor = 'grab';
    };
    const onPointerUp = (evt) => endPointer(evt, true);
    const onPointerCancel = (evt) => endPointer(evt, false);
    const onWheel = (evt) => {
      evt.preventDefault();
      const unit = evt.deltaMode === 1 ? 16 : evt.deltaMode === 2 ? 400 : 1;
      nav.zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, nav.zoom * Math.exp(-evt.deltaY * unit * 0.0012)));
      markInteract();
    };
    const onDblClick = (evt) => {
      evt.preventDefault();
      nav.vLon = 0;
      nav.vLat = 0;
      nav.resetting = true;
      nav.lastInteractMs = -Infinity; // auto-rotation resumes straight away
    };
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointercancel', onPointerCancel);
    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('dblclick', onDblClick);

    // "Fly to impact" from the impact console: turn the collision point to face us.
    const stopFly = onFlyTo((target) => {
      const e = target?.eci;
      if (!Array.isArray(e) || e.length < 3) return;
      const x = Number(e[0]); const y = Number(e[1]); const z = Number(e[2]);
      const r = Math.hypot(x, y, z);
      if (!(r > 0)) return;
      nav.flyLon = Math.atan2(y, x) * RAD2DEG;
      nav.flyLat = Math.max(-MAX_TILT_DEG, Math.min(MAX_TILT_DEG, Math.asin(z / r) * RAD2DEG));
      nav.flying = true;
      nav.resetting = false;
      nav.vLon = 0;
      nav.vLat = 0;
    });

    // Pause when the tab is hidden or the canvas is offscreen.
    const stopActivity = observeActivity(canvas, (active) => (active ? start() : stop()));

    return () => {
      stopFly();
      stopActivity();
      stop();
      ro.disconnect();
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerCancel);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('dblclick', onDblClick);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      data-testid="threat-globe"
      style={{ width: '100%', height: '100%', display: 'block', cursor: 'grab', touchAction: 'none', background: PALETTE.void }}
    />
  );
}
