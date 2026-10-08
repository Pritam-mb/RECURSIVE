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
 * Orthographic view state for one frame. Rotating the globe is a rotation
 * about the polar axis, so a unit vector (ux, uy, uz) projects to
 *   sx = cx + (uy·cosV − ux·sinV)·R,  sy = cy − uz·R,  depth = ux·cosV + uy·sinV
 */
const view = { cx: 0, cy: 0, R: 1, cosV: 1, sinV: 0 };

function strokeUnitPolylines(ctx, lines) {
  const { cx, cy, R, cosV, sinV } = view;
  ctx.beginPath();
  for (const arr of lines) {
    let penDown = false;
    for (let i = 0; i < arr.length; i += 3) {
      const ux = arr[i];
      const uy = arr[i + 1];
      if ((ux * cosV) + (uy * sinV) < 0) { penDown = false; continue; }
      const sx = cx + (((uy * cosV) - (ux * sinV)) * R);
      const sy = cy - (arr[i + 2] * R);
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
  const { cx, cy, R, cosV, sinV } = view;
  const k = R * displayScale(r);
  const px = ((uy * cosV) - (ux * sinV)) * k;
  const py = uz * k;
  out.sx = cx + px;
  out.sy = cy - py;
  out.visible = (ux * cosV) + (uy * sinV) >= 0 || Math.hypot(px, py) > R + 1;
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

const testPosA = { sx: 0, sy: 0, visible: false };
const testPosB = { sx: 0, sy: 0, visible: false };
const scratchProj = { sx: 0, sy: 0, visible: false };
const fragAnchor = { sx: 0, sy: 0, visible: true };
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
    let viewLon = 80;

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

      view.cx = W / 2;
      view.cy = H / 2;
      view.R = Math.min(W, H) * 0.40;
      view.cosV = Math.cos(viewLon * DEG);
      view.sinV = Math.sin(viewLon * DEG);
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

      // ── Debris clouds: thin dashed rings, slow breathing opacity ───────
      if (clouds && clouds.length > 0) {
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
            ctx.rect(fp.sx - 0.75, fp.sy - 0.75, 1.5, 1.5);
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
          ctx.strokeStyle = PALETTE.caution;
          ctx.globalAlpha = breathe;
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, radiusPx, 0, TWO_PI);
          ctx.stroke();
          ctx.globalAlpha = 1;
          ctx.setLineDash([]);
          const text = cloudLabel(cloud);
          if (seenLabels.has(text)) { ctx.setLineDash([3, 3]); continue; }
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
      if (orbit && orbit.length > 1 && selectedId != null) {
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
      for (const alert of ranked) {
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
        const color = threatColor(cpiOf(id));
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

      // ── Limb + readout ─────────────────────────────────────────────────
      ctx.beginPath();
      ctx.arc(cx, cy, R + 0.5, 0, TWO_PI);
      ctx.strokeStyle = PALETTE.lineStrong;
      ctx.lineWidth = 1;
      ctx.stroke();

      // ── Callout labels, outside the disc (near-side objects only) ──────
      ctx.font = MONO_9;
      ctx.textBaseline = 'middle';
      const midpoint = { sx: 0, sy: 0 };
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

      ctx.font = CAPS_10;
      if ('letterSpacing' in ctx) ctx.letterSpacing = '1.4px';
      ctx.fillStyle = PALETTE.dim;
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(`${cpiById.size} AT RISK`, 10, H - 24);
      ctx.fillText(`${alertCount} CONJUNCTIONS`, 10, H - 10);
      if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
    };

    const tick = (nowMs) => {
      rafId = 0;
      if (!running) return;
      rafId = requestAnimationFrame(tick);
      if (nowMs - lastFrameMs < FRAME_INTERVAL_MS - 1) return; // ~30 fps cap
      lastFrameMs = nowMs;
      // Time-based rotation, independent of frame rate and pauses.
      const dt = Math.min((nowMs - lastTickMs) / 1000, 0.1);
      lastTickMs = nowMs;
      viewLon = (viewLon + (ROTATION_DEG_PER_S * dt)) % 360;
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

    // Pause when the tab is hidden or the canvas is offscreen.
    const stopActivity = observeActivity(canvas, (active) => (active ? start() : stop()));

    return () => {
      stopActivity();
      stop();
      ro.disconnect();
    };
  }, []);

  // Click a dot to select it (same selection as the live globe).
  const setSelectedSatelliteId = useStore((s) => s.setSelectedSatelliteId);
  const handleClick = (evt) => {
    const rect = evt.currentTarget.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;
    let bestId = null;
    let bestD2 = 10 * 10; // px pick radius
    for (const [id, entry] of motionRef.current.sats) {
      const p = entry.proj;
      if (!p.visible) continue;
      const d2 = ((p.sx - x) ** 2) + ((p.sy - y) ** 2);
      if (d2 < bestD2) { bestD2 = d2; bestId = id; }
    }
    if (bestId != null) setSelectedSatelliteId(bestId);
  };

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      style={{ width: '100%', height: '100%', display: 'block', cursor: 'crosshair', background: PALETTE.void }}
    />
  );
}
