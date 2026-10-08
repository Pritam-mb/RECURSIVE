/**
 * Cesium primitives for the debris impact replay (live globe).
 *
 * Everything is driven from CesiumGlobe's scene.preUpdate: update() reads the
 * smooth playhead, interpolates the replay samples (linear, labelled in the
 * console) and moves existing primitives — nothing is rebuilt per frame except
 * polyline vertex lists. Layers toggle whole collections (`.show`).
 *
 * Frame: all ECI positions are rotated by GMST at the playhead instant, so the
 * Earth is oriented correctly under the objects at that moment; trajectories
 * and trails are drawn as inertial paths in that same orientation, so every
 * moving point sits exactly on its line.
 */
import * as Cesium from 'cesium';
import { computeGmst } from '../../utils/coords';
import { clockPlaying, playheadNow } from './impactClock';
import {
  DV_EDGES_MS, FRAG_PARENT_COLORS, FRAG_RAMP, SIZE_EDGES_M,
  binIndex, lerpXyz, parseReplay, sampleAt,
} from './replayData';

const RE_KM = 6378.137;
const FLASH_MS = 3200;
const THREAT_WINDOW_S = 15 * 60;
// Only link a threatened satellite to its fragment once they are this close.
const THREAT_LINK_KM = 1500;
const TRAIL_SAMPLES = 4;
const TRAIL_MAX_VERTICES = 1400;
const ENV_MAX_STEP_RAD = (1.5 * Math.PI) / 180;
const SHOCK_MAX_KM = 420;
const LABEL_FONT = "500 11px 'IBM Plex Mono', ui-monospace, monospace";

const col = (hex, a = 1) => Cesium.Color.fromCssColorString(hex).withAlpha(a);
const C = {
  parent: [col('#4c8dff'), col('#7cc4ff')],
  parentDim: [col('#4c8dff', 0.4), col('#7cc4ff', 0.4)],
  parentFuture: [col('#4c8dff', 0.55), col('#7cc4ff', 0.55)],
  void: col('#030508'),
  clear: col('#000000', 0),
  labelText: col('#eef2f6'),
  labelBg: col('#030508', 0.88),
  warning: col('#ff5a4f'),
  warningSoft: col('#ff5a4f', 0.55),
  envGlow: col('#f0b429', 0.13),
  envCore: col('#f0b429', 0.5),
  flashHot: col('#fff6e0'),
  flashAmber: col('#f0b429'),
};
const PARENT_FRAG = FRAG_PARENT_COLORS.map((h) => col(h, 0.92));
const RAMP = FRAG_RAMP.map((h) => col(h, 0.92));
const PX_BY_BIN = [2, 2.8, 3.6, 4.6];

const SCR_A = new Cesium.Color();
// Setters clone; never pass a primitive's own getter object as `result`.
const SCR_POS = new Cesium.Cartesian3();
const SCR_B = new Cesium.Color();
const easeOut = (x) => 1 - ((1 - x) ** 3);

function segmentClearsEarth(a, b) {
  const dx = b[0] - a[0]; const dy = b[1] - a[1]; const dz = b[2] - a[2];
  const len2 = (dx * dx) + (dy * dy) + (dz * dz);
  const t = len2 > 0 ? Math.min(1, Math.max(0, -((a[0] * dx) + (a[1] * dy) + (a[2] * dz)) / len2)) : 0;
  const px = a[0] + (t * dx); const py = a[1] + (t * dy); const pz = a[2] + (t * dz);
  return Math.sqrt((px * px) + (py * py) + (pz * pz)) > RE_KM;
}

const above = (x, y, z) => ((x * x) + (y * y) + (z * z)) > RE_KM * RE_KM;

export default class ImpactScene {
  constructor(scene) {
    this.scene = scene;
    const P = scene.primitives;
    this.envLines = P.add(new Cesium.PolylineCollection());
    this.trailLines = P.add(new Cesium.PolylineCollection());
    this.trackLines = P.add(new Cesium.PolylineCollection());
    this.threatLines = P.add(new Cesium.PolylineCollection());
    this.fragPoints = P.add(new Cesium.PointPrimitiveCollection());
    this.markPoints = P.add(new Cesium.PointPrimitiveCollection());
    this.flashPoints = P.add(new Cesium.PointPrimitiveCollection());
    this.labels = P.add(new Cesium.LabelCollection());
    this.shockMaterial = Cesium.Material.fromType('Color', { color: C.clear });
    this.shock = P.add(new Cesium.Primitive({
      geometryInstances: new Cesium.GeometryInstance({
        geometry: new Cesium.EllipsoidGeometry({
          radii: new Cesium.Cartesian3(1, 1, 1),
          stackPartitions: 24,
          slicePartitions: 32,
          vertexFormat: Cesium.MaterialAppearance.MaterialSupport.BASIC.vertexFormat,
        }),
      }),
      appearance: new Cesium.MaterialAppearance({
        material: this.shockMaterial,
        translucent: true,
        flat: true,
        faceForward: true,
        closed: false,
      }),
      asynchronous: false,
      show: false,
    }));
    this.shockMatrix = new Cesium.Matrix4();
    // Polyline._destroy() destroys its material, so materials are never
    // shared between polylines (same type+uniforms still batch together).
    this.trailMaterial = () => Cesium.Material.fromType('Color', { color: col('#f0b429', 0.35) });
    this.threatMaterial = () => Cesium.Material.fromType('Color', { color: C.warningSoft });

    this.model = null;
    this.layers = {};
    this.colorMode = 'parent';
    this.lastT = null;
    this.dirty = true;
    this.flashStart = -Infinity;
    this.flashVisible = false;
    this.ringsVisible = false;
    this.S = { i: 0, a: 0 };
    this.tmp = new Float64Array(3);
    this.tmp2 = new Float64Array(3);
    this.cos = 1;
    this.sin = 0;
    this.stats = { alive: 0 };
    this.clear();
  }

  // ── setup ──────────────────────────────────────────────────────────────
  clear() {
    this.envLines.removeAll();
    this.trailLines.removeAll();
    this.trackLines.removeAll();
    this.threatLines.removeAll();
    this.fragPoints.removeAll();
    this.markPoints.removeAll();
    this.flashPoints.removeAll();
    this.labels.removeAll();
    this.shock.show = false;
    this.parentItems = [];
    this.fragItems = [];
    this.trailItems = [];
    this.envItems = [];
    this.threatItems = [];
    this.flash = null;
    this.cpEci = null;
    this.fragNow = null;
    this.fragAlive = null;
    this.lastT = null;
    this.flashStart = -Infinity;
  }

  setReplay(replay, satellites) {
    this.clear();
    this.model = replay ? parseReplay(replay, satellites) : null;
    const m = this.model;
    this.dirty = true;
    if (!m) { this.scene.requestRender(); return; }

    this.fragNow = new Float64Array(m.F * 3);
    this.fragAlive = new Uint8Array(m.F);

    m.parents.slice(0, 2).forEach((p, k) => {
      this.parentItems.push({
        p,
        k,
        past: this.trackLines.add({
          positions: [], width: 2, arcType: Cesium.ArcType.NONE,
          material: Cesium.Material.fromType('Color', { color: C.parent[k] }),
        }),
        future: this.trackLines.add({
          positions: [], width: 1.5, arcType: Cesium.ArcType.NONE,
          material: Cesium.Material.fromType(Cesium.Material.PolylineDashType, {
            color: C.parentFuture[k], gapColor: C.clear, dashLength: 12,
          }),
        }),
        point: this.markPoints.add({
          show: false, pixelSize: 9, color: C.parent[k], outlineColor: C.void, outlineWidth: 1.5,
        }),
        label: this.labels.add({
          show: false, text: p.name, font: LABEL_FONT, fillColor: C.parent[k],
          style: Cesium.LabelStyle.FILL, showBackground: true, backgroundColor: C.labelBg,
          backgroundPadding: new Cesium.Cartesian2(5, 3), pixelOffset: new Cesium.Cartesian2(10, k === 0 ? -10 : 10),
          horizontalOrigin: Cesium.HorizontalOrigin.LEFT, verticalOrigin: Cesium.VerticalOrigin.CENTER,
        }),
      });
      this.envItems.push({
        k,
        glow: this.envLines.add({ show: false, positions: [], width: 9, arcType: Cesium.ArcType.NONE, material: Cesium.Material.fromType('Color', { color: C.envGlow }) }),
        core: this.envLines.add({ show: false, positions: [], width: 1.5, arcType: Cesium.ArcType.NONE, material: Cesium.Material.fromType('Color', { color: C.envCore }) }),
      });
    });

    for (let f = 0; f < m.F; f += 1) {
      this.fragItems.push(this.fragPoints.add({ show: false, pixelSize: 3, color: RAMP[2] }));
    }
    const trailCount = Math.min(m.F, Math.floor(TRAIL_MAX_VERTICES / (TRAIL_SAMPLES + 1)));
    const stride = trailCount > 0 ? m.F / trailCount : 1;
    for (let n = 0; n < trailCount; n += 1) {
      const f = Math.floor(n * stride);
      this.trailItems.push({ f, line: this.trailLines.add({ show: false, positions: [], width: 1, arcType: Cesium.ArcType.NONE, material: this.trailMaterial() }) });
    }

    for (const th of m.threatened) {
      this.threatItems.push({
        th,
        dot: this.markPoints.add({ show: false, pixelSize: 6, color: C.warning, outlineColor: C.void, outlineWidth: 1 }),
        ring: this.markPoints.add({ show: false, pixelSize: 18, color: C.clear, outlineColor: C.warning, outlineWidth: 2 }),
        line: this.threatLines.add({ show: false, positions: [], width: 1, arcType: Cesium.ArcType.NONE, material: this.threatMaterial() }),
        label: this.labels.add({
          show: false,
          text: `${th.name}  ${Number.isFinite(th.missKm) ? `${th.missKm < 10 ? th.missKm.toFixed(2) : th.missKm.toFixed(0)} km` : ''}`,
          font: LABEL_FONT, fillColor: C.warning, style: Cesium.LabelStyle.FILL, showBackground: true,
          backgroundColor: C.labelBg, backgroundPadding: new Cesium.Cartesian2(5, 3),
          pixelOffset: new Cesium.Cartesian2(14, 0), horizontalOrigin: Cesium.HorizontalOrigin.LEFT,
          verticalOrigin: Cesium.VerticalOrigin.CENTER,
        }),
      });
    }

    this.flash = {
      core: this.flashPoints.add({ show: false, pixelSize: 24, color: C.flashHot }),
      ring1: this.flashPoints.add({ show: false, pixelSize: 10, color: C.clear, outlineColor: C.flashAmber, outlineWidth: 2 }),
      ring2: this.flashPoints.add({ show: false, pixelSize: 10, color: C.clear, outlineColor: C.flashHot, outlineWidth: 1.5 }),
    };

    this.cpEci = this.collisionPoint();
    this.applyColors();
    this.applyLayers();
  }

  /** Collision point: parents' interpolated positions at T0 (mean). */
  collisionPoint() {
    const m = this.model;
    sampleAt(m.tRel, 0, this.S);
    const { i, a } = this.S;
    const j = a > 0 && i < m.N - 1 ? (i + 1) * 3 : -1;
    const acc = [0, 0, 0];
    let n = 0;
    for (const p of m.parents) {
      if (lerpXyz(p.pos, i * 3, j, a, this.tmp)) {
        acc[0] += this.tmp[0]; acc[1] += this.tmp[1]; acc[2] += this.tmp[2]; n += 1;
      }
    }
    if (n === 0) return null;
    return [acc[0] / n, acc[1] / n, acc[2] / n];
  }

  setLayers(layers) {
    this.layers = layers || {};
    this.applyLayers();
  }

  setColorMode(mode) {
    this.colorMode = mode || 'parent';
    this.applyColors();
  }

  on(key) {
    const v = this.layers[key];
    return typeof v === 'boolean' ? v : key !== 'fragmentTrails';
  }

  applyLayers() {
    this.trackLines.show = this.on('parentTracks');
    this.fragPoints.show = this.on('fragments');
    this.trailLines.show = this.on('fragmentTrails');
    this.envLines.show = this.on('debrisEnvelope');
    this.threatLines.show = this.on('threatened');
    this.labels.show = this.on('labels');
    this.dirty = true;
    this.scene.requestRender();
  }

  applyColors() {
    const m = this.model;
    if (!m) return;
    for (let f = 0; f < this.fragItems.length; f += 1) {
      const pt = this.fragItems[f];
      const sizeBin = Number.isFinite(m.fragSize[f]) ? binIndex(m.fragSize[f], SIZE_EDGES_M) : 1;
      let color;
      if (this.colorMode === 'size') color = RAMP[sizeBin];
      else if (this.colorMode === 'dv') color = RAMP[Number.isFinite(m.fragDv[f]) ? binIndex(m.fragDv[f], DV_EDGES_MS) : 1];
      else color = PARENT_FRAG[m.fragParent[f] >= 0 ? Math.min(m.fragParent[f], 1) : 2];
      pt.color = color;
      pt.pixelSize = PX_BY_BIN[sizeBin];
    }
    this.dirty = true;
    this.scene.requestRender();
  }

  // ── per frame ──────────────────────────────────────────────────────────
  toCart(x, y, z, result = new Cesium.Cartesian3()) {
    result.x = ((x * this.cos) + (y * this.sin)) * 1000;
    result.y = ((-x * this.sin) + (y * this.cos)) * 1000;
    result.z = z * 1000;
    return result;
  }

  /** Called from scene.preUpdate. Returns true when a render is needed. */
  update(perf = performance.now()) {
    const m = this.model;
    if (!m) return false;
    const raw = playheadNow(perf);
    const t = Math.min(m.t1, Math.max(m.t0, Number.isFinite(raw) ? raw : m.t0));
    const changed = this.dirty || t !== this.lastT;
    if (this.lastT != null && t !== this.lastT && ((this.lastT < 0) !== (t < 0))) this.flashStart = perf;
    if (changed) {
      this.renderAt(t);
      this.lastT = t;
      this.dirty = false;
    }
    const animating = this.animate(perf, t);
    return changed || animating;
  }

  renderAt(t) {
    const m = this.model;
    const gmst = computeGmst(new Date(m.collisionMs + (t * 1000)));
    this.cos = Math.cos(gmst);
    this.sin = Math.sin(gmst);
    const { i, a } = sampleAt(m.tRel, t, this.S);
    const hasJ = a > 0 && i < m.N - 1;
    const tmp = this.tmp;
    const after = t >= 0;
    const tracksOn = this.on('parentTracks');
    const labelsOn = this.on('labels');

    // Parents: solid past, dashed future, moving point (dimmed after T0).
    for (const it of this.parentItems) {
      const { pos } = it.p;
      const ok = lerpXyz(pos, i * 3, hasJ ? (i + 1) * 3 : -1, a, tmp) && above(tmp[0], tmp[1], tmp[2]);
      const now = ok ? this.toCart(tmp[0], tmp[1], tmp[2]) : null;
      const past = [];
      for (let s = 0; s <= i; s += 1) {
        const o = s * 3;
        if (!Number.isNaN(pos[o]) && above(pos[o], pos[o + 1], pos[o + 2])) past.push(this.toCart(pos[o], pos[o + 1], pos[o + 2]));
      }
      if (now) past.push(now);
      const future = now ? [now] : [];
      for (let s = i + 1; s < m.N; s += 1) {
        const o = s * 3;
        if (!Number.isNaN(pos[o]) && above(pos[o], pos[o + 1], pos[o + 2])) future.push(this.toCart(pos[o], pos[o + 1], pos[o + 2]));
      }
      it.past.positions = past.length > 1 ? past : [];
      it.past.show = past.length > 1;
      it.future.positions = future.length > 1 ? future : [];
      it.future.show = future.length > 1;
      it.point.show = tracksOn && !!now;
      if (now) {
        it.point.position = now;
        it.label.position = now;
      }
      it.point.color = after ? C.parentDim[it.k] : C.parent[it.k];
      it.point.pixelSize = after ? 6 : 9;
      it.label.show = tracksOn && labelsOn && !!now && !after;
    }

    // Fragments (linear between real samples; hidden outside their lifetime).
    const { F, frag } = m;
    const fragOn = this.on('fragments');
    let alive = 0;
    for (let f = 0; f < F; f += 1) {
      const offI = ((i * F) + f) * 3;
      const offJ = hasJ ? ((((i + 1) * F) + f) * 3) : -1;
      const ok = after && lerpXyz(frag, offI, offJ, a, tmp) && above(tmp[0], tmp[1], tmp[2]);
      this.fragAlive[f] = ok ? 1 : 0;
      const pt = this.fragItems[f];
      if (ok) {
        alive += 1;
        this.fragNow[f * 3] = tmp[0]; this.fragNow[(f * 3) + 1] = tmp[1]; this.fragNow[(f * 3) + 2] = tmp[2];
        pt.position = this.toCart(tmp[0], tmp[1], tmp[2], SCR_POS);
      }
      pt.show = ok && fragOn;
    }
    this.stats.alive = alive;

    if (this.on('fragmentTrails')) this.renderTrails(i, hasJ);
    if (this.on('debrisEnvelope')) this.renderEnvelope(i, a, hasJ);
    this.renderThreats(t, i, a, hasJ);
  }

  renderTrails(i) {
    const m = this.model;
    const { F, frag } = m;
    for (const tr of this.trailItems) {
      const { f } = tr;
      if (!this.fragAlive[f]) { tr.line.show = false; continue; }
      const pts = [];
      for (let s = Math.max(0, i - TRAIL_SAMPLES + 1); s <= i; s += 1) {
        const o = ((s * F) + f) * 3;
        if (!Number.isNaN(frag[o])) pts.push(this.toCart(frag[o], frag[o + 1], frag[o + 2]));
      }
      pts.push(this.toCart(this.fragNow[f * 3], this.fragNow[(f * 3) + 1], this.fragNow[(f * 3) + 2]));
      tr.line.show = pts.length > 1;
      tr.line.positions = pts.length > 1 ? pts : [];
    }
  }

  /**
   * Per-parent debris stream: an arc along the parent's orbital plane that
   * spans the angular extent of that parent's live fragments (complement of
   * the largest empty gap), at their mean radius. Grows with the real spread
   * and never cuts through or engulfs the Earth.
   */
  renderEnvelope(i, a, hasJ) {
    const m = this.model;
    for (const env of this.envItems) {
      env.glow.show = false;
      env.core.show = false;
      const parent = m.parents[env.k];
      if (!parent) continue;
      const r = this.tmp;
      const rn = this.tmp2;
      if (!lerpXyz(parent.pos, i * 3, hasJ ? (i + 1) * 3 : -1, a, r)) continue;
      // Orbital plane normal from two consecutive parent samples (r_s × r_s+1).
      const s0 = Math.min(i, m.N - 2);
      const p0 = [0, 0, 0];
      if (!lerpXyz(parent.pos, s0 * 3, -1, 0, p0) || !lerpXyz(parent.pos, (s0 + 1) * 3, -1, 0, rn)) continue;
      let hx = (p0[1] * rn[2]) - (p0[2] * rn[1]);
      let hy = (p0[2] * rn[0]) - (p0[0] * rn[2]);
      let hz = (p0[0] * rn[1]) - (p0[1] * rn[0]);
      const hn = Math.hypot(hx, hy, hz);
      if (!(hn > 0)) continue;
      hx /= hn; hy /= hn; hz /= hn;
      const rr = Math.hypot(r[0], r[1], r[2]);
      const e1 = [r[0] / rr, r[1] / rr, r[2] / rr];
      const e2 = [(hy * e1[2]) - (hz * e1[1]), (hz * e1[0]) - (hx * e1[2]), (hx * e1[1]) - (hy * e1[0])];

      const angles = [];
      let rSum = 0;
      for (let f = 0; f < m.F; f += 1) {
        if (!this.fragAlive[f] || Math.min(m.fragParent[f], 1) !== env.k) continue;
        const x = this.fragNow[f * 3]; const y = this.fragNow[(f * 3) + 1]; const z = this.fragNow[(f * 3) + 2];
        angles.push(Math.atan2((x * e2[0]) + (y * e2[1]) + (z * e2[2]), (x * e1[0]) + (y * e1[1]) + (z * e1[2])));
        rSum += Math.hypot(x, y, z);
      }
      if (angles.length < 3) continue;
      angles.sort((u, v) => u - v);
      let gap = (angles[0] + (2 * Math.PI)) - angles[angles.length - 1];
      let start = angles[0];
      for (let n = 1; n < angles.length; n += 1) {
        const g = angles[n] - angles[n - 1];
        if (g > gap) { gap = g; start = angles[n]; }
      }
      const span = Math.max((2 * Math.PI) - gap, 0.004);
      const radius = rSum / angles.length;
      const steps = Math.max(2, Math.ceil(span / ENV_MAX_STEP_RAD));
      const pts = [];
      for (let n = 0; n <= steps; n += 1) {
        const th = start + (span * n / steps);
        const c = Math.cos(th) * radius; const s = Math.sin(th) * radius;
        pts.push(this.toCart((c * e1[0]) + (s * e2[0]), (c * e1[1]) + (s * e2[1]), (c * e1[2]) + (s * e2[2])));
      }
      env.glow.positions = pts;
      env.core.positions = pts;
      env.glow.show = true;
      env.core.show = true;
    }
  }

  renderThreats(t, i, a, hasJ) {
    const on = this.on('threatened');
    const labelsOn = this.on('labels');
    const pos = this.tmp;
    let rings = false;
    for (const it of this.threatItems) {
      const { th } = it;
      const ok = on && th.track && lerpXyz(th.track, i * 3, hasJ ? (i + 1) * 3 : -1, a, pos) && above(pos[0], pos[1], pos[2]);
      if (!ok) {
        it.dot.show = false; it.ring.show = false; it.label.show = false; it.line.show = false;
        continue;
      }
      const c = this.toCart(pos[0], pos[1], pos[2]);
      it.dot.position = c;
      it.ring.position = c;
      it.label.position = c;
      it.dot.show = true;
      const inWindow = Math.abs(t - th.tcaRel) <= THREAT_WINDOW_S;
      it.ring.show = inWindow;
      it.label.show = inWindow && labelsOn;
      rings = rings || inWindow;
      const f = th.fragIndex;
      if (inWindow && f >= 0 && this.fragAlive[f]) {
        const fp = [this.fragNow[f * 3], this.fragNow[(f * 3) + 1], this.fragNow[(f * 3) + 2]];
        const sp = [pos[0], pos[1], pos[2]];
        const sepKm = Math.hypot(fp[0] - sp[0], fp[1] - sp[1], fp[2] - sp[2]);
        if (sepKm <= THREAT_LINK_KM && segmentClearsEarth(sp, fp)) {
          it.line.positions = [c, this.toCart(fp[0], fp[1], fp[2])];
          it.line.show = true;
        } else it.line.show = false;
      } else it.line.show = false;
    }
    this.ringsVisible = rings;
  }

  /** Flash + shockwave (wall-clock) and threat-ring pulse (while playing). */
  animate(perf) {
    let busy = false;
    const e = (perf - this.flashStart) / FLASH_MS;
    if (this.flash && this.cpEci && e >= 0 && e < 1) {
      const center = this.toCart(this.cpEci[0], this.cpEci[1], this.cpEci[2]);
      const { core, ring1, ring2 } = this.flash;
      const hot = Math.max(0, 1 - (e / 0.18));
      core.show = true;
      core.position = center;
      core.pixelSize = 10 + (26 * Math.sqrt(Math.max(0, 1 - e)));
      core.color = Cesium.Color.lerp(C.flashAmber, C.flashHot, hot, SCR_A).withAlpha((1 - e) ** 1.6, SCR_B);
      ring1.show = true;
      ring1.position = center;
      ring1.pixelSize = 10 + (230 * easeOut(e));
      ring1.outlineColor = C.flashAmber.withAlpha(0.9 * (1 - e), SCR_A);
      const e2 = Math.max(0, (e - 0.22) / 0.78);
      ring2.show = e2 > 0;
      ring2.position = center;
      ring2.pixelSize = 10 + (160 * easeOut(e2));
      ring2.outlineColor = C.flashHot.withAlpha(0.7 * (1 - e2), SCR_A);
      const rKm = 15 + (SHOCK_MAX_KM * easeOut(e));
      this.shock.modelMatrix = Cesium.Matrix4.multiplyByUniformScale(
        Cesium.Matrix4.fromTranslation(center, this.shockMatrix), rKm * 1000, this.shockMatrix,
      );
      this.shockMaterial.uniforms.color = Cesium.Color.lerp(C.flashHot, C.flashAmber, Math.min(1, e * 2), new Cesium.Color()).withAlpha(0.32 * ((1 - e) ** 1.5));
      this.shock.show = true;
      this.flashVisible = true;
      busy = true;
    } else if (this.flashVisible) {
      const { core, ring1, ring2 } = this.flash || {};
      if (core) { core.show = false; ring1.show = false; ring2.show = false; }
      this.shock.show = false;
      this.flashVisible = false;
      busy = true;
    }

    if (this.ringsVisible) {
      const playing = clockPlaying();
      const k = playing ? 0.5 + (0.5 * Math.sin(perf / 220)) : 0.6;
      for (const it of this.threatItems) {
        if (!it.ring.show) continue;
        it.ring.pixelSize = 16 + (12 * k);
        it.ring.outlineColor = C.warning.withAlpha(0.45 + (0.5 * (1 - k)), SCR_A);
      }
      busy = busy || playing;
    }
    return busy;
  }

  /**
   * Camera: frame the collision point from a raking angle with the limb
   * behind. Target = event collision_point_eci_km rotated at the collision
   * instant (same transform as the T0 frame); fallback = the parents' mean
   * position at T0. Every value is validated — an invalid target is a no-op.
   */
  flyTo(camera, target) {
    const valid = (v) => Array.isArray(v) && v.length >= 3 && v.slice(0, 3).every(Number.isFinite)
      && Math.hypot(v[0], v[1], v[2]) > RE_KM && Math.hypot(v[0], v[1], v[2]) < 100000;
    const m = this.model;
    // After the impact the stream has moved downrange: frame the live
    // fragment nearest the stream mean (never the mean itself, which can sit
    // inside the Earth) in the current playhead frame, from further out.
    if (m && this.lastT != null && this.lastT > 300 && this.stats.alive > 0) {
      let sx = 0; let sy = 0; let sz = 0; let n = 0;
      for (let f = 0; f < m.F; f += 1) {
        if (!this.fragAlive[f]) continue;
        sx += this.fragNow[f * 3]; sy += this.fragNow[(f * 3) + 1]; sz += this.fragNow[(f * 3) + 2]; n += 1;
      }
      sx /= n; sy /= n; sz /= n;
      let best = -1; let bestD = Infinity;
      for (let f = 0; f < m.F; f += 1) {
        if (!this.fragAlive[f]) continue;
        const d = ((this.fragNow[f * 3] - sx) ** 2) + ((this.fragNow[(f * 3) + 1] - sy) ** 2) + ((this.fragNow[(f * 3) + 2] - sz) ** 2);
        if (d < bestD) { bestD = d; best = f; }
      }
      if (best >= 0) {
        const c = this.toCart(this.fragNow[best * 3], this.fragNow[(best * 3) + 1], this.fragNow[(best * 3) + 2]);
        if ([c.x, c.y, c.z].every(Number.isFinite)) {
          camera.cancelFlight();
          camera.flyToBoundingSphere(new Cesium.BoundingSphere(c, 2000e3), {
            offset: new Cesium.HeadingPitchRange(Cesium.Math.toRadians(20), Cesium.Math.toRadians(-55), 1.3e7),
            duration: 2.2,
          });
          return true;
        }
      }
    }
    const eciT = Array.isArray(target?.eci) ? target.eci.slice(0, 3).map(Number) : null;
    let eci = null;
    let ms = Date.parse(target?.utc || '');
    if (valid(eciT) && Number.isFinite(ms)) eci = eciT;
    else if (valid(this.cpEci) && m) { eci = this.cpEci; ms = m.collisionMs; }
    if (!eci) return false;
    const g = computeGmst(new Date(ms));
    const cos = Math.cos(g); const sin = Math.sin(g);
    const center = new Cesium.Cartesian3(
      ((eci[0] * cos) + (eci[1] * sin)) * 1000,
      ((-eci[0] * sin) + (eci[1] * cos)) * 1000,
      eci[2] * 1000,
    );
    if (![center.x, center.y, center.z].every(Number.isFinite)) return false;
    camera.cancelFlight();
    camera.flyToBoundingSphere(new Cesium.BoundingSphere(center, 800e3), {
      offset: new Cesium.HeadingPitchRange(Cesium.Math.toRadians(35), Cesium.Math.toRadians(-28), 6.5e6),
      duration: 2.2,
    });
    return true;
  }

  destroy() {
    const P = this.scene.primitives;
    for (const p of [this.envLines, this.trailLines, this.trackLines, this.threatLines, this.fragPoints, this.markPoints, this.flashPoints, this.labels, this.shock]) {
      if (p && !p.isDestroyed()) P.remove(p);
    }
    this.model = null;
  }
}
