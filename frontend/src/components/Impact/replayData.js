/**
 * Impact replay payload (CONTRACT3, GET /api/debris/events/{id}/replay)
 * → flat typed arrays the globe can interpolate every frame without
 * allocating. Positions stay in ECI km; the renderer rotates by GMST.
 *
 * Interpolation is LINEAR between the backend's propagated samples (default
 * 30 s apart: chord sag < 1 km in LEO). Nothing is extrapolated: a fragment
 * is drawn only while both bracketing samples exist.
 */

// Shared with ThreatGlobe's legend (same bins / colours).
export const FRAG_PARENT_COLORS = ['#f0b429', '#ff8a3d', '#c48d1f'];
export const FRAG_RAMP = ['#8a6418', '#c48d1f', '#f0b429', '#ffe3a0'];
export const SIZE_EDGES_M = [0.1, 0.3, 1.0];
export const DV_EDGES_MS = [50, 150, 300];

export function binIndex(value, edges) {
  let i = 0;
  while (i < edges.length && value >= edges[i]) i += 1;
  return i;
}

const MU = 398600.4418;
const J2 = 1.08262668e-3;
const RE = 6378.137;

function toXyz(p) {
  if (!p) return null;
  if (Array.isArray(p)) return p.length >= 3 && p.every((v) => v != null && Number.isFinite(Number(v))) ? [Number(p[0]), Number(p[1]), Number(p[2])] : null;
  if (Number.isFinite(p.x)) return [p.x, p.y, p.z];
  return null;
}

function accel(x, y, z, out) {
  const r2 = (x * x) + (y * y) + (z * z);
  const r = Math.sqrt(r2);
  const r3 = r2 * r;
  const k = 1.5 * J2 * MU * RE * RE / (r2 * r2 * r);
  const zz = 5 * z * z / r2;
  out[0] = (-MU * x / r3) + (k * x * (zz - 1));
  out[1] = (-MU * y / r3) + (k * y * (zz - 1));
  out[2] = (-MU * z / r3) + (k * z * (zz - 3));
}

/** RK4 two-body + J2 step (km, km/s, s). Mutates s = [x,y,z,vx,vy,vz]. */
const A1 = [0, 0, 0]; const A2 = [0, 0, 0]; const A3 = [0, 0, 0]; const A4 = [0, 0, 0];
function rk4(s, h) {
  const [x, y, z, vx, vy, vz] = s;
  accel(x, y, z, A1);
  const x2 = x + (vx * h / 2); const y2 = y + (vy * h / 2); const z2 = z + (vz * h / 2);
  const vx2 = vx + (A1[0] * h / 2); const vy2 = vy + (A1[1] * h / 2); const vz2 = vz + (A1[2] * h / 2);
  accel(x2, y2, z2, A2);
  const x3 = x + (vx2 * h / 2); const y3 = y + (vy2 * h / 2); const z3 = z + (vz2 * h / 2);
  const vx3 = vx + (A2[0] * h / 2); const vy3 = vy + (A2[1] * h / 2); const vz3 = vz + (A2[2] * h / 2);
  accel(x3, y3, z3, A3);
  const x4 = x + (vx3 * h); const y4 = y + (vy3 * h); const z4 = z + (vz3 * h);
  const vx4 = vx + (A3[0] * h); const vy4 = vy + (A3[1] * h); const vz4 = vz + (A3[2] * h);
  accel(x4, y4, z4, A4);
  s[0] = x + (h / 6) * (vx + (2 * vx2) + (2 * vx3) + vx4);
  s[1] = y + (h / 6) * (vy + (2 * vy2) + (2 * vy3) + vy4);
  s[2] = z + (h / 6) * (vz + (2 * vz2) + (2 * vz3) + vz4);
  s[3] = vx + (h / 6) * (A1[0] + (2 * A2[0]) + (2 * A3[0]) + A4[0]);
  s[4] = vy + (h / 6) * (A1[1] + (2 * A2[1]) + (2 * A3[1]) + A4[1]);
  s[5] = vz + (h / 6) * (A1[2] + (2 * A2[2]) + (2 * A3[2]) + A4[2]);
}

const MAX_STEP_S = 20;

/**
 * Propagate a live state vector (two-body + J2, RK4 ≤ 20 s steps) to every
 * sample instant. Used for threatened satellites, whose tracks the replay
 * payload does not carry. Returns Float64Array(N*3) ECI km.
 */
export function propagateToSamples(sat, sampleMs) {
  const r = toXyz(sat?.position);
  const v = sat?.velocity ? [sat.velocity.vx, sat.velocity.vy, sat.velocity.vz].map(Number) : null;
  const epochMs = Date.parse(sat?.epoch_utc || '');
  if (!r || !v || v.some((c) => !Number.isFinite(c)) || !Number.isFinite(epochMs)) return null;
  const n = sampleMs.length;
  const out = new Float64Array(n * 3).fill(Number.NaN);
  const order = Array.from({ length: n }, (_, i) => i);
  const run = (indices, sign) => {
    const s = [...r, ...v];
    let tMs = epochMs;
    for (const i of indices) {
      let remaining = (sampleMs[i] - tMs) / 1000 * sign;
      while (remaining > 1e-6) {
        const h = Math.min(MAX_STEP_S, remaining);
        rk4(s, h * sign);
        remaining -= h;
      }
      tMs = sampleMs[i];
      out[i * 3] = s[0]; out[(i * 3) + 1] = s[1]; out[(i * 3) + 2] = s[2];
    }
  };
  run(order.filter((i) => sampleMs[i] >= epochMs).sort((a, b) => sampleMs[a] - sampleMs[b]), 1);
  run(order.filter((i) => sampleMs[i] < epochMs).sort((a, b) => sampleMs[b] - sampleMs[a]), -1);
  return out;
}

/**
 * Parse the replay payload. `satellites` (live store) supplies state vectors
 * for threatened satellites. Returns null for an unusable payload.
 */
export function parseReplay(replay, satellites = []) {
  const tRelRaw = replay?.t_rel_s;
  const collisionMs = Date.parse(replay?.collision_utc || '');
  if (!Array.isArray(tRelRaw) || tRelRaw.length < 2 || !Number.isFinite(collisionMs)) return null;
  const N = tRelRaw.length;
  const tRel = Float64Array.from(tRelRaw, Number);
  const sampleMs = Array.from(tRel, (t) => collisionMs + (t * 1000));

  const parentIds = Object.keys(replay.parents || {});
  const parents = parentIds.map((id) => {
    const p = replay.parents[id] || {};
    const pos = new Float64Array(N * 3).fill(Number.NaN);
    (p.positions || []).slice(0, N).forEach((xyz, i) => {
      const c = toXyz(xyz);
      if (c) { pos[i * 3] = c[0]; pos[(i * 3) + 1] = c[1]; pos[(i * 3) + 2] = c[2]; }
    });
    return { id, name: p.name || `#${id}`, agency: p.agency || '', pos };
  });

  const fr = replay.fragments || {};
  const ids = Array.isArray(fr.ids) ? fr.ids : [];
  const F = Math.min(ids.length, 400);
  const frag = new Float32Array(N * F * 3).fill(Number.NaN);
  const posBySample = Array.isArray(fr.positions) ? fr.positions : [];
  for (let s = 0; s < N; s += 1) {
    const row = posBySample[s];
    if (!Array.isArray(row)) continue;
    for (let f = 0; f < F; f += 1) {
      const c = toXyz(row[f]);
      if (!c) continue;
      const k = ((s * F) + f) * 3;
      frag[k] = c[0]; frag[k + 1] = c[1]; frag[k + 2] = c[2];
    }
  }
  const num = (arr, i) => (Array.isArray(arr) && Number.isFinite(Number(arr[i])) ? Number(arr[i]) : null);
  const fragSize = new Float32Array(F);
  const fragDv = new Float32Array(F);
  const fragParent = new Int8Array(F).fill(-1);
  const fragIndexById = new Map();
  for (let f = 0; f < F; f += 1) {
    fragSize[f] = num(fr.size_m, f) ?? Number.NaN;
    fragDv[f] = num(fr.dv_ms, f) ?? Number.NaN;
    const parentOf = Array.isArray(fr.parent_of) ? fr.parent_of[f] : null;
    fragParent[f] = parentOf == null ? -1 : parentIds.indexOf(String(parentOf));
    fragIndexById.set(String(ids[f]), f);
  }

  const satById = new Map((satellites || []).map((s) => [String(s.norad_id), s]));
  const trackCache = new Map();
  const threatened = (Array.isArray(replay.threatened) ? replay.threatened : [])
    .map((t) => {
      const tcaMs = Date.parse(t?.tca_utc || '');
      if (t?.sat_id == null || !Number.isFinite(tcaMs)) return null;
      const key = String(t.sat_id);
      if (!trackCache.has(key)) {
        let track = null;
        if (Array.isArray(t.positions) && t.positions.length === N) {
          track = new Float64Array(N * 3).fill(Number.NaN);
          t.positions.forEach((xyz, i) => {
            const c = toXyz(xyz);
            if (c) { track[i * 3] = c[0]; track[(i * 3) + 1] = c[1]; track[(i * 3) + 2] = c[2]; }
          });
        } else if (satById.has(key)) {
          track = propagateToSamples(satById.get(key), sampleMs);
        }
        trackCache.set(key, track);
      }
      return {
        satId: key,
        name: t.name || `#${key}`,
        tcaRel: (tcaMs - collisionMs) / 1000,
        missKm: Number(t.miss_km),
        pc: Number(t.pc),
        fragIndex: fragIndexById.get(String(t.fragment_id))
          ?? (Number.isInteger(t.fragment_display_index) && t.fragment_display_index < F ? t.fragment_display_index : -1),
        track: trackCache.get(key),
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.tcaRel - b.tcaRel)
    .slice(0, 12);

  return {
    eventId: replay.event_id,
    collisionMs,
    N,
    tRel,
    t0: tRel[0],
    t1: tRel[N - 1],
    parents,
    F,
    frag,
    fragSize,
    fragDv,
    fragParent,
    threatened,
    envelope: Array.isArray(replay.envelope) ? replay.envelope : [],
  };
}

/** Bracketing sample index i and weight a ∈ [0,1) for time t (clamped). */
export function sampleAt(tRel, t, out) {
  const n = tRel.length;
  if (t <= tRel[0]) { out.i = 0; out.a = 0; return out; }
  if (t >= tRel[n - 1]) { out.i = n - 1; out.a = 0; return out; }
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (tRel[mid] <= t) lo = mid; else hi = mid;
  }
  out.i = lo;
  out.a = (t - tRel[lo]) / (tRel[hi] - tRel[lo]);
  return out;
}

/**
 * Linear interpolation of a strided xyz buffer between samples i and i+1.
 * `base(i)` gives the element offset of sample i. Returns false (out untouched)
 * when a needed sample is missing.
 */
export function lerpXyz(buf, offI, offJ, a, out) {
  const x0 = buf[offI];
  if (Number.isNaN(x0)) return false;
  if (a <= 0 || offJ < 0) {
    out[0] = x0; out[1] = buf[offI + 1]; out[2] = buf[offI + 2];
    return true;
  }
  const x1 = buf[offJ];
  if (Number.isNaN(x1)) return false;
  out[0] = x0 + ((x1 - x0) * a);
  out[1] = buf[offI + 1] + ((buf[offJ + 1] - buf[offI + 1]) * a);
  out[2] = buf[offI + 2] + ((buf[offJ + 2] - buf[offI + 2]) * a);
  return true;
}

/** Envelope stats at t, linearly interpolated (numbers only; nulls kept). */
export function envelopeAt(envelope, t) {
  if (!envelope?.length) return null;
  let j = 0;
  while (j < envelope.length - 1 && Number(envelope[j + 1].t_rel_s) <= t) j += 1;
  const e0 = envelope[j];
  const e1 = envelope[Math.min(j + 1, envelope.length - 1)];
  const span = Number(e1.t_rel_s) - Number(e0.t_rel_s);
  const a = span > 0 ? Math.min(1, Math.max(0, (t - Number(e0.t_rel_s)) / span)) : 0;
  const mix = (k) => {
    const v0 = e0[k];
    const v1 = e1[k];
    if (v0 == null) return v1 == null || a < 0.5 ? null : Number(v1);
    if (v1 == null) return Number(v0);
    return Number(v0) + ((Number(v1) - Number(v0)) * a);
  };
  return {
    n_alive: e0.n_alive == null && e1.n_alive == null ? null : Math.round(mix('n_alive') ?? 0),
    p50_km: mix('p50_km'),
    p90_km: mix('p90_km'),
    along_track_spread_km: mix('along_track_spread_km'),
  };
}
