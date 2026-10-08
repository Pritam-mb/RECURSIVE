// Shared smooth-motion helpers for the globe renderers.
//
// Satellites arrive in ~1 Hz snapshots. Renderers dead-reckon each one from
// its latest snapshot at constant velocity and fold any mismatch with the
// previously drawn position into an offset that decays, so motion is
// continuous (no easing stalls, no snaps).
//
// The per-frame helpers write into caller-owned objects so the hot loops
// (hundreds of satellites, every frame) allocate nothing.

// How fast a snapshot correction fades out.
export const CORRECTION_DECAY_S = 0.35;
// Gaps larger than this are a time jump / reset, so snap instead of gliding.
export const MAX_SMOOTHED_GAP_KM = 300;
// Stop extrapolating if the stream stalls for longer than this.
export const MAX_EXTRAPOLATION_S = 5;

/**
 * Sim-seconds per wall-second, from consecutive snapshot timestamps, so
 * extrapolation keeps pace when the simulation clock is accelerated.
 */
export function estimateSimRate(prevSimMs, prevWallMs, simMs, wallMs, current = 1) {
  if (prevSimMs == null || !prevWallMs) return current;
  const elapsedWallMs = wallMs - prevWallMs;
  if (elapsedWallMs < 100) return current;
  const sample = (simMs - prevSimMs) / elapsedWallMs;
  if (!Number.isFinite(sample) || sample <= 0 || sample > 100000) return current;
  // Follow genuine speed changes immediately; smooth network jitter otherwise.
  if (sample > current * 2 || sample < current / 2) return sample;
  return current + ((sample - current) * 0.2);
}

/**
 * Offset from a fresh snapshot position to where the object is drawn now,
 * or null when there is nothing to blend (first sighting or a jump).
 * Writes into `out` when given, so per-satellite objects can be reused.
 */
export function correctionFor(rendered, base, out) {
  if (!rendered || !base) return null;
  const dx = rendered.x - base.x;
  const dy = rendered.y - base.y;
  const dz = rendered.z - base.z;
  if (!Number.isFinite(dx) || !Number.isFinite(dy) || !Number.isFinite(dz)) return null;
  const gap = Math.sqrt((dx * dx) + (dy * dy) + (dz * dz));
  if (gap >= MAX_SMOOTHED_GAP_KM) return null;
  const target = out || {};
  target.x = dx;
  target.y = dy;
  target.z = dz;
  return target;
}

/**
 * Position `simSeconds` past `base` plus the decayed correction, written into
 * `out` (which is also returned). Velocity is {vx, vy, vz} in km/s or null.
 */
export function extrapolate(base, velocity, simSeconds, correction, correctionWeight, out) {
  const target = out || { x: 0, y: 0, z: 0 };
  let x = base.x || 0;
  let y = base.y || 0;
  let z = base.z || 0;
  if (velocity) {
    x += (velocity.vx || 0) * simSeconds;
    y += (velocity.vy || 0) * simSeconds;
    z += (velocity.vz || 0) * simSeconds;
  }
  if (correction && correctionWeight > 0.001) {
    x += correction.x * correctionWeight;
    y += correction.y * correctionWeight;
    z += correction.z * correctionWeight;
  }
  target.x = x;
  target.y = y;
  target.z = z;
  return target;
}

/**
 * Calls `onChange(active)` whenever the element becomes both on-screen and in
 * a visible tab, or stops being either. Returns a cleanup function.
 */
export function observeActivity(element, onChange) {
  let onScreen = true;
  let tabVisible = typeof document === 'undefined' || document.visibilityState !== 'hidden';
  let last = null;
  const emit = () => {
    const active = onScreen && tabVisible;
    if (active !== last) {
      last = active;
      onChange(active);
    }
  };
  const onVisibility = () => {
    tabVisible = document.visibilityState !== 'hidden';
    emit();
  };
  document.addEventListener('visibilitychange', onVisibility);

  let io = null;
  if (element && typeof IntersectionObserver !== 'undefined') {
    io = new IntersectionObserver((entries) => {
      const entry = entries[entries.length - 1];
      onScreen = entry.isIntersecting;
      emit();
    });
    io.observe(element);
  }
  emit();

  return () => {
    document.removeEventListener('visibilitychange', onVisibility);
    if (io) io.disconnect();
  };
}
