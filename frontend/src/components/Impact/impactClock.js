/**
 * Smooth impact-replay playhead shared by the renderers.
 *
 * The store (impact.tRelS / playing / speed) is the source of truth, but while
 * playing it is only published at <= 10 Hz so React does not re-render every
 * frame. Renderers read playheadNow() each frame instead: it extrapolates the
 * last anchor at `speed` sim-seconds per wall-second. Any external write to
 * impact.tRelS (seek / scrub) re-anchors the clock.
 */
import useStore from '../../store/useStore';

const clock = {
  anchorT: 0,
  anchorPerf: 0,
  playing: false,
  speed: 60,
  lastPushed: null,
};

function current(perf) {
  if (!clock.playing) return clock.anchorT;
  return clock.anchorT + (((perf - clock.anchorPerf) / 1000) * clock.speed);
}

export function playheadNow(perf = performance.now()) {
  return current(perf);
}

export function clockPlaying() {
  return clock.playing;
}

/** Publish the smooth playhead to the store (called by the 10 Hz ticker). */
export function pushPlayhead(t) {
  clock.lastPushed = t;
  useStore.getState().setImpact({ tRelS: t });
}

{
  const init = useStore.getState().impact;
  if (init) {
    clock.anchorT = Number(init.tRelS) || 0;
    clock.playing = !!init.playing;
    clock.speed = Number(init.speed) || 60;
  }
  clock.anchorPerf = performance.now();
}

const unsubscribe = useStore.subscribe((state, prev) => {
  const a = state.impact;
  const b = prev.impact;
  if (!a || a === b) return;
  const now = performance.now();
  const seek = a.tRelS !== b?.tRelS && a.tRelS !== clock.lastPushed;
  clock.anchorT = seek ? Number(a.tRelS) || 0 : current(now);
  clock.anchorPerf = now;
  clock.playing = !!a.playing;
  clock.speed = Number(a.speed) || 1;
});

// ── Camera "fly to impact" requests (console → globe) ──────────────────────
const flyListeners = new Set();

export function onFlyTo(cb) {
  flyListeners.add(cb);
  return () => flyListeners.delete(cb);
}

/** target: { eci: [x,y,z] km, utc: ISO } — the collision point and instant. */
export function requestFlyTo(target) {
  for (const cb of flyListeners) cb(target);
}

if (import.meta.hot) import.meta.hot.dispose(() => unsubscribe());
