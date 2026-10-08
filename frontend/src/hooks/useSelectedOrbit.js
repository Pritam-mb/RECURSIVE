import { useEffect } from 'react';
import useStore from '../store/useStore';
import { generateKeplerianOrbit } from '../utils/coords';

const MU_KM3_S2 = 398600.4418;
const SAMPLES_PER_ORBIT = 180;
const REFRESH_MS = 10000;

/** One orbital period in minutes from the state vector (vis-viva), capped at 24 h. */
function periodMinutes(sat) {
  const p = sat.position;
  const v = sat.velocity;
  const r = Math.hypot(p.x, p.y, p.z);
  const speed = Math.hypot(v.vx, v.vy, v.vz);
  const a = 1 / ((2 / r) - ((speed * speed) / MU_KM3_S2));
  if (!(a > 0)) return 120; // escape trajectory: just show the next two hours
  const minutes = (2 * Math.PI * Math.sqrt((a ** 3) / MU_KM3_S2)) / 60;
  return Math.min(Math.max(minutes, 60), 1440);
}

/**
 * Keeps `selectedOrbit` in the store filled for the selected satellite, so the
 * globes can draw its orbit path. Samples one full revolution from the latest
 * snapshot and refreshes periodically (not every 1 Hz frame).
 */
export default function useSelectedOrbit() {
  const selectedId = useStore((s) => s.selectedSatelliteId);
  const setSelectedOrbit = useStore((s) => s.setSelectedOrbit);

  useEffect(() => {
    if (selectedId == null) {
      setSelectedOrbit([]);
      return undefined;
    }

    const compute = () => {
      const { satellites, snapshotTimestamp } = useStore.getState();
      const sat = satellites.find((s) => String(s.norad_id) === String(selectedId));
      if (!sat?.position || !sat?.velocity) return false;
      const span = periodMinutes(sat);
      const step = Math.max(10, Math.round((span * 60) / SAMPLES_PER_ORBIT));
      const withEpoch = sat.epoch_utc ? sat : { ...sat, epoch_utc: snapshotTimestamp };
      setSelectedOrbit(generateKeplerianOrbit(withEpoch, span, step));
      return true;
    };

    // The satellite may not be in the current snapshot yet; retry quickly
    // until it is, then refresh slowly.
    let timer = null;
    const scheduleRefresh = () => { timer = setInterval(compute, REFRESH_MS); };
    if (compute()) {
      scheduleRefresh();
    } else {
      timer = setInterval(() => {
        if (!compute()) return;
        clearInterval(timer);
        scheduleRefresh();
      }, 500);
    }

    return () => clearInterval(timer);
  }, [selectedId, setSelectedOrbit]);
}
