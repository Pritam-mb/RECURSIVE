/**
 * debrisCloud.js — read-only accessors for backend debris clouds.
 * Every value comes from the cloud payload (derived from real fragment states);
 * a missing field yields null, never a stand-in radius or count.
 */

const finite = (v) => {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

function toXyz(v) {
  if (!v) return null;
  if (Array.isArray(v)) {
    const [x, y, z] = v.map(finite);
    return x != null && y != null && z != null ? { x, y, z } : null;
  }
  const x = finite(v.x); const y = finite(v.y); const z = finite(v.z);
  return x != null && y != null && z != null ? { x, y, z } : null;
}

/** Fragment centroid (ECI km) at the cloud epoch. */
export function cloudCentroid(cloud) {
  return toXyz(cloud?.centroid_eci_km) ?? toXyz(cloud?.center_eci_km) ?? toXyz(cloud?.centroid_km);
}

/** Epoch the cloud geometry refers to (ISO), or null. */
export function cloudEpoch(cloud) {
  return cloud?.epoch_utc ?? cloud?.tca_utc ?? null;
}

/**
 * Percentile radius of the fragment spread: { km, percentile } or null.
 * percentile is null when the backend did not say which percentile it is.
 */
export function cloudRadius(cloud) {
  if (!cloud) return null;
  const pct = finite(cloud.radius_percentile ?? cloud.percentile);
  const candidates = [
    [cloud.radius_p90_km, 90],
    [cloud.r90_km, 90],
    [cloud.radius_p50_km, 50],
    [cloud.percentile_radius_km, pct],
    [cloud.radius_km, pct],
    [cloud.radius_km_now, pct],
  ];
  for (const [value, p] of candidates) {
    const km = finite(value);
    if (km != null && km > 0) return { km, percentile: p ?? null };
  }
  return null;
}

/** Real fragment count (null when absent). */
export function cloudFragmentCount(cloud) {
  return finite(cloud?.fragment_count);
}

/** Event id linking a cloud to debris alerts' parent_event.event_id. */
export function cloudEventId(cloud) {
  const pe = cloud?.parent_event;
  if (pe && typeof pe === 'object' && pe.event_id != null) return String(pe.event_id);
  if (pe != null && typeof pe !== 'object') return String(pe);
  if (cloud?.event_id != null) return String(cloud.event_id);
  return cloud?.id != null ? String(cloud.id) : null;
}

/** Downsampled fragment positions [{x,y,z}] (ECI km), capped at `max`. */
export function cloudFragments(cloud, max = 300) {
  const raw = Array.isArray(cloud?.fragments) ? cloud.fragments : [];
  const out = [];
  const step = raw.length > max ? raw.length / max : 1;
  for (let i = 0; i < raw.length && out.length < max; i += step) {
    const p = toXyz(raw[Math.floor(i)]);
    if (p) out.push(p);
  }
  return out;
}

/** "DEBRIS 912 FRAG · R90 34 KM" — omits any part the backend did not provide. */
export function cloudLabel(cloud) {
  const count = cloudFragmentCount(cloud);
  const radius = cloudRadius(cloud);
  const parts = [`DEBRIS ${count == null ? '—' : count} FRAG`];
  if (radius) {
    const rTag = radius.percentile != null ? `R${radius.percentile}` : 'R';
    parts.push(`${rTag} ${radius.km < 10 ? radius.km.toFixed(1) : radius.km.toFixed(0)} KM`);
  }
  return parts.join(' · ');
}
