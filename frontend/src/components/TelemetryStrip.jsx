import { useEffect, useState, useRef, useMemo } from 'react';
import useStore from '../store/useStore';
import { apiGet } from '../utils/api';
import { severityLabel } from '../utils/severity';
import '../styles/telemetry.css';

const MU_KM3_S2 = 398600.4418;
const SEV_RANK = { CRITICAL: 3, WARNING: 2, WATCH: 1 };
const NON_PAYLOAD = new Set(['DEB', 'R/B', 'UNK']);

function useFlash(value) {
  const [flashing, setFlashing] = useState(false);
  const prev = useRef(value);
  useEffect(() => {
    if (prev.current !== value) {
      setFlashing(true);
      const t = setTimeout(() => setFlashing(false), 500);
      prev.current = value;
      return () => clearTimeout(t);
    }
  }, [value]);
  return flashing;
}

function TelemetryValue({ value, unit = '' }) {
  const flash = useFlash(value);
  return (
    <span className={`tm-num${flash ? ' is-changed' : ''}`}>
      {value ?? '—'}
      {unit && value != null && <span className="tm-unit">{unit}</span>}
    </span>
  );
}

function Cell({ label, wide = false, tag = null, title, children }) {
  return (
    <div className={`tm-cell${wide ? ' tm-cell--wide' : ''}`} title={title}>
      <div className="tm-cell-label">
        {label}
        {tag && <span className="tm-sim-tag">{tag}</span>}
      </div>
      <div className="tm-cell-value">{children}</div>
    </div>
  );
}

const stateClass = (status) =>
  status === 'CRITICAL' ? 'is-warning'
    : status === 'WARNING' || status === 'WATCH' ? 'is-caution'
      : 'is-nominal';

const num = (v) => {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};
const fixed = (v, d) => (num(v) == null ? null : num(v).toFixed(d));

/** Osculating period from the propagated state (vis-viva), minutes. */
function periodFromState(pos, vel) {
  const x = num(pos.x); const y = num(pos.y); const z = num(pos.z);
  const vx = num(vel.vx); const vy = num(vel.vy); const vz = num(vel.vz);
  if ([x, y, z, vx, vy, vz].some((c) => c == null)) return null;
  const r = Math.hypot(x, y, z);
  if (r <= 0) return null;
  const v2 = (vx * vx) + (vy * vy) + (vz * vz);
  const invA = (2 / r) - (v2 / MU_KM3_S2);
  if (invA <= 0) return null; // unbound trajectory
  const a = 1 / invA;
  return (2 * Math.PI * Math.sqrt((a ** 3) / MU_KM3_S2)) / 60;
}

export default function TelemetryStrip({ selectedSatId }) {
  // Select only the chosen satellite rather than the full list.
  const sat = useStore((s) => (
    selectedSatId != null
      ? s.satellites.find((x) => x.norad_id === selectedSatId) ?? null
      : null
  ));
  const alerts = useStore((s) => s.alerts);
  const [telemetry, setTelemetry] = useState(null);

  // Poll the telemetry block every 2 seconds while a satellite is selected.
  useEffect(() => {
    setTelemetry(null);
    if (!selectedSatId) return undefined;
    let cancelled = false;

    const load = async () => {
      try {
        const data = await apiGet(`/api/satellites/${selectedSatId}/telemetry`);
        if (!cancelled) setTelemetry(data);
      } catch (_) { /* keep last block; cells show — */ }
    };

    load();
    const id = setInterval(load, 2000);
    return () => { cancelled = true; clearInterval(id); };
  }, [selectedSatId]);

  const satNorad = sat?.norad_id;
  const satAlerts = useMemo(
    () => (satNorad == null
      ? []
      : alerts.filter((a) => String(a.sat1?.id) === String(satNorad) || String(a.sat2?.id) === String(satNorad))),
    [alerts, satNorad]
  );

  if (!sat) {
    return (
      <div className="tm-strip tm-strip--empty">
        <span className="tm-empty-label">No object selected</span>
        <span className="tm-empty-hint">— click a satellite on the globe</span>
      </div>
    );
  }

  const pos = sat.position ?? {};
  const vel = sat.velocity ?? {};
  const altKm = num(sat.altitude_km);
  const spdKmh = num(sat.speed_kmh);
  const spdKms = spdKmh == null ? null : (spdKmh / 3600).toFixed(3);
  const period = periodFromState(pos, vel);

  // Object type: alert payload (SATCAT) first, else the telemetry block.
  const alertSide = satAlerts
    .map((a) => (String(a.sat1?.id) === String(satNorad) ? a.sat1 : a.sat2))
    .find((s) => s?.object_type);
  const objectType = alertSide?.object_type ?? telemetry?.object_type ?? null;
  // Fuel is only meaningful for payloads, and only once the backend says so.
  const isPayload = objectType != null
    && !NON_PAYLOAD.has(String(objectType).toUpperCase())
    && telemetry?.telemetry_available === true;
  const simTag = telemetry?.simulated ? 'SIM' : null;
  const m = telemetry?.model;
  const modelTitle = m
    ? `Simulated, not measured: ${m.fuel ?? 'model'} (wet ${m.assumed_wet_mass_kg ?? '—'} kg, propellant ${m.assumed_propellant_mass_kg ?? '—'} kg, Isp ${m.assumed_isp_s ?? '—'} s)`
    : undefined;

  const fuel = num(telemetry?.fuel_remaining_pct);
  const fuelClass = fuel == null ? '' : fuel < 15 ? 'is-warning' : fuel < 35 ? 'is-caution' : '';
  const illum = telemetry?.illumination ?? null;

  // Risk from the alerts this object is part of (backend severity tier).
  const maxCpi = satAlerts.length > 0
    ? Math.max(...satAlerts.map((a) => Number(a.cpi_score ?? 0)))
    : null;
  const status = satAlerts.length > 0
    ? satAlerts.map(severityLabel).sort((a, b) => (SEV_RANK[b] ?? 0) - (SEV_RANK[a] ?? 0))[0]
    : 'NOMINAL';
  const cpiClass = stateClass(status);

  return (
    <div className="tm-strip">
      <div className="tm-ident">
        <div className="tm-ident-name" title={sat.name}>{sat.name}</div>
        <div className="tm-ident-meta">
          <span>NORAD {sat.norad_id}</span>
          {objectType && <span>{objectType}</span>}
        </div>
      </div>

      <div className="tm-cells">
        <Cell label="Position ECI" wide>
          <span><span className="tm-axis">X</span><TelemetryValue value={fixed(pos.x, 1)} /></span>
          <span><span className="tm-axis">Y</span><TelemetryValue value={fixed(pos.y, 1)} /></span>
          <span><span className="tm-axis">Z</span><TelemetryValue value={fixed(pos.z, 1)} unit="km" /></span>
        </Cell>

        <Cell label="Velocity ECI" wide>
          <span><span className="tm-axis">X</span><TelemetryValue value={fixed(vel.vx, 3)} /></span>
          <span><span className="tm-axis">Y</span><TelemetryValue value={fixed(vel.vy, 3)} /></span>
          <span><span className="tm-axis">Z</span><TelemetryValue value={fixed(vel.vz, 3)} unit="km/s" /></span>
        </Cell>

        <Cell label="Altitude">
          <TelemetryValue value={fixed(altKm, 1)} unit="km" />
        </Cell>

        <Cell label="Speed">
          <TelemetryValue value={spdKms} unit="km/s" />
        </Cell>

        <Cell label="Period" title="Osculating period from propagated r, v (vis-viva)">
          <span>
            {period == null ? '—' : period.toFixed(1)}
            {period != null && <span className="tm-unit">min</span>}
          </span>
        </Cell>

        <Cell
          label="Illum"
          title={telemetry?.illumination_model ? `Computed: ${telemetry.illumination_model}` : undefined}
        >
          <span className={illum === 'eclipse' ? 'is-caution' : ''}>{illum ? illum.toUpperCase() : '—'}</span>
        </Cell>

        {isPayload && (
          <Cell label="Fuel" tag={simTag} title={modelTitle}>
            {fuel == null ? (
              <span>—</span>
            ) : (
              <>
                <span className={fuelClass}>{fuel.toFixed(1)}<span className="tm-unit">%</span></span>
                <span className="tm-bar">
                  <span
                    className={`tm-bar-fill ${fuelClass}`}
                    style={{ width: `${Math.min(Math.max(fuel, 0), 100)}%` }}
                  />
                </span>
              </>
            )}
          </Cell>
        )}

        <Cell label={`Risk · ${satAlerts.length} alert${satAlerts.length === 1 ? '' : 's'}`}>
          <span className={cpiClass}>
            {maxCpi == null ? '—' : maxCpi.toFixed(1)}<span className="tm-unit">CPI</span>
          </span>
          <span className={`tm-chip ${stateClass(status)}`}>{status}</span>
        </Cell>
      </div>
    </div>
  );
}
