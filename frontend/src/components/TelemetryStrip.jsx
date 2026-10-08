import { useEffect, useState, useRef, useMemo } from 'react';
import useStore from '../store/useStore';
import { apiGet } from '../utils/api';
import '../styles/telemetry.css';

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
      {unit && <span className="tm-unit">{unit}</span>}
    </span>
  );
}

function Cell({ label, wide = false, children }) {
  return (
    <div className={`tm-cell${wide ? ' tm-cell--wide' : ''}`}>
      <div className="tm-cell-label">{label}</div>
      <div className="tm-cell-value">{children}</div>
    </div>
  );
}

const stateClass = (status) =>
  status === 'CRITICAL' ? 'is-warning'
    : status === 'WARNING' || status === 'WATCH' ? 'is-caution'
      : 'is-nominal';

export default function TelemetryStrip({ selectedSatId }) {
  // Select only the chosen satellite rather than the full list.
  const sat = useStore((s) => (
    selectedSatId != null
      ? s.satellites.find((x) => x.norad_id === selectedSatId) ?? null
      : null
  ));
  const alerts = useStore((s) => s.alerts);
  const [telemetry, setTelemetry] = useState(null);

  // Poll telemetry every 2 seconds if a satellite is selected
  useEffect(() => {
    if (!selectedSatId) { setTelemetry(null); return; }
    let cancelled = false;

    const fetch = async () => {
      try {
        const data = await apiGet(`/api/satellites/${selectedSatId}/telemetry`);
        if (!cancelled) setTelemetry(data);
      } catch (_) {}
    };

    fetch();
    const id = setInterval(fetch, 2000);
    return () => { cancelled = true; clearInterval(id); };
  }, [selectedSatId]);

  const satNorad = sat?.norad_id;
  const satAlerts = useMemo(
    () => (satNorad == null
      ? []
      : alerts.filter((a) => a.sat1?.id === satNorad || a.sat2?.id === satNorad)),
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
  const altKm = sat.altitude_km ?? 0;
  const spdKmh = sat.speed_kmh ?? 0;
  const spdKms = (spdKmh / 3600).toFixed(3);

  // Compute orbital period from altitude (approximate)
  const periodMin = altKm > 0
    ? ((2 * Math.PI * Math.sqrt(((6371 + altKm) * 1e3) ** 3 / (3.986e14))) / 60).toFixed(1)
    : '—';

  // Telemetry from API or defaults
  const fuel = telemetry?.fuel_remaining_pct ?? 85;
  const bat = telemetry?.battery_pct ?? 92;
  const temp = telemetry?.temperature_c ?? 22;
  const sig = telemetry?.signal_strength_dbm ?? -85;

  // Risk from alerts
  const maxCpi = satAlerts.length > 0
    ? Math.max(...satAlerts.map((a) => Number(a.cpi_score ?? 0)))
    : 0;
  const status = maxCpi >= 8 ? 'CRITICAL' : maxCpi >= 5 ? 'WARNING' : maxCpi > 0 ? 'WATCH' : 'NOMINAL';
  const cpiClass = maxCpi >= 8 ? 'is-warning' : maxCpi >= 5 ? 'is-caution' : 'is-nominal';
  const fuelClass = fuel < 15 ? 'is-warning' : fuel < 35 ? 'is-caution' : '';

  return (
    <div className="tm-strip">
      <div className="tm-ident">
        <div className="tm-ident-name" title={sat.name}>{sat.name}</div>
        <div className="tm-ident-meta">
          <span>NORAD {sat.norad_id}</span>
        </div>
      </div>

      <div className="tm-cells">
        <Cell label="Position ECI" wide>
          <span><span className="tm-axis">X</span><TelemetryValue value={Number(pos.x ?? 0).toFixed(1)} /></span>
          <span><span className="tm-axis">Y</span><TelemetryValue value={Number(pos.y ?? 0).toFixed(1)} /></span>
          <span><span className="tm-axis">Z</span><TelemetryValue value={Number(pos.z ?? 0).toFixed(1)} unit="km" /></span>
        </Cell>

        <Cell label="Velocity ECI" wide>
          <span><span className="tm-axis">X</span><TelemetryValue value={Number(vel.vx ?? 0).toFixed(3)} /></span>
          <span><span className="tm-axis">Y</span><TelemetryValue value={Number(vel.vy ?? 0).toFixed(3)} /></span>
          <span><span className="tm-axis">Z</span><TelemetryValue value={Number(vel.vz ?? 0).toFixed(3)} unit="km/s" /></span>
        </Cell>

        <Cell label="Altitude">
          <TelemetryValue value={altKm.toFixed(1)} unit="km" />
        </Cell>

        <Cell label="Speed">
          <TelemetryValue value={spdKms} unit="km/s" />
        </Cell>

        <Cell label="Period">
          <span>{periodMin}<span className="tm-unit">min</span></span>
        </Cell>

        <Cell label="Fuel">
          <span className={fuelClass}>{fuel}<span className="tm-unit">%</span></span>
          <span className="tm-bar">
            <span className={`tm-bar-fill ${fuelClass}`} style={{ width: `${fuel}%` }} />
          </span>
        </Cell>

        <Cell label="Bat / Temp / Sig">
          <span className="tm-sub">
            {bat}<span className="tm-unit">%</span>{' '}
            {temp}<span className="tm-unit">°C</span>{' '}
            {sig}<span className="tm-unit">dBm</span>
          </span>
        </Cell>

        <Cell label={`Risk · ${satAlerts.length} alert${satAlerts.length === 1 ? '' : 's'}`}>
          <span className={cpiClass}>
            {maxCpi.toFixed(1)}<span className="tm-unit">CPI</span>
          </span>
          <span className={`tm-chip ${stateClass(status)}`}>{status}</span>
        </Cell>
      </div>
    </div>
  );
}
