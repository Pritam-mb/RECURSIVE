import { useEffect, useState, useRef } from 'react';
import useStore from '../store/useStore';
import { apiGet } from '../utils/api';

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
    <span className={flash ? 'ts-changed' : ''}>
      {value ?? '—'}{unit}
    </span>
  );
}

export default function TelemetryStrip({ selectedSatId }) {
  const satellites = useStore((s) => s.satellites);
  const alerts = useStore((s) => s.alerts);
  const [telemetry, setTelemetry] = useState(null);

  // Use selectedSatId (prop) which maps to selectedSatelliteId in the store
  const sat = selectedSatId != null
    ? satellites.find((s) => s.norad_id === selectedSatId) ?? null
    : null;

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

  if (!sat) {
    return (
      <div className="telemetry-strip">
        <div className="ts-empty">Select a satellite on the globe to view telemetry</div>
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
  const satAlerts = alerts.filter(
    (a) => a.sat1?.id === sat.norad_id || a.sat2?.id === sat.norad_id
  );
  const maxCpi = satAlerts.length > 0
    ? Math.max(...satAlerts.map((a) => Number(a.cpi_score ?? 0)))
    : 0;
  const status = maxCpi >= 8 ? 'CRITICAL' : maxCpi >= 5 ? 'WARNING' : maxCpi > 0 ? 'WATCH' : 'NOMINAL';
  const cpiColor = maxCpi >= 8 ? 'var(--alert-red)' : maxCpi >= 5 ? 'var(--alert-yellow)' : 'var(--alert-green)';

  return (
    <div className="telemetry-strip">
      {/* Sat name */}
      <div className="ts-sat-name">{sat.name}</div>

      <div className="ts-divider" />

      {/* Position */}
      <div className="ts-group">
        <div className="ts-group-label">Position (ECI km)</div>
        <div className="ts-value-row">
          X:<TelemetryValue value={Number(pos.x ?? 0).toFixed(1)} />
          {' '}Y:<TelemetryValue value={Number(pos.y ?? 0).toFixed(1)} />
          {' '}Z:<TelemetryValue value={Number(pos.z ?? 0).toFixed(1)} />
        </div>
      </div>

      <div className="ts-divider" />

      {/* Velocity */}
      <div className="ts-group">
        <div className="ts-group-label">Velocity (km/s)</div>
        <div className="ts-value-row">
          Vx:<TelemetryValue value={Number(vel.vx ?? 0).toFixed(3)} />
          {' '}Vy:<TelemetryValue value={Number(vel.vy ?? 0).toFixed(3)} />
          {' '}Vz:<TelemetryValue value={Number(vel.vz ?? 0).toFixed(3)} />
        </div>
      </div>

      <div className="ts-divider" />

      {/* Orbital */}
      <div className="ts-group">
        <div className="ts-group-label">Orbital</div>
        <div className="ts-value-row">
          ALT:<TelemetryValue value={altKm.toFixed(0)} unit="km" />
          {' '}SPD:<TelemetryValue value={spdKms} unit="km/s" />
        </div>
        <div className="ts-value-row">PER:{periodMin}min</div>
      </div>

      <div className="ts-divider" />

      {/* Health */}
      <div className="ts-group">
        <div className="ts-group-label">Health</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-dim)' }}>FUEL</span>
          <div className="ts-fuel-bar">
            <div className="ts-fuel-fill" style={{ width: `${fuel}%` }} />
          </div>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-bright)' }}>{fuel}%</span>
        </div>
        <div className="ts-value-row">
          BAT:{bat}% TMP:{temp}°C SIG:{sig}dBm
        </div>
      </div>

      <div className="ts-divider" />

      {/* Risk */}
      <div className="ts-group">
        <div className="ts-group-label">Risk</div>
        <div className="ts-value-row">
          <span className="ts-risk-cpi" style={{ color: cpiColor }}>
            CPI {maxCpi.toFixed(1)}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: satAlerts.length > 0 ? 'var(--alert-red)' : 'var(--text-dim)' }}>
            ALERTS:{satAlerts.length}
          </span>
          <span className={`ts-status-badge ${status}`}>{status}</span>
        </div>
      </div>
    </div>
  );
}
