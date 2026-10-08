import { useState, useEffect, useRef } from 'react';
import useStore from '../../store/useStore';
import { apiPost } from '../../utils/api';

const DIRECTIONS = [
  { key: 'PROGRADE',   label: 'Prograde',   desc: 'Increase altitude / speed' },
  { key: 'RETROGRADE', label: 'Retrograde', desc: 'Decrease altitude / speed' },
  { key: 'RADIAL',     label: 'Radial',     desc: 'Change orbit shape' },
  { key: 'NORMAL',     label: 'Normal',     desc: 'Change orbital plane' },
];

const TABS = ['MANEUVER', 'ORBIT CHANGE', 'OVERRIDE'];

function hohmannDv(altCurrentKm, altTargetKm) {
  const mu = 3.986e14;
  const R1 = (6371 + altCurrentKm) * 1e3;
  const R2 = (6371 + altTargetKm) * 1e3;
  if (R1 <= 0 || R2 <= 0) return { dv1: 0, dv2: 0, transferMin: 0 };
  const v1 = Math.sqrt(mu / R1);
  const vt1 = Math.sqrt(2 * mu * R2 / (R1 + R2) / R1);
  const vt2 = Math.sqrt(2 * mu * R1 / (R1 + R2) / R2);
  const v2 = Math.sqrt(mu / R2);
  const dv1 = Math.abs(vt1 - v1);
  const dv2 = Math.abs(v2 - vt2);
  const semiMajorTransfer = (R1 + R2) / 2;
  const transferPeriodSec = 2 * Math.PI * Math.sqrt(semiMajorTransfer ** 3 / mu);
  const transferMin = (transferPeriodSec / 60 / 2).toFixed(1);
  return { dv1: (dv1).toFixed(2), dv2: (dv2).toFixed(2), transferMin };
}

function directionToRsw(direction, dvMs) {
  switch (direction) {
    case 'RADIAL': return { dvx: dvMs, dvy: 0, dvz: 0 };
    case 'NORMAL': return { dvx: 0, dvy: 0, dvz: dvMs };
    case 'RETROGRADE': return { dvx: 0, dvy: -dvMs, dvz: 0 };
    default: return { dvx: 0, dvy: dvMs, dvz: 0 }; // PROGRADE
  }
}

export default function UplinkDownlinkV2({ selectedSatId }) {
  const satellites = useStore((s) => s.satellites);
  const addDecisionLogEntry = useStore((s) => s.addDecisionLogEntry);
  const alerts = useStore((s) => s.alerts);

  const sat = selectedSatId != null
    ? satellites.find((s) => s.norad_id === selectedSatId)
    : null;

  const [activeTab, setActiveTab] = useState('MANEUVER');
  const [dvAmount, setDvAmount] = useState(0.1);
  const [direction, setDirection] = useState('PROGRADE');
  const [targetAlt, setTargetAlt] = useState(550);
  const [override, setOverride] = useState({ x: '', y: '', z: '', vx: '', vy: '', vz: '' });
  const [cmdLog, setCmdLog] = useState([]);
  const [downlinkHistory, setDownlinkHistory] = useState([]);
  const [fuelPct, setFuelPct] = useState(null);
  const [maneuverPreview, setManeuverPreview] = useState(null);
  const [previewError, setPreviewError] = useState(null);
  const downlinkTimer = useRef(null);

  // Fuel comes from the backend telemetry tracker, not a hardcoded constant.
  useEffect(() => {
    if (!sat) {
      setFuelPct(null);
      return;
    }
    let cancelled = false;
    const loadFuel = async () => {
      try {
        const res = await fetch(`/api/satellites/${sat.norad_id}/telemetry`);
        if (!res.ok) throw new Error(`status ${res.status}`);
        const data = await res.json();
        if (cancelled) return;
        const value = Number(data?.fuel_remaining_pct);
        setFuelPct(Number.isFinite(value) ? value : null);
      } catch (_) {
        if (!cancelled) setFuelPct(null);
      }
    };
    loadFuel();
    const timer = setInterval(loadFuel, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [sat]);

  // Build downlink history from sat data
  useEffect(() => {
    if (!sat) { setDownlinkHistory([]); return; }
    downlinkTimer.current = setInterval(() => {
      const ts = new Date().toISOString().substring(11, 23);
      const satAlerts = alerts.filter(
        (a) => a.sat1?.id === sat.norad_id || a.sat2?.id === sat.norad_id
      );
      const cpi = satAlerts.length > 0
        ? Math.max(...satAlerts.map((a) => Number(a.cpi_score ?? 0)))
        : 0;
      setDownlinkHistory((prev) => [
        {
          ts,
          alt: sat.altitude_km?.toFixed(0) ?? '—',
          spd: (sat.speed_kmh ? (sat.speed_kmh / 3600).toFixed(2) : '—'),
          fuel: fuelPct != null ? fuelPct.toFixed(1) : '—',
          cpi: cpi.toFixed(1),
        },
        ...prev,
      ].slice(0, 5));
    }, 2000);
    return () => clearInterval(downlinkTimer.current);
  }, [sat, alerts, fuelPct]);

  // Nearest threat for maneuver preview
  const nearestAlert = sat
    ? alerts
        .filter((a) => a.sat1?.id === sat.norad_id || a.sat2?.id === sat.norad_id)
        .sort((a, b) => Number(a.miss_distance_km ?? 999) - Number(b.miss_distance_km ?? 999))[0]
    : null;

  const currentMiss = nearestAlert ? Number(nearestAlert.miss_distance_km ?? 0) : null;
  const altKm = sat?.altitude_km ?? 400;
  const { dv1, dv2, transferMin } = hohmannDv(altKm, targetAlt);

  // Post-burn miss distance is requested from the backend, which actually
  // propagates the burn. It is never synthesised on the client.
  useEffect(() => {
    if (!sat || currentMiss == null) {
      setManeuverPreview(null);
      setPreviewError(null);
      return;
    }
    let cancelled = false;
    const dv = directionToRsw(direction, dvAmount);
    const run = async () => {
      try {
        const res = await fetch('/api/preflight', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            norad_id: sat.norad_id,
            dvx: dv.dvx,
            dvy: dv.dvy,
            dvz: dv.dvz,
            frame: 'RSW',
          }),
        });
        if (!res.ok) throw new Error(`status ${res.status}`);
        const data = await res.json();
        if (cancelled) return;
        setManeuverPreview(data);
        setPreviewError(null);
      } catch (e) {
        if (cancelled) return;
        setManeuverPreview(null);
        setPreviewError(e?.message || 'preview unavailable');
      }
    };

    const timer = setTimeout(run, 400);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [sat, currentMiss, direction, dvAmount]);

  const logCmd = (type, result, ok = true) => {
    const ts = new Date().toISOString().substring(11, 19);
    setCmdLog((prev) => [{ ts, type, sat: sat?.name ?? '—', result, ok }, ...prev].slice(0, 10));
    addDecisionLogEntry({ time: new Date().toISOString(), decision: type, sat: sat?.name ?? '—' });
  };

  const handleManeuver = async () => {
    if (!sat) return;
    try {
      const dv = directionToRsw(direction, dvAmount);
      await apiPost('/api/maneuver', {
        norad_id: sat.norad_id,
        dvx: dv.dvx,
        dvy: dv.dvy,
        dvz: dv.dvz,
        frame: 'RSW',
      });
      logCmd('MANEUVER', `${direction} ${dvAmount}m/s OK`, true);
    } catch (e) {
      logCmd('MANEUVER', e.message, false);
    }
  };

  const handleOrbitChange = async () => {
    if (!sat) return;
    try {
      await apiPost('/api/orbit-change', {
        norad_id: sat.norad_id,
        target_altitude_km: targetAlt,
      });
      logCmd('ORBIT CHANGE', `→ ${targetAlt}km OK`, true);
    } catch (e) {
      logCmd('ORBIT CHANGE', e.message, false);
    }
  };

  const overrideKeys = ['x', 'y', 'z', 'vx', 'vy', 'vz'];
  const overrideComplete = overrideKeys.every((f) => {
    const raw = override[f];
    if (raw === '' || raw === null || raw === undefined) return false;
    return Number.isFinite(Number(raw));
  });

  const handleOverride = async () => {
    if (!sat) return;
    if (!overrideComplete) {
      logCmd('OVERRIDE', 'All six ECI components must be numeric', false);
      return;
    }
    try {
      // Pydantic requires floats; sending the raw strings yields a 422.
      const numeric = Object.fromEntries(
        overrideKeys.map((f) => [f, Number(override[f])])
      );
      await apiPost('/api/override', { norad_id: sat.norad_id, ...numeric });
      logCmd('OVERRIDE', 'State applied', true);
      setOverride({ x: '', y: '', z: '', vx: '', vy: '', vz: '' });
    } catch (e) {
      logCmd('OVERRIDE', e.message, false);
    }
  };

  return (
    <div className="udv2-panel">
      {/* DOWNLINK */}
      <div className="udv2-section">
        <div className="udv2-section-header">
          <div className="udv2-live-dot" />
          <span className="udv2-section-title">↓ Downlink</span>
        </div>
        {downlinkHistory.length === 0 ? (
          <div style={{ padding: '8px 10px', fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-dim)' }}>
            {sat ? 'Waiting for data...' : 'No satellite selected'}
          </div>
        ) : (
          <table className="udv2-downlink-table">
            <thead>
              <tr>
                <th>Time</th><th>Alt</th><th>Spd</th><th>Fuel</th><th>CPI</th>
              </tr>
            </thead>
            <tbody>
              {downlinkHistory.map((row, i) => (
                <tr key={i}>
                  <td>{row.ts}</td>
                  <td>{row.alt}km</td>
                  <td>{row.spd}km/s</td>
                  <td>{row.fuel}%</td>
                  <td>{row.cpi}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* UPLINK */}
      <div className="udv2-section">
        <div className="udv2-section-header">
          <span className="udv2-section-title">↑ Uplink Command</span>
          {sat && (
            <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-dim)', marginLeft: 'auto' }}>
              {sat.name}
            </span>
          )}
        </div>

        <div className="udv2-cmd-tabs">
          {TABS.map((tab) => (
            <button
              key={tab}
              type="button"
              className={`udv2-cmd-tab ${activeTab === tab ? 'active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className="udv2-cmd-body">
          {/* MANEUVER TAB */}
          {activeTab === 'MANEUVER' && (
            <>
              <div className="udv2-field">
                <label>Delta-V Amount: {dvAmount.toFixed(2)} m/s</label>
                <input
                  type="range"
                  min={0.01}
                  max={2.0}
                  step={0.01}
                  value={dvAmount}
                  onChange={(e) => setDvAmount(Number(e.target.value))}
                />
              </div>

              <div className="udv2-field">
                <label>Direction</label>
                <div className="udv2-direction-group">
                  {DIRECTIONS.map((d) => (
                    <label key={d.key} className="udv2-direction-option">
                      <input
                        type="radio"
                        name="direction"
                        value={d.key}
                        checked={direction === d.key}
                        onChange={() => setDirection(d.key)}
                      />
                      <div>
                        <div className="udv2-direction-name">{d.label}</div>
                        <div className="udv2-direction-desc">{d.desc}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {currentMiss != null && (
                <div className="udv2-preview">
                  <div className="udv2-preview-row">
                    <span className="udv2-preview-label">Current miss</span>
                    <span className="udv2-preview-value">{currentMiss.toFixed(2)} km</span>
                  </div>
                  {previewError ? (
                    <div className="udv2-preview-row">
                      <span className="udv2-preview-label">After maneuver</span>
                      <span className="udv2-preview-value" style={{ color: 'var(--alert-red)' }}>
                        unavailable
                      </span>
                    </div>
                  ) : (
                    maneuverPreview?.trajectory && (
                      <>
                        <div className="udv2-preview-row">
                          <span className="udv2-preview-label">After maneuver</span>
                          <span className="udv2-preview-value better">
                            {maneuverPreview.trajectory.min_miss_distance_km != null
                              ? `${Number(maneuverPreview.trajectory.min_miss_distance_km).toFixed(2)} km`
                              : '—'}
                          </span>
                        </div>
                        <div className="udv2-preview-row">
                          <span className="udv2-preview-label">Closest TCA</span>
                          <span className="udv2-preview-value">
                            {maneuverPreview.trajectory.closest_tca_minutes != null
                              ? `${Number(maneuverPreview.trajectory.closest_tca_minutes).toFixed(1)} min`
                              : '—'}
                          </span>
                        </div>
                      </>
                    )
                  )}
                  <div className="udv2-preview-row">
                    <span className="udv2-preview-label">Fuel cost</span>
                    <span className="udv2-preview-value">{(dvAmount * 0.3).toFixed(2)}%</span>
                  </div>
                </div>
              )}

              <button
                type="button"
                className="udv2-execute-btn"
                disabled={!sat}
                onClick={handleManeuver}
              >
                Execute Maneuver
              </button>
            </>
          )}

          {/* ORBIT CHANGE TAB */}
          {activeTab === 'ORBIT CHANGE' && (
            <>
              <div className="udv2-field">
                <label>Current altitude: {altKm.toFixed(0)} km</label>
              </div>
              <div className="udv2-field">
                <label>Target altitude: {targetAlt} km</label>
                <input
                  type="range"
                  min={200}
                  max={35786}
                  step={10}
                  value={targetAlt}
                  onChange={(e) => setTargetAlt(Number(e.target.value))}
                />
              </div>
              <div className="udv2-preview">
                <div className="udv2-preview-row">
                  <span className="udv2-preview-label">Burn 1 (perigee)</span>
                  <span className="udv2-preview-value">{dv1} m/s</span>
                </div>
                <div className="udv2-preview-row">
                  <span className="udv2-preview-label">Burn 2 (apogee)</span>
                  <span className="udv2-preview-value">{dv2} m/s</span>
                </div>
                <div className="udv2-preview-row">
                  <span className="udv2-preview-label">Transfer time</span>
                  <span className="udv2-preview-value">{transferMin} min</span>
                </div>
                <div className="udv2-preview-row">
                  <span className="udv2-preview-label">Total ΔV</span>
                  <span className="udv2-preview-value">{(Number(dv1) + Number(dv2)).toFixed(2)} m/s</span>
                </div>
              </div>
              <button
                type="button"
                className="udv2-execute-btn"
                disabled={!sat}
                onClick={handleOrbitChange}
              >
                Execute Orbit Change
              </button>
            </>
          )}

          {/* OVERRIDE TAB */}
          {activeTab === 'OVERRIDE' && (
            <>
              <div className="udv2-warn-banner">⚠ SIMULATION MODE ONLY — Direct state override</div>
              <div className="udv2-override-fields">
                {['x', 'y', 'z'].map((f) => (
                  <input
                    key={f}
                    type="number"
                    step="any"
                    placeholder={`Position ${f.toUpperCase()} (km)`}
                    value={override[f]}
                    onChange={(e) => setOverride((o) => ({ ...o, [f]: e.target.value }))}
                  />
                ))}
                {['vx', 'vy', 'vz'].map((f) => (
                  <input
                    key={f}
                    type="number"
                    step="any"
                    placeholder={`Velocity ${f.toUpperCase()} (km/s)`}
                    value={override[f]}
                    onChange={(e) => setOverride((o) => ({ ...o, [f]: e.target.value }))}
                  />
                ))}
              </div>
              <button
                type="button"
                className="udv2-execute-btn"
                disabled={!sat || !overrideComplete}
                onClick={handleOverride}
              >
                Apply Override
              </button>
            </>
          )}
        </div>
      </div>

      {/* COMMAND LOG */}
      <div className="udv2-cmd-log">
        <div style={{ fontFamily: 'var(--mono)', fontSize: 8, letterSpacing: '1.5px', textTransform: 'uppercase', color: 'var(--text-dim)', marginBottom: 4 }}>
          Command Log
        </div>
        {cmdLog.length === 0 ? (
          <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-dim)' }}>No commands sent</div>
        ) : (
          cmdLog.map((row, i) => (
            <div key={i} className={`udv2-log-row ${row.ok ? 'ok' : 'fail'}`}>
              <span className="udv2-log-time">{row.ts}</span>
              <span className="udv2-log-type">{row.type}</span>
              <span className="udv2-log-sat">{row.sat}</span>
              <span className={`udv2-log-status udv2-log-result`}>{row.result}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
