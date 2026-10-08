import { useState, useEffect, useRef, useMemo } from 'react';
import useStore from '../../store/useStore';
import { apiPost } from '../../utils/api';
import '../../styles/uplink.css';

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
  // Select only the chosen satellite rather than the full list.
  const sat = useStore((s) => (
    selectedSatId != null
      ? s.satellites.find((x) => x.norad_id === selectedSatId)
      : null
  ));
  const addDecisionLogEntry = useStore((s) => s.addDecisionLogEntry);
  const alerts = useStore((s) => s.alerts);

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
  const satNorad = sat?.norad_id;
  const nearestAlert = useMemo(() => (
    satNorad != null
      ? alerts
          .filter((a) => a.sat1?.id === satNorad || a.sat2?.id === satNorad)
          .sort((a, b) => Number(a.miss_distance_km ?? 999) - Number(b.miss_distance_km ?? 999))[0]
      : null
  ), [alerts, satNorad]);

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
    <div className="ui-panel ul-panel">
      <div className="ui-panel-header">
        <span className="ul-panel-title">
          <span className="ui-label">Command Uplink</span>
        </span>
        <span className="ul-header-meta">
          {sat ? (
            <>
              <span className="ul-target">{sat.name}</span>
              <span>{sat.norad_id}</span>
            </>
          ) : (
            <span>NO TARGET</span>
          )}
        </span>
      </div>

      <div className="ul-tabs" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab}
            type="button"
            role="tab"
            aria-selected={activeTab === tab}
            className={`ul-tab${activeTab === tab ? ' is-active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="ul-body">
        {/* COMMAND FORM */}
        <div className="ul-section">
          <div className="ul-form">
            {/* MANEUVER TAB */}
            {activeTab === 'MANEUVER' && (
              <>
                <div className="ul-row">
                  <span className="ul-row-label">Delta-V</span>
                  <input
                    className="ul-range"
                    type="range"
                    min={0.01}
                    max={2.0}
                    step={0.01}
                    value={dvAmount}
                    onChange={(e) => setDvAmount(Number(e.target.value))}
                  />
                  <span className="ul-row-value">
                    {dvAmount.toFixed(2)}<span className="ul-unit">m/s</span>
                  </span>
                </div>

                <div className="ul-row ul-row--span ul-row--top">
                  <span className="ul-row-label">Direction</span>
                  <div className="ul-dir-grid">
                    {DIRECTIONS.map((d) => (
                      <label
                        key={d.key}
                        className={`ul-dir${direction === d.key ? ' is-active' : ''}`}
                      >
                        <input
                          type="radio"
                          name="direction"
                          value={d.key}
                          checked={direction === d.key}
                          onChange={() => setDirection(d.key)}
                        />
                        <span className="ul-dir-name">{d.label}</span>
                        <span className="ul-dir-desc">{d.desc}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {currentMiss != null && (
                  <div className="ul-row ul-row--span ul-row--top">
                    <span className="ul-row-label">Preview</span>
                    <div className="ul-kv">
                      <div className="ul-kv-row">
                        <span className="ul-kv-key">Current miss</span>
                        <span className="ul-kv-val">{currentMiss.toFixed(2)}<span className="ul-unit">km</span></span>
                      </div>
                      {previewError ? (
                        <div className="ul-kv-row">
                          <span className="ul-kv-key">After maneuver</span>
                          <span className="ul-kv-val is-warning">UNAVAILABLE</span>
                        </div>
                      ) : (
                        maneuverPreview?.trajectory && (
                          <>
                            <div className="ul-kv-row">
                              <span className="ul-kv-key">After maneuver</span>
                              <span className="ul-kv-val is-nominal">
                                {maneuverPreview.trajectory.min_miss_distance_km != null
                                  ? <>{Number(maneuverPreview.trajectory.min_miss_distance_km).toFixed(2)}<span className="ul-unit">km</span></>
                                  : '—'}
                              </span>
                            </div>
                            <div className="ul-kv-row">
                              <span className="ul-kv-key">Closest TCA</span>
                              <span className="ul-kv-val">
                                {maneuverPreview.trajectory.closest_tca_minutes != null
                                  ? <>{Number(maneuverPreview.trajectory.closest_tca_minutes).toFixed(1)}<span className="ul-unit">min</span></>
                                  : '—'}
                              </span>
                            </div>
                          </>
                        )
                      )}
                      <div className="ul-kv-row">
                        <span className="ul-kv-key">Fuel cost</span>
                        <span className="ul-kv-val">{(dvAmount * 0.3).toFixed(2)}<span className="ul-unit">%</span></span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="ul-actions">
                  <button
                    type="button"
                    className="ui-btn ui-btn--primary"
                    disabled={!sat}
                    onClick={handleManeuver}
                  >
                    Execute Maneuver
                  </button>
                </div>
              </>
            )}

            {/* ORBIT CHANGE TAB */}
            {activeTab === 'ORBIT CHANGE' && (
              <>
                <div className="ul-row">
                  <span className="ul-row-label">Current Alt</span>
                  <span />
                  <span className="ul-row-value">
                    {altKm.toFixed(0)}<span className="ul-unit">km</span>
                  </span>
                </div>
                <div className="ul-row">
                  <span className="ul-row-label">Target Alt</span>
                  <input
                    className="ul-range"
                    type="range"
                    min={200}
                    max={35786}
                    step={10}
                    value={targetAlt}
                    onChange={(e) => setTargetAlt(Number(e.target.value))}
                  />
                  <span className="ul-row-value">
                    {targetAlt}<span className="ul-unit">km</span>
                  </span>
                </div>
                <div className="ul-row ul-row--span ul-row--top">
                  <span className="ul-row-label">Hohmann</span>
                  <div className="ul-kv">
                    <div className="ul-kv-row">
                      <span className="ul-kv-key">Burn 1 (perigee)</span>
                      <span className="ul-kv-val">{dv1}<span className="ul-unit">m/s</span></span>
                    </div>
                    <div className="ul-kv-row">
                      <span className="ul-kv-key">Burn 2 (apogee)</span>
                      <span className="ul-kv-val">{dv2}<span className="ul-unit">m/s</span></span>
                    </div>
                    <div className="ul-kv-row">
                      <span className="ul-kv-key">Transfer time</span>
                      <span className="ul-kv-val">{transferMin}<span className="ul-unit">min</span></span>
                    </div>
                    <div className="ul-kv-row">
                      <span className="ul-kv-key">Total ΔV</span>
                      <span className="ul-kv-val">{(Number(dv1) + Number(dv2)).toFixed(2)}<span className="ul-unit">m/s</span></span>
                    </div>
                  </div>
                </div>
                <div className="ul-actions">
                  <button
                    type="button"
                    className="ui-btn ui-btn--primary"
                    disabled={!sat}
                    onClick={handleOrbitChange}
                  >
                    Execute Orbit Change
                  </button>
                </div>
              </>
            )}

            {/* OVERRIDE TAB */}
            {activeTab === 'OVERRIDE' && (
              <>
                <div className="ul-banner is-caution">Simulation mode only — direct state override</div>
                <div className="ul-row ul-row--span">
                  <span className="ul-row-label">Position km</span>
                  <div className="ul-override-grid">
                    {['x', 'y', 'z'].map((f) => (
                      <input
                        key={f}
                        className="ui-input ul-input"
                        type="number"
                        step="any"
                        placeholder={f.toUpperCase()}
                        aria-label={`Position ${f.toUpperCase()} (km)`}
                        value={override[f]}
                        onChange={(e) => setOverride((o) => ({ ...o, [f]: e.target.value }))}
                      />
                    ))}
                  </div>
                </div>
                <div className="ul-row ul-row--span">
                  <span className="ul-row-label">Velocity km/s</span>
                  <div className="ul-override-grid">
                    {['vx', 'vy', 'vz'].map((f) => (
                      <input
                        key={f}
                        className="ui-input ul-input"
                        type="number"
                        step="any"
                        placeholder={f.toUpperCase()}
                        aria-label={`Velocity ${f.toUpperCase()} (km/s)`}
                        value={override[f]}
                        onChange={(e) => setOverride((o) => ({ ...o, [f]: e.target.value }))}
                      />
                    ))}
                  </div>
                </div>
                <div className="ul-actions">
                  <button
                    type="button"
                    className="ui-btn ui-btn--danger"
                    disabled={!sat || !overrideComplete}
                    onClick={handleOverride}
                  >
                    Apply Override
                  </button>
                </div>
              </>
            )}
          </div>
        </div>

        {/* COMMAND LOG */}
        <div className="ul-section">
          <div className="ul-section-head">
            <span className="ui-label">Command Log</span>
          </div>
          {cmdLog.length === 0 ? (
            <div className="ul-empty">No commands sent</div>
          ) : (
            <div className="ul-log">
              <div className="ul-log-row ul-log-row--head">
                <span>UTC</span><span>Dir</span><span>Cmd</span><span>Result</span><span className="ul-log-status">Stat</span>
              </div>
              {cmdLog.map((row, i) => (
                <div key={i} className="ul-log-row" title={`${row.sat} — ${row.result}`}>
                  <span className="ul-log-time">{row.ts}</span>
                  <span className="ul-log-dir">UL</span>
                  <span className="ul-log-type">{row.type}</span>
                  <span className="ul-log-msg">{row.sat} · {row.result}</span>
                  <span className={`ul-log-status ${row.ok ? 'is-nominal' : 'is-warning'}`}>
                    {row.ok ? 'OK' : 'FAIL'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* DOWNLINK */}
        <div className="ul-section">
          <div className="ul-section-head">
            <span className="ui-label">Downlink</span>
            {downlinkHistory.length > 0 && <span className="ui-status is-nominal">Rx</span>}
          </div>
          {downlinkHistory.length === 0 ? (
            <div className="ul-empty">
              {sat ? 'Waiting for data...' : 'No satellite selected'}
            </div>
          ) : (
            <table className="ul-table">
              <thead>
                <tr>
                  <th>UTC</th><th>Alt km</th><th>Spd km/s</th><th>Fuel %</th><th>CPI</th>
                </tr>
              </thead>
              <tbody>
                {downlinkHistory.map((row, i) => (
                  <tr key={i}>
                    <td>{row.ts}</td>
                    <td>{row.alt}</td>
                    <td>{row.spd}</td>
                    <td>{row.fuel}</td>
                    <td>{row.cpi}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
