import { useState } from 'react';
import useStore from '../../store/useStore';
import useTestMode from '../../hooks/useTestMode';
import PreflightModal from './PreflightModal';
import '../../styles/uplink.css';

const apiBaseUrl = '';

const ManualControl = () => {
  const maneuver = useStore((s) => s.maneuver);
  const setManeuver = useStore((s) => s.setManeuver);
  const resetManeuver = useStore((s) => s.resetManeuver);
  const setAlerts = useStore((s) => s.setAlerts);
  const setCascadePlan = useStore((s) => s.setCascadePlan);
  const setHotspots = useStore((s) => s.setHotspots);
  const selectedSatelliteId = useStore((s) => s.selectedSatelliteId);
  const selectedSatellite = useStore((s) => {
    if (!s.selectedSatelliteId) return null;
    return s.satellites.find((sat) => sat.norad_id === s.selectedSatelliteId) || null;
  });
  const setSimulationActive = useStore((s) => s.setSimulationActive);
  const setSelectedSatelliteId = useStore((s) => s.setSelectedSatelliteId);
  const setTestActive = useTestMode((s) => s.setTestActive);
  const setSelectedA = useTestMode((s) => s.setSelectedA);
  const setSelectedB = useTestMode((s) => s.setSelectedB);
  
  const [showPreflight, setShowPreflight] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const totalDeltaV = Math.sqrt((maneuver.dvx ** 2) + (maneuver.dvy ** 2) + (maneuver.dvz ** 2));
  const burnReady = totalDeltaV > 0.01;

  const handleSliderChange = (axis, value) => {
    setManeuver(axis, parseFloat(value));
  };

  const handleExecute = () => {
    if (!selectedSatelliteId || !burnReady) return;
    setShowPreflight(true);
  };

  const handleConfirmBurn = async () => {
    if (!selectedSatelliteId) return;

    setLoading(true);
    setResult(null);

    try {
      const resp = await fetch(`${apiBaseUrl}/api/maneuver`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          norad_id: selectedSatelliteId,
          dvx: maneuver.dvx,
          dvy: maneuver.dvy,
          dvz: maneuver.dvz,
          frame: 'RSW',
        }),
      });
      const data = await resp.json();
      if (data?.alerts) {
        setAlerts(data.alerts || []);
      }
      if (Object.prototype.hasOwnProperty.call(data, 'hotspots')) {
        setHotspots(data.hotspots || []);
      }
      if (Object.prototype.hasOwnProperty.call(data, 'cascade_plan')) {
        setCascadePlan({
          graph: data.graph || {},
          alerts: data.alerts || [],
          cascade_plan: data.cascade_plan || [],
          total_delta_v_ms: data.total_delta_v_ms || 0,
          cascade_depth: data.cascade_depth || 0,
          agencies_involved: data.agencies_involved || [],
          seed_satellites: data.seed_satellites || [],
          cpi_threshold: data.cpi_threshold || 5,
          node_probabilities: data.node_probabilities || {},
        });
      }
      setResult(data);
    } catch (e) {
      setResult({ status: 'ERROR', error: e.message });
    }

    setLoading(false);
    setShowPreflight(false);
  };

  const handleTriggerScenario = async () => {
    try {
      const resp = await fetch(`${apiBaseUrl}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: 'collision_hotspot' }),
      });
      const data = await resp.json();
      console.log('Scenario loaded:', data);
      
      // Auto-target the two dummy satellites that spawn
      setSelectedSatelliteId(99901); 
      setTestActive(true);
      setSelectedA({ name: 'SAT-COLLIDER-A', norad_id: 99901 });
      setSelectedB({ name: 'SAT-COLLIDER-B', norad_id: 99902 });
      
      setResult({ status: 'SUCCESS', message: 'Scenario Triggered!' });
      setSimulationActive(true);
    } catch (e) {
      console.error('Scenario trigger failed:', e);
      setResult({ status: 'ERROR', error: e.message });
    }
  };

  const sliders = [
    { axis: 'dvx', label: 'Radial (R)', hint: 'pushes orbit up or down' },
    { axis: 'dvy', label: 'Along-track (S)', hint: 'speeds up or slows down' },
    { axis: 'dvz', label: 'Cross-track (W)', hint: 'tilts the orbital plane' },
  ];

  return (
    <div className="ui-panel ul-mc">
      <div className="ui-panel-header">
        <span className="ui-label">Manual Control</span>
        <span className="ul-header-meta">RSW frame</span>
      </div>

      <div className="ul-mc-body">
        {!selectedSatelliteId ? (
          <div className="ul-empty">SELECT A SATELLITE TO COMMAND</div>
        ) : (
          <div className="ul-mc-target">
            Commanding: <strong>{selectedSatellite?.name || `#${selectedSatelliteId}`}</strong>
          </div>
        )}

        {sliders.map(({ axis, label, hint }) => (
          <div className="ul-slider" key={axis}>
            <div className="ul-slider-head">
              <span className="ul-row-label">
                {label}<span className="ul-hint">{hint}</span>
              </span>
              <span className="ul-row-value">
                {maneuver[axis]?.toFixed(1)}<span className="ul-unit">m/s</span>
              </span>
            </div>
            <input
              className="ul-range"
              type="range"
              min="-5"
              max="5"
              step="0.05"
              value={maneuver[axis] || 0}
              onChange={(e) => handleSliderChange(axis, e.target.value)}
            />
          </div>
        ))}

        <div className="ul-total">
          <span className="ul-row-label">Total delta-V</span>
          <span className="ul-row-value">
            {totalDeltaV.toFixed(3)}<span className="ul-unit">m/s</span>
          </span>
        </div>

        <div className="ul-mc-actions">
          <button className="ui-btn" type="button" onClick={resetManeuver}>
            Reset
          </button>
          <button
            className="ui-btn ui-btn--primary"
            type="button"
            onClick={handleExecute}
            disabled={!selectedSatelliteId || loading || !burnReady}
          >
            {loading ? 'Applying...' : (burnReady ? 'Run Preflight + Burn' : 'Set a non-zero burn')}
          </button>
        </div>

        <button className="ui-btn ul-btn-full" type="button" onClick={handleTriggerScenario}>
          Trigger Scenario
        </button>

        {result && (
          result.error || result.status === 'ERROR' ? (
            <div className="ul-banner is-warning">
              Rejected: {result.error || result.message || 'Unable to apply maneuver'}
            </div>
          ) : (
            <div className="ul-kv">
              <div className="ul-kv-row">
                <span className="ul-kv-key">Burn applied successfully</span>
                <span className="ul-kv-val is-nominal">OK</span>
              </div>
              <div className="ul-kv-row">
                <span className="ul-kv-key">Frame</span>
                <span className="ul-kv-val">{result.frame_used || 'RSW'}</span>
              </div>
              <div className="ul-kv-row">
                <span className="ul-kv-key">New perigee</span>
                <span className="ul-kv-val">{result.new_perigee_km}<span className="ul-unit">km</span></span>
              </div>
              <div className="ul-kv-row">
                <span className="ul-kv-key">New apogee</span>
                <span className="ul-kv-val">{result.new_apogee_km}<span className="ul-unit">km</span></span>
              </div>
              <div className="ul-kv-row">
                <span className="ul-kv-key">New period</span>
                <span className="ul-kv-val">{result.new_period_min}<span className="ul-unit">min</span></span>
              </div>
              <div className="ul-kv-row">
                <span className="ul-kv-key">Delta-V used</span>
                <span className="ul-kv-val">{Number(result.delta_v_ms || 0).toFixed(3)}<span className="ul-unit">m/s</span></span>
              </div>
            </div>
          )
        )}
      </div>

      {showPreflight && (
        <PreflightModal
          noradId={selectedSatelliteId}
          maneuver={maneuver}
          onConfirm={handleConfirmBurn}
          onCancel={() => setShowPreflight(false)}
        />
      )}
    </div>
  );
};

export default ManualControl;
