import { useMemo, useState } from 'react';
import useStore from '../store/useStore';
import useTestMode from '../hooks/useTestMode';

const API_BASE = '';

const Section = ({ title, defaultOpen = true, children }) => {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="panel test-section">
      <button
        type="button"
        className="section-toggle"
        onClick={() => setOpen((prev) => !prev)}
      >
        <span className="panel-title">{title}</span>
        <span className="section-caret">{open ? '^' : 'v'}</span>
      </button>
      {open && <div className="section-body">{children}</div>}
    </div>
  );
};

const OverrideForm = ({ label, dataKey, override, setOverrideMode, setOverrideVectorField, setOverrideField }) => (
  <div className="override-block">
    <div className="override-header">
      <span className="override-label mono">{label}</span>
      <div className="override-toggle">
        <button
          type="button"
          className={`toggle-btn ${override.mode === 'real' ? 'active' : ''}`}
          onClick={() => setOverrideMode(dataKey, 'real')}
        >
          Use Real TLE
        </button>
        <button
          type="button"
          className={`toggle-btn ${override.mode === 'override' ? 'active' : ''}`}
          onClick={() => setOverrideMode(dataKey, 'override')}
        >
          Override Position
        </button>
      </div>
    </div>

    {override.mode === 'override' && (
      <div className="override-fields">
        <div className="field-row">
          <label>Position X (km)</label>
          <input
            className="field-input"
            value={override.position.x}
            onChange={(e) => setOverrideVectorField(dataKey, 'position', 'x', e.target.value)}
          />
        </div>
        <div className="field-row">
          <label>Position Y (km)</label>
          <input
            className="field-input"
            value={override.position.y}
            onChange={(e) => setOverrideVectorField(dataKey, 'position', 'y', e.target.value)}
          />
        </div>
        <div className="field-row">
          <label>Position Z (km)</label>
          <input
            className="field-input"
            value={override.position.z}
            onChange={(e) => setOverrideVectorField(dataKey, 'position', 'z', e.target.value)}
          />
        </div>
        <div className="field-row">
          <label>Velocity X (km/s)</label>
          <input
            className="field-input"
            value={override.velocity.vx}
            onChange={(e) => setOverrideVectorField(dataKey, 'velocity', 'vx', e.target.value)}
          />
        </div>
        <div className="field-row">
          <label>Velocity Y (km/s)</label>
          <input
            className="field-input"
            value={override.velocity.vy}
            onChange={(e) => setOverrideVectorField(dataKey, 'velocity', 'vy', e.target.value)}
          />
        </div>
        <div className="field-row">
          <label>Velocity Z (km/s)</label>
          <input
            className="field-input"
            value={override.velocity.vz}
            onChange={(e) => setOverrideVectorField(dataKey, 'velocity', 'vz', e.target.value)}
          />
        </div>
        <div className="field-row">
          <label>Epoch (UTC)</label>
          <input
            className="field-input"
            placeholder="YYYY-MM-DDTHH:MM:SS.sssZ"
            value={override.epochUtc}
            onChange={(e) => setOverrideField(dataKey, 'epochUtc', e.target.value)}
          />
        </div>
      </div>
    )}
  </div>
);

const TestModePanel = () => {
  const satellites = useStore((s) => s.satellites);
  const setAlerts = useStore((s) => s.setAlerts);
  const setCascadePlan = useStore((s) => s.setCascadePlan);
  const setHotspots = useStore((s) => s.setHotspots);
  const setDebrisClouds = useStore((s) => s.setDebrisClouds);
  const {
    selectedA,
    selectedB,
    overrideA,
    overrideB,
    prediction,
    sim,
    testActive,
    testSatellites,
    setSelectedA,
    setSelectedB,
    clearSelectedA,
    clearSelectedB,
    setOverrideMode,
    setOverrideField,
    setOverrideVectorField,
    setPredictionField,
    setSimControl,
    setComputed,
    setTestActive,
    setTestSatellites,
  } = useTestMode();

  const [searchQuery, setSearchQuery] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const [showPasteModal, setShowPasteModal] = useState(false);
  const [pasteText, setPasteText] = useState('');
  const [pasteError, setPasteError] = useState('');

  const filteredSatellites = useMemo(() => {
    if (!searchQuery.trim()) return satellites.slice(0, 60);
    const query = searchQuery.trim().toLowerCase();
    return satellites
      .filter((sat) => (
        sat.name?.toLowerCase().includes(query) ||
        String(sat.norad_id).includes(query)
      ))
      .slice(0, 60);
  }, [satellites, searchQuery]);

  const parseNumber = (value) => {
    const parsed = Number(value);
    if (Number.isNaN(parsed)) {
      throw new Error('Invalid numeric value');
    }
    return parsed;
  };

  const buildSatellitePayload = (satellite, override) => {
    if (!satellite) throw new Error('Satellite not selected');

    const payload = {
      name: satellite.name,
      norad_id: String(satellite.norad_id),
      use_real_tle: override.mode !== 'override',
    };

    if (override.mode === 'override') {
      payload.position_eci_km = [
        parseNumber(override.position.x),
        parseNumber(override.position.y),
        parseNumber(override.position.z),
      ];
      payload.velocity_eci_kms = [
        parseNumber(override.velocity.vx),
        parseNumber(override.velocity.vy),
        parseNumber(override.velocity.vz),
      ];
      payload.epoch_utc = override.epochUtc;
    }

    return payload;
  };

  const handleApplySetup = async () => {
    setStatusMessage('');

    try {
      const satelliteA = buildSatellitePayload(selectedA, overrideA);
      const satelliteB = buildSatellitePayload(selectedB, overrideB);
      const epoch = satelliteA.epoch_utc || satelliteB.epoch_utc || '';

      const resp = await fetch(`${API_BASE}/api/test/setup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          satellite_a: satelliteA,
          satellite_b: satelliteB,
          epoch_utc: epoch,
        }),
      });

      if (!resp.ok) {
        throw new Error('Test setup failed');
      }

      setTestActive(true);
      setTestSatellites({
        a: satelliteA.position_eci_km
          ? {
              name: satelliteA.name,
              norad_id: satelliteA.norad_id,
              position: {
                x: satelliteA.position_eci_km[0],
                y: satelliteA.position_eci_km[1],
                z: satelliteA.position_eci_km[2],
              },
              velocity: {
                vx: satelliteA.velocity_eci_kms[0],
                vy: satelliteA.velocity_eci_kms[1],
                vz: satelliteA.velocity_eci_kms[2],
              },
              epochUtc: satelliteA.epoch_utc,
            }
          : {
              name: satelliteA.name,
              norad_id: satelliteA.norad_id,
            },
        b: satelliteB.position_eci_km
          ? {
              name: satelliteB.name,
              norad_id: satelliteB.norad_id,
              position: {
                x: satelliteB.position_eci_km[0],
                y: satelliteB.position_eci_km[1],
                z: satelliteB.position_eci_km[2],
              },
              velocity: {
                vx: satelliteB.velocity_eci_kms[0],
                vy: satelliteB.velocity_eci_kms[1],
                vz: satelliteB.velocity_eci_kms[2],
              },
              epochUtc: satelliteB.epoch_utc,
            }
          : {
              name: satelliteB.name,
              norad_id: satelliteB.norad_id,
            },
      });
      setSimControl({ startEpochUtc: epoch, currentEpochUtc: epoch });
      setStatusMessage('TEST SESSION READY');
    } catch (error) {
      setStatusMessage(error.message || 'Unable to apply setup');
    }
  };

  const handleRunToTca = async () => {
    setStatusMessage('');
    if (!prediction.tcaUtc) {
      setStatusMessage('Predicted TCA is required');
      return;
    }

    try {
      const resp = await fetch(`${API_BASE}/api/test/run-to-tca`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_utc: prediction.tcaUtc,
          step_seconds: sim.stepSeconds,
          max_steps: 10000,
        }),
      });

      if (!resp.ok) {
        throw new Error('Run to TCA failed');
      }

      const data = await resp.json();
      setComputed({
        tcaUtc: data.tca_utc,
        missDistanceM: data.miss_distance_m,
        relVelocityKms: data.relative_velocity_kms,
        collisionDetected: data.collision_detected,
        trajectoryA: data.trajectory_a,
        trajectoryB: data.trajectory_b,
      });
      if (Object.prototype.hasOwnProperty.call(data, 'alerts')) {
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
          cpi_threshold: data.cpi_threshold || 7.0,
          node_probabilities: data.node_probabilities || {},
        });
      }
      if (Object.prototype.hasOwnProperty.call(data, 'debris_clouds')) {
        setDebrisClouds(data.debris_clouds || []);
      }
      setSimControl({ currentEpochUtc: data.tca_utc });
      setTestSatellites({
        a: testSatellites.a
          ? { ...testSatellites.a, position: { x: data.position_a_eci[0], y: data.position_a_eci[1], z: data.position_a_eci[2] } }
          : testSatellites.a,
        b: testSatellites.b
          ? { ...testSatellites.b, position: { x: data.position_b_eci[0], y: data.position_b_eci[1], z: data.position_b_eci[2] } }
          : testSatellites.b,
      });
      setStatusMessage('TCA COMPUTED');
    } catch (error) {
      setStatusMessage(error.message || 'Run to TCA failed');
    }
  };

  const handlePasteParse = () => {
    setPasteError('');
    try {
      const parsed = JSON.parse(pasteText);
      const satA = parsed.satellite_a;
      const satB = parsed.satellite_b;
      if (!satA || !satB) {
        throw new Error('Missing satellite_a or satellite_b data');
      }

      const parsedAId = Number(satA.norad_id);
      const parsedBId = Number(satB.norad_id);
      setSelectedA({ name: satA.name, norad_id: Number.isNaN(parsedAId) ? satA.norad_id : parsedAId });
      setSelectedB({ name: satB.name, norad_id: Number.isNaN(parsedBId) ? satB.norad_id : parsedBId });

      setOverrideMode('overrideA', 'override');
      setOverrideMode('overrideB', 'override');

      setOverrideVectorField('overrideA', 'position', 'x', String(satA.position_eci_km?.[0] ?? ''));
      setOverrideVectorField('overrideA', 'position', 'y', String(satA.position_eci_km?.[1] ?? ''));
      setOverrideVectorField('overrideA', 'position', 'z', String(satA.position_eci_km?.[2] ?? ''));
      setOverrideVectorField('overrideA', 'velocity', 'vx', String(satA.velocity_eci_kms?.[0] ?? ''));
      setOverrideVectorField('overrideA', 'velocity', 'vy', String(satA.velocity_eci_kms?.[1] ?? ''));
      setOverrideVectorField('overrideA', 'velocity', 'vz', String(satA.velocity_eci_kms?.[2] ?? ''));
      setOverrideField('overrideA', 'epochUtc', satA.epoch_utc || '');

      setOverrideVectorField('overrideB', 'position', 'x', String(satB.position_eci_km?.[0] ?? ''));
      setOverrideVectorField('overrideB', 'position', 'y', String(satB.position_eci_km?.[1] ?? ''));
      setOverrideVectorField('overrideB', 'position', 'z', String(satB.position_eci_km?.[2] ?? ''));
      setOverrideVectorField('overrideB', 'velocity', 'vx', String(satB.velocity_eci_kms?.[0] ?? ''));
      setOverrideVectorField('overrideB', 'velocity', 'vy', String(satB.velocity_eci_kms?.[1] ?? ''));
      setOverrideVectorField('overrideB', 'velocity', 'vz', String(satB.velocity_eci_kms?.[2] ?? ''));
      setOverrideField('overrideB', 'epochUtc', satB.epoch_utc || '');

      if (parsed.prediction) {
        setPredictionField('tcaUtc', parsed.prediction.tca_utc || '');
        setPredictionField('missDistanceM', String(parsed.prediction.miss_distance_m ?? ''));
        setPredictionField('relVelocityKms', String(parsed.prediction.relative_velocity_kms ?? ''));
        setPredictionField('collisionProbability', String(parsed.prediction.collision_probability ?? ''));
        setPredictionField('predictedCollision', parsed.prediction.predicted_collision ? 'yes' : 'no');
      }

      setShowPasteModal(false);
    } catch (error) {
      setPasteError(error.message || 'Unable to parse JSON');
    }
  };

  return (
    <>
      <Section title="Satellite Selector">
        <div className="selection-summary">
          <div className="summary-row">
            <span className="summary-label">Satellite A</span>
            <span className="summary-value mono primary">
              {selectedA ? `${selectedA.name || 'SAT A'} (#${selectedA.norad_id})` : 'UNSET'}
            </span>
            <button className="mini-btn" type="button" onClick={clearSelectedA}>Clear</button>
          </div>
          <div className="summary-row">
            <span className="summary-label">Satellite B</span>
            <span className="summary-value mono secondary">
              {selectedB ? `${selectedB.name || 'SAT B'} (#${selectedB.norad_id})` : 'UNSET'}
            </span>
            <button className="mini-btn" type="button" onClick={clearSelectedB}>Clear</button>
          </div>
        </div>

        <input
          className="field-input search-input"
          placeholder="Search name or NORAD ID"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />

        <div className="sat-list">
          {filteredSatellites.map((sat) => {
            const isSelectedA = selectedA?.norad_id === sat.norad_id;
            const isSelectedB = selectedB?.norad_id === sat.norad_id;
            let itemClass = 'sat-list-item';
            if (isSelectedA && isSelectedB) itemClass += ' selected-both';
            else if (isSelectedA) itemClass += ' selected-a';
            else if (isSelectedB) itemClass += ' selected-b';

            return (
              <div key={sat.norad_id} className={itemClass}>
              <div className="sat-list-name">
                <span>{sat.name}</span>
                <span className="mono">#{sat.norad_id}</span>
              </div>
              <div className="sat-list-actions">
                <button
                  className={`mini-btn ${selectedA?.norad_id === sat.norad_id ? 'primary' : ''}`}
                  type="button"
                  onClick={() => setSelectedA({ name: sat.name, norad_id: sat.norad_id })}
                >
                  Set A
                </button>
                <button
                  className={`mini-btn ${selectedB?.norad_id === sat.norad_id ? 'secondary' : ''}`}
                  type="button"
                  onClick={() => setSelectedB({ name: sat.name, norad_id: sat.norad_id })}
                >
                  Set B
                </button>
              </div>
            </div>
            );
          })}
        </div>
      </Section>

      <Section title="Scenario Setup">
        <div className="section-subtitle mono">Initial Conditions Override</div>
        <OverrideForm
          label="Satellite A"
          dataKey="overrideA"
          override={overrideA}
          setOverrideMode={setOverrideMode}
          setOverrideVectorField={setOverrideVectorField}
          setOverrideField={setOverrideField}
        />
        <OverrideForm
          label="Satellite B"
          dataKey="overrideB"
          override={overrideB}
          setOverrideMode={setOverrideMode}
          setOverrideVectorField={setOverrideVectorField}
          setOverrideField={setOverrideField}
        />

        <button className="btn btn-execute" type="button" onClick={handleApplySetup}>
          Apply Setup
        </button>
        <button className="btn" type="button" onClick={() => setShowPasteModal(true)}>
          Paste JSON
        </button>

        <div className="section-divider" />
        <div className="section-subtitle mono">Prediction Model Input</div>
        <div className="field-row">
          <label>Predicted TCA (UTC)</label>
          <input
            className="field-input"
            value={prediction.tcaUtc}
            onChange={(e) => setPredictionField('tcaUtc', e.target.value)}
          />
        </div>
        <div className="field-row">
          <label>Predicted Miss Distance (m)</label>
          <input
            className="field-input"
            value={prediction.missDistanceM}
            onChange={(e) => setPredictionField('missDistanceM', e.target.value)}
          />
        </div>
        <div className="field-row">
          <label>Predicted Rel. Velocity (km/s)</label>
          <input
            className="field-input"
            value={prediction.relVelocityKms}
            onChange={(e) => setPredictionField('relVelocityKms', e.target.value)}
          />
        </div>
        <div className="field-row">
          <label>Collision Probability</label>
          <input
            className="field-input"
            value={prediction.collisionProbability}
            onChange={(e) => setPredictionField('collisionProbability', e.target.value)}
          />
        </div>
        <div className="field-row">
          <label>Predicted Outcome</label>
          <select
            className="field-input"
            value={prediction.predictedCollision}
            onChange={(e) => setPredictionField('predictedCollision', e.target.value)}
          >
            <option value="no">No Collision</option>
            <option value="yes">Collision Occurs</option>
          </select>
        </div>
        <div className="field-row">
          <label>Prediction Model Name</label>
          <input
            className="field-input"
            value={prediction.modelName}
            onChange={(e) => setPredictionField('modelName', e.target.value)}
          />
        </div>
        <div className="field-row">
          <label>Test Case ID</label>
          <input
            className="field-input"
            value={prediction.testCaseId}
            onChange={(e) => setPredictionField('testCaseId', e.target.value)}
          />
        </div>
      </Section>

      <Section title="Simulation Controls">
        <div className="sim-row">
          <div>
            <div className="section-subtitle mono">Start Epoch</div>
            <div className="mono sim-value">{sim.startEpochUtc || '---'}</div>
          </div>
          <div>
            <div className="section-subtitle mono">Current Time</div>
            <div className="mono sim-value">{sim.currentEpochUtc || '---'}</div>
          </div>
        </div>

        <div className="field-row">
          <label>Step Size</label>
          <select
            className="field-input"
            value={sim.stepSeconds}
            onChange={(e) => setSimControl({ stepSeconds: Number(e.target.value) })}
          >
            <option value={1}>1s</option>
            <option value={10}>10s</option>
            <option value={60}>1min</option>
            <option value={300}>5min</option>
          </select>
        </div>

        <div className="btn-row">
          <button
            className="btn btn-execute"
            type="button"
            onClick={handleRunToTca}
            disabled={!testActive}
          >
            Run to TCA
          </button>
          <button className="btn" type="button" disabled>
            Step Forward
          </button>
          <button className="btn" type="button" disabled>
            Step Back
          </button>
        </div>

        <div className="btn-row">
          <button className="btn" type="button" disabled>
            Reset
          </button>
          <button className="btn" type="button" disabled>
            Real-Time
          </button>
        </div>

        <div className="field-row">
          <label>Time Warp</label>
          <select
            className="field-input"
            value={sim.timeWarp}
            onChange={(e) => setSimControl({ timeWarp: Number(e.target.value) })}
          >
            <option value={1}>1x</option>
            <option value={10}>10x</option>
            <option value={60}>60x</option>
            <option value={600}>600x</option>
            <option value={3600}>3600x</option>
          </select>
        </div>

        {statusMessage && (
          <div className="status-banner mono">{statusMessage}</div>
        )}
      </Section>

      {showPasteModal && (
        <div className="modal-overlay" onClick={() => setShowPasteModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Paste Scenario JSON</h2>
            <textarea
              className="field-input textarea"
              rows={10}
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
            />
            {pasteError && <div className="status-banner error mono">{pasteError}</div>}
            <div className="modal-actions">
              <button className="btn" type="button" onClick={() => setShowPasteModal(false)}>
                Cancel
              </button>
              <button className="btn btn-execute" type="button" onClick={handlePasteParse}>
                Parse
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default TestModePanel;
