import { useMemo, useState } from 'react';
import useStore from '../store/useStore';
import useTestMode from '../hooks/useTestMode';
import '../styles/sim.css';

const API_BASE = '';

const STEP_OPTIONS = [
  { value: 1, label: '1s' },
  { value: 10, label: '10s' },
  { value: 60, label: '1m' },
  { value: 300, label: '5m' },
];

const WARP_OPTIONS = [1, 10, 60, 600, 3600];

const OK_MESSAGES = new Set(['TEST SESSION READY', 'TCA COMPUTED']);

const Section = ({ title, defaultOpen = true, children }) => {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className={`sim-section${open ? '' : ' is-collapsed'}`}>
      <button
        type="button"
        className="sim-section-head"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
      >
        <span className="ui-label sim-section-title">{title}</span>
        <span className="sim-chevron" aria-hidden="true" />
      </button>
      {open && <div className="sim-body">{children}</div>}
    </section>
  );
};

const Segmented = ({ options, value, onChange, label }) => (
  <div className="sim-seg" role="group" aria-label={label}>
    {options.map((opt) => (
      <button
        key={String(opt.value)}
        type="button"
        className={`sim-seg-btn${value === opt.value ? ' is-active' : ''}`}
        aria-pressed={value === opt.value}
        onClick={() => onChange(opt.value)}
      >
        {opt.label}
      </button>
    ))}
  </div>
);

const Field = ({ label, children }) => (
  <label className="sim-field">
    <span className="sim-field-label">{label}</span>
    {children}
  </label>
);

const OverrideForm = ({ label, dataKey, override, setOverrideMode, setOverrideVectorField, setOverrideField }) => (
  <div className="sim-object">
    <div className="sim-object-head">
      <span className="sim-object-name">{label}</span>
      <Segmented
        label={`${label} initial conditions`}
        value={override.mode}
        onChange={(mode) => setOverrideMode(dataKey, mode)}
        options={[
          { value: 'real', label: 'Real TLE' },
          { value: 'override', label: 'Override' },
        ]}
      />
    </div>

    {override.mode === 'override' && (
      <div className="sim-fields">
        <div className="sim-vec">
          <span />
          <span className="sim-vec-head">X</span>
          <span className="sim-vec-head">Y</span>
          <span className="sim-vec-head">Z</span>
        </div>
        <div className="sim-vec">
          <span className="sim-field-label">Pos (km)</span>
          <input
            className="ui-input sim-input"
            aria-label={`${label} position X (km)`}
            value={override.position.x}
            onChange={(e) => setOverrideVectorField(dataKey, 'position', 'x', e.target.value)}
          />
          <input
            className="ui-input sim-input"
            aria-label={`${label} position Y (km)`}
            value={override.position.y}
            onChange={(e) => setOverrideVectorField(dataKey, 'position', 'y', e.target.value)}
          />
          <input
            className="ui-input sim-input"
            aria-label={`${label} position Z (km)`}
            value={override.position.z}
            onChange={(e) => setOverrideVectorField(dataKey, 'position', 'z', e.target.value)}
          />
        </div>
        <div className="sim-vec">
          <span className="sim-field-label">Vel (km/s)</span>
          <input
            className="ui-input sim-input"
            aria-label={`${label} velocity X (km/s)`}
            value={override.velocity.vx}
            onChange={(e) => setOverrideVectorField(dataKey, 'velocity', 'vx', e.target.value)}
          />
          <input
            className="ui-input sim-input"
            aria-label={`${label} velocity Y (km/s)`}
            value={override.velocity.vy}
            onChange={(e) => setOverrideVectorField(dataKey, 'velocity', 'vy', e.target.value)}
          />
          <input
            className="ui-input sim-input"
            aria-label={`${label} velocity Z (km/s)`}
            value={override.velocity.vz}
            onChange={(e) => setOverrideVectorField(dataKey, 'velocity', 'vz', e.target.value)}
          />
        </div>
        <Field label="Epoch (UTC)">
          <input
            className="ui-input sim-input"
            placeholder="YYYY-MM-DDTHH:MM:SS.sssZ"
            value={override.epochUtc}
            onChange={(e) => setOverrideField(dataKey, 'epochUtc', e.target.value)}
          />
        </Field>
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
        <div className="sim-pair">
          <div className="sim-pair-row">
            <span className="sim-pair-tag">A</span>
            <span className={`sim-pair-value${selectedA ? '' : ' is-unset'}`}>
              {selectedA ? `${selectedA.name || 'SAT A'} (#${selectedA.norad_id})` : 'UNSET'}
            </span>
            <button className="ui-btn ui-btn--ghost sim-mini" type="button" onClick={clearSelectedA}>Clear</button>
          </div>
          <div className="sim-pair-row">
            <span className="sim-pair-tag">B</span>
            <span className={`sim-pair-value${selectedB ? '' : ' is-unset'}`}>
              {selectedB ? `${selectedB.name || 'SAT B'} (#${selectedB.norad_id})` : 'UNSET'}
            </span>
            <button className="ui-btn ui-btn--ghost sim-mini" type="button" onClick={clearSelectedB}>Clear</button>
          </div>
        </div>

        <input
          className="ui-input sim-input"
          placeholder="Search name or NORAD ID"
          aria-label="Search satellites"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />

        <div className="sim-list">
          {filteredSatellites.map((sat) => {
            const isSelectedA = selectedA?.norad_id === sat.norad_id;
            const isSelectedB = selectedB?.norad_id === sat.norad_id;

            return (
              <div key={sat.norad_id} className={`sim-list-row${isSelectedA || isSelectedB ? ' is-selected' : ''}`}>
                <span className="sim-list-name">{sat.name}</span>
                <span className="sim-list-id">#{sat.norad_id}</span>
                <div className="sim-seg" role="group" aria-label={`Assign ${sat.name}`}>
                  <button
                    className={`sim-seg-btn${isSelectedA ? ' is-active' : ''}`}
                    type="button"
                    aria-pressed={isSelectedA}
                    onClick={() => setSelectedA({ name: sat.name, norad_id: sat.norad_id })}
                  >
                    Set A
                  </button>
                  <button
                    className={`sim-seg-btn${isSelectedB ? ' is-active' : ''}`}
                    type="button"
                    aria-pressed={isSelectedB}
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
        <div className="sim-subhead">
          <span className="ui-label">Initial conditions override</span>
        </div>
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

        <div className="sim-actions">
          <button className="ui-btn ui-btn--primary" type="button" onClick={handleApplySetup}>
            Apply Setup
          </button>
          <button className="ui-btn" type="button" onClick={() => setShowPasteModal(true)}>
            Paste JSON
          </button>
        </div>

        <div className="sim-subhead">
          <span className="ui-label">Prediction model input</span>
        </div>
        <div className="sim-fields">
          <Field label="Predicted TCA (UTC)">
            <input
              className="ui-input sim-input"
              value={prediction.tcaUtc}
              onChange={(e) => setPredictionField('tcaUtc', e.target.value)}
            />
          </Field>
          <Field label="Miss distance (m)">
            <input
              className="ui-input sim-input"
              value={prediction.missDistanceM}
              onChange={(e) => setPredictionField('missDistanceM', e.target.value)}
            />
          </Field>
          <Field label="Rel. velocity (km/s)">
            <input
              className="ui-input sim-input"
              value={prediction.relVelocityKms}
              onChange={(e) => setPredictionField('relVelocityKms', e.target.value)}
            />
          </Field>
          <Field label="Collision probability">
            <input
              className="ui-input sim-input"
              value={prediction.collisionProbability}
              onChange={(e) => setPredictionField('collisionProbability', e.target.value)}
            />
          </Field>
          <div className="sim-field">
            <span className="sim-field-label">Predicted outcome</span>
            <Segmented
              label="Predicted outcome"
              value={prediction.predictedCollision}
              onChange={(v) => setPredictionField('predictedCollision', v)}
              options={[
                { value: 'no', label: 'No collision' },
                { value: 'yes', label: 'Collision' },
              ]}
            />
          </div>
          <Field label="Model name">
            <input
              className="ui-input sim-input"
              value={prediction.modelName}
              onChange={(e) => setPredictionField('modelName', e.target.value)}
            />
          </Field>
          <Field label="Test case ID">
            <input
              className="ui-input sim-input"
              value={prediction.testCaseId}
              onChange={(e) => setPredictionField('testCaseId', e.target.value)}
            />
          </Field>
        </div>
      </Section>

      <Section title="Simulation Controls">
        <div className="sim-readouts">
          <div className="sim-readout">
            <span className="ui-label">Start epoch</span>
            <span className="sim-readout-value">{sim.startEpochUtc || '---'}</span>
          </div>
          <div className="sim-readout">
            <span className="ui-label">Current time</span>
            <span className="sim-readout-value">{sim.currentEpochUtc || '---'}</span>
          </div>
        </div>

        <div className="sim-field">
          <span className="sim-field-label">Step size</span>
          <Segmented
            label="Step size"
            value={sim.stepSeconds}
            onChange={(v) => setSimControl({ stepSeconds: Number(v) })}
            options={STEP_OPTIONS}
          />
        </div>

        <div className="sim-field">
          <span className="sim-field-label">Time warp</span>
          <Segmented
            label="Time warp"
            value={sim.timeWarp}
            onChange={(v) => setSimControl({ timeWarp: Number(v) })}
            options={WARP_OPTIONS.map((w) => ({ value: w, label: `${w}x` }))}
          />
        </div>

        <div className="sim-actions">
          <button
            className="ui-btn ui-btn--primary"
            type="button"
            onClick={handleRunToTca}
            disabled={!testActive}
          >
            Run to TCA
          </button>
          <button className="ui-btn" type="button" disabled>
            Step Fwd
          </button>
          <button className="ui-btn" type="button" disabled>
            Step Back
          </button>
        </div>

        <div className="sim-actions">
          <button className="ui-btn" type="button" disabled>
            Reset
          </button>
          <button className="ui-btn" type="button" disabled>
            Real-Time
          </button>
        </div>

        {statusMessage && (
          <div className={`sim-status ${OK_MESSAGES.has(statusMessage) ? 'is-nominal' : 'is-warning'}`}>
            {statusMessage}
          </div>
        )}
      </Section>

      {showPasteModal && (
        <div className="sim-modal-overlay" onClick={() => setShowPasteModal(false)}>
          <div className="sim-modal" role="dialog" aria-label="Paste scenario JSON" onClick={(e) => e.stopPropagation()}>
            <div className="ui-panel-header">
              <span className="ui-label sim-section-title">Paste Scenario JSON</span>
            </div>
            <div className="sim-modal-body">
              <textarea
                className="ui-input sim-textarea"
                rows={10}
                value={pasteText}
                onChange={(e) => setPasteText(e.target.value)}
              />
              {pasteError && <div className="sim-status is-warning">{pasteError}</div>}
              <div className="sim-modal-actions">
                <button className="ui-btn" type="button" onClick={() => setShowPasteModal(false)}>
                  Cancel
                </button>
                <button className="ui-btn ui-btn--primary" type="button" onClick={handlePasteParse}>
                  Parse
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default TestModePanel;
