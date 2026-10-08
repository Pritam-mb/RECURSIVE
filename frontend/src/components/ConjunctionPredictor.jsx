import { useMemo, useState } from 'react';
import useStore from '../store/useStore';

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

const defaultRefA = { mode: 'norad', noradId: '', lat: '25.0', lon: '55.0', alt: '550', incl: '51.6' };
const defaultRefB = { mode: 'latlon', noradId: '', lat: '25.0', lon: '57.0', alt: '550', incl: '51.6' };

const ConjunctionPredictor = () => {
  const satellites = useStore((s) => s.satellites);
  const setAlerts = useStore((s) => s.setAlerts);
  const setCascadePlan = useStore((s) => s.setCascadePlan);
  const setHotspots = useStore((s) => s.setHotspots);
  const setDebrisClouds = useStore((s) => s.setDebrisClouds);

  const [refA, setRefA] = useState(defaultRefA);
  const [refB, setRefB] = useState(defaultRefB);
  const [forecastHours, setForecastHours] = useState('24');
  const [statusMessage, setStatusMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);

  const filteredSatellites = useMemo(() => satellites.slice(0, 80), [satellites]);

  const parseNumber = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isNaN(parsed) ? fallback : parsed;
  };

  const buildSatellitePayload = (ref) => {
    if (ref.mode === 'norad') {
      if (!ref.noradId) throw new Error('NORAD ID is required');
      return { norad_id: parseNumber(ref.noradId, -1), name: ref.noradName || undefined };
    }
    if (ref.mode === 'latlon') {
      return {
        name: ref.geoName || undefined,
        latitude_deg: parseNumber(ref.lat),
        longitude_deg: parseNumber(ref.lon),
        altitude_km: parseNumber(ref.alt),
        inclination_deg: parseNumber(ref.incl, 51.6),
      };
    }
    throw new Error('Invalid reference mode');
  };

  const handleRun = async () => {
    setStatusMessage('');
    setLoading(true);
    try {
      const satelliteA = buildSatellitePayload(refA);
      const satelliteB = buildSatellitePayload(refB);

      const resp = await fetch(`${API_BASE}/api/predict/conjunction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          satellite_a: satelliteA,
          satellite_b: satelliteB,
          forecast_hours: parseNumber(forecastHours, 24),
          step_seconds: 60,
          max_steps: 2400,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || 'Prediction failed');
      }

      const data = await resp.json();
      setPrediction(data);

      if (Array.isArray(data.alerts)) setAlerts(data.alerts);
      if (Array.isArray(data.hotspots)) setHotspots(data.hotspots);
      if (Array.isArray(data.debris_clouds)) setDebrisClouds(data.debris_clouds);
      if (data.cascade_plan && data.graph) {
        setCascadePlan({
          graph: data.graph || {},
          alerts: data.alerts || [],
          cascade_plan: data.cascade_plan || [],
          total_delta_v_ms: data.total_delta_v_ms || 0,
          cascade_depth: data.cascade_depth || 0,
          agencies_involved: data.agencies_involved || [],
          seed_satellites: data.seed_satellites || [],
          cpi_threshold: data.cpi_threshold || 5.0,
          node_probabilities: data.node_probabilities || {},
        });
      }

      setStatusMessage(
        data.status === 'conjunction'
          ? `CONJUNCTION PREDICTED — TCA ${data.tca.tca_utc}`
          : 'NO CONJUNCTION WITHIN WINDOW'
      );
    } catch (error) {
      setStatusMessage(error.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const setRefField = (setRef, key, value) => setRef((prev) => ({ ...prev, [key]: value }));

  const RefForm = ({ label, ref, setRef, accent }) => (
    <div className="override-block">
      <div className="override-header">
        <span className={`override-label mono ${accent}`}>{label}</span>
        <div className="override-toggle">
          <button
            type="button"
            className={`toggle-btn ${ref.mode === 'norad' ? 'active' : ''}`}
            onClick={() => setRefField(setRef, 'mode', 'norad')}
          >
            Catalog
          </button>
          <button
            type="button"
            className={`toggle-btn ${ref.mode === 'latlon' ? 'active' : ''}`}
            onClick={() => setRefField(setRef, 'mode', 'latlon')}
          >
            Lat/Lon
          </button>
        </div>
      </div>

      {ref.mode === 'norad' && (
        <div className="override-fields">
          <div className="field-row">
            <label>Satellite</label>
            <select
              className="field-input"
              value={ref.noradId}
              onChange={(e) => setRefField(setRef, 'noradId', e.target.value)}
            >
              <option value="">Select ...</option>
              {filteredSatellites.map((sat) => (
                <option key={sat.norad_id} value={sat.norad_id}>
                  {sat.name} (#{sat.norad_id})
                </option>
              ))}
            </select>
          </div>
          <div className="field-row">
            <label>NORAD ID (override)</label>
            <input
              className="field-input mono"
              value={ref.noradId}
              onChange={(e) => setRefField(setRef, 'noradId', e.target.value)}
              placeholder="e.g. 25544"
            />
          </div>
        </div>
      )}

      {ref.mode === 'latlon' && (
        <div className="override-fields">
          <div className="field-row">
            <label>Latitude (°)</label>
            <input
              className="field-input mono"
              value={ref.lat}
              onChange={(e) => setRefField(setRef, 'lat', e.target.value)}
            />
          </div>
          <div className="field-row">
            <label>Longitude (°)</label>
            <input
              className="field-input mono"
              value={ref.lon}
              onChange={(e) => setRefField(setRef, 'lon', e.target.value)}
            />
          </div>
          <div className="field-row">
            <label>Altitude (km)</label>
            <input
              className="field-input mono"
              value={ref.alt}
              onChange={(e) => setRefField(setRef, 'alt', e.target.value)}
            />
          </div>
          <div className="field-row">
            <label>Inclination (°)</label>
            <input
              className="field-input mono"
              value={ref.incl}
              onChange={(e) => setRefField(setRef, 'incl', e.target.value)}
            />
          </div>
        </div>
      )}
    </div>
  );

  const tca = prediction?.tca || {};
  const geoA = prediction?.hotspots?.[0]?.geodetic;
  const clouds = prediction?.debris_clouds || [];

  return (
    <Section title="Conjunction Predictor">
      <div className="section-subtitle mono">Object A</div>
      <RefForm label="Object A" ref={refA} setRef={setRefA} accent="primary" />
      <div className="section-subtitle mono">Object B</div>
      <RefForm label="Object B" ref={refB} setRef={setRefB} accent="secondary" />

      <div className="field-row">
        <label>Forecast Window (hours)</label>
        <input
          className="field-input mono"
          type="number"
          value={forecastHours}
          onChange={(e) => setForecastHours(e.target.value)}
        />
      </div>

      <button
        className="btn btn-execute"
        type="button"
        onClick={handleRun}
        disabled={loading}
      >
        {loading ? 'RUNNING ...' : 'RUN PREDICTION'}
      </button>

      {prediction && (
        <div className="prediction-readout">
          <div className="readout-row">
            <span className="readout-label">Status</span>
            <span className={`readout-value mono ${prediction.status === 'conjunction' ? 'danger' : 'ok'}`}>
              {prediction.status.toUpperCase()}
            </span>
          </div>
          <div className="readout-row">
            <span className="readout-label">TCA</span>
            <span className="readout-value mono">{tca.tca_utc || '---'}</span>
          </div>
          <div className="readout-row">
            <span className="readout-label">Miss Distance</span>
            <span className="readout-value mono">
              {tca.miss_distance_m != null ? `${tca.miss_distance_m.toLocaleString()} m` : '---'}
            </span>
          </div>
          <div className="readout-row">
            <span className="readout-label">Rel. Velocity</span>
            <span className="readout-value mono">
              {tca.relative_velocity_kms != null ? `${tca.relative_velocity_kms} km/s` : '---'}
            </span>
          </div>
          {prediction.alerts?.[0] && (
            <>
              <div className="readout-row">
                <span className="readout-label">P(collision)</span>
                <span className="readout-value mono">{(prediction.alerts[0].p_collision * 100).toFixed(6)}%</span>
              </div>
              <div className="readout-row">
                <span className="readout-label">CPI</span>
                <span className={`readout-value mono ${prediction.alerts[0].severity}`}>
                  {prediction.alerts[0].cpi_score?.toFixed(1)} ({prediction.alerts[0].severity})
                </span>
              </div>
            </>
          )}
          {geoA && (
            <div className="readout-row">
              <span className="readout-label">Ground Region</span>
              <span className="readout-value mono">
                {geoA.latitude_deg?.toFixed(2)}°, {geoA.longitude_deg?.toFixed(2)}° @ {(geoA.altitude_km ?? 0).toFixed(1)} km
              </span>
            </div>
          )}
        </div>
      )}

      {clouds.length > 0 && (
        <Section title={`Affected Region (${clouds.length} cloud${clouds.length > 1 ? 's' : ''})`}>
          {clouds.map((cloud, idx) => (
            <div key={cloud.id || idx} className="cloud-card">
              <div className="cloud-header mono">
                {cloud.affected_region_geodetic
                  ? `${cloud.affected_region_geodetic.latitude_deg?.toFixed(2)}°, ${cloud.affected_region_geodetic.longitude_deg?.toFixed(2)}°`
                  : 'Region unavailable'}
              </div>
              <div className="cloud-meta">
                <span>radius: {(cloud.radius_km_at_tca ?? 0).toFixed(1)} km</span>
                <span>T+{cloud.minutes_to_tca} min</span>
                <span>{cloud.fragment_count} fragments</span>
              </div>
              <div className="cloud-sat-list">
                {cloud.affected_satellites?.length ? (
                  cloud.affected_satellites.map((sat) => (
                    <div key={sat.norad_id} className="cloud-sat">
                      <span className={`risk-dot ${sat.risk_band}`} />
                      <span className="cloud-sat-name">{sat.name}</span>
                      <span className="zero mono">#{sat.norad_id}</span>
                      <span className={`cloud-risk ${sat.risk_band}`}>{sat.risk_band}</span>
                    </div>
                  ))
                ) : (
                  <div className="muted">No affected satellites found at TCA</div>
                )}
              </div>
            </div>
          ))}
        </Section>
      )}

      {statusMessage && (
        <div className="status-banner mono">{statusMessage}</div>
      )}
    </Section>
  );
};

export default ConjunctionPredictor;