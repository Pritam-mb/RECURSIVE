import { useMemo, useState } from 'react';
import useStore from '../store/useStore';
import '../styles/sim.css';

const API_BASE = '';

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

const Field = ({ label, children }) => (
  <label className="sim-field">
    <span className="sim-field-label">{label}</span>
    {children}
  </label>
);

// Severity / status → state colour class
const severityClass = (sev) => {
  const s = String(sev || '').toUpperCase();
  if (/CRIT|RED|HIGH/.test(s)) return 'is-warning';
  if (/WARN|YELLOW|MED/.test(s)) return 'is-caution';
  if (/NOMINAL|GREEN|LOW/.test(s)) return 'is-nominal';
  return '';
};

const statusClass = (msg) => {
  if (msg.startsWith('CONJUNCTION PREDICTED')) return 'is-caution';
  if (msg === 'NO CONJUNCTION WITHIN WINDOW') return 'is-nominal';
  return 'is-warning';
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

  // Plain render helper (not a nested component) so inputs keep focus while typing.
  const renderRefForm = ({ label, refData, setRef }) => (
    <div className="sim-object">
      <div className="sim-object-head">
        <span className="sim-object-name">{label}</span>
        <div className="sim-seg" role="group" aria-label={`${label} reference`}>
          <button
            type="button"
            className={`sim-seg-btn${refData.mode === 'norad' ? ' is-active' : ''}`}
            aria-pressed={refData.mode === 'norad'}
            onClick={() => setRefField(setRef, 'mode', 'norad')}
          >
            Catalog
          </button>
          <button
            type="button"
            className={`sim-seg-btn${refData.mode === 'latlon' ? ' is-active' : ''}`}
            aria-pressed={refData.mode === 'latlon'}
            onClick={() => setRefField(setRef, 'mode', 'latlon')}
          >
            Lat/Lon
          </button>
        </div>
      </div>

      {refData.mode === 'norad' && (
        <div className="sim-fields">
          <Field label="Satellite">
            <select
              className="ui-select sim-input"
              value={refData.noradId}
              onChange={(e) => setRefField(setRef, 'noradId', e.target.value)}
            >
              <option value="">Select ...</option>
              {filteredSatellites.map((sat) => (
                <option key={sat.norad_id} value={sat.norad_id}>
                  {sat.name} (#{sat.norad_id})
                </option>
              ))}
            </select>
          </Field>
          <Field label="NORAD ID (override)">
            <input
              className="ui-input sim-input"
              value={refData.noradId}
              onChange={(e) => setRefField(setRef, 'noradId', e.target.value)}
              placeholder="e.g. 25544"
            />
          </Field>
        </div>
      )}

      {refData.mode === 'latlon' && (
        <div className="sim-fields">
          <Field label="Latitude (°)">
            <input
              className="ui-input sim-input"
              value={refData.lat}
              onChange={(e) => setRefField(setRef, 'lat', e.target.value)}
            />
          </Field>
          <Field label="Longitude (°)">
            <input
              className="ui-input sim-input"
              value={refData.lon}
              onChange={(e) => setRefField(setRef, 'lon', e.target.value)}
            />
          </Field>
          <Field label="Altitude (km)">
            <input
              className="ui-input sim-input"
              value={refData.alt}
              onChange={(e) => setRefField(setRef, 'alt', e.target.value)}
            />
          </Field>
          <Field label="Inclination (°)">
            <input
              className="ui-input sim-input"
              value={refData.incl}
              onChange={(e) => setRefField(setRef, 'incl', e.target.value)}
            />
          </Field>
        </div>
      )}
    </div>
  );

  const tca = prediction?.tca || {};
  const geoA = prediction?.hotspots?.[0]?.geodetic;
  const clouds = prediction?.debris_clouds || [];
  const topAlert = prediction?.alerts?.[0];

  return (
    <Section title="Conjunction Predictor">
      {renderRefForm({ label: 'Object A', refData: refA, setRef: setRefA })}
      {renderRefForm({ label: 'Object B', refData: refB, setRef: setRefB })}

      <Field label="Forecast window (h)">
        <input
          className="ui-input sim-input"
          type="number"
          value={forecastHours}
          onChange={(e) => setForecastHours(e.target.value)}
        />
      </Field>

      <div className="sim-actions">
        <button
          className="ui-btn ui-btn--primary"
          type="button"
          onClick={handleRun}
          disabled={loading}
        >
          {loading ? 'Running…' : 'Run Prediction'}
        </button>
      </div>

      {prediction && (
        <>
          <div className="sim-subhead">
            <span className="ui-label">Result</span>
          </div>
          <dl className="sim-dl">
            <dt>Status</dt>
            <dd className={prediction.status === 'conjunction' ? 'is-caution' : 'is-nominal'}>
              {prediction.status.toUpperCase()}
            </dd>
            <dt>TCA</dt>
            <dd>{tca.tca_utc || '---'}</dd>
            <dt>Miss distance</dt>
            <dd>{tca.miss_distance_m != null ? `${tca.miss_distance_m.toLocaleString()} m` : '---'}</dd>
            <dt>Rel. velocity</dt>
            <dd>{tca.relative_velocity_kms != null ? `${tca.relative_velocity_kms} km/s` : '---'}</dd>
            {topAlert && (
              <>
                <dt>P(collision)</dt>
                <dd>{topAlert.p_collision == null ? '—' : `${(topAlert.p_collision * 100).toFixed(6)}%`}</dd>
                <dt>CPI</dt>
                <dd className={severityClass(topAlert.severity)}>
                  {topAlert.cpi_score?.toFixed(1)} ({topAlert.severity})
                </dd>
              </>
            )}
            {geoA && (
              <>
                <dt>Ground region</dt>
                <dd>
                  {geoA.latitude_deg?.toFixed(2)}°, {geoA.longitude_deg?.toFixed(2)}° @ {(geoA.altitude_km ?? 0).toFixed(1)} km
                </dd>
              </>
            )}
          </dl>
        </>
      )}

      {clouds.length > 0 && (
        <Section title={`Affected Region (${clouds.length} cloud${clouds.length > 1 ? 's' : ''})`}>
          {clouds.map((cloud, idx) => (
            <div key={cloud.id || idx} className="sim-cloud">
              <div className="sim-cloud-head">
                {cloud.affected_region_geodetic
                  ? `${cloud.affected_region_geodetic.latitude_deg?.toFixed(2)}°, ${cloud.affected_region_geodetic.longitude_deg?.toFixed(2)}°`
                  : 'Region unavailable'}
              </div>
              <div className="sim-cloud-meta">
                <span>R {(cloud.radius_km_at_tca ?? 0).toFixed(1)} km</span>
                <span>T+{cloud.minutes_to_tca} min</span>
                <span>{cloud.fragment_count} fragments</span>
              </div>
              <div>
                {cloud.affected_satellites?.length ? (
                  cloud.affected_satellites.map((sat) => (
                    <div key={sat.norad_id} className={`sim-cloud-sat sim-risk-${sat.risk_band || 'none'}`}>
                      <span className="sim-dot" />
                      <span className="sim-cloud-sat-name">{sat.name}</span>
                      <span className="sim-cloud-sat-id">#{sat.norad_id}</span>
                      <span className="sim-band">{sat.risk_band}</span>
                    </div>
                  ))
                ) : (
                  <div className="sim-muted">No affected satellites found at TCA</div>
                )}
              </div>
            </div>
          ))}
        </Section>
      )}

      {statusMessage && (
        <div className={`sim-status ${statusClass(statusMessage)}`}>{statusMessage}</div>
      )}
    </Section>
  );
};

export default ConjunctionPredictor;
