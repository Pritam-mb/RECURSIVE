import { useEffect, useState } from 'react';

const GATE_LABELS = {
  trajectory_clear: 'Trajectory Clear (24h)',
  fuel_budget: 'Fuel Budget',
  agency_auth: 'Agency Authorization',
  tca_window: 'TCA Window Clear',
  physical_limits: 'Physical Limits',
  sim_dryrun: 'Simulation Dry-Run',
};

const PreflightModal = ({ noradId, maneuver, onConfirm, onCancel }) => {
  const apiBaseUrl = '';
  const [preflight, setPreflight] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const runCheck = async () => {
      try {
        const resp = await fetch(`${apiBaseUrl}/api/preflight`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            norad_id: noradId,
            dvx: maneuver.dvx,
            dvy: maneuver.dvy,
            dvz: maneuver.dvz,
          }),
        });

        if (!resp.ok) {
          throw new Error(`preflight request failed with status ${resp.status}`);
        }

        const data = await resp.json();
        if (cancelled) return;
        setPreflight(data);
        setError(null);
      } catch (e) {
        console.error('Preflight check failed:', e);
        if (cancelled) return;
        // Fail closed: an unreachable validator must never enable a burn.
        setError(
          e?.message
            ? `Preflight validation unavailable: ${e.message}`
            : 'Preflight validation unavailable.'
        );
        setPreflight({
          dv_magnitude_ms: Math.sqrt(
            maneuver.dvx ** 2 + maneuver.dvy ** 2 + maneuver.dvz ** 2
          ).toFixed(3),
          gates: {
            trajectory_clear: false,
            fuel_budget: false,
            agency_auth: false,
            tca_window: false,
            physical_limits: false,
            sim_dryrun: false,
          },
          all_clear: false,
        });
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    runCheck();
    return () => {
      cancelled = true;
    };
  }, [noradId, maneuver]);

  if (loading) {
    return (
      <div className="modal-overlay" onClick={onCancel}>
        <div className="modal" onClick={(e) => e.stopPropagation()}>
          <h2>Pre-Flight Check</h2>
          <div className="no-data">RUNNING VALIDATION...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>Pre-Flight Validation</h2>

        {error && (
          <div className="error-banner" style={{ marginBottom: 12 }}>
            {error}
          </div>
        )}

        <div style={{ marginBottom: 12 }}>
          <span className="telemetry-label">ΔV Magnitude</span>
          <span className="mono" style={{ marginLeft: 8, color: 'var(--text-bright)' }}>
            {preflight?.dv_magnitude_ms} m/s
          </span>
        </div>

        {preflight?.trajectory && (
          <div style={{ marginBottom: 12, fontSize: 11, color: 'var(--text-dim)' }}>
            <div>
              Nearest miss: <span className="mono" style={{ color: 'var(--text-bright)' }}>
                {preflight.trajectory.min_miss_distance_km != null
                  ? `${Number(preflight.trajectory.min_miss_distance_km).toFixed(2)} km`
                  : '—'}
              </span>
            </div>
            <div>
              Closest TCA: <span className="mono" style={{ color: 'var(--text-bright)' }}>
                {preflight.trajectory.closest_tca_minutes != null
                  ? `${Number(preflight.trajectory.closest_tca_minutes).toFixed(1)} min`
                  : '—'}
              </span>
            </div>
          </div>
        )}

        {preflight?.gates &&
          Object.entries(preflight.gates).map(([key, passed]) => (
            <div className="gate-item" key={key}>
              <span style={{ color: 'var(--text)' }}>
                {GATE_LABELS[key] || key}
              </span>
              <span className={passed ? 'gate-pass' : 'gate-fail'}>
                {passed ? '✓ PASS' : '✗ FAIL'}
              </span>
            </div>
          ))}

        <div className="modal-actions">
          <button className="btn" onClick={onCancel}>
            Abort
          </button>
          <button
            className="btn btn-execute"
            onClick={onConfirm}
            disabled={!preflight?.all_clear}
          >
            Confirm Burn
          </button>
        </div>
      </div>
    </div>
  );
};

export default PreflightModal;
