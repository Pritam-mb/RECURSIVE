import { useEffect, useState } from 'react';
import '../../styles/uplink.css';

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
      <div className="ul-scrim" onClick={onCancel}>
        <div
          className="ui-panel ul-modal"
          role="dialog"
          aria-modal="true"
          aria-label="Pre-flight check"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="ui-panel-header">
            <span className="ui-label">Pre-Flight Check</span>
          </div>
          <div className="ui-panel-body">
            <div className="ul-empty">RUNNING VALIDATION...</div>
          </div>
        </div>
      </div>
    );
  }

  const gates = preflight?.gates ? Object.entries(preflight.gates) : [];

  return (
    <div className="ul-scrim" onClick={onCancel}>
      <div
        className="ui-panel ul-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Pre-flight validation"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="ui-panel-header">
          <span className="ui-label">Pre-Flight Validation</span>
          <span className={`ui-status ${preflight?.all_clear ? 'is-nominal' : 'is-warning'}`}>
            {preflight?.all_clear ? 'GO' : 'NO-GO'}
          </span>
        </div>

        <div className="ui-panel-body ul-modal-body">
          {error && (
            <div className="ul-banner is-warning">{error}</div>
          )}

          <div className="ul-kv">
            <div className="ul-kv-row">
              <span className="ul-kv-key">ΔV Magnitude</span>
              <span className="ul-kv-val">{preflight?.dv_magnitude_ms}<span className="ul-unit">m/s</span></span>
            </div>
            {preflight?.trajectory && (
              <>
                <div className="ul-kv-row">
                  <span className="ul-kv-key">Nearest miss</span>
                  <span className="ul-kv-val">
                    {preflight.trajectory.min_miss_distance_km != null
                      ? <>{Number(preflight.trajectory.min_miss_distance_km).toFixed(2)}<span className="ul-unit">km</span></>
                      : '—'}
                  </span>
                </div>
                <div className="ul-kv-row">
                  <span className="ul-kv-key">Closest TCA</span>
                  <span className="ul-kv-val">
                    {preflight.trajectory.closest_tca_minutes != null
                      ? <>{Number(preflight.trajectory.closest_tca_minutes).toFixed(1)}<span className="ul-unit">min</span></>
                      : '—'}
                  </span>
                </div>
              </>
            )}
          </div>

          {gates.length > 0 && (
            <div className="ul-checklist">
              {gates.map(([key, passed], idx) => (
                <div className="ul-check-row" key={key}>
                  <span className="ul-check-idx">{String(idx + 1).padStart(2, '0')}</span>
                  <span className="ul-check-name">{GATE_LABELS[key] || key}</span>
                  <span className={`ul-check-state ${passed ? 'is-nominal' : 'is-warning'}`}>
                    {passed ? 'GO' : 'NO-GO'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="ul-modal-footer">
          <button type="button" className="ui-btn" onClick={onCancel}>
            Abort
          </button>
          <button
            type="button"
            className="ui-btn ui-btn--primary"
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
