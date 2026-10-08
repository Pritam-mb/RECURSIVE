import { useState, memo } from 'react';
import { useTCACountdown } from '../utils/tcaCountdown';
import BPlaneDiagram from './BPlaneDiagram';
import { severityLabel } from '../utils/severity';
import { apiPost } from '../utils/api';
import useStore from '../store/useStore';
import '../styles/threats.css';

// Display state (warning / caution / nominal) from CPI, escalated by an
// explicit backend severity when that is higher.
function rowState(cpi, severity) {
  if (cpi >= 8 || severity === 'CRITICAL') return 'warning';
  if (cpi >= 5 || severity === 'WARNING') return 'caution';
  return 'nominal';
}

function formatPc(pc) {
  if (!pc || pc === 0) return '—';
  const n = Number(pc);
  if (!Number.isFinite(n)) return '—';
  return n.toExponential(1);
}

function ThreatCard({ alert, isSelected, onDecision }) {
  const [expanded, setExpanded] = useState(false);
  const [showDvEditor, setShowDvEditor] = useState(false);
  const [customDv, setCustomDv] = useState(0.1);
  const [submitting, setSubmitting] = useState(false);
  const [decisionError, setDecisionError] = useState(null);
  const addDecisionLogEntry = useStore((s) => s.addDecisionLogEntry);
  const setSelectedAlertId = useStore((s) => s.setSelectedAlertId);

  const cpi = Number(alert.cpi_score ?? 0);
  const severity = severityLabel(alert);
  const state = rowState(cpi, severity);
  const pc = Number(alert.p_collision ?? alert.probability_of_collision ?? 0);
  const tcaHours = Number(alert.tca_hours ?? (Number(alert.tca_minutes ?? NaN) / 60));
  const { formatted, urgent, critical } = useTCACountdown(tcaHours, alert.tca_utc);
  const tcaState = critical ? 'is-warning' : urgent ? 'is-caution' : '';

  const sat1 = alert.sat1 ?? {};
  const sat2 = alert.sat2 ?? {};
  const sat1Name = sat1.name ?? `#${sat1.id}`;
  const sat2Name = sat2.name ?? `#${sat2.id}`;
  const missKm = Number(alert.miss_distance_km ?? 0);
  const relVelKmh = Number(alert.relative_speed_kmh ?? alert.relative_speed_kh ?? 0);
  const relVelKms = (relVelKmh / 3600).toFixed(2);
  const cascadeCount = alert.cascade_depth ?? 0;
  const btKm = Number(alert.bt_km ?? alert.b_plane_bt_km ?? 0);
  const bnKm = Number(alert.bn_km ?? alert.b_plane_bn_km ?? 0);
  const cov = alert.covariance_ellipse ?? {};
  const semiMajorM = Number(cov.a ?? 3000);
  const semiMinorM = Number(cov.b ?? 1000);
  const angleRad = Number(cov.angle ?? 0);
  const hbrKm = 0.010;

  const recManeuver = alert.recommended_maneuver ?? {};
  const newMissKm = Number(recManeuver.new_miss_distance_km ?? missKm * 3);
  const newPc = recManeuver.new_pc_collision ?? null;
  const deltaV = Number(recManeuver.delta_v_ms ?? 0.1);
  const fuelCost = Number(recManeuver.fuel_cost_pct ?? 0.5);

  const handleDecision = async (decision, dv) => {
    const payload = {
      alert_id: alert.id,
      sat1_id: sat1.id,
      sat2_id: sat2.id,
      decision,
      delta_v_ms: dv ?? deltaV,
    };

    setSubmitting(true);
    setDecisionError(null);
    try {
      await apiPost('/api/feedback/maneuver', payload);
    } catch (e) {
      // The decision was not recorded: surface the failure and keep the editor
      // open so the operator can retry or change course.
      setDecisionError(e?.message || 'Failed to submit decision');
      setSubmitting(false);
      return;
    }

    addDecisionLogEntry({
      time: new Date().toISOString(),
      decision,
      alert_id: alert.id,
      sat: sat1.name ?? 'unknown',
    });
    setSubmitting(false);
    onDecision?.(decision, alert);
    setShowDvEditor(false);
  };

  const handleCardClick = () => {
    setExpanded((v) => !v);
    setSelectedAlertId(isSelected ? null : alert.id);
  };

  const handleHeadKey = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleCardClick();
    }
  };

  return (
    <div
      className={`tq-row is-${state}${isSelected ? ' is-selected' : ''}${expanded ? ' is-expanded' : ''}`}
      id={`threat-card-${alert.id ?? 'unknown'}`}
    >
      {/* Collapsed summary — always visible */}
      <div
        className="tq-row-head"
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        onClick={handleCardClick}
        onKeyDown={handleHeadKey}
      >
        <div className="tq-row-top">
          <span className="tq-pair" title={`${sat1Name} / ${sat2Name}`}>
            {sat1Name}
            <span className="tq-pair-sep">/</span>
            {sat2Name}
          </span>
          <span className={`tq-sev is-${state}`}>{severity}</span>
          <span className="tq-chev" aria-hidden="true">{expanded ? '−' : '+'}</span>
        </div>

        <div className="tq-row-sub">
          {(sat1.agency || sat2.agency) && (
            <span className="tq-agency">{sat1.agency ?? '—'} / {sat2.agency ?? '—'}</span>
          )}
          {alert.tca_utc && (
            <span className="tq-utc">
              TCA {alert.tca_utc.replace('T', ' ').substring(0, 19)}Z
            </span>
          )}
        </div>

        <div className="tq-metrics">
          <span className={`tq-tca ${tcaState}`}>{formatted}</span>
          <span className="tq-num">{missKm.toFixed(1)}</span>
          <span className="tq-num">{formatPc(pc)}</span>
          <span className="tq-num">{cascadeCount}</span>
          <span className={`tq-num tq-cpi is-${state}`}>{cpi.toFixed(1)}</span>
        </div>

        <div className={`tq-meter is-${state}`} aria-hidden="true">
          <div className="tq-meter-fill" style={{ width: `${Math.min(Math.max(cpi, 0) / 10, 1) * 100}%` }} />
        </div>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="tq-detail">
          <div className="tq-bplane">
            <span className="ui-label tq-section-label">B-Plane Geometry</span>
            <BPlaneDiagram
              btKm={btKm}
              bnKm={bnKm}
              semiMajorM={semiMajorM}
              semiMinorM={semiMinorM}
              angleRad={angleRad}
              hbrKm={hbrKm}
            />
            <div className="tq-bplane-caption">
              σa {semiMajorM.toFixed(0)} m · σb {semiMinorM.toFixed(0)} m
            </div>
          </div>

          <div>
            <span className="ui-label tq-section-label">Encounter</span>
            <dl className="tq-dl">
              <dt>TCA (est)</dt><dd>{tcaHours.toFixed(1)} h</dd>
              <dt>Miss distance</dt><dd>{missKm.toFixed(3)} km</dd>
              <dt>Rel velocity</dt><dd>{relVelKms} km/s</dd>
              <dt>Combined HBR</dt><dd>{hbrKm.toFixed(3)} km</dd>
              <dt>B-plane Bt</dt><dd>{btKm.toFixed(3)} km</dd>
              <dt>B-plane Bn</dt><dd>{bnKm.toFixed(3)} km</dd>
              <dt>Method</dt><dd>Foster 1992</dd>
            </dl>
          </div>

          <div>
            <span className="ui-label tq-section-label">Recommended Maneuver</span>
            <dl className="tq-dl">
              <dt>Satellite</dt><dd title={sat1Name}>{sat1Name}</dd>
              <dt>Delta-V</dt><dd>{deltaV.toFixed(2)} m/s prograde</dd>
              <dt>Fuel cost</dt><dd>{fuelCost.toFixed(1)} %</dd>
              <dt>New miss</dt><dd className="is-nominal">{newMissKm.toFixed(2)} km</dd>
              {newPc != null && (
                <>
                  <dt>New Pc</dt><dd className="is-nominal">{formatPc(newPc)}</dd>
                </>
              )}
            </dl>
          </div>

          {cascadeCount > 0 && (
            <div>
              <span className="ui-label tq-section-label">Cascade Impact</span>
              <div className="tq-note">
                Maneuver affects {cascadeCount} other satellite{cascadeCount !== 1 ? 's' : ''}
              </div>
            </div>
          )}

          {showDvEditor && (
            <div className="tq-dv">
              <label className="ui-label" htmlFor={`tq-dv-${alert.id ?? 'unknown'}`}>Custom Delta-V (m/s)</label>
              <div className="tq-dv-row">
                <input
                  id={`tq-dv-${alert.id ?? 'unknown'}`}
                  className="tq-dv-range"
                  type="range"
                  min={0.01}
                  max={2.0}
                  step={0.01}
                  value={customDv}
                  onChange={(e) => setCustomDv(Number(e.target.value))}
                />
                <span className="tq-dv-value">{customDv.toFixed(2)} m/s</span>
              </div>
              <button
                type="button"
                className="ui-btn ui-btn--primary"
                disabled={submitting}
                onClick={() => handleDecision('MODIFY', customDv)}
              >
                {submitting ? 'Submitting…' : 'Execute Modified Maneuver'}
              </button>
            </div>
          )}

          {decisionError && <div className="tq-error" role="alert">{decisionError}</div>}

          <div className="tq-actions">
            <button
              type="button"
              className="ui-btn ui-btn--primary"
              disabled={submitting}
              onClick={() => handleDecision('APPROVE')}
            >
              Approve
            </button>
            <button
              type="button"
              className="ui-btn"
              aria-pressed={showDvEditor}
              disabled={submitting}
              onClick={() => setShowDvEditor((v) => !v)}
            >
              Modify
            </button>
            <button
              type="button"
              className="ui-btn ui-btn--danger"
              disabled={submitting}
              onClick={() => handleDecision('REJECT')}
            >
              Reject
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default memo(ThreatCard);
