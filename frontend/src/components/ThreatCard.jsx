import { useState, memo } from 'react';
import { formatTCA } from '../utils/tcaCountdown';
import BPlaneDiagram from './BPlaneDiagram';
import { severityLabel, severityState } from '../utils/severity';
import { apiPost } from '../utils/api';
import useStore from '../store/useStore';
import '../styles/threats.css';

// Every value on this card comes from the alert payload. When the backend did
// not compute a field we print "—" (or "not computed"), never a stand-in number.

const DASH = '—';

const PC_METHOD_LABEL = {
  foster: 'Foster 2D',
  chan: 'Chan series',
  max_pc: 'Max Pc (Alfano)',
};

/** Finite number or null (null/undefined/''/NaN all mean "not provided"). */
function num(v) {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function fmt(v, digits, unit = '') {
  const n = num(v);
  return n == null ? DASH : `${n.toFixed(digits)}${unit}`;
}

function formatPc(pc) {
  const n = num(pc);
  if (n == null) return DASH;
  if (n === 0) return '0';
  return n.toExponential(1);
}

function formatRsw(vec) {
  if (!Array.isArray(vec) || vec.length !== 3 || vec.some((c) => num(c) == null)) return null;
  const [r, s, w] = vec.map(Number);
  return `R ${r.toFixed(3)} · S ${s.toFixed(3)} · W ${w.toFixed(3)}`;
}

function ThreatCard({ alert, isSelected, onDecision }) {
  const [expanded, setExpanded] = useState(false);
  const [showDvEditor, setShowDvEditor] = useState(false);
  const [customDv, setCustomDv] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [decisionError, setDecisionError] = useState(null);
  const addDecisionLogEntry = useStore((s) => s.addDecisionLogEntry);
  const setSelectedAlertId = useStore((s) => s.setSelectedAlertId);
  // Simulation clock: the timestamp of the latest propagated frame.
  const snapshotTimestamp = useStore((s) => s.snapshotTimestamp);

  const severity = severityLabel(alert);
  const state = severityState(severity);
  const cpi = num(alert.cpi_score);

  // ── Probability of collision (physics) + ML surrogate (secondary) ─────────
  const pc = num(alert.probability_of_collision ?? alert.p_collision);
  const pcMethod = alert.pc_method ? (PC_METHOD_LABEL[alert.pc_method] ?? String(alert.pc_method)) : null;
  const ml = alert.ml && typeof alert.ml === 'object' ? alert.ml : null;
  const mlPc = num(ml?.pc_surrogate);
  const mlAgreement = num(ml?.agreement);

  // ── Time to closest approach, measured on the simulation clock ────────────
  const tcaMs = alert.tca_utc ? Date.parse(alert.tca_utc) : NaN;
  const simNowMs = snapshotTimestamp ? Date.parse(snapshotTimestamp) : NaN;
  let tcaHours = null;
  if (Number.isFinite(tcaMs) && Number.isFinite(simNowMs)) {
    tcaHours = Math.max(0, (tcaMs - simNowMs) / 3_600_000);
  } else if (num(alert.tca_hours) != null) {
    tcaHours = num(alert.tca_hours);
  } else if (num(alert.tca_minutes) != null) {
    tcaHours = num(alert.tca_minutes) / 60;
  }
  const { formatted, urgent, critical } = formatTCA(tcaHours);
  // Time pressure only colours the timer when the conjunction itself is risky.
  const tcaState = state === 'nominal' ? '' : critical ? 'is-warning' : urgent ? 'is-caution' : '';
  const tcaText = tcaHours == null ? DASH : formatted === '00:00:00' ? 'NOW' : formatted;

  // ── Objects ────────────────────────────────────────────────────────────────
  const sat1 = alert.sat1 ?? {};
  const sat2 = alert.sat2 ?? {};
  const sat1Name = sat1.name ?? (sat1.id != null ? `#${sat1.id}` : DASH);
  const sat2Name = sat2.name ?? (sat2.id != null ? `#${sat2.id}` : DASH);
  const isDebris = alert.source === 'debris';
  const parentEvent = alert.parent_event && typeof alert.parent_event === 'object' ? alert.parent_event : null;

  // ── Encounter geometry ────────────────────────────────────────────────────
  const missKm = num(alert.miss_distance_km);
  const relVelKms = num(alert.relative_speed_kms)
    ?? (num(alert.relative_speed_kmh) != null ? num(alert.relative_speed_kmh) / 3600 : null);
  const btKm = num(alert.b_t_km ?? alert.bt_km);
  const bnKm = num(alert.b_n_km ?? alert.bn_km);
  const cov = alert.covariance_ellipse ?? {};
  const semiMajorM = num(cov.a);
  const semiMinorM = num(cov.b);
  const angleRad = num(cov.angle);
  const hbrKm = num(alert.hbr_km);
  const hasBPlane = [btKm, bnKm, semiMajorM, semiMinorM, hbrKm].every((v) => v != null)
    && semiMajorM > 0 && semiMinorM > 0;

  // ── Cascade ───────────────────────────────────────────────────────────────
  const cascadeDepth = num(alert.cascade_depth);
  const downstream = Array.isArray(alert.downstream_ids) ? alert.downstream_ids : null;

  // ── Recommended manoeuvre (computed by re-propagation on the backend) ─────
  const rec = alert.recommended_maneuver && typeof alert.recommended_maneuver === 'object'
    ? alert.recommended_maneuver
    : null;
  const recDv = num(rec?.delta_v_ms);
  const recRsw = Array.isArray(rec?.delta_v_rsw_ms) ? rec.delta_v_rsw_ms.map(Number) : null;
  const recRswText = formatRsw(rec?.delta_v_rsw_ms);
  const maneuverSatId = rec?.sat_id ?? sat1.id;
  const maneuverSat = String(maneuverSatId) === String(sat2.id) ? sat2 : sat1;
  const otherSat = maneuverSat === sat1 ? sat2 : sat1;
  const maneuverSatName = maneuverSat.name ?? (maneuverSatId != null ? `#${maneuverSatId}` : DASH);
  const verified = rec?.verified_by === 'repropagation';
  const canApprove = rec != null && recDv != null && maneuverSatId != null;
  const dvValue = customDv ?? recDv ?? 0.1;

  const handleDecision = async (decision, dvOverride) => {
    let deltaV = 0;
    let rsw = [0, 0, 0];
    if (decision === 'APPROVE') {
      deltaV = recDv;
      rsw = recRsw ?? [0, recDv, 0];
    } else if (decision === 'MODIFY') {
      deltaV = dvOverride;
      // Keep the computed burn direction, rescaled; prograde if none computed.
      const norm = recRsw ? Math.hypot(...recRsw) : 0;
      rsw = norm > 0 ? recRsw.map((c) => (c / norm) * dvOverride) : [0, dvOverride, 0];
    }
    const payload = {
      alert_id: alert.id,
      // The manoeuvring satellite goes first: the handler burns sat1_id.
      sat1_id: maneuverSatId,
      sat2_id: otherSat.id,
      sat_id: maneuverSatId,
      decision,
      delta_v_ms: deltaV,
      delta_v_rsw_ms: rsw,
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
      sat: maneuverSat.name ?? 'unknown',
      delta_v_ms: deltaV,
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
          <span
            className={`tq-src${isDebris ? ' is-debris' : ''}`}
            title={isDebris && parentEvent
              ? `Fragment of event ${parentEvent.event_id ?? DASH}`
              : 'Found by future-window screening'}
          >
            {isDebris ? 'DEBRIS' : 'SCREENING'}
          </span>
          <span className={`tq-sev is-${state}`}>{severity}</span>
          <span className="tq-chev" aria-hidden="true">{expanded ? '−' : '+'}</span>
        </div>

        <div className="tq-row-sub">
          {(sat1.agency || sat2.agency) && (
            <span className="tq-agency">{sat1.agency ?? DASH} / {sat2.agency ?? DASH}</span>
          )}
          {alert.tca_utc && (
            <span className="tq-utc">
              TCA {String(alert.tca_utc).replace('T', ' ').substring(0, 19)}Z
            </span>
          )}
        </div>

        <div className="tq-metrics">
          <span className={`tq-tca ${tcaState}`}>{tcaText}</span>
          <span className="tq-num">{fmt(missKm, 1)}</span>
          <span className="tq-num" title={pcMethod ? `Physics Pc · ${pcMethod}` : 'Physics Pc'}>{formatPc(pc)}</span>
          <span className="tq-num">{cascadeDepth == null ? DASH : cascadeDepth}</span>
          <span className={`tq-num tq-cpi is-${state}`}>{fmt(cpi, 1)}</span>
        </div>

        <div className={`tq-meter is-${state}`} aria-hidden="true">
          <div className="tq-meter-fill" style={{ width: `${Math.min(Math.max(cpi ?? 0, 0) / 10, 1) * 100}%` }} />
        </div>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="tq-detail">
          <div className="tq-bplane">
            <span className="ui-label tq-section-label">B-Plane Geometry</span>
            {hasBPlane ? (
              <>
                <BPlaneDiagram
                  btKm={btKm}
                  bnKm={bnKm}
                  semiMajorM={semiMajorM}
                  semiMinorM={semiMinorM}
                  angleRad={angleRad ?? 0}
                  hbrKm={hbrKm}
                />
                <div className="tq-bplane-caption">
                  σa {semiMajorM.toFixed(0)} m · σb {semiMinorM.toFixed(0)} m
                  {alert.sigma_source ? ` · σ from ${alert.sigma_source}` : ''}
                </div>
              </>
            ) : (
              <div className="tq-note">B-plane geometry not computed for this pair</div>
            )}
          </div>

          <div>
            <span className="ui-label tq-section-label">Encounter</span>
            <dl className="tq-dl">
              <dt>Source</dt>
              <dd>{isDebris ? 'Debris fragment' : 'Screening'}</dd>
              {isDebris && (
                <>
                  <dt>Parent event</dt>
                  <dd title={parentEvent?.collision_utc ?? ''}>{parentEvent?.event_id ?? DASH}</dd>
                  <dt>Parents</dt>
                  <dd>{Array.isArray(parentEvent?.parent_ids) ? parentEvent.parent_ids.join(' × ') : DASH}</dd>
                  <dt>Fragments</dt>
                  <dd>{num(parentEvent?.fragment_count) ?? DASH}</dd>
                </>
              )}
              <dt>TCA (sim)</dt><dd>{tcaHours == null ? DASH : `T-${tcaHours.toFixed(2)} h`}</dd>
              <dt>Miss distance</dt><dd>{fmt(missKm, 3, ' km')}</dd>
              <dt>Rel velocity</dt><dd>{fmt(relVelKms, 2, ' km/s')}</dd>
              <dt>Pc (physics)</dt><dd>{formatPc(pc)}</dd>
              <dt>Pc method</dt><dd>{pcMethod ?? DASH}</dd>
              <dt>ML surrogate</dt>
              <dd className="tq-dd-secondary" title={ml?.model ? `Model: ${ml.model}` : 'ML scoring not attached'}>
                {ml
                  ? `${formatPc(mlPc)}${ml.risk_class ? ` · ${ml.risk_class}` : ''}${mlAgreement != null ? ` · agree ${(mlAgreement * 100).toFixed(0)}%` : ''}`
                  : 'not computed'}
              </dd>
              <dt>σ source</dt><dd>{alert.sigma_source ?? DASH}</dd>
              <dt>Combined HBR</dt><dd>{hbrKm == null ? DASH : `${(hbrKm * 1000).toFixed(1)} m`}</dd>
              <dt>B-plane Bt</dt><dd>{fmt(btKm, 3, ' km')}</dd>
              <dt>B-plane Bn</dt><dd>{fmt(bnKm, 3, ' km')}</dd>
            </dl>
          </div>

          <div>
            <span className="ui-label tq-section-label">Recommended Maneuver</span>
            {rec ? (
              <>
                <dl className="tq-dl">
                  <dt>Satellite</dt><dd title={maneuverSatName}>{maneuverSatName}</dd>
                  <dt>Δv RSW (m/s)</dt><dd title={recRswText ?? ''}>{recRswText ?? DASH}</dd>
                  <dt>|Δv|</dt><dd>{fmt(recDv, 3, ' m/s')}</dd>
                  <dt>Fuel cost</dt>
                  <dd title={rec.fuel_model ? `Model: ${rec.fuel_model}` : ''}>{fmt(rec.fuel_cost_pct, 2, ' %')}</dd>
                  <dt>New miss</dt><dd className="is-nominal">{fmt(rec.new_miss_distance_km, 3, ' km')}</dd>
                  <dt>New Pc</dt><dd className="is-nominal">{formatPc(rec.new_pc_collision)}</dd>
                </dl>
                <div className={`tq-verify${verified ? ' is-verified' : ''}`}>
                  {verified ? 'Verified by re-propagation' : 'Not verified by re-propagation'}
                </div>
              </>
            ) : (
              <div className="tq-note">No manoeuvre computed for this conjunction</div>
            )}
          </div>

          <div>
            <span className="ui-label tq-section-label">Cascade Impact</span>
            <dl className="tq-dl">
              <dt>Cascade depth</dt><dd>{cascadeDepth == null ? DASH : cascadeDepth}</dd>
              <dt>Downstream</dt>
              <dd title={downstream ? downstream.join(', ') : ''}>
                {downstream == null ? DASH : `${downstream.length} object${downstream.length === 1 ? '' : 's'}`}
              </dd>
              {alert.upstream_event && (
                <>
                  <dt>Upstream</dt><dd>{alert.upstream_event}</dd>
                </>
              )}
            </dl>
          </div>

          {showDvEditor && (
            <div className="tq-dv">
              <label className="ui-label" htmlFor={`tq-dv-${alert.id ?? 'unknown'}`}>
                Custom Δv (m/s){recRsw ? ' · along computed direction' : ' · prograde'}
              </label>
              <div className="tq-dv-row">
                <input
                  id={`tq-dv-${alert.id ?? 'unknown'}`}
                  className="tq-dv-range"
                  type="range"
                  min={0.01}
                  max={2.0}
                  step={0.01}
                  value={dvValue}
                  onChange={(e) => setCustomDv(Number(e.target.value))}
                />
                <span className="tq-dv-value">{dvValue.toFixed(2)} m/s</span>
              </div>
              <button
                type="button"
                className="ui-btn ui-btn--primary"
                disabled={submitting || maneuverSatId == null}
                onClick={() => handleDecision('MODIFY', dvValue)}
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
              disabled={submitting || !canApprove}
              title={canApprove ? 'Execute the computed manoeuvre' : 'No computed manoeuvre to approve'}
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
