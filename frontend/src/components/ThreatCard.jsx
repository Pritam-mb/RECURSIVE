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

function sig(v, p = 3) {
  const n = num(v);
  if (n == null) return DASH;
  if (n === 0) return '0';
  const a = Math.abs(n);
  if (a >= 1e5 || a < 1e-3) return n.toExponential(2);
  return String(Number(n.toPrecision(p)));
}

const ACTION_STATE = { MANOEUVRE: 'warning', PREPARE: 'caution', MONITOR: 'info', NONE: 'nominal' };
const CONF_STATE = { high: 'nominal', medium: 'caution', low: 'warning' };

// ── Decision (score 0–100, action, components) ─────────────────────────────
const DecisionBlock = memo(function DecisionBlock({ decision }) {
  if (!decision || typeof decision !== 'object') {
    return (
      <div>
        <span className="ui-label tq-section-label">Decision</span>
        <div className="tq-note">Decision score not computed</div>
      </div>
    );
  }
  const score = num(decision.score);
  const action = typeof decision.action === 'string' ? decision.action.toUpperCase() : null;
  const st = ACTION_STATE[action] ?? 'dim';
  const comps = Array.isArray(decision.components) ? decision.components : [];
  const agree = decision.model_agreement ?? {};
  const conf = typeof agree.confidence === 'string' ? agree.confidence.toLowerCase() : null;
  // Gauge: 180° arc, score 0..100 mapped left → right.
  const f = score == null ? 0 : Math.min(1, Math.max(0, score / 100));
  const R = 34;
  const cx = 42;
  const cy = 40;
  const ex = cx - R * Math.cos(Math.PI * f);
  const ey = cy - R * Math.sin(Math.PI * f);
  return (
    <div>
      <span className="ui-label tq-section-label">Decision</span>
      <div className="tq-decision-top">
        <svg className="tq-gauge" width="84" height="48" viewBox="0 0 84 48" role="img" aria-label={`Decision score ${score == null ? 'not computed' : score.toFixed(0)} of 100`}>
          <path d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`} fill="none" stroke="var(--c-line-strong)" strokeWidth="5" />
          {score != null && f > 0 && (
            <path d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${ex.toFixed(2)} ${ey.toFixed(2)}`} fill="none" className={`tq-gauge-arc is-${st}`} strokeWidth="5" />
          )}
          <text x={cx} y={cy - 4} textAnchor="middle" className="tq-gauge-val">{score == null ? DASH : score.toFixed(0)}</text>
          <text x={cx} y={cy + 7} textAnchor="middle" className="tq-gauge-sub">/ 100</text>
        </svg>
        <div className="tq-decision-meta">
          <span className={`tq-action is-${st}`}>{action ?? 'NOT COMPUTED'}</span>
          <span className="tq-note">
            Confidence{' '}
            <span className={`tq-conf is-${CONF_STATE[conf] ?? 'dim'}`}>{conf ?? DASH}</span>
          </span>
          <span className="tq-note">
            Physics vs ML {fmt(agree.physics_vs_ml_decades, 2, ' dec')} · methods spread {fmt(agree.physics_methods_spread_decades, 2, ' dec')}
          </span>
        </div>
      </div>
      {comps.length > 0 ? (
        <table className="tq-table">
          <thead>
            <tr><th>Component</th><th>Src</th><th className="num">Raw</th><th className="num">Norm</th><th className="num">w</th><th className="num">Pts</th></tr>
          </thead>
          <tbody>
            {comps.map((c, i) => (
              <tr key={`${c?.name ?? 'c'}-${i}`}>
                <td title={c?.name ?? ''}>{c?.name ?? DASH}</td>
                <td className="tq-dim">{c?.source ?? DASH}</td>
                <td className="num">{sig(c?.raw)}</td>
                <td className="num">{fmt(c?.normalized, 2)}</td>
                <td className="num">{fmt(c?.weight, 2)}</td>
                <td className="num">{fmt(c?.points, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <div className="tq-note">Score components not provided</div>}
      {decision.rationale && <div className="tq-rationale">{decision.rationale}</div>}
    </div>
  );
});

// ── Pc cross-check across independent methods ──────────────────────────────
const PcCrossCheck = memo(function PcCrossCheck({ checks }) {
  if (!checks || typeof checks !== 'object') {
    return (
      <div>
        <span className="ui-label tq-section-label">Pc Cross-check</span>
        <div className="tq-note">Pc cross-check not computed</div>
      </div>
    );
  }
  const mcN = num(checks.mc_samples);
  const consistent = checks.consistent;
  return (
    <div>
      <span className="ui-label tq-section-label">Pc Cross-check</span>
      <dl className="tq-dl">
        <dt>Foster 2D</dt><dd>{formatPc(checks.foster)}</dd>
        <dt>Chan series</dt><dd>{formatPc(checks.chan)}</dd>
        <dt>Alfano max Pc</dt><dd title="Upper bound over covariance scaling">{formatPc(checks.alfano_max)} (bound)</dd>
        <dt>Monte Carlo</dt>
        <dd>{num(checks.monte_carlo) == null ? 'not computed' : `${formatPc(checks.monte_carlo)}${mcN != null ? ` · n=${mcN.toLocaleString('en-US')}` : ''}`}</dd>
        <dt>Spread</dt><dd>{fmt(checks.spread_decades, 2, ' decades')}</dd>
        <dt>Consistent</dt>
        <dd className={checks.consistent === true ? 'is-nominal' : consistent === false ? 'is-caution' : ''}>
          {consistent === true ? 'YES (≤ 0.5 dec)' : consistent === false ? 'NO (> 0.5 dec)' : DASH}
        </dd>
      </dl>
    </div>
  );
});

// ── ML explanation: signed TreeSHAP contributions in decades of Pc ─────────
const MlExplanation = memo(function MlExplanation({ ml }) {
  const contribs = Array.isArray(ml?.contributions)
    ? ml.contributions.filter((c) => c && c.feature && num(c.contribution_log10) != null)
    : [];
  if (!ml || contribs.length === 0) {
    return (
      <div>
        <span className="ui-label tq-section-label">ML Explanation</span>
        <div className="tq-note">{ml ? 'Feature contributions not computed' : 'ML scoring not attached'}</div>
      </div>
    );
  }
  const maxAbs = Math.max(...contribs.map((c) => Math.abs(num(c.contribution_log10)))) || 1;
  const main = ml.main_factor ?? contribs[0].feature;
  return (
    <div>
      <span className="ui-label tq-section-label">ML Explanation</span>
      <div className="tq-note">
        Main factor <span className="tq-strong">{main}</span> · base {fmt(ml.base_log10, 2)} log10 Pc
        {num(ml.pc_surrogate) != null ? ` · surrogate ${formatPc(ml.pc_surrogate)}` : ''}
      </div>
      <div className="tq-shap">
        {contribs.map((c, i) => {
          const v = num(c.contribution_log10);
          const w = (Math.abs(v) / maxAbs) * 50;
          return (
            <div className="tq-shap-row" key={`${c.feature}-${i}`} title={`${c.feature} = ${c.value == null ? 'n/a' : c.value} → ${v >= 0 ? '+' : ''}${v.toFixed(3)} decades`}>
              <span className={`tq-shap-name${c.feature === main ? ' is-main' : ''}`}>{c.feature}</span>
              <span className="tq-shap-track">
                <span className="tq-shap-axis" />
                <span className={`tq-shap-bar ${v >= 0 ? 'is-up' : 'is-down'}`} style={v >= 0 ? { left: '50%', width: `${w}%` } : { right: '50%', width: `${w}%` }} />
              </span>
              <span className="tq-shap-val">{v >= 0 ? '+' : '−'}{Math.abs(v).toFixed(2)}</span>
              <span className="tq-shap-fv">{num(c.value) == null ? DASH : sig(c.value)}</span>
            </div>
          );
        })}
      </div>
      <div className="tq-caption">Decades each feature pushes Pc up (orange) or down (blue) · last column = feature value</div>
    </div>
  );
});

// ── Avoidance options (cascade-safe screening + propulsion) ────────────────
function OptionEngines({ option }) {
  const engines = Array.isArray(option?.engines) ? option.engines : [];
  const ac = option?.analytic_check ?? null;
  const recEngine = engines.find((x) => x?.engine === option?.recommended_engine);
  return (
    <div className="tq-opt-more">
      {engines.length > 0 ? (
        <table className="tq-table">
          <thead>
            <tr><th>Engine</th><th className="num">Isp s</th><th className="num">Prop kg</th><th className="num">Burn s</th><th>Finite</th></tr>
          </thead>
          <tbody>
            {engines.map((e, i) => {
              const isRec = option.recommended_engine != null && e?.engine === option.recommended_engine;
              return (
                <tr key={`${e?.engine ?? 'e'}-${i}`} className={isRec ? 'is-rec' : ''} title={[e?.family, e?.propellant, e?.note].filter(Boolean).join(' · ')}>
                  <td>{isRec ? '★ ' : ''}{e?.engine ?? DASH}</td>
                  <td className="num">{sig(e?.isp_s, 4)}</td>
                  <td className="num">{sig(e?.prop_mass_kg)}</td>
                  <td className="num">{sig(e?.burn_time_s)}</td>
                  <td className={e?.finite_burn_ok === true ? 'is-nominal' : e?.finite_burn_ok === false ? 'is-warning' : ''}>
                    {e?.finite_burn_ok === true ? 'OK' : e?.finite_burn_ok === false ? 'NO' : DASH}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : <div className="tq-note">Engine comparison not computed</div>}
      {option?.recommended_engine && (
        <div className="tq-note">
          Recommended engine: <span className="tq-strong">{option.recommended_engine}</span>
          {recEngine?.note ? ` — ${recEngine.note}` : ''}
        </div>
      )}
      <div className="tq-note">
        {ac ? (
          <>
            {ac.method ?? 'Analytic'} check: along-track {fmt(ac.along_track_shift_km_at_tca, 3, ' km')} vs numeric {fmt(ac.numeric_km, 3, ' km')}
            {' '}· rel err {num(ac.rel_error) == null ? DASH : `${(num(ac.rel_error) * 100).toFixed(1)}%`}
          </>
        ) : 'Analytic (CW) check not computed'}
      </div>
    </div>
  );
}

const AvoidanceOptions = memo(function AvoidanceOptions({ rec, selectedIdx, onSelect, name }) {
  const [open, setOpen] = useState(null);
  if (!rec) return null;
  const options = Array.isArray(rec.options) ? rec.options : [];
  const chosen = num(rec.chosen_index);
  return (
    <div>
      <span className="ui-label tq-section-label">Avoidance Options</span>
      {options.length === 0 ? <div className="tq-note">Alternative options not computed</div> : (
        <div className="tq-opts">
          {options.map((o, i) => {
            const sec = Array.isArray(o?.secondary_conjunctions) ? o.secondary_conjunctions : null;
            const safe = o?.cascade_safe;
            const isSel = selectedIdx === i;
            return (
              <div key={`${o?.candidate ?? 'opt'}-${i}`} className={`tq-opt${isSel ? ' is-selected' : ''}`}>
                <div className="tq-opt-head">
                  <label className="tq-opt-pick">
                    <input type="radio" name={name} checked={isSel} onChange={() => onSelect(i)} />
                    <span className="tq-opt-name" title={o?.candidate ?? ''}>{o?.candidate ?? `Option ${i + 1}`}</span>
                  </label>
                  {chosen === i && <span className="tq-chip is-accent" title="Selected by the planner">CHOSEN</span>}
                  <span className={`tq-chip${safe === true ? ' is-nominal' : safe === false ? ' is-warning' : ''}`}>
                    {safe === true ? 'CASCADE-SAFE' : safe === false ? 'UNSAFE'
                      : (typeof o?.cascade_check === 'string' && o.cascade_check !== 'ok'
                        ? `NOT CHECKED (${o.cascade_check.replace(/_/g, ' ')})` : 'NOT CHECKED')}
                  </span>
                </div>
                <dl className="tq-dl">
                  <dt>|Δv|</dt><dd>{fmt(o?.delta_v_ms, 3, ' m/s')}</dd>
                  <dt>New Pc</dt><dd>{formatPc(o?.new_pc_collision)}</dd>
                  <dt>New miss</dt><dd>{fmt(o?.new_miss_distance_km, 3, ' km')}</dd>
                  <dt>Secondaries</dt>
                  <dd title={sec ? sec.map((c) => `${c?.name ?? c?.id}: ${sig(c?.miss_km)} km, Pc ${formatPc(c?.pc)}`).join('\n') : ''}>
                    {sec == null ? DASH : `${sec.length} · max Pc ${formatPc(o?.secondary_max_pc)}`}
                  </dd>
                </dl>
                <button type="button" className="tq-opt-toggle" aria-expanded={open === i} onClick={() => setOpen(open === i ? null : i)}>
                  {open === i ? '− Engines & checks' : '+ Engines & checks'}
                </button>
                {open === i && <OptionEngines option={o} />}
              </div>
            );
          })}
        </div>
      )}
      {rec.selection_rule && <div className="tq-caption">Selection rule: {rec.selection_rule}</div>}
    </div>
  );
});

function ThreatCard({ alert, isSelected, onDecision }) {
  const [expanded, setExpanded] = useState(false);
  const [showDvEditor, setShowDvEditor] = useState(false);
  const [customDv, setCustomDv] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [decisionError, setDecisionError] = useState(null);
  const [pickedOption, setPickedOption] = useState(null);
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
  // Avoidance options: Approve sends the selected option (default = planner's choice).
  const options = Array.isArray(rec?.options) ? rec.options : [];
  const chosenIdx = num(rec?.chosen_index);
  const defaultIdx = chosenIdx != null && options[chosenIdx] ? chosenIdx : null;
  const selIdx = pickedOption != null && options[pickedOption] ? pickedOption : defaultIdx;
  const selOpt = selIdx != null ? options[selIdx] : null;
  const selValid = formatRsw(selOpt?.delta_v_rsw_ms) != null && num(selOpt?.delta_v_ms) != null;
  const burn = selValid ? selOpt : rec;
  const recDv = num(burn?.delta_v_ms);
  const recRsw = Array.isArray(burn?.delta_v_rsw_ms) ? burn.delta_v_rsw_ms.map(Number) : null;
  const recRswText = formatRsw(burn?.delta_v_rsw_ms);
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
          <span className="tq-num">
            {missKm != null && missKm < 1 ? `${Math.round(missKm * 1000)} m` : fmt(missKm, 1)}
          </span>
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

          <DecisionBlock decision={alert.decision} />
          <PcCrossCheck checks={alert.pc_checks} />
          <MlExplanation ml={ml} />

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
                  {selValid && (
                    <>
                      <dt>Option</dt><dd title={selOpt.candidate ?? ''}>{selOpt.candidate ?? `#${selIdx + 1}`}</dd>
                    </>
                  )}
                  <dt>New miss</dt><dd className="is-nominal">{fmt(burn.new_miss_distance_km, 3, ' km')}</dd>
                  <dt>New Pc</dt><dd className="is-nominal">{formatPc(burn.new_pc_collision)}</dd>
                </dl>
                <div className={`tq-verify${verified ? ' is-verified' : ''}`}>
                  {verified ? 'Verified by re-propagation' : 'Not verified by re-propagation'}
                </div>
              </>
            ) : (
              <div className="tq-note">No manoeuvre computed for this conjunction</div>
            )}
          </div>

          <AvoidanceOptions
            rec={rec}
            selectedIdx={selIdx}
            onSelect={setPickedOption}
            name={`tq-opt-${alert.id ?? 'unknown'}`}
          />

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
