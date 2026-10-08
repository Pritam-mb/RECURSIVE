import { useEffect, useState, useRef } from 'react';
import { apiGet, apiPost } from '../utils/api';
import '../styles/analysis.css';

const POLL_MS = 30_000;

// Short display labels for the surrogate's feature names (backend: train_risk_surrogate.FEATURES).
const FEATURE_LABELS = {
  miss_distance_km: 'miss_dist',
  radial_miss_km: 'radial_miss',
  relative_speed_kms: 'rel_vel',
  tle_age_max_h: 'tle_age_max',
  tle_age_min_h: 'tle_age_min',
  hbr_km: 'hbr',
  altitude_km: 'altitude',
};
const DEFAULT_BUFFER_THRESHOLD = 500;
// ENABLE_EXTENDED_PIPELINE gates only optional extras (trajectory RNN recorder,
// anomaly flags, shadow retraining, Kafka). Screening, Pc, the XGBoost surrogate,
// decisions, cascade and debris always run.
const PIPELINE_OFF_NOTE = 'Core pipeline running. Optional extended ML (trajectory RNN, anomaly flags, shadow retraining) is off to save RAM — set ENABLE_EXTENDED_PIPELINE=1 to enable.';

// '—' when the value is unknown (status endpoint unreachable), never a fake 0.
const fmt = (v) => (v == null ? '—' : Number(v).toLocaleString());

// 0-1 value → clamped percentage for a flat meter; null when unknown.
const pct01 = (v) => (v != null && Number.isFinite(Number(v)) ? Math.max(0, Math.min(1, Number(v))) * 100 : null);
// log10 error in decades.
const dec = (v) => (v != null && Number.isFinite(Number(v)) ? `${Number(v).toFixed(2)} dec` : '—');
const fixed = (v, d) => (v != null && Number.isFinite(Number(v)) ? Number(v).toFixed(d) : '—');

const Row = ({ k, children, valueClass = '' }) => (
  <div className="an-ms-row">
    <span className="an-ms-key">{k}</span>
    <span className={`an-ms-val ${valueClass}`}>{children}</span>
  </div>
);

const MeterRow = ({ k, pct, children, mono = false, valueClass = '' }) => (
  <div className="an-ms-row an-ms-row--meter">
    <span className={`an-ms-key${mono ? ' an-ms-key--mono' : ''}`}>{k}</span>
    <div className="an-meter">
      <div className={`an-meter-fill ${valueClass}`} style={{ width: `${pct ?? 0}%` }} />
    </div>
    <span className={`an-ms-val ${valueClass}`}>{children}</span>
  </div>
);

const ModelHead = ({ name, kind, ok, okText, offText }) => (
  <div className="an-ms-group-head">
    <span className="an-ms-name">
      {name}
      {kind && <span className="an-ms-kind">{kind}</span>}
    </span>
    <span className={`ui-status an-ms-state ${ok ? 'is-nominal' : 'is-caution'}`}>
      {ok ? okText : offText}
    </span>
  </div>
);

export default function ModelStatusV2() {
  const [metrics, setMetrics] = useState(null);
  const [cascadeStatus, setCascadeStatus] = useState(null);
  const [mlStatus, setMlStatus] = useState(null);
  const [mlStatusReachable, setMlStatusReachable] = useState(null);
  const [retrainBusy, setRetrainBusy] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState(null);
  const mountedRef = useRef(true);

  const fetchAll = async () => {
    if (!mountedRef.current) return;
    try {
      const [m, c, s] = await Promise.all([
        apiGet('/api/model-metrics').catch(() => null),
        apiGet('/api/cascade/status').catch(() => null),
        apiGet('/api/ml/status').catch(() => null),
      ]);
      if (!mountedRef.current) return;
      if (m) setMetrics(m);
      if (c) setCascadeStatus(c);
      setMlStatus(s);
      setMlStatusReachable(s != null);
    } catch (_) {}
  };

  const triggerRetrain = async () => {
    setRetrainBusy(true);
    setRetrainMsg(null);
    try {
      const res = await apiPost('/api/ml/retrain');
      if (!mountedRef.current) return;
      setRetrainMsg({
        ok: res?.ok !== false,
        text: res?.message ?? (res?.ok === false ? 'Retrain did not run' : 'Retrain complete'),
      });
    } catch (err) {
      if (!mountedRef.current) return;
      setRetrainMsg({
        ok: false,
        text: err?.body?.message ?? err?.body?.detail ?? err?.message ?? 'Retrain request failed',
      });
    } finally {
      if (mountedRef.current) {
        setRetrainBusy(false);
        fetchAll();
      }
    }
  };

  useEffect(() => {
    mountedRef.current = true;
    fetchAll();
    const id = setInterval(fetchAll, POLL_MS);
    return () => { mountedRef.current = false; clearInterval(id); };
  }, []);

  // ── Pc surrogate (model card: held-out metrics written at training time) ──
  const risk = metrics?.risk_model ?? null;
  const riskOk = risk?.available === true;
  const summary = risk?.summary ?? {};
  const heldout = risk?.heldout ?? {};
  const cls4 = heldout['classification_at_1e-4'] ?? null;
  const live = risk?.live ?? null;
  const importance = Object.entries(risk?.feature_importance ?? {})
    .filter(([, v]) => v != null && Number.isFinite(Number(v)))
    .sort((a, b) => b[1] - a[1]);
  const maxImp = importance.length ? Math.max(...importance.map(([, v]) => Number(v)), 1e-9) : 1;
  const trainedAt = risk?.trained_at_utc ? String(risk.trained_at_utc).substring(0, 10) : '—';

  // ── Trajectory model (benchmarked against persistence + linear baselines) ──
  const traj = metrics?.trajectory_model ?? null;
  const trajMae = traj?.mae_km ?? null;
  const trajBeats = traj?.beats_baselines === true;
  const pipelineEnabled = mlStatus?.pipeline_enabled ?? null;
  const pipelineOff = pipelineEnabled === false;
  const lstm = mlStatus?.lstm ?? null;
  const bufferFill = lstm?.buffer_records ?? null;
  const bufferThreshold = lstm?.buffer_threshold || DEFAULT_BUFFER_THRESHOLD;
  const bufferPct = bufferFill != null ? Math.min((bufferFill / bufferThreshold) * 100, 100) : 0;
  const readyToRetrain = bufferFill != null ? bufferFill >= bufferThreshold : null;

  // ── Graph rankers (cross-check only) ──
  const graph = metrics?.graph_models ?? null;
  const gat = graph?.gat ?? null;
  const cascadePred = cascadeStatus?.cascade_predictions ?? null;
  const meanDepth = cascadeStatus?.mean_cascade_depth ?? null;

  // ── Operator feedback log (nothing trains on it) ──
  const fb = metrics?.operator_feedback?.stats ?? mlStatus?.rlhf ?? null;
  const fbDecisions = fb?.decisions ?? null;
  const approvalRate = fb?.approval_rate ?? null;
  const fbRoundRates = (fb?.round_approval_rates ?? [])
    .filter((r) => r != null && Number.isFinite(Number(r)))
    .map((r) => { const n = Number(r); return Math.max(0, Math.min(1, n > 1 ? n / 100 : n)); });
  const shadow = mlStatus?.shadow ?? null;

  const pipelineState = mlStatusReachable === false
    ? { cls: 'is-warning', text: 'Unreachable' }
    : pipelineEnabled === true
      ? { cls: 'is-nominal', text: 'Core + extended ML' }
      : pipelineOff
        ? { cls: 'is-nominal', text: 'Core pipeline on' }
        : { cls: 'is-dim', text: 'Pending' };

  return (
    <section className="ui-panel an-panel an-models">
      <header className="ui-panel-header">
        <span className="ui-label an-title">Model Status</span>
        <span className={`ui-status ${pipelineState.cls}`}>{pipelineState.text}</span>
      </header>

      <div className="an-ms-body">
        {mlStatusReachable === false && (
          <div className="an-ms-banner">ML status endpoint unreachable</div>
        )}

        {/* Pc surrogate */}
        <div className="an-ms-group">
          <ModelHead
            name="Pc Surrogate"
            kind="XGBOOST"
            ok={riskOk}
            okText="Held-out validated"
            offText={metrics ? 'Unavailable' : 'Pending'}
          />
          <div className="an-ms-note">
            Cross-check only. The physics Foster Pc on each alert is authoritative.
          </div>
          <div className="an-ms-rows">
            <Row k="Model">{risk?.name ?? '—'}</Row>
            <Row k="Task">Pc surrogate (log10)</Row>
            <Row k="Label source" valueClass="is-dim">Foster Pc, simulated encounters</Row>
            <Row k="Training samples">{fmt(risk?.training_samples)}</Row>
            <Row k="Held-out samples">{fmt(risk?.heldout_samples)}</Row>
            <Row k="Held-out MAE (log10 Pc)" valueClass="is-nominal">{dec(summary.heldout_mae_log10_pc)}</Row>
            <Row k="Baseline MAE (miss-dist only)" valueClass="is-dim">{dec(summary.baseline_mae_log10_pc)}</Row>
          </div>

          <div className="ui-label an-ms-subhead">Classify Pc ≥ 1e-4 (held-out)</div>
          <div className="an-ms-rows">
            <MeterRow k="Precision" pct={pct01(summary['precision_at_1e-4'])}>{fixed(summary['precision_at_1e-4'], 3)}</MeterRow>
            <MeterRow k="Recall" pct={pct01(summary['recall_at_1e-4'])}>{fixed(summary['recall_at_1e-4'], 3)}</MeterRow>
            <MeterRow k="F1" pct={pct01(summary['f1_at_1e-4'])} valueClass="is-nominal">{fixed(summary['f1_at_1e-4'], 3)}</MeterRow>
            <MeterRow k="F1 baseline (miss-dist only)" pct={pct01(summary['baseline_f1_at_1e-4'])}>{fixed(summary['baseline_f1_at_1e-4'], 3)}</MeterRow>
            {cls4 && <Row k="Positives / held-out" valueClass="is-dim">{fmt(cls4.positives)} / {fmt(risk?.heldout_samples)}</Row>}
          </div>

          <div className="ui-label an-ms-subhead">Feature importance (gain)</div>
          <div className="an-ms-rows">
            {importance.length === 0 && <Row k="Not reported">—</Row>}
            {importance.map(([feat, v]) => (
              <MeterRow key={feat} k={FEATURE_LABELS[feat] ?? feat} mono pct={(Number(v) / maxImp) * 100}>
                {`${(Number(v) * 100).toFixed(1)}%`}
              </MeterRow>
            ))}
          </div>

          <div className="ui-label an-ms-subhead">Live (this process)</div>
          <div className="an-ms-rows">
            <Row k="Alerts scored">{fmt(live?.alerts_scored)}</Row>
            <Row k="Surrogate class high (≥1e-4)">{fmt(live?.risk_class_high)}</Row>
            <Row k="Mean |Δlog10| vs physics Pc">{dec(live?.mean_agreement_decades)}</Row>
            <Row k="Disagreements > 1 decade" valueClass={live?.disagreements_over_1_decade > 0 ? 'is-caution' : ''}>
              {fmt(live?.disagreements_over_1_decade)}
            </Row>
            <Row k="Trained">{trainedAt}</Row>
          </div>
        </div>

        {/* Trajectory */}
        <div className="an-ms-group">
          <ModelHead
            name="Trajectory Predictor"
            kind="NUMPY RNN"
            ok={trajBeats}
            okText="Beats baselines"
            offText={traj?.status === 'experimental' ? 'Experimental' : 'Unevaluated'}
          />
          {traj?.status === 'experimental' && (
            <div className="an-ms-note is-caution">
              Underperforms linear extrapolation. Not used for alerts, Pc or manoeuvres; SGP4 is the propagator of record.
            </div>
          )}
          <div className="an-ms-rows">
            <Row k="1-step MAE: model" valueClass={trajBeats ? 'is-nominal' : 'is-warning'}>
              {trajMae?.model != null ? `${fixed(trajMae.model, 1)} km` : '—'}
            </Row>
            <Row k="Persistence baseline">{trajMae?.persistence != null ? `${fixed(trajMae.persistence, 1)} km` : '—'}</Row>
            <Row k="Linear extrapolation baseline">{trajMae?.linear_extrapolation != null ? `${fixed(trajMae.linear_extrapolation, 1)} km` : '—'}</Row>
            <Row k="Eval set" valueClass="is-dim">{traj?.eval_samples != null ? `${traj.eval_samples} synthetic arcs` : '—'}</Row>
            {pipelineEnabled === true && (
              <MeterRow k="Buffer fill" pct={bufferPct} valueClass={readyToRetrain ? 'is-nominal' : ''}>
                {fmt(bufferFill)}/{bufferThreshold.toLocaleString()}
              </MeterRow>
            )}
          </div>
        </div>

        {/* Graph rankers */}
        <div className="an-ms-group">
          <ModelHead name="Graph Rankers" kind="GNN / GAT" ok={false} okText="" offText="Cross-check only" />
          <div className="an-ms-note">
            Cascade depth comes from physics alert links, not these models. Trained on synthetic heuristic labels.
          </div>
          <div className="an-ms-rows">
            <Row k="GAT trainer">{gat?.trainer ?? 'unavailable'}</Row>
            <Row k="Heads / hidden dim">
              {gat?.heads != null && gat?.hidden_dim != null ? `${gat.heads} / ${gat.hidden_dim}` : 'n/a'}
            </Row>
            <Row k="Training graphs">{fmt(gat?.training_graphs)}</Row>
            <Row k="Cascade predictions">{fmt(cascadePred)}</Row>
            <Row k="Mean cascade depth">{meanDepth != null ? Number(meanDepth).toFixed(1) : '—'}</Row>
          </div>
        </div>

        {/* Operator feedback log */}
        <div className="an-ms-group">
          <div className="an-ms-group-head">
            <span className="an-ms-name">
              Operator Feedback Log
              <span className="an-ms-kind">AUDIT</span>
            </span>
            <span className="ui-status an-ms-state is-dim">Trains no model</span>
          </div>
          <div className="an-ms-rows">
            <Row k="Decisions logged">{fmt(fbDecisions)}</Row>
            <MeterRow k="Approval rate" pct={pct01(approvalRate)}>
              {approvalRate != null ? `${(approvalRate * 100).toFixed(0)}%` : '—'}
            </MeterRow>
          </div>
          {fbRoundRates.length > 0 && (
            <div className="an-ms-history">
              {fbRoundRates.slice(-10).map((r, i, shown) => {
                const block = fbRoundRates.length - shown.length + i + 1;
                return (
                  <div
                    key={block}
                    className="an-ms-history-bar"
                    style={{ height: `${Math.max(2, r * 24)}px` }}
                    title={`Block ${block} (10 decisions): ${(r * 100).toFixed(0)}% approved`}
                  />
                );
              })}
            </div>
          )}
        </div>

        {/* Retraining */}
        <div className="an-ms-group">
          <div className="an-ms-group-head">
            <span className="an-ms-name">Retraining</span>
          </div>
          {pipelineOff && <div className="an-ms-note">{PIPELINE_OFF_NOTE}</div>}
          <div className="an-ms-rows">
            <Row k="Pc surrogate" valueClass="is-dim">offline · train_risk_surrogate</Row>
            <Row k="Trajectory (LSTM buffer)" valueClass={readyToRetrain ? 'is-nominal' : 'is-dim'}>
              {readyToRetrain == null
                ? '—'
                : readyToRetrain
                  ? 'READY'
                  : `${bufferFill.toLocaleString()}/${bufferThreshold.toLocaleString()}`}
            </Row>
            <Row k="Shadow retrain">{shadow == null ? '—' : shadow.enabled ? 'ENABLED' : 'DISABLED'}</Row>
            <Row k="Last retrain">{shadow == null ? '—' : (shadow.last_retrain ?? 'never')}</Row>
          </div>

          <div className="an-ms-actions">
            <button
              type="button"
              className="ui-btn"
              onClick={triggerRetrain}
              disabled={retrainBusy || pipelineEnabled !== true}
              title={pipelineOff ? PIPELINE_OFF_NOTE : undefined}
            >
              {retrainBusy ? 'Retraining…' : 'Retrain trajectory'}
            </button>
            {retrainMsg && (
              <span className={`an-ms-msg ${retrainMsg.ok ? 'is-nominal' : 'is-warning'}`}>{retrainMsg.text}</span>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
