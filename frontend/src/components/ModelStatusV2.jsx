import { useEffect, useState, useRef } from 'react';
import { apiGet, apiPost } from '../utils/api';
import '../styles/analysis.css';

const POLL_MS = 30_000;

const TOP_FEATURES = ['miss_dist', 'rel_vel', 'altitude', 'tle_age'];
const DEFAULT_BUFFER_THRESHOLD = 500;
const PIPELINE_OFF_NOTE = 'Extended pipeline off — set ENABLE_EXTENDED_PIPELINE=1';

// '—' when the value is unknown (status endpoint unreachable), never a fake 0.
const fmt = (v) => (v == null ? '—' : Number(v).toLocaleString());

// 0-1 value → clamped percentage for a flat meter; null when unknown.
const pct01 = (v) => (v != null && Number.isFinite(Number(v)) ? Math.max(0, Math.min(1, Number(v))) * 100 : null);
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
  const [sampleHistory, setSampleHistory] = useState([]);
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
      setSampleHistory((prev) => {
        const samples = m?.classification_models?.[0]?.metrics?.samples ?? 0;
        const ts = new Date().toISOString().substring(11, 19);
        return [...prev, { ts, count: samples }].slice(-10);
      });
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

  const riskModel = (metrics?.classification_models ?? []).find((m) => m.name === 'Risk Scorer');
  const graphModel = (metrics?.classification_models ?? []).find((m) => m.name === 'Graph Cascade');
  const gatModel = (metrics?.classification_models ?? []).find((m) => m.name === 'GAT Cascade');
  const trajModel = (metrics?.regression_models ?? []).find((m) => m.name === 'Trajectory Predictor');

  const usingTrained = (riskModel?.metrics?.samples ?? 0) > 0;
  const trainSamples = riskModel?.metrics?.samples ?? 0;
  const precision = riskModel?.metrics?.precision ?? null;
  const recall = riskModel?.metrics?.recall ?? null;
  const f1 = riskModel?.metrics?.f1 ?? null;
  const pipelineEnabled = mlStatus?.pipeline_enabled ?? null;
  const pipelineOff = pipelineEnabled === false;
  const xgb = mlStatus?.xgboost ?? null;
  const lstm = mlStatus?.lstm ?? null;
  const rlhf = mlStatus?.rlhf ?? null;
  const meta = mlStatus?.meta_propagator ?? null;
  const shadow = mlStatus?.shadow ?? null;

  const predictions = xgb?.predictions_today ?? null;
  const highRisk = xgb?.high_risk_today ?? null;

  const lstmTrained = lstm?.trained ?? ((trajModel?.metrics?.samples ?? 0) > 0);
  const lstmSeqs = trajModel?.metrics?.samples ?? 0;
  const lstmSats = lstm?.satellites_tracked ?? null;
  const lstmErr = trajModel?.metrics?.mae ?? null;
  const bufferFill = lstm?.buffer_records ?? null;
  const bufferThreshold = lstm?.buffer_threshold || DEFAULT_BUFFER_THRESHOLD;

  const gatStatus = metrics?.graph_attention ?? null;
  const gatDegenerate = gatStatus?.degenerate !== false;
  const cascadePred = cascadeStatus?.cascade_predictions ?? graphModel?.metrics?.samples ?? 0;
  const meanDepth = cascadeStatus?.mean_cascade_depth ?? 0;

  const rlhfDecisions = rlhf?.decisions ?? null;
  const approvalRate = rlhf?.approval_rate ?? null;
  const rlhfRounds = rlhf?.rounds ?? null;
  // Rates are 0-1; tolerate a backend that reports percentages.
  const rlhfRoundRates = (rlhf?.round_approval_rates ?? [])
    .filter((r) => r != null && Number.isFinite(Number(r)))
    .map((r) => { const n = Number(r); return Math.max(0, Math.min(1, n > 1 ? n / 100 : n)); });

  const metaSats = meta?.satellites_with_corrections ?? null;
  const metaImprovement = meta?.mean_improvement_pct ?? null;
  const metaCorrections = meta?.corrections ?? null;

  // Conjunction samples are the XGBoost training set; the retrain trigger is
  // the LSTM position buffer reaching its threshold, so readiness is based on
  // buffer fill (unknown when /api/ml/status is unreachable).
  const conjSamples = metrics?.conjunction_samples ?? trainSamples;
  const readyToRetrain = bufferFill != null ? bufferFill >= bufferThreshold : null;
  const bufferPct = bufferFill != null ? Math.min((bufferFill / bufferThreshold) * 100, 100) : 0;

  const pipelineState = mlStatusReachable === false
    ? { cls: 'is-warning', text: 'Unreachable' }
    : pipelineEnabled === true
      ? { cls: 'is-nominal', text: 'Pipeline on' }
      : pipelineOff
        ? { cls: 'is-caution', text: 'Pipeline off' }
        : { cls: 'is-dim', text: 'Pending' };

  // Sparkline for sample history (simple inline SVG)
  const maxCount = Math.max(...sampleHistory.map((s) => s.count), 1);
  const sparkW = 260;
  const sparkH = 28;

  const sparkPoints = sampleHistory
    .map((s, i) => {
      const x = sampleHistory.length < 2
        ? sparkW / 2
        : (i / (sampleHistory.length - 1)) * sparkW;
      const y = sparkH - (s.count / maxCount) * (sparkH - 4) - 2;
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <section className="ui-panel an-panel an-models">
      <header className="ui-panel-header">
        <span className="ui-label an-title">Model Status</span>
        <span className={`ui-status ${pipelineState.cls}`}>{pipelineState.text}</span>
      </header>

      <div className="an-ms-body">
        {pipelineOff && <div className="an-ms-banner">{PIPELINE_OFF_NOTE}</div>}
        {mlStatusReachable === false && (
          <div className="an-ms-banner">ML status endpoint unreachable</div>
        )}

        {/* XGBoost */}
        <div className="an-ms-group">
          <ModelHead name="Risk Classifier" kind="XGBOOST" ok={usingTrained} okText="Trained" offText="Heuristic" />
          <div className="an-ms-rows">
            <Row k="Training samples">{trainSamples.toLocaleString()}</Row>
            {usingTrained && precision != null && (
              <>
                <MeterRow k="Precision" pct={pct01(precision)}>{fixed(precision, 2)}</MeterRow>
                <MeterRow k="Recall" pct={pct01(recall)}>{fixed(recall, 2)}</MeterRow>
                <MeterRow k="F1" pct={pct01(f1)}>{fixed(f1, 2)}</MeterRow>
              </>
            )}
            <Row k="Predictions today">{fmt(predictions)}</Row>
            <Row k="High-risk detections" valueClass={highRisk > 0 ? 'is-caution' : ''}>{fmt(highRisk)}</Row>
          </div>

          <div className="ui-label an-ms-subhead">Feature importance</div>
          <div className="an-ms-rows">
            {TOP_FEATURES.map((feat) => {
              // No fabricated fallback: unknown importance renders as an empty bar + '—'.
              const raw = riskModel?.feature_importance?.[feat];
              const imp = raw != null && Number.isFinite(Number(raw)) ? Number(raw) : null;
              return (
                <MeterRow key={feat} k={feat} mono pct={imp != null ? Math.min(imp * 100, 100) : 0}>
                  {imp != null ? `${(imp * 100).toFixed(0)}%` : '—'}
                </MeterRow>
              );
            })}
          </div>
        </div>

        {/* LSTM */}
        <div className="an-ms-group">
          <ModelHead name="Trajectory Predictor" kind="LSTM" ok={lstmTrained} okText="Trained" offText="Linear extrap." />
          <div className="an-ms-rows">
            <Row k="Training sequences">{lstmSeqs} / {fmt(lstmSats)} sats</Row>
            {lstmErr != null && <Row k="Mean prediction error">{lstmErr.toFixed(2)} km</Row>}
            <MeterRow k="Buffer fill" pct={bufferPct} valueClass={readyToRetrain ? 'is-nominal' : ''}>
              {fmt(bufferFill)}/{bufferThreshold.toLocaleString()}
            </MeterRow>
          </div>
        </div>

        {/* GNN / GAT */}
        <div className="an-ms-group">
          <ModelHead name="Graph Cascade" kind="GNN / GAT" ok={!gatDegenerate} okText="GAT attention" offText="GAT degenerate" />
          <div className="an-ms-rows">
            {gatDegenerate && (
              <Row k="GAT is not an independent model" valueClass="is-warning">ALIASES GNN</Row>
            )}
            <Row k="Attention trainer">{gatStatus?.trainer ?? 'unavailable'}</Row>
            <Row k="Heads / hidden dim">
              {gatStatus?.heads != null && gatStatus?.hidden_dim != null
                ? `${gatStatus.heads} / ${gatStatus.hidden_dim}`
                : 'n/a'}
            </Row>
            {gatModel?.metrics?.accuracy != null && (
              <MeterRow
                k={`GAT accuracy${gatModel.independent === false ? ' (aliased)' : ''}`}
                pct={pct01(gatModel.metrics.accuracy)}
              >
                {gatModel.metrics.accuracy.toFixed(4)}
              </MeterRow>
            )}
            {graphModel?.metrics?.accuracy != null && (
              <MeterRow k="GNN accuracy" pct={pct01(graphModel.metrics.accuracy)}>
                {graphModel.metrics.accuracy.toFixed(4)}
              </MeterRow>
            )}
            <Row k="Cascade predictions">{cascadePred}</Row>
            <Row k="Mean cascade depth">{Number(meanDepth).toFixed(1)}</Row>
          </div>
        </div>

        {/* Meta-Propagator + RLHF */}
        <div className="an-ms-group">
          <div className="an-ms-group-head">
            <span className="an-ms-name">
              Meta-Propagator
              <span className="an-ms-kind">+ RLHF</span>
            </span>
          </div>
          <div className="an-ms-rows">
            <Row k="Sats with corrections">{fmt(metaSats)}</Row>
            <Row k="Mean improvement vs SGP4">
              {metaImprovement != null ? `${Number(metaImprovement).toFixed(1)}%` : '—'}
            </Row>
            <Row k="Total corrections">{fmt(metaCorrections)}</Row>
            <div className="an-ms-gap" />
            <Row k="RLHF decisions">{fmt(rlhfDecisions)}</Row>
            <MeterRow k="Approval rate" pct={pct01(approvalRate)}>
              {approvalRate != null ? `${(approvalRate * 100).toFixed(0)}%` : '—'}
            </MeterRow>
            <Row k="RLHF rounds">{fmt(rlhfRounds)}</Row>
          </div>

          {/* Per-round approval history */}
          {rlhfRoundRates.length > 0 && (
            <div className="an-ms-history">
              {rlhfRoundRates.slice(-10).map((r, i, shown) => {
                const round = rlhfRoundRates.length - shown.length + i + 1;
                return (
                  <div
                    key={round}
                    className="an-ms-history-bar"
                    style={{ height: `${Math.max(2, r * 24)}px` }}
                    title={`Round ${round}: ${(r * 100).toFixed(0)}%`}
                  />
                );
              })}
            </div>
          )}
        </div>

        {/* Training data accumulation */}
        <div className="an-ms-group">
          <div className="an-ms-group-head">
            <span className="an-ms-name">Training Data</span>
          </div>
          <div className="an-ms-rows">
            <Row k="Conjunction samples">{conjSamples.toLocaleString()}</Row>
            <Row k="Ready to retrain (LSTM buffer)" valueClass={readyToRetrain ? 'is-nominal' : 'is-dim'}>
              {readyToRetrain == null
                ? '—'
                : readyToRetrain
                  ? 'YES'
                  : `NO (${bufferFill.toLocaleString()}/${bufferThreshold.toLocaleString()})`}
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
              {retrainBusy ? 'Retraining…' : 'Retrain now'}
            </button>
            {retrainMsg && (
              <span className={`an-ms-msg ${retrainMsg.ok ? 'is-nominal' : 'is-warning'}`}>{retrainMsg.text}</span>
            )}
          </div>

          {/* Sample count sparkline */}
          {sampleHistory.length > 1 && (
            <svg className="an-ms-spark" viewBox={`0 0 ${sparkW} ${sparkH}`} preserveAspectRatio="none">
              <polyline points={sparkPoints} />
            </svg>
          )}
        </div>
      </div>
    </section>
  );
}
