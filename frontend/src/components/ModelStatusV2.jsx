import { useEffect, useState, useRef } from 'react';
import { apiGet, apiPost } from '../utils/api';

const POLL_MS = 30_000;

const TOP_FEATURES = ['miss_dist', 'rel_vel', 'altitude', 'tle_age'];
const DEFAULT_BUFFER_THRESHOLD = 500;
const PIPELINE_OFF_NOTE = 'Extended pipeline off — set ENABLE_EXTENDED_PIPELINE=1';

// '—' when the value is unknown (status endpoint unreachable), never a fake 0.
const fmt = (v) => (v == null ? '—' : Number(v).toLocaleString());

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
    <div className="model-status-v2">
      {/* Section 1: XGBoost */}
      <div className="msv2-section">
        <div className="msv2-section-title">XGBoost Risk Classifier</div>
        <div className={`msv2-model-badge ${usingTrained ? 'trained' : 'heuristic'}`}>
          {usingTrained ? '● TRAINED' : '◌ HEURISTIC'}
        </div>
        <div className="msv2-row"><span>Training samples</span><span>{trainSamples.toLocaleString()}</span></div>
        {usingTrained && precision != null && (
          <div className="msv2-row">
            <span>Precision / Recall / F1</span>
            <span>{precision.toFixed(2)} / {recall.toFixed(2)} / {f1.toFixed(2)}</span>
          </div>
        )}
        <div className="msv2-row"><span>Predictions today</span><span>{fmt(predictions)}</span></div>
        <div className="msv2-row"><span>High-risk detections</span><span>{fmt(highRisk)}</span></div>
        {pipelineOff && <div className="msv2-note">{PIPELINE_OFF_NOTE}</div>}

        {/* Feature importance bars */}
        <div className="msv2-feature-bars">
          {TOP_FEATURES.map((feat) => {
            // No fabricated fallback: unknown importance renders as an empty bar + '—'.
            const raw = riskModel?.feature_importance?.[feat];
            const imp = raw != null && Number.isFinite(Number(raw)) ? Number(raw) : null;
            return (
              <div className="msv2-feature-bar-row" key={feat}>
                <span className="msv2-feature-name">{feat}</span>
                <div className="msv2-bar-container">
                  <div className="msv2-bar-fill" style={{ width: `${imp != null ? Math.min(imp * 100, 100) : 0}%` }} />
                </div>
                <span className="msv2-feature-val">{imp != null ? `${(imp * 100).toFixed(0)}%` : '—'}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Section 2: LSTM */}
      <div className="msv2-section">
        <div className="msv2-section-title">LSTM Trajectory Predictor</div>
        <div className={`msv2-model-badge ${lstmTrained ? 'trained' : 'heuristic'}`}>
          {lstmTrained ? '● TRAINED' : '◌ LINEAR EXTRAPOLATION'}
        </div>
        <div className="msv2-row">
          <span>Training sequences</span>
          <span>{lstmSeqs} from {fmt(lstmSats)} satellites</span>
        </div>
        {lstmErr != null && (
          <div className="msv2-row"><span>Mean prediction error</span><span>{lstmErr.toFixed(2)} km</span></div>
        )}
        <div className="msv2-row">
          <span>Buffer fill</span>
          <span>{fmt(bufferFill)} / {bufferThreshold.toLocaleString()}</span>
        </div>
        <div className="msv2-bar-container">
          <div className="msv2-bar-fill" style={{ width: `${bufferPct}%` }} />
        </div>
        {pipelineOff && <div className="msv2-note">{PIPELINE_OFF_NOTE}</div>}
      </div>

      {/* Section 3: GNN */}
      <div className="msv2-section">
        <div className="msv2-section-title">Graph Cascade Models</div>
        <div className={`msv2-model-badge ${gatDegenerate ? 'heuristic' : 'trained'}`}>
          {gatDegenerate ? '◌ GAT DEGENERATE' : '● GAT ATTENTION'}
        </div>
        {gatDegenerate && (
          <div className="msv2-row">
            <span>GAT is not an independent model</span>
            <span style={{ color: 'var(--alert-red)' }}>ALIASES GNN</span>
          </div>
        )}
        <div className="msv2-row">
          <span>Attention trainer</span>
          <span>{gatStatus?.trainer ?? 'unavailable'}</span>
        </div>
        <div className="msv2-row">
          <span>Heads / hidden dim</span>
          <span>
            {gatStatus?.heads != null && gatStatus?.hidden_dim != null
              ? `${gatStatus.heads} / ${gatStatus.hidden_dim}`
              : 'n/a'}
          </span>
        </div>
        {gatModel?.metrics?.accuracy != null && (
          <div className="msv2-row">
            <span>GAT accuracy</span>
            <span>
              {gatModel.metrics.accuracy.toFixed(4)}
              {gatModel.independent === false ? ' (aliased)' : ''}
            </span>
          </div>
        )}
        {graphModel?.metrics?.accuracy != null && (
          <div className="msv2-row">
            <span>GNN accuracy</span>
            <span>{graphModel.metrics.accuracy.toFixed(4)}</span>
          </div>
        )}
        <div className="msv2-row"><span>Cascade predictions</span><span>{cascadePred}</span></div>
        <div className="msv2-row"><span>Mean cascade depth</span><span>{Number(meanDepth).toFixed(1)}</span></div>
      </div>

      {/* Section 4: Meta-Propagator + RLHF */}
      <div className="msv2-section">
        <div className="msv2-section-title">Meta-Propagator + RLHF</div>

        {pipelineOff && <div className="msv2-note">{PIPELINE_OFF_NOTE}</div>}
        <div className="msv2-row"><span>Sats with corrections</span><span>{fmt(metaSats)}</span></div>
        <div className="msv2-row">
          <span>Mean improvement vs SGP4</span>
          <span>{metaImprovement != null ? `${Number(metaImprovement).toFixed(1)}%` : '—'}</span>
        </div>
        <div className="msv2-row"><span>Total corrections</span><span>{fmt(metaCorrections)}</span></div>

        <div style={{ height: 6 }} />

        <div className="msv2-row"><span>RLHF decisions</span><span>{fmt(rlhfDecisions)}</span></div>
        <div className="msv2-row">
          <span>Approval rate</span>
          <span>
            {approvalRate != null ? `${(approvalRate * 100).toFixed(0)}%` : '—'}
          </span>
        </div>
        <div className="msv2-row"><span>RLHF rounds</span><span>{fmt(rlhfRounds)}</span></div>

        {/* Approval rate sparkline dots */}
        {rlhfRoundRates.length > 0 && (
          <div className="msv2-rlhf-dots">
            {rlhfRoundRates.slice(-10).map((r, i, shown) => {
              const round = rlhfRoundRates.length - shown.length + i + 1;
              return (
                <div
                  key={round}
                  className="msv2-rlhf-dot"
                  style={{ height: `${Math.max(2, r * 28)}px` }}
                  title={`Round ${round}: ${(r * 100).toFixed(0)}%`}
                />
              );
            })}
          </div>
        )}
      </div>

      {/* Bottom: data accumulation */}
      <div className="msv2-section">
        <div className="msv2-section-title">Training Data Accumulation</div>
        <div className="msv2-row"><span>Conjunction samples</span><span>{conjSamples.toLocaleString()}</span></div>
        <div className="msv2-row">
          <span>Ready to retrain (LSTM buffer)</span>
          <span style={{ color: readyToRetrain ? 'var(--alert-green)' : 'var(--text-dim)' }}>
            {readyToRetrain == null
              ? '—'
              : readyToRetrain
                ? 'YES'
                : `NO (${bufferFill.toLocaleString()}/${bufferThreshold.toLocaleString()})`}
          </span>
        </div>
        <div className="msv2-row">
          <span>Shadow retrain</span>
          <span>{shadow == null ? '—' : shadow.enabled ? 'ENABLED' : 'DISABLED'}</span>
        </div>
        <div className="msv2-row">
          <span>Last retrain</span>
          <span>{shadow == null ? '—' : (shadow.last_retrain ?? 'never')}</span>
        </div>
        {pipelineOff && <div className="msv2-note">{PIPELINE_OFF_NOTE}</div>}
        {mlStatusReachable === false && (
          <div className="msv2-note">ML status endpoint unreachable</div>
        )}

        <button
          type="button"
          className="msv2-btn"
          onClick={triggerRetrain}
          disabled={retrainBusy || pipelineEnabled !== true}
          title={pipelineOff ? PIPELINE_OFF_NOTE : undefined}
        >
          {retrainBusy ? 'RETRAINING…' : 'RETRAIN NOW'}
        </button>
        {retrainMsg && (
          <div className={`msv2-retrain-msg ${retrainMsg.ok ? 'ok' : 'err'}`}>{retrainMsg.text}</div>
        )}

        {/* Sample count sparkline */}
        {sampleHistory.length > 1 && (
          <svg className="msv2-sparkline" width={sparkW} height={sparkH} viewBox={`0 0 ${sparkW} ${sparkH}`}>
            <polyline
              points={sparkPoints}
              fill="none"
              stroke="var(--alert-green)"
              strokeWidth={1.5}
              opacity={0.7}
            />
          </svg>
        )}
      </div>
    </div>
  );
}
