import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import useStore from '../store/useStore';
import { apiGet } from '../utils/api';
import '../styles/drawers.css';
import '../styles/analytics.css';

// Judge-facing analytics. Every number is read from the backend
// (/api/analytics/model, /api/physics/validation, /api/physics/engines).
// Missing fields render "—" / "not computed"; nothing is synthesised here
// except presentation (sorting, colour mapping, top-|loading| summaries).

const DASH = '—';

function num(v) {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

/** Compact significant-figure formatter for mixed-magnitude values. */
function sig(v, p = 4) {
  const n = num(v);
  if (n == null) return DASH;
  if (n === 0) return '0';
  const a = Math.abs(n);
  if (a >= 1e5 || a < 1e-3) return n.toExponential(2);
  return String(Number(n.toPrecision(p)));
}

function fixed(v, d) {
  const n = num(v);
  return n == null ? DASH : n.toFixed(d);
}

function pct(v, d = 1) {
  const n = num(v);
  return n == null ? DASH : `${(n * 100).toFixed(d)}%`;
}

/** Accept either an array or {key: [...]} wrapper. */
function asList(data, ...keys) {
  if (Array.isArray(data)) return data;
  if (data && typeof data === 'object') {
    for (const k of keys) if (Array.isArray(data[k])) return data[k];
  }
  return null;
}

// ── Diverging colour scale: blue (−1) ↔ neutral (0) ↔ orange (+1) ──────────
const NEG = [63, 134, 240];
const POS = [240, 138, 36];
const MID = [22, 28, 37];
function divColor(v) {
  const n = num(v);
  if (n == null) return 'transparent';
  const t = Math.min(1, Math.abs(n)) ** 0.85;
  const end = n < 0 ? NEG : POS;
  const c = MID.map((m, i) => Math.round(m + (end[i] - m) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

async function fetchJson(path) {
  try {
    return { data: await apiGet(path), error: null };
  } catch (e) {
    return { data: null, error: e?.message || 'request failed' };
  }
}

// ── Section shell ──────────────────────────────────────────────────────────
function Section({ title, meta, right, children }) {
  return (
    <section className="ax-section">
      <div className="ax-section-head">
        <span className="ui-label ax-section-title">{title}</span>
        <div className="ax-toolbar">
          {meta && <span className="ax-section-meta">{meta}</span>}
          {right}
        </div>
      </div>
      <div className="ax-section-body">{children}</div>
    </section>
  );
}

function Empty({ children, error }) {
  return <div className={`ax-empty${error ? ' is-error' : ''}`}>{children}</div>;
}

function status(res, loading, what) {
  if (loading && !res) return <Empty>Loading {what}…</Empty>;
  if (res?.error) return <Empty error>{what}: endpoint unavailable ({res.error}) — not computed</Empty>;
  return null;
}

// ── a. Correlation heatmap ─────────────────────────────────────────────────
const CorrelationMatrix = memo(function CorrelationMatrix({ model }) {
  const [method, setMethod] = useState('pearson');
  const [hover, setHover] = useState(null);
  const features = useMemo(() => (Array.isArray(model?.features) ? model.features : []), [model]);
  const defs = model?.feature_definitions ?? {};
  const corr = useMemo(() => model?.correlation ?? {}, [model]);
  // The API matrix spans correlation.variables (the features plus the
  // target as the last variable); re-index it by feature name.
  const matrix = useMemo(() => {
    const raw = Array.isArray(corr[method]) ? corr[method] : null;
    if (!raw) return null;
    const vars = Array.isArray(corr.variables) ? corr.variables : null;
    if (!vars) return raw;
    const idx = features.map((f) => vars.indexOf(f));
    if (idx.some((k) => k < 0)) return null;
    return idx.map((i) => idx.map((j) => raw[i]?.[j]));
  }, [corr, method, features]);
  const withTarget = corr.with_target?.[method] ?? null;
  const n = features.length;
  const valid = matrix && n > 0 && matrix.length === n;

  const LBL = 150;
  const TOP = 92;
  const cell = n > 0 ? Math.max(16, Math.min(34, Math.floor(520 / n))) : 20;
  const hasTarget = withTarget && typeof withTarget === 'object';
  const rows = n + (hasTarget ? 1 : 0);
  const W = LBL + n * cell + 4;
  const H = TOP + rows * cell + (hasTarget ? 6 : 0) + 4;
  const showVals = cell >= 26;

  const cells = useMemo(() => {
    if (!valid) return null;
    const out = [];
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const v = num(matrix[i]?.[j]);
        out.push(
          <g key={`${i}-${j}`}>
            <rect
              x={LBL + j * cell} y={TOP + i * cell} width={cell - 1} height={cell - 1}
              fill={v == null ? 'none' : divColor(v)}
              stroke={v == null ? 'var(--c-line)' : 'none'}
              onMouseEnter={() => setHover({ r: features[i], c: features[j], v })}
            />
            {showVals && v != null && (
              <text className="ax-cell-val" x={LBL + j * cell + cell / 2} y={TOP + i * cell + cell / 2 + 3} textAnchor="middle">
                {v.toFixed(2)}
              </text>
            )}
          </g>,
        );
      }
    }
    if (hasTarget) {
      const y = TOP + n * cell + 6;
      features.forEach((f, j) => {
        const v = num(withTarget[f]);
        out.push(
          <g key={`t-${f}`}>
            <rect
              x={LBL + j * cell} y={y} width={cell - 1} height={cell - 1}
              fill={v == null ? 'none' : divColor(v)}
              stroke="var(--c-text-bright)" strokeWidth={0.6}
              onMouseEnter={() => setHover({ r: 'TARGET (log10 Pc)', c: f, v })}
            />
            {showVals && v != null && (
              <text className="ax-cell-val" x={LBL + j * cell + cell / 2} y={y + cell / 2 + 3} textAnchor="middle">
                {v.toFixed(2)}
              </text>
            )}
          </g>,
        );
      });
    }
    return out;
  }, [valid, matrix, n, cell, features, hasTarget, withTarget, showVals]);

  const right = (
    <div className="ax-seg" role="group" aria-label="Correlation method">
      {['pearson', 'spearman'].map((m) => (
        <button key={m} type="button" aria-pressed={method === m} onClick={() => setMethod(m)}>{m}</button>
      ))}
    </div>
  );

  return (
    <Section title="Feature correlation" meta={n ? `${n} features` : null} right={right}>
      {!valid ? (
        <Empty>{method} correlation matrix not computed</Empty>
      ) : (
        <>
          <div className="ax-readout">
            {hover
              ? <>{hover.r} × {hover.c}: <b>{method === 'pearson' ? 'r' : 'ρ'} = {hover.v == null ? DASH : hover.v.toFixed(3)}</b></>
              : <span className="is-dim">Hover a cell for the coefficient. Bottom row = correlation with target.</span>}
          </div>
          <svg className="ax-svg" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: W }} onMouseLeave={() => setHover(null)} role="img" aria-label={`${method} correlation matrix`}>
            {features.map((f, i) => (
              <text key={`r-${f}`} x={LBL - 6} y={TOP + i * cell + cell / 2 + 3} textAnchor="end">
                <title>{defs[f] ?? f}</title>{f}
              </text>
            ))}
            {features.map((f, j) => (
              <text key={`c-${f}`} transform={`translate(${LBL + j * cell + cell / 2 + 3},${TOP - 6}) rotate(-55)`}>
                <title>{defs[f] ?? f}</title>{f}
              </text>
            ))}
            {hasTarget && (
              <text className="ax-lbl-target" x={LBL - 6} y={TOP + n * cell + 6 + cell / 2 + 3} textAnchor="end">target log10 Pc</text>
            )}
            {cells}
          </svg>
          <div className="ax-legend">
            <span>−1</span><span className="ax-legend-bar" /><span>+1</span>
            <span>blue = negative · orange = positive</span>
          </div>
        </>
      )}
    </Section>
  );
});

// ── b. PCA ─────────────────────────────────────────────────────────────────
function topLoadings(row, features, k = 3) {
  return row
    .map((v, i) => ({ f: features[i], v: num(v) }))
    .filter((x) => x.v != null && x.f)
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    .slice(0, k);
}

const PcaPanel = memo(function PcaPanel({ model }) {
  const pca = model?.pca ?? null;
  const features = Array.isArray(model?.features) ? model.features : [];
  const evr = Array.isArray(pca?.explained_variance_ratio) ? pca.explained_variance_ratio.map(num) : [];
  const cumulative = useMemo(() => {
    if (Array.isArray(pca?.cumulative) && pca.cumulative.length === evr.length) return pca.cumulative.map(num);
    return null;
  }, [pca, evr.length]);
  const n95 = num(pca?.n_components_95);
  const loadings = Array.isArray(pca?.loadings) ? pca.loadings : null;
  const interp = Array.isArray(pca?.interpretation) ? pca.interpretation
    : Array.isArray(pca?.interpretations) ? pca.interpretations : null;

  if (!pca || evr.length === 0) {
    return <Section title="Principal component analysis"><Empty>PCA not computed</Empty></Section>;
  }

  // Scree geometry
  const W = 520; const H = 190; const L = 36; const B = 22; const T = 10; const R = 10;
  const m = evr.length;
  const bw = (W - L - R) / m;
  const y = (v) => T + (H - T - B) * (1 - v);
  const cumPts = cumulative
    ? cumulative.map((v, i) => (v == null ? null : `${L + i * bw + bw / 2},${y(v)}`)).filter(Boolean).join(' ')
    : '';

  const k = Math.min(loadings ? loadings.length : 0, Math.max(2, Math.min(n95 ?? 3, 5)));

  return (
    <Section
      title="Principal component analysis"
      meta={pca.standardized ? 'standardized features' : null}
    >
      <div className="ax-stats">
        <div className="ax-stat">
          <span className="ui-label">Components for 95% variance</span>
          <span className="ax-stat-val">{n95 == null ? DASH : `${n95} of ${m}`}</span>
        </div>
        <div className="ax-stat">
          <span className="ui-label">PC1 explains</span>
          <span className="ax-stat-val">{pct(evr[0])}</span>
        </div>
      </div>
      <div className="ax-grid2">
        <div>
          <span className="ui-label">Scree plot</span>
          <svg className="ax-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="PCA scree plot">
            {[0, 0.25, 0.5, 0.75, 1].map((g) => (
              <g key={g}>
                <line className="ax-gridline" x1={L} x2={W - R} y1={y(g)} y2={y(g)} />
                <text x={L - 4} y={y(g) + 3} textAnchor="end">{g * 100}%</text>
              </g>
            ))}
            {evr.map((v, i) => (v == null ? null : (
              <rect key={i} x={L + i * bw + 2} y={y(v)} width={Math.max(1, bw - 4)} height={y(0) - y(v)}
                fill={n95 != null && i < n95 ? 'var(--c-accent)' : 'var(--c-text-faint)'} opacity={0.85}>
                <title>{`PC${i + 1}: ${pct(v, 2)} (cumulative ${cumulative ? pct(cumulative[i], 1) : DASH})`}</title>
              </rect>
            )))}
            <line x1={L} x2={W - R} y1={y(0.95)} y2={y(0.95)} stroke="var(--c-caution)" strokeDasharray="4 3" />
            <text x={W - R} y={y(0.95) - 3} textAnchor="end" style={{ fill: 'var(--c-caution)' }}>95%</text>
            {cumPts && <polyline points={cumPts} fill="none" stroke="var(--c-text-bright)" strokeWidth={1.4} />}
            {cumulative && cumulative.map((v, i) => (v == null ? null : (
              <circle key={i} cx={L + i * bw + bw / 2} cy={y(v)} r={2.2} fill="var(--c-text-bright)" />
            )))}
            {n95 != null && n95 >= 1 && n95 <= m && (
              <line x1={L + n95 * bw} x2={L + n95 * bw} y1={T} y2={y(0)} stroke="var(--c-caution)" strokeWidth={1} />
            )}
            <line className="ax-axis" x1={L} x2={W - R} y1={y(0)} y2={y(0)} />
            {evr.map((_, i) => (
              (m <= 12 || i % 2 === 0) && <text key={i} x={L + i * bw + bw / 2} y={H - 8} textAnchor="middle">{i + 1}</text>
            ))}
          </svg>
          <div className="ax-note">Bars: explained variance ratio per component · line: cumulative · dashed: 95% threshold.</div>
        </div>
        <div>
          <span className="ui-label">Loadings (top {k} components)</span>
          {!loadings || k === 0 ? <Empty>Loadings not computed</Empty> : (
            <div className="ax-table-wrap">
              <table className="ax-table">
                <thead>
                  <tr><th>Feature</th>{Array.from({ length: k }, (_, c) => <th key={c} className="num">PC{c + 1}</th>)}</tr>
                </thead>
                <tbody>
                  {features.map((f, i) => (
                    <tr key={f}>
                      <td title={model.feature_definitions?.[f] ?? f}>{f}</td>
                      {Array.from({ length: k }, (_, c) => {
                        const v = num(loadings[c]?.[i]);
                        return (
                          <td key={c} className="ax-heat-cell" style={{ background: divColor(v) }}>
                            {v == null ? DASH : v.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
      {loadings && k > 0 && (
        <div className="ax-note">
          {Array.from({ length: k }, (_, c) => {
            const given = interp?.[c];
            const text = typeof given === 'string' ? given
              : topLoadings(loadings[c] ?? [], features).map((x) => `${x.v >= 0 ? '+' : '−'}${x.f} (${Math.abs(x.v).toFixed(2)})`).join(', ');
            return (
              <div key={c}>
                <b>PC{c + 1}</b> ({pct(evr[c])}): {text || DASH}
                {typeof given !== 'string' && text ? <span className="is-dim"> — largest |loadings|</span> : null}
              </div>
            );
          })}
        </div>
      )}
    </Section>
  );
});

// ── c. PCA vs raw ──────────────────────────────────────────────────────────
function PcaVsRaw({ model }) {
  const cmp = model?.pca_vs_raw ?? null;
  if (!cmp) return <Section title="PCA vs raw features (XGBoost)"><Empty>Comparison not computed</Empty></Section>;
  const rows = [
    { name: 'XGBoost on raw features', d: cmp.raw_xgb },
    { name: `XGBoost on PCA${num(cmp.pca_xgb?.n_components) != null ? ` (${cmp.pca_xgb.n_components} comps)` : ''}`, d: cmp.pca_xgb },
  ];
  const maes = rows.map((r) => num(r.d?.mae_log10));
  const f1s = rows.map((r) => num(r.d?.f1_1e4));
  const bestMae = maes.every((v) => v != null) ? (maes[0] <= maes[1] ? 0 : 1) : -1;
  const bestF1 = f1s.every((v) => v != null) ? (f1s[0] >= f1s[1] ? 0 : 1) : -1;
  return (
    <Section title="PCA vs raw features (XGBoost)" meta="held-out">
      <table className="ax-table">
        <thead><tr><th>Model</th><th className="num">MAE (log10 Pc)</th><th className="num">F1 @ Pc ≥ 1e-4</th></tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.name}>
              <td>{r.name}</td>
              <td className={`num${bestMae === i ? ' is-nominal' : ''}`}>{fixed(maes[i], 3)}</td>
              <td className={`num${bestF1 === i ? ' is-nominal' : ''}`}>{fixed(f1s[i], 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="ax-callout">{cmp.verdict ? <><b>Verdict:</b> {cmp.verdict}</> : 'Verdict not provided'}</div>
    </Section>
  );
}

// ── d. Feature attribution ─────────────────────────────────────────────────
function normaliseGain(model) {
  const src = model?.split_importance?.gain ?? model?.gain_importance ?? model?.feature_importance_gain ?? model?.feature_importance
    ?? model?.card?.gain_importance ?? model?.card?.feature_importance ?? null;
  if (!src) return null;
  const out = {};
  if (Array.isArray(src)) {
    for (const r of src) {
      const v = num(r?.gain ?? r?.importance ?? r?.value);
      if (r?.feature && v != null) out[r.feature] = v;
    }
  } else if (typeof src === 'object') {
    for (const [f, v] of Object.entries(src)) if (num(v) != null) out[f] = num(v);
  }
  const total = Object.values(out).reduce((a, b) => a + b, 0);
  if (!(total > 0)) return null;
  for (const f of Object.keys(out)) out[f] /= total;
  return out;
}

const Attribution = memo(function Attribution({ model }) {
  const shap = useMemo(() => {
    const list = Array.isArray(model?.shap_global) ? model.shap_global : [];
    return list
      .map((r) => ({ f: r?.feature, v: num(r?.mean_abs_contribution_log10) }))
      .filter((r) => r.f && r.v != null)
      .sort((a, b) => b.v - a.v);
  }, [model]);
  const gain = useMemo(() => normaliseGain(model), [model]);
  const main = model?.main_factor ?? shap[0]?.f ?? null;
  if (shap.length === 0) {
    return <Section title="XGBoost feature attribution"><Empty>Global SHAP attribution not computed</Empty></Section>;
  }
  const maxV = shap[0].v || 1;
  const maxG = gain ? Math.max(...Object.values(gain)) : 1;
  return (
    <Section title="XGBoost feature attribution" meta={num(model?.sample_size) != null ? `n = ${model.sample_size}` : null}>
      {main && (
        <div className="ax-callout">
          <b>Main factor:</b> {main}
          {model?.feature_definitions?.[main] ? ` — ${model.feature_definitions[main]}` : ''}
        </div>
      )}
      <div className="ax-bars">
        <span className="ax-bars-head">Feature</span>
        <span className="ax-bars-head">Mean |SHAP| (decades of Pc)</span>
        <span className="ax-bars-head">Gain share{gain ? '' : ' — not provided'}</span>
        {shap.map((r) => {
          const g = gain ? num(gain[r.f]) : null;
          return (
            <div key={r.f} style={{ display: 'contents' }}>
              <span className={`ax-bar-name${r.f === main ? ' is-main' : ''}`} title={model?.feature_definitions?.[r.f] ?? r.f}>{r.f}</span>
              <span className="ax-bar">
                <span className="ax-bar-fill" style={{ width: `${(r.v / maxV) * 100}%` }} />
                <span className="ax-bar-val">{r.v.toFixed(3)}</span>
              </span>
              <span className="ax-bar">
                {g != null && <span className="ax-bar-fill is-gain" style={{ width: `${(g / maxG) * 100}%` }} />}
                <span className="ax-bar-val">{g == null ? DASH : pct(g)}</span>
              </span>
            </div>
          );
        })}
      </div>
      <div className="ax-note">
        SHAP = exact TreeSHAP contributions in log10 Pc units (how many decades a feature moves the prediction, on average).
        Gain = XGBoost split-gain share; it measures how often/usefully a feature splits, not its effect size.
      </div>
    </Section>
  );
});

// ── e. Physics cross-validation ────────────────────────────────────────────
function PhysicsValidation({ res, loading }) {
  const checks = asList(res?.data, 'checks', 'results', 'validation');
  const st = status(res, loading, 'Physics validation');
  const passed = checks ? checks.filter((c) => c?.pass === true).length : 0;
  return (
    <Section
      title="Physics cross-validation"
      meta={checks ? `${passed}/${checks.length} pass` : null}
    >
      {st ?? (!checks || checks.length === 0 ? <Empty>No validation checks returned</Empty> : (
        <div className="ax-table-wrap">
          <table className="ax-table">
            <thead>
              <tr>
                <th>Check</th><th>Standard formula</th><th className="num">Ours</th><th className="num">Reference</th>
                <th className="num">Abs err</th><th className="num">Rel err</th><th className="num">Tol</th><th>Result</th>
              </tr>
            </thead>
            <tbody>
              {checks.map((c, i) => (
                <tr key={`${c?.name ?? 'check'}-${i}`}>
                  <td>
                    {c?.name ?? DASH}
                    {c?.source && <div className="ax-note" style={{ fontSize: 'var(--fs-micro)' }}>{c.source}</div>}
                  </td>
                  <td className="mono">{c?.standard_formula ?? DASH}</td>
                  <td className="num">{sig(c?.our_value)}</td>
                  <td className="num">{sig(c?.reference_value)}</td>
                  <td className="num">{sig(c?.abs_error, 3)}</td>
                  <td className="num">{num(c?.rel_error) == null || num(c?.reference_value) === 0 || Math.abs(num(c.rel_error)) > 1e6
                    ? DASH : `${(num(c.rel_error) * 100).toPrecision(3)}%`}</td>
                  <td className="num">{sig(c?.tolerance, 3)}</td>
                  <td>
                    {c?.pass === true ? <span className="ax-chip is-pass">PASS</span>
                      : c?.pass === false ? <span className="ax-chip is-fail">FAIL</span>
                        : <span className="ax-chip is-na">n/a</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </Section>
  );
}

// ── f. Propulsion catalogue ────────────────────────────────────────────────
function EngineCatalogue({ res, loading }) {
  const engines = asList(res?.data, 'engines', 'catalogue', 'catalog');
  const st = status(res, loading, 'Propulsion catalogue');
  return (
    <Section title="Propulsion catalogue" meta={engines ? `${engines.length} engines` : null}>
      {st ?? (!engines || engines.length === 0 ? <Empty>No engines returned</Empty> : (
        <div className="ax-table-wrap">
          <table className="ax-table">
            <thead>
              <tr><th>Engine</th><th>Family</th><th className="num">Isp (s)</th><th className="num">Thrust (N)</th><th>Propellant</th><th>Note</th></tr>
            </thead>
            <tbody>
              {engines.map((e, i) => (
                <tr key={`${e?.engine ?? e?.name ?? 'e'}-${i}`}>
                  <td>{e?.engine ?? e?.name ?? DASH}</td>
                  <td>{e?.family ?? DASH}</td>
                  <td className="num">{sig(e?.isp_s)}</td>
                  <td className="num">{sig(e?.thrust_n)}</td>
                  <td>{e?.propellant ?? DASH}</td>
                  <td className="src">{e?.note ?? e?.source ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </Section>
  );
}

// ── g. Decision score definition ───────────────────────────────────────────
function DecisionDefinition({ model, validation }) {
  // Weights/thresholds come from a live alert's decision block; documented
  // text, if the API supplies it, from the analytics/physics payloads.
  const decision = useStore((s) => {
    for (const a of s.alerts) if (a?.decision && Array.isArray(a.decision.components)) return a.decision;
    return null;
  });
  const doc = model?.decision_score ?? model?.decision_definition ?? validation?.decision_score
    ?? validation?.decision_definition ?? null;
  const docText = typeof doc === 'string' ? doc : (doc?.formula ?? doc?.description ?? null);
  const comps = decision?.components ?? null;
  const th = decision?.thresholds ?? (doc && typeof doc === 'object' ? doc.thresholds : null) ?? null;
  const wsum = comps ? comps.reduce((a, c) => a + (num(c?.weight) ?? 0), 0) : null;

  return (
    <Section title="Decision score definition">
      <div className="ax-note">
        score (0–100) = Σ points<sub>i</sub>, points<sub>i</sub> = 100 · w<sub>i</sub> · n<sub>i</sub>, with each input normalised to n<sub>i</sub> ∈ [0, 1].
        Action is set by the physics Pc thresholds below.
      </div>
      {docText && <div className="ax-callout">{docText}</div>}
      {!comps ? <Empty>No alert currently carries a decision block — weights not available</Empty> : (
        <table className="ax-table">
          <thead><tr><th>Component</th><th>Source</th><th className="num">Weight</th></tr></thead>
          <tbody>
            {comps.map((c, i) => (
              <tr key={`${c?.name}-${i}`}>
                <td>{c?.name ?? DASH}</td><td>{c?.source ?? DASH}</td><td className="num">{fixed(c?.weight, 3)}</td>
              </tr>
            ))}
            <tr><td><b>Σ weights</b></td><td /><td className="num">{fixed(wsum, 3)}</td></tr>
          </tbody>
        </table>
      )}
      {th ? (
        <table className="ax-table">
          <thead><tr><th>Action</th><th className="num">Physics Pc ≥</th></tr></thead>
          <tbody>
            <tr><td className="is-warning">MANOEUVRE</td><td className="num">{sig(th.manoeuvre_pc)}</td></tr>
            <tr><td className="is-caution">PREPARE</td><td className="num">{sig(th.prepare_pc)}</td></tr>
            <tr><td style={{ color: 'var(--c-info)' }}>MONITOR</td><td className="num">{sig(th.monitor_pc)}</td></tr>
            <tr><td className="is-dim">NONE</td><td className="num">below monitor</td></tr>
          </tbody>
        </table>
      ) : <Empty>Thresholds not provided</Empty>}
      {th?.source && <div className="ax-note">Source: {th.source}</div>}
    </Section>
  );
}

function ModelCard({ model }) {
  const card = model?.card && typeof model.card === 'object' ? model.card : null;
  if (!card) return null;
  const entries = Object.entries(card).filter(([, v]) => v == null || ['string', 'number', 'boolean'].includes(typeof v)).slice(0, 14);
  if (entries.length === 0) return null;
  return (
    <Section title="Model card">
      <div className="dr-kv-grid">
        {entries.map(([k, v]) => (
          <div className="dr-kv" key={k}>
            <span className="dr-kv-key">{k.replace(/_/g, ' ')}</span>
            <span className="dr-kv-val">{typeof v === 'number' ? sig(v) : v == null ? DASH : String(v)}</span>
          </div>
        ))}
      </div>
    </Section>
  );
}

// ── Sheet (only mounted while open → fetches only when open) ──────────────
function AnalyticsSheet({ onClose }) {
  const [model, setModel] = useState(null);
  const [validation, setValidation] = useState(null);
  const [engines, setEngines] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fetchedAt, setFetchedAt] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    const [m, v, e1] = await Promise.all([
      fetchJson('/api/analytics/model'),
      fetchJson('/api/physics/validation'),
      fetchJson('/api/physics/engines'),
    ]);
    // Contract allows the engine list under either path.
    const e = e1.error ? await fetchJson('/api/propulsion/engines') : e1;
    setModel(m);
    setValidation(v);
    setEngines(e.error ? e1 : e);
    setFetchedAt(new Date());
    setLoading(false);
  }, []);

  useEffect(() => {
    // One fetch per open; the Refresh button re-runs it.
    load();
  }, [load]);

  const md = model?.data ?? null;
  const modelStatus = status(model, loading, 'Model analytics');

  return (
    <aside className="dr-sheet dr-sheet--right ax-sheet" aria-label="Analytics">
      <div className="dr-sheet-header">
        <span className="ui-label dr-sheet-title">Analytics · model &amp; physics evidence</span>
        <div className="ax-toolbar">
          <span className="ax-section-meta">
            {loading ? 'refreshing…' : fetchedAt ? `fetched ${fetchedAt.toISOString().substring(11, 19)}Z` : ''}
          </span>
          <button type="button" className="ui-btn" onClick={load} disabled={loading} title="Re-fetch analytics">Refresh</button>
          <button type="button" className="dr-close" onClick={onClose} aria-label="Close analytics" title="Close">
            <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
              <path d="M1 1l8 8M9 1l-8 8" stroke="currentColor" strokeWidth="1.2" />
            </svg>
          </button>
        </div>
      </div>

      <div className="dr-sheet-body ax-body">
        {modelStatus ? (
          <Section title="Model analytics">{modelStatus}</Section>
        ) : (
          <>
            <CorrelationMatrix model={md} />
            <PcaPanel model={md} />
            <PcaVsRaw model={md} />
            <Attribution model={md} />
          </>
        )}
        <PhysicsValidation res={validation} loading={loading} />
        <EngineCatalogue res={engines} loading={loading} />
        <DecisionDefinition model={md} validation={validation?.data && !Array.isArray(validation.data) ? validation.data : null} />
        {!modelStatus && <ModelCard model={md} />}
      </div>
    </aside>
  );
}

// Opened/closed from the header "Analytics" toggle.
export default function AnalyticsDrawer() {
  const open = useStore((s) => s.analyticsDrawerOpen);
  const setOpen = useStore((s) => s.setAnalyticsDrawerOpen);
  const close = useCallback(() => setOpen(false), [setOpen]);
  if (!open) return null;
  return <AnalyticsSheet onClose={close} />;
}
