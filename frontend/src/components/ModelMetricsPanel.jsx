import useStore from '../store/useStore';
import '../styles/drawers.css';

const formatMetric = (value) => {
  if (value == null || Number.isNaN(Number(value))) return '---';
  return Number(value).toFixed(3);
};

const MetricRow = ({ label, value }) => (
  <div className="dr-kv">
    <span className="dr-kv-key">{label}</span>
    <span className="dr-kv-val">{value}</span>
  </div>
);

const ModelBlock = ({ item, rows }) => (
  <div className="dr-model">
    <div className="dr-model-head">
      <span className="dr-model-name">{item.name}</span>
      <span className="dr-model-type">{item.type}</span>
    </div>
    <div className="dr-kv-grid">
      {rows.map(([label, key]) => (
        <MetricRow key={key} label={label} value={formatMetric(item.metrics?.[key])} />
      ))}
    </div>
  </div>
);

const CLASSIFICATION_ROWS = [['Accuracy', 'accuracy'], ['Precision', 'precision'], ['Recall', 'recall'], ['F1', 'f1']];
const REGRESSION_ROWS = [['MAE', 'mae'], ['RMSE', 'rmse']];

export default function ModelMetricsPanel() {
  const modelMetrics = useStore((s) => s.modelMetrics);
  const ready = Boolean(modelMetrics?.generated_at);

  return (
    <div className="dr-metrics">
      <div className="dr-row-status">
        <span className="ui-label">Status</span>
        <span className={`ui-status ${ready ? 'is-nominal' : 'is-dim'}`}>
          {ready ? 'Ready' : 'Loading'}
        </span>
      </div>

      {!modelMetrics ? (
        <div className="dr-empty">Loading model metrics</div>
      ) : (
        <>
          <section className="dr-section">
            <div className="dr-section-title ui-label">Classification Models</div>
            {(modelMetrics.classification_models || []).map((item) => (
              <ModelBlock key={item.name} item={item} rows={CLASSIFICATION_ROWS} />
            ))}
          </section>

          <section className="dr-section">
            <div className="dr-section-title ui-label">Regression Models</div>
            {(modelMetrics.regression_models || []).map((item) => (
              <ModelBlock key={item.name} item={item} rows={REGRESSION_ROWS} />
            ))}
          </section>
        </>
      )}
    </div>
  );
}
