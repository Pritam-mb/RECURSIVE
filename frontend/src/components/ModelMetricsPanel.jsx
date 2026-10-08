import useStore from '../store/useStore';

const formatMetric = (value) => {
  if (value == null || Number.isNaN(Number(value))) return '---';
  return Number(value).toFixed(3);
};

const MetricRow = ({ label, value }) => (
  <div className="monitor-row">
    <span>{label}</span>
    <span className="mono">{value}</span>
  </div>
);

export default function ModelMetricsPanel() {
  const modelMetrics = useStore((s) => s.modelMetrics);

  return (
    <div className="panel metrics-panel">
      <div className="panel-header">
        <span className="panel-title">Model Scorecard</span>
        <span className="mono" style={{ fontSize: 11 }}>
          {modelMetrics?.generated_at ? 'READY' : 'LOADING'}
        </span>
      </div>

      {!modelMetrics ? (
        <div className="no-data">LOADING MODEL METRICS</div>
      ) : (
        <>
          <div className="monitor-card">
            <div className="monitor-title">Classification Models</div>
            {(modelMetrics.classification_models || []).map((item) => (
              <div key={item.name} style={{ marginBottom: 8 }}>
                <div className="monitor-row">
                  <span>{item.name}</span>
                  <span className="mono">{item.type}</span>
                </div>
                <MetricRow label="Accuracy" value={formatMetric(item.metrics?.accuracy)} />
                <MetricRow label="Precision" value={formatMetric(item.metrics?.precision)} />
                <MetricRow label="Recall" value={formatMetric(item.metrics?.recall)} />
                <MetricRow label="F1" value={formatMetric(item.metrics?.f1)} />
              </div>
            ))}
          </div>

          <div className="monitor-card" style={{ marginTop: 12 }}>
            <div className="monitor-title">Regression Models</div>
            {(modelMetrics.regression_models || []).map((item) => (
              <div key={item.name} style={{ marginBottom: 8 }}>
                <div className="monitor-row">
                  <span>{item.name}</span>
                  <span className="mono">{item.type}</span>
                </div>
                <MetricRow label="MAE" value={formatMetric(item.metrics?.mae)} />
                <MetricRow label="RMSE" value={formatMetric(item.metrics?.rmse)} />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}