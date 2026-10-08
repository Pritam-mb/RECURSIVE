import useStore from '../store/useStore';
import ModelMetricsPanel from './ModelMetricsPanel';

export default function MetricsDrawer() {
  const open = useStore((s) => s.metricsDrawerOpen);
  const setOpen = useStore((s) => s.setMetricsDrawerOpen);

  if (!open) {
    return (
      <button
        type="button"
        className="metrics-tab"
        onClick={() => setOpen(true)}
      >
        OPEN SCORECARD
      </button>
    );
  }

  return (
    <aside className="metrics-drawer">
      <div className="metrics-drawer-header">
        <span className="panel-title">Model Scorecard</span>
        <button
          type="button"
          className="mini-btn"
          onClick={() => setOpen(false)}
        >
          Hide
        </button>
      </div>

      <div className="metrics-drawer-body">
        <ModelMetricsPanel />
      </div>
    </aside>
  );
}