import useStore from '../store/useStore';
import ModelMetricsPanel from './ModelMetricsPanel';
import '../styles/drawers.css';

// Opened/closed from the header "Scorecard" toggle.
export default function MetricsDrawer() {
  const open = useStore((s) => s.metricsDrawerOpen);
  const setOpen = useStore((s) => s.setMetricsDrawerOpen);

  if (!open) return null;

  return (
    <aside className="dr-sheet dr-sheet--left" aria-label="Model scorecard">
      <div className="dr-sheet-header">
        <span className="ui-label dr-sheet-title">Model Scorecard</span>
        <button
          type="button"
          className="dr-close"
          onClick={() => setOpen(false)}
          aria-label="Close model scorecard"
          title="Close"
        >
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
            <path d="M1 1l8 8M9 1l-8 8" stroke="currentColor" strokeWidth="1.2" />
          </svg>
        </button>
      </div>

      <div className="dr-sheet-body">
        <ModelMetricsPanel />
      </div>
    </aside>
  );
}
