import useStore from '../store/useStore';
import TestModePanel from './TestModePanel';
import ConjunctionPredictor from './ConjunctionPredictor';
import ConjunctionMonitor from './ConjunctionMonitor';
import ManualControl from './Control/ManualControl';
import '../styles/drawers.css';

// Opened/closed from the header "Simulation" toggle.
export default function SimulationDrawer() {
  const open = useStore((s) => s.simulationDrawerOpen);
  const setOpen = useStore((s) => s.setSimulationDrawerOpen);

  if (!open) return null;

  return (
    <aside className="dr-sheet dr-sheet--right" aria-label="Simulation console">
      <div className="dr-sheet-header">
        <span className="ui-label dr-sheet-title">Simulation Console</span>
        <button
          type="button"
          className="dr-close"
          onClick={() => setOpen(false)}
          aria-label="Close simulation console"
          title="Close"
        >
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
            <path d="M1 1l8 8M9 1l-8 8" stroke="currentColor" strokeWidth="1.2" />
          </svg>
        </button>
      </div>

      <div className="dr-sheet-body dr-sheet-body--stack">
        <ConjunctionPredictor />
        <TestModePanel />
        <ConjunctionMonitor />
        <ManualControl />
      </div>
    </aside>
  );
}
