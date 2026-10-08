import useStore from '../store/useStore';
import TestModePanel from './TestModePanel';
import ConjunctionPredictor from './ConjunctionPredictor';
import ConjunctionMonitor from './ConjunctionMonitor';
import ManualControl from './Control/ManualControl';

export default function SimulationDrawer() {
  const open = useStore((s) => s.simulationDrawerOpen);
  const setOpen = useStore((s) => s.setSimulationDrawerOpen);

  if (!open) {
    return (
      <button
        type="button"
        className="simulation-tab"
        onClick={() => setOpen(true)}
      >
        OPEN SIMULATION
      </button>
    );
  }

  return (
    <aside className="simulation-drawer">
      <div className="simulation-drawer-header">
        <span className="panel-title">Simulation Console</span>
        <button
          type="button"
          className="mini-btn"
          onClick={() => setOpen(false)}
        >
          Hide
        </button>
      </div>

      <div className="simulation-drawer-body">
        <ConjunctionPredictor />
        <TestModePanel />
        <ConjunctionMonitor />
        <ManualControl />
      </div>
    </aside>
  );
}