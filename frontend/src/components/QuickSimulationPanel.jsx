import { useState } from 'react';
import useTestMode from '../hooks/useTestMode';
import '../styles/drawers.css';

const QUICK_EPOCH = '2026-05-09T05:40:00Z';

export default function QuickSimulationPanel() {
  const [status, setStatus] = useState('');
  const testActive = useTestMode((s) => s.testActive);
  const setSelectedA = useTestMode((s) => s.setSelectedA);
  const setSelectedB = useTestMode((s) => s.setSelectedB);
  const setOverrideMode = useTestMode((s) => s.setOverrideMode);
  const setOverrideField = useTestMode((s) => s.setOverrideField);
  const setOverrideVectorField = useTestMode((s) => s.setOverrideVectorField);
  const setPredictionField = useTestMode((s) => s.setPredictionField);
  const setSimControl = useTestMode((s) => s.setSimControl);
  const setComputed = useTestMode((s) => s.setComputed);
  const setTestSatellites = useTestMode((s) => s.setTestSatellites);
  const setTestActive = useTestMode((s) => s.setTestActive);

  const loadCollisionDemo = () => {
    const satelliteA = {
      name: 'SAT-COLLIDER-A',
      norad_id: 99901,
      position_eci_km: [6800, 0, 0],
      velocity_eci_kms: [0, 7.6, 0],
      epoch_utc: QUICK_EPOCH,
    };
    const satelliteB = {
      name: 'SAT-COLLIDER-B',
      norad_id: 99902,
      position_eci_km: [6800, 0, 0],
      velocity_eci_kms: [0, 7.6, 0],
      epoch_utc: QUICK_EPOCH,
    };

    setSelectedA({ name: satelliteA.name, norad_id: satelliteA.norad_id });
    setSelectedB({ name: satelliteB.name, norad_id: satelliteB.norad_id });
    setOverrideMode('overrideA', 'override');
    setOverrideMode('overrideB', 'override');
    setOverrideVectorField('overrideA', 'position', 'x', '6800');
    setOverrideVectorField('overrideA', 'position', 'y', '0');
    setOverrideVectorField('overrideA', 'position', 'z', '0');
    setOverrideVectorField('overrideA', 'velocity', 'vx', '0');
    setOverrideVectorField('overrideA', 'velocity', 'vy', '7.6');
    setOverrideVectorField('overrideA', 'velocity', 'vz', '0');
    setOverrideField('overrideA', 'epochUtc', QUICK_EPOCH);
    setOverrideVectorField('overrideB', 'position', 'x', '6800');
    setOverrideVectorField('overrideB', 'position', 'y', '0');
    setOverrideVectorField('overrideB', 'position', 'z', '0');
    setOverrideVectorField('overrideB', 'velocity', 'vx', '0');
    setOverrideVectorField('overrideB', 'velocity', 'vy', '7.6');
    setOverrideVectorField('overrideB', 'velocity', 'vz', '0');
    setOverrideField('overrideB', 'epochUtc', QUICK_EPOCH);
    setPredictionField('tcaUtc', QUICK_EPOCH);
    setPredictionField('missDistanceM', '0');
    setPredictionField('relVelocityKms', '0');
    setPredictionField('collisionProbability', '1');
    setPredictionField('predictedCollision', 'yes');
    setSimControl({ startEpochUtc: QUICK_EPOCH, currentEpochUtc: QUICK_EPOCH, stepSeconds: 10, timeWarp: 1 });
    setTestSatellites({
      a: satelliteA,
      b: satelliteB,
    });
    setComputed({
      tcaUtc: QUICK_EPOCH,
      missDistanceM: 0,
      relVelocityKms: 0,
      collisionDetected: true,
    });
    setTestActive(true);
    setStatus('Collision demo loaded');
  };

  return (
    <div className="ui-panel dr-card">
      <div className="ui-panel-header">
        <span className="ui-label">Simulation Controls</span>
        <span className={`ui-status ${testActive ? 'is-caution' : 'is-dim'}`}>
          {testActive ? 'Test active' : 'Standby'}
        </span>
      </div>

      <div className="dr-card-body">
        <p className="dr-copy">Load the collision demo scenario or toggle test mode.</p>

        <div className="dr-actions">
          <button className="ui-btn ui-btn--primary" type="button" onClick={loadCollisionDemo}>
            Load Collision Demo
          </button>
          <button className="ui-btn" type="button" onClick={() => setTestActive(!testActive)}>
            Toggle Test Mode
          </button>
        </div>

        {status && <div className="dr-note" role="status">{status}</div>}
      </div>
    </div>
  );
}
