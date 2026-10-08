import { useMemo, useState, useEffect } from 'react';
import useTestMode from '../hooks/useTestMode';

const formatDuration = (totalSeconds) => {
  if (totalSeconds == null || Number.isNaN(totalSeconds)) return '---';
  const absSeconds = Math.max(0, Math.floor(totalSeconds));
  const hours = String(Math.floor(absSeconds / 3600)).padStart(2, '0');
  const minutes = String(Math.floor((absSeconds % 3600) / 60)).padStart(2, '0');
  const seconds = String(absSeconds % 60).padStart(2, '0');
  return `${hours}:${minutes}:${seconds}`;
};

const ConjunctionMonitor = () => {
  const {
    testActive,
    prediction,
    computed,
    thresholds,
    separationKm,
    closingRateKms,
    timeToTcaSeconds,
    setSeparation,
    setThresholds,
  } = useTestMode();

  const [showThresholds, setShowThresholds] = useState(false);

  useEffect(() => {
    if (!testActive) {
      setSeparation({ separationKm: null, closingRateKms: null, timeToTcaSeconds: null });
      return;
    }

    const fetchSeparation = async () => {
      try {
        const res = await fetch('/api/test/separation');
        if (!res.ok) return;
        const data = await res.json();
        setSeparation({
          separationKm: data.separation_km,
          closingRateKms: data.closing_rate_kms,
          timeToTcaSeconds: data.time_to_tca_s,
        });
      } catch (e) {
        console.error('Failed to fetch separation:', e);
      }
    };

    fetchSeparation();
    const timer = setInterval(fetchSeparation, 1000);
    return () => clearInterval(timer);
  }, [testActive, setSeparation]);

  const formattedTcaSeconds = useMemo(() => timeToTcaSeconds, [timeToTcaSeconds]);

  const comparison = useMemo(() => {
    if (!computed) return null;

    const predictedMiss = Number(prediction.missDistanceM);
    const computedMiss = Number(computed.missDistanceM);
    const predictedVel = Number(prediction.relVelocityKms);
    const computedVel = Number(computed.relVelocityKms);

    const missPct = predictedMiss > 0
      ? Math.abs(computedMiss - predictedMiss) / predictedMiss * 100
      : null;
    const velPct = predictedVel > 0
      ? Math.abs(computedVel - predictedVel) / predictedVel * 100
      : null;

    const predictedTca = Date.parse(prediction.tcaUtc);
    const computedTca = Date.parse(computed.tcaUtc);
    const tcaDiff = Number.isNaN(predictedTca) || Number.isNaN(computedTca)
      ? null
      : Math.abs((computedTca - predictedTca) / 1000);

    const collisionMatch = prediction.predictedCollision
      ? (prediction.predictedCollision === 'yes') === Boolean(computed.collisionDetected)
      : null;

    const passTca = tcaDiff != null && tcaDiff <= thresholds.tcaSeconds;
    const passMiss = missPct != null && missPct <= thresholds.missDistancePct;
    const passVel = velPct != null && velPct <= thresholds.velocityPct;
    const passCollision = collisionMatch === null ? false : collisionMatch;

    const passes = [passTca, passMiss, passVel, passCollision];
    const passCount = passes.filter(Boolean).length;

    let verdict = 'FAIL';
    if (passCount === passes.length) verdict = 'PASS';
    else if (passCount >= 2) verdict = 'PARTIAL';

    return {
      missPct,
      velPct,
      tcaDiff,
      collisionMatch,
      verdict,
      passTca,
      passMiss,
      passVel,
      passCollision,
    };
  }, [computed, prediction, thresholds]);

  return (
    <div className="panel monitor-panel">
      <div className="panel-header">
        <span className="panel-title">Conjunction Monitor</span>
        <button className="mini-btn" type="button" onClick={() => setShowThresholds((prev) => !prev)}>
          {showThresholds ? 'Hide' : 'Thresholds'}
        </button>
      </div>

      {showThresholds && (
        <div className="thresholds">
          <div className="field-row">
            <label>TCA Threshold (s)</label>
            <input
              className="field-input"
              value={thresholds.tcaSeconds}
              onChange={(e) => setThresholds({ tcaSeconds: Number(e.target.value) })}
            />
          </div>
          <div className="field-row">
            <label>Miss Distance Threshold (%)</label>
            <input
              className="field-input"
              value={thresholds.missDistancePct}
              onChange={(e) => setThresholds({ missDistancePct: Number(e.target.value) })}
            />
          </div>
          <div className="field-row">
            <label>Velocity Threshold (%)</label>
            <input
              className="field-input"
              value={thresholds.velocityPct}
              onChange={(e) => setThresholds({ velocityPct: Number(e.target.value) })}
            />
          </div>
        </div>
      )}

      <div className="monitor-card">
        <div className="monitor-title">Closest Approach Monitor</div>
        <div className="monitor-row">
          <span>Current Separation</span>
          <span className="mono">{separationKm != null ? separationKm.toFixed(2) : '---'} km</span>
        </div>
        <div className="monitor-row">
          <span>Closing Rate</span>
          <span className={`mono ${closingRateKms != null && closingRateKms < 0 ? 'danger' : ''}`}>
            {closingRateKms != null ? closingRateKms.toFixed(3) : '---'} km/s
          </span>
        </div>
        <div className="monitor-row">
          <span>Time to Predicted TCA</span>
          <span className="mono">{formatDuration(formattedTcaSeconds)}</span>
        </div>
        <div className="monitor-divider" />
        <div className="monitor-row">
          <span>Minimum Separation So Far</span>
          <span className="mono">
            {computed ? `${computed.missDistanceM.toFixed(2)} m` : '---'}
          </span>
        </div>
        <div className="monitor-row">
          <span>Computed TCA Time</span>
          <span className="mono">{computed ? computed.tcaUtc : '---'}</span>
        </div>
      </div>

      {computed && (
        <div className="result-card">
          <div className="monitor-title">Simulation Result</div>
          <div className="monitor-row">
            <span>Computed Miss Distance</span>
            <span className="mono">{computed.missDistanceM.toFixed(2)} m</span>
          </div>
          <div className="monitor-row">
            <span>Computed TCA Time</span>
            <span className="mono">{computed.tcaUtc}</span>
          </div>
          <div className="monitor-row">
            <span>Computed Rel. Velocity</span>
            <span className="mono">{computed.relVelocityKms.toFixed(3)} km/s</span>
          </div>
          <div className="monitor-row">
            <span>Collision Detected</span>
            <span className={`mono ${computed.collisionDetected ? 'danger' : 'success'}`}>
              {computed.collisionDetected ? 'YES' : 'NO'}
            </span>
          </div>
        </div>
      )}

      {comparison && (
        <div className="comparison">
          <div className="monitor-title">Prediction Comparison</div>
          <div className="comparison-table">
            <div className="comparison-row header">
              <span>Metric</span>
              <span>Model</span>
              <span>Sim</span>
              <span>Status</span>
            </div>
            <div className="comparison-row">
              <span>Miss Distance</span>
              <span>{prediction.missDistanceM || '---'} m</span>
              <span>{computed.missDistanceM.toFixed(2)} m</span>
              <span className={comparison.passMiss ? 'success' : 'warning'}>
                {comparison.missPct != null ? `${comparison.missPct.toFixed(1)}%` : '---'}
              </span>
            </div>
            <div className="comparison-row">
              <span>TCA Timestamp</span>
              <span>{prediction.tcaUtc || '---'}</span>
              <span>{computed.tcaUtc}</span>
              <span className={comparison.passTca ? 'success' : 'warning'}>
                {comparison.tcaDiff != null ? `${comparison.tcaDiff.toFixed(0)}s` : '---'}
              </span>
            </div>
            <div className="comparison-row">
              <span>Rel. Velocity</span>
              <span>{prediction.relVelocityKms || '---'} km/s</span>
              <span>{computed.relVelocityKms.toFixed(3)} km/s</span>
              <span className={comparison.passVel ? 'success' : 'warning'}>
                {comparison.velPct != null ? `${comparison.velPct.toFixed(1)}%` : '---'}
              </span>
            </div>
            <div className="comparison-row">
              <span>Collision</span>
              <span>{prediction.predictedCollision === 'yes' ? 'YES' : 'NO'}</span>
              <span>{computed.collisionDetected ? 'YES' : 'NO'}</span>
              <span className={comparison.passCollision ? 'success' : 'warning'}>
                {comparison.passCollision ? 'PASS' : 'FAIL'}
              </span>
            </div>
          </div>

          <div className={`verdict ${comparison.verdict.toLowerCase()}`}>
            Overall Verdict: {comparison.verdict}
          </div>
          <div className="threshold-note mono">
            Thresholds: TCA +/-{thresholds.tcaSeconds}s, Miss +/-{thresholds.missDistancePct}%, Vel +/-{thresholds.velocityPct}%
          </div>
        </div>
      )}
    </div>
  );
};

export default ConjunctionMonitor;
