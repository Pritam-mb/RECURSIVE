import { useMemo, useState, useEffect } from 'react';
import useTestMode from '../hooks/useTestMode';
import '../styles/sim.css';

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

  const verdictClass = comparison
    ? { PASS: 'is-nominal', PARTIAL: 'is-caution', FAIL: 'is-warning' }[comparison.verdict]
    : '';

  return (
    <section className="sim-section">
      <div className="sim-section-head">
        <span className="ui-label sim-section-title">Conjunction Monitor</span>
        <div className="sim-section-head-aside">
          {testActive && <span className="ui-status is-nominal">Live</span>}
          <button
            className="ui-btn sim-mini"
            type="button"
            aria-expanded={showThresholds}
            onClick={() => setShowThresholds((prev) => !prev)}
          >
            {showThresholds ? 'Hide' : 'Thresholds'}
          </button>
        </div>
      </div>

      <div className="sim-body">
        {showThresholds && (
          <div className="sim-fields">
            <label className="sim-field">
              <span className="sim-field-label">TCA threshold (s)</span>
              <input
                className="ui-input sim-input"
                value={thresholds.tcaSeconds}
                onChange={(e) => setThresholds({ tcaSeconds: Number(e.target.value) })}
              />
            </label>
            <label className="sim-field">
              <span className="sim-field-label">Miss distance threshold (%)</span>
              <input
                className="ui-input sim-input"
                value={thresholds.missDistancePct}
                onChange={(e) => setThresholds({ missDistancePct: Number(e.target.value) })}
              />
            </label>
            <label className="sim-field">
              <span className="sim-field-label">Velocity threshold (%)</span>
              <input
                className="ui-input sim-input"
                value={thresholds.velocityPct}
                onChange={(e) => setThresholds({ velocityPct: Number(e.target.value) })}
              />
            </label>
          </div>
        )}

        <div className="sim-subhead">
          <span className="ui-label">Closest approach</span>
        </div>
        <dl className="sim-dl">
          <dt>Current separation</dt>
          <dd>{separationKm != null ? separationKm.toFixed(2) : '---'} km</dd>
          <dt>Closing rate</dt>
          <dd className={closingRateKms != null && closingRateKms < 0 ? 'is-caution' : ''}>
            {closingRateKms != null ? closingRateKms.toFixed(3) : '---'} km/s
          </dd>
          <dt>Time to predicted TCA</dt>
          <dd>{formatDuration(formattedTcaSeconds)}</dd>
          <dt>Minimum separation so far</dt>
          <dd>{computed ? `${computed.missDistanceM.toFixed(2)} m` : '---'}</dd>
          <dt>Computed TCA time</dt>
          <dd>{computed ? computed.tcaUtc : '---'}</dd>
        </dl>

        {computed && (
          <>
            <div className="sim-subhead">
              <span className="ui-label">Simulation result</span>
            </div>
            <dl className="sim-dl">
              <dt>Computed miss distance</dt>
              <dd>{computed.missDistanceM.toFixed(2)} m</dd>
              <dt>Computed TCA time</dt>
              <dd>{computed.tcaUtc}</dd>
              <dt>Computed rel. velocity</dt>
              <dd>{computed.relVelocityKms.toFixed(3)} km/s</dd>
              <dt>Collision detected</dt>
              <dd className={computed.collisionDetected ? 'is-warning' : 'is-nominal'}>
                {computed.collisionDetected ? 'YES' : 'NO'}
              </dd>
            </dl>
          </>
        )}

        {comparison && (
          <>
            <div className="sim-subhead">
              <span className="ui-label">Prediction comparison</span>
            </div>
            <table className="sim-table">
              <thead>
                <tr>
                  <th className="ui-label">Metric</th>
                  <th className="ui-label">Model</th>
                  <th className="ui-label">Sim</th>
                  <th className="ui-label">Δ</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Miss distance</td>
                  <td>{prediction.missDistanceM || '---'} m</td>
                  <td>{computed.missDistanceM.toFixed(2)} m</td>
                  <td className={comparison.passMiss ? 'is-nominal' : 'is-caution'}>
                    {comparison.missPct != null ? `${comparison.missPct.toFixed(1)}%` : '---'}
                  </td>
                </tr>
                <tr>
                  <td>TCA timestamp</td>
                  <td title={prediction.tcaUtc || undefined}>{prediction.tcaUtc || '---'}</td>
                  <td title={computed.tcaUtc}>{computed.tcaUtc}</td>
                  <td className={comparison.passTca ? 'is-nominal' : 'is-caution'}>
                    {comparison.tcaDiff != null ? `${comparison.tcaDiff.toFixed(0)}s` : '---'}
                  </td>
                </tr>
                <tr>
                  <td>Rel. velocity</td>
                  <td>{prediction.relVelocityKms || '---'} km/s</td>
                  <td>{computed.relVelocityKms.toFixed(3)} km/s</td>
                  <td className={comparison.passVel ? 'is-nominal' : 'is-caution'}>
                    {comparison.velPct != null ? `${comparison.velPct.toFixed(1)}%` : '---'}
                  </td>
                </tr>
                <tr>
                  <td>Collision</td>
                  <td>{prediction.predictedCollision === 'yes' ? 'YES' : 'NO'}</td>
                  <td>{computed.collisionDetected ? 'YES' : 'NO'}</td>
                  <td className={comparison.passCollision ? 'is-nominal' : 'is-caution'}>
                    {comparison.passCollision ? 'PASS' : 'FAIL'}
                  </td>
                </tr>
              </tbody>
            </table>

            <div className="sim-verdict">
              <span className="ui-label">Overall verdict</span>
              <span className={`ui-status ${verdictClass}`}>{comparison.verdict}</span>
            </div>
            <div className="sim-note">
              Thresholds: TCA +/-{thresholds.tcaSeconds}s, Miss +/-{thresholds.missDistancePct}%, Vel +/-{thresholds.velocityPct}%
            </div>
          </>
        )}
      </div>
    </section>
  );
};

export default ConjunctionMonitor;
