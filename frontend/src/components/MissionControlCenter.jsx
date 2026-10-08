import { useState, useEffect } from 'react';
import ThreatCard from './ThreatCard';
import useStore from '../store/useStore';
import { apiPost, apiDelete } from '../utils/api';
import { markEdgeResolved } from '../utils/cascadeGraph';

const TABS = ['ALL', 'CRITICAL', 'WARNING', 'WATCH'];

export default function MissionControlCenter({ alerts = [] }) {
  const [activeTab, setActiveTab] = useState('ALL');
  const selectedAlertId = useStore((s) => s.selectedAlertId);
  const wsConnected = useStore((s) => s.wsConnected);
  const modelMetrics = useStore((s) => s.modelMetrics);
  const setCascadeGraph = useStore((s) => s.setCascadeGraph);
  const cascadeGraph = useStore((s) => s.cascadeGraph);
  const rankerReview = useStore((s) => s.rankerReview);

  const filteredAlerts = alerts
    .filter((a) => {
      if (activeTab === 'ALL') return true;
      const cpi = Number(a.cpi_score ?? 0);
      if (activeTab === 'CRITICAL') return cpi >= 8 || a.severity?.toUpperCase() === 'CRITICAL';
      if (activeTab === 'WARNING') return (cpi >= 5 && cpi < 8) || a.severity?.toUpperCase() === 'WARNING';
      if (activeTab === 'WATCH') return cpi < 5 && a.severity?.toUpperCase() !== 'CRITICAL' && a.severity?.toUpperCase() !== 'WARNING';
      return true;
    })
    .sort((a, b) => Number(b.cpi_score ?? 0) - Number(a.cpi_score ?? 0));

  // System status indicators
  const pipeline = wsConnected;
  const mlTrained = (modelMetrics?.classification_models ?? []).some(
    (m) => (m.metrics?.samples ?? 0) > 0
  );
  // Physics health reflects whether a physics engine is actually reporting, not
  // a hardcoded true. "unknown" until a signal is observed.
  const [physicsOk, setPhysicsOk] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const checkPhysics = async () => {
      try {
        const resp = await fetch('/api/alerts');
        if (!resp.ok) throw new Error(`status ${resp.status}`);
        const data = await resp.json();
        if (cancelled) return;
        // The alert pipeline is the physics consumer: a well-formed response
        // with a timestamp means the propagator and screening loop are live.
        setPhysicsOk(Boolean(data?.timestamp));
      } catch (_) {
        if (!cancelled) setPhysicsOk(false);
      }
    };

    checkPhysics();
    const timer = setInterval(checkPhysics, 30000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const [actionError, setActionError] = useState(null);
  const [busyAction, setBusyAction] = useState(null);

  const runAction = async (key, label, fn) => {
    setBusyAction(key);
    setActionError(null);
    try {
      await fn();
    } catch (e) {
      setActionError(`${label} failed: ${e?.message || 'unknown error'}`);
    } finally {
      setBusyAction(null);
    }
  };

  const handleDecision = (decision, alert) => {
    if (decision === 'APPROVE' || decision === 'MODIFY') {
      const alertId = alert.id ?? `${alert.sat1?.id}-${alert.sat2?.id}`;
      setCascadeGraph(markEdgeResolved(cascadeGraph, alertId));
    }
  };

  const handleLoadCascadeDemo = () =>
    runAction('demo', 'Load Cascade Demo', () =>
      apiPost('/api/simulate', { scenario: 'cascade_demo' })
    );

  const handleSimulateCollision = () =>
    runAction('collision', 'Simulate Collision', () =>
      apiPost('/api/debris/simulate', {})
    );

  const handleClearDebris = () =>
    runAction('clear', 'Clear Debris', () => apiDelete('/api/debris/active'));

  const handleResetTime = () =>
    runAction('reset', 'Reset Time', () => apiPost('/api/simulation/reset', {}));

  return (
    <div className="mcc-column">
      {/* Section 1: System status */}
      <div className="mcc-section">
        <div className="mcc-section-header">
          <span className="mcc-section-title">System Status</span>
        </div>
        <div className="sys-status-row">
          <div className="sys-indicator">
            <div className={`sys-indicator-dot ${wsConnected ? 'green' : 'red'}`} />
            <div className="sys-indicator-label">BACKEND</div>
          </div>
          <div className="sys-indicator">
            <div className={`sys-indicator-dot ${pipeline ? 'green' : 'red'}`} />
            <div className="sys-indicator-label">PIPELINE</div>
          </div>
          <div className="sys-indicator">
            <div className={`sys-indicator-dot ${mlTrained ? 'green' : 'amber'}`} />
            <div className="sys-indicator-label">ML</div>
          </div>
          <div className="sys-indicator">
            <div className={`sys-indicator-dot ${physicsOk === true ? 'green' : physicsOk === false ? 'red' : 'amber'}`} />
            <div className="sys-indicator-label">PHYSICS</div>
          </div>
          <div
            className="sys-indicator"
            title={
              rankerReview?.degraded
                ? rankerReview.degraded_reason || 'Primary cascade ranker unavailable'
                : `Primary ${(rankerReview?.primary || 'gat').toUpperCase()}, cross-check ${(rankerReview?.cross_check || 'gnn').toUpperCase()}` +
                  (rankerReview?.disagreement_count
                    ? ` — ${rankerReview.disagreement_count} disputed (max Δ ${rankerReview.max_disagreement})`
                    : ' — no disputed nodes')
            }
          >
            <div
              className={`sys-indicator-dot ${
                rankerReview?.degraded ? 'red' : rankerReview?.disagreement_count ? 'amber' : 'green'
              }`}
            />
            <div className="sys-indicator-label">RANKER</div>
          </div>
        </div>
        {rankerReview?.degraded ? (
          <div className="sys-status-note" style={{ color: 'var(--alert-red)' }}>
            CASCADE RANKER DEGRADED: {rankerReview.degraded_reason || 'primary ranker unavailable'}
          </div>
        ) : rankerReview?.disagreement_count ? (
          <div className="sys-status-note" style={{ color: 'var(--alert-amber, #f5a623)' }}>
            {rankerReview.disagreement_count} node(s) disputed by the {String(rankerReview.cross_check || 'gnn').toUpperCase()}{' '}
            cross-check (max Δ {rankerReview.max_disagreement}). GAT scores were used.
          </div>
        ) : null}
      </div>

      {/* Section 2: Threat Queue */}
      <div className="threat-queue">
        <div className="mcc-section-header">
          <span className="mcc-section-title">
            Threat Queue
          </span>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: alerts.length > 0 ? 'var(--alert-red)' : 'var(--text-dim)' }}>
            {alerts.length} ACTIVE
          </span>
        </div>

        {/* Filter tabs */}
        <div className="threat-filter-tabs">
          {TABS.map((tab) => (
            <button
              key={tab}
              type="button"
              className={`threat-tab ${activeTab === tab ? 'active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Cards */}
        <div className="threat-cards-list">
          {filteredAlerts.length === 0 ? (
            <div className="threat-empty">
              {activeTab === 'ALL'
                ? 'No active conjunction alerts'
                : `No ${activeTab} alerts`}
            </div>
          ) : (
            filteredAlerts.map((alert) => (
              <ThreatCard
                key={alert.id ?? `${alert.sat1?.id}-${alert.sat2?.id}`}
                alert={alert}
                isSelected={selectedAlertId === alert.id}
                onDecision={handleDecision}
              />
            ))
          )}
        </div>
      </div>

      {/* Section 3: Quick Actions */}
      <div className="quick-actions">
        <button type="button" className="qa-btn" disabled={busyAction !== null} onClick={handleLoadCascadeDemo}>
          ▶ Load Cascade Demo
        </button>
        <button type="button" className="qa-btn" disabled={busyAction !== null} onClick={handleSimulateCollision}>
          ● Simulate Collision
        </button>
        <button type="button" className="qa-btn" disabled={busyAction !== null} onClick={handleClearDebris}>
          ✕ Clear Debris
        </button>
        <button type="button" className="qa-btn" disabled={busyAction !== null} onClick={handleResetTime}>
          ↺ Reset Time
        </button>
        {actionError && (
          <div className="error-banner" style={{ marginTop: 6 }}>
            {actionError}
          </div>
        )}
      </div>
    </div>
  );
}
