import { useState, useEffect, useMemo, useCallback } from 'react';
import ThreatCard from './ThreatCard';
import { severityLabel } from '../utils/severity';
import useStore from '../store/useStore';
import { apiPost, apiDelete } from '../utils/api';
import { markEdgeResolved } from '../utils/cascadeGraph';
import '../styles/threats.css';

const TABS = ['ALL', 'CRITICAL', 'WARNING', 'WATCH'];

function SysCell({ label, state, title }) {
  return (
    <div className="tq-sys-cell" title={title}>
      <span className={`ui-status is-${state}`}>{label}</span>
    </div>
  );
}

export default function MissionControlCenter({ alerts = [] }) {
  const [activeTab, setActiveTab] = useState('ALL');
  const selectedAlertId = useStore((s) => s.selectedAlertId);
  const wsConnected = useStore((s) => s.wsConnected);
  const modelMetrics = useStore((s) => s.modelMetrics);
  const setCascadeGraph = useStore((s) => s.setCascadeGraph);
  const cascadeGraph = useStore((s) => s.cascadeGraph);
  const rankerReview = useStore((s) => s.rankerReview);

  const filteredAlerts = useMemo(
    () =>
      alerts
        .filter((a) => {
          if (activeTab === 'ALL') return true;
          return severityLabel(a) === activeTab;
        })
        .sort((a, b) => Number(b.cpi_score ?? 0) - Number(a.cpi_score ?? 0)),
    [alerts, activeTab]
  );

  // System status indicators
  const pipeline = wsConnected;
  const mlTrained = useMemo(
    () => (modelMetrics?.classification_models ?? []).some((m) => (m.metrics?.samples ?? 0) > 0),
    [modelMetrics]
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

  // Stable across the per-second re-renders so memoized ThreatCards can skip work.
  const handleDecision = useCallback(
    (decision, alert) => {
      if (decision === 'APPROVE' || decision === 'MODIFY') {
        const alertId = alert.id ?? `${alert.sat1?.id}-${alert.sat2?.id}`;
        setCascadeGraph(markEdgeResolved(cascadeGraph, alertId));
      }
    },
    [cascadeGraph, setCascadeGraph]
  );

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

  const backendState = wsConnected ? 'nominal' : 'warning';
  const pipelineState = pipeline ? 'nominal' : 'warning';
  const mlState = mlTrained ? 'nominal' : 'caution';
  const physicsState = physicsOk === true ? 'nominal' : physicsOk === false ? 'warning' : 'caution';
  const rankerState = rankerReview?.degraded ? 'warning' : rankerReview?.disagreement_count ? 'caution' : 'nominal';
  const rankerTitle = rankerReview?.degraded
    ? rankerReview.degraded_reason || 'Primary cascade ranker unavailable'
    : `Primary ${(rankerReview?.primary || 'gat').toUpperCase()}, cross-check ${(rankerReview?.cross_check || 'gnn').toUpperCase()}` +
      (rankerReview?.disagreement_count
        ? ` — ${rankerReview.disagreement_count} disputed (max Δ ${rankerReview.max_disagreement})`
        : ' — no disputed nodes');

  const sysStates = [backendState, pipelineState, mlState, physicsState, rankerState];
  const overall = sysStates.includes('warning') ? 'warning' : sysStates.includes('caution') ? 'caution' : 'nominal';
  const overallLabel = overall === 'warning' ? 'Sys fault' : overall === 'caution' ? 'Sys degraded' : 'Sys nominal';

  return (
    <section className="tq-root" aria-label="Conjunction queue">
      {/* Panel header */}
      <header className="tq-header">
        <span className="ui-label tq-title">Conjunction Queue</span>
        <span className={`tq-count ${alerts.length > 0 ? 'is-active' : ''}`}>
          {String(alerts.length).padStart(2, '0')} ACTIVE
        </span>
        <span className={`ui-status tq-header-status is-${overall}`}>{overallLabel}</span>
      </header>

      {/* System status */}
      <div className="tq-sys">
        <div className="tq-sys-row">
          <SysCell label="Backend" state={backendState} />
          <SysCell label="Pipeline" state={pipelineState} />
          <SysCell label="ML" state={mlState} />
          <SysCell label="Physics" state={physicsState} />
          <SysCell label="Ranker" state={rankerState} title={rankerTitle} />
        </div>
        {rankerReview?.degraded ? (
          <div className="tq-sys-note is-warning">
            CASCADE RANKER DEGRADED: {rankerReview.degraded_reason || 'primary ranker unavailable'}
          </div>
        ) : rankerReview?.disagreement_count ? (
          <div className="tq-sys-note is-caution">
            {rankerReview.disagreement_count} node(s) disputed by the {String(rankerReview.cross_check || 'gnn').toUpperCase()}{' '}
            cross-check (max Δ {rankerReview.max_disagreement}). GAT scores were used.
          </div>
        ) : null}
      </div>

      {/* Filter tabs */}
      <div className="tq-tabs" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab}
            type="button"
            role="tab"
            aria-selected={activeTab === tab}
            className={`tq-tab ${activeTab === tab ? 'is-active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Column header (aligns with each row's metric line) */}
      <div className="tq-colhead" aria-hidden="true">
        <span className="ui-label">TCA</span>
        <span className="ui-label tq-num">Miss km</span>
        <span className="ui-label tq-num">Pc</span>
        <span className="ui-label tq-num">Casc</span>
        <span className="ui-label tq-num">CPI</span>
      </div>

      {/* Rows */}
      <div className="tq-list">
        {filteredAlerts.length === 0 ? (
          <div className="tq-empty">
            {activeTab === 'ALL' ? 'No active conjunction alerts' : `No ${activeTab} alerts`}
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

      {/* Quick actions */}
      <div className="tq-footer">
        <div className="tq-footer-grid">
          <button type="button" className="ui-btn" disabled={busyAction !== null} onClick={handleLoadCascadeDemo}>
            Load Cascade Demo
          </button>
          <button type="button" className="ui-btn" disabled={busyAction !== null} onClick={handleSimulateCollision}>
            Simulate Collision
          </button>
          <button type="button" className="ui-btn" disabled={busyAction !== null} onClick={handleClearDebris}>
            Clear Debris
          </button>
          <button type="button" className="ui-btn" disabled={busyAction !== null} onClick={handleResetTime}>
            Reset Time
          </button>
        </div>
        {actionError && <div className="tq-error" role="alert">{actionError}</div>}
      </div>
    </section>
  );
}
