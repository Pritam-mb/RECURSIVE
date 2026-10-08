import { useEffect, useRef, Suspense } from 'react';
import CesiumGlobe from './components/Globe/CesiumGlobe';
import ThreatGlobe from './components/ThreatGlobe';
import Header from './components/Header';
import ConnectionBanner from './components/ConnectionBanner';
import MissionControlCenter from './components/MissionControlCenter';
import TelemetryStrip from './components/TelemetryStrip';
import UplinkDownlinkV2 from './components/Control/UplinkDownlinkV2';
import CascadeDiagram from './components/CascadeDiagram';
import ModelStatusV2 from './components/ModelStatusV2';
import SimulationDrawer from './components/SimulationDrawer';
import MetricsDrawer from './components/MetricsDrawer';
import useStore from './store/useStore';
import { buildCascadeGraph } from './utils/cascadeGraph';
import './App.css';

const wsBaseUrl = (window.location.origin || '').replace(/^http/, 'ws');
const alertPollMs = 30000;
const heartbeatPollMs = 20000;

function GlobeLoader() {
  return (
    <div style={{
      width: '100%', height: '100%', background: '#000',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-dim)', letterSpacing: 2 }}>
        LOADING GLOBE…
      </span>
    </div>
  );
}

function App() {
  // ── Store subscriptions ────────────────────────────────────────────────────
  const setSatellites        = useStore((s) => s.setSatellites);
  const setSnapshotTimestamp = useStore((s) => s.setSnapshotTimestamp);
  const setAlerts            = useStore((s) => s.setAlerts);
  const setCascadePlan       = useStore((s) => s.setCascadePlan);
  const setHotspots          = useStore((s) => s.setHotspots);
  const setDebrisClouds      = useStore((s) => s.setDebrisClouds);
  const setModelMetrics      = useStore((s) => s.setModelMetrics);
  const setWsConnected       = useStore((s) => s.setWsConnected);
  const setCascadeGraph      = useStore((s) => s.setCascadeGraph);
  const setRankerReview      = useStore((s) => s.setRankerReview);

  const satellites        = useStore((s) => s.satellites);
  const alerts            = useStore((s) => s.alerts);
  const selectedSatelliteId = useStore((s) => s.selectedSatelliteId);
  const setSelectedSatelliteId = useStore((s) => s.setSelectedSatelliteId);
  const agencyFilter      = useStore((s) => s.agencyFilter);
  const cascadeGraph      = useStore((s) => s.cascadeGraph);

  const wsRef = useRef(null);
  const alertFetchInFlight = useRef(false);

  // ── WebSocket + polling ────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    let reconnectTimer = null;
    let alertPollTimer = null;
    let heartbeatTimer = null;

    const bootstrapSatellites = async () => {
      try {
        const res = await fetch('/api/satellites');
        if (cancelled || !res.ok) return;
        const data = await res.json();
        setSnapshotTimestamp(data.timestamp || null);
        setSatellites(data.satellites || []);
      } catch (e) {
        console.error('Satellite bootstrap failed:', e);
      }
    };

    const bootstrapAlerts = async () => {
      if (cancelled || alertFetchInFlight.current) return;
      alertFetchInFlight.current = true;
      try {
        const res = await fetch('/api/alerts');
        if (cancelled || !res.ok) return;
        const data = await res.json();
        if (!cancelled) {
          setAlerts(data.alerts || []);
          setCascadePlan({
            graph: data.graph || {},
            alerts: data.alerts || [],
            cascade_plan: data.cascade_plan || [],
            total_delta_v_ms: data.total_delta_v_ms || 0,
            cascade_depth: data.cascade_depth || 0,
            agencies_involved: data.agencies_involved || [],
            seed_satellites: data.seed_satellites || [],
            cpi_threshold: data.cpi_threshold || 5.0,
            node_probabilities: data.node_probabilities || {},
          });
          setRankerReview(data.ranker_review || {});
          setHotspots(data.hotspots || []);
          setDebrisClouds(data.debris_clouds || [], data.debris_sources || ['forecast', 'fragment']);
        }
      } catch (e) {
        console.error('Alert bootstrap failed:', e);
      } finally {
        alertFetchInFlight.current = false;
      }
    };

    const bootstrapModelMetrics = async () => {
      try {
        const res = await fetch('/api/model-metrics');
        if (cancelled || !res.ok) return;
        const data = await res.json();
        if (!cancelled) setModelMetrics(data);
      } catch (_) {}
    };

    const connect = () => {
      if (cancelled) return;
      const ws = new WebSocket(`${wsBaseUrl}/ws/satellites`);

      ws.onopen = () => {
        setWsConnected(true);
        heartbeatTimer = setInterval(() => {
          try { if (ws.readyState === WebSocket.OPEN) ws.send('ping'); } catch (_) {}
        }, heartbeatPollMs);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'update') {
            setSnapshotTimestamp(data.timestamp || null);
            setSatellites(data.satellites || []);
            if (Object.prototype.hasOwnProperty.call(data, 'debris_clouds')) {
              setDebrisClouds(data.debris_clouds || [], data.debris_sources || null);
            }
            // The server attaches the alert block only when the alert cache
            // changes, so absence means "unchanged" and must not clear state.
            if (Object.prototype.hasOwnProperty.call(data, 'alerts')) {
              setAlerts(data.alerts || []);
              setHotspots(data.hotspots || []);
              setCascadePlan({
                graph: data.graph || {},
                alerts: data.alerts || [],
                cascade_plan: data.cascade_plan || [],
                total_delta_v_ms: data.total_delta_v_ms || 0,
                cascade_depth: data.cascade_depth || 0,
                agencies_involved: data.agencies_involved || [],
                seed_satellites: data.seed_satellites || [],
                cpi_threshold: data.cpi_threshold || 5.0,
                node_probabilities: data.node_probabilities || {},
              });
              setRankerReview(data.ranker_review || {});
            }
          }
        } catch (e) {
          console.error('WS parse error:', e);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
        if (!cancelled) reconnectTimer = setTimeout(connect, 3000);
      };

      ws.onerror = () => ws.close();
      wsRef.current = ws;
    };

    bootstrapSatellites();
    bootstrapAlerts();
    bootstrapModelMetrics();
    connect();

    alertPollTimer = setInterval(bootstrapAlerts, alertPollMs);

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (alertPollTimer) clearInterval(alertPollTimer);
      if (heartbeatTimer) clearInterval(heartbeatTimer);
      if (wsRef.current) wsRef.current.close();
    };
  }, [setSatellites, setSnapshotTimestamp, setAlerts, setCascadePlan, setHotspots, setDebrisClouds, setModelMetrics, setWsConnected, setRankerReview]);

  // ── Build cascade graph whenever alerts change ─────────────────────────────
  useEffect(() => {
    const graph = buildCascadeGraph(alerts);
    setCascadeGraph(graph);
  }, [alerts, setCascadeGraph]);

  return (
    <div className="mission-control">
      {/* Offline banner (fixed, above everything) */}
      <ConnectionBanner />

      {/* Row 1: Header (48px) */}
      <Header />

      {/* Row 2: Dual globe + mission control center */}
      <div className="main-grid">
        <div className="live-globe-panel">
          <div className="panel-label">LIVE CATALOG</div>
          <Suspense fallback={<GlobeLoader />}>
            <CesiumGlobe
              mode="live"
              satellites={satellites}
              alerts={alerts}
              selectedSatId={selectedSatelliteId}
              onSatelliteSelect={setSelectedSatelliteId}
              agencyFilter={agencyFilter}
            />
          </Suspense>
        </div>

        <MissionControlCenter alerts={alerts} />

        <div className="threat-globe-panel">
          <div className="panel-label">THREAT ANALYSIS</div>
          <ThreatGlobe
            alerts={alerts}
            satellites={satellites}
            selectedSatId={selectedSatelliteId}
          />
        </div>
      </div>

      {/* Row 3: Telemetry strip (72px) */}
      <TelemetryStrip selectedSatId={selectedSatelliteId} />

      {/* Row 4: Bottom three panels */}
      <div className="bottom-grid">
        <UplinkDownlinkV2 selectedSatId={selectedSatelliteId} />
        <CascadeDiagram graph={cascadeGraph} />
        <ModelStatusV2 />
      </div>

      {/* Preserved floating drawers (simulation & metrics controls) */}
      <MetricsDrawer />
      <SimulationDrawer />
    </div>
  );
}

export default App;
