import { memo, useEffect, useMemo, useState } from 'react';
import useStore from '../store/useStore';
import AnalyticsDrawer from './AnalyticsDrawer';

const DEFAULT_AGENCIES = ['SpaceX', 'ISS', 'Roscosmos', 'CNSA', 'NOAA', 'NASA', 'Iridium'];

const pad2 = (n) => String(n).padStart(2, '0');

const formatUtc = (ms) => {
  const d = new Date(ms);
  return `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}:${pad2(d.getUTCSeconds())}`;
};

const formatAge = (secs) => {
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  if (m < 60) return `${m}m ${pad2(secs % 60)}s`;
  return `${Math.floor(m / 60)}h ${pad2(m % 60)}m`;
};

// Isolated 1 Hz ticker so the rest of the header does not re-render every second.
const HeaderClock = memo(function HeaderClock({ snapshotTimestamp, simTimeOffset }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const dataAgeSecs = snapshotTimestamp
    ? Math.max(0, Math.round((now - new Date(snapshotTimestamp).getTime()) / 1000))
    : null;

  const ageClass = dataAgeSecs == null
    ? 'is-dim'
    : dataAgeSecs >= 600 ? 'is-warning' : dataAgeSecs >= 120 ? 'is-caution' : '';

  const isSim = simTimeOffset !== 0;

  return (
    <div className="sh-clock">
      <div className="sh-readout">
        <span className="ui-label">Data Age</span>
        <span className={`sh-readout-value ${ageClass}`}>
          {dataAgeSecs == null ? '---' : formatAge(dataAgeSecs)}
        </span>
      </div>
      <div className="sh-readout">
        <span className="ui-label">{isSim ? `Sim +${simTimeOffset}h` : 'Realtime'}</span>
        <span className={`sh-readout-value ${isSim ? 'is-caution' : ''}`}>
          {formatUtc(now + simTimeOffset * 3600000)}
          <span className="sh-unit">UTC</span>
        </span>
      </div>
    </div>
  );
});

export default function Header() {
  const satellites = useStore((s) => s.satellites);
  const alerts = useStore((s) => s.alerts);
  const wsConnected = useStore((s) => s.wsConnected);
  const snapshotTimestamp = useStore((s) => s.snapshotTimestamp);
  const simTimeOffset = useStore((s) => s.simTimeOffset);
  const agencyFilter = useStore((s) => s.agencyFilter);
  const setAgencyFilter = useStore((s) => s.setAgencyFilter);
  const metricsOpen = useStore((s) => s.metricsDrawerOpen);
  const setMetricsOpen = useStore((s) => s.setMetricsDrawerOpen);
  const simOpen = useStore((s) => s.simulationDrawerOpen);
  const setSimOpen = useStore((s) => s.setSimulationDrawerOpen);
  const analyticsOpen = useStore((s) => s.analyticsDrawerOpen);
  const setAnalyticsOpen = useStore((s) => s.setAnalyticsDrawerOpen);

  const maxCpi = useMemo(() => {
    let max = 0;
    for (const a of alerts) {
      const v = Number(a.cpi_score ?? 0);
      if (v > max) max = v;
    }
    return max;
  }, [alerts]);

  // Derive the agency list from the live catalog so labels always match
  const agencyOptions = useMemo(() => {
    const set = new Set();
    for (const s of satellites) {
      if (s.agency && s.agency !== 'Unknown') set.add(s.agency);
    }
    return set.size > 0 ? [...set].sort() : DEFAULT_AGENCIES;
  }, [satellites]);

  const cpiClass = maxCpi >= 8 ? 'is-warning' : maxCpi >= 5 ? 'is-caution' : maxCpi > 0 ? 'is-nominal' : 'is-dim';
  const alertClass = alerts.length > 0 ? 'is-caution' : 'is-dim';

  const handleAgencyChange = (e) => {
    const val = e.target.value;
    setAgencyFilter(val === 'All' ? null : [val]);
  };

  return (
    <header className="sh-header">
      <div className="sh-brand">
        <span className="sh-brand-mark">Orbital Sentinel</span>
        <span className="ui-label sh-brand-sub">Space Traffic Mission Control</span>
      </div>

      <div className="sh-readouts">
        <div className="sh-readout">
          <span className="ui-label">Tracked</span>
          <span className="sh-readout-value">{satellites.length.toLocaleString('en-US')}</span>
        </div>
        <div className="sh-readout">
          <span className="ui-label">Conjunctions</span>
          <span className={`sh-readout-value ${alertClass}`}>{alerts.length}</span>
        </div>
        <div className="sh-readout">
          <span className="ui-label">Max CPI</span>
          <span className={`sh-readout-value ${cpiClass}`}>{maxCpi.toFixed(1)}</span>
        </div>
        <div className="sh-readout">
          <span className="ui-label">Link</span>
          <span className={`ui-status ${wsConnected ? 'is-nominal' : 'is-warning'}`}>
            {wsConnected ? 'Live' : 'Offline'}
          </span>
        </div>
        <HeaderClock snapshotTimestamp={snapshotTimestamp} simTimeOffset={simTimeOffset} />
      </div>

      <div className="sh-controls">
        <label className="sh-field">
          <span className="ui-label">Agency</span>
          <select
            className="ui-select sh-select"
            value={agencyFilter ? agencyFilter[0] : 'All'}
            onChange={handleAgencyChange}
            title="Filter by agency"
          >
            {['All', ...agencyOptions].map((a) => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        </label>

        <button
          type="button"
          className="ui-btn sh-toggle"
          aria-pressed={metricsOpen}
          onClick={() => setMetricsOpen(!metricsOpen)}
          title="Toggle model scorecard"
        >
          Scorecard
        </button>
        <button
          type="button"
          className="ui-btn sh-toggle"
          aria-pressed={simOpen}
          onClick={() => setSimOpen(!simOpen)}
          title="Toggle simulation console"
        >
          Simulation
        </button>
        <button
          type="button"
          className="ui-btn sh-toggle"
          aria-pressed={analyticsOpen}
          onClick={() => setAnalyticsOpen(!analyticsOpen)}
          title="Toggle model & physics analytics"
        >
          Analytics
        </button>
      </div>
      {/* Fixed-position side sheet; mounted here so it fetches only when open. */}
      <AnalyticsDrawer />
    </header>
  );
}
