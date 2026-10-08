import { useEffect, useState } from 'react';
import useStore from '../store/useStore';

const DEFAULT_AGENCIES = ['SpaceX', 'ISS', 'Roscosmos', 'CNSA', 'NOAA', 'NASA', 'Iridium'];

export default function Header() {
  const satellites = useStore((s) => s.satellites);
  const alerts = useStore((s) => s.alerts);
  const wsConnected = useStore((s) => s.wsConnected);
  const snapshotTimestamp = useStore((s) => s.snapshotTimestamp);
  const simTimeOffset = useStore((s) => s.simTimeOffset);
  const agencyFilter = useStore((s) => s.agencyFilter);
  const setAgencyFilter = useStore((s) => s.setAgencyFilter);

  const [dataAgeSecs, setDataAgeSecs] = useState(0);

  // Update data age every second
  useEffect(() => {
    const tick = () => {
      if (!snapshotTimestamp) { setDataAgeSecs(0); return; }
      const age = Math.round((Date.now() - new Date(snapshotTimestamp).getTime()) / 1000);
      setDataAgeSecs(Math.max(0, age));
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [snapshotTimestamp]);

  const maxCpi = alerts.length > 0
    ? Math.max(...alerts.map((a) => Number(a.cpi_score ?? 0)))
    : 0;

  // Derive the agency list from the live catalog so labels always match
  const catalogAgencies = [...new Set(
    satellites.map((s) => s.agency).filter((a) => a && a !== 'Unknown')
  )].sort();
  const agencyOptions = catalogAgencies.length > 0
    ? catalogAgencies
    : DEFAULT_AGENCIES;

  const cpiClass = maxCpi >= 8 ? 'critical' : maxCpi >= 5 ? 'warning' : maxCpi > 0 ? 'nominal' : '';
  const alertClass = alerts.length > 0 ? 'alert' : '';

  const handleAgencyChange = (e) => {
    const val = e.target.value;
    setAgencyFilter(val === 'All' ? null : [val]);
  };

  return (
    <header className="mc-header">
      {/* Left: logo */}
      <div className="mc-header-left">
        <div className="mc-header-logo">ORBITAL SENTINEL</div>
        <div className="mc-header-subtitle">Space Traffic Mission Control</div>
      </div>

      {/* Center: stat boxes */}
      <div className="mc-header-stats">
        <div className="mc-stat-box">
          <div className="mc-stat-label">Tracked</div>
          <div className="mc-stat-value">{satellites.length}</div>
        </div>
        <div className="mc-stat-box">
          <div className="mc-stat-label">Alerts</div>
          <div className={`mc-stat-value ${alertClass}`}>{alerts.length}</div>
        </div>
        <div className="mc-stat-box">
          <div className="mc-stat-label">Max CPI</div>
          <div className={`mc-stat-value ${cpiClass}`}>{maxCpi.toFixed(1)}</div>
        </div>
      </div>

      {/* Right: connection, age, filter, sim time */}
      <div className="mc-header-right">
        <div className="mc-conn-status">
          <div className={`mc-conn-dot ${wsConnected ? 'live' : ''}`} />
          <span>{wsConnected ? 'LIVE' : 'OFFLINE'}</span>
        </div>

        <div className="mc-tle-age">
          TLE: {dataAgeSecs}s ago
        </div>

        <select
          className="mc-agency-select"
          value={agencyFilter ? agencyFilter[0] : 'All'}
          onChange={handleAgencyChange}
          title="Filter by agency"
        >
          {['All', ...agencyOptions].map((a) => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>

        <div className={`mc-sim-badge ${simTimeOffset !== 0 ? 'sim' : 'realtime'}`}>
          {simTimeOffset !== 0 ? `SIM +${simTimeOffset}h` : 'REALTIME'}
        </div>
      </div>
    </header>
  );
}
