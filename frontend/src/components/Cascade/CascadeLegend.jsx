import '../../styles/cascade.css';

/** Legend shared by the panel and the explorer. */
export default function CascadeLegend({ compact = false }) {
  return (
    <div className={`cx-legend${compact ? ' cx-legend--compact' : ''}`} aria-label="Graph legend">
      <span className="cx-legend-item"><span className="cx-sw cx-sev-critical" />Critical</span>
      <span className="cx-legend-item"><span className="cx-sw cx-sev-warning" />Warning</span>
      <span className="cx-legend-item"><span className="cx-sw cx-sev-watch" />Watch</span>
      <span className="cx-legend-item"><svg width="11" height="11" viewBox="0 0 11 11" className="cx-sev-event"><path d="M5.5 0.5 L6.9 4 L10.5 4.1 L7.7 6.4 L8.7 10 L5.5 7.9 L2.3 10 L3.3 6.4 L0.5 4.1 L4.1 4 Z" fill="currentColor" /></svg>Collision</span>
      <span className="cx-legend-item"><span className="cx-sw cx-sw--ring" />Hotspot zone</span>
      <span className="cx-legend-item"><span className="cx-ln cx-ln--screening" />Conjunction</span>
      <span className="cx-legend-item"><span className="cx-ln cx-ln--debris" />Fragments</span>
      {!compact && <span className="cx-legend-item"><span className="cx-ln cx-ln--parent" />Breakup parent</span>}
      {!compact && <span className="cx-legend-item cx-legend-note">dot size = P(hit)</span>}
    </div>
  );
}
