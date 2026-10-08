import useStore from '../../store/useStore';

const riskColor = (riskBand) => {
  if (riskBand === 'high') return 'var(--alert-red)';
  if (riskBand === 'medium') return 'var(--alert-yellow)';
  return 'var(--alert-green)';
};

const formatUtc = (value) => {
  if (!value) return '---';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '---';
  return date.toISOString().replace('.000Z', 'Z');
};

export default function AlertPanel() {
  const alerts = useStore((s) => s.alerts);
  const cascadePlan = useStore((s) => s.cascadePlan);
  const hotspots = useStore((s) => s.hotspots);
  const debrisClouds = useStore((s) => s.debrisClouds);

  const totalDeltaV = cascadePlan?.total_delta_v_ms != null
    ? Number(cascadePlan.total_delta_v_ms).toFixed(3)
    : '0.000';

  return (
    <div className="panel alert-panel">
      <div className="panel-header">
        <span className="panel-title">Conjunction Alerts</span>
        <span className="mono" style={{ fontSize: 11 }}>
          {alerts.length}
        </span>
      </div>

      {cascadePlan && (
        <div className="monitor-card">
          <div className="monitor-title">Cascade Plan</div>
          <div className="monitor-row">
            <span>Graph Nodes</span>
            <span className="mono">{cascadePlan.graph?.node_count ?? 0}</span>
          </div>
          <div className="monitor-row">
            <span>Graph Edges</span>
            <span className="mono">{cascadePlan.graph?.edge_count ?? 0}</span>
          </div>
          <div className="monitor-row">
            <span>Max Cascade Depth</span>
            <span className="mono">{cascadePlan.cascade_depth ?? 0}</span>
          </div>
          <div className="monitor-row">
            <span>Total Delta-V</span>
            <span className="mono">{totalDeltaV} m/s</span>
          </div>
          <div className="monitor-row">
            <span>Agencies</span>
            <span className="mono">{(cascadePlan.agencies_involved || []).join(', ') || '—'}</span>
          </div>
        </div>
      )}

      {alerts.length === 0 ? (
        <div className="no-data">NO ACTIVE ALERTS</div>
      ) : (
        <div className="terminal-log">
          {alerts.map((alert, i) => {
            const topCascade = cascadePlan?.cascade_plan?.find(
              (maneuver) => maneuver.satellite_id === alert.sat1?.id || maneuver.satellite_id === alert.sat2?.id,
            );
            const alertPosition = alert.position || alert.hotspot_position;
            const relatedCloud = debrisClouds.find(
              (cloud) => cloud?.affected_satellites?.some(
                (sat) => sat.norad_id === alert.sat1?.id || sat.norad_id === alert.sat2?.id,
              ),
            );
            const zoneRadiusKm = Number(alert.zone_radius_km || 100);
            const affectedSatellites = relatedCloud?.affected_satellites || [];
            const collisionHours = Number(alert.tca_hours ?? (Number(alert.tca_minutes || 0) / 60.0));
            const collisionTime = formatUtc(alert.tca_utc);
            const fallbackDeadline = (() => {
              if (!alert.tca_utc) return '---';
              const date = new Date(alert.tca_utc);
              if (Number.isNaN(date.getTime())) return '---';
              return new Date(date.getTime() - (60 * 60 * 1000)).toISOString().replace('.000Z', 'Z');
            })();

            const verdictSats = [alert.sat1?.name || 'SAT-1', alert.sat2?.name || 'SAT-2'].join(' & ');
            const fallbackHHMM = fallbackDeadline !== '---' ? fallbackDeadline.slice(11, 16) : '---';
            const tcaHHMM = collisionTime !== '---' ? collisionTime.slice(11, 16) : '---';
            const otherSatsInZone = affectedSatellites
              .filter((sat) => sat.norad_id !== alert.sat1?.id && sat.norad_id !== alert.sat2?.id)
              .slice(0, 3)
              .map((sat) => sat.name || `#${sat.norad_id}`)
              .join(', ');
            const otherSatsPhrase = otherSatsInZone ? ` Also at location: ${otherSatsInZone}.` : '';
            const verdict = fallbackDeadline !== '---'
              ? `${verdictSats} must switch to fallback trajectory before ${fallbackHHMM} UTC; collision expected ${tcaHHMM} UTC.${otherSatsPhrase}`
              : `${verdictSats} collision risk detected. Maneuver required.${otherSatsPhrase}`;
            const affectedTargets = affectedSatellites
              .filter((sat) => sat.norad_id !== alert.sat1?.id && sat.norad_id !== alert.sat2?.id)
              .slice(0, 6);

            return (
              <div
                className="alert-item terminal-item"
                key={`${alert.sat1?.id || 'a'}-${alert.sat2?.id || 'b'}-${i}`}
                style={{ borderColor: 'var(--alert-red)', background: 'rgba(239, 68, 68, 0.06)' }}
              >
                <div className="terminal-line terminal-title-line" style={{ color: 'var(--alert-red)', marginBottom: 8 }}>
                  <span className="terminal-tag" style={{ color: 'var(--alert-red)' }}>VERDICT</span>
                  <span>{verdict}</span>
                </div>
                <div className="terminal-line terminal-title-line">
                  <span className="terminal-tag">ALERT</span>
                  <span>{alert.sat1?.name || 'SAT-1'} × {alert.sat2?.name || 'SAT-2'}</span>
                </div>
                <div className="terminal-line">
                  <span className="terminal-tag">TIME</span>
                  <span>Collision in {Number.isFinite(collisionHours) ? `${collisionHours.toFixed(2)} hr` : '---'} | {collisionTime}</span>
                </div>
                <div className="terminal-line">
                  <span className="terminal-tag">POINT</span>
                  <span>
                    {alertPosition
                      ? `${Number(alertPosition.x || 0).toFixed(1)}, ${Number(alertPosition.y || 0).toFixed(1)}, ${Number(alertPosition.z || 0).toFixed(1)} km`
                      : '---'}
                  </span>
                </div>
                <div className="terminal-line">
                  <span className="terminal-tag">WAVE</span>
                  <span>Hotspot zone radius {zoneRadiusKm.toFixed(0)} km</span>
                </div>
                <div className="terminal-line">
                  <span className="terminal-tag">ACTION</span>
                  <span>Move the affected satellite(s) to fallback trajectory before {fallbackDeadline}</span>
                </div>
                <div className="terminal-line">
                  <span className="terminal-tag">RISK</span>
                  <span>Miss {Number(alert.miss_distance_km || 0).toFixed(1)} km | Speed {Number(alert.relative_speed_kh || alert.relative_speed_kmh || 0).toFixed(0)} km/h | CPI {Number(alert.cpi_score || 0).toFixed(1)}</span>
                </div>
                {topCascade && (
                  <div className="terminal-line">
                    <span className="terminal-tag">CPI</span>
                    <span>Cascade depth {topCascade.cascade_depth} | Burn {topCascade.maneuver.delta_v_ms} m/s</span>
                  </div>
                )}
                {relatedCloud && (
                  <div className="terminal-line">
                    <span className="terminal-tag">DEBRIS</span>
                    <span>
                      {relatedCloud.id} | affected {Number(relatedCloud.affected_count || 0)} | high risk {Number(relatedCloud.affected_high_risk || 0)}
                    </span>
                  </div>
                )}
                {affectedSatellites.length > 0 && (
                  <div className="terminal-line terminal-wrap">
                    <span className="terminal-tag">TARGETS</span>
                    <span>
                      {affectedSatellites.map((sat) => `${sat.name || `#${sat.norad_id}`}${sat.risk_band ? `(${sat.risk_band})` : ''}`).join(' | ')}
                    </span>
                  </div>
                )}
                {affectedTargets.length > 0 && (
                  <div className="terminal-line terminal-wrap" style={{ alignItems: 'stretch' }}>
                    <span className="terminal-tag">AFFECTED</span>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, flex: 1 }}>
                      {affectedTargets.map((sat) => (
                        <div
                          key={`${alert.sat1?.id || 'a'}-${alert.sat2?.id || 'b'}-${sat.norad_id}`}
                          className="monitor-row"
                          style={{
                            padding: '6px 8px',
                            border: `1px solid ${riskColor(sat.risk_band)}`,
                            background: 'rgba(255, 255, 255, 0.03)',
                            alignItems: 'center',
                          }}
                        >
                          <span style={{ color: 'var(--text-bright)' }}>
                            {sat.name || `#${sat.norad_id}`}
                          </span>
                          <span className="mono" style={{ color: riskColor(sat.risk_band) }}>
                            {sat.risk_band || 'low'} | {sat.recommended_action === 'fallback' ? 'MOVE TO FALLBACK' : 'MONITOR'}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {cascadePlan?.cascade_plan?.length > 0 && (
        <div className="monitor-card" style={{ marginTop: 12 }}>
          <div className="monitor-title">Maneuver Chain</div>
          {cascadePlan.cascade_plan.slice(0, 8).map((maneuver) => (
            <div className="monitor-row" key={`${maneuver.satellite_id}-${maneuver.cascade_depth}`}>
              <span>{maneuver.satellite_name}</span>
              <span className="mono">
                D{maneuver.cascade_depth} | {maneuver.maneuver.delta_v_ms} m/s
              </span>
            </div>
          ))}
        </div>
      )}

      {hotspots.length > 0 && (
        <div className="monitor-card" style={{ marginTop: 12 }}>
          <div className="monitor-title">Hotspot Wave</div>
          {hotspots.slice(0, 5).map((hotspot, index) => (
            <div className="monitor-row" key={`${hotspot.sat1?.id}-${hotspot.sat2?.id}-${index}`}>
              <span>
                {hotspot.sat1?.name || 'SAT-1'} × {hotspot.sat2?.name || 'SAT-2'}
              </span>
              <span className="mono">
                {Number(hotspot.hotspot_score || 0).toFixed(2)} | T{Number(hotspot.tca_minutes || 0).toFixed(0)}m | R{Number(hotspot.zone_radius_km || 100).toFixed(0)}km
              </span>
            </div>
          ))}
        </div>
      )}

      {debrisClouds.length > 0 && (
        <div className="monitor-card" style={{ marginTop: 12 }}>
          <div className="monitor-title">Debris Alerts</div>
          {debrisClouds.slice(0, 5).map((cloud) => (
            <div key={cloud.id} style={{ marginBottom: 8 }}>
              <div className="monitor-row">
                <span>{cloud.id}</span>
                <span className="mono">
                  R{Number(cloud.radius_km_at_tca || 0).toFixed(0)}km | {Number(cloud.minutes_to_tca || 0).toFixed(0)}m
                  {' '}| A{Number(cloud.affected_count || 0)} H{Number(cloud.affected_high_risk || 0)}
                </span>
              </div>
              {(cloud.radius_timeline || []).length > 0 && (
                <div className="monitor-row" style={{ fontSize: 11 }}>
                  <span style={{ color: 'var(--text-dim)' }}>Wave radius</span>
                  <span className="mono">
                    now {Number(cloud.radius_timeline[0]?.radius_km || 0).toFixed(0)}km | mid {Number(cloud.radius_timeline[1]?.radius_km || 0).toFixed(0)}km | tca {Number(cloud.radius_timeline[2]?.radius_km || 0).toFixed(0)}km
                  </span>
                </div>
              )}
              {(cloud.affected_satellites || []).slice(0, 4).map((sat) => (
                <div className="monitor-row" key={`${cloud.id}-${sat.norad_id}`} style={{ fontSize: 11 }}>
                  <span style={{ color: 'var(--text-dim)' }}>
                    {sat.name || `#${sat.norad_id}`}
                  </span>
                  <span className="mono">
                    CPI {Number(sat.cpi_score || 0).toFixed(1)} | {sat.risk_band || 'low'}
                    {sat.recommended_action === 'fallback' ? ' | FALLBACK' : ''}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
