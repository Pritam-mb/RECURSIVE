import useStore from '../../store/useStore';
import '../../styles/sim.css';

// Risk band → state colour class (anything other than high/medium reads as low)
const riskClass = (riskBand) => {
  if (riskBand === 'high') return 'sim-risk-high';
  if (riskBand === 'medium') return 'sim-risk-medium';
  return 'sim-risk-low';
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
    <section className="sim-section">
      <div className="sim-section-head">
        <span className="ui-label sim-section-title">Conjunction Alerts</span>
        <span className={`ui-status ${alerts.length > 0 ? 'is-warning' : 'is-nominal'}`}>
          {alerts.length}
        </span>
      </div>

      <div className="sim-body">
        {cascadePlan && (
          <>
            <div className="sim-subhead">
              <span className="ui-label">Cascade plan</span>
            </div>
            <dl className="sim-dl">
              <dt>Graph nodes</dt>
              <dd>{cascadePlan.graph?.node_count ?? 0}</dd>
              <dt>Graph edges</dt>
              <dd>{cascadePlan.graph?.edge_count ?? 0}</dd>
              <dt>Max cascade depth</dt>
              <dd>{cascadePlan.cascade_depth ?? 0}</dd>
              <dt>Total delta-V</dt>
              <dd>{totalDeltaV} m/s</dd>
              <dt>Agencies</dt>
              <dd>{(cascadePlan.agencies_involved || []).join(', ') || '—'}</dd>
            </dl>
          </>
        )}

        {alerts.length === 0 ? (
          <div className="sim-muted">No active alerts</div>
        ) : (
          <div className="sim-stack">
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
                <article
                  className="sim-alert"
                  key={`${alert.sat1?.id || 'a'}-${alert.sat2?.id || 'b'}-${i}`}
                >
                  <div className="sim-alert-title">
                    {alert.sat1?.name || 'SAT-1'} × {alert.sat2?.name || 'SAT-2'}
                  </div>
                  <div className="sim-alert-verdict">{verdict}</div>
                  <dl className="sim-dl">
                    <dt>Time</dt>
                    <dd>
                      Collision in {Number.isFinite(collisionHours) ? `${collisionHours.toFixed(2)} hr` : '---'} | {collisionTime}
                    </dd>
                    <dt>Point</dt>
                    <dd>
                      {alertPosition
                        ? `${Number(alertPosition.x || 0).toFixed(1)}, ${Number(alertPosition.y || 0).toFixed(1)}, ${Number(alertPosition.z || 0).toFixed(1)} km`
                        : '---'}
                    </dd>
                    <dt>Wave</dt>
                    <dd>Hotspot zone radius {zoneRadiusKm.toFixed(0)} km</dd>
                    <dt>Action</dt>
                    <dd>Move the affected satellite(s) to fallback trajectory before {fallbackDeadline}</dd>
                    <dt>Risk</dt>
                    <dd>
                      Miss {Number(alert.miss_distance_km || 0).toFixed(1)} km | Speed {Number(alert.relative_speed_kh || alert.relative_speed_kmh || 0).toFixed(0)} km/h | CPI {Number(alert.cpi_score || 0).toFixed(1)}
                    </dd>
                    {topCascade && (
                      <>
                        <dt>CPI</dt>
                        <dd>Cascade depth {topCascade.cascade_depth} | Burn {topCascade.maneuver.delta_v_ms} m/s</dd>
                      </>
                    )}
                    {relatedCloud && (
                      <>
                        <dt>Debris</dt>
                        <dd>
                          {relatedCloud.id} | affected {Number(relatedCloud.affected_count || 0)} | high risk {Number(relatedCloud.affected_high_risk || 0)}
                        </dd>
                      </>
                    )}
                    {affectedSatellites.length > 0 && (
                      <>
                        <dt>Targets</dt>
                        <dd>
                          {affectedSatellites.map((sat) => `${sat.name || `#${sat.norad_id}`}${sat.risk_band ? `(${sat.risk_band})` : ''}`).join(' | ')}
                        </dd>
                      </>
                    )}
                    {affectedTargets.length > 0 && (
                      <>
                        <dt>Affected</dt>
                        <dd>
                          <div className="sim-targets">
                            {affectedTargets.map((sat) => (
                              <div
                                key={`${alert.sat1?.id || 'a'}-${alert.sat2?.id || 'b'}-${sat.norad_id}`}
                                className={`sim-target ${riskClass(sat.risk_band)}`}
                              >
                                <span className="sim-dot" />
                                <span className="sim-target-name">{sat.name || `#${sat.norad_id}`}</span>
                                <span className="sim-band">
                                  {sat.risk_band || 'low'} | {sat.recommended_action === 'fallback' ? 'MOVE TO FALLBACK' : 'MONITOR'}
                                </span>
                              </div>
                            ))}
                          </div>
                        </dd>
                      </>
                    )}
                  </dl>
                </article>
              );
            })}
          </div>
        )}

        {cascadePlan?.cascade_plan?.length > 0 && (
          <>
            <div className="sim-subhead">
              <span className="ui-label">Maneuver chain</span>
            </div>
            <dl className="sim-dl">
              {cascadePlan.cascade_plan.slice(0, 8).map((maneuver) => (
                <div key={`${maneuver.satellite_id}-${maneuver.cascade_depth}`} className="sim-dl-group">
                  <dt>{maneuver.satellite_name}</dt>
                  <dd>D{maneuver.cascade_depth} | {maneuver.maneuver.delta_v_ms} m/s</dd>
                </div>
              ))}
            </dl>
          </>
        )}

        {hotspots.length > 0 && (
          <>
            <div className="sim-subhead">
              <span className="ui-label">Hotspot wave</span>
            </div>
            <dl className="sim-dl">
              {hotspots.slice(0, 5).map((hotspot, index) => (
                <div key={`${hotspot.sat1?.id}-${hotspot.sat2?.id}-${index}`} className="sim-dl-group">
                  <dt>{hotspot.sat1?.name || 'SAT-1'} × {hotspot.sat2?.name || 'SAT-2'}</dt>
                  <dd>
                    {Number(hotspot.hotspot_score || 0).toFixed(2)} | T{Number(hotspot.tca_minutes || 0).toFixed(0)}m | R{Number(hotspot.zone_radius_km || 100).toFixed(0)}km
                  </dd>
                </div>
              ))}
            </dl>
          </>
        )}

        {debrisClouds.length > 0 && (
          <>
            <div className="sim-subhead">
              <span className="ui-label">Debris alerts</span>
            </div>
            {debrisClouds.slice(0, 5).map((cloud) => (
              <dl className="sim-dl" key={cloud.id}>
                <dt>{cloud.id}</dt>
                <dd>
                  R{Number(cloud.radius_km_at_tca || 0).toFixed(0)}km | {Number(cloud.minutes_to_tca || 0).toFixed(0)}m
                  {' '}| A{Number(cloud.affected_count || 0)} H{Number(cloud.affected_high_risk || 0)}
                </dd>
                {(cloud.radius_timeline || []).length > 0 && (
                  <>
                    <dt>Wave radius</dt>
                    <dd className="is-dim">
                      now {Number(cloud.radius_timeline[0]?.radius_km || 0).toFixed(0)}km | mid {Number(cloud.radius_timeline[1]?.radius_km || 0).toFixed(0)}km | tca {Number(cloud.radius_timeline[2]?.radius_km || 0).toFixed(0)}km
                    </dd>
                  </>
                )}
                {(cloud.affected_satellites || []).slice(0, 4).map((sat) => (
                  <div key={`${cloud.id}-${sat.norad_id}`} className="sim-dl-group">
                    <dt>{sat.name || `#${sat.norad_id}`}</dt>
                    <dd>
                      CPI {Number(sat.cpi_score || 0).toFixed(1)} | {sat.risk_band || 'low'}
                      {sat.recommended_action === 'fallback' ? ' | FALLBACK' : ''}
                    </dd>
                  </div>
                ))}
              </dl>
            ))}
          </>
        )}
      </div>
    </section>
  );
}
