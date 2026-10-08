import { useMemo } from 'react';
import useImpactReplay, { SPEEDS } from './useImpactReplay';
import { envelopeAt } from './replayData';
import '../../styles/impact.css';

const IMPACT_BAND_S = 5 * 60; // "IMPACT" phase label spans T0 → T+5 min
const THREAT_WINDOW_S = 15 * 60;

function fmtTRel(t) {
  if (!Number.isFinite(t)) return 'T±--:--:--';
  const sign = t < 0 ? '−' : '+';
  const a = Math.round(Math.abs(t));
  const h = Math.floor(a / 3600);
  const m = Math.floor((a % 3600) / 60);
  const s = a % 60;
  return `T${sign}${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function fmtKm(v) {
  if (v == null || !Number.isFinite(v)) return '—';
  if (v >= 1000) return `${(v / 1000).toFixed(1)}k`;
  if (v >= 100) return v.toFixed(0);
  return v.toFixed(1);
}

function hourTicks(t0, t1) {
  const ticks = [t0, 0];
  for (let t = 3600; t < t1 - 900; t += 3600) ticks.push(t);
  ticks.push(t1);
  return ticks.filter((t, i, a) => t >= t0 && t <= t1 && a.indexOf(t) === i);
}

function phaseOf(t) {
  if (t < 0) return 'APPROACH';
  if (t < IMPACT_BAND_S) return 'IMPACT';
  return 'SPREAD';
}

function PlayIcon({ playing }) {
  return playing ? (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="2" y="1.5" width="3" height="9" fill="currentColor" /><rect x="7" y="1.5" width="3" height="9" fill="currentColor" /></svg>
  ) : (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 1.5v9l7.5-4.5z" fill="currentColor" /></svg>
  );
}

export default function ImpactConsole() {
  const {
    available, events, event, impact, loading, error, open, t0, t1,
    play, pause, seek, setSpeed, selectEvent, flyTo, close, reopen,
  } = useImpactReplay();
  const { replay, tRelS, playing, speed } = impact;

  const env = useMemo(() => envelopeAt(replay?.envelope, tRelS), [replay, tRelS]);
  const threat = useMemo(() => {
    const list = Array.isArray(replay?.threatened) ? replay.threatened : [];
    const cMs = Date.parse(replay?.collision_utc || '');
    const ids = new Set(list.map((x) => String(x.sat_id)));
    let active = 0;
    let next = null;
    for (const x of list) {
      const rel = (Date.parse(x.tca_utc || '') - cMs) / 1000;
      if (!Number.isFinite(rel)) continue;
      if (Math.abs(rel - tRelS) <= THREAT_WINDOW_S) active += 1;
      if (rel >= tRelS && (next == null || rel < next)) next = rel;
    }
    return { total: ids.size, active, next };
  }, [replay, tRelS]);

  if (!available || events.length === 0) return null;

  if (!open) {
    return (
      <button type="button" className="im-pill" onClick={reopen}>
        <span className="im-pill-dot" />
        <span className="ui-label">Impact replay</span>
        <span className="im-pill-n">{events.length}</span>
      </button>
    );
  }

  const span = Math.max(1, t1 - t0);
  const pct = (t) => `${(((t - t0) / span) * 100).toFixed(3)}%`;
  const phase = phaseOf(tRelS);
  const ready = !!replay;
  const fragShown = replay?.fragments?.ids?.length ?? null;
  // envelope.n_alive counts every simulated fragment, not just the displayed sample.
  const fragSim = replay?.provenance?.fragments_simulated ?? event?.fragments_simulated ?? null;
  const step = Number(replay?.step_s) || 30;
  const playheadUtc = ready && Number.isFinite(Date.parse(replay.collision_utc))
    ? new Date(Date.parse(replay.collision_utc) + (tRelS * 1000)).toISOString().slice(11, 19)
    : null;
  const title = event
    ? (event.parent_names || event.parent_ids || []).slice(0, 2).join(' × ')
    : '';

  return (
    <div className={`im-console is-${phase.toLowerCase()}`} role="region" aria-label="Debris impact replay">
      <div className="im-row im-head">
        <span className="im-badge"><span className="im-badge-dot" />Impact replay</span>
        {events.length > 1 ? (
          <select
            className="ui-select im-select"
            value={event?.event_id || ''}
            onChange={(e) => selectEvent(e.target.value)}
            aria-label="Collision event"
          >
            {events.map((e) => (
              <option key={e.event_id} value={e.event_id}>
                {(e.parent_names || e.parent_ids || []).slice(0, 2).join(' × ')} · {String(e.collision_utc || '').slice(11, 16)}Z
              </option>
            ))}
          </select>
        ) : (
          <span className="im-title" title={title}>{title}</span>
        )}
        {event && (
          <span className={`im-status ${event.status === 'active' ? 'is-active' : ''}`}>
            {event.status === 'active' ? 'Debris active' : 'Pending impact'}
          </span>
        )}
        <span className="im-spacer" />
        <button type="button" className="ui-btn im-btn" onClick={flyTo} disabled={!event}>{tRelS > 300 ? 'Fly to debris' : 'Fly to impact'}</button>
        <button type="button" className="im-x" onClick={close} aria-label="Close replay (back to live)" title="Back to live">
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"><path d="M1 1l8 8M9 1l-8 8" stroke="currentColor" strokeWidth="1.4" /></svg>
        </button>
      </div>

      <div className="im-row im-transport">
        <button
          type="button"
          className="ui-btn ui-btn--primary im-play"
          onClick={playing ? pause : play}
          disabled={!ready}
          aria-label={playing ? 'Pause' : 'Play'}
        >
          <PlayIcon playing={playing} />
        </button>
        <div className="im-speeds" role="group" aria-label="Playback speed (sim seconds per second)">
          {SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              className={`im-speed${speed === s ? ' is-on' : ''}`}
              onClick={() => setSpeed(s)}
              aria-pressed={speed === s}
            >
              {s}×
            </button>
          ))}
        </div>
        <div className="im-readout">
          <span className={`im-t ${tRelS >= 0 ? 'is-after' : ''}`}>{fmtTRel(tRelS)}</span>
          <span className="im-phase">{phase}{playheadUtc ? ` · ${playheadUtc}Z` : ''}</span>
        </div>
      </div>

      <div className="im-scrub">
        <div className="im-track" aria-hidden="true">
          <span className="im-band im-band--approach" style={{ left: 0, width: pct(0) }} />
          <span className="im-band im-band--impact" style={{ left: pct(0), width: `calc(${pct(Math.min(t1, IMPACT_BAND_S))} - ${pct(0)})` }} />
          <span className="im-band im-band--spread" style={{ left: pct(Math.min(t1, IMPACT_BAND_S)), right: 0 }} />
          <span className="im-t0" style={{ left: pct(0) }} />
          <span className="im-fill" style={{ width: pct(Math.min(t1, Math.max(t0, tRelS))) }} />
          <span className="im-head" style={{ left: pct(Math.min(t1, Math.max(t0, tRelS))) }} />
          {threat.total > 0 && (replay?.threatened || []).map((x) => {
            const rel = (Date.parse(x.tca_utc || '') - Date.parse(replay.collision_utc || '')) / 1000;
            if (!Number.isFinite(rel) || rel < t0 || rel > t1) return null;
            return <span key={`${x.sat_id}-${x.fragment_id}`} className="im-tca" style={{ left: pct(rel) }} />;
          })}
        </div>
        <input
          className="im-range"
          type="range"
          min={t0}
          max={t1}
          step={step / 3}
          value={Math.min(t1, Math.max(t0, tRelS))}
          onChange={(e) => seek(e.target.value)}
          disabled={!ready}
          aria-label="Replay time relative to collision"
        />
        <div className="im-phases" aria-hidden="true">
          <span className={phase === 'APPROACH' ? 'is-on' : ''} style={{ left: 0 }}>Approach</span>
          <span className={`im-phase-impact${phase === 'IMPACT' ? ' is-on' : ''}`} style={{ left: pct(0) }}>Impact</span>
          <span className={phase === 'SPREAD' ? 'is-on' : ''} style={{ left: pct((Math.min(t1, IMPACT_BAND_S) + t1) / 2) }}>Spread</span>
        </div>
        <div className="im-ticks" aria-hidden="true">
          {hourTicks(t0, t1).map((t) => (
            <span key={t} className={t === 0 ? 'im-tick-t0' : ''} style={{ left: pct(t) }}>
              {t === 0 ? 'T0' : `${t < 0 ? 'T−' : 'T+'}${Math.abs(Math.round(t / 60))}m`}
            </span>
          ))}
        </div>
      </div>

      <div className="im-row im-stats">
        {loading && <span className="im-note">Propagating fragments…</span>}
        {!loading && error && <span className="im-note is-error">{error}</span>}
        {ready && (
          <>
            <Stat label="Alive" value={env?.n_alive != null ? `${env.n_alive}` : (tRelS < 0 ? '0' : '—')} sub={fragSim != null ? `/ ${fragSim} sim` : null} />
            <Stat label="Spread p50" value={tRelS < 0 ? '—' : fmtKm(env?.p50_km)} sub="km" />
            <Stat label="p90" value={tRelS < 0 ? '—' : fmtKm(env?.p90_km)} sub="km" />
            <Stat label="Along-track" value={tRelS < 0 ? '—' : fmtKm(env?.along_track_spread_km)} sub="km" />
            <Stat
              label="Threatened"
              value={`${threat.total}`}
              sub={threat.active > 0 ? `${threat.active} in window` : (threat.next != null ? `next ${fmtTRel(threat.next)}` : null)}
              warn={threat.active > 0}
            />
          </>
        )}
      </div>
      {ready && (
        <div className="im-prov" title={replay?.provenance?.note || ''}>
          {replay?.provenance?.propagator || 'propagated'} · linear interp. between {step}s samples
          {fragShown != null ? ` · ${fragShown} of ${fragSim ?? '?'} simulated fragments drawn` : ''}
          {replay?.provenance?.sampled_from ? ` (${replay.provenance.sampled_from} predicted)` : ''}
          {replay?.threatened?.length && !replay.threatened[0]?.positions ? ' · threatened-sat tracks: client RK4 J2 from live state' : ''}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, sub, warn }) {
  return (
    <div className={`im-stat${warn ? ' is-warn' : ''}`}>
      <span className="ui-label">{label}</span>
      <span className="im-stat-v">{value}{sub ? <small>{sub}</small> : null}</span>
    </div>
  );
}
