/**
 * Impact replay controller (CONTRACT3).
 *  - polls GET /api/debris/events every 10 s while mounted (30 s while the
 *    endpoint is missing), and immediately when the live debris set changes;
 *  - fetches each event's replay ONCE (cached) into impact.replay;
 *  - playback: the smooth playhead lives in impactClock (read by the globe on
 *    every Cesium frame); the store's impact.tRelS is published at 10 Hz.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import useStore from '../../store/useStore';
import { playheadNow, pushPlayhead, requestFlyTo } from './impactClock';

const POLL_MS = 10000;
const POLL_MISSING_MS = 30000;
const TICK_MS = 100; // ≤ 10 Hz store / readout updates
const REPLAY_QUERY = 't0_min=-15&t1_min=180&step_s=30&max_fragments=400';

const replayCache = new Map();

export const SPEEDS = [1, 10, 60, 300];

export default function useImpactReplay() {
  const [events, setEvents] = useState([]);
  const [available, setAvailable] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(true);
  const inFlight = useRef(null);

  const impact = useStore((s) => s.impact);
  const setImpact = useStore((s) => s.setImpact);
  const debrisCount = useStore((s) => s.debrisClouds?.length ?? 0);
  const { eventId, replay, playing } = impact;

  // ── Poll the event list ─────────────────────────────────────────────────
  const pollRef = useRef(null);
  useEffect(() => {
    let cancelled = false;
    let timer = null;
    const poll = async () => {
      if (timer) { clearTimeout(timer); timer = null; }
      let delay = POLL_MS;
      try {
        const res = await fetch('/api/debris/events');
        if (res.status === 404) {
          if (!cancelled) { setAvailable(false); setEvents([]); }
          delay = POLL_MISSING_MS;
        } else if (res.ok) {
          const data = await res.json();
          const list = Array.isArray(data?.events) ? data.events : [];
          list.sort((a, b) => Date.parse(b.collision_utc || 0) - Date.parse(a.collision_utc || 0));
          if (!cancelled) { setAvailable(true); setEvents(list); }
        }
      } catch {
        // backend restarting: keep the last list
      }
      if (!cancelled) timer = setTimeout(poll, delay);
    };
    pollRef.current = poll;
    poll();
    return () => {
      cancelled = true;
      pollRef.current = null;
      if (timer) clearTimeout(timer);
    };
  }, []);

  // A new/cleared debris event shows up in the live stream first.
  const firstDebris = useRef(true);
  useEffect(() => {
    if (firstDebris.current) { firstDebris.current = false; return; }
    pollRef.current?.();
  }, [debrisCount]);

  // ── Event selection: follow the list; leave impact mode when it empties ─
  useEffect(() => {
    if (events.length === 0) {
      if (eventId != null || replay != null) setImpact({ eventId: null, replay: null, playing: false });
      return;
    }
    if (!events.some((e) => e.event_id === eventId)) {
      setImpact({ eventId: events[0].event_id, replay: null, playing: false });
    }
  }, [events, eventId, replay, setImpact]);

  // ── Replay fetch: once per event ────────────────────────────────────────
  useEffect(() => {
    if (!open || !eventId) return;
    if (replay?.event_id === eventId) return;
    const cached = replayCache.get(eventId);
    const start = (payload) => {
      const t0 = Number(payload?.t_rel_s?.[0] ?? -900);
      setImpact({ replay: payload, tRelS: t0, playing: false });
    };
    if (cached) { start(cached); return; }
    if (inFlight.current === eventId) return;
    inFlight.current = eventId;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const res = await fetch(`/api/debris/events/${encodeURIComponent(eventId)}/replay?${REPLAY_QUERY}`);
        if (!res.ok) throw new Error(res.status === 404 ? 'Replay endpoint not available yet' : `Replay failed (${res.status})`);
        const payload = await res.json();
        replayCache.set(eventId, payload);
        if (useStore.getState().impact.eventId === eventId) start(payload);
      } catch (e) {
        setError(e.message || 'Replay failed');
      } finally {
        inFlight.current = null;
        setLoading(false);
      }
    })();
  // `events` re-runs this after each poll, so a failed fetch retries every 10 s.
  }, [open, eventId, replay, setImpact, events]);

  // ── 10 Hz publisher while playing; stop at the end of the window ────────
  const t1 = replay?.t_rel_s?.length ? Number(replay.t_rel_s[replay.t_rel_s.length - 1]) : 0;
  const t0 = replay?.t_rel_s?.length ? Number(replay.t_rel_s[0]) : -900;
  useEffect(() => {
    if (!playing || !replay) return undefined;
    const id = setInterval(() => {
      const t = playheadNow();
      if (t >= t1) {
        pushPlayhead(t1);
        useStore.getState().setImpact({ playing: false });
      } else {
        pushPlayhead(t);
      }
    }, TICK_MS);
    return () => clearInterval(id);
  }, [playing, replay, t1]);

  // Pause when the tab is hidden (the globe stops rendering anyway).
  useEffect(() => {
    const onVis = () => {
      if (document.visibilityState === 'hidden' && useStore.getState().impact.playing) {
        useStore.getState().setImpact({ playing: false, tRelS: playheadNow() });
      }
    };
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, []);

  // Leaving the page section: back to live mode.
  useEffect(() => () => useStore.getState().setImpact({ replay: null, playing: false }), []);

  // ── Controls ────────────────────────────────────────────────────────────
  const clamp = useCallback((t) => Math.min(t1, Math.max(t0, t)), [t0, t1]);
  const play = useCallback(() => {
    const t = playheadNow();
    setImpact({ playing: true, tRelS: t >= t1 ? t0 : t });
  }, [setImpact, t0, t1]);
  const pause = useCallback(() => setImpact({ playing: false, tRelS: clamp(playheadNow()) }), [setImpact, clamp]);
  const seek = useCallback((t) => setImpact({ tRelS: clamp(Number(t)) }), [setImpact, clamp]);
  const setSpeed = useCallback((speed) => setImpact({ speed, tRelS: clamp(playheadNow()) }), [setImpact, clamp]);
  const selectEvent = useCallback((id) => setImpact({ eventId: id, replay: null, playing: false }), [setImpact]);
  const event = events.find((e) => e.event_id === eventId) || null;
  const flyTo = useCallback(() => {
    requestFlyTo(event ? { eci: event.collision_point_eci_km, utc: event.collision_utc } : null);
  }, [event]);
  const close = useCallback(() => {
    setOpen(false);
    setImpact({ replay: null, playing: false });
  }, [setImpact]);
  const reopen = useCallback(() => setOpen(true), []);

  return {
    available, events, event, impact, loading, error, open, t0, t1,
    play, pause, seek, setSpeed, selectEvent, flyTo, close, reopen,
  };
}
