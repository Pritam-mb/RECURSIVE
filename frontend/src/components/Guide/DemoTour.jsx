import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { TOUR_STEPS, resetDemo } from './tourSteps';
import '../../styles/guide.css';

const N = TOUR_STEPS.length;

function resolveTarget(sel, ctx) {
  if (sel === '#__gd_row__') return ctx.rowId ? document.getElementById(`threat-card-${ctx.rowId}`) : null;
  try { return document.querySelector(sel); } catch { return null; }
}

function rectOf(el) {
  if (!el) return null;
  const r = el.getBoundingClientRect();
  if (r.width < 4 || r.height < 4) return null;
  return { x: r.left, y: r.top, w: r.width, h: r.height };
}

/** Rings around the elements the current step talks about (no dimming, no pointer capture). */
function Spotlight({ targets, ctx }) {
  const [rects, setRects] = useState([]);
  useEffect(() => {
    const update = () => {
      const out = [];
      for (const sel of targets) {
        const r = rectOf(resolveTarget(sel, ctx));
        if (r) {
          // Clip to the queue list when the target is a scrolled row.
          out.push(r);
          if (sel === '#__gd_row__') break; // the row replaces the queue fallback
        }
      }
      setRects(out);
    };
    update();
    const id = setInterval(update, 400);
    window.addEventListener('resize', update);
    return () => { clearInterval(id); window.removeEventListener('resize', update); };
  }, [targets, ctx]);
  return rects.map((r, i) => (
    <div
      key={i}
      className="gd-spot"
      style={{ left: r.x - 3, top: Math.max(0, r.y - 3), width: r.w + 6, height: Math.min(r.h + 6, window.innerHeight - Math.max(0, r.y - 3)) }}
    />
  ));
}

export default function DemoTour({ open, onClose }) {
  const [idx, setIdx] = useState(0);
  const [states, setStates] = useState(() => TOUR_STEPS.map(() => ({ status: 'idle' })));
  const [resetState, setResetState] = useState(null);
  const ctxRef = useRef({});
  const abortRef = useRef(null);
  const mounted = useRef(true);

  useEffect(() => {
    // StrictMode mounts twice: re-arm on every mount.
    mounted.current = true;
    return () => { mounted.current = false; abortRef.current?.abort(); };
  }, []);

  const patch = useCallback((i, p) => {
    if (!mounted.current) return;
    setStates((prev) => prev.map((s, j) => (j === i ? { ...s, ...p } : s)));
  }, []);

  const runStep = useCallback(async (i) => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    patch(i, { status: 'running', progress: 'Working …', error: null, startedAt: Date.now() });
    try {
      const data = await TOUR_STEPS[i].run({
        signal: ac.signal,
        ctx: ctxRef.current,
        progress: (msg) => { if (!ac.signal.aborted) patch(i, { progress: msg }); },
      });
      if (!ac.signal.aborted) patch(i, { status: 'done', data, progress: null });
    } catch (e) {
      if (!ac.signal.aborted) patch(i, { status: 'error', error: e?.message || String(e), progress: null });
    }
  }, [patch]);

  // Start (or restart) the tour at step 1 when opened.
  const wasOpen = useRef(false);
  useEffect(() => {
    if (open && !wasOpen.current) {
      ctxRef.current = {};
      setStates(TOUR_STEPS.map(() => ({ status: 'idle' })));
      setResetState(null);
      setIdx(0);
      runStep(0);
    }
    if (!open && wasOpen.current) abortRef.current?.abort();
    wasOpen.current = open;
  }, [open, runStep]);

  const go = (next) => {
    if (next < 0 || next >= N) return;
    setIdx(next);
    const st = states[next];
    if (st.status === 'idle' || st.status === 'error') runStep(next);
  };

  const onReset = useCallback(async () => {
    setResetState({ status: 'running', progress: 'Resetting …' });
    try {
      await resetDemo((m) => mounted.current && setResetState({ status: 'running', progress: m }));
      if (mounted.current) setResetState({ status: 'done' });
    } catch (e) {
      if (mounted.current) setResetState({ status: 'error', error: e?.message || String(e) });
    }
  }, []);

  // Esc closes; arrow keys navigate when nothing is running.
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.target && /input|select|textarea/i.test(e.target.tagName)) return;
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const step = TOUR_STEPS[idx];
  const st = states[idx];
  const running = st.status === 'running';
  const elapsed = running && st.startedAt ? Math.round((Date.now() - st.startedAt) / 1000) : null;
  const placeTop = step.panel === 'top';

  return createPortal(
    <>
      <Spotlight targets={step.targets} ctx={ctxRef.current} />
      <div
        className={`gd-tour${placeTop ? ' is-top' : ''}`}
        role="dialog"
        aria-label={`Demo tour, step ${idx + 1} of ${N}: ${step.title}`}
      >
        <div className="gd-tour-head">
          <span className="gd-tour-kicker">Demo tour · step {idx + 1}/{N}</span>
          <span className="gd-tour-title">{step.title}</span>
          <button type="button" className="gd-x" onClick={onClose} aria-label="Close demo tour" title="Close (Esc)">×</button>
        </div>
        <div className="gd-progress" aria-hidden="true">
          {TOUR_STEPS.map((s, i) => (
            <span
              key={s.key}
              className={`gd-progress-seg${i === idx ? ' is-current' : ''}${states[i].status === 'done' ? ' is-done' : ''}${states[i].status === 'error' ? ' is-error' : ''}`}
              title={`${i + 1}. ${s.title}`}
            />
          ))}
        </div>

        <div className="gd-tour-body" aria-live="polite">
          {running && (
            <div className="gd-wait">
              <span className="gd-spinner" aria-hidden="true" />
              <span>{st.progress || 'Working …'}</span>
              {elapsed != null && elapsed > 2 && !/\d+ s/.test(st.progress || '') && <span className="gd-dim"> {elapsed} s</span>}
            </div>
          )}
          {st.status === 'error' && (
            <div className="gd-error" role="alert">
              <span className="gd-err">Step failed: {st.error}</span>
              <button type="button" className="ui-btn" onClick={() => runStep(idx)}>Retry</button>
            </div>
          )}
          {st.status === 'done' && step.render(st.data, { onReset, resetState, ctx: ctxRef.current })}
        </div>

        <div className="gd-tour-foot">
          <button type="button" className="ui-btn" onClick={() => go(idx - 1)} disabled={idx === 0}>Back</button>
          {idx < N - 1 && (
            <button type="button" className="gd-link" onClick={onReset} disabled={resetState?.status === 'running'} title="Clock to real time, clear debris and scenario">
              {resetState?.status === 'running' ? 'Resetting…' : resetState?.status === 'done' ? 'Demo reset ✓' : 'Reset demo'}
            </button>
          )}
          <span className="gd-spacer" />
          {idx < N - 1 ? (
            <button type="button" className="ui-btn ui-btn--primary" onClick={() => go(idx + 1)} disabled={running}>
              {running ? 'Working…' : `Next: ${TOUR_STEPS[idx + 1].title}`}
            </button>
          ) : (
            <button type="button" className="ui-btn ui-btn--primary" onClick={onClose}>Finish</button>
          )}
        </div>
      </div>
    </>,
    document.body,
  );
}
