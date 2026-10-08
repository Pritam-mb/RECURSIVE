/**
 * Small fetch helpers for the guided demo. Every request has its own
 * AbortController timeout, and errors carry the server's message so the tour
 * can show something readable next to a Retry button.
 */

export async function gdFetch(path, { method = 'GET', body, timeoutMs = 20000, signal } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error('timeout')), timeoutMs);
  const onOuterAbort = () => controller.abort(new Error('cancelled'));
  if (signal) {
    if (signal.aborted) controller.abort(new Error('cancelled'));
    else signal.addEventListener('abort', onOuterAbort, { once: true });
  }
  try {
    const res = await fetch(path, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    let data = null;
    try { data = await res.json(); } catch { data = null; }
    if (!res.ok) {
      const detail = data?.detail || data?.message || res.statusText;
      throw new Error(`${method} ${path} returned ${res.status}${detail ? `: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}` : ''}`);
    }
    if (data && data.status === 'ERROR') {
      throw new Error(`${method} ${path}: ${data.message || 'server reported an error'}`);
    }
    return data;
  } catch (e) {
    if (controller.signal.aborted) {
      if (signal?.aborted) throw new Error('Cancelled');
      throw new Error(`${method} ${path} timed out after ${Math.round(timeoutMs / 1000)} s`);
    }
    if (e instanceof TypeError) throw new Error(`${method} ${path}: backend unreachable (${e.message})`);
    throw e;
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', onOuterAbort);
  }
}

export const sleep = (ms, signal) =>
  new Promise((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener('abort', () => { clearTimeout(t); reject(new Error('Cancelled')); }, { once: true });
  });

export function pairKey(a, b) {
  return [Number(a), Number(b)].sort((x, y) => x - y).join('-');
}

export function alertPair(alert) {
  return pairKey(alert?.sat1?.id, alert?.sat2?.id);
}

export function fmtPc(p) {
  const v = Number(p);
  if (p == null || !Number.isFinite(v)) return 'n/a';
  if (v === 0) return '0';
  return v.toExponential(1);
}

export function fmtNum(v, d = 1, unit = '') {
  const n = Number(v);
  if (v == null || !Number.isFinite(n)) return 'n/a';
  return `${n.toFixed(d)}${unit}`;
}

export function fmtMiss(km) {
  const n = Number(km);
  if (km == null || !Number.isFinite(n)) return 'n/a';
  return n < 1 ? `${Math.round(n * 1000)} m` : `${n.toFixed(2)} km`;
}

export function fmtDuration(hours) {
  const h = Number(hours);
  if (!Number.isFinite(h)) return 'n/a';
  const mins = Math.round(h * 60);
  if (Math.abs(mins) < 90) return `${mins} min`;
  return `${h.toFixed(1)} h`;
}
