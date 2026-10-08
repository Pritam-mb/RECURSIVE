/**
 * api.js — Centralised fetch helpers.
 * Derives the API base URL from the current window location so no
 * hardcoded localhost URLs exist anywhere in the frontend.
 */

const getBase = () => {
  if (typeof window === 'undefined') return '';
  // In dev the Vite proxy rewrites /api → backend, so empty base works.
  // In production the frontend is served from the same origin as the API.
  return '';
};

// A request that never answers (backend busy or restarting) must not leave
// the UI waiting forever: abort it and let the caller show an error.
const GET_TIMEOUT_MS = 20000;
const ACTION_TIMEOUT_MS = 60000;

async function fetchWithTimeout(url, options = {}, timeoutMs = GET_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (e) {
    if (e?.name === 'AbortError') throw new Error(`timed out after ${Math.round(timeoutMs / 1000)} s`);
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export async function apiGet(path, timeoutMs = GET_TIMEOUT_MS) {
  const res = await fetchWithTimeout(`${getBase()}${path}`, {}, timeoutMs);
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json();
}

export async function apiPost(path, body = {}) {
  const res = await fetchWithTimeout(`${getBase()}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }, ACTION_TIMEOUT_MS);
  if (!res.ok) {
    // Attach status + parsed JSON body (if any) so callers can surface
    // server-provided messages (e.g. 409 { ok:false, message }).
    const err = new Error(`POST ${path} → ${res.status}`);
    err.status = res.status;
    err.body = await res.json().catch(() => null);
    throw err;
  }
  return res.json();
}

export async function apiDelete(path) {
  const res = await fetchWithTimeout(`${getBase()}${path}`, { method: 'DELETE' }, ACTION_TIMEOUT_MS);
  if (!res.ok) throw new Error(`DELETE ${path} → ${res.status}`);
  return res.json().catch(() => ({}));
}
