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

export async function apiGet(path) {
  const res = await fetch(`${getBase()}${path}`);
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json();
}

export async function apiPost(path, body = {}) {
  const res = await fetch(`${getBase()}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json();
}

export async function apiDelete(path) {
  const res = await fetch(`${getBase()}${path}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`DELETE ${path} → ${res.status}`);
  return res.json().catch(() => ({}));
}
