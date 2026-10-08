// Operator severity tier for a conjunction alert: CRITICAL / WARNING / WATCH.
// Shared by the queue rows, filter tabs, telemetry strip and cascade graph.
//
// The backend tier (CRITICAL / WARNING / WATCH, from Pc and miss distance) is
// authoritative. Legacy colour words (RED / YELLOW / AMBER / GREEN) are still
// accepted. Only when no tier is sent at all do we fall back to the CPI score.
const TIER = {
  CRITICAL: 'CRITICAL',
  RED: 'CRITICAL',
  WARNING: 'WARNING',
  YELLOW: 'WARNING',
  AMBER: 'WARNING',
  ORANGE: 'WARNING',
  WATCH: 'WATCH',
  GREEN: 'WATCH',
  NOMINAL: 'WATCH',
};

export function normalizeSeverity(value) {
  return TIER[String(value ?? '').toUpperCase()] ?? null;
}

export function severityLabel(alert) {
  const tier = normalizeSeverity(alert?.severity);
  if (tier) return tier;
  const cpi = Number(alert?.cpi_score ?? 0);
  if (cpi >= 8) return 'CRITICAL';
  if (cpi >= 5) return 'WARNING';
  return 'WATCH';
}

/** CRITICAL / WARNING / WATCH → flight-console state class suffix. */
export function severityState(tier) {
  if (tier === 'CRITICAL') return 'warning';
  if (tier === 'WARNING') return 'caution';
  return 'nominal';
}
