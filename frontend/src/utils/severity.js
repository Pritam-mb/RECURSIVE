// Operator severity tier for a conjunction alert: CRITICAL / WARNING / WATCH.
// Shared by the queue rows and the queue filter tabs so they always agree.
export function severityLabel(alert) {
  const cpi = Number(alert.cpi_score ?? 0);
  // Backend may send colour words; show operator terms instead.
  const sev = String(alert.severity ?? '').toUpperCase();
  if (sev === 'RED' || sev === 'CRITICAL') return 'CRITICAL';
  if (sev === 'YELLOW' || sev === 'AMBER' || sev === 'WARNING') return 'WARNING';
  if (sev === 'GREEN' || sev === 'WATCH' || sev === 'NOMINAL') return cpi >= 8 ? 'CRITICAL' : cpi >= 5 ? 'WARNING' : 'WATCH';
  if (cpi >= 8) return 'CRITICAL';
  if (cpi >= 5) return 'WARNING';
  return 'WATCH';
}
