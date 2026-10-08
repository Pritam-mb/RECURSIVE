import { useState, useEffect, useMemo } from 'react';

/**
 * formatTCA(tcaHours)
 * Converts a decimal hours value into a HH:MM:SS countdown string.
 * Returns { formatted, hours, minutes, seconds, urgent, critical }
 */
export function formatTCA(tcaHours) {
  if (tcaHours == null || !Number.isFinite(tcaHours) || tcaHours < 0) {
    return { formatted: '--:--:--', hours: 0, minutes: 0, seconds: 0, urgent: false, critical: false };
  }
  const totalSeconds = Math.floor(tcaHours * 3600);
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  const formatted = [
    String(h).padStart(2, '0'),
    String(m).padStart(2, '0'),
    String(s).padStart(2, '0'),
  ].join(':');
  return {
    formatted,
    hours: h,
    minutes: m,
    seconds: s,
    urgent: tcaHours < 1.0,
    critical: tcaHours < 0.5,
  };
}

/**
 * useTCACountdown(tcaHours, tcaUtc)
 * Counts down once per second toward a fixed target time. The target is
 * `tcaUtc` when given, otherwise "now + tcaHours" captured when that value
 * changes. Anchoring to an absolute time (rather than re-deriving hours from
 * Date.now() each render) keeps the hook's inputs stable between renders.
 * Returns { formatted, urgent, critical }
 */
export function useTCACountdown(tcaHours, tcaUtc) {
  const targetMs = useMemo(() => {
    if (tcaUtc) {
      const parsed = Date.parse(tcaUtc);
      if (Number.isFinite(parsed)) return parsed;
    }
    return tcaHours != null && Number.isFinite(tcaHours) ? Date.now() + (tcaHours * 3_600_000) : null;
  }, [tcaHours, tcaUtc]);

  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    if (targetMs == null) return undefined;
    setNowMs(Date.now());
    const id = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(id);
  }, [targetMs]);

  return formatTCA(targetMs == null ? null : Math.max(0, (targetMs - nowMs) / 3_600_000));
}
