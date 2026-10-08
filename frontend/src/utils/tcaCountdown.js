import { useState, useEffect, useRef } from 'react';

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
 * useTCACountdown(tcaHours)
 * React hook that counts down every second from the given TCA hours value.
 * Returns { formatted, urgent, critical }
 */
export function useTCACountdown(tcaHoursInitial) {
  const [remaining, setRemaining] = useState(() =>
    tcaHoursInitial != null && Number.isFinite(tcaHoursInitial) ? tcaHoursInitial : null
  );
  const startRef = useRef(Date.now());
  const initialRef = useRef(tcaHoursInitial);

  // Reset when the prop changes
  useEffect(() => {
    initialRef.current = tcaHoursInitial;
    startRef.current = Date.now();
    setRemaining(tcaHoursInitial);
  }, [tcaHoursInitial]);

  useEffect(() => {
    if (remaining == null) return;
    const id = setInterval(() => {
      const elapsedHours = (Date.now() - startRef.current) / 3_600_000;
      const next = (initialRef.current ?? 0) - elapsedHours;
      setRemaining(next < 0 ? 0 : next);
    }, 1000);
    return () => clearInterval(id);
  }, [remaining == null]); // eslint-disable-line

  return formatTCA(remaining);
}
