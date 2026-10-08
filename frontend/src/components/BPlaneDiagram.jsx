import { useMemo } from 'react';
import '../styles/threats.css';

/**
 * BPlaneDiagram
 * A 200×200 SVG showing the B-plane geometry for a conjunction event.
 *
 * Props:
 *   btKm        — B-plane tangential component (km)
 *   bnKm        — B-plane normal component (km)
 *   semiMajorM  — covariance ellipse semi-major axis (metres)
 *   semiMinorM  — covariance ellipse semi-minor axis (metres)
 *   angleRad    — ellipse rotation angle (radians)
 *   hbrKm       — hard body radius (km, default 0.010)
 */

// Muted chrome, state colours reserved for the threat geometry.
const GRID = 'var(--c-line)';
const AXIS = 'var(--c-line-strong)';
const LABEL = 'var(--c-text-dim)';
const FONT = 'IBM Plex Mono, ui-monospace, Consolas, monospace';
const GRID_STEPS = [-0.3, -0.15, 0.15, 0.3]; // fractions of SIZE from center

export default function BPlaneDiagram({
  btKm = 0,
  bnKm = 0,
  semiMajorM = 3000,
  semiMinorM = 1000,
  angleRad = 0,
  hbrKm = 0.010,
}) {
  const SIZE = 200;
  const CENTER = SIZE / 2;

  const { scale, missX, missY, ellRx, ellRy, hbrPx } = useMemo(() => {
    // Compute the scale so that the covariance ellipse fits within ~80% of the SVG
    const semiMajorKm = semiMajorM / 1000;
    const semiMinorKm = semiMinorM / 1000;

    // The range that needs to fit: max of miss vector distance, semi-major, hbr
    const missDistKm = Math.sqrt(btKm * btKm + bnKm * bnKm);
    const maxRange = Math.max(semiMajorKm * 1.5, missDistKm * 1.3, hbrKm * 5, 0.1);
    const scale = (SIZE * 0.4) / maxRange; // pixels per km

    return {
      scale,
      missX: CENTER + btKm * scale,
      missY: CENTER - bnKm * scale, // SVG Y is inverted
      ellRx: Math.max(semiMajorKm * scale, 2),
      ellRy: Math.max(semiMinorKm * scale, 1),
      hbrPx: Math.max(hbrKm * scale, 2),
    };
  }, [btKm, bnKm, semiMajorM, semiMinorM, hbrKm, CENTER]);

  const angleDeg = (angleRad * 180) / Math.PI;
  const missDistKm = Math.sqrt(btKm * btKm + bnKm * bnKm);
  const insideEllipse =
    ((btKm * Math.cos(angleRad) + bnKm * Math.sin(angleRad)) / (semiMajorM / 1000)) ** 2 +
    ((-btKm * Math.sin(angleRad) + bnKm * Math.cos(angleRad)) / (semiMinorM / 1000)) ** 2 <= 1;

  // Miss vector: warning when it falls inside the 1σ ellipse, nominal otherwise.
  const missColor = insideEllipse ? 'var(--c-warning)' : 'var(--c-nominal)';
  const ellipseColor = 'var(--c-caution)';

  // Scale bar: 1 km
  const scaleBarPx = scale;
  const scaleBarY = SIZE - 16;

  return (
    <svg
      className="tq-bplane-svg"
      width={SIZE}
      height={SIZE}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      role="img"
      aria-label={`B-plane: miss ${missDistKm.toFixed(3)} km, ${insideEllipse ? 'inside' : 'outside'} 1-sigma ellipse`}
    >
      {/* Grid */}
      {GRID_STEPS.map((f) => (
        <g key={f}>
          <line x1={CENTER + f * SIZE} y1={0} x2={CENTER + f * SIZE} y2={SIZE} stroke={GRID} strokeWidth={0.5} />
          <line x1={0} y1={CENTER + f * SIZE} x2={SIZE} y2={CENTER + f * SIZE} stroke={GRID} strokeWidth={0.5} />
        </g>
      ))}

      {/* Axis lines */}
      <line x1={CENTER} y1={0} x2={CENTER} y2={SIZE} stroke={AXIS} strokeWidth={0.75} />
      <line x1={0} y1={CENTER} x2={SIZE} y2={CENTER} stroke={AXIS} strokeWidth={0.75} />

      {/* Axis labels */}
      <text x={CENTER + 4} y={10} fill={LABEL} fontSize={8} fontFamily={FONT}>Bn</text>
      <text x={SIZE - 14} y={CENTER - 4} fill={LABEL} fontSize={8} fontFamily={FONT}>Bt</text>

      {/* 1-sigma covariance ellipse */}
      <ellipse
        cx={CENTER}
        cy={CENTER}
        rx={ellRx}
        ry={ellRy}
        fill="none"
        stroke={ellipseColor}
        strokeWidth={1}
        strokeDasharray="3 2"
        transform={`rotate(${-angleDeg}, ${CENTER}, ${CENTER})`}
      />

      {/* Hard body radius circle */}
      <circle
        cx={CENTER}
        cy={CENTER}
        r={hbrPx}
        fill="none"
        stroke="var(--c-text)"
        strokeWidth={0.75}
      />

      {/* Miss vector line */}
      <line
        x1={CENTER}
        y1={CENTER}
        x2={missX}
        y2={missY}
        stroke={missColor}
        strokeWidth={1}
      />

      {/* Miss vector point */}
      <circle cx={missX} cy={missY} r={2.5} fill={missColor} />

      {/* Origin cross */}
      <line x1={CENTER - 3} y1={CENTER} x2={CENTER + 3} y2={CENTER} stroke="var(--c-text)" strokeWidth={0.75} />
      <line x1={CENTER} y1={CENTER - 3} x2={CENTER} y2={CENTER + 3} stroke="var(--c-text)" strokeWidth={0.75} />

      {/* Scale bar */}
      {scaleBarPx > 4 && (
        <>
          <line
            x1={8}
            y1={scaleBarY}
            x2={8 + Math.min(scaleBarPx, SIZE - 20)}
            y2={scaleBarY}
            stroke={LABEL}
            strokeWidth={1}
          />
          <line x1={8} y1={scaleBarY - 2} x2={8} y2={scaleBarY + 2} stroke={LABEL} strokeWidth={1} />
          <line
            x1={8 + Math.min(scaleBarPx, SIZE - 20)}
            y1={scaleBarY - 2}
            x2={8 + Math.min(scaleBarPx, SIZE - 20)}
            y2={scaleBarY + 2}
            stroke={LABEL}
            strokeWidth={1}
          />
          <text x={8} y={scaleBarY - 4} fill={LABEL} fontSize={7} fontFamily={FONT}>
            {scaleBarPx <= SIZE - 20 ? '1 km' : `${((SIZE - 20) / scale).toFixed(1)} km`}
          </text>
        </>
      )}

      {/* Miss distance label */}
      <text x={8} y={SIZE - 5} fill={LABEL} fontSize={7} fontFamily={FONT}>
        |B| {missDistKm.toFixed(3)} km
      </text>
    </svg>
  );
}
