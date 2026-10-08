import { useMemo } from 'react';

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
  }, [btKm, bnKm, semiMajorM, semiMinorM, hbrKm]);

  const angleDeg = (angleRad * 180) / Math.PI;
  const missDistKm = Math.sqrt(btKm * btKm + bnKm * bnKm);
  const insideEllipse =
    ((btKm * Math.cos(angleRad) + bnKm * Math.sin(angleRad)) / (semiMajorM / 1000)) ** 2 +
    ((-btKm * Math.sin(angleRad) + bnKm * Math.cos(angleRad)) / (semiMinorM / 1000)) ** 2 <= 1;

  const missColor = insideEllipse ? '#ef4444' : '#4a90d9';

  // Scale bar: 1 km
  const scaleBarPx = scale;
  const scaleBarY = SIZE - 12;

  return (
    <svg
      className="bplane-svg"
      width={SIZE}
      height={SIZE}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      style={{ background: '#040408', border: '0.5px solid #1f2937' }}
    >
      {/* Axis lines */}
      <line x1={CENTER} y1={4} x2={CENTER} y2={SIZE - 4} stroke="#1f2937" strokeWidth={0.5} />
      <line x1={4} y1={CENTER} x2={SIZE - 4} y2={CENTER} stroke="#1f2937" strokeWidth={0.5} />

      {/* Axis labels */}
      <text x={CENTER + 3} y={10} fill="#6b7280" fontSize={8} fontFamily="monospace">n</text>
      <text x={SIZE - 9} y={CENTER - 3} fill="#6b7280" fontSize={8} fontFamily="monospace">t</text>

      {/* 1-sigma covariance ellipse */}
      <ellipse
        cx={CENTER}
        cy={CENTER}
        rx={ellRx}
        ry={ellRy}
        fill="rgba(74,144,217,0.08)"
        stroke="#4a90d9"
        strokeWidth={0.8}
        strokeDasharray="3 2"
        transform={`rotate(${-angleDeg}, ${CENTER}, ${CENTER})`}
      />

      {/* Hard body radius circle */}
      <circle
        cx={CENTER}
        cy={CENTER}
        r={hbrPx}
        fill="rgba(34,197,94,0.15)"
        stroke="#22c55e"
        strokeWidth={0.8}
      />

      {/* Miss vector line */}
      <line
        x1={CENTER}
        y1={CENTER}
        x2={missX}
        y2={missY}
        stroke={missColor}
        strokeWidth={0.5}
        strokeDasharray="2 2"
        opacity={0.6}
      />

      {/* Miss vector dot */}
      <circle cx={missX} cy={missY} r={3} fill={missColor} />

      {/* Origin cross */}
      <line x1={CENTER - 3} y1={CENTER} x2={CENTER + 3} y2={CENTER} stroke="#6b7280" strokeWidth={0.8} />
      <line x1={CENTER} y1={CENTER - 3} x2={CENTER} y2={CENTER + 3} stroke="#6b7280" strokeWidth={0.8} />

      {/* Scale bar */}
      {scaleBarPx > 4 && (
        <>
          <line
            x1={10}
            y1={scaleBarY}
            x2={10 + Math.min(scaleBarPx, SIZE - 20)}
            y2={scaleBarY}
            stroke="#6b7280"
            strokeWidth={1}
          />
          <text x={10} y={scaleBarY - 3} fill="#6b7280" fontSize={7} fontFamily="monospace">
            {scaleBarPx <= SIZE - 20 ? '1km' : `${((SIZE - 20) / scale).toFixed(1)}km`}
          </text>
        </>
      )}

      {/* Miss distance label */}
      <text x={4} y={SIZE - 3} fill="#6b7280" fontSize={7} fontFamily="monospace">
        |miss|={missDistKm.toFixed(3)}km
      </text>
    </svg>
  );
}
