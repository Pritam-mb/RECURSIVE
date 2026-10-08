import { useRef, useEffect, useCallback } from 'react';
import useTestMode from '../hooks/useTestMode';
import useStore from '../store/useStore';

const TWO_PI = Math.PI * 2;
const DEG = Math.PI / 180;
// ... (rest of COAST_LINES definition unchanged)
// Let's keep from eciToLatLon to ThreatGlobe component


// Simplified coastline data as lat/lon polylines (very compressed)
// Each sub-array is a connected polyline [[lat,lon],...]
const COAST_LINES = [
  // Africa
  [[37.3,9.5],[36.9,11.0],[22.2,37.1],[11.8,44.9],[1.7,41.6],[-4.7,39.8],[-10.8,40.5],[-26.0,32.9],[-34.8,20.0],[-33.9,18.3],[-29.9,16.7],[-15.8,11.9],[-5.5,5.3],[4.0,2.4],[5.1,1.2],[6.0,2.9],[5.0,5.0],[4.2,7.0],[5.5,11.2],[12.0,15.0],[19.1,12.3],[21.9,23.1],[23.9,32.9],[26.7,33.5],[31.1,32.1],[37.3,9.5]],
  // Europe
  [[71.2,25.8],[69.7,30.0],[65.0,25.5],[60.4,22.0],[56.0,21.0],[54.4,18.5],[54.5,10.0],[57.7,8.0],[58.0,5.4],[55.7,5.0],[51.4,2.6],[47.8,-4.5],[43.3,-8.7],[37.0,-9.0],[36.0,-5.4],[37.5,-0.6],[40.0,0.5],[42.0,3.3],[43.5,7.4],[43.9,15.0],[45.7,13.6],[45.0,14.8],[44.5,14.5],[41.9,12.5],[37.9,15.6],[38.0,15.0],[37.3,15.0],[36.8,11.1],[37.3,9.5]],
  // North America (simplified)
  [[71.4,-156],[70.5,-149],[67.5,-143],[60.4,-145],[59.7,-151],[58.5,-137],[55.5,-133],[49.0,-124],[37.5,-122],[36.6,-121],[34.4,-120],[32.6,-117],[30.0,-110],[25.5,-97],[22.9,-97],[19.6,-87],[15.9,-85],[10.9,-83],[8.0,-77],[8.9,-79],[9.5,-79],[9.4,-82],[9.5,-83],[8.2,-76],[9.6,-75],[11.0,-74],[12.5,-71],[16.0,-61],[20.0,-72],[22.0,-78],[24.0,-81],[25.8,-80],[30.6,-81],[35.0,-75],[38.9,-74],[41.2,-70],[42.0,-69],[44.5,-66],[47.0,-53],[51.2,-55],[53.9,-57],[58.5,-67],[62.0,-70],[66.0,-64],[70.0,-51],[71.0,-52],[71.0,-69],[71.4,-156]],
  // South America (simplified)
  [[11.0,-74],[10.5,-62],[10.6,-61],[10.0,-62],[8.8,-60],[5.2,-52],[4.0,-51],[2.0,-50],[0,-50],[-5,-35],[-8,-35],[-13,-39],[-23,-43],[-23,-44],[-25,-48],[-33,-52],[-34,-53],[-35,-57],[-38,-62],[-42,-65],[-44,-66],[-51,-69],[-55,-65],[-55,-70],[-53,-73],[-50,-75],[-43,-73],[-35,-72],[-25,-70],[-18,-70],[-16,-72],[-18,-70],[-16,-75],[-2,-80],[0,-78],[5,-77],[10,-75],[11,-74]],
  // Asia (simplified)
  [[71.2,25.8],[72.0,52.0],[73.0,68.0],[71.0,87.0],[72.0,105.0],[70.0,131.0],[65.0,141.0],[60.0,163.0],[59.0,164.0],[53.0,159.0],[47.0,142.0],[43.0,132.0],[38.0,121.0],[32.0,122.0],[26.0,120.0],[22.0,114.0],[18.0,110.0],[10.0,104.0],[1.0,104.0],[-5.0,105.0],[-8.0,115.0],[-8.0,124.0],[1.0,131.0],[5.0,126.0],[14.0,120.0],[18.0,122.0],[25.0,122.0],[22.0,114.0],[18.0,110.0],[10.0,104.0],[0.0,103.9],[1.0,103.9],[1.0,104.0],[3.0,103.0],[6.0,102.0],[13.0,100.0],[16.0,98.0],[18.0,92.0],[20.0,86.0],[16.0,81.0],[10.0,79.0],[8.0,77.0],[8.0,76.0],[9.0,78.0],[13.0,80.0],[20.0,87.0],[22.0,91.0],[24.0,90.0],[23.0,89.0],[21.0,88.0],[21.0,86.0],[20.0,86.0],[15.0,74.0],[15.0,73.0],[18.0,73.0],[22.0,70.0],[23.0,68.0],[25.0,67.0],[24.0,63.0],[23.0,58.0],[22.0,59.0],[20.0,58.0],[12.0,44.0],[11.8,44.9],[12.0,45.0],[11.5,43.0],[12.5,43.0],[15.0,42.0],[22.0,37.0],[26.0,33.5],[30.0,33.0],[31.0,32.0],[32.0,34.5],[37.3,35.0],[36.5,36.0],[37.0,37.0],[41.0,36.0],[41.0,30.0],[41.5,28.0],[41.0,29.0],[37.0,27.0],[36.5,28.0],[37.5,26.5],[38.0,26.0],[36.5,22.0],[37.0,22.0],[38.0,21.5],[37.5,22.0],[36.5,22.0],[35.0,24.0],[35.0,25.0],[36.0,28.0],[37.0,27.0],[41.0,29.0],[41.0,28.0],[42.0,28.5],[43.0,28.0],[43.5,28.5],[43.0,30.0],[41.0,30.0],[41.0,36.0],[42.0,41.0],[43.0,41.0],[43.0,40.0],[43.0,51.0],[47.0,53.0],[48.0,59.0],[51.0,60.0],[53.0,59.0],[55.0,60.0],[58.0,62.0],[62.0,60.0],[64.0,40.0],[66.0,33.0],[68.0,31.0],[71.2,25.8]],
  // Australia
  [[-14,130],[-13,136],[-12,136],[-12,135],[-14,130],[-15,129],[-16,123],[-22,114],[-31,115],[-35,117],[-35,118],[-38,140],[-39,144],[-37,147],[-37,150],[-33,152],[-28,153],[-24,152],[-22,150],[-19,147],[-18,147],[-17,146],[-16,145],[-14,144],[-11,143],[-12,142],[-12,136],[-12,132],[-14,130]],
];

/**
 * Convert ECI x,y,z (km) to latitude/longitude (degrees).
 * Treats ECI ≈ ECEF for visualization purposes (acceptable since we only
 * care about approximate globe positions, not precise ground tracks).
 */
function eciToLatLon(x, y, z) {
  const r = Math.sqrt(x * x + y * y + z * z);
  if (r < 1) return null;
  const lat = Math.asin(Math.max(-1, Math.min(1, z / r))) / DEG;
  const lon = Math.atan2(y, x) / DEG;
  return { lat, lon };
}

/**
 * Orthographic projection.
 * viewLon: the longitude (deg) currently centred in the view.
 * Returns { sx, sy, visible } in canvas pixel space.
 */
function orthoProject(lat, lon, viewLon, cx, cy, R) {
  const φ = lat * DEG;
  const λ = (lon - viewLon) * DEG;

  const x = Math.cos(φ) * Math.sin(λ);
  const y = Math.sin(φ);
  const z = Math.cos(φ) * Math.cos(λ); // depth component

  return {
    sx: cx + x * R,
    sy: cy - y * R,
    visible: z >= -0.15, // include slightly past the limb for smooth appearance
    depth: z,
  };
}

function drawCoastlines(ctx, viewLon, cx, cy, R) {
  ctx.strokeStyle = 'rgba(100,140,100,0.35)';
  ctx.lineWidth = 0.7;
  for (const line of COAST_LINES) {
    ctx.beginPath();
    let penDown = false;
    for (const [lat, lon] of line) {
      const p = orthoProject(lat, lon, viewLon, cx, cy, R);
      if (!p.visible) { penDown = false; continue; }
      if (!penDown) { ctx.moveTo(p.sx, p.sy); penDown = true; }
      else ctx.lineTo(p.sx, p.sy);
    }
    ctx.stroke();
  }
}

function drawGrid(ctx, viewLon, cx, cy, R) {
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 0.5;
  // Latitude lines every 30°
  for (let lat = -60; lat <= 60; lat += 30) {
    ctx.beginPath();
    let penDown = false;
    for (let lon = -180; lon <= 180; lon += 3) {
      const p = orthoProject(lat, lon, viewLon, cx, cy, R);
      if (!p.visible) { penDown = false; continue; }
      if (!penDown) { ctx.moveTo(p.sx, p.sy); penDown = true; }
      else ctx.lineTo(p.sx, p.sy);
    }
    ctx.stroke();
  }
  // Longitude lines every 30°
  for (let lon = 0; lon < 360; lon += 30) {
    ctx.beginPath();
    let penDown = false;
    for (let lat = -90; lat <= 90; lat += 3) {
      const p = orthoProject(lat, lon, viewLon, cx, cy, R);
      if (!p.visible) { penDown = false; continue; }
      if (!penDown) { ctx.moveTo(p.sx, p.sy); penDown = true; }
      else ctx.lineTo(p.sx, p.sy);
    }
    ctx.stroke();
  }
}

function satColor(cpi) {
  if (cpi >= 8) return { fill: '#ef4444', glow: 'rgba(239,68,68,0.3)' };
  if (cpi >= 5) return { fill: '#eab308', glow: 'rgba(234,179,8,0.3)' };
  return { fill: '#4a90d9', glow: 'rgba(74,144,217,0.2)' };
}

export default function ThreatGlobe({ alerts = [], satellites = [], selectedSatId }) {
  const canvasRef = useRef(null);
  const stateRef = useRef({ viewLon: 80, animId: null, frame: 0 });
  const testActive = useTestMode((s) => s.testActive);
  const testSatellites = useTestMode((s) => s.testSatellites);
  const computed = useTestMode((s) => s.computed);
  const debrisClouds = useStore((s) => s.debrisClouds);
  // Track satellite index order so labels alternate left/right
  const satIndexRef = useRef(new Map());

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;
    const R = Math.min(W, H) * 0.44;

    const { viewLon, frame } = stateRef.current;

    ctx.clearRect(0, 0, W, H);

    // ── Globe sphere background ────────────────────────────────────────────
    const grd = ctx.createRadialGradient(cx - R * 0.25, cy - R * 0.25, R * 0.05, cx, cy, R);
    grd.addColorStop(0, '#0d1a2a');
    grd.addColorStop(0.6, '#060d14');
    grd.addColorStop(1, '#020608');
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, TWO_PI);
    ctx.fillStyle = grd;
    ctx.fill();

    // Subtle limb glow
    const limbGrd = ctx.createRadialGradient(cx, cy, R * 0.8, cx, cy, R);
    limbGrd.addColorStop(0, 'transparent');
    limbGrd.addColorStop(1, 'rgba(74,144,217,0.12)');
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, TWO_PI);
    ctx.fillStyle = limbGrd;
    ctx.fill();

    // Clip everything to globe circle
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, TWO_PI);
    ctx.clip();

    // ── Grid ─────────────────────────────────────────────────────────────
    drawGrid(ctx, viewLon, cx, cy, R);

    // ── Coastlines ───────────────────────────────────────────────────────
    drawCoastlines(ctx, viewLon, cx, cy, R);

    // ── Debris Clouds ─────────────────────────────────────────────────────
    if (debrisClouds && debrisClouds.length > 0) {
      for (const cloud of debrisClouds) {
        const center = cloud.center_eci_km;
        if (!center) continue;
        const ll = eciToLatLon(center.x, center.y, center.z);
        if (!ll) continue;
        const p = orthoProject(ll.lat, ll.lon, viewLon, cx, cy, R);
        if (!p.visible) continue;

        const radiusPx = Math.max(14, (cloud.radius_km_now || 100) * (R / 6371.0));
        const pulse = 0.5 + 0.5 * Math.sin(frame * 0.07);
        
        ctx.save();
        // Pulsing outer glow
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, radiusPx + 6 + pulse * 4, 0, TWO_PI);
        ctx.fillStyle = `rgba(168, 85, 247, ${0.08 * pulse})`;
        ctx.fill();

        // Translucent purple fill
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, radiusPx, 0, TWO_PI);
        ctx.fillStyle = 'rgba(168, 85, 247, 0.18)';
        ctx.fill();

        // Dashed purple outline (animated)
        ctx.strokeStyle = `rgba(168, 85, 247, ${0.6 + 0.3 * pulse})`;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 3]);
        ctx.lineDashOffset = -frame * 0.3;
        ctx.stroke();
        ctx.setLineDash([]);

        // Inner hazard ring
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, Math.max(4, radiusPx * 0.35), 0, TWO_PI);
        ctx.strokeStyle = 'rgba(251, 113, 133, 0.85)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Center X marker
        const xs = 3;
        ctx.strokeStyle = 'rgba(251, 113, 133, 0.9)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(p.sx - xs, p.sy - xs); ctx.lineTo(p.sx + xs, p.sy + xs); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(p.sx + xs, p.sy - xs); ctx.lineTo(p.sx - xs, p.sy + xs); ctx.stroke();

        // Label — title
        ctx.font = 'bold 9px JetBrains Mono, monospace';
        ctx.fillStyle = 'rgba(216, 180, 254, 1)';
        ctx.textAlign = 'center';
        ctx.fillText('⚠ DEBRIS FIELD', p.sx, p.sy - radiusPx - 14);

        // Fragment count
        ctx.font = '8px JetBrains Mono, monospace';
        ctx.fillStyle = 'rgba(168, 85, 247, 0.95)';
        ctx.fillText(`${cloud.fragment_count || 0} FRAGMENTS`, p.sx, p.sy - radiusPx - 4);

        // Radius
        const radStr = cloud.radius_km_now ? `r=${cloud.radius_km_now.toFixed(0)}km` : '';
        if (radStr) {
          ctx.fillStyle = 'rgba(168, 85, 247, 0.7)';
          ctx.fillText(radStr, p.sx, p.sy + radiusPx + 10);
        }
        ctx.restore();
      }
    }

    // ── Crossing Trajectories (Orbits) for Test Mode ──────────────────────
    if (testActive && computed?.trajectoryA && computed?.trajectoryB) {
      // Draw Trajectory A (Cyan)
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.7)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let penDown = false;
      for (const pt of computed.trajectoryA) {
        const ll = eciToLatLon(pt[1], pt[2], pt[3]);
        if (!ll) continue;
        const p = orthoProject(ll.lat, ll.lon, viewLon, cx, cy, R);
        if (!p.visible) {
          penDown = false;
          continue;
        }
        if (!penDown) {
          ctx.moveTo(p.sx, p.sy);
          penDown = true;
        } else {
          ctx.lineTo(p.sx, p.sy);
        }
      }
      ctx.stroke();

      // Draw Trajectory B (Magenta)
      ctx.strokeStyle = 'rgba(236, 72, 153, 0.7)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      penDown = false;
      for (const pt of computed.trajectoryB) {
        const ll = eciToLatLon(pt[1], pt[2], pt[3]);
        if (!ll) continue;
        const p = orthoProject(ll.lat, ll.lon, viewLon, cx, cy, R);
        if (!p.visible) {
          penDown = false;
          continue;
        }
        if (!penDown) {
          ctx.moveTo(p.sx, p.sy);
          penDown = true;
        } else {
          ctx.lineTo(p.sx, p.sy);
        }
      }
      ctx.stroke();
    }

    // ── Build alert-satellite lookup ──────────────────────────────────────
    const alertSatIds = new Set();
    const satCpiMap = new Map(); // satId → max cpi
    for (const a of alerts) {
      const cpi = Number(a.cpi_score ?? 0);
      if (a.sat1?.id != null) {
        alertSatIds.add(Number(a.sat1.id));
        satCpiMap.set(Number(a.sat1.id), Math.max(satCpiMap.get(Number(a.sat1.id)) ?? 0, cpi));
      }
      if (a.sat2?.id != null) {
        alertSatIds.add(Number(a.sat2.id));
        satCpiMap.set(Number(a.sat2.id), Math.max(satCpiMap.get(Number(a.sat2.id)) ?? 0, cpi));
      }
    }

    // ── Conjunction lines ─────────────────────────────────────────────────
    const satPosMap = new Map(); // satId → {lat,lon,sx,sy,visible}
    let alertSatObjs = satellites.filter((s) => alertSatIds.has(Number(s.norad_id)));

    if (testActive && testSatellites.a && testSatellites.b) {
      const satAId = Number(testSatellites.a.norad_id);
      const satBId = Number(testSatellites.b.norad_id);
      
      const testSatA = {
        norad_id: satAId,
        name: testSatellites.a.name,
        position: testSatellites.a.position,
      };
      const testSatB = {
        norad_id: satBId,
        name: testSatellites.b.name,
        position: testSatellites.b.position,
      };
      
      alertSatObjs = [testSatA, testSatB, ...alertSatObjs];
      
      alertSatIds.add(satAId);
      alertSatIds.add(satBId);
      if (!satCpiMap.has(satAId)) satCpiMap.set(satAId, 10.0);
      if (!satCpiMap.has(satBId)) satCpiMap.set(satBId, 10.0);
    }

    // Build a stable index order for label offset alternation
    let satIdx = 0;
    satIndexRef.current = new Map();
    for (const sat of alertSatObjs) {
      const id = Number(sat.norad_id);
      if (!satIndexRef.current.has(id)) {
        satIndexRef.current.set(id, satIdx++);
      }
    }

    for (const sat of alertSatObjs) {
      const pos = sat.position ?? {};
      const ll = eciToLatLon(pos.x ?? 0, pos.y ?? 0, pos.z ?? 0);
      if (!ll) continue;
      const proj = orthoProject(ll.lat, ll.lon, viewLon, cx, cy, R);
      satPosMap.set(Number(sat.norad_id), { ...proj, ll });
    }

    for (const alert of alerts) {
      const pA = satPosMap.get(alert.sat1?.id);
      const pB = satPosMap.get(alert.sat2?.id);
      if (!pA || !pB || !pA.visible || !pB.visible) continue;
      const cpi = Number(alert.cpi_score ?? 0);
      const color = cpi >= 8 ? 'rgba(239,68,68,0.55)'
        : cpi >= 5 ? 'rgba(234,179,8,0.4)'
        : 'rgba(74,144,217,0.3)';

      // Pulsing dashed line
      ctx.setLineDash([6, 4]);
      ctx.lineDashOffset = -frame * 0.4;
      ctx.strokeStyle = color;
      ctx.lineWidth = cpi >= 8 ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(pA.sx, pA.sy);
      ctx.lineTo(pB.sx, pB.sy);
      ctx.stroke();
      ctx.setLineDash([]);

      // Distance label on midpoint
      const missKm = Number(alert.miss_distance_km ?? 0);
      if (missKm > 0 && pA.visible && pB.visible) {
        const midX = (pA.sx + pB.sx) / 2;
        const midY = (pA.sy + pB.sy) / 2;
        const distStr = missKm < 1 ? `${(missKm * 1000).toFixed(0)}m` : `${missKm.toFixed(1)}km`;
        ctx.save();
        ctx.font = '7px JetBrains Mono, monospace';
        const tw = ctx.measureText(distStr).width;
        ctx.fillStyle = 'rgba(6,13,20,0.88)';
        ctx.fillRect(midX - tw / 2 - 3, midY - 6, tw + 6, 12);
        ctx.fillStyle = cpi >= 8 ? '#ef4444' : cpi >= 5 ? '#eab308' : '#4a90d9';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(distStr, midX, midY);
        ctx.restore();
      }

      // TCA time label
      const tcaHours = Number(alert.tca_hours ?? 0);
      const tcaUtc = alert.tca_utc;
      if ((tcaHours > 0 || tcaUtc) && pA.visible && pB.visible) {
        const midX = (pA.sx + pB.sx) / 2;
        const midY = (pA.sy + pB.sy) / 2 - 14;
        let tcaStr = '';
        if (tcaUtc) {
          // Show time portion only: HH:MM UTC
          tcaStr = `TCA ${tcaUtc.slice(11, 16)} UTC`;
        } else if (tcaHours > 0) {
          tcaStr = `TCA ${tcaHours.toFixed(1)}h`;
        }
        if (tcaStr) {
          ctx.save();
          ctx.font = '7px JetBrains Mono, monospace';
          const tw2 = ctx.measureText(tcaStr).width;
          ctx.fillStyle = 'rgba(6,13,20,0.85)';
          ctx.fillRect(midX - tw2 / 2 - 3, midY - 6, tw2 + 6, 11);
          ctx.fillStyle = 'rgba(251, 191, 36, 0.95)';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(tcaStr, midX, midY);
          ctx.restore();
        }
      }
    }

    // ── Satellites ────────────────────────────────────────────────────────
    for (const [satId, proj] of satPosMap.entries()) {
      if (!proj.visible) continue;
      const cpi = satCpiMap.get(satId) ?? 0;
      const { fill, glow } = satColor(cpi);
      const isSelected = satId === selectedSatId;
      const r = isSelected ? 6 : (cpi >= 8 ? 5 : 4);

      // Glow halo
      ctx.beginPath();
      ctx.arc(proj.sx, proj.sy, r + 4, 0, TWO_PI);
      ctx.fillStyle = glow;
      ctx.fill();

      // Dot
      ctx.beginPath();
      ctx.arc(proj.sx, proj.sy, r, 0, TWO_PI);
      ctx.fillStyle = fill;
      ctx.fill();

      if (isSelected) {
        ctx.beginPath();
        ctx.arc(proj.sx, proj.sy, r + 2, 0, TWO_PI);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Pulse ring for critical
      if (cpi >= 8) {
        const pulse = 0.5 + 0.5 * Math.sin(frame * 0.1);
        ctx.beginPath();
        ctx.arc(proj.sx, proj.sy, r + 4 + pulse * 4, 0, TWO_PI);
        ctx.strokeStyle = `rgba(239,68,68,${0.4 * pulse})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Label — alternate left/right based on stable insertion order
      const sat = alertSatObjs.find((s) => Number(s.norad_id) === satId);
      if (sat) {
        const label = sat.name?.slice(0, 14) ?? `#${satId}`;
        ctx.font = '8px JetBrains Mono, monospace';
        ctx.fillStyle = fill;

        // Even-indexed satellites label right, odd label left
        const idxOrder = satIndexRef.current.get(satId) ?? 0;
        if (idxOrder % 2 === 1) {
          ctx.save();
          ctx.textAlign = 'right';
          ctx.fillText(label, proj.sx - r - 3, proj.sy + 3);
          ctx.restore();
        } else {
          ctx.fillText(label, proj.sx + r + 3, proj.sy + 3);
        }
      }
    }

    ctx.restore();

    // ── Globe border ─────────────────────────────────────────────────────
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, TWO_PI);
    ctx.strokeStyle = 'rgba(74,144,217,0.2)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // ── Labels overlay ───────────────────────────────────────────────────
    ctx.font = '8px JetBrains Mono, monospace';
    ctx.fillStyle = 'rgba(107,114,128,0.8)';
    ctx.fillText(`${alerts.length} ACTIVE CONJUNCTIONS`, 8, H - 8);
    ctx.fillText(`${alertSatIds.size} SATELLITES AT RISK`, 8, H - 18);
  }, [alerts, satellites, selectedSatId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Resize canvas to container
    const ro = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    });
    ro.observe(canvas);
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const tick = () => {
      stateRef.current.viewLon += 0.08; // slow rotation
      stateRef.current.frame++;
      draw();
      stateRef.current.animId = requestAnimationFrame(tick);
    };
    stateRef.current.animId = requestAnimationFrame(tick);

    return () => {
      ro.disconnect();
      if (stateRef.current.animId) cancelAnimationFrame(stateRef.current.animId);
    };
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '100%', display: 'block', cursor: 'crosshair' }}
    />
  );
}
