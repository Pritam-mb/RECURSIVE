/**
 * Coordinate conversion utilities.
 * ECI (Earth-Centered Inertial) -> ECEF (Earth-Centered Earth-Fixed) -> LLA
 * Uses Greenwich Sidereal Time rotation matrix.
 */

/**
 * Compute Greenwich Mean Sidereal Time from a JS Date.
 * Returns angle in radians.
 */
export function computeGmst(date) {
  const JD_J2000 = 2451545.0;
  const msPerDay = 86400000;

  // Julian Date
  const jd = date.getTime() / msPerDay + 2440587.5;
  const T = (jd - JD_J2000) / 36525.0;

  // GMST in degrees
  let gmstDeg =
    280.46061837 +
    360.98564736629 * (jd - JD_J2000) +
    0.000387933 * T * T -
    (T * T * T) / 38710000.0;

  // Normalize to 0-360
  gmstDeg = ((gmstDeg % 360) + 360) % 360;

  return (gmstDeg * Math.PI) / 180.0;
}

/**
 * Convert ECI position to ECEF using GST rotation matrix.
 *
 * R_GST = [ cos(θ)   sin(θ)   0 ]
 *         [-sin(θ)   cos(θ)   0 ]
 *         [  0         0      1 ]
 *
 * @param {Object} eci - {x, y, z} in km (ECI frame)
 * @param {number} gmst - Greenwich Mean Sidereal Time in radians
 * @returns {Object} {x, y, z} in km (ECEF frame)
 */
export function eciToEcef(eci, gmst) {
  const cosG = Math.cos(gmst);
  const sinG = Math.sin(gmst);

  return {
    x: eci.x * cosG + eci.y * sinG,
    y: -eci.x * sinG + eci.y * cosG,
    z: eci.z,
  };
}

/**
 * Convert ECEF (km) to Cesium Cartesian3 (meters).
 * Cesium uses meters internally.
 *
 * @param {Object} ecef - {x, y, z} in km
 * @returns {Object} {x, y, z} in meters
 */
export function ecefToCartesian(ecef) {
  return {
    x: ecef.x * 1000,
    y: ecef.y * 1000,
    z: ecef.z * 1000,
  };
}

/**
 * Convert ECEF position to geodetic coordinates (lat, lon, alt).
 * Uses iterative method for WGS84 ellipsoid.
 *
 * @param {Object} ecef - {x, y, z} in km
 * @returns {Object} {lat, lon, alt} in degrees and km
 */
export function ecefToLla(ecef) {
  const a = 6378.137; // WGS84 semi-major axis (km)
  const f = 1.0 / 298.257223563;
  const b = a * (1 - f);
  const e2 = 1 - (b * b) / (a * a);

  const x = ecef.x;
  const y = ecef.y;
  const z = ecef.z;

  const lon = Math.atan2(y, x);
  const p = Math.sqrt(x * x + y * y);

  // Iterative latitude computation
  let lat = Math.atan2(z, p * (1 - e2));
  for (let i = 0; i < 10; i++) {
    const sinLat = Math.sin(lat);
    const N = a / Math.sqrt(1 - e2 * sinLat * sinLat);
    lat = Math.atan2(z + e2 * N * sinLat, p);
  }

  const sinLat = Math.sin(lat);
  const N = a / Math.sqrt(1 - e2 * sinLat * sinLat);
  const alt = p / Math.cos(lat) - N;

  return {
    lat: (lat * 180) / Math.PI,
    lon: (lon * 180) / Math.PI,
    alt: alt,
  };
}

/**
 * Full pipeline: ECI -> ECEF -> Cesium Cartesian3
 * @param {Object} eciPos - {x, y, z} in km
 * @param {Date} date - timestamp for GMST calculation
 * @returns {Object} {x, y, z} in meters (Cesium-ready)
 */
export function eciToCesiumCartesian(eciPos, date) {
  const gmst = computeGmst(date);
  const ecef = eciToEcef(eciPos, gmst);
  return ecefToCartesian(ecef);
}

/**
 * Full pipeline: ECI -> geodetic
 * @param {Object} eciPos - {x, y, z} in km
 * @param {Date} date - timestamp for GMST calculation
 * @returns {Object} {lat, lon, alt} in degrees and km
 */
export function eciToLla(eciPos, date) {
  const gmst = computeGmst(date);
  const ecef = eciToEcef(eciPos, gmst);
  return ecefToLla(ecef);
}

/**
 * Fast RK4 two-body Keplerian orbit propagator.
 * Generating the orbit locally provides an instantaneous unbroken orbital line.
 * @param {Object} satellite - Full satellite object containing position, velocity, epoch_utc
 * @param {number} spanMinutes - How much time to look ahead
 * @param {number} stepSeconds - Timestep precision
 * @returns {Array} Array of orbit state objects { position, epoch_utc }
 */
export function generateKeplerianOrbit(satellite, spanMinutes = 120, stepSeconds = 20) {
  if (!satellite?.position || !satellite?.velocity || !satellite?.epoch_utc) return [];

  const mu = 398600.4418; // Earth's gravitational parameter, km^3/s^2

  function derivative(state) {
    const r2 = state.x * state.x + state.y * state.y + state.z * state.z;
    const rMag3 = Math.pow(r2, 1.5);
    return {
      vx: state.vx,
      vy: state.vy,
      vz: state.vz,
      ax: (-mu * state.x) / rMag3,
      ay: (-mu * state.y) / rMag3,
      az: (-mu * state.z) / rMag3,
    };
  }

  let state = {
    x: Number(satellite.position.x ?? satellite.position[0] ?? 0),
    y: Number(satellite.position.y ?? satellite.position[1] ?? 0),
    z: Number(satellite.position.z ?? satellite.position[2] ?? 0),
    vx: Number(satellite.velocity.vx ?? satellite.velocity.x ?? satellite.velocity[0] ?? 0),
    vy: Number(satellite.velocity.vy ?? satellite.velocity.y ?? satellite.velocity[1] ?? 0),
    vz: Number(satellite.velocity.vz ?? satellite.velocity.z ?? satellite.velocity[2] ?? 0),
  };
  
  const orbit = [];
  const startMs = new Date(satellite.epoch_utc).getTime();
  
  if (isNaN(startMs)) return [];
  
  const totalSteps = Math.ceil((spanMinutes * 60) / stepSeconds);
  let currentMs = startMs;

  for (let i = 0; i <= totalSteps; i++) {
    orbit.push({
      position: { x: state.x, y: state.y, z: state.z },
      epoch_utc: new Date(currentMs).toISOString(),
    });

    const dt = stepSeconds;
    const d1 = derivative(state);

    const s2 = {
      x: state.x + (d1.vx * dt) / 2, y: state.y + (d1.vy * dt) / 2, z: state.z + (d1.vz * dt) / 2,
      vx: state.vx + (d1.ax * dt) / 2, vy: state.vy + (d1.ay * dt) / 2, vz: state.vz + (d1.az * dt) / 2,
    };
    const d2 = derivative(s2);

    const s3 = {
      x: state.x + (d2.vx * dt) / 2, y: state.y + (d2.vy * dt) / 2, z: state.z + (d2.vz * dt) / 2,
      vx: state.vx + (d2.ax * dt) / 2, vy: state.vy + (d2.ay * dt) / 2, vz: state.vz + (d2.az * dt) / 2,
    };
    const d3 = derivative(s3);

    const s4 = {
      x: state.x + d3.vx * dt, y: state.y + d3.vy * dt, z: state.z + d3.vz * dt,
      vx: state.vx + d3.ax * dt, vy: state.vy + d3.ay * dt, vz: state.vz + d3.az * dt,
    };
    const d4 = derivative(s4);

    state.x += (dt / 6) * (d1.vx + 2 * d2.vx + 2 * d3.vx + d4.vx);
    state.y += (dt / 6) * (d1.vy + 2 * d2.vy + 2 * d3.vy + d4.vy);
    state.z += (dt / 6) * (d1.vz + 2 * d2.vz + 2 * d3.vz + d4.vz);

    state.vx += (dt / 6) * (d1.ax + 2 * d2.ax + 2 * d3.ax + d4.ax);
    state.vy += (dt / 6) * (d1.ay + 2 * d2.ay + 2 * d3.ay + d4.ay);
    state.vz += (dt / 6) * (d1.az + 2 * d2.az + 2 * d3.az + d4.az);

    currentMs += stepSeconds * 1000;
  }

  return orbit;
}
