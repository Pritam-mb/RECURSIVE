"""
Coordinate frame helpers: geodetic (WGS84) <-> ECEF <-> ECI (GMST), plus
geodetic conversion of a propagated ECI position to a sub-satellite point.

Used by the conjunction predictor so that operators can specify a satellite by
ground latitude/longitude/altitude and receive the affected geographic region.

Conventions:
  - ECEF x -> 0° E prime meridian, z -> Earth's rotation axis.
  - ECI x -> vernal equinox at epoch (IAU-82 GMST approximation, demo accuracy).
  - r_eci = Rz(gmst) @ r_ecef  (rotation by GMST about +Z).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

RE_WGS84_A = 6378.137        # km, equatorial radius
RE_WGS84_B = 6356.752314245  # km, polar radius
WGS84_E2 = 1.0 - (RE_WGS84_B / RE_WGS84_A) ** 2

J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def gmst_rad(dt: datetime) -> float:
    """IAU-82 Greenwich Mean Sidereal Time approximation, in radians."""
    dt = _as_utc(dt)
    days = (dt - J2000).total_seconds() / 86400.0
    t = days / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629 * days + 0.000387933 * t * t - (t * t * t) / 38710000.0
    return math.radians(gmst_deg % 360.0)


def geodetic_to_ecef(
    latitude_deg: float,
    longitude_deg: float,
    altitude_km: float,
) -> np.ndarray:
    """Convert WGS84 geodetic coordinates to ECEF position (km)."""
    lat = math.radians(float(latitude_deg))
    lon = math.radians(float(longitude_deg))
    alt = float(altitude_km)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = RE_WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (n + alt) * cos_lat * math.cos(lon)
    y = (n + alt) * cos_lat * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_eci(position_ecef: np.ndarray, dt: datetime) -> np.ndarray:
    """Rotate ECEF -> ECI using GMST at epoch."""
    theta = gmst_rad(dt)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    return rot @ np.asarray(position_ecef, dtype=float)


def eci_to_ecef(position_eci: np.ndarray, dt: datetime) -> np.ndarray:
    """Rotate ECI -> ECEF using GMST at epoch."""
    theta = gmst_rad(dt)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    return rot @ np.asarray(position_eci, dtype=float)


def eci_to_geodetic(
    position_eci: np.ndarray,
    dt: datetime,
) -> dict[str, float]:
    """
    Convert an ECI position to the sub-satellite geodetic point.

    Returns {latitude_deg, longitude_deg, altitude_km}.
    Uses a geodetic-latitude approximation (spherical latitude) consistent
    with the demo propagator accuracy.
    """
    ecef = eci_to_ecef(position_eci, dt)
    x, y, z = float(ecef[0]), float(ecef[1]), float(ecef[2])

    longitude_deg = math.degrees(math.atan2(y, x)) % 360.0
    if longitude_deg > 180.0:
        longitude_deg -= 360.0

    r_xy = math.hypot(x, y)
    r = math.sqrt(x * x + y * y + z * z)
    latitude_deg = math.degrees(math.atan2(z, r_xy)) if r_xy > 1e-9 else (90.0 if z > 0 else -90.0)
    altitude_km = r - RE_WGS84_A
    return {
        "latitude_deg": round(float(latitude_deg), 4),
        "longitude_deg": round(float(longitude_deg), 4),
        "altitude_km": round(max(0.0, float(altitude_km)), 3),
    }