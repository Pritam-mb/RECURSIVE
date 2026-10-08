"""
SGP4 orbit propagator.
Parses TLE lines and propagates to arbitrary epoch.
Returns ECI position (km) and velocity (km/s).
SGP4 inherently models J2 oblateness and BSTAR atmospheric drag.
"""

from datetime import datetime, timezone
from threading import RLock
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
import numpy as np
import logging
from app.core.kalman import KalmanFilter, KalmanStateECI
logger = logging.getLogger(__name__)

MU = 398600.4418  # km^3/s^2
RE = 6378.137  # km
ANGLE_EPS = 1e-10


def _state_to_keplerian(pos, vel):
    r = np.array(pos, dtype=float)
    v = np.array(vel, dtype=float)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    if r_mag <= 0.0:
        raise ValueError("Invalid position vector")

    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    if h_mag <= ANGLE_EPS:
        raise ValueError("Invalid angular momentum vector")

    k_hat = np.array([0.0, 0.0, 1.0])
    n = np.cross(k_hat, h)
    n_mag = np.linalg.norm(n)

    e_vec = ((v_mag**2 - MU / r_mag) * r - np.dot(r, v) * v) / MU
    e = float(np.linalg.norm(e_vec))
    energy = (v_mag**2 / 2.0) - (MU / r_mag)
    if abs(energy) <= ANGLE_EPS:
      raise ValueError("Cannot rebuild orbit from a near-parabolic state")

    a = -MU / (2.0 * energy)
    i = np.degrees(np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0)))

    if n_mag > ANGLE_EPS:
        raan = np.degrees(np.arctan2(n[1], n[0])) % 360.0
    else:
        raan = 0.0

    if e > ANGLE_EPS and n_mag > ANGLE_EPS:
        aop = np.degrees(
            np.arctan2(
                np.dot(np.cross(n, e_vec), h) / (n_mag * e * h_mag),
                np.dot(n, e_vec) / (n_mag * e),
            )
        ) % 360.0
        ta = np.degrees(
            np.arctan2(
                np.dot(np.cross(e_vec, r), h) / (e * r_mag * h_mag),
                np.dot(e_vec, r) / (e * r_mag),
            )
        ) % 360.0
    elif e > ANGLE_EPS:
        aop = 0.0
        ta = np.degrees(np.arctan2(e_vec[1], e_vec[0])) % 360.0
    else:
        aop = 0.0
        ta = np.degrees(np.arctan2(r[1], r[0])) % 360.0

    return {
        "a": a,
        "e": e,
        "i": i,
        "raan": raan,
        "aop": aop,
        "ta": ta,
    }


def _rebuild_satrec(pos_km, vel_kms, epoch: datetime, satnum: int) -> Satrec:
    """
    After a delta-V burn, create a new Satrec from the updated
    position and velocity so future propagation follows the new orbit.
    """
    el = _state_to_keplerian(pos_km, vel_kms)
    a = el["a"]
    e = max(el["e"], 1e-7)

    ta_r = np.radians(el["ta"])
    E = 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(ta_r / 2.0),
        np.sqrt(1.0 + e) * np.cos(ta_r / 2.0),
    )
    M = E - e * np.sin(E)

    n_rads = np.sqrt(MU / (a**3))
    n_radm = n_rads * 60.0

    jd, fr = jday(
        epoch.year, epoch.month, epoch.day,
        epoch.hour, epoch.minute, epoch.second + epoch.microsecond / 1e6,
    )

    sat = Satrec()
    sat.sgp4init(
        WGS72,
        "i",
        satnum,
        (jd + fr) - 2433281.5,
        1e-4,
        0.0,
        0.0,
        e,
        np.radians(el["aop"]),
        np.radians(el["i"]),
        M % (2.0 * np.pi),
        n_radm,
        np.radians(el["raan"]),
    )
    return sat


def rsw_to_eci(dv_rsw, pos, vel):
    """
    Convert delta-V from RSW frame to ECI.
    R = radial, S = along-track, W = cross-track.
    """
    r = np.array(pos, dtype=float)
    v = np.array(vel, dtype=float)
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    if h_mag <= ANGLE_EPS:
        raise ValueError("Cannot convert burn frame for a degenerate orbit")

    R_hat = r / np.linalg.norm(r)
    W_hat = h / h_mag
    S_hat = np.cross(W_hat, R_hat)
    rotation = np.column_stack([R_hat, S_hat, W_hat])
    return (rotation @ np.array(dv_rsw, dtype=float)).tolist()


class SatelliteState:
    """Container for a satellite's propagated state."""

    __slots__ = [
        "norad_id", "name",
        "x", "y", "z",
        "vx", "vy", "vz",
        "epoch_utc", "error_code",
    ]

    def __init__(self, norad_id: int, name: str):
        self.norad_id = norad_id
        self.name = name
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.epoch_utc = ""
        self.error_code = 0

    def to_dict(self) -> dict:
        speed_kmh = np.sqrt(self.vx**2 + self.vy**2 + self.vz**2) * 3600
        alt_km = np.sqrt(self.x**2 + self.y**2 + self.z**2) - 6371.0
        return {
            "norad_id": self.norad_id,
            "name": self.name,
            "position": {"x": self.x, "y": self.y, "z": self.z},
            "velocity": {"vx": self.vx, "vy": self.vy, "vz": self.vz},
            "speed_kmh": round(speed_kmh, 2),
            "altitude_km": round(alt_km, 2),
            "epoch_utc": self.epoch_utc,
        }


class SGP4Propagator:
    """Batch SGP4 propagator for multiple satellites with Kalman covariance tracking."""

    def __init__(self):
        self._satellites: dict[int, tuple[Satrec, str]] = {}
        self._maneuvers: dict[int, tuple[float, float, float]] = {}
        self._kalman_states: dict[int, KalmanStateECI] = {}  # Per-satellite Kalman state
        self._kalman = KalmanFilter()
        self._last_epoch: dict[int, datetime] = {}  # Track epoch for covariance propagation
        self._lock = RLock()

    def load_tles(self, tle_list: list[dict]):
        """
        Load TLEs into the propagator.
        Each entry: {name, norad_id, line1, line2}
        """
        with self._lock:
            self._satellites.clear()
            self._maneuvers.clear()
            self._kalman_states.clear()
            self._last_epoch.clear()
            for tle in tle_list:
                try:
                    sat = Satrec.twoline2rv(tle["line1"], tle["line2"], WGS72)
                    self._satellites[tle["norad_id"]] = (sat, tle["name"])
                except Exception as e:
                    logger.warning(
                        f"Failed to parse TLE for {tle.get('name', '?')}: {e}"
                    )

        logger.info(f"Loaded {len(self._satellites)} satellites into propagator")

    def propagate_all(self, dt: datetime = None) -> list[SatelliteState]:
        """
        Propagate all loaded satellites to the given datetime.
        Returns list of SatelliteState objects.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        jd, fr = jday(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second + dt.microsecond / 1e6,
        )

        results = []
        with self._lock:
            satellites = list(self._satellites.items())

        for norad_id, (sat, name) in satellites:
            state = SatelliteState(norad_id, name)
            state.epoch_utc = dt.isoformat()

            e, r, v = sat.sgp4(jd, fr)
            state.error_code = e

            if e == 0:
                state.x, state.y, state.z = r
                state.vx, state.vy, state.vz = v
            else:
                logger.debug(f"SGP4 error {e} for {name} (NORAD {norad_id})")

            results.append(state)

        return results

    def propagate_one(
        self, norad_id: int, dt: datetime = None
    ) -> SatelliteState | None:
        """Propagate a single satellite and update Kalman covariance."""
        with self._lock:
            if norad_id not in self._satellites:
                return None

            sat, name = self._satellites[norad_id]

        if dt is None:
            dt = datetime.now(timezone.utc)

        state = SatelliteState(norad_id, name)
        state.epoch_utc = dt.isoformat()

        jd, fr = jday(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second + dt.microsecond / 1e6,
        )

        e, r, v = sat.sgp4(jd, fr)
        state.error_code = e

        if e == 0:
            state.x, state.y, state.z = r
            state.vx, state.vy, state.vz = v

            # Update Kalman state with SGP4 observation
            with self._lock:
                if norad_id not in self._kalman_states:
                    # Initialize Kalman state on first propagation
                    self._kalman_states[norad_id] = KalmanStateECI(
                        position_km=[r[0], r[1], r[2]],
                        velocity_kms=[v[0], v[1], v[2]],
                        epoch_utc=dt.isoformat(),
                    )
                    self._last_epoch[norad_id] = dt
                else:
                    # Propagate covariance forward in time
                    last_dt = self._last_epoch[norad_id]
                    dt_seconds = (dt - last_dt).total_seconds()
                    kal_state = self._kalman_states[norad_id]
                    kal_state = self._kalman.propagate_covariance(kal_state, dt_seconds)
                    # Update with new observation from SGP4
                    kal_state = self._kalman.update_with_observation(
                        kal_state,
                        np.array([r[0], r[1], r[2]], dtype=float),
                    )
                    self._kalman_states[norad_id] = kal_state
                    self._last_epoch[norad_id] = dt

        return state

    def apply_delta_v(
        self,
        norad_id: int | str,
        dvx: float,
        dvy: float,
        dvz: float,
        frame: str = "RSW",
    ) -> dict:
        """
        Apply an impulsive burn and rebuild the propagator for this satellite.
        After this call, future propagate_one() calls use the updated orbit.
        """
        try:
            norad_int = int(norad_id)
        except (TypeError, ValueError):
            return {"status": "ERROR", "error": f"Invalid NORAD ID: {norad_id}"}

        frame_name = (frame or "RSW").upper()
        if frame_name not in {"RSW", "ECI"}:
            return {"status": "ERROR", "error": f"Unsupported frame: {frame}"}

        epoch = datetime.now(timezone.utc)

        state = self.propagate_one(norad_int, epoch)
        if state is None:
            return {"status": "ERROR", "error": f"Satellite {norad_int} not found"}
        if state.error_code != 0:
            return {
                "status": "ERROR",
                "error": f"Propagation failed for satellite {norad_int}",
            }

        pos = [state.x, state.y, state.z]
        vel = [state.vx, state.vy, state.vz]
        dv_input = [dvx, dvy, dvz]

        try:
            if frame_name == "RSW":
                dv_eci = rsw_to_eci(dv_input, pos, vel)
            else:
                dv_eci = dv_input

            new_vel = [vel[i] + dv_eci[i] / 1000.0 for i in range(3)]
            el = _state_to_keplerian(pos, new_vel)
            perigee = el["a"] * (1.0 - el["e"]) - RE
            if perigee < 100.0:
                return {
                    "status": "ERROR",
                    "error": "Burn rejected: perigee would drop below 100 km (reentry risk)",
                }

            new_satrec = _rebuild_satrec(pos, new_vel, epoch, norad_int)

            with self._lock:
                _, name = self._satellites[norad_int]
                self._satellites[norad_int] = (new_satrec, name)
                self._maneuvers[norad_int] = tuple(dv_input)

            dv_mag = float(np.linalg.norm(dv_eci))
            return {
                "status": "success",
                "norad_id": norad_int,
                "delta_v_ms": round(dv_mag, 3),
                "new_perigee_km": float(round(float(perigee), 1)),
                "new_apogee_km": float(round(float(el["a"] * (1.0 + el["e"]) - RE), 1)),
                "new_inclination_deg": float(round(float(el["i"]), 3)),
                "new_period_min": float(round(float(2.0 * np.pi * np.sqrt(el["a"]**3 / MU) / 60.0), 2)),
                "frame_used": frame_name,
                "epoch_utc": epoch.isoformat(),
            }
        except ValueError as error:
            return {"status": "ERROR", "error": str(error)}

    def get_maneuver(self, norad_id: int) -> tuple | None:
        with self._lock:
            if norad_id in self._maneuvers:
                return self._maneuvers.pop(norad_id)
        return None

    def apply_state_override(
        self,
        norad_id: int | str,
        x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
    ) -> dict:
        """Directly replace a satellite's orbit from an ECI state vector (km, km/s)."""
        try:
            norad_int = int(norad_id)
        except (TypeError, ValueError):
            return {"status": "ERROR", "error": f"Invalid NORAD ID: {norad_id}"}

        with self._lock:
            if norad_int not in self._satellites:
                return {"status": "ERROR", "error": f"Satellite {norad_int} not found"}

        pos = [float(x), float(y), float(z)]
        vel = [float(vx), float(vy), float(vz)]
        try:
            el = _state_to_keplerian(pos, vel)
            perigee = el["a"] * (1.0 - el["e"]) - RE
            if perigee < 100.0:
                return {
                    "status": "ERROR",
                    "error": "Override rejected: perigee would drop below 100 km (reentry risk)",
                }

            epoch = datetime.now(timezone.utc)
            prev_state = self.propagate_one(norad_int, epoch)
            new_satrec = _rebuild_satrec(pos, vel, epoch, norad_int)

            with self._lock:
                _, name = self._satellites[norad_int]
                self._satellites[norad_int] = (new_satrec, name)

            dv_mag = 0.0
            if prev_state is not None and prev_state.error_code == 0:
                dv_mag = float(np.linalg.norm([
                    vel[0] - prev_state.vx,
                    vel[1] - prev_state.vy,
                    vel[2] - prev_state.vz,
                ]))
            return {
                "status": "success",
                "norad_id": norad_int,
                "applied_delta_v_ms": round(dv_mag * 1000.0, 3),
                "new_perigee_km": float(round(float(perigee), 1)),
                "new_apogee_km": float(round(float(el["a"] * (1.0 + el["e"]) - RE), 1)),
                "new_inclination_deg": float(round(float(el["i"]), 3)),
                "epoch_utc": epoch.isoformat(),
            }
        except ValueError as error:
            return {"status": "ERROR", "error": str(error)}

    @property
    def satellite_count(self) -> int:
        with self._lock:
            return len(self._satellites)

    @property
    def norad_ids(self) -> list[int]:
        with self._lock:
            return list(self._satellites.keys())
