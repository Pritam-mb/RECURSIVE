"""
SGP4 orbit propagator.
Parses TLE lines and propagates to arbitrary epoch.
Returns ECI position (km) and velocity (km/s).
SGP4 inherently models J2 oblateness and BSTAR atmospheric drag.
"""

from datetime import datetime, timezone
from threading import Lock, RLock
import math
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
import numpy as np
import logging
from app.core.kalman import KalmanFilter, KalmanStateECI
from app.core import sim_clock
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


# ══════════════════════════════════════════════════════════════════════════════
# Impulsive burns as a propagated deviation layered on SGP4
# ══════════════════════════════════════════════════════════════════════════════
#
#   r_burned(t) = r_SGP4(t) + sum_b [ r_J2(t; x_b + dv_b) - r_J2(t; x_b) ]   (t >= t_b)
#
# x_b is the object's (already burned) trajectory state at burn epoch t_b. Both
# the burned and the un-burned state are integrated with the SAME two-body + J2
# RK4 dynamics (app.core.screening._j2_accel), so the common-mode mismatch
# between SGP4 mean elements and an osculating J2 propagation cancels: a zero
# burn is exactly zero, and the burn effect (along-track drift, period change,
# J2 nodal effects) is propagated with real orbital dynamics. This is the same
# trajectory model the manoeuvre planner uses for its prediction
# (app/services/maneuver_planner.py), so a planned burn, once executed,
# reproduces the predicted encounter geometry. We deliberately do NOT re-fit a
# TLE from the osculating post-burn state: SGP4 elements are mean elements and
# such a re-fit moves the object by ~10 km at once and ~100+ km within hours.

BURN_CHECKPOINT_S = 120.0   # cached deviation checkpoints along the arc
BURN_RK4_STEP_S = 20.0      # RK4 step (difference of two arcs: sub-metre error)


def _jday_dt(dt: datetime) -> tuple[float, float]:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                dt.second + dt.microsecond / 1e6)


def _rk4_rows(r: np.ndarray, v: np.ndarray, h, nsub: int, accel):
    """RK4 on stacked states r, v (..., 3) with a (per-row broadcastable) step h."""
    for _ in range(nsub):
        a1 = accel(r)
        r2 = r + 0.5 * h * v; v2 = v + 0.5 * h * a1; a2 = accel(r2)
        r3 = r + 0.5 * h * v2; v3 = v + 0.5 * h * a2; a3 = accel(r3)
        r4 = r + h * v3; v4 = v + h * a3; a4 = accel(r4)
        r = r + (h / 6.0) * (v + 2 * v2 + 2 * v3 + v4)
        v = v + (h / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
    return r, v


def _j2_dynamics():
    from app.core.screening import _j2_accel  # shared two-body + J2 dynamics
    return _j2_accel


class BurnDeviation:
    """One impulsive burn: deviation of the burned arc from the un-burned arc."""

    def __init__(self, epoch: datetime, r0, v0, dv_eci_kms, dv_input, frame: str):
        self.epoch = epoch
        self.jd, self.fr = _jday_dt(epoch)
        self.r0 = np.asarray(r0, float)
        self.v0 = np.asarray(v0, float)
        self.dv_eci_kms = np.asarray(dv_eci_kms, float)
        self.dv_input = tuple(float(x) for x in dv_input)
        self.frame = frame
        # checkpoint k holds the stacked [nominal, burned] states at k * BURN_CHECKPOINT_S
        self._ck_r = [np.stack([self.r0, self.r0])]
        self._ck_v = [np.stack([self.v0, self.v0 + self.dv_eci_kms])]
        self._ck_lock = Lock()

    def _extend(self, k_max: int) -> None:
        with self._ck_lock:
            if len(self._ck_r) > k_max:
                return
            accel = _j2_dynamics()
            nsub = max(1, int(math.ceil(BURN_CHECKPOINT_S / BURN_RK4_STEP_S)))
            h = BURN_CHECKPOINT_S / nsub
            r, v = self._ck_r[-1], self._ck_v[-1]
            while len(self._ck_r) <= k_max:
                r, v = _rk4_rows(r, v, h, nsub, accel)
                self._ck_r.append(r)
                self._ck_v.append(v)

    def to_dict(self) -> dict:
        return {
            "epoch_utc": self.epoch.isoformat(),
            "frame": self.frame,
            "delta_v_input_ms": list(self.dv_input),
            "delta_v_eci_ms": [float(x) * 1000.0 for x in self.dv_eci_kms],
        }

    def offsets(self, jd, fr) -> tuple[np.ndarray, np.ndarray]:
        """Position (km) / velocity (km/s) deviation at Julian dates jd + fr (zero before the burn)."""
        jd, fr = np.broadcast_arrays(np.asarray(jd, float), np.asarray(fr, float))
        shape = jd.shape
        t = ((jd - self.jd) + (fr - self.fr)).ravel() * 86400.0
        dr = np.zeros((t.size, 3))
        dv = np.zeros((t.size, 3))
        m = np.isfinite(t) & (t >= -1e-6)
        if m.any():
            tm = np.maximum(t[m], 0.0)
            k = np.floor(tm / BURN_CHECKPOINT_S).astype(int)
            self._extend(int(k.max()))
            R = np.stack([self._ck_r[i] for i in k])   # (n, 2, 3)
            V = np.stack([self._ck_v[i] for i in k])
            tau = tm - k * BURN_CHECKPOINT_S
            if np.any(tau > 0):
                nsub = max(1, int(math.ceil(BURN_CHECKPOINT_S / BURN_RK4_STEP_S)))
                R, V = _rk4_rows(R, V, (tau / nsub)[:, None, None], nsub, _j2_dynamics())
            dr[m] = R[:, 1] - R[:, 0]
            dv[m] = V[:, 1] - V[:, 0]
        return dr.reshape(shape + (3,)), dv.reshape(shape + (3,))


class BurnedSatrec:
    """
    Satrec-compatible trajectory (sgp4 / sgp4_array): SGP4 of the original TLE
    plus the propagated deviation of every executed burn. Other attributes
    (jdsatepoch, satnum, ...) fall through to the underlying Satrec, exposed as
    .base_satrec for vectorised SatrecArray callers, which must then add
    burn_offsets(jd, fr) themselves (see add_burn_offsets / screening._Window).
    """

    def __init__(self, base: Satrec, burns: list[BurnDeviation]):
        self.base_satrec = base
        self.burns = list(burns)

    def __getattr__(self, item):
        if item in ("base_satrec", "burns"):
            raise AttributeError(item)
        return getattr(self.base_satrec, item)

    def burn_offsets(self, jd, fr) -> tuple[np.ndarray, np.ndarray]:
        shape = np.broadcast(np.asarray(jd, float), np.asarray(fr, float)).shape
        dr = np.zeros(shape + (3,)); dv = np.zeros(shape + (3,))
        for b in self.burns:
            r_b, v_b = b.offsets(jd, fr)
            dr += r_b; dv += v_b
        return dr, dv

    def sgp4(self, jd, fr):
        e, r, v = self.base_satrec.sgp4(jd, fr)
        if e != 0:
            return e, r, v
        dr, dv = self.burn_offsets(jd, fr)
        return (e, tuple((np.asarray(r) + dr.reshape(3)).tolist()),
                tuple((np.asarray(v) + dv.reshape(3)).tolist()))

    def sgp4_array(self, jd, fr):
        e, r, v = self.base_satrec.sgp4_array(jd, fr)
        dr, dv = self.burn_offsets(jd, fr)
        return e, r + dr.reshape(r.shape), v + dv.reshape(v.shape)


def add_burn_offsets(propagator, ids, jd, fr, r, v) -> None:
    """
    In place: add executed-burn deviations to SatrecArray output r, v of shape
    (len(ids), nt, 3) evaluated at jd, fr (nt,). No-op for objects without burns.
    """
    burned = getattr(propagator, "burned_ids", None)
    if burned is None:
        return
    ids_b = burned()
    if not ids_b:
        return
    for row, nid in enumerate(ids):
        if nid in ids_b:
            traj = propagator.trajectory(nid)
            if isinstance(traj, BurnedSatrec):
                dr, dv = traj.burn_offsets(jd, fr)
                r[row] += dr.reshape(r[row].shape)
                v[row] += dv.reshape(v[row].shape)


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
        # Executed burns: norad_id -> (base Satrec the burns are layered on, [BurnDeviation]).
        # If the _satellites entry is replaced (scenario load, state override),
        # the base no longer matches and the burns are ignored.
        self._burns: dict[int, tuple[Satrec, list[BurnDeviation]]] = {}
        self._lock = RLock()

    # ── Burned-trajectory access ───────────────────────────────────────────
    def trajectory(self, norad_id: int):
        """Satrec-compatible trajectory incl. executed burns (raw Satrec if none), or None."""
        with self._lock:
            entry = self._satellites.get(norad_id)
            if entry is None:
                return None
            burns = self._burns.get(norad_id)
            if burns is not None and burns[0] is entry[0] and burns[1]:
                return BurnedSatrec(entry[0], burns[1])
            return entry[0]

    def burned_ids(self) -> set:
        """Ids whose trajectory currently carries at least one executed burn."""
        with self._lock:
            return {nid for nid, (base, burns) in self._burns.items()
                    if burns and nid in self._satellites and self._satellites[nid][0] is base}

    def burn_history(self, norad_id: int) -> list[dict]:
        with self._lock:
            entry = self._satellites.get(norad_id)
            burns = self._burns.get(norad_id)
            if entry is None or burns is None or burns[0] is not entry[0]:
                return []
            return [b.to_dict() for b in burns[1]]

    def clone(self) -> "SGP4Propagator":
        """Independent copy for what-if evaluation (burns are copied, not shared lists)."""
        other = SGP4Propagator()
        with self._lock:
            other._satellites = dict(self._satellites)
            other._maneuvers = dict(self._maneuvers)
            other._burns = {nid: (base, list(burns)) for nid, (base, burns) in self._burns.items()}
        return other

    def load_tles(self, tle_list: list[dict]):
        """
        Load TLEs into the propagator.
        Each entry: {name, norad_id, line1, line2}
        """
        with self._lock:
            self._satellites.clear()
            self._maneuvers.clear()
            self._burns.clear()
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
            dt = sim_clock.simulation_now()

        jd, fr = jday(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute, dt.second + dt.microsecond / 1e6,
        )

        results = []
        with self._lock:
            satellites = list(self._satellites.items())
            burned = self.burned_ids()

        for norad_id, (sat, name) in satellites:
            if norad_id in burned:
                sat = self.trajectory(norad_id) or sat
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

            _, name = self._satellites[norad_id]
            sat = self.trajectory(norad_id)

        if dt is None:
            dt = sim_clock.simulation_now()

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
        epoch: datetime | None = None,
    ) -> dict:
        """
        Apply an impulsive burn (dv in m/s) at `epoch` (default: simulation now).

        The burn is stored as a deviation layered on the SGP4 trajectory (see
        BurnDeviation): every later propagate_one / propagate_all / screening
        call evaluates SGP4(t) + [J2(t; x+dv) - J2(t; x)], the same model the
        manoeuvre planner predicts with. The TLE itself is not re-fitted.
        """
        try:
            norad_int = int(norad_id)
        except (TypeError, ValueError):
            return {"status": "ERROR", "error": f"Invalid NORAD ID: {norad_id}"}

        frame_name = (frame or "RSW").upper()
        if frame_name not in {"RSW", "ECI"}:
            return {"status": "ERROR", "error": f"Unsupported frame: {frame}"}

        if epoch is None:
            epoch = sim_clock.simulation_now()
        elif epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)

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

            burn = BurnDeviation(epoch, pos, vel, np.asarray(dv_eci, float) / 1000.0,
                                 dv_input, frame_name)

            with self._lock:
                base = self._satellites[norad_int][0]
                prev = self._burns.get(norad_int)
                burns = list(prev[1]) if prev is not None and prev[0] is base else []
                burns.append(burn)
                self._burns[norad_int] = (base, burns)
                self._maneuvers[norad_int] = tuple(dv_input)

            dv_mag = float(np.linalg.norm(dv_eci))
            return {
                "status": "success",
                "burn_count": len(burns),
                "propagation_model": "sgp4_reference+j2_rk4_deviation",
                "pre_burn_state_eci": {"r_km": [float(x) for x in pos],
                                       "v_kms": [float(x) for x in vel]},
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

            epoch = sim_clock.simulation_now()
            prev_state = self.propagate_one(norad_int, epoch)
            new_satrec = _rebuild_satrec(pos, vel, epoch, norad_int)

            with self._lock:
                _, name = self._satellites[norad_int]
                self._satellites[norad_int] = (new_satrec, name)
                self._burns.pop(norad_int, None)   # explicit new orbit supersedes burn history

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
