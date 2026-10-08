"""
Future-window conjunction screening (all-vs-all) with refined TCA and Foster Pc.

Pipeline (screen()):
  1. Propagate every tracked TLE object over [sim_now, sim_now + window] on a
     coarse grid (default 60 s) in ONE vectorised call to sgp4.api.SatrecArray.
     Objects without a TLE (scenario objects, breakup fragments) are given as
     ECI/TEME state vectors and propagated with two-body + J2 (vectorised RK4).
  2. Apogee/perigee (radial shell) prefilter: an object whose sampled radial
     band [r_min, r_max] (+ threshold + sampling margin) overlaps no other
     object's band can never conjunct and is dropped before the spatial search.
  3. Per grid step, a scipy cKDTree query_pairs with
         radius = threshold + v_rel_max * step / 2
     collects every (pair, step) whose closest approach could fall inside
     +/- step/2 of that sample (v_rel_max = 2 x max orbital speed observed).
  4. A vectorised linear-relative-motion estimate of the closest approach
     around each sample keeps only candidates whose local minimum lies in
     [-step/2, step/2) of that sample and whose linearised miss is within
     threshold + LINEAR_MARGIN_KM.
  5. Each surviving candidate is refined with bounded Brent minimisation on
     [t - step, t + step] using DIRECT sgp4 calls (or J2 RK4 for state-vector
     objects) -> exact TCA, miss distance and relative velocity.
  6. Keep miss <= threshold, dedupe per pair (closest encounter, earliest on
     ties), then compute Foster 2-D Pc on the B-plane with a TLE-age
     covariance model and object-size hard-body radius.

Everything is driven by the SIMULATION clock passed in as sim_time.
"""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from sgp4.api import SatrecArray, jday

try:  # C++ accelerated SatrecArray (sgp4 >= 2.27 ships abi3 wheels for py3.14)
    from sgp4.api import accelerated as SGP4_ACCELERATED
except Exception:  # pragma: no cover
    SGP4_ACCELERATED = False
FORCE_ANCHOR_MODE = os.getenv("SCREEN_FORCE_ANCHOR", "0") == "1"
ANCHOR_S = float(os.getenv("SCREEN_ANCHOR_S", "1200"))

logger = logging.getLogger(__name__)

# ── Configuration (env-overridable) ───────────────────────────────────────────
DEFAULT_WINDOW_HOURS = float(os.getenv("SCREEN_WINDOW_HOURS", "24"))
DEFAULT_STEP_S = float(os.getenv("SCREEN_STEP_S", "60"))
DEFAULT_THRESHOLD_KM = float(os.getenv("SCREEN_THRESHOLD_KM", "25"))

# Linearised-relative-motion error over +/- step/2 is dominated by the gravity
# gradient: |a_rel| <= 3 mu/r^3 * |dr| ~ 2e-3 km/s^2 for |dr| = 500 km in LEO,
# i.e. < 1 km over 30 s. 5 km is a conservative cushion (validated against a
# brute-force fine scan in tests/test_screening_real.py).
LINEAR_MARGIN_KM = 5.0
MAX_REFINES_PER_PAIR = 3

# WGS-72 constants (SGP4/TEME frame consistency)
MU_KM3_S2 = 398600.8
RE_KM = 6378.135
J2 = 0.001082616

# ── Covariance model: TLE-age error growth ────────────────────────────────────
# TLEs carry no covariance, so position uncertainty is modelled from the age of
# the element set at the time of closest approach (|TCA - TLE epoch|).
#   sigma_i(age) = SIGMA0_i + GROWTH_i * age_days        (RTN, km, 1-sigma)
# SIGMA0 follows the typical at-epoch LEO TLE accuracy reported by
#   Flohrer, Krag & Klinkrad (2008), "Assessment and categorization of TLE
#   orbit errors for the US SSN catalogue" (AMOS) - radial ~0.1 km,
#   along-track ~0.5 km, cross-track ~0.15 km for near-circular LEO;
# GROWTH reflects the ~1 km/day along-track degradation of propagated TLEs
#   (Vallado & Cefola (2012) "Two-line element sets - practice and use",
#   IAC-12.C1.6.12; Krag et al. ESA DRAMA/ARES documentation), with radial and
#   cross-track growing roughly an order of magnitude slower.
# These are model parameters, not measurements; every alert carries
# sigma_source="tle_age_model" plus the sigmas and ages actually used.
SIGMA0_RTN_KM = (0.10, 0.50, 0.15)
SIGMA_GROWTH_RTN_KM_PER_DAY = (0.10, 1.00, 0.10)
STALE_TLE_DAYS = 30.0

# ── Hard-body radius model ────────────────────────────────────────────────────
# Per-object radius (m). With SATCAT RCS size class: equivalent-area radius of
# the class upper bound r = sqrt(A/pi): SMALL (<0.1 m^2) -> 0.18 m,
# MEDIUM (<1 m^2) -> 0.56 m, LARGE (>1 m^2) -> type default below.
# Without RCS: conservative type defaults (payload envelope incl. arrays,
# rocket body length/2, debris fragment).
TYPE_DEFAULT_RADIUS_M = {"PAY": 5.0, "R/B": 4.0, "DEB": 0.5, "UNK": 1.0}
RCS_RADIUS_M = {"SMALL": math.sqrt(0.1 / math.pi), "MEDIUM": math.sqrt(1.0 / math.pi)}
LARGE_STRUCTURE_RADIUS_M = {"ISS": 55.0, "ZARYA": 55.0, "TIANHE": 20.0, "CSS": 20.0}

_DEFAULT_PROPAGATOR = None


def set_default_propagator(propagator) -> None:
    """Register the process-wide SGP4Propagator used when none is passed."""
    global _DEFAULT_PROPAGATOR
    _DEFAULT_PROPAGATOR = propagator


# ══════════════════════════════════════════════════════════════════════════════
# Object metadata helpers
# ══════════════════════════════════════════════════════════════════════════════

def _satcat_lookup(norad_id):
    try:
        from app.core.satcat import lookup  # agent C
    except Exception:
        return None
    try:
        return lookup(int(norad_id))
    except Exception:
        return None


def object_type_from_name(name: str) -> str:
    n = (name or "").upper()
    if " DEB" in n or n.startswith("DEB") or "(DEB" in n:
        return "DEB"
    if "R/B" in n or "AKM" in n or "PKM" in n:
        return "R/B"
    if not n:
        return "UNK"
    return "PAY"


def object_meta(norad_id, name: str, agency: str | None = None, object_type: str | None = None) -> dict:
    """Agency, object type, radius (m) and provenance for one object."""
    cat = _satcat_lookup(norad_id) if isinstance(norad_id, (int, np.integer)) else None
    otype = object_type
    if not otype and cat and cat.get("object_type"):
        otype = str(cat["object_type"]).upper()
    if not otype:
        otype = object_type_from_name(name)
    if otype not in TYPE_DEFAULT_RADIUS_M:
        otype = {"PAYLOAD": "PAY", "ROCKET BODY": "R/B", "DEBRIS": "DEB"}.get(otype, "UNK")

    if agency is None:
        try:
            from app.core.agency import infer_agency
            try:
                agency = infer_agency(name, norad_id=norad_id)
            except TypeError:
                agency = infer_agency(name)
        except Exception:
            agency = "UNKNOWN"

    radius_m, radius_src = None, None
    upper = (name or "").upper()
    for key, r in LARGE_STRUCTURE_RADIUS_M.items():
        if upper.startswith(key) or f"({key})" in upper:
            radius_m, radius_src = r, "known_structure"
            break
    if radius_m is None and cat and cat.get("rcs_size"):
        rcs = str(cat["rcs_size"]).upper()
        if rcs in RCS_RADIUS_M:
            radius_m, radius_src = RCS_RADIUS_M[rcs], "satcat_rcs"
        elif rcs == "LARGE":
            radius_m, radius_src = max(1.0, TYPE_DEFAULT_RADIUS_M[otype]), "satcat_rcs"
    if radius_m is None:
        radius_m, radius_src = TYPE_DEFAULT_RADIUS_M[otype], "type_default"
    return {"agency": agency, "object_type": otype, "radius_m": radius_m, "radius_source": radius_src}


# ══════════════════════════════════════════════════════════════════════════════
# Propagation
# ══════════════════════════════════════════════════════════════════════════════

def _j2_accel(r: np.ndarray) -> np.ndarray:
    """Two-body + J2 acceleration, r shape (..., 3) km -> km/s^2."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r2 = x * x + y * y + z * z
    rn = np.sqrt(r2)
    k = -MU_KM3_S2 / (r2 * rn)
    f = 1.5 * J2 * MU_KM3_S2 * RE_KM**2 / (r2 * r2 * rn)
    zz = 5.0 * z * z / r2
    ax = k * x - f * x * (1.0 - zz)
    ay = k * y - f * y * (1.0 - zz)
    az = k * z - f * z * (3.0 - zz)
    return np.stack([ax, ay, az], axis=-1)


def _rk4(r: np.ndarray, v: np.ndarray, dt: float, nsub: int) -> tuple[np.ndarray, np.ndarray]:
    h = dt / nsub
    for _ in range(nsub):
        a1 = _j2_accel(r)
        r2 = r + 0.5 * h * v; v2 = v + 0.5 * h * a1; a2 = _j2_accel(r2)
        r3 = r + 0.5 * h * v2; v3 = v + 0.5 * h * a2; a3 = _j2_accel(r3)
        r4 = r + h * v3; v4 = v + h * a3; a4 = _j2_accel(r4)
        r = r + (h / 6.0) * (v + 2 * v2 + 2 * v3 + v4)
        v = v + (h / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
    return r, v


def propagate_j2(r0, v0, dt_s: float, max_substep_s: float = 10.0):
    """Propagate state(s) by dt_s seconds with two-body + J2 RK4."""
    r = np.asarray(r0, dtype=float)
    v = np.asarray(v0, dtype=float)
    if abs(dt_s) < 1e-9:
        return r.copy(), v.copy()
    nsub = max(1, int(math.ceil(abs(dt_s) / max_substep_s)))
    return _rk4(r, v, dt_s, nsub)


def _j2_grid(r0: np.ndarray, v0: np.ndarray, nt: int, step_s: float):
    """Vectorised J2 RK4 over (M,3) states, sampled every step_s for nt samples."""
    M = r0.shape[0]
    R = np.empty((M, nt, 3)); V = np.empty((M, nt, 3))
    r, v = r0.astype(float).copy(), v0.astype(float).copy()
    R[:, 0], V[:, 0] = r, v
    nsub = max(1, int(math.ceil(step_s / 20.0)))
    for k in range(1, nt):
        r, v = _rk4(r, v, step_s, nsub)
        R[:, k], V[:, k] = r, v
    return R, V


def _jd_fr(dt: datetime) -> tuple[float, float]:
    dt = dt.astimezone(timezone.utc)
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)


class _Obj:
    __slots__ = ("id", "name", "satrec", "r0", "v0", "meta", "epoch_jd", "row")

    def __init__(self, oid, name, satrec=None, r0=None, v0=None, meta=None):
        self.id, self.name, self.satrec, self.r0, self.v0 = oid, name, satrec, r0, v0
        self.meta = meta or {}
        self.epoch_jd = (satrec.jdsatepoch + satrec.jdsatepochF) if satrec is not None else None
        self.row = -1


class _Window:
    """Holds the propagation grid and per-object exact-state evaluators."""

    def __init__(self, objs: list[_Obj], sim_time: datetime, nt: int, step_s: float):
        self.objs, self.sim_time, self.nt, self.step_s = objs, sim_time, nt, step_s
        self.jd0, self.fr0 = _jd_fr(sim_time)
        N = len(objs)
        self.R = np.full((N, nt, 3), np.nan)
        self.V = np.full((N, nt, 3), np.nan)
        tle_rows = [i for i, o in enumerate(objs) if o.satrec is not None]
        ext_rows = [i for i, o in enumerate(objs) if o.satrec is None]
        if tle_rows:
            # Burned objects carry a BurnedSatrec (SGP4 + propagated burn
            # deviation): vectorise the SGP4 part, add the deviation below.
            arr = SatrecArray([getattr(objs[i].satrec, "base_satrec", objs[i].satrec) for i in tle_rows])
            if SGP4_ACCELERATED and not FORCE_ANCHOR_MODE:
                # Full SGP4 on every grid sample (C++ vectorised).
                fr = self.fr0 + np.arange(nt) * step_s / 86400.0
                e, r, v = arr.sgp4(np.full(nt, self.jd0), fr)
                bad = e != 0
                r[bad] = np.nan; v[bad] = np.nan
            else:
                # Pure-python sgp4 (~30 us/call) is too slow for every sample:
                # take exact SGP4 anchors every ANCHOR_S and fill the coarse
                # grid with vectorised J2 RK4 from the latest anchor. The grid
                # only feeds the candidate search (error << LINEAR_MARGIN_KM);
                # TCA refinement always uses direct sgp4 calls.
                per = max(1, int(round(ANCHOR_S / step_s)))
                ks = np.arange(0, nt, per)
                fr = self.fr0 + ks * step_s / 86400.0
                e, ra, va = arr.sgp4(np.full(len(ks), self.jd0), fr)
                r = np.empty((len(tle_rows), nt, 3)); v = np.empty_like(r)
                for a_idx, k0 in enumerate(ks):
                    k1 = min(nt, k0 + per)
                    rr, vv = ra[:, a_idx].copy(), va[:, a_idx].copy()
                    bad = e[:, a_idx] != 0
                    rr[bad] = 7000.0; vv[bad] = 7.5  # placeholder, masked below
                    r[:, k0], v[:, k0] = rr, vv
                    for kk in range(k0 + 1, k1):
                        rr, vv = _rk4(rr, vv, step_s, 1)
                        r[:, kk], v[:, kk] = rr, vv
                    r[bad, k0:k1] = np.nan; v[bad, k0:k1] = np.nan
            burned = [(a, i) for a, i in enumerate(tle_rows) if hasattr(objs[i].satrec, "burn_offsets")]
            if burned:
                jd_g = np.full(nt, self.jd0)
                fr_g = self.fr0 + np.arange(nt) * step_s / 86400.0
                for a, i in burned:
                    dr, dv = objs[i].satrec.burn_offsets(jd_g, fr_g)
                    r[a] += dr; v[a] += dv
            self.R[tle_rows], self.V[tle_rows] = r, v
        if ext_rows:
            r0 = np.array([objs[i].r0 for i in ext_rows], dtype=float)
            v0 = np.array([objs[i].v0 for i in ext_rows], dtype=float)
            R, V = _j2_grid(r0, v0, nt, step_s)
            self.R[ext_rows], self.V[ext_rows] = R, V

    def state(self, i: int, t: float):
        """Exact state of object i at t seconds after sim_time."""
        o = self.objs[i]
        if o.satrec is not None:
            e, r, v = o.satrec.sgp4(self.jd0, self.fr0 + t / 86400.0)
            if e != 0:
                return None, None
            return np.array(r), np.array(v)
        k = int(min(max(math.floor(t / self.step_s), 0), self.nt - 1))
        return propagate_j2(self.R[i, k], self.V[i, k], t - k * self.step_s, max_substep_s=5.0)


# ══════════════════════════════════════════════════════════════════════════════
# Refinement
# ══════════════════════════════════════════════════════════════════════════════

def _refine(win: _Window, i: int, j: int, t_lo: float, t_hi: float, xatol: float = 1e-3):
    def dist(t):
        r1, _ = win.state(i, t)
        r2, _ = win.state(j, t)
        if r1 is None or r2 is None:
            return 1e12
        return float(np.linalg.norm(r1 - r2))

    res = minimize_scalar(dist, bounds=(t_lo, t_hi), method="bounded", options={"xatol": xatol})
    return float(res.x), float(res.fun)


def refine_pair(satrec_or_state_a, satrec_or_state_b, sim_time: datetime, t_guess_s: float,
                half_width_s: float = 60.0) -> dict | None:
    """
    Refine a single encounter near t_guess_s (seconds after sim_time).

    Each argument is either an sgp4 Satrec or a dict {"r_km": [3], "v_kms": [3]}
    giving the state at sim_time. Returns {tca_utc, t_s, miss_km, r1, v1, r2, v2,
    rel_speed_kms} or None.
    """
    objs = []
    for k, a in enumerate((satrec_or_state_a, satrec_or_state_b)):
        if hasattr(a, "sgp4"):
            objs.append(_Obj(k, str(k), satrec=a))
        else:
            objs.append(_Obj(k, str(k), r0=np.asarray(a["r_km"], float), v0=np.asarray(a["v_kms"], float)))
    span = max(0.0, t_guess_s) + half_width_s
    nt = int(math.ceil(span / 60.0)) + 2
    win = _Window(objs, sim_time, nt, 60.0)
    t, miss = _refine(win, 0, 1, max(0.0, t_guess_s - half_width_s), t_guess_s + half_width_s)
    r1, v1 = win.state(0, t)
    r2, v2 = win.state(1, t)
    if r1 is None or r2 is None:
        return None
    return {
        "t_s": t, "tca_utc": (sim_time + timedelta(seconds=t)).isoformat(), "miss_km": miss,
        "r1": r1, "v1": v1, "r2": r2, "v2": v2, "rel_speed_kms": float(np.linalg.norm(v2 - v1)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Collision probability
# ══════════════════════════════════════════════════════════════════════════════

def tle_age_sigmas_km(age_days: float) -> tuple[float, float, float]:
    a = abs(float(age_days))
    return tuple(s0 + g * a for s0, g in zip(SIGMA0_RTN_KM, SIGMA_GROWTH_RTN_KM_PER_DAY))


def rtn_to_eci_cov(r: np.ndarray, v: np.ndarray, sig_rtn) -> np.ndarray:
    r_hat = r / np.linalg.norm(r)
    n_hat = np.cross(r, v); n_hat /= np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    Q = np.column_stack([r_hat, t_hat, n_hat])
    return Q @ np.diag(np.square(sig_rtn)) @ Q.T


# Gauss-Legendre (radial) x trapezoid (angular, spectrally accurate for a
# periodic integrand) quadrature of the bivariate normal over the HBR disk.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)
_N_THETA = 96
_THETA = np.arange(_N_THETA) * (2.0 * np.pi / _N_THETA)


def foster_pc(b_vec_km, cov_2d_km2, hbr_km: float) -> float:
    """
    Foster (1992) 2-D Pc: integral of N(b, C) over the disk |x| <= HBR in the
    B-plane (equivalently the disk centred on the miss vector).
    """
    b = np.asarray(b_vec_km, float)
    C = np.asarray(cov_2d_km2, float)
    det = float(np.linalg.det(C))
    if det <= 0 or hbr_km <= 0:
        return 0.0
    Ci = np.linalg.inv(C)
    rho = 0.5 * hbr_km * (_GL_X + 1.0)              # radial nodes on [0, HBR]
    w_rho = 0.5 * hbr_km * _GL_W
    X = rho[:, None] * np.cos(_THETA)[None, :] - b[0]
    Y = rho[:, None] * np.sin(_THETA)[None, :] - b[1]
    q = Ci[0, 0] * X * X + 2 * Ci[0, 1] * X * Y + Ci[1, 1] * Y * Y
    pdf = np.exp(-0.5 * q) / (2 * np.pi * math.sqrt(det))
    val = float(np.sum((pdf.sum(axis=1) * (2 * np.pi / _N_THETA)) * rho * w_rho))
    return float(min(1.0, max(0.0, val)))


def compute_pc(r1, v1, r2, v2, *, age1_days: float, age2_days: float, hbr_km: float,
               sig1_rtn=None, sig2_rtn=None) -> dict:
    """Foster Pc on the encounter B-plane with TLE-age RTN covariances."""
    r1, v1, r2, v2 = (np.asarray(x, float) for x in (r1, v1, r2, v2))
    sig1 = tuple(sig1_rtn) if sig1_rtn is not None else tle_age_sigmas_km(age1_days)
    sig2 = tuple(sig2_rtn) if sig2_rtn is not None else tle_age_sigmas_km(age2_days)
    C = rtn_to_eci_cov(r1, v1, sig1) + rtn_to_eci_cov(r2, v2, sig2)

    dv = v2 - v1
    dvn = float(np.linalg.norm(dv))
    h_hat = dv / dvn if dvn > 1e-9 else np.array([1.0, 0.0, 0.0])
    t_hat = np.cross(h_hat, r1)
    if np.linalg.norm(t_hat) < 1e-9:
        t_hat = np.cross(h_hat, [0.0, 0.0, 1.0])
    t_hat /= np.linalg.norm(t_hat)
    n_hat = np.cross(t_hat, h_hat)
    B = np.vstack([t_hat, n_hat])
    dr = r2 - r1
    b_vec = B @ dr
    C2 = B @ C @ B.T
    pc = foster_pc(b_vec, C2, hbr_km)

    w, U = np.linalg.eigh(C2)
    a_m = 3000.0 * math.sqrt(max(w[1], 0.0))
    b_m = 3000.0 * math.sqrt(max(w[0], 0.0))
    angle = math.atan2(U[1, 1], U[0, 1])
    return {
        "pc": pc,
        "pc_method": "foster",
        "b_t_km": float(b_vec[0]),
        "b_n_km": float(b_vec[1]),
        "cov_2d_km2": C2.tolist(),
        "covariance_ellipse": {
            "a": round(a_m, 1), "b": round(b_m, 1), "angle": round(angle, 4),
            "sigma_level": 3, "units": "m",
            "sigma_rtn_km_obj1": [round(s, 4) for s in sig1],
            "sigma_rtn_km_obj2": [round(s, 4) for s in sig2],
        },
        # Short-encounter (rectilinear) assumption behind Foster 2-D Pc
        # requires a fast encounter; flag slow, co-orbital geometries.
        "short_encounter_valid": dvn >= 0.1,
    }


def classify_severity(pc: float, miss_km: float) -> str:
    if pc >= 1e-4 or miss_km < 1.0:
        return "CRITICAL"
    if pc >= 1e-6 or miss_km < 5.0:
        return "WARNING"
    return "WATCH"


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def screen(
    states: Iterable[Any],
    sim_time: datetime,
    *,
    window_hours: float | None = None,
    step_s: float | None = None,
    threshold_km: float | None = None,
    extra_objects: list[dict] | None = None,
    propagator=None,
    stats: dict | None = None,
) -> list[dict]:
    """
    Screen all tracked objects (+ extra state-vector objects) for close
    approaches within the next window_hours of SIMULATION time.

    states: SatelliteState-like objects (norad_id, name, error_code, x..vz).
            Objects with a Satrec in the propagator are propagated by SGP4;
            others fall back to two-body+J2 from their state vector.
    Returns alert dicts (contract schema), sorted by Pc then miss distance.
    """
    t_start = time.perf_counter()
    window_hours = DEFAULT_WINDOW_HOURS if window_hours is None else float(window_hours)
    step_s = DEFAULT_STEP_S if step_s is None else float(step_s)
    threshold_km = DEFAULT_THRESHOLD_KM if threshold_km is None else float(threshold_km)
    if sim_time.tzinfo is None:
        sim_time = sim_time.replace(tzinfo=timezone.utc)
    propagator = propagator or _DEFAULT_PROPAGATOR

    satrecs: dict = {}
    burned_ids: set = set()
    if propagator is not None:
        lock = getattr(propagator, "_lock", None)
        if lock is not None:
            with lock:
                satrecs = dict(getattr(propagator, "_satellites", {}))
        else:
            satrecs = dict(getattr(propagator, "_satellites", {}))
        if hasattr(propagator, "burned_ids") and hasattr(propagator, "trajectory"):
            burned_ids = propagator.burned_ids()

    objs: list[_Obj] = []
    seen = set()
    for s in states or []:
        if _get(s, "error_code", 0) not in (0, None):
            continue
        nid = _get(s, "norad_id")
        if nid is None or nid in seen:
            continue
        seen.add(nid)
        name = _get(s, "name", str(nid))
        if nid in satrecs:
            sat = satrecs[nid][0]
            if nid in burned_ids:  # executed burns: SGP4 + propagated deviation
                sat = propagator.trajectory(nid) or sat
            objs.append(_Obj(nid, name, satrec=sat))
        else:
            r0 = np.array([_get(s, "x"), _get(s, "y"), _get(s, "z")], float)
            v0 = np.array([_get(s, "vx"), _get(s, "vy"), _get(s, "vz")], float)
            if np.all(np.isfinite(r0)) and np.linalg.norm(r0) > RE_KM:
                objs.append(_Obj(nid, name, r0=r0, v0=v0))
    for e in extra_objects or []:
        nid = e.get("id")
        if nid is None or nid in seen:
            continue
        seen.add(nid)
        objs.append(_Obj(nid, e.get("name", str(nid)), r0=np.asarray(e["r_km"], float),
                         v0=np.asarray(e["v_kms"], float),
                         meta={k: e.get(k) for k in ("agency", "object_type") if e.get(k)}))

    st = {"objects": len(objs), "window_hours": window_hours, "step_s": step_s,
          "threshold_km": threshold_km}
    if len(objs) < 2:
        if stats is not None:
            stats.update(st)
        return []

    nt = int(round(window_hours * 3600.0 / step_s)) + 1
    win = _Window(objs, sim_time, nt, step_s)
    t_prop = time.perf_counter()

    # ── Radial-shell (apogee/perigee) prefilter ───────────────────────────────
    R, V = win.R, win.V
    rad = np.linalg.norm(R, axis=2)
    valid_obj = np.isfinite(rad).any(axis=1)
    rdot = np.abs(np.nansum(R * V, axis=2) / np.where(rad > 0, rad, 1.0))
    slack = threshold_km + np.nan_to_num(np.nanmax(rdot, axis=1)) * step_s / 2.0
    rmin = np.where(valid_obj, np.nanmin(np.where(np.isfinite(rad), rad, np.inf), axis=1), np.inf) - slack
    rmax = np.where(valid_obj, np.nanmax(np.where(np.isfinite(rad), rad, -np.inf), axis=1), -np.inf) + slack
    order = np.argsort(rmin)
    keep = np.zeros(len(objs), bool)
    running_max, running_idx = -np.inf, -1
    for idx in order:            # interval-overlap sweep
        if not valid_obj[idx]:
            continue
        if rmin[idx] <= running_max:
            keep[idx] = True
            keep[running_idx] = True
        if rmax[idx] > running_max:
            running_max, running_idx = rmax[idx], idx
    # The sweep above marks pairs overlapping the running maximum; do an exact
    # check for objects not yet kept (overlap with ANY other interval).
    rest = np.where(valid_obj & ~keep)[0]
    if rest.size:
        vi = np.where(valid_obj)[0]
        for idx in rest:
            others = vi[vi != idx]
            if np.any((rmin[others] <= rmax[idx]) & (rmax[others] >= rmin[idx])):
                keep[idx] = True
    active = np.where(keep)[0]
    st["prefilter_kept"] = int(active.size)

    # ── Per-step KD-tree candidate search ─────────────────────────────────────
    speed = np.linalg.norm(V[active], axis=2)
    v_rel_max = 2.0 * float(np.nanmax(speed)) if active.size else 0.0
    radius = threshold_km + v_rel_max * step_s / 2.0
    Ra = R[active]
    far = (np.arange(active.size, dtype=float)[:, None] + 1.0) * 1e7
    cand_i, cand_j, cand_k = [], [], []
    for k in range(nt):
        P = Ra[:, k, :]
        bad = ~np.isfinite(P).all(axis=1)
        if bad.any():
            P = P.copy(); P[bad] = far[bad] * np.array([1.0, 0.0, 0.0])
        pairs = cKDTree(P).query_pairs(radius, output_type="ndarray")
        if pairs.size:
            cand_i.append(active[pairs[:, 0]]); cand_j.append(active[pairs[:, 1]])
            cand_k.append(np.full(len(pairs), k))
    t_kd = time.perf_counter()
    if not cand_i:
        st.update({"candidates_kdtree": 0, "candidates_linear": 0, "refined": 0,
                   "t_propagate_s": t_prop - t_start, "t_kdtree_s": t_kd - t_prop,
                   "t_total_s": time.perf_counter() - t_start})
        if stats is not None:
            stats.update(st)
        return []
    I = np.concatenate(cand_i); J = np.concatenate(cand_j); K = np.concatenate(cand_k)
    st["candidates_kdtree"] = int(I.size)

    # ── Linear relative-motion filter ─────────────────────────────────────────
    dr = R[J, K] - R[I, K]
    dv = V[J, K] - V[I, K]
    dv2 = np.einsum("ij,ij->i", dv, dv)
    tstar = np.where(dv2 > 1e-12, -np.einsum("ij,ij->i", dr, dv) / np.maximum(dv2, 1e-12), 0.0)
    half = step_s / 2.0
    in_bin = (tstar >= -half) & (tstar < half)
    in_bin |= (K == nt - 1) & (tstar >= half)  # minimum beyond last sample: checked below
    in_bin &= ~((K == 0) & (tstar < 0))         # encounter already in the past
    slow = dv2 < (0.05) ** 2                     # co-orbital: no well-defined local minimum
    tclamp = np.clip(tstar, -half, half)
    d_est = np.linalg.norm(dr + dv * tclamp[:, None], axis=1)
    sel = (in_bin | slow) & (d_est <= threshold_km + LINEAR_MARGIN_KM)
    I, J, K, d_est, tstar = I[sel], J[sel], K[sel], d_est[sel], tclamp[sel]
    st["candidates_linear"] = int(I.size)

    # Cap refinements per pair (slow co-orbital pairs match at many steps)
    pair_key = I.astype(np.int64) * 1_000_003 + J
    o = np.lexsort((d_est, pair_key))
    I, J, K, d_est, tstar, pair_key = I[o], J[o], K[o], d_est[o], tstar[o], pair_key[o]
    rank = np.zeros(len(pair_key), int)
    if len(pair_key):
        new = np.r_[True, pair_key[1:] != pair_key[:-1]]
        grp = np.cumsum(new) - 1
        first = np.flatnonzero(new)
        rank = np.arange(len(pair_key)) - first[grp]
    m = rank < MAX_REFINES_PER_PAIR
    I, J, K = I[m], J[m], K[m]

    # ── Exact refinement ──────────────────────────────────────────────────────
    t_end = (nt - 1) * step_s
    best: dict[tuple[int, int], dict] = {}
    n_ref = 0
    for i, j, k in zip(I.tolist(), J.tolist(), K.tolist()):
        tk = k * step_s
        lo, hi = max(0.0, tk - step_s), min(t_end, tk + step_s)
        t, miss = _refine(win, i, j, lo, hi)
        n_ref += 1
        if miss > threshold_km:
            continue
        if t >= t_end - 1e-2 or (t <= 1e-2 and k == 0 and _closing(win, i, j, 0.0) is False):
            continue  # minimum lies outside the window
        key = (i, j)
        prev = best.get(key)
        if prev is None or miss < prev["miss"] - 1e-6 or (abs(miss - prev["miss"]) <= 1e-6 and t < prev["t"]):
            cnt = (prev["count"] + 1) if prev else 1
            best[key] = {"t": t, "miss": miss, "count": cnt}
        else:
            prev["count"] += 1
    st["refined"] = n_ref
    t_ref = time.perf_counter()

    # ── Build alerts with Pc ──────────────────────────────────────────────────
    alerts = []
    meta_cache: dict[int, dict] = {}

    def meta(idx):
        if idx not in meta_cache:
            ob = objs[idx]
            m0 = ob.meta or {}
            meta_cache[idx] = object_meta(ob.id, ob.name, m0.get("agency"), m0.get("object_type"))
        return meta_cache[idx]

    for (i, j), enc in best.items():
        a, b = objs[i], objs[j]
        if isinstance(a.id, (int, np.integer)) and isinstance(b.id, (int, np.integer)) and b.id < a.id:
            i, j, a, b = j, i, b, a
        t = enc["t"]
        r1, v1 = win.state(i, t)
        r2, v2 = win.state(j, t)
        if r1 is None or r2 is None:
            continue
        tca_dt = sim_time + timedelta(seconds=t)
        jd_tca = win.jd0 + win.fr0 + t / 86400.0
        age1 = (jd_tca - a.epoch_jd) if a.epoch_jd is not None else t / 86400.0
        age2 = (jd_tca - b.epoch_jd) if b.epoch_jd is not None else t / 86400.0
        ma, mb = meta(i), meta(j)
        hbr_km = (ma["radius_m"] + mb["radius_m"]) / 1000.0
        pcd = compute_pc(r1, v1, r2, v2, age1_days=age1, age2_days=age2, hbr_km=hbr_km)
        miss = float(np.linalg.norm(r2 - r1))
        vrel = float(np.linalg.norm(v2 - v1))
        pc = pcd["pc"]
        tca_hours = t / 3600.0
        age_h = max(abs(age1), abs(age2)) * 24.0
        try:
            from app.core.conjunction import compute_cpi_score
            cpi = compute_cpi_score(pc, miss, tca_hours=tca_hours, relative_velocity_kms=vrel,
                                    tle_age_hours=age_h)
        except Exception:
            cpi = 0.0
        sev = classify_severity(pc, miss)
        alerts.append({
            "id": f"{a.id}-{b.id}",
            "source": "screening",
            "sat1": {"id": a.id, "name": a.name, "agency": ma["agency"], "object_type": ma["object_type"],
                     "tle_age_hours": round(abs(age1) * 24.0, 3), "radius_m": ma["radius_m"]},
            "sat2": {"id": b.id, "name": b.name, "agency": mb["agency"], "object_type": mb["object_type"],
                     "tle_age_hours": round(abs(age2) * 24.0, 3), "radius_m": mb["radius_m"]},
            "altitude_km": round(float(np.linalg.norm(0.5 * (r1 + r2))) - RE_KM, 3),
            "radial_miss_km": round(float(np.dot(r2 - r1, r1 / np.linalg.norm(r1))), 5),
            "tca_utc": tca_dt.isoformat(),
            "tca_hours": round(tca_hours, 4),
            "tca_minutes": round(t / 60.0, 2),
            "miss_distance_km": round(miss, 4),
            "relative_speed_kms": round(vrel, 4),
            "relative_speed_kmh": round(vrel * 3600.0, 2),
            "probability_of_collision": pc,
            "p_collision": pc,
            "pc_method": pcd["pc_method"],
            "short_encounter_valid": pcd["short_encounter_valid"],
            "hbr_km": round(hbr_km, 5),
            "hbr_source": [ma["radius_source"], mb["radius_source"]],
            "covariance_ellipse": pcd["covariance_ellipse"],
            "b_t_km": round(pcd["b_t_km"], 5),
            "b_n_km": round(pcd["b_n_km"], 5),
            "bt_km": round(pcd["b_t_km"], 5),
            "bn_km": round(pcd["b_n_km"], 5),
            "sigma_source": "tle_age_model" if (a.epoch_jd is not None and b.epoch_jd is not None) else "state_age_model",
            "tle_age_days": [round(age1, 3), round(age2, 3)],
            "stale_tle": bool(max(abs(age1), abs(age2)) > STALE_TLE_DAYS),
            "severity": sev,
            "severity_color": {"CRITICAL": "red", "WARNING": "yellow", "WATCH": "green"}[sev],
            "cpi_score": round(cpi, 3),
            "encounters_in_window": enc["count"],
            "tca_position_km": [round(float(x), 3) for x in (0.5 * (r1 + r2))],
            "screening": {"window_hours": window_hours, "step_s": step_s, "threshold_km": threshold_km,
                          "sim_epoch_utc": sim_time.isoformat()},
        })

    alerts.sort(key=lambda x: (-x["p_collision"], x["miss_distance_km"]))
    st.update({
        "alerts": len(alerts),
        "t_propagate_s": round(t_prop - t_start, 3),
        "t_kdtree_s": round(t_kd - t_prop, 3),
        "t_refine_s": round(t_ref - t_kd, 3),
        "t_pc_s": round(time.perf_counter() - t_ref, 3),
        "t_total_s": round(time.perf_counter() - t_start, 3),
        "kd_radius_km": round(radius, 1),
    })
    if stats is not None:
        stats.update(st)
    logger.info("screening: %s", st)
    return alerts


def _closing(win: _Window, i: int, j: int, t: float):
    r1, v1 = win.state(i, t)
    r2, v2 = win.state(j, t)
    if r1 is None or r2 is None:
        return None
    return float(np.dot(r2 - r1, v2 - v1)) < 0.0
