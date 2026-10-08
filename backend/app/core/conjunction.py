"""
Conjunction assessment engine.
Computes pairwise miss distances and identifies close approaches.
Screening itself lives in app/core/screening.py (future-window, refined TCA,
Foster Pc with TLE-age covariance); this module keeps the CPI score, the
legacy distance-band classifier and find_tca (single-pair TCA search).

CPI (Conjunction Priority Index, 0-10) is a heuristic TRIAGE score for
ordering the operator queue. It is a hand-weighted blend of Foster Pc, miss
distance, time-to-TCA, closing speed and TLE age; it is NOT a probability and
not fitted to data. The physics result is always probability_of_collision.

References:
  Foster & Estes (1992) NASA/JSC-25898 — B-plane integration method
  NASA CARA threshold guide: Pc > 1e-4 triggers maneuver consideration
"""

import math
import numpy as np
import logging
from datetime import datetime, timedelta, timezone
from app.core.sgp4_propagator import SGP4Propagator, SatelliteState

logger = logging.getLogger(__name__)

# Legacy distance bands. Used ONLY by find_tca()'s ConjunctionEvent.severity
# label and the pre-flight "trajectory_clear" policy gate (sim_engine). Alert
# severity in the live pipeline is screening.classify_severity (Pc + miss).
ALERT_DISTANCE_KM = 50.0       # Red alert
WARNING_DISTANCE_KM = 200.0    # Yellow warning
WATCH_DISTANCE_KM = 500.0      # Green watch


# ── ConjunctionEvent ──────────────────────────────────────────────────────────

class ConjunctionEvent:
    """Represents a potential collision between two objects."""

    __slots__ = [
        "sat1_id", "sat1_name",
        "sat2_id", "sat2_name",
        "miss_distance_km",
        "relative_speed_kmh",
        "tca_utc",
        "tca_hours",
        "severity",
        "probability_of_collision",
        "cpi_score",
        "covariance_ellipse",
    ]

    def __init__(self):
        self.sat1_id = 0
        self.sat1_name = ""
        self.sat2_id = 0
        self.sat2_name = ""
        self.miss_distance_km = 0.0
        self.relative_speed_kmh = 0.0
        self.tca_utc = ""
        self.tca_hours = 0.0
        self.severity = "green"
        self.probability_of_collision = 0.0
        self.cpi_score = 0.0
        self.covariance_ellipse = None  # populated by Foster integration

    def to_dict(self) -> dict:
        return {
            "id": f"{self.sat1_id}-{self.sat2_id}",
            "sat1": {"id": self.sat1_id, "name": self.sat1_name},
            "sat2": {"id": self.sat2_id, "name": self.sat2_name},
            "miss_distance_km": round(self.miss_distance_km, 3),
            "relative_speed_kmh": round(self.relative_speed_kmh, 2),
            "tca_utc": self.tca_utc,
            "tca_hours": round(self.tca_hours, 3),
            "tca_minutes": round(self.tca_hours * 60.0, 2),
            "severity": self.severity,
            "probability_of_collision": round(self.probability_of_collision, 8),
            "p_collision": round(self.probability_of_collision, 8),
            "cpi_score": round(self.cpi_score, 2),
            "covariance_ellipse": self.covariance_ellipse,
        }


# ── Geometry helpers ──────────────────────────────────────────────────────────

def classify_severity(distance_km: float) -> str:
    if distance_km <= ALERT_DISTANCE_KM:
        return "red"
    elif distance_km <= WARNING_DISTANCE_KM:
        return "yellow"
    elif distance_km <= WATCH_DISTANCE_KM:
        return "green"
    return "none"


def compute_miss_distance(s1: SatelliteState, s2: SatelliteState) -> float:
    """Euclidean distance between two satellite positions in km."""
    dx = s1.x - s2.x
    dy = s1.y - s2.y
    dz = s1.z - s2.z
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def compute_relative_speed(s1: SatelliteState, s2: SatelliteState) -> float:
    """Relative speed in km/h."""
    dvx = s1.vx - s2.vx
    dvy = s1.vy - s2.vy
    dvz = s1.vz - s2.vz
    return np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz) * 3600


# ── CPI score calibrated against Foster Pc ───────────────────────────────────

def compute_cpi_score(
    probability_of_collision: float,
    miss_distance_km: float,
    tca_hours: float = 12.0,
    relative_velocity_kms: float = 7.5,
    tle_age_hours: float = 6.0,
) -> float:
    """
    Conjunction Priority Index (0–10): hand-weighted triage heuristic (design
    choice, not fitted). Monotone in each input; NOT a probability.

    Weights (sum = 1.0):
      0.40 — miss distance component   (strongest predictor of Pc)
      0.30 — Foster Pc component       (log-mapped to 0–10 scale)
      0.15 — TCA urgency component     (time to closest approach)
      0.10 — closing speed component   (relative velocity)
      0.05 — TLE data quality          (age of orbital elements)

    Pc component mapping (log-linear): Pc 1e-6 -> 0, 1e-4 -> 5 (the common
    CARA manoeuvre-consideration threshold), 1e-2 -> 10.
    """
    try:
        # ── Miss distance component (0–10) ─────────────────────────────────
        # Closer approach → higher component
        if miss_distance_km <= 0.1:
            miss_score = 10.0
        elif miss_distance_km <= 1.0:
            miss_score = 9.0 - (miss_distance_km - 0.1) * (2.0 / 0.9)
        elif miss_distance_km <= 10.0:
            miss_score = 7.0 - (miss_distance_km - 1.0) * (3.0 / 9.0)
        elif miss_distance_km <= 50.0:
            miss_score = 4.0 - (miss_distance_km - 10.0) * (2.0 / 40.0)
        elif miss_distance_km <= 200.0:
            miss_score = 2.0 - (miss_distance_km - 50.0) * (1.5 / 150.0)
        else:
            miss_score = max(0.0, 0.5 - (miss_distance_km - 200.0) * 0.001)

        miss_score = float(np.clip(miss_score, 0.0, 10.0))

        # ── Foster Pc component (0–10) — log-mapped ────────────────────────
        # Log-linear mapping of Pc onto 0-10:
        #   Pc = 1e-6 → score 0.0
        #   Pc = 1e-4 → score 5.0   (NASA CARA watch threshold)
        #   Pc = 1e-3 → score 7.5
        #   Pc = 1e-2 → score 10.0
        p_c = float(probability_of_collision)
        if p_c <= 0.0:
            pc_score = 0.0
        elif p_c >= 1e-2:
            pc_score = 10.0
        else:
            # Linear in log10(Pc) space: maps [-6, -2] → [0, 10]
            pc_score = max(0.0, (math.log10(p_c) + 6.0) / 0.4)
            pc_score = min(10.0, pc_score)

        # ── TCA urgency component (0–10) ───────────────────────────────────
        # Events closer in time are higher priority
        tca_hours = max(0.0, float(tca_hours))
        if tca_hours <= 1.0:
            tca_score = 10.0
        elif tca_hours <= 6.0:
            tca_score = 10.0 - (tca_hours - 1.0) * (4.0 / 5.0)
        elif tca_hours <= 24.0:
            tca_score = 6.0 - (tca_hours - 6.0) * (4.0 / 18.0)
        else:
            tca_score = max(0.0, 2.0 - (tca_hours - 24.0) * 0.05)

        tca_score = float(np.clip(tca_score, 0.0, 10.0))

        # ── Relative velocity component (0–10) ─────────────────────────────
        # Higher closing speed → harder to mitigate → higher priority
        vel_kms = max(0.0, float(relative_velocity_kms))
        vel_score = float(np.clip(vel_kms / 15.0 * 10.0, 0.0, 10.0))

        # ── TLE age / data quality component (0–10) ───────────────────────
        # Older TLE → larger uncertainty → slightly higher priority. Scaled
        # over 7 days (was 48 h): real catalogue TLEs are routinely 1–3 days
        # old, so a 48 h scale saturated this term for almost every pair now
        # that the actual |TCA − TLE epoch| is passed instead of a 0.5 h stub.
        age = max(0.0, float(tle_age_hours))
        age_score = float(np.clip(age / 168.0 * 10.0, 0.0, 10.0))

        # ── Weighted CPI (weights sum to 1.0) ──────────────────────────────
        cpi = (
            0.40 * miss_score
            + 0.30 * pc_score
            + 0.15 * tca_score
            + 0.10 * vel_score
            + 0.05 * age_score
        )
        return float(np.clip(cpi, 0.0, 10.0))

    except Exception as exc:
        logger.warning("compute_cpi_score failed: %s — returning 0", exc)
        return 0.0


def find_tca(
    propagator: SGP4Propagator,
    id1: int,
    id2: int,
    start: datetime = None,
    hours_ahead: float = 24.0,
    steps: int = 240,
) -> ConjunctionEvent | None:
    """
    Time of Closest Approach between two catalogued objects over
    [start, start + hours_ahead] (start defaults to the SIMULATION clock).

    Vectorised SGP4 scan on a grid no coarser than 60 s (``steps`` is a lower
    bound on the sample count), then bounded Brent refinement of the global
    minimum on [t - dt, t + dt] with direct sgp4 calls (~1 ms TCA accuracy).
    Does not touch the propagator's Kalman bookkeeping.
    """
    from scipy.optimize import minimize_scalar
    from app.core.screening import _jd_fr

    if start is None:
        from app.core import sim_clock
        start = sim_clock.simulation_now()
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    sats = getattr(propagator, "_satellites", None)
    if not sats:
        return None
    lock = getattr(propagator, "_lock", None)
    if lock is not None:
        with lock:
            e1, e2 = sats.get(id1), sats.get(id2)
    else:
        e1, e2 = sats.get(id1), sats.get(id2)
    if e1 is None or e2 is None:
        return None
    (sat1, name1), (sat2, name2) = e1, e2
    traj = getattr(propagator, "trajectory", None)
    if traj is not None:  # executed burns: SGP4 + propagated deviation
        sat1 = traj(id1) or sat1
        sat2 = traj(id2) or sat2

    span = float(hours_ahead) * 3600.0
    n = max(int(steps), int(math.ceil(span / 60.0))) + 1
    ts = np.linspace(0.0, span, n)
    dt = ts[1] - ts[0] if n > 1 else span
    jd0, fr0 = _jd_fr(start)
    jd = np.full(n, jd0)
    fr = fr0 + ts / 86400.0
    err1, r1, _ = sat1.sgp4_array(jd, fr)
    err2, r2, _ = sat2.sgp4_array(jd, fr)
    d = np.linalg.norm(r1 - r2, axis=1)
    d[(err1 != 0) | (err2 != 0)] = np.inf
    if not np.isfinite(d).any():
        return None
    k = int(np.argmin(d))

    def dist(t):
        ea, ra, _ = sat1.sgp4(jd0, fr0 + t / 86400.0)
        eb, rb, _ = sat2.sgp4(jd0, fr0 + t / 86400.0)
        if ea != 0 or eb != 0:
            return 1e12
        return float(np.linalg.norm(np.subtract(ra, rb)))

    # Every local minimum of the sampled distance is a candidate encounter: a
    # fast crossing can fall between samples, so the grid argmin alone is not
    # reliable. Refine the 20 lowest local minima and keep the true closest.
    left = np.r_[np.inf, d[:-1]]
    right = np.r_[d[1:], np.inf]
    minima = np.flatnonzero((d <= left) & (d <= right) & np.isfinite(d))
    if minima.size == 0:
        minima = np.array([k])
    minima = minima[np.argsort(d[minima])][:20]
    t_best, d_best = float(ts[k]), float(d[k])
    for km in minima:
        lo, hi = max(0.0, ts[km] - dt), min(span, ts[km] + dt)
        res = minimize_scalar(dist, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
        if res.fun < d_best:
            t_best, d_best = float(res.x), float(res.fun)

    ea, ra, va = sat1.sgp4(jd0, fr0 + t_best / 86400.0)
    eb, rb, vb = sat2.sgp4(jd0, fr0 + t_best / 86400.0)
    if ea != 0 or eb != 0:
        return None
    miss = float(np.linalg.norm(np.subtract(ra, rb)))

    event = ConjunctionEvent()
    event.sat1_id, event.sat1_name = id1, name1
    event.sat2_id, event.sat2_name = id2, name2
    event.miss_distance_km = miss
    event.relative_speed_kmh = float(np.linalg.norm(np.subtract(va, vb))) * 3600.0
    event.tca_utc = (start + timedelta(seconds=t_best)).isoformat()
    event.tca_hours = t_best / 3600.0
    event.severity = classify_severity(miss)
    event.covariance_ellipse = None
    return event
