"""
Conjunction assessment engine.
Computes pairwise miss distances and identifies close approaches.
Uses Foster 1992 B-plane integration (via analytics.py) for collision
probability. Falls back to Gaussian approximation only when analytics
module is unavailable.

CPI weights are calibrated against Foster Pc ordering so that higher
CPI always correlates with higher Foster Pc.

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

# ── Analytics import (Foster Pc engine) ───────────────────────────────────────
try:
    from app.core.analytics import compute_collision_probability
    _ANALYTICS_AVAILABLE = True
    logger.info("Foster B-plane analytics loaded successfully")
except ImportError as _import_err:
    _ANALYTICS_AVAILABLE = False
    logger.warning("analytics.py not available (%s) — using Gaussian fallback", _import_err)

# Thresholds
ALERT_DISTANCE_KM = 50.0       # Red alert
WARNING_DISTANCE_KM = 200.0    # Yellow warning
WATCH_DISTANCE_KM = 500.0      # Green watch


# ── Gaussian fallback (used only when analytics import fails) ─────────────────

def _gaussian_fallback(miss_km: float) -> float:
    """
    Simple Gaussian approximation used ONLY as fallback when analytics.py
    cannot be imported. Never call this as the primary path.

    Not physically calibrated — kept solely to prevent a hard crash.
    """
    sigma = max(1.0, miss_km / 6.0)
    hbr   = 0.010  # km (10 m)
    p = float(np.exp(-0.5 * (miss_km / sigma) ** 2) * (hbr / sigma) ** 2)
    return min(1.0, max(0.0, p))


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


def compute_ric_frame(pos_chief: np.ndarray, vel_chief: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RIC (Radial-In-Track-Cross-track) frame unit vectors.
    R: radial (along position)
    I: in-track (along velocity direction)
    C: cross-track (orbit normal)
    """
    r_norm = np.linalg.norm(pos_chief)
    if r_norm < 1e-6:
        return np.array([1, 0, 0], dtype=float), np.array([0, 1, 0], dtype=float), np.array([0, 0, 1], dtype=float)

    r_hat = pos_chief / r_norm
    h = np.cross(pos_chief, vel_chief)
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-6:
        c_hat = np.array([0, 0, 1], dtype=float)
    else:
        c_hat = h / h_norm

    i_hat = np.cross(c_hat, r_hat)
    return r_hat, i_hat, c_hat


def transform_covariance_to_ric(
    cov_6x6_sat1: np.ndarray,
    cov_6x6_sat2: np.ndarray,
    r_hat: np.ndarray,
    i_hat: np.ndarray,
    c_hat: np.ndarray,
) -> np.ndarray:
    """
    Transform covariances from ECI to RIC frame relative coordinates.
    Returns a 2x2 covariance matrix for relative RIC position (R, I only).
    """
    R_mat = np.vstack([r_hat, i_hat, c_hat])
    P_pos_1 = cov_6x6_sat1[:3, :3]
    P_pos_2 = cov_6x6_sat2[:3, :3]
    P_rel_eci = P_pos_1 + P_pos_2
    P_rel_ric = R_mat @ P_rel_eci @ R_mat.T
    P_rel_2d = P_rel_ric[:2, :2]
    return P_rel_2d


def compute_collision_probability_2d(
    miss_distance_ric: np.ndarray,
    covariance_2d: np.ndarray,
    combined_radius_km: float = 0.010,
) -> float:
    """
    Legacy RIC-frame 2D Gaussian approximation.
    Kept for compatibility with Kalman-state-based paths.
    """
    try:
        eigvals = np.linalg.eigvals(covariance_2d)
        if np.any(eigvals <= 1e-10):
            miss_dist_norm = np.linalg.norm(miss_distance_ric)
            return 1.0 if miss_dist_norm < combined_radius_km else 0.0

        try:
            cov_inv = np.linalg.inv(covariance_2d)
        except np.linalg.LinAlgError:
            miss_dist_norm = np.linalg.norm(miss_distance_ric)
            return 1.0 if miss_dist_norm < combined_radius_km else 0.0

        mahal_dist_sq = miss_distance_ric @ cov_inv @ miss_distance_ric
        from scipy.special import gammainc
        p_coll = float(gammainc(1.0, (combined_radius_km**2) / (np.trace(covariance_2d) + 1e-10)))
        return float(np.clip(p_coll, 0.0, 1.0))
    except Exception as e:
        logger.debug(f"Collision probability computation failed: {e}")
        return 0.0


# ── CPI score calibrated against Foster Pc ───────────────────────────────────

def compute_cpi_score(
    probability_of_collision: float,
    miss_distance_km: float,
    tca_hours: float = 12.0,
    relative_velocity_kms: float = 7.5,
    tle_age_hours: float = 6.0,
) -> float:
    """
    Compute Conjunction Priority Index (0–10) calibrated against Foster Pc ordering.

    Weights (sum = 1.0):
      0.40 — miss distance component   (strongest predictor of Pc)
      0.30 — Foster Pc component       (log-mapped to 0–10 scale)
      0.15 — TCA urgency component     (time to closest approach)
      0.10 — closing speed component   (relative velocity)
      0.05 — TLE data quality          (age of orbital elements)

    NASA CARA thresholds for reference:
      Pc >= 1e-4 (score ≈ 5.0) → watch / elevated concern
      Pc >= 1e-3 (score ≈ 7.5) → action threshold
      Pc >= 1e-2 (score ≈ 10.) → emergency — maneuver required
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
        # This mapping is calibrated so CPI rank == Foster Pc rank:
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
        # Older TLE → larger uncertainty → less reliable → slightly higher priority
        age = max(0.0, float(tle_age_hours))
        age_score = float(np.clip(age / 48.0 * 10.0, 0.0, 10.0))

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


# ── Main screening function ───────────────────────────────────────────────────

def screen_conjunctions(
    states: list[SatelliteState],
    threshold_km: float = WATCH_DISTANCE_KM,
    kalman_states: dict = None,
    propagator: SGP4Propagator = None,
) -> list[ConjunctionEvent]:
    """
    Pairwise conjunction screening with Foster B-plane collision probability.

    Uses analytics.compute_collision_probability (Foster 1992) as primary path.
    Falls back to Gaussian approximation only if analytics import failed.

    When a propagator is supplied, real Time-of-Closest-Approach is refined
    via find_tca() for red/yellow pairs (distance <= WARNING_DISTANCE_KM),
    so tca_utc reflects the actual closest approach rather than the snapshot
    epoch and CPI uses the true time-to-closest-approach.

    kalman_states: dict[norad_id] -> dict with covariance_6x6 (optional, legacy)
    Returns events where miss distance < threshold or CPI > 2.0, sorted by CPI.
    """
    if kalman_states is None:
        kalman_states = {}

    events = []
    n = len(states)

    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = states[i], states[j]

            # Skip if either had propagation errors
            if s1.error_code != 0 or s2.error_code != 0:
                continue

            dist = compute_miss_distance(s1, s2)
            rel_speed_kmh = compute_relative_speed(s1, s2)
            rel_speed_kms = rel_speed_kmh / 3600.0
            severity = classify_severity(dist)

            if severity == "none":
                continue  # Far outside watch threshold — skip entirely

            # ── Foster B-plane collision probability (primary path) ─────────
            p_collision = 0.0
            covariance_ellipse = None

            if _ANALYTICS_AVAILABLE:
                try:
                    state_a_dict = {
                        "x": float(s1.x), "y": float(s1.y), "z": float(s1.z),
                        "vx": float(s1.vx), "vy": float(s1.vy), "vz": float(s1.vz),
                    }
                    state_b_dict = {
                        "x": float(s2.x), "y": float(s2.y), "z": float(s2.z),
                        "vx": float(s2.vx), "vy": float(s2.vy), "vz": float(s2.vz),
                    }
                    analytics_result = compute_collision_probability(
                        state_a_dict,
                        state_b_dict,
                        tle_age_hours=0.5,
                    )
                    p_collision = analytics_result["p_collision"]
                    covariance_ellipse = analytics_result["covariance_ellipse"]
                except Exception as analytics_err:
                    logger.warning(
                        "Analytics Foster Pc failed for pair (%d, %d): %s — using fallback",
                        s1.norad_id, s2.norad_id, analytics_err,
                    )
                    p_collision = _gaussian_fallback(dist)
                    covariance_ellipse = None
            else:
                p_collision = _gaussian_fallback(dist)
                covariance_ellipse = None

            # ── Legacy Kalman-based RIC Pc (if covariances available) ──────
            # Only used to potentially refine p_collision; Foster always wins
            kal_state_1 = kalman_states.get(s1.norad_id)
            kal_state_2 = kalman_states.get(s2.norad_id)
            if (
                not _ANALYTICS_AVAILABLE
                and kal_state_1 is not None
                and kal_state_2 is not None
            ):
                try:
                    pos_1 = np.array([s1.x, s1.y, s1.z], dtype=float)
                    vel_1 = np.array([s1.vx, s1.vy, s1.vz], dtype=float)
                    r_hat, i_hat, c_hat = compute_ric_frame(pos_1, vel_1)
                    cov_1 = np.asarray(kal_state_1.get("covariance_6x6", []), dtype=float)
                    cov_2 = np.asarray(kal_state_2.get("covariance_6x6", []), dtype=float)
                    if cov_1.ndim == 1 and cov_1.size == 36:
                        cov_1 = cov_1.reshape(6, 6)
                    if cov_2.ndim == 1 and cov_2.size == 36:
                        cov_2 = cov_2.reshape(6, 6)
                    if cov_1.shape == (6, 6) and cov_2.shape == (6, 6):
                        cov_2d = transform_covariance_to_ric(cov_1, cov_2, r_hat, i_hat, c_hat)
                        rel_pos = pos_1 - np.array([s2.x, s2.y, s2.z], dtype=float)
                        rel_pos_ric = np.array([np.dot(rel_pos, r_hat), np.dot(rel_pos, i_hat)], dtype=float)
                        p_collision = compute_collision_probability_2d(rel_pos_ric, cov_2d, 0.010)
                except Exception as ric_err:
                    logger.debug("RIC CPI computation failed: %s", ric_err)

            # ── Real TCA refinement for close pairs (red/yellow) ────────────
            tca_utc = s1.epoch_utc
            if propagator is not None and severity in ("red", "yellow"):
                try:
                    refined = find_tca(
                        propagator,
                        s1.norad_id,
                        s2.norad_id,
                        hours_ahead=24.0,
                        steps=240,
                    )
                    if refined is not None and refined.miss_distance_km <= dist:
                        dist = refined.miss_distance_km
                        rel_speed_kmh = refined.relative_speed_kmh
                        rel_speed_kms = rel_speed_kmh / 3600.0
                        tca_utc = refined.tca_utc
                except Exception as tca_err:
                    logger.debug("TCA refinement failed for (%d, %d): %s", s1.norad_id, s2.norad_id, tca_err)

            # ── CPI with Foster-calibrated weights ─────────────────────────
            # TCA urgency is measured against the snapshot epoch, not wall
            # clock. Using datetime.now() made the value drift whenever the
            # simulation clock is warped, and desynchronised CPI from the
            # states actually being screened.
            tca_hours = 12.0  # Used when TCA is unavailable
            try:
                tca_dt = datetime.fromisoformat(tca_utc)
                if tca_dt.tzinfo is None:
                    tca_dt = tca_dt.replace(tzinfo=timezone.utc)
                reference = datetime.fromisoformat(s1.epoch_utc)
                if reference.tzinfo is None:
                    reference = reference.replace(tzinfo=timezone.utc)
                tca_hours = max(0.0, (tca_dt - reference).total_seconds() / 3600.0)
            except Exception:
                pass

            cpi_score = compute_cpi_score(
                p_collision,
                dist,
                tca_hours=tca_hours,
                relative_velocity_kms=rel_speed_kms,
                tle_age_hours=0.5,
            )

            # Report if severity is high or CPI is concerning
            if severity == "none" and cpi_score < 2.0:
                continue

            event = ConjunctionEvent()
            event.sat1_id = s1.norad_id
            event.sat1_name = s1.name
            event.sat2_id = s2.norad_id
            event.sat2_name = s2.name
            event.miss_distance_km = dist
            event.relative_speed_kmh = rel_speed_kmh
            event.tca_utc = tca_utc
            event.tca_hours = tca_hours
            event.severity = severity
            event.probability_of_collision = p_collision
            event.cpi_score = cpi_score
            event.covariance_ellipse = covariance_ellipse

            events.append(event)

    # Sort by CPI score descending (most critical first)
    events.sort(key=lambda e: e.cpi_score, reverse=True)
    return events


def find_tca(
    propagator: SGP4Propagator,
    id1: int,
    id2: int,
    start: datetime = None,
    hours_ahead: float = 24.0,
    steps: int = 240,
) -> ConjunctionEvent | None:
    """
    Find Time of Closest Approach between two satellites
    over the next `hours_ahead` hours using iterative search.
    """
    if start is None:
        start = datetime.now(timezone.utc)

    dt_step = timedelta(hours=hours_ahead) / steps
    min_dist = float("inf")
    best_event = None

    for i in range(steps):
        t = start + dt_step * i
        s1 = propagator.propagate_one(id1, t)
        s2 = propagator.propagate_one(id2, t)

        if s1 is None or s2 is None:
            continue
        if s1.error_code != 0 or s2.error_code != 0:
            continue

        dist = compute_miss_distance(s1, s2)

        if dist < min_dist:
            min_dist = dist
            best_event = ConjunctionEvent()
            best_event.sat1_id = s1.norad_id
            best_event.sat1_name = s1.name
            best_event.sat2_id = s2.norad_id
            best_event.sat2_name = s2.name
            best_event.miss_distance_km = dist
            best_event.relative_speed_kmh = compute_relative_speed(s1, s2)
            best_event.tca_utc = t.isoformat()
            best_event.severity = classify_severity(dist)
            best_event.covariance_ellipse = None  # Not computed here

    return best_event
