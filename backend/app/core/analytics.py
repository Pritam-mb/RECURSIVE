"""
analytics.py — Foster 1992 B-plane collision probability integration.

Implements the method used by NASA CARA (Conjunction Assessment
Risk Analysis) for computing collision probability between two
space objects.

Reference: Foster, J.L. & Estes, H.S. (1992). "A Parametric Analysis of
Orbital Debris Collision Probability and Maneuver Rate for Space Vehicles."
NASA/JSC-25898.

All units are km and km/s throughout. Never mix metres and km inside
a single calculation — conversions are explicit and labelled.
"""

from __future__ import annotations

import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

ASSUMED_MASS_KG = 500.0  # used only for the fragment-count estimate when no mass is known

# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — Build RTN covariance matrix in ECI frame
# ──────────────────────────────────────────────────────────────────────────────

def build_rtn_covariance(
    position_km: np.ndarray,
    velocity_kms: np.ndarray,
    tle_age_hours: float = 0.5,
) -> np.ndarray:
    """
    Build a 3x3 ECI position covariance from the TLE-age error-growth model.

    sigma_RTN(age) = SIGMA0 + GROWTH * age_days (km), see
    app.core.screening.SIGMA0_RTN_KM / SIGMA_GROWTH_RTN_KM_PER_DAY for the
    parameters and their sources (Flohrer et al. 2008; Vallado & Cefola 2012).
    tle_age_hours must be |epoch_of_interest - TLE epoch|.

    Returns: 3x3 numpy array in km^2.
    """
    from app.core.screening import rtn_to_eci_cov, tle_age_sigmas_km

    sig = tle_age_sigmas_km(float(tle_age_hours) / 24.0)
    try:
        return rtn_to_eci_cov(np.asarray(position_km, float), np.asarray(velocity_kms, float), sig)
    except Exception as exc:
        logger.warning("build_rtn_covariance degenerate state: %s", exc)
        return np.diag(np.square(sig))


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — Compute B-plane coordinate frame
# ──────────────────────────────────────────────────────────────────────────────

def compute_b_plane_transform(
    state_p: dict,
    state_q: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the B-plane coordinate frame for a conjunction encounter.

    The B-plane is perpendicular to the relative velocity vector at TCA.
    It is the standard reference plane for conjunction analysis.

    Parameters
    ----------
    state_p, state_q : dict
        State dicts with keys x, y, z (km) and vx, vy, vz (km/s).

    Returns
    -------
    h_hat : (3,) array — unit vector along relative velocity
    t_hat : (3,) array — in-plane unit vector (B_T direction)
    n_hat : (3,) array — normal unit vector (B_N direction)
    B     : (2, 3) array — projection matrix onto B-plane [t_hat; n_hat]
    """
    try:
        r_p = np.array([state_p["x"], state_p["y"], state_p["z"]], dtype=float)
        v_p = np.array([state_p["vx"], state_p["vy"], state_p["vz"]], dtype=float)
        r_q = np.array([state_q["x"], state_q["y"], state_q["z"]], dtype=float)
        v_q = np.array([state_q["vx"], state_q["vy"], state_q["vz"]], dtype=float)

        delta_v = v_q - v_p
        dv_norm = float(np.linalg.norm(delta_v))
        if dv_norm < 1e-12:
            # Nearly identical velocities — degenerate encounter
            h_hat = np.array([1.0, 0.0, 0.0])
        else:
            h_hat = delta_v / dv_norm

        # t_hat: in the encounter plane, perpendicular to h_hat
        cross_t = np.cross(h_hat, r_p)
        t_norm = float(np.linalg.norm(cross_t))
        if t_norm < 1e-12:
            # Degenerate — pick any perpendicular
            perp = np.array([1.0, 0.0, 0.0]) if abs(h_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            cross_t = np.cross(h_hat, perp)
            t_norm = float(np.linalg.norm(cross_t))
        t_hat = cross_t / t_norm

        # n_hat: completes right-handed frame
        n_hat = np.cross(t_hat, h_hat)
        n_norm = float(np.linalg.norm(n_hat))
        if n_norm > 1e-12:
            n_hat = n_hat / n_norm

        # B-plane projection matrix (2×3)
        B = np.vstack([t_hat, n_hat])

        return h_hat, t_hat, n_hat, B

    except Exception as exc:
        logger.warning("compute_b_plane_transform failed: %s — using identity", exc)
        h_hat = np.array([1.0, 0.0, 0.0])
        t_hat = np.array([0.0, 1.0, 0.0])
        n_hat = np.array([0.0, 0.0, 1.0])
        B = np.vstack([t_hat, n_hat])
        return h_hat, t_hat, n_hat, B


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 — Project combined covariance onto B-plane
# ──────────────────────────────────────────────────────────────────────────────

def project_covariance_to_bplane(
    cov_3d: np.ndarray,
    B_matrix: np.ndarray,
) -> np.ndarray:
    """
    Project a 3×3 combined position covariance onto the 2×2 B-plane.

    cov_3d   : (3, 3) combined covariance matrix in ECI (km²)
    B_matrix : (2, 3) projection matrix [t_hat; n_hat]

    Returns: (2, 2) covariance in B-plane (km²), guaranteed positive definite.
    """
    try:
        cov_3d = np.asarray(cov_3d, dtype=float)
        B_matrix = np.asarray(B_matrix, dtype=float)

        cov_2d = B_matrix @ cov_3d @ B_matrix.T

        # Ensure positive definiteness by regularising any tiny eigenvalues
        eigvals = np.linalg.eigvalsh(cov_2d)
        min_eig = float(np.min(eigvals))
        if min_eig < 1e-12:
            cov_2d += (abs(min_eig) + 1e-12) * np.eye(2)

        return cov_2d

    except Exception as exc:
        logger.warning("project_covariance_to_bplane failed: %s — returning default", exc)
        return np.diag([1.0, 0.0025])


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 — Foster double integration over collision disk
# ──────────────────────────────────────────────────────────────────────────────

def foster_integrate(
    b_vec_km: np.ndarray,
    cov_2d_km2: np.ndarray,
    hbr_km: float = 0.010,
) -> float:
    """
    Foster 2-D Pc via deterministic Gauss-Legendre x trapezoid quadrature of
    the bivariate normal over the hard-body disk (app.core.screening.foster_pc).
    Cross-checked against the adaptive dblquad version below in
    tests/test_screening_real.py.
    """
    try:
        from app.core.screening import foster_pc
        return foster_pc(b_vec_km, cov_2d_km2, hbr_km)
    except Exception as exc:
        logger.warning("foster_pc failed (%s) - using dblquad", exc)
        return foster_integrate_dblquad(b_vec_km, cov_2d_km2, hbr_km)


def foster_integrate_dblquad(
    b_vec_km: np.ndarray,
    cov_2d_km2: np.ndarray,
    hbr_km: float = 0.010,
) -> float:
    """
    Compute collision probability by integrating the bivariate Gaussian PDF
    over the circular collision disk of radius hbr_km in the B-plane.

    Uses scipy.integrate.dblquad with a 2-second timeout fallback to the
    Chan (1997) series approximation if integration is too slow or fails.

    Parameters
    ----------
    b_vec_km   : (2,) miss vector in B-plane [B_T, B_N] (km)
    cov_2d_km2 : (2, 2) covariance matrix in B-plane (km²)
    hbr_km     : combined hard-body radius (km), default 10 m = 0.010 km

    Returns
    -------
    float in [0, 1]
    """
    try:
        from scipy.integrate import dblquad
        from scipy.stats import multivariate_normal

        b_vec_km = np.asarray(b_vec_km, dtype=float)
        cov_2d_km2 = np.asarray(cov_2d_km2, dtype=float)

        # Validate covariance
        eigvals = np.linalg.eigvalsh(cov_2d_km2)
        if np.any(eigvals < 1e-20):
            return _chan_approximation(b_vec_km, cov_2d_km2, hbr_km)

        rv = multivariate_normal(mean=b_vec_km, cov=cov_2d_km2)

        def integrand(y: float, x: float) -> float:
            return float(rv.pdf([x, y]))

        def y_lower(x: float) -> float:
            disc = hbr_km**2 - x**2
            return -math.sqrt(max(disc, 0.0))

        def y_upper(x: float) -> float:
            disc = hbr_km**2 - x**2
            return math.sqrt(max(disc, 0.0))

        import signal

        # Use a threading-based timeout compatible with Windows
        import threading
        result_holder = [None]
        error_holder = [None]

        def _integrate():
            try:
                val, _ = dblquad(
                    integrand,
                    -hbr_km,
                    hbr_km,
                    y_lower,
                    y_upper,
                    epsabs=1e-8,
                    epsrel=1e-6,
                )
                result_holder[0] = float(np.clip(val, 0.0, 1.0))
            except Exception as exc:
                error_holder[0] = exc

        thread = threading.Thread(target=_integrate, daemon=True)
        thread.start()
        thread.join(timeout=2.0)

        if result_holder[0] is not None:
            return result_holder[0]

        # Timeout or error — fall back to Chan approximation
        if error_holder[0] is not None:
            logger.warning(
                "foster_integrate dblquad error: %s — using Chan approximation",
                error_holder[0],
            )
        else:
            logger.debug("foster_integrate timed out — using Chan approximation")

        return _chan_approximation(b_vec_km, cov_2d_km2, hbr_km)

    except Exception as exc:
        logger.warning("foster_integrate failed entirely: %s — using Chan", exc)
        return _chan_approximation(b_vec_km, cov_2d_km2, hbr_km)


def _chan_approximation(
    b_vec_km: np.ndarray,
    cov_2d_km2: np.ndarray,
    hbr_km: float,
) -> float:
    """
    Chan (1997) series approximation for collision probability.
    Fast fallback when numerical integration is unavailable or slow.

    Reference: Chan, F.K. (1997). "Spacecraft Collision Probability."
    """
    try:
        b_vec_km = np.asarray(b_vec_km, dtype=float)
        cov_2d_km2 = np.asarray(cov_2d_km2, dtype=float)

        sigma_sq = float(np.trace(cov_2d_km2)) / 2.0
        if sigma_sq < 1e-20:
            return 0.0

        miss_sq = float(b_vec_km @ b_vec_km)
        u = hbr_km**2 / (2.0 * sigma_sq)
        x = miss_sq / (2.0 * sigma_sq)

        # Chan series: Pc ≈ u * exp(-(u+x)) * I0(sqrt(4*u*x)) — first term approximation
        # Using modified Bessel function of the first kind, order 0
        from scipy.special import iv as bessel_iv
        val = u * math.exp(-(u + x)) * float(bessel_iv(0, 2.0 * math.sqrt(u * x)))
        return float(np.clip(val, 0.0, 1.0))
    except Exception as exc:
        logger.debug("Chan approximation failed: %s — returning 0", exc)
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 5 — Main entry point: compute_collision_probability
# ──────────────────────────────────────────────────────────────────────────────

def compute_collision_probability(
    state_p: dict,
    state_q: dict,
    cov_p: np.ndarray | None = None,
    cov_q: np.ndarray | None = None,
    tle_age_hours: float = 0.5,
    hbr_km: float = 0.010,
    tle_age_hours_q: float | None = None,
) -> dict:
    """
    Compute Foster B-plane collision probability between two space objects.

    Parameters
    ----------
    state_p, state_q : dict
        Satellite states with keys: x, y, z (km), vx, vy, vz (km/s).
    cov_p, cov_q : (3,3) ndarray or None
        ECI position covariance matrices (km²). If None, builds from RTN model.
    tle_age_hours : float
        Age of TLE data used to set along-track uncertainty.

    Returns
    -------
    dict with keys:
        p_collision, miss_distance_km, b_t_km, b_n_km,
        cov_2d (list-of-lists), covariance_ellipse (dict)
    """
    try:
        r_p = np.array([state_p["x"], state_p["y"], state_p["z"]], dtype=float)
        v_p = np.array([state_p["vx"], state_p["vy"], state_p["vz"]], dtype=float)
        r_q = np.array([state_q["x"], state_q["y"], state_q["z"]], dtype=float)
        v_q = np.array([state_q["vx"], state_q["vy"], state_q["vz"]], dtype=float)

        # Miss vector and distance
        delta_r = r_q - r_p
        miss_distance_km = float(np.linalg.norm(delta_r))
        rel_vel_kms = float(np.linalg.norm(v_q - v_p))

        # Build covariances if not provided
        if cov_p is None:
            cov_p = build_rtn_covariance(r_p, v_p, tle_age_hours)
        sigma_source = "provided" if (cov_p is not None and cov_q is not None) else "tle_age_model"
        if cov_q is None:
            cov_q = build_rtn_covariance(
                r_q, v_q, tle_age_hours if tle_age_hours_q is None else tle_age_hours_q
            )

        cov_combined = np.asarray(cov_p, dtype=float) + np.asarray(cov_q, dtype=float)

        # B-plane transform
        h_hat, t_hat, n_hat, B = compute_b_plane_transform(state_p, state_q)

        # Miss vector projected into B-plane (km)
        b_vec = B @ delta_r  # shape (2,)
        b_t_km = float(b_vec[0])
        b_n_km = float(b_vec[1])

        # Project covariance onto B-plane
        cov_2d = project_covariance_to_bplane(cov_combined, B)

        # Foster integration (hbr_km = combined hard-body radius, caller-supplied)
        hbr_km = float(hbr_km)
        p_collision = foster_integrate(b_vec, cov_2d, hbr_km=hbr_km)

        # Covariance ellipse for frontend rendering
        ellipse = ellipse_from_covariance(cov_2d)

        # Overlap / affection rate: fraction of 3-sigma ellipse overlapping HBR disk
        a_km = ellipse["a_m"] / 1000.0
        b_km = ellipse["b_m"] / 1000.0
        ellipse_area = math.pi * a_km * b_km if a_km > 0 and b_km > 0 else 1e-10
        disk_area = math.pi * hbr_km**2
        overlap_fraction = min(1.0, disk_area / ellipse_area)
        affection_rate = round(overlap_fraction * 100.0, 2)

        # Fragment count: masses are an ASSUMPTION (no mass data here), tagged below.
        mass_p = ASSUMED_MASS_KG
        mass_q = ASSUMED_MASS_KG
        frag_count = nasa_fragment_count(mass_p, mass_q, rel_vel_kms)

        covariance_ellipse = {
            "a": round(ellipse["a_m"], 1),           # metres
            "b": round(ellipse["b_m"], 1),            # metres
            "angle": round(ellipse["angle_rad"], 4),  # radians
            "affection_rate": affection_rate,          # 0–100 %
            "predicted_fragments": frag_count,         # integer
            "assumed_mass_kg": ASSUMED_MASS_KG,
            "mass_source": "default",
        }

        return {
            "p_collision": float(np.clip(p_collision, 0.0, 1.0)),
            "miss_distance_km": round(miss_distance_km, 6),
            "b_t_km": round(b_t_km, 6),
            "b_n_km": round(b_n_km, 6),
            "cov_2d": cov_2d.tolist(),
            "covariance_ellipse": covariance_ellipse,
            "hbr_km": hbr_km,
            "pc_method": "foster",
            "sigma_source": sigma_source,
        }

    except Exception as exc:
        # No fabricated numbers: report the failure explicitly.
        logger.warning("compute_collision_probability failed: %s", exc)
        try:
            r_p2 = np.array([state_p.get("x", 0), state_p.get("y", 0), state_p.get("z", 0)], dtype=float)
            r_q2 = np.array([state_q.get("x", 0), state_q.get("y", 0), state_q.get("z", 0)], dtype=float)
            miss_km = float(np.linalg.norm(r_q2 - r_p2))
        except Exception:
            miss_km = float("nan")
        return {
            "p_collision": 0.0,
            "miss_distance_km": round(miss_km, 6),
            "b_t_km": None,
            "b_n_km": None,
            "cov_2d": None,
            "covariance_ellipse": None,
            "hbr_km": hbr_km,
            "pc_method": "failed",
            "sigma_source": None,
        }


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 6 — Extract ellipse parameters from 2×2 covariance
# ──────────────────────────────────────────────────────────────────────────────

def ellipse_from_covariance(cov_2d: np.ndarray) -> dict:
    """
    Compute the 3-sigma covariance ellipse parameters.

    Parameters
    ----------
    cov_2d : (2, 2) covariance matrix (km²)

    Returns
    -------
    dict with:
        a_m       : semi-major axis in metres
        b_m       : semi-minor axis in metres
        angle_rad : rotation angle of major axis (radians)
    """
    try:
        cov_2d = np.asarray(cov_2d, dtype=float)
        eigvals, eigvecs = np.linalg.eigh(cov_2d)

        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        eigvals = np.maximum(eigvals, 0.0)

        # 3-sigma axes in km → convert to metres
        a_km = 3.0 * math.sqrt(float(eigvals[0]))
        b_km = 3.0 * math.sqrt(float(eigvals[1]))

        a_m = a_km * 1000.0
        b_m = b_km * 1000.0

        # Angle of major axis
        major_vec = eigvecs[:, 0]
        angle_rad = math.atan2(float(major_vec[1]), float(major_vec[0]))

        return {
            "a_m": round(a_m, 2),
            "b_m": round(b_m, 2),
            "angle_rad": round(angle_rad, 6),
        }

    except Exception as exc:
        logger.warning("ellipse_from_covariance failed: %s", exc)
        return {"a_m": 3000.0, "b_m": 150.0, "angle_rad": 0.0}


# ──────────────────────────────────────────────────────────────────────────────
# FUNCTION 7 — NASA fragment count model
# ──────────────────────────────────────────────────────────────────────────────

def nasa_fragment_count(
    mass_p_kg: float,
    mass_q_kg: float,
    rel_vel_kms: float,
) -> int:
    """
    Estimate debris fragment count using the NASA Standard Breakup Model.

    Catastrophic threshold: E/M > 40 J/kg where
      E = 0.5 * m_small * (v_rel * 1000)²   (Joules)
      M = m_large                              (kg)

    Catastrophic collision:
      N ≈ 0.1 * (M1 + M2)^0.75

    Cratering (sub-catastrophic):
      N ≈ 0.1 * (M1 + M2)^0.75 * 0.1

    Reference: NASA/TM-2001-210865 (EVOLVE 4.0 model)

    Returns integer fragment count (minimum 0).
    """
    try:
        mass_p_kg = max(0.0, float(mass_p_kg))
        mass_q_kg = max(0.0, float(mass_q_kg))
        rel_vel_ms = float(rel_vel_kms) * 1000.0  # km/s → m/s

        m_small = min(mass_p_kg, mass_q_kg)
        m_large = max(mass_p_kg, mass_q_kg)

        if m_small < 1e-6 or m_large < 1e-6:
            return 0

        # Specific energy (J/kg)
        kinetic_energy = 0.5 * m_small * rel_vel_ms**2
        specific_energy = kinetic_energy / m_large

        total_mass = mass_p_kg + mass_q_kg

        if specific_energy > 40.0:
            # Catastrophic breakup
            n = 0.1 * (total_mass ** 0.75)
        else:
            # Sub-catastrophic cratering
            n = 0.1 * (total_mass ** 0.75) * 0.1

        return max(0, int(round(n)))

    except Exception as exc:
        logger.warning("nasa_fragment_count failed: %s", exc)
        return 0
