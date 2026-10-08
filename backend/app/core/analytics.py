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

The live pipeline computes Pc in app.core.screening.compute_pc / foster_pc.
This module keeps the reference implementations that cross-check it:
  * build_rtn_covariance  - TLE-age covariance (same model as screening)
  * foster_integrate      - thin wrapper over screening.foster_pc (used to
                            label the ML surrogate's training set)
  * foster_integrate_dblquad - independent adaptive-quadrature Foster Pc used
                            in tests/test_screening_real.py to verify foster_pc
  * _chan_approximation   - Chan (1997) first-term series (dblquad fallback)
(The former compute_collision_probability / nasa_fragment_count helpers were
removed: they were unused by the pipeline and carried an assumed 500 kg mass
and an ad-hoc "affection rate". The real NASA SBM lives in app/core/breakup.py.)
"""

from __future__ import annotations

import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

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
