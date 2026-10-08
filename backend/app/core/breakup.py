"""
NASA Standard Breakup Model (SBM) for collisions + vectorised fragment dynamics.

References
----------
* Johnson, N.L., Krisko, P.H., Liou, J.-C., Anz-Meador, P.D. (2001),
  "NASA's new breakup model of EVOLVE 4.0", Adv. Space Res. 28(9), 1377-1384.
* Krisko, P.H. (2011), "Proper implementation of the 1998 NASA breakup model",
  Orbital Debris Quarterly News 15(4).
* Vallado, D.A. (2013), Fundamentals of Astrodynamics and Applications, 4th ed.,
  Table 8-4 (exponential atmosphere).

What is implemented (collision branch of the SBM)
-------------------------------------------------
1. Catastrophic test: energy-to-mass ratio EMR = 0.5 * m_p * v_imp^2 / M_t,
   catastrophic when EMR >= 40 J/g (= 40 000 J/kg).  m_p is the lighter
   ("projectile") object, M_t the heavier ("target").
2. Reference mass M:
     catastrophic      M = m_t + m_p                      [kg]
     non-catastrophic  M = m_p * v_imp^2  (v in km/s)       [kg]
3. Cumulative size distribution  N(>=Lc) = 0.1 * M^0.75 * Lc^-1.71  (Lc in m).
   Sizes are sampled from this power law by inverse-CDF on [Lc_min, Lc_max].
4. Area-to-mass ratio: bimodal normal in chi = log10(A/M) with the size
   dependent alpha/mu/sigma tables for spacecraft (and rocket bodies) of
   Johnson et al. (2001) eq. 6-7.  Area A = 0.556945 * Lc^2.0047077, mass = A / (A/M).
5. Ejection speed: nu = log10(dV [m/s]) ~ Normal(0.9*chi + 2.9, 0.4), directions
   isotropic.  Fragment velocity = parent velocity + dV.
6. Sanity controls (documented, counted in the result):
   * momentum: per parent, the mass-weighted mean dV of the sampled fragments is
     removed so the fragments carry exactly the parent's momentum;
   * no hyperbolic fragments: any dV that would put a fragment at or above
     0.99 * local escape speed is scaled back onto that limit ("dv_capped");
   * fragments whose perigee is below 100 km are kept (they physically re-enter
     within the first revolution) and are removed by the propagator.
7. Seeded numpy Generator per event (seed recorded); at most ``max_fragments``
   fragments are sampled for propagation/rendering, each representing
   ``weight = n_total / n_sampled`` real fragments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MU_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.137
J2 = 1.08262668e-3
OMEGA_EARTH_RAD_S = 7.292115e-5
DEFAULT_CD = 2.2
REENTRY_ALT_KM = 100.0

CATASTROPHIC_EMR_J_PER_KG = 40_000.0   # 40 J/g
DEFAULT_LC_MIN_M = 0.10               # 10 cm: trackable-size fragments
BULK_DENSITY_KG_M3 = 100.0            # assumed spacecraft bulk density -> Lc_max
ESCAPE_FRACTION = 0.99


# ── Size / count ─────────────────────────────────────────────────────────────

def sbm_reference_mass(m1_kg: float, m2_kg: float, v_rel_kms: float) -> tuple[float, bool, float]:
    """Return (M_ref_kg, catastrophic, EMR_J_per_kg) for a collision."""
    m_p = min(m1_kg, m2_kg)
    m_t = max(m1_kg, m2_kg)
    v_ms = v_rel_kms * 1000.0
    emr = 0.5 * m_p * v_ms * v_ms / m_t if m_t > 0 else 0.0
    catastrophic = emr >= CATASTROPHIC_EMR_J_PER_KG
    if catastrophic:
        m_ref = m_t + m_p
    else:
        m_ref = m_p * v_rel_kms * v_rel_kms
    return float(m_ref), bool(catastrophic), float(emr)


def sbm_cumulative_count(m_ref_kg: float, lc_m: float) -> float:
    """N(>= Lc) = 0.1 M^0.75 Lc^-1.71."""
    return 0.1 * (m_ref_kg ** 0.75) * (lc_m ** -1.71)


def sample_lc(rng: np.random.Generator, n: int, lc_min: float, lc_max: float) -> np.ndarray:
    """Inverse CDF of the truncated power law  pdf ~ Lc^-2.71  on [lc_min, lc_max]."""
    beta = 1.71
    u = rng.random(n)
    lo = lc_min ** -beta
    hi = lc_max ** -beta
    return (lo - u * (lo - hi)) ** (-1.0 / beta)


# ── Area-to-mass (Johnson 2001 eq. 6/7) ────────────────────────────────────

def _am_params_spacecraft(lam: np.ndarray):
    alpha = np.where(lam <= -1.95, 0.0, np.where(lam < 0.55, 0.3 + 0.4 * (lam + 1.2), 1.0))
    mu1 = np.where(lam <= -1.1, -0.6, np.where(lam < 0.0, -0.6 - 0.318 * (lam + 1.1), -0.95))
    s1 = np.where(lam <= -1.3, 0.1, np.where(lam < -0.3, 0.1 + 0.2 * (lam + 1.3), 0.3))
    mu2 = np.where(lam <= -0.7, -1.2, np.where(lam < -0.1, -1.2 - 1.333 * (lam + 0.7), -2.0))
    s2 = np.where(lam <= -0.5, 0.5, np.where(lam < -0.3, 0.5 - (lam + 0.5), 0.3))
    return alpha, mu1, s1, mu2, s2


def _am_params_rocket_body(lam: np.ndarray):
    alpha = np.where(lam <= -1.4, 1.0, np.where(lam < 0.0, 1.0 - 0.3571 * (lam + 1.4), 0.5))
    mu1 = np.where(lam <= -0.5, -0.45, np.where(lam < 0.0, -0.45 - 0.9 * (lam + 0.5), -0.9))
    s1 = np.full_like(lam, 0.55)
    mu2 = np.full_like(lam, -0.9)
    s2 = np.where(lam <= -1.0, 0.28, np.where(lam < 0.1, 0.28 - 0.1636 * (lam + 1.0), 0.1))
    return alpha, mu1, s1, mu2, s2


def sample_log_am(rng: np.random.Generator, lc: np.ndarray, rocket_body: np.ndarray | bool = False) -> np.ndarray:
    """Sample chi = log10(A/M [m^2/kg]) from the bimodal SBM distribution."""
    lam = np.log10(lc)
    sc = _am_params_spacecraft(lam)
    rb = _am_params_rocket_body(lam)
    rbm = np.broadcast_to(np.asarray(rocket_body, dtype=bool), lam.shape)
    alpha, mu1, s1, mu2, s2 = (np.where(rbm, b, a) for a, b in zip(sc, rb))
    first = rng.random(lam.shape) < alpha
    z = rng.standard_normal(lam.shape)
    return np.where(first, mu1 + s1 * z, mu2 + s2 * z)


def area_from_lc(lc: np.ndarray) -> np.ndarray:
    lc = np.asarray(lc, dtype=float)
    return np.where(lc < 0.00167, 0.540424 * lc ** 2, 0.556945 * lc ** 2.0047077)


def sample_dv_ms(rng: np.random.Generator, chi: np.ndarray) -> np.ndarray:
    """Collision ejection speed: log10(dV) ~ N(0.9 chi + 2.9, 0.4)."""
    nu = 0.9 * chi + 2.9 + 0.4 * rng.standard_normal(chi.shape)
    return 10.0 ** nu


def isotropic_unit(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ── Breakup ─────────────────────────────────────────────────────────────────

@dataclass
class BreakupResult:
    catastrophic: bool
    emr_j_per_g: float
    m_ref_kg: float
    lc_min_m: float
    lc_max_m: float
    n_total: int                    # SBM count of fragments >= lc_min (real)
    n_sampled: int                  # fragments actually generated/propagated
    weight: float                   # real fragments represented per sample
    seed: int
    impact_point_km: np.ndarray
    lc_m: np.ndarray
    am_m2_kg: np.ndarray
    area_m2: np.ndarray
    mass_kg: np.ndarray
    dv_kms: np.ndarray              # (n,3) applied ejection velocity
    parent_index: np.ndarray        # 0 -> parent A, 1 -> parent B
    r_km: np.ndarray                # (n,3) initial positions
    v_kms: np.ndarray               # (n,3) initial velocities
    n_dv_capped: int = 0
    n_suborbital: int = 0
    momentum_residual_before: float = 0.0
    momentum_residual_after: float = 0.0
    represented_mass_kg: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)


def _cap_to_bound_orbit(r: np.ndarray, v0: np.ndarray, dv: np.ndarray) -> tuple[np.ndarray, int]:
    """Scale dV so |v0 + f dv| <= ESCAPE_FRACTION * v_esc(r). Returns (dv, n_capped)."""
    rn = np.linalg.norm(r, axis=1)
    vmax = ESCAPE_FRACTION * np.sqrt(2.0 * MU_KM3_S2 / rn)
    v_new = v0 + dv
    over = np.linalg.norm(v_new, axis=1) >= vmax
    if not np.any(over):
        return dv, 0
    a = np.einsum("ij,ij->i", dv[over], dv[over])
    b = 2.0 * np.einsum("ij,ij->i", v0[over], dv[over])
    c = np.einsum("ij,ij->i", v0[over], v0[over]) - vmax[over] ** 2
    f = (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0.0))) / (2 * a)
    dv = dv.copy()
    dv[over] *= np.clip(f, 0.0, 1.0)[:, None]
    return dv, int(over.sum())


def simulate_breakup(
    r_a_km, v_a_kms, m_a_kg: float,
    r_b_km, v_b_kms, m_b_kg: float,
    *,
    seed: int,
    rocket_body_a: bool = False,
    rocket_body_b: bool = False,
    lc_min_m: float = DEFAULT_LC_MIN_M,
    max_fragments: int = 1000,
) -> BreakupResult:
    rng = np.random.default_rng(seed)
    r_a = np.asarray(r_a_km, float)
    r_b = np.asarray(r_b_km, float)
    v_a = np.asarray(v_a_kms, float)
    v_b = np.asarray(v_b_kms, float)
    v_rel = float(np.linalg.norm(v_a - v_b))

    m_ref, catastrophic, emr = sbm_reference_mass(m_a_kg, m_b_kg, v_rel)
    lc_max = max(lc_min_m * 1.01, (max(m_a_kg, m_b_kg) / BULK_DENSITY_KG_M3) ** (1.0 / 3.0))
    n_total = int(round(sbm_cumulative_count(m_ref, lc_min_m) - sbm_cumulative_count(m_ref, lc_max)))
    n_total = max(n_total, 0)
    n = min(n_total, int(max_fragments))
    weight = n_total / n if n > 0 else 0.0
    impact = 0.5 * (r_a + r_b)

    lc = sample_lc(rng, n, lc_min_m, lc_max)
    # Parent assignment: catastrophic -> proportional to parent mass;
    # cratering -> fragments come from the target (heavier) body.
    if catastrophic:
        p_a = m_a_kg / (m_a_kg + m_b_kg)
        parent = (rng.random(n) >= p_a).astype(int)
    else:
        parent = np.full(n, 0 if m_a_kg >= m_b_kg else 1, dtype=int)
    rb_flag = np.where(parent == 0, rocket_body_a, rocket_body_b)
    chi = sample_log_am(rng, lc, rb_flag)
    am = 10.0 ** chi
    area = area_from_lc(lc)
    mass = area / am

    dv = (sample_dv_ms(rng, chi) / 1000.0)[:, None] * isotropic_unit(rng, n)

    # Momentum: remove mass-weighted mean dV per parent.
    v_parent = np.where(parent[:, None] == 0, v_a, v_b)
    ref_p = float(np.sum(mass)) * float(np.mean(np.linalg.norm(dv, axis=1))) if n else 1.0
    p_before = float(np.linalg.norm(np.sum(mass[:, None] * dv, axis=0))) / max(ref_p, 1e-12)
    for k in (0, 1):
        sel = parent == k
        if sel.sum() > 1:
            dv[sel] -= np.sum(mass[sel, None] * dv[sel], axis=0) / mass[sel].sum()

    r0 = np.repeat(impact[None, :], n, axis=0)
    dv, n_capped = _cap_to_bound_orbit(r0, v_parent, dv)
    p_after = float(np.linalg.norm(np.sum(mass[:, None] * dv, axis=0))) / max(ref_p, 1e-12)
    v0 = v_parent + dv

    # perigee check
    n_sub = 0
    if n:
        rn = np.linalg.norm(r0, axis=1)
        vn2 = np.einsum("ij,ij->i", v0, v0)
        energy = 0.5 * vn2 - MU_KM3_S2 / rn
        a = -MU_KM3_S2 / (2.0 * energy)
        h = np.linalg.norm(np.cross(r0, v0), axis=1)
        e = np.sqrt(np.maximum(0.0, 1.0 - h * h / (MU_KM3_S2 * a)))
        n_sub = int(np.sum(a * (1.0 - e) - R_EARTH_KM < REENTRY_ALT_KM))

    dv_mag = np.linalg.norm(dv, axis=1) * 1000.0 if n else np.zeros(0)
    stats = {
        "dv_ms_median": float(np.median(dv_mag)) if n else 0.0,
        "dv_ms_p90": float(np.percentile(dv_mag, 90)) if n else 0.0,
        "lc_m_median": float(np.median(lc)) if n else 0.0,
        "am_m2_kg_median": float(np.median(am)) if n else 0.0,
        "relative_velocity_kms": v_rel,
    }
    return BreakupResult(
        catastrophic=catastrophic,
        emr_j_per_g=emr / 1000.0,
        m_ref_kg=m_ref,
        lc_min_m=lc_min_m,
        lc_max_m=float(lc_max),
        n_total=n_total,
        n_sampled=n,
        weight=float(weight),
        seed=int(seed),
        impact_point_km=impact,
        lc_m=lc, am_m2_kg=am, area_m2=area, mass_kg=mass,
        dv_kms=dv, parent_index=parent, r_km=r0, v_kms=v0,
        n_dv_capped=n_capped,
        n_suborbital=n_sub,
        momentum_residual_before=p_before,
        momentum_residual_after=p_after,
        represented_mass_kg=float(np.sum(mass) * weight),
        stats=stats,
    )


# ── Atmosphere (Vallado 2013, Table 8-4) ────────────────────────────────────

_VALLADO_H0 = np.array([0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 180, 200,
                        250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000], float)
_VALLADO_RHO0 = np.array([1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5, 1.905e-5,
                          3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 2.070e-9, 5.464e-10,
                          2.789e-10, 7.248e-11, 2.418e-11, 9.518e-12, 3.725e-12, 1.585e-12, 6.967e-13,
                          1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15], float)
_VALLADO_H = np.array([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 5.877, 7.263,
                       9.473, 12.636, 16.149, 22.523, 29.740, 37.105, 45.546, 53.628, 53.298, 58.515,
                       60.828, 63.822, 71.835, 88.667, 124.64, 181.05, 268.00], float)


def atmospheric_density(alt_km) -> np.ndarray:
    """Exponential density [kg/m^3] (Vallado 2013 Table 8-4). Vectorised."""
    h = np.maximum(np.asarray(alt_km, float), 0.0)
    idx = np.clip(np.searchsorted(_VALLADO_H0, h, side="right") - 1, 0, len(_VALLADO_H0) - 1)
    return _VALLADO_RHO0[idx] * np.exp(-(h - _VALLADO_H0[idx]) / _VALLADO_H[idx])


# ── Vectorised J2 + drag dynamics ──────────────────────────────────────────

def acceleration(r: np.ndarray, v: np.ndarray, bc_m2_kg: np.ndarray | None) -> np.ndarray:
    """Two-body + J2 + drag acceleration (km/s^2). r, v: (n,3). bc = Cd*A/m [m^2/kg]."""
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    r2 = x * x + y * y + z * z
    rn = np.sqrt(r2)
    a = -MU_KM3_S2 * r / (rn ** 3)[:, None]
    f = 1.5 * J2 * MU_KM3_S2 * R_EARTH_KM ** 2 / rn ** 5
    zr = 5.0 * z * z / r2
    a[:, 0] += f * x * (zr - 1.0)
    a[:, 1] += f * y * (zr - 1.0)
    a[:, 2] += f * z * (zr - 3.0)
    if bc_m2_kg is not None:
        # density evaluated no lower than the re-entry altitude: RK4 stage points
        # below it would otherwise see sea-level density and destabilise the step
        rho = atmospheric_density(np.maximum(rn - R_EARTH_KM, REENTRY_ALT_KM))
        vrel = v.copy()
        vrel[:, 0] += OMEGA_EARTH_RAD_S * y
        vrel[:, 1] -= OMEGA_EARTH_RAD_S * x
        vmag = np.linalg.norm(vrel, axis=1)
        # 0.5*rho*BC*|v|v with v in m/s gives m/s^2; km units -> factor 1000
        a -= (0.5 * rho * bc_m2_kg * vmag * 1000.0)[:, None] * vrel
    return a


def propagate(r: np.ndarray, v: np.ndarray, bc_m2_kg: np.ndarray | None, dt_s: float,
              max_step_s: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step RK4 of all objects over dt_s (may be negative)."""
    if r.size == 0 or dt_s == 0:
        return r.copy(), v.copy()
    n_steps = max(1, int(math.ceil(abs(dt_s) / max_step_s)))
    h = dt_s / n_steps
    r = r.copy()
    v = v.copy()
    # Rows that fall below the re-entry altitude are frozen there (callers mark
    # them decayed); integrating them further would pass through the Earth.
    live = np.linalg.norm(r, axis=1) - R_EARTH_KM >= REENTRY_ALT_KM
    for _ in range(n_steps):
        if live.all():
            r, v = rk4_step(r, v, bc_m2_kg, h)
        else:
            idx = np.nonzero(live)[0]
            if idx.size == 0:
                break
            bc = None if bc_m2_kg is None else bc_m2_kg[idx]
            r[idx], v[idx] = rk4_step(r[idx], v[idx], bc, h)
        live &= np.linalg.norm(r, axis=1) - R_EARTH_KM >= REENTRY_ALT_KM
    return r, v


def rk4_step(r, v, bc, h):
    k1v = acceleration(r, v, bc)
    k1r = v
    k2v = acceleration(r + 0.5 * h * k1r, v + 0.5 * h * k1v, bc)
    k2r = v + 0.5 * h * k1v
    k3v = acceleration(r + 0.5 * h * k2r, v + 0.5 * h * k2v, bc)
    k3r = v + 0.5 * h * k2v
    k4v = acceleration(r + h * k3r, v + h * k3v, bc)
    k4r = v + h * k3v
    r_new = r + (h / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
    v_new = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r_new, v_new


def specific_energy_j2(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Conserved energy for two-body + J2 (no drag): v^2/2 - mu/r + J2 potential term."""
    rn = np.linalg.norm(r, axis=1)
    sin2 = (r[:, 2] / rn) ** 2
    u_j2 = (MU_KM3_S2 / rn) * J2 * (R_EARTH_KM / rn) ** 2 * 0.5 * (3.0 * sin2 - 1.0)
    return 0.5 * np.einsum("ij,ij->i", v, v) - MU_KM3_S2 / rn + u_j2
