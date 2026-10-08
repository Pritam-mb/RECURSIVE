"""
Closed-form orbital-mechanics checks beside the numerical propagation.

Standard formulas (Vallado, "Fundamentals of Astrodynamics and Applications",
4th ed.: vis-viva §2.3, Hohmann §6.3, Clohessy-Wiltshire / Hill §6.8; Curtis,
"Orbital Mechanics for Engineering Students", ch. 6-7):

  vis-viva            v² = μ (2/r − 1/a)          ε = v²/2 − μ/r = −μ/(2a)
  mean motion         n  = sqrt(μ / a³)           T = 2π / n
  Hohmann (r1 → r2)   Δv1 = sqrt(μ/r1) (sqrt(2 r2/(r1+r2)) − 1)
                      Δv2 = sqrt(μ/r2) (1 − sqrt(2 r1/(r1+r2)))
  Clohessy-Wiltshire  relative motion about a circular reference orbit, Hill
  frame x radial, y along-track, z cross-track; for an impulse (ẋ0, ẏ0, ż0)
  applied at x0 = y0 = z0 = 0 the exact linear solution is
        x(t) = (ẋ0/n) sin nt + (2ẏ0/n)(1 − cos nt)
        y(t) = (2ẋ0/n)(cos nt − 1) + (ẏ0/n)(4 sin nt − 3 n t)
        z(t) = (ż0/n) sin nt
  The −3 ẏ0 t term is the secular along-track drift a prograde burn creates
  (higher orbit → longer period → falls behind).

The manoeuvre planner attaches cw_along_track_check() to every option, and the
validation helpers at the bottom (check_* / validation_checks) each return a
dict in the /api/physics/validation schema:
  {name, standard_formula, our_value, reference_value, abs_error, rel_error,
   pass, tolerance, source}
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MU_KM3_S2 = 398600.4418
RE_KM = 6378.137


# ── Standard formulas ───────────────────────────────────────────────────────

def vis_viva_speed(r_km: float, a_km: float, mu: float = MU_KM3_S2) -> float:
    return math.sqrt(mu * (2.0 / r_km - 1.0 / a_km))


def semi_major_axis(r_vec, v_vec, mu: float = MU_KM3_S2) -> float:
    """a from vis-viva: 1/a = 2/r − v²/μ."""
    r = float(np.linalg.norm(r_vec)); v = float(np.linalg.norm(v_vec))
    return 1.0 / (2.0 / r - v * v / mu)


def specific_energy(r_vec, v_vec, mu: float = MU_KM3_S2) -> float:
    r = float(np.linalg.norm(r_vec)); v = float(np.linalg.norm(v_vec))
    return 0.5 * v * v - mu / r


def mean_motion(a_km: float, mu: float = MU_KM3_S2) -> float:
    return math.sqrt(mu / a_km ** 3)


def orbital_period_s(a_km: float, mu: float = MU_KM3_S2) -> float:
    return 2.0 * math.pi / mean_motion(a_km, mu)


def hohmann(r1_km: float, r2_km: float, mu: float = MU_KM3_S2) -> dict[str, float]:
    """Hohmann transfer between circular orbits: Δv1, Δv2 (km/s), total and transfer time."""
    at = 0.5 * (r1_km + r2_km)
    dv1 = math.sqrt(mu / r1_km) * (math.sqrt(2.0 * r2_km / (r1_km + r2_km)) - 1.0)
    dv2 = math.sqrt(mu / r2_km) * (1.0 - math.sqrt(2.0 * r1_km / (r1_km + r2_km)))
    return {"dv1_kms": dv1, "dv2_kms": dv2, "total_kms": abs(dv1) + abs(dv2),
            "transfer_time_s": math.pi * math.sqrt(at ** 3 / mu)}


def cw_position(t_s, n: float, dv_rsw_kms) -> np.ndarray:
    """Exact CW solution for an impulse at the origin; returns (..., 3) [x_R, y_S, z_W] km."""
    t = np.asarray(t_s, float)
    xd, yd, zd = (float(c) for c in dv_rsw_kms)
    nt = n * t
    x = (xd / n) * np.sin(nt) + (2.0 * yd / n) * (1.0 - np.cos(nt))
    y = (2.0 * xd / n) * (np.cos(nt) - 1.0) + (yd / n) * (4.0 * np.sin(nt) - 3.0 * nt)
    z = (zd / n) * np.sin(nt)
    return np.stack([x, y, z], axis=-1)


def cw_along_track_shift_km(t_s: float, n: float, dv_rsw_ms) -> float:
    dv = np.asarray(dv_rsw_ms, float) / 1000.0
    return float(cw_position(t_s, n, dv)[1])


def cw_along_track_check(t_s: float, n: float, dv_rsw_ms, numeric_km: float) -> dict[str, Any]:
    """Option-level check: CW along-track shift vs the numerical (SGP4+J2 deviation) shift.

    rel_error is normalised by max(|numeric|, 10 m) so that W (cross-track)
    burns, whose along-track shift is ~0 in both models, do not divide by zero."""
    cw = cw_along_track_shift_km(t_s, n, dv_rsw_ms)
    err = abs(cw - numeric_km)
    return {"method": "Clohessy-Wiltshire",
            "along_track_shift_km_at_tca": round(cw, 5),
            "numeric_km": round(float(numeric_km), 5),
            "abs_error_km": round(err, 5),
            "rel_error": round(err / max(abs(numeric_km), 0.01), 5),
            "t_since_burn_s": round(float(t_s), 1),
            "mean_motion_rad_s": n,
            "formula": "y(t) = (2·ẋ0/n)(cos nt − 1) + (ẏ0/n)(4 sin nt − 3nt)"}


# ── Numerical reference integrators (two-body, optionally J2) ───────────────

def _accel(r: np.ndarray, j2: bool) -> np.ndarray:
    if j2:
        from app.core.screening import _j2_accel
        return _j2_accel(r)
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    return -MU_KM3_S2 * r / rn ** 3


def propagate(r0, v0, dt_s: float, step_s: float = 10.0, j2: bool = False):
    r = np.asarray(r0, float).copy(); v = np.asarray(v0, float).copy()
    n = max(1, int(math.ceil(abs(dt_s) / step_s)))
    h = dt_s / n
    for _ in range(n):
        a1 = _accel(r, j2)
        r2 = r + 0.5 * h * v; v2 = v + 0.5 * h * a1; a2 = _accel(r2, j2)
        r3 = r + 0.5 * h * v2; v3 = v + 0.5 * h * a2; a3 = _accel(r3, j2)
        r4 = r + h * v3; v4 = v + h * a3; a4 = _accel(r4, j2)
        r = r + (h / 6.0) * (v + 2 * v2 + 2 * v3 + v4)
        v = v + (h / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
    return r, v


def _circular_state(alt_km: float, inc_deg: float = 51.6):
    r = RE_KM + alt_km
    vc = math.sqrt(MU_KM3_S2 / r)
    i = math.radians(inc_deg)
    return np.array([r, 0.0, 0.0]), vc * np.array([0.0, math.cos(i), math.sin(i)])


def numeric_along_track_shift_km(r0, v0, dv_rsw_ms, t_s: float, *, j2: bool = False, step_s: float = 10.0) -> float:
    """Propagate nominal and burned states; project the deviation on the nominal along-track axis."""
    r0 = np.asarray(r0, float); v0 = np.asarray(v0, float)
    Q = _rsw(r0, v0)
    dv = Q @ (np.asarray(dv_rsw_ms, float) / 1000.0)
    both_r, both_v = propagate(np.stack([r0, r0]), np.stack([v0, v0 + dv]), t_s, step_s, j2)
    Qt = _rsw(both_r[0], both_v[0])
    return float((both_r[1] - both_r[0]) @ Qt[:, 1])


def _rsw(r, v) -> np.ndarray:
    r_hat = r / np.linalg.norm(r)
    w = np.cross(r, v); w_hat = w / np.linalg.norm(w)
    return np.column_stack([r_hat, np.cross(w_hat, r_hat), w_hat])


# ── Validation checks (schema of /api/physics/validation) ───────────────────

def _check(name, formula, ours, ref, tol, source, rel=True) -> dict[str, Any]:
    abs_err = abs(float(ours) - float(ref))
    rel_err = abs_err / abs(float(ref)) if float(ref) != 0.0 else None   # undefined for a zero reference
    return {"name": name, "standard_formula": formula, "our_value": float(ours), "reference_value": float(ref),
            "abs_error": abs_err, "rel_error": rel_err,
            "pass": bool((rel_err if rel and rel_err is not None else abs_err) <= tol), "tolerance": tol,
            "tolerance_kind": "relative" if rel else "absolute", "source": source}


def check_cw_vs_numeric(alt_km: float = 500.0, dv_t_ms: float = 0.1, t_s: float = 3 * 3600.0,
                        j2: bool = True) -> dict[str, Any]:
    """CW along-track drift for an along-track impulse vs numerical (two-body+J2 RK4) propagation."""
    r0, v0 = _circular_state(alt_km)
    n = mean_motion(float(np.linalg.norm(r0)))
    cw = cw_along_track_shift_km(t_s, n, [0.0, dv_t_ms, 0.0])
    num = numeric_along_track_shift_km(r0, v0, [0.0, dv_t_ms, 0.0], t_s, j2=j2)
    out = _check(f"CW along-track drift vs numeric ({dv_t_ms} m/s +S, {t_s / 3600:.0f} h, {alt_km:.0f} km)",
                 "y(t) = (ẏ0/n)(4 sin nt − 3nt)", num, cw, 0.03,
                 "Clohessy & Wiltshire 1960, J. Aerospace Sci. 27(9); Vallado §6.8")
    out["units"] = "km"
    return out


def check_hohmann_vs_numeric(alt1_km: float = 500.0, alt2_km: float = 520.0) -> dict[str, Any]:
    """Find numerically (two-body RK4 + secant) the tangential Δv whose propagated apoapsis
    (half a transfer later) reaches r2, and compare it with the Hohmann Δv1."""
    r0, v0 = _circular_state(alt1_km)
    r1, r2 = RE_KM + alt1_km, RE_KM + alt2_km
    h = hohmann(r1, r2)
    t_half = h["transfer_time_s"]
    vhat = v0 / np.linalg.norm(v0)

    def radius_after(dv):
        r, _ = propagate(r0, v0 + dv * vhat, t_half, step_s=10.0)
        return float(np.linalg.norm(r)) - r2

    x0, x1 = 0.0, 2.0 * h["dv1_kms"]
    f0, f1 = radius_after(x0), radius_after(x1)
    for _ in range(8):
        if abs(f1 - f0) < 1e-12:
            break
        x0, x1, f0 = x1, x1 - f1 * (x1 - x0) / (f1 - f0), f1
        f1 = radius_after(x1)
        if abs(f1) < 1e-6:
            break
    out = _check(f"Hohmann Δv1 {alt1_km:.0f}→{alt2_km:.0f} km vs numerical altitude raise",
                 "Δv1 = sqrt(μ/r1)(sqrt(2r2/(r1+r2)) − 1)", x1 * 1000.0, h["dv1_kms"] * 1000.0, 1e-3,
                 "Hohmann 1925; Vallado §6.3")
    out["units"] = "m/s"
    return out


def check_vis_viva_energy(alt_km: float = 700.0, ecc: float = 0.01, orbits: float = 3.0) -> dict[str, Any]:
    """Specific orbital energy ε = v²/2 − μ/r must be conserved by the two-body integrator
    and equal −μ/(2a) (vis-viva)."""
    rp = RE_KM + alt_km
    a = rp / (1.0 - ecc)
    vp = vis_viva_speed(rp, a)
    r0 = np.array([rp, 0.0, 0.0]); v0 = np.array([0.0, vp, 0.0])
    r, v = propagate(r0, v0, orbits * orbital_period_s(a), step_s=10.0)
    eps_num = specific_energy(r, v)
    out = _check(f"Vis-viva energy conservation ({orbits:g} orbits, e={ecc})",
                 "ε = v²/2 − μ/r = −μ/(2a)", eps_num, -MU_KM3_S2 / (2.0 * a), 1e-8,
                 "vis-viva equation; Vallado §2.3")
    out["units"] = "km²/s²"
    return out


def check_rocket_equation(m0_kg: float = 500.0, delta_v_ms: float = 1.0, isp_s: float = 220.0,
                          thrust_n: float = 1.0) -> dict[str, Any]:
    """Tsiolkovsky propellant mass vs RK4 integration of dm/dt = −F/(Isp g0), dv/dt = F/m."""
    from app.core.propulsion import integrate_constant_thrust, tsiolkovsky_propellant_kg

    num = integrate_constant_thrust(m0_kg, delta_v_ms, isp_s, thrust_n)
    out = _check(f"Rocket equation vs numeric mass integration ({delta_v_ms} m/s, Isp {isp_s:g} s)",
                 "m_p = m0 (1 − exp(−Δv/(Isp·g0)))", num["prop_mass_kg"],
                 tsiolkovsky_propellant_kg(m0_kg, delta_v_ms, isp_s), 1e-6,
                 "Tsiolkovsky 1903; Sutton & Biblarz ch. 4")
    out["units"] = "kg"
    return out


def validation_checks() -> list[dict[str, Any]]:
    """All AV-owned standard-formula checks (≈0.3 s)."""
    return [check_cw_vs_numeric(), check_hohmann_vs_numeric(), check_vis_viva_energy(), check_rocket_equation()]
