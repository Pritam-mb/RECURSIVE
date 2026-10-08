"""
/api/physics — live cross-validation of the physics engine against standard
closed-form results, plus the propulsion engine catalogue.

GET /api/physics/validation
    Every check is computed live (cached 60 s) and returned as
    {name, category, standard_formula, our_value, reference_value, abs_error,
     rel_error, pass, tolerance, tolerance_kind, units, source}.
    Covered:
      * Pc: Foster 2-D vs Chan series vs Monte Carlo on 3 encounter geometries,
        Foster vs the exact Rice / non-central chi-square CDF (isotropic case),
        Alfano max-Pc closed form vs numerical maximisation, Alfano >= Foster;
      * propagation: specific-energy (incl. J2 potential) conservation of the
        production J2-RK4 propagator (app.core.screening.propagate_j2),
        vis-viva semi-major axis of an SGP4 arc vs SGP4's mean elements,
        SGP4 vs J2-RK4 short-arc agreement;
      * manoeuvres: Clohessy-Wiltshire drift vs numerical burn, Hohmann Δv vs
        numerical altitude raise, Tsiolkovsky vs numerical mass integration
        (app.core.analytic_checks from the avoidance agent when present, local
        implementations otherwise);
      * breakup: NASA SBM fragment count (app.core.breakup) vs the formula,
        sampled size distribution vs the power law.
GET /api/physics/engines -> app.core.propulsion.ENGINES (503 if unavailable).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/physics", tags=["physics"])

CACHE_TTL_S = 60.0
_cache: dict[str, Any] = {"t": 0.0, "payload": None}
_lock = threading.Lock()

# Public ISS TLE (CelesTrak format example, epoch 2008-09-20) used as a fixed,
# reproducible SGP4 sample.
_TLE_L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
_TLE_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# Encounter geometries (B-plane miss vector km, 1-sigma major/minor km, rotation deg, HBR km)
GEOMETRIES = [
    {"name": "G1 isotropic", "b": (0.15, 0.0), "sig": (0.2, 0.2), "rot_deg": 0.0, "hbr": 0.02},
    {"name": "G2 anisotropic 5:1", "b": (0.2, 0.05), "sig": (0.5, 0.1), "rot_deg": 0.0, "hbr": 0.015},
    {"name": "G3 rotated 3.3:1, large HBR", "b": (0.3, -0.2), "sig": (1.0, 0.3), "rot_deg": 30.0, "hbr": 0.05},
]


def _check(name, category, formula, ours, ref, tol, source, *, rel=True, units="", detail=None) -> dict:
    ours, ref = float(ours), float(ref)
    abs_err = abs(ours - ref)
    rel_err = abs_err / max(abs(ref), 1e-300)
    out = {"name": name, "category": category, "standard_formula": formula, "our_value": ours,
           "reference_value": ref, "abs_error": abs_err, "rel_error": rel_err,
           "pass": bool((rel_err if rel else abs_err) <= tol), "tolerance": tol,
           "tolerance_kind": "relative" if rel else "absolute", "units": units, "source": source}
    if detail:
        out["detail"] = detail
    return out


def _cov(g) -> np.ndarray:
    th = math.radians(g["rot_deg"])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return R @ np.diag(np.square(g["sig"])) @ R.T


# ══════════════════════════════════════════════════════════════════════════════
# Pc checks
# ══════════════════════════════════════════════════════════════════════════════

def _pc_checks() -> list[dict]:
    from app.core.pc_methods import (alfano_max_pc, alfano_max_pc_closed_form, chan_pc,
                                     monte_carlo_pc)
    from app.core.screening import foster_pc

    out = []
    for k, g in enumerate(GEOMETRIES):
        b, C, h = np.array(g["b"]), _cov(g), g["hbr"]
        pf = foster_pc(b, C, h)
        pch = chan_pc(b, C, h)
        out.append(_check(f"Pc Foster vs Chan series — {g['name']}", "pc",
                          "Pc = e^{-v/2} Σ (v/2)^m/m! [1 − e^{-u/2} Σ_{k≤m} (u/2)^k/k!], u=R²/σxσy, v=x²/σx²+y²/σy²",
                          pf, pch, 0.02, "Chan (2008) Spacecraft Collision Probability ch. 5; Foster & Estes (1992)"))
        mc = monte_carlo_pc(b, C, h, seed=1000 + k, target_hits=10**9)  # always full N for a fixed budget
        tol = 4.0 * (mc["stderr"] or 0.0)
        out.append(_check(f"Pc Foster vs Monte Carlo (N={mc['samples']}) — {g['name']}", "pc",
                          "Pc ≈ (1/N) Σ 1[|x_i| ≤ HBR], x_i ~ N(b, C); tolerance 4σ_MC = 4 sqrt(p(1−p)/N)",
                          pf, mc["pc"] if mc["pc"] is not None else 0.0, tol,
                          "direct B-plane Monte Carlo (seeded numpy)", rel=False,
                          detail={"mc_hits": mc["hits"], "mc_stderr": mc["stderr"]}))
    # Isotropic case has an exact closed form: Rice / non-central chi-square CDF
    g = GEOMETRIES[0]
    s = g["sig"][0]
    b, C, h = np.array(g["b"]), _cov(g), g["hbr"]
    try:
        from scipy.special import chndtr  # non-central chi-square CDF
        exact = float(chndtr((h / s) ** 2, 2, float(b @ b) / s ** 2))
        out.append(_check("Pc Foster vs exact Rice distribution (isotropic)", "pc",
                          "Pc = F_χ'²(R²/σ²; k=2, λ=|b|²/σ²)", foster_pc(b, C, h), exact, 1e-6,
                          "Rice (1945) / non-central chi-square CDF (scipy.special.chndtr)"))
    except Exception:
        pass
    pmax_num = alfano_max_pc(b, C, h)
    out.append(_check("Alfano max-Pc: closed form vs numerical covariance-scale maximisation", "pc",
                      "Pc_max = HBR² / (e · σx σy · bᵀC⁻¹b)", pmax_num,
                      alfano_max_pc_closed_form(b, C, h), 0.01,
                      "Alfano (2005) J. Astronaut. Sci. 53(2)"))
    g3 = GEOMETRIES[2]
    b3, C3, h3 = np.array(g3["b"]), _cov(g3), g3["hbr"]
    pf3, pm3 = foster_pc(b3, C3, h3), alfano_max_pc(b3, C3, h3)
    chk = _check("Alfano max-Pc is an upper bound of Foster Pc — G3", "pc", "Pc_max ≥ Pc(k=1)",
                 pm3, pf3, 0.0, "Alfano (2005)")
    chk["pass"] = bool(pm3 >= pf3)
    chk["tolerance_kind"] = "inequality (our_value ≥ reference_value)"
    out.append(chk)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Propagation checks
# ══════════════════════════════════════════════════════════════════════════════

def _propagation_checks() -> list[dict]:
    from sgp4.api import WGS72, Satrec

    from app.core.screening import J2, MU_KM3_S2 as MU, RE_KM, propagate_j2

    sat = Satrec.twoline2rv(_TLE_L1, _TLE_L2, WGS72)
    jd, fr = sat.jdsatepoch, sat.jdsatepochF
    _, r0, v0 = sat.sgp4(jd, fr)
    r0, v0 = np.array(r0), np.array(v0)
    out = []

    def energy(r, v):
        rn = float(np.linalg.norm(r))
        z = r[2]
        return float(v @ v) / 2 - MU / rn + MU * J2 * RE_KM ** 2 / rn ** 3 * (3 * z * z / rn ** 2 - 1) / 2

    T = 3 * 3600.0
    r1, v1 = propagate_j2(r0, v0, T)
    out.append(_check("Energy conservation of production J2-RK4 propagator (3 h, ISS orbit)", "propagation",
                      "ε = v²/2 − μ/r + (μJ2Re²/2r³)(3z²/r² − 1) = const",
                      energy(r1, v1), energy(r0, v0), 1e-8,
                      "conservative two-body + J2 potential (Vallado §9.6); app.core.screening.propagate_j2",
                      units="km²/s²"))

    # vis-viva semi-major axis averaged over one SGP4 revolution vs SGP4's mean a
    P = 2 * math.pi / sat.no_kozai * 60.0
    ts = np.linspace(0.0, P, 361)
    _, rr, vv = sat.sgp4_array(np.full(ts.size, jd), fr + ts / 86400.0)
    rr, vv = np.asarray(rr), np.asarray(vv)
    a_osc = 1.0 / (2.0 / np.linalg.norm(rr, axis=1) - np.sum(vv * vv, axis=1) / sat.mu)
    out.append(_check("Vis-viva semi-major axis of SGP4 arc vs SGP4 mean elements", "propagation",
                      "a = 1 / (2/r − v²/μ), averaged over one revolution",
                      float(a_osc.mean()), sat.a * sat.radiusearthkm, 1e-3,
                      "vis-viva (Vallado §2.3); SGP4 (Vallado et al. 2006, AIAA 2006-6753)", units="km"))

    dt = 1800.0
    _, rs, _ = sat.sgp4(jd, fr + dt / 86400.0)
    rj, _ = propagate_j2(r0, v0, dt)
    out.append(_check("SGP4 vs J2-RK4 short-arc agreement (30 min)", "propagation",
                      "|r_SGP4(t) − r_J2RK4(t)| from the same initial state", float(np.linalg.norm(rj - np.array(rs))),
                      0.0, 1.0, "SGP4 (Hoots & Roehrich 1980, STR#3) vs two-body+J2 numerical integration",
                      rel=False, units="km"))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Manoeuvre checks (CW / Hohmann / Tsiolkovsky)
# ══════════════════════════════════════════════════════════════════════════════

def _local_manoeuvre_checks() -> list[dict]:
    """Fallback used only when app.core.analytic_checks is unavailable."""
    from app.core.screening import propagate_j2

    MU = 398600.4418
    G0 = 9.80665
    out = []
    # Clohessy-Wiltshire along-track drift
    rr = 6378.137 + 500.0
    r0 = np.array([rr, 0.0, 0.0]); v0 = np.array([0.0, math.sqrt(MU / rr), 0.0])
    n = math.sqrt(MU / rr ** 3); dv = 1e-4; t = 3 * 3600.0
    cw = (dv / n) * (4 * math.sin(n * t) - 3 * n * t)
    ra, _ = propagate_j2(r0, v0, t); rb, _ = propagate_j2(r0, v0 + np.array([0.0, dv, 0.0]), t)
    vref = propagate_j2(r0, v0, t)[1]
    num = float((rb - ra) @ (vref / np.linalg.norm(vref)))
    out.append(_check("CW along-track drift vs numeric (0.1 m/s, 3 h, 500 km)", "manoeuvre",
                      "y(t) = (ẏ0/n)(4 sin nt − 3nt)", num, cw, 0.03, "Clohessy & Wiltshire 1960", units="km"))
    # Hohmann: two-body RK4 apoapsis after Δv1
    r1, r2 = 6378.137 + 500.0, 6378.137 + 520.0
    dv1 = math.sqrt(MU / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1)
    at = 0.5 * (r1 + r2); th = math.pi * math.sqrt(at ** 3 / MU)
    r, v = np.array([r1, 0.0, 0.0]), np.array([0.0, math.sqrt(MU / r1) + dv1, 0.0])
    N = int(th / 5.0) + 1; h = th / N
    acc = lambda x: -MU * x / np.linalg.norm(x) ** 3  # noqa: E731
    for _ in range(N):
        k1v = acc(r); k1r = v
        k2v = acc(r + 0.5 * h * k1r); k2r = v + 0.5 * h * k1v
        k3v = acc(r + 0.5 * h * k2r); k3r = v + 0.5 * h * k2v
        k4v = acc(r + h * k3r); k4r = v + h * k3v
        r = r + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r); v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    out.append(_check("Hohmann Δv1 500→520 km: numerical apoapsis radius", "manoeuvre",
                      "Δv1 = sqrt(μ/r1)(sqrt(2r2/(r1+r2)) − 1)", float(np.linalg.norm(r)), r2, 1e-5,
                      "Hohmann 1925; Vallado §6.3", units="km"))
    # Tsiolkovsky vs RK4 mass integration
    m0, isp, F, target = 500.0, 220.0, 1.0, 1.0
    mdot = F / (isp * G0); m = m0; vel = 0.0; hs = 0.05
    while vel < target:
        dvs = hs / 6 * (F / m + 4 * F / (m - 0.5 * hs * mdot) + F / (m - hs * mdot))
        if vel + dvs >= target:
            m -= (target - vel) / dvs * hs * mdot; break
        vel += dvs; m -= hs * mdot
    out.append(_check("Rocket equation vs numeric mass integration (1 m/s, Isp 220 s)", "manoeuvre",
                      "m_p = m0 (1 − exp(−Δv/(Isp·g0)))", m0 - m, m0 * (1 - math.exp(-target / (isp * G0))),
                      1e-5, "Tsiolkovsky 1903", units="kg"))
    return out


def _manoeuvre_checks() -> list[dict]:
    try:
        from app.core.analytic_checks import validation_checks
        checks = [dict(c, category=c.get("category", "manoeuvre"), provider="app.core.analytic_checks")
                  for c in validation_checks()]
        if checks:
            return checks
    except Exception:
        pass
    return _local_manoeuvre_checks()


# ══════════════════════════════════════════════════════════════════════════════
# Breakup (NASA SBM)
# ══════════════════════════════════════════════════════════════════════════════

def _breakup_checks() -> list[dict]:
    from app.core import breakup as bk

    m_a, m_b, vrel = 800.0, 15.0, 10.0
    r = np.array([7000.0, 0.0, 0.0])
    va = np.array([0.0, 7.546, 0.0])
    vb = va + np.array([0.0, 0.0, vrel])
    res = bk.simulate_breakup(r, va, m_a, r, vb, m_b, seed=12345, max_fragments=1000)
    # independent evaluation of the SBM formulas
    emr = 0.5 * m_b * (vrel * 1e3) ** 2 / m_a
    M = (m_a + m_b) if emr >= 40_000.0 else m_b * vrel ** 2
    lc_min = bk.DEFAULT_LC_MIN_M
    lc_max = max(lc_min * 1.01, (m_a / bk.BULK_DENSITY_KG_M3) ** (1 / 3))
    n_formula = 0.1 * M ** 0.75 * (lc_min ** -1.71 - lc_max ** -1.71)
    out = [_check(f"NASA SBM fragment count ≥ {lc_min * 100:.0f} cm ({m_a:.0f} kg + {m_b:.0f} kg @ {vrel:.0f} km/s)",
                  "breakup", "N(≥Lc) = 0.1 M^0.75 Lc^-1.71; catastrophic if EMR ≥ 40 J/g → M = m1+m2",
                  res.n_total, n_formula, 1.0, "Johnson et al. (2001) Adv. Space Res. 28(9); Krisko (2011) ODQN 15(4)",
                  rel=False, units="fragments", detail={"catastrophic": res.catastrophic, "emr_j_per_kg": emr})]
    lc = np.asarray(getattr(res, "lc_m", []), float)
    if lc.size:
        cut = 0.5
        p_theory = (cut ** -1.71 - lc_max ** -1.71) / (lc_min ** -1.71 - lc_max ** -1.71)
        p_obs = float(np.mean(lc >= cut))
        sig = math.sqrt(p_theory * (1 - p_theory) / lc.size)
        out.append(_check(f"SBM sampled size distribution: fraction ≥ {cut} m (n={lc.size})", "breakup",
                          "P(Lc ≥ x) = (x^-1.71 − Lc_max^-1.71)/(Lc_min^-1.71 − Lc_max^-1.71)",
                          p_obs, p_theory, 4 * sig, "Johnson et al. (2001) power law; tolerance 4σ binomial",
                          rel=False))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

def compute_validation() -> dict:
    t0 = time.perf_counter()
    checks: list[dict] = []
    errors: list[str] = []
    for fn in (_pc_checks, _propagation_checks, _manoeuvre_checks, _breakup_checks):
        try:
            checks.extend(fn())
        except Exception as exc:  # report, never hide
            errors.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
    passed = sum(1 for c in checks if c["pass"])
    return {
        "checks": checks,
        "summary": {"total": len(checks), "passed": passed, "failed": len(checks) - passed,
                    "all_pass": passed == len(checks) and not errors},
        "errors": errors,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "computed_at_unix": time.time(),
        "cache_ttl_s": CACHE_TTL_S,
    }


@router.get("/validation")
def physics_validation() -> dict:
    with _lock:
        now = time.monotonic()
        if _cache["payload"] is None or now - _cache["t"] > CACHE_TTL_S:
            _cache["payload"] = compute_validation()
            _cache["t"] = now
            return {**_cache["payload"], "cached": False}
        return {**_cache["payload"], "cached": True}


@router.get("/engines")
def physics_engines() -> dict:
    try:
        from app.core.propulsion import ENGINES
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"propulsion catalogue unavailable: {exc}")
    return {"engines": ENGINES, "count": len(ENGINES), "source": "app.core.propulsion.ENGINES"}
