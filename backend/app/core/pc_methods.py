"""
Standard alternative probability-of-collision (Pc) formulations used to
cross-check the authoritative Foster 2-D Pc (app.core.screening.foster_pc).

All methods work in the encounter B-plane (short-encounter / rectilinear
assumption): the relative position at TCA is a bivariate normal
``x ~ N(b, C)`` with ``b = (b_t, b_n)`` the miss vector and ``C`` the combined
2x2 position covariance; a collision occurs if ``|x| <= HBR`` (combined
hard-body radius).

    Foster (1992)     Pc = (1 / 2 pi sqrt|C|) * Int_{|x|<=HBR} exp(-1/2 (x-b)^T C^-1 (x-b)) dx
                      (numerical quadrature, app.core.screening.foster_pc)

Methods implemented here
------------------------
* ``chan_pc``        Chan's equivalent-area series (Chan 1997, "Collision
  probability analyses for Earth-orbiting satellites", Adv. Astronaut. Sci. 96;
  Chan 2008, *Spacecraft Collision Probability*, Aerospace Press, ch. 5).
  With principal-axis sigmas (sx, sy) and miss components (xm, ym) in those
  axes, the elliptic problem is replaced by an isotropic one of equal area:

      u = HBR^2 / (sx * sy),        v = xm^2 / sx^2 + ym^2 / sy^2
      Pc = e^{-v/2} * Sum_{m>=0} (v/2)^m / m!  *  [1 - e^{-u/2} Sum_{k=0}^{m} (u/2)^k / k!]

  i.e. a Poisson(v/2)-weighted sum of regularised lower incomplete gamma
  functions P(m+1, u/2) (it is the non-central chi-square CDF with 2 d.o.f.,
  exact for an isotropic covariance and accurate when HBR << sigma_min).

* ``alfano_max_pc``  Maximum Pc over an unknown covariance *scale* (Alfano
  2005, "Relating position uncertainty to maximum conjunction probability",
  J. Astronaut. Sci. 53(2), 193-205).  The covariance shape (aspect ratio and
  orientation) is kept, its size k^2 C is varied.  In the small-HBR limit
  Pc(k) ~ HBR^2 / (2 k^2 sx sy) * exp(-q / 2k^2) with q = b^T C^-1 b, which is
  maximised at k^2 = q / 2:

      Pc_max ~= HBR^2 / (e * sx * sy * q)          (closed form, Alfano 2005)

  We refine that with the exact Foster integral (golden-section search on
  log k around k*), and return max(Pc_max, Pc(k=1)), so it is a guaranteed
  upper bound of the reported Foster Pc for this geometry.  If the miss vector
  lies inside the hard-body disk the bound is 1 (k -> 0).

* ``monte_carlo_pc`` Direct sampling of ``x ~ N(b, C)`` (vectorised numpy,
  seeded generator, adaptive N up to 2e5) and counting hits ``|x| <= HBR``.
  Standard error sqrt(p(1-p)/N).  Returns None when fewer than ``min_hits``
  hits are observed at N_max: Pc below ~min_hits/N_max cannot be resolved by
  plain sampling and we do not pretend otherwise.

``pc_checks(alert)`` reconstructs (b, C, HBR) from the published alert fields
(b_t_km, b_n_km, covariance_ellipse a/b/angle (+ sigma_level), hbr_km) and
returns the CONTRACT2 ``pc_checks`` block.
"""

from __future__ import annotations

import math
import zlib
from typing import Any

import numpy as np

from app.core.screening import foster_pc

try:  # vectorised log-Gamma; pure-python fallback keeps the module dependency-free
    from scipy.special import gammaln as _gammaln
except Exception:  # pragma: no cover
    def _gammaln(x):
        return np.array([math.lgamma(float(v)) for v in np.ravel(x)]).reshape(np.shape(x))

# log10 floor for comparing methods: below 1e-20 every method is "operationally
# zero" and decade differences between them carry no information.
LOG_FLOOR_PC = 1e-20
CONSISTENT_SPREAD_DECADES = 0.5
MC_MAX_SAMPLES = 200_000
MC_BATCH = 20_000
MC_MIN_HITS = 10
MC_TARGET_HITS = 400          # stop early when relative stderr <= 5 %

_ALFANO_REF = "Alfano (2005) J. Astronaut. Sci. 53(2):193-205"
_CHAN_REF = "Chan (2008) Spacecraft Collision Probability, ch. 5"
_FOSTER_REF = "Foster & Estes (1992) NASA JSC-25898"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _principal(cov_2d_km2) -> tuple[float, float, np.ndarray]:
    """Return (sx, sy, U): principal 1-sigma (major, minor) and eigenvector matrix."""
    C = np.asarray(cov_2d_km2, float)
    w, U = np.linalg.eigh(0.5 * (C + C.T))
    w = np.clip(w, 0.0, None)
    # eigh -> ascending; put major axis first
    return math.sqrt(w[1]), math.sqrt(w[0]), U[:, ::-1]


def cov_from_ellipse(a: float, b: float, angle_rad: float, sigma_level: float = 1.0,
                     units: str = "m") -> np.ndarray:
    """2x2 B-plane covariance (km^2) from a published ellipse.

    ``a``/``b`` are the semi-major/minor axes at ``sigma_level`` sigma; ``angle``
    is the major-axis direction in the (b_t, b_n) frame (same convention as
    app.core.screening.compute_pc: atan2(U[1,1], U[0,1]) of the major eigvector).
    """
    scale = (1e-3 if units == "m" else 1.0) / max(float(sigma_level), 1e-12)
    s1, s2 = float(a) * scale, float(b) * scale
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag([s1 * s1, s2 * s2]) @ R.T


def bplane_inputs(alert: dict) -> tuple[np.ndarray, np.ndarray, float] | None:
    """(b_vec_km, cov_2d_km2, hbr_km) from alert fields, or None if unavailable."""
    try:
        bt = alert.get("b_t_km", alert.get("bt_km"))
        bn = alert.get("b_n_km", alert.get("bn_km"))
        hbr = alert.get("hbr_km")
        ell = alert.get("covariance_ellipse") or {}
        if bt is None or bn is None or not hbr or ell.get("a") is None or ell.get("b") is None:
            return None
        if alert.get("cov_2d_km2") is not None:
            C = np.asarray(alert["cov_2d_km2"], float)
        else:
            # screening ellipses carry sigma_level=3; debris ellipses are 1-sigma.
            C = cov_from_ellipse(ell["a"], ell["b"], float(ell.get("angle") or 0.0),
                                 float(ell.get("sigma_level", 1.0)), ell.get("units", "m"))
        if not np.all(np.isfinite(C)) or np.linalg.det(C) <= 0:
            return None
        return np.array([float(bt), float(bn)]), C, float(hbr)
    except Exception:
        return None


def _log10(p: float | None) -> float | None:
    if p is None:
        return None
    return math.log10(max(float(p), LOG_FLOOR_PC))


# ══════════════════════════════════════════════════════════════════════════════
# Chan series
# ══════════════════════════════════════════════════════════════════════════════

def _log_reg_lower_gamma(a: np.ndarray, x: float) -> np.ndarray:
    """log P(a, x) for integer a >= 1 (vector) and scalar x >= 0.

    P(a, x) = 1 - e^{-x} Sum_{k<a} x^k/k!.  For x < a + 1 we use the series
    P = x^a e^{-x} / Gamma(a+1) * Sum_j x^j / ((a+1)...(a+j)) which avoids the
    catastrophic cancellation of "1 - ..." when u = HBR^2/(sx sy) is tiny.
    """
    a = np.asarray(a, float)
    out = np.empty_like(a)
    if x <= 0:
        out.fill(-np.inf)
        return out
    lx = math.log(x)
    small = a > x - 1.0
    if np.any(small):
        aa = a[small]
        term = np.ones_like(aa)
        s = np.ones_like(aa)
        for j in range(1, 4000):
            term = term * x / (aa + j)
            s = s + term
            if np.all(term < 1e-17 * s):
                break
        out[small] = aa * lx - x - _gammaln(aa + 1.0) + np.log(s)
    big = ~small
    if np.any(big):
        # Q = e^{-x} Sum_{k<a} x^k/k! is small here; P = 1 - Q is safe.
        for idx in np.nonzero(big)[0]:
            n = int(a[idx])
            k = np.arange(n)
            logt = -x + k * lx - _gammaln(k + 1.0)
            q = float(np.exp(logt).sum())
            out[idx] = math.log(max(1.0 - q, 1e-300))
    return out


def chan_pc(b_vec_km, cov_2d_km2, hbr_km: float) -> float:
    """Chan (1997/2008) equivalent-area series Pc (see module docstring)."""
    b = np.asarray(b_vec_km, float)
    sx, sy, U = _principal(cov_2d_km2)
    if sx <= 0 or sy <= 0 or hbr_km <= 0:
        return 0.0
    xm, ym = U.T @ b
    u = hbr_km * hbr_km / (sx * sy)
    v = (xm / sx) ** 2 + (ym / sy) ** 2
    lam = 0.5 * v
    half_u = 0.5 * u
    # Terms peak near m ~ sqrt(lam * u/2); Poisson mass lies within lam +/- 12 sqrt(lam).
    m_max = int(min(50_000, lam + 12.0 * math.sqrt(lam + 1.0) + 60))
    m = np.arange(m_max + 1, dtype=float)
    log_pois = -lam + (m * math.log(lam) if lam > 0 else np.where(m == 0, 0.0, -np.inf)) \
        - _gammaln(m + 1.0)
    log_terms = log_pois + _log_reg_lower_gamma(m + 1.0, half_u)
    top = float(np.max(log_terms))
    if not np.isfinite(top):
        return 0.0
    val = math.exp(top) * float(np.exp(log_terms - top).sum())
    return float(min(1.0, max(0.0, val)))


# ══════════════════════════════════════════════════════════════════════════════
# Alfano maximum Pc (covariance-scale worst case)
# ══════════════════════════════════════════════════════════════════════════════

def _foster_scaled_factory(b_vec_km, cov_2d_km2, hbr_km: float):
    """Return f(s) = Foster Pc with covariance s*C (vectorised quadrature)."""
    from app.core import screening as _s
    b = np.asarray(b_vec_km, float)
    C = np.asarray(cov_2d_km2, float)
    det = float(np.linalg.det(C))
    Ci = np.linalg.inv(C)
    rho = 0.5 * hbr_km * (_s._GL_X + 1.0)
    w_rho = 0.5 * hbr_km * _s._GL_W
    X = rho[:, None] * np.cos(_s._THETA)[None, :] - b[0]
    Y = rho[:, None] * np.sin(_s._THETA)[None, :] - b[1]
    q = Ci[0, 0] * X * X + 2 * Ci[0, 1] * X * Y + Ci[1, 1] * Y * Y
    dth = 2 * np.pi / _s._N_THETA
    wgt = rho * w_rho * dth

    def f(s: float) -> float:
        pdf = np.exp(-0.5 * q / s) / (2 * np.pi * s * math.sqrt(det))
        return float(min(1.0, max(0.0, np.sum(pdf.sum(axis=1) * wgt))))
    return f


def alfano_max_pc_closed_form(b_vec_km, cov_2d_km2, hbr_km: float) -> float:
    """Alfano (2005) small-HBR maximum Pc: HBR^2 / (e * sx * sy * q)."""
    b = np.asarray(b_vec_km, float)
    C = np.asarray(cov_2d_km2, float)
    q = float(b @ np.linalg.solve(C, b))
    sx, sy, _ = _principal(C)
    if q <= 0:
        return 1.0
    return float(min(1.0, hbr_km ** 2 / (math.e * sx * sy * q)))


def alfano_max_pc(b_vec_km, cov_2d_km2, hbr_km: float, pc_nominal: float | None = None) -> float:
    """Maximum Foster Pc over covariance scale k^2*C (Alfano 2005), see module docstring."""
    b = np.asarray(b_vec_km, float)
    C = np.asarray(cov_2d_km2, float)
    if hbr_km <= 0:
        return 0.0
    if float(np.linalg.norm(b)) <= hbr_km:
        return 1.0
    f = _foster_scaled_factory(b, C, hbr_km)
    p1 = f(1.0) if pc_nominal is None else float(pc_nominal)
    q = float(b @ np.linalg.solve(C, b))
    s_star = max(q / 2.0, 1e-12)
    sx, sy, _ = _principal(C)
    if hbr_km <= 0.05 * sy * math.sqrt(s_star):
        # HBR << scaled sigma: the closed-form optimum is exact to O((HBR/sigma)^2);
        # evaluate the exact Foster integral there (one quadrature).
        return float(max(f(s_star), p1))
    # golden-section search on log10(s) within +/- 1.5 decades of the closed-form optimum
    lo, hi = math.log10(s_star) - 1.5, math.log10(s_star) + 1.5
    g = (math.sqrt(5) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = f(10 ** c), f(10 ** d)
    for _ in range(30):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo); fc = f(10 ** c)
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo); fd = f(10 ** d)
        if hi - lo < 1e-3:
            break
    return float(max(fc, fd, p1))


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo
# ══════════════════════════════════════════════════════════════════════════════

def monte_carlo_pc(b_vec_km, cov_2d_km2, hbr_km: float, *, seed: int = 0,
                   max_samples: int = MC_MAX_SAMPLES, batch: int = MC_BATCH,
                   min_hits: int = MC_MIN_HITS, target_hits: int = MC_TARGET_HITS) -> dict:
    """Monte Carlo B-plane Pc.

    Samples x ~ N(b, C) in batches (Cholesky of C), counts |x| <= HBR, stops
    when ``target_hits`` are reached or ``max_samples`` drawn.  Returns
    {"pc": float|None, "samples": N, "hits": h, "stderr": float|None, "seed": seed}.
    pc is None when h < min_hits (Pc below the resolution ~min_hits/N_max).
    """
    b = np.asarray(b_vec_km, float)
    C = np.asarray(cov_2d_km2, float)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        w, U = np.linalg.eigh(C)
        L = U @ np.diag(np.sqrt(np.clip(w, 0, None)))
    rng = np.random.default_rng(seed)
    r2 = hbr_km * hbr_km
    n = hits = 0
    while n < max_samples:
        m = min(batch, max_samples - n)
        x = rng.standard_normal((m, 2)) @ L.T + b
        hits += int(np.count_nonzero(np.einsum("ij,ij->i", x, x) <= r2))
        n += m
        if hits >= target_hits:
            break
    if hits < min_hits:
        return {"pc": None, "samples": n, "hits": hits, "stderr": None, "seed": seed,
                "resolution": min_hits / n}
    p = hits / n
    return {"pc": p, "samples": n, "hits": hits, "stderr": math.sqrt(p * (1 - p) / n), "seed": seed,
            "resolution": min_hits / n}


def _seed_for(key: Any) -> int:
    return zlib.crc32(str(key).encode()) & 0x7FFFFFFF


# ══════════════════════════════════════════════════════════════════════════════
# Contract block
# ══════════════════════════════════════════════════════════════════════════════

def pc_checks_from_inputs(b_vec_km, cov_2d_km2, hbr_km: float, *, monte_carlo: bool = False,
                          seed: int = 0, pc_reported: float | None = None) -> dict:
    """CONTRACT2 ``pc_checks`` block from explicit B-plane inputs."""
    foster = foster_pc(b_vec_km, cov_2d_km2, hbr_km)
    chan = chan_pc(b_vec_km, cov_2d_km2, hbr_km)
    amax = alfano_max_pc(b_vec_km, cov_2d_km2, hbr_km, pc_nominal=foster)
    mc = monte_carlo_pc(b_vec_km, cov_2d_km2, hbr_km, seed=seed) if monte_carlo else None
    logs = [_log10(foster), _log10(chan)]
    if mc is not None and mc["pc"] is not None:
        logs.append(_log10(mc["pc"]))
    spread = max(logs) - min(logs)
    fc_spread = abs(_log10(foster) - _log10(chan))
    sx, sy, _ = _principal(cov_2d_km2)
    out = {
        "foster": foster,
        "chan": chan,
        "alfano_max": amax,
        "monte_carlo": None if mc is None else mc["pc"],
        "mc_samples": None if mc is None else mc["samples"],
        "mc_hits": None if mc is None else mc["hits"],
        "mc_stderr": None if mc is None else mc["stderr"],
        "mc_note": (None if mc is None else
                    ("seeded vectorised sampling" if mc["pc"] is not None else
                     f"Pc below Monte Carlo resolution (~{mc['resolution']:.1e} with N={mc['samples']})")),
        "spread_decades": round(spread, 4),
        "consistent": bool(fc_spread <= CONSISTENT_SPREAD_DECADES),
        "aspect_ratio": round(sx / sy, 3) if sy > 0 else None,
        "log10_floor": LOG_FLOOR_PC,
        "methods": {"foster": _FOSTER_REF, "chan": _CHAN_REF, "alfano_max": _ALFANO_REF,
                    "monte_carlo": "direct B-plane sampling, seeded"},
    }
    if pc_reported is not None:
        out["foster_reported"] = float(pc_reported)
    return out


def pc_checks(alert: dict, *, monte_carlo: bool = False) -> dict | None:
    """CONTRACT2 ``pc_checks`` for an alert, or None if B-plane inputs are missing."""
    inp = bplane_inputs(alert)
    if inp is None:
        return None
    b, C, hbr = inp
    pc_rep = alert.get("probability_of_collision", alert.get("p_collision"))
    return pc_checks_from_inputs(b, C, hbr, monte_carlo=monte_carlo,
                                 seed=_seed_for(alert.get("id")), pc_reported=pc_rep)
