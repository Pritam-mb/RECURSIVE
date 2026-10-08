"""
Collision-avoidance manoeuvre planning by re-propagation.

For one conjunction alert the planner answers: "which burn, on which object,
brings the re-computed collision probability below the target, for the least
delta-v?" Nothing here is a heuristic score: every candidate burn is applied
to a copy of the object's trajectory, the encounter is re-screened, and Pc is
recomputed with the same Foster routine the screening uses
(app.core.screening.compute_pc).

Trajectory model for a burned object (documented assumption)
-------------------------------------------------------------
    r_burned(t) = r_SGP4(t) + [ r_J2(t; x0 + dv) - r_J2(t; x0) ]

x0 is the object's SGP4 state at the burn epoch (the simulation clock "now").
The bracketed term is the burn-induced deviation, propagated with a two-body +
J2 RK4 integrator for both the burned and the un-burned initial state. Taking
the difference cancels the common-mode mismatch between SGP4 mean elements and
an osculating J2 propagation, so a zero burn reproduces the screening geometry
exactly and the burn effect is propagated with real orbital dynamics (along-
track drift, radial/period change, J2 nodal effects). This is the classic
"reference trajectory + propagated deviation" used in conjunction-assessment
manoeuvre planning; it does not re-fit a TLE (re-fitting a TLE from an
osculating state shifts the orbit by kilometres and would swamp the burn).

Candidate search
----------------
Directions: +/-S (along-track), +/-R (radial), +/-W (cross-track) in the RSW
frame at the burn epoch; magnitudes CANDIDATE_DV_MS. All candidates of all
alerts are integrated as one vectorised batch on a 60 s node grid; inside the
encounter window the deviation is cubic-Hermite interpolated (it varies on the
orbital time-scale, so this is accurate to well below a metre) and TCA is
refined by linear relative motion.

Selection: the smallest |dv| whose re-computed Pc < PC_TARGET; if none
reaches the target, the candidate with the lowest Pc (reported with
achieved_target = False). Only the alert's own pair is re-screened
(rescreen_scope = "pair"): secondary conjunctions created by the burn are not
checked here.

Fuel: Tsiolkovsky rocket equation with a documented mass / Isp / propellant
model (spacecraft_mass_model); every assumed number is tagged with its source.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sgp4.api import jday

logger = logging.getLogger(__name__)

MU_KM3_S2 = 398600.4418
RE_KM = 6378.137
J2 = 1.08262668e-3
G0_MS2 = 9.80665

PC_TARGET = 1e-6
CANDIDATE_DV_MS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
CANDIDATE_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "+S": (0.0, 1.0, 0.0), "-S": (0.0, -1.0, 0.0),
    "+R": (1.0, 0.0, 0.0), "-R": (-1.0, 0.0, 0.0),
    "+W": (0.0, 0.0, 1.0), "-W": (0.0, 0.0, -1.0),
}
NODE_STEP_S = 60.0          # RK4 node spacing for the deviation integration
RK4_SUBSTEPS = 2            # 30 s RK4 steps
WINDOW_HALF_S = 900.0       # re-screening window around the original TCA
FINE_STEP_S = 2.0           # sampling inside the window before TCA refinement
MAX_HORIZON_S = 3.0 * 86400.0

# ── Spacecraft mass / propulsion model (assumptions, all tagged) ────────────
DEFAULT_MASS_KG = 500.0
DEFAULT_ISP_S = 220.0               # hydrazine monopropellant
DEFAULT_PROPELLANT_FRACTION = 0.10  # propellant / wet mass
RCS_CLASS_MASS_KG = {"SMALL": 20.0, "MEDIUM": 300.0, "LARGE": 1500.0}
NAME_CLASS_MODELS: list[tuple[str, dict[str, Any]]] = [
    # (name prefix, model). Public figures, rounded; source tag says so.
    ("ISS", {"mass_kg": 420000.0, "isp_s": 300.0, "propellant_fraction": 0.02,
             "class": "iss"}),
    ("STARLINK", {"mass_kg": 306.0, "isp_s": 1500.0, "propellant_fraction": 0.05,
                  "class": "starlink_v1.5_hall_thruster"}),
    ("ONEWEB", {"mass_kg": 147.0, "isp_s": 1500.0, "propellant_fraction": 0.05,
                "class": "oneweb_hall_thruster"}),
    ("IRIDIUM", {"mass_kg": 860.0, "isp_s": 220.0, "propellant_fraction": 0.10,
                 "class": "iridium_next"}),
]


def spacecraft_mass_model(name: str | None, satcat_rec: dict | None) -> dict[str, Any]:
    """Wet mass, Isp and propellant fraction, each tagged with its provenance."""
    upper = (name or "").upper()
    for prefix, model in NAME_CLASS_MODELS:
        if upper.startswith(prefix):
            out = dict(model)
            if prefix == "STARLINK" and satcat_rec and (satcat_rec.get("launch_year") or 0) >= 2023:
                out.update(mass_kg=800.0, isp_s=1500.0, **{"class": "starlink_v2_mini_hall_thruster"})
            out["mass_source"] = f"name_class:{out.pop('class')}"
            out["isp_source"] = out["mass_source"]
            return out
    rcs = (satcat_rec or {}).get("rcs_size")
    if rcs in RCS_CLASS_MASS_KG:
        return {"mass_kg": RCS_CLASS_MASS_KG[rcs], "mass_source": f"satcat_rcs_class:{rcs}",
                "isp_s": DEFAULT_ISP_S, "isp_source": "default_hydrazine",
                "propellant_fraction": DEFAULT_PROPELLANT_FRACTION}
    return {"mass_kg": DEFAULT_MASS_KG, "mass_source": "default", "isp_s": DEFAULT_ISP_S,
            "isp_source": "default_hydrazine", "propellant_fraction": DEFAULT_PROPELLANT_FRACTION}


def rocket_equation_cost(delta_v_ms: float, model: dict[str, Any]) -> dict[str, float]:
    """Propellant used (kg) and % of the assumed propellant load."""
    m0 = float(model["mass_kg"])
    used = m0 * (1.0 - math.exp(-abs(delta_v_ms) / (float(model["isp_s"]) * G0_MS2)))
    load = m0 * float(model["propellant_fraction"])
    return {"propellant_kg": used, "fuel_cost_pct": 100.0 * used / load if load > 0 else float("nan")}


# ── Dynamics ────────────────────────────────────────────────────────────────

def _j2_accel(r: np.ndarray) -> np.ndarray:
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    r2 = x * x + y * y + z * z
    rn = np.sqrt(r2)
    k = -MU_KM3_S2 / (r2 * rn)
    f = 1.5 * J2 * MU_KM3_S2 * RE_KM ** 2 / (r2 * r2 * rn)
    zz = 5.0 * z * z / r2
    return np.stack([k * x - f * x * (1.0 - zz), k * y - f * y * (1.0 - zz), k * z - f * z * (3.0 - zz)], axis=1)


def _rk4_step(r: np.ndarray, v: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    a1 = _j2_accel(r)
    r2 = r + 0.5 * h * v; v2 = v + 0.5 * h * a1; a2 = _j2_accel(r2)
    r3 = r + 0.5 * h * v2; v3 = v + 0.5 * h * a2; a3 = _j2_accel(r3)
    r4 = r + h * v3; v4 = v + h * a3; a4 = _j2_accel(r4)
    return r + (h / 6.0) * (v + 2 * v2 + 2 * v3 + v4), v + (h / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)


def propagate_j2(r: np.ndarray, v: np.ndarray, dt_s: float, max_step_s: float = 30.0):
    r = np.atleast_2d(np.asarray(r, float)); v = np.atleast_2d(np.asarray(v, float))
    if abs(dt_s) < 1e-12:
        return r.copy(), v.copy()
    n = max(1, int(math.ceil(abs(dt_s) / max_step_s)))
    h = dt_s / n
    for _ in range(n):
        r, v = _rk4_step(r, v, h)
    return r, v


def rsw_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Columns R, S, W (radial, along-track, cross-track) in ECI."""
    r_hat = r / np.linalg.norm(r)
    w = np.cross(r, v); w_hat = w / np.linalg.norm(w)
    s_hat = np.cross(w_hat, r_hat)
    return np.column_stack([r_hat, s_hat, w_hat])


def _hermite(t: np.ndarray, t0: float, step: float, P: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Cubic Hermite on a uniform node grid. P, D: (..., nodes, 3) values/derivatives."""
    n = P.shape[-2]
    u = (np.asarray(t, float) - t0) / step
    k = np.clip(np.floor(u).astype(int), 0, n - 2)
    s = (u - k)[..., None]
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2;    h11 = s ** 3 - s ** 2
    return h00 * P[..., k, :] + h10 * step * D[..., k, :] + h01 * P[..., k + 1, :] + h11 * step * D[..., k + 1, :]


def _hermite_deriv(t: np.ndarray, t0: float, step: float, P: np.ndarray, D: np.ndarray) -> np.ndarray:
    n = P.shape[-2]
    u = (np.asarray(t, float) - t0) / step
    k = np.clip(np.floor(u).astype(int), 0, n - 2)
    s = (u - k)[..., None]
    d00 = (6 * s ** 2 - 6 * s) / step; d10 = 3 * s ** 2 - 4 * s + 1
    d01 = (-6 * s ** 2 + 6 * s) / step; d11 = 3 * s ** 2 - 2 * s
    return d00 * P[..., k, :] + d10 * D[..., k, :] + d01 * P[..., k + 1, :] + d11 * D[..., k + 1, :]


# ── Problem description ────────────────────────────────────────────────────

@dataclass
class ObjectTrack:
    """An object's trajectory source: an SGP4 Satrec, or a state at the burn epoch."""
    norad_id: Any
    name: str
    satrec: Any = None
    r0: np.ndarray | None = None
    v0: np.ndarray | None = None


@dataclass
class Mover:
    track: ObjectTrack
    agency: str = "Unknown"
    owner_known: bool = False
    satcat: dict | None = None


@dataclass
class EncounterJob:
    alert: dict
    is_sat1: list[bool]                # per mover: does the mover play sat1?
    movers: list[Mover]
    others: list[ObjectTrack]          # per mover: the object it avoids
    tca_s: float
    hbr_km: float
    ages_days: tuple[float, float] | None
    iso_sigma_km: float | None = None   # isotropic B-plane sigma (debris alerts), else TLE-age RTN model
    rows: list[np.ndarray] = field(default_factory=list)       # per mover row indices
    other_rows: list[int | None] = field(default_factory=list)
    k0: int = 0
    k1: int = 0
    nodes_R: dict = field(default_factory=dict)
    nodes_V: dict = field(default_factory=dict)


def _sgp4_states(satrec, jd0: float, fr0: float, t_s: np.ndarray):
    t_s = np.asarray(t_s, float)
    jd = np.full(t_s.shape, jd0)
    fr = fr0 + t_s / 86400.0
    e, r, v = satrec.sgp4_array(jd.ravel(), fr.ravel())
    r = np.asarray(r).reshape(t_s.shape + (3,)); v = np.asarray(v).reshape(t_s.shape + (3,))
    bad = np.asarray(e).reshape(t_s.shape) != 0
    if bad.any():
        r = r.copy(); v = v.copy(); r[bad] = np.nan; v[bad] = np.nan
    return r, v


def _compute_pc_fn():
    try:
        from app.core.screening import compute_pc  # agent A: same Foster routine as screening
        return compute_pc, "app.core.screening.compute_pc"
    except Exception:
        return None, None


def _isotropic_pc(r1, v1, r2, v2, sigma_km: float, hbr_km: float) -> float:
    """Foster Pc with an isotropic B-plane covariance (the debris-alert model)."""
    from app.core.screening import foster_pc

    dv = np.asarray(v2, float) - np.asarray(v1, float)
    h = dv / max(np.linalg.norm(dv), 1e-12)
    dr = np.asarray(r2, float) - np.asarray(r1, float)
    b = dr - h * float(dr @ h)                      # miss vector in the B-plane
    return foster_pc(np.array([np.linalg.norm(b), 0.0]), np.eye(2) * sigma_km ** 2, hbr_km)


def plan_maneuvers(jobs: list[EncounterJob], burn_epoch: datetime, *, pc_target: float = PC_TARGET,
                   time_budget_s: float = 3.0) -> dict[str, Any]:
    """Run the batched re-propagation search. Returns stats; writes job results in place."""
    t_start = time.perf_counter()
    compute_pc, pc_source = _compute_pc_fn()
    stats: dict[str, Any] = {"jobs": len(jobs), "pc_source": pc_source, "candidates_evaluated": 0}
    if compute_pc is None or not jobs:
        stats["status"] = "no_pc_routine" if compute_pc is None else "no_jobs"
        for job in jobs:
            job.alert["_maneuver_result"] = None
        return stats

    if burn_epoch.tzinfo is None:
        burn_epoch = burn_epoch.replace(tzinfo=timezone.utc)
    jd0, fr0 = jday(burn_epoch.year, burn_epoch.month, burn_epoch.day, burn_epoch.hour,
                    burn_epoch.minute, burn_epoch.second + burn_epoch.microsecond / 1e6)

    dv_dirs = list(CANDIDATE_DIRECTIONS.items())
    cand_rsw = np.array([np.array(d) * m for _, d in dv_dirs for m in CANDIDATE_DV_MS])   # m/s
    cand_label = [f"{name} {m:g} m/s" for name, _ in dv_dirs for m in CANDIDATE_DV_MS]
    n_cand = len(cand_rsw)

    # ── Build batch rows (initial states at burn epoch) ─────────────────────
    R0: list[np.ndarray] = []
    V0: list[np.ndarray] = []
    n_rows = 0
    for job in jobs:
        job.rows, job.other_rows = [], []
        for mover, other in zip(job.movers, job.others):
            e, r, v = mover.track.satrec.sgp4(jd0, fr0)
            if e != 0:
                job.rows.append(None); job.other_rows.append(None)
                continue
            r = np.asarray(r); v = np.asarray(v)
            Q = rsw_basis(r, v)
            dv_eci_kms = (Q @ cand_rsw.T).T / 1000.0
            R0.append(np.repeat(r[None, :], n_cand + 1, axis=0))
            V0.append(np.vstack([v[None, :], v[None, :] + dv_eci_kms]))
            job.rows.append(np.arange(n_rows, n_rows + n_cand + 1))
            n_rows += n_cand + 1
            if other.satrec is None:
                job.other_rows.append(n_rows)
                R0.append(np.asarray(other.r0, float)[None, :]); V0.append(np.asarray(other.v0, float)[None, :])
                n_rows += 1
            else:
                job.other_rows.append(None)
        lo = max(0.0, job.tca_s - WINDOW_HALF_S)
        job.k0 = int(math.floor(lo / NODE_STEP_S))
        job.k1 = int(math.ceil((job.tca_s + WINDOW_HALF_S) / NODE_STEP_S)) + 1
        job.nodes_R, job.nodes_V = {}, {}

    if not R0:
        stats["status"] = "no_propagatable_movers"
        return stats
    R = np.vstack(R0); V = np.vstack(V0)
    stats["batch_rows"] = int(R.shape[0])

    # Which jobs need node k
    needed: dict[int, list[int]] = {}
    k_max = 0
    for j, job in enumerate(jobs):
        if all(r is None for r in job.rows):
            continue
        for k in range(job.k0, job.k1 + 1):
            needed.setdefault(k, []).append(j)
        k_max = max(k_max, job.k1)

    # Record only the rows each job needs (avoid copying the full batch)
    job_rowsets = []
    for job in jobs:
        idx = [r for r in job.rows if r is not None]
        idx += [o for o in job.other_rows if o is not None]
        job_rowsets.append(np.concatenate([np.atleast_1d(x) for x in idx]) if idx else np.zeros(0, int))

    h = NODE_STEP_S / RK4_SUBSTEPS
    for k in range(0, k_max + 1):
        if k > 0:
            for _ in range(RK4_SUBSTEPS):
                R, V = _rk4_step(R, V, h)
        for j in needed.get(k, ()):
            rs = job_rowsets[j]
            jobs[j].nodes_R[k] = R[rs].copy()
            jobs[j].nodes_V[k] = V[rs].copy()
    stats["t_integrate_s"] = round(time.perf_counter() - t_start, 3)

    # ── Per job: window re-screen + Pc ─────────────────────────────────────
    for j, job in enumerate(jobs):
        if time.perf_counter() - t_start > time_budget_s:
            job.alert["_maneuver_result"] = {"status": "time_budget_exceeded"}
            continue
        rs = job_rowsets[j]
        if rs.size == 0:
            job.alert["_maneuver_result"] = {"status": "mover_propagation_failed"}
            continue
        ks = list(range(job.k0, job.k1 + 1))
        NR = np.stack([job.nodes_R[k] for k in ks], axis=1)   # (rows, nodes, 3)
        NV = np.stack([job.nodes_V[k] for k in ks], axis=1)
        local = {int(g): i for i, g in enumerate(rs)}
        t_node0 = job.k0 * NODE_STEP_S
        lo = max(0.0, job.tca_s - WINDOW_HALF_S)
        t_f = np.arange(lo, job.tca_s + WINDOW_HALF_S + 1e-9, FINE_STEP_S)

        per_mover = []
        for m_idx, (mover, other) in enumerate(zip(job.movers, job.others)):
            rows = job.rows[m_idx]
            if rows is None:
                continue
            li = np.array([local[int(x)] for x in rows])
            dR = NR[li] - NR[li[0]]; dV = NV[li] - NV[li[0]]           # deviations (cand+1, nodes, 3)
            if other.satrec is None:
                oi = local[int(job.other_rows[m_idx])]
                o_R, o_V = NR[oi], NV[oi]

                def other_state(t, o_R=o_R, o_V=o_V):
                    return _hermite(t, t_node0, NODE_STEP_S, o_R, o_V), _hermite_deriv(t, t_node0, NODE_STEP_S, o_R, o_V)
            else:
                def other_state(t, sat=other.satrec):
                    return _sgp4_states(sat, jd0, fr0, t)

            def mover_state(t, sat=mover.track.satrec, dR=dR, dV=dV):
                # t: (ncand+1, nt) or (ncand+1,)
                rr, vv = _sgp4_states(sat, jd0, fr0, t)
                tt = np.asarray(t, float)
                if tt.ndim == 1:
                    dr = np.stack([_hermite(tt[c:c + 1], t_node0, NODE_STEP_S, dR[c], dV[c])[0] for c in range(dR.shape[0])])
                    dv = np.stack([_hermite_deriv(tt[c:c + 1], t_node0, NODE_STEP_S, dR[c], dV[c])[0] for c in range(dR.shape[0])])
                else:
                    dr = np.stack([_hermite(tt[c], t_node0, NODE_STEP_S, dR[c], dV[c]) for c in range(dR.shape[0])])
                    dv = np.stack([_hermite_deriv(tt[c], t_node0, NODE_STEP_S, dR[c], dV[c]) for c in range(dR.shape[0])])
                return rr + dr, vv + dv

            ncr = dR.shape[0]
            T = np.broadcast_to(t_f, (ncr, t_f.size))
            r_m, v_m = mover_state(T)
            r_o, v_o = other_state(t_f)
            rel = r_m - r_o[None]
            dist = np.linalg.norm(rel, axis=2)
            dist = np.where(np.isfinite(dist), dist, np.inf)
            i_min = np.argmin(dist, axis=1)
            t_star = t_f[i_min]
            for _ in range(3):  # linear-relative-motion TCA refinement
                rm, vm = mover_state(t_star)
                ro, vo = other_state(t_star)
                dr = rm - ro; dvr = vm - vo
                dv2 = np.einsum("ij,ij->i", dvr, dvr)
                shift = np.where(dv2 > 1e-12, -np.einsum("ij,ij->i", dr, dvr) / np.maximum(dv2, 1e-12), 0.0)
                t_star = np.clip(t_star + np.clip(shift, -FINE_STEP_S, FINE_STEP_S), lo, job.tca_s + WINDOW_HALF_S)
            rm, vm = mover_state(t_star)
            ro, vo = other_state(t_star)
            miss = np.linalg.norm(rm - ro, axis=1)
            edge = (t_star <= lo + 1e-6) | (t_star >= job.tca_s + WINDOW_HALF_S - 1e-6)

            pcs = np.empty(ncr)
            for c in range(ncr):
                if not np.all(np.isfinite(rm[c])) or not np.all(np.isfinite(ro[c])):
                    pcs[c] = np.nan
                    continue
                if job.is_sat1[m_idx]:
                    r1, v1, r2, v2 = rm[c], vm[c], ro[c], vo[c]
                else:
                    r1, v1, r2, v2 = ro[c], vo[c], rm[c], vm[c]
                if job.iso_sigma_km:
                    pcs[c] = _isotropic_pc(r1, v1, r2, v2, job.iso_sigma_km, job.hbr_km)
                else:
                    a1, a2 = job.ages_days or (0.0, 0.0)
                    pcs[c] = compute_pc(r1, v1, r2, v2, age1_days=a1, age2_days=a2, hbr_km=job.hbr_km)["pc"]
            stats["candidates_evaluated"] += ncr - 1
            per_mover.append({
                "mover": mover, "pcs": pcs, "miss": miss, "t_star": t_star, "edge": edge,
            })

        job.alert["_maneuver_result"] = _select(job, per_mover, cand_rsw, cand_label, burn_epoch, pc_target, jd0, fr0)

    stats["t_total_s"] = round(time.perf_counter() - t_start, 3)
    stats["status"] = "ok"
    return stats


def _select(job: EncounterJob, per_mover: list[dict], cand_rsw: np.ndarray, cand_label: list[str],
            burn_epoch: datetime, pc_target: float, jd0: float, fr0: float) -> dict[str, Any]:
    if not per_mover:
        return {"status": "no_manoeuvrable_object"}
    options = []
    baseline = None
    for pm in per_mover:
        pcs, miss, edge = pm["pcs"], pm["miss"], pm["edge"]
        if baseline is None:
            baseline = {"pc": float(pcs[0]), "miss_km": float(miss[0]),
                        "tca_s": float(pm["t_star"][0])}
        dv_mag = np.linalg.norm(cand_rsw, axis=1)
        for c in range(1, len(pcs)):
            if not np.isfinite(pcs[c]) or edge[c]:
                continue
            options.append((pm, c, float(pcs[c]), float(dv_mag[c - 1]), float(miss[c])))
    if baseline is not None and baseline["pc"] < pc_target:
        return {"status": "below_target_without_burn", "baseline": baseline}
    if not options:
        return {"status": "no_valid_candidate", "baseline": baseline}
    achieving = [o for o in options if o[2] < pc_target]
    if achieving:
        best = min(achieving, key=lambda o: (not o[0]["mover"].owner_known, o[3], -o[4]))
        achieved = True
    else:
        best = min(options, key=lambda o: (o[2], o[3]))
        achieved = False
    pm, c, pc_new, dv, miss_new = best
    mover: Mover = pm["mover"]
    e, r, v = mover.track.satrec.sgp4(jd0, fr0)
    dv_rsw = cand_rsw[c - 1]
    dv_eci = rsw_basis(np.asarray(r), np.asarray(v)) @ dv_rsw
    model = spacecraft_mass_model(mover.track.name, mover.satcat)
    fuel = rocket_equation_cost(dv, model)
    new_tca = burn_epoch + timedelta(seconds=float(pm["t_star"][c]))
    return {
        "status": "ok",
        "recommended_maneuver": {
            "sat_id": mover.track.norad_id,
            "sat_name": mover.track.name,
            "agency": mover.agency,
            "delta_v_rsw_ms": [round(float(x), 4) for x in dv_rsw],
            "delta_v_eci_ms": [round(float(x), 4) for x in dv_eci],
            "delta_v_ms": round(dv, 4),
            "candidate": cand_label[c - 1],
            "burn_epoch_utc": burn_epoch.isoformat(),
            "new_tca_utc": new_tca.isoformat(),
            "new_miss_distance_km": round(miss_new, 4),
            "new_pc_collision": pc_new,
            "pc_before": float(job.alert.get("p_collision") or job.alert.get("probability_of_collision") or 0.0),
            "pc_before_recomputed": baseline["pc"],
            "miss_before_km": round(baseline["miss_km"], 4),
            "target_pc": pc_target,
            "achieved_target": achieved,
            "fuel_cost_pct": round(fuel["fuel_cost_pct"], 4),
            "propellant_kg": round(fuel["propellant_kg"], 5),
            "fuel_model": "rocket_equation",
            "mass_kg": model["mass_kg"], "mass_source": model["mass_source"],
            "isp_s": model["isp_s"], "isp_source": model["isp_source"],
            "propellant_fraction": model["propellant_fraction"],
            "candidates_evaluated": int(sum(len(p["pcs"]) - 1 for p in per_mover)),
            "movers_considered": [p["mover"].track.norad_id for p in per_mover],
            "rescreen_scope": "pair",
            "propagation": "sgp4_reference+j2_rk4_deviation",
            "verified_by": "repropagation",
        },
    }
