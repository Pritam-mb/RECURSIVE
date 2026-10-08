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

Cascade-safe selection (no burn may hamper another object)
---------------------------------------------------------
Up to MAX_OPTIONS pair-verified candidates are shortlisted (smallest Δv that
reaches Pc < PC_TARGET per direction, then further magnitudes / lowest Pc).
Each option's 24 h burned trajectory is re-screened against the WHOLE
catalogue plus live debris fragments (_cascade_check: same grid / linear-
motion filter / TCA refinement / Foster Pc as app.core.screening.screen,
restricted to the burned row). Secondary conjunctions (Pc >= 1e-7 or miss
< 5 km, original threat excluded) are compared with a zero-burn pass; an
option is cascade_safe when no burn-induced or burn-worsened secondary has
Pc >= 1e-6 or miss < 1 km. Selection (_finalize, written to selection_rule):
min Δv among cascade-safe options reaching the target; else the largest Pc
reduction among cascade-safe options; else "no cascade-safe option" is
flagged honestly. Each option also carries the propulsion trade
(app.core.propulsion: Tsiolkovsky propellant, finite-burn time per engine
class) and a Clohessy-Wiltshire along-track check of the numerical shift
(app.core.analytic_checks).

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

# ── Cascade-safety re-screen (secondary conjunctions created by a burn) ─────
MAX_OPTIONS = 4             # shortlisted options per alert that get the catalogue re-screen
SEC_WINDOW_H = 24.0         # re-screen horizon after the burn
SEC_STEP_S = 60.0           # catalogue grid step (same as app.core.screening default)
SEC_DETECT_KM = 10.0        # refine every predicted approach inside this distance
SEC_LINEAR_MARGIN_KM = 5.0  # linear-relative-motion error allowance (as in screening)
SEC_VREL_MAX_KMS = 16.0     # bound on LEO relative speed for the coarse grid gate
SEC_REPORT_MISS_KM = 5.0    # a secondary is reported if miss < 5 km ...
SEC_REPORT_PC = 1e-7        # ... or Pc >= 1e-7
SAFE_PC = 1e-6              # cascade_safe: no burn-induced/-worsened secondary with Pc >= 1e-6
SAFE_MISS_KM = 1.0          #               or miss < 1 km
SHELL_MARGIN_KM = 50.0      # perigee/apogee shell prefilter slack
CASCADE_BUDGET_S = 2.5

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


def _hermite_batch(t: np.ndarray, t0: float, step: float, P: np.ndarray, D: np.ndarray):
    """Batched cubic Hermite value and derivative: t (C, N), P/D (C, nodes, 3) -> (C, N, 3) each."""
    n = P.shape[1]
    u = (t - t0) / step
    k = np.clip(np.floor(u).astype(int), 0, n - 2)
    s = (u - k)[..., None]
    kk = np.repeat(k[..., None], 3, axis=2)
    P0 = np.take_along_axis(P, kk, axis=1); P1 = np.take_along_axis(P, kk + 1, axis=1)
    D0 = np.take_along_axis(D, kk, axis=1); D1 = np.take_along_axis(D, kk + 1, axis=1)
    s2, s3 = s * s, s * s * s
    val = (2 * s3 - 3 * s2 + 1) * P0 + (s3 - 2 * s2 + s) * step * D0 + (-2 * s3 + 3 * s2) * P1 + (s3 - s2) * step * D1
    der = ((6 * s2 - 6 * s) / step) * P0 + (3 * s2 - 4 * s + 1) * D0 + ((-6 * s2 + 6 * s) / step) * P1 + (3 * s2 - 2 * s) * D1
    return val, der


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
    other_nodes: dict = field(default_factory=dict)   # m_idx -> (R, V) node arrays of a fragment (drag model)
    shortlist: Any = None                              # (baseline, [option dicts]) awaiting the cascade check


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


def _fragment_state_for(job: EncounterJob, other: ObjectTrack) -> dict | None:
    """The alert's ``fragment_state`` when ``other`` is the debris fragment of a debris alert.

    Debris alerts (app.core.debris_model.compute_debris_alerts) carry the
    fragment's state at a known sim epoch plus its ballistic coefficient; the
    fragment is then propagated here with the same two-body + J2 + drag model
    (app.core.breakup.propagate) that generated the alert, instead of the
    drag-free J2 batch used for objects given only as a state."""
    if other.satrec is not None:
        return None
    alert = job.alert
    st = alert.get("fragment_state")
    if not isinstance(st, dict) or st.get("r_km") is None or st.get("v_kms") is None or not st.get("epoch_utc"):
        return None
    deb_ids = {str(alert.get(s, {}).get("id")) for s in ("sat1", "sat2")
               if (alert.get(s) or {}).get("object_type") == "DEB"}
    if str(other.norad_id) not in deb_ids:
        return None
    return st


def _fragment_nodes(st: dict, burn_epoch: datetime, k0: int, k1: int):
    """Fragment positions/velocities on the node grid t_k = k * NODE_STEP_S after burn_epoch."""
    from app.core import breakup as sbm

    epoch = datetime.fromisoformat(str(st["epoch_utc"]).replace("Z", "+00:00"))
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    bc = st.get("ballistic_coeff_m2_kg")
    bc_arr = None if bc is None else np.array([float(bc)])
    r = np.asarray(st["r_km"], float)[None, :]
    v = np.asarray(st["v_kms"], float)[None, :]
    dt0 = k0 * NODE_STEP_S - (epoch - burn_epoch).total_seconds()
    if abs(dt0) > 1e-9:
        r, v = sbm.propagate(r, v, bc_arr, dt0, max_step_s=NODE_STEP_S / RK4_SUBSTEPS)
    R = [r[0]]; V = [v[0]]
    for _ in range(k0 + 1, k1 + 1):
        r, v = sbm.propagate(r, v, bc_arr, NODE_STEP_S, max_step_s=NODE_STEP_S / RK4_SUBSTEPS)
        R.append(r[0]); V.append(v[0])
    return np.array(R), np.array(V)


def plan_maneuvers(jobs: list[EncounterJob], burn_epoch: datetime, *, pc_target: float = PC_TARGET,
                   time_budget_s: float = 3.0, catalogue: list[ObjectTrack] | None = None,
                   extra_objects: list[dict] | None = None,
                   cascade_budget_s: float = CASCADE_BUDGET_S) -> dict[str, Any]:
    """Run the batched re-propagation search. Returns stats; writes job results in place.

    catalogue / extra_objects: every other tracked object (SGP4 tracks) and
    state-vector objects (live debris fragments, scenario objects) used for the
    cascade-safety re-screen of each shortlisted option. Without a catalogue
    the options are reported with cascade_safe = None (check not run)."""
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
        job.rows, job.other_rows, job.other_nodes = [], [], {}
        lo = max(0.0, job.tca_s - WINDOW_HALF_S)
        job.k0 = int(math.floor(lo / NODE_STEP_S))
        job.k1 = int(math.ceil((job.tca_s + WINDOW_HALF_S) / NODE_STEP_S)) + 1
        for m_idx, (mover, other) in enumerate(zip(job.movers, job.others)):
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
            frag = _fragment_state_for(job, other)
            if frag is not None:
                job.other_nodes[m_idx] = _fragment_nodes(frag, burn_epoch, job.k0, job.k1)
                job.other_rows.append(None)
            elif other.satrec is None:
                job.other_rows.append(n_rows)
                R0.append(np.asarray(other.r0, float)[None, :]); V0.append(np.asarray(other.v0, float)[None, :])
                n_rows += 1
            else:
                job.other_rows.append(None)
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
            if m_idx in job.other_nodes:
                fR, fV = job.other_nodes[m_idx]

                def other_state(t, o_R=fR, o_V=fV):
                    return _hermite(t, t_node0, NODE_STEP_S, o_R, o_V), _hermite_deriv(t, t_node0, NODE_STEP_S, o_R, o_V)
            elif other.satrec is None:
                oi = local[int(job.other_rows[m_idx])]
                o_R, o_V = NR[oi], NV[oi]

                def other_state(t, o_R=o_R, o_V=o_V):
                    return _hermite(t, t_node0, NODE_STEP_S, o_R, o_V), _hermite_deriv(t, t_node0, NODE_STEP_S, o_R, o_V)
            else:
                def other_state(t, sat=other.satrec):
                    return _sgp4_states(sat, jd0, fr0, t)

            def mover_state(t, sat=mover.track.satrec, dR=dR, dV=dV):
                # t: (ncand+1, nt) or (ncand+1,): candidate c evaluated at its own times
                rr, vv = _sgp4_states(sat, jd0, fr0, t)
                tt = np.asarray(t, float)
                t2 = tt[:, None] if tt.ndim == 1 else tt
                dr, dv = _hermite_batch(t2, t_node0, NODE_STEP_S, dR, dV)
                if tt.ndim == 1:
                    dr, dv = dr[:, 0], dv[:, 0]
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
                "fragment_drag": m_idx in job.other_nodes, "other_id": other.norad_id,
            })

        early, baseline, picked = _shortlist(job, per_mover, cand_rsw, cand_label, pc_target)
        if early is not None:
            job.alert["_maneuver_result"] = early
        else:
            job.shortlist = (baseline, picked)
    stats["t_pair_search_s"] = round(time.perf_counter() - t_start, 3)

    # ── Cascade-safety re-screen of the shortlisted options ────────────────
    entries = [(job, o) for job in jobs if job.shortlist for o in job.shortlist[1]]
    t_c = time.perf_counter()
    cstats = {"status": "no_options"}
    if entries:
        try:
            cstats = _cascade_check(entries, burn_epoch, jd0, fr0, catalogue, extra_objects,
                                    cascade_budget_s, compute_pc)
        except Exception as exc:  # never lose the pair-verified plan
            logger.exception("cascade-safety check failed")
            cstats = {"status": f"error:{type(exc).__name__}"}
            for _, o in entries:
                o.setdefault("cascade_check", "error")
    stats["cascade_check"] = {**cstats, "t_s": round(time.perf_counter() - t_c, 3)}
    for job in jobs:
        if job.shortlist:
            baseline, picked = job.shortlist
            job.alert["_maneuver_result"] = _finalize(job, baseline, picked, burn_epoch, pc_target, jd0, fr0,
                                                      cstats.get("status"))
            job.shortlist = None

    stats["t_total_s"] = round(time.perf_counter() - t_start, 3)
    stats["status"] = "ok"
    return stats




# ── Option shortlist ────────────────────────────────────────────────────────

def _option_key(o: dict) -> tuple:
    """Preference order: operator-known movers first, then least Δv, then largest new miss."""
    return (not o["pm"]["mover"].owner_known, o["dv"], -o["miss"])


def _shortlist(job: EncounterJob, per_mover: list[dict], cand_rsw: np.ndarray, cand_label: list[str],
               pc_target: float):
    """Pick up to MAX_OPTIONS pair-verified candidates for the cascade re-screen.

    Order: for each (mover, direction) the smallest Δv that reaches Pc < target
    (so the shortlist spans different directions — a burn the other way often
    avoids a secondary the naive burn creates), then further target-reaching
    magnitudes, then the lowest-Pc candidates if fewer reach the target.
    Returns (early_result | None, baseline, options)."""
    if not per_mover:
        return {"status": "no_manoeuvrable_object"}, None, []
    n_mag = len(CANDIDATE_DV_MS)
    dv_mag = np.linalg.norm(cand_rsw, axis=1)
    options, baseline = [], None
    for pm in per_mover:
        pcs, miss, edge = pm["pcs"], pm["miss"], pm["edge"]
        if baseline is None:
            baseline = {"pc": float(pcs[0]), "miss_km": float(miss[0]), "tca_s": float(pm["t_star"][0])}
        for c in range(1, len(pcs)):
            if not np.isfinite(pcs[c]) or edge[c]:
                continue
            options.append({"pm": pm, "c": c, "pc": float(pcs[c]), "dv": float(dv_mag[c - 1]),
                            "miss": float(miss[c]), "dir": (c - 1) // n_mag,
                            "label": cand_label[c - 1], "dv_rsw": cand_rsw[c - 1]})
    if baseline is not None and baseline["pc"] < pc_target:
        return {"status": "below_target_without_burn", "baseline": baseline}, baseline, []
    if not options:
        return {"status": "no_valid_candidate", "baseline": baseline}, baseline, []

    achieving = sorted((o for o in options if o["pc"] < pc_target), key=_option_key)
    picked, dirs = [], set()
    for o in achieving:                         # smallest target-reaching Δv per (mover, direction)
        k = (id(o["pm"]), o["dir"])
        if k not in dirs:
            dirs.add(k)
            picked.append(o)
    picked = sorted(picked, key=_option_key)[:MAX_OPTIONS]
    used = {id(o) for o in picked}
    rest = sorted((o for o in options if o["pc"] >= pc_target), key=lambda o: (o["pc"], o["dv"]))
    for pool in (achieving, rest):
        for o in pool:
            if len(picked) >= MAX_OPTIONS:
                break
            if id(o) not in used:
                picked.append(o)
                used.add(id(o))
    picked.sort(key=lambda o: (o["pc"] >= pc_target,) + _option_key(o))
    return None, baseline, picked


# ── Cascade-safety re-screen ────────────────────────────────────────────────

def _shell(sat=None, r=None, v=None) -> tuple[float, float]:
    """Perigee / apogee radius (km): SGP4 mean elements, or osculating from a state."""
    if sat is not None:
        try:
            re_ = float(getattr(sat, "radiusearthkm", RE_KM))
            return (1.0 + float(sat.altp)) * re_, (1.0 + float(sat.alta)) * re_
        except Exception:
            return 0.0, float("inf")
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn = float(np.linalg.norm(r))
    vn2 = float(v @ v)
    inv_a = 2.0 / rn - vn2 / MU_KM3_S2
    if inv_a <= 0:
        return rn, float("inf")
    a = 1.0 / inv_a
    e_vec = ((vn2 - MU_KM3_S2 / rn) * r - float(r @ v) * v) / MU_KM3_S2
    e = float(np.linalg.norm(e_vec))
    return a * (1.0 - e), a * (1.0 + e)


def _cascade_check(entries, burn_epoch, jd0, fr0, catalogue, extra_objects, budget_s, compute_pc) -> dict:
    """Re-screen every shortlisted option's burned trajectory against the catalogue.

    Same algorithm as app.core.screening.screen, restricted to the one burned
    row against all other objects (an all-vs-all screen per option would redo
    every pair the burn cannot affect):
      1. the burn deviation [J2(x0+dv) - J2(x0)] of all options is integrated
         in one vectorised RK4 batch on the 60 s grid (identical model to
         SGP4Propagator.apply_delta_v / BurnDeviation, i.e. to what happens
         when the operator executes the burn),
      2. the catalogue grid is built ONCE with screening._Window (SGP4 for TLE
         objects incl. already executed burns, two-body+J2 for state-vector
         objects such as live debris fragments) after a perigee/apogee shell
         prefilter,
      3. per option, grid samples inside the gate distance are linear-relative-
         motion filtered, the TCA refined with exact states (SGP4 + Hermite
         deviation for the burned object, SGP4 / J2 for the other) and Pc
         computed with the screening's Foster routine (TLE-age covariance,
         catalogue hard-body radii).
    A zero-burn pass gives each mover's pre-existing approaches so the check
    separates burn-induced (or worsened) secondaries from ones that exist with
    or without the burn (those have their own alerts).
    """
    t0 = time.perf_counter()
    nt = int(round(SEC_WINDOW_H * 3600.0 / SEC_STEP_S)) + 1
    t_end = (nt - 1) * SEC_STEP_S
    st: dict[str, Any] = {"window_hours": SEC_WINDOW_H, "step_s": SEC_STEP_S, "options": len(entries)}

    # 1. Burn deviations of all options (also feed the analytic CW check)
    t_need = max([t_end] + [float(o["pm"]["t_star"][o["c"]]) for _, o in entries]) + 2 * SEC_STEP_S
    n_nodes = int(math.ceil(t_need / SEC_STEP_S)) + 1
    R0, V0 = [], []
    for _, o in entries:
        _, r, v = o["pm"]["mover"].track.satrec.sgp4(jd0, fr0)
        r = np.asarray(r, float)
        v = np.asarray(v, float)
        o["_r0"], o["_v0"] = r, v
        dv = rsw_basis(r, v) @ np.asarray(o["dv_rsw"], float) / 1000.0
        R0 += [r, r]
        V0 += [v, v + dv]
    R = np.array(R0)
    V = np.array(V0)
    DR = np.empty((n_nodes, len(entries), 3))
    DV = np.empty_like(DR)
    DR[0] = R[1::2] - R[0::2]
    DV[0] = V[1::2] - V[0::2]
    h = SEC_STEP_S / RK4_SUBSTEPS
    for k in range(1, n_nodes):
        for _ in range(RK4_SUBSTEPS):
            R, V = _rk4_step(R, V, h)
        DR[k] = R[1::2] - R[0::2]
        DV[k] = V[1::2] - V[0::2]
    for i, (_, o) in enumerate(entries):
        o["_dev"] = (DR[:, i].copy(), DV[:, i].copy())
    st["t_deviation_s"] = round(time.perf_counter() - t0, 3)

    if not catalogue and not extra_objects:
        for _, o in entries:
            o["cascade_check"] = "no_catalogue"
        st["status"] = "no_catalogue"
        return st
    try:
        from app.core.screening import _Obj, _Window, object_meta
    except Exception:
        for _, o in entries:
            o["cascade_check"] = "screening_unavailable"
        st["status"] = "screening_unavailable"
        return st

    # 2. Movers' reference grids and radial shells
    jd_g = np.full(nt, jd0)
    fr_g = fr0 + np.arange(nt) * SEC_STEP_S / 86400.0
    movers: dict[str, dict] = {}
    for _, o in entries:
        trk = o["pm"]["mover"].track
        key = str(trk.norad_id)
        m = movers.get(key)
        if m is None:
            e, r, v = trk.satrec.sgp4_array(jd_g, fr_g)
            r = np.array(r, float).reshape(nt, 3)
            v = np.array(v, float).reshape(nt, 3)
            bad = np.asarray(e).reshape(nt) != 0
            r[bad] = np.nan
            v[bad] = np.nan
            rad = np.linalg.norm(r, axis=1)
            m = movers[key] = {"track": trk, "r": r, "v": v, "rmin": float(np.nanmin(rad)),
                               "rmax": float(np.nanmax(rad)), "dv_max": 0.0}
        m["dv_max"] = max(m["dv_max"], o["dv"])

    objs, shells, seen = [], [], set()
    for trk in catalogue or []:
        if str(trk.norad_id) in seen or trk.satrec is None:
            continue
        seen.add(str(trk.norad_id))
        objs.append(_Obj(trk.norad_id, trk.name, satrec=trk.satrec))
        shells.append(_shell(sat=trk.satrec))
    for e in extra_objects or []:
        if e.get("id") is None or str(e["id"]) in seen:
            continue
        r, v = np.asarray(e["r_km"], float), np.asarray(e["v_kms"], float)
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
            continue
        seen.add(str(e["id"]))
        objs.append(_Obj(e["id"], e.get("name", str(e["id"])), r0=r, v0=v,
                         meta={k: e.get(k) for k in ("agency", "object_type") if e.get(k)}))
        shells.append(_shell(r=r, v=v))
    st["catalogue_objects"] = len(objs)
    rp = np.array([s[0] for s in shells], float)
    ra = np.array([s[1] for s in shells], float)
    ids = np.array([str(ob.id) for ob in objs], dtype=object)
    near: dict[str, np.ndarray] = {}
    keep = np.zeros(len(objs), bool)
    for key, m in movers.items():
        a_m = 0.5 * (m["rmin"] + m["rmax"])
        dr_burn = 4.0 * a_m * (m["dv_max"] / 1000.0) / math.sqrt(MU_KM3_S2 / a_m)   # ~2*da radial excursion
        marg = SHELL_MARGIN_KM + SEC_DETECT_KM + dr_burn
        ok = (rp <= m["rmax"] + marg) & (ra >= m["rmin"] - marg) & (ids != key)
        near[key] = ok
        keep |= ok
    kept_idx = np.flatnonzero(keep)
    st["screened_objects"] = int(kept_idx.size)
    if kept_idx.size == 0:
        for _, o in entries:
            o["cascade_check"] = "ok"
            o["_secondaries"] = []
        st["status"] = "ok"
        return st
    win = _Window([objs[i] for i in kept_idx], burn_epoch, nt, SEC_STEP_S)
    remap = np.full(len(objs), -1, int)
    remap[kept_idx] = np.arange(kept_idx.size)
    for key in near:
        near[key] = remap[np.flatnonzero(near[key])]
    st["t_grid_s"] = round(time.perf_counter() - t0, 3)

    meta_cache: dict[int, dict] = {}

    def meta(i):
        if i not in meta_cache:
            ob = win.objs[i]
            mm = ob.meta or {}
            meta_cache[i] = object_meta(ob.id, ob.name, mm.get("agency"), mm.get("object_type"))
        return meta_cache[i]

    mover_meta: dict[str, dict] = {}
    for key, m in movers.items():
        d = np.full((near[key].size, nt), np.inf)
        for c0 in range(0, near[key].size, 256):        # chunked: bounded temporary memory
            sl = near[key][c0:c0 + 256]
            dd = np.linalg.norm(win.R[sl] - m["r"][None], axis=2)
            d[c0:c0 + 256] = np.where(np.isfinite(dd), dd, np.inf)
        m["d_base"] = d
        mover_meta[key] = object_meta(m["track"].norad_id, m["track"].name)

    def mover_state(m, dev, t):
        e, r, v = m["track"].satrec.sgp4(jd0, fr0 + t / 86400.0)
        if e != 0:
            return None, None
        return (np.asarray(r) + _hermite(t, 0.0, SEC_STEP_S, dev[0], dev[1]),
                np.asarray(v) + _hermite_deriv(t, 0.0, SEC_STEP_S, dev[0], dev[1]))

    def detect(key, dev, exclude: set) -> list[dict]:
        m = movers[key]
        idx = near[key]
        if idx.size == 0:
            return []
        dr_n, dv_n = dev[0][:nt], dev[1][:nt]
        gate = SEC_DETECT_KM + SEC_LINEAR_MARGIN_KM + SEC_VREL_MAX_KMS * SEC_STEP_S / 2.0
        oi, kk = np.nonzero(m["d_base"] < gate + np.linalg.norm(dr_n, axis=1)[None, :])
        if oi.size == 0:
            return []
        gi = idx[oi]
        mr = m["r"][kk] + dr_n[kk]
        mv = m["v"][kk] + dv_n[kk]
        drel = win.R[gi, kk] - mr
        vrel = win.V[gi, kk] - mv
        dv2 = np.einsum("ij,ij->i", vrel, vrel)
        tstar = np.where(dv2 > 1e-12, -np.einsum("ij,ij->i", drel, vrel) / np.maximum(dv2, 1e-12), 0.0)
        half = SEC_STEP_S / 2.0
        in_bin = ((tstar >= -half) & (tstar < half)) | ((kk == nt - 1) & (tstar >= half))
        in_bin &= ~((kk == 0) & (tstar < 0))
        tcl = np.clip(tstar, -half, half)
        d_est = np.linalg.norm(drel + vrel * tcl[:, None], axis=1)
        sel = (in_bin | (dv2 < 0.05 ** 2)) & (d_est <= SEC_DETECT_KM + SEC_LINEAR_MARGIN_KM) & np.isfinite(d_est)
        if not sel.any():
            return []
        gi, kk, d_est, tcl = gi[sel], kk[sel], d_est[sel], tcl[sel]
        order = np.lexsort((d_est, gi))
        gi, kk, tcl = gi[order], kk[order], tcl[order]
        first = np.r_[True, gi[1:] != gi[:-1]]        # closest grid sample per object
        out = []
        for i, k, tc in zip(gi[first].tolist(), kk[first].tolist(), tcl[first].tolist()):
            ob = win.objs[i]
            if str(ob.id) in exclude:
                continue
            t = float(k * SEC_STEP_S + tc)
            for _ in range(6):                          # linear-relative-motion TCA refinement
                rm, vm = mover_state(m, dev, t)
                ro, vo = win.state(i, t)
                if rm is None or ro is None:
                    break
                dr_, dv_ = ro - rm, vo - vm
                q = float(dv_ @ dv_)
                shift = -float(dr_ @ dv_) / q if q > 1e-12 else 0.0
                t_new = min(max(t + max(-SEC_STEP_S, min(SEC_STEP_S, shift)), 0.0), t_end)
                if abs(t_new - t) < 1e-3:
                    break
                t = t_new
            rm, vm = mover_state(m, dev, t)
            ro, vo = win.state(i, t)
            if rm is None or ro is None:
                continue
            miss = float(np.linalg.norm(ro - rm))
            if miss > SEC_DETECT_KM:
                continue
            jd_t = jd0 + fr0 + t / 86400.0
            sat = m["track"].satrec
            age1 = jd_t - (sat.jdsatepoch + sat.jdsatepochF)
            age2 = (jd_t - ob.epoch_jd) if ob.epoch_jd is not None else t / 86400.0
            mo = meta(i)
            hbr = (mover_meta[key]["radius_m"] + mo["radius_m"]) / 1000.0
            pc = float(compute_pc(rm, vm, ro, vo, age1_days=age1, age2_days=age2, hbr_km=hbr)["pc"])
            if pc >= SEC_REPORT_PC or miss < SEC_REPORT_MISS_KM:
                out.append({"id": ob.id, "name": ob.name, "object_type": mo["object_type"],
                            "miss_km": round(miss, 4), "pc": pc, "t_s": round(t, 1),
                            "tca_utc": (burn_epoch + timedelta(seconds=t)).isoformat(),
                            "relative_speed_kms": round(float(np.linalg.norm(vo - vm)), 4)})
        out.sort(key=lambda s: (-s["pc"], s["miss_km"]))
        return out

    zero = (np.zeros((nt, 3)), np.zeros((nt, 3)))
    baselines: dict[str, dict] = {}
    checked = unchecked = 0
    for _, o in entries:
        if time.perf_counter() - t0 > budget_s:
            o["cascade_check"] = "time_budget_exceeded"
            unchecked += 1
            continue
        key = str(o["pm"]["mover"].track.norad_id)
        if key not in baselines:
            baselines[key] = {str(s["id"]): s for s in detect(key, zero, {key})}
        threat = str(o["pm"].get("other_id"))
        secs = detect(key, o["_dev"], {key, threat})
        for s in secs:
            b = baselines[key].get(str(s["id"]))
            s["baseline_pc"] = b["pc"] if b else 0.0
            s["baseline_miss_km"] = b["miss_km"] if b else None
            worsened = b is None or s["pc"] > 1.1 * b["pc"] or s["miss_km"] < 0.9 * b["miss_km"]
            s["burn_induced"] = bool(worsened)
            s["blocks_cascade_safe"] = bool(worsened and (s["pc"] >= SAFE_PC or s["miss_km"] < SAFE_MISS_KM))
        o["_secondaries"] = secs
        o["cascade_check"] = "ok"
        checked += 1
    st.update(status="ok" if unchecked == 0 else "partial_time_budget", options_checked=checked,
              options_unchecked=unchecked, t_screen_s=round(time.perf_counter() - t0, 3))
    return st


# ── Final selection ─────────────────────────────────────────────────────────

def _option_output(job: EncounterJob, o: dict, burn_epoch: datetime, pc_target: float,
                   jd0: float, fr0: float) -> dict:
    from app.core import analytic_checks as ac
    from app.core.propulsion import evaluate_engines

    pm, c = o["pm"], o["c"]
    mover: Mover = pm["mover"]
    sat = mover.track.satrec
    r0, v0 = o.get("_r0"), o.get("_v0")
    if r0 is None:
        _, r0, v0 = sat.sgp4(jd0, fr0)
        r0, v0 = np.asarray(r0, float), np.asarray(v0, float)
    dv_rsw = np.asarray(o["dv_rsw"], float)
    dv_eci = rsw_basis(r0, v0) @ dv_rsw
    model = spacecraft_mass_model(mover.track.name, mover.satcat)
    fuel = rocket_equation_cost(o["dv"], model)
    n = ac.mean_motion(ac.semi_major_axis(r0, v0))      # vis-viva a -> n = sqrt(mu/a^3)
    t_new = float(pm["t_star"][c])

    analytic = None
    dev = o.get("_dev")
    if dev is not None:
        e, rn, vn = sat.sgp4(jd0, fr0 + t_new / 86400.0)
        if e == 0:
            d_r = _hermite(t_new, 0.0, SEC_STEP_S, dev[0], dev[1])
            numeric = float(d_r @ rsw_basis(np.asarray(rn), np.asarray(vn))[:, 1])
            analytic = ac.cw_along_track_check(t_new, n, dv_rsw, numeric)
    eng = evaluate_engines(o["dv"], model["mass_kg"], lead_time_s=job.tca_s,
                           orbital_period_s=2.0 * math.pi / n)

    secs = o.get("_secondaries")
    checked = o.get("cascade_check") == "ok" and secs is not None
    fields = ("id", "name", "object_type", "miss_km", "pc", "tca_utc", "relative_speed_kms",
              "baseline_pc", "baseline_miss_km", "burn_induced", "blocks_cascade_safe")
    return {
        "candidate": o["label"],
        "sat_id": mover.track.norad_id, "sat_name": mover.track.name,
        "delta_v_rsw_ms": [round(float(x), 4) for x in dv_rsw],
        "delta_v_eci_ms": [round(float(x), 4) for x in dv_eci],
        "delta_v_ms": round(o["dv"], 4),
        "burn_epoch_utc": burn_epoch.isoformat(),
        "new_tca_utc": (burn_epoch + timedelta(seconds=t_new)).isoformat(),
        "new_miss_distance_km": round(o["miss"], 4),
        "new_pc_collision": o["pc"],
        "achieves_target": bool(o["pc"] < pc_target),
        "secondary_conjunctions": [{k: s[k] for k in fields} for s in (secs or [])[:10]],
        "secondary_count": len(secs or []),
        "secondary_max_pc": max((s["pc"] for s in secs), default=0.0) if checked else None,
        "cascade_safe": (not any(s["blocks_cascade_safe"] for s in secs)) if checked else None,
        "cascade_check": o.get("cascade_check", "not_run"),
        "propellant_kg": round(fuel["propellant_kg"], 5),
        "fuel_cost_pct": round(fuel["fuel_cost_pct"], 4),
        "engines": eng["engines"],
        "recommended_engine": eng["recommended_engine"],
        "engine_rule": eng["engine_rule"],
        "finite_burn_limits": eng["finite_burn_limits"],
        "analytic_check": analytic,
    }


def _finalize(job: EncounterJob, baseline: dict, picked: list[dict], burn_epoch: datetime, pc_target: float,
              jd0: float, fr0: float, cascade_status: str | None) -> dict[str, Any]:
    """Choose among the shortlisted options (rule documented in selection_rule).

    1. min Δv among cascade-safe options achieving Pc < target;
    2. else the largest Pc reduction among cascade-safe options;
    3. else (checked, nothing safe) flag it honestly and show the least-secondary-risk option;
    4. if the check could not run, fall back to the pair-verified min-Δv rule and say so."""
    opts = [_option_output(job, o, burn_epoch, pc_target, jd0, fr0) for o in picked]
    idx = list(range(len(opts)))

    def pref(i):
        return _option_key(picked[i])

    reach = [i for i in idx if opts[i]["achieves_target"]]
    naive = min(reach, key=pref) if reach else min(idx, key=lambda i: (opts[i]["new_pc_collision"], pref(i)))
    safe = [i for i in idx if opts[i]["cascade_safe"] is True]
    safe_reach = [i for i in safe if opts[i]["achieves_target"]]
    any_checked = any(opts[i]["cascade_safe"] is not None for i in idx)
    if safe_reach:
        chosen = min(safe_reach, key=pref)
        rule = f"min Δv among cascade-safe options achieving Pc < {pc_target:g}"
    elif safe:
        chosen = min(safe, key=lambda i: (opts[i]["new_pc_collision"], pref(i)))
        rule = f"no cascade-safe option reaches Pc < {pc_target:g}: largest Pc reduction among cascade-safe options"
    elif any_checked:
        chosen = min(idx, key=lambda i: (opts[i]["secondary_max_pc"] if opts[i]["secondary_max_pc"] is not None else 1.0,
                                         opts[i]["new_pc_collision"], pref(i)))
        rule = ("NO cascade-safe option found: every shortlisted burn creates or worsens a secondary conjunction; "
                "showing the option with the least secondary risk for operator review")
    else:
        chosen = naive
        rule = (f"cascade check not run ({cascade_status or 'n/a'}): min Δv achieving Pc < {pc_target:g}, "
                "verified on the pair only")
    ch, o = opts[chosen], picked[chosen]
    mover: Mover = o["pm"]["mover"]
    model = spacecraft_mass_model(mover.track.name, mover.satcat)
    pms = {id(x["pm"]): x["pm"] for x in picked}
    return {
        "status": "ok",
        "recommended_maneuver": {
            "sat_id": ch["sat_id"],
            "sat_name": ch["sat_name"],
            "agency": mover.agency,
            "delta_v_rsw_ms": ch["delta_v_rsw_ms"],
            "delta_v_eci_ms": ch["delta_v_eci_ms"],
            "delta_v_ms": ch["delta_v_ms"],
            "candidate": ch["candidate"],
            "burn_epoch_utc": ch["burn_epoch_utc"],
            "new_tca_utc": ch["new_tca_utc"],
            "new_miss_distance_km": ch["new_miss_distance_km"],
            "new_pc_collision": ch["new_pc_collision"],
            "pc_before": float(job.alert.get("p_collision") or job.alert.get("probability_of_collision") or 0.0),
            "pc_before_recomputed": baseline["pc"],
            "miss_before_km": round(baseline["miss_km"], 4),
            "target_pc": pc_target,
            "achieved_target": ch["achieves_target"],
            "fuel_cost_pct": ch["fuel_cost_pct"],
            "propellant_kg": ch["propellant_kg"],
            "fuel_model": "rocket_equation",
            "mass_kg": model["mass_kg"], "mass_source": model["mass_source"],
            "isp_s": model["isp_s"], "isp_source": model["isp_source"],
            "propellant_fraction": model["propellant_fraction"],
            "candidates_evaluated": int(sum(len(p["pcs"]) - 1 for p in pms.values())),
            "movers_considered": [p["mover"].track.norad_id for p in pms.values()],
            "rescreen_scope": "pair+catalogue_24h" if ch["cascade_safe"] is not None else "pair",
            "cascade_safe": ch["cascade_safe"],
            "cascade_safe_option_found": bool(safe) if any_checked else None,
            "secondary_max_pc": ch["secondary_max_pc"],
            "recommended_engine": ch["recommended_engine"],
            "analytic_check": ch["analytic_check"],
            "propagation": "sgp4_reference+j2_rk4_deviation",
            "other_object_propagation": ("fragment_state:two_body+J2+drag(app.core.breakup.propagate)"
                                         if o["pm"].get("fragment_drag") else
                                         ("sgp4" if any(t.satrec is not None for t in job.others) else "two_body+J2_rk4")),
            "verified_by": "repropagation",
            "options": opts,
            "selection_rule": rule,
            "chosen_index": chosen,
            "naive_min_dv_index": naive,
            "cascade_criteria": {
                "secondary_reported_if": f"Pc >= {SEC_REPORT_PC:g} or miss < {SEC_REPORT_MISS_KM:g} km",
                "unsafe_if": (f"a burn-induced or burn-worsened secondary with Pc >= {SAFE_PC:g} "
                              f"or miss < {SAFE_MISS_KM:g} km"),
                "window_hours": SEC_WINDOW_H, "original_threat_excluded": True},
        },
    }
