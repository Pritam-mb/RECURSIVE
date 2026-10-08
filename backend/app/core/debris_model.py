"""
debris_model.py -- collision -> NASA SBM breakup -> fragment propagation ->
fragment-vs-satellite screening -> debris clouds for the UI.

Pipeline (everything computed, nothing scripted):

1. ``simulate_collision_from_pair(a, b, propagator, sim_time)``
   * finds the pair's TCA with a vectorised SGP4 coarse scan (20 s grid) and
     refines it with ``conjunction_solver.run_to_tca`` (RK45 + J2 + Hermite
     interpolation + bounded Brent minimisation);
   * parent masses: SATCAT (``app.core.satcat.lookup``) RCS class / object type
     when available, else documented defaults (``mass_source`` says which);
   * breakup via ``app.core.breakup.simulate_breakup`` (NASA SBM, seeded with a
     CRC32 of the event id, seed recorded on the event).
2. Fragments are propagated with vectorised RK4 (two-body + J2 + drag with the
   Vallado 2013 exponential atmosphere) to the SIMULATION clock: every call
   advances to ``sim_clock.simulation_now()`` whatever wall time elapsed, and a
   backwards clock jump re-propagates from the breakup state.  If the TCA is
   still in the future the event is "pending" (no fragments exist yet).
3. ``compute_debris_alerts(states, sim_time)`` screens every live fragment
   against every satellite over a forward window (default 6 h, 30 s grid,
   KD-tree candidate search + linear relative-motion TCA refinement inside the
   step) and computes Pc for each fragment/satellite pair.
4. ``get_frontend_debris_clouds(states)`` reports the real fragment cloud
   (centroid, 50/90th percentile radius, real SBM fragment count, <=300
   rendered positions, affected satellites from step 3).

Units: km, km/s, seconds unless suffixed.
"""

from __future__ import annotations

import logging
import math
import os
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any

import numpy as np

from app.core import breakup as sbm
from app.core import sim_clock

logger = logging.getLogger(__name__)

R_EARTH_KM = sbm.R_EARTH_KM
MU_KM3_S2 = sbm.MU_KM3_S2

# ── Documented assumptions (exposed, tagged on outputs) ─────────────────────
# Mass by SATCAT radar-cross-section class (typical catalogue masses).
RCS_MASS_KG = {"SMALL": 50.0, "MEDIUM": 500.0, "LARGE": 2000.0}
# Mass by object type when no RCS is known.
TYPE_MASS_KG = {"PAY": 800.0, "R/B": 1500.0, "DEB": 20.0, "UNK": 500.0}
# Hard-body radius (m) of a satellite by RCS class / object type.
RCS_HBR_M = {"SMALL": 0.5, "MEDIUM": 1.5, "LARGE": 4.0}
TYPE_HBR_M = {"PAY": 2.0, "R/B": 3.0, "DEB": 0.5, "UNK": 2.0}

# Pc covariance model for fragment/satellite pairs (isotropic, B-plane).
SAT_SIGMA_KM = 0.2                 # catalogue satellite position 1-sigma
FRAG_SIGMA0_KM = 0.1               # fragment 1-sigma at breakup
FRAG_SIGMA_RATE_KMS = 0.001        # 1 m/s velocity knowledge -> growth per second

DEBRIS_WINDOW_HOURS = float(os.getenv("DEBRIS_WINDOW_HOURS", "6"))
DEBRIS_STEP_S = float(os.getenv("DEBRIS_STEP_S", "30"))
DEBRIS_THRESHOLD_KM = float(os.getenv("DEBRIS_THRESHOLD_KM", "5"))
DEBRIS_MAX_ALERTS = int(os.getenv("DEBRIS_MAX_ALERTS", "50"))
MAX_FRAGMENTS_PROPAGATED = int(os.getenv("DEBRIS_MAX_FRAGMENTS", "1000"))
MAX_FRAGMENTS_RENDERED = 300
MAX_REL_SPEED_KMS = 16.0           # bound used to size the KD-tree candidate radius
FRAGMENT_ID_BASE = 90_000_000      # synthetic integer ids for fragments in alerts

# Extra metadata for objects the simulation itself registers (scenario objects).
_REGISTERED_OBJECT_META: dict[int, dict[str, Any]] = {}


def register_object_meta(norad_id: int, **meta: Any) -> None:
    _REGISTERED_OBJECT_META[int(norad_id)] = dict(meta)


def unregister_object_meta(norad_id: int) -> None:
    _REGISTERED_OBJECT_META.pop(int(norad_id), None)


def _satcat_lookup(norad_id: int) -> dict | None:
    try:
        from app.core.satcat import lookup  # agent C
    except Exception:
        return None
    try:
        return lookup(int(norad_id))
    except Exception:
        return None


def _object_type_from_name(name: str | None) -> str:
    n = (name or "").upper()
    if " DEB" in n or n.endswith("DEB") or "FRAG" in n:
        return "DEB"
    if "R/B" in n or "ROCKET BODY" in n:
        return "R/B"
    return "UNK"


def object_physical_properties(norad_id: int, name: str | None = None) -> dict[str, Any]:
    """Mass / HBR / type for a catalogue object, with provenance."""
    meta = _REGISTERED_OBJECT_META.get(int(norad_id))
    if meta:
        return {
            "mass_kg": float(meta.get("mass_kg", TYPE_MASS_KG["PAY"])),
            "hbr_m": float(meta.get("hbr_m", TYPE_HBR_M["PAY"])),
            "object_type": meta.get("object_type", "PAY"),
            "mass_source": meta.get("mass_source", "scenario_config"),
        }
    sc = _satcat_lookup(norad_id)
    if sc:
        otype = (sc.get("object_type") or "UNK").upper()
        otype = {"PAYLOAD": "PAY", "ROCKET BODY": "R/B", "DEBRIS": "DEB"}.get(otype, otype)
        rcs = (sc.get("rcs_size") or "").upper()
        if rcs in RCS_MASS_KG:
            return {"mass_kg": RCS_MASS_KG[rcs], "hbr_m": RCS_HBR_M[rcs],
                    "object_type": otype, "mass_source": f"satcat_rcs_{rcs.lower()}"}
        if otype in TYPE_MASS_KG:
            return {"mass_kg": TYPE_MASS_KG[otype], "hbr_m": TYPE_HBR_M[otype],
                    "object_type": otype, "mass_source": "satcat_object_type"}
    otype = _object_type_from_name(name)
    if otype == "UNK":
        return {"mass_kg": TYPE_MASS_KG["UNK"], "hbr_m": TYPE_HBR_M["UNK"],
                "object_type": "UNK", "mass_source": "default"}
    return {"mass_kg": TYPE_MASS_KG[otype], "hbr_m": TYPE_HBR_M[otype],
            "object_type": otype, "mass_source": "default_by_name_type"}


def _agency(name: str | None, norad_id: int | None = None) -> str:
    try:
        from app.core.agency import infer_agency
        try:
            return infer_agency(name or "", norad_id=norad_id)
        except TypeError:
            return infer_agency(name or "")
    except Exception:
        return "Unknown"


def _jd_arrays(times: list[datetime]) -> tuple[np.ndarray, np.ndarray]:
    from sgp4.api import jday
    jd = np.empty(len(times))
    fr = np.empty(len(times))
    for i, t in enumerate(times):
        jd[i], fr[i] = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)
    return jd, fr


def _jd_grid(start: datetime, offsets_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    jd0, fr0 = _jd_arrays([start])
    fr = fr0[0] + offsets_s / 86400.0
    whole = np.floor(fr)
    return jd0[0] + whole, fr - whole


def _as_utc(t: datetime | None) -> datetime:
    if t is None:
        return sim_clock.simulation_now()
    if isinstance(t, str):
        t = datetime.fromisoformat(t.replace("Z", "+00:00"))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def _state_fields(state: Any) -> tuple | None:
    get = state.get if isinstance(state, dict) else (lambda k, d=None: getattr(state, k, d))
    if get("error_code", 0):
        return None
    nid = get("norad_id", None)
    if nid is None:
        nid = get("id", None)
    if nid is None:
        return None
    pos = get("position", None)
    vel = get("velocity", None)
    if isinstance(pos, dict):
        r = (pos["x"], pos["y"], pos["z"])
    else:
        r = (get("x", 0.0), get("y", 0.0), get("z", 0.0))
    if isinstance(vel, dict):
        v = (vel["vx"], vel["vy"], vel["vz"])
    else:
        v = (get("vx", 0.0), get("vy", 0.0), get("vz", 0.0))
    return int(nid), get("name", None) or f"NORAD {nid}", r, v


def _pc_isotropic(miss_km: np.ndarray, hbr_km: np.ndarray, sigma_km: np.ndarray) -> np.ndarray:
    """Foster 2-D Pc for an isotropic B-plane covariance (exact: non-central chi^2, 2 dof)."""
    try:
        from scipy.stats import ncx2
        return np.clip(ncx2.cdf((hbr_km / sigma_km) ** 2, 2, (miss_km / sigma_km) ** 2), 0.0, 1.0)
    except Exception:
        s2 = sigma_km ** 2
        return np.clip(hbr_km ** 2 / (2 * s2) * np.exp(-miss_km ** 2 / (2 * s2)), 0.0, 1.0)


def _severity(pc: float, miss_km: float) -> str:
    if pc >= 1e-4 or miss_km < 1.0:
        return "CRITICAL"
    if pc >= 1e-6 or miss_km < 5.0:
        return "WARNING"
    return "WATCH"


def _cpi_from_pc(pc: float) -> float:
    """CPI on 0..10 as a log scale of Pc: 1e-9 -> 0, 1e-4 -> 10."""
    if pc <= 0:
        return 0.0
    return float(np.clip(2.0 * (math.log10(pc) + 9.0), 0.0, 10.0))


# ── Event ────────────────────────────────────────────────────────────────────

@dataclass
class DebrisEvent:
    event_id: str
    seq: int
    collision_utc: datetime
    parent_ids: list[int]
    parent_names: list[str]
    parents: list[dict[str, Any]]
    tca: dict[str, Any]
    result: sbm.BreakupResult
    provenance: str
    created_sim_utc: str
    # Live state (at self.epoch)
    r: np.ndarray = field(default=None)
    v: np.ndarray = field(default=None)
    alive: np.ndarray = field(default=None)
    epoch: datetime = field(default=None)
    bc: np.ndarray = field(default=None)

    def __post_init__(self):
        res = self.result
        self.bc = sbm.DEFAULT_CD * res.am_m2_kg
        self._reset()

    def _reset(self):
        self.r = self.result.r_km.copy()
        self.v = self.result.v_kms.copy()
        self.alive = np.ones(len(self.r), dtype=bool)
        self.epoch = self.collision_utc

    # legacy attribute names used by older callers
    @property
    def is_catastrophic(self) -> bool:
        return self.result.catastrophic

    @property
    def predicted_count(self) -> int:
        return self.result.n_total

    @property
    def collision_point_km(self) -> list[float]:
        return [float(x) for x in self.result.impact_point_km]

    @property
    def fragments(self) -> list[dict]:
        return self.fragment_dicts()

    def pending(self, t: datetime) -> bool:
        return t < self.collision_utc

    def advance_to(self, t: datetime) -> None:
        """Propagate the live fragment state to sim time t (forward or backward)."""
        if t <= self.collision_utc:
            self._reset()
            return
        if t < self.epoch:
            self._reset()
        dt = (t - self.epoch).total_seconds()
        if dt <= 0:
            return
        self.r, self.v, self.alive = _advance(self.r, self.v, self.bc, self.alive, dt)
        self.epoch = t

    def state_at(self, t: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Copy of the fragment state at t without mutating the event (t >= collision)."""
        if t <= self.collision_utc:
            return self.result.r_km.copy(), self.result.v_kms.copy(), np.ones(len(self.result.r_km), bool)
        if t >= self.epoch:
            r, v, alive = self.r, self.v, self.alive
            dt = (t - self.epoch).total_seconds()
        else:
            r, v, alive = self.result.r_km, self.result.v_kms, np.ones(len(self.result.r_km), bool)
            dt = (t - self.collision_utc).total_seconds()
        return _advance(r.copy(), v.copy(), self.bc, alive.copy(), dt)

    def fragment_dicts(self) -> list[dict]:
        res = self.result
        alt = np.linalg.norm(self.r, axis=1) - R_EARTH_KM
        epoch = self.epoch.isoformat()
        return [
            {
                "fragment_id": f"{self.event_id}:F{i:04d}",
                "size_m": round(float(res.lc_m[i]), 3),
                "mass_kg": round(float(res.mass_kg[i]), 4),
                "area_to_mass_m2_kg": round(float(res.am_m2_kg[i]), 4),
                "x": round(float(self.r[i, 0]), 3), "y": round(float(self.r[i, 1]), 3), "z": round(float(self.r[i, 2]), 3),
                "vx": round(float(self.v[i, 0]), 5), "vy": round(float(self.v[i, 1]), 5), "vz": round(float(self.v[i, 2]), 5),
                "altitude_km": round(float(alt[i]), 2),
                "is_decayed": bool(not self.alive[i]),
                "epoch_utc": epoch,
            }
            for i in range(len(self.r))
        ]

    def to_summary(self) -> dict:
        res = self.result
        return {
            "event_id": self.event_id,
            "collision_utc": self.collision_utc.isoformat(),
            "parent_ids": self.parent_ids,
            "parent_names": self.parent_names,
            "parents": self.parents,
            "tca": self.tca,
            "provenance": self.provenance,
            "collision_point_km": self.collision_point_km,
            "is_catastrophic": res.catastrophic,
            "emr_j_per_g": round(res.emr_j_per_g, 3),
            "catastrophic_threshold_j_per_g": sbm.CATASTROPHIC_EMR_J_PER_KG / 1000.0,
            "sbm_reference_mass_kg": round(res.m_ref_kg, 2),
            "lc_min_m": res.lc_min_m,
            "lc_max_m": round(res.lc_max_m, 3),
            "total_fragments": res.n_total,
            "fragment_count_formula": "N(>=Lc) = 0.1 * M^0.75 * Lc^-1.71",
            "simulated_fragments": res.n_sampled,
            "fragment_weight": round(res.weight, 4),
            "active_fragments": int(self.alive.sum()),
            "decayed_fragments": int((~self.alive).sum()),
            "suborbital_at_breakup": res.n_suborbital,
            "dv_capped_to_bound_orbit": res.n_dv_capped,
            "momentum_residual_before": round(res.momentum_residual_before, 6),
            "momentum_residual_after": round(res.momentum_residual_after, 6),
            "represented_fragment_mass_kg": round(res.represented_mass_kg, 2),
            "seed": res.seed,
            "stats": {k: round(v, 4) for k, v in res.stats.items()},
            "created_sim_utc": self.created_sim_utc,
            "state_epoch_utc": self.epoch.isoformat(),
            "rel_vel_kms": round(res.stats.get("relative_velocity_kms", 0.0), 4),
            "created_at": self.created_sim_utc,
        }


def _bound_ok(r: np.ndarray) -> np.ndarray:
    """Rows still in orbit: finite and above the 100 km re-entry altitude."""
    rn = np.linalg.norm(r, axis=1)
    # 1e5 km bound: safety net, bound fragments from LEO never reach it
    return np.isfinite(rn) & (rn - R_EARTH_KM >= sbm.REENTRY_ALT_KM) & (rn < 1.0e5)


def _advance(r, v, bc, alive, dt_s, chunk_s: float | None = None):
    """Propagate alive rows over dt_s; rows dropping below 100 km are marked decayed."""
    r = r.copy()
    v = v.copy()
    alive = alive.copy()
    remaining = float(dt_s)
    sign = 1.0 if remaining >= 0 else -1.0
    chunk_s = DEBRIS_STEP_S if chunk_s is None else chunk_s  # decay checked every step
    while abs(remaining) > 1e-9 and alive.any():
        step = sign * min(chunk_s, abs(remaining))
        idx = np.nonzero(alive)[0]
        r_new, v_new = sbm.propagate(r[idx], v[idx], bc[idx], step, max_step_s=DEBRIS_STEP_S)
        r[idx], v[idx] = r_new, v_new
        alive[idx[~_bound_ok(r_new)]] = False
        remaining -= step
    return r, v, alive


# ── Model ────────────────────────────────────────────────────────────────────

class DebrisModel:
    def __init__(self):
        self._events: dict[str, DebrisEvent] = {}
        self._lock = RLock()
        self._seq = 0
        self._propagator = None
        self._last_alerts: list[dict] = []
        self._last_exposure: dict[str, list[dict]] = {}
        self._last_timeline: dict[str, list[dict]] = {}
        self._last_screen_meta: dict[str, Any] = {}

    # ── creation ─────────────────────────────────────────────────────────
    def _register(self, event_id, collision_utc, parent_ids, parent_names, parents, tca, res, provenance):
        with self._lock:
            self._seq += 1
            ev = DebrisEvent(
                event_id=event_id, seq=self._seq, collision_utc=collision_utc,
                parent_ids=parent_ids, parent_names=parent_names, parents=parents, tca=tca,
                result=res, provenance=provenance,
                created_sim_utc=sim_clock.simulation_now().isoformat(),
            )
            self._events[event_id] = ev
        logger.info("Debris event %s: SBM N=%d (sampled %d), catastrophic=%s, EMR=%.1f J/g",
                    event_id, res.n_total, res.n_sampled, res.catastrophic, res.emr_j_per_g)
        return ev

    def simulate_collision(
        self,
        collision_point_eci_km: list[float],
        collision_velocity_eci_kms: list[float],
        mass_p_kg: float = 500.0,
        mass_q_kg: float = 500.0,
        rel_vel_kms: float = 10.0,
        max_fragments_simulated: int = MAX_FRAGMENTS_PROPAGATED,
        event_id: str | None = None,
        collision_utc: datetime | None = None,
    ) -> DebrisEvent:
        """Manual-state breakup (kept for scripts/tests): parent B moves at
        ``rel_vel_kms`` relative to parent A, perpendicular to A's velocity."""
        r = np.asarray(collision_point_eci_km, float)
        v_a = np.asarray(collision_velocity_eci_kms, float)
        h = np.cross(r, v_a)
        w = h / np.linalg.norm(h)
        v_b = v_a + rel_vel_kms * w
        t = _as_utc(collision_utc)
        event_id = event_id or f"manual-{t:%Y%m%dT%H%M%SZ}"
        seed = zlib.crc32(event_id.encode())
        res = sbm.simulate_breakup(r, v_a, mass_p_kg, r, v_b, mass_q_kg, seed=seed,
                                   max_fragments=max_fragments_simulated)
        parents = [{"id": None, "mass_kg": mass_p_kg, "mass_source": "caller"},
                   {"id": None, "mass_kg": mass_q_kg, "mass_source": "caller"}]
        return self._register(event_id, t, [], [], parents,
                              {"tca_utc": t.isoformat(), "relative_velocity_kms": rel_vel_kms},
                              res, "manual_state")

    def simulate_collision_from_pair(
        self,
        sat_a_id: int,
        sat_b_id: int,
        propagator,
        sim_time: datetime | None = None,
        *,
        tca_hint_utc: str | datetime | None = None,
        window_hours: float = 24.0,
        max_fragments: int = MAX_FRAGMENTS_PROPAGATED,
    ) -> dict:
        """Breakup of a catalogue pair at their predicted TCA."""
        sim_time = _as_utc(sim_time)
        self._propagator = propagator
        tca = find_pair_tca(propagator, int(sat_a_id), int(sat_b_id), sim_time,
                            window_hours=window_hours, tca_hint_utc=tca_hint_utc)
        names = tca["names"]
        props = [object_physical_properties(sat_a_id, names[0]), object_physical_properties(sat_b_id, names[1])]
        tca_time = _as_utc(tca["tca_utc"])
        event_id = f"evt-{min(sat_a_id, sat_b_id)}-{max(sat_a_id, sat_b_id)}-{tca_time:%Y%m%dT%H%M%SZ}"
        seed = zlib.crc32(event_id.encode())
        res = sbm.simulate_breakup(
            tca["position_a_eci"], tca["velocity_a_eci"], props[0]["mass_kg"],
            tca["position_b_eci"], tca["velocity_b_eci"], props[1]["mass_kg"],
            seed=seed,
            rocket_body_a=props[0]["object_type"] == "R/B",
            rocket_body_b=props[1]["object_type"] == "R/B",
            max_fragments=max_fragments,
        )
        parents = [
            {"id": int(sat_a_id), "name": names[0], **props[0]},
            {"id": int(sat_b_id), "name": names[1], **props[1]},
        ]
        tca_public = {k: v for k, v in tca.items() if k not in ("names",)}
        ev = self._register(event_id, tca_time, [int(sat_a_id), int(sat_b_id)], list(names),
                            parents, tca_public, res, "predicted_tca_pair")
        ev.advance_to(sim_time)
        return ev.to_summary()

    # ── propagation ──────────────────────────────────────────────────────
    def propagate_fragments(self, event_id: str, dt_seconds: float | None = None) -> list[dict]:
        """Advance the event to the current SIMULATION time.

        ``dt_seconds`` is accepted for backwards compatibility but ignored: the
        step is always (sim_now - state epoch), so clock offsets/jumps are
        honoured exactly instead of assuming 1 s per wall-clock tick."""
        ev = self._events.get(event_id)
        if ev is None:
            return []
        now = sim_clock.simulation_now()
        with self._lock:
            ev.advance_to(now)
        return []

    def advance_all(self, sim_time: datetime | None = None) -> None:
        t = _as_utc(sim_time)
        with self._lock:
            for ev in self._events.values():
                ev.advance_to(t)

    # ── queries ──────────────────────────────────────────────────────────
    def get_cloud(self, event_id: str) -> DebrisEvent | None:
        return self._events.get(event_id)

    def get_all_clouds(self) -> dict[str, dict]:
        return {eid: ev.to_summary() for eid, ev in self._events.items()}

    def list_event_ids(self) -> list[str]:
        return list(self._events.keys())

    def clear(self):
        with self._lock:
            self._events.clear()
            self._last_alerts = []
            self._last_exposure = {}
            self._last_timeline = {}
            self._last_screen_meta = {}

    @property
    def last_screen_meta(self) -> dict:
        return dict(self._last_screen_meta)

    # ── screening ────────────────────────────────────────────────────────
    def compute_debris_alerts(
        self,
        satellite_states: list,
        sim_time: datetime | None = None,
        *,
        window_hours: float | None = None,
        step_s: float | None = None,
        threshold_km: float | None = None,
        max_alerts: int | None = None,
    ) -> list[dict]:
        """Screen all live fragments of all events against all satellites."""
        import time as _time
        t_start = _time.perf_counter()
        sim_time = _as_utc(sim_time)
        window_hours = DEBRIS_WINDOW_HOURS if window_hours is None else float(window_hours)
        step_s = DEBRIS_STEP_S if step_s is None else float(step_s)
        threshold_km = DEBRIS_THRESHOLD_KM if threshold_km is None else float(threshold_km)
        max_alerts = DEBRIS_MAX_ALERTS if max_alerts is None else int(max_alerts)

        with self._lock:
            events = list(self._events.values())
        if not events:
            self._last_alerts, self._last_exposure, self._last_timeline = [], {}, {}
            return []

        sats = [s for s in (_state_fields(st) for st in satellite_states or []) if s is not None]
        n_steps = int(window_hours * 3600.0 / step_s)
        offsets = np.arange(n_steps + 1) * step_s

        # Fragments: state at sim_time. Pending events are back-propagated from
        # breakup to sim_time and masked until their release step.
        fr_r, fr_v, fr_bc, fr_alive, fr_ev, fr_idx, fr_release = [], [], [], [], [], [], []
        for k, ev in enumerate(events):
            if ev.pending(sim_time):
                dt_back = (sim_time - ev.collision_utc).total_seconds()
                if dt_back < -window_hours * 3600:
                    continue
                # Fragments do not exist before breakup: start them at the first
                # grid step after the TCA (propagated by the sub-step remainder)
                # and keep them frozen until then.
                release = int(math.ceil(-dt_back / step_s))
                r, v, alive = _advance(ev.result.r_km, ev.result.v_kms, ev.bc,
                                       np.ones(len(ev.result.r_km), bool), release * step_s + dt_back)
            else:
                with self._lock:
                    r, v, alive = ev.state_at(sim_time)
                release = 0
            n = len(r)
            fr_r.append(r); fr_v.append(v); fr_bc.append(ev.bc); fr_alive.append(alive)
            fr_ev.append(np.full(n, k)); fr_idx.append(np.arange(n)); fr_release.append(np.full(n, release))
        if not fr_r or not sats:
            self._last_alerts, self._last_exposure = [], {}
            return []
        F_r = np.concatenate(fr_r); F_v = np.concatenate(fr_v); F_bc = np.concatenate(fr_bc)
        F_alive = np.concatenate(fr_alive); F_ev = np.concatenate(fr_ev)
        F_idx = np.concatenate(fr_idx); F_rel = np.concatenate(fr_release)

        sat_ids = np.array([s[0] for s in sats])
        sat_names = [s[1] for s in sats]
        S_r_all, S_v_all = self._satellite_tracks(sats, sim_time, offsets)

        # exclude each event's own parents
        parent_ids = [set(ev.parent_ids) for ev in events]

        from scipy.spatial import cKDTree
        cand_radius = threshold_km + MAX_REL_SPEED_KMS * step_s / 2.0
        best: dict[tuple[int, int], tuple] = {}
        timeline: dict[int, list[dict]] = {k: [] for k in range(len(events))}
        timeline_every = max(1, int(round(1800.0 / step_s)))
        r_cur, v_cur, alive = F_r, F_v, F_alive
        for k in range(n_steps + 1):
            if k > 0:
                idx = np.nonzero(alive & (k > F_rel))[0]
                rn, vn = sbm.propagate(r_cur[idx], v_cur[idx], F_bc[idx], step_s, max_step_s=step_s)
                r_cur = r_cur.copy(); v_cur = v_cur.copy(); alive = alive.copy()
                r_cur[idx], v_cur[idx] = rn, vn
                alive[idx[~_bound_ok(rn)]] = False
            active = alive & (k >= F_rel)
            if k % timeline_every == 0:
                for e in range(len(events)):
                    sel = active & (F_ev == e)
                    if sel.sum() >= 3:
                        c = r_cur[sel].mean(axis=0)
                        d = np.linalg.norm(r_cur[sel] - c, axis=1)
                        timeline[e].append({"minutes": round(k * step_s / 60.0, 1),
                                            "radius_km": round(float(np.percentile(d, 90)), 2),
                                            "radius_p50_km": round(float(np.percentile(d, 50)), 2)})
            ai = np.nonzero(active)[0]
            if ai.size == 0:
                continue
            S_r = S_r_all[:, k, :]
            S_v = S_v_all[:, k, :]
            ok = np.isfinite(S_r[:, 0])
            tree_s = cKDTree(np.where(ok[:, None], S_r, 1e9))
            tree_f = cKDTree(r_cur[ai])
            pairs = tree_f.sparse_distance_matrix(tree_s, cand_radius, output_type="ndarray")
            if len(pairs) == 0:
                continue
            fi = ai[pairs["i"]]
            sj = pairs["j"]
            dr = r_cur[fi] - S_r[sj]
            dv = v_cur[fi] - S_v[sj]
            dv2 = np.einsum("ij,ij->i", dv, dv)
            tstar = np.clip(-np.einsum("ij,ij->i", dr, dv) / np.maximum(dv2, 1e-12), -step_s / 2, step_s / 2)
            miss_vec = dr + dv * tstar[:, None]
            miss = np.linalg.norm(miss_vec, axis=1)
            keep = miss < threshold_km
            for q in np.nonzero(keep)[0]:
                f, s = int(fi[q]), int(sj[q])
                if int(sat_ids[s]) in parent_ids[F_ev[f]]:
                    continue
                key = (f, s)
                if key not in best or miss[q] < best[key][0]:
                    best[key] = (float(miss[q]), k * step_s + float(tstar[q]), dv[q].copy(),
                                 miss_vec[q].copy(), S_r[s].copy())

        alerts = []
        exposure: dict[int, dict] = {}
        for (f, s), (miss, t_off, dv, miss_vec, sat_r) in best.items():
            ev = events[F_ev[f]]
            fidx = int(F_idx[f])
            lc = float(ev.result.lc_m[fidx])
            sid = int(sat_ids[s])
            props = object_physical_properties(sid, sat_names[s])
            hbr_km = (props["hbr_m"] + lc / 2.0) / 1000.0
            tca_time = sim_time + timedelta(seconds=t_off)
            age_s = max(0.0, (tca_time - ev.collision_utc).total_seconds())
            sig_f = math.hypot(FRAG_SIGMA0_KM, FRAG_SIGMA_RATE_KMS * age_s)
            sigma = math.hypot(SAT_SIGMA_KM, sig_f)
            pc = float(_pc_isotropic(np.array([miss]), np.array([hbr_km]), np.array([sigma]))[0])
            vrel = float(np.linalg.norm(dv))
            zhat = dv / max(vrel, 1e-12)
            xi = np.cross(zhat, sat_r); xi /= max(np.linalg.norm(xi), 1e-12)
            zeta = np.cross(zhat, xi)
            frag_int_id = FRAGMENT_ID_BASE + ev.seq * 10_000 + fidx
            alert = {
                "id": f"debris:{ev.event_id}:{fidx}-{sid}",
                "source": "debris",
                "sat1": {"id": sid, "name": sat_names[s], "agency": _agency(sat_names[s], sid),
                         "object_type": props["object_type"]},
                "sat2": {"id": frag_int_id, "name": f"FRAG {ev.event_id} #{fidx}",
                         "agency": "Debris", "object_type": "DEB"},
                "tca_utc": tca_time.isoformat(),
                "tca_hours": round(t_off / 3600.0, 4),
                "tca_minutes": round(t_off / 60.0, 2),
                "miss_distance_km": round(miss, 4),
                "relative_speed_kms": round(vrel, 4),
                "relative_speed_kmh": round(vrel * 3600.0, 1),
                "probability_of_collision": pc,
                "p_collision": pc,
                "pc_method": "foster",
                "covariance_model": "isotropic_bplane",
                "hbr_km": round(hbr_km, 6),
                "covariance_ellipse": {"a": round(sigma * 1000, 1), "b": round(sigma * 1000, 1), "angle": 0.0,
                                       "sigma_sat_km": SAT_SIGMA_KM, "sigma_fragment_km": round(sig_f, 4)},
                "b_t_km": round(float(miss_vec @ xi), 4),
                "b_n_km": round(float(miss_vec @ zeta), 4),
                "sigma_source": "fragment_growth_model",
                "severity": _severity(pc, miss),
                "cpi_score": round(_cpi_from_pc(pc), 3),
                "cpi_method": "log10_pc_scale",
                "ml": None,
                "parent_event": {"event_id": ev.event_id, "collision_utc": ev.collision_utc.isoformat(),
                                 "parent_ids": ev.parent_ids, "fragment_count": ev.result.n_total},
                "fragment_id": f"{ev.event_id}:F{fidx:04d}",
                "fragment_size_m": round(lc, 3),
                "fragment_weight": round(ev.result.weight, 4),
                "sat_mass_source": props["mass_source"],
            }
            alerts.append(alert)
            agg = exposure.setdefault((F_ev[f], sid), {
                "norad_id": sid, "name": sat_names[s], "event_id": ev.event_id,
                "fragments_within_threshold": 0, "represented_fragments": 0.0,
                "max_pc": 0.0, "expected_hits": 0.0, "min_miss_km": float("inf"),
                "earliest_tca_utc": None, "_earliest": float("inf"),
            })
            agg["fragments_within_threshold"] += 1
            agg["represented_fragments"] += ev.result.weight
            agg["max_pc"] = max(agg["max_pc"], pc)
            agg["expected_hits"] += ev.result.weight * pc
            agg["min_miss_km"] = min(agg["min_miss_km"], miss)
            if t_off < agg["_earliest"]:
                agg["_earliest"] = t_off
                agg["earliest_tca_utc"] = tca_time.isoformat()

        exposure_by_event: dict[str, list[dict]] = {}
        exp_by_sat: dict[tuple[str, int], dict] = {}
        for (_, sid), agg in exposure.items():
            agg.pop("_earliest", None)
            agg["aggregate_pc"] = float(-math.expm1(-agg["expected_hits"]))
            agg["represented_fragments"] = round(agg["represented_fragments"], 2)
            agg["min_miss_km"] = round(agg["min_miss_km"], 4)
            agg["severity"] = _severity(agg["aggregate_pc"], agg["min_miss_km"])
            agg["cpi_score"] = round(_cpi_from_pc(agg["aggregate_pc"]), 3)
            agg["risk_band"] = {"CRITICAL": "high", "WARNING": "medium"}.get(agg["severity"], "low")
            agg["recommended_action"] = "fallback" if agg["severity"] == "CRITICAL" else "monitor"
            exposure_by_event.setdefault(agg["event_id"], []).append(agg)
            exp_by_sat[(agg["event_id"], sid)] = agg
        for lst in exposure_by_event.values():
            lst.sort(key=lambda a: a["aggregate_pc"], reverse=True)

        alerts.sort(key=lambda a: (a["probability_of_collision"], -a["miss_distance_km"]), reverse=True)
        total = len(alerts)
        alerts = alerts[:max_alerts]
        for a in alerts:
            agg = exp_by_sat.get((a["parent_event"]["event_id"], a["sat1"]["id"]))
            if agg:
                a["debris_aggregate"] = {k: agg[k] for k in ("fragments_within_threshold", "represented_fragments",
                                                             "max_pc", "aggregate_pc", "expected_hits", "min_miss_km")}

        elapsed = _time.perf_counter() - t_start
        with self._lock:
            self._last_alerts = alerts
            self._last_exposure = exposure_by_event
            self._last_timeline = {events[k].event_id: v for k, v in timeline.items()}
            self._last_screen_meta = {
                "sim_time": sim_time.isoformat(), "window_hours": window_hours, "step_s": step_s,
                "threshold_km": threshold_km, "fragments_screened": int(len(F_r)),
                "satellites_screened": len(sats), "pairs_within_threshold": total,
                "alerts_returned": len(alerts), "satellites_exposed": len(exposure),
                "elapsed_s": round(elapsed, 3),
            }
        logger.info("Debris screening: %d fragments x %d sats over %.1f h -> %d pairs (%.2f s)",
                    len(F_r), len(sats), window_hours, total, elapsed)
        return alerts

    def _satellite_tracks(self, sats, sim_time, offsets):
        """Satellite positions/velocities on the grid: SGP4 (SatrecArray) when the
        propagator knows the object, else RK4 two-body + J2 from the given state."""
        m, nt = len(sats), len(offsets)
        R = np.full((m, nt, 3), np.nan)
        V = np.full((m, nt, 3), np.nan)
        sgp_rows, satrecs = [], []
        prop = self._propagator
        if prop is not None:
            try:
                with prop._lock:
                    table = dict(prop._satellites)
                for i, s in enumerate(sats):
                    entry = table.get(s[0])
                    if entry is not None:
                        sgp_rows.append(i)
                        satrecs.append(entry[0])
            except Exception:
                sgp_rows, satrecs = [], []
        if satrecs:
            from sgp4.api import SatrecArray
            jd, fr = _jd_grid(sim_time, offsets)
            err, r, v = SatrecArray(satrecs).sgp4(jd, fr)
            r = np.where((err == 0)[..., None], r, np.nan)
            R[sgp_rows] = r
            V[sgp_rows] = v
        rest = [i for i in range(m) if i not in set(sgp_rows)]
        if rest:
            r = np.array([sats[i][2] for i in rest], float)
            v = np.array([sats[i][3] for i in rest], float)
            R[rest, 0], V[rest, 0] = r, v
            for k in range(1, nt):
                r, v = sbm.propagate(r, v, None, offsets[k] - offsets[k - 1], max_step_s=60.0)
                R[rest, k], V[rest, k] = r, v
        return R, V

    # ── UI ───────────────────────────────────────────────────────────────
    def get_frontend_debris_clouds(self, states: list | None = None) -> list[dict]:
        now = sim_clock.simulation_now()
        out = []
        with self._lock:
            events = list(self._events.values())
            exposure = dict(self._last_exposure)
            timelines = dict(self._last_timeline)
        for ev in events:
            res = ev.result
            pending = ev.pending(now)
            if pending:
                r_alive = np.zeros((0, 3))
                center = res.impact_point_km
            else:
                r_alive = ev.r[ev.alive]
                center = r_alive.mean(axis=0) if len(r_alive) else res.impact_point_km
            if len(r_alive):
                d = np.linalg.norm(r_alive - center, axis=1)
                p50, p90, pmax = (float(np.percentile(d, q)) for q in (50, 90, 100))
                step = max(1, int(math.ceil(len(r_alive) / MAX_FRAGMENTS_RENDERED)))
                frags = np.round(r_alive[::step][:MAX_FRAGMENTS_RENDERED], 2).tolist()
            else:
                p50 = p90 = pmax = 0.0
                frags = []
            affected = exposure.get(ev.event_id, [])
            tl = timelines.get(ev.event_id) or []
            if len(tl) >= 3:
                radius_timeline = [tl[0], tl[len(tl) // 2], tl[-1]]
            else:
                radius_timeline = [{"minutes": 0.0, "radius_km": round(p90, 2)}]
            minutes_to_tca = max(0.0, (ev.collision_utc - now).total_seconds() / 60.0)
            alive_fraction = float(ev.alive.mean()) if len(ev.alive) else 0.0
            out.append({
                "id": ev.event_id,
                "kind": "fragments",
                "status": "pending_impact" if pending else "active",
                "tca_utc": ev.collision_utc.isoformat(),
                "epoch_utc": (now if not pending else ev.collision_utc).isoformat(),
                "minutes_to_tca": round(minutes_to_tca, 2),
                "center_eci_km": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
                "centroid_eci_km": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
                "impact_point_eci_km": [float(x) for x in res.impact_point_km],
                "radius_km_now": round(p90, 3),
                "radius_km_p50": round(p50, 3),
                "radius_km_p90": round(p90, 3),
                "radius_p90_km": round(p90, 3),
                "radius_p50_km": round(p50, 3),
                "percentile_radius_km": round(p90, 3),
                "radius_percentile": 90,
                "radius_km_at_tca": round(p90, 3),
                "max_radius_km": round(pmax, 3),
                "fragment_count": int(round(res.n_total * alive_fraction)) if not pending else res.n_total,
                "fragment_count_at_breakup": res.n_total,
                "fragments_simulated": int(ev.alive.sum()),
                "fragment_weight": round(res.weight, 4),
                "fragments": frags,
                "parent_event": {"event_id": ev.event_id, "collision_utc": ev.collision_utc.isoformat(),
                                 "parent_ids": ev.parent_ids, "parent_names": ev.parent_names,
                                 "fragment_count": res.n_total, "is_catastrophic": res.catastrophic,
                                 "seed": res.seed},
                "affected_satellites": affected[:25],
                "affected_count": len(affected),
                "affected_high_risk": sum(1 for a in affected if a.get("severity") == "CRITICAL"),
                "radius_timeline": radius_timeline,
                "shells": [
                    {"label": "p50", "radius_km": round(max(p50, 0.5), 3)},
                    {"label": "p90", "radius_km": round(max(p90, 1.0), 3)},
                ],
            })
        return out


# ── TCA for a catalogue pair ────────────────────────────────────────────────

def find_pair_tca(propagator, a_id: int, b_id: int, start: datetime, *, window_hours: float = 24.0,
                  tca_hint_utc=None, coarse_step_s: float = 20.0) -> dict:
    """Vectorised SGP4 coarse scan then conjunction_solver.run_to_tca refinement."""
    from sgp4.api import SatrecArray
    from app.services.conjunction_solver import parse_eci_state, run_to_tca

    with propagator._lock:
        ea = propagator._satellites.get(int(a_id))
        eb = propagator._satellites.get(int(b_id))
    if ea is None or eb is None:
        raise KeyError(f"satellite {a_id if ea is None else b_id} not tracked")
    start = _as_utc(start)
    if tca_hint_utc is not None:
        hint = _as_utc(tca_hint_utc)
        scan_start = max(start, hint - timedelta(minutes=30))
        span = (hint + timedelta(minutes=30) - scan_start).total_seconds()
    else:
        scan_start = start
        span = window_hours * 3600.0
    offsets = np.arange(0.0, max(span, coarse_step_s) + coarse_step_s, coarse_step_s)
    jd, fr = _jd_grid(scan_start, offsets)
    err, r, v = SatrecArray([ea[0], eb[0]]).sgp4(jd, fr)
    d = np.linalg.norm(r[0] - r[1], axis=1)
    d[(err[0] != 0) | (err[1] != 0)] = np.inf
    k = int(np.argmin(d))
    if not np.isfinite(d[k]):
        raise ValueError("propagation failed for pair")
    k0 = max(0, k - 1)
    t0 = scan_start + timedelta(seconds=float(offsets[k0]))
    t1 = scan_start + timedelta(seconds=float(offsets[min(k + 1, len(offsets) - 1)]))
    sa = parse_eci_state(r[0, k0], v[0, k0], t0)
    sb = parse_eci_state(r[1, k0], v[1, k0], t0)
    res = run_to_tca(sa, sb, t1, step_seconds=1.0, max_steps=200, use_j2=True)
    return {
        "tca_utc": res.tca_utc,
        "miss_distance_m": res.miss_distance_m,
        "relative_velocity_kms": res.relative_velocity_kms,
        "position_a_eci": res.position_a_eci, "velocity_a_eci": res.velocity_a_eci,
        "position_b_eci": res.position_b_eci, "velocity_b_eci": res.velocity_b_eci,
        "coarse_min_km": round(float(d[k]), 4),
        "method": "sgp4_scan_20s+run_to_tca(rk45_j2,hermite,brent)",
        "names": (ea[1], eb[1]),
    }


# ── Module-level singleton & contract functions ─────────────────────────────
debris_model = DebrisModel()


def compute_debris_alerts(satellite_states, sim_time=None, **kwargs) -> list[dict]:
    return debris_model.compute_debris_alerts(satellite_states, sim_time, **kwargs)


def get_frontend_debris_clouds(active_states=None) -> list[dict]:
    return debris_model.get_frontend_debris_clouds(active_states)


def simulate_collision_from_pair(sat_a_id, sat_b_id, propagator, sim_time=None, **kwargs) -> dict:
    return debris_model.simulate_collision_from_pair(sat_a_id, sat_b_id, propagator, sim_time, **kwargs)


def atmospheric_density(altitude_km: float) -> float:
    """Back-compat scalar wrapper around the Vallado exponential table."""
    return float(sbm.atmospheric_density(altitude_km))
