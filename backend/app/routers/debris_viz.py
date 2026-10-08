"""
/api/debris/events — debris-event catalogue and physically propagated impact
replays for the interactive impact visualisation (sprint 3).

GET /api/debris/events
    Every breakup event held by ``app.core.debris_model.debris_model`` with its
    status on the SIMULATION clock ("pending_impact" before the TCA, "active"
    after), parents, SBM counts and the collision point.

GET /api/debris/events/{event_id}/replay?t0_min=-15&t1_min=180&step_s=30&max_fragments=400
    Time-sampled positions (ECI/TEME km, rounded to 0.1 km) on the grid
    t_rel = k * step_s  (k integer, t0_min*60 <= t_rel <= t1_min*60), i.e. the
    collision instant t_rel = 0 is always a sample.

How each number is produced (nothing is tweened or scripted)
------------------------------------------------------------
* Breakup state.  ``debris_model.simulate_collision_from_pair`` computes the
  NASA-SBM breakup at the predicted TCA *when the event is created*, seeded with
  CRC32(event_id) (``event.result``).  That state exists for pending events
  too, so a replay of a ``pending_impact`` event shows exactly the fragments the
  live model will release at the TCA (same seed, same arrays).
* Fragments.  ALL simulated fragments of the event (<= 1000) are integrated from
  the breakup state with the live model's own integrator
  (``debris_model._advance`` -> ``breakup.propagate``: RK4, two-body + J2 + drag
  with the Vallado 2013 exponential atmosphere, <= 30 s steps, decay below
  100 km).  The loop runs over the time samples only; every step is vectorised
  over fragments.  Positions are null before t_rel = 0 and after decay.
* Display sample (<= max_fragments) — stratified, deterministic:
    1. fragments named in this event's current debris alerts are always kept
       (at most a quarter of the budget) so the threatening fragment's path is
       drawn;
    2. the rest is allocated over strata = parent (A/B) x size quartile of
       Lc (largest-remainder proportional allocation, >= 1 per non-empty
       stratum);
    3. inside a stratum fragments are sorted by ejection |dV| and taken at
       evenly spaced ranks, so the sample spans the stratum's dV range.
* Envelope statistics use ALL simulated fragments (not only the display
  sample): arithmetic centroid, 50th/90th percentile distance from it,
  along-track spread = arc length (mean radius x angle) between the 5th and
  95th percentile in-plane angle about the mean orbit normal, measured from the
  centroid direction.  Note the arithmetic centroid moves towards the Earth's
  centre once the cloud wraps around the orbit.
* Parents.  SGP4 of the propagator's TLE plus executed-burn deviations
  (``sgp4_propagator.add_burn_offsets``) at every sample, before AND after the
  collision: after t_rel = 0 they show the trajectories the parents would have
  flown had they not collided.
* Threatened satellites come from the debris model's last fragment screening
  (``debris_model.compute_debris_alerts``) for this event.

Caching: the fragment/envelope part is cached per (event, seed, collision time,
params, forced alert-fragment set) in a small LRU; parents and the threatened
list are recomputed on every request (cheap) so executed burns / fresh
screening show up immediately.  Heavy work runs in ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.core import breakup as sbm
from app.core import sim_clock

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/debris", tags=["debris-visualisation"])

FRAME = "ECI (TEME≈ECI) km"
PROPAGATOR_LABEL = "two-body+J2+drag (Vallado exp. atmosphere)"
MAX_DISPLAY_FRAGMENTS = 400          # contract: <= 400 animated fragment points
MAX_SAMPLES = 2000
N_SIZE_BINS = 4
THREAT_SHARE = 0.25                  # at most this share of the budget is forced alert fragments
CACHE_SIZE = 6
# Payload guard: fragment position cells (samples x displayed fragments). 160k cells ~ 3.4 MB
# (the default 391 x 400 request). Larger windows / finer steps shrink the display sample
# (reported as provenance.display_budget_reduced) instead of shipping > ~6 MB to a laptop.
MAX_POSITION_CELLS = 250_000

_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_cache_lock = threading.Lock()


# ── Accessors (monkeypatched in tests) ─────────────────────────────────────

def _model():
    from app.core.debris_model import debris_model
    return debris_model


def _propagator():
    try:
        from app.api import routes
        if getattr(routes, "_propagator", None) is not None:
            return routes._propagator
    except Exception:  # pragma: no cover - routes import guard
        pass
    return getattr(_model(), "_propagator", None)


def _agency(name: str | None, norad_id: int | None) -> str:
    try:
        from app.core.debris_model import _agency as agency_fn
        return agency_fn(name, norad_id)
    except Exception:  # pragma: no cover
        return "Unknown"


def _events(model) -> list:
    lock = getattr(model, "_lock", None)
    if lock is not None:
        with lock:
            return list(model._events.values())
    return list(model._events.values())


def _event_alerts(model, event_id: str) -> list[dict]:
    lock = getattr(model, "_lock", None)
    if lock is not None:
        with lock:
            alerts = list(getattr(model, "_last_alerts", []) or [])
    else:
        alerts = list(getattr(model, "_last_alerts", []) or [])
    return [a for a in alerts if (a.get("parent_event") or {}).get("event_id") == event_id]


# ── Event catalogue ─────────────────────────────────────────────────────────

def event_summary(ev, now: datetime) -> dict:
    res = ev.result
    agencies = [_agency(n, i) for n, i in zip(ev.parent_names, ev.parent_ids)]
    rel_v = (ev.tca or {}).get("relative_velocity_kms")
    if rel_v is None:
        rel_v = res.stats.get("relative_velocity_kms", 0.0)
    return {
        "event_id": ev.event_id,
        "collision_utc": ev.collision_utc.isoformat(),
        "status": "pending_impact" if ev.pending(now) else "active",
        "parent_ids": list(ev.parent_ids),
        "parent_names": list(ev.parent_names),
        "parent_agencies": agencies,
        "fragment_count_total": int(res.n_total),
        "fragments_simulated": int(res.n_sampled),
        "catastrophic": bool(res.catastrophic),
        "relative_velocity_kms": round(float(rel_v), 4),
        "collision_point_eci_km": [round(float(x), 3) for x in res.impact_point_km],
        "seed": int(res.seed),
        "minutes_to_impact": round((ev.collision_utc - now).total_seconds() / 60.0, 2),
    }


@router.get("/events")
async def list_debris_events():
    model = _model()
    now = sim_clock.simulation_now()
    events = sorted(_events(model), key=lambda e: e.seq)
    return {"sim_time_utc": now.isoformat(), "events": [event_summary(ev, now) for ev in events]}


# ── Display sample ──────────────────────────────────────────────────────────

def stratified_sample(res, max_fragments: int, forced: list[int] | None = None) -> tuple[np.ndarray, dict]:
    """Deterministic stratified display sample (see module docstring)."""
    n = int(res.n_sampled)
    forced = sorted({int(i) for i in (forced or []) if 0 <= int(i) < n})
    info = {"method": "all_fragments", "strata": [], "forced_alert_fragments": len(forced)}
    if n <= max_fragments:
        return np.arange(n), info
    forced = forced[: max(0, int(max_fragments * THREAT_SHARE))]
    info["forced_alert_fragments"] = len(forced)
    budget = max_fragments - len(forced)
    lc = np.asarray(res.lc_m, float)
    dv = np.linalg.norm(np.asarray(res.dv_kms, float), axis=1)
    edges = np.quantile(lc, np.linspace(0, 1, N_SIZE_BINS + 1))
    size_bin = np.clip(np.searchsorted(edges, lc, side="right") - 1, 0, N_SIZE_BINS - 1)
    mask_free = np.ones(n, bool)
    mask_free[forced] = False
    strata = []
    for p in (0, 1):
        for b in range(N_SIZE_BINS):
            members = np.nonzero(mask_free & (res.parent_index == p) & (size_bin == b))[0]
            if members.size:
                strata.append((p, b, members))
    total = sum(m.size for _, _, m in strata)
    quotas = [max(1, budget * m.size / total) for _, _, m in strata]
    base = [min(int(math.floor(q)), m.size) for q, (_, _, m) in zip(quotas, strata)]
    # largest remainder until the budget is used
    order = np.argsort([-(q - math.floor(q)) for q in quotas])
    left = budget - sum(base)
    while left > 0:
        progressed = False
        for j in order:
            if left <= 0:
                break
            if base[j] < strata[j][2].size:
                base[j] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    while sum(base) > budget:   # the >=1 floor can overshoot a tiny budget
        j = int(np.argmax(base))
        base[j] -= 1
    chosen = list(forced)
    for (p, b, members), k in zip(strata, base):
        if k <= 0:
            continue
        ranked = members[np.argsort(dv[members], kind="stable")]
        picks = np.unique(np.round(np.linspace(0, ranked.size - 1, k)).astype(int))
        chosen.extend(int(x) for x in ranked[picks])
        info["strata"].append({"parent_index": p, "size_bin": b,
                               "lc_range_m": [round(float(edges[b]), 3), round(float(edges[b + 1]), 3)],
                               "members": int(members.size), "shown": int(picks.size)})
    info["method"] = "stratified_parent_x_size_quartile_dv_ranked"
    return np.array(sorted(set(chosen)), dtype=int), info


# ── Physics ─────────────────────────────────────────────────────────────────

def time_grid(t0_min: float, t1_min: float, step_s: float) -> np.ndarray:
    k0 = int(math.ceil(t0_min * 60.0 / step_s - 1e-9))
    k1 = int(math.floor(t1_min * 60.0 / step_s + 1e-9))
    return np.arange(k0, k1 + 1) * float(step_s)


def propagate_fragments(ev, t_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All simulated fragments on the grid. Returns R (nt, n, 3) float, alive (nt, n) bool.

    Uses the live model's integrator (``_advance``); only the time loop is in Python."""
    from app.core.debris_model import _advance

    res = ev.result
    n = len(res.r_km)
    nt = len(t_rel)
    R = np.full((nt, n, 3), np.nan, dtype=np.float64)
    A = np.zeros((nt, n), dtype=bool)
    post = np.nonzero(t_rel >= 0)[0]
    if n == 0 or post.size == 0:
        return R, A
    r = res.r_km.copy()
    v = res.v_kms.copy()
    alive = np.ones(n, bool)
    t_prev = 0.0
    for k in post:
        dt = float(t_rel[k]) - t_prev
        if dt > 0 and alive.any():
            r, v, alive = _advance(r, v, ev.bc, alive, dt)
        t_prev = float(t_rel[k])
        R[k] = r
        A[k] = alive
    return R, A


def envelope_stats(R: np.ndarray, A: np.ndarray, t_rel: np.ndarray) -> list[dict]:
    out = []
    for k in range(len(t_rel)):
        sel = A[k]
        n_alive = int(sel.sum())
        entry = {"t_rel_s": float(t_rel[k]), "centroid": None, "p50_km": 0.0, "p90_km": 0.0,
                 "along_track_spread_km": 0.0, "n_alive": n_alive}
        if n_alive:
            pts = R[k][sel]
            c = pts.mean(axis=0)
            d = np.linalg.norm(pts - c, axis=1)
            entry["centroid"] = [round(float(x), 1) for x in c]
            entry["p50_km"] = round(float(np.percentile(d, 50)), 2)
            entry["p90_km"] = round(float(np.percentile(d, 90)), 2)
            if n_alive >= 3:
                entry["along_track_spread_km"] = round(_along_track_spread(pts, c), 2)
        out.append(entry)
    return out


def _along_track_spread(pts: np.ndarray, c: np.ndarray) -> float:
    """Arc length between the 5th and 95th percentile in-plane angle (mean orbit plane)."""
    # mean orbit normal = normal of the best-fit plane through the Earth's centre
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    h = vt[-1]
    cn = np.linalg.norm(c)
    e1 = c - (c @ h) * h
    if np.linalg.norm(e1) < 1e-6 * max(cn, 1.0):
        e1 = pts[0] - (pts[0] @ h) * h
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(h, e1)
    theta = np.arctan2(pts @ e2, pts @ e1)
    lo, hi = np.percentile(theta, [5, 95])
    rmean = float(np.mean(np.linalg.norm(pts, axis=1)))
    return float(min(hi - lo, 2 * math.pi) * rmean)


def parent_tracks(ev, propagator, times: list[datetime]) -> dict:
    out: dict[str, Any] = {}
    roles = ("A", "B")
    if not ev.parent_ids:
        return out
    entries = []
    if propagator is not None:
        with propagator._lock:
            entries = [propagator._satellites.get(int(pid)) for pid in ev.parent_ids]
    else:
        entries = [None] * len(ev.parent_ids)
    from app.core.debris_model import _jd_arrays
    jd, fr = _jd_arrays(times)
    for k, (pid, entry) in enumerate(zip(ev.parent_ids, entries)):
        name = ev.parent_names[k] if k < len(ev.parent_names) else f"NORAD {pid}"
        rec = {"name": name, "agency": _agency(name, pid), "role": roles[k] if k < 2 else str(k),
               "positions": [None] * len(times), "source": "unavailable (not in propagator catalogue)"}
        if entry is not None:
            from sgp4.api import SatrecArray
            err, r, v = SatrecArray([entry[0]]).sgp4(jd, fr)
            try:
                from app.core.sgp4_propagator import add_burn_offsets
                add_burn_offsets(propagator, [int(pid)], jd, fr, r, v)
            except ImportError:  # pragma: no cover
                pass
            pos = np.round(r[0], 1).tolist()
            ok = (err[0] == 0) & np.all(np.isfinite(r[0]), axis=1)
            rec["positions"] = [p if o else None for p, o in zip(pos, ok.tolist())]
            rec["source"] = "SGP4 + executed-burn deviations"
        out[str(pid)] = rec
    return out


def threatened_list(alerts: list[dict], id_to_index: dict[str, int]) -> list[dict]:
    rows = []
    for a in alerts:
        s = a.get("sat1") or {}
        fid = a.get("fragment_id")
        rows.append({
            "sat_id": s.get("id"), "name": s.get("name"), "agency": s.get("agency"),
            "tca_utc": a.get("tca_utc"), "miss_km": a.get("miss_distance_km"),
            "pc": a.get("probability_of_collision"), "fragment_id": fid,
            "severity": a.get("severity"),
            "relative_speed_kms": a.get("relative_speed_kms"),
            "fragment_display_index": id_to_index.get(fid),
        })
    rows.sort(key=lambda x: (x["pc"] or 0.0), reverse=True)
    return rows


def _alert_fragment_indices(alerts: list[dict], event_id: str) -> list[int]:
    out = []
    prefix = f"{event_id}:F"
    for a in alerts:
        fid = a.get("fragment_id") or ""
        if fid.startswith(prefix):
            try:
                out.append(int(fid[len(prefix):]))
            except ValueError:
                pass
    return out


def _fragment_part(ev, t_rel: np.ndarray, max_fragments: int, forced: list[int]) -> dict:
    """Heavy part: propagation + sampling + serialisation (cached)."""
    t_start = time.perf_counter()
    res = ev.result
    R, A = propagate_fragments(ev, t_rel)
    t_prop = time.perf_counter() - t_start
    env = envelope_stats(R, A, t_rel)
    idx, sample_info = stratified_sample(res, max_fragments, forced)
    ids = [f"{ev.event_id}:F{i:04d}" for i in idx.tolist()]
    pids = list(ev.parent_ids)
    parent_of = [(pids[p] if p < len(pids) else int(p)) for p in res.parent_index[idx].tolist()]
    Rs = np.round(R[:, idx, :], 1)
    As = A[:, idx]
    positions = []
    for k in range(len(t_rel)):
        if not As[k].any():
            positions.append([None] * len(idx))
            continue
        pos = Rs[k].tolist()
        positions.append([p if a else None for p, a in zip(pos, As[k].tolist())])
    fragments = {
        "ids": ids,
        "size_m": np.round(res.lc_m[idx], 3).tolist(),
        "am_m2_kg": np.round(res.am_m2_kg[idx], 4).tolist(),
        "dv_ms": np.round(np.linalg.norm(res.dv_kms[idx], axis=1) * 1000.0, 1).tolist(),
        "parent_of": parent_of,
        "positions": positions,
    }
    frag_json = json.dumps(fragments, separators=(",", ":"))
    env_json = json.dumps(env, separators=(",", ":"))
    return {
        "frag_json": frag_json, "env_json": env_json,
        "id_to_index": {fid: i for i, fid in enumerate(ids)},
        "sample_info": sample_info, "n_display": int(len(idx)),
        "propagate_ms": round(t_prop * 1000.0, 1),
        "compute_ms": round((time.perf_counter() - t_start) * 1000.0, 1),
    }


def build_replay(ev, propagator, alerts: list[dict], *, t0_min: float = -15.0, t1_min: float = 180.0,
                 step_s: float = 30.0, max_fragments: int = MAX_DISPLAY_FRAGMENTS) -> bytes:
    """Full replay payload as compact JSON bytes (schema: CONTRACT3)."""
    t_start = time.perf_counter()
    t_rel = time_grid(t0_min, t1_min, step_s)
    if t_rel.size == 0:
        raise ValueError("empty time window")
    if t_rel.size > MAX_SAMPLES:
        raise ValueError(f"window/step gives {t_rel.size} samples (max {MAX_SAMPLES})")
    times = [ev.collision_utc + timedelta(seconds=float(s)) for s in t_rel]
    requested = int(max_fragments)
    max_fragments = max(1, min(requested, MAX_POSITION_CELLS // int(t_rel.size)))
    forced = sorted(set(_alert_fragment_indices(alerts, ev.event_id)))
    key = (ev.event_id, int(ev.result.seed), ev.collision_utc.isoformat(), int(ev.result.n_sampled),
           float(t0_min), float(t1_min), float(step_s), int(max_fragments), tuple(forced))
    with _cache_lock:
        part = _cache.get(key)
        if part is not None:
            _cache.move_to_end(key)
    cached = part is not None
    if part is None:
        part = _fragment_part(ev, t_rel, max_fragments, forced)
        with _cache_lock:
            _cache[key] = part
            while len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
    parents = parent_tracks(ev, propagator, times)
    res = ev.result
    light = {
        "event_id": ev.event_id,
        "collision_utc": ev.collision_utc.isoformat(),
        "status": "pending_impact" if ev.pending(sim_clock.simulation_now()) else "active",
        "frame": FRAME,
        "step_s": float(step_s),
        "times_utc": [t.isoformat() for t in times],
        "t_rel_s": [float(s) for s in t_rel],
        "parents": parents,
        "threatened": threatened_list(alerts, part["id_to_index"]),
        "provenance": {
            "propagator": PROPAGATOR_LABEL,
            "parents_propagator": "SGP4 (+ executed-burn deviations); after t_rel=0 = trajectory had they not collided",
            "seed": int(res.seed),
            "sampled_from": int(res.n_total),
            "fragments_simulated": int(res.n_sampled),
            "fragments_displayed": part["n_display"],
            "max_fragments_requested": requested,
            "display_budget_reduced": max_fragments < requested,
            "fragment_weight": round(float(res.weight), 4),
            "sample": part["sample_info"],
            "envelope_over": "all simulated fragments",
            "breakup": "NASA SBM state computed at event creation (TCA), seeded CRC32(event_id): "
                       "identical to what the live model releases at impact",
            "rounding_km": 0.1,
            "cached": cached,
            "propagate_ms": part["propagate_ms"],
            "fragment_part_ms": part["compute_ms"],
            "note": "Fragment positions: RK4 two-body+J2+drag from the breakup state at the TCA, null before "
                    "the collision and after decay (<100 km). Positions between samples may be interpolated "
                    "linearly by the client (label as interpolation).",
        },
    }
    light["provenance"]["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 1)
    head = json.dumps(light, separators=(",", ":"))
    body = head[:-1] + ',"fragments":' + part["frag_json"] + ',"envelope":' + part["env_json"] + "}"
    return body.encode("utf-8")


def _find_event(model, event_id: str):
    get = getattr(model, "get_cloud", None)
    ev = get(event_id) if get else None
    if ev is None:
        ev = next((e for e in _events(model) if e.event_id == event_id), None)
    return ev


def _prune_cache(live_ids: set[str]) -> None:
    with _cache_lock:
        for key in [k for k in _cache if k[0] not in live_ids]:
            _cache.pop(key, None)


@router.get("/events/{event_id}/replay")
async def debris_event_replay(
    event_id: str,
    t0_min: float = Query(-15.0, ge=-720.0, le=0.0),
    t1_min: float = Query(180.0, gt=0.0, le=1440.0),
    step_s: float = Query(30.0, ge=5.0, le=600.0),
    max_fragments: int = Query(MAX_DISPLAY_FRAGMENTS, ge=1, le=MAX_DISPLAY_FRAGMENTS),
):
    model = _model()
    ev = _find_event(model, event_id)
    if ev is None:
        raise HTTPException(status_code=404, detail=f"debris event {event_id!r} not found")
    _prune_cache({e.event_id for e in _events(model)})
    alerts = _event_alerts(model, event_id)
    try:
        body = await asyncio.to_thread(build_replay, ev, _propagator(), alerts, t0_min=t0_min,
                                       t1_min=t1_min, step_s=step_s, max_fragments=max_fragments)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return Response(content=body, media_type="application/json",
                    headers={"X-Payload-Bytes": str(len(body))})
