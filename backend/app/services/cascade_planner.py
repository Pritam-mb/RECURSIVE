"""
Cascade planner: alert graph, BFS cascade depth, physics node probabilities,
re-propagated avoidance manoeuvres and alert-derived hotspots.

Everything published here is derived from the alerts it is given (agent A's
screening alerts with Foster Pc, agent B's debris alerts with parent_event)
plus orbital re-propagation. There is no proximity heuristic, no ML score in
the published depth / probability / manoeuvre, and no fixed constants
presented as results.

Graph
-----
* Object nodes: every object that appears in an alert.
* Event nodes ("event:<id>"): every distinct ``parent_event`` (a simulated
  collision). Edges: event -> each parent object (the colliding pair) and
  event -> each fragment that appears in a debris alert.
* Alert edges: sat1 <-> sat2, weighted by the alert's Pc.

Cascade depth (BFS hop count, nothing else)
-------------------------------------------
Roots are the collision events. BFS over the graph gives every object its hop
distance from the nearest event: fragments 1, satellites threatened by those
fragments 2, objects in screened conjunctions with those satellites 3, ...
An alert's ``cascade_depth`` is the hop count of its farther endpoint, i.e.
the number of links from the root event to the threatened object. An alert in
a component with no collision event is itself the primary (potential) event:
its depth is 1. ``downstream_ids`` are objects reachable from the alert's
deeper endpoint(s) moving away from the root (for a primary alert: objects
that have their own screened conjunctions with either party).
``upstream_event`` is the root event id, or None for a primary alert.

Node probability
----------------
P(object is hit by at least one threat) = 1 - prod_i (1 - Pc_i) over all
alerts incident to the object, assuming independent encounters.

Manoeuvres: see app/services/maneuver_planner.py (re-propagation search).

The GAT / GNN rankers trained on synthetic graphs are run, when available, as
an ADVISORY cross-check only (``ranker_review``); their outputs never feed
the depth, probabilities or the plan.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.core.agency import AGENCY_ALIASES as agency_aliases, infer_agency as agency_inference

logger = logging.getLogger(__name__)

AGENCY_ALIASES = agency_aliases
infer_agency = agency_inference

DEFAULT_CPI_THRESHOLD = 5.0          # kept for API compatibility (debris builder reads it)
MANEUVER_PC_THRESHOLD = 1e-6         # alerts at/above this Pc get a recommended manoeuvre
MAX_MANEUVER_ALERTS = 15             # cap re-propagation work per refresh (top-Pc alerts)
MANEUVER_TIME_BUDGET_S = 3.0
MAX_DOWNSTREAM_IDS = 50
MAX_HOTSPOTS = 20
RANKER_DISAGREEMENT_THRESHOLD = 0.25

_EMPTY_RANKER_REVIEW = {
    "role": "advisory",
    "primary": "gat",
    "cross_check": "gnn",
    "primary_ok": False,
    "cross_check_ok": False,
    "fallback_used": False,
    "degraded": False,
    "disagreement_threshold": RANKER_DISAGREEMENT_THRESHOLD,
    "disagreement_count": 0,
    "max_disagreement": 0.0,
    "disputed_satellites": [],
    "spearman_vs_physics": None,
    "note": "Synthetic-trained graph rankers; advisory only, never used for depth, probability or plan.",
}


def _pc(alert: dict) -> float:
    for key in ("p_collision", "probability_of_collision"):
        value = alert.get(key)
        if value is not None:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                pass
    return 0.0


def _parse_time(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _node_key(value) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


class CascadePlanner:
    """Alert-graph cascade analysis with re-propagated manoeuvre planning."""

    def __init__(self) -> None:
        self._primary_ranker = None
        self._cross_check_ranker = None
        self._rankers_loaded = False

    # ── Advisory ML rankers (lazy, optional) ────────────────────────────────
    def _load_rankers(self) -> None:
        if self._rankers_loaded:
            return
        self._rankers_loaded = True
        try:
            from app.ml.gat_cascade import CascadeGAT

            self._primary_ranker = CascadeGAT()
        except Exception as error:  # pragma: no cover - optional
            logger.info("Advisory GAT ranker unavailable: %s", error)
        try:
            from app.ml.gnn_cascade import CascadeGNN

            self._cross_check_ranker = CascadeGNN()
        except Exception as error:  # pragma: no cover - optional
            logger.info("Advisory GNN ranker unavailable: %s", error)

    @property
    def primary_ranker(self):
        self._load_rankers()
        return self._primary_ranker

    @property
    def cross_check_ranker(self):
        self._load_rankers()
        return self._cross_check_ranker

    # ── Public entry point ──────────────────────────────────────────────────
    def analyze_snapshot(
        self,
        states: list[Any] | None,
        alerts: list[dict[str, Any]] | None = None,
        propagator: Any | None = None,
        reference_time: datetime | str | None = None,
        *,
        plan_maneuvers: bool = True,
        max_maneuver_alerts: int = MAX_MANEUVER_ALERTS,
        maneuver_time_budget_s: float = MANEUVER_TIME_BUDGET_S,
        run_rankers: bool = True,
        **_ignored: Any,
    ) -> dict[str, Any]:
        t_start = time.perf_counter()
        alerts = [a.to_dict() if hasattr(a, "to_dict") else a for a in (alerts or [])]
        alerts = [a for a in alerts if isinstance(a, dict) and a.get("sat1") and a.get("sat2")]
        ref = _parse_time(reference_time)
        if ref is None:
            from app.core.sim_clock import simulation_now

            ref = simulation_now()

        state_by_id = {}
        for s in states or []:
            if _get(s, "error_code", 0) in (0, None) and _get(s, "norad_id") is not None:
                state_by_id[_node_key(_get(s, "norad_id"))] = s

        graph = build_alert_graph(alerts)
        depth_info = bfs_cascade(graph)
        node_prob = node_hit_probabilities(graph)
        annotate_alerts(alerts, graph, depth_info)

        maneuver_stats: dict[str, Any] = {"status": "skipped"}
        if plan_maneuvers and alerts:
            maneuver_stats = self._plan_maneuvers(alerts, propagator, ref, state_by_id,
                                                  max_maneuver_alerts, maneuver_time_budget_s)
        for alert in alerts:
            alert.setdefault("recommended_maneuver", None)

        hotspots = build_hotspots(alerts, graph, propagator, ref)
        cascade_plan = build_cascade_plan(alerts, graph, node_prob)
        ranker_review = self._ranker_review(graph, node_prob, state_by_id) if run_rankers else dict(_EMPTY_RANKER_REVIEW)

        involved = set()
        for alert in alerts:
            if _pc(alert) >= MANEUVER_PC_THRESHOLD or alert.get("upstream_event"):
                for side in ("sat1", "sat2"):
                    node = graph["nodes"].get(_node_key(alert[side].get("id")))
                    if node is not None:
                        involved.add(node["agency"])
        involved.discard("Debris")  # synthetic fragments have no operator

        seeds = []
        for alert in sorted(alerts, key=_pc, reverse=True):
            for side in ("sat1", "sat2"):
                sid = _node_key(alert[side].get("id"))
                if sid not in seeds:
                    seeds.append(sid)
        max_depth = max((int(a.get("cascade_depth") or 0) for a in alerts), default=0)

        return {
            "graph": {
                "node_count": len(graph["nodes"]),
                "edge_count": len(graph["edges"]),
                "event_count": len(graph["events"]),
                "edge_source": "alerts",
                "influence_radius_km": None,
                "max_downstream_hops": depth_info["max_hops"],
            },
            "alerts": alerts,
            "hotspots": hotspots,
            "cascade_plan": cascade_plan,
            "seed_satellites": seeds[:20],
            "total_delta_v_ms": round(sum(p["maneuver"]["delta_v_ms"] for p in cascade_plan), 4),
            "cascade_depth": max_depth,
            "cascade_depth_definition": "BFS hops from the root collision event (primary conjunction = 1)",
            "agencies_involved": sorted(involved),
            "cpi_threshold": DEFAULT_CPI_THRESHOLD,
            "node_probabilities": {str(k): v for k, v in node_prob.items()},
            "node_probability_definition": "1 - prod(1 - Pc_i) over alerts incident to the object",
            "optimization": {
                "method": "repropagation_candidate_search",
                **maneuver_stats,
            },
            "ranker_review": ranker_review,
            "runtime_s": round(time.perf_counter() - t_start, 3),
        }

    # ── Manoeuvres ──────────────────────────────────────────────────────────
    def _plan_maneuvers(self, alerts, propagator, ref, state_by_id, max_alerts, budget_s) -> dict:
        from app.services import maneuver_planner as mp

        satrecs: dict = {}
        if propagator is not None:
            lock = getattr(propagator, "_lock", None)
            try:
                if lock is not None:
                    with lock:
                        satrecs = dict(getattr(propagator, "_satellites", {}))
                else:
                    satrecs = dict(getattr(propagator, "_satellites", {}))
            except Exception:
                satrecs = {}
        if not satrecs:
            for alert in alerts:
                if _pc(alert) >= MANEUVER_PC_THRESHOLD:
                    alert["maneuver_status"] = "no_propagator"
            return {"status": "no_propagator"}

        try:
            from app.core import satcat
        except Exception:  # pragma: no cover
            satcat = None

        eligible = sorted((a for a in alerts if _pc(a) >= MANEUVER_PC_THRESHOLD), key=_pc, reverse=True)
        jobs = []
        for alert in eligible[:max_alerts]:
            tca = _parse_time(alert.get("tca_utc"))
            if tca is None:
                alert["maneuver_status"] = "no_tca"
                continue
            tca_s = (tca - ref).total_seconds()
            if tca_s <= 0 or tca_s > mp.MAX_HORIZON_S:
                alert["maneuver_status"] = "tca_outside_planning_horizon"
                continue
            tracks = {}
            for side in ("sat1", "sat2"):
                info = alert[side]
                sid = _node_key(info.get("id"))
                name = info.get("name") or str(sid)
                if sid in satrecs:
                    tracks[side] = mp.ObjectTrack(sid, name, satrec=satrecs[sid][0])
                else:
                    st = alert.get(f"{side}_state") or (alert.get("fragment_state") if info.get("object_type") == "DEB" else None)
                    if st and st.get("r_km") is not None and st.get("v_kms") is not None:
                        r0, v0 = np.asarray(st["r_km"], float), np.asarray(st["v_kms"], float)
                        epoch = _parse_time(st.get("epoch_utc")) or ref
                        if abs((epoch - ref).total_seconds()) > 1e-3:
                            r0, v0 = mp.propagate_j2(r0, v0, (ref - epoch).total_seconds())
                            r0, v0 = r0[0], v0[0]
                        tracks[side] = mp.ObjectTrack(sid, name, r0=r0, v0=v0)
                    elif sid in state_by_id:
                        s = state_by_id[sid]
                        tracks[side] = mp.ObjectTrack(
                            sid, name,
                            r0=np.array([_get(s, "x"), _get(s, "y"), _get(s, "z")], float),
                            v0=np.array([_get(s, "vx"), _get(s, "vy"), _get(s, "vz")], float))
            if len(tracks) < 2:
                alert["maneuver_status"] = "object_state_unavailable"
                continue
            movers, others, is_sat1 = [], [], []
            for side, other_side in (("sat1", "sat2"), ("sat2", "sat1")):
                info = alert[side]
                track = tracks[side]
                rec = satcat.lookup(track.norad_id) if satcat else None
                otype = (info.get("object_type") or (rec or {}).get("object_type") or "UNK")
                decayed = bool(rec and rec.get("decay_date"))
                # SATCAT OPS_STATUS_CODE: '-' nonoperational, 'D' decayed -> no thrust available.
                dead = bool(rec and rec.get("ops_status") in ("-", "D"))
                if track.satrec is None or otype != "PAY" or decayed or dead:
                    continue
                agency = info.get("agency") or infer_agency(track.name, track.norad_id)
                movers.append(mp.Mover(track, agency, agency not in ("Unknown", "UNKNOWN", None), rec))
                others.append(tracks[other_side])
                is_sat1.append(side == "sat1")
            if not movers:
                alert["maneuver_status"] = "no_operational_payload"
                continue
            ages = alert.get("tle_age_days")
            ages_t = (float(ages[0]), float(ages[1])) if isinstance(ages, (list, tuple)) and len(ages) == 2 else None
            if ages_t is None:
                ages_t = tuple(
                    (tca_s / 86400.0) if tracks[s].satrec is None else
                    ((tca - ref).total_seconds() / 86400.0 + _satrec_age_days(tracks[s].satrec, ref))
                    for s in ("sat1", "sat2"))
            iso_sigma = None
            if alert.get("covariance_model") == "isotropic_bplane":
                a_m = (alert.get("covariance_ellipse") or {}).get("a")
                iso_sigma = float(a_m) / 1000.0 if a_m else None
            jobs.append(mp.EncounterJob(
                alert=alert, is_sat1=is_sat1, movers=movers, others=others, tca_s=tca_s,
                hbr_km=float(alert.get("hbr_km") or 0.01), ages_days=ages_t, iso_sigma_km=iso_sigma))

        for alert in eligible[max_alerts:]:
            alert["maneuver_status"] = "not_planned_outside_top_n"
        if not jobs:
            return {"status": "no_jobs", "eligible_alerts": len(eligible)}

        stats = mp.plan_maneuvers(jobs, ref, time_budget_s=budget_s)
        for job in jobs:
            res = job.alert.pop("_maneuver_result", None) or {}
            job.alert["maneuver_status"] = res.get("status", "failed")
            job.alert["recommended_maneuver"] = res.get("recommended_maneuver")
            if res.get("baseline"):
                job.alert["maneuver_baseline"] = res["baseline"]
        stats["eligible_alerts"] = len(eligible)
        stats["planned_alerts"] = sum(1 for j in jobs if j.alert.get("recommended_maneuver"))
        return stats

    # ── Advisory rankers ───────────────────────────────────────────────────
    def _ranker_review(self, graph, node_prob, state_by_id) -> dict:
        review = dict(_EMPTY_RANKER_REVIEW)
        review["disputed_satellites"] = []
        ids = [n for n in graph["nodes"] if n in state_by_id]
        if len(ids) < 2:
            return review
        index = {n: i for i, n in enumerate(ids)}
        agencies = sorted({graph["nodes"][n]["agency"] for n in ids})
        agency_id = {a: i + 1 for i, a in enumerate(agencies)}
        feats = []
        for n in ids:
            s = state_by_id[n]
            pos = np.array([_get(s, "x"), _get(s, "y"), _get(s, "z")], float)
            vel = np.array([_get(s, "vx"), _get(s, "vy"), _get(s, "vz")], float)
            peak = max((e["cpi_score"] for e in graph["adjacency"].get(n, [])), default=0.0)
            feats.append([*pos, *vel, np.linalg.norm(vel) * 3600.0, np.linalg.norm(pos) - 6371.0,
                          agency_id[graph["nodes"][n]["agency"]], peak])
        ei, ea = [], []
        for e in graph["edges"]:
            if e["kind"] != "alert" or e["source"] not in index or e["target"] not in index:
                continue
            for a, b in ((e["source"], e["target"]), (e["target"], e["source"])):
                ei.append([index[a], index[b]])
                ea.append([e["miss_distance_km"], e["relative_speed_kms"], e["tca_minutes"], e["pc"]])
        if not ei:
            return review
        node_features = np.array(feats, float)
        edge_index = np.array(ei, int).T
        edge_attr = np.array(ea, float)
        gat = gnn = None
        try:
            if self.primary_ranker is not None:
                gat = np.asarray(self.primary_ranker.predict(node_features, edge_index, edge_attr)["maneuver_probability"], float)
                review["primary_ok"] = True
        except Exception as error:
            logger.info("Advisory GAT failed: %s", error)
        try:
            if self.cross_check_ranker is not None:
                gnn = np.asarray(self.cross_check_ranker.predict(node_features, edge_index, edge_attr)["maneuver_probability"], float)
                review["cross_check_ok"] = True
        except Exception as error:
            logger.info("Advisory GNN failed: %s", error)
        if gat is None and gnn is not None:
            review["fallback_used"] = True
            review["degraded"] = True
        if gat is not None and gnn is not None and gat.shape == gnn.shape:
            delta = np.abs(gat - gnn)
            disputed = np.flatnonzero(delta > RANKER_DISAGREEMENT_THRESHOLD)
            review["disagreement_count"] = int(disputed.size)
            review["max_disagreement"] = round(float(delta.max()), 4)
            review["disputed_satellites"] = [
                {"satellite_id": ids[i], "gat_probability": round(float(gat[i]), 4),
                 "gnn_probability": round(float(gnn[i]), 4), "delta": round(float(delta[i]), 4)}
                for i in disputed[:20]]
        ml = gat if gat is not None else gnn
        if ml is not None and len(ids) >= 3:
            phys = np.array([node_prob.get(n, 0.0) for n in ids])
            rho = _spearman(ml, phys)
            review["spearman_vs_physics"] = None if rho is None else round(rho, 3)
        return review


def _satrec_age_days(satrec, ref: datetime) -> float:
    from sgp4.api import jday

    jd, fr = jday(ref.year, ref.month, ref.day, ref.hour, ref.minute, ref.second + ref.microsecond / 1e6)
    return (jd + fr) - (satrec.jdsatepoch + satrec.jdsatepochF)


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ══════════════════════════════════════════════════════════════════════════
# Graph construction and BFS (pure functions, unit-tested)
# ══════════════════════════════════════════════════════════════════════════

def build_alert_graph(alerts: list[dict]) -> dict[str, Any]:
    """Nodes from alert endpoints + collision events; edges from alerts and parent links."""
    nodes: dict[Any, dict] = {}
    events: dict[str, dict] = {}
    edges: list[dict] = []
    adjacency: dict[Any, list[dict]] = {}

    def add_node(info: dict) -> Any:
        nid = _node_key(info.get("id"))
        if nid not in nodes:
            name = info.get("name") or str(nid)
            otype = info.get("object_type")
            if not otype:
                try:
                    from app.core.satcat import lookup, object_type_from_name

                    rec = lookup(nid) if isinstance(nid, int) else None
                    otype = (rec or {}).get("object_type") or object_type_from_name(name)
                except Exception:
                    otype = "UNK"
            agency = info.get("agency") or infer_agency(name, nid if isinstance(nid, int) else None)
            nodes[nid] = {"id": nid, "name": name, "agency": agency, "object_type": otype, "kind": "object"}
            adjacency.setdefault(nid, [])
        return nid

    def link(a, b, edge):
        edges.append(edge)
        adjacency.setdefault(a, []).append({**edge, "neighbor": b})
        adjacency.setdefault(b, []).append({**edge, "neighbor": a})

    for alert in alerts:
        a = add_node(alert["sat1"])
        b = add_node(alert["sat2"])
        pc = _pc(alert)
        link(a, b, {
            "kind": "alert", "source": a, "target": b, "alert_id": alert.get("id"), "pc": pc,
            "miss_distance_km": float(alert.get("miss_distance_km") or 0.0),
            "relative_speed_kms": float(alert.get("relative_speed_kms") or
                                        (alert.get("relative_speed_kmh") or 0.0) / 3600.0),
            "tca_minutes": float(alert.get("tca_minutes") or 0.0),
            "cpi_score": float(alert.get("cpi_score") or 0.0),
        })
        pe = alert.get("parent_event")
        if isinstance(pe, dict) and pe.get("event_id"):
            ev_key = f"event:{pe['event_id']}"
            if ev_key not in events:
                events[ev_key] = {"id": ev_key, "event_id": pe["event_id"], "parent_ids": list(pe.get("parent_ids") or []),
                                  "collision_utc": pe.get("collision_utc"), "fragment_count": pe.get("fragment_count"),
                                  "kind": "event"}
                adjacency.setdefault(ev_key, [])
                for pid in events[ev_key]["parent_ids"]:
                    p = add_node({"id": pid, "name": None})
                    link(ev_key, p, {"kind": "event_parent", "source": ev_key, "target": p, "pc": 1.0})
            # The fragment is whichever endpoint is debris from this event.
            frag = None
            for side in ("sat2", "sat1"):
                info = alert[side]
                if info.get("object_type") == "DEB" or str(info.get("name", "")).upper().startswith("FRAG"):
                    frag = _node_key(info.get("id"))
                    break
            if frag is not None and not any(e["neighbor"] == frag for e in adjacency[ev_key]):
                link(ev_key, frag, {"kind": "event_fragment", "source": ev_key, "target": frag, "pc": 1.0})
    return {"nodes": nodes, "events": events, "edges": edges, "adjacency": adjacency}


def bfs_cascade(graph: dict) -> dict[str, Any]:
    """Hop distance of every node from its nearest collision event (multi-source BFS)."""
    adjacency = graph["adjacency"]
    hops: dict[Any, int] = {}
    root: dict[Any, str] = {}
    queue = deque()
    for ev_key, ev in graph["events"].items():
        hops[ev_key] = 0
        root[ev_key] = ev["event_id"]
        queue.append(ev_key)
    while queue:
        node = queue.popleft()
        for edge in adjacency.get(node, []):
            nb = edge["neighbor"]
            if nb not in hops:
                hops[nb] = hops[node] + 1
                root[nb] = root[node]
                queue.append(nb)
    return {"hops": hops, "root": root, "max_hops": max(hops.values(), default=0)}


def _downstream(graph: dict, starts: list, blocked: set, hops: dict | None, limit: int) -> list:
    """Objects reachable from `starts` without passing through `blocked`, moving away from the root."""
    adjacency = graph["adjacency"]
    seen = set(blocked) | set(starts)
    out = []
    queue = deque(starts)
    while queue and len(out) < limit:
        node = queue.popleft()
        for edge in adjacency.get(node, []):
            nb = edge["neighbor"]
            if nb in seen or str(nb).startswith("event:"):
                continue
            if hops is not None and nb in hops and node in hops and hops[nb] <= hops[node]:
                continue
            seen.add(nb)
            out.append(nb)
            queue.append(nb)
    return out[:limit]


def annotate_alerts(alerts: list[dict], graph: dict, depth_info: dict) -> None:
    """Attach cascade_depth, downstream_ids and upstream_event to each alert in place."""
    hops, root = depth_info["hops"], depth_info["root"]
    for alert in alerts:
        a = _node_key(alert["sat1"].get("id"))
        b = _node_key(alert["sat2"].get("id"))
        if a in hops and b in hops:
            far = a if hops[a] >= hops[b] else b
            alert["cascade_depth"] = int(hops[far])
            alert["upstream_event"] = root.get(far)
            starts = [n for n in (a, b) if hops[n] == hops[far]]
            alert["downstream_ids"] = _downstream(graph, starts, {a, b} - set(starts), hops, MAX_DOWNSTREAM_IDS)
        else:
            alert["cascade_depth"] = 1
            alert["upstream_event"] = None
            alert["downstream_ids"] = _downstream(graph, [a, b], set(), None, MAX_DOWNSTREAM_IDS)


def node_hit_probabilities(graph: dict) -> dict[Any, float]:
    """1 - prod(1 - Pc_i) over alerts incident to each object node."""
    out: dict[Any, float] = {}
    for nid in graph["nodes"]:
        survive = 1.0
        for edge in graph["adjacency"].get(nid, []):
            if edge["kind"] == "alert":
                survive *= (1.0 - edge["pc"])
        out[nid] = float(1.0 - survive)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Plan + hotspots
# ══════════════════════════════════════════════════════════════════════════

def build_cascade_plan(alerts: list[dict], graph: dict, node_prob: dict) -> list[dict]:
    plan = []
    for alert in sorted(alerts, key=_pc, reverse=True):
        rec = alert.get("recommended_maneuver")
        if not rec:
            continue
        sat_id = rec["sat_id"]
        threat_side = "sat2" if _node_key(alert["sat1"].get("id")) == sat_id else "sat1"
        threat = alert[threat_side]
        rsw = rec["delta_v_rsw_ms"]
        node = graph["nodes"].get(sat_id, {})
        plan.append({
            "satellite_id": sat_id,
            "satellite_name": rec.get("sat_name") or node.get("name"),
            "agency": rec.get("agency") or node.get("agency"),
            "alert_id": alert.get("id"),
            "cascade_depth": alert.get("cascade_depth", 1),
            "triggered_by": _node_key(threat.get("id")),
            "trigger_label": (f"EVENT_{alert['upstream_event']}" if alert.get("upstream_event")
                              else "PRIMARY_CONJUNCTION"),
            "threat": {
                "satellite_id": _node_key(threat.get("id")),
                "satellite_name": threat.get("name"),
                "miss_distance_km": alert.get("miss_distance_km"),
                "relative_velocity_kmh": alert.get("relative_speed_kmh"),
                "tca_minutes": alert.get("tca_minutes"),
                "p_collision": _pc(alert),
                "cpi_score": alert.get("cpi_score"),
            },
            "maneuver": {
                "frame": "RSW",
                "dv_r": rsw[0], "dv_s": rsw[1], "dv_w": rsw[2],
                "delta_v_ms": rec["delta_v_ms"],
                "eci_ms": rec.get("delta_v_eci_ms"),
                "burn_epoch_utc": rec.get("burn_epoch_utc"),
            },
            "risk_before": rec.get("pc_before"),
            "risk_after": rec.get("new_pc_collision"),
            "risk_metric": "collision_probability",
            "new_miss_distance_km": rec.get("new_miss_distance_km"),
            "achieved_target": rec.get("achieved_target"),
            "fuel_cost_pct": rec.get("fuel_cost_pct"),
            "maneuver_probability": round(float(node_prob.get(sat_id, 0.0)), 8),
            "verified_by": rec.get("verified_by"),
        })
    return plan


def _alert_position(alert: dict, propagator, cache: dict) -> np.ndarray | None:
    pos = alert.get("tca_position_km")
    if pos is not None and len(pos) == 3:
        return np.asarray(pos, float)
    tca = _parse_time(alert.get("tca_utc"))
    if propagator is None or tca is None:
        return None
    pts = []
    for side in ("sat1", "sat2"):
        sid = _node_key(alert[side].get("id"))
        try:
            sats = getattr(propagator, "_satellites", {})
            if sid not in sats:
                return None
            from sgp4.api import jday

            jd, fr = jday(tca.year, tca.month, tca.day, tca.hour, tca.minute, tca.second + tca.microsecond / 1e6)
            e, r, _ = sats[sid][0].sgp4(jd, fr)
            if e != 0:
                return None
            pts.append(np.asarray(r, float))
        except Exception:
            return None
    return 0.5 * (pts[0] + pts[1])


def _objects_near(propagator, when: datetime, center: np.ndarray, radius_km: float, cache: dict) -> list[dict]:
    """All tracked objects within radius_km of center at `when` (one vectorised SGP4 call)."""
    if propagator is None:
        return []
    key = when.isoformat()
    if key not in cache:
        try:
            from sgp4.api import SatrecArray, jday

            lock = getattr(propagator, "_lock", None)
            if lock is not None:
                with lock:
                    sats = dict(getattr(propagator, "_satellites", {}))
            else:
                sats = dict(getattr(propagator, "_satellites", {}))
            ids = list(sats)
            if not ids:
                cache[key] = None
            else:
                jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute, when.second + when.microsecond / 1e6)
                e, r, _ = SatrecArray([sats[i][0] for i in ids]).sgp4(np.array([jd]), np.array([fr]))
                r = r[:, 0, :]
                r[e[:, 0] != 0] = np.nan
                cache[key] = (ids, [sats[i][1] for i in ids], r)
        except Exception as error:
            logger.debug("hotspot neighbour propagation failed: %s", error)
            cache[key] = None
    data = cache[key]
    if data is None:
        return []
    ids, names, r = data
    d = np.linalg.norm(r - center[None, :], axis=1)
    sel = np.flatnonzero(np.isfinite(d) & (d <= radius_km))
    return [{"id": ids[i], "name": names[i], "distance_km": round(float(d[i]), 3)}
            for i in sel[np.argsort(d[sel])][:50]]


def build_hotspots(alerts: list[dict], graph: dict, propagator, ref: datetime) -> list[dict]:
    """
    Hotspots from alerts:
      * screening alert -> location = TCA midpoint, radius = 3-sigma major axis of
        the B-plane covariance (>= miss distance), score = Pc.
      * debris event -> location = centroid of its fragment-encounter TCA points,
        radius = 90th-percentile spread of those points, score = 1 - prod(1 - Pc).
    affected_satellites = tracked objects inside the zone at the hotspot TCA.
    """
    pos_cache: dict = {}
    near_cache: dict = {}
    candidates = []
    by_event: dict[str, list[dict]] = {}
    for alert in alerts:
        if alert.get("parent_event") and alert.get("source") == "debris":
            by_event.setdefault(alert["parent_event"]["event_id"], []).append(alert)
            continue
        candidates.append(("alert", alert, [alert]))
    for ev_id, group in by_event.items():
        candidates.append(("event", ev_id, group))

    def score(group):
        survive = 1.0
        for a in group:
            survive *= (1.0 - _pc(a))
        return 1.0 - survive

    candidates.sort(key=lambda c: score(c[2]), reverse=True)
    hotspots = []
    for kind, ref_obj, group in candidates[:MAX_HOTSPOTS]:
        pts = [p for p in (_alert_position(a, propagator, pos_cache) for a in group) if p is not None]
        if not pts:
            continue
        lead = max(group, key=_pc)
        center = np.mean(pts, axis=0)
        if kind == "alert":
            ell = lead.get("covariance_ellipse") or {}
            a_m = ell.get("a")
            radius = max(float(a_m) / 1000.0 if a_m else 0.0, float(lead.get("miss_distance_km") or 0.0))
            radius_source = "covariance_3sigma_major_axis" if a_m else "miss_distance"
        else:
            spread = np.linalg.norm(np.asarray(pts) - center[None, :], axis=1)
            radius = float(np.percentile(spread, 90)) if len(pts) > 1 else float(lead.get("miss_distance_km") or 0.0)
            radius_source = "fragment_encounter_p90_spread"
        radius = max(radius, 0.01)
        tca_times = [t for t in (_parse_time(a.get("tca_utc")) for a in group) if t is not None]
        when = min(tca_times) if tca_times else ref
        affected = _objects_near(propagator, when, center, radius, near_cache)
        if not affected:
            affected = []
        for a in group:  # the alert's own objects are affected by definition
            for side in ("sat1", "sat2"):
                sid = _node_key(a[side].get("id"))
                if not any(x["id"] == sid for x in affected) and graph["nodes"].get(sid, {}).get("object_type") != "DEB":
                    affected.append({"id": sid, "name": a[side].get("name"), "distance_km": None})
        hs = score(group)
        tca_minutes = max(0.0, (when - ref).total_seconds() / 60.0)
        hotspots.append({
            "kind": "conjunction" if kind == "alert" else "debris_event",
            "event_id": ref_obj if kind == "event" else lead.get("upstream_event"),
            "sat1": lead["sat1"], "sat2": lead["sat2"],
            "tca_utc": when.isoformat(),
            "tca_minutes": round(tca_minutes, 2),
            "tca_hours": round(tca_minutes / 60.0, 3),
            "miss_distance_km": lead.get("miss_distance_km"),
            "relative_speed_kmh": lead.get("relative_speed_kmh"),
            "cpi_score": lead.get("cpi_score"),
            "p_collision": _pc(lead),
            "severity": lead.get("severity"),
            "hotspot_score": hs,
            "hotspot_score_definition": "1 - prod(1 - Pc) over alerts in the zone",
            "alert_count": len(group),
            "position": {"x": round(float(center[0]), 3), "y": round(float(center[1]), 3), "z": round(float(center[2]), 3)},
            "zone_radius_km": round(radius, 3),
            "zone_radius_source": radius_source,
            "affected_satellites": affected,
            "affected_count": len(affected),
        })
    hotspots.sort(key=lambda h: h["hotspot_score"], reverse=True)
    return hotspots
