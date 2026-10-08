"""
GET /api/cascade/explorer — one coherent payload for the Cascade & Hotspot Explorer.

Everything here is *re-arranged* from values the live pipeline already computed
(``state_cache.get_latest_alerts()``: screening + debris alerts, pc_checks,
decision, recommended_maneuver options/engines, hotspots, node probabilities)
plus the debris model's breakup events.  The only new physics is the debris
census of each hotspot: the event's simulated fragments are propagated with the
live model's own integrator (``debris_model._advance``: RK4 two-body + J2 + drag)
to the hotspot TCA and counted inside the hotspot radius.  Nothing is invented;
a value that the pipeline did not produce is ``null``.

Payload
-------
graph.nodes   objects / collision events / hotspots
              {id, kind: "object"|"event"|"hotspot", name, object_type, agency,
               probability (P(hit) = 1 - prod(1 - Pc) from the cascade planner),
               severity (worst incident alert), component, focus, degree, hotspots[]}
graph.edges   {id, source, target, kind: "screening"|"debris"|"event-parent",
               miss_distance_km, pc, tca_utc, severity, alert_id, fragments, label}
hotspots[]    {id, kind, severity, score, area{...}, debris{...}, closest_approach{...},
               avoidance[...], effect_on_others{...}, explanation, explanations{...}}
events[]      debris events (as /api/debris/events) + threatened satellites
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter

from app.core import sim_clock

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cascade", tags=["cascade-visualisation"])

SEV_RANK = {"CRITICAL": 3, "WARNING": 2, "WATCH": 1}
SPREAD_OFFSETS_S = (600.0, 2700.0, 5400.0)   # own-event spread shown 10 / 45 / 90 min after breakup
PC_TARGET = 1e-6                              # maneuver_planner target (reported, not decided here)
MAX_EVENT_HORIZON_S = 36 * 3600.0            # do not propagate fragments further than this past breakup
CACHE_SIZE = 8

_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_cache_lock = threading.Lock()


# ── Accessors (monkeypatched in tests) ─────────────────────────────────────

def _alerts_snapshot() -> dict:
    from app.core.state_cache import get_latest_alerts
    return get_latest_alerts() or {}


def _model():
    from app.core.debris_model import debris_model
    return debris_model


def _events(model) -> list:
    if model is None:
        return []
    lock = getattr(model, "_lock", None)
    if lock is not None:
        with lock:
            return list(getattr(model, "_events", {}).values())
    return list(getattr(model, "_events", {}).values())


# ── Small helpers ──────────────────────────────────────────────────────────

def _f(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _pc(alert: dict) -> float | None:
    for key in ("probability_of_collision", "p_collision"):
        v = _f(alert.get(key))
        if v is not None:
            return v
    return None


def _key(node_id) -> str:
    return str(node_id)


def _parse(ts) -> datetime | None:
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def fmt_dist(km: float | None) -> str:
    if km is None:
        return "n/a"
    return f"{km * 1000:.0f} m" if km < 1 else f"{km:.2f} km" if km < 10 else f"{km:.1f} km"


def fmt_pc(pc: float | None) -> str:
    if pc is None:
        return "n/a"
    return "0" if pc == 0 else f"{pc:.1e}"


def fmt_dt(minutes: float | None) -> str:
    if minutes is None:
        return "n/a"
    sign = "T-" if minutes >= 0 else "T+"
    m = abs(minutes)
    return f"{sign}{m:.0f} min" if m < 90 else f"{sign}{m / 60:.1f} h"


def _is_fragment(side: dict, alert: dict) -> bool:
    if alert.get("source") != "debris":
        return False
    if str(side.get("name", "")).upper().startswith("FRAG"):
        return True
    fid = alert.get("fragment_id")
    return fid is not None and _key(side.get("id")) == _key(fid)


def _side(info: dict) -> dict:
    return {"id": _key(info.get("id")), "name": info.get("name"), "agency": info.get("agency"),
            "object_type": info.get("object_type")}


# ── Graph ──────────────────────────────────────────────────────────────────

def build_graph(alerts: list[dict], node_prob: dict, events: list[dict]) -> dict:
    """Objects + collision events; edges = screening alerts, aggregated debris links, event parents."""
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}

    def node(info: dict, kind: str = "object") -> str:
        nid = _key(info.get("id"))
        n = nodes.get(nid)
        if n is None:
            n = nodes[nid] = {"id": nid, "kind": kind, "name": info.get("name") or nid,
                              "object_type": info.get("object_type"), "agency": info.get("agency"),
                              "probability": _f(node_prob.get(nid)), "severity": None,
                              "degree": 0, "hotspots": []}
        else:
            if info.get("name") and n["name"] == nid:
                n["name"] = info["name"]
            n["object_type"] = n["object_type"] or info.get("object_type")
            n["agency"] = n["agency"] or info.get("agency")
        return nid

    def bump(nid: str, severity: str | None):
        if severity and SEV_RANK.get(severity, 0) > SEV_RANK.get(nodes[nid]["severity"] or "", 0):
            nodes[nid]["severity"] = severity

    event_nodes: dict[str, str] = {}

    def event_node(ev: dict) -> str:
        eid = ev["event_id"]
        if eid not in event_nodes:
            nid = f"event:{eid}"
            nodes[nid] = {"id": nid, "kind": "event", "name": f"EVT {eid}", "event_id": eid,
                          "object_type": None, "agency": None, "probability": None,
                          "severity": "EVENT", "degree": 0, "hotspots": [],
                          "parent_ids": [_key(p) for p in ev.get("parent_ids") or []],
                          "fragment_count": ev.get("fragment_count_total", ev.get("fragment_count")),
                          "collision_utc": ev.get("collision_utc"), "status": ev.get("status")}
            event_nodes[eid] = nid
            for pid, pname in zip(ev.get("parent_ids") or [], ev.get("parent_names") or [None] * 2):
                p = node({"id": pid, "name": pname})
                miss_km = _f(ev.get("miss_distance_km"))
                eid_edge = f"parent:{eid}:{p}"
                edges[eid_edge] = {"id": eid_edge, "source": nid, "target": p, "kind": "event-parent",
                                   "miss_distance_km": miss_km, "pc": None,
                                   "tca_utc": ev.get("collision_utc"), "severity": "EVENT",
                                   "alert_id": None, "fragments": None,
                                   "label": "parent" + (f" · {fmt_dist(miss_km)} at breakup" if miss_km is not None else "")}
        return event_nodes[eid]

    for ev in events:
        event_node(ev)

    for alert in alerts:
        s1, s2 = alert.get("sat1") or {}, alert.get("sat2") or {}
        if s1.get("id") is None or s2.get("id") is None:
            continue
        sev = alert.get("severity")
        pc = _pc(alert)
        miss = _f(alert.get("miss_distance_km"))
        pe = alert.get("parent_event") if alert.get("source") == "debris" else None
        if isinstance(pe, dict) and pe.get("event_id"):
            ev_nid = event_node({"event_id": pe["event_id"], "parent_ids": pe.get("parent_ids"),
                                 "collision_utc": pe.get("collision_utc"),
                                 "fragment_count": pe.get("fragment_count")})
            threatened = [s for s in (s1, s2) if not _is_fragment(s, alert)] or [s1, s2]
            for side in threatened:
                t = node(side)
                bump(t, sev)
                eid = f"debris:{pe['event_id']}:{t}"
                e = edges.get(eid)
                if e is None:
                    edges[eid] = {"id": eid, "source": ev_nid, "target": t, "kind": "debris",
                                  "miss_distance_km": miss, "pc": pc, "tca_utc": alert.get("tca_utc"),
                                  "severity": sev, "alert_id": alert.get("id"), "fragments": 1}
                else:
                    e["fragments"] += 1
                    if pc is not None and (e["pc"] is None or pc > e["pc"]):
                        e["pc"] = pc
                    if miss is not None and (e["miss_distance_km"] is None or miss < e["miss_distance_km"]):
                        e["miss_distance_km"], e["tca_utc"], e["alert_id"] = miss, alert.get("tca_utc"), alert.get("id")
                    if SEV_RANK.get(sev, 0) > SEV_RANK.get(e["severity"] or "", 0):
                        e["severity"] = sev
            continue
        a, b = node(s1), node(s2)
        bump(a, sev)
        bump(b, sev)
        aid = alert.get("id") or f"{a}-{b}"
        edges[aid] = {"id": aid, "source": a, "target": b, "kind": "screening", "miss_distance_km": miss,
                      "pc": pc, "tca_utc": alert.get("tca_utc"), "severity": sev, "alert_id": aid,
                      "fragments": None}

    for e in edges.values():
        if "label" not in e:
            parts = []
            if e["fragments"] and e["fragments"] > 1:
                parts.append(f"{e['fragments']} frag")
            parts.append(fmt_dist(e["miss_distance_km"]))
            if e["pc"] is not None:
                parts.append(f"Pc {fmt_pc(e['pc'])}")
            e["label"] = " · ".join(parts)
        nodes[e["source"]]["degree"] += 1
        nodes[e["target"]]["degree"] += 1

    # connected components; a component is "focus" if it holds a CRITICAL/WARNING alert or an event
    parent = {k: k for k in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges.values():
        ra, rb = find(e["source"]), find(e["target"])
        if ra != rb:
            parent[ra] = rb
    comp_ids: dict[str, int] = {}
    focus: set[int] = set()
    for nid, n in nodes.items():
        c = comp_ids.setdefault(find(nid), len(comp_ids))
        n["component"] = c
        if n["kind"] == "event" or n["severity"] in ("CRITICAL", "WARNING"):
            focus.add(c)
    for n in nodes.values():
        n["focus"] = n["component"] in focus
    for e in edges.values():
        e["focus"] = nodes[e["source"]]["focus"]
    return {"nodes": nodes, "edges": list(edges.values()), "component_count": len(comp_ids),
            "focus_components": sorted(focus)}


# ── Debris census inside a hotspot ─────────────────────────────────────────

def fragment_states(ev, offsets_s: list[float]) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Fragment (r, alive) at each t_rel >= 0 (seconds after breakup), one forward pass, cached."""
    from app.core.debris_model import _advance

    res = ev.result
    wanted = sorted({round(float(t), 0) for t in offsets_s if 0.0 <= t <= MAX_EVENT_HORIZON_S})
    key = (ev.event_id, int(getattr(res, "seed", 0)), ev.collision_utc.isoformat(), tuple(wanted))
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    r = np.asarray(res.r_km, float).copy()
    v = np.asarray(res.v_kms, float).copy()
    alive = np.ones(len(r), bool)
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    t_prev = 0.0
    for t in wanted:
        if t > t_prev and alive.any():
            r, v, alive = _advance(r, v, ev.bc, alive, t - t_prev)
        t_prev = t
        out[t] = (r.copy(), alive.copy())
    with _cache_lock:
        _cache[key] = out
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return out


def _spread(pts: np.ndarray) -> dict:
    if len(pts) == 0:
        return {"n_alive": 0, "p50_km": None, "p90_km": None, "along_track_spread_km": None}
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    along = None
    if len(pts) >= 3:
        try:
            from app.routers.debris_viz import _along_track_spread
            along = round(float(_along_track_spread(pts, c)), 1)
        except Exception:  # pragma: no cover - optional helper
            along = None
    return {"n_alive": int(len(pts)), "p50_km": round(float(np.percentile(d, 50)), 2),
            "p90_km": round(float(np.percentile(d, 90)), 2), "along_track_spread_km": along}


def census_offsets(ev, when: datetime, own: bool) -> list[float]:
    """Seconds after breakup at which debris_census needs the fragment state."""
    t_rel = (when - ev.collision_utc).total_seconds()
    if t_rel < -1.0:
        return []
    t0 = max(t_rel, 0.0)
    return [t0] + ([t0 + s for s in SPREAD_OFFSETS_S] if own else [])


def debris_census(ev, center: np.ndarray, radius_km: float, when: datetime, own: bool,
                  states: dict | None = None) -> dict:
    res = ev.result
    t_rel = (when - ev.collision_utc).total_seconds()
    weight = float(getattr(res, "weight", 1.0) or 1.0)
    base = {"event_id": ev.event_id, "relation": "own_breakup" if own else "other_event",
            "parent_ids": [_key(p) for p in ev.parent_ids], "parent_names": list(ev.parent_names),
            "collision_utc": ev.collision_utc.isoformat(), "t_rel_s": round(t_rel, 1),
            "fragment_count_total": int(res.n_total), "fragments_simulated": int(res.n_sampled),
            "fragment_weight": round(weight, 4), "catastrophic": bool(res.catastrophic)}
    if t_rel < -1.0:
        return {**base, "state": "not_yet_released", "sampled_inside": 0, "represented_inside": 0,
                "nearest_fragment_km": None, "spread": None, "spread_timeline": None}
    if states is None:
        states = fragment_states(ev, census_offsets(ev, when, own))
    t0 = round(max(t_rel, 0.0), 0)
    if t0 not in states:
        return {**base, "state": "beyond_propagation_horizon", "sampled_inside": None,
                "represented_inside": None, "nearest_fragment_km": None, "spread": None, "spread_timeline": None}
    r, alive = states[t0]
    pts = r[alive]
    d = np.linalg.norm(pts - center[None, :], axis=1) if len(pts) else np.zeros(0)
    inside = int(np.count_nonzero(d <= radius_km))
    timeline = None
    if own:
        timeline = []
        for s in SPREAD_OFFSETS_S:
            k = round(max(t_rel, 0.0) + s, 0)
            if k in states:
                rr, aa = states[k]
                timeline.append({"t_after_breakup_s": k, **_spread(rr[aa])})
    return {**base, "state": "released", "sampled_inside": inside,
            "represented_inside": int(round(inside * weight)),
            "nearest_fragment_km": round(float(d.min()), 3) if len(d) else None,
            "spread": _spread(pts), "spread_timeline": timeline}


# ── Hotspot sections ───────────────────────────────────────────────────────

def _geodetic(h: dict, center: np.ndarray, when: datetime | None) -> dict | None:
    geo = h.get("geodetic")
    if isinstance(geo, dict) and geo.get("latitude_deg") is not None:
        return {k: geo.get(k) for k in ("latitude_deg", "longitude_deg", "altitude_km")}
    if when is None:
        return None
    try:
        from app.core.frames import eci_to_geodetic
        g = eci_to_geodetic(center, when)
        return {k: g.get(k) for k in ("latitude_deg", "longitude_deg", "altitude_km")}
    except Exception:  # pragma: no cover
        return None


def _pc_methods(alert: dict) -> dict:
    chk = alert.get("pc_checks") or {}
    return {"foster": _f(chk.get("foster", _pc(alert))), "chan": _f(chk.get("chan")),
            "alfano_max": _f(chk.get("alfano_max")), "monte_carlo": _f(chk.get("monte_carlo")),
            "mc_samples": chk.get("mc_samples"), "mc_note": chk.get("mc_note"),
            "spread_decades": _f(chk.get("spread_decades")), "consistent": chk.get("consistent"),
            "reported": _pc(alert), "method": alert.get("pc_method")}


def _option(o: dict, chosen: bool) -> dict:
    keys = ("candidate", "sat_id", "sat_name", "delta_v_rsw_ms", "delta_v_ms", "burn_epoch_utc",
            "new_tca_utc", "new_miss_distance_km", "new_pc_collision", "achieves_target",
            "secondary_conjunctions", "secondary_count", "secondary_max_pc", "cascade_safe",
            "cascade_check", "propellant_kg", "fuel_cost_pct", "engines", "recommended_engine",
            "engine_rule", "analytic_check")
    out = {k: o.get(k) for k in keys}
    out["secondary_conjunctions"] = [
        {**s, "id": _key(s.get("id"))} for s in (o.get("secondary_conjunctions") or [])]
    out["is_chosen"] = chosen
    return out


def avoidance_for(alert: dict) -> dict | None:
    """All re-propagated options for the alert's mover (chosen one flagged)."""
    rec = alert.get("recommended_maneuver")
    s1, s2 = alert.get("sat1") or {}, alert.get("sat2") or {}
    if not rec:
        payloads = [s for s in (s1, s2) if s.get("object_type") == "PAY" and not _is_fragment(s, alert)]
        if not payloads:
            return None
        return {"alert_id": alert.get("id"), "mover": _side(payloads[0]), "threat": _side(s2 if payloads[0] is s1 else s1),
                "maneuver_status": alert.get("maneuver_status") or "not_planned",
                "pc_before": _pc(alert), "miss_before_km": _f(alert.get("miss_distance_km")),
                "options": [], "chosen_index": None, "selection_rule": None}
    mover = _key(rec.get("sat_id"))
    threat = s2 if _key(s1.get("id")) == mover else s1
    opts = rec.get("options") or [rec]
    ci = rec.get("chosen_index") if isinstance(rec.get("chosen_index"), int) else 0
    return {"alert_id": alert.get("id"), "mover": {"id": mover, "name": rec.get("sat_name"), "agency": rec.get("agency")},
            "threat": _side(threat), "maneuver_status": alert.get("maneuver_status") or "ok",
            "pc_before": _f(rec.get("pc_before", _pc(alert))), "miss_before_km": _f(rec.get("miss_before_km", alert.get("miss_distance_km"))),
            "target_pc": _f(rec.get("target_pc")) or PC_TARGET,
            "mass_kg": _f(rec.get("mass_kg")), "mass_source": rec.get("mass_source"),
            "selection_rule": rec.get("selection_rule"), "chosen_index": ci,
            "naive_min_dv_index": rec.get("naive_min_dv_index"),
            "cascade_criteria": rec.get("cascade_criteria"),
            "options": [_option(o, i == ci) for i, o in enumerate(opts)]}


def _explain(h: dict) -> dict:
    area, deb, ca, av, eff = (h["area"], h["debris"], h["closest_approach"], h["avoidance"], h["effect_on_others"])
    geo = area.get("geodetic") or {}
    loc = (f"{abs(geo['latitude_deg']):.2f}°{'N' if geo['latitude_deg'] >= 0 else 'S'}, "
           f"{abs(geo['longitude_deg']):.2f}°{'E' if geo['longitude_deg'] >= 0 else 'W'}, "
           f"{geo['altitude_km']:.0f} km up") if geo.get("latitude_deg") is not None else "an unresolved location"
    e_area = (f"A sphere of radius {fmt_dist(area['radius_km'])} centred at {loc} at {fmt_dt(area['tca_minutes'])}. "
              f"The radius comes from {str(area['radius_source']).replace('_', ' ')}; {area['member_count']} tracked "
              f"object(s) are inside it at that instant.")
    if deb["events"]:
        bits = []
        for e in deb["events"]:
            if e["state"] == "not_yet_released":
                bits.append(f"event {e['event_id']} has not broken up yet at this time")
            elif e["state"] == "released":
                if e["relation"] == "own_breakup" and abs(e["t_rel_s"]) <= 1.0:
                    tl = e.get("spread_timeline") or []
                    after = "; ".join(f"{t['t_after_breakup_s'] / 60:.0f} min later the cloud's 90 % radius is {fmt_dist(t['p90_km'])}"
                                      for t in tl if t.get("p90_km") is not None)
                    bits.append(f"this is the breakup point of event {e['event_id']}: ~{e['fragment_count_total']} "
                                f"fragments ≥10 cm are released here" + (f" ({after})" if after else ""))
                else:
                    bits.append(f"~{e['represented_inside']} fragment(s) of event {e['event_id']} are inside the zone "
                                f"(nearest {fmt_dist(e['nearest_fragment_km'])})")
        e_deb = "Debris layer: " + "; ".join(bits) + "."
    else:
        fc = deb.get("forecast_cloud")
        e_deb = ("No breakup has been simulated yet. " + (
            f"If this pair collided, the NASA breakup model predicts ~{fc['fragment_count']} fragments ≥10 cm "
            f"({'catastrophic' if fc.get('is_catastrophic') else 'non-catastrophic'})." if fc and fc.get("fragment_count") else
            "Run the debris simulation to see the fragment cloud."))
    if ca:
        pcs = ca["pc"]
        e_ca = (f"{ca['pair'][0]['name']} and {ca['pair'][1]['name']} pass {fmt_dist(ca['miss_distance_km'])} apart "
                f"at {ca['relative_speed_kms']:.1f} km/s ({fmt_dt(ca['tca_minutes'])}). " if ca.get("relative_speed_kms") is not None else
                f"{ca['pair'][0]['name']} and {ca['pair'][1]['name']} pass {fmt_dist(ca['miss_distance_km'])} apart ({fmt_dt(ca['tca_minutes'])}). ")
        e_ca += f"Collision probability (Foster) {fmt_pc(pcs['foster'])}"
        if pcs.get("chan") is not None:
            e_ca += f", Chan {fmt_pc(pcs['chan'])}"
        if pcs.get("alfano_max") is not None:
            e_ca += f", worst-case (Alfano) {fmt_pc(pcs['alfano_max'])}"
        e_ca += f"; severity {ca.get('severity') or 'n/a'}"
        if ca.get("decision"):
            e_ca += f", decision {ca['decision'].get('action')}"
        e_ca += "."
    else:
        e_ca = "No conjunction inside this zone has a computed closest approach."
    planned = [a for a in av if a.get("options")]
    if planned:
        parts = []
        for a in planned:
            o = next((x for x in a["options"] if x["is_chosen"]), a["options"][0])
            safe = {True: "cascade-safe", False: "NOT cascade-safe"}.get(o.get("cascade_safe"),
                                                                       f"cascade check: {str(o.get('cascade_check') or 'not run').replace('_', ' ')}")
            parts.append(f"{a['mover']['name']}: {o['candidate']} ({o['delta_v_ms']:.2f} m/s) moves the miss to "
                         f"{fmt_dist(o['new_miss_distance_km'])} and Pc to {fmt_pc(o['new_pc_collision'])}, {safe}; "
                         f"engine {o.get('recommended_engine') or 'n/a'}")
        e_av = ("Re-propagated avoidance burns — " + "; ".join(parts) + ". Cryogenic LOX/LH2 has the best Isp but "
                "boils off, so it is listed for comparison only.")
    elif av:
        e_av = "No burn was planned: " + "; ".join(f"{a['mover']['name']} ({str(a['maneuver_status']).replace('_', ' ')})" for a in av) + "."
    else:
        e_av = "No operational payload in this zone can manoeuvre (debris / rocket bodies only)."
    if eff["options_checked"]:
        e_eff = (f"Each option was re-screened for 24 h against the catalogue and live fragments: "
                 f"{eff['secondary_total']} new or worsened close approach(es) found, "
                 f"{eff['unsafe_options']} option(s) unsafe. ")
    else:
        e_eff = "The 24 h secondary screening has not run for these options (time budget), so effects on others are unknown. "
    e_eff += f"{eff['downstream_count']} object(s) are downstream of this zone in the cascade graph."
    return {"area": e_area, "debris": e_deb, "closest_approach": e_ca, "avoidance": e_av, "effect_on_others": e_eff}


def _own_event(h: dict, ev_objs: list):
    pair = {_key((h.get("sat1") or {}).get("id")), _key((h.get("sat2") or {}).get("id"))}
    for ev in ev_objs:
        if ev.event_id == h.get("event_id") or {_key(p) for p in ev.parent_ids} == pair:
            return ev
    return None


def build_hotspot(idx: int, h: dict, alerts: list[dict], ev_objs: list, forecast: dict, now: datetime,
                  ev_states: dict | None = None) -> dict:
    pos = h.get("position") or {}
    center = np.array([_f(pos.get("x")) or 0.0, _f(pos.get("y")) or 0.0, _f(pos.get("z")) or 0.0])
    radius = _f(h.get("zone_radius_km")) or 0.0
    when = _parse(h.get("tca_utc"))
    members = [{"id": _key(m.get("id")), "name": m.get("name"), "distance_km": _f(m.get("distance_km"))}
               for m in h.get("affected_satellites") or []]
    member_ids = {m["id"] for m in members}
    pair = {_key((h.get("sat1") or {}).get("id")), _key((h.get("sat2") or {}).get("id"))}
    ev_id = h.get("event_id")

    def in_zone(a):
        pe = a.get("parent_event") or {}
        if h.get("kind") == "debris_event":
            return pe.get("event_id") == ev_id
        ids = {_key((a.get("sat1") or {}).get("id")), _key((a.get("sat2") or {}).get("id"))}
        return ids <= member_ids or ids == pair

    zone_alerts = [a for a in alerts if in_zone(a)]
    tca_min = (when - now).total_seconds() / 60.0 if when else None

    # (b) debris layer
    own_ev = _own_event(h, ev_objs)
    census = []
    if when is not None:
        for ev in ev_objs:
            try:
                census.append(debris_census(ev, center, radius, when, ev is own_ev,
                                            (ev_states or {}).get(ev.event_id)))
            except Exception as error:  # pragma: no cover - never fail the payload on one event
                logger.warning("debris census failed for %s: %s", ev.event_id, error)
    a_, b_ = sorted(pair)
    fc = forecast.get(f"forecast-{a_}-{b_}") or forecast.get(f"forecast-{b_}-{a_}")
    frag_threats = [a for a in alerts if a.get("source") == "debris"
                    and ({_key((a.get("sat1") or {}).get("id")), _key((a.get("sat2") or {}).get("id"))} & member_ids)]
    debris = {
        "events": census,
        "fragments_inside_total": sum(c["represented_inside"] or 0 for c in census if c.get("state") == "released") if census else 0,
        "own_event_id": own_ev.event_id if own_ev else None,
        "forecast_cloud": ({k: fc.get(k) for k in ("fragment_count", "fragment_count_basis", "is_catastrophic",
                                                    "emr_j_per_g", "parent_mass_kg", "mass_source")} if fc else None),
        "fragment_threats_to_members": len(frag_threats),
        "closest_fragment_miss_km": min((_f(a.get("miss_distance_km")) for a in frag_threats
                                         if _f(a.get("miss_distance_km")) is not None), default=None),
        "method": "fragments propagated from the breakup state (RK4 two-body+J2+drag) to the hotspot TCA; "
                  "count = sampled fragments within radius x fragment_weight",
    }

    # (c) closest approach
    lead = min((a for a in zone_alerts if _f(a.get("miss_distance_km")) is not None),
               key=lambda a: _f(a.get("miss_distance_km")), default=None)
    ca = None
    if lead is not None:
        tca = _parse(lead.get("tca_utc"))
        dec = lead.get("decision") or {}
        ml = lead.get("ml") or {}
        ca = {"alert_id": lead.get("id"), "source": lead.get("source"),
              "pair": [_side(lead.get("sat1") or {}), _side(lead.get("sat2") or {})],
              "miss_distance_km": _f(lead.get("miss_distance_km")),
              "radial_miss_km": _f(lead.get("radial_miss_km")),
              "b_t_km": _f(lead.get("b_t_km")), "b_n_km": _f(lead.get("b_n_km")),
              "relative_speed_kms": _f(lead.get("relative_speed_kms")),
              "tca_utc": lead.get("tca_utc"),
              "tca_minutes": round((tca - now).total_seconds() / 60.0, 2) if tca else _f(lead.get("tca_minutes")),
              "hbr_km": _f(lead.get("hbr_km")), "covariance_ellipse": lead.get("covariance_ellipse"),
              "sigma_source": lead.get("sigma_source"), "severity": lead.get("severity"),
              "pc": _pc_methods(lead),
              "ml_pc_surrogate": _f(ml.get("pc_surrogate")),
              "decision": ({"action": dec.get("action"), "score": _f(dec.get("score")),
                            "confidence": (dec.get("model_agreement") or {}).get("confidence"),
                            "rationale": dec.get("rationale")} if dec else None),
              "alerts_in_zone": len(zone_alerts)}

    # (d) avoidance — one entry per threatened payload (best plan of the riskiest alert it is in)
    av_by_mover: dict[str, dict] = {}
    for a in sorted(zone_alerts, key=lambda x: _pc(x) or 0.0, reverse=True):
        entry = avoidance_for(a)
        if entry is None:
            continue
        mid = entry["mover"]["id"]
        if mid not in av_by_mover or (not av_by_mover[mid]["options"] and entry["options"]):
            av_by_mover[mid] = entry
    avoidance = list(av_by_mover.values())

    # (e) effect on others
    opts = [o for a in avoidance for o in a["options"]]
    checked = [o for o in opts if o.get("cascade_check") == "ok"]
    downstream: set[str] = set()
    for a in zone_alerts:
        downstream |= {_key(d) for d in a.get("downstream_ids") or []}
    downstream -= member_ids
    effect = {"options_checked": len(checked), "options_total": len(opts),
              "secondary_total": sum(len(o["secondary_conjunctions"]) for o in checked),
              "unsafe_options": sum(1 for o in checked if o.get("cascade_safe") is False),
              "downstream_ids": sorted(downstream), "downstream_count": len(downstream)}

    out = {
        "id": f"hotspot:{idx}", "kind": h.get("kind"), "severity": h.get("severity"),
        "score": _f(h.get("hotspot_score")), "score_definition": h.get("hotspot_score_definition"),
        "event_id": ev_id or (own_ev.event_id if own_ev else None),
        "area": {"center_eci_km": [round(float(x), 3) for x in center],
                 "geodetic": _geodetic(h, center, when), "radius_km": radius,
                 "radius_source": h.get("zone_radius_source"), "tca_utc": h.get("tca_utc"),
                 "tca_minutes": round(tca_min, 2) if tca_min is not None else None,
                 "members": members, "member_count": len(members)},
        "debris": debris, "closest_approach": ca, "avoidance": avoidance, "effect_on_others": effect,
    }
    out["explanations"] = _explain(out)
    out["explanation"] = " ".join(out["explanations"][k] for k in ("area", "closest_approach", "avoidance"))
    return out


# ── Whole payload ──────────────────────────────────────────────────────────

def _event_info(ev, now: datetime, alerts: list[dict]) -> dict:
    try:
        from app.routers.debris_viz import event_summary
        info = event_summary(ev, now)
    except Exception:  # pragma: no cover
        info = {"event_id": ev.event_id, "collision_utc": ev.collision_utc.isoformat(),
                "parent_ids": list(ev.parent_ids), "parent_names": list(ev.parent_names),
                "fragment_count_total": int(ev.result.n_total)}
    miss_m = _f((ev.tca or {}).get("miss_distance_m"))
    info["miss_distance_km"] = miss_m / 1000.0 if miss_m is not None else None
    threatened: dict[str, dict] = {}
    for a in alerts:
        if (a.get("parent_event") or {}).get("event_id") != ev.event_id:
            continue
        for s in (a.get("sat1") or {}, a.get("sat2") or {}):
            if _is_fragment(s, a):
                continue
            k = _key(s.get("id"))
            t = threatened.setdefault(k, {"id": k, "name": s.get("name"), "fragments": 0,
                                          "min_miss_km": None, "max_pc": None, "tca_utc": None})
            t["fragments"] += 1
            m, p = _f(a.get("miss_distance_km")), _pc(a)
            if m is not None and (t["min_miss_km"] is None or m < t["min_miss_km"]):
                t["min_miss_km"], t["tca_utc"] = m, a.get("tca_utc")
            if p is not None and (t["max_pc"] is None or p > t["max_pc"]):
                t["max_pc"] = p
    info["threatened"] = sorted(threatened.values(), key=lambda t: -(t["max_pc"] or 0))
    return info


def build_explorer(snapshot: dict, ev_objs: list, now: datetime) -> dict:
    alerts = list(snapshot.get("alerts") or [])
    node_prob = {_key(k): v for k, v in (snapshot.get("node_probabilities") or {}).items()}
    events = [_event_info(ev, now, alerts) for ev in ev_objs]
    graph = build_graph(alerts, node_prob, events)
    forecast = {c.get("id"): c for c in snapshot.get("debris_clouds") or [] if c.get("kind") == "forecast"}
    raw_hotspots = list(snapshot.get("hotspots") or [])
    # one forward fragment pass per event covering every hotspot TCA it is needed at
    ev_states: dict[str, dict] = {}
    for ev in ev_objs:
        offsets: list[float] = []
        for h in raw_hotspots:
            when = _parse(h.get("tca_utc"))
            if when is not None:
                offsets += census_offsets(ev, when, _own_event(h, ev_objs) is ev)
        try:
            ev_states[ev.event_id] = fragment_states(ev, offsets) if offsets else {}
        except Exception as error:  # pragma: no cover
            logger.warning("fragment propagation failed for %s: %s", ev.event_id, error)
    hotspots = []
    for i, h in enumerate(raw_hotspots):
        hs = build_hotspot(i, h, alerts, ev_objs, forecast, now, ev_states)
        hotspots.append(hs)
        for m in hs["area"]["members"]:
            n = graph["nodes"].get(m["id"])
            if n is not None:
                n["hotspots"].append(hs["id"])
    nodes = list(graph["nodes"].values())
    for hs in hotspots:
        geo = hs["area"]["geodetic"] or {}
        nodes.append({"id": hs["id"], "kind": "hotspot", "name": f"HOTSPOT {hs['id'].split(':')[1]}",
                      "severity": hs["severity"], "probability": hs["score"],
                      "members": [m["id"] for m in hs["area"]["members"]],
                      "radius_km": hs["area"]["radius_km"], "latitude_deg": geo.get("latitude_deg"),
                      "longitude_deg": geo.get("longitude_deg"), "altitude_km": geo.get("altitude_km"),
                      "focus": any(graph["nodes"].get(m["id"], {}).get("focus") for m in hs["area"]["members"])})
    return {
        "sim_now_utc": now.isoformat(),
        "alerts_timestamp": snapshot.get("timestamp"),
        "graph": {"nodes": nodes, "edges": graph["edges"], "component_count": graph["component_count"],
                  "focus_components": graph["focus_components"]},
        "hotspots": hotspots,
        "events": events,
        "counts": {"alerts": len(alerts), "objects": sum(1 for n in nodes if n["kind"] == "object"),
                   "events": len(events), "hotspots": len(hotspots), "edges": len(graph["edges"])},
        "definitions": {
            "edge.miss_distance_km": "minimum predicted separation at TCA (screening: SGP4 + Brent refinement; debris: closest fragment)",
            "edge.pc": "Foster 2-D collision probability (debris edge: highest fragment Pc)",
            "node.probability": "P(hit) = 1 - prod(1 - Pc) over the object's alerts (independent encounters)",
            "hotspot.radius_km": "from the cascade planner: 3-sigma covariance major axis (>= miss) or fragment-encounter p90 spread",
            "cascade_safe": "no burn-induced or burn-worsened secondary with Pc >= 1e-6 or miss < 1 km within 24 h",
        },
    }


@router.get("/explorer")
async def cascade_explorer():
    snapshot = _alerts_snapshot()
    try:
        ev_objs = _events(_model())
    except Exception as error:  # pragma: no cover - debris model optional
        logger.warning("debris model unavailable: %s", error)
        ev_objs = []
    now = sim_clock.simulation_now()
    return await asyncio.to_thread(build_explorer, snapshot, ev_objs, now)
