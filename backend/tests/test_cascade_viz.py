"""Cascade & hotspot explorer (/api/cascade/explorer): values come from the pipeline / real propagation."""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sgp4.api import jday

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core import sim_clock  # noqa: E402
from app.core.debris_model import DebrisModel  # noqa: E402
from app.core.sgp4_propagator import SGP4Propagator, _rebuild_satrec  # noqa: E402
from app.routers import cascade_viz as cv  # noqa: E402

T0 = datetime(2026, 5, 8, 12, tzinfo=timezone.utc)
TCA = T0 + timedelta(minutes=40)


def _satrec_through(r, v, epoch, satnum):
    r_in, v_in = np.array(r, float), np.array(v, float)
    jd, fr = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second)
    for _ in range(10):
        sat = _rebuild_satrec(r_in, v_in, epoch, satnum)
        _, rr, vv = sat.sgp4(jd, fr)
        r_in += np.asarray(r) - rr
        v_in += np.asarray(v) - vv
    return sat


@pytest.fixture(scope="module")
def scenario():
    speed = 7.6127
    r = np.array([6878.0, 0.0, 0.0])
    va = speed * np.array([0.0, math.cos(math.radians(51.6)), math.sin(math.radians(51.6))])
    vb = speed * np.array([0.0, math.cos(math.radians(131.6)), math.sin(math.radians(131.6))])
    prop = SGP4Propagator()
    prop._satellites = {90001: (_satrec_through(r, va, TCA, 90001), "TEST-A"),
                        90002: (_satrec_through(r + np.array([0.0, 0.02, 0.04]), vb, TCA, 90002), "TEST-B")}
    model = DebrisModel()
    summary = model.simulate_collision_from_pair(90001, 90002, prop, T0, window_hours=2.0, max_fragments=400)
    return model, model.get_cloud(summary["event_id"])


def _engines():
    return [{"engine": "Aerojet Rocketdyne RL10B-2", "family": "cryogenic_biprop", "isp_s": 465.5, "thrust_n": 110100.0,
             "propellant": "LOX/LH2", "prop_mass_kg": 0.03, "burn_time_s": 0.001, "finite_burn_ok": True,
             "practical_for_satellites": False, "note": "LH2 boils off"},
            {"engine": "Aerojet MR-106L", "family": "hydrazine_monoprop", "isp_s": 235.0, "thrust_n": 22.0,
             "propellant": "N2H4", "prop_mass_kg": 0.06, "burn_time_s": 6.8, "finite_burn_ok": True,
             "practical_for_satellites": True, "note": ""}]


def _snapshot(ev):
    eid = ev.event_id
    impact = [float(x) for x in ev.result.impact_point_km]
    pe = {"event_id": eid, "collision_utc": ev.collision_utc.isoformat(), "parent_ids": [90001, 90002], "fragment_count": 99}
    screening = {
        "id": "90001-90002", "source": "screening",
        "sat1": {"id": 90001, "name": "TEST-A", "agency": "ESA", "object_type": "PAY"},
        "sat2": {"id": 90002, "name": "TEST-B", "agency": "NASA", "object_type": "PAY"},
        "tca_utc": ev.collision_utc.isoformat(), "miss_distance_km": 0.045, "relative_speed_kms": 10.3,
        "probability_of_collision": 2e-4, "severity": "CRITICAL",
        "pc_checks": {"foster": 2e-4, "chan": 2.1e-4, "alfano_max": 3e-3, "monte_carlo": None, "consistent": True},
        "decision": {"action": "MANOEUVRE", "score": 71.0, "model_agreement": {"confidence": "high"}, "rationale": "x"},
        "downstream_ids": [777, 90003],
        "recommended_maneuver": {
            "sat_id": 90001, "sat_name": "TEST-A", "pc_before": 2e-4, "miss_before_km": 0.045, "chosen_index": 1,
            "selection_rule": "min dv among cascade-safe", "options": [
                {"candidate": "+S 0.2 m/s", "delta_v_ms": 0.2, "new_miss_distance_km": 0.6, "new_pc_collision": 5e-7,
                 "cascade_safe": False, "cascade_check": "ok", "engines": _engines(), "recommended_engine": "Aerojet MR-106L",
                 "secondary_conjunctions": [{"id": 90003, "name": "THIRD", "miss_km": 0.03, "pc": 4e-5,
                                             "tca_utc": TCA.isoformat(), "burn_induced": True}]},
                {"candidate": "+R 0.5 m/s", "delta_v_ms": 0.5, "new_miss_distance_km": 1.4, "new_pc_collision": 1e-9,
                 "cascade_safe": True, "cascade_check": "ok", "engines": _engines(), "recommended_engine": "Aerojet MR-106L",
                 "secondary_conjunctions": []}]},
    }
    debris = [{"id": f"debris:{eid}:{k}-555", "source": "debris",
               "sat1": {"id": 555, "name": "VICTIM", "agency": "JAXA", "object_type": "PAY"},
               "sat2": {"id": 90020000 + k, "name": f"FRAG {eid} #{k}", "object_type": "DEB"},
               "tca_utc": (TCA + timedelta(minutes=50 + k)).isoformat(), "miss_distance_km": 2.0 - 0.5 * k,
               "probability_of_collision": 1e-7 * (k + 1), "severity": "WARNING", "parent_event": pe,
               "fragment_id": f"{eid}:F{k:04d}"} for k in range(3)]
    watch = {"id": "1-2", "source": "screening", "sat1": {"id": 1, "name": "A"}, "sat2": {"id": 2, "name": "B"},
             "miss_distance_km": 12.0, "probability_of_collision": 1e-9, "severity": "WATCH"}
    hotspot = {"kind": "conjunction", "event_id": None, "sat1": screening["sat1"], "sat2": screening["sat2"],
               "tca_utc": (ev.collision_utc + timedelta(minutes=10)).isoformat(), "severity": "CRITICAL",
               "hotspot_score": 2e-4, "position": {"x": impact[0], "y": impact[1], "z": impact[2]},
               "zone_radius_km": 50.0, "zone_radius_source": "covariance_3sigma_major_axis",
               "affected_satellites": [{"id": 90001, "name": "TEST-A", "distance_km": 0.02},
                                       {"id": 90002, "name": "TEST-B", "distance_km": 0.02}]}
    return {"alerts": [screening, *debris, watch], "hotspots": [hotspot],
            "node_probabilities": {"90001": 2e-4, "555": 6e-7}, "debris_clouds": []}


def test_graph_edges_and_focus(scenario):
    _, ev = scenario
    out = cv.build_explorer(_snapshot(ev), [ev], T0)
    edges = {e["id"]: e for e in out["graph"]["edges"]}
    nodes = {n["id"]: n for n in out["graph"]["nodes"]}
    # screening edge carries the alert's own numbers
    s = edges["90001-90002"]
    assert s["kind"] == "screening" and s["miss_distance_km"] == 0.045 and s["pc"] == 2e-4
    assert "45 m" in s["label"] and "Pc 2.0e-04" in s["label"]
    # debris alerts aggregate into one event -> victim edge (closest miss, highest Pc, fragment count)
    d = edges[f"debris:{ev.event_id}:555"]
    assert d["fragments"] == 3 and d["miss_distance_km"] == pytest.approx(1.0) and d["pc"] == pytest.approx(3e-7)
    assert d["source"] == f"event:{ev.event_id}"
    # event -> parents with the breakup miss from the debris model
    p = edges[f"parent:{ev.event_id}:90001"]
    assert p["kind"] == "event-parent"
    assert p["miss_distance_km"] == pytest.approx(ev.tca["miss_distance_m"] / 1000.0)
    # node probabilities come from the cascade planner, absent -> null
    assert nodes["90001"]["probability"] == 2e-4 and nodes["1"]["probability"] is None
    # focus: the event component is focus, the isolated WATCH pair is not
    assert nodes["555"]["focus"] and nodes["90001"]["focus"] and not nodes["1"]["focus"]
    assert nodes[f"event:{ev.event_id}"]["kind"] == "event"
    assert nodes["hotspot:0"]["kind"] == "hotspot" and set(nodes["hotspot:0"]["members"]) == {"90001", "90002"}


def test_debris_census_matches_live_propagation(scenario):
    _, ev = scenario
    out = cv.build_explorer(_snapshot(ev), [ev], T0)
    h = out["hotspots"][0]
    c = h["debris"]["events"][0]
    assert c["relation"] == "own_breakup" and c["state"] == "released" and c["t_rel_s"] == pytest.approx(600, abs=1)
    # independent: the live model's own propagation to the same time
    r, _, alive = ev.state_at(ev.collision_utc + timedelta(seconds=600))
    center = np.asarray(ev.result.impact_point_km, float)
    d = np.linalg.norm(r[alive] - center, axis=1)
    assert c["sampled_inside"] == int(np.count_nonzero(d <= 50.0))
    assert c["represented_inside"] == int(round(c["sampled_inside"] * ev.result.weight))
    assert c["nearest_fragment_km"] == pytest.approx(float(d.min()), abs=1e-3)
    assert 0 < c["sampled_inside"] <= int(alive.sum())
    # spread grows after the breakup
    tl = c["spread_timeline"]
    assert [t["t_after_breakup_s"] for t in tl] == [1200.0, 3300.0, 6000.0]
    assert tl[-1]["p90_km"] > tl[0]["p90_km"] > 0


def test_debris_not_released_before_breakup(scenario):
    _, ev = scenario
    snap = _snapshot(ev)
    snap["hotspots"][0]["tca_utc"] = (ev.collision_utc - timedelta(minutes=5)).isoformat()
    c = cv.build_explorer(snap, [ev], T0)["hotspots"][0]["debris"]["events"][0]
    assert c["state"] == "not_yet_released" and c["represented_inside"] == 0


def test_sections_closest_avoidance_effect(scenario):
    _, ev = scenario
    h = cv.build_explorer(_snapshot(ev), [ev], T0)["hotspots"][0]
    ca = h["closest_approach"]
    assert ca["miss_distance_km"] == 0.045 and ca["pc"]["chan"] == 2.1e-4 and ca["pc"]["monte_carlo"] is None
    assert ca["decision"]["action"] == "MANOEUVRE"
    assert ca["tca_minutes"] == pytest.approx((ev.collision_utc - T0).total_seconds() / 60.0, abs=0.01)
    av = h["avoidance"]
    assert len(av) == 1 and av[0]["mover"]["id"] == "90001" and av[0]["threat"]["id"] == "90002"
    opts = av[0]["options"]
    assert [o["is_chosen"] for o in opts] == [False, True]
    assert opts[0]["secondary_conjunctions"][0]["id"] == "90003"
    assert any(e["family"] == "cryogenic_biprop" for e in opts[1]["engines"])
    eff = h["effect_on_others"]
    assert eff["options_checked"] == 2 and eff["secondary_total"] == 1 and eff["unsafe_options"] == 1
    assert eff["downstream_ids"] == ["777", "90003"]
    ex = h["explanations"]
    assert set(ex) == {"area", "debris", "closest_approach", "avoidance", "effect_on_others"}
    assert "45 m" in ex["closest_approach"] and "+R 0.5 m/s" in ex["avoidance"] and "cascade-safe" in ex["avoidance"]


def test_missing_values_are_null(scenario):
    snap = {"alerts": [{"id": "1-2", "sat1": {"id": 1}, "sat2": {"id": 2}, "severity": "WARNING"}],
            "hotspots": [{"kind": "conjunction", "sat1": {"id": 1}, "sat2": {"id": 2},
                          "position": {"x": 7000.0, "y": 0.0, "z": 0.0}, "zone_radius_km": 3.0,
                          "affected_satellites": [{"id": 1}, {"id": 2}]}]}
    out = cv.build_explorer(snap, [], T0)
    e = out["graph"]["edges"][0]
    assert e["miss_distance_km"] is None and e["pc"] is None
    h = out["hotspots"][0]
    assert h["closest_approach"] is None and h["area"]["geodetic"] is None and h["avoidance"] == []


def test_endpoint(scenario, monkeypatch):
    model, ev = scenario
    monkeypatch.setattr(cv, "_alerts_snapshot", lambda: _snapshot(ev))
    monkeypatch.setattr(cv, "_model", lambda: model)
    monkeypatch.setattr(sim_clock, "simulation_now", lambda: T0)
    app = FastAPI()
    app.include_router(cv.router)
    r = TestClient(app).get("/api/cascade/explorer")
    assert r.status_code == 200
    body = r.json()
    assert body["counts"]["events"] == 1 and body["counts"]["hotspots"] == 1
    assert body["events"][0]["threatened"][0]["id"] == "555"
    assert body["hotspots"][0]["area"]["geodetic"]["altitude_km"] > 400
