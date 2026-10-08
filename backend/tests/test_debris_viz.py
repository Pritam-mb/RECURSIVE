"""Debris impact replay (/api/debris/events*): real propagation, consistent with the live model."""

from __future__ import annotations

import json
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
from app.routers import debris_viz  # noqa: E402

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
    """Two payloads crossing at 80 deg with ~50 m miss at TCA; event created 40 min before (pending)."""
    speed = 7.6127
    r = np.array([6878.0, 0.0, 0.0])
    va = speed * np.array([0.0, math.cos(math.radians(51.6)), math.sin(math.radians(51.6))])
    vb = speed * np.array([0.0, math.cos(math.radians(131.6)), math.sin(math.radians(131.6))])
    prop = SGP4Propagator()
    prop._satellites = {90001: (_satrec_through(r, va, TCA, 90001), "TEST-A"),
                        90002: (_satrec_through(r + np.array([0.0, 0.02, 0.04]), vb, TCA, 90002), "TEST-B")}
    model = DebrisModel()
    summary = model.simulate_collision_from_pair(90001, 90002, prop, T0, window_hours=2.0, max_fragments=600)
    ev = model.get_cloud(summary["event_id"])
    return model, prop, ev


def _payload(ev, prop, alerts=(), **kw):
    debris_viz._cache.clear()
    return json.loads(debris_viz.build_replay(ev, prop, list(alerts), **kw))


def test_replay_matches_live_model_positions(scenario):
    model, prop, ev = scenario
    p = _payload(ev, prop, t0_min=-15, t1_min=90, step_s=30, max_fragments=400)
    ids = p["fragments"]["ids"]
    idx = np.array([int(f.split(":F")[1]) for f in ids])
    for minutes in (10, 45, 90):
        t = ev.collision_utc + timedelta(minutes=minutes)
        k = p["t_rel_s"].index(minutes * 60.0)
        assert p["times_utc"][k] == t.isoformat()
        ev.advance_to(t)                       # the live model's own propagation to this sim time
        live_r, live_alive = ev.r[idx], ev.alive[idx]
        rep = p["fragments"]["positions"][k]
        for j in range(len(idx)):
            if live_alive[j]:
                assert rep[j] is not None
                assert np.linalg.norm(np.array(rep[j]) - live_r[j]) < 0.5
            else:
                assert rep[j] is None
    ev.advance_to(ev.collision_utc)


def test_fragments_null_before_collision_and_at_impact_point_at_zero(scenario):
    _, prop, ev = scenario
    p = _payload(ev, prop, t0_min=-15, t1_min=30, step_s=30)
    t_rel = np.array(p["t_rel_s"])
    assert t_rel[0] == -900.0 and 0.0 in p["t_rel_s"]
    for k in np.nonzero(t_rel < 0)[0]:
        assert all(x is None for x in p["fragments"]["positions"][k])
        assert p["envelope"][k]["centroid"] is None and p["envelope"][k]["n_alive"] == 0
    k0 = p["t_rel_s"].index(0.0)
    impact = np.array(ev.result.impact_point_km)
    pts = np.array(p["fragments"]["positions"][k0], float)
    assert np.all(np.linalg.norm(pts - impact, axis=1) < 0.1)
    # cloud spreads with time
    assert p["envelope"][-1]["p90_km"] > p["envelope"][k0 + 2]["p90_km"] > 0


def test_parents_converge_at_collision(scenario):
    _, prop, ev = scenario
    p = _payload(ev, prop, t0_min=-15, t1_min=30, step_s=30)
    k0 = p["t_rel_s"].index(0.0)
    a, b = (np.array(p["parents"][str(i)]["positions"], float) for i in ev.parent_ids)
    d = np.linalg.norm(a - b, axis=1)
    assert d[k0] < 1.0
    assert d[0] > 100.0 and d[-1] > 100.0          # far apart 15 min before / 30 min after
    assert int(np.argmin(d)) == k0
    # parents keep their SGP4 trajectories after the collision ("would have flown")
    assert all(x is not None for x in p["parents"][str(ev.parent_ids[0])]["positions"])


def test_stratified_sample_covers_parents_and_sizes(scenario):
    _, _, ev = scenario
    res = ev.result
    assert res.n_sampled > 100
    idx, info = debris_viz.stratified_sample(res, 100, forced=[3, 7])
    assert len(idx) <= 100 and {3, 7} <= set(idx.tolist())
    assert set(res.parent_index[idx].tolist()) == set(res.parent_index.tolist())
    assert res.lc_m[idx].max() == pytest.approx(res.lc_m.max(), rel=0.5)
    assert len({(s["parent_index"], s["size_bin"]) for s in info["strata"]}) >= 4


def _client(model, prop, monkeypatch):
    monkeypatch.setattr(debris_viz, "_model", lambda: model)
    monkeypatch.setattr(debris_viz, "_propagator", lambda: prop)
    app = FastAPI()
    app.include_router(debris_viz.router)
    return TestClient(app)


def test_endpoints_schema(scenario, monkeypatch):
    model, prop, ev = scenario
    monkeypatch.setattr(sim_clock, "simulation_now", lambda: T0)
    fid = f"{ev.event_id}:F0005"
    model._last_alerts = [{
        "sat1": {"id": 12345, "name": "VICTIM", "agency": "ESA"}, "tca_utc": TCA.isoformat(),
        "miss_distance_km": 1.2, "probability_of_collision": 3e-5, "fragment_id": fid,
        "parent_event": {"event_id": ev.event_id}}]
    client = _client(model, prop, monkeypatch)
    debris_viz._cache.clear()

    r = client.get("/api/debris/events")
    assert r.status_code == 200
    e = r.json()["events"][0]
    for key in ("event_id", "collision_utc", "status", "parent_ids", "parent_names", "parent_agencies",
                "fragment_count_total", "fragments_simulated", "catastrophic", "relative_velocity_kms",
                "collision_point_eci_km", "seed"):
        assert key in e
    assert e["status"] == "pending_impact"

    r = client.get(f"/api/debris/events/{ev.event_id}/replay")
    assert r.status_code == 200
    p = r.json()
    for key in ("event_id", "collision_utc", "frame", "step_s", "times_utc", "t_rel_s", "parents",
                "fragments", "envelope", "threatened", "provenance"):
        assert key in p
    for key in ("ids", "size_m", "am_m2_kg", "dv_ms", "parent_of", "positions"):
        assert key in p["fragments"]
    for key in ("t_rel_s", "centroid", "p50_km", "p90_km", "along_track_spread_km", "n_alive"):
        assert key in p["envelope"][0]
    for key in ("propagator", "seed", "sampled_from", "note"):
        assert key in p["provenance"]
    nt = len(p["t_rel_s"])
    assert len(p["times_utc"]) == nt == len(p["envelope"]) == len(p["fragments"]["positions"])
    assert len(p["fragments"]["ids"]) <= 400
    assert all(len(row) == len(p["fragments"]["ids"]) for row in p["fragments"]["positions"])
    th = p["threatened"][0]
    assert th["sat_id"] == 12345 and th["fragment_id"] == fid
    assert p["fragments"]["ids"][th["fragment_display_index"]] == fid    # threatening fragment is shown
    assert int(r.headers["X-Payload-Bytes"]) == len(r.content) < 6_000_000

    r2 = client.get(f"/api/debris/events/{ev.event_id}/replay")
    assert r2.json()["provenance"]["cached"] is True
    assert client.get("/api/debris/events/nope/replay").status_code == 404
    assert client.get(f"/api/debris/events/{ev.event_id}/replay?max_fragments=5000").status_code == 422
    model._last_alerts = []
