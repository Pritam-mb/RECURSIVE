"""Cascade graph / manoeuvre / agency tests against computed (not scripted) values."""

from __future__ import annotations

import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sgp4.api import jday

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core import satcat  # noqa: E402
from app.core.agency import infer_agency  # noqa: E402
from app.core.agency_authority import authority_manager  # noqa: E402
from app.core.sgp4_propagator import _rebuild_satrec  # noqa: E402
from app.services.cascade_planner import CascadePlanner  # noqa: E402

T0 = datetime(2026, 5, 8, 12, tzinfo=timezone.utc)


def _alert(aid, s1, s2, pc, **extra):
    return {"id": aid, "source": extra.pop("source", "screening"), "sat1": s1, "sat2": s2,
            "p_collision": pc, "probability_of_collision": pc, "miss_distance_km": 1.0,
            "tca_utc": (T0 + timedelta(hours=2)).isoformat(), "tca_minutes": 120.0, **extra}


# ── 1. Cascade depth = BFS hops ────────────────────────────────────────────

def test_depth_is_bfs_hops_on_collision_fragment_satellite_chain():
    event = {"event_id": "EVT1", "collision_utc": T0.isoformat(), "parent_ids": [100, 101], "fragment_count": 500}
    frag = {"id": 900001, "name": "FRAG EVT1 #1", "agency": "Debris", "object_type": "DEB"}
    s200 = {"id": 200, "name": "SAT-200", "agency": "USA", "object_type": "PAY"}
    s300 = {"id": 300, "name": "SAT-300", "agency": "China", "object_type": "PAY"}
    alerts = [
        _alert("debris:EVT1:1-200", s200, frag, 2e-4, source="debris", parent_event=event),
        _alert("200-300", s200, s300, 3e-5),
        _alert("400-500", {"id": 400, "name": "A", "object_type": "PAY"}, {"id": 500, "name": "B", "object_type": "PAY"}, 1e-7),
    ]
    out = CascadePlanner().analyze_snapshot([], alerts, None, T0, plan_maneuvers=False, run_rankers=False)
    by_id = {a["id"]: a for a in out["alerts"]}

    # event(0) -> fragment(1) -> threatened satellite(2) -> its conjunction partner(3)
    assert by_id["debris:EVT1:1-200"]["cascade_depth"] == 2
    assert by_id["200-300"]["cascade_depth"] == 3
    assert by_id["200-300"]["upstream_event"] == "EVT1"
    assert 300 in by_id["debris:EVT1:1-200"]["downstream_ids"]
    # An isolated conjunction is its own primary event: one hop, no upstream.
    assert by_id["400-500"]["cascade_depth"] == 1
    assert by_id["400-500"]["upstream_event"] is None
    assert out["cascade_depth"] == 3
    # Node probability: P(hit by any threat) = 1 - prod(1 - Pc)
    expected = 1 - (1 - 2e-4) * (1 - 3e-5)
    assert out["node_probabilities"]["200"] == pytest.approx(expected, rel=1e-9)


def test_empty_alerts_give_zero_depth():
    out = CascadePlanner().analyze_snapshot([], [], None, T0, run_rankers=False)
    assert out["cascade_depth"] == 0
    assert out["cascade_plan"] == []


# ── 2. Manoeuvre lowers the re-propagated Pc ───────────────────────────────

def _satrec_through(r, v, epoch, satnum):
    """SGP4 Satrec whose position at `epoch` is exactly r (fixed-point on the osculating fit)."""
    r_in, v_in = np.array(r, float), np.array(v, float)
    jd, fr = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second)
    for _ in range(8):
        sat = _rebuild_satrec(r_in, v_in, epoch, satnum)
        _, rr, vv = sat.sgp4(jd, fr)
        r_in += np.asarray(r) - rr
        v_in += np.asarray(v) - vv
    return sat


class _Prop:
    def __init__(self, sats):
        self._satellites = sats
        self._lock = threading.RLock()


def test_recommended_maneuver_lowers_repropagated_pc():
    from app.core.screening import compute_pc

    tca = T0 + timedelta(hours=6)
    speed = 7.6127
    r = np.array([6878.0, 0.0, 0.0])
    v1 = speed * np.array([0.0, np.cos(np.radians(51.6)), np.sin(np.radians(51.6))])
    v2 = speed * np.array([0.0, np.cos(np.radians(97.5)), np.sin(np.radians(97.5))])
    offset = np.array([0.0, 0.02, 0.05])           # 54 m miss at TCA
    a = _satrec_through(r, v1, tca, 90001)
    b = _satrec_through(r + offset, v2, tca, 90002)
    pc0 = compute_pc(r, v1, r + offset, v2, age1_days=0.5, age2_days=0.5, hbr_km=0.01)["pc"]
    assert pc0 > 1e-4

    alert = _alert("90001-90002",
                   {"id": 90001, "name": "TEST-A", "agency": "USA", "object_type": "PAY"},
                   {"id": 90002, "name": "TEST-B", "agency": "USA", "object_type": "PAY"},
                   pc0, tca_utc=tca.isoformat(), tca_minutes=360.0, miss_distance_km=0.0539,
                   hbr_km=0.01, tle_age_days=[0.5, 0.5])
    prop = _Prop({90001: (a, "TEST-A"), 90002: (b, "TEST-B")})
    out = CascadePlanner().analyze_snapshot([], [alert], prop, T0, run_rankers=False)
    rec = out["alerts"][0]["recommended_maneuver"]

    assert rec is not None and rec["verified_by"] == "repropagation"
    # Zero-burn re-propagation reproduces the screening geometry and Pc.
    assert rec["pc_before_recomputed"] == pytest.approx(pc0, rel=1e-3)
    assert rec["miss_before_km"] == pytest.approx(0.0539, abs=1e-3)
    # The chosen burn really lowers Pc below the target and opens the miss distance.
    assert rec["achieved_target"] is True
    assert rec["new_pc_collision"] < 1e-6 < pc0
    assert rec["new_miss_distance_km"] > 1.0
    assert 0 < rec["delta_v_ms"] <= 2.0
    assert rec["fuel_model"] == "rocket_equation" and rec["fuel_cost_pct"] > 0
    # The plan entry mirrors the verified recommendation.
    assert out["cascade_plan"][0]["satellite_id"] == rec["sat_id"]
    assert out["cascade_plan"][0]["risk_after"] == rec["new_pc_collision"]


def test_along_track_drift_matches_orbital_mechanics():
    """A +S burn of dv produces ~3*dv*t along-track drift (CW secular term) after t."""
    from app.services.maneuver_planner import propagate_j2, rsw_basis

    r = np.array([6878.0, 0.0, 0.0])
    v = 7.6127 * np.array([0.0, np.cos(np.radians(51.6)), np.sin(np.radians(51.6))])
    dv = rsw_basis(r, v) @ np.array([0.0, 0.05, 0.0]) / 1000.0
    t = 6 * 3600.0
    r1, _ = propagate_j2(r, v, t)
    r2, _ = propagate_j2(r, v + dv, t)
    drift = float(np.linalg.norm(r2 - r1))
    assert drift == pytest.approx(3 * 0.05e-3 * t, rel=0.15)


# ── 3. Agencies from the SATCAT snapshot ───────────────────────────────────

def test_satcat_lookup_from_snapshot():
    iss = satcat.lookup(25544)
    assert iss["source"] == "celestrak_satcat"
    assert iss["owner"] == "ISS" and iss["intl_designator"] == "1998-067A"
    assert iss["object_type"] == "PAY" and iss["rcs_size"] == "LARGE"
    vanguard = satcat.lookup(5)
    assert vanguard["owner"] == "US" and vanguard["launch_year"] == 1958


def test_infer_agency_uses_satcat_owner_first():
    assert infer_agency("VANGUARD 1", 5) == "USA"
    assert infer_agency("ISS (ZARYA)", 25544) == "ISS"
    assert infer_agency("STARLINK-1007") == "SpaceX"


def test_unknown_fraction_for_tracked_catalog_is_small():
    from app.data.tle_fetcher import load_local_tles

    tles = load_local_tles(500)
    unknown = sum(infer_agency(t["name"], t["norad_id"]) == "Unknown" for t in tles)
    assert unknown / len(tles) < 0.05


def test_unknown_owner_is_not_universally_commandable():
    sid = authority_manager.create_session("SpaceX")
    # Real catalogued object with unresolved owner (Alpha-5 analyst object in the TLE file)
    assert authority_manager.explain_authority(sid, 270000, "TBA - TO BE ASSIGNED")["allowed"] is False
    # Debris cannot be commanded by an agency session
    assert authority_manager.check_authority(sid, 745, "THOR ABLESTAR DEB") is False
    # Synthetic scenario object (in no catalogue) is commandable
    assert authority_manager.explain_authority(sid, 99_999_001, "SCENARIO SAT")["rule"] == "synthetic_object_no_owner"
