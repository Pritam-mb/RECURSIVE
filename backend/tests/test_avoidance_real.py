"""Cascade-safe avoidance, propulsion maths and analytic (CW / Hohmann / vis-viva) cross-checks."""

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

from app.core import analytic_checks as ac  # noqa: E402
from app.core import propulsion  # noqa: E402
from app.core.screening import compute_pc, screen  # noqa: E402
from app.core.sgp4_propagator import SGP4Propagator, SatelliteState, _rebuild_satrec  # noqa: E402
from app.services.cascade_planner import CascadePlanner  # noqa: E402

T0 = datetime(2026, 5, 8, 12, tzinfo=timezone.utc)
TCA = T0 + timedelta(hours=6)


def _satrec_through(r, v, epoch, satnum):
    """SGP4 Satrec whose state at `epoch` is (r, v) (fixed point on the osculating fit)."""
    r_in, v_in = np.array(r, float), np.array(v, float)
    jd, fr = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second)
    for _ in range(10):
        sat = _rebuild_satrec(r_in, v_in, epoch, satnum)
        _, rr, vv = sat.sgp4(jd, fr)
        r_in += np.asarray(r) - rr
        v_in += np.asarray(v) - vv
    return sat


def _state(sat, nid, name, t):
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
    _, r, v = sat.sgp4(jd, fr)
    s = SatelliteState(nid, name)
    s.x, s.y, s.z = r
    s.vx, s.vy, s.vz = v
    return s


def _base_scenario():
    """Payload A (mover) and payload B meet with 54 m miss 6 h after T0."""
    speed = 7.6127
    r = np.array([6878.0, 0.0, 0.0])
    va = speed * np.array([0.0, math.cos(math.radians(51.6)), math.sin(math.radians(51.6))])
    vb = speed * np.array([0.0, math.cos(math.radians(97.5)), math.sin(math.radians(97.5))])
    a = _satrec_through(r, va, TCA, 90001)
    b = _satrec_through(r + np.array([0.0, 0.02, 0.05]), vb, TCA, 90002)
    pc0 = compute_pc(r, va, r + np.array([0.0, 0.02, 0.05]), vb, age1_days=0.5, age2_days=0.5, hbr_km=0.01)["pc"]
    return a, b, pc0


def _alert(pc0):
    return {"id": "90001-90002", "source": "screening",
            "sat1": {"id": 90001, "name": "TEST-A", "agency": "USA", "object_type": "PAY"},
            # B is a defunct object: only A can manoeuvre.
            "sat2": {"id": 90002, "name": "TEST-B DEB", "agency": "USA", "object_type": "DEB"},
            "p_collision": pc0, "probability_of_collision": pc0, "miss_distance_km": 0.0539,
            "tca_utc": TCA.isoformat(), "tca_minutes": 360.0, "hbr_km": 0.01, "tle_age_days": [0.5, 0.5]}


def _run(prop, ids_names):
    states = [_state(prop._satellites[i][0], i, n, T0) for i, n in ids_names]
    return CascadePlanner().analyze_snapshot(states, [_alert(_base_scenario()[2])], prop, T0, run_rankers=False)


def test_planner_rejects_burn_that_creates_secondary_conjunction():
    a, b, pc0 = _base_scenario()
    assert pc0 > 1e-4

    # 1. Without a third object: the planner's naive choice (min Δv reaching Pc < 1e-6).
    prop = SGP4Propagator()
    prop._satellites = {90001: (a, "TEST-A"), 90002: (b, "TEST-B DEB")}
    out1 = _run(prop, [(90001, "TEST-A"), (90002, "TEST-B DEB")])
    rec1 = out1["alerts"][0]["recommended_maneuver"]
    assert rec1["cascade_safe"] is True and rec1["achieved_target"] is True
    naive = rec1["options"][rec1["chosen_index"]]
    assert rec1["chosen_index"] == rec1["naive_min_dv_index"]

    # 2. Execute the naive burn on a clone with the REAL burn model and find where A would be
    #    9 h after T0; put a third payload C on a crossing orbit through exactly that point.
    t_c = T0 + timedelta(hours=9)
    what_if = prop.clone()
    res = what_if.apply_delta_v(90001, *naive["delta_v_rsw_ms"], frame="RSW", epoch=T0)
    assert res["status"] == "success"
    s_burned = what_if.propagate_one(90001, t_c)
    s_nominal = prop.propagate_one(90001, t_c)
    r_b = np.array([s_burned.x, s_burned.y, s_burned.z])
    r_n = np.array([s_nominal.x, s_nominal.y, s_nominal.z])
    assert np.linalg.norm(r_b - r_n) > 5.0          # the naive burn moved A by > 5 km at t_c
    u = r_b / np.linalg.norm(r_b)
    h_hat = np.cross(r_b, [s_burned.vx, s_burned.vy, s_burned.vz]); h_hat /= np.linalg.norm(h_hat)
    s_hat = np.cross(h_hat, u)
    vc = math.sqrt(398600.4418 / np.linalg.norm(r_b))
    v_c = vc * (math.cos(math.radians(60)) * s_hat + math.sin(math.radians(60)) * h_hat)   # 60° crossing
    c = _satrec_through(r_b + 0.03 * u, v_c, t_c, 90003)

    # Ground truth with the full screening (agent A's screen) on the burned clone:
    what_if._satellites[90003] = (c, "TEST-C")
    st = [_state(what_if._satellites[i][0], i, n, T0) for i, n in ((90001, "TEST-A"), (90003, "TEST-C"))]
    truth = [x for x in screen(st, T0, propagator=what_if) if {x["sat1"]["id"], x["sat2"]["id"]} == {90001, 90003}]
    assert truth and truth[0]["miss_distance_km"] < 1.0       # naive burn => secondary conjunction with C

    # 3. With C in the catalogue the planner must flag the naive option and choose another.
    prop._satellites[90003] = (c, "TEST-C")
    out2 = _run(prop, [(90001, "TEST-A"), (90002, "TEST-B DEB"), (90003, "TEST-C")])
    rec2 = out2["alerts"][0]["recommended_maneuver"]
    opts = rec2["options"]
    bad = next(o for o in opts if o["candidate"] == naive["candidate"])
    assert bad["cascade_safe"] is False
    sec = next(s for s in bad["secondary_conjunctions"] if s["id"] == 90003)
    assert sec["burn_induced"] and sec["miss_km"] < 1.0
    assert sec["miss_km"] == pytest.approx(truth[0]["miss_distance_km"], abs=0.05)   # agrees with screen()
    chosen = opts[rec2["chosen_index"]]
    assert chosen["candidate"] != naive["candidate"]
    assert chosen["cascade_safe"] is True and chosen["achieves_target"] is True
    assert rec2["cascade_safe"] is True and rec2["rescreen_scope"] == "pair+catalogue_24h"
    assert "cascade-safe" in rec2["selection_rule"]
    # The original threat (B) is never listed as a secondary.
    assert all(s["id"] != 90002 for o in opts for s in o["secondary_conjunctions"])
    # Top-level fields describe the chosen option.
    assert rec2["delta_v_ms"] == chosen["delta_v_ms"] and rec2["new_pc_collision"] == chosen["new_pc_collision"]
    # Every option carries propulsion options and the analytic CW cross-check.
    for o in opts:
        assert len(o["engines"]) == len(propulsion.ENGINES)
        assert o["recommended_engine"] in {e["engine"] for e in propulsion.ENGINES}
        assert o["analytic_check"]["method"] == "Clohessy-Wiltshire"


def test_cw_matches_numerical_option_shift():
    a, b, pc0 = _base_scenario()
    prop = SGP4Propagator()
    prop._satellites = {90001: (a, "TEST-A"), 90002: (b, "TEST-B DEB")}
    rec = _run(prop, [(90001, "TEST-A"), (90002, "TEST-B DEB")])["alerts"][0]["recommended_maneuver"]
    for o in rec["options"]:
        chk = o["analytic_check"]
        if abs(o["delta_v_rsw_ms"][1]) > 0:          # along-track burns: secular drift dominates
            assert abs(chk["numeric_km"]) > 0.1
            assert chk["rel_error"] < 0.05


# ── Propulsion maths ───────────────────────────────────────────────────────

@pytest.mark.parametrize("isp,thrust,dv", [(220.0, 1.0, 1.0), (450.0, 490.0, 2.0), (1600.0, 0.083, 0.5), (65.0, 1.0, 0.2)])
def test_tsiolkovsky_matches_numeric_mass_integration(isp, thrust, dv):
    m0 = 300.0
    num = propulsion.integrate_constant_thrust(m0, dv, isp, thrust)
    assert num["prop_mass_kg"] == pytest.approx(propulsion.tsiolkovsky_propellant_kg(m0, dv, isp), rel=1e-6)
    assert num["burn_time_s"] == pytest.approx(propulsion.burn_time_s(m0, dv, isp, thrust), rel=1e-6)
    # first-order m0·Δv/F agrees to O(Δv / Isp g0)
    assert propulsion.burn_time_first_order_s(m0, dv, thrust) == pytest.approx(num["burn_time_s"], rel=2e-3)


def test_engine_rules():
    res = propulsion.evaluate_engines(1.0, 300.0, lead_time_s=6 * 3600.0, orbital_period_s=5677.0)
    by = {e["engine"]: e for e in res["engines"]}
    # Electric: 1 m/s on 300 kg at 39 mN takes ~2 h -> not an impulsive burn.
    assert by["Busek BHT-600"]["finite_burn_ok"] is False
    assert by["Aerojet MR-106L"]["finite_burn_ok"] is True
    # Cryogenic is never recommended for a satellite even though its Isp is highest chemical.
    assert res["recommended_engine"] != "Aerojet Rocketdyne RL10B-2"
    ok = [e for e in res["engines"] if e["finite_burn_ok"] and e["practical_for_satellites"]]
    assert by[res["recommended_engine"]]["prop_mass_kg"] == min(e["prop_mass_kg"] for e in ok)
    # Higher Isp -> less propellant (Tsiolkovsky monotonicity).
    assert by["NASA/L3 NSTAR"]["prop_mass_kg"] < by["Aerojet MR-106L"]["prop_mass_kg"] < by["Cold-gas N2 thruster"]["prop_mass_kg"]


# ── Standard formulas vs numerics ─────────────────────────────────────────

def test_cw_vs_numeric_small_dv_short_time():
    for dv_rsw in ([0.0, 0.05, 0.0], [0.05, 0.0, 0.0], [0.03, -0.04, 0.0]):
        r0, v0 = ac._circular_state(550.0)
        n = ac.mean_motion(float(np.linalg.norm(r0)))
        t = 2 * 3600.0
        cw = ac.cw_along_track_shift_km(t, n, dv_rsw)
        num = ac.numeric_along_track_shift_km(r0, v0, dv_rsw, t, j2=False)
        assert num == pytest.approx(cw, rel=5e-3, abs=2e-3)


def test_validation_checks_pass():
    checks = ac.validation_checks()
    assert {c["name"].split(" ")[0] for c in checks} >= {"CW", "Hohmann", "Vis-viva", "Rocket"}
    for c in checks:
        assert c["pass"], c
        assert set(c) >= {"name", "standard_formula", "our_value", "reference_value", "abs_error",
                          "rel_error", "pass", "tolerance", "source"}


def test_hohmann_formula():
    h = ac.hohmann(6878.0, 6898.0)
    # small-Δr limit: Δv_total ≈ (Δr / 2) · n
    n = ac.mean_motion(6888.0)
    assert h["total_kms"] == pytest.approx(0.5 * 20.0 * n, rel=1e-3)
