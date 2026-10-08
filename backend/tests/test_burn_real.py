"""
Executed burns: SGP4 reference + propagated J2 deviation (app/core/sgp4_propagator.py).

Checks against physics, not against the implementation:
  * a zero burn leaves the trajectory untouched (the old TLE re-fit moved the
    object ~9 km at once and ~180 km after 6 h);
  * a small along-track burn produces the Clohessy-Wiltshire secular drift
    delta_s(t) ~ -3 * dv * t;
  * the manoeuvre planner's predicted post-burn miss distance equals the miss
    distance the screening pipeline finds after the burn is executed.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sgp4.api import Satrec, WGS72

from app.core import screening as S
from app.core.sgp4_propagator import SGP4Propagator
from app.services import maneuver_planner as mp
from app.simulation.sim_engine import _fit_satrec_to_state

T0 = datetime(2026, 3, 14, 6, 0, 0, tzinfo=timezone.utc)
A_ID, B_ID = 90001, 90002


def _host_satrec(epoch=T0 - timedelta(days=1)):
    jd, fr = S._jd_fr(epoch)
    n = math.sqrt(398600.8 / 6900.0 ** 3) * 60.0
    sat = Satrec()
    sat.sgp4init(WGS72, "i", A_ID, (jd + fr) - 2433281.5, 2e-5, 0.0, 0.0, 1.2e-3,
                 math.radians(40.0), math.radians(53.0), 1.0, n, math.radians(120.0))
    return sat


def _prop(sats: dict) -> SGP4Propagator:
    p = SGP4Propagator()
    for nid, (sat, name) in sats.items():
        p._satellites[nid] = (sat, name)
    return p


def _pos(p: SGP4Propagator, nid: int, t: datetime) -> np.ndarray:
    s = p.propagate_one(nid, t)
    assert s.error_code == 0
    return np.array([s.x, s.y, s.z]), np.array([s.vx, s.vy, s.vz])


def test_zero_burn_does_not_move_satellite():
    p = _prop({A_ID: (_host_satrec(), "HOST")})
    t6 = T0 + timedelta(hours=6)
    r_ref, _ = _pos(p, A_ID, t6)
    res = p.apply_delta_v(A_ID, 0.0, 0.0, 0.0, frame="RSW", epoch=T0)
    assert res["status"] == "success"
    for t in (T0, T0 + timedelta(minutes=1), t6):
        r_new, _ = _pos(p, A_ID, t)
        r_nom = np.array(p._satellites[A_ID][0].sgp4(*S._jd_fr(t))[1])
        assert np.linalg.norm(r_new - r_nom) * 1000.0 < 1.0
    assert np.linalg.norm(_pos(p, A_ID, t6)[0] - r_ref) * 1000.0 < 1.0


def test_along_track_burn_matches_clohessy_wiltshire_drift():
    p = _prop({A_ID: (_host_satrec(), "HOST")})
    dv_ms = 0.1
    nominal = _prop({A_ID: (_host_satrec(), "HOST")})
    assert p.apply_delta_v(A_ID, 0.0, dv_ms, 0.0, frame="RSW", epoch=T0)["status"] == "success"
    for hours in (3.0, 6.0):
        t = T0 + timedelta(hours=hours)
        r_b, _ = _pos(p, A_ID, t)
        r_n, v_n = _pos(nominal, A_ID, t)
        Q = mp.rsw_basis(r_n, v_n)
        d_rsw = Q.T @ (r_b - r_n)                       # km
        expected = -3.0 * (dv_ms / 1000.0) * hours * 3600.0
        assert d_rsw[1] == pytest.approx(expected, rel=0.20), (hours, d_rsw, expected)
        # radial excursion stays bounded (CW: |dr| <= 4 dv / n)
        n = math.sqrt(398600.4418 / np.linalg.norm(r_n) ** 3)
        assert abs(d_rsw[0]) <= 4.0 * dv_ms / 1000.0 / n * 1.2


def test_burn_survives_clone_and_screening_grid():
    p = _prop({A_ID: (_host_satrec(), "HOST")})
    p.apply_delta_v(A_ID, 0.0, 0.5, 0.0, epoch=T0)
    c = p.clone()
    t = T0 + timedelta(hours=2)
    assert np.allclose(_pos(p, A_ID, t)[0], _pos(c, A_ID, t)[0], atol=1e-9)
    # clone is independent
    c.apply_delta_v(A_ID, 0.0, 0.5, 0.0, epoch=T0)
    assert len(p.burn_history(A_ID)) == 1 and len(c.burn_history(A_ID)) == 2
    # vectorised screening grid carries the deviation (matches scalar propagation)
    win = S._Window([S._Obj(A_ID, "HOST", satrec=p.trajectory(A_ID))], T0, 121, 60.0)
    r_scalar, _ = _pos(p, A_ID, T0 + timedelta(seconds=120 * 60.0))
    assert np.linalg.norm(win.R[0, 120] - r_scalar) * 1000.0 < 1e-3


def _crossing_pair():
    """B crosses A at T0 + 3 h, ~10 km/s, ~60 m miss (fitted SGP4 elements)."""
    a = _host_satrec()
    t_enc = T0 + timedelta(hours=3)
    e, r_a, v_a = a.sgp4(*S._jd_fr(t_enc))
    r_a, v_a = np.array(r_a), np.array(v_a)
    r_hat = r_a / np.linalg.norm(r_a)
    v_hor = v_a - (v_a @ r_hat) * r_hat
    theta = math.radians(80.0)
    v_b = (v_a @ r_hat) * r_hat + v_hor * math.cos(theta) + np.cross(r_hat, v_hor) * math.sin(theta)
    r_b = r_a + 0.06 * r_hat
    b, _ = _fit_satrec_to_state(r_b, v_b, t_enc, B_ID)
    return a, b, t_enc


class _St:
    error_code = 0

    def __init__(self, nid, name):
        self.norad_id, self.name = nid, name


def test_planner_prediction_matches_executed_burn():
    a, b, t_enc = _crossing_pair()
    p = _prop({A_ID: (a, "HOST"), B_ID: (b, "CROSSER")})
    # A already flew an earlier burn: the planner must start from that trajectory.
    p.apply_delta_v(A_ID, 0.0, 0.0, 0.0, epoch=T0 - timedelta(hours=1))

    pre = S.screen([_St(A_ID, "HOST"), _St(B_ID, "CROSSER")], T0, window_hours=4.0,
                   threshold_km=25.0, propagator=p)
    assert pre, "crossing not detected"
    alert = dict(pre[0])
    tca_s = (datetime.fromisoformat(alert["tca_utc"]) - T0).total_seconds()
    assert abs(tca_s - 3 * 3600.0) < 5.0 and alert["miss_distance_km"] < 0.2

    job = mp.EncounterJob(
        alert=alert, is_sat1=[alert["sat1"]["id"] == A_ID],
        movers=[mp.Mover(mp.ObjectTrack(A_ID, "HOST", satrec=p.trajectory(A_ID)), owner_known=True)],
        others=[mp.ObjectTrack(B_ID, "CROSSER", satrec=p.trajectory(B_ID))],
        tca_s=tca_s, hbr_km=alert.get("hbr_km", 0.01), ages_days=(1.0, 0.1),
    )
    stats = mp.plan_maneuvers([job], T0, time_budget_s=30.0)
    assert stats["status"] == "ok"
    rec = job.alert["_maneuver_result"]["recommended_maneuver"]

    res = p.apply_delta_v(A_ID, *rec["delta_v_rsw_ms"], frame="RSW", epoch=T0)
    assert res["status"] == "success" and res["burn_count"] == 2

    post = S.screen([_St(A_ID, "HOST"), _St(B_ID, "CROSSER")], T0, window_hours=4.0,
                    threshold_km=50.0, propagator=p)
    assert post, "post-burn encounter not found by screening"
    executed = post[0]
    diff_m = abs(executed["miss_distance_km"] - rec["new_miss_distance_km"]) * 1000.0
    print(f"planner {rec['new_miss_distance_km']*1000:.2f} m vs executed "
          f"{executed['miss_distance_km']*1000:.2f} m (diff {diff_m:.3f} m), dv {rec['delta_v_rsw_ms']}")
    assert diff_m < 10.0
    t_exec = datetime.fromisoformat(executed["tca_utc"])
    assert abs((t_exec - datetime.fromisoformat(rec["new_tca_utc"])).total_seconds()) < 1.0
