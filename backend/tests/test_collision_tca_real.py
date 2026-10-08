"""Pair TCA search (find_pair_tca) and the out-of-process debris screening kernel."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sgp4.api import WGS72, Satrec, jday

from app.core import debris_model as dm
from app.core.sgp4_propagator import SGP4Propagator
from app.simulation.sim_engine import _fit_satrec_to_state

# ISS-like TLE (any LEO object works; epoch is irrelevant for the geometry test)
L1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005"
L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815599 01234"


def _crossing_pair(lead_s: float, miss_m: float, rel_speed_kms: float = 10.0):
    """Catalogue object A + SGP4-fitted B that crosses A at start+lead_s with
    the given miss distance perpendicular to the relative velocity."""
    a = Satrec.twoline2rv(L1, L2, WGS72)
    epoch = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    start = epoch + timedelta(hours=1)
    t_enc = start + timedelta(seconds=lead_s)
    jd, fr = jday(t_enc.year, t_enc.month, t_enc.day, t_enc.hour, t_enc.minute,
                   t_enc.second + t_enc.microsecond / 1e6)
    _, r_a, v_a = a.sgp4(jd, fr)
    r_a, v_a = np.array(r_a), np.array(v_a)
    r_hat = r_a / np.linalg.norm(r_a)
    v_hor = v_a - (v_a @ r_hat) * r_hat
    theta = 2.0 * np.arcsin(rel_speed_kms / (2.0 * np.linalg.norm(v_hor)))
    v_b = (v_a - v_hor) + v_hor * np.cos(theta) + np.cross(r_hat, v_hor) * np.sin(theta)
    u = (v_b - v_a) / np.linalg.norm(v_b - v_a)
    off = np.cross(u, r_hat)
    off /= np.linalg.norm(off)
    r_b = r_a + off * (miss_m / 1000.0)
    b, fit = _fit_satrec_to_state(r_b, v_b, t_enc, 99902)
    assert fit["residual_pos_m"] < 1e-2
    prop = SGP4Propagator()
    with prop._lock:
        prop._satellites[25544] = (a, "A")
        prop._satellites[99902] = (b, "B")
    return prop, start, t_enc


@pytest.mark.parametrize("lead_s,miss_m", [(5 * 3600 + 7.3, 12.0), (1234.5, 30.0)])
def test_find_pair_tca_recovers_constructed_encounter(lead_s, miss_m):
    prop, start, t_enc = _crossing_pair(lead_s, miss_m)
    tca = dm.find_pair_tca(prop, 25544, 99902, start, window_hours=24.0)
    t = datetime.fromisoformat(tca["tca_utc"])
    assert abs((t - t_enc).total_seconds()) < 1.0
    assert abs(tca["miss_distance_m"] - miss_m) < 50.0
    assert tca["passes_refined"] >= 1
    assert tca["relative_velocity_kms"] == pytest.approx(10.0, rel=0.02)
    # the coarse grid can sit up to ~50 km away at a 10 s step; refinement fixes it
    assert tca["coarse_min_km"] < 60.0


def test_find_pair_tca_with_hint_window():
    prop, start, t_enc = _crossing_pair(3 * 3600.0, 20.0)
    tca = dm.find_pair_tca(prop, 25544, 99902, start, tca_hint_utc=t_enc + timedelta(seconds=40))
    assert tca["scan"]["basis"] == "hint_window"
    assert abs((datetime.fromisoformat(tca["tca_utc"]) - t_enc).total_seconds()) < 1.0
    assert abs(tca["miss_distance_m"] - 20.0) < 50.0


def test_screen_kernel_worker_matches_in_process(monkeypatch):
    r0 = np.array([[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0]])
    vc = np.sqrt(dm.MU_KM3_S2 / 7000.0)
    v0 = np.array([[0.0, vc, 0.0], [-vc, 0.0, 0.0]])
    offsets = np.arange(0, 1801, 30.0)
    R = np.full((2, len(offsets), 3), np.nan)
    V = np.full((2, len(offsets), 3), np.nan)
    rng = np.random.default_rng(0)
    F_r = r0[[0, 0, 1]] + rng.normal(0, 1.0, (3, 3))
    F_v = v0[[0, 0, 1]] + rng.normal(0, 0.01, (3, 3))
    args = dict(F_r=F_r, F_v=F_v, F_dt=np.array([0.0, 0.0, 15.0]), F_bc=np.full(3, 0.01),
                F_alive=np.ones(3, bool), F_rel=np.zeros(3, int), F_ev=np.zeros(3, int), n_events=1,
                S_r_all=R, S_v_all=V, rest=[0, 1], rest_r0=r0, rest_v0=v0, offsets=offsets,
                step_s=30.0, threshold_km=50.0, excluded=np.zeros((1, 2), bool))
    local = dm._screen_kernel(**{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in args.items()})
    monkeypatch.setenv("DEBRIS_WORKER", "1")
    remote = dm._worker.call("screen", **args)
    assert len(local["f"]) > 0
    for key in ("f", "s", "miss", "t_off"):
        np.testing.assert_allclose(remote[key], local[key], rtol=0, atol=1e-9)
