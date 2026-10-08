"""
Ground-truth tests for future-window conjunction screening (app/core/screening.py).

Truth is computed independently by brute force: a 1 s scan over the whole
window followed by a 1 ms scan around the minimum, using the same dynamics
(sgp4 for TLE objects, two-body+J2 for state-vector objects).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sgp4.api import Satrec, WGS72

from app.core import screening as S
from app.core.conjunction import find_tca

SIM_T0 = datetime(2026, 3, 14, 6, 0, 0, tzinfo=timezone.utc)
N_RAD_MIN = math.sqrt(398600.8 / 7000.0**3) * 60.0  # circular 7000 km, rad/min


def _satrec(satnum, incl_deg, raan_deg, mo_rad, epoch=SIM_T0, ecc=1e-4):
    jd, fr = S._jd_fr(epoch)
    sat = Satrec()
    sat.sgp4init(WGS72, "i", satnum, (jd + fr) - 2433281.5, 0.0, 0.0, 0.0, ecc,
                 0.0, math.radians(incl_deg), mo_rad, N_RAD_MIN, math.radians(raan_deg))
    return sat


class _Prop:
    """Minimal propagator stand-in: screen() only needs _satellites/_lock."""

    def __init__(self, sats: dict):
        import threading
        self._satellites = sats
        self._lock = threading.RLock()

    def state(self, nid, t):
        sat = self._satellites[nid][0]
        jd, fr = S._jd_fr(SIM_T0)
        e, r, v = sat.sgp4(jd, fr + t / 86400.0)
        assert e == 0
        return np.array(r), np.array(v)


class _State:
    error_code = 0

    def __init__(self, nid, name, r, v):
        self.norad_id, self.name = nid, name
        self.x, self.y, self.z = r
        self.vx, self.vy, self.vz = v
        self.epoch_utc = SIM_T0.isoformat()


def _brute(dist_fn, t_end):
    ts = np.arange(0.0, t_end + 1e-9, 1.0)
    d = np.array([dist_fn(t) for t in ts])
    k = int(np.argmin(d))
    fine = np.arange(max(0.0, ts[k] - 1.0), min(t_end, ts[k] + 1.0), 0.001)
    df = np.array([dist_fn(t) for t in fine])
    j = int(np.argmin(df))
    return float(fine[j]), float(df[j])


def test_tle_pair_tca_and_miss_match_brute_force():
    # Two circular 7000 km orbits sharing a node, 45 deg apart in inclination,
    # nearly co-phased -> they cross at the node ~5 min after epoch.
    a = _satrec(90001, 51.6, 0.0, -0.30)
    b = _satrec(90002, 97.0, 0.0, -0.30 + 0.0005)
    prop = _Prop({90001: (a, "SYN-A"), 90002: (b, "SYN-B")})
    states = [_State(n, nm, *prop.state(n, 0.0)) for n, nm in ((90001, "SYN-A"), (90002, "SYN-B"))]

    window_h = 2.0
    alerts = S.screen(states, SIM_T0, window_hours=window_h, step_s=60, threshold_km=25,
                      propagator=prop)
    assert len(alerts) == 1
    al = alerts[0]

    def dist(t):
        return float(np.linalg.norm(prop.state(90001, t)[0] - prop.state(90002, t)[0]))

    t_true, miss_true = _brute(dist, window_h * 3600.0)
    t_scr = al["tca_hours"] * 3600.0
    assert abs(t_scr - t_true) < 1.0, (t_scr, t_true)
    assert abs(al["miss_distance_km"] - miss_true) < 0.1, (al["miss_distance_km"], miss_true)
    # Contract fields
    for key in ("pc_method", "hbr_km", "b_t_km", "b_n_km", "covariance_ellipse", "sigma_source",
                "severity", "cpi_score", "relative_speed_kms", "tca_minutes"):
        assert key in al
    assert al["sigma_source"] == "tle_age_model"
    assert math.isclose(math.hypot(al["b_t_km"], al["b_n_km"]), al["miss_distance_km"], rel_tol=1e-3)

    # Legacy find_tca now refines too and starts from the given (sim) epoch.
    ev = find_tca(prop, 90001, 90002, start=SIM_T0, hours_ahead=window_h, steps=120)
    assert abs(ev.tca_hours * 3600.0 - t_true) < 1.0
    assert abs(ev.miss_distance_km - miss_true) < 0.1


def test_pair_2000km_apart_now_meeting_in_2h_is_found():
    a = _satrec(91001, 53.0, 40.0, 1.0)
    prop = _Prop({91001: (a, "SYN-SAT")})
    t_enc = 7200.0
    rA, vA = prop.state(91001, t_enc)
    # Debris crossing at 70 deg, 3 km radial offset, at t = 2 h.
    r_hat = rA / np.linalg.norm(rA)
    ang = math.radians(70.0)
    vB = vA * math.cos(ang) + np.cross(r_hat, vA) * math.sin(ang) + r_hat * np.dot(r_hat, vA) * (1 - math.cos(ang))
    rB = rA + 3.0 * r_hat
    rB0, vB0 = S.propagate_j2(rB, vB, -t_enc, max_substep_s=5.0)
    rA0, _ = prop.state(91001, 0.0)
    assert np.linalg.norm(rB0 - rA0) > 2000.0, "setup: must be far apart now"

    states = [_State(91001, "SYN-SAT", *prop.state(91001, 0.0))]
    extra = [{"id": "frag-1", "name": "SYN DEB", "object_type": "DEB", "agency": "TEST",
              "r_km": rB0.tolist(), "v_kms": vB0.tolist()}]
    alerts = S.screen(states, SIM_T0, window_hours=6, step_s=60, threshold_km=25,
                      extra_objects=extra, propagator=prop)
    assert len(alerts) == 1
    al = alerts[0]

    def dist(t):
        rb, _ = S.propagate_j2(rB0, vB0, t, max_substep_s=5.0)
        return float(np.linalg.norm(prop.state(91001, t)[0] - rb))

    # brute force only near the planted encounter (the full 6 h scan is slow in RK4)
    ts = np.arange(t_enc - 120, t_enc + 120, 1.0)
    k = int(np.argmin([dist(t) for t in ts]))
    fine = np.arange(ts[k] - 1, ts[k] + 1, 0.001)
    dd = [dist(t) for t in fine]
    t_true, miss_true = float(fine[int(np.argmin(dd))]), float(min(dd))

    assert abs(al["tca_hours"] * 3600.0 - t_true) < 1.0
    assert abs(al["miss_distance_km"] - miss_true) < 0.1
    assert abs(al["tca_hours"] - 2.0) < 0.01
    assert al["sat2"]["object_type"] == "DEB" or al["sat1"]["object_type"] == "DEB"


def test_foster_quadrature_matches_dblquad():
    from app.core.analytics import foster_integrate_dblquad

    cases = [
        (np.array([0.3, 0.1]), np.array([[0.8, 0.1], [0.1, 0.05]]), 0.02),
        (np.array([0.0, 0.0]), np.diag([0.01, 0.01]), 0.01),
        (np.array([1.5, -0.4]), np.array([[4.0, -0.5], [-0.5, 0.3]]), 0.05),
    ]
    for b, C, hbr in cases:
        fast = S.foster_pc(b, C, hbr)
        ref = foster_integrate_dblquad(b, C, hbr)
        assert fast == pytest.approx(ref, rel=1e-4, abs=1e-14)


def test_covariance_grows_with_tle_age_and_severity_rule():
    s0 = S.tle_age_sigmas_km(0.0)
    s5 = S.tle_age_sigmas_km(5.0)
    assert s5[1] > s0[1] and s5[1] == pytest.approx(S.SIGMA0_RTN_KM[1] + 5 * S.SIGMA_GROWTH_RTN_KM_PER_DAY[1])
    assert S.classify_severity(2e-4, 10.0) == "CRITICAL"
    assert S.classify_severity(0.0, 0.5) == "CRITICAL"
    assert S.classify_severity(2e-6, 10.0) == "WARNING"
    assert S.classify_severity(0.0, 4.0) == "WARNING"
    assert S.classify_severity(1e-9, 20.0) == "WATCH"


def test_hbr_from_object_type():
    deb = S.object_meta(99999, "COSMOS 2251 DEB", agency="X")
    rb = S.object_meta(99998, "SL-8 R/B", agency="X")
    assert deb["object_type"] == "DEB" and rb["object_type"] == "R/B"
    assert deb["radius_m"] < rb["radius_m"]
