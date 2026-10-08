"""Conjunction predictor + fragment-avoidance manoeuvre: values are computed, not invented."""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sgp4.api import jday

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core import breakup as sbm  # noqa: E402
from app.core import sim_clock  # noqa: E402
from app.core.conjunction import compute_cpi_score  # noqa: E402
from app.core.debris_model import object_physical_properties  # noqa: E402
from app.core.screening import compute_pc, foster_pc, object_meta  # noqa: E402
from app.core.sgp4_propagator import _rebuild_satrec  # noqa: E402
from app.services import maneuver_planner as mp  # noqa: E402
from app.services.cascade_planner import CascadePlanner  # noqa: E402
from app.services.conjunction_predictor import build_collision_prediction, parse_epoch  # noqa: E402

T0 = datetime(2026, 5, 8, 12, tzinfo=timezone.utc)
SPEED = 7.6127
R_TCA = np.array([6878.0, 0.0, 0.0])
V1 = SPEED * np.array([0.0, np.cos(np.radians(51.6)), np.sin(np.radians(51.6))])
V2 = SPEED * np.array([0.0, np.cos(np.radians(97.5)), np.sin(np.radians(97.5))])


class _StubPropagator:
    norad_ids: list[int] = []

    def propagate_one(self, norad_id, epoch=None):
        return None


def _result(offset_km, tca):
    r2 = R_TCA + np.asarray(offset_km, float)
    return SimpleNamespace(
        tca_utc=tca.isoformat(),
        miss_distance_m=float(np.linalg.norm(offset_km)) * 1000.0,
        relative_velocity_kms=float(np.linalg.norm(V2 - V1)),
        position_a_eci=list(R_TCA), velocity_a_eci=list(V1),
        position_b_eci=list(r2), velocity_b_eci=list(V2),
        collision_detected=False, trajectory_a=[], trajectory_b=[],
    )


# ── Conjunction predictor ────────────────────────────────────────────────────

@pytest.mark.parametrize("offset", [[0.0, 0.02, 0.05], [0.0, 0.3, 0.8], [0.0, 1.5, 2.0]])
def test_predictor_values_are_computed(offset):
    tca = T0 + timedelta(minutes=10)
    sat_a = SimpleNamespace(norad_id=900001, name="SAT-A", epoch_utc=T0)
    sat_b = SimpleNamespace(norad_id=900002, name="SAT-B", epoch_utc=T0)
    payload = build_collision_prediction(_StubPropagator(), sat_a, sat_b, _result(offset, tca),
                                         collision_confirmed=False)
    a = payload["alerts"][0]
    miss_km = float(np.linalg.norm(offset))

    # Real TCA relative to the reference epoch (not "now = TCA").
    assert a["tca_hours"] == pytest.approx(10 / 60, abs=1e-4)
    assert a["tca_minutes"] == pytest.approx(10.0, abs=1e-3)

    # Foster Pc with TLE-age covariance (state-only objects: age = propagation span) and object_meta HBR.
    age = 600.0 / 86400.0
    hbr = (object_meta(900001, "SAT-A")["radius_m"] + object_meta(900002, "SAT-B")["radius_m"]) / 1000.0
    ref = compute_pc(R_TCA, V1, R_TCA + np.array(offset), V2, age1_days=age, age2_days=age, hbr_km=hbr)
    assert a["p_collision"] == pytest.approx(ref["pc"], rel=1e-9)
    assert a["hbr_km"] == pytest.approx(hbr, abs=1e-6)
    assert a["covariance_ellipse"]["a"] == ref["covariance_ellipse"]["a"]
    assert a["zone_radius_km"] == pytest.approx(ref["covariance_ellipse"]["a"] / 1000.0, abs=1e-3)

    # CPI from compute_cpi_score, not 10 - 1.8 * miss.
    cpi = compute_cpi_score(ref["pc"], miss_km, tca_hours=10 / 60,
                            relative_velocity_kms=float(np.linalg.norm(V2 - V1)), tle_age_hours=24 * age)
    assert a["cpi_score"] == pytest.approx(cpi, abs=1e-4)
    assert a["cpi_score"] != pytest.approx(max(1.0, min(10.0, 10.0 - 1.8 * miss_km)))

    # SBM fragment forecast from the two masses (tagged).
    props = [object_physical_properties(900001, "SAT-A"), object_physical_properties(900002, "SAT-B")]
    m_ref, _, _ = sbm.sbm_reference_mass(props[0]["mass_kg"], props[1]["mass_kg"], float(np.linalg.norm(V2 - V1)))
    bf = payload["breakup_forecast"]
    assert bf["expected_fragments"] == round(sbm.sbm_cumulative_count(m_ref, sbm.DEFAULT_LC_MIN_M))
    assert bf["mass_source"] == [p["mass_source"] for p in props]

    blob = json.dumps(payload, default=str)
    assert "predicted_fragments" not in blob and "affection_rate" not in blob


def test_parse_epoch_fallback_is_sim_clock():
    try:
        sim_clock.set_offset_hours(20.0)
        t = parse_epoch(None)
        assert abs((t - sim_clock.simulation_now()).total_seconds()) < 5
        assert (t - datetime.now(timezone.utc)).total_seconds() > 19 * 3600
    finally:
        sim_clock.reset()


# ── Fragment avoidance manoeuvre (drag-propagated fragment_state) ──────────

def _satrec_through(r, v, epoch, satnum):
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


def test_fragment_alert_gets_repropagated_maneuver_with_drag():
    tca = T0 + timedelta(hours=6)
    bc = 0.05                                      # Cd*A/m, m^2/kg (high-A/m fragment)
    offset = np.array([0.0, 0.02, 0.05])           # 54 m miss at TCA
    sat = _satrec_through(R_TCA, V1, tca, 90011)
    # Fragment state at T0 = the TCA state back-propagated with two-body + J2 + drag.
    rf, vf = sbm.propagate((R_TCA + offset)[None, :], V2[None, :], np.array([bc]),
                           -(tca - T0).total_seconds(), max_step_s=30.0)
    sigma = 0.2
    hbr = 0.0025
    pc0 = foster_pc(np.array([np.linalg.norm(offset), 0.0]), np.eye(2) * sigma ** 2, hbr)
    assert pc0 > 1e-6

    frag_id = 90_000_000 + 10_001
    alert = {
        "id": f"debris:EVT:1-90011", "source": "debris",
        "sat1": {"id": 90011, "name": "TEST-SAT", "agency": "USA", "object_type": "PAY"},
        "sat2": {"id": frag_id, "name": "FRAG EVT #1", "agency": "Debris", "object_type": "DEB"},
        "tca_utc": tca.isoformat(), "tca_minutes": 360.0, "miss_distance_km": 0.0539,
        "p_collision": pc0, "probability_of_collision": pc0, "hbr_km": hbr,
        "covariance_model": "isotropic_bplane", "covariance_ellipse": {"a": sigma * 1000, "b": sigma * 1000},
        "fragment_state": {"r_km": rf[0].tolist(), "v_kms": vf[0].tolist(), "epoch_utc": T0.isoformat(),
                           "ballistic_coeff_m2_kg": bc, "dynamics": "two_body+J2+drag(vallado_exp)"},
    }
    out = CascadePlanner().analyze_snapshot([], [alert], _Prop({90011: (sat, "TEST-SAT")}), T0, run_rankers=False)
    a = out["alerts"][0]
    rec = a.get("recommended_maneuver")
    assert rec is not None, a.get("maneuver_status")
    assert rec["verified_by"] == "repropagation"
    assert "drag" in rec["other_object_propagation"]
    # Zero-burn re-propagation of the fragment with drag reproduces the encounter.
    assert rec["miss_before_km"] == pytest.approx(0.0539, abs=1e-2)
    assert rec["pc_before_recomputed"] == pytest.approx(pc0, rel=0.05)
    assert rec["achieved_target"] is True and rec["new_pc_collision"] < 1e-6 < pc0
    assert rec["sat_id"] == 90011 and 0 < rec["delta_v_ms"] <= 2.0

    # Control: a drag-free J2 propagation of the same fragment state misses the encounter
    # geometry by far more, so the drag model is what makes the plan correct.
    rj, _ = mp.propagate_j2(rf[0], vf[0], (tca - T0).total_seconds())
    assert np.linalg.norm(rj[0] - (R_TCA + offset)) > 0.2
