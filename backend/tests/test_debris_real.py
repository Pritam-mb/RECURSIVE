"""Collision -> NASA SBM breakup -> fragment propagation -> debris alerts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from app.core import breakup as sbm
from app.core.debris_model import DebrisModel


def _circular(alt_km=550.0, inc_deg=53.0):
    r = sbm.R_EARTH_KM + alt_km
    v = np.sqrt(sbm.MU_KM3_S2 / r)
    i = np.radians(inc_deg)
    return np.array([r, 0.0, 0.0]), np.array([0.0, v * np.cos(i), v * np.sin(i)])


def test_catastrophic_threshold_is_40_j_per_g():
    m_ref, cat, emr = sbm.sbm_reference_mass(1000.0, 1.0, 0.25)   # 0.5*1*250^2/1000 = 31 J/kg
    assert not cat and emr == pytest.approx(31.25)
    m_ref, cat, emr = sbm.sbm_reference_mass(1000.0, 10.0, 3.0)   # 45 000 J/kg = 45 J/g
    assert cat and m_ref == pytest.approx(1010.0)


def test_fragment_count_matches_sbm_formula():
    r, va = _circular()
    _, vb = _circular(inc_deg=-53.0)
    res = sbm.simulate_breakup(r, va, 1000.0, r, vb, 800.0, seed=1, max_fragments=5000)
    assert res.catastrophic
    lc_max = res.lc_max_m
    expected = 0.1 * 1800.0 ** 0.75 * (0.1 ** -1.71 - lc_max ** -1.71)
    assert res.n_total == round(expected)
    # untruncated N(>=10cm) for M = 1800 kg is ~1410, i.e. hundreds not tens
    assert 1000 < res.n_total < 2000
    # sampled size distribution follows the Lc^-1.71 power law
    frac_above_20cm = np.mean(res.lc_m >= 0.2)
    expected_frac = (0.2 ** -1.71 - lc_max ** -1.71) / (0.1 ** -1.71 - lc_max ** -1.71)
    assert frac_above_20cm == pytest.approx(expected_frac, abs=0.03)


def test_no_hyperbolic_fragments_and_momentum_conserved():
    r, va = _circular(400.0)
    _, vb = _circular(400.0, inc_deg=97.0)
    res = sbm.simulate_breakup(r, va, 1500.0, r, vb, 1500.0, seed=3, max_fragments=1000)
    energy = 0.5 * np.einsum("ij,ij->i", res.v_kms, res.v_kms) - sbm.MU_KM3_S2 / np.linalg.norm(res.r_km, axis=1)
    assert np.all(energy < 0.0)
    assert res.momentum_residual_after < 1e-2


def test_fragment_energy_drift_small():
    r, va = _circular(800.0)
    _, vb = _circular(800.0, inc_deg=120.0)
    res = sbm.simulate_breakup(r, va, 1000.0, r, vb, 1000.0, seed=5, max_fragments=400)
    rr, vv = res.r_km, res.v_kms
    a = -sbm.MU_KM3_S2 / (2 * (0.5 * (vv ** 2).sum(1) - sbm.MU_KM3_S2 / np.linalg.norm(rr, axis=1)))
    h = np.linalg.norm(np.cross(rr, vv), axis=1)
    e = np.sqrt(np.maximum(0, 1 - h * h / (sbm.MU_KM3_S2 * a)))
    keep = a * (1 - e) - sbm.R_EARTH_KM > 200.0
    e0 = sbm.specific_energy_j2(rr[keep], vv[keep])
    r1, v1 = sbm.propagate(rr[keep], vv[keep], None, 3 * 3600.0, max_step_s=30.0)
    e1 = sbm.specific_energy_j2(r1, v1)
    assert np.max(np.abs((e1 - e0) / e0)) < 1e-5


def test_density_table_is_physical():
    rho = sbm.atmospheric_density([400.0])[0]
    assert 1e-12 < rho < 1e-11   # Vallado: 3.725e-12 kg/m^3 at 400 km


def test_fragment_on_satellite_path_produces_debris_alert():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    model = DebrisModel()
    r, v = _circular(700.0, 98.0)
    ev = model.simulate_collision(r.tolist(), v.tolist(), 900.0, 900.0, 10.0,
                                  max_fragments_simulated=50, event_id="t-evt", collision_utc=t0)
    # Satellite placed 20 minutes "upstream" of fragment #0 on a crossing orbit:
    # back-propagate fragment 0 by 20 min; put a satellite at the same point at
    # that later time going the other way, then back-propagate the satellite.
    f_r, f_v = ev.result.r_km[:1], ev.result.v_kms[:1]
    bc = ev.bc[:1]
    t_hit = 20 * 60.0
    fr_hit, fv_hit = sbm.propagate(f_r, f_v, bc, t_hit, max_step_s=10.0)
    sat_v_hit = -fv_hit[0] + np.cross(fr_hit[0] / np.linalg.norm(fr_hit[0]), fv_hit[0]) * 0.3
    sat_v_hit *= np.linalg.norm(fv_hit[0]) / np.linalg.norm(sat_v_hit)
    sr0, sv0 = sbm.propagate(fr_hit, sat_v_hit[None, :], None, -t_hit, max_step_s=10.0)
    sat = SimpleNamespace(norad_id=12345, name="TEST SAT", error_code=0,
                          x=sr0[0, 0], y=sr0[0, 1], z=sr0[0, 2], vx=sv0[0, 0], vy=sv0[0, 1], vz=sv0[0, 2])
    alerts = model.compute_debris_alerts([sat], t0, window_hours=1.0, step_s=30.0, threshold_km=5.0)
    hits = [a for a in alerts if a["fragment_id"] == "t-evt:F0000"]
    assert hits, "fragment placed on the satellite path must be screened"
    a = hits[0]
    assert a["source"] == "debris" and a["sat2"]["object_type"] == "DEB"
    assert a["miss_distance_km"] < 0.5
    assert abs(a["tca_minutes"] - 20.0) < 1.0
    assert a["probability_of_collision"] > 0.0
    assert a["parent_event"]["event_id"] == "t-evt"
    assert a["parent_event"]["fragment_count"] == ev.result.n_total
    clouds = model.get_frontend_debris_clouds([sat])
    assert clouds and clouds[0]["affected_count"] >= 1


def test_debris_alert_carries_fragment_state_reproducing_tca():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    model = DebrisModel()
    r, v = _circular(700.0, 98.0)
    ev = model.simulate_collision(r.tolist(), v.tolist(), 900.0, 900.0, 10.0,
                                  max_fragments_simulated=20, event_id="fs-evt", collision_utc=t0)
    f_r, f_v, bc = ev.result.r_km[:1], ev.result.v_kms[:1], ev.bc[:1]
    fr_hit, fv_hit = sbm.propagate(f_r, f_v, bc, 600.0, max_step_s=10.0)
    sr0, sv0 = sbm.propagate(fr_hit, -fv_hit, None, -600.0, max_step_s=10.0)
    sat = SimpleNamespace(norad_id=7, name="S", error_code=0, x=sr0[0, 0], y=sr0[0, 1], z=sr0[0, 2],
                          vx=sv0[0, 0], vy=sv0[0, 1], vz=sv0[0, 2])
    a = [x for x in model.compute_debris_alerts([sat], t0, window_hours=0.5) if x["fragment_id"] == "fs-evt:F0000"][0]
    fs = a["fragment_state"]
    assert set(fs) >= {"r_km", "v_kms", "epoch_utc"}
    r2, _ = sbm.propagate(np.array([fs["r_km"]]), np.array([fs["v_kms"]]), np.array([fs["ballistic_coeff_m2_kg"]]),
                          a["tca_minutes"] * 60.0 - (datetime.fromisoformat(fs["epoch_utc"]) - t0).total_seconds())
    assert np.linalg.norm(r2[0] - fr_hit[0]) < 1.0
