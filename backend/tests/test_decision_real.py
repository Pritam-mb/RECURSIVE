"""Decision score + Pc cross-validation (agent DS)."""

import math

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core import decision as D
from app.core.pc_methods import (alfano_max_pc, bplane_inputs, chan_pc, cov_from_ellipse,
                                 monte_carlo_pc, pc_checks, pc_checks_from_inputs)
from app.core.screening import compute_pc, foster_pc


def _alert(pc, *, ml=None, downstream=0, dv=None, safe=True, spread=0.0, aid="1-2"):
    a = {"id": aid, "probability_of_collision": pc, "p_collision": pc,
         "downstream_ids": list(range(downstream)), "short_encounter_valid": True,
         "pc_checks": {"spread_decades": spread, "monte_carlo": None}}
    if ml is not None:
        a["ml"] = {"pc_surrogate": ml}
    if dv is not None:
        a["recommended_maneuver"] = {"delta_v_ms": dv, "options": [{"delta_v_ms": dv, "cascade_safe": safe}],
                                     "chosen_index": 0}
    return a


# ── decision score ──────────────────────────────────────────────────────────

def test_weights_sum_to_one_and_physics_dominates():
    assert math.isclose(sum(D.WEIGHTS.values()), 1.0)
    assert D.WEIGHTS["physics"] > sum(v for k, v in D.WEIGHTS.items() if k != "physics")


@pytest.mark.parametrize("ml,down,dv", [(None, 0, None), (1e-6, 3, 0.2), (1e-3, 10, 5.0)])
def test_score_monotonic_in_pc(ml, down, dv):
    pcs = np.logspace(-12, 0, 121)
    scores = [D.decide(_alert(p, ml=ml, downstream=down, dv=dv))["score"] for p in pcs]
    assert all(b >= a for a, b in zip(scores, scores[1:]))
    assert scores[-1] > scores[0]


def test_components_points_add_up_and_bounded():
    d = D.decide(_alert(3e-5, ml=2e-5, downstream=6, dv=0.4))
    assert math.isclose(sum(c["weight"] for c in d["components"]), 1.0)
    assert math.isclose(sum(c["points"] for c in d["components"]), d["score"], abs_tol=0.02)
    assert 0 <= d["score"] <= 100
    assert {c["source"] for c in d["components"]} == {"physics", "ml", "cascade", "manoeuvre"}
    # no ML -> its weight moves to physics, still sums to 1
    d2 = D.decide(_alert(3e-5))
    assert math.isclose(sum(c["weight"] for c in d2["components"]), 1.0)
    assert d2["components"][0]["weight"] == pytest.approx(0.75)


@pytest.mark.parametrize("pc,action", [(2e-4, "MANOEUVRE"), (1e-4, "MANOEUVRE"), (5e-5, "PREPARE"),
                                       (1e-5, "PREPARE"), (9.9e-6, "MONITOR"), (1e-7, "MONITOR"),
                                       (5e-8, "NONE"), (0.0, "NONE")])
def test_action_thresholds(pc, action):
    assert D.decide(_alert(pc))["action"] == action
    assert D.decide(_alert(pc))["thresholds"]["manoeuvre_pc"] == 1e-4


def test_escalation_requires_ml_and_cascade_agreement():
    assert D.decide(_alert(3e-5, ml=5e-4, downstream=8))["action"] == "MANOEUVRE"
    assert D.decide(_alert(3e-5, ml=5e-4, downstream=1))["action"] == "PREPARE"      # cascade disagrees
    assert D.decide(_alert(3e-5, ml=1e-6, downstream=8))["action"] == "PREPARE"      # ML disagrees
    assert D.decide(_alert(3e-6, ml=1e-2, downstream=50))["action"] == "MONITOR"     # never above PREPARE band
    assert D.decide(_alert(2e-4, ml=1e-12, downstream=0))["action"] == "MANOEUVRE"   # models cannot lower physics


def test_confidence_from_agreement():
    assert D.decide(_alert(1e-5, ml=2e-5, spread=0.05))["model_agreement"]["confidence"] == "high"
    assert D.decide(_alert(1e-5, ml=1e-7, spread=0.05))["model_agreement"]["confidence"] == "medium"
    assert D.decide(_alert(1e-5, ml=1e-9, spread=0.05))["model_agreement"]["confidence"] == "low"
    assert D.decide(_alert(1e-5, ml=2e-5, spread=2.0))["model_agreement"]["confidence"] == "low"
    r = D.decide(_alert(1e-5, ml=2e-5))["rationale"]
    assert "1.00e-05" in r and "PREPARE" in r


# ── Pc methods ──────────────────────────────────────────────────────────────

def _typical_geometry():
    # realistic LEO encounter from the production screening code
    r1 = np.array([7000.0, 0.0, 0.0]); v1 = np.array([0.0, 7.5, 0.0])
    r2 = r1 + np.array([0.05, 0.2, 0.1]); v2 = np.array([0.0, 0.5, 7.4])
    return compute_pc(r1, v1, r2, v2, age1_days=1.0, age2_days=2.0, hbr_km=0.02)


def test_chan_matches_foster_typical_geometry():
    pcd = _typical_geometry()
    b, C = np.array([pcd["b_t_km"], pcd["b_n_km"]]), np.array(pcd["cov_2d_km2"])
    pf, pch = foster_pc(b, C, 0.02), chan_pc(b, C, 0.02)
    assert pf > 1e-8
    assert abs(math.log10(pch) - math.log10(pf)) < 0.1


def test_chan_exact_for_isotropic():
    from scipy.special import chndtr
    s, b, h = 0.3, np.array([0.4, 0.2]), 0.03
    exact = chndtr((h / s) ** 2, 2, float(b @ b) / s ** 2)
    assert chan_pc(b, np.eye(2) * s * s, h) == pytest.approx(exact, rel=1e-9)


@pytest.mark.parametrize("b,sig,rot,h", [((0.15, 0.0), (0.2, 0.2), 0, 0.02),
                                         ((0.2, 0.05), (0.5, 0.1), 0, 0.015),
                                         ((0.3, -0.2), (1.0, 0.3), 30, 0.05)])
def test_monte_carlo_matches_foster(b, sig, rot, h):
    th = math.radians(rot)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    C = R @ np.diag(np.square(sig)) @ R.T
    pf = foster_pc(b, C, h)
    assert pf >= 1e-3
    mc = monte_carlo_pc(b, C, h, seed=7, target_hits=10**9)
    assert abs(mc["pc"] - pf) <= 4 * mc["stderr"]


def test_monte_carlo_returns_none_when_unresolvable():
    mc = monte_carlo_pc([5.0, 0.0], np.eye(2) * 0.25, 0.01, seed=1)
    assert mc["pc"] is None and mc["samples"] == 200_000


@pytest.mark.parametrize("b", [(0.15, 0.0), (1.0, 0.5), (3.0, 1.0), (0.01, 0.0)])
def test_alfano_upper_bound(b):
    C = np.array([[0.25, 0.05], [0.05, 0.04]])
    assert alfano_max_pc(b, C, 0.02) >= foster_pc(b, C, 0.02)


def test_pc_checks_from_alert_reconstructs_foster():
    pcd = _typical_geometry()
    alert = {"id": "x", "b_t_km": round(pcd["b_t_km"], 5), "b_n_km": round(pcd["b_n_km"], 5), "hbr_km": 0.02,
             "covariance_ellipse": pcd["covariance_ellipse"], "probability_of_collision": pcd["pc"]}
    chk = pc_checks(alert, monte_carlo=False)
    assert abs(math.log10(chk["foster"]) - math.log10(pcd["pc"])) < 0.01
    assert chk["alfano_max"] >= chk["foster"] and chk["consistent"]
    assert set(["foster", "chan", "alfano_max", "monte_carlo", "mc_samples", "spread_decades", "consistent"]) <= set(chk)
    assert bplane_inputs({"id": "y"}) is None


def test_cov_from_ellipse_roundtrip():
    C = cov_from_ellipse(3000.0, 600.0, 0.3, sigma_level=3)
    w = np.linalg.eigvalsh(C)
    assert np.allclose(np.sqrt(w[::-1]), [1.0, 0.2])


def test_annotate_alerts_attaches_blocks():
    pcd = _typical_geometry()
    base = {"b_t_km": pcd["b_t_km"], "b_n_km": pcd["b_n_km"], "hbr_km": 0.02,
            "covariance_ellipse": pcd["covariance_ellipse"], "probability_of_collision": pcd["pc"],
            "p_collision": pcd["pc"]}
    alerts = [dict(base, id=f"a{i}") for i in range(3)] + [{"id": "nogeom", "p_collision": 1e-6}]
    stats = D.annotate_alerts(alerts, mc_top_n=1)
    assert stats["pc_checks"] == 3
    assert alerts[0]["pc_checks"]["mc_samples"] is not None
    assert alerts[1]["pc_checks"]["mc_samples"] is None
    assert all(a["decision"]["action"] in ("MANOEUVRE", "PREPARE", "MONITOR", "NONE") for a in alerts)


# ── validation endpoint ─────────────────────────────────────────────────────

def test_validation_endpoint_all_pass():
    from app.routers.physics import router
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    r = client.get("/api/physics/validation")
    assert r.status_code == 200
    body = r.json()
    assert body["errors"] == []
    names = " ".join(c["name"] for c in body["checks"])
    for key in ("Chan", "Monte Carlo", "Energy", "SGP4 vs J2", "CW", "Hohmann", "SBM", "Rocket"):
        assert key in names, key
    failed = [c["name"] for c in body["checks"] if not c["pass"]]
    assert not failed, failed
    for c in body["checks"]:
        assert {"standard_formula", "our_value", "reference_value", "abs_error", "rel_error", "pass",
                "tolerance", "source"} <= set(c)
    assert client.get("/api/physics/validation").json()["cached"] is True
    e = client.get("/api/physics/engines")
    assert e.status_code in (200, 503)
