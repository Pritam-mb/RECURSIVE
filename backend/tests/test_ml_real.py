"""
Honest-ML tests: the Pc surrogate is a real model trained on physics labels,
tracks Foster Pc on held-out encounters, is fast, and never touches physics Pc.
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from app.ml import risk_api
from app.ml.model_metrics import get_model_metrics
from app.ml.train_risk_surrogate import (
    FEATURES,
    feature_matrix,
    foster_pc_quadrature,
    generate_encounters,
)

BACKEND = Path(__file__).resolve().parents[1]


def test_model_and_card_load():
    assert risk_api.model_available()
    card = risk_api.model_card()
    assert card["training_samples"] >= 10_000
    assert card["features"] == FEATURES
    assert "Foster" in card["label_source"]
    # Labels are physics, not a function the model sees: covariance is hidden.
    assert "combined covariance" in card["features_not_seen_by_model"]
    # Card's held-out numbers beat the miss-distance-only baseline.
    model_mae = card["heldout"]["regression"]["mae_log10_pc"]
    base_mae = card["baselines"]["miss_distance_only_xgboost"]["regression"]["mae_log10_pc"]
    assert model_mae < base_mae
    assert card["heldout"]["classification_at_1e-4"]["positives"] > 100


def test_quadrature_matches_closed_form_isotropic_centered():
    # Centered isotropic Gaussian: Pc = 1 - exp(-R^2 / (2 sigma^2)).
    sigma, R = 0.2, 0.05
    pc = foster_pc_quadrature(np.zeros((1, 2)), np.array([np.eye(2) * sigma ** 2]), np.array([R]))[0]
    assert pc == pytest.approx(1.0 - math.exp(-R * R / (2 * sigma * sigma)), rel=1e-6)


def test_surrogate_tracks_foster_pc_on_fresh_encounters():
    # Fresh seed never used in training: label = numerically integrated Foster Pc.
    data = generate_encounters(3000, seed=424242)
    x = feature_matrix(data)
    alerts = []
    for row in x:
        rec = dict(zip(FEATURES, row.tolist()))
        alerts.append({
            "miss_distance_km": rec["miss_distance_km"],
            "radial_miss_km": rec["radial_miss_km"],
            "relative_speed_kms": rec["relative_speed_kms"],
            "sat1": {"tle_age_hours": rec["tle_age_max_h"]},
            "sat2": {"tle_age_hours": rec["tle_age_min_h"]},
            "hbr_km": rec["hbr_km"],
            "altitude_km": rec["altitude_km"],
        })
    risk_api.score_alerts(alerts, record=False)
    pred = np.array([a["ml"]["log10_pc_surrogate"] for a in alerts])
    truth = np.log10(np.clip(data["pc"], 1e-12, 1.0))
    mae = float(np.mean(np.abs(pred - truth)))
    assert mae < 0.4, mae
    hi_t, hi_p = truth >= -4, pred >= -4
    precision = (hi_t & hi_p).sum() / max(hi_p.sum(), 1)
    recall = (hi_t & hi_p).sum() / max(hi_t.sum(), 1)
    assert precision > 0.9 and recall > 0.9


def test_score_alert_contract_and_does_not_touch_physics_pc():
    alert = {"miss_distance_km": 0.15, "b_n_km": 0.02, "relative_speed_kms": 11.0,
             "hbr_km": 0.02, "p_collision": 2e-4, "probability_of_collision": 2e-4,
             "sat1": {"id": 1, "tle_age_hours": 10}, "sat2": {"id": 2, "tle_age_hours": 30}}
    out = risk_api.score_alert(alert, record=False)
    assert set(["pc_surrogate", "risk_class", "model", "agreement"]) <= set(out)
    assert 0.0 <= out["pc_surrogate"] <= 1.0
    assert out["risk_class"] in {"high", "elevated", "low"}
    assert out["agreement"] == pytest.approx(abs(math.log10(out["pc_surrogate"]) - math.log10(2e-4)), abs=2e-3)
    assert alert["p_collision"] == 2e-4 and "ml" not in alert


def test_monotone_in_miss_distance():
    base = {"relative_speed_kms": 10.0, "hbr_km": 0.02, "sat1": {"tle_age_hours": 12}, "sat2": {"tle_age_hours": 12}}
    near = risk_api.score_alert({**base, "miss_distance_km": 0.05}, record=False)["pc_surrogate"]
    far = risk_api.score_alert({**base, "miss_distance_km": 15.0}, record=False)["pc_surrogate"]
    assert near > far * 1e3


def test_score_alert_is_fast():
    alert = {"miss_distance_km": 0.4, "relative_speed_kms": 9.0, "p_collision": 1e-5}
    risk_api.score_alert(alert, record=False)
    n = 500
    t = time.perf_counter()
    for _ in range(n):
        risk_api.score_alert(alert, record=False)
    per_alert_ms = (time.perf_counter() - t) / n * 1e3
    assert per_alert_ms < 1.0, per_alert_ms


def test_metrics_report_card_not_training_formula():
    m = get_model_metrics()
    risk = m["risk_model"]
    assert risk["training_samples"] == risk_api.model_card()["training_samples"]
    assert risk["feature_importance"] and abs(sum(risk["feature_importance"].values()) - 1.0) < 0.01
    assert m["classification_models"][0]["metrics"]["split"] == "held-out"
    traj = m["trajectory_model"]
    assert "mae_km" in traj and traj["status"] in {"production", "experimental"}
    if traj["mae_km"]["model"] >= traj["mae_km"]["linear_extrapolation"]:
        assert traj["status"] == "experimental"
    assert m["operator_feedback"]["trains_any_model"] is False
    assert "cross-check" in m["graph_models"]["role"]


def test_risk_path_does_not_import_torch():
    code = ("import sys; from app.ml import risk_api, model_metrics, xgboost_scorer;"
            "risk_api.score_alert({'miss_distance_km': 1.0}); model_metrics.get_model_metrics();"
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, capture_output=True, text=True,
                         env={**__import__('os').environ, "ORBIT_SENTINEL_SKIP_DOTENV": "1"}, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"
