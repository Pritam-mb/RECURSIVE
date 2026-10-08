"""Explainability of the XGBoost Pc surrogate: TreeSHAP additivity, correlation,
PCA and the /api/analytics/model endpoint."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ml import explain, risk_api
from app.ml.train_risk_surrogate import FEATURES

pytestmark = pytest.mark.skipif(not risk_api.model_available(), reason="surrogate artifact missing")


def _alerts(n=40, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a = {
            "id": f"a{i}",
            "miss_distance_km": float(10 ** rng.uniform(-2.5, 1.2)),
            "b_n_km": float(rng.uniform(-0.3, 0.3)),
            "relative_speed_kms": float(rng.uniform(0.5, 14.5)),
            "sat1": {"tle_age_hours": float(10 ** rng.uniform(0, 3.5))},
            "sat2": {"tle_age_hours": float(10 ** rng.uniform(0, 3.5))},
            "hbr_km": float(rng.uniform(0.004, 0.04)),
            "p_collision": float(10 ** rng.uniform(-9, -3)),
        }
        if i % 3 == 0:
            a["altitude_km"] = float(rng.uniform(400, 1200))  # others: altitude missing (NaN branch)
        out.append(a)
    return out


@pytest.mark.parametrize("budget", [0, 8, 1000])
def test_contributions_plus_bias_equal_prediction(budget):
    alerts = risk_api.score_alerts(_alerts(), record=False, exact_budget=budget)
    booster = risk_api._get_booster()
    for a in alerts:
        ml = a["ml"]
        row, _ = risk_api.extract_features(a)
        pred = float(booster.inplace_predict(row)[0])
        total = ml["base_log10"] + sum(c["contribution_log10"] for c in ml["contributions"]) \
            + ml["other_contributions_log10"]
        # Rounded to 4 dp per term (<= 8 terms) -> tolerance 1e-3.
        assert abs(total - pred) < 1e-3, (total, pred, ml["contribution_method"])
        assert len(ml["contributions"]) == 6
        assert ml["main_factor"] == ml["contributions"][0]["feature"]
        assert ml["main_factor"] in FEATURES
        mags = [abs(c["contribution_log10"]) for c in ml["contributions"]]
        assert mags == sorted(mags, reverse=True)
    methods = {a["ml"]["contribution_method"] for a in alerts}
    if budget == 0:
        assert methods == {"saabas_path"}
    elif budget >= len(alerts):
        assert methods == {"treeshap_exact"}
    else:
        assert methods == {"treeshap_exact", "saabas_path"}


def test_exact_treeshap_full_precision_additivity():
    booster = risk_api._get_booster()
    x = np.asarray([risk_api.extract_features(a)[0][0] for a in _alerts(25, seed=9)], dtype=np.float32)
    contrib = explain.tree_contributions(booster, x, FEATURES)
    pred = booster.inplace_predict(x)
    assert contrib.shape == (25, len(FEATURES) + 1)
    np.testing.assert_allclose(contrib.sum(axis=1), pred, atol=1e-3)


def test_missing_feature_value_reported_as_null():
    a = _alerts(1)[0]
    a.pop("altitude_km", None)
    risk_api.score_alerts([a], record=False, exact_budget=1)
    alt = [c for c in a["ml"]["contributions"] if c["feature"] == "altitude_km"]
    assert not alt or alt[0]["value"] is None


def test_correlation_matrix_symmetric_unit_diagonal_pairwise_nan():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 4))
    x[:, 1] = x[:, 0] * 2 + rng.normal(scale=0.1, size=500)
    x[rng.random(500) < 0.3, 2] = np.nan
    res = explain.correlation_matrices(x, ["a", "b", "c", "d"])
    for key in ("pearson", "spearman"):
        m = np.array(res[key], dtype=float)
        np.testing.assert_allclose(m, m.T)
        np.testing.assert_allclose(np.diag(m), 1.0)
        assert np.all(np.abs(m) <= 1.0 + 1e-9)
        assert m[0, 1] > 0.99
    assert res["pairwise_counts"][2][2] < 500 and res["pairwise_counts"][0][0] == 500


def test_pca_explained_variance_sums_to_one_and_loadings_orthonormal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 5)) * [1, 10, 100, 0.1, 5]
    x[:, 4] = x[:, 3] * 50 + rng.normal(scale=0.5, size=800)
    x[rng.random(800) < 0.2, 0] = np.nan
    pca = explain.StandardPCA().fit(x)
    assert abs(pca.explained_variance_ratio_.sum() - 1.0) < 1e-9
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(5), atol=1e-9)
    summary = explain.pca_summary(pca, list("abcde"))
    assert summary["cumulative"][-1] == pytest.approx(1.0, abs=1e-4)
    assert 1 <= summary["n_components_95"] <= 5
    assert np.isfinite(pca.transform(x)).all()  # NaN imputed for PCA


def test_analytics_endpoint_schema():
    from app.routers.analytics import router

    app = FastAPI()
    app.include_router(router)
    r = TestClient(app).get("/api/analytics/model")
    assert r.status_code == 200
    body = r.json()
    for key in ("features", "feature_definitions", "correlation", "pca", "pca_vs_raw",
                "shap_global", "main_factor", "sample_size", "card"):
        assert key in body, key
    p = len(body["features"])
    assert set(body["features"]) <= set(body["feature_definitions"])
    corr = body["correlation"]
    assert len(corr["pearson"]) == p + 1 and len(corr["spearman"]) == p + 1  # features + target
    assert set(corr["with_target"]["pearson"]) == set(body["features"])
    pca = body["pca"]
    assert len(pca["loadings"]) == p and len(pca["loadings"][0]) == p
    assert abs(sum(pca["explained_variance_ratio"]) - 1.0) < 1e-3
    assert pca["standardized"] is True and 1 <= pca["n_components_95"] <= p
    pvr = body["pca_vs_raw"]
    for k in ("mae_log10", "f1_1e4"):
        assert k in pvr["raw_xgb"] and k in pvr["pca_xgb"]
    assert pvr["pca_xgb"]["n_components"] == pca["n_components_95"]
    assert isinstance(pvr["verdict"], str) and pvr["verdict"]
    assert body["main_factor"] == body["shap_global"][0]["feature"]
    assert body["card"].get("model")
