"""
The risk scorer must not load anything at import/construction time and must not
depend on ENABLE_EXTENDED_PIPELINE: it is a thin adapter over the XGBoost Pc
surrogate in app.ml.risk_api (no torch).
"""

from __future__ import annotations

import app.ml.risk_api as risk_api
import app.ml.xgboost_scorer as scorer_module
from app.ml.xgboost_scorer import XGBoostScorer

EVENT = {
    "miss_distance_km": 2.5,
    "relative_speed_kmh": 27000.0,
    "sat1_altitude_km": 550.0,
    "sat2_altitude_km": 420.0,
    "tca_minutes": 45.0,
}


def test_construction_loads_nothing(monkeypatch):
    monkeypatch.setattr(risk_api, "_get_booster", lambda: (_ for _ in ()).throw(AssertionError("loaded")))
    XGBoostScorer()


def test_score_is_a_probability_independent_of_flag(monkeypatch):
    monkeypatch.setenv("ENABLE_EXTENDED_PIPELINE", "0")
    off = XGBoostScorer().score(EVENT, record=False)
    monkeypatch.setenv("ENABLE_EXTENDED_PIPELINE", "1")
    on = XGBoostScorer().score(EVENT, record=False)
    assert 0.0 <= off <= 1.0
    assert off == on


def test_scorer_reports_surrogate_model():
    assert scorer_module.using_trained_model() is True
    assert scorer_module.active_risk_model_path() == risk_api.MODEL_NAME

