"""
The risk scorer must not load or train anything at import/construction time.

The cascade planner builds an XGBoostScorer when its module is imported, which
happens regardless of ENABLE_EXTENDED_PIPELINE. Any eager bootstrap there would
train the heavy Foster model on every startup, defeating the feature flag.
"""

from __future__ import annotations

import pytest

import app.ml.xgboost_scorer as scorer_module
from app.ml.xgboost_scorer import XGBoostScorer

EVENT = {
    "miss_distance_km": 2.5,
    "relative_speed_kmh": 27000.0,
    "sat1_altitude_km": 550.0,
    "sat2_altitude_km": 420.0,
    "tca_minutes": 45.0,
    "sat1_size_m": 2.0,
    "sat2_size_m": 1.5,
    "uncertainty_km": 0.8,
}


class TestScorerLaziness:
    def test_construction_does_not_bootstrap(self):
        scorer = XGBoostScorer()
        assert scorer.model is None

    def test_construction_does_not_load_the_trained_model(self, monkeypatch):
        monkeypatch.setattr(
            scorer_module,
            "_ensure_trained_model",
            lambda: pytest.fail("trained model loaded on construction"),
        )
        XGBoostScorer()

    def test_first_score_bootstraps_exactly_once(self, monkeypatch):
        calls = []
        original = XGBoostScorer._load_or_train_model

        def counting(self, path):
            calls.append(path)
            original(self, path)

        monkeypatch.setattr(XGBoostScorer, "_load_or_train_model", counting)

        scorer = XGBoostScorer()
        scorer.score(EVENT)
        scorer.score(EVENT)
        scorer.score(EVENT)

        assert len(calls) == 1

    def test_disabled_flag_never_loads_the_trained_model(self, monkeypatch):
        monkeypatch.setenv("ENABLE_EXTENDED_PIPELINE", "0")
        monkeypatch.setattr(
            scorer_module,
            "_ensure_trained_model",
            lambda: pytest.fail("trained model loaded while the flag is off"),
        )

        score = XGBoostScorer().score(EVENT)

        assert 0.0 <= score <= 1.0

    def test_enabled_flag_loads_the_trained_model(self, monkeypatch):
        monkeypatch.setenv("ENABLE_EXTENDED_PIPELINE", "1")
        calls = []
        monkeypatch.setattr(
            scorer_module, "_ensure_trained_model", lambda: calls.append(1)
        )

        XGBoostScorer().score(EVENT)

        assert calls == [1]

    def test_flag_defaults_to_disabled(self, monkeypatch):
        monkeypatch.delenv("ENABLE_EXTENDED_PIPELINE", raising=False)
        assert scorer_module._extended_pipeline_enabled() is False
