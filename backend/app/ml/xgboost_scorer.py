"""
xgboost_scorer.py -- compatibility wrapper around the XGBoost Pc surrogate.

History (kept for reviewers): this module used to serve a 24-stump boosted
ensemble trained on labels that were a sigmoid formula of its own inputs, and a
flag-gated "Foster XGBoost" classifier with 2 positives in 4000 rows. Both were
removed. The only risk model now is ``app.ml.risk_api`` (XGBoost regressor of
log10 Foster Pc trained on numerically integrated Pc labels; see
``artifacts/risk_model_card.json``).

``XGBoostScorer.score(event)`` returns the surrogate's Pc *estimate* (a
probability, typically 1e-12 .. 1e-2), not a 0-1 "risk score". Callers must not
present it as the physics Pc of an alert.
"""

from __future__ import annotations

import threading
from typing import Any

from . import risk_api

MODEL_NAME = risk_api.MODEL_NAME

# A surrogate Pc at or above this counts as "high risk" (same threshold as the
# CRITICAL severity rule and the model card's classification metrics).
HIGH_RISK_THRESHOLD = risk_api.THRESHOLD_HIGH


def active_risk_model_path() -> str:
    """Identify the deployed risk model (reported by the metrics endpoint)."""
    return MODEL_NAME if risk_api.model_available() else "unavailable"


def using_trained_model() -> bool:
    """True when the trained Pc surrogate artifact is loaded and usable."""
    return risk_api.model_available()


class _DailyPredictionCounter:
    """Thread-safe count of production scores, reset at UTC day rollover."""

    def __init__(self):
        self._lock = threading.Lock()
        self._day = None
        self._predictions = 0
        self._high_risk = 0

    @staticmethod
    def _today():
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).date()

    def _roll(self) -> None:
        today = self._today()
        if self._day != today:
            self._day = today
            self._predictions = 0
            self._high_risk = 0

    def record(self, score: float) -> None:
        with self._lock:
            self._roll()
            self._predictions += 1
            if score >= HIGH_RISK_THRESHOLD:
                self._high_risk += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            self._roll()
            return {"predictions_today": self._predictions, "high_risk_today": self._high_risk}

    def reset(self) -> None:
        with self._lock:
            self._day = None
            self._predictions = 0
            self._high_risk = 0


prediction_counter = _DailyPredictionCounter()


class XGBoostScorer:
    """Thin adapter: event dict -> surrogate Pc estimate (0..1).

    Construction loads nothing; the booster is loaded lazily by risk_api on the
    first score. No torch, no feature flag.
    """

    def score(self, features: dict[str, Any], *, record: bool = True) -> float:
        value = self._score(features)
        if record:
            prediction_counter.record(value)
        return value

    def _score(self, features: dict[str, Any]) -> float:
        if not isinstance(features, dict):
            raise TypeError("XGBoostScorer.score expects an encounter/alert dict")
        result = risk_api.score_alert(features, record=False)
        if result is None:
            return 0.0
        return float(result["pc_surrogate"])
