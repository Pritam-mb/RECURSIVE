from __future__ import annotations

from functools import lru_cache
from datetime import datetime
from typing import Any, Iterable

import numpy as np

from app.core.sgp4_propagator import SatelliteState

from .lstm_predictor import LSTMPredictor
from .anomaly import ResidualAnomalyDetector
from .shadow_mode import ShadowMode


class MLRuntime:
    def __init__(self):
        self.trajectory = LSTMPredictor()
        self.anomaly = ResidualAnomalyDetector(self.trajectory)
        self.shadow = ShadowMode(self.trajectory)

    def record_snapshot(self, states: Iterable[SatelliteState], timestamp: datetime):
        epoch = timestamp.isoformat()
        for state in states:
            self.anomaly.update(
                state.norad_id,
                np.array([float(state.x), float(state.y), float(state.z)], dtype=float),
            )
            self.trajectory.record(
                state.norad_id,
                epoch,
                float(state.x),
                float(state.y),
                float(state.z),
            )

        self.shadow.record_snapshot(states, timestamp)
        self.shadow.settle_actuals(states, timestamp)

    def forecast_satellite(self, norad_id: int, steps_ahead: int = 10) -> list[dict]:
        return self.trajectory.predict(norad_id, steps_ahead=steps_ahead)

    def get_anomalies(self) -> dict[int, dict[str, Any]]:
        return self.anomaly.get_latest()

    def log_risk_samples(self, alerts: list[dict]):
        self.shadow.log_risk_samples(alerts)

    def shadow_retrain(self) -> dict[str, dict[str, float]]:
        return self.shadow.retrain_all()


@lru_cache(maxsize=1)
def get_ml_runtime() -> MLRuntime:
    return MLRuntime()
