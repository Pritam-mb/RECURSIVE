from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class ResidualAnomalyDetector:
    """Detects trajectory anomalies using residual z-scores."""

    def __init__(self, predictor, window: int = 50, z_threshold: float = 3.0):
        self.predictor = predictor
        self.window = window
        self.z_threshold = z_threshold
        self._residuals: dict[int, deque] = {}
        self._latest: dict[int, dict[str, Any]] = {}

    def _get_buffer(self, norad_id: int) -> deque:
        if norad_id not in self._residuals:
            self._residuals[norad_id] = deque(maxlen=self.window)
        return self._residuals[norad_id]

    def update(self, norad_id: int, actual_pos: np.ndarray) -> dict[str, Any] | None:
        predicted = self.predictor.predict_next_from_buffer(norad_id)
        if predicted is None:
            return None

        residual = float(np.linalg.norm(actual_pos - predicted))
        history = self._get_buffer(norad_id)
        history.append(residual)

        mean = float(np.mean(history)) if len(history) > 0 else residual
        std = float(np.std(history)) if len(history) > 1 else 0.0
        z_score = 0.0 if std < 1e-6 else float((residual - mean) / std)
        is_anomaly = z_score >= self.z_threshold

        result = {
            "norad_id": norad_id,
            "residual_km": round(residual, 3),
            "z_score": round(z_score, 3),
            "is_anomaly": bool(is_anomaly),
            "mean_km": round(mean, 3),
            "std_km": round(std, 3),
            "samples": len(history),
        }
        self._latest[norad_id] = result
        return result

    def get_latest(self) -> dict[int, dict[str, Any]]:
        return dict(self._latest)
