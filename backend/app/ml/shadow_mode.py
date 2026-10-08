from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np

from app.core.sgp4_propagator import SatelliteState
from .shadow_logger import get_shadow_logger

logger = logging.getLogger(__name__)


class ShadowMode:
    def __init__(self, predictor):
        self.enabled = os.getenv("SHADOW_MODE_ENABLED", "1") == "1"
        self.horizon_minutes = float(os.getenv("SHADOW_HORIZON_MINUTES", "5"))
        self.max_logs = int(os.getenv("SHADOW_MAX_LOGS", "200"))
        self.min_lstm_batch = int(os.getenv("SHADOW_MIN_LSTM_BATCH", "200"))
        self.min_risk_batch = int(os.getenv("SHADOW_MIN_RISK_BATCH", "300"))
        self.predictor = predictor
        self.logger = get_shadow_logger()
        self.last_retrain: str | None = None
        self.last_result: dict | None = None
        self._retrain_lock = threading.Lock()

    def record_snapshot(self, states: Iterable[SatelliteState], timestamp: datetime):
        if not self.enabled:
            return

        log_count = 0
        for state in states:
            if log_count >= self.max_logs:
                break
            sequence = self.predictor.get_sequence(state.norad_id)
            if sequence is None:
                continue

            predicted = self.predictor.predict_next_from_sequence(sequence)
            if predicted is None:
                continue

            target_time = timestamp + timedelta(minutes=self.horizon_minutes)
            self.logger.log_prediction(
                norad_id=state.norad_id,
                predicted_at=timestamp,
                target_time=target_time,
                sequence=sequence.tolist(),
                predicted=[float(x) for x in predicted],
            )
            log_count += 1

    def settle_actuals(self, states: Iterable[SatelliteState], timestamp: datetime):
        if not self.enabled:
            return

        for state in states:
            actual = [float(state.x), float(state.y), float(state.z)]
            self.logger.settle_actuals(state.norad_id, timestamp, actual)

    def log_risk_samples(self, alerts: list[dict]):
        """No-op. Risk samples used to be logged with the alert's own p_collision
        as the label, so "retraining" on them taught the model its own output.
        There is no independent ground truth for collision risk in live data, so
        nothing is logged; the Pc surrogate is trained offline on Foster labels
        (app.ml.train_risk_surrogate)."""
        return None

    def retrain_lstm(self) -> dict[str, float]:
        batch = self.logger.get_trajectory_batch(limit=max(self.min_lstm_batch, 200))
        if len(batch) < self.min_lstm_batch:
            return {"status": "skipped", "reason": "insufficient_data", "count": float(len(batch))}

        sequences = np.asarray([item[1] for item in batch], dtype=float)
        targets = np.asarray([item[2] for item in batch], dtype=float)
        loss = self.predictor.fine_tune(sequences, targets, epochs=3, learning_rate=0.001)
        self.predictor.save_model()
        return {"status": "ok", "loss": float(loss), "count": float(len(batch))}

    def retrain_risk(self) -> dict[str, float | str]:
        return {
            "status": "disabled",
            "reason": "no independent ground truth in live data; retrain offline with "
                      "python -m app.ml.train_risk_surrogate",
        }

    def retrain_all(self) -> dict[str, dict[str, float]]:
        if not self.enabled:
            return {"status": {"status": "disabled"}}

        # Serialised so a manual /api/ml/retrain cannot overlap the scheduler.
        with self._retrain_lock:
            lstm_result = self.retrain_lstm()
            risk_result = self.retrain_risk()
            logger.info("Shadow-mode retrain: LSTM=%s Risk=%s", lstm_result, risk_result)
            result = {"lstm": lstm_result, "risk": risk_result}
            self.last_retrain = datetime.now(timezone.utc).isoformat()
            self.last_result = result
            return result
