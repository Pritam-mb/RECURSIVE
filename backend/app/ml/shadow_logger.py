from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import ensure_artifact_dir

logger = logging.getLogger(__name__)

DEFAULT_DB = "shadow_mode.db"


class ShadowLogger:
    def __init__(self, db_name: str = DEFAULT_DB):
        ensure_artifact_dir()
        override = os.getenv("SHADOW_DB_PATH")
        if override:
            self.db_path = Path(override)
        else:
            self.db_path = Path(tempfile.gettempdir()) / db_name
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trajectory_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    norad_id INTEGER,
                    predicted_at TEXT,
                    target_time TEXT,
                    sequence_json TEXT,
                    predicted_json TEXT,
                    actual_json TEXT,
                    residual_km REAL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    logged_at TEXT,
                    features_json TEXT,
                    target REAL
                )
                """
            )
            conn.commit()
        logger.info("Shadow-mode DB path: %s", self.db_path)

    def log_prediction(
        self,
        norad_id: int,
        predicted_at: datetime,
        target_time: datetime,
        sequence: list[list[float]],
        predicted: list[float],
    ):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trajectory_logs (norad_id, predicted_at, target_time, sequence_json, predicted_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(norad_id),
                    predicted_at.isoformat(),
                    target_time.isoformat(),
                    json.dumps(sequence),
                    json.dumps(predicted),
                ),
            )
            conn.commit()

    def settle_actuals(
        self,
        norad_id: int,
        now: datetime,
        actual: list[float],
        limit: int = 3,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, predicted_json
                FROM trajectory_logs
                WHERE norad_id = ?
                  AND target_time <= ?
                  AND actual_json IS NULL
                ORDER BY target_time DESC
                LIMIT ?
                """,
                (int(norad_id), now.isoformat(), int(limit)),
            )
            rows = cursor.fetchall()
            updated = 0
            for log_id, predicted_json in rows:
                predicted = json.loads(predicted_json)
                residual = float(
                    sum((float(p) - float(a)) ** 2 for p, a in zip(predicted, actual)) ** 0.5
                )
                cursor.execute(
                    """
                    UPDATE trajectory_logs
                    SET actual_json = ?, residual_km = ?
                    WHERE id = ?
                    """,
                    (json.dumps(actual), residual, log_id),
                )
                updated += 1
            conn.commit()
            return updated

    def get_trajectory_batch(self, limit: int = 1000) -> list[tuple[int, list[list[float]], list[float]]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT norad_id, sequence_json, actual_json
                FROM trajectory_logs
                WHERE actual_json IS NOT NULL
                ORDER BY predicted_at DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            rows = cursor.fetchall()

        batch = []
        for norad_id, sequence_json, actual_json in rows:
            sequence = json.loads(sequence_json)
            actual = json.loads(actual_json)
            batch.append((int(norad_id), sequence, actual))
        return batch

    def log_risk_sample(self, features: list[float], target: float, logged_at: datetime | None = None):
        timestamp = (logged_at or datetime.utcnow()).isoformat()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO risk_logs (logged_at, features_json, target)
                VALUES (?, ?, ?)
                """,
                (timestamp, json.dumps(features), float(target)),
            )
            conn.commit()

    def get_risk_batch(self, limit: int = 2000) -> list[tuple[list[float], float]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT features_json, target
                FROM risk_logs
                ORDER BY logged_at DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            rows = cursor.fetchall()

        batch = []
        for features_json, target in rows:
            batch.append((json.loads(features_json), float(target)))
        return batch


_shadow_logger: ShadowLogger | None = None


def get_shadow_logger() -> ShadowLogger:
    global _shadow_logger
    if _shadow_logger is None:
        _shadow_logger = ShadowLogger()
    return _shadow_logger
