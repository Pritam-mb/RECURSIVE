"""
rlhf_store.py — Persistent record of operator maneuver decisions.

Every decision posted to ``/api/feedback/maneuver`` is appended to a small
SQLite table so the RLHF panel reports real counts instead of hardcoded zeros.

Semantics:
  - ``APPROVE`` and ``MODIFY`` count as approved (both execute a maneuver).
  - ``REJECT`` counts as rejected.
  - Any other decision string is not recorded.
  - A "round" is every ``ROUND_SIZE`` (10) recorded decisions. Only completed
    rounds produce an entry in ``round_approval_rates`` (most recent 20).

The decisions are stored for a future preference-learning loop; nothing in the
backend retrains on them yet.

The database lives at ``RLHF_DB_PATH`` when set, otherwise
``app/ml/artifacts/rlhf_feedback.db`` (gitignored). The path is resolved on
every call so tests can point it at a temp file.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import artifact_path, ensure_artifact_dir

DEFAULT_DB = "rlhf_feedback.db"
ROUND_SIZE = 10
MAX_ROUNDS_REPORTED = 20
APPROVE_DECISIONS = frozenset({"APPROVE", "MODIFY"})
REJECT_DECISIONS = frozenset({"REJECT"})

_lock = threading.Lock()


def _db_path() -> Path:
    override = os.getenv("RLHF_DB_PATH")
    if override:
        path = Path(override)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    ensure_artifact_dir()
    return artifact_path(DEFAULT_DB)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS operator_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decided_at TEXT NOT NULL,
            decision TEXT NOT NULL,
            approved INTEGER NOT NULL,
            alert_id TEXT,
            sat1_id INTEGER,
            sat2_id INTEGER,
            delta_v_ms REAL
        )
        """
    )
    return conn


def record_decision(
    decision: str,
    *,
    alert_id: str | None = None,
    sat1_id: int | None = None,
    sat2_id: int | None = None,
    delta_v_ms: float | None = None,
) -> bool:
    """Persist one operator decision. Returns False if the decision is not counted."""
    normalized = (decision or "").strip().upper()
    if normalized in APPROVE_DECISIONS:
        approved = 1
    elif normalized in REJECT_DECISIONS:
        approved = 0
    else:
        return False

    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO operator_decisions
                    (decided_at, decision, approved, alert_id, sat1_id, sat2_id, delta_v_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    normalized,
                    approved,
                    alert_id,
                    sat1_id,
                    sat2_id,
                    delta_v_ms,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    return True


def get_stats() -> dict[str, Any]:
    """Aggregate counts in the shape used by ``/api/ml/status`` -> ``rlhf``."""
    with _lock:
        conn = _connect()
        try:
            rows = [int(r[0]) for r in conn.execute(
                "SELECT approved FROM operator_decisions ORDER BY id ASC"
            )]
        finally:
            conn.close()

    decisions = len(rows)
    approved = sum(rows)
    rejected = decisions - approved
    rounds = decisions // ROUND_SIZE
    round_rates = [
        round(sum(rows[i * ROUND_SIZE:(i + 1) * ROUND_SIZE]) / ROUND_SIZE, 4)
        for i in range(rounds)
    ][-MAX_ROUNDS_REPORTED:]

    return {
        "decisions": decisions,
        "approved": approved,
        "rejected": rejected,
        "approval_rate": round(approved / decisions, 4) if decisions else None,
        "rounds": rounds,
        "round_approval_rates": round_rates,
    }
