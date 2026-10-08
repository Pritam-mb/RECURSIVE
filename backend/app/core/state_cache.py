"""
In-memory snapshot cache for the latest propagated satellite state and alerts.
"""

from typing import Any

_latest_snapshot: dict[str, Any] = {
    "type": "update",
    "timestamp": None,
    "satellites": [],
    "count": 0,
    "states": [],
    "payload": None,
}

_latest_alerts: dict[str, Any] = {
    "timestamp": None,
    "alerts": [],
    "count": 0,
    "hotspots": [],
    "cascade_plan": [],
    "graph": {},
    "cascade_depth": 0,
    "total_delta_v_ms": 0.0,
    "agencies_involved": [],
    "debris_clouds": [],
}

def set_latest_snapshot(snapshot: dict[str, Any]):
    global _latest_snapshot
    _latest_snapshot = snapshot


def update_snapshot(snapshot: dict[str, Any]):
    set_latest_snapshot(snapshot)


def get_latest_snapshot() -> dict[str, Any]:
    return _latest_snapshot


def get_snapshot() -> dict[str, Any]:
    return get_latest_snapshot()


def set_latest_alerts(alerts_snapshot: dict[str, Any]):
    global _latest_alerts
    _latest_alerts = alerts_snapshot


def update_alerts(alerts_snapshot: dict[str, Any]):
    set_latest_alerts(alerts_snapshot)


def get_latest_alerts() -> dict[str, Any]:
    return _latest_alerts


def get_alerts_snapshot() -> dict[str, Any]:
    return get_latest_alerts()
