"""
In-memory snapshot cache for the latest propagated satellite state and covariance.
"""

from typing import Any

_latest_snapshot: dict[str, Any] = {
    "type": "update",
    "timestamp": None,
    "satellites": [],
    "count": 0,
    "states": [],
    "payload": None,
    "covariances": {},  # norad_id -> covariance_6x6 matrix
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

_kalman_states: dict[int, dict[str, Any]] = {}  # norad_id -> Kalman state dict


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


def set_kalman_state(norad_id: int, kalman_state_dict: dict[str, Any]):
    """Store Kalman state (position, velocity, covariance) for a satellite."""
    global _kalman_states
    _kalman_states[norad_id] = kalman_state_dict


def get_kalman_state(norad_id: int) -> dict[str, Any] | None:
    """Retrieve Kalman state for a satellite."""
    return _kalman_states.get(norad_id)


def get_all_kalman_states() -> dict[int, dict[str, Any]]:
    """Get all stored Kalman states."""
    return dict(_kalman_states)