"""
WebSocket connection manager.
Handles multi-client connections and 1Hz satellite broadcast.

Enhanced broadcast payload includes covariance_ellipse for satellites
that are in active conjunction alerts, enabling the frontend to render
uncertainty ellipses around threatened satellites.
"""

import asyncio
import json
import logging
import os

from fastapi import WebSocket
from app.core.state_cache import get_latest_snapshot, get_latest_alerts

logger = logging.getLogger(__name__)

BROADCAST_INTERVAL_SECONDS = float(os.getenv("BROADCAST_INTERVAL_SECONDS", "1"))


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"Client connected. Total: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"Client disconnected. Total: {len(self.active_connections)}"
        )

    async def broadcast(self, message: str):
        """Send message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


def _build_covariance_index(alerts: list) -> dict:
    """
    Build a lookup mapping norad_id -> covariance_ellipse from active alerts.

    Each alert has sat1 and sat2 fields (both are threatened by the conjunction).
    The ellipse stored is from the Foster integration for that pair.
    """
    index: dict[str, dict | None] = {}
    for alert in alerts:
        ellipse = alert.get("covariance_ellipse")
        # Both satellites in the conjunction get the ellipse
        sat1_id = str(alert.get("sat1", {}).get("id", ""))
        sat2_id = str(alert.get("sat2", {}).get("id", ""))
        if sat1_id:
            # Use highest-severity ellipse if satellite is in multiple alerts
            if sat1_id not in index or ellipse is not None:
                index[sat1_id] = ellipse
        if sat2_id:
            if sat2_id not in index or ellipse is not None:
                index[sat2_id] = ellipse
    return index


async def satellite_broadcast_loop(
    manager: ConnectionManager,
):
    """
    Background task: broadcast the latest cached snapshot at 1Hz.

    Enriches the satellite list with covariance_ellipse fields from active
    conjunction alerts. Field is None for non-threatened satellites and
    contains ellipse parameters {a, b, angle, affection_rate, predicted_fragments}
    for satellites in active conjunctions.

    Alerts, hotspots and the cascade plan are refreshed on a much slower cadence
    than the position stream, so they are attached to the broadcast only when
    the alert cache timestamp actually changes. This keeps the 1 Hz payload small
    while still pushing a new alert set to the UI as soon as it is computed,
    instead of waiting for the frontend's REST poll.
    """
    last_alerts_stamp: str | None = None

    while True:
        try:
            if manager.active_connections:
                snapshot = get_latest_snapshot()

                try:
                    alerts_snapshot = get_latest_alerts()
                except Exception:
                    logger.debug("Alert cache unavailable for broadcast", exc_info=True)
                    alerts_snapshot = {}

                active_alerts = alerts_snapshot.get("alerts", [])
                covariance_index = _build_covariance_index(active_alerts)
                alerts_stamp = alerts_snapshot.get("timestamp")

                satellites = snapshot.get("satellites", [])
                if covariance_index:
                    satellites = [
                        {
                            **sat,
                            "covariance_ellipse": covariance_index.get(
                                str(sat.get("norad_id", "")), None
                            ),
                        }
                        for sat in satellites
                    ]

                payload_obj = {
                    "type": snapshot.get("type", "update"),
                    "timestamp": snapshot.get("timestamp"),
                    "satellites": satellites,
                    "count": snapshot.get("count", len(satellites)),
                }

                # Fragment clouds from the core EVOLVE model. These radii grow
                # every second as fragments propagate, so they ride along on
                # every position frame. `debris_sources` names which sources
                # this frame actually carries, so the client can clear a source
                # that has emptied without wiping one that is merely absent.
                fragment_clouds = [
                    {**cloud, "debris_source": "fragment"}
                    for cloud in snapshot.get("debris_clouds", []) or []
                ]
                payload_obj["debris_sources"] = ["fragment"]
                if fragment_clouds:
                    payload_obj["debris_clouds"] = fragment_clouds

                # The alert set is refreshed far more slowly than the position
                # stream, so it is attached only when the cache stamp changes.
                # This keeps the 1 Hz payload small while still pushing a new
                # alert set as soon as it is computed, and it lets the frontend
                # tell "alerts cleared" apart from "alerts unchanged".
                if alerts_stamp != last_alerts_stamp:
                    last_alerts_stamp = alerts_stamp
                    forecast_clouds = [
                        {**cloud, "debris_source": "forecast"}
                        for cloud in alerts_snapshot.get("debris_clouds", []) or []
                    ]
                    payload_obj["debris_sources"].append("forecast")
                    # Merge with this frame's fragment clouds. The client keeps
                    # the two sources separate, so a tick without forecast clouds
                    # does not erase them.
                    payload_obj["debris_clouds"] = [
                        *forecast_clouds,
                        *fragment_clouds,
                    ]
                    payload_obj.update(
                        {
                            "alerts": active_alerts,
                            "alert_count": len(active_alerts),
                            "hotspots": alerts_snapshot.get("hotspots", []),
                            "cascade_plan": alerts_snapshot.get("cascade_plan", []),
                            "cascade_depth": alerts_snapshot.get("cascade_depth", 0),
                            "graph": alerts_snapshot.get("graph", {}),
                            "ranker_review": alerts_snapshot.get("ranker_review", {}),
                            "alerts_timestamp": alerts_stamp,
                        }
                    )

                await manager.broadcast(
                    json.dumps(payload_obj, separators=(",", ":"))
                )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Broadcast error: {e}")

        await asyncio.sleep(BROADCAST_INTERVAL_SECONDS)
