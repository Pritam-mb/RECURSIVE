"""
WebSocket broadcast contract tests.

The 1 Hz position stream is the frontend's primary update path. Two defects were
found in the audit:

1. Alerts, hotspots and the cascade plan were never pushed over the WebSocket,
   so the dashboard could be up to a full REST poll interval stale even though
   the alert cache had already been recomputed.
2. `App.jsx` already had an `hasOwnProperty('alerts')` branch, i.e. the client
   expected a streamed alert block that the server never sent.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


class _FakeWebSocket:
    def __init__(self):
        self.sent: list[str] = []
        self.fail = False

    async def accept(self):
        return None

    async def send_text(self, message: str):
        if self.fail:
            raise RuntimeError("client gone")
        self.sent.append(message)

    def frames(self) -> list[dict]:
        return [json.loads(m) for m in self.sent]


@pytest.fixture
def broadcast(monkeypatch):
    """Run the broadcast loop for one tick and return the emitted frames."""
    import app.api.ws_handler as ws  # noqa: PLC0415

    def _run(ticks: int = 1) -> list[dict]:
        async def _drive():
            manager = ws.ConnectionManager()
            socket = _FakeWebSocket()
            await manager.connect(socket)

            iterations = {"n": 0}
            real_sleep = ws.asyncio.sleep

            async def _fake_sleep(_seconds):
                iterations["n"] += 1
                if iterations["n"] >= ticks:
                    raise asyncio.CancelledError
                await real_sleep(0)

            monkeypatch.setattr(ws.asyncio, "sleep", _fake_sleep)
            with pytest.raises(asyncio.CancelledError):
                await ws.satellite_broadcast_loop(manager)
            return socket.frames()

        return asyncio.run(_drive())

    return _run


def _seed(snapshot: dict, alerts: dict):
    import app.core.state_cache as cache  # noqa: PLC0415

    cache.set_latest_snapshot(snapshot)
    cache.set_latest_alerts(alerts)


SNAPSHOT = {
    "type": "update",
    "timestamp": "2026-01-01T12:00:00+00:00",
    "satellites": [{"norad_id": 5, "name": "A"}, {"norad_id": 9, "name": "B"}],
    "count": 2,
    "debris_clouds": [],
}

ALERTS = {
    "timestamp": "2026-01-01T12:00:00+00:00",
    "alerts": [
        {
            "id": "5-9",
            "sat1": {"id": 5, "name": "A"},
            "sat2": {"id": 9, "name": "B"},
            "severity": "red",
            "covariance_ellipse": {"a": 100.0, "b": 50.0},
        }
    ],
    "count": 1,
    "hotspots": [{"sat1": {"id": 5}, "sat2": {"id": 9}}],
    "cascade_plan": [{"satellite_id": 5}],
    "cascade_depth": 2,
    "graph": {"node_count": 2, "edge_count": 1},
}


class TestBroadcastCarriesAlerts:
    def test_alert_block_is_pushed_with_the_snapshot(self, broadcast):
        _seed(SNAPSHOT, ALERTS)

        frame = broadcast()[0]

        assert "alerts" in frame, "alerts were not pushed over the WebSocket"
        assert frame["alerts"] == ALERTS["alerts"]
        assert frame["alert_count"] == 1
        assert frame["hotspots"] == ALERTS["hotspots"]
        assert frame["cascade_plan"] == ALERTS["cascade_plan"]
        assert frame["cascade_depth"] == 2
        assert frame["graph"] == ALERTS["graph"]

    def test_cleared_alerts_are_pushed_not_suppressed(self, broadcast):
        """An empty alert set must still reach the client, or the UI keeps stale rows."""
        _seed(SNAPSHOT, {**ALERTS, "alerts": [], "count": 0, "hotspots": []})

        frame = broadcast()[0]

        assert "alerts" in frame
        assert frame["alerts"] == []
        assert frame["alert_count"] == 0

    def test_covariance_ellipse_reaches_threatened_satellites(self, broadcast):
        _seed(SNAPSHOT, ALERTS)

        frame = broadcast()[0]
        by_id = {s["norad_id"]: s for s in frame["satellites"]}

        assert by_id[5]["covariance_ellipse"] == {"a": 100.0, "b": 50.0}
        assert by_id[9]["covariance_ellipse"] == {"a": 100.0, "b": 50.0}

    def test_unchanged_alert_set_is_not_resent(self, broadcast):
        """
        The alert block is attached only when the cache stamp changes, keeping the
        1 Hz frame small. Satellites and positions are still pushed every tick.
        """
        _seed(SNAPSHOT, ALERTS)

        frames = broadcast(ticks=3)

        assert len(frames) == 3, "positions must stream every tick"
        assert "alerts" in frames[0], "first tick must carry the alert block"
        assert "alerts" not in frames[1], "unchanged alert set should be omitted"
        assert "alerts" not in frames[2]
        for frame in frames:
            assert frame["satellites"], "every tick must carry positions"


class TestDebrisCloudSources:
    """
    Two debris sources exist and the client needs both:

      * 'forecast' -- build_debris_alerts(), for imminent conjunctions above the
        CPI threshold. Carries the multi-shell `shells` array the globe renders.
      * 'fragment' -- the core EVOLVE model, for collisions actually simulated.

    `/api/alerts` read only the snapshot's fragment clouds, so the forecast
    clouds were computed and cached on every alert refresh and then discarded.
    """

    FORECAST = {
        "id": "debris-1",
        "shells": [
            {"label": "now", "minutes": 0.0, "radius_km": 1.5},
            {"label": "tca", "minutes": 30.0, "radius_km": 210.0},
        ],
        "minutes_to_tca": 30.0,
    }

    def test_alerts_endpoint_serves_forecast_clouds(self):
        import asyncio  # noqa: PLC0415

        import app.core.state_cache as cache  # noqa: PLC0415
        from app.api.routes import get_alerts  # noqa: PLC0415

        cache.set_latest_snapshot({"debris_clouds": []})
        cache.set_latest_alerts(
            {"timestamp": "t", "alerts": [], "count": 0, "debris_clouds": [self.FORECAST]}
        )

        payload = asyncio.run(get_alerts())

        assert len(payload["debris_clouds"]) == 1
        cloud = payload["debris_clouds"][0]
        assert cloud["debris_source"] == "forecast"
        assert len(cloud["shells"]) == 2, "multi-shell forecast structure was lost"

    def test_alerts_endpoint_merges_both_sources(self):
        import asyncio  # noqa: PLC0415

        import app.core.state_cache as cache  # noqa: PLC0415
        from app.api.routes import get_alerts  # noqa: PLC0415

        cache.set_latest_snapshot({"debris_clouds": [{"id": "frag-1"}]})
        cache.set_latest_alerts(
            {"timestamp": "t", "alerts": [], "count": 0, "debris_clouds": [self.FORECAST]}
        )

        payload = asyncio.run(get_alerts())

        sources = {c["debris_source"] for c in payload["debris_clouds"]}
        assert sources == {"forecast", "fragment"}

    def test_broadcast_names_the_sources_it_carries(self, broadcast):
        import app.core.state_cache as cache  # noqa: PLC0415

        cache.set_latest_snapshot(
            {**SNAPSHOT, "debris_clouds": [{"id": "frag-1", "radius_km": 42.0}]}
        )
        cache.set_latest_alerts({**ALERTS, "debris_clouds": [self.FORECAST]})

        frame = broadcast()[0]

        assert "debris_sources" in frame
        assert set(frame["debris_sources"]) == {"fragment", "forecast"}
        sources = {c["debris_source"] for c in frame["debris_clouds"]}
        assert sources == {"fragment", "forecast"}
        assert len(frame["debris_clouds"]) == 2

    def test_absent_fragment_clouds_are_announced_as_empty(self, broadcast):
        """
        `debris_sources` names fragment even when the snapshot has none, so the
        client knows to clear a previously-seen fragment cloud rather than keep
        it forever.
        """
        _seed(SNAPSHOT, {**ALERTS, "debris_clouds": [self.FORECAST]})

        frame = broadcast()[0]

        assert "fragment" in frame["debris_sources"]
        assert [c for c in frame["debris_clouds"] if c["debris_source"] == "fragment"] == []

    def test_empty_forecast_source_is_still_named(self, broadcast):
        """
        A cleared forecast set must be announced, otherwise the client keeps
        rendering forecast shells that no longer exist.
        """
        _seed(SNAPSHOT, {**ALERTS, "debris_clouds": []})

        frame = broadcast()[0]

        assert "forecast" in frame["debris_sources"]
        forecast = [c for c in frame.get("debris_clouds", []) if c.get("debris_source") == "forecast"]
        assert forecast == []


class TestFrontendConsumesStreamedAlerts:
    def test_app_jsx_reads_the_streamed_alert_block(self):
        source = (
            BACKEND_ROOT.parent / "frontend" / "src" / "App.jsx"
        ).read_text(encoding="utf-8")
        assert "hasOwnProperty.call(data, 'alerts')" in source
        assert "setHotspots(data.hotspots || [])" in source

    def test_backend_broadcast_includes_the_alert_fields(self):
        source = (
            BACKEND_ROOT / "app" / "api" / "ws_handler.py"
        ).read_text(encoding="utf-8")
        for field in ('"alerts"', '"hotspots"', '"cascade_plan"', '"graph"'):
            assert field in source, f"{field} missing from broadcast payload"
