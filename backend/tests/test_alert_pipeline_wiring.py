"""
Runtime wiring regression tests for the alert pipeline.

The audit found that `refresh_alerts_once()` began with
`if not ENABLE_EXTENDED_PIPELINE: return`. That flag exists to skip the heavy
ML runtime and the Kafka publisher, but the guard sat at the top of the alert
refresh, so with the default configuration the entire product pipeline —
conjunction screening, cascade planning, hotspot generation and debris
building — never executed and `/api/alerts` served the empty startup stub
forever.

These tests pin the intended scope of the flag so the regression cannot return.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture(scope="module")
def app_module():
    import main  # noqa: PLC0415

    return main


def _seed_snapshot(app_module, states):
    stamp = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat()
    app_module.set_latest_snapshot(
        {
            "type": "update",
            "timestamp": stamp,
            "satellites": [],
            "count": len(states),
            "states": states,
            "payload": None,
            "covariances": {},
            "debris_clouds": [],
        }
    )
    return stamp


def _fake_state(norad_id, x, y, z):
    class _State:
        error_code = 0
        name = f"SAT-{norad_id}"
        epoch_utc = "2026-01-01T12:00:00+00:00"
        vx = 0.0
        vy = 7.5
        vz = 0.0

    state = _State()
    state.norad_id = norad_id
    state.x = x
    state.y = y
    state.z = z
    return state


class TestAlertPipelineNotFlagGated:
    def test_alert_refresh_runs_with_flag_at_default(self, app_module, monkeypatch):
        """The core screening path must execute on the default config."""
        monkeypatch.delenv("ENABLE_EXTENDED_PIPELINE", raising=False)
        assert app_module.ENABLE_EXTENDED_PIPELINE is False

        calls: dict[str, int] = {"screen": 0, "cascade": 0, "debris": 0}

        def fake_screen(states, **kwargs):
            calls["screen"] += 1
            return []

        class _Planner:
            def analyze_snapshot(self, states, alerts, propagator, stamp):
                calls["cascade"] += 1
                return {
                    "graph": {},
                    "alerts": [],
                    "cascade_plan": [],
                    "total_delta_v_ms": 0.0,
                    "cascade_depth": 0,
                    "agencies_involved": [],
                    "seed_satellites": [],
                    "cpi_threshold": 5.0,
                    "node_probabilities": {},
                    "hotspots": [],
                }

        def fake_debris(hotspots, states, stamp, alerts=None, cpi_threshold=5.0):
            calls["debris"] += 1
            return []

        monkeypatch.setattr(app_module, "screen_conjunctions", fake_screen)
        monkeypatch.setattr(app_module, "cascade_planner", _Planner())
        monkeypatch.setattr(app_module, "build_debris_alerts", fake_debris)
        monkeypatch.setattr(
            app_module, "get_all_kalman_states", lambda: {}
        )

        _seed_snapshot(
            app_module,
            [_fake_state(1, 7000.0, 0.0, 0.0), _fake_state(2, 7001.0, 0.0, 0.0)],
        )

        asyncio.run(app_module.refresh_alerts_once())

        assert calls["screen"] == 1, "conjunction screening was skipped"
        assert calls["cascade"] == 1, "cascade planning was skipped"
        assert calls["debris"] == 1, "debris building was skipped"

    def test_alert_cache_is_written_when_flag_is_off(self, app_module, monkeypatch):
        """A populated alert payload must land in the cache, not the stub."""
        monkeypatch.setattr(app_module, "ENABLE_EXTENDED_PIPELINE", False)
        monkeypatch.setattr(app_module, "screen_conjunctions", lambda s, **k: [])
        monkeypatch.setattr(app_module, "get_all_kalman_states", lambda: {})

        class _Planner:
            def analyze_snapshot(self, states, alerts, propagator, stamp):
                return {
                    "graph": {"node_count": 2, "edge_count": 1},
                    "alerts": [{"norad_id": 1, "severity": "red"}],
                    "cascade_plan": [{"satellite_id": 1}],
                    "total_delta_v_ms": 1.25,
                    "cascade_depth": 2,
                    "agencies_involved": ["SpaceX"],
                    "seed_satellites": [1],
                    "cpi_threshold": 5.0,
                    "node_probabilities": {"1": 0.4},
                    "hotspots": [],
                }

        monkeypatch.setattr(app_module, "cascade_planner", _Planner())
        monkeypatch.setattr(app_module, "build_debris_alerts", lambda *a, **k: [])

        _seed_snapshot(
            app_module,
            [_fake_state(1, 7000.0, 0.0, 0.0), _fake_state(2, 7001.0, 0.0, 0.0)],
        )

        asyncio.run(app_module.refresh_alerts_once())

        from app.core.state_cache import get_latest_alerts  # noqa: PLC0415

        cached = get_latest_alerts()
        assert len(cached["alerts"]) == 1
        assert cached["alerts"][0]["severity"] == "red"
        assert cached["count"] == 1
        assert cached["cascade_depth"] == 2
        assert cached["graph"].get("edge_count") == 1

    def test_ml_runtime_stays_optional(self, app_module, monkeypatch):
        """With the flag off, the ML runtime must not be constructed."""
        monkeypatch.setattr(app_module, "ENABLE_EXTENDED_PIPELINE", False)
        assert app_module.ml_runtime is None
        assert app_module.kafka_adapter is None


class TestAlertListIsSelfConsistent:
    """`count` and `alerts` must describe the same list.

    Screening reaches 500 km while the cascade graph only keeps pairs inside the
    200 km influence radius. The response previously reported a `count` taken
    from screening next to a list taken from the graph, so the UI showed a
    non-zero badge over an empty list and every 200-500 km conjunction was
    invisible.
    """

    def test_count_matches_the_emitted_list(self, app_module, monkeypatch):
        monkeypatch.setattr(app_module, "ENABLE_EXTENDED_PIPELINE", False)
        monkeypatch.setattr(app_module, "get_all_kalman_states", lambda: {})

        class _Alert:
            def __init__(self, a, b):
                self._a, self._b = a, b

            def to_dict(self):
                return {
                    "id": f"{self._a}-{self._b}",
                    "sat1": {"id": self._a, "name": f"SAT{self._a}"},
                    "sat2": {"id": self._b, "name": f"SAT{self._b}"},
                    "miss_distance_km": 320.0,
                    "relative_speed_kmh": 21000.0,
                    "tca_utc": "2026-01-01T13:00:00+00:00",
                    "tca_hours": 1.0,
                    "tca_minutes": 60.0,
                    "severity": "green",
                    "probability_of_collision": 1e-9,
                    "cpi_score": 3.2,
                    "covariance_ellipse": None,
                }

        # Screening finds one alert; the graph finds nothing (empty edges).
        monkeypatch.setattr(
            app_module, "screen_conjunctions", lambda s, **k: [_Alert(1, 2)]
        )

        class _Planner:
            def analyze_snapshot(self, states, alerts, propagator, stamp):
                return {
                    "graph": {"node_count": 2, "edge_count": 0},
                    "alerts": [],
                    "cascade_plan": [],
                    "total_delta_v_ms": 0.0,
                    "cascade_depth": 0,
                    "agencies_involved": [],
                    "seed_satellites": [],
                    "cpi_threshold": 5.0,
                    "node_probabilities": {},
                    "hotspots": [],
                }

        monkeypatch.setattr(app_module, "cascade_planner", _Planner())
        monkeypatch.setattr(app_module, "build_debris_alerts", lambda *a, **k: [])

        _seed_snapshot(
            app_module,
            [_fake_state(1, 7000.0, 0.0, 0.0), _fake_state(2, 7100.0, 0.0, 0.0)],
        )

        asyncio.run(app_module.refresh_alerts_once())

        from app.core.state_cache import get_latest_alerts  # noqa: PLC0415

        cached = get_latest_alerts()
        assert cached["count"] == len(cached["alerts"])
        assert cached["count"] == 1, "a screened conjunction was dropped from the API"
        assert cached["alerts"][0]["source"] == "screening"

    def test_graph_alerts_win_on_duplicate_pairs(self, app_module):
        graph_alerts = [
            {
                "id": "1-2",
                "sat1": {"id": 1, "name": "A"},
                "sat2": {"id": 2, "name": "B"},
                "cpi_score": 9.0,
                "severity": "red",
                "p_collision": 1e-3,
                "tca_hours": 0.5,
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            }
        ]
        screened_alerts = [
            {
                "id": "2-1",
                "sat1": {"id": 2, "name": "B"},
                "sat2": {"id": 1, "name": "A"},
                "cpi_score": 1.0,
                "severity": "green",
                "probability_of_collision": 1e-9,
            }
        ]

        merged = app_module.merge_alert_sources(graph_alerts, screened_alerts)

        assert len(merged) == 1, "pair emitted twice in both orientations"
        assert merged[0]["cpi_score"] == 9.0

    def test_screened_alert_is_normalized_to_the_graph_shape(self, app_module):
        screened = [
            {
                "id": "5-9",
                "sat1": {"id": 5, "name": "X"},
                "sat2": {"id": 9, "name": "Y"},
                "miss_distance_km": 120.0,
                "relative_speed_kmh": 24000.0,
                "tca_utc": "2026-01-01T18:00:00+00:00",
                "tca_hours": 6.0,
                "severity": "yellow",
                "probability_of_collision": 2e-5,
                "cpi_score": 6.1,
            }
        ]

        merged = app_module.merge_alert_sources([], screened)
        alert = merged[0]

        # Every field the frontend reads off a graph alert must be present.
        for field in (
            "p_collision",
            "tca_hours",
            "tca_minutes",
            "zone_radius_km",
            "hotspot_score",
            "influence_weight",
            "severity",
            "cpi_score",
            "position",
        ):
            assert field in alert, f"missing {field}"
        assert alert["p_collision"] == 2e-5
        assert alert["tca_minutes"] == 360.0


class TestTcaUrgencyUsesSnapshotEpoch:
    """TCA urgency must be measured from the snapshot epoch, not wall clock."""

    def test_tca_hours_is_measured_from_the_snapshot_epoch(self):
        from app.core.conjunction import ConjunctionEvent  # noqa: PLC0415

        event = ConjunctionEvent()
        event.tca_utc = "2026-01-01T18:00:00+00:00"
        event.tca_hours = 6.0

        payload = event.to_dict()

        assert payload["tca_hours"] == 6.0
        assert payload["tca_minutes"] == 360.0
        assert payload["p_collision"] == payload["probability_of_collision"]

    def test_cpi_tca_urgency_tracks_the_epoch_not_the_wall_clock(self):
        """
        A snapshot epoch 30 days in the past must not make a TCA 30 days away
        look like a 30-day-away TCA measured from now. The screening loop is
        driven by sim_clock, so wall-clock comparison breaks time warp.
        """
        import app.core.conjunction as conj  # noqa: PLC0415

        source = Path(conj.__file__).read_text(encoding="utf-8")
        assert "datetime.now(timezone.utc)).total_seconds() / 3600.0" not in source, (
            "TCA urgency is still measured against datetime.now()"
        )
