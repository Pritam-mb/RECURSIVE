"""
Contract tests for /api/ml/status, /api/ml/retrain, RLHF decision counting and
the deterministic telemetry fallback.

The TestClient is used without a context manager so the app lifespan (TLE
fetch, background loops) never runs; routes are initialised with fakes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes
import app.ml.xgboost_scorer as scorer_module
from app.ml import rlhf_store


@pytest.fixture(scope="module")
def app_module():
    import main  # noqa: PLC0415

    return main


@pytest.fixture
def client(app_module):
    return TestClient(app_module.app)


@pytest.fixture(autouse=True)
def isolated_rlhf_db(tmp_path, monkeypatch):
    monkeypatch.setenv("RLHF_DB_PATH", str(tmp_path / "rlhf.db"))
    monkeypatch.delenv("ENABLE_EXTENDED_PIPELINE", raising=False)


class _FakeState:
    error_code = 0
    epoch_utc = "2026-01-01T12:00:00+00:00"

    def __init__(self, norad_id):
        self.norad_id = norad_id
        self.name = f"SAT-{norad_id}"


class _FakePropagator:
    def propagate_one(self, norad_id, *_args, **_kwargs):
        return _FakeState(norad_id)


class _FakeSimEngine:
    def __init__(self):
        self.calls = []

    def apply_maneuver(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"status": "noop"}


@pytest.fixture
def fake_routes(monkeypatch):
    monkeypatch.setattr(routes, "_propagator", _FakePropagator())
    monkeypatch.setattr(routes, "_sim_engine", _FakeSimEngine())
    monkeypatch.setattr(routes, "_require_satellite_authority", lambda *_a, **_k: "SAT")


class TestMlStatusShape:
    def test_shape_with_pipeline_off(self, client):
        response = client.get("/api/ml/status")
        assert response.status_code == 200
        body = response.json()

        assert set(body) == {"pipeline_enabled", "xgboost", "lstm", "rlhf", "meta_propagator", "shadow"}
        assert body["pipeline_enabled"] is False

        assert set(body["xgboost"]) == {"using_trained", "predictions_today", "high_risk_today"}
        assert isinstance(body["xgboost"]["using_trained"], bool)
        assert isinstance(body["xgboost"]["predictions_today"], int)
        assert isinstance(body["xgboost"]["high_risk_today"], int)

        assert body["lstm"] == {
            "trained": False,
            "satellites_tracked": 0,
            "buffer_records": 0,
            "buffer_threshold": 500,
        }

        assert body["rlhf"] == {
            "decisions": 0,
            "approved": 0,
            "rejected": 0,
            "approval_rate": None,
            "rounds": 0,
            "round_approval_rates": [],
        }

        assert body["meta_propagator"] == {
            "satellites_with_corrections": 0,
            "mean_improvement_pct": 0.0,
            "corrections": 0,
        }

        assert body["shadow"] == {"enabled": False, "last_retrain": None, "last_result": None}

    def test_status_does_not_build_the_runtime(self, client, monkeypatch):
        import app.ml.runtime as runtime_module

        monkeypatch.setattr(
            runtime_module.MLRuntime,
            "__init__",
            lambda self: pytest.fail("ML runtime constructed by /api/ml/status"),
        )
        assert client.get("/api/ml/status").status_code == 200

    def test_xgboost_counter_tracks_scores(self, client, monkeypatch):
        scorer_module.prediction_counter.reset()
        monkeypatch.setattr(scorer_module.XGBoostScorer, "_score", lambda self, f: f)

        scorer = scorer_module.XGBoostScorer()
        scorer.score(0.9)
        scorer.score(1e-7)  # surrogate Pc below the 1e-4 high-risk threshold
        scorer.score(0.7, record=False)  # evaluation path is not counted

        xgb = client.get("/api/ml/status").json()["xgboost"]
        assert xgb["predictions_today"] == 2
        assert xgb["high_risk_today"] == 1
        scorer_module.prediction_counter.reset()


class TestRetrainEndpoint:
    def test_retrain_reports_disabled_pipeline(self, client):
        response = client.post("/api/ml/retrain")
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is False
        assert body["result"] is None
        assert "ENABLE_EXTENDED_PIPELINE" in body["message"]

    def test_retrain_runs_shadow_retrain(self, client, monkeypatch):
        class _Runtime:
            def shadow_retrain(self):
                return {"lstm": {"status": "skipped"}, "risk": {"status": "ok"}}

        monkeypatch.setattr(routes, "get_ml_runtime_if_enabled", lambda: _Runtime())
        body = client.post("/api/ml/retrain").json()
        assert body["ok"] is True
        assert body["result"]["risk"]["status"] == "ok"


class TestRlhfCounter:
    def _post(self, client, decision):
        return client.post(
            "/api/feedback/maneuver",
            json={"alert_id": "a1", "sat1_id": 25544, "sat2_id": 1, "decision": decision},
        )

    def test_feedback_increments_counters(self, client, fake_routes):
        assert self._post(client, "APPROVE").status_code == 200
        assert self._post(client, "REJECT").status_code == 200
        assert self._post(client, "MODIFY").status_code == 200

        rlhf = client.get("/api/ml/status").json()["rlhf"]
        assert rlhf["decisions"] == 3
        assert rlhf["approved"] == 2
        assert rlhf["rejected"] == 1
        assert rlhf["approval_rate"] == pytest.approx(2 / 3, abs=1e-3)
        assert rlhf["rounds"] == 0
        assert rlhf["round_approval_rates"] == []

    def test_rounds_every_ten_decisions(self):
        for i in range(25):
            rlhf_store.record_decision("APPROVE" if i < 10 or i % 2 == 0 else "REJECT")
        stats = rlhf_store.get_stats()
        assert stats["decisions"] == 25
        assert stats["rounds"] == 2
        assert stats["round_approval_rates"] == [1.0, 0.5]

    def test_unknown_decision_is_not_counted(self):
        assert rlhf_store.record_decision("MAYBE") is False
        assert rlhf_store.get_stats()["decisions"] == 0


class TestTelemetryFallback:
    def test_fallback_reports_missing_not_invented(self, client, fake_routes, monkeypatch):
        """Without the tracker there is no telemetry source: values must be
        null, never synthesised numbers."""
        monkeypatch.setattr(routes, "_TRACKER_AVAILABLE", False)
        monkeypatch.setattr(routes, "satellite_tracker", None)

        body = client.get("/api/satellites/25544/telemetry").json()

        assert body["telemetry_source"] == "unavailable"
        assert body["telemetry_available"] is False
        for key in ("fuel_remaining_pct", "battery_pct", "temperature_c", "signal_strength_dbm"):
            assert body[key] is None

    def test_tracker_module_imports_cleanly(self):
        from app.core.satellite_state_tracker import satellite_tracker

        assert satellite_tracker is not None
