"""
Regression tests for the collision-risk threshold separation.

Warning, confirmed collision, and debris generation must use three distinct
thresholds. The specific regression guarded here: a kilometre-scale near miss
used to produce a debris cloud, implying a collision that never happened.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.conjunction_predictor import build_collision_prediction, classify_approach
from app.services.conjunction_solver import (
    COLLISION_THRESHOLD_M,
    CONJUNCTION_REPORT_DISTANCE_M,
    parse_eci_state,
    run_to_tca,
)
from app.services.debris_model import build_debris_alerts

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
RADIUS_KM = 7000.0
ORBIT_SPEED_KMS = float(np.sqrt(398600.4418 / RADIUS_KM))


def _synthetic_result(miss_distance_m: float, collision_detected: bool):
    """Build a solver-shaped result object with a controlled miss distance."""
    return SimpleNamespace(
        tca_utc=(EPOCH + timedelta(seconds=600)).isoformat(),
        miss_distance_m=miss_distance_m,
        relative_velocity_kms=0.01,
        position_a_eci=[RADIUS_KM, 0.0, 0.0],
        position_b_eci=[RADIUS_KM, miss_distance_m / 1000.0, 0.0],
        velocity_a_eci=[0.0, ORBIT_SPEED_KMS, 0.0],
        velocity_b_eci=[0.0, ORBIT_SPEED_KMS, 0.0],
        collision_detected=collision_detected,
        trajectory_a=[],
        trajectory_b=[],
    )


def _satellites():
    sat_a = SimpleNamespace(norad_id=900001, name="SAT-A", epoch_utc=EPOCH)
    sat_b = SimpleNamespace(norad_id=900002, name="SAT-B", epoch_utc=EPOCH)
    return sat_a, sat_b


class _StubPropagator:
    """Minimal propagator stub: no catalog states, so only the pair is analyzed."""

    norad_ids: list[int] = []

    def propagate_one(self, norad_id, epoch=None):
        return None


class TestThresholdsAreDistinct:
    def test_collision_gate_is_one_metre(self):
        assert COLLISION_THRESHOLD_M == 1.0

    def test_report_band_is_wider_than_the_collision_gate(self):
        assert CONJUNCTION_REPORT_DISTANCE_M > COLLISION_THRESHOLD_M

    @pytest.mark.parametrize(
        ("miss_m", "expected"),
        [
            (0.0, "collision"),
            (0.5, "collision"),
            (1.0, "collision"),
            (1.0001, "warning"),
            (100.0, "warning"),
            (5000.0, "warning"),
            (5000.1, "clear"),
            (50_000.0, "clear"),
        ],
    )
    def test_classification_bands(self, miss_m, expected):
        assert classify_approach(miss_m) == expected


class TestDebrisRequiresConfirmedCollision:
    def test_wide_near_miss_produces_no_debris(self):
        sat_a, sat_b = _satellites()
        result = _synthetic_result(miss_distance_m=2500.0, collision_detected=False)

        payload = build_collision_prediction(
            _StubPropagator(),
            sat_a,
            sat_b,
            result,
            collision_confirmed=False,
        )

        assert payload["debris_clouds"] == []
        assert payload["collision_confirmed"] is False
        # The warning report is still produced.
        assert payload["count"] >= 1

    def test_confirmed_collision_may_produce_debris(self):
        sat_a, sat_b = _satellites()
        result = _synthetic_result(miss_distance_m=0.2, collision_detected=True)

        payload = build_collision_prediction(
            _StubPropagator(),
            sat_a,
            sat_b,
            result,
            collision_confirmed=True,
        )

        assert payload["collision_confirmed"] is True

    def test_confirmed_flag_inferred_from_result_when_omitted(self):
        sat_a, sat_b = _satellites()

        near_miss = _synthetic_result(miss_distance_m=800.0, collision_detected=False)
        payload = build_collision_prediction(_StubPropagator(), sat_a, sat_b, near_miss)
        assert payload["debris_clouds"] == []
        assert payload["collision_confirmed"] is False

    def test_caller_cannot_promote_a_near_miss_to_a_collision(self):
        """
        The hard contact gate is authoritative.

        A caller passing collision_confirmed=True for a kilometre-scale near
        miss must still be rejected, otherwise debris is generated for a
        collision that never happened.
        """
        sat_a, sat_b = _satellites()
        result = _synthetic_result(miss_distance_m=2500.0, collision_detected=False)

        payload = build_collision_prediction(
            _StubPropagator(),
            sat_a,
            sat_b,
            result,
            collision_confirmed=True,
        )

        assert payload["collision_confirmed"] is False
        assert payload["debris_clouds"] == []

    def test_gate_is_evaluated_against_the_reported_miss_distance(self):
        """A flag inconsistent with miss_distance_m is resolved by the distance."""
        sat_a, sat_b = _satellites()

        just_inside = _synthetic_result(miss_distance_m=0.9, collision_detected=False)
        payload = build_collision_prediction(
            _StubPropagator(), sat_a, sat_b, just_inside, collision_confirmed=False
        )
        assert payload["collision_confirmed"] is False

        just_outside = _synthetic_result(miss_distance_m=1.1, collision_detected=True)
        payload = build_collision_prediction(
            _StubPropagator(), sat_a, sat_b, just_outside, collision_confirmed=True
        )
        assert payload["collision_confirmed"] is False
        assert payload["debris_clouds"] == []


class TestDebrisModelContract:
    def test_debris_requires_hotspots_and_states(self):
        assert build_debris_alerts([], [], EPOCH.isoformat()) == []
        assert build_debris_alerts([{"cpi_score": 9.0}], [], EPOCH.isoformat()) == []

    def test_cloud_shells_are_layered_and_monotonic(self):
        class _State:
            error_code = 0
            norad_id = 1
            name = "SAT"
            x = RADIUS_KM
            y = 0.0
            z = 0.0

        hotspot = {
            "cpi_score": 9.0,
            "position": {"x": RADIUS_KM, "y": 0.0, "z": 0.0},
            "tca_utc": (EPOCH + timedelta(minutes=30)).isoformat(),
            "tca_minutes": 30.0,
        }
        clouds = build_debris_alerts(
            [hotspot],
            [_State()],
            EPOCH.isoformat(),
            alerts=[],
            cpi_threshold=5.0,
        )

        assert len(clouds) == 1
        shells = clouds[0]["shells"]
        assert len(shells) == 3
        radii = [shell["radius_km"] for shell in shells]
        assert radii == sorted(radii)


class TestSolverAgreementWithThresholds:
    def test_solver_flag_matches_the_one_metre_gate(self):
        state_a = parse_eci_state(
            [RADIUS_KM, 0.0, 0.0], [0.0, ORBIT_SPEED_KMS, 0.0], EPOCH
        )

        for offset_km, expected_collision in [(0.0005, True), (0.5, False), (2.5, False)]:
            state_b = parse_eci_state(
                [RADIUS_KM, offset_km, 0.0], [0.0, ORBIT_SPEED_KMS, 0.0], EPOCH
            )

            result = run_to_tca(
                state_a=state_a,
                state_b=state_b,
                target_utc=EPOCH + timedelta(seconds=600),
                step_seconds=30.0,
                max_steps=1000,
            )
            assert result.collision_detected is expected_collision
            assert result.collision_detected == (
                result.miss_distance_m <= COLLISION_THRESHOLD_M
            )
            assert classify_approach(result.miss_distance_m) in ("collision", "warning")
