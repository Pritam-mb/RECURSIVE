"""Deterministic regression tests for the conjunction solver numerics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.services.conjunction_solver import (
    COLLISION_THRESHOLD_M,
    MU_EARTH_KM3_S2,
    _build_time_grid,
    _hermite_eval,
    _refine_tca,
    parse_eci_state,
    run_to_tca,
)

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
RADIUS_KM = 7000.0
ORBIT_SPEED_KMS = float(np.sqrt(MU_EARTH_KM3_S2 / RADIUS_KM))


def _co_orbital_pair(offset_km: float):
    """Two objects on the same circular orbit separated by an along-track offset."""
    state_a = parse_eci_state(
        [RADIUS_KM, 0.0, 0.0],
        [0.0, ORBIT_SPEED_KMS, 0.0],
        EPOCH,
    )
    state_b = parse_eci_state(
        [RADIUS_KM, offset_km, 0.0],
        [0.0, ORBIT_SPEED_KMS, 0.0],
        EPOCH,
    )
    return state_a, state_b


def _crossing_pair(sep_km: float, tca_seconds: float = 300.0):
    """
    Two co-planar circular orbits whose separation has a true interior minimum.

    The radial separation sets the floor of the range; the phase drift is solved
    so the objects are radially aligned at ``tca_seconds``. This is the geometry
    that forces the sub-sample refinement to move the reported TCA off the
    coarse sampling grid.
    """
    radius_b = RADIUS_KM + sep_km
    omega_a = float(np.sqrt(MU_EARTH_KM3_S2 / RADIUS_KM**3))
    omega_b = float(np.sqrt(MU_EARTH_KM3_S2 / radius_b**3))
    phase = (omega_a - omega_b) * tca_seconds

    state_a = parse_eci_state(
        [RADIUS_KM, 0.0, 0.0],
        [0.0, ORBIT_SPEED_KMS, 0.0],
        EPOCH,
    )
    state_b = parse_eci_state(
        [
            radius_b * float(np.cos(phase)),
            radius_b * float(np.sin(phase)),
            0.0,
        ],
        [
            -omega_b * radius_b * float(np.sin(phase)),
            omega_b * radius_b * float(np.cos(phase)),
            0.0,
        ],
        EPOCH,
    )
    return state_a, state_b


class TestTimeGrid:
    @pytest.mark.parametrize(
        ("total_seconds", "step_seconds", "max_steps"),
        [
            (100.0, 30.0, 1000),
            (100.0, 30.0, 3),
            (7.5, 2.0, 1000),
            (0.0, 5.0, 10),
            (1.0, 1.0, 2),
            (86400.0, 60.0, 100),
        ],
    )
    def test_grid_ends_exactly_at_target(self, total_seconds, step_seconds, max_steps):
        grid = _build_time_grid(total_seconds, step_seconds, max_steps)
        assert grid[-1] == pytest.approx(total_seconds, abs=1e-9)

    @pytest.mark.parametrize("max_steps", [2, 3, 10, 1000])
    def test_grid_respects_step_cap(self, max_steps):
        grid = _build_time_grid(3600.0, 10.0, max_steps)
        assert len(grid) <= max_steps

    def test_grid_is_monotonic_and_non_negative(self):
        grid = _build_time_grid(600.0, 7.0, 1000)
        assert np.all(np.diff(grid) > 0)
        assert grid[0] == 0.0

    def test_zero_duration_returns_single_sample(self):
        grid = _build_time_grid(0.0, 10.0, 100)
        assert grid.tolist() == [0.0]


class TestHermiteRefinement:
    def _linear_relative_motion(self, nodes):
        def rel_pos(t):
            return np.array([1.0, 0.2 - 0.3 * t, 0.0])

        def rel_vel(_t):
            return np.array([0.0, -0.3, 0.0])

        positions = np.array([rel_pos(t) for t in nodes])
        velocities = np.array([rel_vel(t) for t in nodes])
        return positions, velocities

    def test_refinement_finds_off_grid_minimum(self):
        nodes = np.array([0.0, 1.0, 2.0, 3.0])
        positions, velocities = self._linear_relative_motion(nodes)

        distances = np.linalg.norm(positions, axis=1)
        coarse_index = int(np.argmin(distances))
        coarse_t = float(nodes[coarse_index])
        coarse_distance = float(distances[coarse_index])

        refined_t = _refine_tca(nodes, positions, velocities, coarse_index)
        refined_positions, _ = _hermite_eval(
            np.array([refined_t]), nodes, positions, velocities
        )
        refined_distance = float(np.linalg.norm(refined_positions[0]))

        # True minimum of the linear model is at t = 0.2 / 0.3.
        assert refined_t == pytest.approx(2.0 / 3.0, abs=1e-6)
        assert abs(refined_t - 2.0 / 3.0) < abs(coarse_t - 2.0 / 3.0)
        assert refined_distance <= coarse_distance

    def test_hermite_reproduces_node_values(self):
        nodes = np.array([0.0, 1.0, 2.0, 3.0])
        positions, velocities = self._linear_relative_motion(nodes)

        pos_out, vel_out = _hermite_eval(nodes, nodes, positions, velocities)

        assert np.allclose(pos_out, positions)
        assert np.allclose(vel_out, velocities)

    def test_refinement_stays_within_bracket(self):
        nodes = np.array([0.0, 2.0, 4.0])
        positions, velocities = self._linear_relative_motion(nodes)
        refined_t = _refine_tca(nodes, positions, velocities, 1)
        assert nodes[0] <= refined_t <= nodes[-1]


class TestRunToTca:
    def test_trajectory_covers_the_exact_requested_target(self):
        state_a, state_b = _co_orbital_pair(0.5)
        target = EPOCH + timedelta(seconds=600)

        result = run_to_tca(
            state_a=state_a,
            state_b=state_b,
            target_utc=target,
            step_seconds=30.0,
            max_steps=1000,
        )

        assert result.trajectory_a[-1][0] == target.isoformat()
        assert result.trajectory_b[-1][0] == target.isoformat()

    def test_reported_tca_lies_inside_the_window(self):
        state_a, state_b = _co_orbital_pair(0.5)
        target = EPOCH + timedelta(seconds=600)

        result = run_to_tca(
            state_a=state_a,
            state_b=state_b,
            target_utc=target,
            step_seconds=30.0,
            max_steps=1000,
        )

        tca = datetime.fromisoformat(result.tca_utc)
        assert EPOCH <= tca <= target

    def test_miss_distance_never_exceeds_coarse_grid_minimum(self):
        state_a, state_b = _co_orbital_pair(0.5)
        result = run_to_tca(
            state_a=state_a,
            state_b=state_b,
            target_utc=EPOCH + timedelta(seconds=600),
            step_seconds=30.0,
            max_steps=1000,
        )

        coarse = []
        for track_a, track_b in zip(result.trajectory_a, result.trajectory_b):
            delta = np.array(track_a[1:4]) - np.array(track_b[1:4])
            coarse.append(float(np.linalg.norm(delta)) * 1000.0)

        assert result.miss_distance_m <= min(coarse) + 1e-6

    def test_tight_collision_is_detected(self):
        state_a, state_b = _co_orbital_pair(0.0005)
        result = run_to_tca(
            state_a=state_a,
            state_b=state_b,
            target_utc=EPOCH + timedelta(seconds=600),
            step_seconds=30.0,
            max_steps=1000,
        )

        assert result.collision_detected is True
        assert result.miss_distance_m <= COLLISION_THRESHOLD_M

    def test_multi_kilometre_near_miss_is_not_a_collision(self):
        state_a, state_b = _co_orbital_pair(2.5)
        result = run_to_tca(
            state_a=state_a,
            state_b=state_b,
            target_utc=EPOCH + timedelta(seconds=600),
            step_seconds=30.0,
            max_steps=1000,
        )

        assert result.collision_detected is False
        assert result.miss_distance_m > COLLISION_THRESHOLD_M

    def test_zero_step_size_is_tolerated(self):
        state_a, state_b = _co_orbital_pair(0.5)
        result = run_to_tca(
            state_a=state_a,
            state_b=state_b,
            target_utc=EPOCH + timedelta(seconds=120),
            step_seconds=0.0,
            max_steps=100,
        )
        assert result.miss_distance_m >= 0.0

    @pytest.mark.parametrize("sep_km", [50.0, 100.0, 300.0])
    def test_reported_states_match_the_reported_tca(self, sep_km):
        """
        The reported absolute states must describe the same instant as tca_utc.

        The coarse grid forces the sub-sample refinement to engage; the returned
        positions/velocities still have to reproduce the reported miss distance
        and relative velocity, which they cannot do if they are sampled from a
        neighbouring grid node.
        """
        state_a, state_b = _crossing_pair(sep_km)
        result = run_to_tca(
            state_a=state_a,
            state_b=state_b,
            target_utc=EPOCH + timedelta(seconds=600),
            step_seconds=200.0,
            max_steps=100,
        )

        separation_m = float(
            np.linalg.norm(
                np.array(result.position_a_eci) - np.array(result.position_b_eci)
            )
        ) * 1000.0
        # miss_distance_m is rounded to millimetres, velocity to 1e-5 km/s.
        assert separation_m == pytest.approx(result.miss_distance_m, rel=1e-6, abs=1e-3)

        relative_velocity_kms = float(
            np.linalg.norm(
                np.array(result.velocity_a_eci) - np.array(result.velocity_b_eci)
            )
        )
        assert relative_velocity_kms == pytest.approx(
            result.relative_velocity_kms, rel=1e-6, abs=1e-4
        )

    def test_refinement_engages_on_a_coarse_grid(self):
        """Guards the test above: refinement must actually be exercised."""
        state_a, state_b = _crossing_pair(100.0)
        result = run_to_tca(
            state_a=state_a,
            state_b=state_b,
            target_utc=EPOCH + timedelta(seconds=600),
            step_seconds=200.0,
            max_steps=100,
        )

        coarse = []
        for track_a, track_b in zip(result.trajectory_a, result.trajectory_b):
            delta = np.array(track_a[1:4]) - np.array(track_b[1:4])
            coarse.append(float(np.linalg.norm(delta)) * 1000.0)

        # Strictly better than every sample => the sub-sample path was used.
        assert result.miss_distance_m < min(coarse)

        tca = datetime.fromisoformat(result.tca_utc)
        sampled_times = [datetime.fromisoformat(row[0]) for row in result.trajectory_a]
        assert tca not in sampled_times

    def test_refined_tca_lands_near_the_true_closest_approach(self):
        state_a, state_b = _crossing_pair(100.0, tca_seconds=300.0)
        result = run_to_tca(
            state_a=state_a,
            state_b=state_b,
            target_utc=EPOCH + timedelta(seconds=600),
            step_seconds=200.0,
            max_steps=100,
        )

        tca = datetime.fromisoformat(result.tca_utc)
        offset = (tca - EPOCH).total_seconds()
        # Interior minimum, close to the exact alignment at 300 s, and the
        # separation floor is the 100 km radial difference.
        assert 200.0 < offset < 400.0
        assert offset == pytest.approx(300.0, abs=30.0)
        assert result.miss_distance_m == pytest.approx(100_000.0, rel=0.02)
