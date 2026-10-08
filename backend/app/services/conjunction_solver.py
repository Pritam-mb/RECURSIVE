"""
Conjunction solver for test mode.
Propagates two injected satellites and finds the minimum separation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.137
J2_EARTH = 1.08262668e-3

# A conjunction is only "confirmed" once the miss distance is at or below this
# hard contact gate. Debris clouds are generated exclusively for confirmed
# collisions; wider bands are warnings only.
COLLISION_THRESHOLD_M = 1.0

# Report band: inside this range a full conjunction report is built and the
# approach is classified as a warning. This is intentionally much wider than
# COLLISION_THRESHOLD_M, so a multi-kilometre near miss is reported and given a
# cascade plan, but never produces debris.
CONJUNCTION_REPORT_DISTANCE_M = 5000.0


@dataclass
class EciState:
    position_km: np.ndarray
    velocity_kms: np.ndarray
    epoch_utc: datetime


@dataclass
class TcaResult:
    tca_utc: str
    miss_distance_m: float
    relative_velocity_kms: float
    position_a_eci: list[float]
    position_b_eci: list[float]
    velocity_a_eci: list[float]
    velocity_b_eci: list[float]
    collision_detected: bool
    trajectory_a: list[list]
    trajectory_b: list[list]


def _accel_two_body(
    position_km: np.ndarray,
    use_j2: bool = True,
) -> np.ndarray:
    r2 = float(np.dot(position_km, position_km))
    r = np.sqrt(r2)
    if r == 0.0:
        return np.zeros(3)

    accel = -MU_EARTH_KM3_S2 * position_km / (r**3)

    if use_j2:
        z2 = position_km[2] ** 2
        factor = 1.5 * J2_EARTH * MU_EARTH_KM3_S2 * (R_EARTH_KM**2) / (r**5)
        scale = (5.0 * z2 / r2) - 1.0
        accel_j2 = np.array([
            position_km[0] * scale,
            position_km[1] * scale,
            position_km[2] * ((5.0 * z2 / r2) - 3.0),
        ])
        accel = accel + factor * accel_j2

    return accel


def _rk45_propagate(
    state: EciState,
    t_eval: np.ndarray,
    use_j2: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if len(t_eval) == 1:
        positions = np.array([state.position_km])
        velocities = np.array([state.velocity_kms])
        return positions, velocities

    def dynamics(_t: float, y: np.ndarray) -> np.ndarray:
        position = y[:3]
        velocity = y[3:]
        accel = _accel_two_body(position, use_j2=use_j2)
        return np.concatenate((velocity, accel))

    y0 = np.concatenate((state.position_km, state.velocity_kms))
    solution = solve_ivp(
        dynamics,
        (float(t_eval[0]), float(t_eval[-1])),
        y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8,
    )

    if not solution.success or solution.y.shape[1] != len(t_eval):
        # Fallback: linear propagation for robustness
        dt = t_eval - t_eval[0]
        positions = state.position_km + np.outer(dt, state.velocity_kms)
        velocities = np.repeat(state.velocity_kms[None, :], len(t_eval), axis=0)
        return positions, velocities

    positions = solution.y[:3].T
    velocities = solution.y[3:].T
    return positions, velocities


def _build_time_grid(total_seconds: float, step_seconds: float, max_steps: int) -> np.ndarray:
    """
    Build the sampling grid for a propagation window.

    Guarantees:
      * the final sample is exactly `total_seconds`, so the requested target
        time is always covered;
      * at most `max_steps` samples are produced.
    """
    if total_seconds <= 0.0:
        return np.array([0.0], dtype=float)

    max_steps = max(2, int(max_steps))
    step_seconds = max(float(step_seconds), 1e-6)

    # Uniform samples strictly inside the window, capped so that the exact
    # target can always be appended as the final node.
    interior = int(np.floor(total_seconds / step_seconds))
    interior = min(interior, max_steps - 2)
    grid = np.arange(interior + 1, dtype=float) * step_seconds
    grid = grid[grid < total_seconds]

    if grid.size == 0:
        return np.array([0.0, total_seconds], dtype=float)

    if not np.isclose(grid[-1], total_seconds, rtol=0.0, atol=1e-9):
        grid = np.append(grid, total_seconds)

    return grid


def _hermite_eval(
    t_query: np.ndarray,
    nodes: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a piecewise cubic Hermite interpolant of position and velocity.

    Node positions/velocities are the RK45 samples, so the interpolant is
    C1-continuous across every node and reproduces the integrated state
    exactly at the nodes.
    """
    t_query = np.atleast_1d(np.asarray(t_query, dtype=float))
    out_pos = np.zeros((t_query.size, 3), dtype=float)
    out_vel = np.zeros((t_query.size, 3), dtype=float)

    for idx in range(len(nodes) - 1):
        t0 = float(nodes[idx])
        t1 = float(nodes[idx + 1])
        mask = (t_query >= t0) & (t_query <= t1)
        if not np.any(mask):
            continue

        h = t1 - t0
        if h <= 0.0:
            continue

        s = (t_query[mask] - t0) / h
        s2 = s * s
        s3 = s2 * s

        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2

        p0 = positions[idx]
        v0 = velocities[idx]
        p1 = positions[idx + 1]
        v1 = velocities[idx + 1]

        out_pos[mask] = (
            h00[:, None] * p0
            + (h10 * h)[:, None] * v0
            + h01[:, None] * p1
            + (h11 * h)[:, None] * v1
        )
        out_vel[mask] = (
            ((6.0 * s2 - 6.0 * s) / h)[:, None] * p0
            + (3.0 * s2 - 4.0 * s + 1.0)[:, None] * v0
            + ((-6.0 * s2 + 6.0 * s))[:, None] * p1
            + (3.0 * s2 - 2.0 * s)[:, None] * v1
        )

    return out_pos, out_vel


def _refine_tca(
    nodes: np.ndarray,
    rel_positions: np.ndarray,
    rel_velocities: np.ndarray,
    min_index: int,
) -> float:
    """
    Sub-sample refinement of the closest approach inside the bracketing step.

    Uses a cubic Hermite interpolant of the relative state and a bounded
    minimisation of the squared separation, so the reported TCA is not limited
    to the coarse sampling grid.
    """
    lo = max(0, min_index - 1)
    hi = min(len(nodes) - 1, min_index + 1)
    if hi <= lo:
        return float(nodes[min_index])

    t_lo = float(nodes[lo])
    t_hi = float(nodes[hi])
    if t_hi <= t_lo:
        return float(nodes[min_index])

    def objective(t_value: float) -> float:
        rel_pos, _ = _hermite_eval(
            np.array([t_value], dtype=float),
            nodes,
            rel_positions,
            rel_velocities,
        )
        return float(np.dot(rel_pos[0], rel_pos[0]))

    try:
        result = minimize_scalar(
            objective,
            bounds=(t_lo, t_hi),
            method="bounded",
            options={"xatol": max(1e-9, (t_hi - t_lo) * 1e-10)},
        )
    except Exception:
        return float(nodes[min_index])

    if result.success and t_lo <= float(result.x) <= t_hi:
        return float(result.x)

    return float(nodes[min_index])


def run_to_tca(
    state_a: EciState,
    state_b: EciState,
    target_utc: datetime,
    step_seconds: float,
    max_steps: int,
    use_j2: bool = True,
) -> TcaResult:
    start = state_a.epoch_utc
    if target_utc.tzinfo is None:
        target_utc = target_utc.replace(tzinfo=timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    total_seconds = max(0.0, (target_utc - start).total_seconds())
    if step_seconds <= 0:
        step_seconds = 1.0

    t_eval = _build_time_grid(total_seconds, step_seconds, max_steps)

    positions_a, velocities_a = _rk45_propagate(state_a, t_eval, use_j2=use_j2)
    positions_b, velocities_b = _rk45_propagate(state_b, t_eval, use_j2=use_j2)

    rel_positions = positions_a - positions_b
    rel_velocities = velocities_a - velocities_b
    distances_km = np.linalg.norm(rel_positions, axis=1)
    min_index = int(np.argmin(distances_km))

    refined_offset = _refine_tca(t_eval, rel_positions, rel_velocities, min_index)
    refined_pos = None
    refined_vel = None
    refined_distance_km = None
    if refined_offset != float(t_eval[min_index]):
        refined_pos, refined_vel = _hermite_eval(
            np.array([refined_offset], dtype=float),
            t_eval,
            rel_positions,
            rel_velocities,
        )
        refined_distance_km = float(np.linalg.norm(refined_pos[0]))

    # Accept the refined point only when it actually improves on the sample.
    use_refined = (
        refined_distance_km is not None
        and refined_distance_km <= float(distances_km[min_index])
    )

    if use_refined:
        # The reported states must describe the same instant as the refined
        # TCA, so re-evaluate each satellite's absolute state at that offset.
        # At a grid node the interpolant reproduces the integrated state
        # exactly, so this stays consistent with the sampled trajectories.
        tca_offset = float(refined_offset)
        min_distance_km = refined_distance_km
        rel_velocity_kms = float(np.linalg.norm(refined_vel[0]))
        refined_a_pos, refined_a_vel = _hermite_eval(
            np.array([tca_offset], dtype=float),
            t_eval,
            positions_a,
            velocities_a,
        )
        refined_b_pos, refined_b_vel = _hermite_eval(
            np.array([tca_offset], dtype=float),
            t_eval,
            positions_b,
            velocities_b,
        )
        position_a_eci = [float(v) for v in refined_a_pos[0]]
        position_b_eci = [float(v) for v in refined_b_pos[0]]
        velocity_a_eci = [float(v) for v in refined_a_vel[0]]
        velocity_b_eci = [float(v) for v in refined_b_vel[0]]
    else:
        tca_offset = float(t_eval[min_index])
        min_distance_km = float(distances_km[min_index])
        rel_velocity_kms = float(np.linalg.norm(rel_velocities[min_index]))
        position_a_eci = [float(v) for v in positions_a[min_index]]
        position_b_eci = [float(v) for v in positions_b[min_index]]
        velocity_a_eci = [float(v) for v in velocities_a[min_index]]
        velocity_b_eci = [float(v) for v in velocities_b[min_index]]

    tca_time = start + timedelta(seconds=tca_offset)

    tca_utc = tca_time.replace(tzinfo=timezone.utc).isoformat()

    trajectory_a = []
    trajectory_b = []
    for i, t_offset in enumerate(t_eval):
        ts = start + timedelta(seconds=float(t_offset))
        iso = ts.replace(tzinfo=timezone.utc).isoformat()
        trajectory_a.append([iso, float(positions_a[i][0]), float(positions_a[i][1]), float(positions_a[i][2])])
        trajectory_b.append([iso, float(positions_b[i][0]), float(positions_b[i][1]), float(positions_b[i][2])])

    collision_detected = (min_distance_km * 1000.0) <= COLLISION_THRESHOLD_M

    return TcaResult(
        tca_utc=tca_utc,
        miss_distance_m=round(min_distance_km * 1000.0, 3),
        relative_velocity_kms=round(rel_velocity_kms, 5),
        position_a_eci=position_a_eci,
        position_b_eci=position_b_eci,
        velocity_a_eci=velocity_a_eci,
        velocity_b_eci=velocity_b_eci,
        collision_detected=collision_detected,
        trajectory_a=trajectory_a,
        trajectory_b=trajectory_b,
    )


def parse_eci_state(
    position_eci_km: Iterable[float],
    velocity_eci_kms: Iterable[float],
    epoch_utc: datetime,
) -> EciState:
    return EciState(
        position_km=np.array(list(position_eci_km), dtype=float),
        velocity_kms=np.array(list(velocity_eci_kms), dtype=float),
        epoch_utc=epoch_utc,
    )
