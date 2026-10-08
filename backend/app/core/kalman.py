"""
Lightweight Kalman filter bookkeeping attached to SGP4Propagator.propagate_one.

HONESTY NOTE: there are no independent observations in this system - the
"measurement" fed to update_with_observation() is the SGP4 output itself, so
the resulting covariance is NOT a physical orbit uncertainty. It is therefore
NOT used for collision probability anywhere: Pc covariance comes from the
documented TLE-age error-growth model in app/core/screening.py
(sigma_source="tle_age_model"). The filter is kept only because snapshot /
state-cache code still serialises it for diagnostics.
"""

from __future__ import annotations

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Process noise (orbital dynamics uncertainty per step)
Q_POSITION = 0.1**2  # km^2 per second (model error)
Q_VELOCITY = 0.01**2  # (km/s)^2 per second

# Measurement noise (typical TLE/SGP4 uncertainty)
R_POSITION = 1.0**2  # km^2 (SGP4 accuracy ~1-5 km)


class KalmanStateECI:
    """Maintains state and covariance for one satellite in ECI."""

    __slots__ = [
        "position_km",
        "velocity_kms",
        "covariance_6x6",
        "last_epoch_utc",
    ]

    def __init__(
        self,
        position_km: list[float] | np.ndarray,
        velocity_kms: list[float] | np.ndarray,
        epoch_utc: str,
        initial_covariance: np.ndarray | None = None,
    ):
        self.position_km = np.asarray(position_km, dtype=float)
        self.velocity_kms = np.asarray(velocity_kms, dtype=float)
        self.last_epoch_utc = epoch_utc

        if initial_covariance is not None:
            self.covariance_6x6 = np.asarray(initial_covariance, dtype=float)
        else:
            # Default: 10 km uncertainty in position, 0.1 km/s in velocity
            self.covariance_6x6 = np.diag([10.0**2, 10.0**2, 10.0**2, 0.1**2, 0.1**2, 0.1**2])

    def to_dict(self) -> dict:
        """Serialize state and covariance."""
        return {
            "position_km": self.position_km.tolist(),
            "velocity_kms": self.velocity_kms.tolist(),
            "covariance_6x6": self.covariance_6x6.tolist(),
            "epoch_utc": self.last_epoch_utc,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> KalmanStateECI:
        """Deserialize from dict."""
        return cls(
            position_km=payload["position_km"],
            velocity_kms=payload["velocity_kms"],
            epoch_utc=payload["epoch_utc"],
            initial_covariance=np.asarray(payload.get("covariance_6x6"), dtype=float),
        )


class KalmanFilter:
    """
    Simple EKF-like Kalman filter for satellite orbit state.
    State: [x, y, z, vx, vy, vz] in ECI coordinates.
    Propagates covariance using linear model + process noise.
    Updates with position observations.
    """

    def __init__(self):
        self.Q = self._build_process_noise()
        self.R = R_POSITION  # scalar for position measurement noise

    def _build_process_noise(self) -> np.ndarray:
        """Construct 6x6 process noise matrix (position + velocity)."""
        Q = np.zeros((6, 6), dtype=float)
        # Position component (km^2)
        Q[0, 0] = Q[1, 1] = Q[2, 2] = Q_POSITION
        # Velocity component ((km/s)^2)
        Q[3, 3] = Q[4, 4] = Q[5, 5] = Q_VELOCITY
        return Q

    def propagate_covariance(
        self,
        state: KalmanStateECI,
        dt_seconds: float,
        accel_magnitude: float = 9.8e-3,  # m/s^2, typical for orbital decay
    ) -> KalmanStateECI:
        """
        Propagate covariance forward in time using a linear model.
        dt_seconds: time step in seconds
        accel_magnitude: acceleration perturbation magnitude (km/s^2)

        Returns an updated KalmanStateECI with propagated covariance.
        """
        if dt_seconds <= 0:
            return state

        # Linear state transition: position += velocity * dt, velocity unchanged
        F = np.eye(6, dtype=float)
        F[0, 3] = dt_seconds  # x += vx*dt
        F[1, 4] = dt_seconds  # y += vy*dt
        F[2, 5] = dt_seconds  # z += vz*dt

        # Propagate covariance: P_new = F @ P @ F^T + Q
        P_prop = F @ state.covariance_6x6 @ F.T
        P_new = P_prop + (self.Q * dt_seconds)

        # Clamp to avoid numerical blow-up
        P_new = np.clip(P_new, 1e-6, 1e6)

        return KalmanStateECI(
            position_km=state.position_km,
            velocity_kms=state.velocity_kms,
            epoch_utc=state.last_epoch_utc,
            initial_covariance=P_new,
        )

    def update_with_observation(
        self,
        state: KalmanStateECI,
        observed_position_km: np.ndarray,
    ) -> KalmanStateECI:
        """
        Update state and covariance using a new position observation (from SGP4 propagation).
        observed_position_km: [x, y, z] in km

        Returns updated KalmanStateECI.
        """
        observed_position_km = np.asarray(observed_position_km, dtype=float)

        # Measurement matrix: we observe only position [x, y, z]
        H = np.zeros((3, 6), dtype=float)
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0

        # State vector as 6-element: [x, y, z, vx, vy, vz]
        x_state = np.concatenate([state.position_km, state.velocity_kms])

        # Innovation (residual): z - H@x
        innovation = observed_position_km - (H @ x_state)

        # Innovation covariance: S = H @ P @ H^T + R*I
        R_matrix = np.eye(3, dtype=float) * self.R
        S = H @ state.covariance_6x6 @ H.T + R_matrix

        # Kalman gain: K = P @ H^T @ S^-1
        try:
            S_inv = np.linalg.inv(S)
            K = state.covariance_6x6 @ H.T @ S_inv
        except np.linalg.LinAlgError:
            logger.warning("Kalman update singularity, skipping")
            return state

        # State update: x_new = x + K @ innovation
        state_update = K @ innovation
        x_new = state.position_km + state_update[:3]
        vx_new = state.velocity_kms + state_update[3:]

        # Covariance update: P_new = (I - K @ H) @ P
        I_KH = np.eye(6, dtype=float) - K @ H
        P_new = I_KH @ state.covariance_6x6

        # Clamp again
        P_new = np.clip(P_new, 1e-6, 1e6)

        return KalmanStateECI(
            position_km=x_new,
            velocity_kms=vx_new,
            epoch_utc=state.last_epoch_utc,
            initial_covariance=P_new,
        )


def estimate_uncertainty_from_covariance(
    covariance_6x6: np.ndarray,
) -> dict[str, float]:
    """
    Extract scalar uncertainty metrics from a 6x6 covariance matrix.
    Returns position std-dev (km) and velocity std-dev (km/s).
    """
    try:
        position_var = float(np.mean(np.diag(covariance_6x6)[:3]))
        velocity_var = float(np.mean(np.diag(covariance_6x6)[3:]))
        position_std = float(np.sqrt(max(position_var, 0.0)))
        velocity_std = float(np.sqrt(max(velocity_var, 0.0)))
    except Exception as e:
        logger.debug(f"Covariance extraction error: {e}")
        position_std = 0.0
        velocity_std = 0.0

    return {
        "position_std_km": position_std,
        "velocity_std_kms": velocity_std,
    }
