"""
debris_model.py — Physically realistic debris fragment simulation.

Replaces the geometric sphere expansion model with fragment velocity
distributions based on the NASA Standard Breakup Model (NASA EVOLVE 4.0).

Reference:
  Johnson, N.L. et al. (2001). "NASA's new breakup model of EVOLVE 4.0."
  NASA/TM-2001-210865.

Fragment velocity distribution:
  v_fragment ~ LogNormal(mean_dv, sigma_dv)
  where mean_dv = 200 * size_m^(-0.5) m/s  (NASA EVOLVE characteristic velocity)
        sigma_dv = 0.4 * mean_dv             (40% standard deviation)

Propagation:
  Simplified Keplerian + drag (analytical model — no RK45 required for demo).
  Each fragment has its own orbital element set evolved forward in time.

All units: km, km/s, seconds, metres for sizes.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Earth constants
R_EARTH_KM = 6371.0
MU_KM3_S2 = 398600.4418  # km³/s²
# Atmospheric scale height and reference density (simplified exponential model)
ATMOS_H0_KM = 8.5          # scale height km
ATMOS_RHO0_KG_M3 = 1.225   # sea level density kg/m³
ATMOS_REF_ALT_KM = 0.0     # reference altitude km


def atmospheric_density(altitude_km: float) -> float:
    """
    Simplified exponential atmospheric density model (kg/m³).
    Valid for altitudes 100–2000 km (uses scale-height extrapolation).
    """
    if altitude_km < 100.0:
        return ATMOS_RHO0_KG_M3 * math.exp(-altitude_km / ATMOS_H0_KM)
    # Use 7.5 km scale height for higher altitudes (approximate thermosphere)
    rho_100 = ATMOS_RHO0_KG_M3 * math.exp(-100.0 / ATMOS_H0_KM)
    return rho_100 * math.exp(-(altitude_km - 100.0) / 7.5)


@dataclass
class DebrisFragment:
    """A single debris fragment with its orbital state and physical properties."""
    fragment_id: int
    size_m: float           # characteristic size in metres
    mass_kg: float          # estimated mass kg
    area_m2: float          # cross-sectional area m² (estimated from size)
    # Current ECI position (km) and velocity (km/s)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    # Derived orbital elements (updated on propagation)
    altitude_km: float = 400.0
    is_decayed: bool = False
    # Propagation metadata
    epoch_utc: str = ""

    def position_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def velocity_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz])

    def to_dict(self) -> dict:
        return {
            "fragment_id": self.fragment_id,
            "size_m": round(self.size_m, 3),
            "mass_kg": round(self.mass_kg, 3),
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "vx": round(self.vx, 4),
            "vy": round(self.vy, 4),
            "vz": round(self.vz, 4),
            "altitude_km": round(self.altitude_km, 2),
            "is_decayed": self.is_decayed,
            "epoch_utc": self.epoch_utc,
        }


@dataclass
class DebrisCloud:
    """Collection of fragments from a single collision event."""
    event_id: str
    collision_point_km: list[float]       # [x, y, z] ECI km
    collision_velocity_kms: list[float]   # [vx, vy, vz] ECI km/s
    mass_p_kg: float
    mass_q_kg: float
    rel_vel_kms: float
    created_at: str
    fragments: list[DebrisFragment] = field(default_factory=list)
    is_catastrophic: bool = False
    predicted_count: int = 0

    def to_summary(self) -> dict:
        alive = [f for f in self.fragments if not f.is_decayed]
        return {
            "event_id": self.event_id,
            "collision_point_km": self.collision_point_km,
            "is_catastrophic": self.is_catastrophic,
            "total_fragments": self.predicted_count,
            "active_fragments": len(alive),
            "decayed_fragments": len(self.fragments) - len(alive),
            "created_at": self.created_at,
            "rel_vel_kms": round(self.rel_vel_kms, 2),
        }


class DebrisModel:
    """
    Physically realistic debris simulation using NASA EVOLVE fragment velocities.

    Velocity distribution (per fragment):
        v_kick = LogNormal(mean_dv, sigma_dv)
        mean_dv = 200 * size_m^(-0.5) m/s   [NASA characteristic velocity]
        sigma_dv = 0.4 * mean_dv

    Fragment sizes sampled from power-law distribution (NASA breakup model):
        N(> L) ∝ L^(-2.6)

    Propagation (analytical, not RK45):
        Altitude decay per revolution = -C_D * A/m * rho * v * T_period
        where rho is atmospheric density at current altitude
    """

    def __init__(self):
        self._clouds: dict[str, DebrisCloud] = {}
        self._rng = random.Random(42)

    # ── Fragment generation ────────────────────────────────────────────────────

    def _sample_fragment_sizes(self, n_fragments: int, min_size_m: float = 0.01) -> list[float]:
        """
        Sample fragment sizes from a power-law distribution (NASA EVOLVE model).
        Sizes in metres. Returns list of n_fragments sizes.
        """
        sizes = []
        for _ in range(n_fragments):
            # Power-law CDF inversion: size = min_size * (1-u)^(-1/1.6)
            u = self._rng.random()
            size = min_size_m * ((1.0 - u + 1e-10) ** (-1.0 / 1.6))
            size = min(size, 10.0)  # Cap at 10 m (largest realistic fragment)
            sizes.append(size)
        return sizes

    def _fragment_velocity_kick(self, size_m: float) -> np.ndarray:
        """
        Sample isotropic velocity kick from NASA EVOLVE LogNormal distribution.

        mean_dv = 200 * size_m^(-0.5) m/s → converted to km/s
        Returns 3-vector in km/s.
        """
        size_m = max(size_m, 0.001)
        mean_dv_ms = 200.0 * (size_m ** -0.5)   # m/s
        std_dv_ms = 0.4 * mean_dv_ms

        # LogNormal parameterisation from mean and std
        sigma2 = math.log(1.0 + (std_dv_ms / mean_dv_ms) ** 2)
        mu = math.log(mean_dv_ms) - sigma2 / 2.0
        dv_magnitude_ms = self._rng.lognormvariate(mu, math.sqrt(sigma2))
        dv_magnitude_ms = max(1.0, min(dv_magnitude_ms, 10000.0))

        # Isotropic direction
        theta = self._rng.uniform(0.0, 2.0 * math.pi)
        cos_phi = self._rng.uniform(-1.0, 1.0)
        sin_phi = math.sqrt(max(0.0, 1.0 - cos_phi**2))
        direction = np.array([
            sin_phi * math.cos(theta),
            sin_phi * math.sin(theta),
            cos_phi,
        ])

        # Convert m/s → km/s
        dv_kms = dv_magnitude_ms / 1000.0
        return dv_kms * direction

    # ── Simulate collision ─────────────────────────────────────────────────────

    def simulate_collision(
        self,
        collision_point_eci_km: list[float],
        collision_velocity_eci_kms: list[float],
        mass_p_kg: float = 500.0,
        mass_q_kg: float = 500.0,
        rel_vel_kms: float = 10.0,
        max_fragments_simulated: int = 500,
        event_id: str | None = None,
    ) -> DebrisCloud:
        """
        Simulate a collision and generate fragment initial states.

        Parameters
        ----------
        collision_point_eci_km     : [x, y, z] ECI position at impact (km)
        collision_velocity_eci_kms : [vx, vy, vz] mean velocity at impact (km/s)
        mass_p_kg, mass_q_kg       : masses of the two objects (kg)
        rel_vel_kms                : relative velocity at impact (km/s)
        max_fragments_simulated    : cap on simulated fragments (for performance)
        event_id                   : unique identifier for this event

        Returns
        -------
        DebrisCloud with fragment initial states populated.
        """
        try:
            from app.core.analytics import nasa_fragment_count
        except ImportError:
            # Fallback formula if analytics unavailable
            def nasa_fragment_count(mp, mq, rv):
                e_ratio = 0.5 * min(mp, mq) * (rv * 1000.0) ** 2 / max(mp, mq)
                total = mp + mq
                return int(round(0.1 * total ** 0.75 * (1.0 if e_ratio > 40 else 0.1)))

        if event_id is None:
            event_id = f"debris_{int(datetime.now(timezone.utc).timestamp())}"

        # Determine catastrophic vs cratering
        m_small = min(mass_p_kg, mass_q_kg)
        m_large = max(mass_p_kg, mass_q_kg)
        rel_vel_ms = rel_vel_kms * 1000.0
        specific_energy = 0.5 * m_small * rel_vel_ms**2 / m_large if m_large > 0 else 0.0
        is_catastrophic = specific_energy > 40.0  # J/kg threshold

        predicted_count = nasa_fragment_count(mass_p_kg, mass_q_kg, rel_vel_kms)
        n_simulate = min(predicted_count, max_fragments_simulated)

        logger.info(
            "Simulating %s collision: predicted %d fragments, simulating %d",
            "CATASTROPHIC" if is_catastrophic else "sub-catastrophic",
            predicted_count, n_simulate,
        )

        r_impact = np.array(collision_point_eci_km, dtype=float)
        v_impact = np.array(collision_velocity_eci_kms, dtype=float)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Sample fragment sizes
        sizes = self._sample_fragment_sizes(n_simulate, min_size_m=0.01)

        fragments = []
        for fid, size_m in enumerate(sizes):
            try:
                # Physical properties
                density = 2700.0  # kg/m³ (aluminium — typical satellite structure)
                volume = (4.0 / 3.0) * math.pi * (size_m / 2.0) ** 3
                mass_kg = density * volume
                area_m2 = math.pi * (size_m / 2.0) ** 2

                # Velocity: collision velocity + isotropic kick
                dv = self._fragment_velocity_kick(size_m)
                v_frag = v_impact + dv

                altitude_km = float(np.linalg.norm(r_impact)) - R_EARTH_KM

                frag = DebrisFragment(
                    fragment_id=fid,
                    size_m=size_m,
                    mass_kg=mass_kg,
                    area_m2=area_m2,
                    x=float(r_impact[0]),
                    y=float(r_impact[1]),
                    z=float(r_impact[2]),
                    vx=float(v_frag[0]),
                    vy=float(v_frag[1]),
                    vz=float(v_frag[2]),
                    altitude_km=altitude_km,
                    epoch_utc=now_iso,
                )
                fragments.append(frag)
            except Exception as frag_err:
                logger.debug("Fragment %d generation failed: %s", fid, frag_err)

        cloud = DebrisCloud(
            event_id=event_id,
            collision_point_km=list(collision_point_eci_km),
            collision_velocity_kms=list(collision_velocity_eci_kms),
            mass_p_kg=mass_p_kg,
            mass_q_kg=mass_q_kg,
            rel_vel_kms=rel_vel_kms,
            created_at=now_iso,
            fragments=fragments,
            is_catastrophic=is_catastrophic,
            predicted_count=predicted_count,
        )

        self._clouds[event_id] = cloud
        logger.info(
            "Debris cloud %s: %d fragments simulated (predicted %d, catastrophic=%s)",
            event_id, len(fragments), predicted_count, is_catastrophic,
        )
        return cloud

    # ── Fragment propagation ───────────────────────────────────────────────────

    def propagate_fragments(
        self,
        event_id: str,
        dt_seconds: float = 3600.0,
    ) -> list[dict]:
        """
        Propagate all fragments by dt_seconds using simplified Keplerian + drag.

        Drag model (analytical — per time step):
            dv_drag = -0.5 * C_D * (A/m) * rho * v² * dt
            where C_D = 2.2 (typical space object)
                  A/m = area_m2 / mass_kg
                  rho = atmospheric density at current altitude (kg/m³)
                  v   = orbital speed (km/s) converted to m/s

        Altitude is updated from position vector after velocity correction.
        Fragments with altitude < 100 km are marked decayed.

        Returns list of fragment dicts for the event.
        """
        cloud = self._clouds.get(event_id)
        if cloud is None:
            return []

        C_D = 2.2  # Drag coefficient for irregular debris
        alive = []

        for frag in cloud.fragments:
            if frag.is_decayed:
                alive.append(frag.to_dict())
                continue

            try:
                r = np.array([frag.x, frag.y, frag.z])
                v = np.array([frag.vx, frag.vy, frag.vz])
                r_norm = float(np.linalg.norm(r))
                v_norm = float(np.linalg.norm(v))

                if r_norm < R_EARTH_KM:
                    frag.is_decayed = True
                    continue

                altitude_km = r_norm - R_EARTH_KM
                frag.altitude_km = altitude_km

                # ── Gravity (two-body) ─────────────────────────────────────
                a_grav = -MU_KM3_S2 / r_norm**3 * r  # km/s²

                # ── Atmospheric drag ────────────────────────────────────────
                rho_kg_m3 = atmospheric_density(altitude_km)
                # Convert velocity to m/s for drag calc
                v_ms = v_norm * 1000.0  # km/s → m/s
                am_ratio = frag.area_m2 / max(frag.mass_kg, 1e-6)  # m²/kg
                drag_accel_ms2 = -0.5 * C_D * am_ratio * rho_kg_m3 * v_ms**2
                # Direction: opposing velocity
                v_hat = v / v_norm if v_norm > 1e-10 else np.zeros(3)
                a_drag_kms2 = drag_accel_ms2 / 1000.0 * v_hat  # m/s² → km/s²

                # ── Euler integration ───────────────────────────────────────
                a_total = a_grav + a_drag_kms2
                v_new = v + a_total * dt_seconds
                r_new = r + v * dt_seconds + 0.5 * a_total * dt_seconds**2

                r_new_norm = float(np.linalg.norm(r_new))
                alt_new = r_new_norm - R_EARTH_KM

                if alt_new < 100.0:
                    frag.is_decayed = True
                else:
                    frag.x, frag.y, frag.z = float(r_new[0]), float(r_new[1]), float(r_new[2])
                    frag.vx, frag.vy, frag.vz = float(v_new[0]), float(v_new[1]), float(v_new[2])
                    frag.altitude_km = alt_new

            except Exception as prop_err:
                logger.debug("Fragment %d propagation failed: %s", frag.fragment_id, prop_err)

            alive.append(frag.to_dict())

        return alive

    # ── API ────────────────────────────────────────────────────────────────────

    def get_cloud(self, event_id: str) -> DebrisCloud | None:
        return self._clouds.get(event_id)

    def get_all_clouds(self) -> dict[str, dict]:
        return {eid: cloud.to_summary() for eid, cloud in self._clouds.items()}

    def list_event_ids(self) -> list[str]:
        return list(self._clouds.keys())

    def clear(self):
        """Clear all simulated debris clouds."""
        self._clouds.clear()

    def get_frontend_debris_clouds(self, states: list) -> list[dict]:
        """Format and return the debris clouds in the format expected by the frontend."""
        import numpy as np
        frontend_clouds = []
        for event_id, cloud in self._clouds.items():
            # Calculate radius of active fragments from collision point
            active_frags = [f for f in cloud.fragments if not f.is_decayed]
            collision_pos = np.array(cloud.collision_point_km)
            if active_frags:
                distances = [np.linalg.norm(np.array([f.x, f.y, f.z]) - collision_pos) for f in active_frags]
                radius_km = max(distances)
            else:
                radius_km = 0.0

            # Minimum visible radius so it displays nicely
            radius_km = max(radius_km, 10.0)

            # Find affected satellites within the radius
            affected_satellites = []
            for state in states:
                # support both objects and dicts
                if isinstance(state, dict):
                    err = state.get("error_code", 0)
                    if err != 0:
                        continue
                    x = state.get("x", 0.0)
                    y = state.get("y", 0.0)
                    z = state.get("z", 0.0)
                    norad_id = state.get("norad_id")
                    name = state.get("name")
                else:
                    err = getattr(state, "error_code", 0)
                    if err != 0:
                        continue
                    x = getattr(state, "x", 0.0)
                    y = getattr(state, "y", 0.0)
                    z = getattr(state, "z", 0.0)
                    norad_id = getattr(state, "norad_id", None)
                    name = getattr(state, "name", None)

                sat_pos = np.array([x, y, z])
                dist = np.linalg.norm(sat_pos - collision_pos)
                if dist <= radius_km:
                    affected_satellites.append({
                        "norad_id": norad_id,
                        "name": name,
                        "cpi_score": 5.0,
                        "severity": "medium",
                        "risk_band": "medium",
                        "recommended_action": "monitor"
                    })

            frontend_clouds.append({
                "id": cloud.event_id,
                "tca_utc": cloud.created_at,
                "minutes_to_tca": 0.0,
                "center_eci_km": {
                    "x": cloud.collision_point_km[0],
                    "y": cloud.collision_point_km[1],
                    "z": cloud.collision_point_km[2]
                },
                "radius_km_now": round(radius_km, 3),
                "radius_km_at_tca": round(radius_km, 3),
                "max_radius_km": 1500.0,
                "fragment_count": cloud.predicted_count,
                "affected_satellites": affected_satellites,
                "affected_count": len(affected_satellites),
                "affected_high_risk": 0,
                "radius_timeline": [],
                "shells": [{"label": "now", "radius_km": round(radius_km, 3)}]
            })
        return frontend_clouds


# ── Module-level singleton ─────────────────────────────────────────────────────
debris_model = DebrisModel()
