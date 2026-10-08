"""
Test mode router for manual conjunction validation.
"""

from __future__ import annotations

import asyncio

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from types import SimpleNamespace
from threading import RLock
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.core.sgp4_propagator import SGP4Propagator
from app.core import sim_clock
from app.core.state_cache import set_latest_alerts
from app.services.conjunction_solver import (
    CONJUNCTION_REPORT_DISTANCE_M,
    parse_eci_state,
    run_to_tca,
)
from app.services.conjunction_predictor import build_collision_prediction, classify_approach

router = APIRouter(prefix="/api/test")


class SatelliteInput(BaseModel):
    name: Optional[str] = None
    norad_id: Optional[str] = None
    use_real_tle: bool = True
    position_eci_km: Optional[list[float]] = None
    velocity_eci_kms: Optional[list[float]] = None
    epoch_utc: Optional[str] = None

    @model_validator(mode="after")
    def validate_override(self):
        if self.position_eci_km and self.velocity_eci_kms and self.use_real_tle:
            self.use_real_tle = False

        if not self.use_real_tle:
            if not self.position_eci_km or not self.velocity_eci_kms:
                raise ValueError("Override requires position_eci_km and velocity_eci_kms")
            if len(self.position_eci_km) != 3 or len(self.velocity_eci_kms) != 3:
                raise ValueError("Override vectors must have 3 elements")

        return self


class TestSetupRequest(BaseModel):
    satellite_a: SatelliteInput
    satellite_b: SatelliteInput
    epoch_utc: Optional[str] = None


class RunToTcaRequest(BaseModel):
    target_utc: str
    step_seconds: float = Field(default=10.0, gt=0.0)
    max_steps: int = Field(default=3600, ge=1)


@dataclass
class TestSatellite:
    name: str
    norad_id: str
    position_eci_km: list[float]
    velocity_eci_kms: list[float]
    epoch_utc: datetime
    use_real_tle: bool


@dataclass
class TestSession:
    active: bool = False
    satellite_a: Optional[TestSatellite] = None
    satellite_b: Optional[TestSatellite] = None


_session = TestSession()
_session_lock = RLock()
_propagator: Optional[SGP4Propagator] = None


def init_test_mode(propagator: SGP4Propagator):
    global _propagator
    _propagator = propagator


def _parse_epoch(value: Optional[str], fallback: Optional[datetime] = None) -> datetime:
    if not value:
        return fallback or sim_clock.simulation_now()

    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"

    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_norad_id(value: Optional[str]) -> int:
    if value is None:
        raise HTTPException(status_code=400, detail="NORAD ID is required")

    try:
        return int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="NORAD ID must be numeric")


def _load_state_from_propagator(norad_id: int, epoch: datetime):
    if _propagator is None:
        raise HTTPException(status_code=503, detail="Propagator not initialized")

    state = _propagator.propagate_one(norad_id, epoch)
    if state is None:
        raise HTTPException(status_code=404, detail="Satellite not found")
    if state.error_code != 0:
        raise HTTPException(status_code=500, detail="Propagation failed")
    return state


def _resolve_satellite_state(satellite: TestSatellite, now: datetime):
    if satellite.use_real_tle:
        norad_id = _normalize_norad_id(satellite.norad_id)
        state = _load_state_from_propagator(norad_id, now)
        position = [state.x, state.y, state.z]
        velocity = [state.vx, state.vy, state.vz]
        return position, velocity

    dt = (now - satellite.epoch_utc).total_seconds()
    position = [
        satellite.position_eci_km[0] + satellite.velocity_eci_kms[0] * dt,
        satellite.position_eci_km[1] + satellite.velocity_eci_kms[1] * dt,
        satellite.position_eci_km[2] + satellite.velocity_eci_kms[2] * dt,
    ]
    velocity = list(satellite.velocity_eci_kms)
    return position, velocity


def _build_satellite(input_data: SatelliteInput, epoch: datetime) -> TestSatellite:
    if input_data.use_real_tle:
        epoch = _parse_epoch(input_data.epoch_utc, epoch)
        norad_id = _normalize_norad_id(input_data.norad_id)
        state = _load_state_from_propagator(norad_id, epoch)
        name = input_data.name or state.name or "UNKNOWN"
        return TestSatellite(
            name=name,
            norad_id=str(state.norad_id),
            position_eci_km=[float(state.x), float(state.y), float(state.z)],
            velocity_eci_kms=[float(state.vx), float(state.vy), float(state.vz)],
            epoch_utc=epoch,
            use_real_tle=True,
        )

    name = input_data.name or "UNKNOWN"
    norad_id = str(input_data.norad_id or "0")
    return TestSatellite(
        name=name,
        norad_id=norad_id,
        position_eci_km=[float(x) for x in input_data.position_eci_km or []],
        velocity_eci_kms=[float(x) for x in input_data.velocity_eci_kms or []],
        epoch_utc=epoch,
        use_real_tle=False,
    )


def _to_test_state(satellite: TestSatellite) -> SimpleNamespace:
    return SimpleNamespace(
        error_code=0,
        norad_id=_normalize_norad_id(satellite.norad_id),
        name=satellite.name,
        x=float(satellite.position_eci_km[0]),
        y=float(satellite.position_eci_km[1]),
        z=float(satellite.position_eci_km[2]),
        vx=float(satellite.velocity_eci_kms[0]),
        vy=float(satellite.velocity_eci_kms[1]),
        vz=float(satellite.velocity_eci_kms[2]),
        epoch_utc=satellite.epoch_utc,
    )


def _publish_collision_alerts(
    sat_a: TestSatellite,
    sat_b: TestSatellite,
    result,
    collision_confirmed: bool,
) -> dict:
    return build_collision_prediction(
        _propagator,
        sat_a,
        sat_b,
        result,
        collision_confirmed=collision_confirmed,
    )



@router.post("/setup")
async def setup_test(req: TestSetupRequest):
    epoch = _parse_epoch(req.epoch_utc)
    epoch_a = _parse_epoch(req.satellite_a.epoch_utc, epoch)
    epoch_b = _parse_epoch(req.satellite_b.epoch_utc, epoch)

    if abs((epoch_a - epoch_b).total_seconds()) > 1.0:
        # Force both satellites onto the same epoch for deterministic run-to-tca
        epoch_b = epoch_a

    sat_a = _build_satellite(req.satellite_a, epoch_a)
    sat_b = _build_satellite(req.satellite_b, epoch_b)

    with _session_lock:
        _session.active = True
        _session.satellite_a = sat_a
        _session.satellite_b = sat_b

    return {
        "success": True,
        "satellite_a_id": sat_a.norad_id,
        "satellite_b_id": sat_b.norad_id,
    }


@router.post("/run-to-tca")
async def run_to_tca_endpoint(req: RunToTcaRequest):
    with _session_lock:
        if not _session.active or _session.satellite_a is None or _session.satellite_b is None:
            raise HTTPException(status_code=400, detail="Test mode is not initialized")

        sat_a = _session.satellite_a
        sat_b = _session.satellite_b

    target_utc = _parse_epoch(req.target_utc, sat_a.epoch_utc)

    state_a = parse_eci_state(sat_a.position_eci_km, sat_a.velocity_eci_kms, sat_a.epoch_utc)
    state_b = parse_eci_state(sat_b.position_eci_km, sat_b.velocity_eci_kms, sat_b.epoch_utc)

    result = await asyncio.to_thread(run_to_tca, 
        state_a=state_a,
        state_b=state_b,
        target_utc=target_utc,
        step_seconds=req.step_seconds,
        max_steps=req.max_steps,
        use_j2=True,
    )

    alert_payload = {
        "timestamp": result.tca_utc,
        "alerts": [],
        "count": 0,
        "hotspots": [],
        "graph": {},
        "cascade_plan": [],
        "cascade_depth": 0,
        "total_delta_v_ms": 0.0,
        "agencies_involved": [],
        "seed_satellites": [],
        "cpi_threshold": 7.0,
        "node_probabilities": {},
        "debris_clouds": [],
        "collision_confirmed": False,
    }

    if result.miss_distance_m <= CONJUNCTION_REPORT_DISTANCE_M:
        alert_payload = _publish_collision_alerts(
            sat_a, sat_b, result, result.collision_detected
        )
    else:
        set_latest_alerts(alert_payload)

    return {
        "tca_utc": result.tca_utc,
        "miss_distance_m": result.miss_distance_m,
        "relative_velocity_kms": result.relative_velocity_kms,
        "position_a_eci": result.position_a_eci,
        "position_b_eci": result.position_b_eci,
        "collision_detected": result.collision_detected,
        "classification": classify_approach(result.miss_distance_m),
        "trajectory_a": result.trajectory_a,
        "trajectory_b": result.trajectory_b,
        "alerts": alert_payload.get("alerts", []),
        "hotspots": alert_payload.get("hotspots", []),
        "cascade_plan": alert_payload.get("cascade_plan", []),
        "graph": alert_payload.get("graph", {}),
        "cascade_depth": alert_payload.get("cascade_depth", 0),
        "total_delta_v_ms": alert_payload.get("total_delta_v_ms", 0.0),
        "agencies_involved": alert_payload.get("agencies_involved", []),
        "seed_satellites": alert_payload.get("seed_satellites", []),
        "cpi_threshold": alert_payload.get("cpi_threshold", 7.0),
        "node_probabilities": alert_payload.get("node_probabilities", {}),
        "debris_clouds": alert_payload.get("debris_clouds", []),
    }


@router.get("/separation")
async def get_separation():
    with _session_lock:
        if not _session.active or _session.satellite_a is None or _session.satellite_b is None:
            raise HTTPException(status_code=400, detail="Test mode is not initialized")

        sat_a = _session.satellite_a
        sat_b = _session.satellite_b

    now = sim_clock.simulation_now()
    pos_a, vel_a = _resolve_satellite_state(sat_a, now)
    pos_b, vel_b = _resolve_satellite_state(sat_b, now)

    rx = pos_a[0] - pos_b[0]
    ry = pos_a[1] - pos_b[1]
    rz = pos_a[2] - pos_b[2]
    separation_km = math.sqrt(rx * rx + ry * ry + rz * rz)

    rvx = vel_a[0] - vel_b[0]
    rvy = vel_a[1] - vel_b[1]
    rvz = vel_a[2] - vel_b[2]
    rel_speed_sq = rvx * rvx + rvy * rvy + rvz * rvz

    if separation_km > 0:
        closing_rate_kms = (rx * rvx + ry * rvy + rz * rvz) / separation_km
    else:
        closing_rate_kms = 0.0

    if rel_speed_sq > 0:
        time_to_tca_s = -((rx * rvx + ry * rvy + rz * rvz) / rel_speed_sq)
    else:
        time_to_tca_s = 0.0

    return {
        "separation_km": round(separation_km, 6),
        "closing_rate_kms": round(closing_rate_kms, 6),
        "time_to_tca_s": round(max(time_to_tca_s, 0.0), 3),
        "epoch_utc": now.isoformat(),
    }
