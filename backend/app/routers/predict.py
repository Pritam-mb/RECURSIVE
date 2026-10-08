"""
Manual conjunction prediction router.

POST /api/predict/conjunction — given two object references (NORAD ID, geodetic
sub-satellite point, or explicit ECI override) plus a forecast window, returns
the predicted Time of Closest Approach, miss distance, Foster Pc, cascade plan,
debris clouds, the satellites present at the collision time, and the affected
ground region.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.sgp4_propagator import SGP4Propagator
from app.services.conjunction_predictor import predict_conjunction

router = APIRouter(prefix="/api/predict")

_propagator: Optional[SGP4Propagator] = None


def init_predict_router(propagator: SGP4Propagator):
    global _propagator
    _propagator = propagator


class SatelliteRef(BaseModel):
    name: Optional[str] = None
    norad_id: Optional[int] = None
    inclination_deg: Optional[float] = None
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    altitude_km: Optional[float] = None
    position_eci_km: Optional[list[float]] = None
    velocity_eci_kms: Optional[list[float]] = None


class PredictConjunctionRequest(BaseModel):
    satellite_a: SatelliteRef
    satellite_b: SatelliteRef
    target_utc: Optional[str] = None
    forecast_hours: float = Field(default=24.0, gt=0.0, le=168.0)
    step_seconds: float = Field(default=60.0, ge=1.0, le=3600.0)
    max_steps: int = Field(default=2400, ge=10, le=20000)
    epoch_utc: Optional[str] = None


@router.post("/conjunction")
async def predict_conjunction_endpoint(req: PredictConjunctionRequest):
    """Predict a future close approach between two user-specified objects."""
    if _propagator is None:
        raise HTTPException(status_code=503, detail="Propagator not initialized")

    try:
        result = await asyncio.to_thread(
            predict_conjunction,
            _propagator,
            req.satellite_a.model_dump(exclude_none=True),
            req.satellite_b.model_dump(exclude_none=True),
            step_seconds=req.step_seconds,
            max_steps=req.max_steps,
            target_utc=req.target_utc,
            forecast_hours=req.forecast_hours,
            epoch_utc=req.epoch_utc,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return result