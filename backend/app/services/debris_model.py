"""
Debris cloud modeling for post-collision alerts.
Uses a simple isotropic expansion model suitable for real-time risk messaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree


@dataclass
class DebrisCloud:
    id: str
    tca_utc: str
    minutes_to_tca: float
    center_eci_km: dict[str, float]
    radius_km_now: float
    radius_km_at_tca: float
    max_radius_km: float
    fragment_count: int
    affected_satellites: list[dict[str, Any]]
    affected_count: int
    affected_high_risk: int
    radius_timeline: list[dict[str, float]]
    shells: list[dict[str, Any]]


def _build_cpi_lookup(alerts: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for alert in alerts:
        sat1 = alert.get("sat1", {})
        sat2 = alert.get("sat2", {})
        cpi_score = float(alert.get("cpi_score", 0.0))
        severity = alert.get("severity", "none")

        for sat in (sat1, sat2):
            sat_id = sat.get("id")
            if sat_id is None:
                continue
            existing = lookup.get(sat_id)
            if existing is None or cpi_score > float(existing.get("cpi_score", 0.0)):
                lookup[sat_id] = {
                    "cpi_score": round(cpi_score, 2),
                    "severity": severity,
                }
    return lookup


def _parse_utc(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    cleaned = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _radius_at_time(
    seconds_since_collision: float,
    base_radius_km: float,
    expansion_rate_mps: float,
    max_radius_km: float,
) -> float:
    radius_km = base_radius_km + max(0.0, seconds_since_collision) * (expansion_rate_mps / 1000.0)
    return float(min(radius_km, max_radius_km))


def _integrate_radius_rk45(
    seconds: float,
    base_radius_km: float,
    expansion_rate_mps: float,
    max_radius_km: float,
    decay_tau_s: float,
) -> float:
    if seconds <= 0:
        return float(base_radius_km)

    v0_kms = max(0.0, expansion_rate_mps / 1000.0)

    def dynamics(_t: float, y: np.ndarray) -> np.ndarray:
        radius_km, velocity_kms = y
        dv_dt = -velocity_kms / max(decay_tau_s, 1.0)
        dr_dt = velocity_kms
        return np.array([dr_dt, dv_dt], dtype=float)

    y0 = np.array([base_radius_km, v0_kms], dtype=float)

    try:
        solution = solve_ivp(
            dynamics,
            (0.0, float(seconds)),
            y0,
            method="RK45",
            t_eval=[float(seconds)],
            rtol=1e-6,
            atol=1e-8,
        )
        if solution.success and solution.y.shape[1] == 1:
            radius = float(solution.y[0, 0])
        else:
            radius = _radius_at_time(seconds, base_radius_km, expansion_rate_mps, max_radius_km)
    except Exception:
        radius = _radius_at_time(seconds, base_radius_km, expansion_rate_mps, max_radius_km)

    return float(min(radius, max_radius_km))


def _build_radius_timeline(
    checkpoints_minutes: list[float],
    base_radius_km: float,
    expansion_rate_mps: float,
    max_radius_km: float,
    decay_tau_s: float,
) -> list[dict[str, float]]:
    timeline = []
    for minutes in checkpoints_minutes:
        seconds = max(0.0, float(minutes) * 60.0)
        radius = _integrate_radius_rk45(
            seconds,
            base_radius_km,
            expansion_rate_mps,
            max_radius_km,
            decay_tau_s,
        )
        timeline.append({"minutes": float(minutes), "radius_km": round(radius, 3)})
    return timeline


def build_debris_alerts(
    hotspots: list[dict[str, Any]],
    states: list[Any],
    snapshot_timestamp: str | None,
    alerts: list[dict[str, Any]] | None = None,
    cpi_threshold: float = 7.0,
    max_tca_minutes: float = 180.0,
    base_radius_km: float = 1.5,
    expansion_rate_mps: float = 80.0,
    max_radius_km: float = 1500.0,
    fragment_count: int = 420,
    integration_hours: float = 24.0,
    decay_tau_s: float = 5400.0,
) -> list[dict[str, Any]]:
    """Create debris cloud alerts from high-risk hotspots."""

    if not hotspots or not states:
        return []

    now = _parse_utc(snapshot_timestamp, datetime.now(timezone.utc))
    cpi_lookup = _build_cpi_lookup(alerts or [])

    # Build a KD-tree for current satellite positions to find affected neighbors.
    positions = []
    sat_meta = []
    for state in states:
        if getattr(state, "error_code", 0) != 0:
            continue
        positions.append([float(state.x), float(state.y), float(state.z)])
        sat_meta.append({"norad_id": state.norad_id, "name": state.name})

    if not positions:
        return []

    tree = cKDTree(np.array(positions, dtype=float))
    cloud_alerts: list[dict[str, Any]] = []

    for idx, hotspot in enumerate(hotspots[:8]):
        cpi_score = float(hotspot.get("cpi_score", 0.0))
        tca_minutes = hotspot.get("tca_minutes")
        if cpi_score < cpi_threshold:
            continue
        if tca_minutes is not None and float(tca_minutes) > max_tca_minutes:
            continue

        position = hotspot.get("position") or {}
        center = {
            "x": float(position.get("x", 0.0)),
            "y": float(position.get("y", 0.0)),
            "z": float(position.get("z", 0.0)),
        }

        tca_utc = hotspot.get("tca_utc")
        tca_time = _parse_utc(tca_utc, now)
        minutes_to_tca = max(0.0, (tca_time - now).total_seconds() / 60.0)

        radius_now = _integrate_radius_rk45(0.0, base_radius_km, expansion_rate_mps, max_radius_km, decay_tau_s)
        radius_at_tca = _integrate_radius_rk45(
            minutes_to_tca * 60.0,
            base_radius_km,
            expansion_rate_mps,
            max_radius_km,
            decay_tau_s,
        )

        tca_minutes_safe = max(0.0, float(minutes_to_tca))
        mid_minutes = tca_minutes_safe * 0.5
        timeline_minutes = [0.0, mid_minutes, tca_minutes_safe]
        radius_timeline = _build_radius_timeline(
            timeline_minutes,
            base_radius_km,
            expansion_rate_mps,
            max_radius_km,
            decay_tau_s,
        )

        shells = [
            {"label": "now", "minutes": radius_timeline[0]["minutes"], "radius_km": radius_timeline[0]["radius_km"]},
            {"label": "mid", "minutes": radius_timeline[1]["minutes"], "radius_km": radius_timeline[1]["radius_km"]},
            {"label": "tca", "minutes": radius_timeline[2]["minutes"], "radius_km": radius_timeline[2]["radius_km"]},
        ]

        affected_indices = tree.query_ball_point([center["x"], center["y"], center["z"]], r=radius_at_tca)
        affected = []
        high_risk_count = 0
        for idx_hit in affected_indices:
            meta = sat_meta[idx_hit]
            cpi_info = cpi_lookup.get(meta["norad_id"], {})
            cpi_score = float(cpi_info.get("cpi_score", 0.0))
            risk_band = "high" if cpi_score >= 7.0 else "medium" if cpi_score >= 5.0 else "low"
            recommended_action = "fallback" if risk_band == "high" else "monitor"
            if risk_band == "high":
                high_risk_count += 1
            affected.append(
                {
                    **meta,
                    "cpi_score": round(cpi_score, 2),
                    "severity": cpi_info.get("severity", "none"),
                    "risk_band": risk_band,
                    "recommended_action": recommended_action,
                }
            )

        cloud = DebrisCloud(
            id=f"debris-{idx + 1}",
            tca_utc=tca_time.isoformat(),
            minutes_to_tca=round(minutes_to_tca, 2),
            center_eci_km=center,
            radius_km_now=round(radius_now, 3),
            radius_km_at_tca=round(radius_at_tca, 3),
            max_radius_km=max_radius_km,
            fragment_count=fragment_count,
            affected_satellites=affected,
            affected_count=len(affected),
            affected_high_risk=high_risk_count,
            radius_timeline=radius_timeline,
            shells=shells,
        )

        cloud_alerts.append(cloud.__dict__)

    return cloud_alerts
