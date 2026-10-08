"""
Forecast debris clouds for predicted (not yet happened) high-risk conjunctions.

For each hotspot above the CPI threshold whose two objects are present in
``states`` (states propagated to the hotspot TCA by the caller), the NASA
Standard Breakup Model (app.core.breakup) is run on the actual pair state:
masses from SATCAT / documented defaults, relative velocity from the states.
The expected SBM fragment count is reported, and a sample of fragments is
propagated (RK4, two-body + J2 + drag) to measure the cloud's 90th-percentile
radius at fixed times after the hypothetical breakup.  Nothing here is a
constant presented as a result; every cloud is labelled ``kind: "forecast"``.

Fragment clouds that exist because a breakup was simulated come from
``app.core.debris_model`` (re-exported below for the contract).
"""

from __future__ import annotations

import zlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from app.core import breakup as sbm
from app.core.debris_model import (  # noqa: F401  (contract re-exports)
    compute_debris_alerts,
    debris_model,
    get_frontend_debris_clouds,
    object_physical_properties,
    simulate_collision_from_pair,
)

FORECAST_SAMPLE_FRAGMENTS = 300
FORECAST_TIMELINE_MIN = (10.0, 45.0, 90.0)   # minutes after the hypothetical breakup
MAX_FORECAST_CLOUDS = 8


def _build_cpi_lookup(alerts: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for alert in alerts:
        cpi_score = float(alert.get("cpi_score", 0.0) or 0.0)
        severity = alert.get("severity", "none")
        for sat in (alert.get("sat1", {}), alert.get("sat2", {})):
            sat_id = sat.get("id")
            if sat_id is None:
                continue
            existing = lookup.get(sat_id)
            if existing is None or cpi_score > float(existing.get("cpi_score", 0.0)):
                lookup[sat_id] = {"cpi_score": round(cpi_score, 2), "severity": severity}
    return lookup


def _parse_utc(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def forecast_breakup_cloud(r_a, v_a, id_a, name_a, r_b, v_b, id_b, name_b, *, seed: int,
                           sample: int = FORECAST_SAMPLE_FRAGMENTS,
                           timeline_min=FORECAST_TIMELINE_MIN) -> dict[str, Any]:
    """Run the SBM on a predicted pair state and measure the cloud spread."""
    pa = object_physical_properties(id_a, name_a)
    pb = object_physical_properties(id_b, name_b)
    res = sbm.simulate_breakup(r_a, v_a, pa["mass_kg"], r_b, v_b, pb["mass_kg"], seed=seed,
                               rocket_body_a=pa["object_type"] == "R/B",
                               rocket_body_b=pb["object_type"] == "R/B",
                               max_fragments=sample)
    bc = sbm.DEFAULT_CD * res.am_m2_kg
    r, v = res.r_km, res.v_kms
    t_prev = 0.0
    timeline = []
    for minutes in timeline_min:
        r, v = sbm.propagate(r, v, bc, (minutes - t_prev) * 60.0, max_step_s=30.0)
        t_prev = minutes
        rn = np.linalg.norm(r, axis=1)
        ok = np.isfinite(rn) & (rn - sbm.R_EARTH_KM > sbm.REENTRY_ALT_KM)
        if ok.sum() >= 3:
            c = r[ok].mean(axis=0)
            d = np.linalg.norm(r[ok] - c, axis=1)
            p90 = float(np.percentile(d, 90))
            p50 = float(np.percentile(d, 50))
        else:
            p90 = p50 = 0.0
        timeline.append({"minutes": float(minutes), "radius_km": round(p90, 3), "radius_p50_km": round(p50, 3)})
    return {"result": res, "parents": [pa, pb], "timeline": timeline}


def build_debris_alerts(
    hotspots: list[dict[str, Any]],
    states: list[Any],
    snapshot_timestamp: str | None,
    alerts: list[dict[str, Any]] | None = None,
    cpi_threshold: float = 7.0,
    max_tca_minutes: float = 180.0,
    **_ignored: Any,
) -> list[dict[str, Any]]:
    """Forecast clouds for imminent high-CPI hotspots (signature kept for callers)."""
    if not hotspots or not states:
        return []

    now = _parse_utc(snapshot_timestamp, datetime.now(timezone.utc))
    cpi_lookup = _build_cpi_lookup(alerts or [])

    positions, velocities, meta, index = [], [], [], {}
    for state in states:
        if getattr(state, "error_code", 0) != 0:
            continue
        index[int(state.norad_id)] = len(positions)
        positions.append([float(state.x), float(state.y), float(state.z)])
        velocities.append([float(state.vx), float(state.vy), float(state.vz)])
        meta.append({"norad_id": state.norad_id, "name": state.name})
    if not positions:
        return []
    positions = np.array(positions)
    velocities = np.array(velocities)
    tree = cKDTree(positions)

    clouds: list[dict[str, Any]] = []
    for hotspot in hotspots:
        if len(clouds) >= MAX_FORECAST_CLOUDS:
            break
        cpi_score = float(hotspot.get("cpi_score", 0.0) or 0.0)
        tca_minutes = hotspot.get("tca_minutes")
        if cpi_score < cpi_threshold:
            continue
        if tca_minutes is not None and float(tca_minutes) > max_tca_minutes:
            continue
        id_a = (hotspot.get("sat1") or {}).get("id")
        id_b = (hotspot.get("sat2") or {}).get("id")
        if id_a is None or id_b is None or int(id_a) not in index or int(id_b) not in index:
            continue
        ia, ib = index[int(id_a)], index[int(id_b)]
        tca_time = _parse_utc(hotspot.get("tca_utc"), now)
        minutes_to_tca = max(0.0, (tca_time - now).total_seconds() / 60.0)
        seed = zlib.crc32(f"forecast-{min(id_a, id_b)}-{max(id_a, id_b)}-{tca_time:%Y%m%dT%H%M}".encode())
        fc = forecast_breakup_cloud(positions[ia], velocities[ia], int(id_a), meta[ia]["name"],
                                    positions[ib], velocities[ib], int(id_b), meta[ib]["name"], seed=seed)
        res = fc["result"]
        timeline = fc["timeline"]
        center_vec = res.impact_point_km
        center = {"x": float(center_vec[0]), "y": float(center_vec[1]), "z": float(center_vec[2])}
        radius_ref = timeline[1]["radius_km"] if len(timeline) > 1 else timeline[0]["radius_km"]

        affected, high = [], 0
        for hit in tree.query_ball_point(center_vec, r=max(radius_ref, 1.0)):
            m = meta[hit]
            if m["norad_id"] in (id_a, id_b):
                continue
            info = cpi_lookup.get(m["norad_id"], {})
            cpi = float(info.get("cpi_score", 0.0))
            band = "high" if cpi >= 7.0 else "medium" if cpi >= 5.0 else "low"
            high += band == "high"
            affected.append({**m, "cpi_score": round(cpi, 2), "severity": info.get("severity", "none"),
                             "risk_band": band, "recommended_action": "fallback" if band == "high" else "monitor",
                             "exposure_basis": f"inside p90 forecast radius at +{timeline[1]['minutes']:.0f} min"})

        clouds.append({
            "id": f"forecast-{min(id_a, id_b)}-{max(id_a, id_b)}",
            "kind": "forecast",
            "tca_utc": tca_time.isoformat(),
            "epoch_utc": tca_time.isoformat(),
            "minutes_to_tca": round(minutes_to_tca, 2),
            "center_eci_km": center,
            "centroid_eci_km": center,
            "radius_km_now": 0.0,
            "radius_km_at_tca": radius_ref,
            "radius_p90_km": radius_ref,
            "percentile_radius_km": radius_ref,
            "radius_percentile": 90,
            "radius_reference_minutes_after_breakup": timeline[1]["minutes"] if len(timeline) > 1 else timeline[0]["minutes"],
            "max_radius_km": timeline[-1]["radius_km"],
            "fragment_count": res.n_total,
            "fragment_count_basis": "expected SBM N(>=10 cm) if this pair collides",
            "is_catastrophic": res.catastrophic,
            "emr_j_per_g": round(res.emr_j_per_g, 2),
            "parent_ids": [int(id_a), int(id_b)],
            "parent_mass_kg": [fc["parents"][0]["mass_kg"], fc["parents"][1]["mass_kg"]],
            "mass_source": [fc["parents"][0]["mass_source"], fc["parents"][1]["mass_source"]],
            "seed": res.seed,
            "fragments": [],
            "affected_satellites": affected,
            "affected_count": len(affected),
            "affected_high_risk": int(high),
            "radius_timeline": timeline,
            "shells": [{"label": f"+{t['minutes']:.0f}m", "minutes": t["minutes"], "radius_km": max(t["radius_km"], 1.0)}
                       for t in timeline],
        })
    return clouds
