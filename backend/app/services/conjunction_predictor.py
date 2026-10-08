"""
Shared conjunction prediction pipeline.

Encapsulates the full two-satellite prediction chain so that the manual
predictor endpoint (/api/predict/conjunction) and the live alert loop agree:

  1. Resolve each satellite reference to an ECI state at a common epoch:
     - by NORAD ID (real TLE through the SGP4 propagator), or
     - by geodetic sub-satellite point (latitude/longitude/altitude) with a
       synthesized circular orbit, or
     - by explicit ECI position/velocity override.
  2. Propagate both objects with an RK45 + J2 integrator to the requested
     target time and locate the Time of Closest Approach.
  3. If the miss distance is low enough, build the full conjunction payload at
     the TCA epoch: Foster Pc, cascade plan, debris clouds, and the affected
     satellites present at that time (catalog states propagated to the TCA).
  4. Annotate with the affected ground region (sub-satellite lat/lon).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from app.core.frames import ecef_to_eci, eci_to_geodetic, geodetic_to_ecef
from app.core.sgp4_propagator import SGP4Propagator, SatelliteState, MU
from app.core import sim_clock
from app.core.breakup import DEFAULT_LC_MIN_M, sbm_cumulative_count, sbm_reference_mass
from app.core.conjunction import compute_cpi_score
from app.core.debris_model import object_physical_properties
from app.core.screening import classify_severity, compute_pc, object_meta, propagate_j2
from app.core.state_cache import set_latest_alerts
from app.services.conjunction_solver import (
    COLLISION_THRESHOLD_M,
    CONJUNCTION_REPORT_DISTANCE_M,
    EciState,
    parse_eci_state,
    run_to_tca,
)
from app.services.cascade_planner import CascadePlanner
from app.services.debris_model import build_debris_alerts

logger = logging.getLogger(__name__)

_cascade_planner = CascadePlanner()

_DEFAULT_INCLINATION_DEG = 51.6
_MAX_SAMPLE_STATES = 250


def classify_approach(miss_distance_m: float) -> str:
    """
    Classify a miss distance into an operational band.

    Two thresholds, deliberately distinct:
      * ``collision`` - inside the hard contact gate (COLLISION_THRESHOLD_M);
        the only band for which debris may be generated.
      * ``warning``   - inside the wider report band
        (CONJUNCTION_REPORT_DISTANCE_M); alerts, hotspots and a cascade plan
        are produced, but no debris.
      * ``clear``     - no action required.
    """
    if miss_distance_m <= COLLISION_THRESHOLD_M:
        return "collision"
    if miss_distance_m <= CONJUNCTION_REPORT_DISTANCE_M:
        return "warning"
    return "clear"


def parse_epoch(value: Optional[str], fallback: Optional[datetime] = None) -> datetime:
    """Parse an ISO-8601 UTC timestamp with a 'Z' suffix tolerance."""
    if not value:
        return fallback or sim_clock.simulation_now()

    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"

    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _circular_orbit_state_from_point(
    latitude_deg: float,
    longitude_deg: float,
    altitude_km: float,
    epoch_utc: datetime,
    inclination_deg: float = _DEFAULT_INCLINATION_DEG,
) -> EciState:
    """
    Synthesize a circular orbit whose sub-satellite point passes through the
    given geodetic location at the given epoch.

    The position comes from the geodetic -> ECEF -> ECI chain; the velocity is
    a horizontal, circular-speed vector tangent to a great circle inclined by
    `inclination_deg` through that point (prograde heading from the ascending
    side of the latitude). This gives a deterministic circular orbit that the
    operator can reason about without entering Keplerian elements.
    """
    ecef = geodetic_to_ecef(latitude_deg, longitude_deg, altitude_km)
    pos_eci = ecef_to_eci(ecef, epoch_utc)

    r = float(np.linalg.norm(pos_eci))
    if r <= 0.0:
        raise ValueError("Invalid geodetic position -> radius <= 0")

    v_circ = math.sqrt(MU / r)

    r_hat = pos_eci / r
    k_hat = np.array([0.0, 0.0, 1.0], dtype=float)

    # Local horizontal basis: East and North (orthonormal).
    east = np.cross(k_hat, pos_eci)
    east_norm = float(np.linalg.norm(east))
    if east_norm < 1e-9:
        east = np.array([1.0, 0.0, 0.0], dtype=float)
        east_norm = 1.0
    east = east / east_norm

    north = k_hat - np.dot(k_hat, r_hat) * r_hat
    north_norm = float(np.linalg.norm(north))
    if north_norm < 1e-9:
        north = np.array([0.0, 1.0, 0.0], dtype=float)
        north_norm = 1.0
    north = north / north_norm

    # Heading azimuth A from North such that the great circle has the desired
    # inclination: sin(A) = cos(i) / cos(latitude). Clamp the divergence.
    latitude_rad = math.radians(float(latitude_deg))
    cos_lat = max(1e-9, math.cos(latitude_rad))
    inclination_deg = max(abs(float(latitude_deg)) + 0.5, float(inclination_deg))
    sin_heading = min(1.0, max(-1.0, math.cos(math.radians(inclination_deg)) / cos_lat))
    heading_deg = math.degrees(math.asin(sin_heading))

    # Prograde heading: East component positive (moves east as it rises).
    heading_rad = math.radians(heading_deg)
    heading_vec = north * math.cos(heading_rad) + east * math.sin(heading_rad)
    velocity = heading_vec * v_circ

    return EciState(
        position_km=pos_eci,
        velocity_kms=velocity,
        epoch_utc=epoch_utc,
    )


def _resolve_satellite_reference(
    propagator: SGP4Propagator,
    ref: dict[str, Any],
    epoch: datetime,
) -> tuple[EciState, dict[str, Any]]:
    """
    Resolve a satellite reference to an ECI state plus a label dict.

    Supported shapes for `ref`:
      {"norad_id": <int>}                                real TLE through SGP4
      {"latitude_deg":..,"longitude_deg":..,"altitude_km":..,[inclination_deg]}
      {"position_eci_km":[..],"velocity_eci_kms":[..],["epoch_utc"]} override
    """
    norad_id = ref.get("norad_id")
    has_geo = (
        "latitude_deg" in ref
        and "longitude_deg" in ref
        and "altitude_km" in ref
    )
    has_override = (
        "position_eci_km" in ref
        and "velocity_eci_kms" in ref
    )

    if norad_id is not None and (not has_geo) and (not has_override):
        state = propagator.propagate_one(int(norad_id), epoch)
        if state is None:
            raise ValueError(f"Satellite {norad_id} not found in catalog")
        if state.error_code != 0:
            raise ValueError(f"Satellite {norad_id} failed to propagate (error {state.error_code})")
        label = {"id": int(state.norad_id), "name": state.name, "source": "tle"}
        eci_state = EciState(
            position_km=np.array([state.x, state.y, state.z], dtype=float),
            velocity_kms=np.array([state.vx, state.vy, state.vz], dtype=float),
            epoch_utc=epoch,
        )
        return eci_state, label

    if has_geo:
        incl = float(ref.get("inclination_deg", _DEFAULT_INCLINATION_DEG))
        eci_state = _circular_orbit_state_from_point(
            float(ref["latitude_deg"]),
            float(ref["longitude_deg"]),
            float(ref["altitude_km"]),
            epoch,
            inclination_deg=incl,
        )
        label = {
            "id": int(norad_id) if norad_id is not None else None,
            "name": ref.get("name") or f"GEO({ref['latitude_deg']}, {ref['longitude_deg']})",
            "source": "latlon",
        }
        return eci_state, label

    if has_override:
        ref_epoch = parse_epoch(ref.get("epoch_utc"), epoch)
        eci_state = parse_eci_state(
            ref["position_eci_km"],
            ref["velocity_eci_kms"],
            ref_epoch,
        )
        label = {
            "id": int(norad_id) if norad_id is not None else None,
            "name": ref.get("name") or (f"SAT-{norad_id}" if norad_id is not None else "OVERRIDE"),
            "source": "override",
        }
        return eci_state, label

    raise ValueError("Satellite reference must provide norad_id, lat/lon/alt, or ECI override")


def propagate_states_to(
    propagator: SGP4Propagator,
    states: Optional[list[SatelliteState]] = None,
    target_utc: Optional[str | datetime] = None,
    max_sample: int = _MAX_SAMPLE_STATES,
) -> list[SatelliteState]:
    """
    Propagate a bounded sample of satellite states to a target epoch.

    When `states` is provided, only those NORAD IDs are preserved (useful for
    keeping the neighboring objects around a predicted hit). Otherwise a
    subsample of the full catalog is propagated. States that fail to propagate
    are skipped.
    """
    if target_utc is None:
        return []

    if isinstance(target_utc, str):
        target_utc = parse_epoch(target_utc)

    active_states = []
    if propagator is None:
        return active_states

    if states:
        seen = set()
        for state in states:
            if state.error_code != 0 or state.norad_id in seen:
                continue
            seen.add(state.norad_id)
            propagated = propagator.propagate_one(state.norad_id, target_utc)
            if propagated is not None and propagated.error_code == 0:
                active_states.append(propagated)
        return active_states

    ids = propagator.norad_ids
    if len(ids) > max_sample:
        step = max(1, len(ids) // max_sample)
        sampled_ids = ids[::step][:max_sample]
    else:
        sampled_ids = ids

    for norad_id in sampled_ids:
        state = propagator.propagate_one(norad_id, target_utc)
        if state is not None and state.error_code == 0:
            active_states.append(state)
    return active_states


def _build_test_state(
    norad_id: int,
    name: str,
    position_eci: list[float],
    velocity_eci: list[float],
    tca_utc: str,
) -> SatelliteState:
    state = SatelliteState(norad_id=norad_id, name=name)
    state.x, state.y, state.z = position_eci[0], position_eci[1], position_eci[2]
    state.vx, state.vy, state.vz = velocity_eci[0], velocity_eci[1], velocity_eci[2]
    state.epoch_utc = tca_utc
    state.error_code = 0
    return state


def enrich_payload_with_geodetic(payload: dict[str, Any], tca_time: datetime) -> dict[str, Any]:
    """Add the affected ground region (lat/lon sub-satellite point) to hotspot + debris."""
    hotspots = payload.get("hotspots", [])
    for hotspot in hotspots:
        position = hotspot.get("position")
        if not position:
            continue
        hotspot_geo_time = parse_epoch(hotspot.get("tca_utc"), tca_time)
        geo = eci_to_geodetic(
            np.array([position["x"], position["y"], position["z"]], dtype=float),
            hotspot_geo_time,
        )
        hotspot["geodetic"] = geo

    clouds = payload.get("debris_clouds", [])
    for cloud in clouds:
        center = cloud.get("center_eci_km", {})
        if not center:
            continue
        cloud_geo_time = parse_epoch(cloud.get("tca_utc"), tca_time)
        geo = eci_to_geodetic(
            np.array([center["x"], center["y"], center["z"]], dtype=float),
            cloud_geo_time,
        )
        cloud["affected_region_geodetic"] = {
            **geo,
            "radius_km": cloud.get("radius_km_at_tca", 0.0),
        }
    return payload


def _tle_age_days(propagator, norad_id: int, at: datetime) -> float | None:
    """|at - TLE epoch| in days for a catalogue object, else None."""
    try:
        with propagator._lock:
            entry = propagator._satellites.get(int(norad_id))
    except Exception:
        return None
    if not entry:
        return None
    satrec = entry[0]
    from sgp4.api import jday

    jd, fr = jday(at.year, at.month, at.day, at.hour, at.minute, at.second + at.microsecond / 1e6)
    return abs((jd - satrec.jdsatepoch) + (fr - satrec.jdsatepochF))


def assess_pair_risk(propagator, obj_a: tuple, obj_b: tuple, tca_time: datetime, ref_time: datetime) -> dict:
    """
    Foster Pc, B-plane geometry and SBM fragment forecast for one pair at TCA.

    obj_* = (norad_id, name, r_tca_km, v_tca_kms). Covariance: TLE-age RTN
    model of app.core.screening (age = |TCA - TLE epoch| for catalogue objects;
    for objects given only as a state, the propagation span ref -> TCA).
    HBR: sum of app.core.screening.object_meta radii. Fragment count: NASA SBM
    N(>=10 cm) = 0.1 M_ref^0.75 Lc^-1.71 with masses from
    app.core.debris_model.object_physical_properties (mass_source tagged).
    Anything that cannot be computed is returned as None, never substituted.
    """
    span_days = max(0.0, (tca_time - ref_time).total_seconds()) / 86400.0
    metas, ages, age_src, props = [], [], [], []
    for nid, name, _, _ in (obj_a, obj_b):
        metas.append(object_meta(int(nid), name))
        age = _tle_age_days(propagator, nid, tca_time)
        ages.append(span_days if age is None else age)
        age_src.append("propagation_span_from_state" if age is None else "tle_epoch_to_tca")
        props.append(object_physical_properties(int(nid), name))
    hbr_km = (metas[0]["radius_m"] + metas[1]["radius_m"]) / 1000.0
    r1, v1, r2, v2 = (np.asarray(x, float) for x in (obj_a[2], obj_a[3], obj_b[2], obj_b[3]))
    out: dict[str, Any] = {
        "hbr_km": round(hbr_km, 6),
        "tle_age_days": [round(a, 4) for a in ages],
        "tle_age_source": age_src,
        "meta": metas,
        "p_collision": None, "b_t_km": None, "b_n_km": None,
        "covariance_ellipse": None, "short_encounter_valid": None,
    }
    try:
        pc = compute_pc(r1, v1, r2, v2, age1_days=ages[0], age2_days=ages[1], hbr_km=hbr_km)
        out.update(
            p_collision=float(pc["pc"]),
            b_t_km=round(pc["b_t_km"], 6),
            b_n_km=round(pc["b_n_km"], 6),
            covariance_ellipse={**pc["covariance_ellipse"],
                                "radius_source": [m["radius_source"] for m in metas]},
            short_encounter_valid=pc["short_encounter_valid"],
        )
    except Exception as exc:
        logger.warning("compute_pc failed for %s-%s: %s", obj_a[0], obj_b[0], exc)

    v_rel = float(np.linalg.norm(v2 - v1))
    m_ref, catastrophic, emr = sbm_reference_mass(props[0]["mass_kg"], props[1]["mass_kg"], v_rel)
    out["breakup_forecast"] = {
        "expected_fragments": int(round(sbm_cumulative_count(m_ref, DEFAULT_LC_MIN_M))),
        "lc_min_m": DEFAULT_LC_MIN_M,
        "is_catastrophic": catastrophic,
        "emr_j_per_g": round(emr / 1000.0, 3),
        "sbm_reference_mass_kg": round(m_ref, 2),
        "relative_velocity_kms": round(v_rel, 5),
        "masses_kg": [props[0]["mass_kg"], props[1]["mass_kg"]],
        "mass_source": [props[0]["mass_source"], props[1]["mass_source"]],
        "model": "NASA SBM: N(>=Lc) = 0.1 * M_ref^0.75 * Lc^-1.71",
        "conditional_on": "collision_occurring",
    }
    return out


def build_collision_prediction(
    propagator: SGP4Propagator,
    sat_a: SimpleNamespace,
    sat_b: SimpleNamespace,
    result,
    collision_confirmed: Optional[bool] = None,
) -> dict[str, Any]:
    """
    Build the full conjunction prediction payload at the TCA epoch.

    Mirrors the former test-mode pipeline: catalog states propagated to the
    TCA, Foster Pc, cascade plan, and debris clouds with affected satellites.

    Debris clouds are generated only for a *confirmed* collision (miss distance
    at or below the hard contact gate). A close approach inside the wider report
    band still produces alerts, hotspots, and a cascade plan, but no debris.
    """
    if collision_confirmed is None:
        collision_confirmed = bool(getattr(result, "collision_detected", False))

    # The hard contact gate is authoritative and cannot be overridden by a
    # caller: a kilometre-scale near miss must never spawn debris, even if a
    # stale or optimistic `collision_confirmed=True` is passed in.
    miss_distance_m = float(getattr(result, "miss_distance_m", float("inf")))
    if collision_confirmed and miss_distance_m > COLLISION_THRESHOLD_M:
        logger.info(
            "Rejecting collision_confirmed=True: miss %.3f m exceeds the %.1f m collision gate",
            miss_distance_m,
            COLLISION_THRESHOLD_M,
        )
        collision_confirmed = False

    tca_time = parse_epoch(result.tca_utc, sat_a.epoch_utc)
    sat1_id = int(sat_a.norad_id)
    sat2_id = int(sat_b.norad_id)

    # Reference epoch = the epoch the two states were resolved at (the sim
    # clock "now" for the live predictor). TCA is reported relative to it and
    # the cascade / manoeuvre analysis runs from it (a burn happens BEFORE TCA).
    ref_time = getattr(sat_a, "epoch_utc", None)
    if not isinstance(ref_time, datetime):
        ref_time = parse_epoch(ref_time if isinstance(ref_time, str) else None)
    elif ref_time.tzinfo is None:
        ref_time = ref_time.replace(tzinfo=timezone.utc)
    tca_s = max(0.0, (tca_time - ref_time).total_seconds())

    # Catalogue states at TCA (debris / affected satellites) ...
    active_states = propagate_states_to(propagator, target_utc=tca_time)

    state_a_tca = _build_test_state(
        sat1_id, sat_a.name,
        result.position_a_eci, result.velocity_a_eci, result.tca_utc,
    )
    state_b_tca = _build_test_state(
        sat2_id, sat_b.name,
        result.position_b_eci, result.velocity_b_eci, result.tca_utc,
    )
    all_states = active_states + [state_a_tca, state_b_tca]

    # ... and at the reference epoch (cascade / manoeuvre planner). The pair is
    # brought back from its TCA state with the same two-body + J2 dynamics the
    # TCA solver integrated forward.
    states_now = propagate_states_to(propagator, target_utc=ref_time) if tca_s > 0 else list(active_states)
    pair_now = []
    for sid, name, r_t, v_t in ((sat1_id, sat_a.name, result.position_a_eci, result.velocity_a_eci),
                                (sat2_id, sat_b.name, result.position_b_eci, result.velocity_b_eci)):
        r0, v0 = propagate_j2(np.asarray(r_t, float)[None, :], np.asarray(v_t, float)[None, :], -tca_s)
        pair_now.append(_build_test_state(sid, name, [float(x) for x in r0[0]], [float(x) for x in v0[0]],
                                          ref_time.isoformat()))
    pair_ids = {sat1_id, sat2_id}
    cascade_states = [st for st in states_now if int(st.norad_id) not in pair_ids] + pair_now

    risk = assess_pair_risk(
        propagator,
        (sat1_id, sat_a.name, result.position_a_eci, result.velocity_a_eci),
        (sat2_id, sat_b.name, result.position_b_eci, result.velocity_b_eci),
        tca_time, ref_time,
    )
    p_col = risk["p_collision"]
    bt_km = risk["b_t_km"]
    bn_km = risk["b_n_km"]
    covariance_ellipse = risk["covariance_ellipse"]
    miss_km = result.miss_distance_m / 1000.0
    rel_kms = float(result.relative_velocity_kms)
    tca_hours = tca_s / 3600.0
    if p_col is not None:
        cpi = round(compute_cpi_score(p_col, miss_km, tca_hours=tca_hours, relative_velocity_kms=rel_kms,
                                      tle_age_hours=24.0 * max(risk["tle_age_days"])), 4)
        severity = classify_severity(p_col, miss_km)
    else:
        cpi = None
        severity = None
    # Zone radius = 3-sigma semi-major axis of the combined B-plane covariance.
    zone_radius_km = round(covariance_ellipse["a"] / 1000.0, 3) if covariance_ellipse else None

    primary_alert = {
        "sat1": {"id": sat1_id, "name": sat_a.name, "object_type": risk["meta"][0]["object_type"],
                 "agency": risk["meta"][0]["agency"]},
        "sat2": {"id": sat2_id, "name": sat_b.name, "object_type": risk["meta"][1]["object_type"],
                 "agency": risk["meta"][1]["agency"]},
        "miss_distance_km": round(miss_km, 6),
        "relative_speed_kmh": round(rel_kms * 3600.0, 3),
        "relative_speed_kms": round(rel_kms, 5),
        "p_collision": p_col,
        "probability_of_collision": p_col,
        "pc_method": "foster" if p_col is not None else None,
        "hbr_km": risk["hbr_km"],
        "sigma_source": "tle_age_model",
        "tle_age_days": risk["tle_age_days"],
        "tle_age_source": risk["tle_age_source"],
        "short_encounter_valid": risk["short_encounter_valid"],
        "cpi_score": cpi,
        "cpi_method": "app.core.conjunction.compute_cpi_score" if cpi is not None else None,
        "hotspot_score": cpi,
        "severity": severity,
        "tca_minutes": round(tca_s / 60.0, 3),
        "tca_hours": round(tca_hours, 5),
        "tca_utc": result.tca_utc,
        "reference_utc": ref_time.isoformat(),
        "bt_km": bt_km,
        "bn_km": bn_km,
        "b_t_km": bt_km,
        "b_n_km": bn_km,
        "b_plane_bt_km": bt_km,
        "b_plane_bn_km": bn_km,
        "hotspot_position": {
            "x": round(float(state_a_tca.x + state_b_tca.x) / 2.0, 3),
            "y": round(float(state_a_tca.y + state_b_tca.y) / 2.0, 3),
            "z": round(float(state_a_tca.z + state_b_tca.z) / 2.0, 3),
        },
        "zone_radius_km": zone_radius_km,
        "zone_radius_source": "3sigma_bplane_semi_major" if zone_radius_km is not None else None,
        "covariance_ellipse": covariance_ellipse,
        "breakup_forecast": risk["breakup_forecast"],
    }

    cascade_summary = _cascade_planner.analyze_snapshot(
        states=cascade_states,
        alerts=[primary_alert],
        propagator=propagator,
        reference_time=ref_time,
        cpi_threshold=5.0,
    )

    planner_alerts = cascade_summary.get("alerts", [])
    has_primary = any(
        (a["sat1"]["id"] == sat1_id and a["sat2"]["id"] == sat2_id) or
        (a["sat1"]["id"] == sat2_id and a["sat2"]["id"] == sat1_id)
        for a in planner_alerts
    )
    if not has_primary:
        planner_alerts.insert(0, primary_alert)
    else:
        for a in planner_alerts:
            if (a["sat1"]["id"] == sat1_id and a["sat2"]["id"] == sat2_id) or \
               (a["sat1"]["id"] == sat2_id and a["sat2"]["id"] == sat1_id):
                a["covariance_ellipse"] = primary_alert["covariance_ellipse"]
                a["bt_km"] = bt_km
                a["bn_km"] = bn_km
                a["b_plane_bt_km"] = bt_km
                a["b_plane_bn_km"] = bn_km
    # Other alerts keep the covariance their own screening computed; none is invented here.

    hotspots = cascade_summary.get("hotspots", [])
    primary_hotspot = {
        "sat1": {"id": sat1_id, "name": sat_a.name},
        "sat2": {"id": sat2_id, "name": sat_b.name},
        "miss_distance_km": round(miss_km, 6),
        "relative_speed_kmh": round(float(result.relative_velocity_kms) * 3600.0, 3),
        "p_collision": p_col,
        "cpi_score": cpi,
        "hotspot_score": cpi,
        "severity": severity,
        "tca_minutes": primary_alert["tca_minutes"],
        "tca_hours": primary_alert["tca_hours"],
        "tca_utc": result.tca_utc,
        "position": primary_alert["hotspot_position"],
        "zone_radius_km": zone_radius_km,
    }
    has_primary_hotspot = any(
        (h["sat1"]["id"] == sat1_id and h["sat2"]["id"] == sat2_id) or
        (h["sat1"]["id"] == sat2_id and h["sat2"]["id"] == sat1_id)
        for h in hotspots
    )
    if not has_primary_hotspot:
        hotspots.insert(0, primary_hotspot)

    if collision_confirmed:
        debris_clouds = build_debris_alerts(
            hotspots,
            all_states,
            tca_time.isoformat(),
            alerts=planner_alerts,
            cpi_threshold=5.0,
        )
    else:
        logger.info(
            "Skipping debris generation: miss %.3f m exceeds the %.1f m collision gate",
            result.miss_distance_m,
            COLLISION_THRESHOLD_M,
        )
        debris_clouds = []

    payload = {
        "timestamp": tca_time.isoformat(),
        "reference_utc": ref_time.isoformat(),
        "breakup_forecast": risk["breakup_forecast"],
        "alerts": planner_alerts,
        "count": len(planner_alerts),
        "hotspots": hotspots,
        "graph": cascade_summary.get("graph", {}),
        "cascade_plan": cascade_summary.get("cascade_plan", []),
        "cascade_depth": cascade_summary.get("cascade_depth", 1),
        "total_delta_v_ms": cascade_summary.get("total_delta_v_ms", 0.0),
        "agencies_involved": cascade_summary.get("agencies_involved", []),
        "seed_satellites": cascade_summary.get("seed_satellites", [sat_a.name, sat_b.name]),
        "cpi_threshold": 5.0,
        "node_probabilities": cascade_summary.get("node_probabilities", {}),
        "debris_clouds": debris_clouds,
        "collision_confirmed": collision_confirmed,
    }
    enrich_payload_with_geodetic(payload, tca_time)
    set_latest_alerts(payload)
    return payload


def predict_conjunction(
    propagator: SGP4Propagator,
    satellite_a: dict[str, Any],
    satellite_b: dict[str, Any],
    step_seconds: float = 60.0,
    max_steps: int = 2400,
    target_utc: Optional[str] = None,
    forecast_hours: float = 24.0,
    epoch_utc: Optional[str] = None,
) -> dict[str, Any]:
    """
    Predict a future conjunction between two objects.

    Returns a dict describing the closest approach within the forecast window:
    TCA time, miss distance, relative velocity, Foster Pc, cascade plan, and —
    when the miss is low enough — debris clouds with the satellites present at
    the collision time and the affected ground region.
    """
    epoch = parse_epoch(epoch_utc)
    target = parse_epoch(target_utc, fallback=epoch + timedelta(hours=forecast_hours))

    state_a, label_a = _resolve_satellite_reference(propagator, satellite_a, epoch)
    state_b, label_b = _resolve_satellite_reference(propagator, satellite_b, epoch)

    result = run_to_tca(
        state_a=state_a,
        state_b=state_b,
        target_utc=target,
        step_seconds=step_seconds,
        max_steps=max_steps,
        use_j2=True,
    )

    sat_a = SimpleNamespace(
        norad_id=label_a["id"] if label_a["id"] is not None else 999999,
        name=label_a["name"],
        epoch_utc=epoch,
    )
    sat_b = SimpleNamespace(
        norad_id=label_b["id"] if label_b["id"] is not None else 999998,
        name=label_b["name"],
        epoch_utc=epoch,
    )

    classification = classify_approach(result.miss_distance_m)

    base = {
        "epoch_utc": epoch.isoformat(),
        "target_utc": target.isoformat(),
        "satellites": {"a": label_a, "b": label_b},
        "tca": {
            "tca_utc": result.tca_utc,
            "miss_distance_km": round(result.miss_distance_m / 1000.0, 6),
            "miss_distance_m": round(result.miss_distance_m, 3),
            "relative_velocity_kms": round(result.relative_velocity_kms, 5),
            "collision_detected": result.collision_detected,
            "classification": classification,
        },
        "trajectories": {"a": result.trajectory_a, "b": result.trajectory_b},
    }

    if result.miss_distance_m <= CONJUNCTION_REPORT_DISTANCE_M:
        prediction = build_collision_prediction(
            propagator,
            sat_a,
            sat_b,
            result,
            collision_confirmed=result.collision_detected,
        )
        status = "collision" if result.collision_detected else "conjunction"
        return {**base, "status": status, **prediction}

    empty_payload: dict[str, Any] = {
        "alerts": [], "count": 0, "hotspots": [], "graph": {},
        "cascade_plan": [], "cascade_depth": 0, "total_delta_v_ms": 0.0,
        "agencies_involved": [], "seed_satellites": [],
        "cpi_threshold": 5.0, "node_probabilities": {}, "debris_clouds": [],
        "collision_confirmed": False,
    }
    return {**base, "status": "clear", **empty_payload}