"""
REST API routes for maneuvers, scenarios, and alerts.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.core.state_cache import get_latest_alerts, get_latest_snapshot, set_latest_alerts, set_latest_snapshot
from app.core.screening import screen as screen_alerts
from app.core.sgp4_propagator import MU, RE
from app.core.agency import infer_agency
from app.core.agency_authority import authority_manager
from app.core import sim_clock
from app.services.cascade_planner import CascadePlanner
from app.services.debris_model import build_debris_alerts
from app.services.conjunction_predictor import propagate_states_to, enrich_payload_with_geodetic

# ML modules are imported lazily inside the handlers that need them so that
# importing the API layer does not pull in torch/xgboost or trigger any model
# load. See get_ml_runtime_if_enabled() and _get_model_metrics().

try:
    from app.core.satellite_state_tracker import satellite_tracker
    _TRACKER_AVAILABLE = True
except Exception:  # any init failure, not just ImportError, must hit the fallback
    satellite_tracker = None
    _TRACKER_AVAILABLE = False

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


# ── Agency authority ──────────────────────────────────────────────────────────
# Commanding is a privileged operation, so every state-changing endpoint is
# gated on an agency session presented in the X-Agency-Session header.
#
# ALLOW_UNAUTHENTICATED_COMMANDS controls what happens when the header is
# absent. It defaults to 1 so the out-of-the-box demo keeps working, but that
# is a deployment choice made explicitly here rather than a hardcoded True
# buried in the authority manager. Set it to 0 in any real deployment and
# commands without a valid session are rejected with 401.

SESSION_HEADER = "X-Agency-Session"
DEMO_SESSION = "DEMO_SESSION"

_UNAUTHENTICATED_COMMANDS_DEFAULT = "1"


def _unauthenticated_commands_allowed() -> bool:
    return os.getenv("ALLOW_UNAUTHENTICATED_COMMANDS", _UNAUTHENTICATED_COMMANDS_DEFAULT) == "1"


def _session_minting_allowed() -> bool:
    # Minting a session for an arbitrary agency is a bootstrap operation. It
    # stays enabled for the demo and must be disabled alongside
    # ALLOW_UNAUTHENTICATED_COMMANDS in a real deployment.
    return os.getenv("ALLOW_SESSION_MINTING", _UNAUTHENTICATED_COMMANDS_DEFAULT) == "1"


def resolve_session_id(
    x_agency_session: str | None = Header(default=None, alias=SESSION_HEADER),
) -> str:
    """FastAPI dependency: resolve the caller's agency session."""
    if x_agency_session and x_agency_session.strip():
        return x_agency_session.strip()

    if _unauthenticated_commands_allowed():
        return DEMO_SESSION

    raise HTTPException(
        status_code=401,
        detail=f"Missing {SESSION_HEADER} header. Commanding requires an agency session.",
    )


def _require_satellite_authority(session_id: str, norad_id: int) -> str:
    """
    Enforce that the session may command this satellite.

    Returns the resolved satellite name. Raises 403 when the session is not
    authorized for the satellite's controlling agency.
    """
    satellite_name = f"NORAD-{norad_id}"
    if _propagator is not None:
        try:
            state = _propagator.propagate_one(norad_id)
            if state is not None and not state.error_code and state.name:
                satellite_name = state.name
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail=f"Could not resolve satellite {norad_id} for authorization: {error}",
            ) from error

    decision = authority_manager.explain_authority(session_id, norad_id, satellite_name)
    if not decision["allowed"]:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Session agency '{decision.get('session_agency') or 'unknown'}' is not authorized "
                f"to command '{satellite_name}' (NORAD {norad_id}, type {decision['object_type']}), "
                f"controlled by '{decision['controlling_agency']}' [rule: {decision['rule']}]."
            ),
        )

    return satellite_name


def _require_valid_session(session_id: str) -> None:
    """Enforce a real, unexpired session for global state changes."""
    if session_id == DEMO_SESSION:
        return
    info = authority_manager.get_session_info(session_id)
    if info is None:
        raise HTTPException(status_code=401, detail=f"Unknown agency session '{session_id}'.")
    if info.get("expired"):
        raise HTTPException(status_code=401, detail=f"Agency session '{session_id}' has expired.")


class ManeuverRequest(BaseModel):
    norad_id: int
    dvx: float  # m/s
    dvy: float
    dvz: float
    frame: str = "RSW"


class ScenarioRequest(BaseModel):
    name: str | None = None
    scenario: str | None = None


class JudgeManipulateRequest(BaseModel):
    """Override a satellite's state and predict downstream effects."""
    norad_id: int
    position_eci_km: list[float] | None = None  # [x, y, z] in km
    velocity_eci_kms: list[float] | None = None  # [vx, vy, vz] in km/s


class OrbitChangeRequest(BaseModel):
    norad_id: int
    target_altitude_km: float


class OverrideRequest(BaseModel):
    """Direct ECI state override (simulation mode only)."""
    norad_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float


class ManeuverFeedbackRequest(BaseModel):
    alert_id: str | None = None
    sat1_id: int
    sat2_id: int | None = None
    decision: str
    delta_v_ms: float = 0.1
    # Recommended manoeuvre as computed by the cascade planner (RSW, m/s).
    sat_id: int | None = None
    delta_v_rsw_ms: list[float] | None = None


# These will be set by main.py on startup
_propagator = None
_sim_engine = None
_cascade_planner = CascadePlanner()


def init_routes(propagator, sim_engine):
    global _propagator, _sim_engine
    _propagator = propagator
    _sim_engine = sim_engine


def _build_snapshot(current_propagator, dt: datetime) -> dict:
    states = current_propagator.propagate_all(dt)

    active_states = [state for state in states if state.error_code == 0]
    satellites = [
        {
            "norad_id": state.norad_id,
            "name": state.name,
            "agency": infer_agency(state.name, state.norad_id),
            "position": {
                "x": state.x,
                "y": state.y,
                "z": state.z,
            },
            "velocity": {
                "vx": state.vx,
                "vy": state.vy,
                "vz": state.vz,
            },
            "speed_kmh": round(float((state.vx**2 + state.vy**2 + state.vz**2) ** 0.5 * 3600.0), 2),
            "altitude_km": round(float(((state.x**2 + state.y**2 + state.z**2) ** 0.5) - 6371.0), 2),
            "epoch_utc": state.epoch_utc,
        }
        for state in active_states
    ]

    from app.core.debris_model import debris_model
    debris_clouds = debris_model.get_frontend_debris_clouds(active_states)

    snapshot = {
        "type": "update",
        "timestamp": dt.isoformat(),
        "satellites": satellites,
        "count": len(states),
        "states": active_states,
        "debris_clouds": debris_clouds,
    }
    snapshot["payload"] = json.dumps({
        "type": snapshot["type"],
        "timestamp": snapshot["timestamp"],
        "satellites": satellites,
        "count": snapshot["count"],
        "debris_clouds": debris_clouds,
    }, separators=(",", ":"))
    return snapshot


@router.get("/satellites")
async def get_satellites():
    """Get current satellite positions."""
    snapshot = get_latest_snapshot()

    payload = snapshot.get("payload")
    if payload:
        return Response(content=payload, media_type="application/json")

    return {
        "satellites": snapshot.get("satellites", []),
        "count": snapshot.get("count", 0),
        "timestamp": snapshot.get("timestamp"),
    }


@router.get("/satellites/{norad_id}")
async def get_satellite_detail(norad_id: int):
    """Get the full state for a single satellite."""
    if _propagator is None:
        raise HTTPException(status_code=503, detail="Propagator not initialized")

    state = _propagator.propagate_one(norad_id, sim_clock.simulation_now())
    if state is None:
        raise HTTPException(status_code=404, detail="Satellite not found")

    from app.core import satcat
    from app.core.agency import agency_attribution

    detail = state.to_dict()
    detail.update(agency_attribution(state.name, norad_id))
    # CelesTrak SATCAT record (offline snapshot) or TLE-derived fallback; None
    # for synthetic scenario objects that exist in no catalogue.
    detail["satcat"] = satcat.lookup(norad_id, name=state.name)
    return detail


def _fallback_telemetry(norad_id: int, satellite_name: str) -> dict:
    """Telemetry used only when the state tracker is unavailable.

    There is no telemetry source in that case, so every housekeeping value is
    reported as missing rather than invented.
    """
    return {
        "norad_id": norad_id,
        "satellite_name": satellite_name,
        "telemetry_available": False,
        "fuel_remaining_pct": None,
        "battery_pct": None,
        "temperature_c": None,
        "signal_strength_dbm": None,
        "solar_power_w": None,
        "total_delta_v_used_ms": 0.0,
        "maneuver_count": 0,
        "telemetry_source": "unavailable",
    }


@router.get("/satellites/{norad_id}/telemetry")
async def get_satellite_telemetry(norad_id: int):
    """Get stateful telemetry for a satellite (fuel, battery, temperature, etc.)."""
    if _propagator is None:
        raise HTTPException(status_code=503, detail="Propagator not initialized")

    state = _propagator.propagate_one(norad_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Satellite not found")

    if _TRACKER_AVAILABLE and satellite_tracker is not None:
        telemetry = satellite_tracker.get_telemetry(norad_id, state.name)
    else:
        telemetry = _fallback_telemetry(norad_id, state.name)

    return {
        **telemetry,
        "norad_id": norad_id,
        "epoch_utc": state.epoch_utc,
    }


@router.get("/satellites/{norad_id}/history")
async def get_satellite_maneuver_history(norad_id: int):
    """Get the full maneuver history for a satellite."""
    if _TRACKER_AVAILABLE and satellite_tracker is not None:
        history = satellite_tracker.get_maneuver_history(norad_id)
    else:
        history = []
    return {
        "norad_id": norad_id,
        "maneuver_count": len(history),
        "history": history,
    }


@router.get("/alerts")
async def get_alerts():
    """Get current conjunction alerts."""
    alerts_snapshot = get_latest_alerts()
    snapshot = get_latest_snapshot()

    # Two debris sources exist and both are needed by the client:
    #   * forecast clouds from build_debris_alerts() -- imminent conjunctions
    #     above the CPI threshold, carrying the multi-shell `shells` array and
    #     per-satellite risk bands.
    #   * fragment clouds from the core EVOLVE model -- clouds that exist only
    #     because a collision was actually simulated.
    # The forecast clouds were previously computed and cached but never served:
    # this handler read the snapshot's fragment clouds only, so the multi-shell
    # forecast structure never reached the globe.
    forecast_clouds = alerts_snapshot.get("debris_clouds", []) or []
    fragment_clouds = snapshot.get("debris_clouds", []) or []
    debris_clouds = [
        *(
            {**cloud, "debris_source": "forecast"}
            for cloud in forecast_clouds
        ),
        *(
            {**cloud, "debris_source": "fragment"}
            for cloud in fragment_clouds
        ),
    ]

    return {
        "alerts": alerts_snapshot.get("alerts", []),
        "count": alerts_snapshot.get("count", 0),
        "timestamp": alerts_snapshot.get("timestamp"),
        "hotspots": alerts_snapshot.get("hotspots", []),
        "cascade_plan": alerts_snapshot.get("cascade_plan", []),
        "graph": alerts_snapshot.get("graph", {}),
        "cascade_depth": alerts_snapshot.get("cascade_depth", 0),
        "total_delta_v_ms": alerts_snapshot.get("total_delta_v_ms", 0.0),
        "agencies_involved": alerts_snapshot.get("agencies_involved", []),
        "seed_satellites": alerts_snapshot.get("seed_satellites", []),
        "cpi_threshold": alerts_snapshot.get("cpi_threshold", 5.0),
        "node_probabilities": alerts_snapshot.get("node_probabilities", {}),
        "ranker_review": alerts_snapshot.get("ranker_review", {}),
        "debris_clouds": debris_clouds,
        "debris_sources": ["forecast", "fragment"],
    }


def get_ml_runtime_if_enabled():
    """Return the ML runtime only when the extended pipeline is enabled."""
    import os
    if os.getenv("ENABLE_EXTENDED_PIPELINE", "0") != "1":
        return None
    from app.ml.runtime import get_ml_runtime
    return get_ml_runtime()


@router.get("/anomalies")
async def get_anomalies():
    runtime = get_ml_runtime_if_enabled()
    if runtime is None:
        return {"count": 0, "anomalies": [], "enabled": False}
    anomalies = runtime.get_anomalies()
    return {"count": len(anomalies), "anomalies": anomalies, "enabled": True}


@router.get("/model-metrics")
async def get_model_metrics_endpoint():
    from app.ml.model_metrics import get_model_metrics
    return get_model_metrics()


def _ml_status_payload() -> dict:
    """Build the /api/ml/status body. Never constructs the ML runtime."""
    import app.ml.xgboost_scorer as xgb_scorer
    from app.ml.rlhf_store import get_stats as rlhf_stats

    pipeline_enabled = os.getenv("ENABLE_EXTENDED_PIPELINE", "0") == "1"

    # Read the runtime only if something already built it: status polling must
    # not trigger the (slow) model load/training that get_ml_runtime() does.
    runtime = None
    if pipeline_enabled:
        from app.ml.runtime import get_ml_runtime
        if get_ml_runtime.cache_info().currsize:
            runtime = get_ml_runtime()

    xgboost = {
        "using_trained": xgb_scorer.using_trained_model(),
        **xgb_scorer.prediction_counter.snapshot(),
    }

    lstm = {"trained": False, "satellites_tracked": 0, "buffer_records": 0, "buffer_threshold": 500}
    shadow = {
        "enabled": pipeline_enabled and os.getenv("SHADOW_MODE_ENABLED", "1") == "1",
        "last_retrain": None,
        "last_result": None,
    }
    if runtime is not None:
        try:
            lstm = runtime.trajectory.stats()
        except Exception as error:
            logger.warning("LSTM stats unavailable: %s", error)
        shadow["enabled"] = bool(runtime.shadow.enabled)
        shadow["last_retrain"] = runtime.shadow.last_retrain
        shadow["last_result"] = runtime.shadow.last_result
    else:
        from app.ml.lstm_predictor import LSTM_RETRAIN_THRESHOLD
        lstm["buffer_threshold"] = LSTM_RETRAIN_THRESHOLD

    try:
        rlhf = rlhf_stats()
    except Exception as error:
        logger.warning("RLHF stats unavailable: %s", error)
        rlhf = {
            "decisions": 0, "approved": 0, "rejected": 0,
            "approval_rate": None, "rounds": 0, "round_approval_rates": [],
        }

    # There is no meta-propagator: nothing produces corrected-vs-raw-SGP4
    # error pairs. Shadow mode logs LSTM predictions against SGP4 positions
    # (SGP4 is the "actual"), so an improvement over SGP4 cannot be measured
    # from it. Report zeros rather than a fabricated number.
    meta_propagator = {"satellites_with_corrections": 0, "mean_improvement_pct": 0.0, "corrections": 0}

    return {
        "pipeline_enabled": pipeline_enabled,
        "xgboost": xgboost,
        "lstm": lstm,
        "rlhf": rlhf,
        "meta_propagator": meta_propagator,
        "shadow": shadow,
    }


@router.get("/ml/status")
async def get_ml_status():
    """Aggregate live ML pipeline counters for the model status panel.

    Safe with ENABLE_EXTENDED_PIPELINE=0: LSTM/shadow fields report
    zeros/false/null and the ML runtime is never constructed by this call.
    ``meta_propagator`` is always zeros (no such model exists; see
    _ml_status_payload).
    """
    return await run_in_threadpool(_ml_status_payload)


@router.post("/ml/retrain")
async def trigger_ml_retrain():
    """Run a shadow-mode retrain (LSTM fine-tune + risk model) on demand.

    Always responds HTTP 200 with ``{"ok", "result", "message"}``; failure is
    signalled by ``ok: false``, never by the status code:
      - pipeline disabled (ENABLE_EXTENDED_PIPELINE != 1): ok=false, result=null
      - shadow mode disabled: ok=false, result is the "disabled" marker
      - retrain raised: ok=false, result=null, message carries the error
    "skipped" sub-results (insufficient data) still return ok=true; inspect
    ``result.lstm.status`` / ``result.risk.status``. Runs in a threadpool so
    the event loop keeps serving WebSocket/REST traffic during training.
    """
    runtime = get_ml_runtime_if_enabled()
    if runtime is None:
        return {
            "ok": False,
            "result": None,
            "message": "Extended ML pipeline is disabled. Set ENABLE_EXTENDED_PIPELINE=1 and restart the backend.",
        }
    try:
        result = await run_in_threadpool(runtime.shadow_retrain)
    except Exception as error:
        logger.exception("Manual shadow retrain failed")
        return {"ok": False, "result": None, "message": f"Retrain failed: {error}"}

    if "status" in result and (result.get("status") or {}).get("status") == "disabled":
        return {
            "ok": False,
            "result": result,
            "message": "Shadow mode is disabled (SHADOW_MODE_ENABLED=0).",
        }

    parts = [f"{name}: {(sub or {}).get('status', 'unknown')}" for name, sub in result.items()]
    return {"ok": True, "result": result, "message": "Retrain finished (" + ", ".join(parts) + ")."}


@router.get("/cascade/status")
async def get_cascade_status():
    """Get the current cascade prediction status."""
    alerts_snapshot = get_latest_alerts()
    plan = alerts_snapshot.get("cascade_plan", [])
    alerts = alerts_snapshot.get("alerts", []) or []
    depths = [int(a["cascade_depth"]) for a in alerts if isinstance(a.get("cascade_depth"), (int, float))]

    return {
        "cascade_predictions": len(plan),
        # BFS hop counts from the root collision event (primary conjunction = 1).
        "mean_cascade_depth": round(sum(depths) / len(depths), 3) if depths else 0.0,
        "max_cascade_depth": max(depths) if depths else 0,
        "alerts_with_depth": len(depths),
        "events": sorted({a["upstream_event"] for a in alerts if a.get("upstream_event")}),
        "planned_maneuvers": sum(1 for a in alerts if a.get("recommended_maneuver")),
        "depth_definition": "BFS hops from root collision event",
    }


@router.get("/satellites/{norad_id}/orbit")
async def get_satellite_orbit(
    norad_id: int,
    span_minutes: int = 90,
    step_seconds: int = 60,
):
    """Get a short orbit track for the selected satellite."""
    if _propagator is None:
        raise HTTPException(status_code=503, detail="Propagator not initialized")

    from datetime import timedelta

    span_minutes = max(15, min(span_minutes, 180))
    step_seconds = max(10, min(step_seconds, 300))

    now = sim_clock.simulation_now()
    start = now - timedelta(minutes=span_minutes / 2)
    end = now + timedelta(minutes=span_minutes / 2)

    orbit = []
    current = start
    while current <= end:
      state = _propagator.propagate_one(norad_id, current)
      if state is not None and state.error_code == 0:
          orbit.append(
              {
                  "epoch_utc": state.epoch_utc,
                  "position": {
                      "x": state.x,
                      "y": state.y,
                      "z": state.z,
                  },
              }
          )
      current += timedelta(seconds=step_seconds)

    if not orbit:
        raise HTTPException(status_code=404, detail="Satellite not found")

    return {
        "norad_id": norad_id,
        "generated_at": now.isoformat(),
        "orbit": orbit,
    }


@router.post("/maneuver")
async def execute_maneuver(
    req: ManeuverRequest,
    session_id: str = Depends(resolve_session_id),
):
    """Execute a delta-V maneuver with pre-flight validation."""
    if _sim_engine is None or _propagator is None:
        return {"status": "ERROR", "message": "Sim engine not initialized"}

    _require_satellite_authority(session_id, req.norad_id)

    result = _sim_engine.apply_maneuver(
        req.norad_id,
        req.dvx,
        req.dvy,
        req.dvz,
        frame=req.frame,
        session_id=session_id,
    )

    if result.get("status") == "success":
        # ── Record maneuver in state tracker (fuel depletion) ─────────────
        import math as _math
        delta_v_ms = _math.sqrt(req.dvx**2 + req.dvy**2 + req.dvz**2)
        if _TRACKER_AVAILABLE and satellite_tracker is not None:
            try:
                state = _propagator.propagate_one(req.norad_id)
                sat_name = state.name if state is not None else f"SAT-{req.norad_id}"
                satellite_tracker.record_maneuver(
                    req.norad_id, delta_v_ms, req.frame, sat_name
                )
            except Exception as tracker_err:
                import logging as _log
                _log.getLogger(__name__).warning("Tracker record_maneuver failed: %s", tracker_err)
        # Same pipeline as the periodic refresh (screening + debris + ML + cascade).
        payload = await _recompute_alerts_pipeline(_propagator)
        result = {
            **result,
            **{k: payload.get(k, default) for k, default in (
                ("alerts", []), ("hotspots", []), ("cascade_plan", []), ("graph", {}),
                ("cascade_depth", 0), ("total_delta_v_ms", 0.0), ("agencies_involved", []),
                ("seed_satellites", []), ("cpi_threshold", 5.0), ("node_probabilities", {}),
                ("ranker_review", {}), ("debris_clouds", []),
            )},
        }

    return result


# The canonical alert refresh lives in main.py (screening + debris alerts + ML
# + cascade). main.py registers it here at startup to avoid a circular import.
_alerts_refresher = None


def set_alerts_refresher(fn) -> None:
    """Register main.refresh_alerts_once as the single alert pipeline."""
    global _alerts_refresher
    _alerts_refresher = fn


async def _recompute_alerts_pipeline(propagator) -> dict:
    """Rebuild snapshot + conjunction/cascade/debris alerts after a state change."""
    snapshot = await asyncio.to_thread(
        _build_snapshot,
        propagator,
        sim_clock.simulation_now(),
    )
    set_latest_snapshot(snapshot)

    if _alerts_refresher is not None:
        # Same pipeline as the periodic refresh, so a clock change or burn
        # never publishes a payload without debris alerts / ML / cascade.
        await _alerts_refresher()
        return get_latest_alerts()

    # Fallback when main.py has not registered its refresher (e.g. routes used
    # stand-alone in a test): same future-window screening, no debris/ML.
    sampled_states = snapshot.get("states", [])
    sim_now = datetime.fromisoformat(snapshot["timestamp"])
    alerts = await asyncio.to_thread(
        screen_alerts, sampled_states, sim_now, propagator=propagator,
    )
    cascade_summary = await asyncio.to_thread(
        _cascade_planner.analyze_snapshot,
        sampled_states,
        alerts,
        propagator,
        sim_now,
    )

    debris_context_states = sampled_states
    top_hotspot_tca = None
    for hotspot in cascade_summary.get("hotspots", []):
        if hotspot.get("tca_utc"):
            top_hotspot_tca = hotspot["tca_utc"]
            break
    if top_hotspot_tca:
        debris_context_states = await asyncio.to_thread(
            propagate_states_to,
            propagator,
            sampled_states,
            top_hotspot_tca,
        )

    debris_clouds = build_debris_alerts(
        cascade_summary.get("hotspots", []),
        debris_context_states,
        snapshot.get("timestamp"),
        alerts=cascade_summary.get("alerts", alerts),
        cpi_threshold=cascade_summary.get("cpi_threshold", 5.0),
    )

    serialized_alerts = cascade_summary.get("alerts", alerts)
    hotspots = cascade_summary.get("hotspots", [])
    enrich_payload_with_geodetic(
        {"hotspots": hotspots, "debris_clouds": debris_clouds},
        datetime.fromisoformat(snapshot["timestamp"]),
    )
    payload = {
        "timestamp": snapshot.get("timestamp"),
        "alerts": serialized_alerts,
        "count": len(serialized_alerts),
        "hotspots": cascade_summary.get("hotspots", []),
        "graph": cascade_summary.get("graph", {}),
        "cascade_plan": cascade_summary.get("cascade_plan", []),
        "cascade_depth": cascade_summary.get("cascade_depth", 0),
        "total_delta_v_ms": cascade_summary.get("total_delta_v_ms", 0.0),
        "agencies_involved": cascade_summary.get("agencies_involved", []),
        "seed_satellites": cascade_summary.get("seed_satellites", []),
        "cpi_threshold": cascade_summary.get("cpi_threshold", 5.0),
        "node_probabilities": cascade_summary.get("node_probabilities", {}),
        "ranker_review": cascade_summary.get("ranker_review", {}),
        "debris_clouds": debris_clouds,
    }
    set_latest_alerts(payload)
    return payload


@router.post("/orbit-change")
async def orbit_change(
    req: OrbitChangeRequest,
    session_id: str = Depends(resolve_session_id),
):
    """
    Execute the first burn of a Hohmann transfer toward a target circular
    altitude. The second (circularization) burn is computed and returned, to be
    applied at transfer apogee — this demo applies burn 1 immediately.
    """
    if _propagator is None or _sim_engine is None:
        return {"status": "ERROR", "message": "Sim engine not initialized"}

    _require_satellite_authority(session_id, req.norad_id)

    state = _propagator.propagate_one(req.norad_id)
    if state is None or state.error_code != 0:
        return {"status": "ERROR", "message": "Satellite not found"}

    r_norm = float((state.x**2 + state.y**2 + state.z**2) ** 0.5)
    current_alt_km = r_norm - RE
    target_alt_km = max(100.0, float(req.target_altitude_km))
    r2 = RE + target_alt_km
    a_transfer = (r_norm + r2) / 2.0
    v_circ_1 = (MU / r_norm) ** 0.5
    v_transfer_1 = (2.0 * MU / r_norm - MU / a_transfer) ** 0.5
    v_transfer_2 = (2.0 * MU / r2 - MU / a_transfer) ** 0.5
    v_circ_2 = (MU / r2) ** 0.5
    dv1_ms = abs(v_transfer_1 - v_circ_1) * 1000.0
    dv2_ms = abs(v_circ_2 - v_transfer_2) * 1000.0

    prograde = r2 >= r_norm
    result = _sim_engine.apply_maneuver(
        req.norad_id,
        0.0,
        dv1_ms if prograde else -dv1_ms,
        0.0,
        frame="RSW",
    )

    payload = None
    if result.get("status") == "success":
        payload = await _recompute_alerts_pipeline(_propagator)

    return {
        **result,
        "current_altitude_km": round(current_alt_km, 2),
        "target_altitude_km": round(target_alt_km, 2),
        "burn_1_ms": round(dv1_ms, 3),
        "burn_2_ms": round(dv2_ms, 3),
        "transfer_phase": "BURN_1_APPLIED" if result.get("status") == "success" else "NOT_APPLIED",
        "alerts": (payload or {}).get("alerts", []),
        "debris_clouds": (payload or {}).get("debris_clouds", []),
    }


@router.post("/override")
async def state_override(
    req: OverrideRequest,
    session_id: str = Depends(resolve_session_id),
):
    """Apply a direct ECI state override (simulation mode only)."""
    if _propagator is None:
        return {"status": "ERROR", "message": "Propagator not initialized"}

    _require_satellite_authority(session_id, req.norad_id)

    result = _propagator.apply_state_override(
        req.norad_id,
        req.x, req.y, req.z,
        req.vx, req.vy, req.vz,
    )
    if result.get("status") != "success":
        return result

    payload = await _recompute_alerts_pipeline(_propagator)
    return {**result, **payload}


@router.post("/feedback/maneuver")
async def maneuver_feedback(
    req: ManeuverFeedbackRequest,
    session_id: str = Depends(resolve_session_id),
):
    """Record an operator decision and (for approve/modify) apply the manoeuvre.

    The burn applied is, in order of preference:
      1. the alert's server-side ``recommended_maneuver`` (looked up by alert_id
         in the latest alert cache; computed by re-propagation in the cascade
         planner) - full RSW vector on its ``sat_id``;
      2. the RSW vector the client sent (``delta_v_rsw_ms`` on ``sat_id``,
         falling back to ``sat1_id``);
      3. legacy: a scalar along-track burn of ``delta_v_ms`` on ``sat1_id``.
    For MODIFY, the chosen direction is kept and rescaled to ``delta_v_ms``.
    """
    import math as _math

    if _propagator is None or _sim_engine is None:
        return {"status": "ERROR", "message": "Not initialized"}

    recommended = None
    if req.alert_id:
        for cached in get_latest_alerts().get("alerts", []) or []:
            if cached.get("id") == req.alert_id and cached.get("recommended_maneuver"):
                recommended = cached["recommended_maneuver"]
                break

    if recommended is not None:
        target_id = int(recommended["sat_id"])
        dv_rsw = [float(x) for x in recommended["delta_v_rsw_ms"]]
        maneuver_source = "server_recommended_maneuver"
    elif req.delta_v_rsw_ms and len(req.delta_v_rsw_ms) == 3:
        target_id = int(req.sat_id if req.sat_id is not None else req.sat1_id)
        dv_rsw = [float(x) for x in req.delta_v_rsw_ms]
        maneuver_source = "client_rsw_vector"
    else:
        target_id = int(req.sat_id if req.sat_id is not None else req.sat1_id)
        dv_rsw = [0.0, float(req.delta_v_ms), 0.0]
        maneuver_source = "operator_scalar_along_track"

    if req.decision == "MODIFY":
        norm = _math.sqrt(sum(x * x for x in dv_rsw))
        if norm > 0:
            dv_rsw = [x * float(req.delta_v_ms) / norm for x in dv_rsw]
            maneuver_source += "+operator_rescaled"

    _require_satellite_authority(session_id, target_id)

    # Persist the decision for the RLHF counters exposed by /api/ml/status.
    try:
        from app.ml.rlhf_store import record_decision
        await run_in_threadpool(
            record_decision,
            req.decision,
            alert_id=req.alert_id,
            sat1_id=req.sat1_id,
            sat2_id=req.sat2_id,
            delta_v_ms=req.delta_v_ms,
        )
    except Exception as error:  # never block a maneuver on bookkeeping
        logger.warning("Failed to record operator decision: %s", error)

    executed = None
    if req.decision in ("APPROVE", "MODIFY"):
        executed = _sim_engine.apply_maneuver(
            target_id,
            dv_rsw[0],
            dv_rsw[1],
            dv_rsw[2],
            frame="RSW",
            session_id=session_id,
        )

    payload = None
    if executed is not None and executed.get("status") == "success":
        payload = await _recompute_alerts_pipeline(_propagator)

    return {
        "status": "OK" if executed is None else executed.get("status"),
        "decision": req.decision,
        "alert_id": req.alert_id,
        "sat1_id": req.sat1_id,
        "sat_id": target_id,
        "delta_v_rsw_ms": [round(x, 5) for x in dv_rsw],
        "maneuver_source": maneuver_source,
        "recommended_maneuver": recommended,
        "executed_maneuver": executed,
        "alerts": (payload or {}).get("alerts", []),
    }


@router.post("/simulate")
async def trigger_scenario(
    req: ScenarioRequest,
    session_id: str = Depends(resolve_session_id),
):
    """Load a deterministic simulation scenario."""
    if _sim_engine is None:
        return {"status": "ERROR", "message": "Sim engine not initialized"}

    _require_valid_session(session_id)

    name = req.name or req.scenario
    if not name:
        raise HTTPException(status_code=400, detail="Scenario name is required")

    # Map cascade_demo to the actual collision_hotspot scenario
    if name == "cascade_demo":
        name = "collision_hotspot"

    try:
        result = _sim_engine.load_scenario(name)
        # Update snapshot immediately so the new satellites show up on the UI
        if _propagator:
            snapshot = await asyncio.to_thread(_build_snapshot, _propagator, sim_clock.simulation_now())
            set_latest_snapshot(snapshot)
        return {"status": "LOADED", **result}
    except (FileNotFoundError, ValueError) as e:
        return {"status": "ERROR", "message": str(e)}


@router.delete("/simulate")
async def clear_scenario(session_id: str = Depends(resolve_session_id)):
    """Clear the active scenario."""
    _require_valid_session(session_id)
    if _sim_engine:
        _sim_engine.clear_scenario()
    return {"status": "CLEARED"}


def _drop_debris_alerts() -> None:
    """Remove published fragment alerts/clouds once all debris events are cleared."""
    current = dict(get_latest_alerts() or {})
    if not current:
        return
    current["alerts"] = [a for a in current.get("alerts", []) or [] if a.get("source") != "debris"]
    current["count"] = len(current["alerts"])
    current["debris_alert_count"] = 0
    set_latest_alerts(current)


class DebrisSimulateRequest(BaseModel):
    sat_a: int | None = None
    sat_b: int | None = None
    tca_utc: str | None = None
    window_hours: float = 24.0


def _default_debris_pair() -> tuple[int, int, str | None, str]:
    """Scenario pair if a computed crossing is loaded, else the highest-Pc alert."""
    pair = getattr(_sim_engine, "scenario_pair", None) if _sim_engine else None
    if pair:
        return int(pair[0]), int(pair[1]), None, "scenario_pair"
    tracked = set(_propagator.norad_ids) if _propagator else set()
    best = None
    for alert in (get_latest_alerts() or {}).get("alerts", []) or []:
        if alert.get("source") == "debris":
            continue
        a = (alert.get("sat1") or {}).get("id")
        b = (alert.get("sat2") or {}).get("id")
        if a not in tracked or b not in tracked:
            continue
        pc = float(alert.get("probability_of_collision", alert.get("p_collision", 0.0)) or 0.0)
        if best is None or pc > best[0]:
            best = (pc, int(a), int(b), alert.get("tca_utc"))
    if best is None:
        raise HTTPException(status_code=409, detail="No pair given, no scenario loaded and no screened alert available")
    return best[1], best[2], best[3], "highest_pc_alert"


@router.post("/debris/simulate")
async def simulate_collision_event(
    req: DebrisSimulateRequest | None = None,
    session_id: str = Depends(resolve_session_id),
):
    """Break up a catalogue pair at its predicted TCA (NASA SBM).

    Body (optional): {sat_a, sat_b, tca_utc?}. Default pair: the loaded
    computed-crossing scenario pair, else the highest-Pc screened alert.
    """
    if _propagator is None:
        raise HTTPException(status_code=503, detail="Propagator not initialized")

    _require_valid_session(session_id)

    from app.core.debris_model import debris_model
    req = req or DebrisSimulateRequest()
    if req.sat_a is not None and req.sat_b is not None:
        sat_a, sat_b, hint, pair_source = int(req.sat_a), int(req.sat_b), req.tca_utc, "request"
    else:
        sat_a, sat_b, hint, pair_source = _default_debris_pair()

    now = sim_clock.simulation_now()
    try:
        event = await asyncio.to_thread(
            debris_model.simulate_collision_from_pair, sat_a, sat_b, _propagator, now,
            tca_hint_utc=hint, window_hours=req.window_hours,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    # All heavy work off the event loop. One snapshot (states) feeds the
    # fragment screening; the clouds are re-read from the debris model's cache
    # so cloud.affected_satellites and the debris alerts come from the same run.
    snapshot = await asyncio.to_thread(_build_snapshot, _propagator, now)
    debris_alerts = await asyncio.to_thread(
        debris_model.compute_debris_alerts, snapshot.get("states", []), now,
    )
    snapshot = {**snapshot, "debris_clouds": debris_model.get_frontend_debris_clouds()}
    try:
        payload = json.loads(snapshot["payload"])
        payload["debris_clouds"] = snapshot["debris_clouds"]
        snapshot["payload"] = json.dumps(payload, separators=(",", ":"))
    except Exception:
        pass
    set_latest_snapshot(snapshot)
    # Publish the debris alerts now instead of waiting for the next 30 s refresh.
    current = dict(get_latest_alerts() or {})
    kept = [a for a in current.get("alerts", []) or [] if a.get("source") != "debris"]
    current["alerts"] = kept + debris_alerts
    current["count"] = len(current["alerts"])
    current["debris_alert_count"] = len(debris_alerts)
    set_latest_alerts(current)

    return {
        "status": "SUCCESS",
        "pair_source": pair_source,
        "event": event,
        "fragments_generated": event["simulated_fragments"],
        "total_fragments": event["total_fragments"],
        "debris_alerts": len(debris_alerts),
        "debris_screening": debris_model.last_screen_meta,
    }


@router.delete("/debris/active")
async def clear_debris(session_id: str = Depends(resolve_session_id)):
    """Clear all active debris clouds."""
    if _propagator is None:
        raise HTTPException(status_code=503, detail="Propagator not initialized")

    _require_valid_session(session_id)

    from app.core.debris_model import debris_model
    debris_model.clear()
    _drop_debris_alerts()
    
    snapshot = await asyncio.to_thread(_build_snapshot, _propagator, sim_clock.simulation_now())
    set_latest_snapshot(snapshot)
    
    return {"status": "CLEARED"}


@router.post("/simulation/reset")
async def reset_simulation(session_id: str = Depends(resolve_session_id)):
    """Reset the simulation and clear debris."""
    if _propagator is None:
        raise HTTPException(status_code=503, detail="Propagator not initialized")

    _require_valid_session(session_id)

    if _sim_engine:
        _sim_engine.clear_scenario()
    from app.core.debris_model import debris_model
    debris_model.clear()
    _drop_debris_alerts()
    
    snapshot = await asyncio.to_thread(_build_snapshot, _propagator, sim_clock.simulation_now())
    set_latest_snapshot(snapshot)
    
    return {"status": "RESET"}


@router.get("/scenarios")
async def list_scenarios():
    """List available simulation scenarios."""
    if _sim_engine is None:
        return {"scenarios": []}
    return {"scenarios": _sim_engine.list_scenarios()}


AGENCY_COLORS = {
    "SpaceX": "#4a90d9",
    "NASA": "#1d9e75",
    "ISS": "#e24b4a",
    "ESA": "#8f5fd6",
    "India": "#ef9f27",
    "Russia/CIS": "#c94f4f",
    "China": "#d9534f",
    "USA": "#5b8def",
    "US Space Force": "#3b5fa0",
    "NOAA": "#3fa9c9",
    "Iridium": "#f5c518",
    "OneWeb": "#9bc53d",
    "Amazon": "#ff9900",
    "Japan": "#e8a0bf",
    "France": "#6c8ebf",
    "UK": "#7f6fbf",
    "Unknown": "#8a8f98",
}


@router.get("/agencies")
async def list_agencies():
    """Agencies in the current catalog, attributed from CelesTrak SATCAT owners
    (operator name patterns refine e.g. US-owned STARLINK -> SpaceX)."""
    if _propagator is None:
        return {"agencies": []}

    from app.core.agency import agency_attribution

    try:
        with _propagator._lock:
            catalog = [(nid, entry[1]) for nid, entry in _propagator._satellites.items()]
    except Exception:
        catalog = []

    counts: dict[str, int] = {}
    owners: dict[str, dict[str, int]] = {}
    sources: dict[str, int] = {}
    for norad_id, name in catalog:
        info = agency_attribution(name, norad_id)
        agency = info["agency"]
        counts[agency] = counts.get(agency, 0) + 1
        if info["owner"]:
            owners.setdefault(agency, {})
            owners[agency][info["owner"]] = owners[agency].get(info["owner"], 0) + 1
        sources[info["agency_source"]] = sources.get(info["agency_source"], 0) + 1

    agencies = [
        {
            "name": name,
            "count": count,
            "color": AGENCY_COLORS.get(name, "#8a8f98"),
            "satcat_owner_codes": owners.get(name, {}),
        }
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    total = sum(counts.values())
    return {
        "agencies": agencies,
        "total": total,
        "unknown_fraction": round(counts.get("Unknown", 0) / total, 4) if total else 0.0,
        "attribution_sources": sources,
    }


class SimulationTimeRequest(BaseModel):
    offset_hours: float = 0.0


@router.get("/simulation/time")
async def get_simulation_time():
    """Get the active simulation time offset."""
    return {
        "offset_hours": sim_clock.get_offset_hours(),
        "simulation_time": sim_clock.simulation_now().isoformat(),
        "real_time": sim_clock.real_now().isoformat(),
    }


@router.post("/simulation/time")
async def set_simulation_time(
    req: SimulationTimeRequest,
    session_id: str = Depends(resolve_session_id),
):
    """
    Shift the simulation clock and rebuild the snapshot at that epoch.

    The offset is bounded so a typo cannot request an absurd epoch, and it is
    stored on the shared clock so the background refresh loop keeps honouring
    it instead of snapping the view back to wall-clock UTC.
    """
    _require_valid_session(session_id)

    try:
        sim_clock.set_offset_hours(req.offset_hours)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if _propagator is None:
        raise HTTPException(status_code=503, detail="Propagator not initialized")

    sim_time = sim_clock.simulation_now()
    snapshot = await asyncio.to_thread(_build_snapshot, _propagator, sim_time)
    set_latest_snapshot(snapshot)

    alerts = 0
    try:
        payload = await _recompute_alerts_pipeline(_propagator)
        alerts = int(payload.get("count", 0) or 0)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Alert recompute failed: {error}") from error

    return {
        "status": "OK",
        "offset_hours": sim_clock.get_offset_hours(),
        "simulation_time": sim_time.isoformat(),
        "alerts_found": alerts,
    }


class UplinkCommandRequest(BaseModel):
    type: str
    delta_v_ms: float | None = None
    direction: str | None = None
    altitude_km: float | None = None


@router.post("/satellites/{norad_id}/uplink")
async def satellite_uplink(
    norad_id: int,
    req: UplinkCommandRequest,
    session_id: str = Depends(resolve_session_id),
):
    """
    Accept an uplink command for a satellite.

    MANEUVER and SET_ORBIT are executed through the same validated path as the
    REST maneuver endpoints; any other command is acknowledged without effect
    so the link protocol stays explicit.
    """
    if _propagator is None or _sim_engine is None:
        raise HTTPException(status_code=503, detail="Runtime not initialized")

    state = _propagator.propagate_one(norad_id)
    if state is None or state.error_code != 0:
        raise HTTPException(status_code=404, detail=f"Satellite {norad_id} not found")

    _require_satellite_authority(session_id, norad_id)

    if req.type == "MANEUVER":
        delta_v = float(req.delta_v_ms or 0.0)
        components = {
            "PROGRADE": (0.0, delta_v, 0.0),
            "RETROGRADE": (0.0, -delta_v, 0.0),
            "RADIAL": (delta_v, 0.0, 0.0),
            "NORMAL": (0.0, 0.0, delta_v),
        }
        dvx, dvy, dvz = components.get(req.direction or "PROGRADE", (0.0, delta_v, 0.0))
        result = _sim_engine.apply_maneuver(
            norad_id, dvx, dvy, dvz, frame="RSW", session_id=session_id
        )
        if result.get("status") == "success":
            await _recompute_alerts_pipeline(_propagator)
        return {"result": result}

    if req.type == "SET_ORBIT":
        return {
            "result": await orbit_change(
                OrbitChangeRequest(
                    norad_id=norad_id,
                    target_altitude_km=float(req.altitude_km or 550.0),
                ),
                session_id=session_id,
            )
        }

    return {
        "result": {
            "success": True,
            "message": f"Command '{req.type}' acknowledged (no state change)",
            "status": "ACK",
        }
    }


@router.post("/preflight")
async def preflight_check(req: ManeuverRequest):
    """Run pre-flight validation without executing."""
    if _sim_engine is None:
        return {"status": "ERROR"}

    result = _sim_engine.preflight_check(req.norad_id, req.dvx, req.dvy, req.dvz, frame=req.frame)
    return result


@router.post("/judge/manipulate-satellite")
async def judge_manipulate_satellite(
    req: JudgeManipulateRequest,
    session_id: str = Depends(resolve_session_id),
):
    """
    Judge manipulation mode: override a satellite's state and immediately predict
    downstream effects, neighbor risks, and cascade implications.
    Returns: overridden state, KD-tree neighbors within 200km, CPI scores, and alerts.
    """
    if _propagator is None:
        raise HTTPException(status_code=503, detail="Propagator not initialized")

    _require_satellite_authority(session_id, req.norad_id)

    now = datetime.now(timezone.utc)
    
    # Build base snapshot from current constellation
    snapshot = await asyncio.to_thread(_build_snapshot, _propagator, now)
    states = snapshot.get("states", [])
    
    # Find and override the target satellite
    target_idx = None
    target_state = None
    for idx, state in enumerate(states):
        if state.norad_id == req.norad_id:
            target_idx = idx
            target_state = state
            break
    
    if target_state is None:
        raise HTTPException(status_code=404, detail=f"Satellite {req.norad_id} not found")
    
    # Apply override if provided
    if req.position_eci_km is not None and len(req.position_eci_km) == 3:
        target_state.x, target_state.y, target_state.z = req.position_eci_km
    
    if req.velocity_eci_kms is not None and len(req.velocity_eci_kms) == 3:
        target_state.vx, target_state.vy, target_state.vz = req.velocity_eci_kms
    
    # Rebuild satellite list with override
    satellites = [
        {
            "norad_id": state.norad_id,
            "name": state.name,
            "position": {
                "x": state.x,
                "y": state.y,
                "z": state.z,
            },
            "velocity": {
                "vx": state.vx,
                "vy": state.vy,
                "vz": state.vz,
            },
            "speed_kmh": round(float((state.vx**2 + state.vy**2 + state.vz**2) ** 0.5 * 3600.0), 2),
            "altitude_km": round(float(((state.x**2 + state.y**2 + state.z**2) ** 0.5) - 6371.0), 2),
            "epoch_utc": state.epoch_utc,
        }
        for state in states
    ]
    
    # Recompute alerts and cascade plan with the overridden state
    alerts = []
    cascade_summary = {
        "graph": {"node_count": 0, "edge_count": 0, "influence_radius_km": 200.0},
        "alerts": [],
        "cascade_plan": [],
        "total_delta_v_ms": 0.0,
        "cascade_depth": 0,
        "agencies_involved": [],
        "seed_satellites": [],
        "cpi_threshold": 5.0,
        "node_probabilities": {},
        "ranker_review": {},
    }
    
    sampled_states = []
    if states:
        # Screen the full catalogue (the vectorised screen handles it).
        sampled_states = states
        
        kalman_states = get_all_kalman_states()
        alerts = await asyncio.to_thread(screen_conjunctions, sampled_states, kalman_states=kalman_states, propagator=_propagator)
        cascade_summary = await asyncio.to_thread(
            _cascade_planner.analyze_snapshot,
            sampled_states,
            [alert.to_dict() for alert in alerts],
        )

    debris_clouds = build_debris_alerts(
        cascade_summary.get("hotspots", []),
        sampled_states if states else [],
        snapshot.get("timestamp"),
        alerts=cascade_summary.get("alerts", []),
        cpi_threshold=cascade_summary.get("cpi_threshold", 5.0),
    )
    
    # Build a neighbor map: all satellites within 200 km of the target
    target_pos = [target_state.x, target_state.y, target_state.z]
    neighbors = []
    
    for state in states:
        if state.norad_id == req.norad_id:
            continue
        
        other_pos = [state.x, state.y, state.z]
        distance_km = ((other_pos[0] - target_pos[0])**2 + 
                       (other_pos[1] - target_pos[1])**2 + 
                       (other_pos[2] - target_pos[2])**2) ** 0.5
        
        if distance_km <= 200.0:
            # Find CPI score for this pair in the alerts
            cpi_score = 0.0
            for alert in cascade_summary.get("alerts", []):
                if ((alert["sat1"]["id"] == req.norad_id and alert["sat2"]["id"] == state.norad_id) or
                    (alert["sat1"]["id"] == state.norad_id and alert["sat2"]["id"] == req.norad_id)):
                    cpi_score = alert.get("cpi_score", 0.0)
                    break
            
            neighbors.append({
                "satellite_id": state.norad_id,
                "satellite_name": state.name,
                "distance_km": round(distance_km, 3),
                "position": {
                    "x": state.x,
                    "y": state.y,
                    "z": state.z,
                },
                "velocity": {
                    "vx": state.vx,
                    "vy": state.vy,
                    "vz": state.vz,
                },
                "speed_kmh": round(float((state.vx**2 + state.vy**2 + state.vz**2) ** 0.5 * 3600.0), 2),
                "cpi_score": round(cpi_score, 2),
                "collision_risk": "high" if cpi_score >= 7.0 else "medium" if cpi_score >= 5.0 else "low",
            })
    
    # Sort neighbors by distance
    neighbors.sort(key=lambda n: n["distance_km"])
    
    return {
        "status": "success",
        "manipulated_satellite": {
            "norad_id": req.norad_id,
            "name": target_state.name,
            "position": {
                "x": target_state.x,
                "y": target_state.y,
                "z": target_state.z,
            },
            "velocity": {
                "vx": target_state.vx,
                "vy": target_state.vy,
                "vz": target_state.vz,
            },
            "speed_kmh": round(float((target_state.vx**2 + target_state.vy**2 + target_state.vz**2) ** 0.5 * 3600.0), 2),
            "altitude_km": round(float(((target_state.x**2 + target_state.y**2 + target_state.z**2) ** 0.5) - 6371.0), 2),
        },
        "neighbors_within_200km": neighbors,
        "neighbor_count": len(neighbors),
        "alerts": cascade_summary.get("alerts", []),
        "cascade_plan": cascade_summary.get("cascade_plan", []),
        "graph": cascade_summary.get("graph", {}),
        "cascade_depth": cascade_summary.get("cascade_depth", 0),
        "total_delta_v_ms": cascade_summary.get("total_delta_v_ms", 0.0),
        "agencies_involved": cascade_summary.get("agencies_involved", []),
        "debris_clouds": debris_clouds,
    }
