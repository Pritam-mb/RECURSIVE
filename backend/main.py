"""
Orbital Sentinel — FastAPI Entry Point.
Initializes propagator, sim engine, WebSocket broadcast, and REST routes.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.core.sgp4_propagator import SGP4Propagator
from app.core.agency import infer_agency
from app.core import sim_clock
from app.core.screening import screen as screen_alerts, set_default_propagator
from app.core.state_cache import (
    get_latest_snapshot,
    set_latest_alerts,
    set_latest_snapshot,
    get_all_kalman_states,  # noqa: F401  (diagnostics only; never used for Pc)
)
from app.data.tle_fetcher import fetch_tles
from app.simulation.sim_engine import SimEngine
from app.services.cascade_planner import CascadePlanner
from app.services.debris_model import build_debris_alerts
from app.services.conjunction_predictor import propagate_states_to, enrich_payload_with_geodetic
from app.api.ws_handler import ConnectionManager, satellite_broadcast_loop
from app.api.routes import router, init_routes, set_alerts_refresher
from app.routers.test_mode import router as test_mode_router, init_test_mode
from app.routers.predict import router as predict_router, init_predict_router
from app.ml.runtime import get_ml_runtime
from app.streaming.kafka_adapter import KafkaAdapter

_BACKEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BACKEND_DIR.parent


def load_environment() -> list[Path]:
    """Load .env files resolved from this file's location, not the cwd.

    ``backend/.env`` is loaded first, then the repo-root ``.env`` as a
    fallback. ``override=False`` everywhere, so real environment variables and
    earlier files win. Set ORBIT_SENTINEL_SKIP_DOTENV=1 to skip loading
    entirely (the test suite does this so a developer's local .env cannot flip
    feature flags such as ENABLE_EXTENDED_PIPELINE under the tests).
    """
    if os.getenv("ORBIT_SENTINEL_SKIP_DOTENV", "0") == "1":
        return []
    loaded: list[Path] = []
    for candidate in (_BACKEND_DIR / ".env", _REPO_ROOT / ".env"):
        if candidate.is_file():
            load_dotenv(dotenv_path=candidate, override=False)
            loaded.append(candidate)
    return loaded


_LOADED_ENV_FILES = load_environment()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if _LOADED_ENV_FILES:
    logger.info("Loaded environment from: %s", ", ".join(str(p) for p in _LOADED_ENV_FILES))

# Global instances
propagator = SGP4Propagator()
sim_engine = SimEngine(propagator)
ws_manager = ConnectionManager()
cascade_planner = CascadePlanner()
USE_APSCHEDULER = os.getenv("USE_APSCHEDULER", "0") == "1"
SNAPSHOT_REFRESH_SECONDS = float(os.getenv("SNAPSHOT_REFRESH_SECONDS", "1"))
ALERT_REFRESH_SECONDS = float(os.getenv("ALERT_REFRESH_SECONDS", "30"))
ENABLE_EXTENDED_PIPELINE = os.getenv("ENABLE_EXTENDED_PIPELINE", "0") == "1"

# Only construct the ML runtime when the extended pipeline is enabled.
# get_ml_runtime() is an lru_cache singleton: calling it eagerly would load or
# train models, create the shadow database, and write artifacts on every
# startup even when the feature is disabled.
ml_runtime = get_ml_runtime() if ENABLE_EXTENDED_PIPELINE else None
kafka_adapter = KafkaAdapter() if ENABLE_EXTENDED_PIPELINE else None


def build_snapshot(current_propagator: SGP4Propagator, dt: datetime) -> dict:
    """Compute a serializable snapshot for the latest satellite state."""
    from app.core.state_cache import set_kalman_state
    
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

    # Capture Kalman covariance per satellite
    covariances = {}
    for state in active_states:
        kal_state = current_propagator._kalman_states.get(state.norad_id)
        if kal_state is not None:
            kal_dict = kal_state.to_dict()
            covariances[state.norad_id] = kal_dict.get("covariance_6x6")
            set_kalman_state(state.norad_id, kal_dict)

    from app.core.debris_model import debris_model
    debris_clouds = debris_model.get_frontend_debris_clouds(active_states)

    snapshot = {
        "type": "update",
        "timestamp": dt.isoformat(),
        "satellites": satellites,
        "count": len(states),
        "states": active_states,
        "covariances": covariances,
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


async def refresh_snapshot_once():
    try:
        now = sim_clock.simulation_now()

        # Propagate debris fragments in the core model
        from app.core.debris_model import debris_model
        for event_id in debris_model.list_event_ids():
            await asyncio.to_thread(debris_model.propagate_fragments, event_id, dt_seconds=1.0)

        snapshot = await asyncio.to_thread(build_snapshot, propagator, now)
        set_latest_snapshot(snapshot)
        if ENABLE_EXTENDED_PIPELINE and ml_runtime is not None and snapshot.get("states"):
            ml_runtime.record_snapshot(snapshot.get("states", []), now)
        payload = snapshot.get("payload")
        if ENABLE_EXTENDED_PIPELINE and kafka_adapter is not None and payload:
            kafka_adapter.publish_snapshot(payload)
    except Exception as error:
        logger.error(f"Snapshot refresh error: {error}")


async def refresh_snapshot_loop():
    """Keep the latest propagated satellite snapshot warm in memory."""
    while True:
        try:
            await refresh_snapshot_once()
        except asyncio.CancelledError:
            raise
        await asyncio.sleep(SNAPSHOT_REFRESH_SECONDS)


def _alert_pair_key(alert: dict) -> tuple:
    sat1 = (alert.get("sat1") or {}).get("id")
    sat2 = (alert.get("sat2") or {}).get("id")
    if sat1 is None or sat2 is None:
        return ()
    return (sat1, sat2) if sat1 <= sat2 else (sat2, sat1)


# Fields computed by physics (screening / debris propagation). When the cascade
# planner returns its own alert for the same pair, these values are NOT
# overwritten; the planner only adds enrichment (cascade depth, manoeuvre...).
_PHYSICS_KEYS = frozenset({
    "id", "source", "sat1", "sat2", "tca_utc", "tca_hours", "tca_minutes",
    "miss_distance_km", "relative_speed_kmh", "relative_speed_kms",
    "probability_of_collision", "p_collision", "pc_method", "hbr_km",
    "covariance_ellipse", "b_t_km", "b_n_km", "bt_km", "bn_km", "sigma_source",
    "severity", "cpi_score", "tle_age_days", "short_encounter_valid",
    "parent_event", "fragment_id", "tca_position_km",
})


def _tca_hours_from_utc(tca_utc, reference: datetime | None):
    if not tca_utc or reference is None:
        return None
    try:
        tca_dt = datetime.fromisoformat(str(tca_utc))
        if tca_dt.tzinfo is None:
            tca_dt = tca_dt.replace(tzinfo=timezone.utc)
        ref = reference if reference.tzinfo else reference.replace(tzinfo=timezone.utc)
        return max(0.0, (tca_dt - ref).total_seconds() / 3600.0)
    except Exception:
        return None


def _normalize_screened_alert(alert: dict, reference: datetime | None = None) -> dict:
    """Project a physics alert (screening or debris) onto the graph-edge shape.

    All computed fields are preserved verbatim. tca_hours is never invented:
    if absent it is derived from tca_utc against the simulation epoch, else
    left as None. Graph-only fields get neutral defaults, and the hotspot
    position is the computed midpoint of the two objects at TCA when known.
    """
    out = dict(alert)
    tca_hours = out.get("tca_hours")
    if tca_hours is None:
        tca_hours = _tca_hours_from_utc(out.get("tca_utc"), reference)
    if tca_hours is not None:
        tca_hours = float(tca_hours)
        out["tca_hours"] = round(tca_hours, 4)
        if out.get("tca_minutes") is None:
            out["tca_minutes"] = round(tca_hours * 60.0, 2)
    else:
        out["tca_hours"] = None
        out.setdefault("tca_minutes", None)
    pc = out.get("probability_of_collision", out.get("p_collision", 0.0))
    out["probability_of_collision"] = pc
    out["p_collision"] = pc
    out.setdefault("severity", "WATCH")
    out.setdefault("cpi_score", 0.0)
    out.setdefault("hotspot_score", 0.0)
    out.setdefault("influence_weight", 0.0)
    pos = out.get("position")
    if pos is None and out.get("tca_position_km"):
        x, y, z = out["tca_position_km"]
        pos = {"x": x, "y": y, "z": z}
    out["position"] = pos
    out.setdefault("zone_radius_km", 100.0)
    out.setdefault("covariance_ellipse", None)
    out.setdefault("source", "screening")
    return out


def merge_alert_sources(
    graph_alerts: list[dict] | None,
    screened_alerts: list[dict] | None,
    reference: datetime | None = None,
) -> list[dict]:
    """Union physics alerts (screening + debris) with cascade-graph alerts.

    One entry per pair. Physics alerts are authoritative for every computed
    field (_PHYSICS_KEYS); a cascade-graph alert for the same pair contributes
    only its enrichment fields (cascade_depth, downstream_ids,
    recommended_maneuver, hotspot geometry, ...). Graph-only pairs are kept
    with source "cascade_graph".
    """
    merged: list[dict] = []
    index: dict[tuple, int] = {}

    for alert in screened_alerts or []:
        key = _alert_pair_key(alert) or (alert.get("id"),)
        if key in index:
            continue
        index[key] = len(merged)
        merged.append(_normalize_screened_alert(alert, reference))

    for alert in graph_alerts or []:
        key = _alert_pair_key(alert)
        if key and key in index:
            target = merged[index[key]]
            for k, v in alert.items():
                if k not in _PHYSICS_KEYS and (k not in target or target.get(k) in (None, 0.0, [], {})):
                    target[k] = v
            continue
        if key:
            index[key] = len(merged)
        merged.append({**alert, "source": alert.get("source", "cascade_graph")})

    return merged


def _compute_debris_alerts(states, sim_now):
    """Agent B's fragment-vs-catalogue alerts (feature-detected)."""
    for mod_name in ("app.services.debris_model", "app.core.debris_model"):
        try:
            mod = __import__(mod_name, fromlist=["compute_debris_alerts"])
        except Exception:
            continue
        fn = getattr(mod, "compute_debris_alerts", None)
        if fn is None:
            obj = getattr(mod, "debris_model", None)
            fn = getattr(obj, "compute_debris_alerts", None)
        if fn is not None:
            return list(fn(states, sim_now) or [])
    return []


def _attach_ml(alerts: list[dict]) -> None:
    """Agent D's ML surrogate (never replaces the physics Pc)."""
    try:
        from app.ml.risk_api import score_alerts
        score_alerts(alerts)
        return
    except Exception as exc:
        logger.debug("score_alerts unavailable/failed: %s", exc)
    try:
        from app.ml.risk_api import score_alert
    except Exception:
        for a in alerts:
            a.setdefault("ml", None)
        return
    for a in alerts:
        try:
            a["ml"] = score_alert(a)
        except Exception as exc:
            logger.debug("score_alert failed for %s: %s", a.get("id"), exc)
            a["ml"] = None


async def refresh_alerts_once():
    """Refresh conjunction alerts at a slower cadence (ALERT_REFRESH_SECONDS).

    Screens ALL tracked objects (catalogue + injected scenario objects) over a
    future window on the SIMULATION clock (app.core.screening.screen), adds
    debris-fragment alerts (agent B), ML surrogate scores (agent D), then runs
    the cascade planner (agent C) and publishes. Heavy work runs in threads.
    ENABLE_EXTENDED_PIPELINE gates only the optional ML runtime and Kafka.
    """
    try:
        snapshot = get_latest_snapshot()
        states = snapshot.get("states", [])
        if not states:
            return
        stamp = snapshot.get("timestamp")
        sim_now = datetime.fromisoformat(stamp) if stamp else sim_clock.simulation_now()
        if sim_now.tzinfo is None:
            sim_now = sim_now.replace(tzinfo=timezone.utc)

        screen_stats: dict = {}
        screened = await asyncio.to_thread(
            screen_alerts, states, sim_now, propagator=propagator, stats=screen_stats,
        )
        screened = list(screened or [])

        debris_alerts: list[dict] = []
        try:
            debris_alerts = await asyncio.to_thread(_compute_debris_alerts, states, sim_now)
        except Exception as exc:
            logger.warning("compute_debris_alerts failed: %s", exc)

        physics_alerts = screened + debris_alerts
        _attach_ml(physics_alerts)

        cascade_summary = await asyncio.to_thread(
            cascade_planner.analyze_snapshot,
            states,
            physics_alerts,
            propagator,
            sim_now,
        )

        merged_alerts = merge_alert_sources(
            cascade_summary.get("alerts", []),
            physics_alerts,
            reference=sim_now,
        )

        debris_context_states = states
        top_hotspot_tca = None
        for hotspot in cascade_summary.get("hotspots", []):
            if hotspot.get("tca_utc"):
                top_hotspot_tca = hotspot["tca_utc"]
                break
        if top_hotspot_tca:
            debris_context_states = await asyncio.to_thread(
                propagate_states_to, propagator, states, top_hotspot_tca,
            )

        debris_clouds = build_debris_alerts(
            cascade_summary.get("hotspots", []),
            debris_context_states,
            stamp,
            alerts=merged_alerts,
            cpi_threshold=cascade_summary.get("cpi_threshold", 5.0),
        )
        hotspots = cascade_summary.get("hotspots", [])
        enrich_payload_with_geodetic({"hotspots": hotspots, "debris_clouds": debris_clouds}, sim_now)

        alerts_payload = {
            "type": "alerts",
            "timestamp": stamp,
            "alerts": merged_alerts,
            "count": len(merged_alerts),
            "screened_count": len(screened),
            "debris_alert_count": len(debris_alerts),
            "graph_alert_count": len(cascade_summary.get("alerts", [])),
            "screening": screen_stats,
            "hotspots": hotspots,
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
        set_latest_alerts(alerts_payload)
        if kafka_adapter is not None:
            kafka_adapter.publish_alerts(alerts_payload)
        if ml_runtime is not None:
            ml_runtime.log_risk_samples(alerts_payload.get("alerts", []))
    except Exception as error:
        logger.exception(f"Alert refresh error: {error}")


async def refresh_alerts_loop():
    while True:
        try:
            await refresh_alerts_once()
        except asyncio.CancelledError:
            raise
        await asyncio.sleep(ALERT_REFRESH_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: fetch TLEs and begin broadcast loop."""
    logger.info("=== Orbital Sentinel starting ===")

    # ── Model initialization sequence ─────────────────────────────────────────
    print("=== Orbital Sentinel Startup ===")

    print("Initializing Foster B-plane physics engine...")
    try:
        from app.core import analytics as _analytics_module  # noqa: F401
        print("  Foster B-plane analytics ready")
    except Exception as _analytics_err:
        print(f"  WARNING: analytics.py failed to load: {_analytics_err}")

    print("Initializing satellite state tracker...")
    try:
        from app.core.satellite_state_tracker import satellite_tracker as _tracker
        print(f"  State tracker ready (tracking 0 satellites at startup)")
    except Exception as _tracker_err:
        print(f"  WARNING: satellite_state_tracker failed: {_tracker_err}")

    print("Initializing agency authority model...")
    try:
        from app.core.agency_authority import authority_manager as _auth
        # Create default sessions for each known agency
        for _agency in [
            "SpaceX", "ESA", "ISRO", "ROSCOSMOS",
            "ISS/NASA", "CNSA", "US Space Force",
            "NOAA", "NASA",
        ]:
            _auth.create_session(_agency)
        print(f"  Authority model ready with {len(_auth.sessions)} sessions (DEMO_SESSION + {len(_auth.sessions)-1} agency sessions)")
    except Exception as _auth_err:
        print(f"  WARNING: agency_authority failed: {_auth_err}")

    if ENABLE_EXTENDED_PIPELINE:
        print("Initializing XGBoost risk classifier (loads on first use)...")
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: __import__("app.ml.xgboost_scorer", fromlist=["_trained_model"]),
            )
            print("  XGBoost risk classifier ready")
        except Exception as _xgb_err:
            print(f"  WARNING: XGBoost init failed: {_xgb_err}")
    else:
        # Flag is off: the trained model is never touched at startup, and
        # scoring uses the cheap boosted-stump heuristic.
        print("  XGBoost risk classifier deferred (ENABLE_EXTENDED_PIPELINE=0)")

    print("=== Model initialization complete ===")

    # Load live TLEs when Space-Track credentials are configured, otherwise fall back.
    max_sats = int(os.getenv("MAX_SATS", "500"))
    tles = await fetch_tles(max_sats=max_sats)
    propagator.load_tles(tles)
    set_default_propagator(propagator)
    logger.info(f"Tracking {propagator.satellite_count} satellites from current TLE source")

    now = sim_clock.simulation_now()
    set_latest_snapshot(await asyncio.to_thread(build_snapshot, propagator, now))
    set_latest_alerts({"timestamp": now.isoformat(), "alerts": [], "count": 0, "hotspots": [], "debris_clouds": []})

    # Initialize REST routes with shared instances
    init_routes(propagator, sim_engine)
    set_alerts_refresher(refresh_alerts_once)
    init_test_mode(propagator)
    init_predict_router(propagator)

    scheduler = None
    snapshot_task = None
    alerts_task = None
    if USE_APSCHEDULER:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler

            scheduler = AsyncIOScheduler()
            scheduler.add_job(refresh_snapshot_once, "interval", seconds=SNAPSHOT_REFRESH_SECONDS)
            scheduler.add_job(refresh_alerts_once, "interval", seconds=ALERT_REFRESH_SECONDS)
            if ENABLE_EXTENDED_PIPELINE and ml_runtime is not None:
                shadow_hours = float(os.getenv("SHADOW_RETRAIN_HOURS", "6"))
                scheduler.add_job(ml_runtime.shadow_retrain, "interval", hours=shadow_hours)
            scheduler.start()
            logger.info("APScheduler enabled for snapshot/alert refresh")
        except Exception as error:
            logger.warning("APScheduler unavailable, falling back to loops: %s", error)

    if scheduler is None:
        snapshot_task = asyncio.create_task(refresh_snapshot_loop())
        alerts_task = asyncio.create_task(refresh_alerts_loop())

    broadcast_task = asyncio.create_task(satellite_broadcast_loop(ws_manager))

    yield

    # Shutdown
    if scheduler is not None:
        scheduler.shutdown(wait=False)
    if snapshot_task is not None:
        snapshot_task.cancel()
    if alerts_task is not None:
        alerts_task.cancel()
    broadcast_task.cancel()
    logger.info("=== Orbital Sentinel stopped ===")


app = FastAPI(
    title="Orbital Sentinel API",
    description="Satellite tracking and collision simulation",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST routes
app.include_router(router)
app.include_router(test_mode_router)
app.include_router(predict_router)


@app.get("/")
async def root():
    return {
        "service": "Orbital Sentinel",
        "status": "operational",
        "satellites": propagator.satellite_count,
    }


@app.websocket("/ws/satellites")
async def websocket_satellites(websocket: WebSocket):
    """WebSocket endpoint for real-time satellite position updates.

    The server pushes snapshots from `satellite_broadcast_loop`; this handler
    only drains inbound frames (currently the client's `ping` heartbeat) so the
    connection stays healthy. Frames are logged at debug level rather than
    assigned to an unused local.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            frame = await websocket.receive_text()
            logger.debug("WebSocket frame from client: %s", frame)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
        raise


if __name__ == "__main__":
    import uvicorn
    reload_enabled = os.getenv("UVICORN_RELOAD", "0") == "1"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=reload_enabled)
