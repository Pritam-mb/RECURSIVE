"""
satellite_state_tracker.py — per-object propellant bookkeeping + honest telemetry.

We have NO downlink from any tracked spacecraft, so this module never claims
measured housekeeping telemetry. What it returns:

  * object_type            PAY / R/B / DEB / UNK (SATCAT via app.core.satcat when
                           available, else a documented name heuristic).
  * Rocket bodies, debris and unknown objects carry no bus at all → every
    housekeeping field is ``None`` and ``telemetry_available`` is False.
  * Payloads get a block flagged ``"simulated": True`` containing only values
    that follow from a documented model and real inputs:
      - fuel_remaining_pct  : Tsiolkovsky rocket equation applied to the Δv of
                              manoeuvres actually executed in this session,
                              starting from an ASSUMED full tank (parameters
                              below, echoed in the response under ``model``).
      - illumination        : "sunlit" / "eclipse" computed from the object's
                              propagated position (sim clock) and a low-precision
                              solar ephemeris with a cylindrical Earth shadow.
    battery / temperature / signal / solar power are ``None`` ("not modeled"):
    without a downlink any number there would be invented.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Documented model assumptions (exposed in every payload response) ──────────
ASSUMED_WET_MASS_KG = 500.0         # generic small LEO payload
ASSUMED_PROPELLANT_MASS_KG = 100.0  # 20 % propellant fraction, full at session start
ASSUMED_ISP_S = 220.0               # hydrazine monoprop
G0 = 9.80665
EARTH_RADIUS_KM = 6378.137
AU_KM = 149_597_870.7

NON_PAYLOAD_TYPES = {"R/B", "DEB", "UNK"}


def _sim_now() -> datetime:
    try:
        from app.core.sim_clock import simulation_now
        return simulation_now()
    except Exception:  # pragma: no cover - sim clock always importable in app
        return datetime.now(timezone.utc)


def classify_object(norad_id: int, name: str = "", object_type: str | None = None) -> tuple[str, str]:
    """Return (object_type, source). SATCAT first, then name heuristic."""
    if object_type:
        return str(object_type).upper(), "caller"
    try:
        from app.core.satcat import lookup  # agent C's module; optional
        rec = lookup(int(norad_id))
        if rec and rec.get("object_type"):
            return str(rec["object_type"]).upper(), "satcat"
    except Exception:
        pass
    upper = (name or "").upper()
    tokens = upper.replace("(", " ").replace(")", " ").split()
    if "R/B" in tokens or "ROCKET BODY" in upper or "AKM" in tokens or "PKM" in tokens:
        return "R/B", "name_heuristic"
    if "DEB" in tokens or "DEBRIS" in tokens or "FRAG" in upper or upper.startswith("FRAGMENT"):
        return "DEB", "name_heuristic"
    if not upper or upper.startswith("UNKNOWN") or upper.startswith("TBA"):
        return "UNK", "name_heuristic"
    return "PAY", "name_heuristic"


def sun_position_eci_km(when: datetime) -> tuple[float, float, float]:
    """Low-precision solar ephemeris (Astronomical Almanac, ~0.01 deg)."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    jd = when.timestamp() / 86400.0 + 2440587.5
    n = jd - 2451545.0
    mean_lon = math.radians((280.460 + 0.9856474 * n) % 360.0)
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    ecl_lon = mean_lon + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    r_au = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)
    r = r_au * AU_KM
    return (
        r * math.cos(ecl_lon),
        r * math.cos(eps) * math.sin(ecl_lon),
        r * math.sin(eps) * math.sin(ecl_lon),
    )


def illumination(position_km: tuple[float, float, float], when: datetime) -> str:
    """Cylindrical-shadow test: 'eclipse' if behind Earth inside its shadow cylinder."""
    sx, sy, sz = sun_position_eci_km(when)
    s_norm = math.sqrt(sx * sx + sy * sy + sz * sz)
    ux, uy, uz = sx / s_norm, sy / s_norm, sz / s_norm
    x, y, z = position_km
    along = x * ux + y * uy + z * uz
    if along >= 0:
        return "sunlit"
    px, py, pz = x - along * ux, y - along * uy, z - along * uz
    return "eclipse" if math.sqrt(px * px + py * py + pz * pz) < EARTH_RADIUS_KM else "sunlit"


def propellant_used_kg(mass_kg: float, delta_v_ms: float, isp_s: float = ASSUMED_ISP_S) -> float:
    """Rocket equation: m_prop = m0 * (1 - exp(-dv / (Isp g0)))."""
    return mass_kg * (1.0 - math.exp(-abs(delta_v_ms) / (isp_s * G0)))


class SatelliteStateTracker:
    """Thread-safe propellant ledger keyed by NORAD id."""

    def __init__(self):
        self.states: dict[int, dict[str, Any]] = {}
        self.maneuver_history: dict[int, list] = defaultdict(list)
        self._lock = threading.RLock()

    def _now_iso(self) -> str:
        return _sim_now().isoformat()

    def get_or_create(self, norad_id: int, satellite_name: str = "") -> dict[str, Any]:
        with self._lock:
            if norad_id not in self.states:
                now = self._now_iso()
                self.states[norad_id] = {
                    "norad_id": norad_id,
                    "satellite_name": satellite_name or f"SAT-{norad_id}",
                    "mass_kg": ASSUMED_WET_MASS_KG,
                    "propellant_kg": ASSUMED_PROPELLANT_MASS_KG,
                    "fuel_remaining_pct": 100.0,
                    "total_delta_v_used_ms": 0.0,
                    "maneuver_count": 0,
                    "created_at": now,
                    "last_updated": now,
                }
            return self.states[norad_id]

    def record_maneuver(
        self,
        norad_id: int,
        delta_v_ms: float,
        direction: str = "radial",
        satellite_name: str = "",
    ) -> dict[str, Any]:
        """Deduct propellant for an executed manoeuvre via the rocket equation."""
        with self._lock:
            state = self.get_or_create(norad_id, satellite_name)
            delta_v_ms = abs(float(delta_v_ms))
            used = min(state["propellant_kg"], propellant_used_kg(state["mass_kg"], delta_v_ms))
            old_fuel = state["fuel_remaining_pct"]
            state["propellant_kg"] -= used
            state["mass_kg"] -= used
            state["fuel_remaining_pct"] = round(100.0 * state["propellant_kg"] / ASSUMED_PROPELLANT_MASS_KG, 4)
            state["total_delta_v_used_ms"] = round(state["total_delta_v_used_ms"] + delta_v_ms, 4)
            state["maneuver_count"] += 1
            state["last_updated"] = self._now_iso()

            self.maneuver_history[norad_id].append({
                "timestamp": state["last_updated"],
                "delta_v_ms": round(delta_v_ms, 4),
                "direction": direction,
                "propellant_used_kg": round(used, 5),
                "fuel_before_pct": round(old_fuel, 4),
                "fuel_after_pct": state["fuel_remaining_pct"],
                "maneuver_number": state["maneuver_count"],
                "fuel_model": "rocket_equation",
            })
            logger.info("NORAD %d manoeuvre #%d: dv=%.3f m/s, propellant %.4f kg",
                        norad_id, state["maneuver_count"], delta_v_ms, used)
            return state

    @staticmethod
    def _position_for(norad_id: int, when: datetime):
        """Propagated ECI position (km) at sim time, or None if unavailable."""
        try:
            from app.api import routes as _routes
            prop = getattr(_routes, "_propagator", None)
            if prop is None:
                return None
            st = prop.propagate_one(int(norad_id), when)
            if st is None or getattr(st, "error_code", 0):
                return None
            return (float(st.x), float(st.y), float(st.z))
        except Exception:
            return None

    def get_telemetry(
        self,
        norad_id: int,
        satellite_name: str = "",
        object_type: str | None = None,
        position_km: tuple[float, float, float] | None = None,
    ) -> dict[str, Any]:
        """Telemetry block. Never invents housekeeping values (see module doc)."""
        now = _sim_now()
        obj_type, type_source = classify_object(norad_id, satellite_name, object_type)
        pos = position_km if position_km is not None else self._position_for(norad_id, now)
        illum = illumination(pos, now) if pos is not None else None

        base = {
            "norad_id": norad_id,
            "satellite_name": satellite_name or f"SAT-{norad_id}",
            "object_type": obj_type,
            "object_type_source": type_source,
            "measured": False,
            "illumination": illum,
            "illumination_model": "cylindrical_shadow+low_precision_sun" if illum else None,
            "battery_pct": None,
            "temperature_c": None,
            "signal_strength_dbm": None,
            "solar_power_w": None,
            "sim_time_utc": now.isoformat(),
        }

        if obj_type in NON_PAYLOAD_TYPES:
            return {
                **base,
                "telemetry_available": False,
                "simulated": False,
                "reason": f"{obj_type} object: no spacecraft bus, no propellant",
                "fuel_remaining_pct": None,
                "total_delta_v_used_ms": None,
                "maneuver_count": 0,
            }

        with self._lock:
            state = self.get_or_create(norad_id, satellite_name)
            return {
                **base,
                "telemetry_available": True,
                "simulated": True,
                "fuel_remaining_pct": round(state["fuel_remaining_pct"], 3),
                "propellant_kg": round(state["propellant_kg"], 4),
                "total_delta_v_used_ms": state["total_delta_v_used_ms"],
                "maneuver_count": state["maneuver_count"],
                "last_updated": state["last_updated"],
                "created_at": state["created_at"],
                "model": {
                    "fuel": "rocket_equation on executed delta-v; tank assumed full at session start",
                    "assumed_wet_mass_kg": ASSUMED_WET_MASS_KG,
                    "assumed_propellant_mass_kg": ASSUMED_PROPELLANT_MASS_KG,
                    "assumed_isp_s": ASSUMED_ISP_S,
                    "not_modeled": ["battery_pct", "temperature_c", "signal_strength_dbm", "solar_power_w"],
                },
            }

    def get_maneuver_history(self, norad_id: int) -> list[dict]:
        with self._lock:
            return list(self.maneuver_history.get(norad_id, []))

    def get_all_states(self) -> dict[int, dict[str, Any]]:
        with self._lock:
            return dict(self.states)


# Module-level singleton — import this, never instantiate directly.
satellite_tracker = SatelliteStateTracker()
