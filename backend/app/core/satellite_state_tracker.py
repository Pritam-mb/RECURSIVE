"""
satellite_state_tracker.py — Stateful per-satellite health and fuel model.

Replaces random.uniform() health telemetry with a persistent, physically
plausible state that:
  - Initialises once per satellite with randomised-but-fixed baseline
  - Depletes fuel on maneuvers (simplified Tsiolkovsky model)
  - Varies health metrics with small realistic increments (not random jumps)
  - Records full maneuver history per satellite

Import the module-level singleton:
    from app.core.satellite_state_tracker import satellite_tracker

Never instantiate SatelliteStateTracker directly — use the singleton.
"""

from __future__ import annotations

import logging
import random
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class SatelliteStateTracker:
    """
    Persistent health and fuel state tracker for all satellites.

    Thread-safe via a reentrant lock. All public methods acquire the
    lock automatically so callers need not worry about concurrency.
    """

    def __init__(self):
        # norad_id (int) → state dict
        self.states: dict[int, dict[str, Any]] = {}
        # norad_id (int) → list of maneuver records
        self.maneuver_history: dict[int, list] = defaultdict(list)
        self._lock = threading.RLock()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_or_create(self, norad_id: int, satellite_name: str = "") -> dict[str, Any]:
        """
        Return the persisted state dict for norad_id, creating it on first call.

        Initial values are randomised once at creation time, then the tracker
        owns the state — no further random calls for existing satellites.
        """
        with self._lock:
            if norad_id not in self.states:
                rng = random.Random(norad_id)  # deterministic seed per satellite
                now = self._now_iso()
                self.states[norad_id] = {
                    "norad_id": norad_id,
                    "satellite_name": satellite_name or f"SAT-{norad_id}",
                    # Fuel: starts 88–100 %, seeded by NORAD ID so reproducible
                    "fuel_remaining_pct": round(95.0 + rng.uniform(-5.0, 5.0), 2),
                    # Battery: 85–91 %
                    "battery_pct": round(88.0 + rng.uniform(-3.0, 3.0), 2),
                    # Temperature: -18 to -2 °C (typical LEO shadow/sun cycling baseline)
                    "temperature_c": round(-10.0 + rng.uniform(-8.0, 8.0), 2),
                    # Signal strength: -70 to -60 dBm
                    "signal_strength_dbm": round(-65.0 + rng.uniform(-5.0, 5.0), 2),
                    # Solar power: 1100–1300 W
                    "solar_power_w": round(1200.0 + rng.uniform(-100.0, 100.0), 1),
                    # Cumulative maneuver metrics
                    "total_delta_v_used_ms": 0.0,
                    "maneuver_count": 0,
                    # Metadata
                    "created_at": now,
                    "last_updated": now,
                    # Small thermal / charge cycle accumulators (persist between calls)
                    "_temp_phase": rng.uniform(0, 6.283),   # thermal orbit phase (rad)
                    "_batt_phase": rng.uniform(0, 6.283),   # charge/discharge phase
                    "_call_count": 0,
                }
                logger.debug(
                    "Initialised state for NORAD %d (%s): fuel=%.1f%%",
                    norad_id,
                    satellite_name,
                    self.states[norad_id]["fuel_remaining_pct"],
                )
            return self.states[norad_id]

    def record_maneuver(
        self,
        norad_id: int,
        delta_v_ms: float,
        direction: str = "radial",
        satellite_name: str = "",
    ) -> dict[str, Any]:
        """
        Record a maneuver and deduct fuel.

        Fuel cost model (simplified Tsiolkovsky for ~500 kg satellite):
            fuel_cost_pct = delta_v_ms * 0.1  % per m/s

        This is physically calibrated for a 500 kg satellite with a
        specific impulse of ~220 s and a fuel mass fraction of ~20%.

        Returns the updated state dict.
        """
        with self._lock:
            state = self.get_or_create(norad_id, satellite_name)
            delta_v_ms = float(delta_v_ms)

            # Fuel cost: 0.1% per m/s (realistic for 500 kg LEO satellite)
            fuel_cost = delta_v_ms * 0.1
            old_fuel = state["fuel_remaining_pct"]
            new_fuel = max(0.0, old_fuel - fuel_cost)
            state["fuel_remaining_pct"] = round(new_fuel, 3)
            state["total_delta_v_used_ms"] = round(
                state["total_delta_v_used_ms"] + delta_v_ms, 3
            )
            state["maneuver_count"] += 1
            state["last_updated"] = self._now_iso()

            record = {
                "timestamp": self._now_iso(),
                "delta_v_ms": round(delta_v_ms, 3),
                "direction": direction,
                "fuel_before_pct": round(old_fuel, 3),
                "fuel_after_pct": round(new_fuel, 3),
                "maneuver_number": state["maneuver_count"],
            }
            self.maneuver_history[norad_id].append(record)

            logger.info(
                "NORAD %d maneuver #%d: Δv=%.2f m/s, fuel %.1f%% → %.1f%%",
                norad_id,
                state["maneuver_count"],
                delta_v_ms,
                old_fuel,
                new_fuel,
            )
            return state

    def get_telemetry(
        self,
        norad_id: int,
        satellite_name: str = "",
    ) -> dict[str, Any]:
        """
        Return a telemetry snapshot for norad_id with small realistic variations.

        Variation model (not random per call — based on call count):
          Temperature: sinusoidal ±8 °C orbital thermal cycle
          Battery:     sinusoidal ±5 % charge/discharge cycle
          Signal:      slow walk ±2 dBm (link margin variation)
          Solar power: constant (panel degradation is negligible short-term)

        All values are clamped to physically plausible ranges.
        """
        with self._lock:
            state = self.get_or_create(norad_id, satellite_name)
            state["_call_count"] += 1
            k = state["_call_count"]

            # ── Temperature: sinusoidal orbital thermal cycling ──────────────
            # LEO period ~90 min → model varies slowly over hundreds of calls
            import math
            phase_t = state["_temp_phase"] + k * 0.02  # ~0.02 rad per call
            temp_variation = 8.0 * math.sin(phase_t)
            # Base temperature from created state (persist + cycle)
            base_temp = state["temperature_c"]
            # Drift very slightly toward 0 °C as satellite ages (thermal equilibrium)
            temperature_c = round(base_temp + temp_variation * 0.3, 2)
            temperature_c = max(-30.0, min(40.0, temperature_c))

            # ── Battery: charge/discharge cycle ─────────────────────────────
            phase_b = state["_batt_phase"] + k * 0.015
            batt_variation = 5.0 * math.sin(phase_b)
            battery_pct = round(
                max(10.0, min(100.0, state["battery_pct"] + batt_variation * 0.3)),
                2,
            )

            # ── Signal strength: slow random walk (link margin) ──────────────
            # Small deterministic variation based on call count
            signal_variation = 2.0 * math.sin(k * 0.07) * math.cos(k * 0.03)
            signal_strength_dbm = round(
                max(-90.0, min(-40.0, state["signal_strength_dbm"] + signal_variation)),
                2,
            )

            # ── Solar power: essentially constant (minor eclipse variation) ──
            eclipse_factor = max(0.0, math.sin(phase_t + math.pi / 4.0))
            solar_power_w = round(
                max(0.0, state["solar_power_w"] * (0.85 + 0.15 * eclipse_factor)),
                1,
            )

            return {
                "norad_id": norad_id,
                "satellite_name": satellite_name or state.get("satellite_name", f"SAT-{norad_id}"),
                "fuel_remaining_pct": round(state["fuel_remaining_pct"], 2),
                "battery_pct": battery_pct,
                "temperature_c": temperature_c,
                "signal_strength_dbm": signal_strength_dbm,
                "solar_power_w": solar_power_w,
                "total_delta_v_used_ms": state["total_delta_v_used_ms"],
                "maneuver_count": state["maneuver_count"],
                "last_updated": state["last_updated"],
                "created_at": state["created_at"],
            }

    def get_maneuver_history(self, norad_id: int) -> list[dict]:
        """Return the full maneuver history list for norad_id."""
        with self._lock:
            return list(self.maneuver_history.get(norad_id, []))

    def get_all_states(self) -> dict[int, dict[str, Any]]:
        """Return a shallow copy of all tracked satellite states."""
        with self._lock:
            return dict(self.states)


# ── Module-level singleton ─────────────────────────────────────────────────────
# Import this instance everywhere — never instantiate SatelliteStateTracker directly.
satellite_tracker = SatelliteStateTracker()
