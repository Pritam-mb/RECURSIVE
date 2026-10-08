import math
from datetime import datetime, timezone

from app.core.satellite_state_tracker import (
    ASSUMED_ISP_S,
    ASSUMED_PROPELLANT_MASS_KG,
    ASSUMED_WET_MASS_KG,
    G0,
    SatelliteStateTracker,
    classify_object,
    illumination,
    sun_position_eci_km,
)


def test_non_payloads_have_no_telemetry():
    t = SatelliteStateTracker()
    for name, expected in [("COSMOS 2251 DEB", "DEB"), ("SL-16 R/B", "R/B")]:
        out = t.get_telemetry(999998, name, position_km=(7000.0, 0.0, 0.0))
        assert out["object_type"] == expected
        assert out["telemetry_available"] is False
        for key in ("fuel_remaining_pct", "battery_pct", "temperature_c", "signal_strength_dbm"):
            assert out[key] is None


def test_payload_block_is_simulated_and_fuel_follows_rocket_equation():
    t = SatelliteStateTracker()
    out = t.get_telemetry(25544, "ISS (ZARYA)", object_type="PAY", position_km=(7000.0, 0.0, 0.0))
    assert out["simulated"] is True and out["measured"] is False
    assert out["fuel_remaining_pct"] == 100.0
    assert out["battery_pct"] is None and out["signal_strength_dbm"] is None

    t.record_maneuver(25544, 10.0)
    expected_used = ASSUMED_WET_MASS_KG * (1 - math.exp(-10.0 / (ASSUMED_ISP_S * G0)))
    after = t.get_telemetry(25544, "ISS (ZARYA)", object_type="PAY", position_km=(7000.0, 0.0, 0.0))
    assert math.isclose(after["fuel_remaining_pct"], 100 * (1 - expected_used / ASSUMED_PROPELLANT_MASS_KG), abs_tol=1e-3)


def test_illumination_geometry():
    when = datetime(2026, 3, 20, 12, tzinfo=timezone.utc)
    sx, sy, sz = sun_position_eci_km(when)
    norm = math.sqrt(sx * sx + sy * sy + sz * sz)
    assert 1.47e8 < norm < 1.53e8
    toward = tuple(7000.0 * c / norm for c in (sx, sy, sz))
    away = tuple(-c for c in toward)
    assert illumination(toward, when) == "sunlit"
    assert illumination(away, when) == "eclipse"


def test_classify_heuristic():
    assert classify_object(999999, "STARLINK-1234", None)[0] in {"PAY", "UNK", "R/B", "DEB"}
    assert classify_object(999999, "FENGYUN 1C DEB", None)[0] == "DEB"
