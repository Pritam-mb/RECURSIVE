"""
Regression tests for the bugs found by the independent maths verification
(docs/report/verification.json). Each test checks against an independent
reference, not against the implementation.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
from scipy.integrate import solve_ivp

from app.core import breakup as BK
from app.core import debris_model as DM


def _j2_rhs(t, y):
    x, yy, z = y[:3]
    r = math.sqrt(x * x + yy * yy + z * z)
    c = 1.5 * BK.J2 * BK.MU_KM3_S2 * BK.R_EARTH_KM ** 2 / r ** 5
    k = 5.0 * z * z / (r * r)
    a = -BK.MU_KM3_S2 * np.array([x, yy, z]) / r ** 3 + c * np.array([x * (k - 1), yy * (k - 1), z * (k - 3)])
    return np.r_[y[3:], a]


def test_state_vector_satellite_tracks_are_metre_accurate():
    """Satellites without a TLE are propagated (two-body + J2 RK4) for debris screening.
    With 60 s substeps the 6 h error was ~200 m, i.e. ~1 sigma of the 0.2 km satellite
    sigma used for debris Pc; it must stay well below that."""
    r0 = np.array([6778.137, 0.0, 0.0])
    vc = math.sqrt(BK.MU_KM3_S2 / r0[0])
    v0 = np.array([0.0, vc * math.cos(0.9), vc * math.sin(0.9)])
    offs = np.arange(0.0, 6 * 3600.0 + 1.0, 60.0)       # coarse grid: substeps must be internal
    model = DM.DebrisModel()
    R, _ = model._satellite_tracks([(999999, "STATE-ONLY", r0, v0)], datetime(2026, 5, 12, tzinfo=timezone.utc), offs)
    ref = solve_ivp(_j2_rhs, (0.0, offs[-1]), np.r_[r0, v0], rtol=1e-13, atol=1e-12, method="DOP853")
    err_m = float(np.linalg.norm(R[0, -1] - ref.y[:3, -1])) * 1000.0
    assert err_m < 20.0, err_m
    assert err_m < 0.1 * DM.SAT_SIGMA_KM * 1000.0


def test_model_card_covariance_recipe_matches_live_model():
    """The card must describe the covariance the labels were generated with."""
    from app.core.screening import SIGMA0_RTN_KM, SIGMA_GROWTH_RTN_KM_PER_DAY
    from app.ml.artifacts import ARTIFACT_DIR
    from app.ml.train_risk_surrogate import CARD_FILE, _covariance_recipe_text

    text = _covariance_recipe_text()
    assert str(tuple(SIGMA0_RTN_KM)) in text and str(tuple(SIGMA_GROWTH_RTN_KM_PER_DAY)) in text
    card = json.loads((ARTIFACT_DIR / CARD_FILE).read_text(encoding="utf-8"))
    assert card["recipe"]["covariance"] == text


def test_validation_rel_error_undefined_for_zero_reference():
    """abs-tolerance checks against a zero reference must not report abs/1e-300 ~ 1e298."""
    from app.core.analytic_checks import _check as ac_check
    from app.routers.physics import _check as ph_check

    c = ph_check("zero ref", "x", "f", 0.04, 0.0, 0.5, "src", rel=False)
    assert c["rel_error"] is None and c["pass"] is True
    c = ac_check("zero ref", "f", 0.04, 0.0, 0.5, "src", rel=False)
    assert c["rel_error"] is None and c["pass"] is True
    c = ph_check("nonzero", "x", "f", 1.01, 1.0, 0.02, "src")
    assert abs(c["rel_error"] - 0.01) < 1e-12 and c["pass"] is True
