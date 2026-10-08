"""
risk_api.py -- fast ML Pc surrogate attached to alerts as ``alert["ml"]``.

Purpose (honest scope)
----------------------
``score_alert`` estimates log10(Foster Pc) from cheap encounter features with an
XGBoost regressor trained by ``app.ml.train_risk_surrogate`` on simulated
encounters whose labels are numerically integrated Foster Pc values (see the
model card ``artifacts/risk_model_card.json`` for held-out metrics).

It is a triage aid and a cross-check of the physics Pc:
* it NEVER replaces ``alert["p_collision"]`` / ``probability_of_collision``;
* ``agreement`` = |log10(pc_surrogate) - log10(physics Pc)| (decades) flags
  alerts whose physics Pc disagrees with what the geometry usually implies
  (e.g. an unusual covariance orientation) -- worth a human look.

No torch, no feature flag, ~0.1 ms per alert (single-row inplace predict).

Per-alert explanation (``contributions`` / ``base_log10`` / ``main_factor``)
---------------------------------------------------------------------------
XGBoost ``pred_contribs`` splits each alert's log10 Pc prediction into one
additive contribution per feature plus a bias (the model's expected value):
``base_log10 + sum(contributions) == log10 Pc prediction`` exactly.
* The ``EXACT_SHAP_BUDGET`` (default 8, env ORBIT_SENTINEL_EXACT_SHAP_BUDGET)
  highest-risk alerts of each batch (by max of physics and surrogate Pc) get
  **exact path-dependent TreeSHAP** (~2.4 ms/row single-threaded; run on a
  4-thread model copy, ~1 ms/row).
* All other alerts get **Saabas path attribution** (``approx_contribs=True``,
  ~0.02 ms/row batched), which is also exactly additive but not a Shapley value.
Typical cost: ~10 ms + 0.02 ms/alert per refresh.
``contribution_method`` on each alert says which one was used.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

import numpy as np

import os

from .artifacts import artifact_path, load_artifact
from .explain import label as feature_label
from .train_risk_surrogate import (
    CARD_FILE,
    FEATURES,
    LOG_PC_FLOOR,
    MODEL_FILE,
    MODEL_NAME,
    THRESHOLD_ELEVATED,
    THRESHOLD_HIGH,
)

logger = logging.getLogger(__name__)

DEFAULT_HBR_KM = 0.010  # same default as app.core.analytics when an alert has no hbr_km
TOP_CONTRIBUTIONS = 6
try:
    EXACT_SHAP_BUDGET = max(0, int(os.getenv("ORBIT_SENTINEL_EXACT_SHAP_BUDGET", "8")))
except ValueError:  # pragma: no cover
    EXACT_SHAP_BUDGET = 8
SHAP_THREADS = max(1, min(4, os.cpu_count() or 1))

_lock = threading.Lock()
_booster = None
_load_failed: str | None = None


def _get_booster():
    global _booster, _load_failed
    if _booster is not None or _load_failed is not None:
        return _booster
    with _lock:
        if _booster is not None or _load_failed is not None:
            return _booster
        try:
            import xgboost as xgb

            path = artifact_path(MODEL_FILE)
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run python -m app.ml.train_risk_surrogate")
            booster = xgb.Booster()
            booster.load_model(str(path))
            booster.set_param({"nthread": 1})
            _booster = booster
        except Exception as exc:
            _load_failed = str(exc)
            logger.warning("Pc surrogate unavailable: %s", exc)
    return _booster


_shap_booster = None


def _get_shap_booster():
    """Separate copy of the model for exact TreeSHAP with a few threads, so the
    single-threaded scoring booster's parameters are never mutated concurrently."""
    global _shap_booster
    if _shap_booster is None:
        booster = _get_booster()
        if booster is None:
            return None
        with _lock:
            if _shap_booster is None:
                b = booster.copy()
                b.set_param({"nthread": SHAP_THREADS})
                _shap_booster = b
    return _shap_booster


def model_card() -> dict[str, Any] | None:
    return load_artifact(CARD_FILE)


def model_available() -> bool:
    return _get_booster() is not None


# ── Live counters (exposed via model_metrics) ───────────────────────────────

class _Counters:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.scored = 0
        self.high = 0
        self.elevated = 0
        self.agreement_n = 0
        self.agreement_sum = 0.0
        self.agreement_max = 0.0
        self.disagree_gt_1_decade = 0
        self.missing_inputs: dict[str, int] = {}

    def record(self, risk_class: str, agreement: float | None, missing: list[str]):
        with self.lock:
            self.scored += 1
            if risk_class == "high":
                self.high += 1
            elif risk_class == "elevated":
                self.elevated += 1
            if agreement is not None:
                self.agreement_n += 1
                self.agreement_sum += agreement
                self.agreement_max = max(self.agreement_max, agreement)
                if agreement > 1.0:
                    self.disagree_gt_1_decade += 1
            for name in missing:
                self.missing_inputs[name] = self.missing_inputs.get(name, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "alerts_scored": self.scored,
                "risk_class_high": self.high,
                "risk_class_elevated": self.elevated,
                "agreement_samples": self.agreement_n,
                "mean_agreement_decades": round(self.agreement_sum / self.agreement_n, 3) if self.agreement_n else None,
                "max_agreement_decades": round(self.agreement_max, 3) if self.agreement_n else None,
                "disagreements_over_1_decade": self.disagree_gt_1_decade,
                "missing_inputs": dict(self.missing_inputs),
            }


live_counters = _Counters()


# ── Feature extraction ──────────────────────────────────────────────────────

def _num(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _first(*values) -> float | None:
    for value in values:
        f = _num(value)
        if f is not None:
            return f
    return None


def extract_features(alert: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Map an alert (contract schema) to the surrogate's feature row.

    Returns the row (NaN = missing, handled by XGBoost's missing branches) and
    the list of inputs that were missing or defaulted.
    """
    sat1 = alert.get("sat1") if isinstance(alert.get("sat1"), dict) else {}
    sat2 = alert.get("sat2") if isinstance(alert.get("sat2"), dict) else {}
    missing: list[str] = []

    miss = _first(alert.get("miss_distance_km"))
    if miss is None:
        missing.append("miss_distance_km")

    # B-plane "n" axis of app.core.analytics is ~radial for near-circular
    # orbits, so |b_n| is the radial miss component.
    radial = _first(alert.get("radial_miss_km"),
                    abs(_num(alert.get("b_n_km"))) if _num(alert.get("b_n_km")) is not None else None)
    if radial is None:
        missing.append("radial_miss_km")

    vrel = _first(alert.get("relative_speed_kms"))
    if vrel is None:
        kmh = _num(alert.get("relative_speed_kmh"))
        vrel = kmh / 3600.0 if kmh is not None else None
    if vrel is None:
        missing.append("relative_speed_kms")

    common_age = _num(alert.get("tle_age_hours"))
    age1 = _first(sat1.get("tle_age_hours"), alert.get("tle_age_hours_1"), alert.get("sat1_tle_age_hours"), common_age)
    age2 = _first(sat2.get("tle_age_hours"), alert.get("tle_age_hours_2"), alert.get("sat2_tle_age_hours"), common_age)
    if age1 is None and age2 is None:
        missing.append("tle_age_hours")
        age_max = age_min = None
    else:
        ages = [a for a in (age1, age2) if a is not None]
        age_max, age_min = max(ages), min(ages)

    hbr = _first(alert.get("hbr_km"))
    if hbr is None:
        hbr = DEFAULT_HBR_KM
        missing.append("hbr_km(defaulted)")

    alt = _first(alert.get("altitude_km"), alert.get("tca_altitude_km"))
    if alt is None:
        a1, a2 = _first(sat1.get("altitude_km"), alert.get("sat1_altitude_km")), _first(sat2.get("altitude_km"), alert.get("sat2_altitude_km"))
        if a1 is not None and a2 is not None:
            alt = 0.5 * (a1 + a2)
        else:
            alt = a1 if a1 is not None else a2
    if alt is None:
        missing.append("altitude_km")

    values = {
        "miss_distance_km": miss, "radial_miss_km": radial, "relative_speed_kms": vrel,
        "tle_age_max_h": age_max, "tle_age_min_h": age_min, "hbr_km": hbr, "altitude_km": alt,
    }
    row = np.array([[np.nan if values[f] is None else values[f] for f in FEATURES]], dtype=np.float32)
    return row, missing


def risk_class_for(pc: float) -> str:
    if pc >= THRESHOLD_HIGH:
        return "high"
    if pc >= THRESHOLD_ELEVATED:
        return "elevated"
    return "low"


def _physics_pc(alert: dict[str, Any]) -> float | None:
    return _first(alert.get("p_collision"), alert.get("probability_of_collision"))


def _finalize(alert: dict[str, Any], log_pc: float, missing: list[str], record: bool) -> dict[str, Any]:
    log_pc = min(0.0, max(LOG_PC_FLOOR, float(log_pc)))
    pc = 10.0 ** log_pc
    risk_class = risk_class_for(pc)

    physics = _physics_pc(alert)
    agreement = None
    if physics is not None:
        agreement = round(abs(log_pc - math.log10(min(1.0, max(physics, 10.0 ** LOG_PC_FLOOR)))), 3)

    if record:
        # Disagreement only matters when either estimate is operationally relevant.
        relevant = pc >= THRESHOLD_ELEVATED or (physics is not None and physics >= THRESHOLD_ELEVATED)
        live_counters.record(risk_class, agreement if relevant else None, missing)

    return {
        "pc_surrogate": pc,
        "log10_pc_surrogate": round(log_pc, 3),
        "risk_class": risk_class,
        "model": MODEL_NAME,
        "agreement": agreement,
        "inputs_missing": missing,
        "role": "surrogate cross-check; physics Pc is authoritative",
    }


def _contributions(booster, rows: np.ndarray, *, exact: bool) -> np.ndarray:
    """(n, p+1) additive contributions (log10 Pc); last column = bias."""
    import xgboost as xgb

    if exact:
        booster = _get_shap_booster() or booster

    d = xgb.DMatrix(np.asarray(rows, dtype=np.float32), feature_names=list(FEATURES), missing=np.nan)
    return booster.predict(d, pred_contribs=True, approx_contribs=not exact)


def explain_block(row: np.ndarray, contrib: np.ndarray, *, exact: bool) -> dict[str, Any]:
    """Format one contributions row for alert["ml"] (top-6 by |value|)."""
    feats = contrib[:-1]
    order = np.argsort(-np.abs(feats))
    top = order[:TOP_CONTRIBUTIONS]
    items = []
    for i in top:
        v = float(row[i])
        items.append({
            "feature": FEATURES[i],
            "label": feature_label(FEATURES[i]),
            "value": round(v, 6) if math.isfinite(v) else None,
            "contribution_log10": round(float(feats[i]), 4),
        })
    rest = float(np.sum(feats[order[TOP_CONTRIBUTIONS:]])) if len(order) > TOP_CONTRIBUTIONS else 0.0
    return {
        "contributions": items,
        "base_log10": round(float(contrib[-1]), 4),
        "main_factor": FEATURES[int(order[0])],
        "main_factor_label": feature_label(FEATURES[int(order[0])]),
        "other_contributions_log10": round(rest, 4),
        "contribution_method": "treeshap_exact" if exact else "saabas_path",
    }


def score_alert(alert: dict[str, Any], *, record: bool = True, explain: bool = False,
                exact: bool = False) -> dict[str, Any] | None:
    """Return the ML block for ``alert["ml"]`` or None if the model is unavailable.

    Keys: pc_surrogate, log10_pc_surrogate, risk_class, model, agreement
    (|log10 ml - log10 physics| in decades, both clipped at 1e-12; None if no
    physics Pc), inputs_missing, role. With explain=True also contributions,
    base_log10, main_factor (Saabas; exact=True for TreeSHAP). Explanation is
    off by default here to keep single-alert scoring < 1 ms; the live pipeline
    uses score_alerts, which always explains in one batched call.
    """
    booster = _get_booster()
    if booster is None:
        return None
    row, missing = extract_features(alert)
    if "miss_distance_km" in missing:
        return None
    block = _finalize(alert, booster.inplace_predict(row)[0], missing, record)
    if not explain:
        return block
    try:
        block.update(explain_block(row[0], _contributions(booster, row, exact=exact)[0], exact=exact))
    except Exception as exc:  # pragma: no cover - explanation is best-effort
        logger.debug("contributions failed: %s", exc)
    return block


def score_alerts(alerts: list[dict[str, Any]], *, record: bool = True,
                 exact_budget: int | None = None) -> list[dict[str, Any]]:
    """Attach ``alert["ml"]`` in place with one batched prediction (never touches physics Pc).

    Contributions: one batched exact-TreeSHAP call for the ``exact_budget``
    highest-risk alerts and one batched Saabas call for the rest.
    """
    booster = _get_booster()
    if booster is None or not alerts:
        for alert in alerts:
            alert["ml"] = None
        return alerts
    rows, metas = [], []
    for alert in alerts:
        row, missing = extract_features(alert)
        metas.append(missing)
        rows.append(row[0])
    x = np.asarray(rows, dtype=np.float32)
    preds = booster.inplace_predict(x)
    scored = []
    for k, (alert, missing, log_pc) in enumerate(zip(alerts, metas, preds)):
        if "miss_distance_km" in missing:
            alert["ml"] = None
            continue
        alert["ml"] = _finalize(alert, log_pc, missing, record)
        scored.append(k)
    if not scored:
        return alerts

    budget = EXACT_SHAP_BUDGET if exact_budget is None else max(0, int(exact_budget))

    def priority(k: int) -> float:
        physics = _physics_pc(alerts[k]) or 0.0
        return max(physics, alerts[k]["ml"]["pc_surrogate"])

    ranked = sorted(scored, key=priority, reverse=True)
    groups = ((ranked[:budget], True), (ranked[budget:], False))
    try:
        for idx, exact in groups:
            if not idx:
                continue
            contrib = _contributions(booster, x[idx], exact=exact)
            for j, k in enumerate(idx):
                alerts[k]["ml"].update(explain_block(x[k], contrib[j], exact=exact))
    except Exception as exc:  # pragma: no cover - explanation is best-effort
        logger.debug("contributions failed: %s", exc)
    return alerts
