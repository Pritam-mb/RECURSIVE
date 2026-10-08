"""
train_risk_surrogate.py -- train the XGBoost Pc surrogate on physics labels.

What this model is (and is not)
-------------------------------
The physics collision probability shown on every alert is Foster 2D Pc, computed
by the screening code (app.core.screening / app.core.analytics). This module
trains a *surrogate* for it: a gradient-boosted regressor that estimates
log10(Pc) from cheap encounter features. It is used for triage and as an
independent cross-check of the physics Pc. It never overwrites the physics Pc.

Why the labels are not circular
-------------------------------
The previous risk model was trained on labels that were a sigmoid formula of its
own input features. Here every label is a Foster Pc obtained by numerically
integrating the bivariate Gaussian of the *combined, B-plane-projected
covariance* over the hard-body disk. The covariance comes from each object's
TLE-age error model, rotated from its own RTN frame into the encounter frame,
so it depends on the full 3D encounter geometry (crossing angle, flight-path
angle, B-plane miss direction). The model never sees the covariance or the
projected terms -- only:

    miss_distance_km, radial_miss_km, relative_speed_kms,
    tle_age_max_h, tle_age_min_h, hbr_km, altitude_km

so it has to learn the mapping, and its error against the integral is a real,
held-out measurement.

Data generation recipe (seeded, reproducible)
---------------------------------------------
* altitude      ~ U(300, 2000) km, circular speed sqrt(mu/r) * (1 + U(-1%, 1%))
* headings      psi1 ~ U(0, 2pi), crossing angle theta: 85% U(10, 175) deg,
                  15% U(0.5, 10) deg (co-orbital / overtaking)
* flight-path   gamma_i ~ U(-1.5, 1.5) deg (mildly eccentric orbits)
* miss vector   in the B-plane (perpendicular to relative velocity),
                  |b| log-uniform: 60% [3 m, 25 km], 40% [3 m, 3 km],
                  direction ~ U(0, 2pi)
* TLE ages      log-uniform [0.5, 8760] h per object (covers stale catalogue TLEs)
* HBR           log-uniform [3, 50] m (combined hard-body radius)
* covariance    per object diag(sigma_r^2, sigma_t^2, sigma_n^2) in RTN with the
                  TLE-age model sigma = SIGMA0 + GROWTH * age_days per RTN axis,
                  imported from app.core.screening (the live Pc model; checked numerically
                  below and recorded in the model card)
* label         Foster Pc = integral over |x| <= HBR of N(x; b, C_2d), evaluated
                  by polar Gauss-Legendre (radial) x periodic trapezoid (angular)
                  quadrature; cross-checked against scipy dblquad.
* target        log10(max(Pc, 1e-12))
* robustness    30% of rows have radial_miss_km = NaN, 30% altitude = NaN and
                  15% both TLE ages = NaN, so the model degrades gracefully when
                  an alert does not carry those fields (XGBoost missing-value
                  branches). Metrics are reported on full and masked features.

Run:  python -m app.ml.train_risk_surrogate      (from backend/, < 1 min)
"""

from __future__ import annotations

import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .artifacts import ARTIFACT_DIR

MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km
PC_FLOOR = 1e-12
LOG_PC_FLOOR = -12.0

MODEL_FILE = "risk_model_xgb.json"
CARD_FILE = "risk_model_card.json"
ANALYTICS_FILE = "risk_model_analytics.json"
MODEL_NAME = "xgboost-pc-surrogate v1"

FEATURES = [
    "miss_distance_km",
    "radial_miss_km",
    "relative_speed_kms",
    "tle_age_max_h",
    "tle_age_min_h",
    "hbr_km",
    "altitude_km",
]

SEED = 20261008
SAMPLE_COUNT = 60_000

THRESHOLD_HIGH = 1e-4
THRESHOLD_ELEVATED = 1e-6


# ── TLE-age covariance model ─────────────────────────────────────────────────

# Up to one year: bundled/offline TLE snapshots can be months old.
TLE_AGE_MAX_H = 24.0 * 365.0


def sigma_rtn_km(tle_age_hours: np.ndarray) -> np.ndarray:
    """Per-object 1-sigma position error (R, T, N) in km from TLE age.

    Uses the exact constants of the live screening covariance model
    (app.core.screening.SIGMA0_RTN_KM / SIGMA_GROWTH_RTN_KM_PER_DAY), imported
    rather than copied so the training labels can never drift from the Pc the
    screening actually reports.
    """
    from app.core.screening import SIGMA0_RTN_KM, SIGMA_GROWTH_RTN_KM_PER_DAY

    age_days = np.asarray(tle_age_hours, dtype=float)[..., None] / 24.0
    sigma0 = np.asarray(SIGMA0_RTN_KM, dtype=float)
    growth = np.asarray(SIGMA_GROWTH_RTN_KM_PER_DAY, dtype=float)
    return sigma0 + (growth * age_days)


def _covariance_recipe_text() -> str:
    """Model-card description of the label covariance, generated from the live constants."""
    from app.core.screening import SIGMA0_RTN_KM, SIGMA_GROWTH_RTN_KM_PER_DAY

    return (f"per object diagonal RTN, sigma_RTN = {tuple(SIGMA0_RTN_KM)} km + "
            f"{tuple(SIGMA_GROWTH_RTN_KM_PER_DAY)} km/day x TLE age (app.core.screening "
            f"TLE-age model); both objects summed in ECI, projected onto the B-plane")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _rtn_covariance(r: np.ndarray, v: np.ndarray, age_h: np.ndarray) -> np.ndarray:
    """Vectorised RTN->ECI covariance, (N,3,3) km^2."""
    r_hat = _unit(r)
    n_hat = _unit(np.cross(r, v))
    t_hat = _unit(np.cross(n_hat, r_hat))
    rot = np.stack([r_hat, t_hat, n_hat], axis=-1)  # columns = RTN basis
    sig = sigma_rtn_km(age_h)
    diag = np.zeros((len(r), 3, 3))
    diag[:, 0, 0] = sig[:, 0] ** 2
    diag[:, 1, 1] = sig[:, 1] ** 2
    diag[:, 2, 2] = sig[:, 2] ** 2
    return rot @ diag @ np.transpose(rot, (0, 2, 1))


# ── Foster Pc by quadrature ──────────────────────────────────────────────────

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(20)
_N_ANG = 64


def foster_pc_quadrature(b: np.ndarray, cov2: np.ndarray, hbr: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Foster 2D Pc for N encounters.

    b: (N,2) km miss vector in B-plane, cov2: (N,2,2) km^2, hbr: (N,) km.
    Integral of N(x; b, cov2) over the disk |x| <= hbr in polar coordinates:
    Gauss-Legendre in radius (20 nodes), periodic trapezoid in angle (64 nodes).
    """
    n = len(b)
    out = np.empty(n)
    phi = np.arange(_N_ANG) * (2.0 * np.pi / _N_ANG)
    cphi, sphi = np.cos(phi), np.sin(phi)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        R = hbr[s:e]
        rho = 0.5 * R[:, None] * (_GL_NODES[None, :] + 1.0)  # (m,20)
        w_r = 0.5 * R[:, None] * _GL_WEIGHTS[None, :]
        x = rho[:, :, None] * cphi[None, None, :]  # (m,20,64)
        y = rho[:, :, None] * sphi[None, None, :]
        c = cov2[s:e]
        det = c[:, 0, 0] * c[:, 1, 1] - c[:, 0, 1] ** 2
        inv00 = c[:, 1, 1] / det
        inv11 = c[:, 0, 0] / det
        inv01 = -c[:, 0, 1] / det
        dx = x - b[s:e, 0][:, None, None]
        dy = y - b[s:e, 1][:, None, None]
        q = (inv00[:, None, None] * dx * dx + 2 * inv01[:, None, None] * dx * dy
             + inv11[:, None, None] * dy * dy)
        pdf = np.exp(-0.5 * q) / (2.0 * np.pi * np.sqrt(det))[:, None, None]
        ang = pdf.sum(axis=2) * (2.0 * np.pi / _N_ANG)  # (m,20)
        out[s:e] = np.sum(ang * rho * w_r, axis=1)
    return np.clip(out, 0.0, 1.0)


# ── Encounter generator ──────────────────────────────────────────────────────

def _loguniform(rng, lo, hi, size):
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size))


def generate_encounters(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    alt = rng.uniform(300.0, 2000.0, n)
    radius = R_EARTH + alt
    vcirc = np.sqrt(MU_EARTH / radius)

    psi1 = rng.uniform(0.0, 2 * np.pi, n)
    co = rng.random(n) < 0.15
    theta = np.where(co, rng.uniform(0.5, 10.0, n), rng.uniform(10.0, 175.0, n))
    theta = np.deg2rad(theta) * np.where(rng.random(n) < 0.5, 1.0, -1.0)
    psi2 = psi1 + theta
    gam1 = np.deg2rad(rng.uniform(-1.5, 1.5, n))
    gam2 = np.deg2rad(rng.uniform(-1.5, 1.5, n))
    s1 = vcirc * (1 + rng.uniform(-0.01, 0.01, n))
    s2 = vcirc * (1 + rng.uniform(-0.01, 0.01, n))

    # Local frame at TCA point: x radial, y/z horizontal.
    r1 = np.stack([radius, np.zeros(n), np.zeros(n)], axis=1)

    def vel(speed, psi, gam):
        return np.stack([
            speed * np.sin(gam),
            speed * np.cos(gam) * np.cos(psi),
            speed * np.cos(gam) * np.sin(psi),
        ], axis=1)

    v1 = vel(s1, psi1, gam1)
    v2 = vel(s2, psi2, gam2)
    dv = v2 - v1
    vrel = np.linalg.norm(dv, axis=1)
    h_hat = dv / vrel[:, None]
    e1 = _unit(np.cross(h_hat, r1))  # horizontal B-plane axis
    e2 = _unit(np.cross(e1, h_hat))  # ~radial B-plane axis
    bmat = np.stack([e1, e2], axis=1)  # (n,2,3)

    wide = rng.random(n) < 0.6
    miss = np.where(wide, _loguniform(rng, 0.003, 25.0, n), _loguniform(rng, 0.003, 3.0, n))
    phi = rng.uniform(0.0, 2 * np.pi, n)
    b2 = np.stack([miss * np.cos(phi), miss * np.sin(phi)], axis=1)
    miss_vec = b2[:, 0:1] * e1 + b2[:, 1:2] * e2
    r2 = r1 + miss_vec

    age1 = _loguniform(rng, 0.5, TLE_AGE_MAX_H, n)
    age2 = _loguniform(rng, 0.5, TLE_AGE_MAX_H, n)
    hbr = _loguniform(rng, 0.003, 0.050, n)

    cov = _rtn_covariance(r1, v1, age1) + _rtn_covariance(r2, v2, age2)
    cov2 = bmat @ cov @ np.transpose(bmat, (0, 2, 1))
    pc = foster_pc_quadrature(b2, cov2, hbr)

    radial_miss = np.abs(np.sum(miss_vec * (r1 / radius[:, None]), axis=1))
    return {
        "miss_distance_km": miss,
        "radial_miss_km": radial_miss,
        "relative_speed_kms": vrel,
        "tle_age_max_h": np.maximum(age1, age2),
        "tle_age_min_h": np.minimum(age1, age2),
        "hbr_km": hbr,
        "altitude_km": alt,
        "crossing_angle_deg": np.rad2deg(np.abs(theta)),
        "pc": pc,
        "_b2": b2,
        "_cov2": cov2,
        "_r1": r1, "_v1": v1, "_r2": r2, "_v2": v2, "_age1": age1, "_age2": age2,
    }


def feature_matrix(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([data[f] for f in FEATURES]).astype(np.float32)


def mask_features(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x.copy()
    n = len(x)
    x[rng.random(n) < 0.30, FEATURES.index("radial_miss_km")] = np.nan
    x[rng.random(n) < 0.30, FEATURES.index("altitude_km")] = np.nan
    age_missing = rng.random(n) < 0.15
    x[age_missing, FEATURES.index("tle_age_max_h")] = np.nan
    x[age_missing, FEATURES.index("tle_age_min_h")] = np.nan
    return x


# ── Metrics ──────────────────────────────────────────────────────────────────

def classification_metrics(true_log: np.ndarray, pred_log: np.ndarray, threshold: float) -> dict:
    t = true_log >= math.log10(threshold)
    p = pred_log >= math.log10(threshold)
    tp = int(np.sum(t & p)); fp = int(np.sum(~t & p)); fn = int(np.sum(t & ~p)); tn = int(np.sum(~t & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold_pc": threshold,
        "positives": int(t.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def regression_metrics(true_log: np.ndarray, pred_log: np.ndarray) -> dict:
    err = pred_log - true_log
    relevant = true_log >= -8.0
    return {
        "mae_log10_pc": round(float(np.mean(np.abs(err))), 4),
        "rmse_log10_pc": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "mae_log10_pc_where_pc_ge_1e-8": round(float(np.mean(np.abs(err[relevant]))), 4) if relevant.any() else None,
        "samples_pc_ge_1e-8": int(relevant.sum()),
        "samples": int(len(true_log)),
    }


def _train_booster(xgb, x_tr, y_tr, x_va, y_va, feature_names, seed):
    params = {
        "objective": "reg:squarederror",
        "eta": 0.08,
        "max_depth": 8,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "seed": seed,
        "nthread": 4,
    }
    dtr = xgb.DMatrix(x_tr, label=y_tr, feature_names=feature_names, missing=np.nan)
    dva = xgb.DMatrix(x_va, label=y_va, feature_names=feature_names, missing=np.nan)
    return xgb.train(params, dtr, num_boost_round=800, evals=[(dva, "val")],
                     early_stopping_rounds=40, verbose_eval=False)


def _predict(xgb, booster, x, feature_names):
    try:
        rng = (0, int(booster.best_iteration) + 1)
    except (AttributeError, TypeError, ValueError):
        rng = (0, 0)  # sliced booster: all trees
    return booster.predict(xgb.DMatrix(x, feature_names=feature_names, missing=np.nan),
                           iteration_range=rng)


def _slice_best(booster):
    """Keep only the trees up to the early-stopping best iteration, so the saved
    model, inplace_predict and TreeSHAP pred_contribs all use the same trees."""
    best = int(booster.best_iteration) + 1
    sliced = booster[:best]
    sliced.set_attr(best_iteration=str(best - 1), best_score=booster.attr("best_score"))
    return sliced


def _check_against_reference(data: dict[str, np.ndarray], count: int = 12) -> dict:
    """Cross-check covariance model and quadrature against app.core.analytics."""
    result: dict = {"reference": "app.core.analytics", "checked": 0}
    try:
        from app.core.analytics import build_rtn_covariance, foster_integrate
    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = f"reference unavailable: {exc}"
        return result
    idx = np.argsort(-data["pc"])[:count // 2].tolist() + list(range(count // 2))
    cov_err, pc_err = [], []
    for i in idx:
        ref_cov = (build_rtn_covariance(data["_r1"][i], data["_v1"][i], float(data["_age1"][i]))
                   + build_rtn_covariance(data["_r2"][i], data["_v2"][i], float(data["_age2"][i])))
        mine_cov = (_rtn_covariance(data["_r1"][i:i + 1], data["_v1"][i:i + 1], data["_age1"][i:i + 1])
                    + _rtn_covariance(data["_r2"][i:i + 1], data["_v2"][i:i + 1], data["_age2"][i:i + 1]))[0]
        cov_err.append(float(np.max(np.abs(ref_cov - mine_cov))))
        ref_pc = foster_integrate(data["_b2"][i], data["_cov2"][i], hbr_km=float(data["hbr_km"][i]))
        mine_pc = float(data["pc"][i])
        if ref_pc > 1e-14 and mine_pc > 1e-14:
            pc_err.append(abs(math.log10(ref_pc) - math.log10(mine_pc)))
    result.update({
        "checked": len(idx),
        "max_abs_covariance_diff_km2": max(cov_err) if cov_err else None,
        "max_abs_log10_pc_diff_vs_dblquad": max(pc_err) if pc_err else None,
    })
    return result


# ── Trajectory model honest baseline comparison ──────────────────────────────

def evaluate_trajectory_model(seed: int = 27, samples: int = 512) -> dict:
    """Compare the numpy recurrent trajectory model with trivial baselines."""
    from .lstm_predictor import load_or_train_trajectory_model
    from .synthetic_data import generate_trajectory_dataset

    seqs, targets, _ = generate_trajectory_dataset(sample_count=samples, history_length=8, seed=seed)
    model = load_or_train_trajectory_model()
    pred = np.array([model.predict_next(s) for s in seqs])
    persist = seqs[:, -1, :]
    linear = seqs[:, -1, :] + (seqs[:, -1, :] - seqs[:, -2, :])

    def mae(p):
        return round(float(np.mean(np.linalg.norm(p - targets, axis=1))), 3)

    ml, per, lin = mae(pred), mae(persist), mae(linear)
    beats = ml < min(per, lin)
    return {
        "name": "Trajectory Predictor",
        "architecture": "numpy tanh RNN (12 hidden units, 8-step history)",
        "label_source": "synthetic circular arcs (app.ml.synthetic_data.generate_trajectory_dataset)",
        "eval_samples": int(samples),
        "eval_seed": seed,
        "mae_km": {"model": ml, "persistence": per, "linear_extrapolation": lin},
        "beats_baselines": bool(beats),
        "status": "production" if beats else "experimental",
        "note": ("Underperforms the linear-extrapolation baseline; not used for any "
                 "alert, Pc or manoeuvre. SGP4 is the propagator of record.") if not beats else "",
        "torch_lstm": "optional (ENABLE_EXTENDED_PIPELINE=1); not loaded by predict()",
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def train(sample_count: int = SAMPLE_COUNT, seed: int = SEED, out_dir: Path | None = None) -> dict:
    import xgboost as xgb

    t0 = time.perf_counter()
    out_dir = Path(out_dir or ARTIFACT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = generate_encounters(sample_count, seed)
    t_gen = time.perf_counter() - t0
    y = np.log10(np.clip(data["pc"], PC_FLOOR, 1.0)).astype(np.float32)
    x_full = feature_matrix(data)
    rng = np.random.default_rng(seed + 1)
    x_masked = mask_features(x_full, rng)

    perm = rng.permutation(sample_count)
    n_te = int(0.2 * sample_count)
    n_va = int(0.1 * sample_count)
    te, va, tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]

    booster = _slice_best(_train_booster(xgb, x_masked[tr], y[tr], x_masked[va], y[va], FEATURES, seed))

    pred_full = _predict(xgb, booster, x_full[te], FEATURES)
    pred_masked = _predict(xgb, booster, x_masked[te], FEATURES)
    x_no_radial = x_full[te].copy()
    x_no_radial[:, FEATURES.index("radial_miss_km")] = np.nan
    pred_no_radial = _predict(xgb, booster, x_no_radial, FEATURES)

    # Baseline 1: miss-distance-only XGBoost (same data, same split).
    base_feats = ["miss_distance_km"]
    bcols = [FEATURES.index(f) for f in base_feats]
    base = _train_booster(xgb, x_full[tr][:, bcols], y[tr], x_full[va][:, bcols], y[va], base_feats, seed)
    pred_base = _predict(xgb, base, x_full[te][:, bcols], base_feats)
    # Baseline 2: constant (training-set mean of log10 Pc).
    pred_const = np.full(len(te), float(np.mean(y[tr])))

    y_te = y[te]
    gain = booster.get_score(importance_type="gain")
    total_gain = sum(gain.values()) or 1.0
    importance = {f: round(gain.get(f, 0.0) / total_gain, 4) for f in FEATURES}

    model_path = out_dir / MODEL_FILE
    booster.save_model(str(model_path))

    # Explainability analytics (correlation, PCA + PCA-vs-raw XGBoost, TreeSHAP).
    from .explain import build_analytics

    t_an = time.perf_counter()
    analytics = build_analytics(
        booster=booster, names=FEATURES,
        x_train=x_masked[tr], y_train=y[tr], x_val=x_masked[va], y_val=y[va],
        x_test=x_full[te], y_test=y_te, x_test_masked=x_masked[te],
        pred_raw_full=pred_full, pred_raw_masked=pred_masked,
        train_fn=lambda a, b, c, d, n, s: _train_booster(xgb, a, b, c, d, n, s),
        predict_fn=lambda b, a, n: _predict(xgb, b, a, n),
        metrics_fn=classification_metrics, regression_fn=regression_metrics, seed=seed,
    )
    analytics.update({
        "model": MODEL_NAME,
        "artifact": MODEL_FILE,
        "analytics_seconds": round(time.perf_counter() - t_an, 2),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
    })
    (out_dir / ANALYTICS_FILE).write_text(json.dumps(analytics, indent=2), encoding="utf-8")

    card = {
        "model": MODEL_NAME,
        "task": "Pc surrogate (log10 Foster Pc regression)",
        "purpose": ("Fast surrogate of the physics Foster Pc for triage and cross-check. "
                    "Never overwrites the physics Pc on an alert."),
        "label_source": "Foster Pc on simulated encounters (numerical integration of the "
                        "combined TLE-age covariance over the hard-body disk)",
        "features": FEATURES,
        "features_not_seen_by_model": ["combined covariance", "B-plane projected sigmas",
                                       "B-plane miss direction (only |radial| component)"],
        "training_samples": int(len(tr)),
        "validation_samples": int(len(va)),
        "heldout_samples": int(len(te)),
        "total_generated": int(sample_count),
        "positive_fraction_pc_ge_1e-4": round(float(np.mean(data["pc"] >= THRESHOLD_HIGH)), 4),
        "positive_fraction_pc_ge_1e-6": round(float(np.mean(data["pc"] >= THRESHOLD_ELEVATED)), 4),
        "boosting_rounds": int(booster.best_iteration + 1),
        "heldout": {
            "regression": regression_metrics(y_te, pred_full),
            "classification_at_1e-4": classification_metrics(y_te, pred_full, THRESHOLD_HIGH),
            "classification_at_1e-6": classification_metrics(y_te, pred_full, THRESHOLD_ELEVATED),
        },
        "heldout_masked_features": {
            "regression": regression_metrics(y_te, pred_masked),
            "classification_at_1e-4": classification_metrics(y_te, pred_masked, THRESHOLD_HIGH),
        },
        "heldout_without_radial_miss": {
            "regression": regression_metrics(y_te, pred_no_radial),
            "classification_at_1e-4": classification_metrics(y_te, pred_no_radial, THRESHOLD_HIGH),
        },
        "baselines": {
            "miss_distance_only_xgboost": {
                "regression": regression_metrics(y_te, pred_base),
                "classification_at_1e-4": classification_metrics(y_te, pred_base, THRESHOLD_HIGH),
            },
            "constant_mean": {
                "regression": regression_metrics(y_te, pred_const),
            },
        },
        "feature_importance_gain": importance,
        "explainability": {
            "artifact": ANALYTICS_FILE,
            "main_factor": analytics["main_factor"],
            "shap_global_top3": analytics["shap_global"][:3],
            "pca_n_components_95": analytics["pca"]["n_components_95"],
            "pca_vs_raw_verdict": analytics["pca_vs_raw"]["verdict"],
        },
        "reference_check": _check_against_reference(data),
        "recipe": {
            "generator": "app.ml.train_risk_surrogate.generate_encounters",
            "seed": seed,
            "split": "70% train / 10% validation (early stopping) / 20% held-out test",
            "altitude_km": "U(300, 2000)",
            "crossing_angle_deg": "85% U(10,175), 15% U(0.5,10)",
            "flight_path_angle_deg": "U(-1.5, 1.5)",
            "miss_distance_km": "60% loguniform(0.003, 25), 40% loguniform(0.003, 3)",
            "tle_age_h": f"loguniform(0.5, {TLE_AGE_MAX_H:.0f}) per object",
            "hbr_km": "loguniform(0.003, 0.050)",
            "covariance": _covariance_recipe_text(),
            "foster_quadrature": "polar: 20-node Gauss-Legendre radius x 64-node trapezoid angle",
            "target": "log10(max(Pc, 1e-12))",
            "missing_feature_masking": "radial 30%, altitude 30%, both TLE ages 15%",
        },
        "trajectory_model": None,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_seconds": None,
        "generation_seconds": round(t_gen, 2),
        "xgboost_version": xgb.__version__,
        "python": platform.python_version(),
        "artifact": MODEL_FILE,
    }
    try:
        card["trajectory_model"] = evaluate_trajectory_model()
    except Exception as exc:  # pragma: no cover
        card["trajectory_model"] = {"name": "Trajectory Predictor", "status": "unavailable", "error": str(exc)}
    card["train_seconds"] = round(time.perf_counter() - t0, 2)
    (out_dir / CARD_FILE).write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


if __name__ == "__main__":
    result = train()
    print(json.dumps({k: result[k] for k in ("training_samples", "heldout", "baselines",
                                             "feature_importance_gain", "explainability", "reference_check",
                                             "trajectory_model", "train_seconds",
                                             "positive_fraction_pc_ge_1e-4")}, indent=2))
