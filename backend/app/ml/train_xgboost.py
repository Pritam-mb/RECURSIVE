"""
train_xgboost.py — Generate Foster-based training data and train XGBoost risk classifier.

On first run (no model file found), this module:
  1. Generates 5 000 physically realistic conjunction samples using the
     Foster B-plane integration from analytics.py
  2. Trains an XGBoost binary classifier to predict high-risk conjunctions
     (Pc > 1e-4 is label=1)
  3. Prints sklearn metrics (precision, recall, F1, confusion matrix) to stdout
  4. Saves the trained model as risk_model_trained.json (XGBoost native format)

On subsequent runs it loads the saved model instantly.

Usage:
    from app.ml.train_xgboost import load_or_train
    model = load_or_train(model_path)
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Feature names for DMatrix construction (8 features)
FEATURE_NAMES = [
    "miss_distance_km",
    "relative_velocity_kms",
    "tca_hours",
    "tle_age_hours",
    "altitude_km",
    "c_radial",
    "c_transverse",
    "c_normal",
]


# ── Data generation ───────────────────────────────────────────────────────────

def _build_state_at_altitude(altitude_km: float, phase_rad: float = 0.0, inc_deg: float = 51.6) -> dict:
    """
    Build a circular orbit state vector in ECI frame.

    Uses a simple circular orbit at given altitude and inclination.
    Position is placed at phase_rad along the orbit.
    """
    R_EARTH = 6371.0  # km
    MU = 398600.4418  # km³/s²

    r = R_EARTH + altitude_km
    v = math.sqrt(MU / r)  # km/s

    inc = math.radians(inc_deg)
    # Position in orbital plane
    x_orb = r * math.cos(phase_rad)
    y_orb = r * math.sin(phase_rad)

    # Rotate to ECI (simplified — ignores RAAN, argument of perigee)
    cos_i = math.cos(inc)
    sin_i = math.sin(inc)

    x = x_orb
    y = y_orb * cos_i
    z = y_orb * sin_i

    # Velocity (perpendicular to position, in orbital plane)
    vx = -v * math.sin(phase_rad)
    vy = v * math.cos(phase_rad) * cos_i
    vz = v * math.cos(phase_rad) * sin_i

    return {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz}


def generate_training_dataset(n_samples: int = 5000, seed: int = 42) -> list[dict]:
    """
    Generate physically realistic conjunction samples using Foster B-plane integration.

    Each sample represents a conjunction scenario with:
      - Randomly generated orbital parameters
      - Foster Pc computed via analytics.compute_collision_probability
      - Binary label: 1 if Pc > 1e-4 (NASA CARA watch threshold), else 0

    Returns list of sample dicts with features and labels.
    """
    try:
        from app.core.analytics import compute_collision_probability
        _analytics_ok = True
    except ImportError:
        logger.error("analytics.py not available — cannot generate Foster-based training data")
        _analytics_ok = False

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    samples = []

    # Target ~20% positive class to ensure stratified split works.
    # Close-approach scenarios (miss < 0.5 km) reliably produce Pc > 1e-4.
    POSITIVE_FRACTION = 0.20

    print(f"Generating {n_samples} Foster-based conjunction samples...")
    t_start = time.time()

    for i in range(n_samples):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            print(f"  {i+1}/{n_samples} samples ({elapsed:.1f}s elapsed)")

        try:
            # ── Force positive-class scenarios at the target rate ────────
            force_positive = rng.random() < POSITIVE_FRACTION

            # ── Physically realistic parameter sampling ──────────────────
            altitude_km = rng.uniform(200.0, 2000.0)

            if force_positive:
                # Very close approach: miss 0.001 – 0.5 km, nearly coplanar,
                # high rel-velocity crossings reliably push Pc above 1e-4.
                miss_distance_km = float(np_rng.uniform(0.001, 0.5))
            else:
                # Background conjunction: log-normal, most are far
                miss_raw = np_rng.lognormal(mean=2.5, sigma=1.2)
                miss_distance_km = float(np.clip(miss_raw, 0.5, 500.0))

            # Relative velocity km/s
            rel_vel_kms = rng.uniform(1.0, 15.0)

            # TLE age hours
            tle_age_hours = rng.uniform(0.1, 48.0)

            # Inclination difference (drives crossing geometry)
            inc_diff_deg = rng.uniform(0.0, 180.0)

            # TCA: physically, close approaches happen minutes before TCA;
            # use a realistic range rather than deriving from miss/vel.
            if force_positive:
                tca_hours = rng.uniform(0.05, 2.0)   # 3 min – 2 hr: urgent
            else:
                tca_hours = rng.uniform(0.5, 48.0)

            if _analytics_ok:
                # ── Build states for Foster integration ──────────────────
                phase_p = rng.uniform(0.0, 2 * math.pi)
                inc_p = rng.uniform(20.0, 98.0)
                state_p = _build_state_at_altitude(altitude_km, phase_p, inc_p)

                # Place sat_q at miss_distance_km offset
                inc_q = inc_p + inc_diff_deg * rng.uniform(-0.5, 0.5)
                # angular offset corresponding to miss distance
                phase_q = phase_p + (miss_distance_km / (6371 + altitude_km))
                state_q_base = _build_state_at_altitude(altitude_km, phase_q, inc_q)

                # Add velocity perturbation to simulate rel_vel_kms
                dv = rel_vel_kms
                dv_dir = np.array([
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 1),
                ], dtype=float)
                dv_norm = float(np.linalg.norm(dv_dir))
                if dv_norm > 1e-8:
                    dv_dir /= dv_norm
                state_q = {
                    "x": state_q_base["x"],
                    "y": state_q_base["y"],
                    "z": state_q_base["z"],
                    "vx": state_q_base["vx"] + dv * dv_dir[0],
                    "vy": state_q_base["vy"] + dv * dv_dir[1],
                    "vz": state_q_base["vz"] + dv * dv_dir[2],
                }

                analytics_result = compute_collision_probability(
                    state_p, state_q, tle_age_hours=tle_age_hours
                )
                p_collision = analytics_result["p_collision"]
                cov_2d = np.array(analytics_result["cov_2d"])
                c_radial = float(cov_2d[0, 0])
                c_transverse = float(cov_2d[1, 1])
                c_normal = (c_radial + c_transverse) / 2.0
            else:
                # Fallback: simple Gaussian approximation
                sigma = max(0.5, miss_distance_km / 6.0)
                hbr = 0.010
                p_collision = float(math.exp(-0.5 * (miss_distance_km / sigma) ** 2) * (hbr / sigma) ** 2)
                p_collision = min(1.0, max(0.0, p_collision))
                c_radial = 0.0025
                c_transverse = (1.0 * (1 + tle_age_hours / 24.0)) ** 2
                c_normal = 0.0025

            # Binary label: Pc > 1e-4 is high risk (NASA CARA watch threshold)
            label = 1 if p_collision > 1e-4 else 0

            samples.append({
                "miss_distance_km": miss_distance_km,
                "relative_velocity_kms": rel_vel_kms,
                "tca_hours": tca_hours,
                "tle_age_hours": tle_age_hours,
                "altitude_km": altitude_km,
                "c_radial": c_radial,
                "c_transverse": c_transverse,
                "c_normal": c_normal,
                "p_collision": p_collision,
                "label": label,
            })

        except Exception as sample_err:
            logger.debug("Sample %d failed: %s", i, sample_err)
            continue

    elapsed = time.time() - t_start
    n_pos = sum(s["label"] for s in samples)
    n_neg = len(samples) - n_pos
    print(f"Generated {len(samples)} samples in {elapsed:.1f}s  (positive={n_pos}, negative={n_neg})")
    return samples


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_save(samples: list[dict], model_path: str) -> Any:
    """
    Train XGBoost binary classifier on Foster-generated samples and save.

    Prints full sklearn metrics to stdout on completion.
    Returns the trained XGBoost Booster object.
    """
    import xgboost as xgb
    from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split

    if not samples:
        raise ValueError("No training samples provided")

    X = np.array([[
        s["miss_distance_km"],
        s["relative_velocity_kms"],
        s["tca_hours"],
        s["tle_age_hours"],
        s["altitude_km"],
        s["c_radial"],
        s["c_transverse"],
        s["c_normal"],
    ] for s in samples], dtype=float)

    y = np.array([s["label"] for s in samples], dtype=int)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0:
        raise ValueError("No positive samples — cannot train binary classifier")
    if n_neg == 0:
        raise ValueError("No negative samples — cannot train binary classifier")

    class_weight = float(n_neg) / float(n_pos)

    # Stratified 80/20 split — fall back to random if class too rare for stratify
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        logger.warning("Stratified split failed (too few positive samples), using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "scale_pos_weight": class_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "nthread": -1,
    }

    print("Training XGBoost with 200 rounds...")
    t0 = time.time()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtest, "test")],
        verbose_eval=50,
    )
    elapsed = time.time() - t0

    # Evaluate
    y_prob = booster.predict(dtest)
    y_pred = (y_prob >= 0.5).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== XGBoost Risk Classifier Training Complete ===")
    print(f"Training time:        {elapsed:.1f}s")
    print(f"Training samples:     {len(X_train)}  (positive: {int(y_train.sum())}, negative: {int(len(y_train) - y_train.sum())})")
    print(f"Test set size:        {len(X_test)}")
    print(f"Test set precision:   {precision:.4f}")
    print(f"Test set recall:      {recall:.4f}")
    print(f"Test set F1:          {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["low-risk (Pc<1e-4)", "high-risk (Pc>=1e-4)"]))
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print("=" * 50)

    # Save model
    booster.save_model(model_path)
    print(f"Model saved to: {model_path}")

    # Save metrics sidecar
    metrics = {
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "n_positive_train": int(y_train.sum()),
        "n_negative_train": int(len(y_train) - y_train.sum()),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "feature_names": FEATURE_NAMES,
    }
    metrics_path = model_path.replace(".json", "_metrics.json")
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to:  {metrics_path}")
    except Exception as save_err:
        logger.warning("Could not save metrics: %s", save_err)

    return booster


# ── Load or train ─────────────────────────────────────────────────────────────

def load_or_train(model_path: str, force_retrain: bool = False) -> Any:
    """
    Load a previously trained XGBoost model from model_path, or train one
    from scratch using Foster-based data generation.

    Parameters
    ----------
    model_path     : path to save / load the XGBoost JSON model
    force_retrain  : if True, always regenerate data and retrain

    Returns
    -------
    XGBoost Booster object
    """
    import xgboost as xgb

    if not force_retrain and os.path.exists(model_path):
        try:
            booster = xgb.Booster()
            booster.load_model(model_path)
            logger.info("XGBoost model loaded from %s", model_path)
            return booster
        except Exception as load_err:
            logger.warning("Failed to load XGBoost model: %s — retraining", load_err)

    print("Generating Foster-based training data...")
    samples = generate_training_dataset(5000)
    booster = train_and_save(samples, model_path)
    return booster


if __name__ == "__main__":
    # Run training directly: python -m app.ml.train_xgboost
    import sys
    model_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(model_dir, "risk_model_trained.json")
    force = "--force" in sys.argv
    load_or_train(out_path, force_retrain=force)
