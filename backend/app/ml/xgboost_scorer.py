"""
xgboost_scorer.py — Collision risk scorer backed by a Foster-trained XGBoost model.

Nothing is loaded or trained at import time. The heavy Foster-trained model in
risk_model_trained.json is loaded on first use *and only when the extended ML
pipeline is enabled* (ENABLE_EXTENDED_PIPELINE=1); otherwise scoring uses the
cheap boosted-stump heuristic. The bootstrap model is loaded lazily on first
score as well, so constructing a scorer is free.

Falls back to the legacy boosted-stump heuristic if XGBoost is unavailable.

Usage:
    from app.ml.xgboost_scorer import XGBoostScorer
    scorer = XGBoostScorer()
    risk = scorer.score(conjunction_event_dict)  # returns float 0–1
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import joblib
    import xgboost as xgb
except Exception:
    joblib = None
    xgb = None

from .artifacts import load_artifact, save_artifact
from .synthetic_data import generate_risk_dataset, sigmoid

logger = logging.getLogger(__name__)

RISK_ARTIFACT = "risk_model.json"
RISK_ARTIFACT_JOBLIB = "risk_model.joblib"


def _extended_pipeline_enabled() -> bool:
    """Mirror of main.ENABLE_EXTENDED_PIPELINE, read lazily so tests can patch it."""
    return os.getenv("ENABLE_EXTENDED_PIPELINE", "0") == "1"


def active_risk_model_path() -> str:
    """
    Identify which model scoring will actually use right now.

    Reported by the metrics endpoint so the numbers shown in the UI describe
    the deployed scorer rather than a parallel evaluation of a different model.
    """
    if not _extended_pipeline_enabled():
        return "boosted-stump"
    if _using_trained and _trained_model is not None:
        return "foster-xgboost"
    return "boosted-stump"


# ── Module-level trained model (loaded on first use, flag-gated) ─────────────
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAINED_MODEL_PATH = os.path.join(_MODULE_DIR, "risk_model_trained.json")

_trained_model = None
_using_trained = False
_trained_model_loaded = False
_trained_model_lock = threading.Lock()


def _ensure_trained_model() -> None:
    """
    Lazily load (or train) the Foster XGBoost model on first use.

    Loading is deferred so that merely importing this module has no side
    effects: no model is read, no training is triggered, and no artifacts are
    written until a score is actually requested. This lets the ML stack be
    gated behind a feature flag instead of running on every process start.
    """
    global _trained_model, _using_trained, _trained_model_loaded

    if _trained_model_loaded:
        return

    with _trained_model_lock:
        if _trained_model_loaded:
            return

        if xgb is None:
            logger.info("XGBoost library not installed - using heuristic fallback")
        else:
            try:
                from app.ml.train_xgboost import load_or_train as _load_or_train

                _trained_model = _load_or_train(_TRAINED_MODEL_PATH)
                _using_trained = True
                logger.info("XGBoost risk model loaded from %s", _TRAINED_MODEL_PATH)
            except Exception as _xgb_init_err:
                _trained_model = None
                _using_trained = False
                logger.warning("XGBoost model load/train failed: %s", _xgb_init_err)

        _trained_model_loaded = True


# ── Legacy boosted-stump model (fallback) ─────────────────────────────────────

@dataclass
class DecisionStump:
    feature_index: int
    threshold: float
    left_value: float
    right_value: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        mask = features[:, self.feature_index] <= self.threshold
        return np.where(mask, self.left_value, self.right_value)

    def to_dict(self) -> dict[str, float]:
        return {
            "feature_index": int(self.feature_index),
            "threshold": float(self.threshold),
            "left_value": float(self.left_value),
            "right_value": float(self.right_value),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecisionStump":
        return cls(
            feature_index=int(payload["feature_index"]),
            threshold=float(payload["threshold"]),
            left_value=float(payload["left_value"]),
            right_value=float(payload["right_value"]),
        )


class BoostedRiskModel:
    def __init__(
        self,
        base_logit: float,
        learning_rate: float,
        stumps: list[DecisionStump],
        feature_names: list[str],
    ):
        self.base_logit = float(base_logit)
        self.learning_rate = float(learning_rate)
        self.stumps = stumps
        self.feature_names = feature_names

    def predict_row(self, row: np.ndarray) -> float:
        raw = self.base_logit
        row_2d = row.reshape(1, -1)
        for stump in self.stumps:
            raw += self.learning_rate * float(stump.predict(row_2d)[0])
        return float(sigmoid(raw))

    def predict_batch(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        raw = np.full(matrix.shape[0], self.base_logit, dtype=float)
        for stump in self.stumps:
            raw += self.learning_rate * stump.predict(matrix)
        return sigmoid(raw).astype(float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_logit": float(self.base_logit),
            "learning_rate": float(self.learning_rate),
            "feature_names": list(self.feature_names),
            "stumps": [stump.to_dict() for stump in self.stumps],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoostedRiskModel":
        return cls(
            base_logit=float(payload["base_logit"]),
            learning_rate=float(payload["learning_rate"]),
            stumps=[DecisionStump.from_dict(item) for item in payload.get("stumps", [])],
            feature_names=list(payload.get("feature_names", [])),
        )


def _best_stump(features: np.ndarray, residual: np.ndarray) -> DecisionStump | None:
    best_stump = None
    best_loss = math.inf

    for feature_index in range(features.shape[1]):
        column = features[:, feature_index]
        thresholds = np.unique(np.quantile(column, np.linspace(0.1, 0.9, 9)))

        for threshold in thresholds:
            left_mask = column <= threshold
            right_mask = ~left_mask
            if not left_mask.any() or not right_mask.any():
                continue

            left_value = float(residual[left_mask].mean())
            right_value = float(residual[right_mask].mean())
            predictions = np.where(left_mask, left_value, right_value)
            loss = float(np.mean((residual - predictions) ** 2))

            if loss < best_loss:
                best_loss = loss
                best_stump = DecisionStump(
                    feature_index=feature_index,
                    threshold=float(threshold),
                    left_value=left_value,
                    right_value=right_value,
                )

    return best_stump


def train_risk_model(
    sample_count: int = 4096,
    rounds: int = 24,
    learning_rate: float = 0.18,
    seed: int = 7,
) -> BoostedRiskModel:
    features, targets, feature_names = generate_risk_dataset(sample_count=sample_count, seed=seed)
    base_prob = float(np.clip(targets.mean(), 1e-4, 1.0 - 1e-4))
    base_logit = float(np.log(base_prob / (1.0 - base_prob)))
    raw_scores = np.full(features.shape[0], base_logit, dtype=float)

    stumps: list[DecisionStump] = []
    for _ in range(rounds):
        probabilities = sigmoid(raw_scores)
        residual = targets - probabilities
        stump = _best_stump(features, residual)
        if stump is None:
            break

        stumps.append(stump)
        raw_scores += learning_rate * stump.predict(features)

    model = BoostedRiskModel(
        base_logit=base_logit,
        learning_rate=learning_rate,
        stumps=stumps,
        feature_names=feature_names,
    )
    save_artifact(
        RISK_ARTIFACT,
        {
            "trained_from": "synthetic_conjunctions",
            "sample_count": sample_count,
            "rounds": rounds,
            "model": model.to_dict(),
        },
    )
    logger.info("Trained risk model with %d stumps", len(stumps))
    return model


def train_risk_model_from_samples(
    features: np.ndarray,
    targets: np.ndarray,
    rounds: int = 24,
    learning_rate: float = 0.18,
) -> BoostedRiskModel:
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("features must be 2D and targets must be 1D")

    base_prob = float(np.clip(targets.mean(), 1e-4, 1.0 - 1e-4))
    base_logit = float(np.log(base_prob / (1.0 - base_prob)))
    raw_scores = np.full(features.shape[0], base_logit, dtype=float)

    stumps: list[DecisionStump] = []
    for _ in range(rounds):
        probabilities = sigmoid(raw_scores)
        residual = targets - probabilities
        stump = _best_stump(features, residual)
        if stump is None:
            break
        stumps.append(stump)
        raw_scores += learning_rate * stump.predict(features)

    model = BoostedRiskModel(
        base_logit=base_logit,
        learning_rate=learning_rate,
        stumps=stumps,
        feature_names=[],
    )
    save_artifact(
        RISK_ARTIFACT,
        {
            "trained_from": "shadow_mode_finetune",
            "sample_count": int(features.shape[0]),
            "rounds": rounds,
            "model": model.to_dict(),
        },
    )
    logger.info("Shadow-mode risk model fine-tune with %d samples", features.shape[0])
    return model


def load_or_train_risk_model(retrain: bool = False) -> BoostedRiskModel:
    from .artifacts import artifact_path

    if not retrain:
        try:
            if joblib is not None:
                path = artifact_path(RISK_ARTIFACT_JOBLIB)
                if path.exists():
                    clf = joblib.load(path)

                    class XGBWrapper:
                        def __init__(self, model):
                            self.model = model

                        def predict_proba(self, X):
                            return self.model.predict_proba(X)

                    wrapper = XGBWrapper(clf)
                    logger.info("Loaded XGBoost risk model from %s", path)
                    return _wrap_xgb_as_boosted(wrapper, feature_names=None)
        except Exception:
            logger.debug("Failed to load joblib xgboost model, falling back")

        payload = load_artifact(RISK_ARTIFACT)
        if payload and "model" in payload:
            return BoostedRiskModel.from_dict(payload["model"])

    return train_risk_model()


def _wrap_xgb_as_boosted(xgb_model, feature_names: list[str] | None = None) -> BoostedRiskModel:
    class Wrapped:
        def __init__(self, model):
            self.model = model

        def predict_row(self, row: np.ndarray) -> float:
            X = row.reshape(1, -1)
            proba = float(self.model.predict_proba(X)[0, 1])
            return proba

        def predict_batch(self, matrix: np.ndarray) -> np.ndarray:
            mat = np.asarray(matrix, dtype=float)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            return self.model.predict_proba(mat)[:, 1].astype(float)

    wrapper = Wrapped(xgb_model)
    adapter = BoostedRiskModel(base_logit=0.0, learning_rate=1.0, stumps=[], feature_names=feature_names or [])
    adapter.predict_row = wrapper.predict_row  # type: ignore
    adapter.predict_batch = wrapper.predict_batch  # type: ignore
    return adapter


def train_xgboost_model(sample_count: int = 16384, seed: int = 7) -> str:
    """Train a real XGBoost classifier on synthetic data and save as joblib."""
    if xgb is None or joblib is None:
        raise RuntimeError("xgboost or joblib not installed")

    features, targets, feature_names = generate_risk_dataset(sample_count=sample_count, seed=seed)
    labels = (targets >= 0.5).astype(int)

    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=seed, n_estimators=200)
    clf.fit(features, labels)

    from .artifacts import artifact_path
    out_path = artifact_path(RISK_ARTIFACT_JOBLIB)
    joblib.dump(clf, out_path)
    logger.info("Trained and saved XGBoost model to %s", out_path)
    return str(out_path)


# ── XGBoostScorer public class ────────────────────────────────────────────────

class XGBoostScorer:
    """
    Tabular collision-risk scorer.

    Primary path: Foster-trained XGBoost model loaded from risk_model_trained.json.
    Fallback: compact boosted-stump heuristic (legacy) when XGBoost unavailable.
    """

    # Feature keys expected for the trained Foster model (8 features)
    TRAINED_FEATURE_KEYS = [
        "miss_distance_km",
        "relative_velocity_kms",
        "tca_hours",
        "tle_age_hours",
        "altitude_km",
        "c_radial",
        "c_transverse",
        "c_normal",
    ]

    def __init__(self, model_path: str = None):
        self.model: BoostedRiskModel | None = None
        self.model_path = model_path
        # Bootstrapping is deferred: constructing a scorer must not load or
        # train anything, because scorers are built at import time by the
        # cascade planner regardless of the ML feature flag.
        self._bootstrap_lock = threading.Lock()

    def _ensure_bootstrap_model(self) -> None:
        if self.model is not None:
            return
        with self._bootstrap_lock:
            if self.model is None:
                self._load_or_train_model(self.model_path)

    def _load_or_train_model(self, path: str | None):
        try:
            if path:
                payload = load_artifact(path)
            else:
                payload = load_artifact(RISK_ARTIFACT)

            if payload and "model" in payload:
                self.model = BoostedRiskModel.from_dict(payload["model"])
                logger.info("Risk model loaded from %s", path or RISK_ARTIFACT)
                return

            self.model = train_risk_model()
        except Exception as error:
            logger.warning("Risk model bootstrap failed, retraining: %s", error)
            self.model = train_risk_model()

    def extract_features(self, event: dict[str, Any]) -> list[float]:
        """
        Extract the compact conjunction feature vector used by the scorer.
        """
        return [
            float(event.get("miss_distance_km", 0.0)),
            float(event.get("relative_speed_kmh", 0.0)),
            float(event.get("sat1_altitude_km", 400.0)),
            float(event.get("sat2_altitude_km", 400.0)),
            float(event.get("tca_minutes", 0.0)),
            float(event.get("sat1_size_m", 1.0)),
            float(event.get("sat2_size_m", 1.0)),
            float(event.get("uncertainty_km", 0.0)),
        ]

    def extract_trained_features(self, event: dict[str, Any]) -> np.ndarray:
        """Extract the 8-feature vector required by the Foster-trained model."""
        miss_km = float(event.get("miss_distance_km", 1.0))
        rel_vel_kmh = float(event.get("relative_speed_kmh", 27000.0))
        rel_vel_kms = rel_vel_kmh / 3600.0

        alt1 = float(event.get("sat1_altitude_km", 400.0))
        alt2 = float(event.get("sat2_altitude_km", 400.0))
        altitude_km = (alt1 + alt2) / 2.0

        tca_hours = float(event.get("tca_hours", event.get("tca_minutes", 720.0) / 60.0))
        tle_age_hours = float(event.get("tle_age_hours", 6.0))

        # Covariance features — use defaults if not available
        cov_ellipse = event.get("covariance_ellipse") or {}
        c_radial = float(event.get("c_radial", 0.0025))
        c_transverse = float(event.get("c_transverse",
            (cov_ellipse.get("a", 3000.0) / 1000.0 / 3.0) ** 2
        ))
        c_normal = (c_radial + c_transverse) / 2.0

        return np.array([
            miss_km,
            rel_vel_kms,
            tca_hours,
            tle_age_hours,
            altitude_km,
            c_radial,
            c_transverse,
            c_normal,
        ], dtype=float)

    def score(self, features: dict | list[float] | np.ndarray) -> float:
        """
        Score collision risk from 0.0 (safe) to 1.0 (critical).

        Primary path: Foster-trained XGBoost model, used only when the extended
        ML pipeline is enabled.
        Fallback: legacy boosted-stump heuristic.
        """
        # ── Primary path: Foster-trained XGBoost (flag-gated) ────────────────
        if _extended_pipeline_enabled():
            _ensure_trained_model()

        if _using_trained and _trained_model is not None and xgb is not None:
            try:
                if isinstance(features, dict):
                    feature_array = self.extract_trained_features(features).reshape(1, -1)
                else:
                    feature_array = np.asarray(features, dtype=float).reshape(1, -1)

                dmatrix = xgb.DMatrix(feature_array, feature_names=None)
                score_val = float(_trained_model.predict(dmatrix)[0])
                return float(np.clip(score_val, 0.0, 1.0))
            except Exception as trained_err:
                logger.warning(
                    "Trained XGBoost scorer failed: %s — using heuristic fallback",
                    trained_err,
                )

        # ── Fallback: legacy boosted-stump heuristic ───────────────────────
        self._ensure_bootstrap_model()

        if isinstance(features, dict):
            row = np.asarray(self.extract_features(features), dtype=float)
        else:
            row = np.asarray(features, dtype=float)

        if row.ndim != 1:
            row = row.reshape(-1)

        if self.model is None:
            return 0.0

        return float(np.clip(self.model.predict_row(row), 0.0, 1.0))
