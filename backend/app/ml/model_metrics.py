from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timezone
from math import sqrt
from typing import Any

import numpy as np

import logging

from .gat_cascade import GAT_META, TORCH_AVAILABLE, CascadeGAT
from .gnn_cascade import CascadeGNN
from .lstm_predictor import load_or_train_trajectory_model
from .xgboost_scorer import XGBoostScorer, active_risk_model_path
from .artifacts import load_artifact
from .synthetic_data import generate_graph_snapshots, generate_risk_dataset, generate_trajectory_dataset


logger = logging.getLogger(__name__)


_CACHE_PATH = Path(__file__).resolve().parent / "artifacts" / "model_metrics_cache.json"


def _artifact_signature() -> str:
    """Fingerprint the deployed model artifacts.

    The cache previously only checked ``risk_model_path``, so changing the graph
    rankers (for example replacing the aliased GAT with a real attention model)
    left the stale metrics in place and the UI kept reporting them. Any change to
    the risk model, graph model or attention bank now invalidates the cache.
    """
    parts: list[str] = [active_risk_model_path()]
    artifacts = Path(__file__).resolve().parent / "artifacts"
    for name in ("graph_model.json", "gat_model.json", "trajectory_model.json"):
        path = artifacts / name
        if not path.exists():
            parts.append(f"{name}:absent")
            continue
        stat = path.stat()
        parts.append(f"{name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=int).reshape(-1)
    probs = np.asarray(y_prob, dtype=float).reshape(-1)
    predicted = (probs >= threshold).astype(int)

    tp = int(np.sum((predicted == 1) & (truth == 1)))
    tn = int(np.sum((predicted == 0) & (truth == 0)))
    fp = int(np.sum((predicted == 1) & (truth == 0)))
    fn = int(np.sum((predicted == 0) & (truth == 1)))
    total = max(len(truth), 1)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "samples": int(total),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    diff = pred - truth
    mae = float(np.mean(np.abs(diff)))
    rmse = float(sqrt(np.mean(diff ** 2)))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "samples": int(len(truth)),
    }


def _graph_attention_status() -> dict[str, Any]:
    """Report the real state of the attention ranker for the status panel.

    The UI used to hardcode ``torchGeo = null`` and therefore always rendered
    "NOT INSTALLED" for a dependency it never queried. These values come from
    the loaded artifact instead, so the panel cannot show a fabricated status.
    """
    gat = CascadeGAT()
    bank = gat.attention
    status: dict[str, Any] = {
        "degenerate": gat.is_degenerate,
        "torch_available": TORCH_AVAILABLE,
        "trainer": None,
        "heads": None,
        "hidden_dim": None,
        "trained": False,
    }
    if bank is not None:
        status.update(
            {
                "heads": int(bank.heads),
                "hidden_dim": int(bank.hidden_dim),
                "trainer": "torch_autograd" if TORCH_AVAILABLE else "seeded_numpy",
            }
        )
    try:
        payload = load_artifact(GAT_META)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        status["trained"] = payload.get("trainer") == "torch_autograd"
        status["trainer"] = payload.get("trainer", status["trainer"])
        status["epochs"] = payload.get("epochs")
    return status


def _evaluate_graph_models() -> list[dict[str, Any]]:
    graphs = generate_graph_snapshots(sample_count=160, seed=31)
    gnn = CascadeGNN()
    gat = CascadeGAT()

    gnn_true = []
    gnn_prob = []
    gat_prob = []

    for graph in graphs:
      gnn_result = gnn.predict(graph["node_features"], graph["edge_index"], graph["edge_attr"])
      gat_result = gat.predict(graph["node_features"], graph["edge_index"], graph["edge_attr"])
      targets = np.asarray(graph["targets"], dtype=float).reshape(-1)

      gnn_true.extend((targets >= 0.5).astype(int).tolist())
      gnn_prob.extend(np.asarray(gnn_result["maneuver_probability"], dtype=float).reshape(-1).tolist())
      gat_prob.extend(np.asarray(gat_result["maneuver_probability"], dtype=float).reshape(-1).tolist())

    gnn_metrics = _binary_metrics(np.asarray(gnn_true), np.asarray(gnn_prob))
    gat_metrics = _binary_metrics(np.asarray(gnn_true), np.asarray(gat_prob))

    # Recorded explicitly so identical scores cannot be presented as two
    # independent pieces of evidence. When the GAT has no attention bank it
    # delegates to the GNN and the metrics are identical by construction.
    degenerate = gat.is_degenerate
    if degenerate:
        logger.warning(
            "GAT Cascade is delegating to CascadeGNN; its metrics are identical "
            "to Graph Cascade by construction and are not independent evidence."
        )

    return [
        {
            "name": "Graph Cascade",
            "type": "classification",
            "independent": not degenerate,
            "metrics": gnn_metrics,
        },
        {
            "name": "GAT Cascade",
            "type": "classification",
            "independent": not degenerate,
            "metrics": gat_metrics,
        },
    ]


def _evaluate_risk_model() -> dict[str, Any]:
    """
    Evaluate the risk scorer through its production entry point.

    This deliberately scores via ``XGBoostScorer.score`` rather than
    ``load_or_train_risk_model`` so the reported metrics describe the same
    model the cascade planner actually uses in production — including the
    Foster-trained XGBoost primary path and its feature-flag gating. Evaluating
    the legacy boosted-stump model directly here is what previously made the
    UI display metrics for a model that was never serving alerts.
    """
    features, targets, feature_names = generate_risk_dataset(sample_count=4096, seed=29)
    scorer = XGBoostScorer()

    # The scorer consumes an event mapping, so rebuild one per feature row: this
    # exercises the real production feature extraction instead of shortcutting
    # straight to a feature matrix.
    probabilities = np.array(
        [scorer.score(dict(zip(feature_names, row))) for row in features],
        dtype=float,
    )

    return {
        "name": "Risk Scorer",
        "type": "classification",
        "model_path": active_risk_model_path(),
        "metrics": _binary_metrics((targets >= 0.5).astype(int), probabilities),
    }


def _evaluate_trajectory_model() -> dict[str, Any]:
    sequences, targets, _ = generate_trajectory_dataset(sample_count=256, history_length=8, seed=27)
    model = load_or_train_trajectory_model()

    predictions = []
    truth = []
    for sequence, target in zip(sequences, targets, strict=False):
        predicted = model.predict_next(sequence)
        if predicted is None:
            continue
        predictions.append(predicted)
        truth.append(target)

    if not predictions:
        return {
            "name": "Trajectory Predictor",
            "type": "regression",
            "metrics": {"mae": 0.0, "rmse": 0.0, "samples": 0},
        }

    predicted = np.asarray(predictions, dtype=float)
    truth_array = np.asarray(truth, dtype=float)
    return {
        "name": "Trajectory Predictor",
        "type": "regression",
        "metrics": _regression_metrics(truth_array, predicted),
    }


def _load_cached_metrics() -> dict[str, Any] | None:
    if not _CACHE_PATH.exists():
        return None

    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if not isinstance(cached, dict):
            return None
        if not (cached.get("classification_models") and cached.get("regression_models")):
            return None
        # Invalidate the cache when any deployed model artifact changed, otherwise
        # the UI would keep displaying metrics computed for models that are no
        # longer the ones serving alerts.
        if cached.get("risk_model_path") != active_risk_model_path():
            return None
        if cached.get("artifact_signature") != _artifact_signature():
            return None
        return cached
    except Exception:
        return None


def _write_cached_metrics(payload: dict[str, Any]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CACHE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception:
        # Cache is an optimization only; fall back to the in-memory result.
        return


@lru_cache(maxsize=1)
def get_model_metrics() -> dict[str, Any]:
    cached = _load_cached_metrics()
    if cached is not None:
        return cached

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "risk_model_path": active_risk_model_path(),
        "artifact_signature": _artifact_signature(),
        "graph_attention": _graph_attention_status(),
        "classification_models": [
            _evaluate_risk_model(),
            *_evaluate_graph_models(),
        ],
        "regression_models": [
            _evaluate_trajectory_model(),
        ],
    }

    _write_cached_metrics(payload)
    return payload