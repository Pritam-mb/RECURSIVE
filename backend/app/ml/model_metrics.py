"""
model_metrics.py -- honest model status for /api/model-metrics.

Everything reported here is either
* read from a model card written at training time with held-out metrics
  (``artifacts/risk_model_card.json`` from ``app.ml.train_risk_surrogate``), or
* a live counter of what the deployed model actually did this process, or
* a plain description of what a model is trained on and what it is used for.

Nothing is evaluated on the training formula's own labels and presented as
accuracy. Importing this module does not import torch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from . import risk_api
from .artifacts import load_artifact


@lru_cache(maxsize=1)
def _risk_card() -> dict[str, Any] | None:
    return risk_api.model_card()


def _risk_model_section() -> dict[str, Any]:
    card = _risk_card()
    available = risk_api.model_available()
    if card is None:
        return {
            "name": risk_api.MODEL_NAME,
            "available": available,
            "trained": False,
            "note": "No model card; run python -m app.ml.train_risk_surrogate",
            "live": risk_api.live_counters.snapshot(),
        }
    heldout = card.get("heldout", {})
    cls = heldout.get("classification_at_1e-4", {})
    baseline = card.get("baselines", {}).get("miss_distance_only_xgboost", {})
    return {
        "name": card.get("model"),
        "available": available,
        "trained": available,
        "status": "production" if available else "artifact missing",
        "role": "triage + cross-check of physics Foster Pc; never overwrites alert Pc",
        "task": card.get("task"),
        "label_source": card.get("label_source"),
        "features": card.get("features"),
        "features_not_seen_by_model": card.get("features_not_seen_by_model"),
        "training_samples": card.get("training_samples"),
        "heldout_samples": card.get("heldout_samples"),
        "heldout": heldout,
        "heldout_masked_features": card.get("heldout_masked_features"),
        "heldout_without_radial_miss": card.get("heldout_without_radial_miss"),
        "baselines": card.get("baselines"),
        "summary": {
            "heldout_mae_log10_pc": heldout.get("regression", {}).get("mae_log10_pc"),
            "precision_at_1e-4": cls.get("precision"),
            "recall_at_1e-4": cls.get("recall"),
            "f1_at_1e-4": cls.get("f1"),
            "baseline_name": "miss-distance-only XGBoost",
            "baseline_mae_log10_pc": baseline.get("regression", {}).get("mae_log10_pc"),
            "baseline_f1_at_1e-4": baseline.get("classification_at_1e-4", {}).get("f1"),
        },
        "feature_importance": card.get("feature_importance_gain"),
        "reference_check": card.get("reference_check"),
        "recipe": card.get("recipe"),
        "trained_at_utc": card.get("trained_at_utc"),
        "train_seconds": card.get("train_seconds"),
        "live": risk_api.live_counters.snapshot(),
    }


def _trajectory_section() -> dict[str, Any]:
    card = _risk_card() or {}
    traj = card.get("trajectory_model")
    if isinstance(traj, dict):
        return traj
    return {
        "name": "Trajectory Predictor",
        "status": "unevaluated",
        "note": "Run python -m app.ml.train_risk_surrogate to benchmark against baselines.",
    }


def _graph_section() -> dict[str, Any]:
    """Graph rankers (GNN / GAT): status from artifacts only (no torch import)."""
    gat = load_artifact("gat_model.json") or {}
    gnn = load_artifact("graph_model.json") or {}
    label_note = ("Trained on synthetic graph snapshots whose labels are a heuristic "
                  "sigmoid of miss distance / speed / TCA (app.ml.synthetic_data). "
                  "Agreement with those labels is not evidence of real-world skill.")
    return {
        "role": "cross-check only: cascade depth and edges come from physics alerts (BFS), "
                "not from these models",
        "label_source": "synthetic heuristic labels (circular)",
        "note": label_note,
        "gat": {
            "trained": bool(gat),
            "trainer": gat.get("trainer"),
            "heads": gat.get("heads"),
            "hidden_dim": gat.get("hidden_dim"),
            "epochs": gat.get("epochs"),
            "training_graphs": gat.get("sample_count"),
        },
        "gnn": {
            "trained": bool(gnn),
            "training_samples": gnn.get("sample_count"),
        },
    }


def _operator_feedback_section() -> dict[str, Any]:
    try:
        from .rlhf_store import get_stats

        stats = get_stats()
    except Exception:
        stats = None
    return {
        "name": "Operator feedback log",
        "trains_any_model": False,
        "note": "Approve/reject decisions are stored for audit; no model is trained on them "
                "(this is not RLHF).",
        "stats": stats,
    }


def get_model_metrics() -> dict[str, Any]:
    risk = _risk_model_section()
    traj = _trajectory_section()
    summary = risk.get("summary") or {}
    mae = traj.get("mae_km") if isinstance(traj.get("mae_km"), dict) else {}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "risk_model_path": risk.get("name") if risk.get("available") else "unavailable",
        "risk_model": risk,
        "trajectory_model": traj,
        "graph_models": _graph_section(),
        "operator_feedback": _operator_feedback_section(),
        # Legacy list shape (kept so older clients keep working); values are the
        # held-out numbers from the model card, not re-evaluations on training labels.
        "classification_models": [
            {
                "name": "Risk Scorer",
                "type": "pc_surrogate",
                "model_path": risk.get("name"),
                "training_samples": risk.get("training_samples"),
                "feature_importance": risk.get("feature_importance"),
                "metrics": {
                    "precision": summary.get("precision_at_1e-4"),
                    "recall": summary.get("recall_at_1e-4"),
                    "f1": summary.get("f1_at_1e-4"),
                    "mae_log10_pc": summary.get("heldout_mae_log10_pc"),
                    "threshold_pc": 1e-4,
                    "samples": risk.get("heldout_samples") or 0,
                    "split": "held-out",
                },
            }
        ],
        "regression_models": [
            {
                "name": "Trajectory Predictor",
                "type": "regression",
                "status": traj.get("status"),
                "metrics": {
                    "mae": mae.get("model"),
                    "persistence_mae": mae.get("persistence"),
                    "linear_mae": mae.get("linear_extrapolation"),
                    "samples": traj.get("eval_samples", 0),
                },
            }
        ],
    }
