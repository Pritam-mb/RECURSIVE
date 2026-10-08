"""
analytics.py -- GET /api/analytics/model: explainability of the Pc surrogate.

Serves ``app/ml/artifacts/risk_model_analytics.json`` written at training time
by ``app.ml.train_risk_surrogate`` (via ``app.ml.explain``): Pearson/Spearman
correlation matrices (features + log10 Pc target), standardized PCA with
loadings and n_components_95, an honest XGBoost raw-vs-PCA comparison on the
same held-out split, and global exact TreeSHAP importances. Nothing here is
computed per request except reading the JSON (cached by file mtime).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException

from app.ml.artifacts import artifact_path
from app.ml.train_risk_surrogate import ANALYTICS_FILE, CARD_FILE

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_cache: dict[str, tuple[float, Any]] = {}


def _load(name: str) -> Any | None:
    path = artifact_path(name)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _cache.get(name)
    if hit and hit[0] == mtime:
        return hit[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    _cache[name] = (mtime, data)
    return data


@router.get("/model")
def model_analytics() -> dict[str, Any]:
    analytics = _load(ANALYTICS_FILE)
    if analytics is None:
        raise HTTPException(status_code=503,
                            detail="Analytics artifact missing; run python -m app.ml.train_risk_surrogate")
    corr = analytics.get("correlation", {})
    return {
        "features": analytics.get("features"),
        "target": analytics.get("target"),
        "feature_definitions": analytics.get("feature_definitions"),
        "feature_labels": analytics.get("feature_labels"),
        "correlation": {
            "variables": corr.get("variables"),
            "pearson": corr.get("pearson"),
            "spearman": corr.get("spearman"),
            "with_target": corr.get("with_target"),
            "pairwise_counts": corr.get("pairwise_counts"),
            "strongest_feature_pairs_spearman": corr.get("strongest_feature_pairs_spearman"),
            "missing_handling": corr.get("missing_handling"),
            "sample_size": corr.get("sample_size"),
        },
        "pca": analytics.get("pca"),
        "pca_vs_raw": analytics.get("pca_vs_raw"),
        "shap_global": analytics.get("shap_global"),
        "shap": {k: v for k, v in (analytics.get("shap") or {}).items() if k != "ranked"},
        "split_importance": analytics.get("split_importance"),
        "main_factor": analytics.get("main_factor"),
        "main_factor_label": analytics.get("main_factor_label"),
        "sample_size": analytics.get("sample_size"),
        "sample_sizes": analytics.get("sample_sizes"),
        "generated_at_utc": analytics.get("generated_at_utc"),
        "card": _load(CARD_FILE) or {},
    }
