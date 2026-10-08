"""
explain.py -- explainability analytics for the XGBoost Pc surrogate.

Answers three questions about ``app.ml.train_risk_surrogate`` with numbers that
are computed at training time and persisted to
``artifacts/risk_model_analytics.json`` (served by GET /api/analytics/model):

1. **How are the inputs related?**  Pearson and Spearman correlation matrices
   over the 7 features plus the target log10(Pc), and every feature's
   correlation with the target.  Missing values (the training set deliberately
   masks radial miss / altitude / TLE ages so the model learns to cope with
   alerts that lack them) are handled by *pairwise deletion*: each coefficient
   uses the rows where both variables are present, and the per-pair row count
   is reported.

2. **What does PCA say, and does PCA-before-XGBoost help?**  PCA by numpy SVD
   on standardized features (z-scores using training-split mean/std).  PCA
   cannot take NaN, so -- for PCA only -- missing values are imputed with the
   training-split median of that feature before standardizing; the XGBoost
   model itself never sees imputed values (it uses its learned missing-value
   branches).  We report explained-variance ratios, cumulative variance, the
   loadings (unit eigenvectors, components x features), the number of
   components needed for 95 % variance, and a plain-language reading of each
   leading component.  We then train the *same* XGBoost recipe on the first
   k = n_components_95 principal components (same rows, same split, same seed)
   and compare held-out MAE(log10 Pc) and F1 at Pc >= 1e-4 with the raw-feature
   model.  Trees split on one axis at a time, so rotating correlated inputs
   into components usually does not help them; the verdict string states what
   the numbers show either way.

3. **What does XGBoost boost?**  Exact TreeSHAP (``Booster.predict(...,
   pred_contribs=True)``; Lundberg et al. 2020, "From local explanations to
   global understanding with explainable AI for trees") on held-out rows.  For
   every row the contributions plus the bias term sum exactly to the model's
   log10 Pc prediction, so mean |contribution| per feature is in units of
   decades of Pc.  The feature with the largest mean |contribution| is the
   ``main_factor``.  Gain / weight / cover split importances are reported for
   comparison (they measure how trees are built, not how much each feature
   moves the prediction).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

TARGET_NAME = "log10_pc"

FEATURE_LABELS: dict[str, str] = {
    "miss_distance_km": "Miss distance",
    "radial_miss_km": "Radial miss component",
    "relative_speed_kms": "Relative speed",
    "tle_age_max_h": "Older TLE age",
    "tle_age_min_h": "Newer TLE age",
    "hbr_km": "Hard-body radius",
    "altitude_km": "Altitude",
    TARGET_NAME: "log10 Foster Pc (target)",
}

FEATURE_DEFINITIONS: dict[str, str] = {
    "miss_distance_km": "Predicted separation at time of closest approach (km).",
    "radial_miss_km": "|Radial component| of the miss vector (km); radial position error is "
                      "small, so radial misses are hard to explain away by uncertainty.",
    "relative_speed_kms": "Relative velocity magnitude at TCA (km/s); sets the B-plane orientation.",
    "tle_age_max_h": "Age of the older of the two TLEs (hours); drives the along-track covariance.",
    "tle_age_min_h": "Age of the fresher of the two TLEs (hours).",
    "hbr_km": "Combined hard-body radius of the two objects (km); the integration disk.",
    "altitude_km": "Altitude of the encounter (km).",
    TARGET_NAME: "log10 of the Foster 2D collision probability (physics label, floored at 1e-12).",
}


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def _r(x: float | None, nd: int = 4) -> float | None:
    if x is None or not math.isfinite(float(x)):
        return None
    return round(float(x), nd)


# ── Correlation (pairwise deletion) ─────────────────────────────────────────

def _rank(v: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), like scipy.stats.rankdata."""
    try:
        from scipy.stats import rankdata

        return rankdata(v).astype(float)
    except Exception:  # pragma: no cover - scipy is optional
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(len(v), dtype=float)
        ranks[order] = np.arange(1, len(v) + 1, dtype=float)
        return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def correlation_matrices(x: np.ndarray, names: list[str]) -> dict[str, Any]:
    """Pearson + Spearman matrices with pairwise-complete rows.

    x: (n, p) with NaN for missing. Returns symmetric matrices with a unit
    diagonal (rounded to 4 dp) and the per-pair complete-row counts.
    """
    x = np.asarray(x, dtype=float)
    p = x.shape[1]
    present = np.isfinite(x)
    pear = np.eye(p)
    spear = np.eye(p)
    counts = np.zeros((p, p), dtype=int)
    for i in range(p):
        counts[i, i] = int(present[:, i].sum())
        for j in range(i + 1, p):
            m = present[:, i] & present[:, j]
            counts[i, j] = counts[j, i] = int(m.sum())
            if m.sum() < 3:
                pear[i, j] = pear[j, i] = spear[i, j] = spear[j, i] = float("nan")
                continue
            a, b = x[m, i], x[m, j]
            pear[i, j] = pear[j, i] = _pearson(a, b)
            spear[i, j] = spear[j, i] = _pearson(_rank(a), _rank(b))

    def rnd(mat):
        return [[_r(v) for v in row] for row in mat]

    return {
        "variables": list(names),
        "pearson": rnd(pear),
        "spearman": rnd(spear),
        "pairwise_counts": counts.tolist(),
        "missing_handling": "pairwise deletion (each coefficient uses rows where both variables are present)",
    }


def strongest_pairs(matrix: list[list[float | None]], names: list[str], top: int = 5,
                    exclude: str | None = None) -> list[dict[str, Any]]:
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if exclude in (names[i], names[j]):
                continue
            v = matrix[i][j]
            if v is not None:
                pairs.append({"a": names[i], "b": names[j], "r": v})
    pairs.sort(key=lambda d: -abs(d["r"]))
    return pairs[:top]


# ── PCA (numpy SVD, standardized) ───────────────────────────────────────────

class StandardPCA:
    """Median-impute -> z-score -> SVD. Fitted on training rows only."""

    def fit(self, x: np.ndarray) -> "StandardPCA":
        x = np.asarray(x, dtype=float)
        self.median_ = np.nanmedian(x, axis=0)
        z = self._impute(x)
        self.mean_ = z.mean(axis=0)
        std = z.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)
        z = (z - self.mean_) / self.scale_
        _, s, vt = np.linalg.svd(z, full_matrices=False)
        # Deterministic sign: largest-|loading| entry of each component is positive.
        signs = np.sign(vt[np.arange(len(vt)), np.argmax(np.abs(vt), axis=1)])
        signs[signs == 0] = 1.0
        self.components_ = vt * signs[:, None]
        var = s ** 2
        self.explained_variance_ratio_ = var / var.sum()
        self.missing_fraction_ = np.mean(~np.isfinite(x), axis=0)
        return self

    def _impute(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=float, copy=True)
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = np.take(self.median_, np.nonzero(bad)[1])
        return x

    def transform(self, x: np.ndarray, k: int | None = None) -> np.ndarray:
        z = (self._impute(x) - self.mean_) / self.scale_
        comps = self.components_ if k is None else self.components_[:k]
        return z @ comps.T


def n_components_for(cumulative: np.ndarray, level: float = 0.95) -> int:
    return int(min(len(cumulative), np.searchsorted(cumulative, level - 1e-12) + 1))


def interpret_component(loadings: np.ndarray, names: list[str], evr: float, idx: int,
                        threshold: float = 0.3) -> dict[str, Any]:
    order = np.argsort(-np.abs(loadings))
    dominant = [(names[i], float(loadings[i])) for i in order if abs(loadings[i]) >= threshold] \
        or [(names[order[0]], float(loadings[order[0]]))]
    terms = ", ".join(f"{'+' if w >= 0 else '-'}{abs(w):.2f} {label(n)}" for n, w in dominant)
    if len(dominant) == 1:
        reading = f"essentially {label(dominant[0][0])} on its own"
    else:
        same = all(np.sign(w) == np.sign(dominant[0][1]) for _, w in dominant)
        reading = ("these move together" if same else
                   "a contrast: the + and - groups move in opposite directions")
    return {
        "component": f"PC{idx + 1}",
        "explained_variance_ratio": _r(evr),
        "dominant_loadings": [{"feature": n, "label": label(n), "loading": _r(w, 3)} for n, w in dominant],
        "interpretation": f"PC{idx + 1} ({100 * evr:.1f}% of variance): {terms} -- {reading}.",
    }


def pca_summary(pca: StandardPCA, names: list[str], top: int = 4) -> dict[str, Any]:
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n95 = n_components_for(cum, 0.95)
    return {
        "explained_variance_ratio": [_r(v, 5) for v in evr],
        "cumulative": [_r(v, 5) for v in cum],
        "loadings": [[_r(v) for v in row] for row in pca.components_],
        "loadings_kind": "unit eigenvectors of the feature correlation matrix (rows = components)",
        "features": list(names),
        "n_components_95": n95,
        "standardized": True,
        "imputation": "training-split median per feature, PCA only (XGBoost uses native missing-value branches)",
        "missing_fraction_imputed": {n: _r(f) for n, f in zip(names, pca.missing_fraction_)},
        "medians": {n: _r(m, 6) for n, m in zip(names, pca.median_)},
        "components": [interpret_component(pca.components_[i], names, float(evr[i]), i)
                       for i in range(min(len(evr), max(top, n95)))],
    }


# ── TreeSHAP ────────────────────────────────────────────────────────────────

def tree_contributions(booster, x: np.ndarray, names: list[str], *, approx: bool = False) -> np.ndarray:
    """(n, p+1) contributions in log10 Pc; last column is the bias (expected value).

    approx=False -> exact path-dependent TreeSHAP; approx=True -> Saabas
    path attribution. Both are additive: row sum == model prediction.
    """
    import xgboost as xgb

    d = xgb.DMatrix(np.asarray(x, dtype=np.float32), feature_names=list(names), missing=np.nan)
    return booster.predict(d, pred_contribs=True, approx_contribs=approx)


def shap_global(booster, x: np.ndarray, names: list[str]) -> dict[str, Any]:
    contrib = tree_contributions(booster, x, names)
    feats = contrib[:, :-1]
    mean_abs = np.mean(np.abs(feats), axis=0)
    mean_signed = np.mean(feats, axis=0)
    order = np.argsort(-mean_abs)
    total = float(mean_abs.sum()) or 1.0
    ranked = [{
        "feature": names[i],
        "label": label(names[i]),
        "mean_abs_contribution_log10": _r(mean_abs[i]),
        "mean_contribution_log10": _r(mean_signed[i]),
        "share": _r(mean_abs[i] / total),
    } for i in order]
    return {
        "ranked": ranked,
        "main_factor": names[order[0]],
        "base_log10": _r(float(np.mean(contrib[:, -1]))),
        "additivity_max_abs_error": None,  # filled by caller (needs the prediction)
        "_contrib": contrib,
    }


def split_importances(booster, names: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for kind in ("gain", "total_gain", "weight", "cover"):
        raw = booster.get_score(importance_type=kind)
        tot = sum(raw.values()) or 1.0
        out[kind] = {n: _r(raw.get(n, 0.0) / tot) for n in names}
    return out


# ── Full analytics bundle (called by the trainer) ───────────────────────────

def build_analytics(*, booster, names: list[str], x_train: np.ndarray, y_train: np.ndarray,
                    x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                    x_test_masked: np.ndarray, pred_raw_full: np.ndarray, pred_raw_masked: np.ndarray,
                    train_fn: Callable, predict_fn: Callable, metrics_fn: Callable,
                    regression_fn: Callable, seed: int, shap_rows: int = 2000) -> dict[str, Any]:
    """Compute correlation, PCA (+ PCA-vs-raw XGBoost) and global TreeSHAP.

    train_fn(x_tr, y_tr, x_va, y_va, names, seed) -> booster;
    predict_fn(booster, x, names) -> log10 Pc; metrics_fn(y, pred, thr) -> dict with f1;
    regression_fn(y, pred) -> dict with mae_log10_pc.
    """
    # 1. Correlation over features + target on the training split (as the model saw it).
    all_names = list(names) + [TARGET_NAME]
    corr = correlation_matrices(np.column_stack([x_train, y_train]), all_names)
    p = len(names)
    with_target = {
        "pearson": {n: corr["pearson"][i][p] for i, n in enumerate(names)},
        "spearman": {n: corr["spearman"][i][p] for i, n in enumerate(names)},
    }
    corr["with_target"] = with_target
    corr["strongest_feature_pairs_spearman"] = strongest_pairs(corr["spearman"], all_names, 5, exclude=TARGET_NAME)
    corr["sample_size"] = int(len(x_train))

    # 2. PCA on standardized (median-imputed) training features.
    pca = StandardPCA().fit(x_train)
    pca_info = pca_summary(pca, names)
    k = pca_info["n_components_95"]
    pairs = corr["strongest_feature_pairs_spearman"]
    strong = [f"{label(d['a'])} ~ {label(d['b'])} (rho={d['r']:.2f})" for d in pairs if abs(d["r"]) >= 0.3]
    pca_info["summary"] = (
        f"{k} of {p} components are needed for 95% of the variance. Only "
        f"{len(strong)} feature pair(s) are materially correlated ({'; '.join(strong) or 'none'}); the "
        f"remaining features are nearly independent by construction of the encounter sampler, so their "
        f"variance spreads almost evenly over components and there is little redundancy for PCA to compress.")

    def pca_names(n):
        return [f"PC{i + 1}" for i in range(n)]

    def run_pca_model(n_comp):
        nm = pca_names(n_comp)
        b = train_fn(pca.transform(x_train, n_comp), y_train, pca.transform(x_val, n_comp), y_val, nm, seed)
        full = predict_fn(b, pca.transform(x_test, n_comp), nm)
        masked = predict_fn(b, pca.transform(x_test_masked, n_comp), nm)
        return b, full, masked

    def score(pred_full, pred_masked):
        return {
            "mae_log10": regression_fn(y_test, pred_full)["mae_log10_pc"],
            "f1_1e4": metrics_fn(y_test, pred_full, 1e-4)["f1"],
            "masked_features": {
                "mae_log10": regression_fn(y_test, pred_masked)["mae_log10_pc"],
                "f1_1e4": metrics_fn(y_test, pred_masked, 1e-4)["f1"],
            },
        }

    raw = score(pred_raw_full, pred_raw_masked)
    raw["n_features"] = p
    _, pk_full, pk_masked = run_pca_model(k)
    pca_k = score(pk_full, pk_masked)
    pca_k["n_components"] = k
    if k < p:
        _, pa_full, pa_masked = run_pca_model(p)
        pca_all = score(pa_full, pa_masked)
        pca_all["n_components"] = p
    else:
        pca_all = dict(pca_k)

    d_mae = pca_k["mae_log10"] - raw["mae_log10"]
    d_f1 = pca_k["f1_1e4"] - raw["f1_1e4"]
    if d_mae > 0.01:
        verdict = (f"Raw features win: XGBoost on the 7 raw features scores held-out MAE "
                   f"{raw['mae_log10']:.3f} dex / F1@1e-4 {raw['f1_1e4']:.3f}, versus "
                   f"{pca_k['mae_log10']:.3f} dex / {pca_k['f1_1e4']:.3f} on the first {k} principal "
                   f"components (95% variance). Even with all {p} components (a pure rotation, no "
                   f"information lost) MAE is {pca_all['mae_log10']:.3f} dex: trees split one axis at a "
                   f"time, so rotating physically meaningful axes (miss distance, TLE age, HBR) into "
                   f"mixtures makes the Pc boundary harder to carve, and the dropped low-variance "
                   f"components still carry Pc signal. PCA is used here to describe the inputs, not "
                   f"as a preprocessing step.")
    elif d_mae < -0.01:
        verdict = (f"PCA helps: {k} components give MAE {pca_k['mae_log10']:.3f} dex / F1@1e-4 "
                   f"{pca_k['f1_1e4']:.3f} vs raw {raw['mae_log10']:.3f} / {raw['f1_1e4']:.3f}.")
    else:
        verdict = (f"No material difference: raw MAE {raw['mae_log10']:.3f} vs PCA({k}) "
                   f"{pca_k['mae_log10']:.3f} dex (F1@1e-4 change {d_f1:+.3f}); raw features are kept "
                   f"because they are directly interpretable.")

    pca_vs_raw = {
        "raw_xgb": raw,
        "pca_xgb": pca_k,
        "pca_xgb_all_components": pca_all,
        "delta_mae_log10_pca_minus_raw": _r(d_mae),
        "delta_f1_1e4_pca_minus_raw": _r(d_f1),
        "protocol": ("same rows, same 70/10/20 split, same seed and XGBoost hyper-parameters; PCA fitted "
                     "on training rows only (median imputation + z-score); evaluated on the held-out 20% "
                     "with full features (primary) and with the training-time missing-feature masking"),
        "verdict": verdict,
    }

    # 3. Global exact TreeSHAP on held-out rows (full features).
    rng = np.random.default_rng(seed + 7)
    idx = rng.choice(len(x_test), size=min(shap_rows, len(x_test)), replace=False)
    try:
        booster.set_param({"nthread": 4})
    except Exception:
        pass
    sg = shap_global(booster, x_test[idx], names)
    contrib = sg.pop("_contrib")
    pred = predict_fn(booster, x_test[idx], names)
    sg["additivity_max_abs_error"] = _r(float(np.max(np.abs(contrib.sum(axis=1) - pred))), 8)
    sg["sample_size"] = int(len(idx))
    sg["method"] = "exact path-dependent TreeSHAP (xgboost pred_contribs=True) on held-out rows"
    sg["units"] = "decades of Pc (log10)"

    return {
        "features": list(names),
        "target": TARGET_NAME,
        "feature_definitions": {n: FEATURE_DEFINITIONS.get(n, n) for n in all_names},
        "feature_labels": {n: label(n) for n in all_names},
        "correlation": corr,
        "pca": pca_info,
        "pca_vs_raw": pca_vs_raw,
        "shap": sg,
        "shap_global": [{"feature": r["feature"], "label": r["label"],
                         "mean_abs_contribution_log10": r["mean_abs_contribution_log10"],
                         "mean_contribution_log10": r["mean_contribution_log10"],
                         "share": r["share"]} for r in sg["ranked"]],
        "main_factor": sg["main_factor"],
        "main_factor_label": label(sg["main_factor"]),
        "split_importance": split_importances(booster, names),
        "sample_size": int(len(x_train)),
        "sample_sizes": {"correlation_pca_rows": int(len(x_train)), "shap_rows": int(len(idx)),
                         "heldout_rows": int(len(x_test))},
    }
