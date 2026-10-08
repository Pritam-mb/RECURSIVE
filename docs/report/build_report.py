#!/usr/bin/env python
"""Build the Orbital Sentinel judges' report (PDF).

Run from the repo root:

    python docs/report/build_report.py                # live backend + scenario run
    python docs/report/build_report.py --no-scenario  # reuse cached scenario data
    python docs/report/build_report.py --offline      # cached data only, no HTTP

Every number and chart is taken from real data:
  * live backend endpoints on http://127.0.0.1:8000 (cached to docs/report/data/),
  * the trained artifacts in backend/app/ml/artifacts/*.json,
  * backend/app/data/satcat_snapshot.csv,
  * direct calls into the backend modules (breakup, screening, CW, propulsion,
    the seeded surrogate recipe) with ORBIT_SENTINEL_SKIP_DOTENV=1.
docs/report/verification.json (written by an independent verifier) is rendered
when present.

Output: docs/report/Orbital_Sentinel_Judges_Report.pdf, figures in docs/report/fig/.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── paths / environment ──────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKEND = ROOT / "backend"
FIG = HERE / "fig"
DATA = HERE / "data"
PDF_PATH = HERE / "Orbital_Sentinel_Judges_Report.pdf"
VERIFICATION = HERE / "verification.json"
VERIFICATION_MD = HERE / "VERIFICATION.md"
ARTIFACTS = BACKEND / "app" / "ml" / "artifacts"
SATCAT_CSV = BACKEND / "app" / "data" / "satcat_snapshot.csv"
API = os.environ.get("ORBIT_SENTINEL_API", "http://127.0.0.1:8000/api")

os.environ.setdefault("ORBIT_SENTINEL_SKIP_DOTENV", "1")
sys.path.insert(0, str(BACKEND))
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: E402

# ── palette ──────────────────────────────────────────────────────────────────
NAVY = "#0b1f3a"
NAVY2 = "#15305a"
TEAL = "#0fa3a3"
TEAL_L = "#e3f5f4"
BLUE = "#2a78d6"
INK = "#1d2433"
INK2 = "#52607a"
MUTED = "#8a94a6"
GRID = "#e5e8ee"
SURF = "#f6f8fb"
GOOD = "#1f9d55"      # nominal
CAUTION = "#e0a100"   # caution
BAD = "#d64545"       # warning / critical
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEV_COL = {"CRITICAL": BAD, "WARNING": CAUTION, "WATCH": GOOD}
ACT_COL = {"MANOEUVRE": BAD, "PREPARE": CAUTION, "MONITOR": BLUE, "NONE": MUTED}
DIVERGE = LinearSegmentedColormap.from_list("bgr", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a8f", "#c0302f"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8, "axes.edgecolor": "#9aa3b2", "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "legend.fontsize": 7, "figure.dpi": 100, "savefig.dpi": 200, "axes.titlecolor": INK,
    "axes.titlelocation": "left",
})

LOG: list[str] = []          # notes about missing data, shown in the console
CAPTIONS: dict[str, str] = {}


def note(msg: str) -> None:
    LOG.append(msg)
    print("  [note]", msg)


def save(fig, name: str, caption: str) -> Path:
    path = FIG / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    CAPTIONS[name] = caption
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Data collection
# ══════════════════════════════════════════════════════════════════════════════

def _http(method: str, path: str, body=None, timeout: float = 240.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, method=method, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def cache_path(key: str) -> Path:
    return DATA / f"{key}.json"


def load_cache(key: str):
    p = cache_path(key)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def store(key: str, obj) -> None:
    cache_path(key).write_text(json.dumps(obj), encoding="utf-8")


def backend_up() -> bool:
    try:
        _http("GET", "/simulation/time", timeout=10)
        return True
    except Exception:
        return False


LIVE_ENDPOINTS = {
    "analytics_model": "/analytics/model",
    "physics_validation": "/physics/validation",
    "physics_engines": "/physics/engines",
    "model_metrics": "/model-metrics",
    "alerts": "/alerts",
    "satellites": "/satellites",
    "agencies": "/agencies",
    "sim_time": "/simulation/time",
}


def collect_live(offline: bool) -> dict:
    out = {}
    for key, ep in LIVE_ENDPOINTS.items():
        val = None
        if not offline:
            try:
                val = _http("GET", ep)
                val = {"_fetched_utc": datetime.now(timezone.utc).isoformat(), "_endpoint": "GET /api" + ep, "data": val}
                store(key, val)
            except Exception as exc:
                note(f"GET /api{ep} failed ({exc}); using cache")
        if val is None:
            val = load_cache(key)
            if val is None:
                note(f"no data for {ep}")
        out[key] = val
    return out


def run_scenario() -> dict:
    """Load the cascade_demo scenario, break up its pair, capture everything, clean up."""
    res: dict = {"_run_utc": datetime.now(timezone.utc).isoformat()}
    try:
        res["scenario"] = _http("POST", "/simulate", {"name": "cascade_demo"})
        t0 = time.perf_counter()
        res["recompute"] = _http("POST", "/simulation/time", {"offset_hours": 0})
        res["pipeline_refresh_s"] = round(time.perf_counter() - t0, 3)
        res["alerts"] = _http("GET", "/alerts")
        t0 = time.perf_counter()
        res["debris"] = _http("POST", "/debris/simulate", {})
        res["debris_call_s"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        res["recompute_debris"] = _http("POST", "/simulation/time", {"offset_hours": 0})
        res["pipeline_refresh_debris_s"] = round(time.perf_counter() - t0, 3)
        res["alerts_debris"] = _http("GET", "/alerts")
        res["satellites_debris"] = _http("GET", "/satellites")
    finally:
        for m, p, b in (("DELETE", "/debris/active", None), ("DELETE", "/simulate", None),
                        ("POST", "/simulation/time", {"offset_hours": 0})):
            try:
                _http(m, p, b)
            except Exception as exc:  # pragma: no cover
                note(f"cleanup {m} {p} failed: {exc}")
    store("scenario_run", res)
    return res


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def D(live: dict, key: str):
    v = live.get(key)
    return (v or {}).get("data") if v else None


# ══════════════════════════════════════════════════════════════════════════════
# Computations through the backend modules
# ══════════════════════════════════════════════════════════════════════════════

def inclination_alt(sats: list[dict]):
    inc, alt = [], []
    for s in sats:
        p, v = s.get("position") or {}, s.get("velocity") or {}
        try:
            r = np.array([p["x"], p["y"], p["z"]], float)
            vv = np.array([v.get("vx", v.get("x")), v.get("vy", v.get("y")), v.get("vz", v.get("z"))], float)
        except Exception:
            continue
        h = np.cross(r, vv)
        inc.append(math.degrees(math.acos(h[2] / np.linalg.norm(h))))
        alt.append(float(s.get("altitude_km", np.linalg.norm(r) - 6371.0)))
    return np.array(inc), np.array(alt)


def read_satcat():
    rows = []
    with open(SATCAT_CSV, encoding="utf-8") as f:
        header = f.readline()
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return header.strip("# \n"), rows


def heldout_predictions():
    """Regenerate the held-out split with the recorded seed and score the saved model."""
    from app.ml import train_risk_surrogate as trs
    import xgboost as xgb

    card = load_json(ARTIFACTS / "risk_model_card.json") or {}
    seed = int((card.get("recipe") or {}).get("seed", trs.SEED))
    n = int(card.get("total_generated", trs.SAMPLE_COUNT))
    t0 = time.perf_counter()
    data = trs.generate_encounters(n, seed)
    y = np.log10(np.clip(data["pc"], trs.PC_FLOOR, 1.0)).astype(np.float32)
    x_full = trs.feature_matrix(data)
    rng = np.random.default_rng(seed + 1)
    trs.mask_features(x_full, rng)
    perm = rng.permutation(n)
    te = perm[: int(0.2 * n)]
    booster = xgb.Booster()
    booster.load_model(str(ARTIFACTS / trs.MODEL_FILE))
    pred = booster.predict(xgb.DMatrix(x_full[te], feature_names=trs.FEATURES, missing=np.nan))
    return {"true": y[te], "pred": pred, "seed": seed, "n": n, "seconds": time.perf_counter() - t0,
            "metrics": trs.classification_metrics(y[te], pred, 1e-4), "reg": trs.regression_metrics(y[te], pred)}


def screening_benchmark(sim_time: datetime):
    from app.core import screening
    from app.core.sgp4_propagator import SGP4Propagator
    from app.data.tle_fetcher import load_local_tles

    rows = []
    for n in (125, 250, 500, 1000):
        prop = SGP4Propagator()
        prop.load_tles(load_local_tles(n))
        states = prop.propagate_all(sim_time)
        st: dict = {}
        alerts = screening.screen(states, sim_time, propagator=prop, stats=st)
        st["n_requested"] = n
        st["alerts"] = len(alerts)
        rows.append(st)
    return rows


def reproduce_breakup(event: dict):
    from app.core import breakup as sbm

    tca = event["tca"]
    par = event["parents"]
    res = sbm.simulate_breakup(
        tca["position_a_eci"], tca["velocity_a_eci"], par[0]["mass_kg"],
        tca["position_b_eci"], tca["velocity_b_eci"], par[1]["mass_kg"],
        seed=int(event["seed"]),
        rocket_body_a=par[0].get("object_type") == "R/B",
        rocket_body_b=par[1].get("object_type") == "R/B",
        max_fragments=1000,
    )
    bc = sbm.DEFAULT_CD * res.am_m2_kg
    snaps = {}
    r, v, t_prev = res.r_km, res.v_kms, 0.0
    for t_min in (5, 30, 90):
        r, v = sbm.propagate(r, v, bc, (t_min - t_prev) * 60.0, max_step_s=20.0)
        t_prev = t_min
        snaps[t_min] = r.copy()
    return res, snaps, sbm


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")
    groups = {"phys": (BLUE, "#e8f1fc"), "ml": (CAT[6], "#eeecf9"), "dec": (TEAL, TEAL_L),
              "deb": (CAT[1], "#fdeee7"), "ag": (GOOD, "#e6f5ec")}
    boxes = [
        ("TLE\ncatalogue", "Space-Track / CDN /\nbundled snapshot", "phys"),
        ("SGP4", "WGS-72, TEME,\nsimulation clock", "phys"),
        ("24 h\nscreening", "KD-tree per 60 s\n+ Brent TCA", "phys"),
        ("Foster Pc", "TLE-age RTN\ncovariance", "phys"),
        ("ML surrogate\n+ SHAP", "XGBoost log10 Pc\n(advisory)", "ml"),
        ("Decision\nscore", "0-100,\nNASA CARA tiers", "dec"),
        ("Cascade-safe\nmanoeuvre", "re-propagated,\n12 engines", "dec"),
        ("Predicted\ncollision", "pair at\npredicted TCA", "deb"),
        ("NASA SBM\nbreakup", "N = 0.1 M^0.75\nLc^-1.71", "deb"),
        ("Fragment\npropagation", "RK4: two-body\n+ J2 + drag", "deb"),
        ("Debris\nscreening", "KD-tree 6 h,\nisotropic Pc", "deb"),
        ("Cascade\ngraph", "BFS depth,\nP(hit)=1-Π(1-Pc)", "dec"),
        ("Agencies", "CelesTrak\nSATCAT owners", "ag"),
    ]
    w, h = 17.0, 14.0
    rows_y = [50, 29, 8]
    xs = [0.8, 20.8, 40.8, 60.8, 80.8]
    pos = [(xs[i], rows_y[0]) for i in range(5)] + [(xs[4 - i], rows_y[1]) for i in range(4)] + \
          [(xs[1 + i], rows_y[2]) for i in range(4)]
    for (title, sub, g), (x, y) in zip(boxes, pos):
        edge, face = groups[g]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc=face, ec=edge, lw=1.4, ls="--" if g == "ml" else "-"))
        ax.text(x + w / 2, y + h * 0.69, title, ha="center", va="center", fontsize=7.0, weight="bold",
                color=INK, linespacing=1.0)
        ax.text(x + w / 2, y + h * 0.24, sub, ha="center", va="center", fontsize=5.5, color=INK2, linespacing=1.05)

    def arrow(p, q, col=INK2, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.1, color=col, ls=ls,
                                     shrinkA=1, shrinkB=1, connectionstyle=f"arc3,rad={rad}"))
    for i in range(4):
        x, y = pos[i]
        arrow((x + w + 0.3, y + h / 2), (pos[i + 1][0] - 0.3, y + h / 2))
    x, y = pos[4]
    arrow((x + w / 2, y - 0.4), (x + w / 2, pos[5][1] + h + 0.4))
    for i in range(5, 8):
        x, y = pos[i]
        arrow((x - 0.3, y + h / 2), (pos[i + 1][0] + w + 0.3, y + h / 2))
    fx, fy = pos[3]
    arrow((fx + w * 0.75, fy - 0.4), (pos[5][0] + 2.5, pos[5][1] + h + 0.4), col=BLUE, rad=0.2)
    ax.text(fx + w * 0.2, fy - 4.8, "physics Pc\n(authoritative)", fontsize=5.6, color=BLUE, ha="center")
    x, y = pos[8]
    arrow((x + w / 2, y - 0.4), (pos[9][0] + w / 2, pos[9][1] + h + 0.4))
    for i in range(9, 12):
        x, y = pos[i]
        arrow((x + w + 0.3, y + h / 2), (pos[i + 1][0] - 0.3, y + h / 2))
    cx, cy = pos[11]
    arrow((cx + w / 2, cy + h + 0.4), (pos[6][0] + w / 2, pos[6][1] - 0.4), col=TEAL, ls="--")
    ax.text(cx + w / 2 + 1.0, cy + h + 2.6, "downstream objects\nfeed n_cascade", fontsize=5.6, color=TEAL)
    legend = [("physics", "phys"), ("ML (advisory)", "ml"), ("decision / cascade", "dec"),
              ("collision → debris", "deb"), ("agency", "ag")]
    for i, (lab, g) in enumerate(legend):
        ax.add_patch(FancyBboxPatch((1.0 + i * 19.5, 1.2), 2.2, 2.2, boxstyle="round,pad=0.1",
                                    fc=groups[g][1], ec=groups[g][0], lw=1.2))
        ax.text(4.2 + i * 19.5, 2.3, lab, fontsize=6.3, va="center", color=INK2)
    return save(fig, "01_pipeline", "Pipeline as implemented in backend/main.py::refresh_alerts_once and the "
                "/api/debris/simulate route (see docs/HOW_IT_WORKS.md for each module). Dashed box: advisory "
                "model that never writes the physics Pc.")


def fig_catalogue(sats, satcat_rows, satcat_hdr):
    inc_l, alt_l = inclination_alt(sats)
    alt_c, inc_c, typ_c = [], [], []
    for r in satcat_rows:
        try:
            a = 0.5 * (float(r["apogee_km"]) + float(r["perigee_km"]))
            i = float(r["inclination_deg"])
        except (ValueError, TypeError):
            continue
        if r.get("decay_date"):
            continue
        alt_c.append(a); inc_c.append(i); typ_c.append(r.get("object_type") or "UNK")
    alt_c, inc_c, typ_c = np.array(alt_c), np.array(inc_c), np.array(typ_c)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    tcols = {"PAY": CAT[0], "R/B": CAT[1], "DEB": CAT[2], "UNK": MUTED}
    for t in ("DEB", "R/B", "PAY", "UNK"):
        m = typ_c == t
        if m.any():
            ax.scatter(alt_c[m], inc_c[m], s=1.2, alpha=0.35, color=tcols[t], lw=0,
                       label=f"SATCAT {t} ({m.sum():,})", rasterized=True)
    ax.scatter(alt_l, inc_l, s=10, facecolor="none", edgecolor=NAVY, lw=0.7,
               label=f"tracked live ({len(alt_l)})")
    ax.set_xscale("log")
    ax.set_xlim(150, 60000)
    for x, lab in ((2000, "LEO | MEO"), (35786, "GEO")):
        ax.axvline(x, color=INK2, lw=0.7, ls=":")
        ax.text(x * 1.04, 3, lab, fontsize=6.5, color=INK2)
    ax.set_xlabel("mean altitude (apogee+perigee)/2  [km, log]")
    ax.set_ylabel("inclination [deg]")
    ax.set_title("Orbital regimes: full on-orbit catalogue vs the objects tracked live")
    leg = ax.legend(loc="upper right", markerscale=4, ncol=1)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    return save(fig, "02_regimes", f"Grey/colour dots: backend/app/data/satcat_snapshot.csv ({satcat_hdr[:90]}...), "
                "non-decayed rows with orbit data. Rings: live states from GET /api/satellites, inclination "
                "computed as acos(h_z/|h|) with h = r × v.")


def fig_agencies(agencies, satcat_rows, sats_live):
    ag = agencies.get("agencies", [])
    owners = Counter(r.get("owner") or "?" for r in satcat_rows)
    top = owners.most_common(10)
    live_ids = {s["norad_id"] for s in sats_live}
    by_id = {}
    for r in satcat_rows:
        try:
            by_id[int(r["norad_id"])] = r
        except Exception:
            pass
    live_types = Counter((by_id.get(i, {}).get("object_type") or "not in SATCAT") for i in live_ids)
    cat_types = Counter(r.get("object_type") or "UNK" for r in satcat_rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    a = axes[0]
    names = [x["name"] for x in ag][::-1]
    counts = [x["count"] for x in ag][::-1]
    a.barh(names, counts, color=[MUTED if n == "Unknown" else BLUE for n in names], height=0.65)
    for y, c in enumerate(counts):
        a.text(c + 4, y, str(c), va="center", fontsize=6.5, color=INK2)
    a.set_title(f"Live, by agency (n={agencies.get('total')})")
    a.set_xlim(0, max(counts) * 1.22); a.grid(axis="y", visible=False)
    a.tick_params(axis="y", labelsize=6.5)
    b = axes[1]
    b.barh([o for o, _ in top][::-1], [c for _, c in top][::-1], color=CAT[2], height=0.65)
    b.set_title("SATCAT owners, top 10")
    b.set_xscale("log"); b.grid(axis="y", visible=False); b.tick_params(axis="y", labelsize=6.5)
    b.set_xlabel("objects (log)")
    c = axes[2]
    tl = ["PAY", "R/B", "DEB", "UNK"]
    xs = np.arange(len(tl))
    tot_c = sum(cat_types.values()); tot_l = sum(live_types.values())
    c.bar(xs - 0.2, [100 * cat_types.get(t, 0) / tot_c for t in tl], 0.38, color=MUTED, label=f"SATCAT ({tot_c:,})")
    c.bar(xs + 0.2, [100 * live_types.get(t, 0) / tot_l for t in tl], 0.38, color=NAVY, label=f"live ({tot_l})")
    c.set_xticks(xs, tl); c.set_ylabel("% of objects"); c.set_title("Object types")
    c.legend(loc="upper right"); c.set_ylim(0, 80); c.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.2)
    return save(fig, "03_agencies", "Left: GET /api/agencies (SATCAT owner + operator-pattern attribution, "
                f"attribution sources {agencies.get('attribution_sources')}). Middle/right: satcat_snapshot.csv "
                "owner and object_type columns; live types by NORAD-id lookup.")


def fig_screening(alerts, bench):
    al = [a for a in alerts if a.get("source", "screening") == "screening"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1.15, 1.05]})
    a = axes[0]
    bins = np.arange(0, 25, 2)
    bottom = np.zeros(len(bins) - 1)
    for sev in ("WATCH", "WARNING", "CRITICAL"):
        h, _ = np.histogram([x["tca_hours"] for x in al if x["severity"] == sev], bins)
        a.bar(bins[:-1] + 1, h, 1.8, bottom=bottom, color=SEV_COL[sev], label=sev)
        bottom += h
    a.set_xlabel("time to TCA [h]"); a.set_ylabel("alerts"); a.set_title("TCA distribution")
    a.set_ylim(0, bottom.max() * 1.45)
    a.legend(loc="upper center", fontsize=5.6, ncol=3, columnspacing=0.6, handlelength=1); a.grid(axis="x", visible=False)
    b = axes[1]
    floor = 1e-20
    for sev in ("WATCH", "WARNING", "CRITICAL"):
        xs = [x["miss_distance_km"] for x in al if x["severity"] == sev]
        ys = [max(x["probability_of_collision"], floor) for x in al if x["severity"] == sev]
        b.scatter(xs, ys, s=11, color=SEV_COL[sev], edgecolor="white", lw=0.4, label=sev, zorder=3)
    b.set_xscale("log"); b.set_yscale("log")
    b.axhline(1e-4, color=BAD, lw=0.8, ls="--"); b.axhline(1e-6, color=CAUTION, lw=0.8, ls="--")
    b.text(b.get_xlim()[0] if False else 0.12, 2e-4, "Pc 1e-4", fontsize=6, color=BAD)
    b.text(0.12, 2e-6, "Pc 1e-6", fontsize=6, color=CAUTION)
    b.set_xlim(0.1, 30)
    b.set_xlabel("miss distance [km]"); b.set_ylabel("Foster Pc (floored 1e-20)")
    b.set_title("Miss vs Pc")
    c = axes[2]
    if bench:
        ns = [str(r["objects"]) for r in bench]
        parts = [("t_propagate_s", "SGP4 grid", CAT[0]), ("t_kdtree_s", "KD-tree", CAT[2]),
                 ("t_refine_s", "Brent TCA", CAT[1]), ("t_pc_s", "Foster Pc", CAT[6])]
        bot = np.zeros(len(bench))
        for k, lab, col in parts:
            v = np.array([r.get(k, 0.0) for r in bench])
            c.bar(ns, v, 0.6, bottom=bot, color=col, label=lab, edgecolor="white", lw=0.6)
            bot += v
        for i, r in enumerate(bench):
            c.text(i, bot[i] * 1.02, f"{r['t_total_s']:.2f}s", ha="center", fontsize=6.3, color=INK)
        c.set_xlabel("objects screened"); c.set_ylabel("wall time [s]")
        c.set_title("Runtime vs catalogue size")
        c.legend(fontsize=5.8, loc="upper left"); c.grid(axis="x", visible=False)
        c.set_ylim(0, bot.max() * 1.25)
    fig.tight_layout(w_pad=1.0)
    return save(fig, "04_screening", f"Left/middle: GET /api/alerts, {len(al)} screened pairs from the live 500-object "
                "catalogue. Right: in-process app.core.screening.screen() on the first N TLEs of the bundled snapshot, "
                "per-stage timings from its stats dict (this laptop).")


def fig_bplane(alert):
    from app.core.screening import foster_pc

    ce = alert["covariance_ellipse"]
    k = float(ce.get("sigma_level", 3))
    sa, sb = ce["a"] / k / 1000.0, ce["b"] / k / 1000.0   # 1-sigma km
    ang = float(ce["angle"])
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    C2 = R @ np.diag([sa ** 2, sb ** 2]) @ R.T
    b = np.array([alert["b_t_km"], alert["b_n_km"]])
    hbr = alert["hbr_km"]
    lim = ce["a"] * 1.08
    pcs = alert.get("pc_checks") or {}
    fig, (a, zi) = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [2.6, 1]})
    for kk, alpha in ((3, 0.12), (2, 0.18), (1, 0.28)):
        a.add_patch(Ellipse((0, 0), 2 * kk * sa * 1000, 2 * kk * sb * 1000, angle=math.degrees(ang),
                            fc=BLUE, alpha=alpha, ec=BLUE, lw=0.6))
    a.annotate("", xy=(b[0] * 1000, b[1] * 1000), xytext=(0, 0),
               arrowprops=dict(arrowstyle="-|>", color=BAD, lw=1.2))
    a.add_patch(Circle((b[0] * 1000, b[1] * 1000), max(hbr * 1000, 1), fc=BAD, ec=BAD))
    a.text(-lim * 0.95, -lim * 0.36, f"miss {np.linalg.norm(b)*1000:.0f} m, HBR {hbr*1000:.1f} m",
           fontsize=6.3, color=BAD)
    a.set_xlim(-lim, lim); a.set_ylim(-lim * 0.45, lim * 0.45); a.set_aspect("equal")
    zl = max(2.5 * np.linalg.norm(b) * 1000, 4 * hbr * 1000, 1.0)
    zi.add_patch(Ellipse((0, 0), 2 * sa * 1000, 2 * sb * 1000, angle=math.degrees(ang), fc=BLUE, alpha=0.15, ec=BLUE, lw=0.5))
    zi.annotate("", xy=(b[0] * 1000, b[1] * 1000), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BAD, lw=1))
    zi.add_patch(Circle((b[0] * 1000, b[1] * 1000), hbr * 1000, fc=BAD, alpha=0.5, ec=BAD))
    zi.scatter([0], [0], s=6, color=INK)
    zi.set_xlim(-zl, zl); zi.set_ylim(-zl, zl); zi.set_aspect("equal")
    zi.tick_params(labelsize=6); zi.set_title(f"Zoom on the hard body (±{zl:.0f} m)", fontsize=7.5)
    zi.set_xlabel("T [m]")
    a.set_xlabel("B-plane T [m]"); a.set_ylabel("B-plane N [m]")
    a.set_title("Encounter B-plane, true scale (1/2/3σ)")
    a.text(-lim * 0.95, -lim * 0.42, f"1σ = {sa*1000:.0f} × {sb*1000:.0f} m", fontsize=6.3, color=INK2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "05_bplane", f"Alert {alert['id']} ({alert['sat1']['name']} × {alert['sat2']['name']}) from GET /api/alerts "
         "during the cascade_demo run: b_t_km/b_n_km, covariance_ellipse (a, b, angle at sigma_level) and hbr_km.")
    fig, (c, d) = plt.subplots(1, 2, figsize=(7.2, 2.4))
    # methods
    meths = [("Foster", pcs.get("foster"), BLUE), ("Chan", pcs.get("chan"), CAT[2]),
             ("Monte Carlo", pcs.get("monte_carlo"), CAT[6]), ("Alfano max", pcs.get("alfano_max"), CAT[1])]
    labs, vals, cols = [], [], []
    for lab, v, col in meths:
        labs.append(lab); vals.append(v if v else np.nan); cols.append(col)
    ys = np.arange(len(labs))[::-1]
    c.barh(ys, [math.log10(v) + 12 if v == v else 0 for v in vals], left=-12, color=cols, height=0.6)
    for y, v in zip(ys, vals):
        c.text((math.log10(v) if v == v else -8) + 0.15, y, f"{v:.2e}" if v == v else "below MC resolution",
               va="center", fontsize=6.3, color=INK)
    if pcs.get("monte_carlo") and pcs.get("mc_stderr"):
        mc, se = pcs["monte_carlo"], pcs["mc_stderr"]
        c.errorbar([math.log10(mc)], [ys[2]], xerr=[[math.log10(mc) - math.log10(max(mc - 2 * se, 1e-12))],
                                                    [math.log10(mc + 2 * se) - math.log10(mc)]],
                   fmt="none", ecolor=INK, capsize=2, lw=0.8)
    c.set_yticks(ys, labs); c.set_xlim(-8, 0)
    c.set_xlabel("log10 Pc"); c.set_title("Four independent Pc methods")
    c.grid(axis="y", visible=False)
    # dilution curve
    ks = np.logspace(-3, 2, 101)
    pk = [foster_pc(b, (kk ** 2) * C2, hbr) for kk in ks]
    d.plot(ks, pk, color=BLUE, lw=1.8)
    d.axvline(1, color=INK2, ls=":", lw=0.8)
    imax = int(np.argmax(pk))
    d.scatter([ks[imax]], [pk[imax]], color=CAT[1], s=22, zorder=4)
    d.text(ks[imax] * 1.15, pk[imax], f"max {pk[imax]:.1e}\n(Alfano)", fontsize=6.3, color=CAT[1], va="top")
    d.set_xscale("log"); d.set_yscale("log")
    d.set_ylim(max(min(pk), 1e-12), max(pk) * 5)
    d.set_xlabel("covariance scale k (C → k²C)"); d.set_ylabel("Foster Pc")
    d.set_title("Covariance dilution")
    fig.tight_layout(w_pad=1.5)
    return save(fig, "05b_methods", f"Left: alert.pc_checks of {alert['id']} (app/core/pc_methods.py; MC with "
                f"{pcs.get('mc_samples')} samples). Right: app.core.screening.foster_pc re-evaluated on the same B-plane "
                "with the covariance scaled by k² (Alfano's maximum is the peak).")


def fig_pcchecks(alerts):
    rows = [a.get("pc_checks") for a in alerts if a.get("pc_checks")]
    rows = [r for r in rows if r.get("foster") and r.get("chan") and r["foster"] > 1e-30 and r["chan"] > 1e-30]
    if not rows:
        return None
    fo = np.array([r["foster"] for r in rows]); ch = np.array([r["chan"] for r in rows])
    al = np.array([r.get("alfano_max") or np.nan for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    a = axes[0]
    a.scatter(fo, ch, s=12, color=CAT[2], edgecolor="white", lw=0.3, zorder=3)
    lo, hi = min(fo.min(), ch.min()) / 3, max(fo.max(), ch.max()) * 3
    a.plot([lo, hi], [lo, hi], color=BAD, lw=0.8, ls="--")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel("Foster Pc"); a.set_ylabel("Chan series Pc")
    sp = np.abs(np.log10(fo) - np.log10(ch))
    a.set_title(f"Foster vs Chan on {len(rows)} live alerts")
    a.text(0.04, 0.9, f"median |Δlog10| = {np.median(sp):.1e}\nmax = {sp.max():.2f} decades", transform=a.transAxes,
           fontsize=6.3, color=INK2, va="top")
    b = axes[1]
    ratio = np.log10(al / fo)
    ratio = ratio[np.isfinite(ratio)]
    b.hist(ratio, bins=30, color=CAT[1], edgecolor="white", lw=0.4)
    b.set_xlabel("log10 (Alfano max Pc / Foster Pc)"); b.set_ylabel("alerts")
    b.set_title("Head-room if the covariance size is wrong")
    b.axvline(0, color=INK, lw=0.8)
    fig.tight_layout(w_pad=1.5)
    return save(fig, "05c_pcchecks", "alert.pc_checks of every screened alert in GET /api/alerts (cascade_demo run); "
                "Alfano ≥ Foster by construction (upper bound).")


def fig_tle_sigma(alerts):
    from app.core.screening import SIGMA0_RTN_KM, SIGMA_GROWTH_RTN_KM_PER_DAY, tle_age_sigmas_km

    ages = np.linspace(0, 60, 200)
    s = np.array([tle_age_sigmas_km(x) for x in ages])
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, (lab, col) in enumerate((("radial", CAT[2]), ("along-track", BLUE), ("cross-track", CAT[1]))):
        ax.plot(ages, s[:, i], color=col, lw=1.8, label=f"{lab}: {SIGMA0_RTN_KM[i]:.2f} + {SIGMA_GROWTH_RTN_KM_PER_DAY[i]:.2f}·age")
    obs = [t for a in alerts for t in (a.get("tle_age_days") or [])]
    if obs:
        ax.plot(obs, np.full(len(obs), -1.5), "|", color=INK2, ms=6, alpha=0.5)
        ax.text(max(obs) + 1.5, -1.5, f"TLE ages in live alerts ({min(obs):.1f}–{max(obs):.1f} d)", fontsize=5.8, va="center", color=INK2)
    ax.set_xlabel("TLE age at TCA [days]"); ax.set_ylabel("1σ [km]")
    ax.set_ylim(-3, s.max() * 1.05)
    ax.set_title("TLE-age covariance model")
    ax.legend(fontsize=5.8, loc="upper left")
    return save(fig, "06_tle_sigma", "app.core.screening.tle_age_sigmas_km (σ0 Flohrer et al. 2008; growth Vallado & "
                "Cefola 2012). Rug: tle_age_days of the live alerts.")


def fig_corr(an):
    corr = an["correlation"]
    labels = [an["feature_labels"].get(v, v) for v in corr["variables"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    for a, key in zip(axes, ("pearson", "spearman")):
        m = np.array(corr[key])
        im = a.imshow(m, cmap=DIVERGE, vmin=-1, vmax=1)
        for i in range(len(m)):
            for j in range(len(m)):
                a.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5.4,
                       color="white" if abs(m[i, j]) > 0.6 else INK)
        a.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=6)
        a.set_yticks(range(len(labels)), labels if key == "pearson" else [""] * len(labels), fontsize=6)
        a.grid(False); a.set_title(key.capitalize())
    fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    return save(fig, "07_corr", f"GET /api/analytics/model → correlation ({corr.get('missing_handling')}; "
                f"{corr.get('sample_size')} training rows). Last row/column = target log10 Pc.")


def fig_pca(an):
    pca = an["pca"]
    evr = np.array(pca["explained_variance_ratio"]); cum = np.array(pca["cumulative"])
    L = np.array(pca["loadings"])
    feats = [an["feature_labels"].get(f, f) for f in pca["features"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.3]})
    a = axes[0]
    xs = np.arange(1, len(evr) + 1)
    a.bar(xs, evr * 100, color=BLUE, width=0.6, label="per component")
    a.plot(xs, cum * 100, color=CAT[1], marker="o", ms=4, lw=1.6, label="cumulative")
    a.axhline(95, color=INK2, ls=":", lw=0.8)
    a.text(0.6, 96.5, "95 %", fontsize=6.3, color=INK2)
    n95 = pca.get("n_components_95")
    a.axvline(n95, color=CAT[1], ls="--", lw=0.8)
    a.text(n95 - 0.1, 50, f"{n95} PCs for 95 %", rotation=90, fontsize=6.3, color=CAT[1], ha="right")
    a.set_xticks(xs, [f"PC{i}" for i in xs]); a.set_ylabel("explained variance [%]")
    a.set_ylim(0, 108); a.set_title("PCA scree"); a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9)); a.grid(axis="x", visible=False)
    b = axes[1]
    im = b.imshow(L, cmap=DIVERGE, vmin=-1, vmax=1, aspect="auto")
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if abs(L[i, j]) >= 0.3:
                b.text(j, i, f"{L[i, j]:+.2f}", ha="center", va="center", fontsize=5.6,
                       color="white" if abs(L[i, j]) > 0.6 else INK)
    b.set_xticks(range(len(feats)), feats, rotation=40, ha="right", fontsize=6)
    b.set_yticks(range(L.shape[0]), [f"PC{i+1}" for i in range(L.shape[0])], fontsize=6.5)
    b.grid(False); b.set_title("Loadings (|loading| ≥ 0.3 labelled)")
    fig.colorbar(im, ax=b, shrink=0.8, pad=0.02)
    fig.tight_layout()
    return save(fig, "08_pca", f"GET /api/analytics/model → pca ({pca.get('loadings_kind')}; {pca.get('imputation')}).")


def fig_pca_vs_raw(an):
    pv = an["pca_vs_raw"]
    rows = [("Raw 7 features", pv["raw_xgb"], GOOD),
            (f"PCA {pv['pca_xgb']['n_components']} PCs (95 %)", pv["pca_xgb"], CAT[1])]
    if pv.get("pca_xgb_all_components"):
        rows.append((f"PCA all {pv['pca_xgb_all_components']['n_components']} PCs", pv["pca_xgb_all_components"], CAT[3]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.0))
    for a, key, lab, better in ((axes[0], "mae_log10", "held-out MAE [dex] (lower is better)", "low"),
                                (axes[1], "f1_1e4", "held-out F1 @ Pc 1e-4 (higher is better)", "high")):
        ys = np.arange(len(rows))[::-1]
        vals = [r[1][key] for r in rows]
        a.barh(ys, vals, color=[r[2] for r in rows], height=0.6)
        for y, v in zip(ys, vals):
            a.text(v + max(vals) * 0.01, y, f"{v:.3f}", va="center", fontsize=6.6)
        a.set_yticks(ys, [r[0] for r in rows] if key == "mae_log10" else [""] * len(rows), fontsize=6.6)
        a.set_title(lab, fontsize=8); a.grid(axis="y", visible=False)
        a.set_xlim(0 if key == "mae_log10" else 0.85, max(vals) * 1.15 if key == "mae_log10" else 1.0)
    fig.tight_layout()
    return save(fig, "09_pca_vs_raw", "GET /api/analytics/model → pca_vs_raw: same rows, split, seed and XGBoost "
                "hyper-parameters; PCA fitted on training rows only.")


def fig_shap(an):
    sg = an["shap_global"]
    gain = (an.get("split_importance") or {}).get("gain", {})
    labs = [x["label"] for x in sg][::-1]
    sh = [x["mean_abs_contribution_log10"] for x in sg][::-1]
    gn = [gain.get(x["feature"], 0.0) for x in sg][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    ys = np.arange(len(labs))
    axes[0].barh(ys, sh, color=[CAT[6] if i == len(labs) - 1 else "#8f86d6" for i in range(len(labs))], height=0.62)
    for y, v in zip(ys, sh):
        axes[0].text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=6.5)
    axes[0].set_yticks(ys, labs, fontsize=6.8); axes[0].set_xlabel("mean |SHAP| [decades of Pc]")
    axes[0].set_title("TreeSHAP global importance"); axes[0].grid(axis="y", visible=False)
    axes[1].barh(ys, gn, color=MUTED, height=0.62)
    for y, v in zip(ys, gn):
        axes[1].text(v + 0.005, y, f"{v:.2f}", va="center", fontsize=6.5)
    axes[1].set_yticks(ys, [""] * len(labs)); axes[1].set_xlabel("normalised split gain")
    axes[1].set_title("XGBoost gain (for comparison)"); axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    s = an.get("shap", {})
    return save(fig, "10_shap", f"GET /api/analytics/model → shap_global ({s.get('method')}, {s.get('sample_size')} "
                f"held-out rows, additivity error {s.get('additivity_max_abs_error')}) and split_importance.gain.")


def fig_waterfall(alert):
    ml = alert["ml"]
    contribs = ml["contributions"]
    base = ml["base_log10"]
    other = ml.get("other_contributions_log10", 0.0) or 0.0
    steps = [(f"{c['label']} = {c['value']:.4g}" if c.get("value") is not None else c["label"], c["contribution_log10"])
             for c in contribs]
    if abs(other) > 1e-6:
        steps.append(("other features", other))
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    cur = base
    ax.barh(0, base + 12, left=-12, color=MUTED, height=0.55)
    ax.text(base + 0.08, 0, f"base {base:.2f}", va="center", fontsize=6.5)
    labels = ["model expectation"]
    for i, (lab, v) in enumerate(steps, start=1):
        ax.barh(i, v, left=cur, color=BAD if v > 0 else GOOD, height=0.55)
        ax.text(max(cur, cur + v) + 0.08, i, f"{v:+.2f}", va="center", fontsize=6.5)
        cur += v
        labels.append(lab)
    final = len(steps) + 1
    ax.barh(final, cur + 12, left=-12, color=CAT[6], height=0.55)
    ax.text(cur + 0.08, final, f"surrogate {cur:.2f}  (physics {math.log10(alert['probability_of_collision']):.2f})",
            va="center", fontsize=6.5)
    labels.append("prediction log10 Pc")
    ax.set_yticks(range(len(labels)), labels, fontsize=6.6); ax.invert_yaxis()
    lo = min(base, cur, -8) - 0.5
    ax.set_xlim(lo, max(base, cur) + 2.2)
    ax.set_xlabel("log10 Pc  (red raises risk, green lowers it)")
    ax.set_title(f"Per-alert SHAP waterfall — {alert['id']}")
    ax.grid(axis="y", visible=False)
    return save(fig, "11_waterfall", f"alert.ml of {alert['id']} from GET /api/alerts ({ml.get('contribution_method')}); "
                "base_log10 + Σ contributions = log10 surrogate Pc.")


def fig_heldout(ho, card):
    t, p = ho["true"], ho["pred"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})
    a = axes[0]
    hb = a.hexbin(t, p, gridsize=55, bins="log", cmap=LinearSegmentedColormap.from_list(
        "b", ["#dbe9fb", "#3987e5", "#0d366b"]), mincnt=1, lw=0)
    a.plot([-12, 0], [-12, 0], color=BAD, lw=0.8, ls="--")
    a.axvline(-4, color=INK2, lw=0.6, ls=":"); a.axhline(-4, color=INK2, lw=0.6, ls=":")
    a.set_xlabel("true log10 Pc (Foster label)"); a.set_ylabel("predicted log10 Pc")
    a.set_title(f"Held-out: n={len(t):,}, MAE {ho['reg']['mae_log10_pc']:.3f} dex")
    fig.colorbar(hb, ax=a, shrink=0.85, pad=0.02, label="count (log)")
    b = axes[1]
    m = ho["metrics"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    b.imshow(cm, cmap=LinearSegmentedColormap.from_list("t", ["#f0f7f7", TEAL]), norm=matplotlib.colors.LogNorm())
    for i in range(2):
        for j in range(2):
            b.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=9, weight="bold",
                   color="white" if cm[i, j] > 1000 else INK)
    b.set_xticks([0, 1], ["pred < 1e-4", "pred ≥ 1e-4"]); b.set_yticks([0, 1], ["true < 1e-4", "true ≥ 1e-4"])
    b.grid(False)
    b.set_title(f"Confusion @1e-4: P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f}", fontsize=8)
    fig.tight_layout()
    chk = (card.get("heldout") or {}).get("classification_at_1e-4", {})
    same = all(chk.get(k) == m[k] for k in ("tp", "fp", "fn", "tn")) if chk else False
    return save(fig, "12_heldout", f"Held-out split regenerated with app.ml.train_risk_surrogate.generate_encounters "
                f"(seed {ho['seed']}, n={ho['n']:,}, 70/10/20) and scored with the saved risk_model_xgb.json "
                f"({ho['seconds']:.0f} s). Confusion counts {'match' if same else 'differ from'} the model card.")


def fig_decision(alerts, scen_alert):
    from app.core import decision as dec

    W = dict(dec.WEIGHTS)
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.85, 1.25])
    a = fig.add_subplot(gs[0])
    cols = {"physics": BLUE, "ml": CAT[6], "cascade": TEAL, "manoeuvre": CAT[1]}
    vals = [W[k] for k in cols]
    a.barh(range(len(vals))[::-1], [v * 100 for v in vals], color=list(cols.values()), height=0.6)
    for y, (k, v) in zip(range(len(vals))[::-1], zip(cols, vals)):
        a.text(v * 100 + 1.5, y, f"{v:.2f}", va="center", fontsize=6.6)
    a.set_yticks(range(len(vals))[::-1], list(cols), fontsize=7)
    a.set_xlim(0, 75); a.set_xlabel("max points (weight × 100)")
    a.set_title("Component weights (Σ = 1)"); a.grid(axis="y", visible=False)
    b = fig.add_subplot(gs[1])
    tiers = [("MANOEUVRE", dec.PC_MANOEUVRE, 1e0), ("PREPARE", dec.PC_PREPARE, dec.PC_MANOEUVRE),
             ("MONITOR", dec.PC_MONITOR, dec.PC_PREPARE), ("NONE", 1e-12, dec.PC_MONITOR)]
    for lab, lo, hi in tiers:
        b.fill_between([0, 1], math.log10(lo), math.log10(hi), color=ACT_COL[lab], alpha=0.9, lw=0)
        b.text(0.5, (math.log10(lo) + math.log10(hi)) / 2, f"{lab}\nPc ≥ {lo:.0e}" if lab != "NONE" else "NONE",
               ha="center", va="center", fontsize=6.3, color="white", weight="bold")
    if scen_alert:
        y = math.log10(scen_alert["probability_of_collision"])
        b.plot([0, 1], [y, y], color=INK, lw=1.2, ls="--")
        b.text(0.03, y + 0.15, "demo alert", fontsize=5.8, color=INK, va="bottom")
    b.set_xlim(0, 1); b.set_ylim(-12, 0); b.set_xticks([])
    b.set_ylabel("log10 Foster Pc"); b.set_title("Action tiers")
    b.grid(False)
    c = fig.add_subplot(gs[2])
    sc = [x["decision"]["score"] for x in alerts if x.get("decision")]
    act = [x["decision"]["action"] for x in alerts if x.get("decision")]
    bins = np.arange(0, 101, 5)
    bottom = np.zeros(len(bins) - 1)
    for lab in ("NONE", "MONITOR", "PREPARE", "MANOEUVRE"):
        h, _ = np.histogram([s for s, ac in zip(sc, act) if ac == lab], bins)
        if h.sum():
            c.bar(bins[:-1] + 2.5, h, 4.4, bottom=bottom, color=ACT_COL[lab], label=f"{lab} ({h.sum()})", log=True)
            bottom += h
    c.set_yscale("log"); c.set_ylim(0.7, max(bottom.max() * 3, 10))
    c.set_xlabel("decision score"); c.set_ylabel("alerts (log)")
    c.set_title(f"Scores of {len(sc)} alerts (demo run)"); c.legend(fontsize=5.8, loc="upper right")
    c.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.0)
    return save(fig, "13_decision", "app.core.decision.WEIGHTS / PC_* constants; score histogram from alert.decision "
                "of every alert in GET /api/alerts while the cascade_demo scenario was loaded.")


def fig_decision_breakdown(alert):
    d = alert["decision"]
    comps = d["components"]
    cols = {"physics": BLUE, "ml": CAT[6], "cascade": TEAL, "manoeuvre": CAT[1]}
    desc = {"physics": "Foster Pc", "ml": "ML surrogate Pc", "cascade": "downstream objects D", "manoeuvre": "Δv of plan [m/s]"}
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ys = np.arange(len(comps))[::-1]
    for y, c in zip(ys, comps):
        mx = c["weight"] * 100
        ax.barh(y, mx, color="none", ec=cols[c["source"]], lw=0.8, ls=":", height=0.6)
        ax.barh(y, c["points"], color=cols[c["source"]], height=0.6)
        raw = c["raw"]
        rtxt = "n/a" if raw is None else (f"{raw:.2e}" if isinstance(raw, float) and abs(raw) < 0.01 else f"{raw:g}")
        ax.text(mx + 1.2, y, f"raw {rtxt} → n = {c['normalized']:.2f} × w {c['weight']:.2f} = {c['points']:.1f} pts",
                va="center", fontsize=6.3, color=INK)
    ax.set_yticks(ys, [f"{c['source']}\n({desc.get(c['source'], c['name'])})" for c in comps], fontsize=6.3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("points (dotted = maximum for the component)")
    ax.set_title(f"Breakdown for {alert['id']}: score {d['score']:.1f}/100 → {d['action']} "
                 f"(confidence {d['model_agreement']['confidence']})")
    ax.grid(axis="y", visible=False)
    return save(fig, "14_decision_breakdown", f"alert.decision.components of {alert['id']} (GET /api/alerts, cascade_demo run).")


def fig_options(rm, alert):
    opts = rm.get("options") or []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.1, 1]})
    a = axes[0]
    floor = 1e-14
    seen = Counter()
    for i, o in enumerate(opts):
        pc = max(o.get("new_pc_collision") or 0.0, floor)
        col = GOOD if o.get("cascade_safe") else BAD
        star = i == rm.get("chosen_index")
        a.scatter(o["delta_v_ms"], pc, s=150 if star else 45, marker="*" if star else "o", color=col,
                  edgecolor=INK, lw=0.5, zorder=4)
        k = seen[round(o["delta_v_ms"], 3)]
        seen[round(o["delta_v_ms"], 3)] += 1
        a.annotate(f"#{i + 1}", (o["delta_v_ms"], pc), xytext=(7 + 17 * k, -3), textcoords="offset points",
                   fontsize=6.3, weight="bold", color=INK)
    a.axhline(alert["probability_of_collision"], color=BAD, ls="--", lw=0.8)
    a.text(0.02, alert["probability_of_collision"] * 1.8, f"before burn {alert['probability_of_collision']:.1e}",
           fontsize=6, color=BAD, transform=a.get_yaxis_transform())
    a.axhline(rm.get("target_pc", 1e-6), color=CAUTION, ls=":", lw=0.8)
    a.text(0.02, rm.get("target_pc", 1e-6) * 1.8, "target 1e-6", fontsize=6, color=CAUTION, transform=a.get_yaxis_transform())
    a.set_yscale("log"); a.set_xlabel("Δv [m/s]"); a.set_ylabel("re-propagated Pc after burn")
    if opts:
        a.set_xlim(0, max(o["delta_v_ms"] for o in opts) * 1.5)
    a.set_title(f"Options ({rm.get('candidates_evaluated', '?')} candidates evaluated)")
    a.scatter([], [], color=GOOD, label="cascade-safe"); a.scatter([], [], color=BAD, label="hampers others")
    a.scatter([], [], marker="*", color=MUTED, s=80, label="chosen")
    a.legend(fontsize=6, loc="lower right")
    b = axes[1]
    from app.core import analytic_checks as ac
    chk = rm.get("analytic_check") or {}
    n = chk.get("mean_motion_rad_s")
    t_tca = chk.get("t_since_burn_s", 1800.0)
    dv = rm.get("delta_v_rsw_ms", [0, -0.5, 0])
    if n:
        a_km = (ac.MU_KM3_S2 / n ** 2) ** (1 / 3)
        r0, v0 = ac._circular_state(a_km - ac.RE_KM, 60.0)
        T = max(3 * 3600.0, 1.5 * t_tca)
        ts = np.linspace(0, T, 300)
        cw = ac.cw_position(ts, n, np.asarray(dv, float) / 1000.0)[:, 1]
        tn = np.linspace(T / 12, T, 12)
        num = [ac.numeric_along_track_shift_km(r0, v0, dv, t, j2=False, step_s=10.0) for t in tn]
        b.plot(ts / 3600, cw, color=BLUE, lw=1.8, label="Clohessy–Wiltshire (analytic)")
        b.scatter(tn / 3600, num, color=CAT[1], s=16, zorder=4, label="RK4 two-body (numeric)")
        if chk:
            b.scatter([t_tca / 3600], [chk["numeric_km"]], marker="D", s=30, color=BAD, zorder=5,
                      label=f"planner check @TCA: rel err {100*chk['rel_error']:.2f} %")
        b.set_xlabel("time since burn [h]"); b.set_ylabel("along-track shift [km]")
        b.set_title(f"Along-track drift, Δv_RSW = {[round(float(x), 3) for x in dv]} m/s")
        b.legend(fontsize=5.8, loc="upper left" if cw[-1] > 0 else "lower left")
    fig.tight_layout(w_pad=1.0)
    return save(fig, "15_options", f"recommended_maneuver.options of {alert['id']} (GET /api/alerts, cascade_demo run; "
                "numbers # match the table below). Right: app.core.analytic_checks cw_position vs "
                "numeric_along_track_shift_km on a circular orbit with the option's mean motion; red diamond = the "
                "planner's own analytic_check.")


def fig_engines(rm):
    opts = rm.get("options") or []
    ci = rm.get("chosen_index", 0) or 0
    opt = opts[ci] if opts else rm
    eng = opt.get("engines") or []
    if not eng:
        return None
    eng = sorted(eng, key=lambda e: e["isp_s"])
    names = [e["engine"].replace("Aerojet Rocketdyne ", "").replace("Aerojet ", "") for e in eng]

    def col(e):
        if not e.get("practical_for_satellites", True):
            return MUTED
        return GOOD if e["finite_burn_ok"] else CAUTION
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ys = np.arange(len(eng))
    for a, key, lab in ((axes[0], "prop_mass_kg", "propellant mass [kg, log]"), (axes[1], "burn_time_s", "burn time [s, log]")):
        v = [max(e[key], 1e-4) for e in eng]
        a.barh(ys, v, color=[col(e) for e in eng], height=0.62)
        for y, x in zip(ys, v):
            a.text(x * 1.15, y, f"{x:.3g}", va="center", fontsize=5.8)
        a.set_xscale("log"); a.set_xlabel(lab); a.grid(axis="y", visible=False)
        a.set_xlim(min(v) / 2, max(v) * 12)
        a.set_yticks(ys, [f"{n} ({e['isp_s']:.0f} s)" for n, e in zip(names, eng)] if key == "prop_mass_kg" else [""] * len(eng), fontsize=6.2)
    rec = rm.get("recommended_engine")
    axes[0].set_title(f"{opt.get('candidate', '')}: Tsiolkovsky propellant")
    axes[1].set_title(f"Constant-thrust burn time (recommended: {rec})", fontsize=8)
    from matplotlib.patches import Patch
    hs = [Patch(color=c, label=lab) for lab, c in (("practical, impulsive OK", GOOD),
                                                   ("needs finite / low-thrust arc", CAUTION),
                                                   ("impractical (cryogenic)", MUTED))]
    fig.legend(handles=hs, loc="lower center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=0.5, rect=(0, 0.06, 1, 1))
    return save(fig, "16_engines", f"options[{ci}].engines of the demo alert (app.core.propulsion.evaluate_engines; "
                f"mass {rm.get('mass_kg')} kg [{rm.get('mass_source')}]); finite_burn_ok = burn ≤ 10 % of lead time and ≤ 1/10 orbit.")


def fig_breakup(res, sbm, event):
    lc = res.lc_m
    w = res.weight
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    a = axes[0]
    xs = np.logspace(math.log10(res.lc_min_m), math.log10(res.lc_max_m), 60)
    emp = np.array([np.sum(lc >= x) * w for x in xs])
    law = np.array([sbm.sbm_cumulative_count(res.m_ref_kg, x) - sbm.sbm_cumulative_count(res.m_ref_kg, res.lc_max_m) for x in xs])
    a.loglog(xs, law, color=INK, lw=1.4, ls="--", label=r"$0.1\,M^{0.75}L_c^{-1.71}$ (truncated at $L_{c,max}$)")
    a.step(xs, np.maximum(emp, 0.8), color=CAT[1], lw=1.6, where="post", label=f"sampled fragments (n={res.n_sampled})")
    a.set_xlabel("characteristic length Lc [m]"); a.set_ylabel("N(≥ Lc)")
    a.set_title(f"Size distribution, M_ref = {res.m_ref_kg:.0f} kg")
    a.legend(fontsize=6)
    b = axes[1]
    dv = np.linalg.norm(res.dv_kms, axis=1) * 1000
    b.hist(np.log10(dv), bins=40, color=TEAL, edgecolor="white", lw=0.4)
    med = np.median(dv)
    b.axvline(math.log10(med), color=INK, ls="--", lw=0.8)
    b.text(math.log10(med) + 0.05, b.get_ylim()[1] * 0.9, f"median {med:.0f} m/s", fontsize=6.3)
    b.set_xlabel("log10 ejection Δv [m/s]"); b.set_ylabel("fragments")
    b.set_title("Ejection speed: log10 Δv ~ N(0.9χ + 2.9, 0.4)")
    fig.tight_layout()
    ok = abs(res.stats["dv_ms_median"] - event.get("stats", {}).get("dv_ms_median", -1)) < 1e-2
    return save(fig, "17_breakup", f"app.core.breakup.simulate_breakup re-run with the event's own TCA states, masses "
                f"and seed {res.seed} (event {event['event_id']}); n_total {res.n_total} vs event {event['total_fragments']}, "
                f"Δv median {'identical to' if ok else 'differs from'} the backend event stats.")


def fig_cloud(res, snaps, event):
    """One row per parent: that parent's fragments projected on ITS OWN orbital plane at TCA."""
    RE = 6378.137
    tca = event["tca"]
    states = [(np.asarray(tca["position_a_eci"], float), np.asarray(tca["velocity_a_eci"], float)),
              (np.asarray(tca["position_b_eci"], float), np.asarray(tca["velocity_b_eci"], float))]
    hs = [np.cross(r, v) / np.linalg.norm(np.cross(r, v)) for r, v in states]
    plane_angle = math.degrees(math.acos(np.clip(abs(hs[0] @ hs[1]), 0, 1)))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.75))
    for k, ((r0, v0), h) in enumerate(zip(states, hs)):
        e1 = r0 / np.linalg.norm(r0)
        e2 = np.cross(h, e1)
        col = CAT[0] if k == 0 else CAT[1]
        for c, (t, r) in enumerate(snaps.items()):
            a = axes[k, c]
            m = (res.parent_index == k) & (np.linalg.norm(snaps[t], axis=1) - RE >= 100)
            x, y, z = r[m] @ e1, r[m] @ e2, r[m] @ h
            a.add_patch(Circle((0, 0), RE, fc="#dfe9f5", ec="#9fb6d3", lw=0.6, zorder=1))
            a.scatter(x, y, s=1.8, color=col, lw=0, zorder=3)
            a.scatter([r0 @ e1], [r0 @ e2], marker="x", color=BAD, s=20, lw=1.1, zorder=4)
            a.set_aspect("equal"); a.set_xlim(-8800, 8800); a.set_ylim(-8800, 8800)
            a.tick_params(labelsize=6)
            oop = np.percentile(np.abs(z), 90) if len(z) else 0.0
            a.set_title(f"+{t} min: {int(m.sum())} fragments" + chr(10) + f"90 % within {oop:.0f} km of plane",
                        fontsize=7, loc="center", linespacing=1.1)
            a.set_xticks([-8000, -4000, 0, 4000, 8000]); a.set_yticks([-8000, -4000, 0, 4000, 8000])
            if c == 0:
                a.set_ylabel(f"{event['parents'][k]['name'][:24]}\nalong-track axis [km]", fontsize=6.6)
            if k == 1:
                a.set_xlabel("radial axis at TCA [km]", fontsize=6.3)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    return save(fig, "18_cloud", "Fragments from the reproduced breakup propagated with app.core.breakup.propagate "
                "(RK4, two-body + J2 + Vallado drag, Cd·A/M). Each row shows ONE parent's fragments projected on that "
                f"parent's own orbital plane at TCA (the two planes are {plane_angle:.0f}° apart), so the stream stays "
                "outside the Earth disc; title gives fragments above 100 km and their 90th-percentile out-of-plane "
                "distance. × = impact point. Fragments below 100 km are removed.")


def fig_debris_alerts(alerts_deb):
    deb = [a for a in alerts_deb if a.get("source") == "debris"]
    if not deb:
        return None
    by_sat = defaultdict(list)
    for a in deb:
        by_sat[(a["sat1"]["id"], a["sat1"]["name"], a["sat1"].get("agency", "?"))].append(a)
    items = sorted(by_sat.items(), key=lambda kv: -max(x["probability_of_collision"] for x in kv[1]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.2, 1]})
    a = axes[0]
    ys = np.arange(len(items))[::-1]
    for y, ((sid, name, ag), lst) in zip(ys, items):
        pcs = [max(x["probability_of_collision"], 1e-20) for x in lst]
        a.scatter(np.log10(pcs), [y] * len(pcs), color=CAT[1], s=18, zorder=3)
        a.text(max(np.log10(pcs)) + 0.5, y, f"{len(lst)} frag., min miss {min(x['miss_distance_km'] for x in lst):.2f} km",
               va="center", fontsize=5.8, color=INK2)
    a.set_yticks(ys, [f"{n[:20]} #{s} ({ag})" for (s, n, ag), _ in items], fontsize=5.9)
    lo = min(math.log10(max(x["probability_of_collision"], 1e-20)) for x in deb)
    hi = max(math.log10(max(x["probability_of_collision"], 1e-20)) for x in deb)
    a.set_xlim(lo - 0.4, hi + 2.2)
    a.set_xlabel("log10 fragment Pc"); a.set_title(f"{len(deb)} fragment alerts on {len(items)} satellites")
    a.grid(axis="y", visible=False)
    b = axes[1]
    for i, ((sid, name, ag), lst) in enumerate(items):
        b.scatter([x["tca_minutes"] for x in lst], [x["miss_distance_km"] for x in lst], s=22,
                  color=CAT[i % len(CAT)], edgecolor="white", lw=0.4, label=f"#{sid}", zorder=3)
    b.axhline(5, color=INK2, ls=":", lw=0.8)
    b.text(0.02, 5.1, "5 km screening threshold", fontsize=5.6, color=INK2, va="bottom", transform=b.get_yaxis_transform())
    b.set_xlabel("minutes from now to fragment TCA"); b.set_ylabel("miss distance [km]")
    b.set_ylim(0, 6.2); b.set_title("When and how close")
    b.legend(fontsize=5.4, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.02))
    fig.tight_layout()
    return save(fig, "19_debris_alerts", "source=='debris' alerts from GET /api/alerts after POST /api/debris/simulate "
                "and a pipeline recompute (6 h window, 30 s step, 5 km threshold, isotropic Foster Pc); agency = SATCAT owner.")


def fig_cascade(alerts_deb, node_prob):
    import networkx as nx

    deb = [a for a in alerts_deb if a.get("source") == "debris"]
    if not deb:
        return None
    G = nx.DiGraph()
    ev = deb[0]["parent_event"]["event_id"]
    G.add_node(ev, layer=0, label="collision\n" + ev.split("-2026")[0].replace("evt-", "evt "))
    threatened = {}
    for a in deb:
        f = a["sat2"]["id"]
        s = a["sat1"]["id"]
        G.add_node(f"F{f}", layer=1, label=a["fragment_id"].split(":")[-1])
        G.add_edge(ev, f"F{f}")
        G.add_node(s, layer=2, label=f"{a['sat1']['name'][:18]}\n#{s}")
        G.add_edge(f"F{f}", s, pc=a["probability_of_collision"])
        threatened[s] = a["sat1"].get("agency", "?")
    nbr = []
    for a in alerts_deb:
        if a.get("source") != "screening":
            continue
        i1, i2 = a["sat1"]["id"], a["sat2"]["id"]
        for s, o, on in ((i1, i2, a["sat2"]), (i2, i1, a["sat1"])):
            if s in threatened and o not in threatened:
                nbr.append((a["probability_of_collision"], s, o, on))
    nbr.sort(key=lambda x: -x[0])
    for pc, s, o, on in nbr[:8]:
        G.add_node(o, layer=3, label=f"{on['name'][:18]} #{o}")
        G.add_edge(s, o, pc=pc)
    for s, ag in threatened.items():
        G.add_node(f"AG:{ag}", layer=4, label=ag)
        G.add_edge(s, f"AG:{ag}", agency=True)
    layers = defaultdict(list)
    for n, d in G.nodes(data=True):
        layers[d["layer"]].append(n)
    pos = {}
    for L, ns in layers.items():
        k = len(ns)
        sp = 1.0 if L == 1 else 1.55
        for i, n in enumerate(ns):
            pos[n] = (L, ((k - 1) / 2 - i) * sp)
    nmax = max(len(v) for k, v in layers.items() if k in (2, 3))
    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.62 * nmax + 1.2)))
    for u, v, d in G.edges(data=True):
        (x1, y1), (x2, y2) = pos[u], pos[v]
        style = ":" if d.get("agency") else "-"
        ax.annotate("", xy=(x2 - 0.06, y2), xytext=(x1 + 0.06, y1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED if d.get("agency") else INK2, lw=0.7, ls=style))
    lcol = {0: BAD, 1: CAT[1], 2: BLUE, 3: CAT[6], 4: GOOD}
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        size = 260 if d["layer"] != 1 else 70
        ax.scatter([x], [y], s=size, color=lcol[d["layer"]], edgecolor="white", lw=1, zorder=3)
        lab = d["label"]
        if d["layer"] == 2:
            p = node_prob.get(str(n)) if node_prob else None
            if p is not None:
                lab = lab.replace("\n#", " #") + f"\nP(hit) {p:.1e}"
        ax.text(x, y - (0.3 if d["layer"] != 1 else 0.22), lab, ha="center", va="top", fontsize=5.0 if d["layer"] == 1 else 5.5,
                color=INK, linespacing=1.05, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.3) if d["layer"] in (2, 3) else None)
    top_y = max(p[1] for p in pos.values()) + 0.6
    for L, t in enumerate(["collision\nevent", "fragments\n(depth 1)", "threatened satellites\n(depth 2)",
                           "their conjunctions\n(depth 3)", "agencies\n(SATCAT owner)"]):
        if L in layers:
            ax.text(L, top_y, t, ha="center", va="bottom", fontsize=6.6, weight="bold", color=lcol[L], linespacing=1.0)
    ax.axis("off")
    allys = [p[1] for p in pos.values()]
    ax.set_ylim(min(allys) - 1.4, max(allys) + 1.5); ax.set_xlim(-0.5, 4.5)
    return save(fig, "20_cascade", "Graph built from GET /api/alerts after the debris recompute: debris alerts give "
                "event → fragment → satellite edges, screening alerts give the satellites' own conjunctions (top 8 by Pc), "
                "agency from alert.sat1.agency; P(hit) from node_probabilities.")


# ══════════════════════════════════════════════════════════════════════════════
# PDF assembly (reportlab)
# ══════════════════════════════════════════════════════════════════════════════

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.pdfgen import canvas as rl_canvas  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether,  # noqa: E402
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", str(FONT_DIR / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(FONT_DIR / "DejaVuSansMono.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402

registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

C = colors.HexColor
PW, PH = A4
MARGIN = 15 * mm
TW = PW - 2 * MARGIN

ST = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=8.6, leading=12, textColor=C(INK)),
    "small": ParagraphStyle("small", fontName="DV", fontSize=7.4, leading=10, textColor=C(INK2)),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=6.4, leading=8.2, textColor=C(MUTED), spaceBefore=1, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=10.5, leading=14, textColor=C(NAVY), spaceBefore=6, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=6.6, leading=8.4, textColor=C(INK)),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=6.6, leading=8.4, textColor=C(INK)),
    "cellh": ParagraphStyle("cellh", fontName="DV-B", fontSize=6.8, leading=8.6, textColor=colors.white),
    "eq": ParagraphStyle("eq", fontName="DV", fontSize=8.6, leading=12.5, textColor=C(NAVY)),
    "bul": ParagraphStyle("bul", fontName="DV", fontSize=8.4, leading=11.6, textColor=C(INK), leftIndent=9,
                          bulletIndent=0, spaceAfter=1.5, bulletFontName="DV", bulletColor=C(TEAL)),
    "kpi": ParagraphStyle("kpi", fontName="DV-B", fontSize=17, leading=21, textColor=C(INK)),
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=34, leading=40, textColor=colors.white),
    "lab": ParagraphStyle("lab", fontName="DV-B", fontSize=6.6, leading=8, textColor=C(TEAL)),
}

SUPER = {"²": "2", "³": "3", "⁻": "−", "¹": "1", "⁰": "0", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "ᵀ": "T"}
SUB = {"₀": "0", "₁": "1", "₂": "2", "ᵢ": "i", "₃": "3", "ₖ": "k", "₄": "4", "₅": "5", "₆": "6", "₇": "7",
       "₈": "8", "₉": "9", "ₓ": "x", "ₙ": "n"}


def rl(text) -> str:
    """Escape for reportlab Paragraph and replace unicode super/subscripts by markup."""
    s = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = []
    for ch in s:
        if ch in SUPER:
            out.append(f"<super>{SUPER[ch]}</super>")
        elif ch in SUB:
            out.append(f"<sub>{SUB[ch]}</sub>")
        else:
            out.append(ch)
    return "".join(out)


def P(text, style="body"):
    return Paragraph(text, ST[style])


def fmt(v, digits=3):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    try:
        v = float(v)
    except Exception:
        return rl(v)
    if v == 0:
        return "0"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        m, e = f"{v:.{digits-1}e}".split("e")
        return f"{m}×10<super>{int(e)}</super>"
    return f"{v:.{digits}g}"


def section_header(num, title, subtitle=""):
    t = Table([[P(f'<font color="{TEAL}" size="15"><b>{num:02d}</b></font>', "body"),
                P(f'<font color="white" size="13"><b>{rl(title)}</b></font><br/>'
                  f'<font color="#b9c7dc" size="7.5">{rl(subtitle)}</font>', "body")]],
              colWidths=[14 * mm, TW - 14 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C(NAVY)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    t._section_title = f"{num:02d}  {title}"
    return [SectionMarker(f"{num:02d} · {title}"), t, Spacer(1, 5)]


def what_how_why(what, how, why):
    cells = []
    for lab, txt, col in (("WHAT IT IS", what, BLUE), ("HOW IT'S COMPUTED", how, TEAL), ("WHY TRUST IT", why, GOOD)):
        cells.append([P(f'<font color="{col}"><b>{lab}</b></font>', "lab"), P(txt, "small")])
    inner = [Table([[c[0]], [c[1]]], colWidths=[TW / 3 - 14]) for c in cells]
    for it in inner:
        it.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    t = Table([inner], colWidths=[TW / 3] * 3)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C(SURF)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LINEBEFORE", (0, 0), (0, 0), 2.5, C(BLUE)), ("LINEBEFORE", (1, 0), (1, 0), 2.5, C(TEAL)),
                           ("LINEBEFORE", (2, 0), (2, 0), 2.5, C(GOOD)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return [t, Spacer(1, 6)]


def callout(title, eq_lines, note_text=None):
    rows = [[P(f'<font color="{TEAL}"><b>{rl(title)}</b></font>', "lab")]]
    for ln in eq_lines:
        rows.append([P(ln, "eq")])
    if note_text:
        rows.append([P(note_text, "small")])
    t = Table(rows, colWidths=[TW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C(TEAL_L)), ("LINEBEFORE", (0, 0), (0, -1), 3, C(TEAL)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (0, 0), 5),
                           ("BOTTOMPADDING", (0, -1), (-1, -1), 6)]))
    return [t, Spacer(1, 6)]


def figure(name, width=TW, path=None, keep=True):
    path = path or FIG / f"{name}.png"
    if not Path(path).exists():
        return [P(f'<font color="{BAD}">Figure {rl(name)} unavailable (data missing).</font>', "small")]
    ir = ImageReader(str(path))
    iw, ih = ir.getSize()
    img = Image(str(path), width=width, height=width * ih / iw)
    parts = [img, P("Source: " + rl(CAPTIONS.get(name, "")), "cap")]
    return [KeepTogether(parts)] if keep else parts


def bullets(items):
    return [Paragraph(t, ST["bul"], bulletText="▸") for t in items]


def kpi_table(tiles, dark=False, ncol=None):
    ncol = ncol or len(tiles)
    w = TW / ncol
    cells = []
    for val, lab, col in tiles:
        cells.append([P(f'<font color="{col}"><b>{val}</b></font>', "kpi"),
                      P(f'<font color="{"#b9c7dc" if dark else INK2}" size="6.8">{lab}</font>', "body")])
    rows = []
    for i in range(0, len(cells), ncol):
        chunk = cells[i:i + ncol]
        inner = []
        for c in chunk:
            it = Table([[c[0]], [c[1]]], colWidths=[w - 8])
            it.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
            inner.append(it)
        while len(inner) < ncol:
            inner.append("")
        rows.append(inner)
    t = Table(rows, colWidths=[w] * ncol)
    bg = C(NAVY2) if dark else C(SURF)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 8),
             ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]
    for r in range(len(rows)):
        for cidx in range(ncol):
            style.append(("BACKGROUND", (cidx, r), (cidx, r), bg))
            style.append(("LINEABOVE", (cidx, r), (cidx, r), 2, C(TEAL)))
    style.append(("LINEAFTER", (0, 0), (-2, -1), 3, C(NAVY) if dark else colors.white))
    style.append(("LINEBELOW", (0, 0), (-1, -2), 3, C(NAVY) if dark else colors.white))
    t.setStyle(TableStyle(style))
    return t


def data_table(header, rows, widths, zebra=True, pass_col=None):
    data = [[P(rl(h), "cellh") for h in header]]
    for r in rows:
        data.append([c if not isinstance(c, str) else P(c, "cell") for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), C(NAVY)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("LINEBELOW", (0, 0), (-1, -1), 0.3, C(GRID))]
    if zebra:
        for i in range(1, len(data)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), C(SURF)))
    t.setStyle(TableStyle(st))
    return t


def badge(ok):
    if ok is None:
        return P(f'<font color="{MUTED}"><b>n/a</b></font>', "cell")
    return P(f'<font color="{GOOD if ok else BAD}"><b>{"PASS" if ok else "FAIL"}</b></font>', "cell")


from reportlab.platypus.flowables import Flowable  # noqa: E402


class SectionMarker(Flowable):
    """Zero-size flowable that records the current section for the running header."""

    def __init__(self, title):
        super().__init__()
        self.title = title

    def wrap(self, aw, ah):
        return 0, 0

    def draw(self):
        self.canv._section = self.title


class NumberedCanvas(rl_canvas.Canvas):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._saved = []
        self._section = ""

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            if self._pageNumber > 1:
                self.setFont("DV", 6.8)
                self.setFillColor(C(MUTED))
                self.drawRightString(PW - MARGIN, 9 * mm, f"Page {self._pageNumber} / {n}")
                self.drawString(MARGIN, 9 * mm, "Every figure is generated from live endpoints, model artifacts "
                                "or backend modules — see the caption under each chart.")
            super().showPage()
        super().save()


def on_cover(canv, doc):
    canv.saveState()
    canv.setFillColor(C(NAVY))
    canv.rect(0, 0, PW, PH, fill=1, stroke=0)
    # orbit art
    canv.setStrokeColor(C("#1d3b66"))
    canv.setLineWidth(0.8)
    cx, cy = PW * 0.80, PH * 0.80
    for i, (rx, ry) in enumerate(((95, 38), (130, 56), (170, 78), (215, 100))):
        canv.saveState()
        canv.translate(cx, cy)
        canv.rotate(-18 + i * 9)
        canv.ellipse(-rx * mm / 2.2, -ry * mm / 2.2, rx * mm / 2.2, ry * mm / 2.2, stroke=1, fill=0)
        canv.restoreState()
    canv.setFillColor(C("#1f4f86"))
    canv.circle(cx, cy, 13 * mm, stroke=0, fill=1)
    canv.setFillColor(C(TEAL))
    for ang, rr in ((30, 52), (200, 70), (120, 92), (300, 108)):
        canv.circle(cx + rr * mm / 2.2 * math.cos(math.radians(ang)) * 1.1,
                    cy + rr * mm / 2.2 * math.sin(math.radians(ang)) * 0.45, 1.3 * mm, stroke=0, fill=1)
    canv.setFillColor(C(TEAL))
    canv.rect(0, PH - 6 * mm, PW, 6 * mm, fill=1, stroke=0)
    canv.restoreState()


def on_page(canv, doc):
    canv.saveState()
    canv.setFillColor(C(NAVY))
    canv.rect(0, PH - 11 * mm, PW, 11 * mm, fill=1, stroke=0)
    canv.setFillColor(C(TEAL))
    canv.rect(0, PH - 11.8 * mm, PW, 0.8 * mm, fill=1, stroke=0)
    canv.setFillColor(colors.white)
    canv.setFont("DV-B", 8)
    canv.drawString(MARGIN, PH - 7 * mm, "ORBITAL SENTINEL")
    canv.setFont("DV", 7.5)
    canv.setFillColor(C("#b9c7dc"))
    canv.drawString(MARGIN + 31 * mm, PH - 7 * mm, "Judges' technical report")
    sec = getattr(canv, "_section", "")
    canv.drawRightString(PW - MARGIN, PH - 7 * mm, sec)
    canv.setStrokeColor(C(GRID))
    canv.line(MARGIN, 13 * mm, PW - MARGIN, 13 * mm)
    canv.restoreState()


def build_pdf(story):
    doc = BaseDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=17 * mm, bottomMargin=16 * mm, title="Orbital Sentinel — Judges' Report",
                          author="Orbital Sentinel team", subject="Space traffic management: physics, ML and cascade analysis")
    cover = Frame(MARGIN, MARGIN, TW, PH - 2 * MARGIN, id="cover", showBoundary=0)
    body = Frame(MARGIN, 16 * mm, TW, PH - 33 * mm, id="body", showBoundary=0)
    doc.addPageTemplates([PageTemplate("cover", [cover], onPage=on_cover),
                          PageTemplate("body", [body], onPageEnd=on_page)])
    doc.build(story, canvasmaker=NumberedCanvas)


# ══════════════════════════════════════════════════════════════════════════════
# Story
# ══════════════════════════════════════════════════════════════════════════════

def _ver_ratio(x):
    """error / tolerance using whichever error metric (abs or rel) satisfies the tolerance; None if tol is 0."""
    t = x.get("tolerance")
    if not isinstance(t, (int, float)) or isinstance(t, bool) or t <= 0:
        return None
    vals = []
    for k in ("abs_err", "rel_err"):
        e = x.get(k)
        if isinstance(e, (int, float)) and not isinstance(e, bool) and math.isfinite(e) and e < 1e10:
            vals.append(abs(e) / t)
    return min(vals) if vals else None


def fig_verification(ver):
    areas = [a for a, _ in Counter(x["area"] for x in ver).most_common()]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [0.85, 1.4]})
    a = axes[0]
    ys = np.arange(len(areas))[::-1]
    for y, ar in zip(ys, areas):
        xs = [x for x in ver if x["area"] == ar]
        npass = sum(1 for x in xs if x.get("pass"))
        nfix = sum(1 for x in xs if x.get("fixed"))
        a.barh(y, npass, color=GOOD, height=0.62)
        if len(xs) - npass:
            a.barh(y, len(xs) - npass, left=npass, color=BAD, height=0.62)
        a.text(len(xs) + 0.4, y, f"{npass}/{len(xs)}" + (f"  ({nfix} fixed)" if nfix else ""), va="center", fontsize=6)
    a.set_yticks(ys, areas, fontsize=6.4); a.set_xlabel("checks"); a.grid(axis="y", visible=False)
    a.set_xlim(0, max(Counter(x["area"] for x in ver).values()) * 1.45)
    a.set_title("Checks per area")
    b = axes[1]
    rng = np.random.default_rng(0)
    n_exact = 0
    for y, ar in zip(ys, areas):
        rs = []
        for x in ver:
            if x["area"] != ar:
                continue
            r = _ver_ratio(x)
            if r is None:
                n_exact += 1
                continue
            rs.append(max(r, 1e-16))
        if rs:
            b.scatter(np.log10(rs), y + rng.uniform(-0.18, 0.18, len(rs)), s=12, color=BLUE, edgecolor="white", lw=0.3, zorder=3)
    b.axvline(0, color=BAD, lw=1, ls="--")
    b.text(0.05, ys.max() + 0.3, "tolerance", color=BAD, fontsize=6.3)
    b.set_yticks(ys, [""] * len(areas)); b.set_xlim(-16.5, 1.2); b.set_ylim(-0.7, len(areas) - 0.2)
    b.set_xlabel("log10 (error ÷ tolerance)   — left of the red line = inside tolerance")
    b.set_title(f"Error ÷ tolerance  ({n_exact} exact checks not shown)")
    fig.tight_layout(w_pad=0.6)
    return save(fig, "21_verification", "docs/report/verification.json (independent re-implementations: scipy, sklearn, "
                "raw sgp4, DOP853 integrators). error = abs_err or rel_err, whichever the tolerance applies to; "
                "checks with tolerance 0 (exact equality / inequality) are counted on the left only.")


def _md_inline(t: str) -> str:
    t = t.strip().replace(chr(92) + "|", "|")
    t = rl(t)
    import re
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r'<font name="DVM" size="6.4">\1</font>', t)
    t = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    return t


def _md_section(md: str, heading: str) -> list[str]:
    lines, on = [], False
    for ln in md.splitlines():
        if ln.startswith("## "):
            on = ln[3:].strip().lower().startswith(heading.lower())
            continue
        if on:
            lines.append(ln)
    return lines


def _md_table(lines):
    rows = []
    for ln in lines:
        if not ln.strip().startswith("|"):
            continue
        body = ln.strip().replace(chr(92) + "|", "\x00")
        cells = [c.replace("\x00", "|") for c in body.strip("|").split("|")]
        if all(set(c.strip()) <= set("-: ") for c in cells):
            continue
        rows.append(cells)
    return rows


def _md_bullets(lines):
    items, cur = [], None
    for ln in lines:
        if ln.startswith("* ") or ln.startswith("- "):
            if cur:
                items.append(cur)
            cur = ln[2:].strip()
        elif cur is not None and ln.strip():
            cur += " " + ln.strip()
    if cur:
        items.append(cur)
    return items


def verification_section(ver, md: str):
    out = []
    n = len(ver)
    npass = sum(1 for x in ver if x.get("pass"))
    fixed = [x for x in ver if x.get("fixed")]
    areas = Counter(x["area"] for x in ver)
    out.append(kpi_table([(f"{n}", "independent checks", NAVY), (f"{npass}/{n}", "pass (after fixes)", GOOD if npass == n else BAD),
                          (f"{len(fixed)}", "genuine defects found & fixed", CAUTION), (f"{len(areas)}", "areas covered", BLUE)]))
    out.append(Spacer(1, 6))
    out += what_how_why(
        "A separate agent recomputed every mathematical component with a different implementation, without reusing "
        "the project's code path.",
        "scipy dblquad / ncx2 / solve_ivp (DOP853), sklearn PCA & metrics, raw sgp4 brute-force scans, KS tests, an own "
        "re-implementation of the decision formula; each record logs ours vs independent, error and tolerance.",
        "It found real bugs (below) and they were fixed and re-tested — evidence that the checks have teeth, "
        "not just that they pass.")
    out += figure("21_verification")
    defects = _md_table(_md_section(md, "Defects found"))
    if len(defects) > 1:
        out.append(P("Defects found and fixed", "h2"))
        hdr = [c.strip() for c in defects[0]]
        rows = [[_md_inline(c) for c in r] for r in defects[1:]]
        out.append(data_table(hdr, rows, [TW * 0.10, TW * 0.36, TW * 0.27, TW * 0.27]))
    elif fixed:
        out.append(P("Defects found and fixed", "h2"))
        out.append(data_table(["id", "check", "fix"], [[rl(x["id"]), rl(x["check"]), rl(x.get("fix_summary", ""))] for x in fixed],
                              [TW * 0.12, TW * 0.38, TW * 0.50]))
    hl = _md_table(_md_section(md, "What was verified"))
    if len(hl) > 1:
        out.append(CondPageBreak(60 * mm))
        out.append(P("What was verified — independent method and worst error per area", "h2"))
        hdr = [c.strip() for c in hl[0]]
        rows = [[f"<b>{_md_inline(r[0])}</b>"] + [_md_inline(c) for c in r[1:]] for r in hl[1:]]
        out.append(data_table(hdr, rows, [TW * 0.17, TW * 0.48, TW * 0.35]))
    else:
        rows = []
        for ar, cnt in areas.most_common():
            xs = [x for x in ver if x["area"] == ar]
            worst = max(xs, key=lambda x: _ver_ratio(x) or 0)
            rows.append([f"<b>{rl(ar)}</b>", str(cnt), rl(worst.get("method_independent", ""))[:120],
                         f"{rl(worst['id'])}: abs {fmt(worst.get('abs_err'))}"])
        out.append(data_table(["Area", "n", "Independent method", "Worst"], rows, [TW * 0.17, TW * 0.06, TW * 0.5, TW * 0.27]))
    lim = _md_bullets(_md_section(md, "Honest limitations"))
    if lim:
        out.append(Spacer(1, 6))
        out.append(P("Limitations stated by the verifier", "h2"))
        out += bullets([_md_inline(x) for x in lim])
    return out


def verification_flowables(ver):
    out = []
    if ver is None:
        out.append(P(f'<font color="{CAUTION}"><b>Independent verification file not found</b></font> '
                     "(docs/report/verification.json). Re-run this script after the verifier has finished; the "
                     "section then lists every independent check.", "small"))
        return out
    # generic: find a list of check dicts
    checks = None
    if isinstance(ver, list):
        checks = ver
    elif isinstance(ver, dict):
        for k in ("checks", "results", "verifications", "items", "tests"):
            if isinstance(ver.get(k), list):
                checks = ver[k]
                break
        meta = {k: v for k, v in ver.items() if not isinstance(v, (list, dict))}
        summ = ver.get("summary") if isinstance(ver.get("summary"), dict) else None
        info = "; ".join(f"<b>{rl(k)}</b>: {rl(v)}" for k, v in list(meta.items())[:8])
        if summ:
            info += ("; " if info else "") + "; ".join(f"<b>{rl(k)}</b>: {rl(v)}" for k, v in summ.items()
                                                        if not isinstance(v, (list, dict)))
        if info:
            out.append(P(info, "small"))
            out.append(Spacer(1, 4))
        if checks is None:
            # dict of dicts
            checks = [{"name": k, **v} if isinstance(v, dict) else {"name": k, "value": v}
                      for k, v in ver.items() if isinstance(v, (dict,)) and k != "summary"]
    if not checks:
        out.append(P("verification.json present but contains no check list.", "small"))
        return out
    rows = []
    for c in checks:
        if not isinstance(c, dict):
            rows.append([rl(c), "", "", badge(None)])
            continue
        name = c.get("name") or c.get("check") or c.get("id") or c.get("title") or "?"
        ok = None
        for k in ("pass", "passed", "ok", "status", "result", "verdict"):
            if k in c:
                v = c[k]
                if isinstance(v, bool):
                    ok = v
                elif isinstance(v, str):
                    u = v.upper()
                    ok = True if u in ("PASS", "PASSED", "OK", "TRUE", "VERIFIED", "AGREE", "CONFIRMED") else (
                        False if u in ("FAIL", "FAILED", "FALSE", "MISMATCH", "BUG", "DISAGREE") else None)
                break
        ours = c.get("ours", c.get("our_value", c.get("backend_value", c.get("value"))))
        ref = c.get("independent", c.get("reference", c.get("reference_value", c.get("independent_value", c.get("expected")))))
        detail = c.get("detail") or c.get("details") or c.get("note") or c.get("notes") or c.get("method") or c.get("description") or ""
        if isinstance(detail, (dict, list)):
            detail = json.dumps(detail)[:220]
        vals = []
        if ours is not None:
            vals.append(f"ours {fmt(ours) if not isinstance(ours, (dict, list)) else rl(json.dumps(ours)[:60])}")
        if ref is not None:
            vals.append(f"indep. {fmt(ref) if not isinstance(ref, (dict, list)) else rl(json.dumps(ref)[:60])}")
        err = c.get("rel_error", c.get("abs_error", c.get("error")))
        if err is not None and not isinstance(err, (dict, list)):
            vals.append(f"err {fmt(err)}")
        rows.append([f"<b>{rl(name)}</b>", "<br/>".join(vals), rl(str(detail)[:260]), badge(ok)])
    out.append(data_table(["Independent check", "Values", "Detail", ""], rows,
                          [TW * 0.30, TW * 0.20, TW * 0.42, TW * 0.08]))
    npass = sum(1 for r in rows if "PASS" in getattr(r[3], "text", ""))
    out.insert(0, P(f"<b>{npass} / {len(rows)}</b> independent checks pass.", "body"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use cached data only (no HTTP)")
    ap.add_argument("--no-scenario", action="store_true", help="reuse the cached scenario run")
    ap.add_argument("--no-bench", action="store_true", help="reuse cached screening benchmark")
    args = ap.parse_args()

    t_start = time.perf_counter()
    up = (not args.offline) and backend_up()
    if not up:
        note("backend not reachable: using cached endpoint data")
    print("collecting live endpoints ...")
    live = collect_live(offline=not up)
    if up and not args.no_scenario:
        print("running cascade_demo scenario (with cleanup) ...")
        try:
            scen = run_scenario()
        except Exception as exc:
            note(f"scenario run failed: {exc}; using cache")
            scen = load_cache("scenario_run")
    else:
        scen = load_cache("scenario_run")
    if scen is None:
        note("no scenario data available")
        scen = {}

    an = D(live, "analytics_model") or load_json(ARTIFACTS / "risk_model_analytics.json") or {}
    if "feature_labels" not in an:
        an.setdefault("feature_labels", {})
    card = load_json(ARTIFACTS / "risk_model_card.json") or (an.get("card") or {})
    val = D(live, "physics_validation") or {}
    alerts_payload = D(live, "alerts") or {}
    alerts = alerts_payload.get("alerts", [])
    sats_payload = D(live, "satellites") or {}
    sats = sats_payload.get("satellites", [])
    agencies = D(live, "agencies") or {}
    mm_ = D(live, "model_metrics") or {}
    satcat_hdr, satcat_rows = read_satcat()
    ver = load_json(VERIFICATION) if VERIFICATION.exists() else None
    if ver is None:
        note("docs/report/verification.json not present yet")

    scen_alerts = (scen.get("alerts") or {}).get("alerts", [])
    deb_payload = scen.get("alerts_debris") or {}
    deb_alerts = deb_payload.get("alerts", [])
    syn_id = (((scen.get("scenario") or {}).get("object_b")) or {}).get("norad_id", 99901)
    pair_ids = {syn_id, (((scen.get("scenario") or {}).get("object_a")) or {}).get("norad_id")}
    scen_alert = None
    for a in sorted(scen_alerts, key=lambda x: -x["probability_of_collision"]):
        if {a["sat1"]["id"], a["sat2"]["id"]} == pair_ids:
            scen_alert = a
            break
    if scen_alert is None and scen_alerts:
        scen_alert = max(scen_alerts, key=lambda x: x["probability_of_collision"])
        note("scenario pair alert not found; using highest-Pc alert of the run")
    if scen_alert is None and alerts:
        scen_alert = max(alerts, key=lambda x: x["probability_of_collision"])
    event = (scen.get("debris") or {}).get("event")

    # ── computations ────────────────────────────────────────────────────────
    print("regenerating held-out split ...")
    ho = None
    try:
        ho = heldout_predictions()
    except Exception as exc:
        note(f"held-out regeneration failed: {exc}")
        traceback.print_exc()
    bench = load_cache("screening_bench") if args.no_bench else None
    if bench is None:
        print("screening benchmark ...")
        try:
            ts = alerts_payload.get("timestamp")
            sim_time = datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc)
            bench = screening_benchmark(sim_time)
            store("screening_bench", bench)
        except Exception as exc:
            note(f"screening benchmark failed: {exc}")
            bench = load_cache("screening_bench") or []
    res = snaps = sbm = None
    if event:
        try:
            res, snaps, sbm = reproduce_breakup(event)
        except Exception as exc:
            note(f"breakup reproduction failed: {exc}")

    # ── figures ─────────────────────────────────────────────────────────────
    print("drawing figures ...")
    figs = {}

    def mk(name, fn, *a):
        try:
            figs[name] = fn(*a)
        except Exception as exc:
            note(f"figure {name} failed: {exc}")
            traceback.print_exc()
            plt.close("all")

    mk("pipeline", fig_pipeline)
    mk("regimes", fig_catalogue, sats, satcat_rows, satcat_hdr)
    mk("agencies", fig_agencies, agencies, satcat_rows, sats)
    mk("screening", fig_screening, alerts, bench)
    if scen_alert:
        mk("bplane", fig_bplane, scen_alert)
        mk("waterfall", fig_waterfall, scen_alert)
        mk("decision_breakdown", fig_decision_breakdown, scen_alert)
        if scen_alert.get("recommended_maneuver"):
            mk("options", fig_options, scen_alert["recommended_maneuver"], scen_alert)
            mk("engines", fig_engines, scen_alert["recommended_maneuver"])
    mk("tle_sigma", fig_tle_sigma, alerts)
    mk("pcchecks", fig_pcchecks, [a for a in (scen_alerts or alerts) if a.get("source", "screening") == "screening"])
    if an.get("correlation"):
        mk("corr", fig_corr, an)
        mk("pca", fig_pca, an)
        mk("pca_vs_raw", fig_pca_vs_raw, an)
        mk("shap", fig_shap, an)
    if ho:
        mk("heldout", fig_heldout, ho, card)
    mk("decision", fig_decision, scen_alerts or alerts, scen_alert)
    if res is not None:
        mk("breakup", fig_breakup, res, sbm, event)
        mk("cloud", fig_cloud, res, snaps, event)
    if deb_alerts:
        mk("debris_alerts", fig_debris_alerts, deb_alerts)
        mk("cascade", fig_cascade, deb_alerts, deb_payload.get("node_probabilities", {}))

    # ── numbers used in text ────────────────────────────────────────────────
    n_obj = sats_payload.get("count", len(sats))
    bench500 = next((r.get("t_total_s", float("nan")) for r in (bench or []) if r.get("objects") == 500), float("nan"))
    n_alerts = alerts_payload.get("count", len(alerts))
    sev = Counter(a["severity"] for a in alerts)
    summ = val.get("summary", {})
    n_pass, n_tot = summ.get("passed", 0), summ.get("total", 0)
    held = card.get("heldout", {})
    c14 = held.get("classification_at_1e-4", {})
    reg = held.get("regression", {})
    base_mae = (((card.get("baselines") or {}).get("miss_distance_only_xgboost") or {}).get("regression") or {}).get("mae_log10_pc")
    unk = next((x["count"] for x in agencies.get("agencies", []) if x["name"] == "Unknown"), 0)
    ag_total = agencies.get("total", n_obj)
    rm = (scen_alert or {}).get("recommended_maneuver") or {}
    chosen = (rm.get("options") or [{}])[rm.get("chosen_index", 0) or 0] if rm.get("options") else rm
    n_deb = sum(1 for a in deb_alerts if a.get("source") == "debris")
    n_deb_sats = len({a["sat1"]["id"] for a in deb_alerts if a.get("source") == "debris"})
    fetched = (live.get("alerts") or {}).get("_fetched_utc", "")[:16].replace("T", " ")
    sim_epoch = alerts_payload.get("timestamp", "")[:16].replace("T", " ")
    if isinstance(ver, list) and ver and isinstance(ver[0], dict) and "area" in ver[0]:
        try:
            mk("verification", fig_verification, ver)
            vf = verification_section(ver, VERIFICATION_MD.read_text(encoding="utf-8") if VERIFICATION_MD.exists() else "")
        except Exception as exc:
            note(f"verification renderer failed: {exc}")
            traceback.print_exc()
            vf = verification_flowables(ver)
    else:
        vf = verification_flowables(ver)

    # ══ story ══════════════════════════════════════════════════════════════
    S = []
    # cover
    S.append(Spacer(1, 30 * mm))
    S.append(P(f'<font color="{TEAL}" size="9"><b>SPACE TRAFFIC MANAGEMENT · HACKATHON SUBMISSION</b></font>', "body"))
    S.append(Spacer(1, 4 * mm))
    S.append(P('Orbital Sentinel', "title"))
    S.append(Spacer(1, 4 * mm))
    S.append(P('<font color="#d6e2f2" size="13">From two-line elements to a cascade-safe avoidance burn — '
               "every number computed, cross-checked and traceable.</font>", "body"))
    S.append(Spacer(1, 12 * mm))
    tiles = [
        (f"{n_obj:,}", "objects tracked live (SGP4)", "white"),
        (f"{n_alerts}", f"conjunction alerts in 24 h ({sev.get('CRITICAL', 0)} critical)", "white"),
        (f"{n_pass}/{n_tot}", "physics cross-checks passing", "#5fd39a" if n_pass == n_tot and n_tot else "#f0c674"),
        (f"{c14.get('f1', 0):.3f}", "ML surrogate F1 @ Pc 1e-4 (held-out)", "white"),
        (f"{unk}/{ag_total}", "objects with unknown agency", "white"),
        ((f"{sum(1 for x in ver if x.get('pass'))}/{len(ver)}", "independent maths checks passing", "#5fd39a")
         if isinstance(ver, list) and ver else (f"{len(satcat_rows):,}", "SATCAT rows for owner / type / RCS", "white")),
    ]
    S.append(kpi_table(tiles, dark=True, ncol=3))
    S.append(Spacer(1, 14 * mm))
    S.append(P(f'<font color="#b9c7dc" size="8">Live backend snapshot: simulation epoch {rl(sim_epoch)} UTC '
               f"(fetched {rl(fetched)} UTC). Demo scenario run: {rl(scen.get('_run_utc', 'n/a')[:16].replace('T', ' '))} UTC. "
               f"Report generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by docs/report/build_report.py.</font>", "body"))
    S.append(Spacer(1, 3 * mm))
    S.append(P('<font color="#b9c7dc" size="8">Repository: orbit-sentinal · FastAPI backend + React/Cesium frontend · '
               "deep-dive: docs/HOW_IT_WORKS.md</font>", "body"))
    S.append(Spacer(1, 12 * mm))
    S.append(P(f'<font color="{TEAL}" size="9"><b>INSIDE THIS REPORT</b></font>', "body"))
    S.append(Spacer(1, 3 * mm))
    toc = [("00", "Executive summary"), ("01", "System pipeline"), ("02", "Catalogue & agencies"),
           ("03", "Conjunction screening"), ("04", "Collision probability"), ("05", "ML surrogate & explainability"),
           ("06", "Decision score"), ("07", "Cascade-safe avoidance"), ("08", "Collision → debris → cascade"),
           ("09", "Physics cross-validation"), ("10", "Independent verification"), ("11", "Assumptions, limitations & how to verify")]
    half = (len(toc) + 1) // 2
    trows = []
    for i in range(half):
        row = []
        for j in (i, i + half):
            if j < len(toc):
                row.append(P(f'<font color="{TEAL}" size="10"><b>{toc[j][0]}</b></font>&nbsp;&nbsp;'
                             f'<font color="#d6e2f2" size="10">{rl(toc[j][1])}</font>', "body"))
            else:
                row.append("")
        trows.append(row)
    tt = Table(trows, colWidths=[TW / 2] * 2)
    tt.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, C("#1d3b66")), ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    S.append(tt)
    S.append(NextPageTemplate("body"))
    S.append(PageBreak())

    # executive summary
    S += section_header(0, "Executive summary", "the problem, what we built, and the five results that matter")
    S.append(P("<b>The problem.</b> Low Earth orbit holds tens of thousands of catalogued objects. Operators need to "
               "know <i>which</i> close approaches are dangerous, <i>what</i> to do about them, and whether the fix "
               "creates a new problem. A single collision produces hundreds of fragments that threaten other "
               "satellites, sometimes owned by other agencies (the Kessler cascade)."))
    S.append(Spacer(1, 4))
    S.append(P("<b>What we built.</b> Orbital Sentinel runs the whole chain on a simulation clock: SGP4 propagation → "
               "24-hour conjunction screening → Foster collision probability with a TLE-age covariance → an "
               "explainable ML surrogate (advisory) → a single, documented decision score → re-propagated, "
               "cascade-safe avoidance burns costed on 12 real engines → a NASA Standard Breakup Model debris "
               "cloud → fragment screening → a cascade graph mapped to owning agencies through the CelesTrak SATCAT. "
               "A Cesium globe shows it live."))
    S.append(Spacer(1, 6))
    S.append(P("Five headline results", "h2"))
    hl = [
        f"<b>Screening is a real look-ahead.</b> {n_obj} objects are screened over 24 h at 60 s with a KD-tree and "
        f"Brent-refined TCA; this snapshot yields {n_alerts} pairs ({sev.get('CRITICAL', 0)} critical, "
        f"{sev.get('WARNING', 0)} warning). Screening 500 objects takes {bench500:.2f} s in-process on a laptop; "
        f"a full API pipeline recompute took {scen.get('pipeline_refresh_s', float('nan')):.1f} s during this report run.",
        f"<b>Collision probability is cross-validated.</b> {n_pass} of {n_tot} live physics checks pass "
        "(Foster vs Chan series, Monte Carlo, exact Rice CDF and Alfano maximum; propagation energy; CW, Hohmann "
        "and rocket-equation manoeuvre checks; NASA SBM fragment counts).",
        f"<b>The ML surrogate is measured, explained and kept in its place.</b> Held-out MAE "
        f"{reg.get('mae_log10_pc', float('nan')):.3f} dex vs {base_mae:.2f} dex for a miss-distance-only baseline; "
        f"precision {c14.get('precision', 0):.3f} / recall {c14.get('recall', 0):.3f} at Pc 1e-4. TreeSHAP shows the "
        f"main driver is <i>{rl(an.get('main_factor_label', '?'))}</i>. PCA was tested and rejected with numbers. "
        "The model never overwrites the physics Pc.",
        (f"<b>Avoidance is cascade-safe and costed.</b> For the demo encounter (Pc "
         f"{fmt(scen_alert['probability_of_collision'])}) the planner evaluated {rm.get('candidates_evaluated', '?')} burns, "
         f"re-screened shortlisted options against the whole catalogue, and chose <b>{rl(chosen.get('candidate', '?'))}</b> "
         f"(new Pc {fmt(chosen.get('new_pc_collision'))}, {chosen.get('propellant_kg', rm.get('propellant_kg', 0)):.3f} kg "
         f"propellant, engine {rl(rm.get('recommended_engine', '?'))}); the CW analytic drift agrees with propagation to "
         f"{100 * (rm.get('analytic_check') or {}).get('rel_error', float('nan')):.2f} %.") if scen_alert and rm else
        "<b>Avoidance</b>: scenario data unavailable in this build.",
        (f"<b>Collision → debris → cascade is end-to-end.</b> Breaking up the demo pair at its predicted TCA gives "
         f"{event['total_fragments']} fragments ≥ 10 cm (EMR {event['emr_j_per_g']:.0f} J/g, catastrophic); screening "
         f"them produced {n_deb} fragment alerts on {n_deb_sats} satellites, wired into the cascade graph and the owning "
         f"agencies.") if event else "<b>Debris</b>: scenario data unavailable in this build.",
    ]
    S += bullets(hl)
    S.append(Spacer(1, 6))
    S += callout("Design principle", [
        "physics decides → ML explains and cross-checks → the decision score combines both, transparently"],
        "Every assumption (mass, covariance growth, hard-body radius, Isp) is a named constant and each output "
        "carries a provenance tag (mass_source, sigma_source, hbr_source, isp_source). Section 11 lists what the "
        "system does <i>not</i> do.")
    S.append(PageBreak())

    # 1 pipeline
    S += section_header(1, "System pipeline", "what runs on every 30 s refresh and on a simulated collision")
    S += what_how_why(
        "One backend loop (refresh_alerts_once) that turns TLEs into alerts, decisions and burns; a second path "
        "turns a predicted collision into a debris cloud and new alerts.",
        "Each box is a module with its own tests: SGP4 (sgp4 lib), screening.py, pc_methods.py, ml/risk_api.py, "
        "decision.py, maneuver_planner.py, propulsion.py, breakup.py, debris_model.py, cascade_planner.py, satcat.py.",
        "All physics runs on one simulation clock; the ML branch is advisory and cannot lower the physics action; "
        "outputs carry provenance tags so judges can see which numbers are assumptions.")
    S += figure("01_pipeline")
    S.append(P("Stage-by-stage key equations", "h2"))
    rows = [
        ["SGP4", "TEME position/velocity from TLE; executed burns: r(t) = r<sub>SGP4</sub>(t) + [r<sub>J2</sub>(t; x<sub>0</sub>+Δv) − r<sub>J2</sub>(t; x<sub>0</sub>)]"],
        ["Screening", "KD-tree radius = threshold + v<sub>rel,max</sub>·step/2; t* = −(Δr·Δv)/|Δv|<super>2</super>; Brent refine (xatol 1 ms)"],
        ["Foster Pc", "Pc = ∬<sub>|x|≤HBR</sub> N(x; b, C<sub>B</sub>) dx,  C<sub>B</sub> = B(C<sub>1</sub>+C<sub>2</sub>)B<super>T</super>"],
        ["Surrogate", "XGBoost f(7 features) ≈ log<sub>10</sub> Pc;  base + Σ SHAP<sub>i</sub> = prediction"],
        ["Decision", "score = 100 Σ w<sub>i</sub> n<sub>i</sub>; action from physics Pc tiers 1e-4 / 1e-5 / 1e-7"],
        ["Manoeuvre", "m<sub>p</sub> = m<sub>0</sub>(1 − e<super>−Δv/(I<sub>sp</sub>g<sub>0</sub>)</super>);  t<sub>burn</sub> = m<sub>p</sub> I<sub>sp</sub> g<sub>0</sub> / F"],
        ["Breakup", "N(≥L<sub>c</sub>) = 0.1 M<super>0.75</super> L<sub>c</sub><super>−1.71</super>; catastrophic if EMR ≥ 40 J/g"],
        ["Cascade", "P(hit) = 1 − Π(1 − Pc<sub>i</sub>);  depth = BFS hops from the collision event"],
    ]
    S.append(data_table(["Stage", "Core relation (as implemented)"], rows, [TW * 0.16, TW * 0.84]))
    S.append(PageBreak())

    # 2 catalogue
    S += section_header(2, "Catalogue & agencies", "who is up there, where, and who owns it")
    S += what_how_why(
        f"{n_obj} objects propagated live, attributed to an owner; a {len(satcat_rows):,}-row SATCAT snapshot gives "
        "owner, object type, RCS size and orbit for the whole catalogue.",
        "Owner from the CelesTrak SATCAT owner code first, refined by operator name patterns (e.g. US-owned "
        "STARLINK-* → SpaceX). Inclination of live objects from the angular-momentum vector.",
        f"Only {unk} of {ag_total} live objects ({100 * agencies.get('unknown_fraction', 0):.1f} %) have no "
        "SATCAT owner; unknown owners, debris and rocket bodies are never commandable.")
    S += figure("02_regimes")
    S += figure("03_agencies")
    S.append(PageBreak())

    # 3 screening
    S += section_header(3, "Conjunction screening", "a 24-hour future window, not 'distance now'")
    S += what_how_why(
        "Every pair that comes within 25 km in the next 24 h, with its time of closest approach (TCA), miss "
        "distance, relative speed and B-plane geometry.",
        "SatrecArray SGP4 grid every 60 s → perigee/apogee shell prefilter → KD-tree query_pairs per step → linear "
        "relative-motion filter → bounded Brent minimisation of the exact SGP4 distance.",
        "tests/test_screening_real.py compares against a brute-force 1 s + 1 ms scan (agree to 1 s / 0.1 km) and "
        "finds a pair that is > 2000 km apart now but meets in 2 h.")
    S += callout("Closest approach inside a step", [
        "t* = −(Δr · Δv) / |Δv|<super>2</super>,   miss ≈ |Δr + Δv t*|   →   Brent refine on |r<sub>1</sub>(t) − r<sub>2</sub>(t)|<sub>SGP4</sub>",
        "Severity: CRITICAL if Pc ≥ 10<super>−4</super> or miss &lt; 1 km;  WARNING if Pc ≥ 10<super>−6</super> or miss &lt; 5 km;  else WATCH"])
    S += figure("04_screening")
    if bench:
        r500 = next((r for r in bench if r.get("objects") == 500), bench[-1])
        S.append(P(f"Measured: {r500.get('objects')} objects → {r500.get('alerts')} pairs in {r500.get('t_total_s')} s "
                   f"(KD radius {r500.get('kd_radius_km')} km). Runtime grows sub-quadratically because the KD-tree only "
                   "visits spatial neighbours.", "small"))
    top = sorted([a for a in alerts if a.get("source", "screening") == "screening"],
                 key=lambda a: (-a["probability_of_collision"], a["miss_distance_km"]))[:8]
    if top:
        S.append(Spacer(1, 5))
        S.append(P(f"Top {len(top)} live alerts by Foster Pc (GET /api/alerts)", "h2"))
        rows = []
        for a in top:
            dcs = a.get("decision") or {}
            rows.append([f"{rl(a['sat1']['name'][:24])} × {rl(a['sat2']['name'][:24])}",
                         f"{a['tca_hours']:.1f} h", f"{a['miss_distance_km']:.3f} km", f"{a['relative_speed_kms']:.2f} km/s",
                         fmt(a["probability_of_collision"]),
                         f'<font color="{SEV_COL.get(a["severity"], INK)}"><b>{a["severity"]}</b></font>',
                         f"{rl(dcs.get('action', '–'))} ({dcs.get('score', 0):.0f})",
                         "/".join(f"{x:.1f}" for x in (a.get("tle_age_days") or []))])
        S.append(data_table(["Pair", "TCA in", "Miss", "v_rel", "Foster Pc", "Severity", "Decision", "TLE age [d]"], rows,
                            [TW * 0.33, TW * 0.07, TW * 0.09, TW * 0.10, TW * 0.12, TW * 0.10, TW * 0.11, TW * 0.08]))
    S.append(PageBreak())

    # 4 Pc
    S += section_header(4, "Collision probability", "Foster 2-D Pc with a modelled TLE-age covariance — checked four ways")
    S += what_how_why(
        "The probability that the two hard bodies overlap at TCA, given position uncertainty. This number — not the "
        "ML model — drives severity and the action.",
        "Project the combined covariance onto the encounter (B-)plane perpendicular to Δv and integrate the 2-D Gaussian over the "
        "hard-body disk: 48-point Gauss–Legendre radially × 96-point trapezoid in angle.",
        "Every alert carries pc_checks: Chan series (analytic), Monte Carlo (sampling), Alfano max (upper bound). "
        "Foster also matches adaptive dblquad to 1e-4 in the test suite.")
    S += callout("Foster (1992) 2-D probability", [
        "Pc = (1 / 2π√|C<sub>B</sub>|) ∬<sub>|x|≤HBR</sub> exp(−½ (x−b)<super>T</super> C<sub>B</sub><super>−1</super> (x−b)) dx",
        "σ<sub>R,T,N</sub>(age) = σ<sub>0</sub> + g · |TCA − epoch|,  σ<sub>0</sub> = (0.10, 0.50, 0.15) km,  g = (0.10, 1.00, 0.10) km/day"],
        "HBR = sum of per-object radii (SATCAT RCS class, else type default; ISS 55 m).")
    S += figure("05_bplane")
    S += figure("05b_methods")
    if scen_alert:
        pcs = scen_alert.get("pc_checks", {})
        mc_txt = (f"gives {fmt(pcs.get('monte_carlo'))}" if pcs.get("monte_carlo") else
                  "is below its resolution, so it reports null instead of a noisy number")
        S.append(P(f"For the demo alert, Foster {fmt(pcs.get('foster'))} vs Chan {fmt(pcs.get('chan'))}: spread "
                   f"{pcs.get('spread_decades')} decades → <b>consistent = {pcs.get('consistent')}</b>. Monte Carlo "
                   f"({pcs.get('mc_samples')} samples, {pcs.get('mc_hits')} hits) {mc_txt}. "
                   f"Alfano's maximum {fmt(pcs.get('alfano_max'))} is what Pc could reach if the covariance size were wrong.", "small"))
    t2 = Table([[figure("06_tle_sigma", width=TW * 0.5, keep=False),
                 [P("Why model the covariance?", "h2"),
                  P("TLEs carry no covariance. We grow a diagonal RTN 1σ linearly with TLE age, dominated by the "
                    "along-track term (~1 km/day). Stale TLEs therefore <i>dilute</i> Pc: with a huge ellipse the "
                    "probability mass spreads far beyond the hard body. The dilution curve above shows the effect "
                    "directly — shrinking the covariance first raises Pc, then lowers it once the miss vector lies "
                    "outside the ellipse.", "small"),
                  Spacer(1, 4),
                  P("Alerts are flagged <b>stale_tle</b> when either TLE is more than 30 days old; short-encounter "
                    "validity (|Δv| ≥ 0.1 km/s) is flagged per alert.", "small")]]],
               colWidths=[TW * 0.52, TW * 0.48])
    t2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    S.append(t2)
    S.append(Spacer(1, 6))
    S.append(P("Do the methods agree on every alert, not just the demo?", "h2"))
    S += figure("05c_pcchecks")
    S.append(PageBreak())

    # 5 ML
    S += section_header(5, "ML surrogate & explainability", "XGBoost on Foster-labelled encounters — correlation, PCA, SHAP")
    S += what_how_why(
        f"A gradient-boosted regressor of log<sub>10</sub> Pc from 7 encounter features, trained on "
        f"{card.get('training_samples', '?'):,} simulated encounters labelled by numerically integrated Foster Pc.",
        f"Seeded generator (seed {(card.get('recipe') or {}).get('seed', '?')}), 70/10/20 split with early stopping; "
        "explainability computed at training time and served by GET /api/analytics/model.",
        "It beats a miss-distance-only baseline by an order of magnitude, is evaluated on a held-out split, "
        "and is only advisory: it never writes probability_of_collision.")
    S += figure("07_corr")
    corr_pairs = (an.get("correlation") or {}).get("strongest_feature_pairs_spearman", [])[:2]
    wt = (an.get("correlation") or {}).get("with_target", {}).get("spearman", {})
    if corr_pairs and wt:
        top_t = min(wt.items(), key=lambda kv: kv[1])
        S.append(P("<b>Reading it:</b> " + "; ".join(
            f"{rl(an['feature_labels'].get(p['a'], p['a']))} ~ {rl(an['feature_labels'].get(p['b'], p['b']))} ρ = {p['r']:.2f}"
            for p in corr_pairs) + f". Strongest link to the target: {rl(an['feature_labels'].get(top_t[0], top_t[0]))} "
            f"(Spearman {top_t[1]:.2f}) — older TLEs mean bigger ellipses and lower Pc.", "small"))
    S += figure("08_pca")
    comps = (an.get("pca") or {}).get("components", [])[:3]
    if comps:
        S += bullets([rl(c.get("interpretation", "")) for c in comps])
    S.append(PageBreak())
    S += figure("09_pca_vs_raw")
    S += callout("Verdict on PCA (measured, not assumed)", [rl((an.get("pca_vs_raw") or {}).get("verdict", "n/a"))])
    S += figure("10_shap")
    S += figure("11_waterfall")
    S.append(PageBreak())
    S += figure("12_heldout")
    mrows = [
        ["Model", rl(card.get("model", "?")), "Task", rl(card.get("task", "?"))],
        ["Train / val / held-out", f"{card.get('training_samples', 0):,} / {card.get('validation_samples', 0):,} / {card.get('heldout_samples', 0):,}",
         "Boosting rounds", str(card.get("boosting_rounds", "?"))],
        ["Held-out MAE / RMSE", f"{reg.get('mae_log10_pc')} / {reg.get('rmse_log10_pc')} dex",
         "MAE with masked inputs", f"{((card.get('heldout_masked_features') or {}).get('regression') or {}).get('mae_log10_pc')} dex"],
        ["P / R / F1 @ 1e-4", f"{c14.get('precision')} / {c14.get('recall')} / {c14.get('f1')}",
         "P / R / F1 @ 1e-6", "{precision} / {recall} / {f1}".format(**(held.get("classification_at_1e-6") or {"precision": "?", "recall": "?", "f1": "?"}))],
        ["Baseline (miss only) MAE", f"{base_mae} dex", "Constant baseline MAE",
         f"{(((card.get('baselines') or {}).get('constant_mean') or {}).get('regression') or {}).get('mae_log10_pc')} dex"],
        ["Features", rl(", ".join(card.get("features", []))), "Not seen by model", rl(", ".join(card.get("features_not_seen_by_model", [])))],
        ["Label", rl(card.get("label_source", "")), "Trained", f"{rl(str(card.get('trained_at_utc', ''))[:16])} · xgboost {rl(card.get('xgboost_version', ''))}"],
    ]
    S.append(P("Model card", "h2"))
    t = Table([[P(f"<b>{r[0]}</b>", "cell"), P(r[1], "cell"), P(f"<b>{r[2]}</b>", "cell"), P(r[3], "cell")] for r in mrows],
              colWidths=[TW * 0.18, TW * 0.32, TW * 0.18, TW * 0.32])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), C(SURF)), ("BACKGROUND", (2, 0), (2, -1), C(SURF)),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.3, C(GRID)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    S.append(t)
    S.append(Spacer(1, 6))
    traj = card.get("trajectory_model") or {}
    if traj:
        S.append(P(f"<b>Honesty note:</b> the separate trajectory RNN ({rl(traj.get('architecture', ''))}) scores MAE "
                   f"{traj.get('mae_km', {}).get('model')} km vs {traj.get('mae_km', {}).get('linear_extrapolation')} km for linear "
                   f"extrapolation; it is marked <b>{rl(traj.get('status', ''))}</b> and feeds nothing in the alert path.", "small"))
    S.append(PageBreak())

    # 6 decision
    S += section_header(6, "Decision score", "one well-defined number across physics, ML, cascade and manoeuvre")
    S += what_how_why(
        "A 0–100 score and an action (MANOEUVRE / PREPARE / MONITOR / NONE) with a confidence level and a "
        "generated one-sentence rationale on every alert.",
        "Weighted sum of four normalised components; the action comes from the physics Pc alone, and ML + cascade "
        "can only escalate PREPARE → MANOEUVRE when both agree.",
        "Unit-tested: weights sum to 1, physics outweighs everything else combined, score monotone in Pc, points add "
        "up to the score, thresholds follow NASA CARA practice.")
    S += callout("Decision score (app/core/decision.py)", [
        "L(p) = clip((log<sub>10</sub> p + 7) / 3, 0, 1)   — 0 at 10<super>−7</super>, 1 at 10<super>−4</super>",
        "score = 100 · [0.60 L(Pc<sub>Foster</sub>) + 0.15 L(Pc<sub>ML</sub>) + 0.15 (1 − e<super>−D/5</super>) + 0.10 e<super>−Δv/1 m/s</super> · s<sub>safe</sub>]",
        "D = downstream objects in the cascade graph;  s<sub>safe</sub> = 1 (cascade-safe), 0.5 (not), 0 (no plan)"])
    S += figure("13_decision")
    if scen_alert and scen_alert.get("decision"):
        S += figure("14_decision_breakdown")
        S.append(P(f"<b>Rationale (generated):</b> <i>{rl(scen_alert['decision'].get('rationale', ''))}</i>", "small"))
    S.append(PageBreak())

    # 7 avoidance
    S += section_header(7, "Cascade-safe avoidance", "re-propagated burns, secondary-conjunction check, 12 engines")
    S += what_how_why(
        "For each risky alert: a ranked set of burns (±R/±S/±W × 0.01–2 m/s), each with its new Pc, secondary "
        "conjunctions, propellant and per-engine feasibility.",
        "Each candidate is applied to a copy of the trajectory, the encounter re-screened and Pc recomputed with the "
        "production compute_pc; shortlisted options are re-screened for 24 h against the whole catalogue + live debris.",
        "The chosen burn is the smallest Δv that is cascade-safe; tests show the naive min-Δv burn can hit a third "
        "object and is rejected. CW analytic drift and RK4 mass integration verify each option.")
    S += callout("Checks attached to every option", [
        "Clohessy–Wiltshire: y(t) = (2ẋ<sub>0</sub>/n)(cos nt − 1) + (ẏ<sub>0</sub>/n)(4 sin nt − 3nt)",
        "Tsiolkovsky: m<sub>p</sub> = m<sub>0</sub>(1 − e<super>−Δv/(I<sub>sp</sub> g<sub>0</sub>)</super>);   finite_burn_ok ⇔ t<sub>burn</sub> ≤ 0.1 · lead time  and  ≤ 1/10 orbit"])
    S += figure("15_options")
    if rm.get("options"):
        rows = []
        for i, o in enumerate(rm["options"]):
            sec = o.get("secondary_conjunctions") or []
            chosen_mark = " ★" if i == rm.get("chosen_index") else ""
            rows.append([f"<b>#{i + 1}{chosen_mark}</b>", rl(o["candidate"]), f"{o['delta_v_ms']:g}",
                         f"{o.get('new_miss_distance_km', 0):.3f}", fmt(o.get("new_pc_collision")),
                         f"{len(sec)} (max Pc {fmt(o.get('secondary_max_pc'))})",
                         f'<font color="{GOOD if o.get("cascade_safe") else BAD}"><b>{"yes" if o.get("cascade_safe") else "no"}</b></font>',
                         f"{o.get('propellant_kg', 0):.3f}", rl(o.get("recommended_engine", "") or "")])
        S.append(data_table(["#", "Burn", "Δv [m/s]", "New miss [km]", "New Pc", "Secondaries (24 h)", "Cascade-safe",
                             "Prop. [kg]", "Engine"], rows,
                            [TW * 0.06, TW * 0.12, TW * 0.08, TW * 0.10, TW * 0.12, TW * 0.19, TW * 0.10, TW * 0.08, TW * 0.15]))
        S.append(Spacer(1, 6))
    S += figure("16_engines")
    if rm:
        S.append(P(f"Selection rule: <i>{rl(rm.get('selection_rule', ''))}</i> · mover mass {rm.get('mass_kg')} kg "
                   f"({rl(rm.get('mass_source'))}), Isp {rm.get('isp_s')} s ({rl(rm.get('isp_source'))}).", "small"))
    S.append(PageBreak())

    # 8 debris & cascade
    S += section_header(8, "Collision → debris → cascade", "NASA Standard Breakup Model, propagated fragments, graph to agencies")
    if event:
        S += what_how_why(
            f"What happens if the demo pair collides: {event['parent_names'][0]} ({event['parents'][0]['mass_kg']:.0f} kg, "
            f"{event['parents'][0]['mass_source']}) × {event['parent_names'][1]} ({event['parents'][1]['mass_kg']:.0f} kg) at "
            f"{event['rel_vel_kms']:.2f} km/s.",
            "Both objects propagated to the predicted TCA; SBM samples size, area-to-mass and ejection Δv per fragment "
            "(seeded); fragments are integrated with J2 + drag and screened against the catalogue for 6 h.",
            "Fragment count equals 0.1·M<super>0.75</super>·L<sub>c</sub><super>−1.71</super> (live check); momentum is "
            "re-centred, Δv capped below escape, energy drift < 1e-5 without drag (tests/test_debris_real.py).")
        S += callout("NASA SBM (Johnson et al. 2001)", [
            f"EMR = ½ m<sub>p</sub> v<super>2</super> / M<sub>t</sub> = {event['emr_j_per_g']:.0f} J/g ≥ 40 J/g → catastrophic;  "
            f"M = m<sub>1</sub> + m<sub>2</sub> = {event['sbm_reference_mass_kg']:.0f} kg",
            f"N(≥ 10 cm) = 0.1 · M<super>0.75</super> · 0.1<super>−1.71</super> → {event['total_fragments']} fragments;  "
            "log<sub>10</sub> Δv ~ N(0.9χ + 2.9, 0.4),  χ = log<sub>10</sub>(A/M)"])
    S += figure("17_breakup")
    S += figure("18_cloud", width=TW * 0.78)
    S.append(PageBreak())
    S += figure("19_debris_alerts")
    S += figure("20_cascade")
    if event:
        ds = (scen.get("debris") or {}).get("debris_screening") or {}
        S.append(P(f"Debris screening: {ds.get('fragments_screened')} fragments × {ds.get('satellites_screened')} "
                   f"satellites, {ds.get('window_hours'):g} h at {ds.get('step_s'):g} s steps → {ds.get('alerts_returned')} alerts in "
                   f"{ds.get('elapsed_s'):.1f} s. Cascade depth is the BFS hop count from the collision event; P(hit) "
                   "combines all incident alerts assuming independence. This is a graph over predicted encounters in the "
                   "screening window, not a multi-year population model.", "small"))
    S.append(PageBreak())

    # 9 validation
    S += section_header(9, "Physics cross-validation", f"GET /api/physics/validation — {n_pass}/{n_tot} checks pass (computed live)")
    S += what_how_why(
        "A battery of standard-formula checks that runs inside the live backend (cached 60 s), so judges can hit "
        "the endpoint and see the same table.",
        "Each check recomputes a quantity with an independent formulation (closed form, series, sampling or "
        "numerical integration) and compares it to our production code against a stated tolerance.",
        "Methods come from the literature (Foster 1992, Chan 2008, Alfano 2005, Johnson 2001, Vallado); the "
        "tolerance kind (relative / absolute / inequality) is reported per check.")
    vrows = []
    for c in val.get("checks", []):
        kind = c.get("tolerance_kind", "")
        if kind == "absolute":
            err = f"abs {fmt(c.get('abs_error'))}"
            tol = f"≤ {fmt(c.get('tolerance'))}"
        elif kind == "relative":
            err = f"rel {fmt(c.get('rel_error'))}"
            tol = f"≤ {fmt(c.get('tolerance'))}"
        else:
            err = f"Δ {fmt(c.get('abs_error'))}"
            tol = rl(kind)
        u = c.get("units") or ""
        vrows.append([f"<b>{rl(c['name'])}</b><br/><font color='{MUTED}' size='5.6'>{rl(c.get('standard_formula', ''))[:170]}</font>",
                      f"{fmt(c.get('our_value'), 6)} {rl(u)}", f"{fmt(c.get('reference_value'), 6)} {rl(u)}", err, tol,
                      badge(c.get("pass"))])
    S.append(data_table(["Check (standard formula)", "Ours", "Reference", "Error", "Tolerance", ""], vrows,
                        [TW * 0.47, TW * 0.13, TW * 0.13, TW * 0.11, TW * 0.10, TW * 0.06]))
    S.append(PageBreak())
    S += section_header(10, "Independent verification", "a separate agent re-derived results outside the production code path")
    S += vf
    S.append(Spacer(1, 8))
    S.append(P("Test suite (backend/tests)", "h2"))
    trows = [
        ["test_screening_real.py", "TCA/miss vs brute-force 1 s + 1 ms scan; far-now-close-later pair; Foster = dblquad; σ grows with age"],
        ["test_debris_real.py", "40 J/g threshold; SBM count and size law; no hyperbolic fragments; momentum; energy drift; Vallado density"],
        ["test_cascade_real.py", "BFS depth; P(hit) = 1 − Π(1 − Pc); re-propagated manoeuvre reaches Pc < 1e-6; SATCAT agencies"],
        ["test_avoidance_real.py", "naive min-Δv burn hits a third object → rejected; CW < 0.5 %; Tsiolkovsky vs RK4 1e-6"],
        ["test_burn_real.py", "zero burn moves nothing; predicted post-burn miss = executed (< 10 m)"],
        ["test_ml_real.py / test_ml_explain.py", "beats baseline; MAE < 0.4 dex on fresh encounters; SHAP additivity; never touches physics Pc"],
        ["test_decision_real.py", "monotone score; weights; thresholds; Chan ≈ Foster; MC within 4σ; validation endpoint all-pass"],
    ]
    S.append(data_table(["File", "What it proves"], trows, [TW * 0.28, TW * 0.72]))
    S.append(PageBreak())

    # 11 limitations
    S += section_header(11, "Assumptions, limitations & how to verify", "what we model, what we assume, what we do not claim")
    lim = [
        "<b>No measured covariance.</b> TLEs have none; σ grows linearly with TLE age (σ<sub>0</sub> Flohrer 2008, growth "
        "Vallado &amp; Cefola 2012). With the bundled snapshot (epochs April–May 2026) TLEs are months old at the "
        "simulation date, the ellipse becomes huge and Pc collapses towards tiny values (dilution) — such alerts are "
        "flagged stale_tle. Live Space-Track/CDN TLEs give meaningful Pc.",
        "<b>Masses are assumed.</b> SATCAT has no masses: scenario config → RCS class (50/500/2000 kg) → type default "
        "→ 500 kg, tagged mass_source. The fragment count scales with M<super>0.75</super>.",
        "<b>Frames.</b> SGP4 output is TEME and is used as ECI (TEME ≈ ECI, sub-km at these timescales); displayed "
        "altitude is |r| − 6371 km (spherical Earth).",
        "<b>Breakup is one Monte-Carlo draw</b> (seed recorded per event), with an exponential atmosphere and no "
        "solar-activity-dependent density; at most 1000 fragments are propagated with weights.",
        "<b>Manoeuvres are impulsive.</b> Finite-burn validity is reported, not simulated; simultaneous burns of "
        "different alerts are not screened against each other; cascade check propagates fragments without drag.",
        f"<b>ML is synthetic-trained.</b> Labels are physics Pc on generated geometry, not real CDMs; MAE degrades to "
        f"{((card.get('heldout_masked_features') or {}).get('regression') or {}).get('mae_log10_pc')} dex when radial miss / altitude are missing. "
        "The trajectory RNN underperforms linear extrapolation and is marked experimental; GAT/GNN rankers are "
        "advisory only and operator feedback trains nothing.",
        "<b>Cascade ≠ Kessler model.</b> The graph spans predicted encounters inside the screening window, not a "
        "multi-year population evolution. P(hit) assumes independent encounters.",
        "<b>Policy thresholds.</b> Pre-flight gates (> 200 km trajectory clearance, > 60 min TCA window) and the "
        "PREPARE/MONITOR tiers (1e-5, 1e-7) are our policy choices; only 1e-4 comes from NASA CARA / ESA practice.",
    ]
    S += bullets(lim)
    S.append(Spacer(1, 6))
    S.append(P("How to verify in five minutes", "h2"))
    S += callout("Commands", [
        '<font name="DVM" size="7.6">cd backend &amp;&amp; set ORBIT_SENTINEL_SKIP_DOTENV=1 &amp;&amp; python -m pytest -q</font>',
        '<font name="DVM" size="7.6">python -m app.ml.train_risk_surrogate      # reproduces the model + card (seeded)</font>',
        '<font name="DVM" size="7.6">python docs/report/build_report.py         # rebuilds this report from live data</font>'])
    erows = [
        ["GET /api/physics/validation", "all live cross-checks with formula, values, error, tolerance"],
        ["GET /api/analytics/model", "correlation, PCA, PCA-vs-raw, SHAP, model card"],
        ["GET /api/model-metrics", "held-out metrics of every model incl. the weak ones"],
        ["GET /api/alerts", "alerts with pc_checks, ml.contributions, decision, recommended_maneuver.options"],
        ["POST /api/simulate {\"name\":\"cascade_demo\"}", "computed-crossing scenario found by screening"],
        ["POST /api/debris/simulate", "SBM breakup at predicted TCA + fragment alerts"],
        ["GET /api/physics/engines · /api/agencies", "engine catalogue with sources · SATCAT agency attribution"],
    ]
    S.append(data_table(["Endpoint", "What you will see"], erows, [TW * 0.38, TW * 0.62]))
    if LOG:
        S.append(Spacer(1, 6))
        S.append(P("Build notes: " + rl(" | ".join(LOG)), "cap"))

    print("building PDF ...")
    build_pdf(S)
    try:
        from pypdf import PdfReader
        n = len(PdfReader(str(PDF_PATH)).pages)
    except Exception:
        n = "?"
    print(f"wrote {PDF_PATH} ({n} pages) in {time.perf_counter() - t_start:.0f} s")
    if LOG:
        print("notes:\n  " + "\n  ".join(LOG))


if __name__ == "__main__":
    main()
