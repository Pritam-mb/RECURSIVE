"""
Multi-model decision score for a conjunction alert.

Orbit Sentinel runs several models on every conjunction: the physics Pc
(Foster 2-D, cross-checked by Chan's series / Alfano max-Pc / Monte Carlo in
app.core.pc_methods), an ML surrogate of the Pc (alert["ml"]), the cascade
graph (alert["downstream_ids"]), and the avoidance planner
(alert["recommended_maneuver"]).  This module combines them into ONE
well-defined, documented, monotonic score and an operational action.

1. Normalisations (each n_i in [0, 1])
--------------------------------------
    L(p)        = clip( (log10 p - log10 1e-7) / (log10 1e-4 - log10 1e-7), 0, 1 )
                  -> 0 at the MONITOR threshold (1e-7), 1 at the MANOEUVRE threshold (1e-4)
    n_physics   = L(Pc_Foster)                       (authoritative physics Pc)
    n_ml        = L(Pc_surrogate)                    (ML surrogate; omitted if no ML)
    n_cascade   = 1 - exp(-D / D0),  D = len(downstream_ids), D0 = 5
                  (consequence: how many objects sit downstream in the conjunction
                   graph if this pair collides; 5 downstream objects -> 0.63)
    n_manoeuvre = exp(-dv / DV0) * (1 if cascade_safe else 0.5),  DV0 = 1 m/s
                  (actionability: a cheap, cascade-safe verified manoeuvre exists;
                   0 when the planner produced none; secondary risk halves it)

2. Weights (sum to 1, physics dominant)
---------------------------------------
    w_physics = 0.60, w_ml = 0.15, w_cascade = 0.15, w_manoeuvre = 0.10
    If the ML surrogate is unavailable its weight is moved to physics (the
    weights still sum to 1; nothing is imputed).

3. Score
--------
    score = 100 * Sum_i w_i * n_i        in [0, 100]
    Each component reports raw, normalized, weight and points = 100 * w_i * n_i.
    The score is monotonic non-decreasing in Pc_Foster (only n_physics depends
    on it, and L is non-decreasing); unit-tested in tests/test_decision_real.py.

4. Action (physics Pc is authoritative)
---------------------------------------
    Pc >= 1e-4            MANOEUVRE   (NASA CARA / ESA SDO typical manoeuvre threshold)
    1e-5 <= Pc < 1e-4     PREPARE     (plan and screen a manoeuvre one decade early)
    1e-7 <= Pc < 1e-5     MONITOR     (keep tracking; request fresh orbit data)
    Pc < 1e-7             NONE
    Escalation rule: PREPARE is raised to MANOEUVRE only when the ML surrogate
    AND the cascade model agree - surrogate Pc >= 1e-4 AND n_cascade >= 0.5
    (D >= 4 downstream objects).  No model can LOWER the physics action.

    Sources: NASA CARA, "NASA Spacecraft Conjunction Assessment and Collision
    Avoidance Best Practices Handbook", NASA/SP-20205011318 (2020) - Pc 1e-4 is
    the common risk-mitigation threshold; ESA Space Debris Office practice
    (Merz et al., "Current collision avoidance service by ESA's Space Debris
    Office", 7th European Conf. on Space Debris, 2017) - manoeuvres typically
    planned for Pc in the 1e-4 range.  The 1e-5 / 1e-7 tiers are our
    planning/monitoring bands one and three decades below that threshold.

5. Confidence (model agreement)
-------------------------------
    physics_methods_spread = max - min of log10 Pc over Foster, Chan (and
                             Monte Carlo when resolvable)  [decades]
    physics_vs_ml          = |log10 Pc_Foster - log10 Pc_surrogate|  [decades]
    (Pc floored at 1e-20 for logs.)
    high   : spread <= 0.5 and physics_vs_ml <= 1.0 and short-encounter valid
    medium : spread <= 1.0 and (physics_vs_ml <= 2.0 or ML absent)
    low    : otherwise (or Foster's short-encounter assumption violated)
"""

from __future__ import annotations

import math
from typing import Any

PC_MANOEUVRE = 1e-4
PC_PREPARE = 1e-5
PC_MONITOR = 1e-7
THRESHOLD_SOURCE = (
    "NASA CARA Best Practices Handbook (NASA/SP-20205011318, 2020): Pc 1e-4 manoeuvre threshold; "
    "ESA SDO practice (Merz et al. 2017). PREPARE 1e-5 / MONITOR 1e-7 are planning/monitoring tiers below it."
)

WEIGHTS = {"physics": 0.60, "ml": 0.15, "cascade": 0.15, "manoeuvre": 0.10}
CASCADE_D0 = 5.0
MANOEUVRE_DV0_MS = 1.0
ESCALATE_CASCADE_MIN = 0.5
LOG_FLOOR = 1e-20

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-12


def log_pc_norm(p: float | None) -> float:
    """L(p): 0 at 1e-7, 1 at 1e-4, linear in log10 p, clipped."""
    if p is None or not (p > 0):
        return 0.0
    lo, hi = math.log10(PC_MONITOR), math.log10(PC_MANOEUVRE)
    return min(1.0, max(0.0, (math.log10(p) - lo) / (hi - lo)))


def physics_action(pc: float) -> str:
    if pc >= PC_MANOEUVRE:
        return "MANOEUVRE"
    if pc >= PC_PREPARE:
        return "PREPARE"
    if pc >= PC_MONITOR:
        return "MONITOR"
    return "NONE"


def _lg(p: float) -> float:
    return math.log10(max(float(p), LOG_FLOOR))


def _f(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def decide(alert: dict, pc_checks: dict | None = None) -> dict:
    """CONTRACT2 ``decision`` block for one alert (see module docstring)."""
    pc = _f(alert.get("probability_of_collision", alert.get("p_collision"))) or 0.0
    pc_checks = pc_checks if pc_checks is not None else alert.get("pc_checks")

    ml = alert.get("ml") or {}
    pc_ml = _f(ml.get("pc_surrogate")) if isinstance(ml, dict) else None

    downstream = alert.get("downstream_ids") or []
    D = len(downstream) if isinstance(downstream, (list, tuple, set)) else 0

    man = alert.get("recommended_maneuver") or None
    dv = None
    safe = None
    if isinstance(man, dict):
        opts = man.get("options") or []
        chosen = man.get("chosen_index")
        opt = opts[chosen] if isinstance(chosen, int) and 0 <= chosen < len(opts) else None
        dv = _f((opt or man).get("delta_v_ms"))
        safe = (opt or {}).get("cascade_safe", man.get("cascade_safe"))

    weights = dict(WEIGHTS)
    if pc_ml is None:
        weights["physics"] += weights["ml"]
        weights["ml"] = 0.0

    n_phys = log_pc_norm(pc)
    n_ml = log_pc_norm(pc_ml) if pc_ml is not None else 0.0
    n_casc = 1.0 - math.exp(-D / CASCADE_D0)
    n_man = 0.0 if dv is None else math.exp(-max(dv, 0.0) / MANOEUVRE_DV0_MS) * (0.5 if safe is False else 1.0)

    comps = [
        ("physics_pc_foster", "physics", pc, n_phys, weights["physics"]),
        ("ml_pc_surrogate", "ml", pc_ml, n_ml, weights["ml"]),
        ("cascade_downstream", "cascade", float(D), n_casc, weights["cascade"]),
        ("manoeuvre_actionability", "manoeuvre", dv, n_man, weights["manoeuvre"]),
    ]
    components = [{"name": n, "source": s, "raw": r, "normalized": round(x, 4), "weight": w,
                   "points": round(100.0 * w * x, 3)} for n, s, r, x, w in comps]
    score = round(sum(100.0 * w * x for _, _, _, x, w in comps), 2)

    action = physics_action(pc)
    escalated = False
    if action == "PREPARE" and pc_ml is not None and pc_ml >= PC_MANOEUVRE and n_casc >= ESCALATE_CASCADE_MIN:
        action, escalated = "MANOEUVRE", True

    spread = _f(pc_checks.get("spread_decades")) if isinstance(pc_checks, dict) else None
    d_ml = abs(_lg(pc) - _lg(pc_ml)) if pc_ml is not None else None
    short_ok = alert.get("short_encounter_valid", True) is not False
    s = spread if spread is not None else 0.0
    if short_ok and spread is not None and s <= 0.5 and d_ml is not None and d_ml <= 1.0:
        conf = "high"
    elif short_ok and s <= 1.0 and (d_ml is None or d_ml <= 2.0):
        conf = "medium"
    else:
        conf = "low"

    band = {"MANOEUVRE": f">= {PC_MANOEUVRE:.0e}", "PREPARE": f"[{PC_PREPARE:.0e}, {PC_MANOEUVRE:.0e})",
            "MONITOR": f"[{PC_MONITOR:.0e}, {PC_PREPARE:.0e})", "NONE": f"< {PC_MONITOR:.0e}"}[physics_action(pc)]
    parts = [f"Foster Pc {pc:.2e} is in the {physics_action(pc)} band {band}"]
    if pc_ml is not None:
        parts.append(f"ML surrogate {pc_ml:.2e} ({d_ml:.2f} decades from physics)")
    else:
        parts.append("no ML surrogate (weight moved to physics)")
    if spread is not None:
        parts.append(f"Foster/Chan{'/MC' if (pc_checks or {}).get('monte_carlo') is not None else ''} spread {spread:.2f} decades")
    parts.append(f"{D} downstream object{'s' if D != 1 else ''}")
    if dv is not None:
        parts.append(f"verified manoeuvre {dv:.3f} m/s{'' if safe is not False else ' (not cascade-safe)'}")
    rationale = "; ".join(parts) + (
        f" -> {action}{' (escalated: ML and cascade agree)' if escalated else ''}, "
        f"confidence {conf}, score {score:.0f}/100."
    )

    return {
        "score": score,
        "action": action,
        "physics_action": physics_action(pc),
        "escalated": escalated,
        "components": components,
        "model_agreement": {
            "physics_vs_ml_decades": None if d_ml is None else round(d_ml, 3),
            "physics_methods_spread_decades": spread,
            "confidence": conf,
        },
        "thresholds": {"manoeuvre_pc": PC_MANOEUVRE, "prepare_pc": PC_PREPARE, "monitor_pc": PC_MONITOR,
                       "source": THRESHOLD_SOURCE},
        "formula": "score = 100 * sum(w_i * n_i); n = clip((log10 Pc + 7) / 3, 0, 1) for Pc terms; "
                   "n_cascade = 1 - exp(-D/5); n_manoeuvre = exp(-dv/1 m/s) * (0.5 if not cascade-safe)",
        "rationale": rationale,
    }


def annotate_alerts(alerts: list[dict], *, mc_top_n: int = 10) -> dict:
    """Attach ``pc_checks`` and ``decision`` to every alert in place.

    Cheap methods (Foster re-evaluation, Chan, Alfano max) for all alerts;
    Monte Carlo only for the ``mc_top_n`` highest-Pc alerts to bound time.
    Returns timing / count statistics.
    """
    import time
    from app.core.pc_methods import pc_checks

    t0 = time.perf_counter()
    order = sorted(range(len(alerts)),
                   key=lambda i: -(_f(alerts[i].get("probability_of_collision", alerts[i].get("p_collision"))) or 0.0))
    mc_set = set(order[:mc_top_n])
    n_checks = n_mc = n_incons = 0
    for i, a in enumerate(alerts):
        try:
            chk = pc_checks(a, monte_carlo=i in mc_set)
        except Exception:
            chk = None
        a["pc_checks"] = chk
        if chk is not None:
            n_checks += 1
            n_mc += chk.get("monte_carlo") is not None
            n_incons += not chk.get("consistent", True)
    t1 = time.perf_counter()
    actions: dict[str, int] = {}
    for a in alerts:
        try:
            a["decision"] = decide(a, a.get("pc_checks"))
            actions[a["decision"]["action"]] = actions.get(a["decision"]["action"], 0) + 1
        except Exception:
            a["decision"] = None
    return {"alerts": len(alerts), "pc_checks": n_checks, "monte_carlo_resolved": n_mc,
            "inconsistent": n_incons, "actions": actions,
            "t_pc_checks_s": round(t1 - t0, 3), "t_total_s": round(time.perf_counter() - t0, 3)}
