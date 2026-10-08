"""
Propulsion options for a collision-avoidance burn.

For a required impulsive Δv the module evaluates a catalogue of real engine
classes with the standard rocket-propulsion relations (Sutton & Biblarz,
"Rocket Propulsion Elements", ch. 2-4; Wertz & Larson, "Space Mission Analysis
and Design", §17):

    Tsiolkovsky:      m_p = m0 · (1 − exp(−Δv / (Isp · g0)))
    mass flow:        ṁ   = F / (Isp · g0)
    burn time:        t_b = m_p / ṁ = m0 · Isp · g0 · (1 − exp(−Δv/(Isp·g0))) / F
                            (exact for constant thrust; first order: m0·Δv/F)
    burn arc:         Δθ  = 2π · t_b / T_orbit

Impulsive-burn validity (finite_burn_ok): the planner models the burn as an
instantaneous Δv. That approximation holds when the burn is short compared
with both the orbital period and the time left before the conjunction:

    t_b ≤ FINITE_BURN_LEAD_FRACTION · (t_TCA − t_burn)   and
    t_b ≤ FINITE_BURN_ARC_FRACTION  · T_orbit            (arc ≤ 36°)

(the ~1/10-orbit arc limit is the usual rule of thumb under which finite-burn
losses are < ~1-2 %, e.g. Robbins 1966, "An analytical study of the impulsive
approximation", AIAA J. 4(8)). Electric thrusters routinely violate it: their
avoidance manoeuvres are executed as long low-thrust arcs that must be planned
days ahead, which is what the note says when finite_burn_ok is False.

Engine data are rounded public datasheet values (vacuum Isp, nominal thrust);
each entry names its source. They are *engine classes offered as options*,
not a claim about what a given satellite carries — the satellite's actual
onboard Isp is still the tagged spacecraft_mass_model in maneuver_planner.

Recommendation rule (documented, deterministic):
    among engines with practical_for_satellites AND finite_burn_ok,
    pick the one with the lowest propellant mass (= highest Isp);
    if none qualifies, pick the practical engine with the shortest burn time
    and say so in the reason.
"""

from __future__ import annotations

import math
from typing import Any

G0_MS2 = 9.80665                   # standard gravity (CGPM 1901), used in Isp·g0
FINITE_BURN_LEAD_FRACTION = 0.10   # burn must fit in 10 % of the lead time to TCA
FINITE_BURN_ARC_FRACTION = 0.10    # and span ≤ 1/10 orbit (impulsive approximation)

# ── Engine catalogue ────────────────────────────────────────────────────────
# isp_s: vacuum specific impulse; thrust_n: nominal steady-state thrust.
ENGINES: list[dict[str, Any]] = [
    {
        "engine": "Aerojet Rocketdyne RL10B-2",
        "family": "cryogenic_biprop",
        "propellant": "LOX/LH2",
        "isp_s": 465.5, "thrust_n": 110_100.0,
        "practical_for_satellites": False,
        "source": "Aerojet Rocketdyne RL10 datasheet (RL10B-2: 465.5 s vac, 24,750 lbf); upper-stage engine",
        "note": ("Highest chemical Isp, but LH2 boils off (20 K storage) within days-weeks, so cryogenic "
                 "propulsion is used on upper stages, not on long-lived satellites; thrust is also orders "
                 "of magnitude above what a cm/s avoidance burn can be metered with."),
    },
    {
        "engine": "Aerojet R-4D-11",
        "family": "hypergolic_biprop",
        "propellant": "MMH/NTO",
        "isp_s": 312.0, "thrust_n": 490.0,
        "practical_for_satellites": True,
        "source": "Aerojet Rocketdyne R-4D datasheet (R-4D-11: 490 N, ~312 s); GEO apogee engines / Orion ESM aux",
        "note": "Storable hypergolic biprop: high thrust, years-long storage; typical on GEO buses and large LEO platforms.",
    },
    {
        "engine": "Aerojet AMBR",
        "family": "hypergolic_biprop",
        "propellant": "N2H4/NTO",
        "isp_s": 333.0, "thrust_n": 623.0,
        "practical_for_satellites": True,
        "source": "Aerojet Rocketdyne Advanced Material Bipropellant Rocket (AMBR) datasheet, ~333 s / 623 N",
        "note": "Iridium-rhenium chamber dual-mode biprop: best storable chemical Isp in this list.",
    },
    {
        "engine": "Aerojet MR-106L",
        "family": "hydrazine_monoprop",
        "propellant": "N2H4",
        "isp_s": 235.0, "thrust_n": 22.0,
        "practical_for_satellites": True,
        "source": "Aerojet Rocketdyne MR-106L 22 N datasheet (Isp ~228-235 s)",
        "note": "Workhorse catalytic hydrazine thruster for orbit maintenance and avoidance; toxic propellant.",
    },
    {
        "engine": "Aerojet MR-103G",
        "family": "hydrazine_monoprop",
        "propellant": "N2H4",
        "isp_s": 224.0, "thrust_n": 1.0,
        "practical_for_satellites": True,
        "source": "Aerojet Rocketdyne MR-103G 1 N datasheet (Isp 202-224 s over blowdown)",
        "note": "Small 1 N hydrazine thruster used for attitude/fine Δv; fine impulse resolution for cm/s burns.",
    },
    {
        "engine": "Bradford ECAPS 1N HPGP",
        "family": "green_monoprop",
        "propellant": "LMP-103S (ADN blend)",
        "isp_s": 231.0, "thrust_n": 1.0,
        "practical_for_satellites": True,
        "source": "Bradford ECAPS 1N HPGP datasheet (Isp 204-235 s); flown on PRISMA (2010), SkySat",
        "note": "Low-toxicity ADN monoprop; ~6 % higher Isp and ~24 % higher density-Isp than hydrazine.",
    },
    {
        "engine": "Aerojet GR-1",
        "family": "green_monoprop",
        "propellant": "AF-M315E (ASCENT, HAN blend)",
        "isp_s": 231.0, "thrust_n": 1.0,
        "practical_for_satellites": True,
        "source": "NASA Green Propellant Infusion Mission (GPIM, 2019) GR-1 thruster, ~231 s",
        "note": "HAN-based green monoprop flight-proven on GPIM; needs catalyst-bed preheat before firing.",
    },
    {
        "engine": "Cold-gas N2 thruster",
        "family": "cold_gas",
        "propellant": "N2",
        "isp_s": 65.0, "thrust_n": 1.0,
        "practical_for_satellites": True,
        "source": "Sutton & Biblarz, Rocket Propulsion Elements (N2 cold gas Isp ~60-73 s); e.g. Moog 58-series valves",
        "note": "Simplest, cleanest system (no combustion); very low Isp so large Δv costs a lot of gas.",
    },
    {
        "engine": "Fakel SPT-100",
        "family": "hall_effect_electric",
        "propellant": "Xenon",
        "isp_s": 1600.0, "thrust_n": 0.083,
        "practical_for_satellites": True,
        "power_w": 1350.0,
        "source": "OKB Fakel SPT-100 datasheet (83 mN, ~1600 s at 1.35 kW); >100 GEO satellites",
        "note": "Hall thruster: ~7x less propellant than hydrazine but tens of mN, so avoidance needs long arcs planned ahead.",
    },
    {
        "engine": "Busek BHT-600",
        "family": "hall_effect_electric",
        "propellant": "Xenon",
        "isp_s": 1530.0, "thrust_n": 0.039,
        "practical_for_satellites": True,
        "power_w": 600.0,
        "source": "Busek BHT-600 datasheet (39 mN, ~1530 s at 600 W)",
        "note": "Small-sat Hall thruster (constellation class, cf. Starlink/OneWeb Hall propulsion).",
    },
    {
        "engine": "NASA/L3 NSTAR",
        "family": "gridded_ion_electric",
        "propellant": "Xenon",
        "isp_s": 3100.0, "thrust_n": 0.092,
        "practical_for_satellites": True,
        "power_w": 2300.0,
        "source": "NASA NSTAR ion engine (Deep Space 1, Dawn): 92 mN, 3100 s at 2.3 kW",
        "note": "Gridded ion: highest Isp, lowest propellant, but burn times of hours for m/s-class Δv.",
    },
    {
        "engine": "QinetiQ T6",
        "family": "gridded_ion_electric",
        "propellant": "Xenon",
        "isp_s": 4120.0, "thrust_n": 0.145,
        "practical_for_satellites": True,
        "power_w": 4500.0,
        "source": "QinetiQ T6 gridded ion thruster (BepiColombo): 145 mN, ~4120 s at 4.5 kW",
        "note": "Kaufman gridded ion engine: very high Isp; power-hungry (kW class).",
    },
]


def tsiolkovsky_propellant_kg(m0_kg: float, delta_v_ms: float, isp_s: float) -> float:
    """m_p = m0 (1 − exp(−Δv / (Isp g0)))."""
    return float(m0_kg) * (1.0 - math.exp(-abs(float(delta_v_ms)) / (float(isp_s) * G0_MS2)))


def burn_time_s(m0_kg: float, delta_v_ms: float, isp_s: float, thrust_n: float) -> float:
    """Constant-thrust burn time t = m_p / ṁ with ṁ = F / (Isp g0) (exact for constant F, Isp)."""
    if thrust_n <= 0:
        return float("inf")
    return tsiolkovsky_propellant_kg(m0_kg, delta_v_ms, isp_s) * float(isp_s) * G0_MS2 / float(thrust_n)


def burn_time_first_order_s(m0_kg: float, delta_v_ms: float, thrust_n: float) -> float:
    """First-order burn time t ≈ m0 Δv / F (mass change neglected)."""
    return float(m0_kg) * abs(float(delta_v_ms)) / float(thrust_n)


def integrate_constant_thrust(m0_kg: float, delta_v_ms: float, isp_s: float, thrust_n: float,
                              n_steps: int = 20000) -> dict[str, float]:
    """Numerically integrate dv/dt = F/m, dm/dt = −F/(Isp g0) (RK4) until Δv is reached.

    Independent check of the closed forms above (used by tests and the
    physics-validation endpoint)."""
    mdot = float(thrust_n) / (float(isp_s) * G0_MS2)
    t_end = burn_time_first_order_s(m0_kg, delta_v_ms, thrust_n) * 1.5 + 1e-9
    h = t_end / n_steps
    m, v, t = float(m0_kg), 0.0, 0.0
    target = abs(float(delta_v_ms))

    def f(mm):
        return float(thrust_n) / mm

    while v < target and t < t_end:
        k1 = f(m); k2 = f(m - 0.5 * h * mdot); k3 = k2; k4 = f(m - h * mdot)
        dv = h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if v + dv >= target:   # sub-step to hit the target exactly (dv/dt ≈ F/m locally)
            frac = (target - v) / dv
            t += frac * h; m -= frac * h * mdot; v = target
            break
        v += dv; m -= h * mdot; t += h
    return {"prop_mass_kg": float(m0_kg) - m, "burn_time_s": t, "delta_v_ms": v}


def evaluate_engines(delta_v_ms: float, m0_kg: float, *, lead_time_s: float,
                     orbital_period_s: float) -> dict[str, Any]:
    """Per-engine propellant mass, burn time and impulsive-approximation validity."""
    lead_limit = FINITE_BURN_LEAD_FRACTION * max(float(lead_time_s), 0.0)
    arc_limit = FINITE_BURN_ARC_FRACTION * float(orbital_period_s)
    rows = []
    for e in ENGINES:
        mp = tsiolkovsky_propellant_kg(m0_kg, delta_v_ms, e["isp_s"])
        tb = burn_time_s(m0_kg, delta_v_ms, e["isp_s"], e["thrust_n"])
        ok = bool(tb <= lead_limit and tb <= arc_limit)
        note = e["note"]
        if not ok:
            why = []
            if tb > arc_limit:
                why.append(f"burn {tb:.0f} s spans {360.0 * tb / orbital_period_s:.0f}° of orbit (> {360 * FINITE_BURN_ARC_FRACTION:.0f}°)")
            if tb > lead_limit:
                why.append(f"burn {tb:.0f} s > {100 * FINITE_BURN_LEAD_FRACTION:.0f}% of the {lead_time_s / 3600.0:.1f} h lead time")
            note = f"{note} Impulsive approximation invalid: {'; '.join(why)} — needs a planned finite/low-thrust arc."
        elif tb < 0.05:
            note = f"{note} Burn would last only {tb * 1000:.1f} ms — below typical minimum impulse bit; use a smaller thruster."
        rows.append({
            "engine": e["engine"], "family": e["family"], "isp_s": e["isp_s"], "thrust_n": e["thrust_n"],
            "propellant": e["propellant"],
            "prop_mass_kg": round(mp, 6),
            "burn_time_s": round(tb, 4),
            "burn_arc_deg": round(360.0 * tb / orbital_period_s, 4),
            "finite_burn_ok": ok,
            "practical_for_satellites": e["practical_for_satellites"],
            "note": note,
        })
    eligible = [r for r in rows if r["practical_for_satellites"] and r["finite_burn_ok"]]
    if eligible:
        best = min(eligible, key=lambda r: (r["prop_mass_kg"], r["burn_time_s"]))
        reason = "lowest propellant mass among satellite-practical engines whose burn is short enough for the impulsive model"
    else:
        practical = [r for r in rows if r["practical_for_satellites"]]
        best = min(practical, key=lambda r: r["burn_time_s"])
        reason = "no engine satisfies the impulsive-burn limits; shortest burn among satellite-practical engines"
    return {
        "engines": rows,
        "recommended_engine": best["engine"],
        "engine_rule": reason,
        "finite_burn_limits": {"lead_fraction": FINITE_BURN_LEAD_FRACTION, "arc_fraction_of_orbit": FINITE_BURN_ARC_FRACTION,
                               "lead_time_s": round(float(lead_time_s), 1), "orbital_period_s": round(float(orbital_period_s), 1)},
    }
