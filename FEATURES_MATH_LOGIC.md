# Features, Math, and Logic Map

> This file used to describe an earlier version that relied on heuristic
> scores: a distance-band "probability", a sigmoid "XGBoost-like" risk score,
> a hand-written graph scorer for cascade depth, and a 200 km proximity
> graph. **All of these have been replaced.** The authoritative, judge-facing
> description is **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)**. The
> equation index below points into it.

| Quantity | Model / equation | Code |
|---|---|---|
| Position / velocity | SGP4 (WGS-72, TEME); executed burns as an RK4 two-body+J2 deviation added to SGP4 | `app/core/sgp4_propagator.py` |
| Close approaches | 24 h, 60 s grid → KD-tree per step → linear relative-motion filter → bounded Brent TCA | `app/core/screening.py::screen` |
| Covariance | RTN σ = σ₀ + g·\|TCA − TLE epoch\| (σ₀ = 0.1/0.5/0.15 km, g = 0.1/1.0/0.1 km/day) | `screening.tle_age_sigmas_km` |
| Collision probability | Foster 2-D: ∬ over the HBR disk of N(b, C_B) in the B-plane | `screening.compute_pc`, `foster_pc` |
| Severity | CRITICAL Pc ≥ 1e-4 or miss < 1 km; WARNING Pc ≥ 1e-6 or miss < 5 km | `screening.classify_severity` |
| CPI | hand-weighted triage index (not a probability) | `conjunction.compute_cpi_score` |
| ML surrogate | XGBoost regression of log10 Foster Pc, held-out metrics | `app/ml/risk_api.py`, `train_risk_surrogate.py` |
| Breakup | NASA SBM: 40 J/g, N(≥Lc) = 0.1·M^0.75·Lc^-1.71, A/M and Δv distributions | `app/core/breakup.py` |
| Fragment motion | two-body + J2 + exponential-atmosphere drag (RK4) | `breakup.propagate` |
| Cascade depth | BFS hops from the collision event over the alert graph | `app/services/cascade_planner.py` |
| P(object hit) | 1 − Π(1 − Pc_i) | `cascade_planner.node_hit_probabilities` |
| Manoeuvre | candidate RSW burns, re-propagated, Pc recomputed, smallest Δv with Pc < 1e-6 | `app/services/maneuver_planner.py` |
| Fuel | Tsiolkovsky: m₀(1 − e^(−Δv/(Isp·g₀))) | `maneuver_planner.rocket_equation_cost` |
| Hohmann orbit change | Δv₁ = \|v_t1 − v_c1\|, Δv₂ = \|v_c2 − v_t2\| | `routes.orbit_change` |
| Owner / agency | CelesTrak SATCAT owner + operator name refinement | `app/core/satcat.py`, `agency.py` |
