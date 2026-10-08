# How Orbital Sentinel works (for reviewers)

Every number shown in the UI comes from a model you can find in the code and
reproduce. Where an input is not known (a satellite's mass, for example), the
value is an assumption. It is stored as a named constant, and the output it
feeds is tagged with where it came from (`mass_source`, `sigma_source`,
`hbr_source`, `isp_source`, `inclination_source`, ...). This page lists, for
each stage, the model, where it lives in the code, its parameters, and what it
cannot do.

All physics runs on the **simulation clock** (`app/core/sim_clock.simulation_now()`).
That clock is wall-clock UTC plus an operator offset of up to ±24 h. Wall-clock
time is only used for session expiry, log stamps and model-card timestamps.

## Pipeline

```
TLEs (Space-Track | CDN | bundled snapshot)            app/data/tle_fetcher.py
  │
  ▼  SGP4 (WGS-72, TEME)                               app/core/sgp4_propagator.py
  │  + executed burns as propagated J2 deviation        (BurnedSatrec)
  ▼
24 h future screening: SatrecArray grid → KD-tree per step
  → linear relative-motion filter → bounded Brent TCA   app/core/screening.py::screen
  ▼
Foster 2-D Pc with TLE-age covariance                  screening.compute_pc / foster_pc
  ├─► ML surrogate: XGBoost log10(Pc) (advisory)       app/ml/risk_api.py::score_alert(s)
  ▼
predicted collision (highest-Pc pair or scenario)      POST /api/debris/simulate
  ▼  NASA Standard Breakup Model                        app/core/breakup.py::simulate_breakup
  ▼  fragment propagation: two-body + J2 + drag (RK4)   breakup.propagate
  ▼  fragment-vs-catalogue screening (KD-tree + Pc)     app/core/debris_model.py::compute_debris_alerts
  ▼
cascade graph from alerts, BFS depth, P(hit)           app/services/cascade_planner.py
  ▼
manoeuvre: candidate burns → re-propagate → new Pc     app/services/maneuver_planner.py::plan_maneuvers
  │        fuel by rocket equation
  ▼
agency / commanding authority from CelesTrak SATCAT    app/core/satcat.py, agency.py, agency_authority.py
```

Orchestration: `backend/main.py::refresh_alerts_once` runs every
`ALERT_REFRESH_SECONDS` (default 30 s) and after every state change (burn,
override, clock shift) through `routes._recompute_alerts_pipeline`. The
sequence is screening, then debris alerts, then ML scores, then cascade and
manoeuvres, then publish.

---

## 1. Orbit propagation: SGP4

* **Model:** SGP4/SDP4 (`sgp4` library, WGS-72) from two-line element sets.
  Positions are in the TEME frame, in km and km/s.
* **Code:** `SGP4Propagator.propagate_all/propagate_one`. For vectorised work,
  `screening._Window` uses `sgp4.api.SatrecArray`.
* **Executed burns:** the TLE is *not* re-fitted. Instead,
  `r(t) = r_SGP4(t) + [r_J2(t; x0+Δv) − r_J2(t; x0)]`, with an RK4 two-body + J2
  integrator (`BurnedSatrec`). A zero burn changes nothing, and the planner and
  the executed burn use the same model.
* **TLE source**, in order: Space-Track (if `SPACETRACK_USER/PASS` are set),
  then a public CDN mirror, then the bundled snapshot `app/simulation/tle-data.txt`
  (31 068 objects, epochs **2026-04-06 to 2026-05-11**). `MAX_SATS` (default 500)
  takes the first N objects in the file.
* **Limitations:** SGP4 accuracy is about 1 km at epoch and grows by roughly
  1–3 km/day. With the bundled snapshot the TLEs are months old by the
  simulation date, so alerts are flagged `stale_tle: true` (age > 30 days) and
  Pc is dominated by covariance dilution. Use live TLEs for meaningful Pc
  values. Displayed altitude is |r| − 6371 km (spherical Earth).

## 2. Conjunction screening: future window, not "distance now"

* **Code:** `app/core/screening.py::screen(states, sim_time, window_hours=24, step_s=60, threshold_km=25)`.
* **Algorithm:**
  1. Propagate every object on a 60 s grid over 24 h.
  2. Drop objects whose radial shell (perigee/apogee ± slack) overlaps no
     other object's shell.
  3. At each step, build a `cKDTree` and run `query_pairs` with radius
     `threshold + v_rel,max·step/2`.
  4. Apply a linear relative-motion filter: for `t* = −(Δr·Δv)/|Δv|²` inside
     the step and an estimated miss below threshold + 5 km.
  5. Refine with bounded Brent (`scipy.optimize.minimize_scalar`, xatol 1 ms)
     on the exact SGP4 distance.
* **Outputs per pair:** TCA (sim time), miss distance, relative speed,
  B-plane miss (`b_t_km`, `b_n_km`), TLE ages, Pc, severity, and the
  screening parameters used.
* **Severity:** CRITICAL if Pc ≥ 1e-4 or miss < 1 km. WARNING if Pc ≥ 1e-6 or
  miss < 5 km. Otherwise WATCH.
* **CPI (0–10)** (`conjunction.compute_cpi_score`) is a *hand-weighted triage
  index* (0.4 miss, 0.3 log Pc, 0.15 time-to-TCA, 0.1 speed, 0.05 TLE age).
  It is not a probability and is not fitted to data. Pc is the physics result.

## 3. Collision probability: Foster 2-D with TLE-age covariance

* **Equation:** `Pc = ∬_{|x|≤HBR} N(x; b, C_B) dx`. Here `b` is the miss vector
  in the encounter (B-)plane ⟂ Δv, and `C_B = B (C₁ + C₂) Bᵀ`. The integral uses
  48-point Gauss–Legendre radially and a 96-point trapezoid in angle
  (`screening.foster_pc`). It is cross-checked against adaptive `dblquad`
  (`analytics.foster_integrate_dblquad`).
* **Covariance (`sigma_source: "tle_age_model"`):** per object, RTN 1σ is
  `σ = σ₀ + g·|TCA − TLE epoch|`, with σ₀ = (0.10, 0.50, 0.15) km and
  g = (0.10, 1.00, 0.10) km/day. Sources: Flohrer et al. 2008 (TLE accuracy)
  and Vallado & Cefola 2012 (~1 km/day along-track growth).
* **Hard-body radius (`hbr_source`):** SATCAT RCS class (SMALL 0.18 m,
  MEDIUM 0.56 m), else type defaults (PAY 5 m, R/B 4 m, DEB 0.5 m, UNK 1 m),
  plus named large structures (ISS 55 m). HBR is the sum of the two radii.
* **Limitations:** TLEs carry no covariance, so the covariance is *modelled*,
  not measured. The short-encounter assumption is flagged by
  `short_encounter_valid` (|Δv| ≥ 0.1 km/s). Covariance is Gaussian and
  diagonal in RTN.

## 4. ML: XGBoost Pc surrogate (advisory only)

* **What it is:** an XGBoost regressor of log10(Foster Pc)
  (`app/ml/train_risk_surrogate.py`, artifact `app/ml/artifacts/risk_model_xgb.json`,
  card `risk_model_card.json`). Its labels are Foster Pc numerically integrated
  on 60 000 simulated encounters. The encounters use seeded, documented
  sampling (seed 20261008) and a 70/10/20 train/val/**held-out** split.
* **Features (7):** miss distance, |radial miss|, relative speed, max/min
  TLE age, HBR, altitude. The model **does not see** the covariance or the
  B-plane geometry, which is why it is a surrogate.
* **Held-out results:**
  * MAE 0.106 dex in log10 Pc.
  * At the 1e-4 threshold: precision 0.975, recall 0.950.
  * Baseline that only knows miss distance: MAE 1.57 dex, recall 0.
* **Use:** `alert.ml = {pc_surrogate, risk_class, model, agreement}`, where
  `agreement` is |Δlog10| against the physics Pc. **The model never writes
  `probability_of_collision`.** Scoring takes < 1 ms per alert and does not
  import torch.
* **Limitations:** the training distribution is synthetic encounter geometry,
  not real CDMs. Accuracy drops when `radial_miss_km`/`altitude` are missing
  (masked-feature MAE is 0.34 dex). The model card reports both figures.
* **Other models, stated honestly:**
  * The trajectory RNN (`lstm_predictor.py`, a 12-unit tanh RNN, *not an
    LSTM*) **underperforms linear extrapolation**: MAE 459 km vs 20.5 km. It is
    marked `experimental` and feeds nothing in the alert path.
  * The GAT/GNN cascade rankers are trained on synthetic graphs and only
    appear in `ranker_review` as an advisory cross-check (including Spearman ρ
    against the physics P(hit)). They never set depth, probabilities or the plan.
  * `/api/anomalies` flags residuals of that trajectory model against SGP4. It
    checks model consistency; it does not detect real manoeuvres.
  * Operator decisions (`rlhf_store`) are only *counted*. They train no model
    (`operator_feedback.trains_any_model: false`).

## 5. Collision to debris: NASA Standard Breakup Model

* **Code:** `app/core/breakup.py::simulate_breakup`, triggered by
  `debris_model.simulate_collision_from_pair`. The pair is either the loaded
  scenario pair or the highest-Pc screened alert. Both objects are propagated
  to the *predicted TCA* (`find_pair_tca`).
* **Equations** (Johnson et al. 2001; Krisko 2011):
  * Energy-to-mass ratio: EMR = ½ m_p v² / M_t. The collision is catastrophic
    if EMR ≥ 40 J/g.
  * Reference mass: M = m₁+m₂ if catastrophic, else m_p·v²[km/s].
  * Cumulative fragment count: `N(≥Lc) = 0.1 M^0.75 Lc^-1.71`, Lc_min = 10 cm.
  * χ = log10(A/M) is drawn from the bimodal spacecraft/rocket-body tables.
  * Ejection speed: log10 Δv ~ N(0.9χ+2.9, 0.4), in isotropic directions.
  * Momentum is re-centred per parent. Δv is capped below escape speed.
* **Fragment propagation:** RK4 two-body + J2 + drag, with ballistic
  coefficient C_d·A/M (C_d = 2.2) and the Vallado exponential atmosphere
  (Table 8-4). Fragments are removed below 100 km. At most 1 000 sampled
  fragments are propagated, each carrying weight `N_total/N_sampled`.
* **Fragment-vs-catalogue screening:** KD-tree over a 6 h window at a 30 s
  step with a 5 km threshold. Pc uses an isotropic Foster (non-central χ²).
  Satellite σ is 0.2 km; fragment σ grows by 1 m/s × t. The resulting alerts
  have `source: "debris"`, a `parent_event`, and the fragment state at the
  alert epoch.
* **Masses (`mass_source`):** scenario config, then SATCAT RCS class (S/M/L
  → 50/500/2000 kg), then object type (PAY 800, R/B 1500, DEB 20 kg), then a
  default of 500 kg. **SATCAT has no masses**, so these are assumptions, and
  they set the fragment count through M^0.75.
* **Limitations:**
  * Fragment sampling is seeded random (the seed is recorded per event); it is
    a Monte-Carlo draw, not a unique answer.
  * There is no solar-activity-dependent density.
  * Debris CPI uses a log-Pc scale (`_cpi_from_pc`) that differs from the
    screening CPI.

## 6. Cascade graph

* **Code:** `cascade_planner.build_alert_graph`, `bfs_cascade`, `annotate_alerts`, `node_hit_probabilities`.
* **Graph:** nodes are objects and collision events. Edges are alerts:
  screening pairs plus debris links event → fragment → satellite.
* **Depth:** `cascade_depth` is the number of BFS hops from the root collision
  event (fragment 1, threatened satellite 2, that satellite's own
  conjunctions 3, ...). An alert with no event upstream has depth 1.
* **Node probability:** `P(hit) = 1 − Π(1 − Pc_i)` over the node's incident
  alerts, assuming the encounters are independent.
* **Limitation:** this is a graph over *predicted* encounters inside the
  screening window. It is not a long-term (years) Kessler population model.

## 7. Manoeuvre planning: verified by re-propagation

* **Code:** `maneuver_planner.plan_maneuvers`. It runs for the top 15 alerts
  with Pc ≥ 1e-6, with a 3 s budget per refresh.
* **Search:** candidates are ±R/±S/±W × {0.01 … 2 m/s}, applied at sim-now.
  Each burn is applied to a copy of the trajectory (same deviation model as
  §1). The encounter is re-screened (±15 min, 2 s sampling then TCA
  refinement) and Pc is **recomputed with `screening.compute_pc`**. The planner
  picks the smallest |Δv| with Pc < 1e-6, or else the lowest-Pc candidate
  (`achieved_target: false`).
* **Fuel:** Tsiolkovsky, `m_prop = m₀(1 − e^{−Δv/(Isp·g₀)})`, reported as a
  percentage of the assumed propellant load. Mass and Isp come from name class
  (ISS, Starlink, OneWeb, Iridium, each tagged), else the SATCAT RCS class,
  else a default of 500 kg / 220 s / 10 % propellant. The pre-flight
  `fuel_budget` gate (`sim_engine._fuel_check`) uses the same model.
* **Limitations:**
  * Only the alert's own pair is re-screened (`rescreen_scope: "pair"`), so
    secondary conjunctions caused by the burn are not checked by the planner.
    The next full refresh catches them.
  * Burns are impulsive.
  * The pre-flight gates `trajectory_clear` (> 200 km), `tca_window`
    (> 60 min) and `physical_limits` are **policy thresholds**, listed in
    `gate_policy`.

## 8. Agencies and commanding authority

* **Owner:** the CelesTrak SATCAT snapshot `app/data/satcat_snapshot.csv`
  (`satcat.lookup`) provides the owner code, object type, RCS and launch date.
  If an id is not in SATCAT, the record is TLE-derived with `owner: None`.
* **Operator refinement:** name patterns refine the owner (US-owned
  `STARLINK-*` → SpaceX). `agency_attribution()` returns `agency_source`.
* **Authority:** a session may command only objects whose controlling agency
  it represents. Debris, rocket bodies, unknown owners and synthetic objects
  are not commandable by anyone. `DEMO_SESSION` (allowed when
  `ALLOW_UNAUTHENTICATED_COMMANDS=1`) is tagged `rule: "demo_session"` in
  decisions.

## 9. Telemetry

There is no spacecraft downlink. Payloads therefore get a block flagged
`simulated: true`:

* fuel from the rocket equation on burns actually executed (tank assumed
  full at session start);
* sunlit/eclipse from geometry (low-precision solar ephemeris, cylindrical
  shadow).

Battery, temperature, signal strength and solar power are `null` ("not
modeled"). Debris and rocket bodies return no telemetry.

---

## How to verify

```bash
cd backend
set ORBIT_SENTINEL_SKIP_DOTENV=1        # PowerShell: $env:ORBIT_SENTINEL_SKIP_DOTENV=1
python -m pytest -q                      # full suite
```

| Test file | What it proves |
|---|---|
| `tests/test_screening_real.py` | Screening TCA/miss agree with an independent brute-force 1 s + 1 ms scan (within 1 s / 0.1 km). A pair **> 2000 km apart now** that meets in 2 h is found at the right TCA. Foster quadrature equals `dblquad` (rel 1e-4). σ grows with TLE age. Severity rule. HBR by object type. `find_tca` matches truth. |
| `tests/test_debris_real.py` | 40 J/g catastrophic threshold. Fragment count equals `0.1·M^0.75·Lc^-1.71`. Size distribution matches. No hyperbolic fragments; momentum conserved. Energy drift < 1e-5 without drag. Density table matches Vallado. A fragment placed on a satellite's path produces a `source: "debris"` alert with the right TCA/parent event. The fragment state reproduces the TCA. |
| `tests/test_cascade_real.py` | Depth is BFS hops on event → fragment → satellite → neighbour. P(hit) = 1 − Π(1 − Pc). The recommended manoeuvre lowers the **re-propagated** Pc below 1e-6 with ≤ 2 m/s and rocket-equation fuel. Along-track drift ≈ 3·Δv·t. SATCAT lookup (ISS, Vanguard). Agency from SATCAT owner first; < 5 % unknown owners in the bundled catalogue. Unknown owners are not commandable. |
| `tests/test_burn_real.py` | A zero burn moves nothing (< 1 m). An along-track burn matches the Clohessy–Wiltshire drift. Burns survive cloning and the screening grid. The planner's predicted post-burn miss equals what screening finds after execution (< 10 m, TCA < 1 s). |
| `tests/test_predictor_real.py` | `/api/predict/conjunction` values (TCA, Pc, HBR, ellipse, CPI, SBM fragment count, mass sources) equal independent recomputation. There are no `affection_rate`/`predicted_fragments` leftovers. Epoch fallback is the sim clock. A fragment alert gets a re-propagated manoeuvre with drag. |
| `tests/test_ml_real.py` | The model and card load (≥ 10 000 training samples, Foster labels). The model beats the baseline on held-out data. On fresh encounters, MAE < 0.4 dex and precision/recall > 0.9 at 1e-4. `score_alert` never touches the physics Pc and is monotone in miss distance. Scoring takes < 1 ms per alert and does not import torch. The metrics endpoint reports the card, not a formula. |
| `tests/test_telemetry_real.py` | Non-payloads have no telemetry. Payload fuel follows the rocket equation. Illumination geometry is correct. |

Reproduce the ML model with `python -m app.ml.train_risk_surrogate`. It is
seeded, and it writes the artifact and the model card.

## Known gaps

* There is no observation data: covariance is modelled from TLE age, not
  taken from CDMs.
* Masses are assumed (SATCAT has none). The planner and the breakup model use
  slightly different default mass tables, and each output is tagged with the
  table it used.
* The bundled TLE snapshot is old relative to the simulation date (see §1).
* The trajectory RNN and the graph rankers are experimental or advisory
  (see §4).
