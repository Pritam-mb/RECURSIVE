# Independent maths verification

Each mathematical component was recomputed with a **different implementation**:
scipy / sklearn / raw sgp4 / own integrators, written without reusing the
project's code path. The machine-readable log is
[`verification.json`](verification.json). Each record has the fields `id, area,
check, method_ours, method_independent, ours, independent, abs_err, rel_err,
tolerance, pass, note, fixed, fix_summary`. The scripts were run from
`backend/` with `ORBIT_SENTINEL_SKIP_DOTENV=1`.

**Result: 149 checks, 149 pass after fixes. 3 genuine defects were found and
fixed (below).** The full backend suite was re-run after the fixes.

## Defects found and fixed

| id | Defect | Impact | Fix |
|---|---|---|---|
| PROP-2b | `debris_model._satellite_tracks` propagated satellites without a TLE (two-body + J2 RK4) with **60 s** substeps. The error was **207 m after 6 h** against DOP853. | About 1σ of the 0.2 km satellite sigma used in the debris Pc, so a real Pc error for scenario / state-vector satellites. | Substep is now `SAT_TRACK_MAX_STEP_S = 20 s`, which gives 0.45 m on the production 30 s grid. The test is `tests/test_verification_real.py`. |
| ML-CARD-1 | The model-card text `recipe.covariance` described an obsolete covariance (`σr=σn=0.05 km, σt=1·(1+age_h/24) km`). The labels actually use the live TLE-age model σ = (0.1, 0.5, 0.15) km + (0.1, 1.0, 0.1) km/day · age. | Documentation only. Labels and metrics were already correct (re-verified bit-for-bit). | The text is now generated from the screening constants (`_covariance_recipe_text`), and the card JSON was corrected. No retrain. |
| VAL-2 | `/api/physics/validation` reported `rel_error = 3.95e+298` for the zero-reference check "SGP4 vs J2-RK4". The cause was `abs/1e-300`. | A nonsensical number in the API. | `rel_error = null` when the reference is 0, and pass/fail falls back to the absolute error (`routers/physics.py`, `core/analytic_checks.py`). |

## What was verified (highlights)

| Area | Independent method | Worst error |
|---|---|---|
| PCA | Regenerated the 60 000-row set (seed 20261008, same split), then sklearn `SimpleImputer(median)` + `StandardScaler` + `PCA` | EVR 4e-6, loadings \|cos\| ≥ 0.99996, medians 5e-7, n₉₅ = 6 identical |
| Correlation | `scipy.stats.pearsonr` / `spearmanr` per pair, pairwise deletion | 5e-5 (4-dp rounding); row counts exact; symmetric |
| XGBoost | Reloaded the artifact on the regenerated held-out set; sklearn metrics | MAE 0.1056, P/R/F1 at 1e-4 and 1e-6, baselines and PCA(6/7) → XGB all reproduced to ≤ 5e-5 |
| TreeSHAP | `pred_contribs` sum vs `output_margin`; own permutation importance | additivity 7e-6; ranking ρ = 0.96, same top factor (older TLE age) |
| Foster Pc | `scipy.integrate.dblquad`, 6 geometries (offset, rotated, AR 20, HBR > σ, 1e-9 tail) | rel ≤ 1e-14 |
| B-plane / covariance | Own RTN covariances + QR-built encounter basis; 3-D Monte Carlo (4·10⁶) of the combined error | Pc identical (rel 1e-15); MC within 0.5σ |
| Chan | `scipy.stats.ncx2` (isotropic, exact) and dblquad (anisotropic) | exact case 1e-16; approximation 0.2–2 % (26 % for HBR = 2σ_min, as expected) |
| Alfano / MC | Numerical maximisation of dblquad over the covariance scale; own scipy MC (different seed) | Alfano 2e-9 rel; MC within 4σ; MC refuses below its resolution |
| Screening | 60 real catalogue conjunctions: raw sgp4 brute force (1 s, then 1 ms scan) | TCA 1.8 ms, miss 0.05 m |
| Completeness | 150-object shell, all pairs, sgp4 every 10 s with every local minimum refined | 51 / 51 pairs found, 0 missed, 0 extra |
| Frames | Vallado Alg. 15 GMST (Example 3-5); frontend `computeGmst` run in node | 8e-8 deg; sign convention correct |
| CW / Hohmann / vis-viva | HCW ODE by DOP853; nonlinear two-body DOP853; vis-viva derivation | CW vs nonlinear 0.005 %; Hohmann 1e-15; transfer lands on r₂ (3e-10 km), e after Δv₂ 4e-14 |
| J2 propagators | DOP853 with own J2 potential gradient | production RK4: 0.44 m after 1 day |
| NASA SBM | Hand EMR/M; KS tests for Lc, A/M mixture and ΔV; np.interp re-implementation of the Johnson 2001 eq. 6–7 tables | all KS p > 0.3; tables 2e-4; momentum re-centred to 1e-4; no hyperbolic fragments |
| Fragment propagation | DOP853 with own J2 + co-rotating drag; Vallado Table 8-4 retyped | 19 m after 6 h (≤ 1 % of the fragment σ); energy drift 1.2e-7 per day |
| Propulsion | `solve_ivp` of dm/dt, dv/dt with a terminal event; datasheets; jet efficiency F·Isp·g₀/2P | 1e-13 rel; all 12 engines within datasheet values; electric efficiencies 0.48–0.65 |
| Decision | Own re-implementation of the HOW_IT_WORKS §11 formula on 500 random alerts; Pc sweep | 0 mismatches; monotone; weights sum to 1; 1e-4 = CARA threshold |
| Manoeuvres | 18 planner options on 6 real alerts, each executed with `apply_delta_v` and then re-screened | miss 0.46 m, TCA 35 µs, Pc 0.005 decades |

## Honest limitations

* **TEME frame.** SGP4 outputs TEME, and the code treats TEME as inertial.
  Relative geometry and Pc are invariant to the rigid TEME↔GCRF rotation
  (verified, FRM-6). The ~0.37° of precession since J2000 only shifts
  *absolute* positions, by about 44 km. TEME→ECEF by GMST82 is the defined
  transform (Vallado 2006). UT1−UTC and polar motion are ignored, which costs
  at most 0.47 km on display. skyfield and astropy are not installed, so this
  bound is analytic.
* **Engine datasheet values** were compared against public figures recalled
  from manufacturer datasheets and papers, not fetched live. The per-engine
  notes give the ranges.
* **Covariance and HBR** are modelled (TLE age and object class), not taken
  from CDMs. The verification proves the maths is implemented correctly, not
  that the inputs are realistic.
* **SBM.** For fragments between 10 and 11 cm the bimodal (> 11 cm) A/M table
  is used, without the 8–11 cm bridging function. This affects the A/M
  distribution only.
* **Chan series.** This is an approximation for anisotropic covariances. It
  reaches 26 % relative error when HBR ≈ 2σ_min, and this is reported, not
  hidden. Foster remains the authoritative value.
* **Manoeuvre check.** It used executed burns on a cloned propagator, not the
  live :8000 server.
