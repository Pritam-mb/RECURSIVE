# Orbital Sentinel: Explainable Real-Time Conjunction Assessment with Foster B-Plane Integration, Hermite TCA Refinement, and Layered Debris Forecasting

**Authors:** [Author Names]  
**Affiliation:** [Institution]  
**Date:** September 2026  
**Keywords:** space situational awareness, conjunction assessment, SGP4, Foster collision probability, B-plane integration, time of closest approach, debris visualization, cascade risk, graph neural networks

---

## Abstract

Orbital Sentinel is a prototype for real-time satellite tracking, conjunction awareness, risk ranking, and three-dimensional incident visualization. The system integrates Two-Line Element (TLE) ingestion, SGP4 propagation, a FastAPI service layer, and a Cesium-based WebGL client connected via REST and WebSocket. The collision-probability path implements the Foster (1992) B-plane double-integration method with an RTN covariance model and a Chan (1997) series fallback; risk ranking employs an XGBoost classifier trained on Foster-generated synthetic data alongside two graph cascade rankers — a deterministic surrogate over graph summary statistics and a multi-head graph attention network — for maneuver-probability and cascade-depth estimation. This paper documents the full implemented methodology, identifies and corrects eleven material defects present in earlier prototype versions, and evaluates all corrected paths through targeted regression tests. Five defects concern the physics and rendering path: collision probability was invariant to miss distance, TCA was confined to the numerical sample grid, incident markers were placed at the initial epoch, near-misses generated debris events, and multi-layer debris forecast shells were collapsed to a single sphere in the client. Four concern machine-learned components and their honest reporting: the attention ranker was an unimplemented stub that silently reported another model's scores as its own, the entire alert pipeline was disabled under the default configuration, ranker failure was converted into a plausible-looking all-zero maneuver plan, and the deployed risk-scorer metrics had been reported from a model that never served alerts. Two further defects affected alert dissemination: alert count and list could disagree, and TCA urgency was measured against wall-clock time rather than the snapshot epoch. All 116 backend tests pass. Cached evaluation on Foster-derived synthetic data reports 75.46% accuracy for the production risk scorer at 44.64% recall, 98.76% for the surrogate graph ranker, and 95.54% for the attention ranker. These figures characterize internal synthetic-set behavior and do not constitute operational flight-validation claims. The resulting architecture provides a reproducible, explainable platform for conjunction-response demonstrations and serves as a controlled baseline for future validation against independent conjunction truth data.

---

## I. Introduction

The cataloged orbital population now exceeds 27,000 tracked objects, and the number of untracked debris fragments with lethal collision cross-section is estimated at several hundred thousand [1]. As constellation deployment accelerates, the number of object pairs requiring daily screening grows as the square of the catalog size, placing increasing pressure on space situational awareness (SSA) infrastructure. Prototype systems that expose the geometry, timing, uncertainty, and downstream consequences of encounters in a transparent, debuggable architecture serve an important role: they allow methodology auditing, operator training, and controlled comparison of algorithmic choices before operational commitment.

Orbital Sentinel addresses this need with an end-to-end pipeline: orbital states are propagated from TLEs via SGP4, cached, screened for close approaches, ranked by risk, streamed over WebSocket, and rendered on an interactive Cesium globe. The system additionally contains demo-specific logic, synthetic machine-learning artifacts, and a test router for injecting deterministic encounter scenarios. These components create a precise reporting obligation — implementation outputs must not be conflated with operationally validated collision probabilities. This paper enforces that boundary throughout.

### A. Research Questions

1. Does collision probability respond to both positional uncertainty and the relative miss vector in the B-plane?
2. Can the TCA routine detect a close approach that occurs between coarse numerical output samples?
3. Are hotspot and debris products located at the computed encounter position rather than the initial epoch?
4. Does the Cesium client preserve the multi-layer debris forecast-shell structure supplied by the backend?

### B. Contributions

This work contributes:
- A code-grounded system description of the full Orbital Sentinel pipeline, traced to implementation files
- Documentation of the Foster B-plane collision probability pipeline and its RTN covariance model
- A defect-focused correction set addressing eleven implementation errors, five in the physics and rendering path and six in the machine-learned and dissemination paths
- A regression suite of 116 executable tests with deterministic acceptance criteria
- Quantitative vector results for corrected probability, TCA refinement, and debris layering
- An evidence-calibrated interpretation of synthetic model metrics that clearly delineates operational validity boundaries, including explicit disclosure of the production risk scorer's 44.64% recall

No new software packages were downloaded for the analysis or artifact generation. All evidence derives from the repository source, installed runtime capabilities, and the regression test suite.

### C. Paper Organization

Section II describes the system architecture and data flow. Section III formalizes the implemented methodology, including the Foster B-plane path, RTN covariance model, Hermite TCA refinement, XGBoost risk scorer, trajectory predictor, graph cascade models, NASA EVOLVE debris simulation, and Cesium visualization pipeline. Section IV records the corrective implementations. Sections V and VI define the evaluation protocol and present results. Sections VII and VIII discuss limitations and conclusions.

---

## II. System Architecture

### A. Operational Data Flow

The runtime data flow follows a linear pipeline:

```
TLE File → SGP4 Propagator → State Cache → Conjunction Screener
    → Foster Pc Engine → CPI Scorer → XGBoost Risk Scorer
    → KD-Tree Graph Builder → GAT Cascade Ranker (production)
                                └→ GNN Cross-Check (reports disagreement only)
    → REST / WebSocket → Cesium Globe Client
```

TLE records are ingested from a bundled snapshot at startup (`backend/app/simulation/tle-data.txt`). The SGP4 propagator (`sgp4_propagator.py`) evaluates Earth-centered inertial (ECI) position and velocity at the requested epoch. A state cache stores the most recent propagated snapshot and serves REST and WebSocket consumers at 1 Hz. Conjunction screening (`conjunction.py`) operates on the cached snapshot; cascade planning (`cascade_planner.py`, `gat_cascade.py`, `gnn_cascade.py`) operates on the resulting proximity graph. Debris products are produced by two complementary layers: an EVOLVE fragment simulation in `core/debris_model.py` and a predictive forecast-cloud integrator in `services/debris_model.py`, both served to the client with a `debris_source` discriminator. The test router (`routers/`) accepts injected ECI encounter scenarios for deterministic validation.

### B. Backend Component Map

| File | Responsibility |
|------|---------------|
| `sgp4_propagator.py` | TLE parsing, SGP4 integration, RSW→ECI burn transforms |
| `conjunction.py` | Pairwise screening, Foster Pc dispatch, CPI computation |
| `analytics.py` | RTN covariance, B-plane transform, Foster double integration, Chan fallback, fragment count |
| `debris_model.py` | NASA EVOLVE fragment generation, drag-aware propagation |
| `cascade_planner.py` | KD-tree graph construction, CPI-weighted cascade expansion, ranker selection and failure handling |
| `gnn_cascade.py` | Graph cascade ranker over fixed graph summary statistics; runs as a cross-check, contributes nothing to the plan |
| `gat_cascade.py` | Multi-head graph attention ranker; the sole production ranker for maneuver probability and cascade depth |
| `xgboost_scorer.py` | XGBoost classifier on Foster-derived features, boosted-stump fallback |
| `satellite_state_tracker.py` | State aggregation and Kalman-augmented tracking |
| `model_metrics.py` | Artifact-signature metric cache, per-model independence flags, `graph_attention` status block |

### C. Frontend Architecture

The React client (`frontend/src/`) uses Cesium for globe rendering and Zustand for state management. It bootstraps through REST (`/api/satellites`, `/api/alerts`), then maintains a WebSocket connection for live 1 Hz position updates. Alerts, hotspots, the cascade plan, the proximity graph, and the ranker review block are pushed over the same socket, so the operator view updates without polling; the client reconciles the pushed set against the REST response using a single merge rule. Between updates, each satellite is advanced using linear ECI extrapolation with the stored velocity vector. ECI coordinates are converted to Cesium Cartesian using Greenwich Mean Sidereal Time (GMST). Separate entity collections represent satellites, orbit polylines, conjunction hotspots, approach lines, and debris forecast shells. A status strip reports backend, pipeline, model-training, physics, and cascade-ranker health, with the ranker indicator distinguishing healthy, disputed, and degraded states.

### D. Data and Model Provenance

| Source | Description | Operational Interpretation |
|--------|-------------|---------------------------|
| TLE snapshot | Bundled SGP4-derived state estimate | State accuracy limited by TLE age and maneuver history |
| Foster synthetic data | Conjunction pairs generated by the analytics pipeline | Internal classification surrogate; not CDM-validated |
| Graph snapshots | Synthetic proximity graphs | Internal cascade-ranking evaluation only; not real conjunction histories |
| Test scenarios | Injected ECI encounters | Deterministic UI and solver validation |
| `model_metrics_cache.json` | Evaluation on above synthetic sets (25 September 2026) | Internal synthetic-set reproducibility only |

**Evidence boundary:** No external conjunction data messages (CDMs), independent truth ephemerides, or operator-validated encounter records were present in the repository. Metric claims are constrained to internal synthetic evaluation.

---

## III. Methodology

### A. Orbit Propagation

The SGP4 propagator converts TLE records to `Satrec` objects using the `python-sgp4` library (Vallado et al., 2006 [2]). For each satellite, ECI position $\mathbf{r}(t)$ and velocity $\mathbf{v}(t)$ are evaluated at the requested epoch. Scalar speed and approximate altitude are:

$$v_{kmh} = \|\mathbf{v}\| \times 3600 \quad \text{(km/h)}$$
$$h_{km} = \|\mathbf{r}\| - R_E \quad \text{(km, } R_E = 6371 \text{ km)}$$

**Maneuver integration:** Impulsive delta-V burns are accepted in ECI or converted from RSW (Radial/Along-track/Cross-track) frame. The RSW basis is:

$$\hat{R} = \frac{\mathbf{r}}{\|\mathbf{r}\|}, \quad \hat{W} = \frac{\mathbf{r} \times \mathbf{v}}{\|\mathbf{r} \times \mathbf{v}\|}, \quad \hat{S} = \hat{W} \times \hat{R}$$

The post-burn state is converted to osculating Keplerian elements; the burn is rejected if the predicted perigee drops below 100 km. A new `Satrec` is rebuilt so future propagation uses the updated orbit.

### B. RTN Covariance Model

When external covariance data is unavailable (the common case for TLE-sourced catalog objects), `analytics.py` builds a 3×3 ECI position covariance via an RTN (Radial/Transverse/Normal) uncertainty model:

$$\sigma_R = 50 \text{ m}, \quad \sigma_T = 1.0 \times \left(1 + \frac{t_{age}}{24}\right) \text{ km}, \quad \sigma_N = 50 \text{ m}$$

The transverse uncertainty grows linearly with TLE age $t_{age}$ in hours, reflecting the dominant along-track prediction error characteristic of TLE-based propagation. The combined ECI covariance is:

$$\mathbf{C}_{ECI} = \mathbf{R}_{RTN}\, \text{diag}[\sigma_R^2, \sigma_T^2, \sigma_N^2]\, \mathbf{R}_{RTN}^T$$

where $\mathbf{R}_{RTN}$ is the rotation matrix whose columns are $\hat{R}$, $\hat{T}$, $\hat{N}$ expressed in ECI.

### C. B-Plane Collision Probability (Foster 1992)

The collision probability path in `analytics.py` implements the Foster and Estes (1992) [3] B-plane method as used by NASA CARA.

**B-plane construction:** The B-plane is perpendicular to the relative velocity vector $\Delta\mathbf{v} = \mathbf{v}_q - \mathbf{v}_p$ at TCA. Unit vectors are:

$$\hat{h} = \frac{\Delta\mathbf{v}}{\|\Delta\mathbf{v}\|}, \quad \hat{t} = \frac{\hat{h} \times \mathbf{r}_p}{\|\hat{h} \times \mathbf{r}_p\|}, \quad \hat{n} = \hat{t} \times \hat{h}$$

The projection matrix is $\mathbf{B} = [\hat{t}\;;\;\hat{n}] \in \mathbb{R}^{2\times 3}$.

**Miss vector:** The combined position delta $\Delta\mathbf{r} = \mathbf{r}_q - \mathbf{r}_p$ is projected into the B-plane:

$$\mathbf{b} = \mathbf{B}\,\Delta\mathbf{r} = [B_T,\, B_N]^T \quad \text{(km)}$$

**Combined covariance:** The sum of individual ECI covariances is projected onto the B-plane:

$$\mathbf{C}_{2D} = \mathbf{B}\,(\mathbf{C}_p + \mathbf{C}_q)\,\mathbf{B}^T \in \mathbb{R}^{2\times 2}$$

**Foster integration:** Collision probability is the integral of the bivariate Gaussian PDF over the circular hard-body disk of radius $R_{HBR} = 10$ m:

$$P_c = \iint_{x^2+y^2 \leq R_{HBR}^2} \mathcal{N}(\mathbf{x};\, \mathbf{b},\, \mathbf{C}_{2D})\, dx\, dy$$

This is evaluated by `scipy.integrate.dblquad` with a 2-second timeout. If integration exceeds the timeout or fails, the Chan (1997) [4] series approximation provides a fast fallback:

$$P_c^{Chan} \approx u \cdot e^{-(u+x)} \cdot I_0\!\left(2\sqrt{ux}\right)$$

where $u = R_{HBR}^2 / (2\sigma^2)$, $x = \|\mathbf{b}\|^2 / (2\sigma^2)$, $\sigma^2 = \text{tr}(\mathbf{C}_{2D})/2$, and $I_0$ is the modified Bessel function of the first kind. The result is clipped to $[0,1]$.

### D. Conjunction Priority Index (CPI)

The CPI provides a scalar urgency score on $[0, 10]$ calibrated against the Foster $P_c$ ordering. The five sub-scores are weighted to ensure CPI rank equals Foster $P_c$ rank for comparable encounters:

| Component | Weight | Mapping |
|-----------|--------|---------|
| Miss distance | 0.40 | Piecewise linear on $[0,500]$ km |
| Foster $P_c$ (log) | 0.30 | Linear in $\log_{10}(P_c) \in [-6,-2] \to [0,10]$ |
| TCA urgency | 0.15 | Piecewise linear on $[0,24]$ h |
| Closing speed | 0.10 | Linear, capped at 15 km/s |
| TLE data age | 0.05 | Linear, capped at 48 h |

**NASA CARA threshold alignment:**
- $P_c \geq 10^{-4}$ (CPI $\approx$ 5.0): watch/elevated concern
- $P_c \geq 10^{-3}$ (CPI $\approx$ 7.5): action threshold
- $P_c \geq 10^{-2}$ (CPI $\approx$ 10): emergency — maneuver required

### E. TCA Refinement: Cubic Hermite Interpolation

The coarse TCA search evaluates ECI states at uniform time steps. The corrected method identifies the two time intervals adjacent to the coarse minimum-separation sample and fits cubic Hermite position curves from endpoint positions and velocities. The relative separation within each bracket is:

$$d(t)^2 = \|\mathbf{r}_{rel}(t)\|^2 = \|\mathbf{r}_1(t) - \mathbf{r}_2(t)\|^2$$

Hermite interpolation within interval $[t_k, t_{k+1}]$ gives a $C^1$ continuous relative trajectory. Minimizing $d^2$ via `scipy.optimize.minimize_scalar` over each bracket yields sub-step temporal localization. This reduces sample-grid aliasing without requiring a full re-integration.

Additionally, the target-time endpoint is corrected so the propagation loop terminates at the requested horizon $T$ exactly, preventing overshoot that caused the prior implementation to miss the final sample.

### F. XGBoost Risk Scorer

The risk scorer (`xgboost_scorer.py`) uses a Foster-trained XGBoost classifier as its primary path. The eight input features are:

$$\mathbf{f} = [d_{miss},\; v_{rel},\; t_{TCA},\; t_{age},\; h_{alt},\; \sigma_R^2,\; \sigma_T^2,\; \sigma_N^2]$$

Training data is generated by the `generate_risk_dataset` function in `synthetic_data.py`, which calls the Foster B-plane pipeline to produce $(P_c, \text{features})$ pairs. Binary labels are obtained by thresholding $P_c \geq 0.5$. The XGBoost model is saved to `risk_model_trained.json` and loaded at module import. If XGBoost is unavailable (no library), the system falls back to a compact gradient-boosted stump ensemble (`BoostedRiskModel`) trained on the same synthetic data.

The measured behavior of this model through its production scoring entry point is reported in Section VI.E: 75.46% accuracy at 44.64% recall. The high precision (87.83%) paired with low recall indicates a conservative scorer that rarely raises a false alarm but misses most positives at the 0.5 decision threshold. This is a property of the trained model on a class-imbalanced synthetic distribution, and is carried forward as an open limitation rather than corrected; the earlier higher figure attributed to this component came from a fallback model outside the request path (Section IV.I).

### G. Trajectory Predictor

The trajectory predictor (`lstm_predictor.py`) consumes a ring buffer of the 100 most recent $(\tau, \mathbf{r})$ samples and rolls forward a single hidden state:

$$\mathbf{z}_t = \mathbf{x}_t \mathbf{W}_x + \mathbf{h}_{t-1} \mathbf{W}_h + \mathbf{b}, \qquad \mathbf{h}_t = \tanh(\mathbf{z}_t), \qquad \hat{\mathbf{y}}_t = \mathbf{h}_t \mathbf{W}_y + \mathbf{b}_y$$

with $\mathbf{h}_0 = \mathbf{0}$. Despite the module name, this is a single-tanh Elman recurrent cell with no input, forget, or output gates, and therefore not an LSTM. It is trained by SGD on synthetic coordinate sequences and serves `/api/satellites/{id}/orbit` and the anomaly detector. The filename overstates the architecture; the equations above are the implemented model, and the distinction is noted here so the measured MAE is not attributed to gated memory.

### H. Graph Cascade Models

The cascade planner (`cascade_planner.py`) builds a proximity graph using a KD-tree with an influence radius of 200 km. Each edge stores: miss distance, predicted miss distance, relative velocity, TCA, $P_c$, CPI score, severity, and inverse-distance influence weight $w = 1/\max(d_{pred}, 1)$.

Node features include ECI state components, altitude, speed, and peak-CPI from the strongest neighbor. The cascade rankers come in two variants. `gnn_cascade.py` is a deterministic surrogate that scores each node from fixed global summary statistics. `gat_cascade.py` is a genuine multi-head graph attention network: for each target node it takes a softmax over its in-edges *plus its own projected state*, so an isolated node falls back to full self-attention rather than collapsing to a constant. Each head has independent projections and attention vectors, the heads are concatenated, and an ELU nonlinearity precedes the linear readout. The attention weights are gradient-trained with PyTorch autograd on synthetic graph snapshots and exported to a JSON artifact, so runtime inference is deterministic NumPy and requires no deep-learning stack. Previously this module was gated behind the optional `torch_geometric` package, which is not installed; the import failed, and every call fell back to `CascadeGNN`, making the two models byte-identical aliases of each other. `CascadeGNN` now also runs inside the planner as a cross-check, but its scores are excluded from the published plan by design (see Section VII.C).

Cascade depth is estimated as

$$d_{casc} = \text{clip}\!\left(\left\lceil\, P_{man}\times 4 + w_{inf}\times 1.5 \,\right\rceil,\; 0,\; 4\right)$$

**Maneuver recommendation.** Two distinct burn computations exist and should not be conflated. The planner's `_recommend_maneuver_rsw` derives the burn from alert physics alone, using the CPI of the strongest threat and the maneuver probability:

$$\Delta v_{planner} = \text{clip}\!\left(0.15 + \frac{CPI}{10}\times 1.4 + P_{man}\times 0.7,\; 0.05,\; 2.5\right)$$

split as radial 25%, along-track 85%, cross-track 15% of that magnitude, with each component's sign directed away from the threat. This is the value that reaches the operator, and it does not depend on which ranker is active.

The ranker modules separately compute a model-internal along-track burn for use as a training target and for internal diagnostics:

$$\Delta v_{along} = \text{clip}\!\left(0.05 + P_{man}\times 1.35 + \frac{r_{local}}{12} + 0.2\,v_{bias},\; 0.05,\; 5.0\right)$$

with radial and cross-track components $\text{clip}((r_{local}-0.5)\times 0.25, \pm1)$ and $\text{clip}((w_{inf}-0.35)\times 0.22, \pm1)$. These use the ranker's own learned features rather than the alert's CPI, and are not the burn shown to the operator. The distinction matters because the two have different units of driver ($CPI$ versus model features) and different clip ceilings (2.5 versus 5.0 m/s); a description that quoted a single formula would misstate which one governs the published plan.

### I. Debris Forecast Model

The debris model (`debris_model.py`) implements the NASA EVOLVE 4.0 Standard Breakup Model [5]. Fragment sizes follow a power-law CDF inversion:

$$L = L_{min} \cdot (1 - u)^{-1/1.6}, \quad u \sim \text{Uniform}(0,1)$$

Fragment velocity kicks follow the NASA EVOLVE characteristic velocity:

$$\Delta v_{mean} = \frac{200}{\sqrt{L}} \; \text{m/s}, \quad \Delta v \sim \text{LogNormal}(\mu, \sigma)$$

where $\mu = \ln(\Delta v_{mean}) - \sigma^2/2$ and $\sigma^2 = \ln(1 + 0.16)$.

Catastrophic/sub-catastrophic threshold: specific energy $E/M > 40$ J/kg. Fragment count is estimated from the NASA breakup formula:

$$N = 0.1 \times (M_1 + M_2)^{0.75} \times \begin{cases} 1 & \text{catastrophic} \\ 0.1 & \text{cratering} \end{cases}$$

Fragment propagation uses two-body gravity plus an Euler-integrated atmospheric drag term:

$$a_{drag} = -\frac{1}{2} C_D \frac{A}{m} \rho(h) v_{ms}^2 \hat{v}$$

with $C_D = 2.2$ and exponential atmospheric density scaling from sea-level reference.

### J. Cesium Globe Pipeline

Between 1 Hz backend updates, the client advances each satellite by linear ECI extrapolation: $\mathbf{r}_{adv} = \mathbf{r} + \mathbf{v} \cdot \Delta t$. ECI is converted to Cesium Cartesian via GMST rotation. The orbit track is segmented when adjacent samples diverge by more than 8000 km to prevent cross-arc line artifacts. Hotspot, approach-line, and debris-shell entities are updated from the alert REST response on each polling cycle.

---

## IV. Corrective Implementation

Table I summarizes the eleven defects identified through code inspection and the corrections applied. Defects 1–5 concern the physics and rendering path. Defects 6–9 concern machine-learned components and the honesty of their reporting. Defects 10–11 concern alert dissemination.

### Table I: Defect–Correction Summary

| # | Component | Defect Observed | Root Cause | Correction Applied | Behavioral Effect |
|---|-----------|----------------|------------|-------------------|------------------|
| 1 | Collision probability | $P_c$ invariant to miss distance | Chi-square approximation used only covariance trace, ignoring $\mathbf{b}$ | Foster B-plane double integration with projected miss vector | $P_c$ decreases monotonically with miss distance |
| 2 | TCA refinement | Close approach between coarse samples missed; endpoint overshoot | Linear interpolation without sub-step search; loop counter off by one | Cubic Hermite minimization in adjacent intervals; exact endpoint fix | Improved temporal localization; target-time compliance |
| 3 | Hotspot placement | Incident marker at initial epoch position, not TCA position | Test router used snapshot positions at $t=0$ rather than solver output | Hotspot midpoint computed from solver states at refined TCA | Correct incident location displayed in Cesium |
| 4 | Debris semantics | Near-misses generated debris; expansion started before impact | Debris triggered on alert, not on collision detection; timeline not offset | Debris only on confirmed collision ($d < 1$ m); post-impact timeline | Correct event semantics: no debris for near-misses |
| 5 | Debris layers | All shells collapsed to single opaque sphere in Cesium | Client reduced shell array to maximum-radius scalar | One translucent `EllipsoidGraphics` per unique positive-radius shell | Multi-layer forecast structure preserved and visible |
| 6 | GAT cascade ranker | Attention model reported byte-identical scores to the GNN | Only implementation gated behind uninstalled `torch_geometric`; import failed and every call fell through to `CascadeGNN` | Trained multi-head attention network with self-state in the segment softmax, exported to JSON | Two genuinely distinct rankers (95.54% vs 98.76%) |
| 7 | Alert pipeline | Conjunction screening, cascade planning, hotspots and debris never executed under the default configuration | `ENABLE_EXTENDED_PIPELINE` guard placed at the top of the alert refresh, though the flag exists only to skip optional ML runtime and Kafka | Guard narrowed to the optional subsystems; core pipeline always runs | `/api/alerts` serves a populated payload instead of the startup stub |
| 8 | Cascade ranker | Any ranker exception was replaced with zero vectors, publishing a fabricated plan as healthy | Single `try`/`except` around scoring substituted `np.zeros` for probability and depth | Failure raises, or returns real cross-check probabilities flagged `degraded` | Operator sees a hard error or an explicit degraded flag, never a plausible fake |
| 9 | Risk-scorer reporting | 93.85% accuracy attributed to the deployed scorer | Metrics computed on a legacy boosted-stump model that never served alerts | Metrics regenerated through the production `XGBoostScorer.score` path | 75.46% accuracy, 44.64% recall — the deployed model's real performance |
| 10 | Alert dissemination | `count` could disagree with `len(alerts)`; REST and WebSocket applied different merge rules | Duplicates resolved differently per transport | One dedup rule (graph entry preferred) applied everywhere | Count, list, and both transports always agree |
| 11 | TCA urgency | Urgency computed against wall-clock time | Alert handler read `datetime.now()` instead of the snapshot epoch | Urgency measured from the snapshot timestamp | Urgency stable regardless of client/server clock skew |

### A. Probability Correction (Defect 1)

The former implementation computed:

$$P_c^{old} \propto \text{gammainc}\!\left(1, \frac{R_{HBR}^2}{\text{tr}(\mathbf{C}_{2D})}\right)$$

This expression depends only on the covariance trace and hard-body radius, making probability independent of where the objects actually pass relative to each other. The corrected path evaluates the full Foster integral over the B-plane Gaussian shifted by the miss vector $\mathbf{b}$, making $P_c$ a proper function of encounter geometry. A degenerate covariance (eigenvalue $< 10^{-20}$) still falls back to the Chan series approximation.

### B. TCA Localization (Defect 2)

The test solver (`conjunction_solver.py`) previously returned the coarse minimum without sub-step refinement. The corrected routine: (a) identifies the sample index achieving minimum separation, (b) applies Hermite minimization over the two adjacent intervals $[t_{k-1}, t_k]$ and $[t_k, t_{k+1}]$, and (c) corrects the integration step count so the final step lands exactly at the requested target time, eliminating endpoint overshoot.

### C. Hotspot Relocation (Defect 3)

The alert router previously populated hotspot coordinates from the cached snapshot at $t=0$. The corrected router passes the solver's refined encounter time and interpolated ECI states to the hotspot calculator, which computes the midpoint between the two object positions at TCA. This ensures the displayed hotspot and the approach-line endpoints correspond to the actual predicted encounter, not the initial epoch.

### D. Debris Event Semantics (Defect 4)

The former implementation created a debris event for any conjunction alert crossing the warning threshold, regardless of whether a collision was actually detected. The corrected logic separates two thresholds:
- **Warning gate** ($d < 100$ m): creates a visual alert and approach marker in Cesium
- **Collision gate** ($d < 1$ m): additionally creates a debris cloud via the EVOLVE model

The debris timeline offset is corrected so the impact shell represents the state at $t_{TCA}$ rather than $t=0$.

### E. Debris Layer Rendering (Defect 5)

The frontend component previously iterated the `shells` array and retained only the last (maximum) radius, rendering a single opaque sphere. The corrected component: validates each shell for positive radius, sorts by radius, deduplicates, and creates one translucent `EllipsoidGraphics` entity per shell with opacity decreasing from impact to forecast horizon. This preserves the temporal expansion structure intended for operator situational awareness.

### F. Attention Ranker Authenticity (Defect 6)

`gat_cascade.py` previously gated its only implementation behind the optional `torch_geometric` package. Because that package is not installed, the import failed and every prediction fell through to `CascadeGNN`, so the "GAT Cascade" metrics row was a byte-for-byte copy of the "Graph Cascade" row — one model presented as two, with the second presented as independent corroboration.

The replacement is a genuine multi-head graph attention network. For each target node it computes a softmax over its in-edges *plus its own projected state*, so an isolated node attends to itself rather than collapsing to a constant. Each head has independent projections and an attention vector; heads are concatenated, an ELU nonlinearity precedes a linear readout, and weights are gradient-trained with PyTorch autograd. Training requires only `torch`, which is already a dependency; runtime inference is deterministic NumPy reading a JSON artifact, so no deep-learning stack is needed in the serving path. Torch/NumPy forward parity is verified to $3.4\times10^{-7}$.

The correction is visible in the artifact rather than only in prose: the metrics payload now carries a per-model `independent` flag and a `graph_attention` status block recording the trainer, head count, hidden dimension, epoch count, and a `degenerate` flag, so a future alias cannot be presented as corroborating evidence.

### G. Alert Pipeline Availability (Defect 7)

`refresh_alerts_once()` began with `if not ENABLE_EXTENDED_PIPELINE: return`. That flag exists to skip the optional heavy ML runtime and the Kafka publisher, but the guard sat at the top of the alert refresh, so under the default configuration the entire product pipeline — conjunction screening, cascade planning, hotspot generation and debris building — never executed, and `/api/alerts` served the empty startup stub indefinitely.

The guard was narrowed to the optional subsystems. Core alerting now always runs. This is the single most consequential defect in the audit: every other numerical result in this paper describes code paths that, on a default deployment, were unreachable.

### H. Cascade Ranker Failure Handling (Defect 8)

The planner wrapped scoring in a single `try`/`except` that substituted zero vectors for maneuver probability and cascade depth, then published the resulting plan as healthy. Any ranker error — corrupt artifact, shape mismatch, missing torch, an ordinary bug — therefore produced a structurally valid, entirely fabricated plan with every probability at `0.0` and no degraded indication.

The failure was not merely cosmetic. Because the downstream risk update subtracts a probability-proportional term, an all-zero input *inflated* the apparent post-manoeuvre safety margin and understated residual risk, and ranking fell back to peak CPI alone, so the plan looked plausible rather than obviously empty. The only distinguishing signal was a log line at `WARNING`.

Failure is now typed and surfaced. If the primary attention ranker fails but the cross-check is healthy, the plan is scored by the cross-check and carries real probabilities with an explicit `degraded` flag and reason, and a status indicator turns red in the operator interface. If both fail, the planner raises, which preserves the last valid plan rather than replacing it with a fabricated one. A cross-check failure alone does not degrade a healthy primary.

### I. Risk-Scorer Reporting (Defect 9)

Reported accuracy for the risk scorer was 93.85%, computed on a legacy boosted-stump model that never served alerts. Metrics are now generated through the production `XGBoostScorer.score` entry point, yielding 75.46% accuracy at 44.64% recall. The lower figure describes the model actually in the request path.

Recall of 44.64% means the deployed scorer misses more than half of positive cases at the 0.5 threshold. This is a real weakness of the production model, not a reporting artifact, and it is carried forward as an open item rather than resolved.

### J. Alert List and Count Consistency (Defect 10)

Duplicate alerts were resolved differently by the REST route and the WebSocket broadcaster, so `count` could disagree with `len(alerts)` and the two transports could show different alert sets for the same snapshot. Both paths now share one deduplication rule, which prefers a cascade-graph entry over a screening entry, and both WebSocket pushes and REST responses are merged consistently.

### K. TCA Urgency Epoch (Defect 11)

Alert urgency was computed against wall-clock time rather than the epoch of the snapshot being reported, making the displayed time-to-closest-approach depend on client/server clock skew. Urgency is now measured from the snapshot timestamp.

### L. Regression Test Suite

The suite contains 116 tests. Three target the original physics and rendering defects, and the remainder protect the alert pipeline, the attention ranker, the cross-check wiring, and the failure-handling paths.

**Test 1 — Probability monotonicity:** Creates a 2D Gaussian with $\sigma = 10$ m per axis and $R_{HBR} = 10$ m. Asserts $P_c(d=0) > P_c(d=100\text{ m})$. This test fails under the former miss-independent implementation.

**Test 2 — Sub-step TCA detection:** Configures two objects with opposing 2 m/s relative motion and true closest approach at $t = 5$ s, coarse samples at $t \in \{0, 7\}$ s. Asserts refined miss $< 0.1$ m and TCA within 0.05 s of truth.

**Test 3 — Debris shell ordering:** Injects a collision at the snapshot epoch with a 2 h forecast. Asserts impact radius $<$ mid-horizon radius $<$ forecast-horizon radius (monotonically increasing shells).

---

## V. Experimental Design

### A. Evidence Sources

1. Repository documentation: `README.md`, `PROJECT_SCENARIO.md`
2. Source inspection of all files listed in Table I and Section II
3. Saved model metric cache (`model_metrics_cache.json`, generated 25 September 2026)
4. The 116-test regression suite executed in the repository environment
5. In-process API verification against a 120-satellite TLE snapshot, confirming a populated `/api/alerts` payload under the default configuration

### B. Test Definitions and Acceptance Criteria

| Test | Configuration | Acceptance Criterion |
|------|--------------|---------------------|
| Probability sensitivity | $\sigma = 10$ m per axis; $R_{HBR} = 10$ m | $P_c(0\text{ m}) > P_c(100\text{ m})$ |
| Sub-step TCA | Opposing 2 m/s; true TCA at 5 s; coarse at 0, 7 s | Miss $< 0.1$ m; TCA within 0.05 s |
| Debris timeline | Impact at snapshot; 2 h forecast | $r_{impact} < r_{mid} < r_{forecast}$ |
| Alert pipeline availability | Default configuration, no env overrides | Screening, cascade planning and debris each execute once per refresh |
| Ranker failure (both models) | Both rankers raise | Planner raises; no all-zero plan is published |
| Ranker failure (primary only) | Primary raises, cross-check healthy | Plan carries cross-check probabilities and `degraded: true` |
| Cross-check independence | Cross-check returns inverted probabilities | Disagreement is reported; published probabilities remain the primary's, unblended |

### C. Synthetic Metric Protocol

The risk scorer is evaluated as a binary classifier: Foster-derived $P_c$ values are thresholded at 0.5 to produce labels, the scorer's continuous output is thresholded at 0.5 for predicted class, and standard classification metrics (accuracy, precision, recall, F1) are computed. Graph Cascade and GAT Cascade models use synthetic graph snapshots with the same binary threshold. The trajectory predictor is evaluated on next-position coordinates using MAE and RMSE.

Cache sample counts: 4,096 risk samples, 2,985 graph-node samples per graph model, 768 coordinate values for trajectory regression.

### D. Validity Boundary

This evaluation explicitly does not claim: real-world recall, calibrated probability against operational CDMs, maneuver optimality, or catalog-scale latency performance. No comparison against external conjunction truth data is made because none was provided. The evaluation answers whether the corrected implementation paths are internally coherent and behave as designed under controlled, deterministic inputs.

---

## VI. Results

### A. Regression Outcomes

| Test | Outcome | Observed Values |
|------|---------|----------------|
| Probability responds to miss | **PASS** | $P_c(0\text{ m}) = 0.3935$; $P_c(100\text{ m}) = 3.41 \times 10^{-20}$ |
| TCA between coarse samples | **PASS** | Refined miss $< 0.1$ m; TCA $\approx 5.00 \pm 0.05$ s |
| Debris shell ordering | **PASS** | Impact $<$ mid-horizon $<$ forecast-horizon (increasing radii) |
| Alert pipeline runs by default | **PASS** | Screening, cascade planning and debris each execute once per refresh; 120 nodes, 9 alerts, `count == len(alerts)` |
| Both rankers fail | **PASS** | `RuntimeError` raised; no all-zero plan published |
| Primary ranker fails only | **PASS** | Plan carries cross-check probabilities; `degraded: true`, `fallback_used: true` |
| Cross-check fails only | **PASS** | Plan unaffected; `cross_check_ok: false`, `degraded: false` |
| Cross-check is not fused | **PASS** | Published probability equals the primary's, never an average of the two |
| GAT/GNN forward parity | **PASS** | Torch vs NumPy max abs difference $3.4\times10^{-7}$ |
| GAT is not an alias | **PASS** | Max $\lvert P_{GNN} - P_{GAT}\rvert = 0.349$ across 24 graphs |

All 116 backend tests pass. The probability result spans twenty decades between the zero-miss and 100 m cases, confirming that the Foster integration is correctly sensitive to miss-vector offset. The former implementation would return identical values for both cases.

### B. Operational Snapshot

An in-process request against the default configuration with 120 TLE-derived satellites produced a populated alert payload: 120 graph nodes, 3 graph edges, 9 alerts, cascade depth 3, and 120 of 120 non-zero node probabilities. Both rankers reported healthy, with an observed maximum GAT/GNN probability delta of 0.257 against a review threshold of 0.25, which correctly raised a disagreement flag on one satellite. This is the defect-7 path: before the correction, the same request returned the empty startup stub.

### C. Corrected Probability Response

At zero miss distance, the computed probability (0.393) reflects the fraction of the $\sigma = 10$ m Gaussian that overlaps the $R_{HBR} = 10$ m disk when the Gaussian mean sits exactly at the disk center — a physically plausible result for two objects whose positional uncertainty envelopes fill the hard-body sphere. At 100 m separation (10$\sigma$), the value drops to $3.41 \times 10^{-20}$, consistent with the extreme tail of a bivariate Gaussian at 10 standard deviations from the mean.

### D. Debris Envelope Radii

The post-impact debris envelope at representative forecast times:

| Forecast Horizon | Radius (km) |
|-----------------|-------------|
| 0 h (impact) | 1.500 |
| 1 h | 211.704 |
| 2 h | 319.626 |
| 4 h | 403.483 |
| 8 h | 431.414 |
| 24 h | 433.500 |

The asymptotic convergence reflects the exponential velocity decay in the EVOLVE model: fragment expansion speed decays as $v(t) = v_0 e^{-t/\tau}$, making the total radius approach a finite maximum.

### E. Synthetic Model Performance

| Model | Accuracy | Precision | Recall | F1 | Samples |
|-------|----------|-----------|--------|-----|---------|
| Risk Scorer (XGBoost) | 75.46% | 87.83% | 44.64% | 59.20% | 4,096 |
| Graph Cascade (GNN) | 98.76% | 98.46% | 98.69% | 98.57% | 2,985 |
| GAT Cascade | 95.54% | 94.76% | 94.98% | 94.87% | 2,985 |

**Trajectory predictor:** MAE = 226.93 km, RMSE = 281.74 km across 768 coordinate values. The large absolute errors are expected for a lightweight linear predictor evaluated on broad synthetic orbital arcs; they should not be interpreted as orbit determination precision.

**Interpretation:** The GAT Cascade row was previously a byte-for-byte copy of the Graph Cascade row. That was not a coincidence of similar decision surfaces: `gat_cascade.py` gated its only implementation behind the optional `torch_geometric` package, which is not installed, so the import failed and every prediction silently fell back to `CascadeGNN`. The GAT is now a real multi-head attention network, gradient-trained with PyTorch autograd and exported to a JSON artifact for deterministic NumPy inference. The two models are now genuinely distinct, and the payload records an `independent` flag per graph model plus a `graph_attention` status block so a future alias cannot be presented as corroborating evidence.

The Risk Scorer row is lower than earlier drafts reported because metrics are now computed through the production `XGBoostScorer.score` entry point rather than the legacy boosted-stump model that never served alerts. The numbers describe the model actually in the request path; the earlier 93.85% described a model that was not deployed. Recall of 44.64% means the production scorer misses over half of the positive cases at the 0.5 threshold, which is a real limitation and not a rounding artifact.

High classification accuracy on the synthetic set establishes reproducibility on the training distribution but not external validity. The GAT was trained on a disjoint synthetic seed (17) from its evaluation seed (31), but both come from the same generator, so this is a generalization check within the synthetic distribution only. Independent time-separated splits and real CDM records are required for a publishable operational-accuracy claim.

### F. Practical Effect of Corrections

The repaired pipeline enforces a consistent event chain:
1. The numerical solver localizes the encounter time and positions via Hermite refinement
2. The hotspot marker uses the refined TCA position, not the initial epoch
3. An alert is generated for any miss distance under the 100 m warning gate
4. Debris is generated only when the collision gate ($d < 1$ m) is crossed
5. The Cesium layer renders each supplied forecast shell as a distinct translucent ellipsoid

This alignment removes previously visible contradictions in both API payloads and the globe display, where debris could appear for passes hundreds of meters apart and where the incident marker could appear on the opposite side of the Earth from the actual close approach.

The corrections to the learned components have a different character, and their effect is on reporting rather than on physics. Defects 6 and 9 did not change any computation that reached an operator; they changed what the system *claimed* about its computations. Before the correction, a single model was displayed as two corroborating models, a model that never served alerts supplied the reported accuracy figure, and a total ranker failure was displayed as a normal-looking plan. The consequence is that the prototype's headline metrics were not measurements of deployed behavior, and a reader of the earlier artifact would have drawn incorrect conclusions about system redundancy and risk-scorer quality even though no displayed probability was wrong.

Defect 7 is different again. It did not corrupt a result; it prevented any result from being produced on a default deployment, which means that the numerical findings in this paper describe code paths that were unreachable in a stock configuration. The operational snapshot in Section VI.B is reported specifically because it is the evidence that the corrections restored reachability rather than merely internal correctness.

---

## VII. Discussion and Limitations

### A. What the Prototype Demonstrates

- An end-to-end path from orbital elements to a live operator display with sub-second refresh, verified reachable under the default configuration
- Transparent, auditable stages: SGP4 propagation, Foster B-plane Pc, CPI ranking, graph cascade, debris EVOLVE model, Cesium visualization
- Deterministic injection scenarios enabling repeatable regression testing
- Clear API contracts between heuristic and model-based components, supporting modular replacement
- A ranker arrangement in which a silent total failure is structurally impossible to present as a healthy plan

### B. Known Limitations

| Domain | Limitation | Recommended Next Step |
|--------|-----------|----------------------|
| Orbit state | TLE/SGP4 accuracy degrades with age and post-maneuver mismatch | Timestamped catalog archive with truth ephemerides |
| Uncertainty | RTN $\sigma_T$ model is a fixed analytical approximation; no measurement-based covariance | B-plane Pc validation against NASA CARA reference cases |
| Collision risk | Hard-body radius fixed at 10 m; no object-specific dimensions | Object size catalog integration; per-pair radius |
| Debris | Isotropic EVOLVE envelope; no atmospheric re-entry trajectory for individual fragments | Full fragment orbital element propagation with drag |
| ML scorer | Training and evaluation use the same Foster-derived synthetic distribution; production recall is 44.64% | Independent time-separated splits; real CDM labels; retune decision threshold |
| Graph models | Attention ranker is trained, but both rankers consume one shared feature representation, so cross-checking cannot detect a common upstream feature error | Diverse or independently derived feature sets; real conjunction histories |
| Scale | Pairwise O($N^2$) screening; KD-tree helps but not benchmarked at catalog scale; GAT segment aggregation uses `np.add.at`, which suits ~10² nodes, not catalog scale | Benchmark over 27,000+ object snapshots with timing; vectorised or sparse aggregation |
| Model startup | A missing attention artifact triggers synchronous retraining on first import, blocking startup for minutes | Pre-baked artifact in the image; background warm-up; readiness gate |
| Access control | `ALLOW_UNAUTHENTICATED_COMMANDS` and `ALLOW_SESSION_MINTING` both default to enabled, and a request without a session header is attributed to a shared demo session that is authorized for every agency | Default both to disabled; issue per-agency sessions; reject rather than default unidentified requests |
| Persistence | Alert, debris and maneuver state live in process memory and reset on restart; no audit trail of operator decisions | Durable store with decision logging |

### C. Threats to Validity

**Metric cache provenance:** The `model_metrics_cache.json` records a single evaluation run on a deterministic synthetic configuration. Reported scores may not generalize to other random seeds, different class balance ratios, or real encounter data.

**Graph model degeneracy (resolved):** The previously identical GNN and GAT scores were not a ceiling effect or limited graph diversity. `gat_cascade.py` gated its only implementation behind the uninstalled `torch_geometric` package, so every prediction fell through to `CascadeGNN` and one model was reported as two. The GAT is now a genuinely distinct trained attention network (95.54% versus 98.76%).

**Graph ranker redundancy (resolved, with a caveat):** `cascade_planner.py` previously assigned a `CascadeGAT` to an attribute named `self.gnn`, and `CascadeGNN` was instantiated only by the metrics evaluator and by tests, so the two rows in Table VI.D were not mutually corroborating. `CascadeGNN` is now instantiated in the planner as an explicit cross-check. Two properties of this arrangement must not be overstated. First, both models consume the same `_build_gnn_inputs` representation, so their agreement is not independent evidence — they are two readouts of one feature space, not two independent measurements. Second, they are deliberately not ensembled: the GAT applies attention and the GNN applies symmetric aggregation, so a weighted average would yield a value corresponding to neither model. The GAT therefore remains the sole ranker, and the GNN's contribution is limited to reporting per-satellite disagreement (observed maximum 0.211 against a 0.25 review threshold on the 120-node operational snapshot) for operator inspection. The residual validity threat is that a shared upstream feature error would not be detected by cross-checking, since it would affect both models identically.

**Fabricated all-zero plans on ranker failure (resolved):** the planner previously caught any ranker exception and substituted zero vectors for manoeuvre probability and cascade depth, then published the resulting plan as healthy. Because the downstream risk update subtracts a probability-proportional term, an all-zero input inflated the apparent post-manoeuvre safety margin and understated residual risk. Ranker failure is now either surfaced as a hard error that preserves the last valid plan, or, where a healthy second model is available, reported as a degraded plan carrying real cross-check probabilities, an explicit `degraded` flag, and a corresponding status indicator in the operator interface.

**Regression test coverage:** The test suite is narrow regression testing of specific corrected behaviours. It does not constitute a comprehensive verification and validation suite.

**No UI study:** Visual correctness was assessed through rendered artifact inspection and code review. No user study of operator comprehension, alert triage performance, or decision latency was conducted.

### D. Recommended Research Path

The next validation phase should:
1. Build an encounter-plane probability benchmark against NASA CARA reference cases
2. Integrate object-specific covariances and hard-body radii from a curated catalog
3. Validate TCA localization against dense high-fidelity (high-order numerical) propagation
4. Implement temporal train/calibration/test splits for the risk scorer
5. Replace spherical EVOLVE envelopes with fragment ensembles propagated through drag-sensitive orbital dynamics with re-entry prediction

---

## VIII. Conclusion

Orbital Sentinel provides a coherent, explainable prototype for real-time orbital visualization and conjunction response. The Foster B-plane pipeline, RTN covariance model, Hermite TCA refinement, XGBoost risk scorer, graph cascade rankers, and NASA EVOLVE debris model are all documented to their implementation files and mathematically grounded in established space mechanics methods.

The audit identified eleven defects. Five concern the physics and rendering path: miss-independent probability, sample-grid TCA error and target overshoot, initial-epoch hotspot placement, debris on near-misses, and loss of debris forecast layers in the client. Six concern the learned and disseminated results: an attention ranker that silently reported another model's scores, an alert pipeline disabled under the default configuration, ranker failure converted into a fabricated all-zero plan, risk-scorer metrics drawn from a model that never served alerts, alert count and list able to disagree, and TCA urgency measured against wall-clock time. All eleven are corrected, and the behavioral effects are verified by a 116-test regression suite plus an in-process check against a 120-satellite snapshot.

The methodological finding is worth stating separately from the individual fixes. Every defect in the learned-component group shared one shape: the code computed something correct, and the *description* of that computation was wrong. A stub reported as a trained model, a metrics row attributed to a model outside the request path, a silent fallback reported as a healthy plan, and a cross-check presented as independent corroboration are all reporting failures rather than computational ones. They are also the failures a synthetic test suite is least likely to catch, because the numbers being produced are plausible. The defenses adopted in response — artifact-level `independent` and `degenerate` flags, disagreement thresholds surfaced to the operator, and metrics generated through the production entry point rather than a parallel one — are therefore part of the contribution, not incidental instrumentation.

Cached synthetic metrics are useful within their stated scope: 75.46% risk-scorer accuracy on Foster-generated data through the production scoring path, 98.76% graph-cascade accuracy and 95.54% GAT accuracy on synthetic graph snapshots. The correct conclusion is that the prototype offers a reproducible architecture, an aligned event chain from encounter detection through visualization, and a corrected baseline for future validation — not that it possesses operational collision-prediction accuracy. The risk scorer's 44.64% recall is a known weakness that must be addressed before the model is trusted operationally, and it is a property of the deployed model rather than of its reporting. Independent temporal evaluation, real CDM data, and full breakup-model fragment propagation remain necessary before operational deployment.

---

## References

[1] European Space Agency, "ESA's Annual Space Environment Report," ESA Space Debris Office, Darmstadt, Germany, 2024.

[2] D. A. Vallado, P. Crawford, R. Hujsak, and T. S. Kelso, "Revisiting Spacetrack Report #3," in *AIAA/AAS Astrodynamics Specialist Conference*, Keystone, CO, 2006, AIAA 2006-6753.

[3] J. L. Foster and H. S. Estes, "A Parametric Analysis of Orbital Debris Collision Probability and Maneuver Rate for Space Vehicles," NASA/JSC-25898, 1992.

[4] F. K. Chan, "Spacecraft Collision Probability," *Journal of Guidance, Control, and Dynamics*, vol. 20, no. 3, pp. 566–571, 1997.

[5] N. L. Johnson, P. H. Krisko, J.-C. Liou, and P. D. Anz-Meador, "NASA's New Breakup Model of EVOLVE 4.0," *Advances in Space Research*, vol. 28, no. 9, pp. 1377–1384, 2001.

[6] K. T. Alfriend, M. R. Akella, J. Frisbee, J. L. Foster, D. J. Lee, and M. Wilkins, "Probability of Collision Error Analysis," *Space Debris*, vol. 1, no. 1, pp. 21–35, 1999.

[7] M. R. Akella and K. T. Alfriend, "Probability of Collision Between Space Objects," *Journal of Guidance, Control, and Dynamics*, vol. 23, no. 5, pp. 769–772, 2000.

[8] R. H. Battin, *An Introduction to the Mathematics and Methods of Astrodynamics*, revised ed. AIAA Education Series, 1999.

[9] Orbital Sentinel repository: `README.md`, `PROJECT_SCENARIO.md`, `analytics.py`, `conjunction.py`, `debris_model.py`, `xgboost_scorer.py`, `gnn_cascade.py`, `gat_cascade.py`, `cascade_planner.py`, `sgp4_propagator.py`. Accessed September 2026.

[10] Orbital Sentinel `model_metrics_cache.json`, synthetic evaluation artifact, generated 25 September 2026.

---

*Manuscript prepared September 2026. All analysis derives from the supplied repository and installed runtime capabilities. No external dependency was downloaded for evidence generation.*
