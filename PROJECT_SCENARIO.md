# Orbital Sentinel — Project Scenario & Component Map

> A plain-language walkthrough of **what this project actually does**, **which
> technology is used where**, and **what state each part is in**.
>
> Companion documents: `README.md` (marketing-level), `AUDIT_NOTES.md` (defect audit
> and fix log), `research_paper/orbital_sentinel_paper.md` (formal writeup),
> `FEATURES_MATH_LOGIC.md` (equations).

---

## 1. The one-paragraph scenario

Orbital Sentinel is a **real-time satellite conjunction-awareness (collision
warning) dashboard**. Every second it pulls two-line element sets (TLEs) for
Earth-orbiting satellites, propagates each one to "now" with the SGP4 orbital
model, and works out which satellites are on collision courses. Any pair that
comes within 500 km becomes a *conjunction event*; the closest 200 km pairs are
additionally wired into a graph so the system can reason about **cascades** — one
satellite's evasive manoeuvre pushing it into a neighbour's path. Ranked threats
are turned into an ordered manoeuvre plan, and if a collision is confirmed the
system simulates the debris field that would result. All of it streams to a
browser at 1 Hz and is drawn on a 3D Earth via Cesium, with a second globe
zoomed in on the threat geometry.

It is a **prototype and demonstrator**, not an operational collision-avoidance
system. See §10 for what that means concretely.

---

## 2. What happens in one run (the 60-second version)

1. **Startup.** TLEs are fetched (Space-Track if `SPACETRACK_USER` /
   `SPACETRACK_PASS` are set, otherwise a bundled snapshot in
   `app/simulation/tle-data.txt`). ~120–500 satellites are loaded. Agency
   authority sessions are created for 10 agencies.
2. **Every 1 second** the position loop fires: propagate all satellites to the
   current epoch, advance any live debris fragments, cache the snapshot, and
   push it down the WebSocket. The globe moves.
3. **Every 30 seconds** the alert loop fires: screen all pairs for conjunctions,
   build the cascade graph, score and rank it, pick hotspots, simulate debris,
   and cache the result. Alerts are pushed only when this cache changes.
4. **In the browser**, the store receives the frame, splits debris clouds by
   source, and the globes, alert panel, cascade diagram, telemetry strip and
   model-status panel all re-render from that single store.

Two different cadences is deliberate: positions change continuously, but
re-running full N² conjunction screening every second is too expensive.

---

## 3. Technology → where it is used

| Technology | Used for | Where |
|---|---|---|
| **Python 3.11** | whole backend | `backend/` |
| **FastAPI** | REST + WebSocket server, lifespan startup | `backend/main.py`, `app/api/routes.py` |
| **Uvicorn** | ASGI server | `uvicorn main:app` |
| **sgp4** | TLE → position/velocity propagation | `app/core/sgp4_propagator.py` |
| **NumPy** | all numeric work, incl. hand-written ML | throughout `app/ml/` |
| **SciPy** | KD-tree neighbour search, optimisation | `app/services/cascade_planner.py` |
| **XGBoost** | conjunction risk classifier | `app/ml/xgboost_scorer.py` |
| **scikit-learn** | metric computation, model plumbing | `app/ml/model_metrics.py` |
| **PyTorch** | autograd training of the GAT only | `app/ml/gat_cascade.py` |
| **filterpy** | Kalman filtering of covariance | `app/core/kalman.py` |
| **pandas / joblib** | artifact IO, model persistence | `app/ml/artifacts.py`, trainers |
| **websockets** | 1 Hz push channel | `app/api/ws_handler.py` |
| **confluent-kafka / kafka-python** | *optional* event streaming | `app/streaming/kafka_adapter.py` |
| **APScheduler** | *optional* replacement for asyncio loops | `backend/main.py` |
| **pytest** | 104 backend tests | `backend/tests/` |
| **React 18** | UI | `frontend/src/` |
| **Zustand** | single global store | `frontend/src/store/useStore.js` |
| **CesiumJS 1.114** | 3D globe rendering | `components/Globe/CesiumGlobe.jsx`, `ThreatGlobe.jsx` |
| **Vite 6** | dev server + production bundle | `frontend/vite.config.js` |
| **ESLint** | linting (react-hooks rules) | `frontend/eslint.config.js` |

Runtime dependencies are split: `backend/requirements.txt` is the required set;
`backend/requirements-optional.txt` holds APScheduler and the Kafka clients. The
GAT needs **only `torch`**, which is already in the required set.

---

## 4. The backend pipeline, stage by stage

```
TLE file / Space-Track
   └─ app/data/tle_fetcher.py            fetch or fall back to bundled snapshot
      └─ app/core/sgp4_propagator.py     SGP4 → ECI position+velocity, 6×6 covariance
         ├─ app/core/kalman.py           filter the covariance per satellite
         └─ app/core/state_cache.py      keep the newest snapshot / alert set
            │
            ├── 1 Hz ──> app/api/ws_handler.py ──> WebSocket /ws/satellites
            │             (+ app/core/debris_model.py fragment positions)
            │
            └── 30 s ──> app/core/conjunction.py     screen_conjunctions()
                          │  WATCH_DISTANCE_KM = 500
                          │  · miss distance, relative speed
                          │  · RIC frame transform of covariance
                          │  · 2-D collision probability (Foster B-plane, Chan fallback)
                          │  · CPI score, severity band, TCA search
                          │
                          └─ app/services/cascade_planner.py   analyze_snapshot()
                             │  influence radius = 200 km → proximity graph
                              │  · ranks nodes with the GAT (primary_ranker)
                              │  · cross-checks with the GNN (disagreement only,
                              │    never contributes to the plan)
                              │  · resolves the cascade, optimises the plan
                              │  · recommends RSW burn vectors
                              │
                              ├─ app/ml/gat_cascade.py    the production ranker
                              ├─ app/ml/gnn_cascade.py    cross-check; reports
                              │                             disagreement, scores nothing
                             │
                             ├─ app/services/debris_model.py   build_debris_alerts()
                             │     predictive multi-shell FORECAST clouds (alert cadence)
                             │
                             └─ app/core/debris_model.py      DebrisModel
                                   EVOLVE fragment simulation (1 Hz cadence)

                         ──> merged alert cache ──> REST /api/alerts + WebSocket
```

### The two debris layers (commonly confused — they are not duplicates)

| | `app/core/debris_model.py` | `app/services/debris_model.py` |
|---|---|---|
| Purpose | Simulate an **actual** breakup | Predict a **hypothetical** debris field |
| Physics | NASA EVOLVE: fragment count from object size, size distribution, velocity kick, exponential drag decay, atmospheric density by altitude | RK45 radius integration over a radius timeline |
| Cadence | every 1 s (radii grow continuously) | only when the alert cache recomputes |
| `debris_source` | `fragment` | `forecast` |
| Both are served | yes, tagged and merged client-side | yes |

---

## 5. Backend file map — what each file owns

### `backend/app/core/` — physics and state
| File | Responsibility |
|---|---|
| `sgp4_propagator.py` | Loads TLEs, propagates to epoch, holds per-satellite Kalman state |
| `kalman.py` | 6×6 state covariance filter |
| `state_cache.py` | Module-level cache of the newest snapshot and alert payload |
| `conjunction.py` | **Screening.** Pair geometry, RIC transform, collision probability, CPI, severity, TCA search |
| `analytics.py` | **Foster B-plane maths.** RTN covariance, B-plane projection, Foster/Chan integration, Pc, covariance ellipse, NASA fragment count |
| `debris_model.py` | EVOLVE fragment-cloud simulation |
| `satellite_state_tracker.py` | Per-satellite tracking bookkeeping |
| `frames.py` | ECI ↔ RTN ↔ B-plane frame helpers |
| `sim_clock.py` | Simulation clock (supports time offset for demos) |
| `agency.py`, `agency_authority.py` | Agency inference and authorisation sessions |

### `backend/app/services/` — orchestration
| File | Responsibility |
|---|---|
| `cascade_planner.py` | Builds the 200 km proximity graph, calls GNN+GAT, resolves cascades, produces the manoeuvre plan, hotspots, node probabilities |
| `conjunction_predictor.py` | Propagation to a future TCA, geodetic enrichment, collision-prediction payloads |
| `conjunction_solver.py` | Miss-distance / classification helpers |
| `debris_model.py` | Forecast cloud construction from alerts |

### `backend/app/ml/` — models
| File | Responsibility |
|---|---|
| `xgboost_scorer.py` | XGBoost risk classifier + a `BoostedRiskModel` fallback; `XGBoostScorer.score` is the production entry point |
| `gnn_cascade.py` | `CascadeGNN` — ranks cascade nodes from fixed global summary statistics |
| `gat_cascade.py` | `CascadeGAT` — **trained multi-head graph attention** ranker |
| `lstm_predictor.py` | Recurrent trajectory forecaster (see caveat in §7) |
| `anomaly.py` | Residual anomaly detection against the forecaster |
| `shadow_mode.py`, `shadow_logger.py` | Shadow-mode retraining, logs risk samples, promotes only on improvement |
| `runtime.py` | `MLRuntime` — bundles predictor+anomaly+shadow; `get_ml_runtime()` is an lru_cache singleton |
| `synthetic_data.py` | Deterministic synthetic graph/risk/trajectory generators (the training set) |
| `model_metrics.py` | Cached metrics for the status panel, with artifact-signature invalidation |
| `train_models.py` | CLI: `python -m app.ml.train_models --gat --graph ... --risk ...` |
| `artifacts.py` | Artifact path resolution, load/save |
| `artifacts/` | `graph_model.json`, `gat_model.json`, `trajectory_model.json`, `model_metrics_cache.json`, XGBoost model |

### `backend/app/api/` and `routers/`
| File | Responsibility |
|---|---|
| `api/routes.py` | 25 REST endpoints (satellites, alerts, metrics, manoeuvres, simulations, agencies, scenarios) |
| `api/ws_handler.py` | 1 Hz broadcast loop; attaches the alert block only when the cache stamp changes |
| `routers/test_mode.py` | Injects deterministic ECI encounter scenarios for repeatable demos |
| `routers/predict.py` | Prediction sub-router |
| `streaming/kafka_adapter.py` | Optional Kafka publisher, off by default |

---

## 6. Frontend file map — what each file owns

### Entry and state
| File | Responsibility |
|---|---|
| `main.jsx` | React root |
| `App.jsx` | **The data hub.** Bootstraps via REST, opens the WebSocket, reconnects on close, heartbeats, polls alerts every 30 s, wires the layout |
| `store/useStore.js` | Zustand store — satellites, alerts, hotspots, cascade plan, `debrisBySource`, model metrics, connection flags |
| `utils/api.js` | `apiGet` / `apiPost` / `apiDelete` helpers (**not yet used everywhere — see §9**) |
| `utils/coords.js` | ECI → lat/lon/alt conversion |
| `utils/cascadeGraph.js` | Turns the alert list into nodes/edges for the diagram |
| `utils/tcaCountdown.js` | Countdown formatting |

### Visualisation
| Component | Responsibility |
|---|---|
| `Globe/CesiumGlobe.jsx` | **Live catalogue globe** — every tracked satellite, agency filter, selection |
| `ThreatGlobe.jsx` | **Threat-analysis globe** — the two conflict pair, B-plane, trajectories, debris shells |
| `CascadeDiagram.jsx` | Cascade propagation graph (nodes + edges + depth) |
| `BPlaneDiagram.jsx` | Foster B-plane / covariance ellipse inset |
| `ConjunctionMonitor.jsx` | Alert list and severity banding |

### Panels and controls
`MissionControlCenter`, `TelemetryStrip`, `Telemetry/TelemetryPanel`, `Alert/AlertPanel`,
`ThreatCard`, `ModelStatusV2`, `ModelMetricsPanel`, `ConjunctionPredictor`,
`Header`, `ConnectionBanner`, `QuickSimulationPanel`, `TestModePanel`,
`Control/ManualControl`, `Control/PreflightModal`, `Control/SimulationTimeControl`,
`Control/UplinkDownlinkPanel`, `Control/UplinkDownlinkV2`, `SimulationDrawer`,
`MetricsDrawer`, `Globe/AgencyFilter`, `hooks/useTestMode`.

---

## 7. The models — what each is, and honest status

| Model | File | Kind | Trained? | Used by | Measured accuracy |
|---|---|---|---|---|---|
| Risk Scorer | `xgboost_scorer.py` | gradient-boosted trees | yes, XGBoost | `cascade_planner` risk scoring, `/api/model-metrics` | **75.46%** (recall 44.64%) |
| GAT Cascade | `gat_cascade.py` | **multi-head graph attention** | yes, PyTorch autograd → JSON | **the production ranker** | 95.54% |
| Graph Cascade (GNN) | `gnn_cascade.py` | logistic model over graph summary statistics | yes, SGD | **cross-check only — never scores a plan** | 98.76% |
| Trajectory predictor | `lstm_predictor.py` | single-tanh recurrent cell | yes, SGD | `/api/satellites/{id}/orbit`, anomalies | MAE 226.93 km |

Three honesty notes worth carrying forward:

- **`CascadeGNN` is in the request path, but only as a cross-check.** The audit
  found it was absent entirely (`self.gnn` held a `CascadeGAT`, so the code read
  as if a GAT were being shadowed by a GNN). FIX-9 renamed that attribute to
  `primary_ranker` and added `cross_check_ranker`. The GNN now runs beside the GAT
  and reports disagreement via `ranker_review`; its scores never enter the plan.
  The two are deliberately **not** ensembled: they share `_build_gnn_inputs`, so
  agreement is not independent evidence, and they encode different biases, so an
  average would imply precision neither model has. A GAT-vs-GNN delta above
  `RANKER_DISAGREEMENT_THRESHOLD` (0.25) raises a review flag; observed max on the
  120-node live snapshot is 0.211, so healthy runs stay quiet.
- **`lstm_predictor.py` is not an LSTM.** It is a hand-written NumPy recurrent
  cell (`Wx, Wh, b, Wy, by`, `tanh` activation) with no gates. The filename
  overstates the architecture. It works, but don't describe it as an LSTM.
- **The GAT was, until this audit, a silent alias of the GNN.** Its only
  implementation was gated behind `torch_geometric`, which is not installed, so
  every prediction fell through to `CascadeGNN`. The paper reported one model as
  two. It is now a real attention network. See `AUDIT_NOTES.md` FIX-3.

A related failure mode was also closed: the old code caught *any* ranker exception
and substituted `np.zeros` for probabilities and cascade depth, then emitted the
plan as if healthy. A fabricated all-zero plan made `risk_after` look *better*,
because it subtracted a `node_prob` term that was always zero. Ranker failures now
raise (preserving the previous real plan) or return real cross-check numbers
flagged `degraded`, and the UI shows a RANKER status light. See `AUDIT_NOTES.md`
FIX-9.

All metrics are on **synthetic data** from one generator. They demonstrate
reproducibility on the training distribution, not operational accuracy.

---

## 8. Configuration

All optional; defaults give a working demo with no `.env` file.

| Variable | Default | Effect |
|---|---|---|
| `MAX_SATS` | 500 | satellite cap |
| `SNAPSHOT_REFRESH_SECONDS` | 1 | position/WebSocket cadence |
| `ALERT_REFRESH_SECONDS` | 30 | conjunction re-screen cadence |
| `BROADCAST_INTERVAL_SECONDS` | 1 | WebSocket push rate |
| `ENABLE_EXTENDED_PIPELINE` | 0 | **only** gates the heavy ML runtime + Kafka. Core screening, cascade, hotspots and debris always run |
| `USE_APSCHEDULER` | 0 | use APScheduler instead of asyncio loops |
| `SPACETRACK_USER` / `SPACETRACK_PASS` | unset | live TLEs instead of the bundled snapshot |
| `KAFKA_ENABLED` / `KAFKA_BOOTSTRAP` / `KAFKA_TOPIC_*` | 0 / localhost:9092 | optional event streaming |
| `SHADOW_MODE_ENABLED` / `SHADOW_RETRAIN_HOURS` | 1 / 6 | shadow retraining cadence |
| `ALLOW_UNAUTHENTICATED_COMMANDS` | see code | command authorisation gate |

---

## 9. Current status — what is fixed and what is not

### Fixed and verified in this audit
- **P1** alert pipeline no longer disabled by `ENABLE_EXTENDED_PIPELINE=0`
- **P3** GAT is a genuinely distinct, trained attention model; metrics differ
- **P6, P11, P12, P13** WebSocket cleanup, streamed alerts/hotspots/cascade, no socket leak, cleared alerts still pushed
- **P8** `/api/alerts` `count` always equals `len(alerts)`; both alert sources unioned
- **P9** TCA urgency measured from the snapshot epoch, not wall-clock
- **P14** both debris sources reach the client (restores AC-5)
- **P2, P10** investigated and confirmed as false positives

Evidence: **104 backend tests pass**, `npm run build` succeeds, runtime check
returns 120 satellites / 7 alerts / 120-node graph with `count == len(alerts)`.

### Still open
- **P4** — the frontend still calls raw `fetch('/api/...')` in `App.jsx` and
  elsewhere instead of routing through `utils/api.js`, so error and base-URL
  handling is inconsistent.
- **P5** — 6 `eslint` react-hooks warnings remain (0 errors). Three are the
  stale-ref-in-cleanup class, which is a real bug when the ref holds a live
  Cesium primitive (`CesiumGlobe.jsx:885`, `ThreatGlobe.jsx:507`,
  `CascadeDiagram.jsx:248`).
- **P7** — `PROJECT_STRUCTURE.md`, `FEATURES_MATH_LOGIC.md` and
  `PROJECT_OVERVIEW.txt` are stale. The paper was updated as part of P3.

### Known-open limitations (design gaps, not regressions)
- **L1** no per-object hard-body radii — collision detection uses a flat 1 m gate
- **L2** covariance is modelled, not measured from real observations
- **L3** no catalog-scale screening benchmark (only synthetic graphs)
- **L4** no temporal train/test split for the risk scorer
- **L6** fragments are not propagated to re-entry
- **L8** no persistent storage and no real route auth
- Risk Scorer recall is 44.64% — it misses over half of positive cases at the
  0.5 threshold. This is the single biggest accuracy problem in the project.

---

## 10. How to run and verify

```bash
# backend
cd backend
pip install -r requirements.txt
python -m pytest -q                 # 104 tests
python -m app.ml.train_models --gat # retrain the attention ranker
uvicorn main:app --port 8000

# frontend
cd frontend
npm install
npm run lint                        # 0 errors, 6 warnings (P5)
npm run build
npm run dev
```

Full audit trail, per-defect root causes and the verification log are in
`AUDIT_NOTES.md`.
