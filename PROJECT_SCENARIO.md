# Orbital Sentinel — Project Scenario & Component Map

> A plain-language walkthrough of what the project does and which file does
> what. For the equations, parameter sources, limitations and verification
> tests of every computation, see **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)**.

---

## 1. The one-paragraph scenario

Orbital Sentinel is a **space-traffic / conjunction-awareness dashboard**. It
loads TLEs, propagates them with SGP4 on a simulation clock, and every 30 s
screens all tracked objects for close approaches over the **next 24 hours**.
For each approach it computes the TCA, the miss distance and a Foster
collision probability from a TLE-age covariance. An XGBoost surrogate gives an
advisory second opinion on Pc. The operator can then break up the most
dangerous pair at its predicted TCA with the NASA Standard Breakup Model. The
fragments are propagated with J2 and drag and screened against the catalogue,
so the cascade graph shows which satellites the debris threatens next, and
how many hops each sits from the collision. Every threatened pair gets a
recommended avoidance burn, verified by re-propagation and costed with the
rocket equation. Commanding is restricted to the owning agency, taken from the
CelesTrak SATCAT. Results stream to a Cesium globe over WebSocket.

It is a **prototype**, not an operational system. The main reasons: the
covariance is modelled rather than measured, masses are assumed, and the
bundled TLEs are a snapshot. See the limitations in `docs/HOW_IT_WORKS.md`.

---

## 2. What happens in one run

1. **Startup:** load TLEs (Space-Track, then a public mirror, then the
   bundled `app/simulation/tle-data.txt`) and create agency sessions.
2. **Every 1 s:** propagate all objects to sim-now, advance debris fragments,
   cache the snapshot, and push it over WebSocket.
3. **Every 30 s, and after any burn, override or clock change**
   (`main.refresh_alerts_once`): future-window screening → debris alerts →
   ML surrogate scores → cascade graph and manoeuvre planning → publish.
4. **Browser:** the store receives the frame, and the globes, alert panel,
   cascade diagram and telemetry re-render.

---

## 3. Technology → where it is used

| Technology | Used for | Where |
|---|---|---|
| FastAPI / Uvicorn | REST + WebSocket | `backend/main.py`, `app/api/` |
| sgp4 | TLE propagation (incl. `SatrecArray` batches) | `app/core/sgp4_propagator.py`, `app/core/screening.py` |
| NumPy | vectorised propagation, breakup, planning | throughout |
| SciPy | `cKDTree` screening, bounded Brent TCA, `ncx2` Pc, `dblquad` cross-check | `app/core/screening.py`, `debris_model.py`, `analytics.py` |
| XGBoost | Pc surrogate (regression of log10 Foster Pc) | `app/ml/risk_api.py`, `train_risk_surrogate.py` |
| PyTorch | offline training of the advisory GAT ranker only | `app/ml/gat_cascade.py` |
| React + Zustand + CesiumJS + Vite | UI | `frontend/src/` |
| pytest | backend tests (`tests/test_*_real.py` prove the physics) | `backend/tests/` |

---

## 4. Backend file map

| File | Responsibility |
|---|---|
| `app/core/sgp4_propagator.py` | TLE loading, SGP4, executed burns (SGP4 + J2 deviation) |
| `app/core/screening.py` | 24 h screening, TCA refinement, TLE-age covariance, HBR model, Foster Pc, severity |
| `app/core/conjunction.py` | CPI triage index, `find_tca` (pair TCA), legacy wrapper |
| `app/core/analytics.py` | reference Foster integrators (cross-checks), TLE-age covariance helper |
| `app/core/breakup.py` | NASA SBM + fragment dynamics (J2 + drag) |
| `app/core/debris_model.py` | collision events, fragment propagation, fragment-vs-catalogue alerts, clouds |
| `app/core/satcat.py`, `agency.py`, `agency_authority.py` | SATCAT owner lookup, agency attribution, command authority |
| `app/core/satellite_state_tracker.py` | simulated payload block (fuel by rocket equation, illumination); no invented housekeeping |
| `app/core/sim_clock.py` | simulation clock (wall clock + bounded offset) |
| `app/services/cascade_planner.py` | alert graph, BFS depth, P(hit), hotspots, plan, advisory ranker review |
| `app/services/maneuver_planner.py` | burn search by re-propagation, rocket-equation fuel |
| `app/services/conjunction_predictor.py` | `/api/predict/conjunction` for user-specified objects |
| `app/services/debris_model.py` | forecast (hypothetical) breakup clouds for top alerts |
| `app/simulation/sim_engine.py` | scenarios (computed crossings), burn execution, pre-flight gates |
| `app/ml/risk_api.py` | XGBoost Pc surrogate scoring (no torch) |
| `app/ml/model_metrics.py` | held-out metrics from the model cards |
| `app/ml/lstm_predictor.py` | experimental tanh RNN trajectory model (not an LSTM; loses to linear extrapolation) |
| `app/ml/gat_cascade.py`, `gnn_cascade.py` | advisory graph rankers trained on synthetic graphs |

---

## 5. Frontend file map — what each file owns

### Entry and state
| File | Responsibility |
|---|---|
| `main.jsx` | React root |
| `App.jsx` | **The data hub.** Bootstraps via REST, opens the WebSocket, reconnects on close, heartbeats, polls alerts every 30 s, wires the layout |
| `store/useStore.js` | Zustand store — satellites, alerts, hotspots, cascade plan, `debrisBySource`, model metrics, connection flags |
| `utils/api.js` | `apiGet` / `apiPost` / `apiDelete` helpers |
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

## 6. The models: honest status

| Model | Role | Evidence |
|---|---|---|
| XGBoost Pc surrogate | advisory `alert.ml`, never overwrites Pc | held-out MAE 0.106 dex in log10 Pc; precision 0.975 / recall 0.950 at 1e-4 (`risk_model_card.json`) |
| Trajectory RNN | experimental, not in the alert path | MAE 459 km vs linear extrapolation 20.5 km: **it underperforms the baseline** |
| GAT / GNN rankers | advisory `ranker_review` only | trained on synthetic graphs; Spearman ρ vs physics P(hit) reported |
| Operator feedback | counted only, trains nothing | `operator_feedback.trains_any_model: false` |

---

## 7. Configuration

All optional; defaults give a working demo with no `.env` file.

| Variable | Default | Effect |
|---|---|---|
| `MAX_SATS` | 500 | satellite cap |
| `SNAPSHOT_REFRESH_SECONDS` | 1 | position/WebSocket cadence |
| `ALERT_REFRESH_SECONDS` | 30 | conjunction re-screen cadence |
| `BROADCAST_INTERVAL_SECONDS` | 1 | WebSocket push rate |
| `ENABLE_EXTENDED_PIPELINE` | 0 | **only** gates the heavy ML runtime + Kafka. Core screening, cascade, hotspots and debris always run |
| `USE_APSCHEDULER` | 0 | use APScheduler instead of asyncio loops |
| `SPACETRACK_USER` / `SPACETRACK_PASS` | unset | live TLEs (else public mirror, else bundled snapshot) |
| `SCREEN_WINDOW_HOURS` / `SCREEN_STEP_S` / `SCREEN_THRESHOLD_KM` | 24 / 60 / 25 | future-window screening parameters |
| `KAFKA_ENABLED` / `KAFKA_BOOTSTRAP` / `KAFKA_TOPIC_*` | 0 / localhost:9092 | optional event streaming |
| `SHADOW_MODE_ENABLED` / `SHADOW_RETRAIN_HOURS` | 1 / 6 | shadow retraining cadence |
| `ALLOW_UNAUTHENTICATED_COMMANDS` | see code | command authorisation gate |

---

## 8. How to run and verify

```bash
cd backend
pip install -r requirements.txt
set ORBIT_SENTINEL_SKIP_DOTENV=1     # PowerShell: $env:ORBIT_SENTINEL_SKIP_DOTENV=1
python -m pytest -q
python -m app.ml.train_risk_surrogate   # reproduce the Pc surrogate + model card (seeded)
uvicorn main:app --port 8000

cd ../frontend
npm install
npm run dev
```
