# Orbital Sentinel: What's Running (Judges' Talk Track)

A plain-language walkthrough of every task the system runs, in the order it happens. Each section has **what it does**, **how** (the real method), and **what to say** (a line you can read out).

---

## 0. One-line pitch

> "Orbital Sentinel tracks real satellites live, predicts close approaches before they happen, computes a real collision probability, simulates the debris if they hit, and plans the smallest burn that avoids the collision and the cascade that follows."

---

## 1. Startup sequence (`backend/main.py` → `lifespan`)

When the server boots it does these steps in order:

| Step | What happens | What to say |
|---|---|---|
| 1 | Load the Foster B-plane physics engine (`core/analytics.py`) | "First we load the collision-probability math." |
| 2 | Start the satellite state tracker | "This keeps per-satellite history for telemetry and anomaly checks." |
| 3 | Create agency authority sessions (SpaceX, ESA, ISRO, ROSCOSMOS, NASA, CNSA, US Space Force, NOAA) | "Each agency can only command its own satellites. That's the authority model." |
| 4 | XGBoost risk model: loaded only if `ENABLE_EXTENDED_PIPELINE=1` | "The ML is optional. Physics always runs; ML adds a second opinion on top." |
| 5 | Fetch TLEs (Space-Track if credentials exist, otherwise the bundled catalogue). `MAX_SATS` defaults to 500 | "We pull real orbital elements (TLEs) for the catalogue." |
| 6 | Build the first snapshot and wire up the REST/WebSocket routes | "The globe has data from the first second." |

---

## 2. The three background tasks (always running)

### Task A: Snapshot loop, every 1 s (`refresh_snapshot_loop`)
- **Does:** propagates every satellite to the current *simulation clock* time with **SGP4**, the standard model used with NORAD TLEs. It also moves any debris fragments forward by 1 s and attaches each satellite's Kalman covariance.
- **Output:** position, velocity, altitude, speed and agency for each object, plus debris clouds.
- **Say:** "Every second we recompute where every object is using SGP4, the same propagator the space-surveillance community uses with TLEs."

### Task B: Alert loop, every 30 s (`refresh_alerts_loop`)
This is the main pipeline. Each run does the following:
1. **Screening** (`core/screening.py`): checks *all* tracked objects over a future window on the sim clock, finds each pair's **time of closest approach (TCA)** and miss distance, and assigns a severity.
2. **Collision probability (Pc)**: uses the **Foster 2D B-plane** method. The covariance grows with **TLE age** (older data means more uncertainty). The hard-body radius is the sum of the two objects' radii.
3. **Debris alerts**: tracks fragments from simulated breakups against the catalogue.
4. **ML score** (`ml/risk_api.py`): an XGBoost surrogate attaches a second risk estimate. **It never replaces the physics Pc.**
5. **Cascade planner** (`services/cascade_planner.py`): builds a graph of conjunctions and ranks which satellites trigger chain reactions (graph ranker / CPI score). It computes cascade depth, agencies involved and total Δv.
6. **Merge**: physics fields always win. The graph only adds enrichment such as cascade depth and the recommended manoeuvre.
7. **Hotspots + debris clouds**: shows where the risk is concentrated, converted to lat/lon/alt for the globe.
- **Say:** "Every 30 seconds we screen the whole catalogue for future close approaches, compute a real collision probability, then ask: if this one hits, what does it hit next? That's the cascade."

### Task C: WebSocket broadcast (`api/ws_handler.py → satellite_broadcast_loop`)
- **Does:** pushes the latest snapshot to every connected browser over `/ws/satellites`.
- **Say:** "The 3D Cesium globe gets live positions pushed to it. Nothing is faked on the frontend."

*(Optional, extended pipeline only: shadow-mode ML retraining every 6 h and Kafka publishing.)*

---

## 3. What the judge can trigger (REST API)

| Endpoint | What it does | What to say |
|---|---|---|
| `POST /judge/manipulate-satellite` | Judge moves a satellite into a dangerous orbit | "Pick any satellite and push it toward something; watch the alert appear." |
| `POST /predict/conjunction` | Full prediction for one pair: TCA, Pc, B-plane, breakup forecast, avoidance burn | "Here's the full risk report for that pair." |
| `POST /maneuver` | Executes an avoidance burn | "We execute the burn, and the new orbit feeds straight back into screening." |
| `POST /simulate`, `POST /debris/simulate` | Runs a collision and generates the fragment cloud | "If they do hit, this is the debris field it creates." |
| `POST /test-mode/setup` → `run-to-tca` → `GET /separation` | Deterministic demo: set up a pair, fast-forward to TCA, measure separation | "Fast-forward to the moment of closest approach and measure the gap." |
| `POST /simulation/time` | Controls the simulation clock | "We can scrub time forward." |
| `GET /alerts`, `/cascade/status`, `/model-metrics`, `/ml/status` | Live alerts, cascade plan, model accuracy | "Every number on screen comes from these endpoints." |

---

## 4. Work in progress right now (uncommitted changes on branch `acm`)

**Theme: no invented numbers. Every value shown is computed, and a planned burn reproduces exactly when it is executed.**

### 4.1 Burns are now real propagated deviations (`core/sgp4_propagator.py`)
- **Before:** an executed burn re-fitted a new TLE from the post-burn state. Because SGP4 uses *mean* elements, that re-fit made the satellite jump about 10 km immediately and drift 100+ km within hours.
- **Now:** position = SGP4(t) + [J2(t; state + Δv) − J2(t; state)].
  Both the burned and un-burned arcs are integrated with the **same two-body + J2 RK4 dynamics** (20 s step, cached every 120 s), so a zero burn gives exactly zero and the model errors cancel.
- It is the **same model the manoeuvre planner predicts with**, so what we plan is what happens.
- Burn history is kept per satellite. `clone()` lets us run what-if checks without touching the live catalogue.
- **Say:** "When you fire a thruster, we don't fake a new orbit. We integrate the real physics difference the burn makes, with Earth's oblateness (J2), on top of SGP4."

### 4.2 Burned trajectories are used everywhere
Screening, TCA search, the debris model, the cascade planner and the sim engine now all read the burned trajectory (`propagator.trajectory()` / `add_burn_offsets`). **After a burn, the alert actually clears, or doesn't, based on physics.**

### 4.3 Conjunction predictor uses real values only (`services/conjunction_predictor.py`)
- Removed the old fallback that made up numbers (`exp(-miss²)` "Pc", hard-coded ellipse, fixed 150 fragments).
- New `assess_pair_risk()`:
  - **Pc:** Foster method with a covariance based on TLE age, measured from TLE epoch to TCA.
  - **Hard-body radius:** taken from the object catalogue (SATCAT), with the source tagged.
  - **Breakup forecast:** NASA Standard Breakup Model, **N(≥Lc) = 0.1 · M_ref^0.75 · Lc^-1.71**, with real masses, a catastrophic check (energy-to-mass ratio) and the mass source tagged.
  - If a value can't be computed, it returns **None**. It never substitutes a made-up number.
- All times use the **simulation clock**, not wall-clock time.
- **Say:** "If we can't compute a number, we show it as unknown. We don't make one up."

### 4.4 Manoeuvre planner handles debris with drag (`services/maneuver_planner.py`)
For fragment alerts, the fragment is now propagated with **two-body + J2 + atmospheric drag** (using its ballistic coefficient), the same model that generated the alert. Before, it used a drag-free approximation.
- **Say:** "Small debris decays because of drag, so we model drag when we plan around it."

### 4.5 New tests (`backend/tests/test_predictor_real.py`)
- The predictor's values are computed across near, mid and far miss distances.
- Epoch fallback uses the sim clock.
- A fragment alert gets a re-propagated manoeuvre with drag.

---

## 5. Recently finished (last commits)
- XGBoost surrogate for collision probability, plus ML scoring on debris alerts
- Batched cubic Hermite interpolation to speed up manoeuvre planning
- SATCAT snapshot rebuild scripts (real masses and sizes)
- Ranker tests that check physics integrity and make failures visible

---

## 6. Honest limits (say these before a judge asks)
- TLE/SGP4 accuracy is roughly a kilometre, so Pc uses a TLE-age covariance model rather than operator-grade covariance.
- Burns are impulsive (instantaneous), not finite-duration thrusts.
- ML is advisory only. The physics Pc is always the source of truth.
- Default demo size is 500 satellites (`MAX_SATS`); the pipeline is vectorised and can scale higher.

---

## 7. 30-second closing
> "Real TLEs in, SGP4 every second, full screening every 30 seconds, Foster collision probability, NASA breakup model for debris, a cascade graph to find the satellite that starts a chain reaction, and an avoidance burn propagated with real J2 physics, so the burn we recommend is the burn that actually works."
