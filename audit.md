# 🔍 Orbital Sentinel — Full Project Audit

> Audit date: 2026-10-08 | Both servers running (FastAPI :8000, Vite :5173)

---

## 📊 Summary Table

| Area | Status | Severity |
|---|---|---|
| ML pipeline gating (`ENABLE_EXTENDED_PIPELINE`) | **OFF by default — all ML disabled** | 🔴 Critical |
| ModelStatusV2 hardcoded zeros | Several counters always show `0` | 🔴 Critical |
| Telemetry fallback (`random.uniform`) | Random noise when tracker unavailable | 🟡 Medium |
| GAT Cascade degeneracy (fixed) | Was aliasing GNN — now real attention network | ✅ Fixed |
| CascadeGraph node layout | Uses `Math.random()` for initial positions | 🟡 Medium |
| LSTM buffer fill / "Predictions today" | Always shows `0` in UI | 🔴 Critical |
| Kafka / shadow-mode logging | Off unless env vars set | ℹ️ By design |
| Conjunction predictor | Fully working ✅ | ✅ OK |
| SGP4 propagation | Fully working ✅ | ✅ OK |
| WebSocket broadcast | Fully working ✅ | ✅ OK |
| Foster Pc / CPI alerts | Fully working ✅ | ✅ OK |
| Maneuver execution | Fully working ✅ | ✅ OK |
| Cascade / GNN scoring | Working (boosted-stump fallback) | 🟡 Medium |
| RLHF / Meta-Propagator section | Hardcoded zeros — no backend | 🔴 Showpiece |

---

## 🔴 Critical Issues

### 1. Extended ML Pipeline is OFF by default — XGBoost and anomaly detection are dead

**File:** [`main.py`](file:///d:/hakckolution/hackolution/backend/main.py#L48-L58), [`routes.py`](file:///d:/hakckolution/hackolution/backend/app/api/routes.py#L350-L365)

```python
ENABLE_EXTENDED_PIPELINE = os.getenv("ENABLE_EXTENDED_PIPELINE", "0") == "1"
ml_runtime = get_ml_runtime() if ENABLE_EXTENDED_PIPELINE else None
```

- `/api/anomalies` returns `{"count": 0, "anomalies": [], "enabled": false}` unless you set `ENABLE_EXTENDED_PIPELINE=1`
- The XGBoost **Foster-trained** model (`risk_model_trained.json`) is **never loaded** in the default run
- The LSTM trajectory recorder never records satellite positions
- Shadow-mode retraining never runs

**Fix:** Add to a `.env` file in `/backend`:
```
ENABLE_EXTENDED_PIPELINE=1
```

---

### 2. ModelStatusV2 has hardcoded-zero counters that can never update

**File:** [`ModelStatusV2.jsx`](file:///d:/hakckolution/hackolution/frontend/src/components/ModelStatusV2.jsx#L49-L71)

```js
const predictions = 0;       // ← never changes
const highRisk = 0;          // ← never changes
const lstmSats = 0;          // ← never changes
const bufferFill = 0;        // ← never changes (LSTM buffer)
const rlhfDecisions = 0;     // ← never changes
const approvalRate = null;   // ← never changes
const rlhfRounds = 0;        // ← never changes
const metaSats = 0;          // ← never changes
const metaImprovement = 0;   // ← never changes
const metaCorrections = 0;   // ← never changes
const rlhfRoundRates = [];   // ← always empty (no sparkline)
```

These values need to come from API endpoints. Currently there is **no backend endpoint** for RLHF decisions, approval rates, or meta-propagator corrections — those sections are **purely visual showpieces**.

---

### 3. RLHF + Meta-Propagator section — entirely fictional

**File:** [`ModelStatusV2.jsx`](file:///d:/hakckolution/hackolution/frontend/src/components/ModelStatusV2.jsx#L191-L223)

The "Meta-Propagator + RLHF" panel displays:
- "Sats with corrections: **0**"
- "Mean improvement vs SGP4: **0.0%**"
- "RLHF decisions: **0**"
- "Approval rate: **—**"
- "RLHF rounds: **0**"

There is **no backend code** that implements RLHF, stores decisions to a training loop, or measures SGP4 improvement. The `maneuver_feedback` endpoint records operator decisions but they are never fed into a retraining loop. This section exists only as a UI concept.

---

## 🟡 Medium Issues

### 4. Telemetry values fall back to `random.uniform()` noise

**File:** [`routes.py`](file:///d:/hakckolution/hackolution/backend/app/api/routes.py#L265-L280)

```python
if _TRACKER_AVAILABLE and satellite_tracker is not None:
    telemetry = satellite_tracker.get_telemetry(norad_id, state.name)
else:
    # fallback — random every call
    telemetry = {
        "fuel_remaining_pct": round(max(0.0, 85.0 + random.uniform(-5, 10)), 2),
        "battery_pct": round(88.0 + random.uniform(-3, 3), 2),
        ...
    }
```

The proper `SatelliteStateTracker` with deterministic seeding exists in [`satellite_state_tracker.py`](file:///d:/hakckolution/hackolution/backend/app/core/satellite_state_tracker.py). It replaces random calls with persistent, physically plausible state. The fallback only fires if the import fails — check that `satellite_state_tracker.py` imports cleanly on startup.

---

### 5. CascadeGraph node layout uses `Math.random()` on every re-render

**File:** [`cascadeGraph.js`](file:///d:/hakckolution/hackolution/frontend/src/utils/cascadeGraph.js#L26-L27), [`CascadeDiagram.jsx`](file:///d:/hakckolution/hackolution/frontend/src/components/CascadeDiagram.jsx#L44)

```js
x: cached?.x ?? (Math.random() * 400 - 200),   // random if not cached
y: cached?.y ?? (Math.random() * 300 - 150),
```

Graph node positions jump on every alert refresh because the cache key isn't stable. The force-directed simulation in CascadeDiagram also initializes with `Math.random()`. This is cosmetic but makes the cascade diagram unstable.

---

### 6. GAT Cascade was previously degenerate (now fixed in code, but not retrained)

**File:** [`gat_cascade.py`](file:///d:/hakckolution/hackolution/backend/app/ml/gat_cascade.py)

The docstring explicitly documents that GAT used to silently delegate to CascadeGNN because `torch_geometric` wasn't installed. The code has been rewritten as a real multi-head attention network using raw PyTorch + NumPy. However:

- The GAT **artifact** (`gat_model.json`) may not exist yet — it only trains when the metrics endpoint is first called after the rewrite
- Without PyTorch installed, it still falls back to GNN
- PyTorch is in `requirements-optional.txt`, not `requirements.txt`

---

### 7. XGBoost risk scoring uses boosted-stump fallback by default

**File:** [`xgboost_scorer.py`](file:///d:/hakckolution/hackolution/backend/app/ml/xgboost_scorer.py#L485-L502)

```python
if _extended_pipeline_enabled():   # False by default
    _ensure_trained_model()

if _using_trained and _trained_model is not None and xgb is not None:
    # Use Foster-trained XGBoost
    ...

# Falls through to legacy boosted-stump heuristic
```

With `ENABLE_EXTENDED_PIPELINE=0` (default), all conjunctions are scored by the **legacy 24-round boosted decision stump**, not the 200-tree XGBoost trained on Foster-method data. The `risk_model_trained.json` file exists but is never loaded.

---

## ℹ️ By Design (not broken, just disabled)

| Feature | Status | How to enable |
|---|---|---|
| Kafka streaming | Disabled | `KAFKA_ENABLED=1` + broker running |
| LSTM auto-retraining | Needs `ENABLE_EXTENDED_PIPELINE=1` + 500 position records | Set env var |
| Shadow-mode logging | On by default (`SHADOW_MODE_ENABLED=1`), but no data until LSTM records | Automatic |
| XGBoost (real) | Needs `ENABLE_EXTENDED_PIPELINE=1` + xgboost installed | `pip install xgboost` |
| PyTorch GAT training | Needs torch installed | `pip install torch` |

---

## ✅ What's Actually Working

| Feature | Implementation |
|---|---|
| **SGP4 orbit propagation** | Real SGP4 via `sgp4` library, thousands of satellites |
| **WebSocket 1Hz broadcast** | Real-time positions pushed to frontend |
| **Conjunction screening** | Pairwise SGP4 + Foster Pc calculation |
| **CPI scoring** | Real formula, drives alert severity |
| **Cascade planner (GNN)** | Boosted-stump risk model + graph propagation |
| **Debris model** | EVOLVE-inspired fragmentation with shells |
| **B-plane geometry** | Real Bt/Bn calculation in ThreatCard |
| **TCA countdown** | Real UTC-based countdown from alert timestamps |
| **Maneuver execution** | Delta-V applied to propagator state |
| **Hohmann transfer** | Real orbital mechanics in `/api/orbit-change` |
| **Conjunction Predictor tool** | Full SGP4 + cascade + debris pipeline |
| **Agency authority system** | Session-based authorization checks |
| **Telemetry strip** | Orbital parameters from live propagation |

---

## 🛠️ Recommended Fixes (Priority Order)

### Immediate (demo-breaking)
1. **Add `.env` to backend** with `ENABLE_EXTENDED_PIPELINE=1` — activates LSTM, XGBoost, anomaly detection
2. **Fix ModelStatusV2 hardcoded zeros** — wire `lstmSats`, `bufferFill`, `predictions` to actual API data or add new endpoints
3. **Either remove or wire RLHF section** — it currently shows fake zeros for real features

### Short-term
4. **Install optional deps**: `pip install xgboost torch` and re-test model metrics
5. **Stabilize cascade graph layout** — use deterministic NORAD-seeded node positions
6. **Add `/api/ml/status` endpoint** — aggregate LSTM buffer fill, prediction count, retrain status for the UI

### Polish
7. **Add retraining trigger UI** — a button to call `shadow_retrain` and show results
8. **Wire RLHF decisions** — store approve/reject in a counter and expose via endpoint

---

## Environment Variables Reference

Create `d:\hakckolution\hackolution\backend\.env`:
```env
ENABLE_EXTENDED_PIPELINE=1
SHADOW_MODE_ENABLED=1
SHADOW_HORIZON_MINUTES=5
SNAPSHOT_REFRESH_SECONDS=1
ALERT_REFRESH_SECONDS=30
ALLOW_UNAUTHENTICATED_COMMANDS=1
KAFKA_ENABLED=0
```
