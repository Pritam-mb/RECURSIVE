# Orbital Sentinel — Build Plan, Audit Notes & Fix Log

> Single source of truth for the audit pass. Every finding, every fix, and the
> final verification checklist live here. Update the checkbox as work lands.

---

## PART 1 — The Full Build Plan (reconstructed from repo sources)

The plan is not written in a single file. It is the union of four documents plus
the actual code, which has drifted well past all of them.

### 1.1 Sources of the plan

| Source | What it defines |
|--------|-----------------|
| `README.md` | Intended architecture, runtime flow, collision-demo flow, view layers |
| `FEATURES_MATH_LOGIC.md` | Per-subsystem math: SGP4, RSW→ECI, TCA, risk score, CPI, cascade, GNN, hotspot, Cesium |
| `PROJECT_OVERVIEW.txt` | Capability list, REST surface, data flow, *known simplifications*, future work |
| `PROJECT_STRUCTURE.md` | File-by-file responsibility map and reading order |
| `research_paper/orbital_sentinel_paper.md` | Formal methodology + **Table I defect list** + **Section VII limitations** + **VII.D research path** |

### 1.2 The intended target pipeline

```
TLE file (backend/app/simulation/tle-data.txt)
  → SGP4 propagator            sgp4_propagator.py      ECI r,v @ epoch
  → state cache                state_cache.py          1 Hz snapshot, in-memory
  → conjunction screening      conjunction.py          pairwise miss, severity
  → Foster Pc engine           analytics.py            RTN cov → B-plane → Pc (dblquad) / Chan fallback
  → CPI scorer                 conjunction.py          5-component weighted index on [0,10]
  → XGBoost risk scorer        xgboost_scorer.py       trained classifier, stump fallback
  → KD-tree graph builder      cascade_planner.py      influence radius 200 km
  → cascade rankers            gnn_cascade.py / gat_cascade.py   maneuver prob, depth
  → debris forecast            debris_model.py         NASA EVOLVE 4.0, multi-shell
  → REST + WebSocket           routes.py / ws_handler.py
  → Cesium client              CesiumGlobe.jsx / ThreatGlobe.jsx
```

### 1.3 Plan acceptance criteria (from the paper + docs)

- **AC-1** Pc must be monotonic in miss distance (Foster B-plane, not covariance trace).
- **AC-2** TCA must be localized below the coarse sample grid (Hermite refinement).
- **AC-3** Hotspot marker must sit at the refined TCA position, not the initial epoch.
- **AC-4** Debris must be generated **only** on confirmed collision (`d < 1 m`), never on a near-miss.
- **AC-5** All supplied debris forecast shells must render as distinct translucent ellipsoids.
- **AC-6** Every documented subsystem must actually be reachable at runtime.
- **AC-7** Model metrics must be reported with an explicit synthetic-only evidence boundary.
- **AC-8** Backend tests pass; frontend lints and builds clean.

### 1.4 Open items the plan itself admits

| # | Limitation (paper VII.B / overview) | Status |
|---|---------------------------------------|--------|
| L1 | Hard-body radius fixed at 10 m, no object sizes | open |
| L2 | RTN sigma model is analytical, not measurement covariance | open |
| L3 | Pairwise O(N²) screening; KD-tree not benchmarked at catalog scale | partially open |
| L4 | ML scorer train/eval share one synthetic distribution | open |
| L5 | GNN and GAT report identical metrics — suspicious degeneracy | **open, investigated below** |
| L6 | Debris is an isotropic envelope, no fragment re-entry propagation | open |
| L7 | TLE/SGP4 accuracy degrades with age; no truth ephemerides | inherent |
| L8 | No persistent storage, no auth on most routes | open |

---

## PART 2 — Audit Findings

Baseline verification run before any change:

- `pytest tests -q` → **64 passed**
- `npm run lint` → **0 errors, 6 warnings**
- `python -c "import main"` → OK
- `uvicorn main:app` boot → loads 60 TLEs, "Application startup complete"

So the suite was green while the product was not working. That is the core
theme of this audit: **the tests do not cover the runtime wiring.**

### P1 — CRITICAL: the entire alert pipeline never runs by default

- `backend/main.py:48` → `ENABLE_EXTENDED_PIPELINE = os.getenv("ENABLE_EXTENDED_PIPELINE", "0") == "1"`
- `backend/main.py:153` → `if not ENABLE_EXTENDED_PIPELINE: return` at the top of `refresh_alerts_once()`

Consequence: with default config, conjunction screening, cascade planning,
hotspot generation and the EVOLVE debris model are **all skipped**. `/api/alerts`
permanently serves the empty stub written at `backend/main.py:314`, and never
refreshes. Every downstream consumer — `AlertPanel`, `CascadeDiagram`,
`ThreatGlobe`, `ConjunctionPredictor`, `MissionControlCenter` — renders nothing.

Root cause: the flag was introduced to avoid eagerly constructing the heavy ML
runtime and Kafka producer (`backend/main.py:53-58`). That intent is correct. But
the guard was placed at the *top of the alert refresh function* instead of around
only the ML/Kafka calls. The flag name and the test in
`backend/tests/test_ml_feature_flag.py` both scope it to ML only.

Violates: **AC-6**, and destroys the entire product surface.

### P2 — Investigated, NOT a duplicate (finding corrected)

The audit initially flagged two debris models as duplicated logic:

- `backend/app/core/debris_model.py` (19 KB) — NASA EVOLVE fragment generation, per-fragment drag propagation, cloud storage, `get_frontend_debris_clouds`
- `backend/app/services/debris_model.py` (8.8 KB) — `build_debris_alerts`, isotropic forecast envelopes with radius timelines and affected-satellite risk bands

**Verdict: false positive — this is a legitimate two-layer split, not duplication.**
`core/` is the physics engine (per-fragment state, drag integration, only
populated when a collision is actually simulated). `services/` is the predictive
alert layer (imminent conjunctions above the CPI threshold, multi-shell
forecast, per-satellite risk bands). They have different inputs, different
outputs and different consumers, and the paper documents both. No code change
was made and no consolidation is warranted.

However, investigating this pair uncovered **P14**, which is a real defect.

### P14 — Found during P2 investigation: forecast debris clouds were never served

`build_debris_alerts()` runs on every alert refresh and its result was cached at
`main.py:324` as `alerts_payload["debris_clouds"]`. But the consumer,
`GET /api/alerts` at `routes.py:308`, read:

```python
debris_clouds = snapshot.get("debris_clouds", [])
```

i.e. only the **snapshot's** fragment clouds. The alert cache's
`debris_clouds` entry was never read by any caller. So on every refresh cycle
the planner built a full debris forecast — CPI-gated hotspots, RK45 radius
integration, KD-tree affected-satellite search, multi-shell timelines — and the
result was silently thrown away.

The practical consequence is that paper acceptance criterion **AC-5** (all
supplied forecast shells rendered as distinct translucent ellipsoids) was
unreachable: the shells only exist in the forecast layer, and the forecast layer
was never on the wire.

Two sources are genuinely needed and are now both served, tagged with
`debris_source` (`forecast` / `fragment`) and accompanied by a `debris_sources`
list naming which sources the payload carries. Because the two refresh on
different cadences — fragment radii grow every second, forecast clouds only
change when the alert cache is recomputed — the frontend store now keeps them in
separate buckets (`debrisBySource`) and merges on read. Replacing the whole list
per frame would have dropped whichever source did not ride along, so this was
coordinated across `ws_handler.py`, `routes.py`, `useStore.js` and `App.jsx`.

### P3 — GNN and GAT cascade rankers are degenerate

`gnn_cascade.py` and `gat_cascade.py` are separate files, but the cached metrics
in `model_metrics_cache.json` are byte-identical (98.76 / 98.46 / 98.69 / 98.57).
Paper section VI.D and VII.C both call this out as an unresolved threat to
validity. Needs to be either genuinely differentiated or explicitly documented
as a single shared surrogate.

### P4 — Frontend bypasses its own API layer

`frontend/src/utils/api.js` exports `apiGet` / `apiPost` / `apiDelete` with
consistent error handling. `App.jsx:65,79,106` and other components call raw
`fetch('/api/...')` instead, so error semantics and base-URL handling are
inconsistent across the codebase.

### P5 — Six react-hooks warnings, three of them real bugs

Three are the stale-ref-in-cleanup class, which is a genuine bug when the ref
holds a live Cesium primitive:

- `frontend/src/components/Globe/CesiumGlobe.jsx:885` — `motionRef.current` in cleanup
- `frontend/src/components/ThreatGlobe.jsx:507` — `stateRef.current` in cleanup
- `frontend/src/components/CascadeDiagram.jsx:248` — `simRef.current` in cleanup

Three are missing-dependency warnings that indicate stale closures:

- `frontend/src/components/Globe/CesiumGlobe.jsx:254` — missing `mode`, `onSatelliteSelect`
- `frontend/src/components/ThreatGlobe.jsx:482` — missing 6 deps
- `frontend/src/components/BPlaneDiagram.jsx:44` — missing `CENTER`

### P6 — Dead statement in the WebSocket handler

`backend/main.py:393` → `data = await websocket.receive_text()` — `data` is never
read. Harmless but it is dead code in the keep-alive path.

### P7 — Documentation is badly stale

`PROJECT_STRUCTURE.md` and `FEATURES_MATH_LOGIC.md` describe a tree of ~50 files.
The repo now has ~120 source files. Neither document mentions: `analytics.py`,
`agency.py`, `agency_authority.py`, `kalman.py`, `satellite_state_tracker.py`,
`anomaly.py`, `shadow_mode.py`, `shadow_logger.py`, `gat_cascade.py`,
`model_metrics.py`, `train_models.py`, `train_xgboost.py`, `synthetic_data.py`,
`predict.py`, `conjunction_predictor.py`, `frames.py`, `sim_clock.py`,
`kafka_adapter.py`, `artifacts/`, or the whole `backend/tests/` suite. On the
frontend: `ThreatGlobe.jsx`, `MissionControlCenter.jsx`, `ModelStatusV2.jsx`,
`UplinkDownlinkV2.jsx`, `CascadeDiagram.jsx`, `BPlaneDiagram.jsx`,
`TelemetryStrip.jsx`, `ConjunctionPredictor.jsx`, `api.js`, `tcaCountdown.js`.

Also note the doc contradiction: `FEATURES_MATH_LOGIC.md:141` points at
`backend/app/ml/xgboost_scorer.py` for "Risk Scoring and CPI Math" and describes a
deterministic weighted heuristic, while the paper documents a trained XGBoost
classifier on Foster features. The docs describe the *old* system.

### P8 — Found during FIX-1 verification: alert list and count disagree

Once the pipeline was running, `GET /api/alerts` returned `count: 10` with an
empty `alerts` list. Two different sources were being mixed:

- `count` came from `len(alerts)` — the **screened** conjunctions (500 km watch radius)
- `alerts` came from `cascade_summary["alerts"]` — the **graph edges** (200 km influence radius)

Consequences: the alert badge showed a non-zero count over an empty list, and
every conjunction between 200 km and 500 km was invisible to the operator. The
frontend also could not recover the missing relative TCA, because
`ConjunctionEvent.to_dict()` exposed only the absolute `tca_utc` string.

Fixed by `merge_alert_sources()` in `backend/main.py`, which unions both sources,
lets the richer graph entry win on duplicate pairs in either orientation, and
normalizes screened alerts onto the graph-edge shape the frontend reads.
`count` is now always `len()` of the emitted list. The payload also reports
`screened_count` and `graph_alert_count` separately for diagnostics.

### P9 — Found during FIX-1: TCA urgency measured against wall clock

`backend/app/core/conjunction.py:425` computed

```python
tca_hours = max(0.0, (tca_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0)
```

The screening loop is driven by `app/core/sim_clock.py`, so under any time warp
the CPI "TCA urgency" term was measured against the wrong reference epoch, and
under test runs it drifted arbitrarily. The snapshot epoch (`s1.epoch_utc`) is
the correct reference and is now used. `ConjunctionEvent` also gained a
`tca_hours` field so the relative TCA reaches the API instead of forcing the
frontend to re-derive it from a timezone-sensitive string.

### P10 — Investigated, NOT a bug (false positive)

An initial read suggested `backend/app/core/conjunction.py` had a corrupted
comment (a "Geometry helpers" banner rendering as replacement characters).

**Verdict: false positive.** The file decodes cleanly as UTF-8 and contains no
U+FFFD. The corruption was an artifact of the Windows cp1252 console being
unable to print the box-drawing characters in the section banner. A repo-wide
scan of every `.py/.js/.jsx/.json/.md/.txt/.css/.html` file (excluding
`node_modules`, `dist`, `__pycache__`, `.git`) found **zero** files with invalid
UTF-8 or embedded replacement characters.

No code change was made. Recorded here so the finding is not re-investigated.

### P11 — Found during FIX-6: alerts were never pushed over the WebSocket

`frontend/src/App.jsx` had a live branch reading a streamed alert block:

```js
if (Object.prototype.hasOwnProperty.call(data, 'alerts')) { setAlerts(data.alerts || []); }
```

but `app/api/ws_handler.py` only ever emitted `type`, `timestamp`, `satellites`,
`count` and `debris_clouds`. That branch was **dead code**. Alerts therefore only
reached the UI through the 30-second REST poll in `App.jsx`, even though the
backend had already recomputed them. The same applied to `hotspots` and
`cascade_plan`, so the alert panel, hotspot rings and cascade diagram could all
be up to 30 s stale while the globe looked live.

### P12 — Found during FIX-6: WebSocket socket leak

`websocket_satellites()` in `main.py` only called `ws_manager.disconnect()` on
`WebSocketDisconnect`. Any other exception (client reset, malformed frame,
network error) left the dead socket in `ConnectionManager.active_connections`
forever, and it was never garbage collected because the manager holds a
reference. Repeated reconnects would grow the list without bound, and the
broadcast loop would retry every dead socket each tick.

### P13 — Found during FIX-6: broadcast payload dropped alerts when there were none

The first attempt at FIX-6 reintroduced a variant of the P8 class of bug: the
alert block was attached only on the enriched path, so a *cleared* alert set
would never be pushed and the UI would keep rendering stale rows. Caught before
landing; the final implementation always builds the payload object and gates only
the attachment of the alert block on the cache stamp.

### P15 — Found during FIX-3: `CascadeGNN` was not in the production path

Original finding: `cascade_planner.py:113` read

```python
self.gnn = CascadeGAT()
```

The attribute was named `gnn` but held a `CascadeGAT`, and the result was assigned
to a local called `gnn_output`. `CascadeGNN` was constructed only in
`model_metrics.py` and in tests. Consequences, same class of problem as the P3
degeneracy:

- The paper and status panel presented "Graph Cascade (GNN)" and "GAT Cascade"
  as two models, so the 98.76% GNN row described a model that scored nothing and
  the two rows were never corroborating each other.
- The misleading name invited the reverse bug: a reader of `self.gnn` would
  reasonably assume `gnn_cascade.py` was live.

**Status: FIXED by FIX-9.** See FIX-9 for the design and the reasoning.

### P16 — Found during FIX-9: ranker failure silently became an all-zero plan

`cascade_planner.py` previously wrapped the entire scoring step in one
`try/except`:

```python
try:
    ...
    maneuver_probability = gnn_output["maneuver_probability"]
    gnn_depth = gnn_output["cascade_depth"]
except Exception as error:
    logger.warning("GNN cascade scoring failed, using fallback: %s", error)
    maneuver_probability = np.zeros(len(nodes), dtype=float)
    gnn_depth = np.zeros(len(nodes), dtype=int)
```

Any ranker error — corrupt artifact, wrong shape, missing torch, a bug — produced
`np.zeros` for **both** manoeuvre probability and cascade depth, and the pipeline
continued. The emitted plan was structurally valid and completely fabricated:
every probability `0.0`, every depth `0`, `degraded` nowhere, exit code normal.

The operators' two likely failure signals were both suppressed:

- `risk_after = max(0.0, risk_before - (node_prob * 3.0) - (total_delta_v_ms * 0.6))`
  so the fabricated plan *inflated* the apparent post-manoeuvre safety, understating
  residual risk.
- `ranked_nodes` fell back to sorting on peak CPI alone, so the plan looked
  plausible rather than obviously empty.

Only a `WARNING` line in the log distinguished the fabricated plan from a real
one. There was no way to tell from the API response.

**Status: FIXED by FIX-9.**

---

## PART 3 — Fix Log

### FIX-1 — Alert pipeline restored (fixes P1)

- **Files:** `backend/main.py`, `backend/tests/test_alert_pipeline_wiring.py` (new)
- **Change:** Removed the blanket early-return from `refresh_alerts_once()`. The
  `ENABLE_EXTENDED_PIPELINE` flag now guards only the ML-runtime and Kafka calls,
  which is what it was designed for. Screening, cascade planning, hotspot
  generation and debris building now run unconditionally.
- **Verified:** `GET /api/alerts` on default config went from a permanent empty
  stub to 7 real conjunctions.
- **Status:** DONE

### FIX-1b — Alert list/count consistency (fixes P8)

- **Files:** `backend/main.py`, `backend/app/core/conjunction.py`
- **Change:** added `merge_alert_sources()`; `count` derived from the emitted
  list; `screened_count` / `graph_alert_count` added for diagnostics; screened
  alerts normalized onto the graph-edge shape.
- **Verified:** `count: 7` with `len(alerts) == 7`.
- **Status:** DONE

### FIX-1c — TCA urgency epoch reference (fixes P9)

- **Files:** `backend/app/core/conjunction.py`
- **Change:** `tca_hours` measured from `s1.epoch_utc` instead of
  `datetime.now(timezone.utc)`; `ConjunctionEvent` gained `tca_hours`, and
  `to_dict()` now emits `tca_hours` / `tca_minutes` / `p_collision`.
- **Status:** DONE

### FIX-6 — WebSocket keep-alive cleanup (fixes P6, P12)

- **Files:** `backend/main.py`
- **Change:** the inbound frame is logged at debug level instead of being bound
  to an unused local; `disconnect()` is now called on any exception, not only
  `WebSocketDisconnect`, so a reset client cannot leak a socket.
- **Status:** DONE

### FIX-7 — Alerts pushed over the WebSocket (fixes P11, P13)

- **Files:** `backend/app/api/ws_handler.py`, `frontend/src/App.jsx`,
  `backend/tests/test_ws_broadcast_contract.py` (new)
- **Change:** the broadcast frame now carries `alerts`, `alert_count`,
  `hotspots`, `cascade_plan`, `cascade_depth`, `graph` and `alerts_timestamp`,
  attached only when the alert cache stamp changes so the 1 Hz payload stays
  small. A cleared alert set is still pushed. `App.jsx` now consumes the
  streamed hotspots and cascade plan, not just the alert list.
- **Status:** DONE

### FIX-8 — Both debris sources served (fixes P14, restores AC-5)

- **Files:** `backend/app/api/routes.py`, `backend/app/api/ws_handler.py`,
  `frontend/src/store/useStore.js`, `frontend/src/App.jsx`,
  `backend/tests/test_ws_broadcast_contract.py`
- **Change:** `/api/alerts` and the WebSocket now both emit forecast *and*
  fragment clouds, tagged by `debris_source` with a `debris_sources` list; the
  store buckets them by source and merges on read so neither cadence erases the
  other.
- **Status:** DONE

### FIX-2 — Single debris model (fixes P2)

**WITHDRAWN.** P2 was a false positive: `core/debris_model.py` and
`services/debris_model.py` are a legitimate physics/alert layer split, not
duplicate implementations. No consolidation performed. The investigation instead
produced FIX-8 (P14), which was the actual defect in this area.

### FIX-3 — GAT cascade ranker was a silent alias of the GNN (fixes P3)

- **Status:** DONE (verified, not asserted)
- **Files:** `backend/app/ml/gat_cascade.py`, `backend/app/ml/model_metrics.py`,
  `backend/tests/test_gat_cascade.py` (new), `backend/app/ml/artifacts/gat_model.json` (new),
  `frontend/src/components/ModelStatusV2.jsx`,
  `research_paper/orbital_sentinel_paper.md`

**Root cause.** `gat_cascade.py` gated its entire implementation behind an import of
`torch_geometric`, which appears in neither `requirements.txt` nor
`requirements-optional.txt` and is not installed. The import failed, so
`TORCH_AVAILABLE` was `False` and every `CascadeGAT.predict()` call silently
delegated to `CascadeGNN.predict()`. The two models in the paper's table were
therefore one model presented twice. The identical metrics were not a coincidence
of similar decision surfaces, as the paper speculated.

**Fix.** Replaced the module with a real multi-head graph attention network:

- attention is a softmax over each target node's in-edges **plus its own projected
  state**, so an isolated node gets full self-attention instead of collapsing to a
  constant, and a connected node can learn how much of itself to keep;
- per-head projections and attention vectors, heads concatenated, ELU before a
  linear readout;
- weights are gradient-trained with **PyTorch autograd only** — `torch_geometric`
  is no longer needed anywhere, which removes the root cause rather than papering
  over it — and exported to JSON so runtime inference is deterministic NumPy;
- `CascadeGAT.predict()` now **raises** if no attention bank is available instead
  of falling back to the GNN, so a degenerate model can never again be reported
  as independent evidence.

**Why self-attention mattered.** The first working version scored nodes only from
neighbour messages. The synthetic target is a function of *both* a node's own
peak-CPI/altitude *and* its neighbourhood, so that version scored 48.81% — worse
than chance. Putting the node's own state inside the attention softmax and adding
the ELU nonlinearity is what made it trainable.

**Verification (all measured, not assumed):**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Graph Cascade (GNN) | 98.76% | 98.46% | 98.69% | 98.57% |
| GAT Cascade | 95.54% | 94.76% | 94.98% | 94.87% |

- Torch and NumPy forward passes agree to `3.4e-07` (float32 precision).
- Max per-graph `|GNN − GAT|` maneuver-probability gap: `0.349`.
- `CascadeGAT().is_degenerate` is `False`; 21 new tests in `test_gat_cascade.py`,
  104 backend tests total.

**Collateral fixes found while doing this:**

- `model_metrics.py` only invalidated its cache on `risk_model_path`, so swapping
  the GAT implementation left the stale identical metrics on screen. Added
  `_artifact_signature()` over the risk, graph, GAT and trajectory artifacts.
- The metrics payload now carries a per-model `independent` flag and a
  `graph_attention` status block, so a future alias is visible in the API rather
  than only in a diff.
- `ModelStatusV2.jsx` had `const torchGeo = null;` hardcoded and therefore always
  rendered "NOT INSTALLED" for a dependency it never queried. That row is replaced
  with the real trainer, heads, hidden dim and per-model accuracy, and the badge
  now shows "GAT DEGENERATE" if the flag ever trips.
- **Risk Scorer accuracy is 75.46%, not the 93.85% the paper previously claimed.**
  `_evaluate_risk_model` scores through the production `XGBoostScorer.score`
  entry point, so the number describes the model actually serving alerts; the old
  figure described a legacy boosted-stump model that was never deployed. Recall of
  44.64% is a real weakness and the paper now says so.

### FIX-4 — Frontend uses one API layer (fixes P4)

- **Files:** `frontend/src/App.jsx` and any other raw `fetch('/api/...')` callers
- **Change:** all call sites route through `utils/api.js`.
- **Status:** NOT STARTED

### FIX-5 — Hook warnings cleared (fixes P5)

- **Files:** `CesiumGlobe.jsx`, `ThreatGlobe.jsx`, `CascadeDiagram.jsx`, `BPlaneDiagram.jsx`
- **Change:** ref values copied to locals inside effects before use in cleanup;
  missing dependencies added or intentionally captured via refs with an
  explanatory comment.
- **Status:** NOT STARTED

### FIX-7 — Docs resynced (fixes P7)

- **Files:** `PROJECT_STRUCTURE.md`, `FEATURES_MATH_LOGIC.md`, `PROJECT_OVERVIEW.txt`
- **Change:** file tree, component map, math sections and endpoint list updated to
  match the code as it now stands.
- **Status:** PARTIAL — the paper's GAT/rankers/metrics sections and the
  `graph_attention` payload are updated as part of FIX-3. The three top-level
  project docs are still outstanding.

### FIX-9 — Ranker made honest (fixes P15, P16)

**Files:** `backend/app/services/cascade_planner.py`, `backend/app/api/routes.py`,
`backend/app/api/ws_handler.py`, `backend/main.py`, `frontend/src/store/useStore.js`,
`frontend/src/App.jsx`, `frontend/src/components/MissionControlCenter.jsx`,
`backend/tests/test_ranker_wiring.py`

**P15 design decision.** The audit rejected the lazy options: renaming the
attribute alone would have hidden the finding, and dropping the GNN metrics row
would have destroyed a working, higher-accuracy model. `CascadeGNN` is now wired
in as a deliberate **cross-check**. Two properties make the scores non-fusable and
so rule out ensembling them:

1. They share `_build_gnn_inputs`, so they are not independent evidence.
2. They score different inductive biases (GAT attention vs GCN aggregation) on
   identical features, so averaging two differently-biased numbers yields a value
   neither model supports.

The GAT therefore remains the sole production ranker. `CascadeGNN` runs beside it
only to report disagreement, and its output can never change the plan.

- `self.gnn` → `self.primary_ranker`; added `self.cross_check_ranker`.
- New `ranker_review` block on the result and on `/api/alerts`, the WebSocket
  payload, and the alert cache: `primary`, `cross_check`, `primary_ok`,
  `cross_check_ok`, `fallback_used`, `degraded`, `degraded_reason`,
  `disagreement_threshold`, `disagreement_count`, `max_disagreement`, and
  `disputed_satellites` (with both raw scores and the delta).
- `RANKER_DISAGREEMENT_THRESHOLD = 0.25`; satellites above it are listed for
  operator review. Observed max delta on the 120-node live snapshot is 0.211,
  below the threshold, so a healthy run is silent.
- Local renames `gnn_depth` → `ranker_depth` at both the definition and the
  `_resolve_cascade` call site, so no trace of the old misnaming remains.

**P16 fix.** Failures are now typed and surfaced, never fabricated:

- Primary fails, cross-check healthy → cross-check scores, `fallback_used: true`
  and `degraded: true` with a `degraded_reason`, and a red RANKER indicator in
  the UI. Real numbers, explicitly marked as not attention-derived.
- Both fail → `RuntimeError`. This propagates to `refresh_alerts_once`, which
  logs and leaves the previous alert cache intact, so the operator keeps seeing
  the last real plan rather than a fresh fabricated one.
- Cross-check fails, primary healthy → no degradation; the plan is unaffected.
- Graph-input construction failure → `RuntimeError` instead of a zero plan.

Also fixed a latent `KeyError('node_index')`: `build_graph`'s empty-graph return
omitted the key that `analyze_snapshot` reads *before* its own `if not nodes`
guard, so passing an empty or all-error state list raised.

**UI.** A RANKER status dot joins BACKEND/PIPELINE/ML/PHYSICS (red when degraded,
amber when nodes are disputed), with a plain-language note underneath naming the
count and max delta and stating that GAT scores were used.

- **Tests:** 12 new in `backend/tests/test_ranker_wiring.py` covering ranker
  identity, absence of the misleading attribute, all four failure permutations,
  no-all-zero fallback, sub-threshold silence, key-set parity between the empty
  and populated return paths, and — importantly — that a published probability is
  never a blend of the two models' outputs.
- **Status:** DONE. 116 tests pass, lint 0 errors, build passes.

---

## PART 4 — Final TODO / Verification Checklist

### Code fixes

- [x] P1 — alert pipeline no longer gated behind `ENABLE_EXTENDED_PIPELINE`
- [x] P2 — investigated: false positive, legitimate layer split, no change needed
- [x] P3 — GAT is a trained attention network, metrics no longer identical
- [ ] P4 — frontend routes all HTTP through `utils/api.js`
- [ ] P5 — 0 eslint errors, 0 warnings
- [x] P6 — no dead statements in the WebSocket keep-alive path
- [ ] P7 — docs match the code (paper done via FIX-3; three project docs outstanding)
- [x] P8 — `count` always equals `len(alerts)`; both sources unioned
- [x] P9 — TCA urgency measured from the snapshot epoch
- [x] P10 — investigated: false positive (cp1252 console artifact), no code change
- [x] P11 — alerts/hotspots/cascade pushed over the WebSocket
- [x] P12 — WebSocket socket leak on non-disconnect errors closed
- [x] P13 — cleared alert set is pushed, not suppressed
- [x] P14 — both debris sources served; forecast shells reach the client (AC-5)
- [x] P15 — `CascadeGNN` is now in the request path as a cross-check that reports
  disagreement without ever contributing to the plan; the misleading `self.gnn`
  attribute is gone (FIX-9)
- [x] P16 — ranker failure no longer silently produces an all-zero plan; failures
  raise or are marked `degraded`, and the condition reaches the UI (FIX-9)

### Verification (re-run after every fix)

- [x] `pytest tests -q` — all green, count recorded below
- [x] `npm run lint` — clean (0 errors, 6 warnings)
- [x] `npm run build` — production bundle builds
- [x] `GET /api/alerts` returns a populated, non-stub payload on default config
- [x] `GET /api/satellites` returns live positions
- [ ] `GET /` reports operational status

### Known-open limitations (carried forward, not regressions)

- [ ] L1 per-object hard-body radii
- [ ] L2 measurement-based covariance
- [ ] L3 catalog-scale screening benchmark
- [ ] L4 temporal train/test split for the risk scorer
- [ ] L6 fragment-level debris re-entry propagation
- [ ] L8 persistent storage and route auth

### Verification log

| # | Command | Result |
|---|---------|--------|
| 1 | `pytest tests -q` (baseline) | 64 passed |
| 2 | `npm run lint` (baseline) | 0 errors, 6 warnings |
| 3 | `import main` (baseline) | OK |
| 4 | `uvicorn` boot (baseline) | 60 TLEs loaded |
| 5 | `pytest tests -q` (post-FIX-1) | 72 passed |
| 5b | `pytest tests -q` (post-FIX-7/8) | 83 passed |
| 6 | `npm run lint` (post-fix) | PENDING |
| 7 | `npm run build` (post-fix) | PENDING |
| 8 | `GET /api/alerts` (post-FIX-1) | PASS — 120 sats, 7 alerts, count==len, graph 120 nodes |
| 9 | `GET /api/satellites` (post-FIX-1) | PASS — 120 satellites, live position+velocity |
| 10 | `GET /` (post-FIX-1) | PASS — operational, 120 satellites |
| 11 | `python -m app.ml.train_models --gat` | PASS — writes the attention artifact |
| 12 | torch vs NumPy forward parity | PASS — max abs diff 3.4e-07 |
| 13 | GAT vs GNN probability gap | PASS — max 0.349 over 24 graphs, not an alias |
| 14 | `pytest tests -q` (post-FIX-3) | 104 passed (83 + 21 new) |
| 15 | metrics regeneration | GAT 95.54% acc, independent=True; Risk Scorer 75.46% |
| 16 | `CascadeGNN` reachability scan | metrics/tests only — **not** in the request path (P15) |
| 17 | `pytest tests -q` (post-FIX-9) | 116 passed (104 + 12 new) |
| 18 | `npm run lint` (post-FIX-9) | 0 errors, 6 warnings (unchanged baseline) |
| 19 | `npm run build` (post-FIX-9) | PASS — 65 modules, 269.25 kB |
| 20 | `GET /api/alerts` (post-FIX-9) | PASS — 120 sats, 9 alerts, 120/120 non-zero probabilities, `ranker_review` present with both rankers healthy, max delta 0.211 < 0.25 |
| 21 | ranker failure matrix (unit) | PASS — both-fail raises; primary-fail uses cross-check + `degraded`; cross-check-fail does not degrade; empty graph does not invoke either ranker |

### Note on the removal of `torch_geometric`

`gat_cascade.py` no longer imports `torch_geometric` at all, so it must **not** be
added to `requirements-optional.txt` to "fix" the original problem — the
dependency was the bug, not a missing requirement. Training needs only `torch`,
which is already in `requirements.txt`. Runtime inference needs only NumPy.
