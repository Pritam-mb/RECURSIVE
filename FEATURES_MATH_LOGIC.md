# Features, Math, and Logic Map

This document explains what the system does, what math each area uses, and how the main ranking and hotspot logic works.

## 1. System Features

### Backend Features

- Satellite propagation from TLEs with SGP4.
- Live snapshot caching for positions, velocities, and timestamps.
- Pairwise conjunction screening with severity levels.
- Maneuver execution with delta-V burns in RSW or ECI frames.
- Burn safety checks to avoid unsafe reentry trajectories.
- Cascade planning for multi-satellite risk chains.
- Graph-based ranking of risky satellites and edges.
- Shared alert cache so polling, WebSocket updates, and maneuver responses stay aligned.

### Frontend Features

- Cesium globe visualization of live satellite motion.
- Smooth extrapolation between backend snapshots using position and velocity.
- Selection and hover highlighting for satellites.
- Orbit track rendering for the selected object.
- Conjunction alert sidebar with risk and cascade summaries.
- Manual maneuver panel with preflight and execution flow.

## 2. Data Flow

1. TLEs are loaded into the SGP4 propagator.
2. The propagator produces ECI position and velocity for each satellite.
3. The backend stores a snapshot with timestamp, position, velocity, altitude, and speed.
4. Conjunction screening checks each satellite pair for close approaches.
5. The cascade planner builds a proximity graph and ranks the highest-risk nodes.
6. The frontend reads the snapshot and alerts, then renders satellites on Cesium.
7. When a maneuver is executed, the burn is applied, the orbit is rebuilt, and the cache is refreshed immediately.

## 3. Orbit Propagation Math

The orbit core is in [backend/app/core/sgp4_propagator.py](backend/app/core/sgp4_propagator.py).

### What it does

- Converts TLE lines into a `Satrec` object.
- Uses SGP4 to propagate to a requested epoch.
- Returns ECI position `(x, y, z)` in km and velocity `(vx, vy, vz)` in km/s.
- Applies impulsive delta-V burns and rebuilds the satellite state so future propagation follows the new orbit.

### Math used

- ECI position and velocity are propagated by SGP4, which already includes major perturbation terms like J2 and drag through the TLE model.
- Satellite speed is computed by Euclidean norm:

  `speed_kmh = sqrt(vx^2 + vy^2 + vz^2) * 3600`

- Altitude is approximated from radius magnitude:

  `altitude_km = sqrt(x^2 + y^2 + z^2) - 6371.0`

### Burn rebuild logic

When a maneuver is applied:

- If the burn is in RSW, the delta-V vector is converted into ECI.
- The updated velocity is computed as:

  `v_new = v_old + dv_eci / 1000`

- The new state is converted into Keplerian elements using:
  - specific angular momentum `h = r x v`
  - eccentricity vector
  - orbital energy
  - semi-major axis `a = -mu / (2 * energy)`
- The burn is rejected if the new perigee would drop below 100 km.
- If the burn is safe, a new `Satrec` is rebuilt so future propagation uses the updated orbit.

### Why this matters

This is what makes the burn real. The maneuver is not just logged; it changes the actual future orbit state used by propagation and the dashboard.

## 4. RSW to ECI Conversion

Also in [backend/app/core/sgp4_propagator.py](backend/app/core/sgp4_propagator.py).

RSW means:

- R = radial direction, along the position vector.
- S = along-track direction, tangent to motion.
- W = cross-track direction, along the orbit normal.

The basis is built from:

- `R_hat = r / ||r||`
- `W_hat = (r x v) / ||r x v||`
- `S_hat = W_hat x R_hat`

The conversion matrix is built from those three unit vectors, then the burn vector is rotated into ECI space.

This math is used for maneuver input, burn validation, and cascade recommendation output.

## 5. Conjunction Screening Math

The conjunction engine is in [backend/app/core/conjunction.py](backend/app/core/conjunction.py).

### Pairwise miss distance

For two satellites with position vectors `r1` and `r2`:

`miss_distance_km = ||r1 - r2||`

This is a straight Euclidean distance in km.

### Relative speed

For velocities `v1` and `v2`:

`relative_speed_kmh = ||v1 - v2|| * 3600`

The `* 3600` converts km/s to km/h.

### Severity thresholds

- `<= 50 km` -> red
- `<= 200 km` -> yellow
- `<= 500 km` -> green
- above that -> no alert

These thresholds are simple and deterministic. They are used for the alert list and for sorting critical approaches.

### Time of closest approach estimate

The planner uses a linear relative-motion approximation:

`t = - (r_rel dot v_rel) / ||v_rel||^2`

Then it clamps negative time to zero and computes the closest approach distance with the same linear model.

This is a lightweight approximation, not a full nonlinear conjunction solver.

## 6. Risk Scoring and CPI Math

The risk scorer is in [backend/app/ml/xgboost_scorer.py](backend/app/ml/xgboost_scorer.py).

### Current behavior

The code keeps an XGBoost-like interface, but the live score is a deterministic heuristic so the result is stable across refreshes.

### Inputs used

- miss distance
- relative speed
- altitude of both objects
- time to closest approach

### Heuristic score

The score is a weighted mix of normalized terms:

- `miss_component = max(0, 1 - miss_dist / 500)`
- `speed_component = min(relative_speed_kmh / 20000, 1)`
- `tca_component = 1 / (1 + tca_minutes / 60)`
- `altitude_component = max(0, 1 - combined_altitude / 2000)`

Final score:

`score = 0.50 * miss_component + 0.20 * speed_component + 0.15 * tca_component + 0.15 * altitude_component`

The output is clipped to `[0, 1]`.

### CPI calculation

The cascade planner turns the risk into a Conjunction Priority Index:

`CPI = 0.40 * probability_component + 0.25 * miss_component + 0.20 * velocity_component + 0.15 * time_component`

Where:

- `probability_component = clamp(p_collision * 10, 0, 10)`
- `miss_component = max(0, 10 - miss_distance_km / 5)`
- `velocity_component = min(relative_velocity_kms / 1.5, 10)`
- `time_component = max(0, 10 - tca_minutes / 2.4)`

### Why there are two scores

- The heuristic risk score is a normalized probability-like value.
- CPI is the planner priority score used to rank which objects should be handled first.

## 7. Cascade Planner Math and Logic

The cascade planner is in [backend/app/services/cascade_planner.py](backend/app/services/cascade_planner.py).

### Graph construction

The planner builds a proximity graph from the live satellite snapshot.

- It uses `cKDTree` to find neighbors inside an influence radius.
- The default radius is `200 km`.
- Each pair becomes a graph edge with risk metadata.

This keeps graph generation fast even when the satellite count grows.

### Edge metrics

For each candidate pair the planner stores:

- actual miss distance
- predicted miss distance from linear TCA
- relative velocity
- TCA
- collision probability
- CPI score
- severity
- influence weight

Influence weight is:

`influence_weight = 1 / max(predicted_miss_distance_km, 1)`

So closer threats matter more.

### Node features for graph ranking

Each node gets a feature vector with:

- position components
- velocity components
- speed
- altitude
- agency id
- peak CPI from its strongest neighbor

These are fed into the lightweight graph ranker.

## 8. Graph Ranker Math

The graph ranker is in [backend/app/ml/gnn_cascade.py](backend/app/ml/gnn_cascade.py).

### What it is

This is not a trained neural network. It is a deterministic graph-attention style scorer that behaves like a small GNN surrogate.

### Edge attention

For each edge it computes an attention value using:

- collision probability
- miss distance
- relative velocity
- TCA

The core idea is that risk rises when probability is high, distance is small, speed is high, and TCA is soon.

### Influence accumulation

Each node collects influence from its neighbors.

The source node gets part of the weight and the target node gets the full weight, so the graph can model risk spread and downstream cascade pressure.

### Maneuver probability

The node logits are built from:

`logits = 1.35 * influence + 0.25 * local_risk + 0.10 * altitude_term`

Then a sigmoid converts logits into a probability:

`maneuver_probability = sigmoid(logits - 0.75)`

### Cascade depth estimate

The estimated depth is:

`cascade_depth = ceil(maneuver_probability * 4)`

clipped to the range `0..4`.

## 9. Maneuver Recommendation Logic

The planner chooses a maneuver for the strongest threat on each node.

### RSW burn magnitude

The recommended burn magnitude is based on local CPI and maneuver probability:

`magnitude = clamp(0.15 + (local_cpi / 10) * 1.4 + maneuver_probability * 0.7, 0.05, 2.5)`

Then the burn is split into components:

- radial = 25% of magnitude
- along-track = 85% of magnitude
- cross-track = 15% of magnitude

The sign of each component is chosen from the relative geometry between the host satellite and the threat.

### Risk-after estimate

The planner also computes a simple post-burn risk estimate:

`risk_after = max(0, risk_before - (maneuver_probability * 3.0) - (delta_v * 0.6))`

This is a heuristic control estimate, not a physics solver.

### Cascade queue logic

The planner seeds the queue from the highest-risk alerts or the highest-ranked graph nodes.

It then expands outward while:

- cascade depth is within the max depth
- edge CPI is above the threshold, or
- target maneuver probability is high enough

Default planner limits:

- `cpi_threshold = 5.0`
- `max_depth = 3`
- `max_maneuvers = 50`

## 10. Hotspot Calculation

There is no separate single `hotspot_score` function in the repo right now. Hotspots are derived from the strongest risk indicators already computed by the planner.

### What counts as a hotspot

A hotspot is a satellite or pair that has:

- high CPI
- high collision probability
- low predicted miss distance
- high influence weight
- high maneuver probability
- high peak CPI in its neighborhood

### Practical hotspot ranking used by the app

The app effectively ranks hotspots by a combination of:

- `cpi_score` on edges
- `p_collision`
- `node_peak_cpi`
- `maneuver_probability`
- graph density inside the influence radius

### Why this works

The best hotspots are not just the closest objects. They are the objects that are both geometrically close and graph-central, meaning they can trigger more cascade reactions.

### Where hotspot information appears

- In the alert panel as the most critical conjunctions.
- In the cascade plan as the first satellites to maneuver.
- In the graph summary as nodes, edges, and cascade depth.

## 11. Cesium Globe Math

The globe rendering code is in [frontend/src/components/Globe/CesiumGlobe.jsx](frontend/src/components/Globe/CesiumGlobe.jsx).

### Snapshot extrapolation

Between backend updates, the globe advances each satellite using linear motion:

`advancedEci = position + velocity * elapsedSeconds`

This is only a short-term interpolation, not a full orbit propagator.

### ECI to Cesium conversion

The app converts ECI coordinates to Cesium Cartesian coordinates using GMST:

- compute GMST for the current frame date
- rotate ECI into Earth-fixed space
- scale km to meters for Cesium

### Orbit track rendering

The selected orbit is split into segments if adjacent samples jump too far:

- if separation > `8000 km`, start a new segment

This prevents the viewer from drawing lines across broken or discontinuous arcs.

### Why the dots move smoothly

The frontend uses:

- backend velocity when available
- derived velocity from previous snapshots when needed
- a pre-render motion loop capped to a short extrapolation window

That combination keeps the satellites moving even between API updates.

## 12. Alert Panel Logic

The sidebar logic is in [frontend/src/components/Alert/AlertPanel.jsx](frontend/src/components/Alert/AlertPanel.jsx).

### What it shows

- number of active alerts
- graph node and edge counts
- max cascade depth
- total delta-V across the chain
- agencies involved
- top pairwise alerts sorted by risk
- maneuver chain entries with depth and delta-V

### Why it matters

This is the judge-facing summary. It turns the planner output into something an operator can understand quickly.

## 13. Tunable Constants

These are the main knobs in the current implementation:

- `WATCH_DISTANCE_KM = 500`
- `WARNING_DISTANCE_KM = 200`
- `ALERT_DISTANCE_KM = 50`
- `DEFAULT_INFLUENCE_RADIUS_KM = 200`
- `DEFAULT_CPI_THRESHOLD = 5.0`
- `max_depth = 3`
- `max_maneuvers = 50`
- burn safety floor of `100 km` perigee

## 14. Important Limitation

The current graph scorer is deterministic and lightweight. It is shaped like a GNN pipeline, but it is not a trained model yet.

That is deliberate: it keeps the system self-contained and makes the ranking reproducible for demos and judge runs.

## 15. Best Reading Order in the Code

If you want to follow the logic in code order, read these files first:

1. [backend/app/core/sgp4_propagator.py](backend/app/core/sgp4_propagator.py)
2. [backend/app/core/conjunction.py](backend/app/core/conjunction.py)
3. [backend/app/ml/xgboost_scorer.py](backend/app/ml/xgboost_scorer.py)
4. [backend/app/ml/gnn_cascade.py](backend/app/ml/gnn_cascade.py)
5. [backend/app/services/cascade_planner.py](backend/app/services/cascade_planner.py)
6. [frontend/src/components/Globe/CesiumGlobe.jsx](frontend/src/components/Globe/CesiumGlobe.jsx)
7. [frontend/src/components/Alert/AlertPanel.jsx](frontend/src/components/Alert/AlertPanel.jsx)

## 16. Short Summary

The repo uses simple, explainable math at every major step:

- vector norms for distance and speed
- linear relative-motion estimates for TCA
- weighted heuristic scoring for risk and CPI
- KD-tree neighbor search for graph building
- sigmoid-based ranking for maneuver probability
- linear extrapolation for short-term globe motion

The result is a practical MVP that is easy to explain and easy to demo.