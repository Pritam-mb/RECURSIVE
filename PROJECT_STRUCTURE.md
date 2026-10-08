# Hackolution / Orbital Sentinel Detailed Structure

This document expands the short notes in `PROJECT_OVERVIEW.txt` and explains what the project does, how the pieces connect, and what each important file is responsible for.

## What The Project Does

Orbital Sentinel is a real-time satellite tracking and conjunction-awareness prototype.

The backend loads TLE data, propagates satellite positions with SGP4, caches the most recent snapshot in memory, screens for conjunctions, and exposes the results through REST and WebSocket endpoints.

The frontend is a React + Cesium dashboard that renders a globe, shows live satellite motion, displays telemetry for the selected object, and provides a test-mode workflow for deterministic collision and maneuver scenarios.

## Runtime Flow

1. `backend/main.py` starts FastAPI, loads a bundled TLE snapshot, initializes the propagator and simulation engine, and warms the snapshot and alert caches.
2. A background loop refreshes propagated satellite state every second.
3. A slower background loop refreshes conjunction alerts from the latest cached states.
4. The WebSocket broadcaster pushes the latest snapshot to connected clients at 1 Hz.
5. `frontend/src/App.jsx` bootstraps the first snapshot and alert list through REST, keeps a satellite polling fallback running, then opens the WebSocket stream for live updates.
6. `frontend/src/components/Globe/CesiumGlobe.jsx` renders satellites on the globe and keeps the scene in sync with the store using live position and velocity data.
7. `frontend/src/components/Telemetry/TelemetryPanel.jsx` fetches detailed state for the selected satellite and orbit track samples.
8. `frontend/src/components/TestModePanel.jsx` configures deterministic two-satellite scenarios for the test workflow.
9. `frontend/src/components/Control/ManualControl.jsx` and `frontend/src/components/Control/PreflightModal.jsx` handle the maneuver demo flow.

## Repository Tree

```text
hackolution/
  PROJECT_OVERVIEW.txt
  PROJECT_STRUCTURE.md
  backend/
    main.py
    requirements.txt
    app/
      __init__.py
      api/
        __init__.py
        routes.py
        ws_handler.py
      core/
        __init__.py
        conjunction.py
        sgp4_propagator.py
        state_cache.py
      data/
        __init__.py

        tle_fetcher.py
      ml/
        __init__.py
        lstm_predictor.py
        xgboost_scorer.py
      routers/
        __init__.py
        test_mode.py
      services/
        __init__.py
        conjunction_solver.py
      simulation/
        __init__.py
        approaching.json
        sim_engine.py
        tle-data.txt
  frontend/
    index.html
    package.json
    package-lock.json
    README.md
    vite.config.js
    eslint.config.js
    public/
      dot-medium.png
      earth-realistic-8k.webp
      favicon.svg
      icons.svg
    src/
      App.css
      App.jsx
      index.css
      main.jsx
      assets/
        hero.png
        react.svg
        vite.svg
      components/
        ConjunctionMonitor.jsx
        TestModePanel.jsx
        Alert/
          AlertPanel.jsx
        Control/
          ManualControl.jsx
          PreflightModal.jsx
        Globe/
          CesiumGlobe.jsx
        Telemetry/
          TelemetryPanel.jsx
      hooks/
        useTestMode.js
      store/
        useStore.js
      utils/
        coords.js
```

## Root Files

- `PROJECT_OVERVIEW.txt` is the high-level summary of the system: what it does, what services exist, and what the current limitations are.
- `PROJECT_STRUCTURE.md` is this expanded guide.
- `.git/` is the Git metadata directory and not part of the application logic.
- `.vscode/` contains editor configuration and workspace settings.

## Backend

### Entry Point

- `backend/main.py` is the FastAPI entry point.
  - It creates the `SGP4Propagator`, `SimEngine`, and WebSocket manager.
  - On startup it loads local TLE data from the bundled snapshot in `backend/app/simulation/tle-data.txt`.
  - It builds an initial snapshot, seeds the alert cache, and starts three background loops: snapshot refresh, alert refresh, and WebSocket broadcast.
  - It registers the REST routers and the `/ws/satellites` WebSocket endpoint.
  - It also configures CORS so the React frontend can talk to the API during local development.

- `backend/requirements.txt` lists the backend Python dependencies used by the prototype, including FastAPI, Uvicorn, SGP4, NumPy, httpx, and python-dotenv.

### `backend/app`

- `backend/app/__init__.py` marks the package.

#### API Layer

- `backend/app/api/__init__.py` marks the API package.
- `backend/app/api/routes.py` defines the REST API under `/api`.
  - `GET /api/satellites` returns the latest cached snapshot.
  - `GET /api/satellites/{norad_id}` returns a single propagated satellite state.
  - `GET /api/satellites/{norad_id}/orbit` generates a short orbit track around the current time.
  - `GET /api/alerts` returns the cached conjunction alerts.
  - `POST /api/maneuver` applies a demo delta-V maneuver after preflight validation.
  - `POST /api/simulate` loads a deterministic scenario from the simulation folder.
  - `DELETE /api/simulate` clears the active scenario.
  - `GET /api/scenarios` lists available scenario files.
  - `POST /api/preflight` runs the maneuver pre-check without executing the burn.
  - `init_routes()` stores shared runtime objects so the router can access the propagator and simulation engine.

- `backend/app/api/ws_handler.py` manages the live WebSocket feed.
  - `ConnectionManager` tracks active clients, accepts and removes sockets, and broadcasts text frames to everyone connected.
  - `satellite_broadcast_loop()` pushes the latest cached snapshot at 1 Hz when clients are present.

#### Core Layer

- `backend/app/core/__init__.py` marks the core package.
- `backend/app/core/sgp4_propagator.py` is the orbit propagation engine.
  - `SatelliteState` stores one propagated object state and can serialize itself into a dictionary.
  - `SGP4Propagator.load_tles()` parses TLE lines into `Satrec` objects.
  - `propagate_all()` and `propagate_one()` compute ECI position and velocity for a given UTC timestamp.
  - `apply_delta_v()` rebuilds the affected `Satrec` after a burn so future propagations use the new orbit.
  - `get_maneuver()` retrieves and clears a stored maneuver.
  - `satellite_count` and `norad_ids` expose lightweight introspection helpers.

- `backend/app/core/conjunction.py` handles conjunction screening.
  - It computes miss distance and relative speed for satellite pairs.
  - `screen_conjunctions()` performs pairwise checks on the current states and assigns a severity based on distance bands.
  - `find_tca()` iteratively searches for the time of closest approach over a future time window.
  - The current thresholds are watch at 500 km, warning at 200 km, and alert at 50 km.

- `backend/app/core/state_cache.py` is the in-memory cache for live data.
  - It stores the latest propagated snapshot.
  - It stores the latest alert list.
  - The cache is process-local and does not persist to disk.

#### Data Layer

- `backend/app/data/__init__.py` marks the data package.
- Fallback TLEs: the bundled real snapshot `backend/app/simulation/tle-data.txt` (see docs/HOW_IT_WORKS.md).
- `backend/app/data/tle_fetcher.py` loads satellite TLE data.
  - `parse_3line_tle()` converts the 3-line `0/1/2` TLE format into dictionaries.
  - `load_local_tles()` reads the bundled snapshot from `backend/app/simulation/tle-data.txt`.
  - `fetch_remote_tles()` downloads the snapshot from the SatelliteTracker3D CDN.
  - `fetch_tles()` tries remote first and falls back to the local copy if needed.

#### ML Layer

- `backend/app/ml/__init__.py` marks the ML package.
- `backend/app/ml/lstm_predictor.py` is a placeholder trajectory forecaster.
  - `RingBuffer` keeps recent position history per satellite.
  - `LSTMPredictor.record()` appends snapshots to the history buffer.
  - `predict()` currently uses linear extrapolation from the last two samples instead of a trained model.

- `backend/app/ml/xgboost_scorer.py` is a placeholder risk scorer.
  - `extract_features()` converts a conjunction event into a small numeric feature vector.
  - `score()` now returns a deterministic collision-risk heuristic that feeds CPI and cascade ranking.

- `backend/app/ml/gnn_cascade.py` provides a deterministic graph-ranking layer for cascade planning.
  - It scores each satellite by graph influence and local risk.
  - It outputs maneuver probability and cascade depth estimates without requiring a heavyweight GNN runtime.

#### Test and Service Layer

- `backend/app/routers/__init__.py` marks the router package.
- `backend/app/routers/test_mode.py` exposes the test-mode API under `/api/test`.
  - `POST /api/test/setup` loads two satellites into a deterministic test session.
  - `POST /api/test/run-to-tca` propagates both satellites to a target TCA and returns the computed encounter result.
  - `GET /api/test/separation` returns live separation, closing rate, and estimated time to TCA.
  - The module supports either real TLE-based satellites or manually overridden ECI states.

- `backend/app/services/__init__.py` marks the services package.
- `backend/app/services/conjunction_solver.py` is the numerical solver used by test mode.
  - It defines an `EciState` data model and a `TcaResult` output model.
  - `_accel_two_body()` computes two-body gravity with an optional J2 term.
  - `_rk45_propagate()` integrates both satellites forward in time with `solve_ivp`.
  - `run_to_tca()` compares the two trajectories, finds the minimum separation, and returns the result timeline.
  - `parse_eci_state()` converts raw lists into solver input objects.

#### Simulation Layer

- `backend/app/simulation/__init__.py` marks the simulation package.
- `backend/app/simulation/sim_engine.py` controls deterministic demo scenarios and maneuvers.
  - `load_scenario()` reads a scenario JSON file, injects its satellites, and records the active scenario.
  - `clear_scenario()` removes injected scenario satellites.
  - `apply_maneuver()` runs a preflight check and forwards the burn to the propagator so the updated orbit is immediately available.
  - `preflight_check()` implements the six-gate demo validation logic.
  - `list_scenarios()` lists the JSON scenarios available in the folder.

- `backend/app/services/cascade_planner.py` builds the constellation graph, ranks CPI risk, and produces a cascade maneuver plan.
  - It uses KD-tree pruning to find influence neighbors efficiently.
  - It scores each edge with a deterministic CPI heuristic and a graph-style ranker.
  - It returns a maneuver chain, agencies involved, and a graph summary for the dashboard.

- `backend/app/simulation/approaching.json` is a deterministic demo scenario with two satellites and an expected close approach.
- `backend/app/simulation/tle-data.txt` is the bundled local TLE snapshot used for startup and fallback loading.

## Frontend

### App Shell And Bootstrapping

- `frontend/index.html` is the Vite HTML shell.
  - It defines the root mount point.
  - It points the app to `src/main.jsx`.
  - It also sets the favicon.

- `frontend/src/main.jsx` mounts the React app into `#root`.

- `frontend/src/App.jsx` is the top-level UI container.
  - It boots the initial satellite and alert data with REST calls.
  - It polls satellites as a fallback so the globe keeps animating even if the WebSocket stalls.
  - It opens the satellite WebSocket and reconnects if the socket closes.
  - It keeps the main store in sync with snapshot timestamps, satellites, alerts, and connection state.
  - It renders the globe panel on the left and the control/sidebar stack on the right.
  - It currently mounts `CesiumGlobe`, `TelemetryPanel`, `TestModePanel`, `ConjunctionMonitor`, and `ManualControl`.

- `frontend/src/App.css` defines the page layout and component styling.
  - It creates the left globe and right sidebar bento layout.
  - It styles the header, panels, buttons, modal, tables, and telemetry overlay.
  - It also contains the Cesium viewport overrides and theme-specific classes.

- `frontend/src/index.css` defines the global theme.
  - It loads the font stack.
  - It defines color tokens, monospace styles, severity colors, and base page resets.

### State And Utilities

- `frontend/src/store/useStore.js` is the main application store.
  - It stores the live satellite list, selected and hovered satellite IDs, selected orbit samples, alerts, maneuver values, WebSocket status, and simulation state.
  - It exposes simple setters and a helper for resolving the currently selected satellite.

- `frontend/src/hooks/useTestMode.js` stores the test-mode workflow state.
  - It tracks the selected satellite pair, override mode and values, prediction fields, simulation controls, computed results, and separation metrics.
  - It also generates a timestamp-based test case ID.

- `frontend/src/utils/coords.js` contains coordinate conversion helpers.
  - `computeGmst()` calculates Greenwich Mean Sidereal Time from a JavaScript `Date`.
  - `eciToEcef()` rotates ECI coordinates into the Earth-fixed frame.
  - `ecefToCartesian()` converts kilometers to Cesium meters.
  - `ecefToLla()` converts Earth-fixed coordinates to latitude, longitude, and altitude.
  - `eciToCesiumCartesian()` and `eciToLla()` combine the pipeline into one call.

### Components

- `frontend/src/components/Globe/CesiumGlobe.jsx` renders the globe and satellite scene.
  - It creates a Cesium viewer without the default UI chrome.
  - It uses `frontend/public/earth-realistic-8k.webp` as the globe texture.
  - It renders satellites as billboards, labels the hovered object, and draws the selected orbit track.
  - It uses backend-provided velocity where available so satellites keep moving smoothly between updates.
  - It highlights the selected satellite, hovered satellite, and test-mode satellites with different colors and scales.
  - It also draws a line between the two override satellites when test mode is active.
  - It uses `VITE_CESIUM_TOKEN` if a Cesium Ion token is provided in the environment.

- `frontend/src/components/Telemetry/TelemetryPanel.jsx` shows details for the selected satellite.
  - It fetches the selected satellite state from the backend.
  - It fetches orbit samples for the selected satellite and stores them in the main store.
  - It calculates latitude and longitude on the client using the coordinate helper functions.
  - It displays speed, height, latitude, and longitude.

- `frontend/src/components/ConjunctionMonitor.jsx` compares model predictions with the computed test-mode result.
  - It reads the prediction input from the test-mode store.
  - It compares predicted and computed miss distance, TCA time, relative velocity, and collision outcome.
  - It applies threshold-based pass, partial, and fail verdicts.
  - It also shows live separation, closing rate, and the computed encounter summary.

- `frontend/src/components/TestModePanel.jsx` is the main test-scenario builder.
  - It lets the user choose satellite A and satellite B from the live satellite list.
  - It supports either real TLE data or manual ECI overrides for each satellite.
  - It can paste a JSON scenario payload and fill the form automatically.
  - It sends the setup to `/api/test/setup`.
  - It sends the current prediction TCA to `/api/test/run-to-tca` and stores the returned encounter result.
  - It provides simulation controls for step size and time warp.
  - It is structured as three visible sections: Satellite Selector, Scenario Setup, and Simulation Controls.

- `frontend/src/components/Control/ManualControl.jsx` handles the maneuver demo workflow.
  - It exposes RSW delta-V sliders for radial, along-track, and cross-track burns.
  - It opens the preflight modal when the user chooses to execute a burn.
  - It can trigger the demo scenario by calling `/api/simulate` with the `approaching` scenario name.

- `frontend/src/components/Control/PreflightModal.jsx` is the pre-burn validation dialog.
  - It calls `/api/preflight` as soon as it opens.
  - It shows the delta-V magnitude and each of the six validation gates.
  - It disables burn confirmation until all gates pass.

- `frontend/src/components/Alert/AlertPanel.jsx` is a reusable conjunction-alert list component.
  - It reads alert data and the cascade plan from the store and renders a compact list of pairwise alerts.
  - It also shows the cascade summary, total delta-v, and maneuver chain.
  - It is mounted in the sidebar in `App.jsx`.

### Static Assets

- `frontend/public/dot-medium.png` is the Cesium billboard icon for satellites.
- `frontend/public/earth-realistic-8k.webp` is the globe texture used by the Cesium scene.
- `frontend/public/favicon.svg` is the browser tab icon.
- `frontend/public/icons.svg` is a shared SVG asset file for the frontend.
- `frontend/src/assets/hero.png`, `frontend/src/assets/react.svg`, and `frontend/src/assets/vite.svg` are bundled image assets kept in the source tree.

### Frontend Config Files

- `frontend/package.json` defines the Vite scripts and frontend dependencies.
  - `dev` starts the local development server.
  - `build` creates the production bundle.
  - `preview` serves the production build locally.

- `frontend/package-lock.json` locks the exact npm dependency versions.
- `frontend/vite.config.js` configures Vite.
  - It enables the React plugin.
  - It enables `vite-plugin-cesium` so Cesium assets are handled correctly.

- `frontend/eslint.config.js` defines the lint rules for JavaScript and React files.
  - It uses the recommended JS rules.
  - It enables React hooks and React refresh rules.
  - It ignores the `dist` folder.

- `frontend/README.md` is still the default Vite scaffold README and does not yet describe the Orbital Sentinel app in detail.

## Notes On Current Design

- The backend uses in-memory caches only; there is no persistent database.
- The maneuver workflow still uses a simplified impulsive-burn reconstruction, and conjunction screening remains intentionally lightweight.
- The ML files are placeholders for future model integration rather than fully trained production models.
- The main React app currently uses fixed local URLs for the API and WebSocket endpoints, so it is tuned for local development.
- Generated folders such as `backend/app/**/__pycache__/`, `frontend/node_modules/`, and `.git/` are part of the workspace state, not the application source.

## Quick Reading Order

If you want to understand the project quickly, read the files in this order:

1. `PROJECT_OVERVIEW.txt`
2. `backend/main.py`
3. `backend/app/core/sgp4_propagator.py`
4. `backend/app/api/routes.py`
5. `frontend/src/App.jsx`
6. `frontend/src/components/Globe/CesiumGlobe.jsx`
7. `frontend/src/components/TestModePanel.jsx`
8. `frontend/src/components/Control/ManualControl.jsx`
