/**
 * Screen-region catalogue for the help overlay. Each region is located at
 * open time by DOM query (first selector that matches a visible element);
 * regions that are not on screen are skipped, so the overlay keeps working
 * when other panels change.
 *
 * place: where the explanation card goes relative to the region rect
 *   'below-left' | 'below-right' | 'inside-tl' | 'inside-bl' | 'inside-tr' | 'inside-wide'
 *   | 'beside-right' (to the right of `anchor`'s rect)
 * badgeOnly: draw outline + number only; the text lives in `parent`'s card.
 */

export const HELP_REGIONS = [
  {
    key: 'readouts',
    title: 'Status readouts',
    selectors: ['.sh-readouts'],
    place: 'below-left',
    what: 'Tracked = catalogue objects propagated right now; Conjunctions = predicted close approaches in the next 24 h; Max CPI = worst triage index; Link = live WebSocket feed; Data age = time since the last position snapshot; clock = simulation UTC.',
    how: 'Positions by SGP4 from TLEs on the simulation clock (wall clock + operator offset).',
    colours: 'Amber/red values need attention: data older than 2 / 10 min, CPI ≥ 5 / ≥ 8, clock label "Sim +Nh" when time is shifted.',
  },
  {
    key: 'controls',
    title: 'Controls & drawers',
    selectors: ['.sh-controls'],
    place: 'below-right',
    what: 'Agency filter limits the globes to one operator. Scorecard = held-out ML metrics. Simulation = scenarios and sim-clock control. Analytics = correlation, PCA, SHAP and the 18 live physics cross-checks.',
    how: 'Drawers open as side sheets over the dashboard; nothing on the main screen changes until you act.',
    colours: 'A lit (pressed) button means its drawer is open.',
  },
  {
    key: 'live-globe',
    title: 'Live catalogue globe',
    selectors: ['section[aria-label="Live catalog globe"]'],
    place: 'inside-bl',
    what: 'Every tracked object at its current position. Click a dot to select it: its orbit is drawn and the telemetry strip fills in.',
    how: 'SGP4 propagation of each TLE (executed burns added as J2 deviations), refreshed about once per second over the WebSocket.',
    colours: 'Grey = no alert, amber = in a WARNING pair, red = in a CRITICAL pair, blue = selected, amber cloud = debris fragments.',
  },
  {
    key: 'layers',
    title: 'Globe layers',
    selectors: ['.im-layers'],
    parent: 'live-globe',
    badgeOnly: true,
    what: 'Layers button: toggle satellites, orbit, conjunction lines, labels, hotspots and debris layers; colour fragments by parent, size or ejection Δv.',
  },
  {
    key: 'impact',
    title: 'Impact replay console',
    selectors: ['.im-console', '.im-pill'],
    parent: 'live-globe',
    badgeOnly: true,
    what: 'Impact replay (appears after a simulated collision): plays the breakup from T−15 min to T+3 h; fragment positions are real propagated samples every 30 s, interpolated between samples.',
  },
  {
    key: 'queue',
    title: 'Conjunction queue',
    selectors: ['.tq-root'],
    place: 'inside-tl',
    offsetY: 100,
    what: 'One row per predicted close approach. SCREENING = found by catalogue screening; DEBRIS = a fragment threatening a satellite. TCA = countdown to closest approach, Miss = distance at TCA, Pc = collision probability, Casc = cascade depth, CPI = triage index (bar).',
    how: 'Pc is Foster 2-D from a TLE-age covariance model; TCA refined by Brent search on SGP4 over a 24 h window.',
    colours: 'CRITICAL (red): Pc ≥ 1e-4 or miss < 1 km · WARNING (amber): Pc ≥ 1e-6 or miss < 5 km · WATCH: the rest. The top lights show backend/pipeline/ML/physics/ranker health.',
  },
  {
    key: 'threat-card',
    title: 'Threat card (click a row)',
    selectors: ['.tq-row.is-expanded', '.tq-list'],
    place: 'beside-right',
    anchor: '.tq-root',
    what: null,
    bullets: [
      ['Decision score', '0–100 = 60 % physics Pc + 15 % ML + 15 % cascade + 10 % manoeuvre cost; the action (MANOEUVRE ≥ 1e-4, PREPARE ≥ 1e-5, MONITOR ≥ 1e-7) is set by physics Pc.'],
      ['Pc cross-check', 'Foster vs Chan series vs Alfano maximum vs Monte Carlo on the same B-plane; "consistent" when Foster and Chan agree within 0.5 decades.'],
      ['ML explanation', 'TreeSHAP: how many decades of Pc each feature adds (orange) or removes (blue) in the XGBoost surrogate; advisory only.'],
      ['Avoidance options', 'Candidate burns, each re-propagated; cascade-safe = creates no new conjunction with Pc ≥ 1e-6 in 24 h.'],
      ['Engines', 'Propellant (rocket equation) and burn time per engine; the recommended one uses the least propellant among practical engines.'],
    ],
  },
  {
    key: 'threat-globe',
    title: 'Threat analysis globe',
    selectors: ['section[aria-label="Threat analysis globe"]'],
    place: 'inside-bl',
    what: 'A wireframe view of only the objects in conjunctions, labelled with miss distance and TCA. Drag to rotate, scroll to zoom, double-click to reset.',
    how: 'Same SGP4 positions and alerts as the queue; selecting a row or satellite highlights it here.',
    colours: 'Red / amber / green dots follow the alert severity; amber points are debris fragments.',
  },
  {
    key: 'telemetry',
    title: 'Telemetry strip',
    selectors: ['.tm-strip'],
    place: 'inside-wide',
    what: 'Position, velocity, altitude and state of the selected object.',
    how: 'No real downlink exists: fuel comes from the rocket equation on executed burns, sunlit/eclipse from sun geometry; values tagged SIM are simulated, battery/temperature show as not modelled.',
    colours: null,
  },
  {
    key: 'uplink',
    title: 'Command uplink',
    selectors: ['.ul-panel'],
    place: 'inside-tl',
    offsetY: 32,
    what: 'Send a Δv burn (prograde, retrograde, radial, normal), an orbit change or a state override to the selected satellite; the log shows each command result.',
    how: 'Pre-flight gates check fuel, TCA window and limits; only satellites your agency session controls can be commanded.',
    colours: 'Green log line = accepted, red = rejected.',
  },
  {
    key: 'cascade',
    title: 'Cascade risk graph',
    selectors: ['.an-cascade'],
    place: 'inside-tl',
    offsetY: 32,
    what: 'Nodes are objects and collision events; each edge is an alert linking two of them. It shows how one collision can spread.',
    how: 'Depth = BFS hops from a collision event (fragment 1, threatened satellite 2, …); node P(hit) = 1 − Π(1 − Pc) over its alerts.',
    colours: 'Node colour = worst severity (red critical, amber warning, blue watch, green nominal); event nodes are collisions.',
  },
  {
    key: 'models',
    title: 'Model status',
    selectors: ['.sh-bottom > .sh-cell:nth-child(3)'],
    place: 'inside-tl',
    offsetY: 32,
    what: 'Honest scorecard of the ML Pc surrogate: held-out error and precision/recall at Pc ≥ 1e-4, next to a miss-distance-only baseline.',
    how: 'XGBoost trained on Foster-labelled simulated encounters, evaluated on a 20 % held-out split; it never replaces the physics Pc.',
    colours: 'Green = beats the baseline / validated; amber = degraded or experimental.',
  },
  {
    key: 'drawer-sim',
    title: 'Simulation drawer',
    selectors: ['aside[aria-label="Simulation console"]'],
    place: 'inside-tl',
    offsetY: 40,
    what: 'Load scenarios (e.g. the computed crossing), shift the simulation clock, run test mode.',
    how: 'Scenarios inject objects whose conjunction must still be found by the screening pipeline.',
  },
  {
    key: 'drawer-metrics',
    title: 'Scorecard drawer',
    selectors: ['aside[aria-label="Model scorecard"]'],
    place: 'inside-tl',
    offsetY: 40,
    what: 'Model cards with held-out metrics for every ML component, including the ones that are only experimental.',
    how: 'Read from /api/model-metrics, which serves the training-time model cards.',
  },
  {
    key: 'drawer-analytics',
    title: 'Analytics drawer',
    selectors: ['aside[aria-label="Analytics"]'],
    place: 'inside-tl',
    offsetY: 40,
    what: 'Feature correlation, PCA, global SHAP and live physics validation (Foster vs Chan vs Monte Carlo, SGP4 vs J2, CW, Hohmann, NASA SBM).',
    how: 'Each check lists formula, our value, reference value, error and tolerance; computed live, cached 60 s.',
  },
];

export const THREAT_SECTION_BADGES = [
  { label: 'Decision', bullet: 0 },
  { label: 'Pc Cross-check', bullet: 1 },
  { label: 'ML Explanation', bullet: 2 },
  { label: 'Avoidance Options', bullet: 3 },
];
