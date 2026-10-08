import { create } from 'zustand';

const useStore = create((set, get) => ({
  // ── Satellite data ──────────────────────────────────────────────────────────
  satellites: [],
  selectedSatelliteId: null,
  hoveredSatelliteId: null,
  selectedOrbit: [],
  snapshotTimestamp: null,

  // ── Conjunction alerts ──────────────────────────────────────────────────────
  alerts: [],
  cascadePlan: null,
  hotspots: [],
  debrisClouds: [],
  debrisBySource: {},
  modelMetrics: null,
  simulationDrawerOpen: true,
  metricsDrawerOpen: true,

  // ── Maneuver controls ───────────────────────────────────────────────────────
  maneuver: { dvx: 0, dvy: 0, dvz: 0 },

  // ── Connection status ───────────────────────────────────────────────────────
  wsConnected: false,

  // ── Simulation ──────────────────────────────────────────────────────────────
  simulationActive: false,

  // ── Agency filtering ────────────────────────────────────────────────────────
  agencyFilter: null,

  // ── NEW: Mission control state ──────────────────────────────────────────────
  selectedAlertId: null,
  simTimeOffset: 0,
  backendConnected: false,
  dataAgeSeconds: 0,
  debrisFragments: [],
  cascadeGraph: { nodes: [], edges: [] },
  rankerReview: {},
  decisionLog: [],

  // ── Existing actions ─────────────────────────────────────────────────────────
  setSatellites: (sats) => set({ satellites: sats }),

  setSnapshotTimestamp: (timestamp) => set({ snapshotTimestamp: timestamp }),

  setSelectedSatelliteId: (id) => set({ selectedSatelliteId: id }),

  setHoveredSatelliteId: (id) => set({ hoveredSatelliteId: id }),

  setSelectedOrbit: (orbit) => set({ selectedOrbit: orbit }),

  clearSelectedOrbit: () => set({ selectedOrbit: [] }),

  getSelectedSatellite: () => {
    const { satellites, selectedSatelliteId } = get();
    if (!selectedSatelliteId) return satellites[0] || null;
    return satellites.find((s) => s.norad_id === selectedSatelliteId) || null;
  },

  setAlerts: (alerts) => set({ alerts }),

  setCascadePlan: (cascadePlan) => set({ cascadePlan }),

  setHotspots: (hotspots) => set({ hotspots }),

  /**
   * Debris clouds arrive from two independent backend sources that refresh on
   * different cadences:
   *   - 'fragment' clouds come from the core EVOLVE model and are pushed on
   *     every 1 Hz position frame because their radii grow continuously.
   *   - 'forecast' clouds come from the alert planner and only change when the
   *     alert cache is recomputed.
   *
   * Replacing the whole list on every frame would drop whichever source did not
   * ride along with that frame, so the two are stored separately and merged on
   * read. `sources` names the sources this payload actually carries: those are
   * replaced outright (including being cleared when empty), and any source it
   * omits is carried forward untouched. When `sources` is not supplied the
   * payload is treated as authoritative for every source it mentions.
   */
  setDebrisClouds: (incoming, sources = null) =>
    set((state) => {
      const next = {};
      const present = new Set();
      for (const cloud of incoming || []) {
        const source = cloud?.debris_source || 'fragment';
        present.add(source);
        (next[source] ||= []).push(cloud);
      }
      // A named-but-empty source is a clear signal, not an absence.
      for (const source of sources || []) {
        if (!present.has(source)) next[source] = [];
      }
      for (const [source, clouds] of Object.entries(state.debrisBySource || {})) {
        if (!(source in next)) next[source] = clouds;
      }
      const debrisBySource = next;
      return {
        debrisBySource,
        debrisClouds: Object.values(debrisBySource).flat(),
      };
    }),

  resetDebrisClouds: () => set({ debrisBySource: {}, debrisClouds: [] }),

  setModelMetrics: (modelMetrics) => set({ modelMetrics }),

  setSimulationDrawerOpen: (simulationDrawerOpen) => set({ simulationDrawerOpen }),

  setMetricsDrawerOpen: (metricsDrawerOpen) => set({ metricsDrawerOpen }),

  setManeuver: (axis, value) =>
    set((state) => ({
      maneuver: { ...state.maneuver, [axis]: value },
    })),

  resetManeuver: () => set({ maneuver: { dvx: 0, dvy: 0, dvz: 0 } }),

  setWsConnected: (connected) => set({ wsConnected: connected }),

  setSimulationActive: (active) => set({ simulationActive: active }),

  setAgencyFilter: (filter) => set({ agencyFilter: filter }),

  // ── NEW actions ──────────────────────────────────────────────────────────────
  setSelectedAlertId: (id) => set({ selectedAlertId: id }),

  setSimTimeOffset: (h) => set({ simTimeOffset: h }),

  setBackendConnected: (v) => set({ backendConnected: v }),

  setDataAgeSeconds: (s) => set({ dataAgeSeconds: s }),

  setDebrisFragments: (f) => set({ debrisFragments: f }),

  setCascadeGraph: (g) => set({ cascadeGraph: g }),

  setRankerReview: (r) => set({ rankerReview: r || {} }),

  addDecisionLogEntry: (e) =>
    set((state) => ({
      decisionLog: [e, ...state.decisionLog].slice(0, 100),
    })),
}));

export default useStore;
