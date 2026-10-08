import { create } from 'zustand';

const generateTestCaseId = () => {
  const now = new Date();
  const yyyy = now.getUTCFullYear();
  const mm = String(now.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(now.getUTCDate()).padStart(2, '0');
  const hh = String(now.getUTCHours()).padStart(2, '0');
  const min = String(now.getUTCMinutes()).padStart(2, '0');
  return `TEST-${yyyy}-${mm}${dd}-${hh}${min}`;
};

const defaultOverride = () => ({
  mode: 'real',
  position: { x: '', y: '', z: '' },
  velocity: { vx: '', vy: '', vz: '' },
  epochUtc: '',
});

const defaultPrediction = () => ({
  tcaUtc: '',
  missDistanceM: '',
  relVelocityKms: '',
  collisionProbability: '',
  predictedCollision: 'no',
  modelName: '',
  testCaseId: generateTestCaseId(),
});

const defaultThresholds = () => ({
  tcaSeconds: 30,
  missDistancePct: 20,
  velocityPct: 5,
});

const useTestMode = create((set) => ({
  testActive: false,
  selectedA: null,
  selectedB: null,
  overrideA: defaultOverride(),
  overrideB: defaultOverride(),
  prediction: defaultPrediction(),
  thresholds: defaultThresholds(),
  sim: {
    stepSeconds: 10,
    timeWarp: 1,
    startEpochUtc: '',
    currentEpochUtc: '',
  },
  computed: null,
  testSatellites: { a: null, b: null },
  separationKm: null,
  closingRateKms: null,
  timeToTcaSeconds: null,

  setSelectedA: (sat) => set({ selectedA: sat }),
  setSelectedB: (sat) => set({ selectedB: sat }),
  clearSelectedA: () => set({ selectedA: null }),
  clearSelectedB: () => set({ selectedB: null }),

  setOverrideMode: (key, mode) =>
    set((state) => ({
      [key]: { ...state[key], mode },
    })),

  setOverrideField: (key, field, value) =>
    set((state) => ({
      [key]: { ...state[key], [field]: value },
    })),

  setOverrideVectorField: (key, vectorKey, axis, value) =>
    set((state) => ({
      [key]: {
        ...state[key],
        [vectorKey]: { ...state[key][vectorKey], [axis]: value },
      },
    })),

  setPredictionField: (field, value) =>
    set((state) => ({ prediction: { ...state.prediction, [field]: value } })),

  resetPrediction: () => set({ prediction: defaultPrediction() }),

  setThresholds: (updates) =>
    set((state) => ({ thresholds: { ...state.thresholds, ...updates } })),

  setSimControl: (updates) =>
    set((state) => ({ sim: { ...state.sim, ...updates } })),

  setComputed: (computed) => set({ computed }),

  setTestSatellites: (satellites) => set({ testSatellites: satellites }),

  setSeparation: ({ separationKm, closingRateKms, timeToTcaSeconds }) =>
    set({
      separationKm,
      closingRateKms,
      timeToTcaSeconds,
    }),

  setTestActive: (active) => set({ testActive: active }),
}));

export default useTestMode;
