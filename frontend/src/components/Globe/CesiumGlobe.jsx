import { useEffect, useMemo, useRef, useState } from 'react';
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import useStore from '../../store/useStore';
import useTestMode from '../../hooks/useTestMode';
import { computeGmst, eciToCesiumCartesian } from '../../utils/coords';
import {
  cloudCentroid, cloudEpoch, cloudFragments, cloudLabel, cloudRadius,
} from '../../utils/debrisCloud';
import {
  CORRECTION_DECAY_S,
  MAX_EXTRAPOLATION_S,
  correctionFor,
  estimateSimRate,
  extrapolate,
  observeActivity,
} from '../../utils/motion';

// Only talk to Cesium Ion when a token is configured. Nothing below needs Ion
// (no base layer, no geocoder, ellipsoid terrain), so without a token the app
// makes no Ion/Bing requests at all.
const CESIUM_TOKEN = import.meta.env.VITE_CESIUM_TOKEN;
if (CESIUM_TOKEN) Cesium.Ion.defaultAccessToken = CESIUM_TOKEN;

// Mirrors the design tokens in index.css (canvas/WebGL can't read CSS vars).
const PALETTE = {
  void: '#030508',
  line: '#1a212b',
  lineStrong: '#27303c',
  text: '#c3cbd5',
  dim: '#6b7685',
  bright: '#eef2f6',
  accent: '#4c8dff',
  info: '#7cc4ff',
  nominal: '#3ccf7a',
  caution: '#f0b429',
  warning: '#ff5a4f',
};
const css = (hex, alpha = 1) => Cesium.Color.fromCssColorString(hex).withAlpha(alpha);
const COLORS = {
  void: css(PALETTE.void),
  satNominal: css(PALETTE.text, 0.5),
  satHover: css(PALETTE.bright, 1),
  satSelected: css(PALETTE.accent, 1),
  satSelectedA: css(PALETTE.accent, 1),
  satSelectedB: css(PALETTE.info, 1),
  satCaution: css(PALETTE.caution, 0.95),
  satWarning: css(PALETTE.warning, 0.95),
  orbit: css(PALETTE.accent, 0.85),
  labelText: css(PALETTE.bright, 1),
  labelBg: css(PALETTE.void, 0.92),
  hotspotCore: css(PALETTE.warning, 0.95),
  hotspotZone: css(PALETTE.warning, 0.07),
  debrisCore: css(PALETTE.caution, 0.95),
  debrisText: css(PALETTE.caution, 1),
  debrisFragment: css(PALETTE.caution, 0.75),
  approach: {
    nominal: css(PALETTE.nominal, 0.8),
    caution: css(PALETTE.caution, 0.85),
    warning: css(PALETTE.warning, 0.9),
  },
};
const LABEL_FONT = "500 11px 'IBM Plex Mono', ui-monospace, monospace";
const SMALL_LABEL_FONT = "500 10px 'IBM Plex Mono', ui-monospace, monospace";

const SATELLITE_DEFAULT_SCALE = 0.26;
const SATELLITE_HOVER_SCALE = 0.34;
const SATELLITE_SELECTED_SCALE = 0.42;
const SATELLITE_SCALE_BY_DISTANCE = new Cesium.NearFarScalar(1.0e6, 1.4, 3.0e7, 0.2);
const ORBIT_BREAK_DISTANCE_KM = 8000;
const MAX_HOTSPOTS = 20;
const HOVER_PICK_INTERVAL_MS = 60;
const EMPTY = [];

// Threat level per satellite: 0 nominal, 1 caution, 2 warning.
const THREAT_CAUTION = 1;
const THREAT_WARNING = 2;

// Reused every frame by the motion loop. BillboardCollection copies the value
// on assignment (Billboard.position setter does Cartesian3.clone into its own
// storage), so one scratch is safe to share across all billboards.
const SCRATCH_CARTESIAN = new Cesium.Cartesian3();

function normalizeId(id) {
  if (id == null) return null;
  const n = Number(id);
  return Number.isNaN(n) ? id : n;
}

function toCartesian3(eci, date) {
  const c = eciToCesiumCartesian(eci, date);
  return new Cesium.Cartesian3(c.x, c.y, c.z);
}

function freshPrimitives() {
  return {
    billboards: null,
    labels: null,
    orbitLines: null,
    focusLabel: null,
    orbitPolyline: null,
    orbitSegmentPolylines: [],
    orbitMaterial: null,
    approachLine: null,
    approachMaterial: null,
    hotspotPoints: null,
    hotspotEntities: [],
    hotspotKey: null,
    debrisEntities: [],
    debrisPoints: null,
    debrisKey: null,
    focusedItem: null,
    map: new Map(),
  };
}

function freshMotion() {
  return {
    snapshotPositions: new Map(),
    lastSnapshotTimestampMs: null,
    lastSnapshotReceivedPerfMs: 0,
    simRate: 1,
    settled: false,
  };
}

const CesiumGlobe = ({ mode = 'live', alerts = EMPTY, onSatelliteSelect }) => {
  const containerRef = useRef(null);
  const viewerRef = useRef(null);
  const motionRef = useRef(freshMotion());
  const primitivesRef = useRef(freshPrimitives());
  const onSelectRef = useRef(onSatelliteSelect);
  onSelectRef.current = onSatelliteSelect;
  // Flips once the viewer exists so data effects re-run against it.
  const [viewerReady, setViewerReady] = useState(false);

  // ── Viewer lifecycle ────────────────────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    let viewer = null;
    let handler = null;
    let stopActivity = null;
    let pickTimer = null;
    let removePreUpdate = null;
    let clearHover = null;

    // Defer one frame so the CSS grid has committed its layout; Cesium reads
    // the container size at construction.
    const startRaf = requestAnimationFrame(() => {
      if (!containerRef.current) return;

      viewer = new Cesium.Viewer(container, {
        baseLayer: false, // Cesium >= 1.104: no Bing/Ion imagery
        baseLayerPicker: false,
        geocoder: false,
        homeButton: false,
        sceneModePicker: false,
        navigationHelpButton: false,
        animation: false,
        timeline: false,
        fullscreenButton: false,
        vrButton: false,
        infoBox: false,
        selectionIndicator: false,
        creditContainer: document.createElement('div'),
        skyBox: false,
        skyAtmosphere: false,
        shadows: false,
        // Render only when something changed; the motion loop requests
        // frames while satellites move. Capped at 30 fps.
        requestRenderMode: true,
        maximumRenderTimeChange: Infinity,
        targetFrameRate: 30,
        useBrowserRecommendedResolution: false,
        msaaSamples: 1,
        orderIndependentTranslucency: false,
        contextOptions: {
          webgl: {
            alpha: false,
            antialias: false,
            depth: true,
            stencil: false,
            preserveDrawingBuffer: false,
            powerPreference: 'default',
          },
        },
      });
      viewerRef.current = viewer;

      // Effective DPR <= 1.25.
      viewer.resolutionScale = Math.min(1, 1.25 / (window.devicePixelRatio || 1));

      const { scene } = viewer;
      scene.backgroundColor = COLORS.void;
      if (scene.postProcessStages?.fxaa) scene.postProcessStages.fxaa.enabled = false;
      if (scene.sun) scene.sun.show = false;
      if (scene.moon) scene.moon.show = false;
      scene.fog.enabled = false;
      scene.highDynamicRange = false;

      const { globe } = scene;
      globe.baseColor = COLORS.void;
      globe.showGroundAtmosphere = false;
      globe.enableLighting = false;
      globe.depthTestAgainstTerrain = false;
      globe.maximumScreenSpaceError = 3;
      globe.tileCacheSize = 100;
      globe.showSkirts = false;
      globe.preloadAncestors = false;

      Cesium.SingleTileImageryProvider.fromUrl('/earth-4k.webp')
        .then((provider) => {
          if (!viewer || viewer.isDestroyed()) return;
          const layer = viewer.imageryLayers.addImageryProvider(provider);
          // Muted, desaturated earth so state colours carry the signal.
          layer.brightness = 0.55;
          layer.contrast = 1.1;
          layer.saturation = 0.3;
          scene.requestRender();
        })
        .catch((err) => console.error('[CesiumGlobe] imagery', err));

      viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(30, 20, 25000000) });

      // Primitive collections (one draw call each, far cheaper than entities).
      const prims = primitivesRef.current;
      prims.billboards = scene.primitives.add(new Cesium.BillboardCollection({
        blendOption: Cesium.BlendOption.TRANSLUCENT,
      }));
      prims.labels = scene.primitives.add(new Cesium.LabelCollection());
      prims.orbitLines = scene.primitives.add(new Cesium.PolylineCollection());
      prims.hotspotPoints = scene.primitives.add(new Cesium.PointPrimitiveCollection());
      prims.debrisPoints = scene.primitives.add(new Cesium.PointPrimitiveCollection());

      prims.focusLabel = prims.labels.add({
        show: false,
        text: '',
        font: LABEL_FONT,
        fillColor: COLORS.labelText,
        style: Cesium.LabelStyle.FILL,
        showBackground: true,
        backgroundColor: COLORS.labelBg,
        backgroundPadding: new Cesium.Cartesian2(6, 4),
        horizontalOrigin: Cesium.HorizontalOrigin.LEFT,
        verticalOrigin: Cesium.VerticalOrigin.CENTER,
        pixelOffset: new Cesium.Cartesian2(12, 0),
      });

      prims.orbitMaterial = Cesium.Material.fromType('Color');
      prims.orbitMaterial.uniforms.color = COLORS.orbit;
      prims.approachMaterial = Cesium.Material.fromType('Color');
      prims.approachMaterial.uniforms.color = COLORS.approach.nominal;

      prims.orbitPolyline = prims.orbitLines.add({
        show: false,
        positions: [],
        width: 1.5,
        material: prims.orbitMaterial,
        arcType: Cesium.ArcType.NONE,
        id: 'selected-orbit',
      });
      prims.approachLine = prims.orbitLines.add({
        show: false,
        positions: [],
        width: 1,
        material: prims.approachMaterial,
        arcType: Cesium.ArcType.NONE,
        id: 'approach-line',
      });

      // ── Per-frame dead reckoning, driven by Cesium's own (30 fps, pausable)
      // render loop instead of a separate rAF. preUpdate fires before the
      // request-render check, so requesting here renders this same frame.
      removePreUpdate = scene.preUpdate.addEventListener(() => {
        const motion = motionRef.current;
        const { map, focusedItem, focusLabel } = primitivesRef.current;
        if (motion.lastSnapshotTimestampMs == null || map.size === 0) return;

        const rawWallSeconds = (performance.now() - motion.lastSnapshotReceivedPerfMs) / 1000;
        // Stream stalled: positions are frozen at the cap; stop rendering.
        if (rawWallSeconds > MAX_EXTRAPOLATION_S) {
          if (motion.settled) return;
          motion.settled = true;
        }
        const wallSeconds = Math.min(rawWallSeconds, MAX_EXTRAPOLATION_S);
        const simSeconds = wallSeconds * (motion.simRate || 1);
        const correctionWeight = Math.exp(-wallSeconds / CORRECTION_DECAY_S);

        const gmst = computeGmst(new Date(motion.lastSnapshotTimestampMs + (simSeconds * 1000)));
        const cosG = Math.cos(gmst);
        const sinG = Math.sin(gmst);

        for (const item of map.values()) {
          const r = extrapolate(
            item.eciPosition,
            item.eciVelocity,
            simSeconds,
            item.correction,
            correctionWeight,
            item.renderEci,
          );
          SCRATCH_CARTESIAN.x = ((r.x * cosG) + (r.y * sinG)) * 1000;
          SCRATCH_CARTESIAN.y = ((-r.x * sinG) + (r.y * cosG)) * 1000;
          SCRATCH_CARTESIAN.z = r.z * 1000;
          item.billboard.position = SCRATCH_CARTESIAN;
        }

        if (focusedItem && focusLabel.show) {
          focusLabel.position = focusedItem.billboard.position;
        }
        scene.requestRender();
      });

      // ── Hover / click picking (scene.pick is a render pass: throttle it).
      handler = new Cesium.ScreenSpaceEventHandler(scene.canvas);
      const pendingPos = new Cesium.Cartesian2();
      const pickAt = (position) => {
        const picked = scene.pick(position);
        const id = Cesium.defined(picked) && typeof picked.id === 'number' ? picked.id : null;
        const store = useStore.getState();
        if (store.hoveredSatelliteId !== id) store.setHoveredSatelliteId(id);
        return id;
      };
      handler.setInputAction((movement) => {
        Cesium.Cartesian2.clone(movement.endPosition, pendingPos);
        if (pickTimer != null) return;
        pickTimer = setTimeout(() => {
          pickTimer = null;
          if (viewer && !viewer.isDestroyed()) pickAt(pendingPos);
        }, HOVER_PICK_INTERVAL_MS);
      }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
      handler.setInputAction((click) => {
        const id = pickAt(click.position);
        if (id != null) {
          useStore.getState().setSelectedSatelliteId(id);
          onSelectRef.current?.(id);
        }
      }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

      clearHover = () => {
        if (pickTimer != null) { clearTimeout(pickTimer); pickTimer = null; }
        const store = useStore.getState();
        if (store.hoveredSatelliteId !== null) store.setHoveredSatelliteId(null);
      };
      scene.canvas.addEventListener('mouseleave', clearHover);

      // ── Pause the whole render loop when the tab is hidden or the globe
      // is scrolled offscreen.
      stopActivity = observeActivity(container, (active) => {
        if (!viewer || viewer.isDestroyed()) return;
        viewer.useDefaultRenderLoop = active;
        if (active) scene.requestRender();
      });

      setViewerReady(true);
    });

    return () => {
      cancelAnimationFrame(startRaf);
      if (pickTimer != null) clearTimeout(pickTimer);
      if (stopActivity) stopActivity();
      if (removePreUpdate) removePreUpdate();
      if (viewer && !viewer.isDestroyed()) {
        if (clearHover) viewer.scene.canvas.removeEventListener('mouseleave', clearHover);
        if (handler && !handler.isDestroyed()) handler.destroy();
        // Destroys every primitive, entity and the WebGL context.
        viewer.destroy();
      }
      viewerRef.current = null;
      primitivesRef.current = freshPrimitives();
      motionRef.current = freshMotion();
      setViewerReady(false);
    };
  }, []);

  // ── Store subscriptions ─────────────────────────────────────────────────
  const satellites = useStore((s) => s.satellites);
  const agencyFilter = useStore((s) => s.agencyFilter);
  const selectedSatelliteId = useStore((s) => s.selectedSatelliteId);
  const hoveredSatelliteId = useStore((s) => s.hoveredSatelliteId);
  const selectedOrbit = useStore((s) => s.selectedOrbit);
  const hotspots = useStore((s) => s.hotspots);
  const debrisClouds = useStore((s) => s.debrisClouds);
  const snapshotTimestamp = useStore((s) => s.snapshotTimestamp);
  const testActive = useTestMode((s) => s.testActive);
  const testSatellites = useTestMode((s) => s.testSatellites);
  const testSim = useTestMode((s) => s.sim);
  const overrideAMode = useTestMode((s) => s.overrideA.mode);
  const overrideBMode = useTestMode((s) => s.overrideB.mode);
  const computed = useTestMode((s) => s.computed);
  const selectedAId = useTestMode((s) => s.selectedA?.norad_id ?? null);
  const selectedBId = useTestMode((s) => s.selectedB?.norad_id ?? null);

  // In threat mode only render satellites that appear in alerts.
  const alertSatIds = useMemo(() => {
    if (mode !== 'threat') return null;
    const ids = new Set();
    for (const a of alerts) {
      if (a.sat1?.id != null) ids.add(normalizeId(a.sat1.id));
      if (a.sat2?.id != null) ids.add(normalizeId(a.sat2.id));
    }
    return ids;
  }, [mode, alerts]);

  const visibleSats = useMemo(() => {
    let base = agencyFilter
      ? satellites.filter((sat) => agencyFilter.includes(sat.agency))
      : satellites;
    if (alertSatIds !== null) base = base.filter((s) => alertSatIds.has(normalizeId(s.norad_id)));
    return base;
  }, [satellites, agencyFilter, alertSatIds]);

  // Threat level per satellite from conjunction CPI and debris exposure.
  const threatLevels = useMemo(() => {
    const levels = new Map();
    const bump = (id, level) => {
      if (id == null || level <= 0) return;
      const key = normalizeId(id);
      if ((levels.get(key) ?? 0) < level) levels.set(key, level);
    };
    for (const a of alerts) {
      const cpi = Number(a.cpi_score ?? 0);
      const level = cpi >= 8 ? THREAT_WARNING : cpi >= 5 ? THREAT_CAUTION : 0;
      bump(a.sat1?.id, level);
      bump(a.sat2?.id, level);
    }
    for (const cloud of debrisClouds || EMPTY) {
      for (const sat of cloud.affected_satellites || EMPTY) {
        bump(sat.norad_id, sat.risk_band === 'high' ? THREAT_WARNING : THREAT_CAUTION);
      }
    }
    return levels;
  }, [alerts, debrisClouds]);

  // ── Snapshot ingest: add/remove billboards, re-baseline dead reckoning ──
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewerReady || !viewer || viewer.isDestroyed()) return;
    const prims = primitivesRef.current;
    const { billboards, map } = prims;
    if (!billboards) return;
    const motion = motionRef.current;

    const parsed = snapshotTimestamp ? Date.parse(snapshotTimestamp) : Number.NaN;
    const snapshotMs = Number.isFinite(parsed) ? parsed : Date.now();
    const prevMs = motion.lastSnapshotTimestampMs;
    // REST/WS races can deliver the same or an older snapshot; re-baselining
    // on those would yank dots backwards. Still sync membership (filters).
    const isNewSnapshot = prevMs == null
      || (snapshotMs !== prevMs && !(snapshotMs < prevMs && snapshotMs > prevMs - 10000));
    const deltaSeconds = prevMs != null ? Math.max((snapshotMs - prevMs) / 1000, 0.001) : 1;
    const prevPositions = motion.snapshotPositions;
    const nextPositions = isNewSnapshot ? new Map() : prevPositions;
    const seen = new Set();

    for (const sat of visibleSats) {
      if (!sat.position) continue;
      const id = sat.norad_id;
      seen.add(id);
      let item = map.get(id);
      if (item && !isNewSnapshot) continue;

      let velocity = null;
      if (sat.velocity) {
        velocity = { vx: sat.velocity.vx || 0, vy: sat.velocity.vy || 0, vz: sat.velocity.vz || 0 };
      } else {
        const prev = prevPositions.get(id);
        if (prev) {
          velocity = {
            vx: (sat.position.x - prev.x) / deltaSeconds,
            vy: (sat.position.y - prev.y) / deltaSeconds,
            vz: (sat.position.z - prev.z) / deltaSeconds,
          };
        }
      }
      if (isNewSnapshot) nextPositions.set(id, sat.position);

      if (item) {
        item.name = sat.name;
        item.eciPosition = sat.position;
        item.eciVelocity = velocity;
        // Carry the gap to where the dot is drawn now as a decaying offset.
        item.correction = correctionFor(item.renderEci, sat.position, item.correctionBuf);
      } else {
        const { x, y, z } = sat.position;
        item = {
          billboard: billboards.add({
            position: new Cesium.Cartesian3(x * 1000, y * 1000, z * 1000),
            image: '/dot-medium.png',
            color: COLORS.satNominal,
            scale: SATELLITE_DEFAULT_SCALE,
            scaleByDistance: SATELLITE_SCALE_BY_DISTANCE,
            id,
          }),
          name: sat.name,
          eciPosition: sat.position,
          eciVelocity: velocity,
          renderEci: { x, y, z },
          correction: null,
          correctionBuf: { x: 0, y: 0, z: 0 },
        };
        map.set(id, item);
      }
    }

    for (const [id, item] of map) {
      if (!seen.has(id)) {
        billboards.remove(item.billboard);
        map.delete(id);
        if (prims.focusedItem === item) prims.focusedItem = null;
      }
    }

    if (isNewSnapshot) {
      const nowPerf = performance.now();
      motion.simRate = estimateSimRate(prevMs, motion.lastSnapshotReceivedPerfMs, snapshotMs, nowPerf, motion.simRate || 1);
      motion.snapshotPositions = nextPositions;
      motion.lastSnapshotTimestampMs = snapshotMs;
      motion.lastSnapshotReceivedPerfMs = nowPerf;
      motion.settled = false;
    }
    viewer.scene.requestRender();
  }, [viewerReady, visibleSats, snapshotTimestamp]);

  // ── Billboard styling: colour only for state ────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewerReady || !viewer || viewer.isDestroyed()) return;
    const prims = primitivesRef.current;
    const { map, focusLabel } = prims;
    if (!focusLabel) return;

    const aId = normalizeId(selectedAId);
    const bId = normalizeId(selectedBId);

    for (const [id, item] of map) {
      const isA = aId != null && id === aId;
      const isB = bId != null && id === bId;
      const isSelected = id === selectedSatelliteId;
      const isHovered = id === hoveredSatelliteId;
      const threat = threatLevels.get(id) ?? 0;
      const bb = item.billboard;

      if (isA) bb.color = COLORS.satSelectedA;
      else if (isB) bb.color = COLORS.satSelectedB;
      else if (isSelected) bb.color = COLORS.satSelected;
      else if (threat === THREAT_WARNING) bb.color = COLORS.satWarning;
      else if (threat === THREAT_CAUTION) bb.color = COLORS.satCaution;
      else if (isHovered) bb.color = COLORS.satHover;
      else bb.color = COLORS.satNominal;

      bb.scale = (isA || isB || isSelected)
        ? SATELLITE_SELECTED_SCALE
        : (isHovered ? SATELLITE_HOVER_SCALE : SATELLITE_DEFAULT_SCALE);
    }

    const hoverItem = hoveredSatelliteId != null ? map.get(hoveredSatelliteId) : null;
    prims.focusedItem = hoverItem || null;
    if (hoverItem) {
      focusLabel.text = hoverItem.name || `#${hoveredSatelliteId}`;
      focusLabel.position = hoverItem.billboard.position;
      focusLabel.show = true;
    } else {
      focusLabel.show = false;
    }
    viewer.scene.requestRender();
  }, [viewerReady, visibleSats, hoveredSatelliteId, selectedSatelliteId, selectedAId, selectedBId, threatLevels]);

  // ── Selected orbit ──────────────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewerReady || !viewer || viewer.isDestroyed()) return;
    const prims = primitivesRef.current;
    const { orbitLines, orbitPolyline, orbitMaterial } = prims;
    if (!orbitLines || !orbitPolyline) return;

    for (const polyline of prims.orbitSegmentPolylines) orbitLines.remove(polyline);
    prims.orbitSegmentPolylines = [];

    // Split where consecutive samples jump (propagation gaps).
    const segments = [];
    let current = [];
    let previous = null;
    for (const sample of selectedOrbit || EMPTY) {
      if (!sample?.position || !sample?.epoch_utc) continue;
      const pos = sample.position;
      if (previous) {
        const dx = pos.x - previous.x;
        const dy = pos.y - previous.y;
        const dz = pos.z - previous.z;
        if (Math.sqrt((dx * dx) + (dy * dy) + (dz * dz)) > ORBIT_BREAK_DISTANCE_KM) {
          if (current.length > 1) segments.push(current);
          current = [];
        }
      }
      current.push(sample);
      previous = pos;
    }
    if (current.length > 1) segments.push(current);

    // Draw the orbit as a closed ring in space: every sample uses the Earth
    // orientation at the first sample's time. Rotating each sample by its own
    // time instead traces a ground track that drifts ~25° per revolution, so
    // the path never joins up with itself.
    const frameDate = segments.length > 0 ? new Date(segments[0][0].epoch_utc) : null;
    const toPositions = (segment) => segment.map((s) => toCartesian3(s.position, frameDate));
    if (segments.length === 1 && segments[0].length > 2) {
      segments[0] = [...segments[0], segments[0][0]]; // close the loop
    }

    if (segments.length > 0) {
      orbitPolyline.positions = toPositions(segments[0]);
      orbitPolyline.show = true;
    } else {
      orbitPolyline.show = false;
    }
    for (let i = 1; i < segments.length; i += 1) {
      prims.orbitSegmentPolylines.push(orbitLines.add({
        show: true,
        positions: toPositions(segments[i]),
        width: 1.5,
        material: orbitMaterial,
        arcType: Cesium.ArcType.NONE,
        id: `selected-orbit-${i}`,
      }));
    }
    viewer.scene.requestRender();
  }, [viewerReady, selectedOrbit]);

  // ── Conjunction hotspots (rebuilt only when their content changes) ──────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewerReady || !viewer || viewer.isDestroyed()) return;
    const prims = primitivesRef.current;
    if (!prims.hotspotPoints) return;

    const list = (hotspots || EMPTY)
      .filter((h) => h?.position && h?.tca_utc)
      .slice(0, MAX_HOTSPOTS);
    const key = list.map((h) => `${h.sat1?.id}:${h.sat2?.id}:${h.tca_utc}:${h.zone_radius_km}:${h.hotspot_score}`).join('|');
    if (key === prims.hotspotKey) return;
    prims.hotspotKey = key;

    for (const entity of prims.hotspotEntities) viewer.entities.remove(entity);
    prims.hotspotEntities = [];
    prims.hotspotPoints.removeAll();

    viewer.entities.suspendEvents();
    for (const hotspot of list) {
      const position = toCartesian3(hotspot.position, new Date(hotspot.tca_utc));
      const score = Number(hotspot.hotspot_score || 0);
      // No computed zone radius → draw the marker only, never an invented sphere.
      const radiusKm = Number(hotspot.zone_radius_km);
      const radiusM = Number.isFinite(radiusKm) && radiusKm > 0 ? radiusKm * 1000 : null;

      prims.hotspotPoints.add({
        position,
        color: COLORS.hotspotCore,
        pixelSize: 5 + (Math.min(score, 1) * 3),
        outlineColor: COLORS.void,
        outlineWidth: 1,
      });
      if (radiusM == null) continue;
      prims.hotspotEntities.push(viewer.entities.add({
        position,
        ellipsoid: {
          radii: new Cesium.Cartesian3(radiusM, radiusM, radiusM),
          material: COLORS.hotspotZone,
          outline: false,
          slicePartitions: 24,
          stackPartitions: 12,
        },
      }));
    }
    viewer.entities.resumeEvents();
    viewer.scene.requestRender();
  }, [viewerReady, hotspots]);

  // ── Debris clouds (rebuilt only when their geometry changes) ────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewerReady || !viewer || viewer.isDestroyed()) return;
    const prims = primitivesRef.current;

    // Geometry comes only from the backend cloud: fragment centroid, the
    // percentile radius of the real fragment spread, and the fragments themselves.
    const clouds = [];
    for (const cloud of debrisClouds || EMPTY) {
      const center = cloudCentroid(cloud);
      if (!center) continue;
      const radius = cloudRadius(cloud);
      const shellRadii = Array.isArray(cloud.shells)
        ? cloud.shells.map((sh) => Number(sh?.radius_km)).filter((r) => Number.isFinite(r) && r > 0)
        : [];
      const radii = (shellRadii.length > 0 ? shellRadii : radius ? [radius.km] : []).sort((a, b) => a - b);
      const fragments = cloudFragments(cloud, 300);
      clouds.push({ cloud, center, radii, fragments, label: cloudLabel(cloud) });
    }
    const key = clouds.map(({ cloud, center, radii, fragments, label }) => (
      `${cloud.id}:${cloudEpoch(cloud)}:${Math.round(center.x)},${Math.round(center.y)},${Math.round(center.z)}:${radii.map((r) => r.toFixed(1)).join(',')}:${fragments.length}:${label}`
    )).join('|');
    if (key === prims.debrisKey) return;
    prims.debrisKey = key;

    for (const entity of prims.debrisEntities) viewer.entities.remove(entity);
    prims.debrisEntities = [];
    if (prims.debrisPoints) prims.debrisPoints.removeAll();

    viewer.entities.suspendEvents();
    // The same event can arrive from both the forecast and fragment sources;
    // label it once so identical tags don't stack on top of each other.
    const labelled = new Set();
    for (const { cloud, center, radii, fragments, label } of clouds) {
      const epoch = cloudEpoch(cloud);
      const date = epoch ? new Date(epoch) : new Date();
      const position = toCartesian3(center, date);
      const showLabel = !labelled.has(label);
      labelled.add(label);

      // Real fragment positions (downsampled ≤ 300 by the backend / here).
      if (prims.debrisPoints) {
        for (const frag of fragments) {
          prims.debrisPoints.add({
            position: toCartesian3(frag, date),
            pixelSize: 2,
            color: COLORS.debrisFragment,
          });
        }
      }

      // Faint percentile shell(s); innermost slightly denser.
      radii.forEach((radiusKm, index) => {
        const t = radii.length > 1 ? 1 - (index / (radii.length - 1)) : 1;
        const r = radiusKm * 1000;
        prims.debrisEntities.push(viewer.entities.add({
          position,
          ellipsoid: {
            radii: new Cesium.Cartesian3(r, r, r),
            material: css(PALETTE.caution, 0.03 + (0.04 * t)),
            outline: false,
            slicePartitions: 24,
            stackPartitions: 12,
          },
        }));
      });

      prims.debrisEntities.push(viewer.entities.add({
        position,
        point: {
          pixelSize: 6,
          color: COLORS.debrisCore,
          outlineColor: COLORS.void,
          outlineWidth: 1,
        },
        label: showLabel ? {
          text: label,
          font: SMALL_LABEL_FONT,
          fillColor: COLORS.debrisText,
          style: Cesium.LabelStyle.FILL,
          showBackground: true,
          backgroundColor: COLORS.labelBg,
          backgroundPadding: new Cesium.Cartesian2(6, 3),
          pixelOffset: new Cesium.Cartesian2(0, -12),
          horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        } : undefined,
      }));
    }
    viewer.entities.resumeEvents();
    viewer.scene.requestRender();
  }, [viewerReady, debrisClouds]);

  // ── Test-mode approach line ─────────────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewerReady || !viewer || viewer.isDestroyed()) return;
    const { approachLine, approachMaterial } = primitivesRef.current;
    if (!approachLine || !approachMaterial) return;

    const satA = testSatellites.a;
    const satB = testSatellites.b;
    if (!testActive || !satA?.position || !satB?.position || !computed
      || overrideAMode !== 'override' || overrideBMode !== 'override') {
      if (approachLine.show) {
        approachLine.show = false;
        viewer.scene.requestRender();
      }
      return;
    }

    const epochDate = new Date(testSim.currentEpochUtc || satA.epochUtc || satB.epochUtc || Date.now());
    const separationKm = Math.sqrt(
      ((satA.position.x - satB.position.x) ** 2)
      + ((satA.position.y - satB.position.y) ** 2)
      + ((satA.position.z - satB.position.z) ** 2),
    );
    approachMaterial.uniforms.color = separationKm < 1
      ? COLORS.approach.warning
      : separationKm < 100 ? COLORS.approach.caution : COLORS.approach.nominal;
    approachLine.positions = [toCartesian3(satA.position, epochDate), toCartesian3(satB.position, epochDate)];
    approachLine.show = true;
    viewer.scene.requestRender();
  }, [viewerReady, testActive, testSatellites, testSim, overrideAMode, overrideBMode, computed]);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', background: PALETTE.void }}
    />
  );
};

export default CesiumGlobe;
