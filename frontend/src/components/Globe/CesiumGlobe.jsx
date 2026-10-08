import React, { useEffect, useRef } from 'react';
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import useStore from '../../store/useStore';
import useTestMode from '../../hooks/useTestMode';
import { computeGmst, eciToCesiumCartesian } from '../../utils/coords';

// Set Ion token
Cesium.Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_TOKEN;

const SATELLITE_DEFAULT_SCALE = 0.26;
const SATELLITE_HOVER_SCALE = 0.34;
const SATELLITE_SELECTED_SCALE = 0.42;
const SATELLITE_SCALE_BY_DISTANCE = new Cesium.NearFarScalar(
  1000000.0,
  1.4,
  30000000.0,
  0.2,
);
const ORBIT_BREAK_DISTANCE_KM = 8000;
const ZERO_VELOCITY = { vx: 0, vy: 0, vz: 0 };

function eciToCesiumCartesianFast(eciPos, cosG, sinG) {
  return {
    x: (eciPos.x * cosG + eciPos.y * sinG) * 1000,
    y: (-eciPos.x * sinG + eciPos.y * cosG) * 1000,
    z: eciPos.z * 1000,
  };
}

function setBillboardPosition(billboard, cartesian) {
  // Always assign a new position instance to ensure Cesium detects the property change
  billboard.position = new Cesium.Cartesian3(cartesian.x, cartesian.y, cartesian.z);
}

const CesiumGlobe = ({ mode = 'live', alerts = [], onSatelliteSelect }) => {
  const viewerRef = useRef(null);
  const containerRef = useRef(null);
  const motionRef = useRef({
    snapshotPositions: new Map(),
    previousSnapshotPositions: new Map(),
    lastSnapshotTimestampMs: null,
    previousSnapshotTimestampMs: null,
    lastSnapshotReceivedPerfMs: 0,
    snapshotIntervalMs: 1000,
    animationFrameId: null,
  });
  const primitivesRef = useRef({
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
    hotspotZones: [],
    debrisEntities: [],
    map: new Map(),
  });
  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;
    let viewer = null;
    let rafId = null;
    let ro = null;

    // Defer by one animation frame so the CSS Grid has committed its layout.
    // Without this, Cesium reads zero/wrong container dimensions and either
    // places the camera at the wrong distance or overflows the container.
    rafId = requestAnimationFrame(() => {
      if (!containerRef.current) return;

      viewer = new Cesium.Viewer(containerRef.current, {
        imageryProvider: false,
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
      });

      viewer.scene.requestRenderMode = false;

      // Threat mode: red-tinted space background
      if (mode === 'threat') {
        viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#0a0005');
      }
      viewer.scene.maximumRenderTimeChange = 0;
      viewer.scene.fxaa = false;

      // Add local imagery
      Cesium.SingleTileImageryProvider.fromUrl('/earth-realistic-8k.webp').then((provider) => {
        if (!viewer.isDestroyed()) {
          viewer.imageryLayers.addImageryProvider(provider);
          viewer.scene.requestRender();
        }
      }).catch(console.error);

      // Black background, no atmosphere glow
      viewer.scene.backgroundColor = Cesium.Color.BLACK;
      viewer.scene.sun.show = false;
      viewer.scene.moon.show = false;
      viewer.scene.skyBox.show = false;
      viewer.scene.skyAtmosphere.show = false;
      viewer.scene.fog.enabled = false;
      viewer.scene.globe.showGroundAtmosphere = false;
      viewer.scene.globe.enableLighting = false;
      viewer.scene.globe.depthTestAgainstTerrain = false;

      // Set initial camera to see Earth from space
      viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(30, 20, 25000000),
      });

      // Force Cesium to measure the real container size right now,
      // then re-render so the globe fills the panel correctly.
      viewer.resize();
      viewer.scene.requestRender();

      // Track container size changes (flex/grid can resize it later)
      ro = new ResizeObserver(() => {
        if (viewer && !viewer.isDestroyed()) {
          viewer.resize();
          viewer.scene.requestRender();
        }
      });
      ro.observe(containerRef.current);

      // Initialize Collections for high-performance rendering
      const billboards = new Cesium.BillboardCollection({
        blendOption: Cesium.BlendOption.TRANSLUCENT,
      });
      const labels = new Cesium.LabelCollection();
      const orbitLines = new Cesium.PolylineCollection();
      const hotspotPoints = new Cesium.PointPrimitiveCollection();
      viewer.scene.primitives.add(billboards);
      viewer.scene.primitives.add(labels);
      viewer.scene.primitives.add(orbitLines);
      viewer.scene.primitives.add(hotspotPoints);

      const focusLabel = labels.add({
        show: false,
        text: '',
        font: '11px JetBrains Mono',
        fillColor: Cesium.Color.fromCssColorString('#f3f4f6'),
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(15, -5),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      });

      const orbitMaterial = Cesium.Material.fromType('Color');
      orbitMaterial.uniforms.color = Cesium.Color.fromCssColorString('#22c55e').withAlpha(0.8);

      const approachMaterial = Cesium.Material.fromType('Color');
      approachMaterial.uniforms.color = Cesium.Color.fromCssColorString('#22c55e').withAlpha(0.9);

      const orbitPolyline = orbitLines.add({
        show: false,
        positions: [],
        width: 2,
        material: orbitMaterial,
        arcType: Cesium.ArcType.NONE,
        id: 'selected-orbit',
      });

      const approachLine = orbitLines.add({
        show: false,
        positions: [],
        width: 2,
        material: approachMaterial,
        arcType: Cesium.ArcType.NONE,
        id: 'approach-line',
      });

      primitivesRef.current.billboards = billboards;
      primitivesRef.current.labels = labels;
      primitivesRef.current.orbitLines = orbitLines;
      primitivesRef.current.focusLabel = focusLabel;
      primitivesRef.current.orbitPolyline = orbitPolyline;
      primitivesRef.current.orbitMaterial = orbitMaterial;
      primitivesRef.current.approachLine = approachLine;
      primitivesRef.current.approachMaterial = approachMaterial;
      primitivesRef.current.hotspotPoints = hotspotPoints;
      primitivesRef.current.hotspotZones = [];
      primitivesRef.current.debrisEntities = [];

      // Hover and click handlers for satellite focus
      const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
      const updateHoveredSatellite = (position) => {
        const picked = viewer.scene.pick(position);
        const nextHoveredId = Cesium.defined(picked) && typeof picked.id === 'number'
          ? picked.id
          : null;
        const store = useStore.getState();

        if (store.hoveredSatelliteId !== nextHoveredId) {
          store.setHoveredSatelliteId(nextHoveredId);
        }

        viewer.scene.requestRender();
        return nextHoveredId;
      };

      handler.setInputAction((movement) => {
        updateHoveredSatellite(movement.endPosition);
      }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);

      handler.setInputAction((click) => {
        const pickedId = updateHoveredSatellite(click.position);
        if (pickedId) {
          useStore.getState().setSelectedSatelliteId(pickedId);
          if (onSatelliteSelect) onSatelliteSelect(pickedId);
        }
      }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

      const clearHover = () => {
        const store = useStore.getState();
        if (store.hoveredSatelliteId !== null) {
          store.setHoveredSatelliteId(null);
        }
        viewer.scene.requestRender();
      };

      viewer.scene.canvas.addEventListener('mouseleave', clearHover);
      viewerRef.current = viewer;

      // Store cleanup handles so the return callback can reach them
      viewerRef._handler = handler;
      viewerRef._clearHover = clearHover;
    });

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      if (ro) ro.disconnect();
      const v = viewerRef.current;
      if (v) {
        try { viewerRef._handler?.destroy(); } catch (_) {}
        try { v.scene.canvas.removeEventListener('mouseleave', viewerRef._clearHover); } catch (_) {}
        if (!v.isDestroyed()) v.destroy();
      }
      viewerRef.current = null;
    };
  }, []);

  // Update satellite positions when store changes
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
  // In threat mode only render satellites that appear in alerts
  const alertSatIds = React.useMemo(() => {
    if (mode !== 'threat') return null;
    const ids = new Set();
    for (const a of alerts) {
      if (a.sat1?.id != null) ids.add(a.sat1.id);
      if (a.sat2?.id != null) ids.add(a.sat2.id);
    }
    return ids;
  }, [mode, alerts]);

  const visibleSats = React.useMemo(() => {
    let base = agencyFilter
      ? satellites.filter((sat) => agencyFilter.includes(sat.agency))
      : satellites;
    if (alertSatIds !== null) {
      base = base.filter((s) => alertSatIds.has(s.norad_id));
    }
    return base;
  }, [satellites, agencyFilter, alertSatIds]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;

    const { billboards, focusLabel, map } = primitivesRef.current;
    if (!billboards || !focusLabel) return;

    const parsedSnapshotTimestampMs = snapshotTimestamp ? Date.parse(snapshotTimestamp) : Number.NaN;
    const snapshotTimestampMs = Number.isFinite(parsedSnapshotTimestampMs)
      ? parsedSnapshotTimestampMs
      : Date.now();
    const previousSnapshotTimestampMs = motionRef.current.lastSnapshotTimestampMs;

    // IMPORTANT: Ignore duplicate updates from REST vs WS race conditions.
    // If we process a duplicate timestamp, it resets our animation elapsed time to 0,
    // violently ripping the satellite back to its starting block.
    if (previousSnapshotTimestampMs != null) {
      if (snapshotTimestampMs === previousSnapshotTimestampMs) {
        return; // Ignore exact duplicates
      }
      // If timestamp went backwards (REST poll resolving after WS update), ignore it
      if (snapshotTimestampMs < previousSnapshotTimestampMs && snapshotTimestampMs > previousSnapshotTimestampMs - 10000) {
        return; 
      }
    }

    const previousSnapshotPositions = motionRef.current.snapshotPositions;
    const nextSnapshotPositions = new Map();
    const nextSnapshotVelocities = new Map();
    const existingIds = new Set();
    const focusedSatelliteId = useStore.getState().hoveredSatelliteId;
    const defaultColor = Cesium.Color.WHITE.withAlpha(0.55);
    const deltaSeconds = previousSnapshotTimestampMs != null
      ? Math.max((snapshotTimestampMs - previousSnapshotTimestampMs) / 1000, 0.001)
      : 1;

    for (const sat of visibleSats) {
      existingIds.add(sat.norad_id);
      nextSnapshotPositions.set(sat.norad_id, sat.position);

      const previousPosition = previousSnapshotPositions.get(sat.norad_id);
      const snapshotVelocity = sat.velocity
        ? {
            vx: sat.velocity.vx || 0,
            vy: sat.velocity.vy || 0,
            vz: sat.velocity.vz || 0,
          }
        : null;
      const derivedVelocity = previousPosition
        ? {
            vx: (sat.position.x - previousPosition.x) / deltaSeconds,
            vy: (sat.position.y - previousPosition.y) / deltaSeconds,
            vz: (sat.position.z - previousPosition.z) / deltaSeconds,
          }
        : ZERO_VELOCITY;
      const velocity = snapshotVelocity || derivedVelocity;

      let item = map.get(sat.norad_id);
      if (item) {
        item.previousEciPosition = item.eciPosition || sat.position;
        item.previousEciVelocity = item.eciVelocity || velocity;
        item.name = sat.name;
        item.agencyColor = sat.agency_color || null;
        item.eciPosition = sat.position;
        item.eciVelocity = velocity;
        item.snapshotEpochUtc = sat.epoch_utc || null;
      } else {
        const satColor = sat.agency_color
          ? Cesium.Color.fromCssColorString(sat.agency_color).withAlpha(0.7)
          : defaultColor;
        const billboard = billboards.add({
          position: new Cesium.Cartesian3(),
          image: '/dot-medium.png',
          color: satColor,
          scale: SATELLITE_DEFAULT_SCALE,
          scaleByDistance: SATELLITE_SCALE_BY_DISTANCE,
          id: sat.norad_id,
        });

        item = {
          billboard,
          name: sat.name,
          agencyColor: sat.agency_color || null,
          previousEciPosition: sat.position,
          previousEciVelocity: velocity,
          eciPosition: sat.position,
          eciVelocity: velocity,
          snapshotEpochUtc: sat.epoch_utc || null,
        };
        map.set(sat.norad_id, item);
      }

      nextSnapshotVelocities.set(sat.norad_id, velocity);
    }

    // Remove entities no longer in the satellite list
    for (const [id, item] of map.entries()) {
      if (!existingIds.has(id)) {
        billboards.remove(item.billboard);
        map.delete(id);
      }
    }

    motionRef.current.snapshotPositions = nextSnapshotPositions;
    motionRef.current.previousSnapshotPositions = previousSnapshotPositions;
    motionRef.current.snapshotVelocities = nextSnapshotVelocities;
    motionRef.current.previousSnapshotTimestampMs = previousSnapshotTimestampMs;
    motionRef.current.lastSnapshotTimestampMs = snapshotTimestampMs;
    motionRef.current.lastSnapshotReceivedPerfMs = performance.now();
    motionRef.current.snapshotIntervalMs = previousSnapshotTimestampMs != null
      ? Math.max(snapshotTimestampMs - previousSnapshotTimestampMs, 250)
      : 1000;

    const updateDate = new Date(snapshotTimestampMs);
    const gmst = computeGmst(updateDate);
    const cosG = Math.cos(gmst);
    const sinG = Math.sin(gmst);

    for (const item of map.values()) {
      if (!item.eciPosition) continue;
      const cartesian = eciToCesiumCartesianFast(item.eciPosition, cosG, sinG);
      setBillboardPosition(item.billboard, cartesian);
    }

    const focusedItem = focusedSatelliteId != null ? map.get(focusedSatelliteId) : null;
    if (focusedItem) {
      focusLabel.show = true;
      focusLabel.text = focusedItem.name;
      focusLabel.position = focusedItem.billboard.position;
    } else {
      focusLabel.show = false;
    }

    viewer.scene.requestRender();
  }, [visibleSats, snapshotTimestamp]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;

    const {
      billboards,
      orbitPolyline,
      orbitLines,
      orbitMaterial,
      orbitSegmentPolylines,
      map,
      focusLabel,
    } = primitivesRef.current;
    if (!billboards || !orbitPolyline || !orbitLines || !orbitMaterial || !focusLabel) return;

    for (const polyline of orbitSegmentPolylines || []) {
      orbitLines.remove(polyline);
    }
    primitivesRef.current.orbitSegmentPolylines = [];

    const selectedColor = Cesium.Color.fromCssColorString('#22c55e');
    const selectedAColor = Cesium.Color.fromCssColorString('#22c55e');
    const selectedBColor = Cesium.Color.fromCssColorString('#eab308');
    const hoverColor = Cesium.Color.WHITE.withAlpha(0.95);
    const defaultColor = Cesium.Color.WHITE.withAlpha(0.55);
    const affectedHighColor = Cesium.Color.fromCssColorString('#ef4444').withAlpha(0.9);
    const affectedMediumColor = Cesium.Color.fromCssColorString('#f97316').withAlpha(0.85);
    const affectedLowColor = Cesium.Color.fromCssColorString('#eab308').withAlpha(0.8);
    const normalizedAId = selectedAId == null
      ? null
      : (Number.isNaN(Number(selectedAId)) ? selectedAId : Number(selectedAId));
    const normalizedBId = selectedBId == null
      ? null
      : (Number.isNaN(Number(selectedBId)) ? selectedBId : Number(selectedBId));

    const affectedRisk = new Map();
    for (const cloud of debrisClouds || []) {
      for (const sat of cloud.affected_satellites || []) {
        const satId = sat.norad_id;
        if (satId == null) continue;
        const band = sat.risk_band || 'low';
        const existing = affectedRisk.get(satId);
        if (!existing || (existing === 'low' && band !== 'low') || (existing === 'medium' && band === 'high')) {
          affectedRisk.set(satId, band);
        }
      }
    }

    for (const [id, item] of map.entries()) {
      const isSelected = id === selectedSatelliteId;
      const isHovered = id === hoveredSatelliteId;
      const isSelectedA = normalizedAId != null && id === normalizedAId;
      const isSelectedB = normalizedBId != null && id === normalizedBId;

      if (isSelectedA) {
        item.billboard.color = selectedAColor;
      } else if (isSelectedB) {
        item.billboard.color = selectedBColor;
      } else if (isSelected) {
        item.billboard.color = selectedColor;
      } else if (isHovered) {
        item.billboard.color = hoverColor;
      } else {
        const riskBand = affectedRisk.get(id);
        if (riskBand === 'high') {
          item.billboard.color = affectedHighColor;
        } else if (riskBand === 'medium') {
          item.billboard.color = affectedMediumColor;
        } else if (riskBand === 'low') {
          item.billboard.color = affectedLowColor;
        } else {
          const agencyColor = item.agencyColor
            ? Cesium.Color.fromCssColorString(item.agencyColor).withAlpha(0.7)
            : defaultColor;
          item.billboard.color = agencyColor;
        }
      }

      if (isSelectedA || isSelectedB || isSelected) {
        item.billboard.scale = SATELLITE_SELECTED_SCALE;
      } else if (isHovered) {
        item.billboard.scale = SATELLITE_HOVER_SCALE;
      } else {
        item.billboard.scale = SATELLITE_DEFAULT_SCALE;
      }
    }

    const hoverItem = hoveredSatelliteId != null ? map.get(hoveredSatelliteId) : null;
    if (hoverItem) {
      focusLabel.show = true;
      focusLabel.text = hoverItem.name;
      focusLabel.position = hoverItem.billboard.position;
    } else {
      focusLabel.show = false;
    }

    const segments = [];
    let currentSegment = [];
    let previous = null;

    for (const sample of selectedOrbit) {
      if (!sample?.position || !sample?.epoch_utc) continue;
      const pos = sample.position;

      if (previous) {
        const dx = pos.x - previous.x;
        const dy = pos.y - previous.y;
        const dz = pos.z - previous.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist > ORBIT_BREAK_DISTANCE_KM) {
          if (currentSegment.length > 1) {
            segments.push(currentSegment);
          }
          currentSegment = [];
        }
      }

      currentSegment.push(sample);
      previous = pos;
    }

    if (currentSegment.length > 1) {
      segments.push(currentSegment);
    }

    if (segments.length > 0) {
      const orbitPositions = segments[0]
        .map((sample) => {
          const sampleDate = new Date(sample.epoch_utc);
          const cartesian = eciToCesiumCartesian(sample.position, sampleDate);
          return new Cesium.Cartesian3(cartesian.x, cartesian.y, cartesian.z);
        });

      if (orbitPositions.length > 1) {
        orbitPolyline.show = true;
        orbitPolyline.positions = orbitPositions;
        orbitMaterial.uniforms.color = selectedColor.withAlpha(0.8);
      } else {
        orbitPolyline.show = false;
      }
    } else {
      orbitPolyline.show = false;
    }

    const nextSegmentPolylines = [];
    for (const segment of segments.slice(1)) {
      const segmentPositions = segment.map((sample) => {
        const sampleDate = new Date(sample.epoch_utc);
        const cartesian = eciToCesiumCartesian(sample.position, sampleDate);
        return new Cesium.Cartesian3(cartesian.x, cartesian.y, cartesian.z);
      });

      if (segmentPositions.length > 1) {
        const polyline = orbitLines.add({
          show: true,
          positions: segmentPositions,
          width: 2,
          material: orbitMaterial,
          arcType: Cesium.ArcType.NONE,
          id: `selected-orbit-${nextSegmentPolylines.length + 1}`,
        });
        nextSegmentPolylines.push(polyline);
      }
    }

    primitivesRef.current.orbitSegmentPolylines = nextSegmentPolylines;

    viewer.scene.requestRender();
  }, [hoveredSatelliteId, selectedSatelliteId, selectedOrbit, selectedAId, selectedBId, debrisClouds]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;

    const { hotspotPoints } = primitivesRef.current;
    const currentHotspotZones = primitivesRef.current.hotspotZones || [];
    if (!hotspotPoints) return;

    for (const entity of currentHotspotZones) {
      viewer.entities.remove(entity);
    }
    primitivesRef.current.hotspotZones = [];

    hotspotPoints.removeAll();

    const visibleHotspots = (hotspots || []).slice(0, 40);
    for (const hotspot of visibleHotspots) {
      if (!hotspot?.position || !hotspot?.tca_utc) continue;

      const tcaDate = new Date(hotspot.tca_utc);
      const cartesian = eciToCesiumCartesian(hotspot.position, tcaDate);
      const score = Number(hotspot.hotspot_score || 0);
      const zoneRadiusKm = Number(hotspot.zone_radius_km || 100.0);
      const waveColor = Cesium.Color.fromCssColorString('#ef4444');
      const coreColor = Cesium.Color.fromCssColorString('#ef4444');

      hotspotPoints.add({
        position: new Cesium.Cartesian3(cartesian.x, cartesian.y, cartesian.z),
        color: coreColor.withAlpha(0.95),
        pixelSize: 8 + (score * 8),
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 1,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      });

      const shellFractions = [0.28, 0.56, 0.82, 1.0];
      const shellAlphas = [0.16, 0.12, 0.08, 0.05];
      const outlineAlphas = [0.35, 0.3, 0.24, 0.18];

      shellFractions.forEach((fraction, index) => {
        const shellEntity = viewer.entities.add({
          position: new Cesium.Cartesian3(cartesian.x, cartesian.y, cartesian.z),
          name: `hotspot-wave-${hotspot.sat1?.id || 'a'}-${hotspot.sat2?.id || 'b'}-${index}`,
          ellipsoid: {
            radii: new Cesium.Cartesian3(
              zoneRadiusKm * fraction * 1000.0,
              zoneRadiusKm * fraction * 1000.0,
              zoneRadiusKm * fraction * 1000.0,
            ),
            material: waveColor.withAlpha(shellAlphas[index]),
            outline: true,
            outlineColor: waveColor.withAlpha(outlineAlphas[index]),
            outlineWidth: 1,
          },
        });
        primitivesRef.current.hotspotZones.push(shellEntity);
      });
    }

    viewer.scene.requestRender();
  }, [hotspots]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;

    const currentEntities = primitivesRef.current.debrisEntities || [];
    for (const entity of currentEntities) {
      viewer.entities.remove(entity);
    }

    const nextEntities = [];
    for (const cloud of debrisClouds || []) {
      const center = cloud.center_eci_km || {};
      if (![center.x, center.y, center.z].every((value) => Number.isFinite(Number(value)))) {
        continue;
      }
      const tcaDate = cloud.tca_utc
        ? new Date(cloud.tca_utc)
        : (snapshotTimestamp ? new Date(snapshotTimestamp) : new Date());
      const cartesian = eciToCesiumCartesian(center, tcaDate);
      const shells = Array.isArray(cloud.shells) && cloud.shells.length > 0
        ? cloud.shells
        : [{ label: 'now', radius_km: Number(cloud.radius_km_now ?? cloud.radius_km_at_tca ?? 0) }];

      const debrisColor = Cesium.Color.fromCssColorString('#a855f7');

      // One ellipsoid per temporal shell, so the cloud keeps its layered
      // structure instead of collapsing into a single envelope sphere.
      const orderedShells = [...shells]
        .filter((shell) => Number.isFinite(Number(shell.radius_km)) && Number(shell.radius_km) > 0)
        .sort((a, b) => Number(a.radius_km) - Number(b.radius_km));

      if (orderedShells.length === 0) {
        continue;
      }

      const debrisRadiusKm = Math.max(...orderedShells.map((s) => Number(s.radius_km)));

      orderedShells.forEach((shell, index) => {
        const radiusKm = Number(shell.radius_km);
        // Shells are ordered inner -> outer, so invert the ramp: the outermost
        // shell is the faintest envelope and the innermost is most opaque.
        const t = orderedShells.length > 1 ? 1 - index / (orderedShells.length - 1) : 1;
        const fillAlpha = 0.10 + 0.24 * t;
        const outlineAlpha = 0.35 + 0.55 * t;

        nextEntities.push(viewer.entities.add({
          position: new Cesium.Cartesian3(cartesian.x, cartesian.y, cartesian.z),
          name: `${cloud.id}-shell-${shell.label ?? index}`,
          description: `${shell.label ?? 'shell'} — r=${radiusKm.toFixed(1)} km`,
          ellipsoid: {
            radii: new Cesium.Cartesian3(radiusKm * 1000.0, radiusKm * 1000.0, radiusKm * 1000.0),
            material: debrisColor.withAlpha(fillAlpha),
            outline: true,
            outlineColor: debrisColor.withAlpha(outlineAlpha),
            outlineWidth: index === orderedShells.length - 1 ? 2 : 1,
          },
        }));
      });

      // Bright inner hazard core point
      const debrisCore = viewer.entities.add({
        position: new Cesium.Cartesian3(cartesian.x, cartesian.y, cartesian.z),
        name: `${cloud.id}-core`,
        point: {
          pixelSize: 14,
          color: Cesium.Color.fromCssColorString('#f0abfc').withAlpha(0.98),
          outlineColor: Cesium.Color.fromCssColorString('#a855f7').withAlpha(0.9),
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
          scaleByDistance: new Cesium.NearFarScalar(1e5, 2.0, 3e7, 0.6),
        },
      });
      nextEntities.push(debrisCore);

      // Floating label — always visible
      const fragCount = cloud.fragment_count ?? 0;
      const labelText = `⚠ DEBRIS FIELD\n${fragCount} FRAGMENTS · r=${debrisRadiusKm.toFixed(0)}km`;
      const debrisLabel = viewer.entities.add({
        position: new Cesium.Cartesian3(cartesian.x, cartesian.y, cartesian.z),
        name: `${cloud.id}-label`,
        label: {
          text: labelText,
          font: 'bold 11px JetBrains Mono, monospace',
          fillColor: Cesium.Color.fromCssColorString('#e879f9'),
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 3,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          pixelOffset: new Cesium.Cartesian2(0, -(debrisRadiusKm > 500 ? 60 : 40)),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
          scaleByDistance: new Cesium.NearFarScalar(1e5, 1.4, 3e7, 0.5),
          translucencyByDistance: new Cesium.NearFarScalar(1e6, 1.0, 3e7, 0.7),
          horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
          verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
          showBackground: true,
          backgroundColor: Cesium.Color.fromCssColorString('#1a0030').withAlpha(0.82),
          backgroundPadding: new Cesium.Cartesian2(8, 5),
        },
      });
      nextEntities.push(debrisLabel);
    }

    primitivesRef.current.debrisEntities = nextEntities;
    viewer.scene.requestRender();
  }, [debrisClouds, snapshotTimestamp]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;

    const { approachLine, approachMaterial } = primitivesRef.current;
    if (!approachLine || !approachMaterial) return;

    const satA = testSatellites.a;
    const satB = testSatellites.b;

    if (!testActive || !satA || !satB || !computed || overrideAMode !== 'override' || overrideBMode !== 'override') {
      approachLine.show = false;
      viewer.scene.requestRender();
      return;
    }

    const epoch = testSim.currentEpochUtc || satA.epochUtc || satB.epochUtc || new Date().toISOString();
    const epochDate = new Date(epoch);

    const separationKm = Math.sqrt(
      (satA.position.x - satB.position.x) ** 2 +
      (satA.position.y - satB.position.y) ** 2 +
      (satA.position.z - satB.position.z) ** 2
    );

    let color = '#22c55e';
    if (separationKm < 1) {
      color = '#ef4444';
    } else if (separationKm < 10) {
      color = '#f97316';
    } else if (separationKm < 100) {
      color = '#eab308';
    }

    const cartA = eciToCesiumCartesian(satA.position, epochDate);
    const cartB = eciToCesiumCartesian(satB.position, epochDate);
    approachLine.positions = [
      new Cesium.Cartesian3(cartA.x, cartA.y, cartA.z),
      new Cesium.Cartesian3(cartB.x, cartB.y, cartB.z),
    ];
    approachMaterial.uniforms.color = Cesium.Color.fromCssColorString(color).withAlpha(0.9);
    approachLine.show = true;

    viewer.scene.requestRender();
  }, [testActive, testSatellites, testSim, overrideAMode, overrideBMode, computed]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || viewer.isDestroyed()) return;

    const updateMotion = () => {
      // Re-queue immediately to ensure the loop never dies
      motionRef.current.animationFrameId = requestAnimationFrame(updateMotion);

      const currentPerfMs = performance.now();

      const { map, focusLabel } = primitivesRef.current;
      const snapshotTimestampMs = motionRef.current.lastSnapshotTimestampMs;
      const snapshotReceivedPerfMs = motionRef.current.lastSnapshotReceivedPerfMs;

      if (snapshotTimestampMs == null || map.size === 0) return;

      const snapshotIntervalMs = motionRef.current.snapshotIntervalMs || 1000;
      const renderLagMs = Math.min(snapshotIntervalMs * 0.5, 500);
      const phase = ((currentPerfMs - snapshotReceivedPerfMs) + renderLagMs) / snapshotIntervalMs;

      const previousSnapshotTimestampMs = motionRef.current.previousSnapshotTimestampMs != null
        ? motionRef.current.previousSnapshotTimestampMs
        : snapshotTimestampMs - snapshotIntervalMs;
      const lerpT = Math.min(Math.max(phase, 0), 1);
      const easedT = lerpT * lerpT * (3 - (2 * lerpT));
      const extraSeconds = Math.max(phase - 1, 0) * (snapshotIntervalMs / 1000);

      const frameDate = phase <= 1
        ? new Date(previousSnapshotTimestampMs + (easedT * snapshotIntervalMs))
        : new Date(snapshotTimestampMs + (extraSeconds * 1000));
      const gmst = computeGmst(frameDate);
      const cosG = Math.cos(gmst);
      const sinG = Math.sin(gmst);
      
      const hoveredSatelliteId = useStore.getState().hoveredSatelliteId;
      const focusedItem = hoveredSatelliteId != null ? map.get(hoveredSatelliteId) : null;

      for (const item of map.values()) {
        if (!item.eciPosition) continue;

        const velocity = item.eciVelocity || ZERO_VELOCITY;

        const previousPosition = motionRef.current.previousSnapshotPositions.get(item.billboard.id) || item.eciPosition;
        const currentPosition = item.eciPosition;

        const targetEci = phase <= 1
          ? {
              x: previousPosition.x + ((currentPosition.x - previousPosition.x) * easedT),
              y: previousPosition.y + ((currentPosition.y - previousPosition.y) * easedT),
              z: previousPosition.z + ((currentPosition.z - previousPosition.z) * easedT),
            }
          : {
              x: currentPosition.x + (velocity.vx * extraSeconds),
              y: currentPosition.y + (velocity.vy * extraSeconds),
              z: currentPosition.z + (velocity.vz * extraSeconds),
            };

        const cartesian = eciToCesiumCartesianFast(targetEci, cosG, sinG);
        setBillboardPosition(item.billboard, cartesian);
      }

      if (focusedItem && focusLabel.show) {
        focusLabel.position = focusedItem.billboard.position;
      }
      
      viewer.scene.requestRender();
    };

    motionRef.current.animationFrameId = requestAnimationFrame(updateMotion);

    return () => {
      if (motionRef.current.animationFrameId) {
        cancelAnimationFrame(motionRef.current.animationFrameId);
      }
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', background: mode === 'threat' ? '#0a0005' : '#000' }}
    />
  );
};

export default CesiumGlobe;
