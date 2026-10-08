/**
 * cascadeGraph.js
 * Builds a force-graph data structure from the alerts array stored in Zustand.
 */

// Keyed by String(id) so numeric (sat_a_norad_id) and string (sat1.id)
// forms of the same NORAD id share one entry across alert refreshes.
const _nodePositionCache = new Map();

/** Canonical cache/dedup key for a node id (NORAD id may arrive as number or string). */
export const nodeKey = (id) => String(id);

// FNV-1a string hash → 32-bit seed
function hashString(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

// mulberry32 PRNG
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * seededUnit(id) → { u, v } in [-0.5, 0.5), deterministic for a given node id.
 * Used instead of Math.random() so the same satellite always starts in the
 * same place.
 */
export function seededUnit(id) {
  const rand = mulberry32(hashString(nodeKey(id)));
  return { u: rand() - 0.5, v: rand() - 0.5 };
}

/** Last known simulated position (centre-relative) for a node, written by CascadeDiagram. */
export function getCachedPosition(id) {
  return _nodePositionCache.get(nodeKey(id));
}

export function setCachedPosition(id, x, y) {
  _nodePositionCache.set(nodeKey(id), { x, y });
}

/**
 * buildCascadeGraph(alerts)
 * Returns { nodes: [], edges: [] } suitable for CascadeDiagram.
 */
export function buildCascadeGraph(alerts) {
  const nodeMap = new Map();
  const edges = [];

  // Returns the canonical id of the node (first-seen form) so edges reference
  // exactly the same value as node.id even if the alert mixes number/string ids.
  const ensureNode = (id, name, agency, cpi, severity) => {
    const key = nodeKey(id);
    if (!nodeMap.has(key)) {
      // Preserve previous position so the graph doesn't jump on re-render;
      // otherwise fall back to a deterministic seeded position.
      const cached = getCachedPosition(id);
      const seed = seededUnit(id);
      nodeMap.set(key, {
        id,
        label: name || `#${id}`,
        agency: agency || '',
        cpi: cpi ?? 0,
        severity: severity || 'WATCH',
        // Centre-relative coordinates (CascadeDiagram adds the canvas centre).
        x: cached?.x ?? seed.u * 200,
        y: cached?.y ?? seed.v * 150,
        vx: 0,
        vy: 0,
      });
    } else {
      // Update dynamic fields without resetting position
      const node = nodeMap.get(key);
      if (cpi != null && cpi > node.cpi) {
        node.cpi = cpi;
        node.severity = severity || node.severity;
      }
    }
    return nodeMap.get(key).id;
  };

  for (const alert of alerts || []) {
    const aId = alert.sat1?.id ?? alert.sat_a_norad_id;
    const bId = alert.sat2?.id ?? alert.sat_b_norad_id;
    const aName = alert.sat1?.name ?? alert.sat_a_name ?? `#${aId}`;
    const bName = alert.sat2?.name ?? alert.sat_b_name ?? `#${bId}`;
    const cpi = Number(alert.cpi_score ?? 0);
    const severity = alert.severity ?? (cpi >= 8 ? 'CRITICAL' : cpi >= 5 ? 'WARNING' : 'WATCH');

    const aNode = aId != null ? ensureNode(aId, aName, alert.sat1?.agency, cpi, severity) : null;
    const bNode = bId != null ? ensureNode(bId, bName, alert.sat2?.agency, cpi, severity) : null;

    if (aNode != null && bNode != null) {
      const tcaHours = alert.tca_hours != null
        ? Number(alert.tca_hours)
        : alert.tca_minutes != null
          ? Number(alert.tca_minutes) / 60
          : null;
      edges.push({
        id: alert.id ?? `${aId}-${bId}`,
        source: aNode,
        target: bNode,
        p_collision: Number(alert.p_collision ?? alert.probability_of_collision ?? alert.probability ?? 0),
        miss_distance: Number(alert.miss_distance_km ?? 0),
        tca_hours: tcaHours,
        tca_utc: alert.tca_utc ?? null,
        severity,
        status: 'active',
      });
    }
  }

  // Positions are cached by CascadeDiagram from the live simulation
  // (centre-relative), so a node keeps its place across alert refreshes.
  const nodes = Array.from(nodeMap.values());

  return { nodes, edges };
}

/**
 * markEdgeResolved(graph, alertId)
 * Sets edge.status = 'resolved' for the given alert ID.
 */
export function markEdgeResolved(graph, alertId) {
  return {
    ...graph,
    edges: graph.edges.map((e) =>
      e.id === alertId ? { ...e, status: 'resolved' } : e
    ),
  };
}

/**
 * addCascadeEdges(graph, primaryAlertId, affectedSatIds)
 * Adds orange cascade edges from the primary pair's maneuvering satellite
 * to all affected downstream satellites.
 */
export function addCascadeEdges(graph, primaryAlertId, affectedSatIds) {
  const primaryEdge = graph.edges.find((e) => e.id === primaryAlertId);
  if (!primaryEdge) return graph;

  const sourceId = primaryEdge.source;
  const newEdges = (affectedSatIds || [])
    .filter((id) => nodeKey(id) !== nodeKey(sourceId) && nodeKey(id) !== nodeKey(primaryEdge.target))
    .map((id) => ({
      id: `cascade-${primaryAlertId}-${id}`,
      source: sourceId,
      target: id,
      p_collision: 0,
      miss_distance: 0,
      severity: 'CASCADE',
      status: 'cascade',
    }));

  return { ...graph, edges: [...graph.edges, ...newEdges] };
}
