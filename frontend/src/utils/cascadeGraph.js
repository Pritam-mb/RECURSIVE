/**
 * cascadeGraph.js
 * Builds a force-graph data structure from the alerts array stored in Zustand.
 */

let _nodePositionCache = new Map();

/**
 * buildCascadeGraph(alerts)
 * Returns { nodes: [], edges: [] } suitable for CascadeDiagram.
 */
export function buildCascadeGraph(alerts) {
  const nodeMap = new Map();
  const edges = [];

  const ensureNode = (id, name, agency, cpi, severity) => {
    if (!nodeMap.has(id)) {
      // Preserve previous position so the graph doesn't jump on re-render
      const cached = _nodePositionCache.get(id);
      nodeMap.set(id, {
        id,
        label: name || `#${id}`,
        agency: agency || '',
        cpi: cpi ?? 0,
        severity: severity || 'WATCH',
        x: cached?.x ?? (Math.random() * 400 - 200),
        y: cached?.y ?? (Math.random() * 300 - 150),
        vx: 0,
        vy: 0,
      });
    } else {
      // Update dynamic fields without resetting position
      const node = nodeMap.get(id);
      if (cpi != null && cpi > node.cpi) {
        node.cpi = cpi;
        node.severity = severity || node.severity;
      }
    }
  };

  for (const alert of alerts || []) {
    const aId = alert.sat1?.id ?? alert.sat_a_norad_id;
    const bId = alert.sat2?.id ?? alert.sat_b_norad_id;
    const aName = alert.sat1?.name ?? alert.sat_a_name ?? `#${aId}`;
    const bName = alert.sat2?.name ?? alert.sat_b_name ?? `#${bId}`;
    const cpi = Number(alert.cpi_score ?? 0);
    const severity = alert.severity ?? (cpi >= 8 ? 'CRITICAL' : cpi >= 5 ? 'WARNING' : 'WATCH');

    if (aId != null) ensureNode(aId, aName, alert.sat1?.agency, cpi, severity);
    if (bId != null) ensureNode(bId, bName, alert.sat2?.agency, cpi, severity);

    if (aId != null && bId != null) {
      const tcaHours = alert.tca_hours != null
        ? Number(alert.tca_hours)
        : alert.tca_minutes != null
          ? Number(alert.tca_minutes) / 60
          : null;
      edges.push({
        id: alert.id ?? `${aId}-${bId}`,
        source: aId,
        target: bId,
        p_collision: Number(alert.p_collision ?? alert.probability_of_collision ?? alert.probability ?? 0),
        miss_distance: Number(alert.miss_distance_km ?? 0),
        tca_hours: tcaHours,
        tca_utc: alert.tca_utc ?? null,
        severity,
        status: 'active',
      });
    }
  }

  const nodes = Array.from(nodeMap.values());

  // Cache positions for continuity
  for (const n of nodes) {
    _nodePositionCache.set(n.id, { x: n.x, y: n.y });
  }

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
    .filter((id) => id !== sourceId && id !== primaryEdge.target)
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
