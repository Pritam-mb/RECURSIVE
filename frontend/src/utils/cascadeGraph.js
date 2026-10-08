/**
 * cascadeGraph.js
 * Builds the cascade risk graph from backend data only:
 *   - screening alerts            → satellite ↔ satellite conjunction edges
 *   - debris alerts + parent_event → collision event node → parent satellites,
 *                                    and event → every satellite its fragments threaten
 *   - cascade_plan                → triggered_by → satellite cascade edges (+ depth)
 *   - node_probabilities          → per-node manoeuvre probability (tooltip)
 * Node colour = worst backend severity of the alerts touching it.
 */
import { severityLabel } from './severity';

// Keyed by String(id) so numeric and string forms of the same NORAD id share
// one entry across alert refreshes.
const _nodePositionCache = new Map();

/** Canonical cache/dedup key for a node id (NORAD id may arrive as number or string). */
export const nodeKey = (id) => String(id);

const SEV_RANK = { EVENT: 4, CRITICAL: 3, WARNING: 2, WATCH: 1 };

// FNV-1a string hash → 32-bit seed
function hashString(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

// mulberry32 PRNG (layout seeding only — never shown as data)
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

/** Deterministic initial layout position in [-0.5, 0.5) for a node id. */
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

const finite = (v) => {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

const tcaHoursOf = (alert) => (
  finite(alert.tca_hours) ?? (finite(alert.tca_minutes) != null ? finite(alert.tca_minutes) / 60 : null)
);

const pcOf = (alert) => finite(alert.probability_of_collision ?? alert.p_collision ?? alert.probability);

const isFragmentSide = (side, alert) => {
  if (!side) return false;
  if (String(side.object_type ?? '').toUpperCase() === 'DEB') return true;
  return alert.fragment_id != null && String(side.id) === String(alert.fragment_id);
};

/**
 * buildCascadeGraph(alerts, cascade)
 * `cascade` is the stored cascade block ({ cascade_plan, node_probabilities, ... }).
 * Returns { nodes, edges } for CascadeDiagram.
 */
export function buildCascadeGraph(alerts, cascade = null) {
  const nodeMap = new Map();
  const edgeMap = new Map();
  const nodeProb = cascade?.node_probabilities ?? {};

  const ensureNode = (id, fields = {}) => {
    const key = nodeKey(id);
    let node = nodeMap.get(key);
    if (!node) {
      const cached = getCachedPosition(id);
      const seed = seededUnit(id);
      node = {
        id,
        kind: 'sat',
        label: `#${id}`,
        agency: '',
        object_type: null,
        cpi: null,
        severity: null,
        probability: finite(nodeProb[key]),
        cascade_depth: null,
        x: cached?.x ?? seed.u * 200,
        y: cached?.y ?? seed.v * 150,
        vx: 0,
        vy: 0,
      };
      nodeMap.set(key, node);
    }
    if (fields.kind) node.kind = fields.kind;
    if (fields.label && (node.label === `#${id}` || fields.kind === 'event')) node.label = fields.label;
    if (fields.agency && !node.agency) node.agency = fields.agency;
    if (fields.object_type && !node.object_type) node.object_type = fields.object_type;
    if (fields.event && (!node.event || node.event.fragment_count == null)) node.event = fields.event;
    const cpi = finite(fields.cpi);
    if (cpi != null && (node.cpi == null || cpi > node.cpi)) node.cpi = cpi;
    if (fields.severity && (SEV_RANK[fields.severity] ?? 0) > (SEV_RANK[node.severity] ?? 0)) {
      node.severity = fields.severity;
    }
    const depth = finite(fields.cascade_depth);
    if (depth != null && (node.cascade_depth == null || depth > node.cascade_depth)) node.cascade_depth = depth;
    return node.id;
  };

  const satFields = (side, alert, severity) => ({
    label: side?.name,
    agency: side?.agency,
    object_type: side?.object_type,
    cpi: alert.cpi_score,
    severity,
    cascade_depth: alert.cascade_depth,
  });

  const addEdge = (id, edge) => {
    const prev = edgeMap.get(id);
    if (!prev) {
      edgeMap.set(id, edge);
      return;
    }
    // Aggregate repeated (event → satellite) links from many fragments.
    prev.count = (prev.count ?? 1) + 1;
    if (edge.p_collision != null && (prev.p_collision == null || edge.p_collision > prev.p_collision)) {
      prev.p_collision = edge.p_collision;
    }
    if (edge.miss_distance != null && (prev.miss_distance == null || edge.miss_distance < prev.miss_distance)) {
      prev.miss_distance = edge.miss_distance;
      prev.tca_hours = edge.tca_hours;
      prev.tca_utc = edge.tca_utc;
      prev.alert_id = edge.alert_id;
    }
    if ((SEV_RANK[edge.severity] ?? 0) > (SEV_RANK[prev.severity] ?? 0)) prev.severity = edge.severity;
  };

  for (const alert of alerts || []) {
    const s1 = alert.sat1 ?? (alert.sat_a_norad_id != null ? { id: alert.sat_a_norad_id, name: alert.sat_a_name } : null);
    const s2 = alert.sat2 ?? (alert.sat_b_norad_id != null ? { id: alert.sat_b_norad_id, name: alert.sat_b_name } : null);
    if (s1?.id == null || s2?.id == null) continue;
    const severity = severityLabel(alert);
    const measure = {
      p_collision: pcOf(alert),
      miss_distance: finite(alert.miss_distance_km),
      tca_hours: tcaHoursOf(alert),
      tca_utc: alert.tca_utc ?? null,
      severity,
      alert_id: alert.id ?? `${s1.id}-${s2.id}`,
    };

    const event = alert.source === 'debris' && alert.parent_event && alert.parent_event.event_id != null
      ? alert.parent_event
      : null;

    if (event) {
      // collision event → parents, event → fragment-threatened satellite
      const eventId = `event:${event.event_id}`;
      ensureNode(eventId, {
        kind: 'event',
        label: `EVT ${String(event.event_id).slice(0, 8)}`,
        severity: 'EVENT',
        event,
      });
      for (const pid of Array.isArray(event.parent_ids) ? event.parent_ids : []) {
        const parentSide = [s1, s2].find((s) => String(s.id) === String(pid));
        ensureNode(pid, parentSide ? satFields(parentSide, alert, null) : {});
        addEdge(`breakup:${event.event_id}:${pid}`, {
          id: `breakup:${event.event_id}:${pid}`,
          source: eventId,
          target: nodeMap.get(nodeKey(pid)).id,
          kind: 'breakup',
          status: 'breakup',
          severity: 'EVENT',
          p_collision: null,
          miss_distance: null,
          tca_hours: null,
          tca_utc: event.collision_utc ?? null,
          alert_id: null,
        });
      }
      const frag1 = isFragmentSide(s1, alert);
      const frag2 = isFragmentSide(s2, alert);
      const threatened = frag1 && !frag2 ? [s2] : frag2 && !frag1 ? [s1] : [s1, s2];
      for (const side of threatened) {
        const tid = ensureNode(side.id, satFields(side, alert, severity));
        addEdge(`debris:${event.event_id}:${nodeKey(side.id)}`, {
          id: `debris:${event.event_id}:${nodeKey(side.id)}`,
          source: eventId,
          target: tid,
          kind: 'debris',
          status: 'debris',
          count: 1,
          ...measure,
        });
      }
      continue;
    }

    const aNode = ensureNode(s1.id, satFields(s1, alert, severity));
    const bNode = ensureNode(s2.id, satFields(s2, alert, severity));
    addEdge(measure.alert_id, {
      id: measure.alert_id,
      source: aNode,
      target: bNode,
      kind: alert.source === 'debris' ? 'debris' : 'conjunction',
      status: alert.source === 'debris' ? 'debris' : 'active',
      ...measure,
    });
  }

  // Cascade plan: each manoeuvre was triggered by an upstream satellite.
  for (const step of Array.isArray(cascade?.cascade_plan) ? cascade.cascade_plan : []) {
    const satId = step?.satellite_id ?? step?.sat_id;
    if (satId == null) continue;
    ensureNode(satId, {
      label: step.satellite_name,
      agency: step.agency,
      cascade_depth: step.cascade_depth,
    });
    const trig = step.triggered_by;
    if (trig == null || nodeKey(trig) === nodeKey(satId)) continue;
    ensureNode(trig, {});
    const id = `cascade:${nodeKey(trig)}:${nodeKey(satId)}`;
    if (edgeMap.has(id)) continue;
    addEdge(id, {
      id,
      source: nodeMap.get(nodeKey(trig)).id,
      target: nodeMap.get(nodeKey(satId)).id,
      kind: 'cascade',
      status: 'cascade',
      severity: null,
      p_collision: finite(step.threat?.p_collision),
      miss_distance: finite(step.threat?.miss_distance_km),
      tca_hours: finite(step.threat?.tca_minutes) != null ? finite(step.threat.tca_minutes) / 60 : null,
      tca_utc: null,
      cascade_depth: finite(step.cascade_depth),
      alert_id: null,
    });
  }

  for (const node of nodeMap.values()) {
    if (node.severity == null) node.severity = node.kind === 'event' ? 'EVENT' : 'NOMINAL';
  }

  return { nodes: Array.from(nodeMap.values()), edges: Array.from(edgeMap.values()) };
}

/**
 * markEdgeResolved(graph, alertId)
 * Sets edge.status = 'resolved' for the given alert ID.
 */
export function markEdgeResolved(graph, alertId) {
  return {
    ...graph,
    edges: graph.edges.map((e) =>
      (e.id === alertId || e.alert_id === alertId ? { ...e, status: 'resolved' } : e)
    ),
  };
}
