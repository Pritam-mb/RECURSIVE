import { useMemo, useState } from 'react';
import useStore from '../store/useStore';
import CxGraphView from './Cascade/CxGraphView';
import CascadeExplorer from './Cascade/CascadeExplorer';
import CascadeLegend from './Cascade/CascadeLegend';
import { focusSubset, fromStoreGraph, fmtDist } from './Cascade/cxGraph';
import '../styles/analysis.css';
import '../styles/cascade.css';

const EMPTY_GRAPH = { nodes: [], edges: [] };

export default function CascadeDiagram({ graph = EMPTY_GRAPH }) {
  const hotspots = useStore((s) => s.hotspots);
  const setSelectedSatId = useStore((s) => s.setSelectedSatelliteId);
  const setSelectedAlertId = useStore((s) => s.setSelectedAlertId);
  const selectedSatId = useStore((s) => s.selectedSatelliteId);
  const [showAll, setShowAll] = useState(false);
  const [explorer, setExplorer] = useState(null); // null | { selection }

  const model = useMemo(() => fromStoreGraph(graph, hotspots), [graph, hotspots]);
  const shown = useMemo(() => focusSubset(model, showAll), [model, showAll]);
  const hasGraph = model.nodes.length > 0;

  const onSelect = (sel) => {
    if (sel.type === 'node') {
      if (sel.node?.kind === 'event') setExplorer({ selection: { type: 'node', id: sel.id } });
      else setSelectedSatId(sel.node?.rawId ?? sel.id);
    } else if (sel.type === 'edge') {
      setSelectedAlertId(sel.edge?.alert_id ?? sel.id);
    } else if (sel.type === 'hotspot') {
      setExplorer({ selection: sel });
    }
  };

  const selection = selectedSatId != null ? { type: 'node', id: String(selectedSatId) } : null;

  return (
    <section className="ui-panel an-panel an-cascade cx-panel">
      <header className="ui-panel-header">
        <span className="ui-label an-title">Cascade Risk Graph</span>
        <div className="an-header-meta">
          {hasGraph && (
            <span className="an-count" title="Shown nodes / edges (hidden nodes are in components with no CRITICAL/WARNING alert or collision)">
              {shown.nodes.length}N / {shown.edges.length}E{shown.hidden ? ` · ${shown.hidden} hidden` : ''}
            </span>
          )}
          {hasGraph && (
            <button
              type="button"
              className={`cx-chip${showAll ? ' is-on' : ''}`}
              onClick={() => setShowAll((v) => !v)}
              title="Show every component, including WATCH-only pairs"
            >
              {showAll ? 'All' : 'Focus'}
            </button>
          )}
          <button
            type="button"
            className="ui-btn cx-expand"
            onClick={() => setExplorer({ selection: null })}
            title="Open the Cascade & Hotspot Explorer"
          >
            Expand
          </button>
        </div>
      </header>

      <div className="an-graph-body cx-panel-body">
        {hasGraph && shown.nodes.length > 0 && (
          <CxGraphView
            model={shown}
            selection={selection}
            onSelect={onSelect}
            hotspotLabel={(h) => `HS${h.index + 1} · ${fmtDist(h.radius_km)}`}
          />
        )}
        {hasGraph && shown.nodes.length === 0 && (
          <div className="an-empty">
            <span className="ui-label">No critical or warning cascade</span>
            <span className="an-empty-sub">Only WATCH-level pairs — press All to show them</span>
          </div>
        )}
        {!hasGraph && (
          <div className="an-empty">
            <span className="ui-label">No active cascade scenarios</span>
            <span className="an-empty-sub">Load the cascade demo to see cascade resolution</span>
          </div>
        )}
        {hasGraph && <CascadeLegend compact />}
      </div>

      {explorer && (
        <CascadeExplorer initialSelection={explorer.selection} onClose={() => setExplorer(null)} />
      )}
    </section>
  );
}
