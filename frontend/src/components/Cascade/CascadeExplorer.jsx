import { useCallback, useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import CxGraphView from './CxGraphView';
import HotspotDetail from './HotspotDetail';
import CascadeLegend from './CascadeLegend';
import { focusSubset, fromExplorer, fmtDist, fmtPc, SEV_CLASS } from './cxGraph';
import '../../styles/cascade.css';

const POLL_MS = 30000;
const utc = (iso) => (iso ? `${String(iso).slice(0, 19).replace('T', ' ')}Z` : '—');

function NodeSummary({ node, edges, nodesById, onSelect }) {
  const inc = edges
    .filter((e) => e.source === node.id || e.target === node.id)
    .sort((a, b) => (a.miss_distance_km ?? Infinity) - (b.miss_distance_km ?? Infinity));
  return (
    <div className="cx-summary">
      <div className="cx-detail-title">
        <span className={`cx-badge ${SEV_CLASS[node.severity] ?? ''}`}>{node.severity ?? 'NOMINAL'}</span>
        <span className="cx-detail-name">{node.name}</span>
        <span className="cx-dim">{node.object_type ?? ''} {node.agency ? `· ${node.agency}` : ''} · P(hit) {fmtPc(node.probability)}</span>
      </div>
      {inc.length > 0 && (
        <table className="cx-table">
          <thead><tr><th>Connected to</th><th>Link</th><th className="r">Distance</th><th className="r">Pc</th><th>TCA</th></tr></thead>
          <tbody>
            {inc.slice(0, 12).map((e) => {
              const other = nodesById.get(e.source === node.id ? e.target : e.source);
              return (
                <tr key={e.id} className="cx-click" onClick={() => onSelect({ type: 'edge', id: e.id })}>
                  <td>{other?.name ?? '—'}</td>
                  <td>{e.kind}{e.fragments > 1 ? ` (${e.fragments} frag)` : ''}</td>
                  <td className="r">{fmtDist(e.miss_distance_km)}</td>
                  <td className="r">{fmtPc(e.pc)}</td>
                  <td>{e.tca_utc ? `${String(e.tca_utc).slice(11, 19)}Z` : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

function EventSummary({ ev }) {
  if (!ev) return null;
  return (
    <div className="cx-summary">
      <div className="cx-detail-title">
        <span className="cx-badge cx-sev-event">COLLISION</span>
        <span className="cx-detail-name">{(ev.parent_names || []).join(' × ')}</span>
        <span className="cx-dim">{ev.status === 'pending_impact' ? 'predicted' : 'occurred'} {utc(ev.collision_utc)}</span>
      </div>
      <dl className="cx-kv">
        <div className="cx-kv-row"><dt>Fragments (SBM ≥10 cm)</dt><dd>{ev.fragment_count_total} · simulated {ev.fragments_simulated}</dd></div>
        <div className="cx-kv-row"><dt>Catastrophic</dt><dd>{ev.catastrophic ? 'yes' : 'no'}</dd></div>
        <div className="cx-kv-row"><dt>Impact speed</dt><dd>{ev.relative_velocity_kms != null ? `${Number(ev.relative_velocity_kms).toFixed(2)} km/s` : '—'}</dd></div>
        <div className="cx-kv-row"><dt>Breakup miss</dt><dd>{fmtDist(ev.miss_distance_km)}</dd></div>
      </dl>
      {ev.threatened?.length > 0 && (
        <table className="cx-table">
          <thead><tr><th>Threatened by fragments</th><th className="r">Frag</th><th className="r">Closest</th><th className="r">Max Pc</th><th>TCA</th></tr></thead>
          <tbody>
            {ev.threatened.map((t) => (
              <tr key={t.id}><td>{t.name}</td><td className="r">{t.fragments}</td><td className="r">{fmtDist(t.min_miss_km)}</td><td className="r">{fmtPc(t.max_pc)}</td><td>{t.tca_utc ? `${String(t.tca_utc).slice(11, 19)}Z` : '—'}</td></tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default function CascadeExplorer({ initialSelection = null, onClose }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [fetchedAt, setFetchedAt] = useState(0);
  const [showAll, setShowAll] = useState(false);
  const [selection, setSelection] = useState(initialSelection);
  const [hoverOption, setHoverOption] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch('/api/cascade/explorer');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = await res.json();
        if (cancelled) return;
        setData(body);
        setFetchedAt(Date.now());
        setError(null);
      } catch (e) {
        if (!cancelled) setError(String(e.message || e));
      }
    };
    load();
    const t = setInterval(load, POLL_MS);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const model = useMemo(() => fromExplorer(data), [data]);
  const shown = useMemo(() => focusSubset(model, showAll), [model, showAll]);
  const nodesById = useMemo(() => new Map(model.nodes.map((n) => [n.id, n])), [model]);
  const hotspotsById = useMemo(() => new Map((data?.hotspots ?? []).map((h) => [h.id, h])), [data]);

  // Default selection: the highest-score hotspot (the backend sorts by score).
  const effectiveSel = selection ?? (data?.hotspots?.length ? { type: 'hotspot', id: data.hotspots[0].id } : null);

  // Which hotspot's detail is shown for the current selection.
  const context = useMemo(() => {
    if (!data || !effectiveSel) return {};
    const hs = data.hotspots ?? [];
    const best = (pred) => hs.find(pred) ?? null;   // list is score-sorted
    if (effectiveSel.type === 'hotspot') return { hotspot: hotspotsById.get(effectiveSel.id) };
    if (effectiveSel.type === 'edge') {
      const e = model.edges.find((x) => x.id === effectiveSel.id);
      if (!e) return {};
      const memb = (h) => h.area.members.some((m) => String(m.id) === e.source) && h.area.members.some((m) => String(m.id) === e.target);
      return { edge: e, hotspot: best(memb) ?? best((h) => h.area.members.some((m) => [e.source, e.target].includes(String(m.id)))) };
    }
    const n = nodesById.get(String(effectiveSel.id));
    if (!n) return {};
    if (n.kind === 'event') {
      return { event: (data.events ?? []).find((ev) => ev.event_id === n.event_id), hotspot: best((h) => h.debris.own_event_id === n.event_id) };
    }
    return { node: n, hotspot: best((h) => h.area.members.some((m) => String(m.id) === n.id)) };
  }, [data, effectiveSel, hotspotsById, model.edges, nodesById]);

  const highlight = useMemo(() => {
    if (!hoverOption) return null;
    return (hoverOption.option.secondary_conjunctions ?? []).map((s) => ({
      source: hoverOption.mover, target: s.id, name: s.name, miss_km: s.miss_km, pc: s.pc, unsafe: s.blocks_cascade_safe,
    }));
  }, [hoverOption]);

  const onSelectNode = useCallback((id) => setSelection({ type: 'node', id: String(id) }), []);
  const edge = context.edge;

  return createPortal(
    <div className="cx-overlay" role="dialog" aria-modal="true" aria-label="Cascade and hotspot explorer">
      <div className="cx-modal">
        <header className="cx-modal-head">
          <span className="ui-label cx-modal-title">Cascade &amp; Hotspot Explorer</span>
          {data && (
            <span className="cx-modal-meta">
              {data.counts.hotspots} hotspots · {data.counts.events} collision events · {data.counts.objects} objects · {data.counts.edges} links · sim {utc(data.sim_now_utc)}
            </span>
          )}
          <span className="cx-grow" />
          <button type="button" className={`cx-chip${showAll ? ' is-on' : ''}`} onClick={() => setShowAll((v) => !v)}>
            {showAll ? 'All components' : `Focus${shown.hidden ? ` (${shown.hidden} hidden)` : ''}`}
          </button>
          <button type="button" className="ui-btn" onClick={onClose} aria-label="Close explorer">Close</button>
        </header>

        <div className="cx-modal-body">
          <div className="cx-left">
            {data && (
              <CxGraphView
                model={shown}
                interactive
                selection={effectiveSel}
                onSelect={(s) => setSelection({ type: s.type, id: s.id })}
                highlight={highlight}
                hotspotLabel={(h) => `HS${h.index + 1} · r ${fmtDist(h.radius_km)}`}
                ariaLabel="Interactive cascade graph"
              />
            )}
            {!data && <div className="cx-loading">{error ? `Explorer data unavailable (${error})` : 'Loading cascade data…'}</div>}
            <div className="cx-left-foot">
              <CascadeLegend />
              <span className="cx-hint">Drag to pan · wheel to zoom · click a node, link or zone</span>
            </div>
          </div>

          <aside className="cx-right">
            {edge && (
              <div className="cx-summary">
                <div className="cx-detail-title">
                  <span className={`cx-badge ${SEV_CLASS[edge.severity] ?? ''}`}>{edge.severity ?? edge.kind}</span>
                  <span className="cx-detail-name">{nodesById.get(edge.source)?.name} ↔ {nodesById.get(edge.target)?.name}</span>
                </div>
                <dl className="cx-kv">
                  <div className="cx-kv-row"><dt>{edge.kind === 'event-parent' ? 'Breakup miss' : 'Miss distance'}</dt><dd>{fmtDist(edge.miss_distance_km)}</dd></div>
                  <div className="cx-kv-row"><dt>Pc</dt><dd>{fmtPc(edge.pc)}</dd></div>
                  <div className="cx-kv-row"><dt>TCA</dt><dd>{utc(edge.tca_utc)}</dd></div>
                  {edge.fragments > 1 && <div className="cx-kv-row"><dt>Fragments</dt><dd>{edge.fragments}</dd></div>}
                </dl>
              </div>
            )}
            {context.node && <NodeSummary node={context.node} edges={model.edges} nodesById={nodesById} onSelect={setSelection} />}
            {context.event && <EventSummary ev={context.event} />}
            {context.hotspot ? (
              <HotspotDetail
                key={context.hotspot.id}
                hs={context.hotspot}
                fetchedAt={fetchedAt}
                onHoverOption={setHoverOption}
                onSelectNode={onSelectNode}
              />
            ) : data && (
              <p className="cx-none cx-pad">
                {effectiveSel ? 'This selection is not inside any hotspot zone.' : 'No hotspots in the current alert set.'}
              </p>
            )}
          </aside>
        </div>
      </div>
    </div>,
    document.body,
  );
}
