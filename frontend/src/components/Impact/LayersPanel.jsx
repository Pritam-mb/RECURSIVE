import { useEffect, useRef, useState } from 'react';
import useStore from '../../store/useStore';
import {
  DV_EDGES_MS, FRAG_PARENT_COLORS, FRAG_RAMP, SIZE_EDGES_M,
} from './replayData';
import '../../styles/impact.css';

const GROUPS = [
  {
    title: 'Catalog',
    items: [
      ['satellites', 'Satellites'],
      ['selectedOrbit', 'Selected orbit'],
      ['conjunctionLines', 'Conjunction lines'],
      ['hotspots', 'Hotspots'],
      ['labels', 'Labels'],
    ],
  },
  {
    title: 'Debris impact',
    items: [
      ['parentTracks', 'Parent tracks'],
      ['fragments', 'Fragments'],
      ['fragmentTrails', 'Fragment trails'],
      ['debrisEnvelope', 'Debris envelope'],
      ['threatened', 'Threatened satellites'],
    ],
  },
];

const MODES = [
  ['parent', 'Parent'],
  ['size', 'Size'],
  ['dv', 'Δv'],
];

function rampLabels(edges, unit) {
  return [`< ${edges[0]}`, ...edges.slice(0, -1).map((e, i) => `${e}–${edges[i + 1]}`), `≥ ${edges[edges.length - 1]}`]
    .map((s) => `${s} ${unit}`);
}

function LayersIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
      <path d="M7 1.5l5.5 3L7 7.5 1.5 4.5z" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
      <path d="M1.5 7l5.5 3 5.5-3" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
      <path d="M1.5 9.5l5.5 3 5.5-3" fill="none" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
    </svg>
  );
}

export default function LayersPanel() {
  const [open, setOpen] = useState(false);
  const layers = useStore((s) => s.layers);
  const setLayer = useStore((s) => s.setLayer);
  const mode = useStore((s) => s.fragmentColorMode);
  const setMode = useStore((s) => s.setFragmentColorMode);
  const rootRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    const onDown = (e) => { if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false); };
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onDown);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onDown);
    };
  }, [open]);

  const offCount = Object.entries(layers || {}).filter(([k, v]) => (k === 'fragmentTrails' ? false : !v)).length;

  let legend;
  if (mode === 'size') legend = rampLabels(SIZE_EDGES_M, 'm').map((l, i) => [FRAG_RAMP[i], l]);
  else if (mode === 'dv') legend = rampLabels(DV_EDGES_MS, 'm/s').map((l, i) => [FRAG_RAMP[i], l]);
  else legend = [[FRAG_PARENT_COLORS[0], 'from parent A'], [FRAG_PARENT_COLORS[1], 'from parent B']];

  return (
    <div className="im-layers" ref={rootRef}>
      <button
        type="button"
        className={`im-layers-btn${open ? ' is-open' : ''}`}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label="Globe layers"
        title="Globe layers"
      >
        <LayersIcon />
        {offCount > 0 && <span className="im-layers-badge">{offCount}</span>}
      </button>
      {open && (
        <div className="im-layers-panel" role="dialog" aria-label="Globe layers">
          {GROUPS.map((g) => (
            <fieldset key={g.title} className="im-layers-group">
              <legend className="ui-label">{g.title}</legend>
              {g.items.map(([key, label]) => (
                <label key={key} className="im-check">
                  <input
                    type="checkbox"
                    checked={!!layers?.[key]}
                    onChange={(e) => setLayer(key, e.target.checked)}
                  />
                  <span className="im-check-box" aria-hidden="true" />
                  <span>{label}</span>
                </label>
              ))}
            </fieldset>
          ))}
          <div className="im-layers-group">
            <span className="ui-label">Fragment colour</span>
            <div className="im-seg" role="radiogroup" aria-label="Fragment colour mode">
              {MODES.map(([m, label]) => (
                <button
                  key={m}
                  type="button"
                  role="radio"
                  aria-checked={mode === m}
                  className={`im-seg-btn${mode === m ? ' is-on' : ''}`}
                  onClick={() => setMode(m)}
                >
                  {label}
                </button>
              ))}
            </div>
            <ul className="im-legend">
              {legend.map(([c, l]) => (
                <li key={l}><span className="im-sw im-sw--dot" style={{ background: c }} />{l}</li>
              ))}
              <li><span className="im-sw im-sw--line" style={{ background: 'var(--c-accent)' }} />Parent A track</li>
              <li><span className="im-sw im-sw--line" style={{ background: 'var(--c-info)' }} />Parent B track <em>(dashed = future)</em></li>
              <li><span className="im-sw im-sw--band" />Debris stream (envelope)</li>
              <li><span className="im-sw im-sw--ring" />Threatened satellite</li>
              <li><span className="im-sw im-sw--flash" />Impact flash (T0)</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
