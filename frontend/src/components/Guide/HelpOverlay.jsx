import { useCallback, useEffect, useLayoutEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { HELP_REGIONS, THREAT_SECTION_BADGES } from './helpRegions';
import InfoTip from './InfoTip';
import '../../styles/guide.css';

const CARD_W = 360;
const GAP = 8;

function visibleRect(el) {
  if (!el) return null;
  const r = el.getBoundingClientRect();
  if (r.width < 4 || r.height < 4) return null;
  if (r.right <= 0 || r.bottom <= 0 || r.left >= window.innerWidth || r.top >= window.innerHeight) return null;
  const style = window.getComputedStyle(el);
  if (style.visibility === 'hidden' || style.display === 'none' || Number(style.opacity) === 0) return null;
  return { x: r.left, y: r.top, w: r.width, h: r.height };
}

function findRect(selectors) {
  for (const sel of selectors) {
    let el = null;
    try { el = document.querySelector(sel); } catch { el = null; }
    const r = visibleRect(el);
    if (r) return r;
  }
  return null;
}

function clampBox(left, top, w, vw, vh) {
  return {
    left: Math.max(GAP, Math.min(left, vw - w - GAP)),
    top: Math.max(GAP, Math.min(top, vh - 80)),
  };
}

function cardStyle(region, rect, anchorRect) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const w = Math.min(CARD_W, Math.max(220, rect.w - 2 * 12));
  const oy = region.offsetY ?? 36;
  switch (region.place) {
    case 'below-left': {
      const p = clampBox(rect.x, rect.y + rect.h + GAP, CARD_W, vw, vh);
      return { ...p, width: CARD_W };
    }
    case 'below-right': {
      const p = clampBox(rect.x + rect.w - CARD_W, rect.y + rect.h + GAP, CARD_W, vw, vh);
      return { ...p, width: CARD_W };
    }
    case 'beside-right': {
      const a = anchorRect || rect;
      const p = clampBox(a.x + a.w + 12, a.y + 12, 420, vw, vh);
      return { ...p, width: 420 };
    }
    case 'inside-bl':
      return { left: rect.x + 12, bottom: Math.max(GAP, vh - (rect.y + rect.h) + 40), width: w };
    case 'inside-tr':
      return { left: rect.x + rect.w - w - 12, top: rect.y + oy, width: w };
    case 'inside-wide': {
      const ww = Math.min(1180, rect.w - 280);
      return { left: rect.x + rect.w - ww - 12, top: rect.y + 6, width: ww, maxHeight: rect.h - 12 };
    }
    case 'inside-tl':
    default:
      return { left: rect.x + 12, top: rect.y + oy, width: w, maxHeight: rect.h - oy - 8 };
  }
}

function measure() {
  const out = [];
  let n = 0;
  const numberOf = {};
  for (const region of HELP_REGIONS) {
    const rect = findRect(region.selectors);
    if (!rect) continue;
    n += 1;
    numberOf[region.key] = n;
    const anchorRect = region.anchor ? findRect([region.anchor]) : null;
    out.push({ region, rect, n, anchorRect });
  }
  // Section badges inside an expanded threat card (only those visible in the list).
  const sections = [];
  const row = document.querySelector('.tq-row.is-expanded');
  const list = document.querySelector('.tq-list');
  const listRect = list ? list.getBoundingClientRect() : null;
  if (row && listRect && numberOf['threat-card']) {
    const labels = Array.from(row.querySelectorAll('.tq-section-label'));
    for (const badge of THREAT_SECTION_BADGES) {
      const lab = labels.find((l) => l.textContent.trim().toLowerCase() === badge.label.toLowerCase());
      const box = lab?.parentElement?.getBoundingClientRect();
      if (!box || box.top < listRect.top || box.top > listRect.bottom - 12) continue;
      sections.push({
        key: badge.label,
        n: `${numberOf['threat-card']}${String.fromCharCode(97 + badge.bullet)}`,
        rect: { x: box.left, y: box.top, w: box.width, h: Math.min(box.height, listRect.bottom - box.top) },
      });
    }
  }
  return { items: out, numberOf, sections };
}

function RegionCard({ item, numberOf, children, style }) {
  const { region, n } = item;
  return (
    <div className="gd-callout" style={style} role="note" aria-label={`${n}. ${region.title}`}>
      <div className="gd-callout-head">
        <span className="gd-num">{n}</span>
        <span className="gd-callout-title">{region.title}</span>
        {region.how && (
          <InfoTip title="How it is computed" side="bottom">{region.how}</InfoTip>
        )}
      </div>
      {region.what && <p className="gd-callout-text">{region.what}</p>}
      {region.bullets && (
        <ul className="gd-callout-list">
          {region.bullets.map(([k, v], i) => (
            <li key={k}>
              <span className="gd-num gd-num--sm">{`${n}${String.fromCharCode(97 + i)}`}</span>
              <span><b>{k}.</b> {v}</span>
            </li>
          ))}
        </ul>
      )}
      {region.how && <p className="gd-callout-how"><span className="gd-k">How</span>{region.how}</p>}
      {region.colours && <p className="gd-callout-how"><span className="gd-k">Colours</span>{region.colours}</p>}
      {children}
      {!children && numberOf && null}
    </div>
  );
}

export default function HelpOverlay({ open, onClose, onStartTour }) {
  const [layout, setLayout] = useState(null);

  const remeasure = useCallback(() => setLayout(measure()), []);

  useLayoutEffect(() => {
    if (!open) return undefined;
    remeasure();
    // Panels may still be settling (fonts, Cesium canvas): re-measure shortly after.
    const t = setTimeout(remeasure, 250);
    window.addEventListener('resize', remeasure);
    return () => {
      clearTimeout(t);
      window.removeEventListener('resize', remeasure);
    };
  }, [open, remeasure]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') { e.stopPropagation(); onClose(); }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [open, onClose]);

  if (!open || !layout) return null;

  const { items, numberOf, sections } = layout;
  const children = {};
  for (const it of items) {
    if (it.region.badgeOnly && it.region.parent) (children[it.region.parent] ||= []).push(it);
  }

  return createPortal(
    <div className="gd-help" role="dialog" aria-modal="true" aria-label="Screen guide">
      <div className="gd-help-dim" onClick={onClose} />

      {items.map((it) => (
        <div
          key={`o-${it.region.key}`}
          className={`gd-region${it.region.badgeOnly ? ' is-small' : ''}`}
          style={{ left: it.rect.x + 2, top: it.rect.y + 2, width: it.rect.w - 4, height: it.rect.h - 4 }}
        >
          <span className="gd-num gd-region-num">{it.n}</span>
        </div>
      ))}

      {sections.map((s) => (
        <div
          key={`s-${s.key}`}
          className="gd-region is-small is-section"
          style={{ left: s.rect.x, top: s.rect.y, width: s.rect.w, height: s.rect.h }}
        >
          <span className="gd-num gd-num--sm gd-region-num">{s.n}</span>
        </div>
      ))}

      {items.filter((it) => !it.region.badgeOnly).map((it) => (
        <RegionCard key={`c-${it.region.key}`} item={it} style={cardStyle(it.region, it.rect, it.anchorRect)}>
          {(children[it.region.key] || []).length > 0 && (
            <ul className="gd-callout-list">
              {children[it.region.key].map((c) => (
                <li key={c.region.key}>
                  <span className="gd-num gd-num--sm">{c.n}</span>
                  <span>{c.region.what}</span>
                </li>
              ))}
            </ul>
          )}
        </RegionCard>
      ))}

      <div className="gd-help-bar">
        <span className="gd-help-bar-title">Screen guide</span>
        <span className="gd-help-bar-meta">{items.length} regions · every value is computed live · Esc closes</span>
        {onStartTour && (
          <button type="button" className="ui-btn ui-btn--primary gd-help-bar-btn" onClick={onStartTour}>
            Start demo tour
          </button>
        )}
        <button type="button" className="ui-btn gd-help-bar-btn" onClick={onClose} autoFocus>
          Close
        </button>
      </div>
      {numberOf.queue && !numberOf['threat-card'] && null}
    </div>,
    document.body,
  );
}
