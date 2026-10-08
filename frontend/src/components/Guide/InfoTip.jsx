import { useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import '../../styles/guide.css';

const POP_W = 260;

/**
 * Small "i" badge with a hover / focus popover.
 *   <InfoTip title="Pc">Foster 2-D probability from TLE-age covariance.</InfoTip>
 * `side` = 'top' | 'bottom' | 'left' | 'right' (default 'top'). The popover is
 * portalled to <body> with fixed positioning, so scrolling / clipped parents
 * never cut it off.
 */
export default function InfoTip({ title, children, side = 'top', className = '' }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState(null);
  const btnRef = useRef(null);
  const popRef = useRef(null);
  const id = useId();

  useLayoutEffect(() => {
    if (!open || !btnRef.current) return;
    const b = btnRef.current.getBoundingClientRect();
    const h = popRef.current?.offsetHeight || 60;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let left;
    let top;
    if (side === 'left') { left = b.left - POP_W - 6; top = b.top + b.height / 2 - h / 2; }
    else if (side === 'right') { left = b.right + 6; top = b.top + b.height / 2 - h / 2; }
    else {
      left = b.left + b.width / 2 - POP_W / 2;
      top = side === 'bottom' ? b.bottom + 6 : b.top - h - 6;
      if (top < 4) top = b.bottom + 6;
      if (top + h > vh - 4) top = b.top - h - 6;
    }
    left = Math.max(4, Math.min(left, vw - POP_W - 4));
    top = Math.max(4, Math.min(top, vh - h - 4));
    setPos({ left, top });
  }, [open, side]);

  return (
    <span
      className={`gd-tip ${className}`}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => { setOpen(false); setPos(null); }}
    >
      <button
        ref={btnRef}
        type="button"
        className="gd-tip-btn"
        aria-label={title ? `About ${title}` : 'More information'}
        aria-describedby={open ? id : undefined}
        onFocus={() => setOpen(true)}
        onBlur={() => { setOpen(false); setPos(null); }}
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
      >
        i
      </button>
      {open && createPortal(
        <span
          ref={popRef}
          role="tooltip"
          id={id}
          className="gd-tip-pop"
          style={{ width: POP_W, left: pos?.left ?? -9999, top: pos?.top ?? -9999 }}
        >
          {title && <span className="gd-tip-title">{title}</span>}
          <span className="gd-tip-body">{children}</span>
        </span>,
        document.body,
      )}
    </span>
  );
}
