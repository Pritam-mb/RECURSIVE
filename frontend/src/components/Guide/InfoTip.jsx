import { useId, useState } from 'react';
import '../../styles/guide.css';

/**
 * Small "i" badge with a hover / focus popover.
 *   <InfoTip title="Pc">Foster 2-D probability from TLE-age covariance.</InfoTip>
 * `side` = 'top' | 'bottom' | 'left' | 'right' (default 'top').
 */
export default function InfoTip({ title, children, side = 'top', className = '' }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span
      className={`gd-tip ${className}`}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        className="gd-tip-btn"
        aria-label={title ? `About ${title}` : 'More information'}
        aria-describedby={open ? id : undefined}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
      >
        i
      </button>
      {open && (
        <span role="tooltip" id={id} className={`gd-tip-pop is-${side}`}>
          {title && <span className="gd-tip-title">{title}</span>}
          <span className="gd-tip-body">{children}</span>
        </span>
      )}
    </span>
  );
}
