import InfoTip from './InfoTip';

/** Label / value rows used by the demo tour; optional third item = InfoTip text. */
export default function Kv({ rows }) {
  return (
    <dl className="gd-kv">
      {rows.filter(Boolean).map(([k, v, tip]) => (
        <div className="gd-kv-row" key={k}>
          <dt>{k}{tip && <InfoTip title={k} side="top">{tip}</InfoTip>}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}
