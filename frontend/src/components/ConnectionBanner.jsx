import useStore from '../store/useStore';

export default function ConnectionBanner() {
  const wsConnected = useStore((s) => s.wsConnected);

  if (wsConnected) return null;

  return (
    <div className="sh-banner" role="alert">
      <span className="ui-status is-warning">Backend offline</span>
      <span className="sh-banner-text">Attempting to reconnect. Data may be stale.</span>
    </div>
  );
}
