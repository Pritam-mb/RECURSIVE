import useStore from '../store/useStore';

export default function ConnectionBanner() {
  const wsConnected = useStore((s) => s.wsConnected);

  if (wsConnected) return null;

  return (
    <div className="connection-banner" role="alert">
      <div className="connection-banner-dot" />
      <span>BACKEND OFFLINE — Attempting to reconnect. Data may be stale.</span>
    </div>
  );
}
