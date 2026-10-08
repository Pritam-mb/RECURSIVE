import { useState, useEffect } from "react"

const API = ""

export default function SimulationTimeControl() {
  const [offset, setOffset] = useState(0)
  const [simTime, setSimTime] = useState(null)
  const [applying, setApplying] = useState(false)
  const [nAlerts, setNAlerts] = useState(0)

  useEffect(() => {
    fetch(`${API}/api/simulation/time`)
      .then(r => r.json())
      .then(d => {
        setOffset(d.offset_hours || 0)
        setSimTime(d.simulation_time)
      })
      .catch(() => {})
  }, [])

  const applyTime = async (hours) => {
    setApplying(true)
    try {
      const res = await fetch(`${API}/api/simulation/time`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ offset_hours: hours })
      })
      const data = await res.json()
      setSimTime(data.simulation_time)
      setNAlerts(data.alerts_found || 0)
      setOffset(hours)
    } catch (e) {
      console.error("Time control error:", e)
    } finally {
      setApplying(false)
    }
  }

  const formattedTime = simTime
    ? new Date(simTime).toUTCString().slice(0, 25)
    : "Loading..."

  return (
    <div className="panel sim-time-panel">
      <div className="panel-header">
        Simulation Time Control
        {offset > 0 && (
          <span className="sim-badge">SIM +{offset.toFixed(1)}h</span>
        )}
      </div>

      <div className="sim-time-display">
        <span className="sim-time-label">Current Time</span>
        <span className="sim-time-value">{formattedTime} UTC</span>
        {nAlerts > 0 && (
          <span className="sim-alerts">
            {nAlerts} alerts at this time
          </span>
        )}
      </div>

      <div className="control-row">
        <label>
          Time Offset: <strong>+{offset.toFixed(1)} hours</strong>
        </label>
        <input
          type="range"
          min="0" max="24" step="0.5"
          value={offset}
          onChange={e => setOffset(parseFloat(e.target.value))}
          onMouseUp={e => applyTime(parseFloat(e.target.value))}
          onTouchEnd={e => applyTime(parseFloat(e.target.value))}
        />
      </div>

      <div className="preset-buttons">
        {[0, 2, 4, 6, 12, 24].map(h => (
          <button
            key={h}
            className={`preset-btn ${offset === h ? "active" : ""}`}
            onClick={() => applyTime(h)}
            disabled={applying}
          >
            {h === 0 ? "NOW" : `+${h}h`}
          </button>
        ))}
      </div>

      {applying && (
        <div className="applying-msg">
          Propagating all satellites to T+{offset.toFixed(1)}h...
        </div>
      )}
    </div>
  )
}
