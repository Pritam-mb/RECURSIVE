import { useState, useEffect } from "react"
import "../../styles/drawers.css"

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
    <div className="ui-panel dr-card">
      <div className="ui-panel-header">
        <span className="ui-label">Simulation Time</span>
        {offset > 0 ? (
          <span className="ui-status is-caution">SIM +{offset.toFixed(1)}h</span>
        ) : (
          <span className="ui-status is-nominal">Realtime</span>
        )}
      </div>

      <div className="dr-card-body">
        <div className="dr-kv-grid">
          <div className="dr-kv">
            <span className="dr-kv-key">Current time</span>
            <span className="dr-kv-val">{formattedTime} UTC</span>
          </div>
          {nAlerts > 0 && (
            <div className="dr-kv">
              <span className="dr-kv-key">Alerts at time</span>
              <span className="dr-kv-val is-caution">{nAlerts}</span>
            </div>
          )}
        </div>

        <label className="dr-range">
          <span className="dr-range-head">
            <span className="ui-label">Time offset</span>
            <span className="dr-kv-val">+{offset.toFixed(1)} h</span>
          </span>
          <input
            type="range"
            className="dr-range-input"
            min="0" max="24" step="0.5"
            value={offset}
            onChange={e => setOffset(parseFloat(e.target.value))}
            onMouseUp={e => applyTime(parseFloat(e.target.value))}
            onTouchEnd={e => applyTime(parseFloat(e.target.value))}
          />
        </label>

        <div className="dr-segmented" role="group" aria-label="Time presets">
          {[0, 2, 4, 6, 12, 24].map(h => (
            <button
              key={h}
              type="button"
              className="dr-segment"
              aria-pressed={offset === h}
              onClick={() => applyTime(h)}
              disabled={applying}
            >
              {h === 0 ? "NOW" : `+${h}h`}
            </button>
          ))}
        </div>

        {applying && (
          <div className="dr-note" role="status">
            Propagating all satellites to T+{offset.toFixed(1)}h
          </div>
        )}
      </div>
    </div>
  )
}
