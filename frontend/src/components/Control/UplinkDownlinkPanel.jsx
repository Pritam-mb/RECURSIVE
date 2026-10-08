import { useState, useEffect, useRef } from "react"
import useStore from "../../store/useStore"
import "../../styles/uplink.css"

const API = "" // relative URLs - works on any host

const COMMAND_TYPES = [
  { value: "MANEUVER", label: "Maneuver (delta-V)" },
  { value: "SET_ORBIT", label: "Change Orbit Altitude" },
  { value: "SET_VELOCITY", label: "Override Velocity" },
  { value: "SET_POSITION", label: "Override Position" },
  { value: "SET_TRAJECTORY", label: "Set Trajectory" }
]

const DIRECTIONS = [
  { value: "prograde", label: "Prograde (speed up)" },
  { value: "retrograde", label: "Retrograde (slow down)" },
  { value: "radial", label: "Radial (away from Earth)" },
  { value: "normal", label: "Normal (change plane)" }
]

export default function UplinkDownlinkPanel() {
  const selectedSatId = useStore((s) => s.selectedSatelliteId)

  const [telemetry, setTelemetry] = useState(null)
  const [loading, setLoading] = useState(false)
  const [cmdType, setCmdType] = useState("MANEUVER")
  const [deltaV, setDeltaV] = useState(0.1)
  const [direction, setDirection] = useState("prograde")
  const [altitudeKm, setAltitudeKm] = useState(550)
  const [sending, setSending] = useState(false)
  const [cmdLog, setCmdLog] = useState([])
  const [error, setError] = useState(null)
  const pollRef = useRef(null)

  // Poll telemetry every 2 seconds when a satellite is selected
  useEffect(() => {
    if (!selectedSatId) {
      setTelemetry(null)
      return
    }
    const poll = async () => {
      try {
        setLoading(true)
        const res = await fetch(
          `${API}/api/satellites/${selectedSatId}/telemetry`
        )
        if (res.ok) {
          const data = await res.json()
          setTelemetry(data)
          setError(null)
        }
      } catch (_e) {
        setError("Telemetry offline")
      } finally {
        setLoading(false)
      }
    }
    poll()
    pollRef.current = setInterval(poll, 2000)
    return () => clearInterval(pollRef.current)
  }, [selectedSatId])

  const sendUplink = async () => {
    if (!selectedSatId) return
    setSending(true)
    const command = { type: cmdType }
    if (cmdType === "MANEUVER") {
      command.delta_v_ms = deltaV
      command.direction = direction
    } else if (cmdType === "SET_ORBIT") {
      command.altitude_km = altitudeKm
    }
    try {
      const res = await fetch(
        `${API}/api/satellites/${selectedSatId}/uplink`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(command)
        }
      )
      const result = await res.json()
      const entry = {
        id: Date.now(),
        time: new Date().toLocaleTimeString(),
        type: cmdType,
        satellite: selectedSatId,
        success: result.result?.success ?? false,
        message: result.result?.message ?? "Sent"
      }
      setCmdLog(prev => [entry, ...prev].slice(0, 20))
    } catch (_e) {
      setCmdLog(prev => [{
        id: Date.now(),
        time: new Date().toLocaleTimeString(),
        type: cmdType,
        satellite: selectedSatId,
        success: false,
        message: "Network error"
      }, ...prev].slice(0, 20))
    } finally {
      setSending(false)
    }
  }

  if (!selectedSatId) {
    return (
      <div className="ui-panel ul-panel">
        <div className="ui-panel-header">
          <span className="ui-label">Uplink / Downlink</span>
        </div>
        <div className="ui-panel-body">
          <div className="ul-empty">
            Click a satellite on the globe to open telemetry
          </div>
        </div>
      </div>
    )
  }

  const risk = telemetry?.risk
  const riskClass = (
    risk?.status === "CRITICAL" ? "is-warning" :
    risk?.status === "WARNING" ? "is-caution" :
    risk?.status === "WATCH" ? "is-caution" :
    "is-nominal"
  )

  return (
    <div className="ui-panel ul-panel">

      {/* Header */}
      <div className="ui-panel-header">
        <span className="ul-panel-title">
          <span className="ui-label">Uplink / Downlink</span>
          {telemetry &&
            <span className="ul-header-meta">[{telemetry.agency}]</span>
          }
        </span>
        <span className={`ui-status ${telemetry ? "is-nominal" : "is-dim"}`}>
          {loading ? "..." : telemetry ? "LIVE" : "OFFLINE"}
        </span>
      </div>

      <div className="ul-body">
        {error && (
          <div className="ul-section">
            <div className="ul-banner is-warning">{error}</div>
          </div>
        )}

        {/* Downlink section */}
        {telemetry && (
          <div className="ul-section">
            <div className="ul-section-head">
              <span className="ui-label">Downlink</span>
              <span className="ul-header-meta">{telemetry.name}</span>
            </div>

            <div className="ul-telem-grid">
            <div className="ul-kv-row">
              <span className="ul-kv-key">Altitude</span>
              <span className="ul-kv-val">{telemetry.orbital?.altitude_km?.toFixed(1)}<span className="ul-unit">km</span></span>
            </div>
            <div className="ul-kv-row">
              <span className="ul-kv-key">Speed</span>
              <span className="ul-kv-val">{telemetry.orbital?.speed_kms?.toFixed(3)}<span className="ul-unit">km/s</span></span>
            </div>
            <div className="ul-kv-row">
              <span className="ul-kv-key">Period</span>
              <span className="ul-kv-val">{telemetry.orbital?.period_min?.toFixed(1)}<span className="ul-unit">min</span></span>
            </div>
            <div className="ul-kv-row">
              <span className="ul-kv-key">Pos X</span>
              <span className="ul-kv-val">{telemetry.position_eci_km?.x?.toFixed(0)}<span className="ul-unit">km</span></span>
            </div>
            <div className="ul-kv-row">
              <span className="ul-kv-key">Pos Y</span>
              <span className="ul-kv-val">{telemetry.position_eci_km?.y?.toFixed(0)}<span className="ul-unit">km</span></span>
            </div>
            <div className="ul-kv-row">
              <span className="ul-kv-key">Pos Z</span>
              <span className="ul-kv-val">{telemetry.position_eci_km?.z?.toFixed(0)}<span className="ul-unit">km</span></span>
            </div>
            <div className="ul-kv-row">
              <span className="ul-kv-key">Vel X</span>
              <span className="ul-kv-val">{telemetry.velocity_kms?.vx?.toFixed(4)}<span className="ul-unit">km/s</span></span>
            </div>
            <div className="ul-kv-row">
              <span className="ul-kv-key">Vel Y</span>
              <span className="ul-kv-val">{telemetry.velocity_kms?.vy?.toFixed(4)}<span className="ul-unit">km/s</span></span>
            </div>
            <div className="ul-kv-row">
              <span className="ul-kv-key">Vel Z</span>
              <span className="ul-kv-val">{telemetry.velocity_kms?.vz?.toFixed(4)}<span className="ul-unit">km/s</span></span>
            </div>
            <div className="ul-kv-row">
              <span className="ul-kv-key">Fuel</span>
              <span className="ul-kv-val">{telemetry.health?.fuel_remaining_pct?.toFixed(1)}<span className="ul-unit">%</span></span>
            </div>
            <div className="ul-kv-row">
              <span className="ul-kv-key">Battery</span>
              <span className="ul-kv-val">{telemetry.health?.battery_pct?.toFixed(1)}<span className="ul-unit">%</span></span>
            </div>
            <div className="ul-kv-row">
              <span className="ul-kv-key">Temp</span>
              <span className="ul-kv-val">{telemetry.health?.temperature_c?.toFixed(1)}<span className="ul-unit">C</span></span>
            </div>
            <div className="ul-kv-row">
              <span className="ul-kv-key">Signal</span>
              <span className="ul-kv-val">{telemetry.health?.signal_strength_dbm?.toFixed(1)}<span className="ul-unit">dBm</span></span>
            </div>
              <div className="ul-kv-row">
                <span className="ul-kv-key">Alerts</span>
                <span className="ul-kv-val">{telemetry.risk?.active_alerts ?? 0}</span>
              </div>
              <div className="ul-kv-row">
                <span className="ul-kv-key">CPI</span>
                <span className={`ul-kv-val ${riskClass}`}>
                  {telemetry.risk?.cpi_score?.toFixed(1)} - {" "}
                  {telemetry.risk?.status}
                </span>
              </div>
              <div className="ul-kv-row">
                <span className="ul-kv-key">Data Age</span>
                <span className="ul-kv-val">{telemetry.data_age_seconds}<span className="ul-unit">s ago</span></span>
              </div>
            </div>
          </div>
        )}

        {/* Uplink section */}
        <div className="ul-section">
          <div className="ul-section-head">
            <span className="ui-label">Uplink</span>
          </div>

          <div className="ul-form">
            <div className="ul-row ul-row--span">
              <label className="ul-row-label" htmlFor="ul-cmd-type">Command</label>
              <select
                id="ul-cmd-type"
                className="ui-select ul-select"
                value={cmdType}
                onChange={e => setCmdType(e.target.value)}
              >
                {COMMAND_TYPES.map(c => (
                  <option key={c.value} value={c.value}>
                    {c.label}
                  </option>
                ))}
              </select>
            </div>

            {cmdType === "MANEUVER" && (
              <>
                <div className="ul-row">
                  <span className="ul-row-label">Delta-V</span>
                  <input
                    className="ul-range"
                    type="range"
                    min="0.01" max="2.0" step="0.01"
                    value={deltaV}
                    onChange={e => setDeltaV(parseFloat(e.target.value))}
                  />
                  <span className="ul-row-value">
                    {deltaV.toFixed(2)}<span className="ul-unit">m/s</span>
                  </span>
                </div>
                <div className="ul-row ul-row--span">
                  <label className="ul-row-label" htmlFor="ul-cmd-dir">Direction</label>
                  <select
                    id="ul-cmd-dir"
                    className="ui-select ul-select"
                    value={direction}
                    onChange={e => setDirection(e.target.value)}
                  >
                    {DIRECTIONS.map(d => (
                      <option key={d.value} value={d.value}>
                        {d.label}
                      </option>
                    ))}
                  </select>
                </div>
              </>
            )}

            {cmdType === "SET_ORBIT" && (
              <div className="ul-row">
                <span className="ul-row-label">Target Alt</span>
                <input
                  className="ul-range"
                  type="range"
                  min="200" max="35786" step="10"
                  value={altitudeKm}
                  onChange={e => setAltitudeKm(parseInt(e.target.value))}
                />
                <span className="ul-row-value">
                  {altitudeKm}<span className="ul-unit">km</span>
                </span>
              </div>
            )}

            <div className="ul-actions">
              <button
                type="button"
                className="ui-btn ui-btn--primary"
                onClick={sendUplink}
                disabled={sending || !selectedSatId}
              >
                {sending ? "SENDING UPLINK..." : "SEND UPLINK COMMAND"}
              </button>
            </div>
          </div>
        </div>

        {/* Command log */}
        {cmdLog.length > 0 && (
          <div className="ul-section">
            <div className="ul-section-head">
              <span className="ui-label">Command Log</span>
            </div>
            <div className="ul-log">
              <div className="ul-log-row ul-log-row--head">
                <span>Time</span><span>Dir</span><span>Cmd</span><span>Message</span><span className="ul-log-status">Stat</span>
              </div>
              {cmdLog.map(entry => (
                <div key={entry.id} className="ul-log-row" title={entry.message}>
                  <span className="ul-log-time">{entry.time}</span>
                  <span className="ul-log-dir">UL</span>
                  <span className="ul-log-type">{entry.type}</span>
                  <span className="ul-log-msg">{entry.message}</span>
                  <span className={`ul-log-status ${entry.success ? "is-nominal" : "is-warning"}`}>
                    {entry.success ? "OK" : "FAIL"}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
