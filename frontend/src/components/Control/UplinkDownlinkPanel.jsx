import { useState, useEffect, useRef } from "react"
import useStore from "../../store/useStore"

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
      <div className="panel uplink-panel empty">
        <div className="panel-header">Uplink / Downlink</div>
        <div className="empty-msg">
          Click a satellite on the globe to open telemetry
        </div>
      </div>
    )
  }

  const risk = telemetry?.risk
  const riskColor = (
    risk?.status === "CRITICAL" ? "#E24B4A" :
    risk?.status === "WARNING" ? "#EF9F27" :
    risk?.status === "WATCH" ? "#F5C518" :
    "#1D9E75"
  )

  return (
    <div className="panel uplink-panel">

      {/* Header */}
      <div className="panel-header">
        <span>
          Uplink / Downlink
          {telemetry &&
            <span
              style={{ color: telemetry.agency_color || "#888",
                       marginLeft: 8, fontSize: 12 }}
            >
              [{telemetry.agency}]
            </span>
          }
        </span>
        <span className={`live-dot ${telemetry ? "live" : "dead"}`}>
          {loading ? "..." : telemetry ? "LIVE" : "OFFLINE"}
        </span>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {/* Downlink section */}
      {telemetry && (
        <div className="telem-section">
          <div className="section-label">
            DOWNLINK - {telemetry.name}
          </div>

          <div className="telem-grid">
            <div className="telem-row">
              <span>Altitude</span>
              <span>
                {telemetry.orbital?.altitude_km?.toFixed(1)} km
              </span>
            </div>
            <div className="telem-row">
              <span>Speed</span>
              <span>
                {telemetry.orbital?.speed_kms?.toFixed(3)} km/s
              </span>
            </div>
            <div className="telem-row">
              <span>Period</span>
              <span>
                {telemetry.orbital?.period_min?.toFixed(1)} min
              </span>
            </div>
            <div className="telem-row">
              <span>Pos X</span>
              <span>
                {telemetry.position_eci_km?.x?.toFixed(0)} km
              </span>
            </div>
            <div className="telem-row">
              <span>Pos Y</span>
              <span>
                {telemetry.position_eci_km?.y?.toFixed(0)} km
              </span>
            </div>
            <div className="telem-row">
              <span>Pos Z</span>
              <span>
                {telemetry.position_eci_km?.z?.toFixed(0)} km
              </span>
            </div>
            <div className="telem-row">
              <span>Vel X</span>
              <span>
                {telemetry.velocity_kms?.vx?.toFixed(4)} km/s
              </span>
            </div>
            <div className="telem-row">
              <span>Vel Y</span>
              <span>
                {telemetry.velocity_kms?.vy?.toFixed(4)} km/s
              </span>
            </div>
            <div className="telem-row">
              <span>Vel Z</span>
              <span>
                {telemetry.velocity_kms?.vz?.toFixed(4)} km/s
              </span>
            </div>
            <div className="telem-row">
              <span>Fuel</span>
              <span>
                {telemetry.health?.fuel_remaining_pct?.toFixed(1)}%
              </span>
            </div>
            <div className="telem-row">
              <span>Battery</span>
              <span>
                {telemetry.health?.battery_pct?.toFixed(1)}%
              </span>
            </div>
            <div className="telem-row">
              <span>Temp</span>
              <span>
                {telemetry.health?.temperature_c?.toFixed(1)} C
              </span>
            </div>
            <div className="telem-row">
              <span>Signal</span>
              <span>
                {telemetry.health?.signal_strength_dbm?.toFixed(1)} dBm
              </span>
            </div>
            <div className="telem-row">
              <span>Alerts</span>
              <span>{telemetry.risk?.active_alerts ?? 0}</span>
            </div>
            <div className="telem-row">
              <span>CPI</span>
              <span style={{ color: riskColor, fontWeight: 600 }}>
                {telemetry.risk?.cpi_score?.toFixed(1)} - {" "}
                {telemetry.risk?.status}
              </span>
            </div>
            <div className="telem-row">
              <span>Data Age</span>
              <span>{telemetry.data_age_seconds}s ago</span>
            </div>
          </div>
        </div>
      )}

      {/* Uplink section */}
      <div className="uplink-section">
        <div className="section-label">UPLINK - Send Command</div>

        <div className="control-row">
          <label>Command Type</label>
          <select
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
            <div className="control-row">
              <label>
                Delta-V: <strong>{deltaV.toFixed(2)} m/s</strong>
              </label>
              <input
                type="range"
                min="0.01" max="2.0" step="0.01"
                value={deltaV}
                onChange={e => setDeltaV(parseFloat(e.target.value))}
              />
            </div>
            <div className="control-row">
              <label>Direction</label>
              <select
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
          <div className="control-row">
            <label>
              Target Altitude: <strong>{altitudeKm} km</strong>
            </label>
            <input
              type="range"
              min="200" max="35786" step="10"
              value={altitudeKm}
              onChange={e => setAltitudeKm(parseInt(e.target.value))}
            />
          </div>
        )}

        <button
          className="uplink-btn"
          onClick={sendUplink}
          disabled={sending || !selectedSatId}
        >
          {sending ? "SENDING UPLINK..." : "SEND UPLINK COMMAND"}
        </button>
      </div>

      {/* Command log */}
      {cmdLog.length > 0 && (
        <div className="cmd-log-section">
          <div className="section-label">COMMAND LOG</div>
          <div className="cmd-log">
            {cmdLog.map(entry => (
              <div
                key={entry.id}
                className={`log-row ${entry.success ? "ok" : "fail"}`}
              >
                <span className="log-time">{entry.time}</span>
                <span className="log-type">{entry.type}</span>
                <span className="log-status">
                  {entry.success ? "OK" : "FAIL"}
                </span>
                <span className="log-msg">{entry.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
