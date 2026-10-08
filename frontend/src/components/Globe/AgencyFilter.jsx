import { useState, useEffect } from "react"
import useStore from "../../store/useStore"

const API = ""

export default function AgencyFilter() {
  const [agencies, setAgencies] = useState([])
  const [selected, setSelected] = useState(new Set(["ALL"]))
  const { setAgencyFilter } = useStore()

  useEffect(() => {
    fetch(`${API}/api/agencies`)
      .then(r => r.json())
      .then(d => setAgencies(d.agencies || []))
      .catch(() => {})
  }, [])

  const toggle = (name) => {
    if (name === "ALL") {
      const next = new Set(["ALL"])
      setSelected(next)
      setAgencyFilter?.(null)
      return
    }
    const next = new Set(selected)
    next.delete("ALL")
    if (next.has(name)) {
      next.delete(name)
      if (next.size === 0) next.add("ALL")
    } else {
      next.add(name)
    }
    setSelected(next)
    setAgencyFilter?.(
      next.has("ALL") ? null : Array.from(next)
    )
  }

  return (
    <div className="agency-filter">
      <div className="filter-header">Filter by Agency</div>
      <div className="filter-chips">
        <button
          className={`agency-chip ${selected.has("ALL") ? "active" : ""}`}
          style={{ borderColor: "#aaa", color: "#aaa" }}
          onClick={() => toggle("ALL")}
        >
          ALL
        </button>
        {agencies.map(ag => (
          <button
            key={ag.name}
            className={`agency-chip ${selected.has(ag.name) ? "active" : ""}`}
            style={{
              borderColor: ag.color,
              color: selected.has(ag.name) ? "#fff" : ag.color,
              backgroundColor: selected.has(ag.name)
                ? ag.color : "transparent"
            }}
            onClick={() => toggle(ag.name)}
            title={`${ag.name}: ${ag.count} satellites`}
          >
            {ag.name} ({ag.count})
          </button>
        ))}
      </div>
    </div>
  )
}
