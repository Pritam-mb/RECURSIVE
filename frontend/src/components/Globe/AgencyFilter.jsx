import { useState, useEffect } from "react"
import useStore from "../../store/useStore"
import "../../styles/drawers.css"

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
    <div className="dr-filter">
      <div className="ui-label dr-filter-title">Filter by Agency</div>
      <div className="dr-chips">
        <button
          type="button"
          className="dr-chip"
          aria-pressed={selected.has("ALL")}
          onClick={() => toggle("ALL")}
        >
          ALL
        </button>
        {agencies.map(ag => (
          <button
            key={ag.name}
            type="button"
            className="dr-chip"
            aria-pressed={selected.has(ag.name)}
            onClick={() => toggle(ag.name)}
            title={`${ag.name}: ${ag.count} satellites`}
          >
            <span className="dr-chip-swatch" style={{ background: ag.color }} />
            {ag.name}
            <span className="dr-chip-count">{ag.count}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
