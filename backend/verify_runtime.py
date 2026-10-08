"""Ad-hoc runtime verification of the Orbital Sentinel API surface."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
os.environ.setdefault("MAX_SATS", "120")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

import asyncio  # noqa: E402


def main_run() -> int:
    with TestClient(main.app) as client:
        print("ROOT:", json.dumps(client.get("/").json()))

        sats = client.get("/api/satellites")
        print("SATELLITES: status", sats.status_code)
        sj = sats.json()
        print("  count:", sj.get("count"), "list:", len(sj.get("satellites", [])))
        print("  timestamp:", sj.get("timestamp"))
        if sj.get("satellites"):
            print("  sample:", json.dumps(sj["satellites"][0]))

        print("--- forcing one alert refresh cycle ---")
        asyncio.get_event_loop().run_until_complete(main.refresh_alerts_once())

        a = client.get("/api/alerts")
        print("ALERTS: status", a.status_code)
        aj = a.json()
        print("  count:", aj.get("count"))
        print("  alerts:", len(aj.get("alerts", [])))
        print("  hotspots:", len(aj.get("hotspots", [])))
        print("  debris_clouds:", len(aj.get("debris_clouds", [])))
        print("  graph:", aj.get("graph"))
        print("  cascade_depth:", aj.get("cascade_depth"))
        print("  agencies:", aj.get("agencies_involved"))
        print("  timestamp:", aj.get("timestamp"))
        rr = aj.get("ranker_review", {})
        print("  ranker_review:", json.dumps(rr, indent=2)[:900])
        probs = aj.get("node_probabilities", {})
        print(
            "  node_probabilities:",
            len(probs),
            "entries,",
            sum(1 for v in probs.values() if v > 0),
            "non-zero (P16: must not be all-zero)",
        )
        if aj.get("alerts"):
            print("  sample alert:", json.dumps(aj["alerts"][0])[:600])
        if aj.get("hotspots"):
            print("  sample hotspot:", json.dumps(aj["hotspots"][0])[:400])
        if aj.get("debris_clouds"):
            print("  sample debris:", json.dumps(aj["debris_clouds"][0])[:400])

        for path in (
            "/api/model-metrics",
            "/api/cascade/status",
            "/api/scenarios",
            "/api/agencies",
            "/api/anomalies",
        ):
            r = client.get(path)
            body = r.text
            print(f"{path}: status {r.status_code} len={len(body)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main_run())
