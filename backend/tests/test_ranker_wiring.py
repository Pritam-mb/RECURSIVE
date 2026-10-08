"""
The GAT / GNN cascade rankers are synthetic-trained and therefore ADVISORY.

History: P15 (misnamed production ranker) and P16 (silent all-zero fallback)
were fixed by making failures loud. The planner has since been rebuilt so the
published cascade depth, node probabilities and manoeuvre plan come only from
the alert graph (BFS hops, 1 - prod(1 - Pc)) and re-propagation. These tests
pin that contract: ranker outputs and ranker failures must never change the
published physics numbers, and failures must be visible in ranker_review.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.cascade_planner import CascadePlanner  # noqa: E402


def _state(norad_id: int, x: float):
    class _State:
        error_code = 0
        name = f"SAT-{norad_id}"
        vx, vy, vz = 0.0, 7.5, 0.0
        y = z = 0.0

    s = _State()
    s.norad_id = norad_id
    s.x = x
    return s


def _scene():
    states = [_state(1000 + i, 7000.0 + 10 * i) for i in range(6)]
    alerts = []
    for i in range(5):
        pc = 10.0 ** -(3 + i)
        alerts.append({"id": f"{1000+i}-{1001+i}", "sat1": {"id": 1000 + i, "name": f"SAT-{1000+i}", "object_type": "PAY"},
                       "sat2": {"id": 1001 + i, "name": f"SAT-{1001+i}", "object_type": "PAY"},
                       "p_collision": pc, "miss_distance_km": 1.0 + i, "relative_speed_kms": 10.0,
                       "tca_minutes": 60.0, "tca_utc": "2026-05-08T13:00:00+00:00"})
    return states, alerts


@pytest.fixture()
def planner():
    return CascadePlanner()


def _physics(out):
    return out["node_probabilities"], [a["cascade_depth"] for a in out["alerts"]]


def test_ranker_failure_does_not_change_physics_outputs(planner, monkeypatch):
    states, alerts = _scene()
    healthy = planner.analyze_snapshot(states, [dict(a) for a in alerts], None, "2026-05-08T12:00:00+00:00",
                                       plan_maneuvers=False)
    if planner.primary_ranker is not None:
        monkeypatch.setattr(planner.primary_ranker, "predict", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gat down")))
    if planner.cross_check_ranker is not None:
        monkeypatch.setattr(planner.cross_check_ranker, "predict", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gnn down")))
    broken = planner.analyze_snapshot(states, [dict(a) for a in alerts], None, "2026-05-08T12:00:00+00:00",
                                      plan_maneuvers=False)
    assert _physics(healthy) == _physics(broken)
    assert broken["ranker_review"]["primary_ok"] is False
    assert broken["ranker_review"]["cross_check_ok"] is False
    assert broken["ranker_review"]["role"] == "advisory"


def test_node_probabilities_are_physics_not_ranker(planner):
    states, alerts = _scene()
    out = planner.analyze_snapshot(states, alerts, None, "2026-05-08T12:00:00+00:00", plan_maneuvers=False)
    # SAT-1001 is in two alerts: 1e-3 and 1e-4
    assert out["node_probabilities"]["1001"] == pytest.approx(1 - (1 - 1e-3) * (1 - 1e-4))


def test_empty_and_populated_ranker_review_share_a_key_set(planner):
    states, alerts = _scene()
    empty = planner.analyze_snapshot([], [], None, "2026-05-08T12:00:00+00:00")["ranker_review"]
    populated = planner.analyze_snapshot(states, alerts, None, "2026-05-08T12:00:00+00:00",
                                         plan_maneuvers=False)["ranker_review"]
    assert set(empty) == set(populated)
