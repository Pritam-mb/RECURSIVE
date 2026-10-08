"""
Regression tests for P15 (misleading production ranker) and P16 (silent zero
fallback) in the cascade planner.

P15: the planner stored a `CascadeGAT` on an attribute named `self.gnn` and
called it "gnn_output", which made the code read as though a graph attention
network was being shadowed by a GNN. In reality `CascadeGNN` was never in the
request path at all -- it existed only for metrics and tests. The planner now
names the primary ranker for what it is and runs `CascadeGNN` deliberately as a
cross-check whose disagreements are reported, never blended.

P16: any ranker exception was caught and replaced with
`np.zeros(...)` for both manoeuvre probability and cascade depth, and the
pipeline continued to emit a plan. Every probability became 0.0, so the plan
looked normal while being entirely fabricated. The planner must now surface the
failure instead of inventing numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.gat_cascade import CascadeGAT  # noqa: E402
from app.ml.gnn_cascade import CascadeGNN  # noqa: E402
from app.services.cascade_planner import (  # noqa: E402
    RANKER_DISAGREEMENT_THRESHOLD,
    CascadePlanner,
)


def _state(norad_id: int, x: float, y: float = 0.0, z: float = 0.0):
    class _State:
        error_code = 0
        name = f"SAT-{norad_id}"
        epoch_utc = "2026-01-01T12:00:00+00:00"
        vx = 0.0
        vy = 7.5
        vz = 0.0

    state = _State()
    state.norad_id = norad_id
    state.x = x
    state.y = y
    state.z = z
    return state


def _cluster(count: int = 8, spacing_km: float = 30.0):
    """Satellites close enough to build a non-trivial cascade graph."""
    return [
        _state(1000 + i, 7000.0 + i * spacing_km, float(i), float(i) * 0.5)
        for i in range(count)
    ]


@pytest.fixture(scope="module")
def planner():
    return CascadePlanner()


def _score(planner, states, **kwargs):
    graph = planner.build_graph(states)
    nodes = graph["nodes"]
    adjacency = graph["adjacency"]
    node_index = graph["node_index"]
    features, edge_index, edge_attr = planner._build_gnn_inputs(
        nodes, graph["edges"], node_index, adjacency
    )
    return planner, nodes, features, edge_index, edge_attr, kwargs


class TestP15RankerIdentity:
    def test_primary_ranker_is_the_gat(self, planner):
        """The object driving the plan must be the attention model, and it must
        no longer be advertised under a `gnn` attribute name."""
        assert isinstance(planner.primary_ranker, CascadeGAT)
        assert isinstance(planner.cross_check_ranker, CascadeGNN)

    def test_no_ambiguous_gnn_attribute_remains(self, planner):
        assert not hasattr(planner, "gnn"), (
            "the ambiguous `self.gnn` attribute was the root of P15: it held a "
            "CascadeGAT, which misrepresented which model ranks production alerts"
        )

    def test_both_rankers_are_distinct_objects(self, planner):
        assert planner.primary_ranker is not planner.cross_check_ranker


class TestP16NoSilentZeroFallback:
    def test_both_rankers_failing_raises(self, planner, monkeypatch):
        """The core P16 assertion: with no ranker able to score, analyze_snapshot
        must raise rather than return a plan full of 0.0 probabilities."""
        monkeypatch.setattr(
            planner.primary_ranker,
            "predict",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gat down")),
        )
        monkeypatch.setattr(
            planner.cross_check_ranker,
            "predict",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gnn down")),
        )

        with pytest.raises(RuntimeError, match="Both cascade rankers failed"):
            planner.analyze_snapshot(_cluster())

    def test_primary_failure_falls_back_to_cross_check_and_flags_degraded(
        self, planner, monkeypatch
    ):
        monkeypatch.setattr(
            planner.primary_ranker,
            "predict",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gat down")),
        )

        result = planner.analyze_snapshot(_cluster())

        review = result["ranker_review"]
        assert review["degraded"] is True
        assert review["fallback_used"] is True
        assert review["primary_ok"] is False
        assert review["cross_check_ok"] is True
        # Probabilities must be real cross-check values, not zeros.
        assert any(p > 0 for p in result["node_probabilities"].values())

    def test_fallback_population_is_not_all_zero(self, planner, monkeypatch):
        """A cheap, direct check that the fabricated-all-zero output is gone."""
        monkeypatch.setattr(
            planner.primary_ranker,
            "predict",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gat down")),
        )
        result = planner.analyze_snapshot(_cluster())
        assert max(result["node_probabilities"].values()) > 0.0

    def test_empty_graph_does_not_invoke_ranker_and_is_not_degraded(
        self, planner, monkeypatch
    ):
        monkeypatch.setattr(
            planner.primary_ranker,
            "predict",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")),
        )
        result = planner.analyze_snapshot([])
        assert result["ranker_review"]["degraded"] is False
        assert result["cascade_plan"] == []

    def test_empty_and_populated_ranker_review_share_a_key_set(self, planner):
        """The two return paths must not drift: a client reading
        `ranker_review` should not have to special-case an empty graph."""
        empty = planner.analyze_snapshot([])["ranker_review"]
        populated = planner.analyze_snapshot(_cluster())["ranker_review"]
        assert set(empty) == set(populated)


class TestP15CrossCheckDisagreementReporting:
    def test_healthy_run_reports_both_rankers_ok(self, planner):
        result = planner.analyze_snapshot(_cluster())
        review = result["ranker_review"]
        assert review["primary"] == "gat"
        assert review["cross_check"] == "gnn"
        assert review["primary_ok"] is True
        assert review["cross_check_ok"] is True
        assert review["degraded"] is False
        assert review["fallback_used"] is False

    def test_disagreement_is_reported_not_fused(self, planner, monkeypatch):
        """The cross-check may disagree, but its scores must not be averaged
        into the plan: the published probabilities must remain the GAT's.

        The stub deliberately does not use `1 - p`, because the GAT's output is
        near-constant (~0.52) on this cluster, so that construction would land
        under the threshold and test nothing.
        """
        states = _cluster()
        graph = planner.build_graph(states)
        nodes = graph["nodes"]
        features, edge_index, edge_attr = planner._build_gnn_inputs(
            nodes, graph["edges"], graph["node_index"], graph["adjacency"]
        )
        gat = planner.primary_ranker.predict(features, edge_index, edge_attr)
        gat_probs = np.asarray(gat["maneuver_probability"], dtype=float)
        inverted = np.where(gat_probs > 0.5, gat_probs - 0.5, gat_probs + 0.5)
        monkeypatch.setattr(
            planner.cross_check_ranker,
            "predict",
            lambda *a, **k: {"maneuver_probability": inverted, "cascade_depth": gat["cascade_depth"]},
        )

        result = planner.analyze_snapshot(states)
        review = result["ranker_review"]

        assert review["disagreement_count"] > 0
        assert review["max_disagreement"] > RANKER_DISAGREEMENT_THRESHOLD
        for entry in review["disputed_satellites"]:
            # Published probability is the GAT's, not a blend of the two.
            assert entry["gat_probability"] == pytest.approx(
                result["node_probabilities"][str(entry["satellite_id"])], abs=1e-3
            )
            assert entry["delta"] > RANKER_DISAGREEMENT_THRESHOLD
            # A blend would land between the two; the published value must not.
            lo, hi = sorted((entry["gat_probability"], entry["gnn_probability"]))
            assert not (lo < entry["gat_probability"] < hi)

    def test_small_disagreement_is_not_flagged(self, planner, monkeypatch):
        """Below-threshold noise must not spam the operator's console."""
        states = _cluster()
        graph = planner.build_graph(states)
        features, edge_index, edge_attr = planner._build_gnn_inputs(
            graph["nodes"],
            graph["edges"],
            graph["node_index"],
            graph["adjacency"],
        )
        gat = planner.primary_ranker.predict(features, edge_index, edge_attr)
        nudged = np.asarray(gat["maneuver_probability"], dtype=float) + 0.01
        monkeypatch.setattr(
            planner.cross_check_ranker,
            "predict",
            lambda *a, **k: {"maneuver_probability": nudged, "cascade_depth": gat["cascade_depth"]},
        )

        result = planner.analyze_snapshot(states)
        assert result["ranker_review"]["disagreement_count"] == 0
        assert result["ranker_review"]["disputed_satellites"] == []

    def test_cross_check_failure_does_not_degrade_a_healthy_primary(
        self, planner, monkeypatch
    ):
        monkeypatch.setattr(
            planner.cross_check_ranker,
            "predict",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gnn down")),
        )
        result = planner.analyze_snapshot(_cluster())
        review = result["ranker_review"]
        assert review["primary_ok"] is True
        assert review["cross_check_ok"] is False
        assert review["degraded"] is False
        assert max(result["node_probabilities"].values()) > 0.0
