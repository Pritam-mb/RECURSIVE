"""Regression tests for the graph-attention cascade ranker.

The audit found that ``CascadeGAT`` was a silent alias of ``CascadeGNN`` because
its only implementation path was gated behind the uninstalled ``torch_geometric``
package. The cached metrics for the two models were byte-identical as a result.
These tests pin down the properties that must hold so the degeneracy cannot come
back unnoticed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.ml.artifacts import artifact_path
from app.ml.gat_cascade import (
    ARTIFACT_VERSION,
    GAT_META,
    TORCH_AVAILABLE,
    CascadeGAT,
    GraphAttentionBank,
    TorchGraphAttentionBank,
    _scale_edges,
    _segment_softmax,
    load_or_build_attention_bank,
)
from app.ml.gnn_cascade import CascadeGNN
from app.ml.synthetic_data import generate_graph_snapshots

EDGE_DIM = 4


def _graphs(count: int = 6, seed: int = 31):
    return [
        graph
        for graph in generate_graph_snapshots(sample_count=count, seed=seed)
        if graph["node_features"].shape[0] > 0
    ]


def _bank(input_dim: int = 10, hidden_dim: int = 8, heads: int = 4, seed: int = 17):
    rng = np.random.default_rng(seed)
    return GraphAttentionBank(
        input_dim=input_dim,
        edge_dim=EDGE_DIM,
        hidden_dim=hidden_dim,
        heads=heads,
        seed=seed,
        node_mean=rng.normal(0.0, 1.0, size=input_dim),
        node_std=rng.uniform(0.5, 2.0, size=input_dim),
    )


def _toy_graph(node_count: int = 12, seed: int = 5):
    rng = np.random.default_rng(seed)
    node_features = rng.normal(0.0, 1.0, size=(node_count, 10))
    if node_count >= 3:
        edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=int)
        edge_attr = np.column_stack(
            [
                rng.uniform(1.0, 500.0, edge_index.shape[1]),
                rng.uniform(0.1, 1.5, edge_index.shape[1]),
                rng.uniform(0.0, 180.0, edge_index.shape[1]),
                rng.uniform(0.0, 1e-3, edge_index.shape[1]),
            ]
        )
    else:
        edge_index = np.zeros((2, 0), dtype=int)
        edge_attr = np.zeros((0, EDGE_DIM), dtype=float)
    return node_features, edge_index, edge_attr


class TestAttentionIsNotAnAlias:
    def test_deployed_bank_exists_and_is_current(self):
        bank = load_or_build_attention_bank()
        assert bank is not None, "no GAT attention bank could be loaded or trained"
        assert bank.input_dim == 10
        assert bank.edge_dim == EDGE_DIM
        assert bank.heads >= 1
        assert bank.hidden_dim >= 1

    def test_artifact_is_versioned_and_records_training(self):
        with artifact_path(GAT_META).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["version"] == ARTIFACT_VERSION
        assert payload["trainer"] == "torch_autograd"
        assert payload["trained_from"] == "synthetic_graph_snapshots"
        assert payload["epochs"] > 0
        assert "model" in payload

    def test_cascade_gat_is_not_degenerate(self):
        assert CascadeGAT().is_degenerate is False

    def test_probabilities_differ_from_the_gnn(self):
        gat = CascadeGAT()
        gnn = CascadeGNN()
        graphs = _graphs()
        assert graphs

        worst_gap = 0.0
        for graph in graphs:
            args = (graph["node_features"], graph["edge_index"], graph["edge_attr"])
            gat_p = gat.predict(*args)["maneuver_probability"]
            gnn_p = gnn.predict(*args)["maneuver_probability"]
            assert gat_p.shape == gnn_p.shape
            worst_gap = max(
                worst_gap, float(np.max(np.abs(np.asarray(gat_p) - np.asarray(gnn_p))))
            )

        # A pass-through alias would give an exact zero gap on every graph.
        assert worst_gap > 1e-3, (
            f"GAT output is effectively identical to the GNN (max gap {worst_gap}); "
            "the GAT is degenerate again."
        )

    def test_output_is_deterministic(self):
        node_features, edge_index, edge_attr = _toy_graph()
        first = GraphAttentionBank.from_dict(_bank().to_dict())
        second = GraphAttentionBank.from_dict(_bank().to_dict())
        assert np.array_equal(
            first.forward(node_features, edge_index, edge_attr),
            second.forward(node_features, edge_index, edge_attr),
        )

    def test_reloaded_artifact_reproduces_predictions(self):
        node_features, edge_index, edge_attr = _toy_graph()
        reloaded = GraphAttentionBank.from_dict(_bank().to_dict())
        original = _bank()
        assert np.array_equal(
            reloaded.forward(node_features, edge_index, edge_attr),
            original.forward(node_features, edge_index, edge_attr),
        )


class TestAttentionSemantics:
    def test_self_attention_gives_an_isolated_node_its_own_state(self):
        bank = _bank()
        node_features, edge_index, edge_attr = _toy_graph(node_count=6)
        scores, targets, _ = bank.attention_coefficients(
            node_features, np.zeros((2, 0), dtype=int), np.zeros((0, EDGE_DIM))
        )
        # Only the six self entries exist, so every alpha must be exactly 1.
        alpha = _segment_softmax(scores, targets, 6)
        assert np.allclose(alpha, 1.0)

    def test_isolated_nodes_are_not_collapsed_to_one_value(self):
        bank = _bank()
        node_features = np.random.default_rng(11).normal(size=(7, 10))
        logits = bank.forward(
            node_features, np.zeros((2, 0), dtype=int), np.zeros((0, EDGE_DIM))
        )
        assert logits.shape == (7,)
        assert len(np.unique(np.round(logits, 9))) > 1

    def test_attention_rows_sum_to_one(self):
        bank = _bank()
        node_features, edge_index, edge_attr = _toy_graph()
        scores, targets, _ = bank.attention_coefficients(
            node_features, edge_index, edge_attr
        )
        alpha = _segment_softmax(scores, targets, node_features.shape[0])
        summed = np.zeros((alpha.shape[0], node_features.shape[0]))
        np.add.at(summed.T, targets, alpha.T)
        assert np.allclose(summed, 1.0, atol=1e-9)

    def test_attention_depends_on_edge_attributes(self):
        bank = _bank()
        node_features, edge_index, _ = _toy_graph()
        base = bank.attention_coefficients(
            node_features, edge_index, np.full((edge_index.shape[1], EDGE_DIM), 0.1)
        )[0]
        shifted = bank.attention_coefficients(
            node_features, edge_index, np.full((edge_index.shape[1], EDGE_DIM), 0.9)
        )[0]
        assert not np.allclose(base, shifted)

    def test_segment_softmax_is_stable_for_large_scores(self):
        scores = np.array([[1000.0, 1001.0, -1000.0, 999.0]])
        targets = np.array([0, 0, 1, 1])
        alpha = _segment_softmax(scores, targets, 2)
        assert np.all(np.isfinite(alpha))
        # Each segment normalises independently, so the per-segment total is 1.
        assert np.isclose(alpha[0, targets == 0].sum(), 1.0, atol=1e-9)
        assert np.isclose(alpha[0, targets == 1].sum(), 1.0, atol=1e-9)


class TestEdgeCases:
    @pytest.mark.parametrize("node_count", [0, 1, 2, 5])
    def test_output_shape_tracks_node_count(self, node_count):
        bank = _bank()
        node_features = np.random.default_rng(node_count).normal(size=(node_count, 10))
        logits = bank.forward(
            node_features, np.zeros((2, 0), dtype=int), np.zeros((0, EDGE_DIM))
        )
        assert logits.shape == (node_count,)

    def test_predict_on_empty_graph(self):
        result = CascadeGAT().predict(
            np.zeros((0, 10)), np.zeros((2, 0), dtype=int), np.zeros((0, EDGE_DIM))
        )
        assert result["maneuver_probability"].shape == (0,)
        assert result["delta_v_rsw_ms"].shape == (0, 3)
        assert result["cascade_depth"].shape == (0,)
        assert result["graph_influence"].shape == (0,)

    def test_predict_shapes_on_a_connected_graph(self):
        node_features, edge_index, edge_attr = _toy_graph(node_count=14)
        result = CascadeGAT().predict(node_features, edge_index, edge_attr)
        count = node_features.shape[0]
        assert result["maneuver_probability"].shape == (count,)
        assert result["delta_v_rsw_ms"].shape == (count, 3)
        assert result["cascade_depth"].shape == (count,)
        assert result["graph_influence"].shape == (count,)
        assert np.all((result["maneuver_probability"] >= 0.0))
        assert np.all((result["maneuver_probability"] <= 1.0))
        assert np.all((result["cascade_depth"] >= 0) & (result["cascade_depth"] <= 4))

    def test_rejects_malformed_inputs(self):
        bank = _bank()
        with pytest.raises(ValueError):
            bank.forward(
                np.zeros(10), np.zeros((2, 0), dtype=int), np.zeros((0, EDGE_DIM))
            )
        with pytest.raises(ValueError):
            bank.forward(
                np.zeros((3, 10)),
                np.array([[0], [99]], dtype=int),
                np.zeros((1, EDGE_DIM)),
            )

    def test_edge_attribute_padding_and_scaling(self):
        scaled = _scale_edges(np.ones((2, 2)), EDGE_DIM)
        assert scaled.shape == (2, EDGE_DIM)
        # The two missing columns are padded with zeros, not garbage.
        assert np.allclose(scaled[:, 2:], 0.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
class TestTorchTrainingMirror:
    def test_torch_and_numpy_agree(self):
        torch = pytest.importorskip("torch")
        graphs = _graphs(count=6, seed=31)
        features = np.vstack([g["node_features"] for g in graphs])
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-6

        model = TorchGraphAttentionBank(features.shape[1], EDGE_DIM, 8, 4, 17)
        # The bank must normalize with the same statistics the torch model was
        # given, exactly as the exported artifact does at runtime.
        bank = GraphAttentionBank(
            input_dim=features.shape[1],
            edge_dim=EDGE_DIM,
            hidden_dim=8,
            heads=4,
            seed=17,
            node_mean=mean,
            node_std=std,
        )
        bank.W_src = model.W_src.detach().numpy()
        bank.a_src = model.a_src.detach().numpy()
        bank.a_tgt = model.a_tgt.detach().numpy()
        bank.a_edge = model.a_edge.detach().numpy()
        bank.W_out = model.W_out.detach().numpy()
        bank.b_out = model.b_out.detach().numpy()

        for graph in graphs:
            edge_index = graph["edge_index"].reshape(2, -1)
            with torch.no_grad():
                trained = model(
                    torch.tensor((graph["node_features"] - mean) / std, dtype=torch.float32),
                    torch.tensor(edge_index.astype(np.int64), dtype=torch.long),
                    torch.tensor(_scale_edges(graph["edge_attr"], EDGE_DIM), dtype=torch.float32),
                ).numpy()
            inferred = bank.forward(
                graph["node_features"], graph["edge_index"], graph["edge_attr"]
            )
            assert np.allclose(trained, inferred, atol=1e-4)

    def test_attention_parameters_receive_gradients(self):
        torch = pytest.importorskip("torch")
        graph = _graphs(count=2, seed=31)[0]
        model = TorchGraphAttentionBank(graph["node_features"].shape[1], EDGE_DIM, 8, 4, 17)
        # Normalize as training does. Raw ECI components reach 1e4, which would
        # saturate the segment softmax and zero out the attention gradients.
        features = graph["node_features"]
        normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
        logits = model(
            torch.tensor(normalized, dtype=torch.float32),
            torch.tensor(graph["edge_index"].reshape(2, -1).astype(np.int64), dtype=torch.long),
            torch.tensor(_scale_edges(graph["edge_attr"], EDGE_DIM), dtype=torch.float32),
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor(graph["targets"], dtype=torch.float32)
        )
        loss.backward()
        for name in ("W_src", "a_src", "a_tgt", "a_edge", "W_out", "b_out"):
            parameter = getattr(model, name)
            assert parameter.grad is not None, f"{name} received no gradient"
            assert torch.isfinite(parameter.grad).all(), f"{name} gradient is not finite"
            assert float(parameter.grad.abs().sum()) > 0.0, f"{name} gradient is all zero"
