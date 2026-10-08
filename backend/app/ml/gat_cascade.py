"""
Graph-attention cascade ranker (GAT).

The audit found this module was not actually a graph-attention network. It gated
all of its logic behind ``torch_geometric``, which is not present in
``requirements.txt`` or ``requirements-optional.txt`` and is not installed. With
the import failing, ``TORCH_AVAILABLE`` was ``False`` and every call to
``CascadeGAT.predict()`` silently delegated to ``CascadeGNN.predict()``. The GAT
was therefore a pass-through alias of the GNN, which is exactly why the cached
metrics for the two models were byte-identical (accuracy 98.76, precision 98.46,
recall 98.69, F1 98.57) -- the paper flagged this as an unexplained degeneracy.

This module implements a real multi-head graph attention network:

* attention is a softmax over each target node's in-edges *plus its own state*,
  so an isolated node falls back to full self-attention instead of collapsing to
  a constant, and a connected node can learn how much of its own state to keep;
* every head gets its own projections and its own attention vectors, and the
  heads are concatenated before the readout;
* the weights are gradient-trained with PyTorch (autograd only -- no
  ``torch_geometric`` dependency) and exported to JSON, so runtime inference is
  deterministic NumPy and needs no deep-learning stack.

That is structurally different from ``CascadeGNN``, which scores each node from
fixed global summary statistics and never computes attention weights.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .artifacts import artifact_path, load_artifact, save_artifact
from .gnn_cascade import CascadeGNN
from .synthetic_data import build_graph_summary_features, sigmoid

logger = logging.getLogger(__name__)

GAT_META = "gat_model.json"
ARTIFACT_VERSION = 3

# Edge feature scaling used to bring raw edge attributes onto a comparable
# range before they enter the attention score. Without this the miss-distance
# term (order 1e2 km) would swamp the probability term (order 1e-4).
EDGE_FEATURE_SCALE = np.array([1.0 / 500.0, 1.0 / 1.5, 1.0 / 180.0, 1.0e4], dtype=float)

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def _scale_edges(edge_attr: np.ndarray, edge_dim: int) -> np.ndarray:
    """Trim/pad edge attributes to ``edge_dim`` and apply the fixed scaling."""
    edge_attr = np.asarray(edge_attr, dtype=float)
    if edge_attr.ndim != 2:
        edge_attr = edge_attr.reshape(edge_attr.shape[0], -1)
    if edge_attr.shape[1] < edge_dim:
        pad = np.zeros((edge_attr.shape[0], edge_dim - edge_attr.shape[1]), dtype=float)
        edge_attr = np.concatenate([edge_attr, pad], axis=1)
    return edge_attr[:, :edge_dim] * EDGE_FEATURE_SCALE[:edge_dim]


def _split_graph_inputs(
    node_features: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Validate and normalise the graph inputs shared by both forward passes."""
    node_features = np.asarray(node_features, dtype=float)
    edge_index = np.asarray(edge_index, dtype=int)
    edge_attr = np.asarray(edge_attr, dtype=float)

    if node_features.ndim != 2:
        raise ValueError("node_features must be a 2D array")
    node_count = int(node_features.shape[0])

    if edge_index.size == 0:
        edge_index = edge_index.reshape(2, 0)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")

    edge_count = int(edge_index.shape[1])
    if edge_count and (
        edge_index.min() < 0 or edge_index.max() >= node_count
    ):
        raise ValueError("edge_index contains an out-of-range node reference")
    if edge_attr.shape[0] < edge_count:
        raise ValueError("edge_attr has fewer rows than edge_index has edges")

    return node_features, edge_index, edge_attr, node_count, edge_count


def _segment_softmax(
    scores: np.ndarray, targets: np.ndarray, node_count: int
) -> np.ndarray:
    """Softmax within each node's segment, independently per head.

    ``scores`` is ``(heads, M)`` and ``targets`` is ``(M,)``. Every node has at
    least its own self-attention entry, so no segment is ever empty.
    """
    head_count, item_count = scores.shape
    if item_count == 0:
        return scores

    out = np.zeros((head_count, item_count), dtype=float)
    order = np.argsort(targets, kind="stable")
    sorted_targets = targets[order]
    sorted_scores = scores[:, order]

    boundaries = np.flatnonzero(
        np.concatenate(([True], sorted_targets[1:] != sorted_targets[:-1]))
    )
    ends = np.concatenate((boundaries[1:], [item_count]))

    for start, end in zip(boundaries, ends):
        block = sorted_scores[:, start:end]
        exponent = np.exp(block - block.max(axis=1, keepdims=True))
        out[:, order[start:end]] = exponent / np.maximum(
            exponent.sum(axis=1, keepdims=True), 1e-300
        )
    return out


class GraphAttentionBank:
    """Deterministic multi-head graph attention implemented in NumPy.

    Inference only -- the weights come from :func:`train_gat_model`. Given the
    same input this always produces byte-identical output, which the project
    needs for repeatable demo and judge runs.
    """

    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int = 16,
        heads: int = 4,
        seed: int = 17,
        node_mean: np.ndarray | None = None,
        node_std: np.ndarray | None = None,
    ):
        self.input_dim = int(input_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.seed = int(seed)
        self.node_mean = (
            np.asarray(node_mean, dtype=float)
            if node_mean is not None
            else np.zeros(self.input_dim, dtype=float)
        )
        self.node_std = (
            np.asarray(node_std, dtype=float)
            if node_std is not None
            else np.ones(self.input_dim, dtype=float)
        )
        if self.node_mean.shape[0] != self.input_dim or self.node_std.shape[0] != self.input_dim:
            raise ValueError("node_mean/node_std length must match input_dim")

        d = self.hidden_dim
        rng = np.random.default_rng(seed)
        self.W_src = rng.normal(0.0, 0.35, size=(self.heads, self.input_dim, d))
        self.a_src = rng.normal(0.0, 0.35, size=(self.heads, d))
        self.a_tgt = rng.normal(0.0, 0.35, size=(self.heads, d))
        self.a_edge = rng.normal(0.0, 0.35, size=(self.heads, self.edge_dim))
        self.W_out = rng.normal(0.0, 0.10, size=(self.heads * d,))
        self.b_out = np.zeros(1, dtype=float)

    @staticmethod
    def _leaky_relu(values: np.ndarray, slope: float = 0.2) -> np.ndarray:
        return np.where(values >= 0.0, values, slope * values)

    @staticmethod
    def _elu(values: np.ndarray) -> np.ndarray:
        return np.where(values >= 0.0, values, np.expm1(np.minimum(values, 0.0)))

    def attention_coefficients(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(alpha, segment_targets, projected_states)``.

        ``alpha`` is ``(heads, node_count + edge_count)`` and covers the self
        entries first, then one entry per edge.
        """
        node_features, edge_index, edge_attr, node_count, edge_count = _split_graph_inputs(
            node_features, edge_index, edge_attr
        )
        normalized = (node_features - self.node_mean) / self.node_std
        states = np.einsum("ni,hid->nhd", normalized, self.W_src)  # (N, heads, d)

        if edge_count == 0:
            self_score = self._leaky_relu(
                (states * self.a_src).sum(axis=-1)
            )  # (N, heads)
            scores = self_score.T  # (heads, N)
            targets = np.arange(node_count, dtype=int)
            return scores, targets, states

        source = edge_index[0]
        target = edge_index[1]
        edge_features = _scale_edges(edge_attr, self.edge_dim)

        self_score = self._leaky_relu((states * self.a_src).sum(axis=-1))  # (N, heads)
        node_tgt_score = (states * self.a_tgt).sum(axis=-1)  # (N, heads)
        edge_score = edge_features @ self.a_edge.T  # (E, heads)

        edge_logits = self._leaky_relu(
            self_score[source] + node_tgt_score[target] + edge_score
        )  # (E, heads)

        scores = np.concatenate([self_score.T, edge_logits.T], axis=1)
        targets = np.concatenate(
            [np.arange(node_count, dtype=int), target.astype(int)]
        )
        return scores, targets, states

    def forward(
        self, node_features: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray
    ) -> np.ndarray:
        node_features, edge_index, edge_attr, node_count, edge_count = _split_graph_inputs(
            node_features, edge_index, edge_attr
        )
        if node_count == 0:
            return np.zeros(0, dtype=float)

        scores, targets, states = self.attention_coefficients(
            node_features, edge_index, edge_attr
        )
        alpha = _segment_softmax(scores, targets, node_count)  # (heads, N+E)

        if edge_count:
            source = edge_index[0]
            message_states = np.concatenate([states, states[source]], axis=0)
        else:
            message_states = states

        # (N+E, heads) -> (N+E, heads, 1) so it broadcasts over the hidden dim.
        weight = alpha.T[:, :, None]
        aggregated = np.zeros((node_count, self.heads, self.hidden_dim), dtype=float)
        np.add.at(aggregated, targets, message_states * weight)

        hidden = self._elu(aggregated.reshape(node_count, self.heads * self.hidden_dim))
        return (hidden @ self.W_out + self.b_out[0]).reshape(-1)

    # ── serialization ────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "edge_dim": self.edge_dim,
            "hidden_dim": self.hidden_dim,
            "heads": self.heads,
            "seed": self.seed,
            "node_mean": self.node_mean.tolist(),
            "node_std": self.node_std.tolist(),
            "W_src": self.W_src.tolist(),
            "a_src": self.a_src.tolist(),
            "a_tgt": self.a_tgt.tolist(),
            "a_edge": self.a_edge.tolist(),
            "W_out": self.W_out.tolist(),
            "b_out": self.b_out.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphAttentionBank":
        bank = cls(
            input_dim=int(payload["input_dim"]),
            edge_dim=int(payload["edge_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            heads=int(payload["heads"]),
            seed=int(payload["seed"]),
            node_mean=np.asarray(payload["node_mean"], dtype=float),
            node_std=np.asarray(payload["node_std"], dtype=float),
        )
        bank.W_src = np.asarray(payload["W_src"], dtype=float)
        bank.a_src = np.asarray(payload["a_src"], dtype=float)
        bank.a_tgt = np.asarray(payload["a_tgt"], dtype=float)
        bank.a_edge = np.asarray(payload["a_edge"], dtype=float)
        bank.W_out = np.asarray(payload["W_out"], dtype=float)
        bank.b_out = np.asarray(payload["b_out"], dtype=float)
        return bank


def _normalization_from(sample_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = sample_features.mean(axis=0)
    std = sample_features.std(axis=0) + 1e-6
    return mean.astype(float), std.astype(float)


# ── Training (PyTorch autograd, no torch_geometric) ──────────────────────────

if TORCH_AVAILABLE:

    class TorchGraphAttentionBank(nn.Module):
        """Autograd mirror of :class:`GraphAttentionBank`.

        The parameter names and shapes match the NumPy bank exactly so training
        results can be exported straight into the JSON artifact.
        """

        def __init__(
            self,
            input_dim: int,
            edge_dim: int,
            hidden_dim: int = 16,
            heads: int = 4,
            seed: int = 17,
        ):
            super().__init__()
            generator = torch.Generator().manual_seed(int(seed))
            d = int(hidden_dim)
            self.heads = int(heads)
            self.hidden_dim = d
            self.edge_dim = int(edge_dim)

            def draw(*shape: int, scale: float) -> "torch.Tensor":
                return torch.randn(*shape, generator=generator) * scale

            self.W_src = nn.Parameter(draw(self.heads, int(input_dim), d, scale=0.35))
            self.a_src = nn.Parameter(draw(self.heads, d, scale=0.35))
            self.a_tgt = nn.Parameter(draw(self.heads, d, scale=0.35))
            self.a_edge = nn.Parameter(draw(self.heads, int(edge_dim), scale=0.35))
            self.W_out = nn.Parameter(draw(self.heads * d, scale=0.10))
            self.b_out = nn.Parameter(torch.zeros(1))

        def forward(
            self,
            normalized: "torch.Tensor",
            edge_index: "torch.Tensor",
            edge_scaled: "torch.Tensor",
        ) -> "torch.Tensor":
            node_count = int(normalized.shape[0])
            states = torch.einsum("ni,hid->nhd", normalized, self.W_src)

            self_score = nn.functional.leaky_relu(
                (states * self.a_src).sum(-1), 0.2
            )  # (N, heads)

            if edge_index.shape[1] == 0:
                targets = torch.arange(node_count, dtype=torch.long)
                alpha = torch.ones(
                    self.heads, node_count, dtype=normalized.dtype
                )
                message_states = states
            else:
                source = edge_index[0]
                target = edge_index[1]
                node_tgt_score = (states * self.a_tgt).sum(-1)
                edge_score = edge_scaled @ self.a_edge.T
                edge_logits = nn.functional.leaky_relu(
                    self_score[source] + node_tgt_score[target] + edge_score, 0.2
                )
                targets = torch.cat(
                    [torch.arange(node_count, dtype=torch.long), target]
                )
                message_states = torch.cat([states, states[source]], dim=0)

                scores = torch.cat([self_score, edge_logits], dim=0).T  # (heads, M)

                # Flatten (head, node) into a single segment axis so the
                # segment reductions can use the 1D index index_add_ requires.
                head_offsets = (
                    torch.arange(self.heads, dtype=torch.long).unsqueeze(1) * node_count
                )
                flat_targets = (head_offsets + targets).reshape(-1)  # (heads * M,)

                seg_max = torch.full(
                    (self.heads * node_count,),
                    float("-inf"),
                    dtype=scores.dtype,
                ).scatter_reduce_(0, flat_targets, scores.reshape(-1), "amax")
                exponent = torch.exp(scores.reshape(-1) - seg_max[flat_targets])
                seg_sum = torch.zeros(
                    self.heads * node_count, dtype=scores.dtype
                ).index_add_(0, flat_targets, exponent)
                alpha = (exponent / seg_sum[flat_targets].clamp_min(1e-12)).reshape(
                    self.heads, -1
                )

            weighted = message_states * alpha.T.unsqueeze(-1)
            aggregated = torch.zeros(
                node_count, self.heads, self.hidden_dim, dtype=normalized.dtype
            ).index_add_(0, targets, weighted)

            hidden = nn.functional.elu(
                aggregated.reshape(node_count, self.heads * self.hidden_dim)
            )
            return (hidden @ self.W_out + self.b_out).reshape(-1)


def train_gat_model(
    sample_count: int = 320,
    hidden_dim: int = 16,
    heads: int = 4,
    epochs: int = 120,
    learning_rate: float = 0.01,
    seed: int = 17,
    save: bool = True,
) -> "GraphAttentionBank | None":
    """Gradient-train the attention bank and export it for NumPy inference.

    Requires only PyTorch (``torch`` is in ``requirements.txt``); unlike the
    previous implementation this no longer needs ``torch_geometric``.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required to train the GAT. The runtime inference path "
            "reads the pre-trained JSON artifact and needs only NumPy."
        )

    from .synthetic_data import generate_graph_snapshots

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    graphs = generate_graph_snapshots(sample_count=sample_count, seed=seed)
    graphs = [g for g in graphs if g["node_features"].shape[0] > 0]
    if not graphs:
        raise RuntimeError("no usable graph snapshots were generated")

    sample_features = np.vstack([g["node_features"] for g in graphs])
    node_mean, node_std = _normalization_from(sample_features)
    input_dim = int(sample_features.shape[1])
    edge_dim = int(graphs[0]["edge_attr"].shape[1]) if graphs[0]["edge_attr"].size else 4

    model = TorchGraphAttentionBank(input_dim, edge_dim, hidden_dim, heads, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    mean_t = torch.tensor(node_mean, dtype=torch.float32)
    std_t = torch.tensor(node_std, dtype=torch.float32)
    scale_t = torch.tensor(EDGE_FEATURE_SCALE, dtype=torch.float32)

    def to_tensors(graph: dict[str, np.ndarray]):
        x = (graph["node_features"] - node_mean) / node_std
        edge_index = torch.tensor(
            graph["edge_index"].reshape(2, -1).astype(np.int64), dtype=torch.long
        )
        edge_scaled = _scale_edges(graph["edge_attr"], edge_dim)
        y = torch.tensor(
            np.asarray(graph["targets"], dtype=np.float32).reshape(-1),
            dtype=torch.float32,
        )
        return (
            torch.tensor(x, dtype=torch.float32),
            edge_index,
            torch.tensor(edge_scaled, dtype=torch.float32),
            y,
        )

    cache = [to_tensors(g) for g in graphs]

    model.train()
    for _ in range(int(epochs)):
        order = rng.permutation(len(cache))
        for index in order:
            x, edge_index, edge_scaled, y = cache[int(index)]
            optimizer.zero_grad()
            loss = loss_fn(model(x, edge_index, edge_scaled), y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        bank = GraphAttentionBank(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            seed=seed,
            node_mean=node_mean,
            node_std=node_std,
        )
        bank.W_src = model.W_src.detach().cpu().numpy().astype(float)
        bank.a_src = model.a_src.detach().cpu().numpy().astype(float)
        bank.a_tgt = model.a_tgt.detach().cpu().numpy().astype(float)
        bank.a_edge = model.a_edge.detach().cpu().numpy().astype(float)
        bank.W_out = model.W_out.detach().cpu().numpy().astype(float)
        bank.b_out = model.b_out.detach().cpu().numpy().astype(float)

    del mean_t, std_t, scale_t

    if save:
        save_artifact(
            GAT_META,
            {
                "version": ARTIFACT_VERSION,
                "trained_from": "synthetic_graph_snapshots",
                "trainer": "torch_autograd",
                "sample_count": int(sample_count),
                "epochs": int(epochs),
                "hidden_dim": int(hidden_dim),
                "heads": int(heads),
                "learning_rate": float(learning_rate),
                "seed": int(seed),
                "note": (
                    "Multi-head graph attention with self-attention inside each "
                    "node's softmax. Trained with PyTorch autograd; inference is "
                    "deterministic NumPy."
                ),
                "model": bank.to_dict(),
            },
        )
        logger.info("Trained GAT attention bank and cached it in %s", GAT_META)

    return bank


def load_or_build_attention_bank() -> GraphAttentionBank | None:
    """Load the cached attention bank, training and saving it if absent."""
    try:
        payload = load_artifact(GAT_META)
        if (
            payload
            and payload.get("version") == ARTIFACT_VERSION
            and "model" in payload
        ):
            return GraphAttentionBank.from_dict(payload["model"])
        logger.warning(
            "GAT artifact missing or stale (version %s, expected %s); retraining.",
            (payload or {}).get("version"),
            ARTIFACT_VERSION,
        )
    except Exception as error:
        logger.warning("GAT artifact load failed, rebuilding: %s", error)

    if TORCH_AVAILABLE:
        try:
            bank = train_gat_model()
            if bank is not None:
                return bank
        except Exception as error:
            logger.warning("GAT training failed: %s", error)

    logger.error(
        "No trained GAT attention bank is available. Refusing to fall back to "
        "CascadeGNN, because that would make the GAT metrics identical to the "
        "GNN metrics by construction."
    )
    return None


class CascadeGAT:
    """Graph attention cascade ranker.

    If no attention bank can be loaded or trained, ``predict`` raises instead of
    quietly aliasing the GNN, so a degenerate result can never again be reported
    as an independent model. ``is_degenerate`` stays available for callers that
    want to check.
    """

    def __init__(self):
        self.attention: GraphAttentionBank | None = load_or_build_attention_bank()
        self._fallback: CascadeGNN | None = (
            CascadeGNN() if self.attention is None else None
        )
        if self.attention is not None:
            logger.info(
                "CascadeGAT active: %d-head attention, hidden dim %d, trained=%s",
                self.attention.heads,
                self.attention.hidden_dim,
                not self.is_degenerate,
            )

    @property
    def is_degenerate(self) -> bool:
        """True when this instance is not an independent model."""
        return self.attention is None

    def _maneuver_geometry(
        self, node_features, edge_index, edge_attr, maneuver_probability
    ):
        """Shared post-processing: cascade depth and the RSW burn recommendation."""
        summary_features = build_graph_summary_features(
            node_features, edge_index, edge_attr
        )
        if summary_features.shape[0]:
            influence = summary_features[:, 9]
            local_risk = summary_features[:, 4]
            speed_bias = np.clip(summary_features[:, 1], 0.0, 1.0)
        else:
            influence = local_risk = speed_bias = np.zeros(0, dtype=float)

        cascade_depth = np.clip(
            np.ceil(maneuver_probability * 4.0 + influence * 1.5), 0.0, 4.0
        ).astype(int)

        delta_v = np.zeros((maneuver_probability.shape[0], 3), dtype=float)
        if maneuver_probability.shape[0]:
            delta_v[:, 1] = np.clip(
                0.05
                + maneuver_probability * 1.35
                + local_risk / 12.0
                + speed_bias * 0.2,
                0.05,
                5.0,
            )
            delta_v[:, 0] = np.clip((local_risk - 0.5) * 0.25, -1.0, 1.0)
            delta_v[:, 2] = np.clip((influence - 0.35) * 0.22, -1.0, 1.0)

        return {
            "delta_v_rsw_ms": delta_v,
            "cascade_depth": cascade_depth,
            "graph_influence": influence,
        }

    def predict(self, node_features, edge_index, edge_attr):
        node_features = np.asarray(node_features, dtype=float)
        edge_index = np.asarray(edge_index, dtype=int)
        edge_attr = np.asarray(edge_attr, dtype=float)

        if node_features.ndim != 2:
            raise ValueError("node_features must be a 2D array")

        if self.attention is None:
            raise RuntimeError(
                "CascadeGAT has no attention bank. Run "
                "`python -m app.ml.train_models --gat` to train one; it will not "
                "silently reuse CascadeGNN."
            )

        if node_features.shape[0] == 0:
            return {
                "maneuver_probability": np.zeros(0, dtype=float),
                "delta_v_rsw_ms": np.zeros((0, 3), dtype=float),
                "cascade_depth": np.zeros(0, dtype=int),
                "graph_influence": np.zeros(0, dtype=float),
            }

        logits = self.attention.forward(node_features, edge_index, edge_attr)
        maneuver_probability = sigmoid(logits).reshape(-1)
        return {
            "maneuver_probability": np.asarray(maneuver_probability, dtype=float),
            **self._maneuver_geometry(
                node_features, edge_index, edge_attr, maneuver_probability
            ),
        }


if __name__ == "__main__":  # pragma: no cover - manual retraining entry point
    logging.basicConfig(level=logging.INFO)
    trained = train_gat_model()
    if trained is not None:
        logger.info("Saved weights to %s", artifact_path(GAT_META))
