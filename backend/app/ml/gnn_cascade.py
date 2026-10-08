from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .artifacts import load_artifact, save_artifact
from .synthetic_data import build_graph_summary_features, generate_graph_dataset, sigmoid


logger = logging.getLogger(__name__)

GRAPH_ARTIFACT = "graph_model.json"


@dataclass
class GraphCascadeModel:
    input_mean: np.ndarray
    input_std: np.ndarray
    hidden_size: int
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        normalized = (features - self.input_mean) / self.input_std
        hidden = np.tanh(normalized @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        return sigmoid(logits).astype(float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "hidden_size": int(self.hidden_size),
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphCascadeModel":
        return cls(
            input_mean=np.asarray(payload["input_mean"], dtype=float),
            input_std=np.asarray(payload["input_std"], dtype=float),
            hidden_size=int(payload["hidden_size"]),
            W1=np.asarray(payload["W1"], dtype=float),
            b1=np.asarray(payload["b1"], dtype=float),
            W2=np.asarray(payload["W2"], dtype=float),
            b2=np.asarray(payload["b2"], dtype=float),
        )


def train_graph_model(
    sample_count: int = 4096,
    hidden_size: int = 16,
    epochs: int = 20,
    learning_rate: float = 0.01,
    seed: int = 17,
) -> GraphCascadeModel:
    features, probabilities, _, feature_names = generate_graph_dataset(sample_count=sample_count, seed=seed)
    input_mean = features.mean(axis=0)
    input_std = features.std(axis=0) + 1e-6
    normalized = (features - input_mean) / input_std

    rng = np.random.default_rng(seed + 1)
    W1 = rng.normal(0.0, 0.08, size=(normalized.shape[1], hidden_size))
    b1 = np.zeros(hidden_size, dtype=float)
    W2 = rng.normal(0.0, 0.08, size=(hidden_size, 1))
    b2 = np.zeros(1, dtype=float)

    for _ in range(epochs):
        indices = rng.permutation(normalized.shape[0])
        for index in indices:
            x = normalized[index]
            target = probabilities[index]

            hidden = np.tanh(x @ W1 + b1)
            logits = hidden @ W2 + b2
            prediction = sigmoid(logits)[0] if logits.ndim > 0 else float(logits)
            error = prediction - target

            # Compute gradients: ensure proper shapes for scalar loss
            d_logits = float(error * prediction * (1.0 - prediction))
            grad_W2 = np.outer(hidden, np.array([d_logits])).reshape(W2.shape)
            grad_b2 = np.array([d_logits]).reshape(b2.shape)
            grad_hidden = (d_logits * W2).flatten()
            grad_z1 = grad_hidden * (1.0 - hidden**2)
            grad_W1 = np.outer(x, grad_z1)
            grad_b1 = grad_z1.reshape(b1.shape)

            np.clip(grad_W1, -5.0, 5.0, out=grad_W1)
            np.clip(grad_W2, -5.0, 5.0, out=grad_W2)
            np.clip(grad_b1, -5.0, 5.0, out=grad_b1)
            np.clip(grad_b2, -5.0, 5.0, out=grad_b2)

            W1 -= learning_rate * grad_W1
            b1 -= learning_rate * grad_b1
            W2 -= learning_rate * grad_W2
            b2 -= learning_rate * grad_b2

    model = GraphCascadeModel(
        input_mean=input_mean.astype(float),
        input_std=input_std.astype(float),
        hidden_size=hidden_size,
        W1=W1.astype(float),
        b1=b1.astype(float),
        W2=W2.astype(float),
        b2=b2.astype(float),
    )
    save_artifact(
        GRAPH_ARTIFACT,
        {
            "trained_from": "synthetic_graph_snapshots",
            "sample_count": sample_count,
            "feature_names": feature_names,
            "model": model.to_dict(),
        },
    )
    logger.info("Trained graph cascade model with hidden size %d", hidden_size)
    return model


def load_or_train_graph_model(retrain: bool = False) -> GraphCascadeModel:
    if not retrain:
        payload = load_artifact(GRAPH_ARTIFACT)
        if payload and "model" in payload:
            return GraphCascadeModel.from_dict(payload["model"])

    return train_graph_model()


class CascadeGNN:
    """Deterministic graph scorer backed by a compact trained graph model."""

    def __init__(self):
        self.model: GraphCascadeModel | None = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        try:
            payload = load_artifact(GRAPH_ARTIFACT)
            if payload and "model" in payload:
                self.model = GraphCascadeModel.from_dict(payload["model"])
                logger.info("Graph model loaded from %s", GRAPH_ARTIFACT)
                return

            # Try to train, but if it fails, use None (fallback in predict)
            try:
                self.model = train_graph_model()
            except Exception as train_error:
                logger.warning("Graph model training failed, will use heuristic fallback: %s", train_error)
                self.model = None
        except Exception as error:
            logger.warning("Graph model bootstrap failed, will use heuristic fallback: %s", error)
            self.model = None

    def predict(self, node_features, edge_index, edge_attr):
        node_features = np.asarray(node_features, dtype=float)
        edge_index = np.asarray(edge_index, dtype=int)
        edge_attr = np.asarray(edge_attr, dtype=float)

        if node_features.ndim != 2:
            raise ValueError("node_features must be a 2D array")

        summary_features = build_graph_summary_features(node_features, edge_index, edge_attr)
        if summary_features.shape[0] == 0:
            return {
                "maneuver_probability": np.zeros(0, dtype=float),
                "delta_v_rsw_ms": np.zeros((0, 3), dtype=float),
                "cascade_depth": np.zeros(0, dtype=int),
                "graph_influence": np.zeros(0, dtype=float),
            }

        if self.model is not None:
            maneuver_probability = self.model.predict_probability(summary_features).reshape(-1)
        else:
            local_risk = summary_features[:, 4]
            influence = summary_features[:, 9]
            altitude_term = 1.0 - np.clip(summary_features[:, 2], 0.0, 1.0)
            logits = (1.2 * influence) + (0.4 * local_risk) + (0.15 * altitude_term)
            maneuver_probability = sigmoid(logits - 0.75).reshape(-1)

        influence = summary_features[:, 9]
        local_risk = summary_features[:, 4]
        speed_bias = np.clip(summary_features[:, 1], 0.0, 1.0)

        cascade_depth = np.clip(np.ceil(maneuver_probability * 4.0 + influence * 1.5), 0.0, 4.0).astype(int)

        delta_v = np.zeros((summary_features.shape[0], 3), dtype=float)
        delta_v[:, 1] = np.clip(
            0.05 + maneuver_probability * 1.35 + local_risk / 12.0 + speed_bias * 0.2,
            0.05,
            5.0,
        )
        delta_v[:, 0] = np.clip((local_risk - 0.5) * 0.25, -1.0, 1.0)
        delta_v[:, 2] = np.clip((influence - 0.35) * 0.22, -1.0, 1.0)

        return {
            "maneuver_probability": np.asarray(maneuver_probability, dtype=float).reshape(-1),
            "delta_v_rsw_ms": delta_v,
            "cascade_depth": cascade_depth,
            "graph_influence": influence,
        }
