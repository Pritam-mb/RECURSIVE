from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .artifacts import load_artifact, save_artifact
from .synthetic_data import generate_trajectory_dataset

logger = logging.getLogger(__name__)

TRAJECTORY_ARTIFACT = "trajectory_model.json"
# Total buffered position records that trigger a real-data LSTM retrain.
LSTM_RETRAIN_THRESHOLD = 500


class RingBuffer:
    """
    Fixed-size circular buffer for satellite position history.
    Stores (timestamp, x, y, z) tuples.
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def push(self, timestamp: str, x: float, y: float, z: float):
        self._buffer.append((timestamp, x, y, z))

    def get_positions(self) -> np.ndarray:
        if len(self._buffer) == 0:
            return np.array([]).reshape(0, 3)
        return np.array([(x, y, z) for _, x, y, z in self._buffer], dtype=float)

    def get_all(self) -> list[tuple]:
        return list(self._buffer)

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.capacity


@dataclass
class RecurrentTrajectoryModel:
    hidden_size: int
    sequence_length: int
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    Wx: np.ndarray
    Wh: np.ndarray
    b: np.ndarray
    Wy: np.ndarray
    by: np.ndarray

    def _normalize_input(self, sequence: np.ndarray) -> np.ndarray:
        return (sequence - self.input_mean) / self.input_std

    def _denormalize_output(self, values: np.ndarray) -> np.ndarray:
        return values * self.output_std + self.output_mean

    def _forward(self, sequence: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        h = np.zeros(self.hidden_size, dtype=float)
        cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for xt in sequence:
            z = xt @ self.Wx + h @ self.Wh + self.b
            h = np.tanh(z)
            cache.append((xt, h.copy(), z.copy()))

        y_norm = h @ self.Wy + self.by
        return y_norm, cache

    def predict_next(self, sequence: np.ndarray) -> np.ndarray:
        sequence = np.asarray(sequence, dtype=float)
        if sequence.ndim != 2 or sequence.shape[0] == 0:
            raise ValueError("sequence must be a 2D array with at least one step")
        if sequence.shape[0] > self.sequence_length:
            sequence = sequence[-self.sequence_length :]

        if sequence.shape[0] < self.sequence_length:
            pad_count = self.sequence_length - sequence.shape[0]
            pad = np.repeat(sequence[:1], pad_count, axis=0)
            sequence = np.vstack([pad, sequence])

        normalized = self._normalize_input(sequence)
        y_norm, _ = self._forward(normalized)
        return self._denormalize_output(y_norm)

    def predict_steps(self, sequence: np.ndarray, steps_ahead: int = 10) -> list[dict[str, float]]:
        history = np.asarray(sequence, dtype=float)
        if history.ndim != 2 or history.shape[0] == 0:
            return []

        predictions: list[dict[str, float]] = []
        cursor = history.copy()
        for step in range(1, steps_ahead + 1):
            next_position = self.predict_next(cursor)
            predictions.append(
                {
                    "step": step,
                    "x": round(float(next_position[0]), 3),
                    "y": round(float(next_position[1]), 3),
                    "z": round(float(next_position[2]), 3),
                }
            )
            cursor = np.vstack([cursor, next_position])
            if cursor.shape[0] > self.sequence_length:
                cursor = cursor[-self.sequence_length :]
        return predictions

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": int(self.hidden_size),
            "sequence_length": int(self.sequence_length),
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "output_mean": self.output_mean.tolist(),
            "output_std": self.output_std.tolist(),
            "Wx": self.Wx.tolist(),
            "Wh": self.Wh.tolist(),
            "b": self.b.tolist(),
            "Wy": self.Wy.tolist(),
            "by": self.by.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RecurrentTrajectoryModel":
        return cls(
            hidden_size=int(payload["hidden_size"]),
            sequence_length=int(payload["sequence_length"]),
            input_mean=np.asarray(payload["input_mean"], dtype=float),
            input_std=np.asarray(payload["input_std"], dtype=float),
            output_mean=np.asarray(payload["output_mean"], dtype=float),
            output_std=np.asarray(payload["output_std"], dtype=float),
            Wx=np.asarray(payload["Wx"], dtype=float),
            Wh=np.asarray(payload["Wh"], dtype=float),
            b=np.asarray(payload["b"], dtype=float),
            Wy=np.asarray(payload["Wy"], dtype=float),
            by=np.asarray(payload["by"], dtype=float),
        )


def _tanh_derivative(value: np.ndarray) -> np.ndarray:
    return 1.0 - np.square(value)


def train_trajectory_model(
    sample_count: int = 2048,
    history_length: int = 8,
    hidden_size: int = 12,
    epochs: int = 18,
    learning_rate: float = 0.003,
    seed: int = 11,
) -> RecurrentTrajectoryModel:
    sequences, targets, stats = generate_trajectory_dataset(
        sample_count=sample_count,
        history_length=history_length,
        seed=seed,
    )

    input_mean = sequences.reshape(-1, 3).mean(axis=0)
    input_std = sequences.reshape(-1, 3).std(axis=0) + 1e-6
    output_mean = targets.mean(axis=0)
    output_std = targets.std(axis=0) + 1e-6

    normalized_sequences = (sequences - input_mean) / input_std
    normalized_targets = (targets - output_mean) / output_std

    rng = np.random.default_rng(seed + 1)
    Wx = rng.normal(0.0, 0.1, size=(3, hidden_size))
    Wh = rng.normal(0.0, 0.1, size=(hidden_size, hidden_size))
    b = np.zeros(hidden_size, dtype=float)
    Wy = rng.normal(0.0, 0.1, size=(hidden_size, 3))
    by = np.zeros(3, dtype=float)

    for _ in range(epochs):
        indices = rng.permutation(sample_count)
        for index in indices:
            sequence = normalized_sequences[index]
            target = normalized_targets[index]

            hidden_states: list[np.ndarray] = []
            pre_activations: list[np.ndarray] = []
            previous_hidden = np.zeros(hidden_size, dtype=float)
            for xt in sequence:
                pre = xt @ Wx + previous_hidden @ Wh + b
                hidden = np.tanh(pre)
                pre_activations.append(pre)
                hidden_states.append(hidden)
                previous_hidden = hidden

            predicted = hidden_states[-1] @ Wy + by
            error = predicted - target

            grad_Wy = np.outer(hidden_states[-1], error)
            grad_by = error
            grad_Wx = np.zeros_like(Wx)
            grad_Wh = np.zeros_like(Wh)
            grad_b = np.zeros_like(b)

            hidden_grad = error @ Wy.T
            for step in reversed(range(history_length)):
                hidden = hidden_states[step]
                previous = hidden_states[step - 1] if step > 0 else np.zeros(hidden_size, dtype=float)
                dz = hidden_grad * _tanh_derivative(hidden)
                grad_Wx += np.outer(sequence[step], dz)
                grad_Wh += np.outer(previous, dz)
                grad_b += dz
                hidden_grad = dz @ Wh.T

            clip_value = 5.0
            for grad in (grad_Wx, grad_Wh, grad_b, grad_Wy, grad_by):
                np.clip(grad, -clip_value, clip_value, out=grad)

            Wx -= learning_rate * grad_Wx
            Wh -= learning_rate * grad_Wh
            b -= learning_rate * grad_b
            Wy -= learning_rate * grad_Wy
            by -= learning_rate * grad_by

    model = RecurrentTrajectoryModel(
        hidden_size=hidden_size,
        sequence_length=history_length,
        input_mean=np.asarray(input_mean, dtype=float),
        input_std=np.asarray(input_std, dtype=float),
        output_mean=np.asarray(output_mean, dtype=float),
        output_std=np.asarray(output_std, dtype=float),
        Wx=Wx,
        Wh=Wh,
        b=b,
        Wy=Wy,
        by=by,
    )
    save_artifact(
        TRAJECTORY_ARTIFACT,
        {
            "trained_from": "synthetic_trajectory_arcs",
            "sample_count": sample_count,
            "epochs": epochs,
            "stats": stats,
            "model": model.to_dict(),
        },
    )
    logger.info("Trained trajectory model for %d-step histories", history_length)
    return model


def load_or_train_trajectory_model(retrain: bool = False) -> RecurrentTrajectoryModel:
    if not retrain:
        payload = load_artifact(TRAJECTORY_ARTIFACT)
        if payload and "model" in payload:
            return RecurrentTrajectoryModel.from_dict(payload["model"])

    return train_trajectory_model()


class LSTMPredictor:
    """
    EXPERIMENTAL trajectory forecaster: a compact numpy tanh RNN (not an
    LSTM) trained on synthetic circular arcs. On held-out arcs it is worse
    than both persistence and linear extrapolation (see
    artifacts/risk_model_card.json -> trajectory_model), so nothing in the
    alert / Pc / manoeuvre path uses it; SGP4 is the propagator of record.
    Only constructed when ENABLE_EXTENDED_PIPELINE=1.
    """

    def __init__(self, model_path: str = None):
        self.model: RecurrentTrajectoryModel | None = None
        self._buffers: dict[int, RingBuffer] = {}
        self.model_path = model_path
        # True once train_on_buffer_data() has succeeded in this process.
        self.trained_on_real_data = False
        self._load_or_train_model(model_path)

    def _load_or_train_model(self, path: str | None):
        try:
            if path:
                payload = load_artifact(path)
            else:
                payload = load_artifact(TRAJECTORY_ARTIFACT)

            if payload and "model" in payload:
                self.model = RecurrentTrajectoryModel.from_dict(payload["model"])
                logger.info("Trajectory model loaded from %s", path or TRAJECTORY_ARTIFACT)
                return

            self.model = train_trajectory_model()
        except Exception as error:
            logger.warning("Trajectory model bootstrap failed, retraining: %s", error)
            self.model = train_trajectory_model()

    def stats(self) -> dict[str, Any]:
        """Buffer/training summary for /api/ml/status."""
        import os

        buffers = list(self._buffers.values())
        records = sum(b.size for b in buffers)
        tracked = sum(1 for b in buffers if b.size > 0)
        weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lstm_model.pt")
        return {
            "trained": bool(self.trained_on_real_data or os.path.exists(weights)),
            "satellites_tracked": int(tracked),
            "buffer_records": int(records),
            "buffer_threshold": int(LSTM_RETRAIN_THRESHOLD),
        }

    def get_buffer(self, norad_id: int) -> RingBuffer:
        if norad_id not in self._buffers:
            self._buffers[norad_id] = RingBuffer(capacity=120)
        return self._buffers[norad_id]

    def record(self, norad_id: int, timestamp: str, x: float, y: float, z: float,
               vx: float = 0.0, vy: float = 0.0, vz: float = 0.0):
        buf = self.get_buffer(norad_id)
        buf.push(timestamp, x, y, z)

        # ── Auto-retrain trigger: when enough real data accumulates ────────
        # Count total entries across all buffers
        total_entries = sum(b.size for b in self._buffers.values())
        if total_entries > LSTM_RETRAIN_THRESHOLD and not getattr(self, "_training_active", False):
            self._training_active = True
            t = threading.Thread(target=self._background_train, daemon=True)
            t.start()

    def _background_train(self):
        """Background thread wrapper for train_on_buffer_data."""
        try:
            self.train_on_buffer_data()
        except Exception as exc:
            logger.warning("Background LSTM retraining failed: %s", exc)
        finally:
            self._training_active = False

    def train_on_buffer_data(self) -> bool:
        """
        Train a PyTorch LSTM on real position sequences from ring buffers.

        Called automatically when total buffer entries exceed 500.
        Uses a 2-layer LSTM with hidden size 64:
          - input_size  = 6  (x, y, z, vx, vy, vz — but only x/y/z stored)
            Note: velocities are approximated from consecutive positions
          - hidden_size = 64
          - num_layers  = 2
          - output_size = 3  (predict next x, y, z)

        Positions are normalised by dividing by 7000 km (typical LEO orbit radius).
        Trains for 20 epochs with Adam lr=1e-3.
        Saves state dict to lstm_model.pt in the artifacts directory.

        Returns True on success, False on failure.
        """
        WINDOW = 10
        NORM = 7000.0  # km — typical LEO orbit radius for normalisation

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.warning("PyTorch not installed — LSTM retraining skipped")
            return False

        try:
            # ── Collect sliding-window samples from ring buffers ─────────
            X_list, y_list = [], []
            n_sats = 0
            for norad_id, buf in self._buffers.items():
                positions = buf.get_positions()  # shape (N, 3)
                if positions.shape[0] < WINDOW + 1:
                    continue
                n_sats += 1
                normed = positions / NORM
                for k in range(len(normed) - WINDOW):
                    seq = normed[k : k + WINDOW]       # (WINDOW, 3)
                    target = normed[k + WINDOW]         # (3,)
                    # Approximate velocities from differences (normalised km/step)
                    vels = np.diff(seq, axis=0)         # (WINDOW-1, 3)
                    # Pad first velocity
                    vels = np.vstack([vels[:1], vels])  # (WINDOW, 3)
                    inp = np.concatenate([seq, vels], axis=1)  # (WINDOW, 6)
                    X_list.append(inp)
                    y_list.append(target)

            if len(X_list) < 10:
                logger.info("Not enough buffer data for LSTM retraining (%d sequences)", len(X_list))
                return False

            X = torch.tensor(np.array(X_list, dtype=np.float32))   # (N, WINDOW, 6)
            y = torch.tensor(np.array(y_list, dtype=np.float32))   # (N, 3)

            # ── Define LSTM model ─────────────────────────────────────────
            class OrbitalLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=6, hidden_size=64, num_layers=2,
                        batch_first=True, dropout=0.1
                    )
                    self.fc = nn.Linear(64, 3)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])  # last timestep

            torch_model = OrbitalLSTM()
            optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            # ── Train for 20 epochs ───────────────────────────────────────
            torch_model.train()
            for epoch in range(20):
                perm = torch.randperm(X.shape[0])
                epoch_loss = 0.0
                for batch_start in range(0, X.shape[0], 64):
                    idx = perm[batch_start : batch_start + 64]
                    xb, yb = X[idx], y[idx]
                    optimizer.zero_grad()
                    pred = torch_model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(torch_model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item() * len(xb)

            # ── Save state dict ───────────────────────────────────────────
            import os
            model_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(model_dir, "lstm_model.pt")
            torch.save(torch_model.state_dict(), save_path)
            logger.info(
                "LSTM retrained on %d sequences from %d satellites (saved to %s)",
                len(X_list), n_sats, save_path,
            )
            self.trained_on_real_data = True
            return True

        except Exception as exc:
            logger.warning("train_on_buffer_data failed: %s", exc)
            return False

    def predict(self, norad_id: int, steps_ahead: int = 10) -> list[dict]:
        buf = self.get_buffer(norad_id)
        positions = buf.get_positions()

        if len(positions) < 2:
            return []

        # ── Primary path: use trained RecurrentTrajectoryModel ───────────
        if self.model is not None and len(positions) >= 2:
            try:
                sequence = positions[-self.model.sequence_length :]
                if len(sequence) < self.model.sequence_length:
                    pad_count = self.model.sequence_length - len(sequence)
                    sequence = np.vstack([np.repeat(sequence[:1], pad_count, axis=0), sequence])
                return self.model.predict_steps(sequence, steps_ahead=steps_ahead)
            except Exception as model_err:
                logger.warning("RecurrentTrajectoryModel prediction failed: %s", model_err)

        # ── FALLBACK: linear extrapolation ────────────────────────────────
        # Labelled explicitly as fallback — has no predictive advantage over SGP4.
        # Only used when model is None or raises.
        logger.debug(
            "FALLBACK: LSTM predict for NORAD %d using linear extrapolation "
            "(model not loaded or insufficient data)",
            norad_id,
        )
        p1 = positions[-2]
        p2 = positions[-1]
        velocity = p2 - p1

        predictions = []
        for i in range(1, steps_ahead + 1):
            future = p2 + velocity * i
            predictions.append(
                {
                    "step": i,
                    "x": round(float(future[0]), 3),
                    "y": round(float(future[1]), 3),
                    "z": round(float(future[2]), 3),
                }
            )
        return predictions

    def get_sequence(self, norad_id: int) -> np.ndarray | None:
        buf = self.get_buffer(norad_id)
        positions = buf.get_positions()
        if self.model is None or len(positions) < 2:
            return None

        sequence = positions[-self.model.sequence_length :]
        if len(sequence) < self.model.sequence_length:
            pad_count = self.model.sequence_length - len(sequence)
            sequence = np.vstack([np.repeat(sequence[:1], pad_count, axis=0), sequence])
        return sequence

    def predict_next_from_sequence(self, sequence: np.ndarray) -> np.ndarray | None:
        if self.model is None:
            return None
        try:
            return self.model.predict_next(sequence)
        except Exception:
            return None

    def fine_tune(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        epochs: int = 3,
        learning_rate: float = 0.001,
    ) -> float:
        if self.model is None:
            return 0.0

        sequences = np.asarray(sequences, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if sequences.ndim != 3 or targets.ndim != 2:
            return 0.0

        input_mean = self.model.input_mean
        input_std = self.model.input_std
        output_mean = self.model.output_mean
        output_std = self.model.output_std

        normalized_sequences = (sequences - input_mean) / input_std
        normalized_targets = (targets - output_mean) / output_std

        Wx = self.model.Wx
        Wh = self.model.Wh
        b = self.model.b
        Wy = self.model.Wy
        by = self.model.by

        rng = np.random.default_rng(42)
        last_loss = 0.0
        for _ in range(epochs):
            indices = rng.permutation(normalized_sequences.shape[0])
            for idx in indices:
                sequence = normalized_sequences[idx]
                target = normalized_targets[idx]

                hidden_states: list[np.ndarray] = []
                previous_hidden = np.zeros(self.model.hidden_size, dtype=float)
                for xt in sequence:
                    pre = xt @ Wx + previous_hidden @ Wh + b
                    hidden = np.tanh(pre)
                    hidden_states.append(hidden)
                    previous_hidden = hidden

                predicted = hidden_states[-1] @ Wy + by
                error = predicted - target
                last_loss = float(np.mean(error**2))

                grad_Wy = np.outer(hidden_states[-1], error)
                grad_by = error
                grad_Wx = np.zeros_like(Wx)
                grad_Wh = np.zeros_like(Wh)
                grad_b = np.zeros_like(b)

                hidden_grad = error @ Wy.T
                for step in reversed(range(sequence.shape[0])):
                    hidden = hidden_states[step]
                    previous = hidden_states[step - 1] if step > 0 else np.zeros(self.model.hidden_size, dtype=float)
                    dz = hidden_grad * _tanh_derivative(hidden)
                    grad_Wx += np.outer(sequence[step], dz)
                    grad_Wh += np.outer(previous, dz)
                    grad_b += dz
                    hidden_grad = dz @ Wh.T

                clip_value = 3.0
                for grad in (grad_Wx, grad_Wh, grad_b, grad_Wy, grad_by):
                    np.clip(grad, -clip_value, clip_value, out=grad)

                Wx -= learning_rate * grad_Wx
                Wh -= learning_rate * grad_Wh
                b -= learning_rate * grad_b
                Wy -= learning_rate * grad_Wy
                by -= learning_rate * grad_by

        self.model.Wx = Wx
        self.model.Wh = Wh
        self.model.b = b
        self.model.Wy = Wy
        self.model.by = by
        return last_loss

    def save_model(self):
        if self.model is None:
            return
        save_artifact(
            TRAJECTORY_ARTIFACT,
            {
                "trained_from": "shadow_mode_finetune",
                "sample_count": 0,
                "epochs": 0,
                "stats": {},
                "model": self.model.to_dict(),
            },
        )

    def predict_next_from_buffer(self, norad_id: int) -> np.ndarray | None:
        buf = self.get_buffer(norad_id)
        positions = buf.get_positions()

        if len(positions) < 2 or self.model is None:
            return None

        sequence = positions[-self.model.sequence_length :]
        if len(sequence) < self.model.sequence_length:
            pad_count = self.model.sequence_length - len(sequence)
            sequence = np.vstack([np.repeat(sequence[:1], pad_count, axis=0), sequence])

        try:
            return self.model.predict_next(sequence)
        except Exception:
            return None
