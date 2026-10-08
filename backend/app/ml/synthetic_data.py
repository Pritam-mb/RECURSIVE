from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


R_EARTH_KM = 6371.0


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-values))


def _rotate_z(vec: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            c * vec[0] - s * vec[1],
            s * vec[0] + c * vec[1],
            vec[2],
        ],
        dtype=float,
    )


def _rotate_x(vec: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            vec[0],
            c * vec[1] - s * vec[2],
            s * vec[1] + c * vec[2],
        ],
        dtype=float,
    )


def generate_trajectory_dataset(
    sample_count: int = 2048,
    history_length: int = 8,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    sequences = np.zeros((sample_count, history_length, 3), dtype=float)
    targets = np.zeros((sample_count, 3), dtype=float)

    for idx in range(sample_count):
        radius_km = rng.uniform(6800.0, 14000.0)
        angular_rate = rng.uniform(0.00025, 0.0014) * rng.choice([-1.0, 1.0])
        inclination = rng.uniform(0.0, math.radians(95.0))
        raan = rng.uniform(0.0, 2.0 * math.pi)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        z_amplitude = rng.uniform(0.0, 1200.0)
        radial_drift = rng.uniform(-3.0, 3.0)
        step_seconds = rng.uniform(20.0, 60.0)
        perturb_scale = rng.uniform(0.0, 25.0)

        samples = []
        for step in range(history_length + 1):
            t = step * step_seconds
            theta = phase + angular_rate * t
            radius = radius_km + radial_drift * step
            orbital = np.array(
                [
                    radius * math.cos(theta),
                    radius * math.sin(theta),
                    z_amplitude * math.sin(theta * 0.63 + phase * 0.15),
                ],
                dtype=float,
            )
            orbital = _rotate_x(orbital, inclination)
            orbital = _rotate_z(orbital, raan)
            perturb = np.array(
                [
                    math.sin(0.031 * t + phase) * perturb_scale,
                    math.cos(0.027 * t + phase * 0.5) * perturb_scale * 0.7,
                    math.sin(0.021 * t + phase * 1.3) * perturb_scale * 0.4,
                ],
                dtype=float,
            )
            samples.append(orbital + perturb)

        sequences[idx] = np.asarray(samples[:-1], dtype=float)
        targets[idx] = np.asarray(samples[-1], dtype=float)

    stats = {
        "position_mean": float(np.mean(sequences)),
        "position_std": float(np.std(sequences) + 1e-6),
    }
    return sequences, targets, stats


def build_graph_summary_features(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> np.ndarray:
    node_features = np.asarray(node_features, dtype=float)
    edge_index = np.asarray(edge_index, dtype=int)
    edge_attr = np.asarray(edge_attr, dtype=float)

    node_count = node_features.shape[0]
    if node_count == 0:
        return np.zeros((0, 10), dtype=float)

    influence = np.zeros(node_count, dtype=float)
    degree = np.zeros(node_count, dtype=float)
    mean_p_collision = np.zeros(node_count, dtype=float)
    mean_miss_inv = np.zeros(node_count, dtype=float)
    mean_tca = np.zeros(node_count, dtype=float)
    mean_rel_speed = np.zeros(node_count, dtype=float)

    if edge_index.size and edge_index.ndim == 2 and edge_index.shape[0] == 2:
        counts = np.zeros(node_count, dtype=float)
        p_sum = np.zeros(node_count, dtype=float)
        miss_sum = np.zeros(node_count, dtype=float)
        tca_sum = np.zeros(node_count, dtype=float)
        rel_sum = np.zeros(node_count, dtype=float)

        for edge_idx in range(edge_index.shape[1]):
            src = int(edge_index[0, edge_idx])
            dst = int(edge_index[1, edge_idx])
            if src < 0 or dst < 0 or src >= node_count or dst >= node_count:
                continue

            miss_distance_km = float(edge_attr[edge_idx, 0]) if edge_attr.shape[0] > edge_idx else 0.0
            relative_velocity_kmh = float(edge_attr[edge_idx, 1]) if edge_attr.shape[0] > edge_idx else 0.0
            tca_minutes = float(edge_attr[edge_idx, 2]) if edge_attr.shape[0] > edge_idx else 0.0
            p_collision = float(edge_attr[edge_idx, 3]) if edge_attr.shape[0] > edge_idx else 0.0

            attention = (
                0.45 * p_collision
                + 0.25 * math.exp(-miss_distance_km / 120.0)
                + 0.15 * min(relative_velocity_kmh / 18000.0, 1.0)
                + 0.15 * math.exp(-max(tca_minutes, 0.0) / 180.0)
            )

            influence[src] += attention * 0.75
            influence[dst] += attention

            counts[src] += 1.0
            counts[dst] += 1.0
            p_sum[src] += p_collision
            p_sum[dst] += p_collision
            miss_sum[src] += 1.0 / (1.0 + miss_distance_km)
            miss_sum[dst] += 1.0 / (1.0 + miss_distance_km)
            tca_sum[src] += tca_minutes
            tca_sum[dst] += tca_minutes
            rel_sum[src] += relative_velocity_kmh
            rel_sum[dst] += relative_velocity_kmh

        degree = counts
        mean_p_collision = np.divide(p_sum, counts, out=np.zeros_like(p_sum), where=counts > 0)
        mean_miss_inv = np.divide(miss_sum, counts, out=np.zeros_like(miss_sum), where=counts > 0)
        mean_tca = np.divide(tca_sum, counts, out=np.zeros_like(tca_sum), where=counts > 0)
        mean_rel_speed = np.divide(rel_sum, counts, out=np.zeros_like(rel_sum), where=counts > 0)

    position_norm = np.linalg.norm(node_features[:, :3], axis=1) / 10000.0
    speed_norm = np.linalg.norm(node_features[:, 3:6], axis=1) / 30000.0
    altitude_norm = node_features[:, 7] / 2000.0 if node_features.shape[1] > 7 else np.zeros(node_count)
    agency_scaled = node_features[:, 8] / 10.0 if node_features.shape[1] > 8 else np.zeros(node_count)
    local_peak_cpi = node_features[:, 9] / 10.0 if node_features.shape[1] > 9 else np.zeros(node_count)

    return np.column_stack(
        [
            position_norm,
            speed_norm,
            altitude_norm,
            agency_scaled,
            local_peak_cpi,
            degree / max(node_count - 1, 1),
            mean_p_collision,
            mean_miss_inv,
            mean_tca / 180.0,
            influence,
            mean_rel_speed / 20000.0,
        ]
    ).astype(float)


def generate_graph_dataset(
    sample_count: int = 4096,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    node_rows = []
    probabilities = []
    depths = []

    for _ in range(sample_count):
        node_count = int(rng.integers(8, 32))
        positions = rng.normal(0.0, 1.0, size=(node_count, 3)) * rng.uniform(4500.0, 10000.0)
        velocities = rng.normal(0.0, 1.0, size=(node_count, 3)) * rng.uniform(1.0, 8.0)
        speed = np.linalg.norm(velocities, axis=1) * 3600.0
        altitude = np.linalg.norm(positions, axis=1) - R_EARTH_KM
        agency_id = rng.integers(0, 8, size=node_count).astype(float)
        local_peak_cpi = rng.uniform(0.0, 10.0, size=node_count)
        node_features = np.column_stack(
            [
                positions,
                velocities,
                speed,
                altitude,
                agency_id,
                local_peak_cpi,
            ]
        )

        edge_pairs = []
        edge_attrs = []
        for i in range(node_count):
            for j in range(i + 1, node_count):
                distance = float(np.linalg.norm(positions[i] - positions[j]))
                if distance > rng.uniform(750.0, 3200.0):
                    continue

                rel_speed = float(np.linalg.norm(velocities[i] - velocities[j]) * 3600.0)
                tca_minutes = float(rng.uniform(0.0, 720.0))
                p_collision = float(sigmoid((2.8 - distance / 180.0) + rel_speed / 18000.0 - tca_minutes / 220.0))
                edge_pairs.append((i, j))
                edge_attrs.append([distance, rel_speed, tca_minutes, p_collision])

        if edge_pairs:
            edge_index = np.array(edge_pairs, dtype=int).T
            edge_attr = np.array(edge_attrs, dtype=float)
        else:
            edge_index = np.zeros((2, 0), dtype=int)
            edge_attr = np.zeros((0, 4), dtype=float)

        summary = build_graph_summary_features(node_features, edge_index, edge_attr)
        influence = summary[:, 9]
        cpi_proxy = summary[:, 4] * 10.0
        p_collision_proxy = summary[:, 6]
        altitude_term = 1.0 - np.clip(summary[:, 2], 0.0, 1.0)
        degree_term = np.clip(summary[:, 5], 0.0, 1.0)

        logit = (
            -1.7
            + 2.8 * cpi_proxy / 10.0
            + 2.4 * influence
            + 0.75 * p_collision_proxy
            + 0.35 * altitude_term
            + 0.25 * degree_term
        )
        probability = sigmoid(logit)
        depth = np.clip(np.round(probability * 4.0 + influence * 1.4), 0, 4)

        node_rows.append(summary)
        probabilities.append(probability)
        depths.append(depth)

    return (
        np.vstack(node_rows).astype(float),
        np.concatenate(probabilities).astype(float),
        np.concatenate(depths).astype(int),
        [
            "position_norm",
            "speed_norm",
            "altitude_norm",
            "agency_scaled",
            "local_peak_cpi",
            "degree_norm",
            "mean_p_collision",
            "mean_miss_inv",
            "mean_tca_norm",
            "influence",
            "mean_rel_speed_norm",
        ],
    )


def generate_graph_snapshots(
    sample_count: int = 256,
    seed: int = 17,
) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    graphs: list[dict[str, np.ndarray]] = []

    for _ in range(sample_count):
        node_count = int(rng.integers(8, 32))
        positions = rng.normal(0.0, 1.0, size=(node_count, 3)) * rng.uniform(4500.0, 10000.0)
        velocities = rng.normal(0.0, 1.0, size=(node_count, 3)) * rng.uniform(1.0, 8.0)
        speed = np.linalg.norm(velocities, axis=1) * 3600.0
        altitude = np.linalg.norm(positions, axis=1) - R_EARTH_KM
        agency_id = rng.integers(0, 8, size=node_count).astype(float)
        local_peak_cpi = rng.uniform(0.0, 10.0, size=node_count)
        node_features = np.column_stack(
            [
                positions,
                velocities,
                speed,
                altitude,
                agency_id,
                local_peak_cpi,
            ]
        )

        edge_pairs = []
        edge_attrs = []
        for i in range(node_count):
            for j in range(i + 1, node_count):
                distance = float(np.linalg.norm(positions[i] - positions[j]))
                if distance > rng.uniform(750.0, 3200.0):
                    continue

                rel_speed = float(np.linalg.norm(velocities[i] - velocities[j]) * 3600.0)
                tca_minutes = float(rng.uniform(0.0, 720.0))
                p_collision = float(sigmoid((2.8 - distance / 180.0) + rel_speed / 18000.0 - tca_minutes / 220.0))
                edge_pairs.append((i, j))
                edge_attrs.append([distance, rel_speed, tca_minutes, p_collision])

        if edge_pairs:
            edge_index = np.array(edge_pairs, dtype=int).T
            edge_attr = np.array(edge_attrs, dtype=float)
        else:
            edge_index = np.zeros((2, 0), dtype=int)
            edge_attr = np.zeros((0, 4), dtype=float)

        summary = build_graph_summary_features(node_features, edge_index, edge_attr)
        influence = summary[:, 9]
        cpi_proxy = summary[:, 4] * 10.0
        p_collision_proxy = summary[:, 6]
        altitude_term = 1.0 - np.clip(summary[:, 2], 0.0, 1.0)
        degree_term = np.clip(summary[:, 5], 0.0, 1.0)

        logit = (
            -1.7
            + 2.8 * cpi_proxy / 10.0
            + 2.4 * influence
            + 0.75 * p_collision_proxy
            + 0.35 * altitude_term
            + 0.25 * degree_term
        )
        probability = sigmoid(logit)

        graphs.append(
            {
                "node_features": node_features.astype(float),
                "edge_index": edge_index.astype(int),
                "edge_attr": edge_attr.astype(float),
                "targets": probability.astype(float),
            }
        )

    return graphs
