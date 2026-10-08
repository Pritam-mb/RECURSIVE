"""
Constellation cascade planner.

Builds a proximity graph with KD-tree pruning, scores each conjunction with a
deterministic CPI heuristic, and produces a cascade maneuver plan that can be
shown on the dashboard.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from datetime import datetime, timezone
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

from app.core.conjunction import classify_severity, find_tca
from app.core.sgp4_propagator import SatelliteState, rsw_to_eci
from app.ml.gat_cascade import CascadeGAT
from app.ml.gnn_cascade import CascadeGNN
from app.ml.xgboost_scorer import XGBoostScorer
from app.core.agency import AGENCY_ALIASES as agency_aliases, infer_agency as agency_inference

logger = logging.getLogger(__name__)

R_EARTH_KM = 6371.0
DEFAULT_INFLUENCE_RADIUS_KM = 200.0
DEFAULT_CPI_THRESHOLD = 5.0
# Probability gap above which the GAT and the GNN cross-check are considered to
# disagree about a satellite. Tuned to sit above ordinary disagreement on the
# synthetic graphs while still catching a ranker that is structurally wrong
# rather than merely noisy.
RANKER_DISAGREEMENT_THRESHOLD = 0.25


AGENCY_ALIASES = agency_aliases
infer_agency = agency_inference


@dataclass
class CascadeNode:
    norad_id: int
    name: str
    position: np.ndarray
    velocity: np.ndarray
    speed_kmh: float
    altitude_km: float
    agency: str


def _norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def _estimate_linear_tca_minutes(
    pos_a: np.ndarray,
    vel_a: np.ndarray,
    pos_b: np.ndarray,
    vel_b: np.ndarray,
) -> tuple[float, float]:
    relative_position = pos_a - pos_b
    relative_velocity = vel_a - vel_b
    relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))

    if relative_speed_sq <= 1e-12:
        return 0.0, _norm(relative_position)

    t_seconds = -float(np.dot(relative_position, relative_velocity)) / relative_speed_sq
    if t_seconds < 0.0:
        t_seconds = 0.0

    closest_position = relative_position + (relative_velocity * t_seconds)
    return t_seconds / 60.0, _norm(closest_position)


def _midpoint_position(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    return (pos_a + pos_b) / 2.0


def _build_node(state: SatelliteState) -> CascadeNode:
    speed_kmh = float(np.linalg.norm([state.vx, state.vy, state.vz]) * 3600.0)
    altitude_km = float(np.linalg.norm([state.x, state.y, state.z]) - R_EARTH_KM)
    return CascadeNode(
        norad_id=state.norad_id,
        name=state.name,
        position=np.array([state.x, state.y, state.z], dtype=float),
        velocity=np.array([state.vx, state.vy, state.vz], dtype=float),
        speed_kmh=speed_kmh,
        altitude_km=altitude_km,
        agency=infer_agency(state.name),
    )


class CascadePlanner:
    """KD-tree based proximity graph planner with cascade scoring."""

    def __init__(self) -> None:
        self.risk_model = XGBoostScorer()
        # The primary ranker was historically stored on an attribute called
        # `self.gnn` while actually holding a CascadeGAT, which hid the fact
        # that CascadeGNN was never in the request path. It is now named for
        # what it is, and the GNN is kept as a deliberate cross-check.
        self.primary_ranker = CascadeGAT()
        # Second, structurally different ranker. It does not contribute to the
        # manoeuvre plan; it exists so a broken or degenerate primary ranker is
        # detectable instead of silently degrading every probability to zero.
        self.cross_check_ranker = CascadeGNN()

    def build_graph(
        self,
        states: list[SatelliteState],
        influence_radius_km: float = DEFAULT_INFLUENCE_RADIUS_KM,
        propagator: Any | None = None,
        reference_time: datetime | None = None,
        refine_limit: int = 30,
    ) -> dict[str, Any]:
        nodes = [_build_node(state) for state in states if state.error_code == 0]
        if not nodes:
            return {
                "nodes": [],
                "edges": [],
                "hotspots": [],
                "adjacency": {},
                # analyze_snapshot() reads this key before its own `if not nodes`
                # guard, so the empty graph must supply it too.
                "node_index": {},
                "summary": {
                    "node_count": 0,
                    "edge_count": 0,
                    "influence_radius_km": influence_radius_km,
                },
            }

        positions = np.array([node.position for node in nodes], dtype=float)
        velocities = np.array([node.velocity for node in nodes], dtype=float)
        tree = cKDTree(positions)
        candidate_pairs = sorted(tree.query_pairs(r=influence_radius_km))

        edges: list[dict[str, Any]] = []
        adjacency: dict[int, list[dict[str, Any]]] = {node.norad_id: [] for node in nodes}
        node_index = {node.norad_id: idx for idx, node in enumerate(nodes)}
        node_lookup = {node.norad_id: node for node in nodes}

        for idx_a, idx_b in candidate_pairs:
            node_a = nodes[idx_a]
            node_b = nodes[idx_b]
            miss_distance_km = _norm(node_a.position - node_b.position)
            relative_velocity_kms = _norm(node_a.velocity - node_b.velocity)
            tca_minutes, predicted_miss_km = _estimate_linear_tca_minutes(
                node_a.position,
                node_a.velocity,
                node_b.position,
                node_b.velocity,
            )

            tca_utc = None
            hotspot_position = _midpoint_position(node_a.position, node_b.position)

            if propagator is not None:
                start_time = reference_time or datetime.now(timezone.utc)
                tca_event = find_tca(
                    propagator,
                    node_a.norad_id,
                    node_b.norad_id,
                    start=start_time,
                    hours_ahead=24.0,
                    steps=120,
                )

                if tca_event is not None:
                    exact_tca = _parse_datetime(tca_event.tca_utc)
                    if exact_tca is not None:
                        state_a = propagator.propagate_one(node_a.norad_id, exact_tca)
                        state_b = propagator.propagate_one(node_b.norad_id, exact_tca)
                        if state_a is not None and state_b is not None and state_a.error_code == 0 and state_b.error_code == 0:
                            pos_a = np.array([state_a.x, state_a.y, state_a.z], dtype=float)
                            pos_b = np.array([state_b.x, state_b.y, state_b.z], dtype=float)
                            vel_a = np.array([state_a.vx, state_a.vy, state_a.vz], dtype=float)
                            vel_b = np.array([state_b.vx, state_b.vy, state_b.vz], dtype=float)

                            miss_distance_km = _norm(pos_a - pos_b)
                            relative_velocity_kms = _norm(vel_a - vel_b)
                            tca_minutes = max(0.0, (exact_tca - start_time).total_seconds() / 60.0)
                            predicted_miss_km = miss_distance_km
                            tca_utc = exact_tca.isoformat()
                            hotspot_position = _midpoint_position(pos_a, pos_b)

            risk_features = {
                "miss_distance_km": predicted_miss_km,
                "relative_speed_kmh": relative_velocity_kms * 3600.0,
                "sat1_altitude_km": node_a.altitude_km,
                "sat2_altitude_km": node_b.altitude_km,
                "tca_minutes": tca_minutes,
            }
            p_collision = self.risk_model.score(risk_features)
            cpi_score = self._compute_cpi(predicted_miss_km, relative_velocity_kms, tca_minutes, p_collision)
            severity = classify_severity(predicted_miss_km)
            influence_weight = 1.0 / max(predicted_miss_km, 1.0)
            hotspot_score = self._compute_hotspot_score(cpi_score, p_collision, predicted_miss_km, tca_minutes)

            edge_ab = {
                "source_id": node_a.norad_id,
                "target_id": node_b.norad_id,
                "source_name": node_a.name,
                "target_name": node_b.name,
                "source_index": idx_a,
                "target_index": idx_b,
                "miss_distance_km": round(float(miss_distance_km), 3),
                "predicted_miss_distance_km": round(float(predicted_miss_km), 3),
                "relative_velocity_kmh": round(float(relative_velocity_kms * 3600.0), 2),
                "tca_minutes": round(float(tca_minutes), 2),
                "tca_utc": tca_utc,
                "p_collision": round(float(p_collision), 4),
                "cpi_score": round(float(cpi_score), 2),
                "hotspot_score": round(float(hotspot_score), 3),
                "severity": severity,
                "influence_weight": round(float(influence_weight), 6),
                "hotspot_position": {
                    "x": round(float(hotspot_position[0]), 3),
                    "y": round(float(hotspot_position[1]), 3),
                    "z": round(float(hotspot_position[2]), 3),
                },
            }
            edge_ba = {**edge_ab, "source_id": node_b.norad_id, "target_id": node_a.norad_id, "source_name": node_b.name, "target_name": node_a.name, "source_index": idx_b, "target_index": idx_a}

            edges.append(edge_ab)
            adjacency[node_a.norad_id].append(edge_ab)
            adjacency[node_b.norad_id].append(edge_ba)

        if propagator is not None and edges:
            refine_count = min(len(edges), max(refine_limit, 1))
            edges_to_refine = sorted(edges, key=lambda item: item["cpi_score"], reverse=True)[:refine_count]
            for edge in edges_to_refine:
                refined = self._refine_edge_with_tca(
                    node_lookup[edge["source_id"]],
                    node_lookup[edge["target_id"]],
                    edge,
                    propagator,
                    reference_time or datetime.now(timezone.utc),
                )
                if refined is not None:
                    edge.update(refined)

            adjacency = {node.norad_id: [] for node in nodes}
            for edge in edges:
                edge_ba = {
                    **edge,
                    "source_id": edge["target_id"],
                    "target_id": edge["source_id"],
                    "source_name": edge["target_name"],
                    "target_name": edge["source_name"],
                    "source_index": edge["target_index"],
                    "target_index": edge["source_index"],
                }
                adjacency[edge["source_id"]].append(edge)
                adjacency[edge["target_id"]].append(edge_ba)

        hotspots = self._serialize_hotspots(edges)

        return {
            "nodes": nodes,
            "edges": edges,
            "hotspots": hotspots,
            "adjacency": adjacency,
            "node_index": node_index,
            "positions": positions,
            "velocities": velocities,
            "summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "influence_radius_km": influence_radius_km,
            },
        }

    def analyze_snapshot(
        self,
        states: list[SatelliteState],
        alerts: list[dict[str, Any]] | None = None,
        propagator: Any | None = None,
        reference_time: datetime | None = None,
        influence_radius_km: float = DEFAULT_INFLUENCE_RADIUS_KM,
        cpi_threshold: float = DEFAULT_CPI_THRESHOLD,
        max_depth: int = 3,
        max_maneuvers: int = 50,
    ) -> dict[str, Any]:
        graph = self.build_graph(
            states,
            influence_radius_km=influence_radius_km,
            propagator=propagator,
            reference_time=reference_time,
        )
        nodes: list[CascadeNode] = graph["nodes"]
        edges: list[dict[str, Any]] = graph["edges"]
        hotspots: list[dict[str, Any]] = graph.get("hotspots", [])
        adjacency: dict[int, list[dict[str, Any]]] = graph["adjacency"]
        node_index: dict[int, int] = graph["node_index"]

        if not nodes:
            return {
                "graph": graph["summary"],
                "alerts": [],
                "cascade_plan": [],
                "hotspots": [],
                "total_delta_v_ms": 0.0,
                "cascade_depth": 0,
                "agencies_involved": [],
                "ranker_review": {
                    "primary": "gat",
                    "cross_check": "gnn",
                    "primary_ok": True,
                    "cross_check_ok": True,
                    "fallback_used": False,
                    "degraded": False,
                    "disagreement_threshold": RANKER_DISAGREEMENT_THRESHOLD,
                    "disagreement_count": 0,
                    "max_disagreement": 0.0,
                    "disputed_satellites": [],
                },
            }

        # ── Ranker scoring (P15/P16) ─────────────────────────────────────────
        # The primary ranker drives the plan. The cross-check ranker never
        # contributes; it only reports where the two disagree so a silently
        # broken or degenerate primary is visible. Neither model's score is
        # blended: they share a data generator, so agreement between them is
        # weak evidence and a fused number would imply precision neither has.
        ranker_review: dict[str, Any] = {
            "primary": "gat",
            "cross_check": "gnn",
            "primary_ok": False,
            "cross_check_ok": False,
            "fallback_used": False,
            "degraded": False,
            "disagreement_threshold": RANKER_DISAGREEMENT_THRESHOLD,
            "disagreement_count": 0,
            "max_disagreement": 0.0,
            "disputed_satellites": [],
        }
        maneuver_probability = np.zeros(len(nodes), dtype=float)
        ranker_depth = np.zeros(len(nodes), dtype=int)

        try:
            node_features, edge_index, edge_attr = self._build_gnn_inputs(
                nodes, edges, node_index, adjacency
            )
        except Exception as error:
            # Without a graph there is nothing for any ranker to score, and
            # fabricating zeros here would produce a plausible-looking plan
            # with no basis in the physics.
            raise RuntimeError(
                f"Could not build cascade graph inputs: {error}"
            ) from error

        primary_error: Exception | None = None
        try:
            primary_output = self.primary_ranker.predict(
                node_features, edge_index, edge_attr
            )
            maneuver_probability = np.asarray(
                primary_output["maneuver_probability"], dtype=float
            )
            ranker_depth = np.asarray(primary_output["cascade_depth"], dtype=int)
            ranker_review["primary_ok"] = True
        except Exception as error:
            primary_error = error
            logger.error("Primary cascade ranker (GAT) failed: %s", error)

        try:
            cross_output = self.cross_check_ranker.predict(
                node_features, edge_index, edge_attr
            )
            cross_probability = np.asarray(
                cross_output["maneuver_probability"], dtype=float
            )
            ranker_review["cross_check_ok"] = True
        except Exception as error:
            cross_probability = None
            logger.error("Cross-check cascade ranker (GNN) failed: %s", error)

        if not ranker_review["primary_ok"]:
            if cross_probability is None:
                # Both rankers are unavailable. Raising leaves the previous
                # alert cache in place, so the UI keeps showing the last real
                # plan. Emitting zeros instead would be a fresh, entirely
                # fabricated plan that looks normal to the operator.
                raise RuntimeError(
                    "Both cascade rankers failed; refusing to emit a plan with "
                    f"fabricated probabilities. primary={primary_error!r}"
                )
            # The primary failed but the cross-check is healthy: use it, and
            # mark the result degraded so the UI can say so.
            maneuver_probability = cross_probability
            ranker_depth = np.asarray(cross_output["cascade_depth"], dtype=int)
            ranker_review["fallback_used"] = True
            ranker_review["degraded"] = True
            ranker_review["degraded_reason"] = (
                "Primary GAT ranker failed; probabilities came from the GNN "
                "cross-check and are not attention-derived."
            )
        elif cross_probability is not None:
            # Both healthy: report where they disagree. The primary still wins.
            if cross_probability.shape == maneuver_probability.shape:
                delta = np.abs(cross_probability - maneuver_probability)
                disputed = np.flatnonzero(
                    delta > RANKER_DISAGREEMENT_THRESHOLD
                )
                ranker_review["disagreement_count"] = int(disputed.size)
                ranker_review["max_disagreement"] = (
                    round(float(delta.max()), 4) if delta.size else 0.0
                )
                ranker_review["disputed_satellites"] = [
                    {
                        "satellite_id": int(nodes[i].norad_id),
                        "satellite_name": nodes[i].name,
                        "gat_probability": round(float(maneuver_probability[i]), 4),
                        "gnn_probability": round(float(cross_probability[i]), 4),
                        "delta": round(float(delta[i]), 4),
                    }
                    for i in disputed
                ]

        ranked_nodes = sorted(
            nodes,
            key=lambda node: (
                maneuver_probability[node_index[node.norad_id]],
                self._node_peak_cpi(node.norad_id, adjacency),
            ),
            reverse=True,
        )

        if alerts:
            seed_ids = []
            for alert in sorted(alerts, key=lambda item: item.get("cpi_score", 0.0), reverse=True):
                seed_ids.append(alert["sat1"]["id"])
                seed_ids.append(alert["sat2"]["id"])
            seed_ids = list(dict.fromkeys(seed_ids))
        else:
            seed_ids = [node.norad_id for node in ranked_nodes[:3]]

        resolved_maneuvers = self._resolve_cascade(
            seed_ids=seed_ids,
            nodes=nodes,
            node_index=node_index,
            adjacency=adjacency,
            maneuver_probability=maneuver_probability,
            ranker_depth=ranker_depth,
            max_depth=max_depth,
            max_maneuvers=max_maneuvers,
            cpi_threshold=cpi_threshold,
        )

        optimized_maneuvers, optimization_summary = self._optimize_cascade_plan(
            resolved_maneuvers,
            cpi_threshold=cpi_threshold,
        )

        agencies_involved = sorted({maneuver["agency"] for maneuver in optimized_maneuvers})
        total_delta_v_ms = round(sum(maneuver["maneuver"]["delta_v_ms"] for maneuver in optimized_maneuvers), 3)
        max_cascade_depth = max([1] + [maneuver["cascade_depth"] for maneuver in optimized_maneuvers])

        return {
            "graph": graph["summary"],
            "alerts": self._serialize_alerts(edges),
            "hotspots": hotspots,
            "cascade_plan": optimized_maneuvers,
            "seed_satellites": seed_ids,
            "total_delta_v_ms": total_delta_v_ms,
            "cascade_depth": max_cascade_depth,
            "agencies_involved": agencies_involved,
            "cpi_threshold": cpi_threshold,
            "optimization": optimization_summary,
            "ranker_review": ranker_review,
            "node_probabilities": {
                str(node.norad_id): round(float(maneuver_probability[node_index[node.norad_id]]), 3)
                for node in nodes
            },
        }

    def _resolve_cascade(
        self,
        seed_ids: list[int],
        nodes: list[CascadeNode],
        node_index: dict[int, int],
        adjacency: dict[int, list[dict[str, Any]]],
        maneuver_probability: np.ndarray,
        ranker_depth: np.ndarray,
        max_depth: int,
        max_maneuvers: int,
        cpi_threshold: float,
    ) -> list[dict[str, Any]]:
        queue = deque((sat_id, 1, None, "PRIMARY_CONJUNCTION") for sat_id in seed_ids)
        visited: set[int] = set()
        resolved_maneuvers: list[dict[str, Any]] = []

        while queue and len(resolved_maneuvers) < max_maneuvers:
            sat_id, depth, trigger_sat_id, trigger_label = queue.popleft()
            if sat_id in visited or depth > max_depth:
                continue
            visited.add(sat_id)

            threat = self._strongest_threat(sat_id, adjacency)
            if threat is None:
                continue

            node = nodes[node_index[sat_id]]
            threat_node = nodes[node_index[threat["target_id"]]]
            node_prob = float(maneuver_probability[node_index[sat_id]])
            local_cpi = float(threat["cpi_score"])
            cascade_depth_score = int(max(depth, int(ranker_depth[node_index[sat_id]])))

            maneuver_rsw = self._recommend_maneuver_rsw(node, threat_node, local_cpi, node_prob)
            maneuver_eci = rsw_to_eci(maneuver_rsw, node.position, node.velocity)
            total_delta_v_ms = float(np.linalg.norm(maneuver_rsw))
            risk_before = local_cpi
            risk_after = max(0.0, risk_before - (node_prob * 3.0) - (total_delta_v_ms * 0.6))

            resolved_maneuvers.append(
                {
                    "satellite_id": sat_id,
                    "satellite_name": node.name,
                    "agency": node.agency,
                    "cascade_depth": cascade_depth_score,
                    "triggered_by": trigger_sat_id,
                    "trigger_label": trigger_label,
                    "threat": {
                        "satellite_id": threat["target_id"],
                        "satellite_name": threat["target_name"],
                        "miss_distance_km": threat["predicted_miss_distance_km"],
                        "relative_velocity_kmh": threat["relative_velocity_kmh"],
                        "tca_minutes": threat["tca_minutes"],
                        "p_collision": threat["p_collision"],
                        "cpi_score": threat["cpi_score"],
                    },
                    "maneuver": {
                        "frame": "RSW",
                        "dv_r": round(float(maneuver_rsw[0]), 3),
                        "dv_s": round(float(maneuver_rsw[1]), 3),
                        "dv_w": round(float(maneuver_rsw[2]), 3),
                        "delta_v_ms": round(total_delta_v_ms, 3),
                        "eci_ms": [
                            round(float(maneuver_eci[0]), 3),
                            round(float(maneuver_eci[1]), 3),
                            round(float(maneuver_eci[2]), 3),
                        ],
                    },
                    "risk_before": round(risk_before, 2),
                    "risk_after": round(risk_after, 2),
                    "maneuver_probability": round(node_prob, 3),
                }
            )

            for neighbor_edge in sorted(adjacency.get(sat_id, []), key=lambda item: item["cpi_score"], reverse=True):
                if neighbor_edge["target_id"] in visited:
                    continue
                if neighbor_edge["cpi_score"] >= cpi_threshold or maneuver_probability[node_index[neighbor_edge["target_id"]]] >= 0.55:
                    queue.append(
                        (
                            neighbor_edge["target_id"],
                            depth + 1,
                            sat_id,
                            f"CASCADE_FROM_{sat_id}",
                        )
                    )

        return resolved_maneuvers

    def _optimize_cascade_plan(
        self,
        maneuvers: list[dict[str, Any]],
        cpi_threshold: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not maneuvers:
            return [], {
                "status": "empty",
                "initial_total_delta_v_ms": 0.0,
                "optimized_total_delta_v_ms": 0.0,
                "optimization_gain_pct": 0.0,
                "iterations": 0,
            }

        base_delta_v = np.array([float(m["maneuver"]["delta_v_ms"]) for m in maneuvers], dtype=float)
        risk_before = np.array([float(m.get("risk_before", 0.0)) for m in maneuvers], dtype=float)
        maneuver_prob = np.array([float(m.get("maneuver_probability", 0.0)) for m in maneuvers], dtype=float)
        node_targets = np.array([max(0.5, min(risk_before[i] * 0.75, cpi_threshold - 0.25)) for i in range(len(maneuvers))], dtype=float)

        x0 = np.ones(len(maneuvers), dtype=float)
        bounds = [(0.45, 1.9) for _ in maneuvers]

        def adjusted_risk(scale_vector: np.ndarray) -> np.ndarray:
            total_dv = base_delta_v * scale_vector
            return np.maximum(0.0, risk_before - (maneuver_prob * 3.0) - (total_dv * 0.6))

        def objective(scale_vector: np.ndarray) -> float:
            total_dv = float(np.sum(base_delta_v * scale_vector))
            risk_penalty = float(np.sum(np.square(np.maximum(0.0, adjusted_risk(scale_vector) - node_targets))))
            smoothness_penalty = float(np.sum(np.square(scale_vector - 1.0)))
            return total_dv + (1.5 * risk_penalty) + (0.05 * smoothness_penalty)

        constraints = []
        for idx in range(len(maneuvers)):
            constraints.append({
                "type": "ineq",
                "fun": lambda scale_vector, i=idx: float(node_targets[i] - adjusted_risk(scale_vector)[i]),
            })

        try:
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-6},
            )
            scales = np.asarray(result.x if result.success and result.x is not None else x0, dtype=float)
            status = "optimized" if result.success else "fallback"
            iterations = int(getattr(result, "nit", 0) or 0)
            objective_value = float(result.fun) if result.success and result.fun is not None else float(objective(scales))
        except Exception as error:
            logger.warning("Cascade optimizer failed, using unoptimized maneuvers: %s", error)
            scales = x0
            status = "fallback"
            iterations = 0
            objective_value = float(objective(scales))

        optimized_maneuvers = []
        for idx, (maneuver, scale) in enumerate(zip(maneuvers, scales, strict=False)):
            updated = dict(maneuver)
            base_vec = np.array([
                float(maneuver["maneuver"]["dv_r"]),
                float(maneuver["maneuver"]["dv_s"]),
                float(maneuver["maneuver"]["dv_w"]),
            ], dtype=float)
            optimized_vec = base_vec * float(scale)
            optimized_delta_v = float(np.linalg.norm(optimized_vec))
            updated["optimization_scale"] = round(float(scale), 3)
            updated["maneuver"]["dv_r"] = round(float(optimized_vec[0]), 3)
            updated["maneuver"]["dv_s"] = round(float(optimized_vec[1]), 3)
            updated["maneuver"]["dv_w"] = round(float(optimized_vec[2]), 3)
            updated["maneuver"]["delta_v_ms"] = round(optimized_delta_v, 3)
            updated["optimization_status"] = status
            per_scale = np.array([scale if j == idx else 1.0 for j in range(len(maneuvers))], dtype=float)
            updated["optimized_risk_after"] = round(float(adjusted_risk(per_scale)[idx]), 2)
            optimized_maneuvers.append(updated)

        initial_total = float(np.sum(base_delta_v))
        optimized_total = float(np.sum([m["maneuver"]["delta_v_ms"] for m in optimized_maneuvers]))
        optimization_summary = {
            "status": status,
            "iterations": iterations,
            "initial_total_delta_v_ms": round(initial_total, 3),
            "optimized_total_delta_v_ms": round(optimized_total, 3),
            "optimization_gain_pct": round(max(0.0, (1.0 - (optimized_total / max(initial_total, 1e-6))) * 100.0), 2),
            "objective_value": round(float(objective_value), 4),
            "maneuver_count": len(maneuvers),
            "scale_factors": [round(float(scale), 3) for scale in scales],
        }

        return optimized_maneuvers, optimization_summary

    def _build_gnn_inputs(
        self,
        nodes: list[CascadeNode],
        edges: list[dict[str, Any]],
        node_index: dict[int, int],
        adjacency: dict[int, list[dict[str, Any]]],
    ):
        agency_ids = self._agency_id_map(nodes)
        node_features = []
        for node in nodes:
            node_peak_cpi = self._node_peak_cpi(node.norad_id, adjacency)
            node_features.append(
                [
                    node.position[0],
                    node.position[1],
                    node.position[2],
                    node.velocity[0],
                    node.velocity[1],
                    node.velocity[2],
                    node.speed_kmh,
                    node.altitude_km,
                    agency_ids[node.agency],
                    node_peak_cpi,
                ]
            )

        edge_index = []
        edge_attr = []
        for edge in edges:
            edge_index.append([node_index[edge["source_id"]], node_index[edge["target_id"]]])
            edge_attr.append([
                edge["predicted_miss_distance_km"],
                edge["relative_velocity_kmh"] / 3600.0,
                edge["tca_minutes"],
                edge["p_collision"],
            ])

        if not edge_index:
            edge_index = np.zeros((2, 0), dtype=int)
            edge_attr = np.zeros((0, 4), dtype=float)
        else:
            edge_index = np.array(edge_index, dtype=int).T
            edge_attr = np.array(edge_attr, dtype=float)

        return np.array(node_features, dtype=float), edge_index, edge_attr

    def _agency_id_map(self, nodes: list[CascadeNode]) -> dict[str, int]:
        agencies = sorted({node.agency for node in nodes})
        return {agency: idx + 1 for idx, agency in enumerate(agencies)}

    def _compute_cpi(self, miss_distance_km: float, relative_velocity_kms: float, tca_minutes: float, p_collision: float) -> float:
        from app.core.conjunction import compute_cpi_score
        return compute_cpi_score(
            probability_of_collision=float(p_collision),
            miss_distance_km=float(miss_distance_km),
            tca_hours=max(0.0, float(tca_minutes)) / 60.0,
            relative_velocity_kms=float(relative_velocity_kms),
        )

    def _strongest_threat(self, sat_id: int, adjacency: dict[int, list[dict[str, Any]]]) -> dict[str, Any] | None:
        threats = adjacency.get(sat_id, [])
        if not threats:
            return None
        return max(threats, key=lambda item: item["cpi_score"])

    def _node_peak_cpi(self, sat_id: int, adjacency: dict[int, list[dict[str, Any]]]) -> float:
        threats = adjacency.get(sat_id, [])
        if not threats:
            return 0.0
        return float(max(item["cpi_score"] for item in threats))

    def _recommend_maneuver_rsw(
        self,
        node: CascadeNode,
        threat: CascadeNode,
        local_cpi: float,
        maneuver_probability: float,
    ) -> np.ndarray:
        r_hat = node.position / max(_norm(node.position), 1e-6)
        h_vec = np.cross(node.position, node.velocity)
        h_norm = max(_norm(h_vec), 1e-6)
        w_hat = h_vec / h_norm
        s_hat = np.cross(w_hat, r_hat)

        relative = threat.position - node.position
        radial_sign = -1.0 if float(np.dot(relative, r_hat)) > 0 else 1.0
        along_sign = -1.0 if float(np.dot(relative, s_hat)) > 0 else 1.0
        cross_sign = -1.0 if float(np.dot(relative, w_hat)) > 0 else 1.0

        magnitude = max(0.05, min(2.5, 0.15 + (local_cpi / 10.0) * 1.4 + maneuver_probability * 0.7))
        return np.array([
            radial_sign * magnitude * 0.25,
            along_sign * magnitude * 0.85,
            cross_sign * magnitude * 0.15,
        ], dtype=float)

    def _serialize_alerts(self, edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for item in sorted(edges, key=lambda edge: edge["cpi_score"], reverse=True):
            hotspot_position = item.get("hotspot_position") or {"x": 0.0, "y": 0.0, "z": 0.0}
            normalized.append(
                {
                    "id": f"{item['source_id']}-{item['target_id']}",
                    "sat1": {"id": item["source_id"], "name": item["source_name"]},
                    "sat2": {"id": item["target_id"], "name": item["target_name"]},
                    "miss_distance_km": item["predicted_miss_distance_km"],
                    "relative_speed_kmh": item["relative_velocity_kmh"],
                    "tca_utc": item.get("tca_utc"),
                    "tca_minutes": item.get("tca_minutes"),
                    "severity": item["severity"],
                    "cpi_score": item["cpi_score"],
                    "p_collision": item["p_collision"],
                    "hotspot_score": item.get("hotspot_score", 0.0),
                    "influence_weight": item["influence_weight"],
                    "position": hotspot_position,
                    "zone_radius_km": 100.0,
                    "tca_hours": round(float(item.get("tca_minutes", 0.0)) / 60.0, 2),
                }
            )

        return normalized

    def _serialize_hotspots(self, edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hotspots = []
        for item in sorted(edges, key=lambda edge: edge.get("hotspot_score", 0.0), reverse=True):
            hotspot_position = item.get("hotspot_position") or {"x": 0.0, "y": 0.0, "z": 0.0}
            hotspots.append(
                {
                    "sat1": {"id": item["source_id"], "name": item["source_name"]},
                    "sat2": {"id": item["target_id"], "name": item["target_name"]},
                    "tca_utc": item.get("tca_utc"),
                    "tca_minutes": item.get("tca_minutes"),
                    "miss_distance_km": item["predicted_miss_distance_km"],
                    "relative_speed_kmh": item["relative_velocity_kmh"],
                    "cpi_score": item["cpi_score"],
                    "p_collision": item["p_collision"],
                    "severity": item["severity"],
                    "hotspot_score": item.get("hotspot_score", 0.0),
                    "position": hotspot_position,
                    "affected_satellites": item.get("affected_satellites", []),
                    "affected_count": item.get("affected_count", 0),
                    "zone_radius_km": 100.0,
                    "tca_hours": round(float(item.get("tca_minutes", 0.0)) / 60.0, 2),
                }
            )

        return hotspots

    def _compute_hotspot_score(
        self,
        cpi_score: float,
        p_collision: float,
        miss_distance_km: float,
        tca_minutes: float,
    ) -> float:
        proximity_component = max(0.0, 1.0 - (miss_distance_km / 500.0))
        time_component = 1.0 / (1.0 + (max(tca_minutes, 0.0) / 60.0))
        cpi_component = min(max(cpi_score / 10.0, 0.0), 1.0)
        score = (0.40 * p_collision) + (0.30 * cpi_component) + (0.20 * proximity_component) + (0.10 * time_component)
        return max(0.0, min(1.0, score))

    def _refine_edge_with_tca(
        self,
        node_a: CascadeNode,
        node_b: CascadeNode,
        edge: dict[str, Any],
        propagator: Any,
        reference_time: datetime,
    ) -> dict[str, Any] | None:
        tca_event = find_tca(
            propagator,
            node_a.norad_id,
            node_b.norad_id,
            start=reference_time,
            hours_ahead=24.0,
            steps=120,
        )
        if tca_event is None:
            return None

        exact_tca = _parse_datetime(tca_event.tca_utc)
        if exact_tca is None:
            return None

        state_a = propagator.propagate_one(node_a.norad_id, exact_tca)
        state_b = propagator.propagate_one(node_b.norad_id, exact_tca)
        if state_a is None or state_b is None or state_a.error_code != 0 or state_b.error_code != 0:
            return None

        pos_a = np.array([state_a.x, state_a.y, state_a.z], dtype=float)
        pos_b = np.array([state_b.x, state_b.y, state_b.z], dtype=float)
        vel_a = np.array([state_a.vx, state_a.vy, state_a.vz], dtype=float)
        vel_b = np.array([state_b.vx, state_b.vy, state_b.vz], dtype=float)

        miss_distance_km = _norm(pos_a - pos_b)
        relative_velocity_kms = _norm(vel_a - vel_b)
        tca_minutes = max(0.0, (exact_tca - reference_time).total_seconds() / 60.0)
        p_collision = self.risk_model.score({
            "miss_distance_km": miss_distance_km,
            "relative_speed_kmh": relative_velocity_kms * 3600.0,
            "sat1_altitude_km": node_a.altitude_km,
            "sat2_altitude_km": node_b.altitude_km,
            "tca_minutes": tca_minutes,
        })
        cpi_score = self._compute_cpi(miss_distance_km, relative_velocity_kms, tca_minutes, p_collision)
        hotspot_score = self._compute_hotspot_score(cpi_score, p_collision, miss_distance_km, tca_minutes)

        return {
            "miss_distance_km": round(float(miss_distance_km), 3),
            "predicted_miss_distance_km": round(float(miss_distance_km), 3),
            "relative_velocity_kmh": round(float(relative_velocity_kms * 3600.0), 2),
            "tca_minutes": round(float(tca_minutes), 2),
            "tca_utc": exact_tca.isoformat(),
            "p_collision": round(float(p_collision), 4),
            "cpi_score": round(float(cpi_score), 2),
            "hotspot_score": round(float(hotspot_score), 3),
            "hotspot_position": {
                "x": round(float((pos_a[0] + pos_b[0]) / 2.0), 3),
                "y": round(float((pos_a[1] + pos_b[1]) / 2.0), 3),
                "z": round(float((pos_a[2] + pos_b[2]) / 2.0), 3),
            },
        }