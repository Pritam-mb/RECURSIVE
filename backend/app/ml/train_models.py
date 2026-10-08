from __future__ import annotations

import argparse
import logging

from .gnn_cascade import train_graph_model
from .gat_cascade import train_gat_model
from .lstm_predictor import train_trajectory_model
from .xgboost_scorer import train_risk_model, train_xgboost_model


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ML artifacts for the cascade demo.")
    parser.add_argument("--graph", action="store_true", help="Train the graph cascade model")
    parser.add_argument("--trajectory", action="store_true", help="Train the trajectory model")
    parser.add_argument("--risk", action="store_true", help="Train the boosted risk model")
    parser.add_argument("--xgboost", action="store_true", help="Train the XGBoost risk model")
    parser.add_argument("--gat", action="store_true", help="Train the PyTorch GAT model")
    parser.add_argument("--all", action="store_true", help="Train all models (except XGBoost)")
    parser.add_argument("--graph-samples", type=int, default=4096)
    parser.add_argument("--graph-epochs", type=int, default=20)
    parser.add_argument("--graph-hidden", type=int, default=16)
    parser.add_argument("--trajectory-samples", type=int, default=2048)
    parser.add_argument("--trajectory-history", type=int, default=8)
    parser.add_argument("--trajectory-epochs", type=int, default=18)
    parser.add_argument("--trajectory-hidden", type=int, default=12)
    parser.add_argument("--risk-samples", type=int, default=4096)
    parser.add_argument("--risk-rounds", type=int, default=24)
    parser.add_argument("--risk-lr", type=float, default=0.18)
    parser.add_argument("--xgboost-samples", type=int, default=16384)
    parser.add_argument("--gat-samples", type=int, default=256)
    parser.add_argument("--gat-epochs", type=int, default=30)
    parser.add_argument("--gat-hidden", type=int, default=32)
    parser.add_argument("--gat-heads", type=int, default=4)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()

    run_graph = args.all or args.graph
    run_trajectory = args.all or args.trajectory
    run_risk = args.all or args.risk
    run_gat = args.gat

    if not (run_graph or run_trajectory or run_risk or run_gat or args.xgboost):
        logger.info("No model selected. Use --all or a specific flag.")
        return 1

    if run_graph:
        train_graph_model(
            sample_count=args.graph_samples,
            hidden_size=args.graph_hidden,
            epochs=args.graph_epochs,
        )

    if run_trajectory:
        train_trajectory_model(
            sample_count=args.trajectory_samples,
            history_length=args.trajectory_history,
            hidden_size=args.trajectory_hidden,
            epochs=args.trajectory_epochs,
        )

    if run_risk:
        train_risk_model(
            sample_count=args.risk_samples,
            rounds=args.risk_rounds,
            learning_rate=args.risk_lr,
        )

    if run_gat:
        train_gat_model(
            sample_count=args.gat_samples,
            hidden_dim=args.gat_hidden,
            heads=args.gat_heads,
            epochs=args.gat_epochs,
        )

    if args.xgboost:
        train_xgboost_model(sample_count=args.xgboost_samples)

    logger.info("Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
