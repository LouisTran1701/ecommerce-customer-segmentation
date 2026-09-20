"""Command-line interface for reproducible model training and scoring."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .features import DataValidationError
from .model import load_model, save_model, score_customers, train_model


def _read_transactions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Transaction file does not exist: {path}")
    return pd.read_csv(path, encoding="unicode_escape")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segment",
        description="Train and run the e-commerce customer segmentation pipeline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Fit and persist a segmentation model.")
    train.add_argument("--input", type=Path, required=True, help="Raw transaction CSV.")
    train.add_argument(
        "--model-out",
        type=Path,
        default=Path("artifacts/segmentation.joblib"),
        help="Destination for the fitted model artifact.",
    )
    train.add_argument(
        "--assignments-out",
        type=Path,
        default=Path("outputs/training_segments.csv"),
        help="Destination for customer assignments.",
    )
    train.add_argument("--as-of", help="Exclusive ISO date/time cutoff for training data.")
    train.add_argument("--clusters", type=int, default=4, help="Number of retail clusters.")

    score = commands.add_parser("score", help="Score customer transaction histories.")
    score.add_argument("--input", type=Path, required=True, help="Transaction history CSV.")
    score.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/segmentation.joblib"),
        help="Previously fitted model artifact.",
    )
    score.add_argument("--output", type=Path, required=True, help="Prediction CSV destination.")
    score.add_argument("--as-of", help="Exclusive ISO date/time cutoff for scoring data.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            transactions = _read_transactions(args.input)
            artifact, assignments = train_model(
                transactions,
                as_of_date=args.as_of,
                n_clusters=args.clusters,
            )
            save_model(artifact, args.model_out)
            _write_csv(assignments, args.assignments_out)
            print(
                json.dumps(
                    {
                        "model": str(args.model_out),
                        "assignments": str(args.assignments_out),
                        "customers": artifact["training_customers"],
                        "retail_customers": artifact["training_retail_customers"],
                        "metrics": artifact["metrics"],
                    },
                    indent=2,
                )
            )
        else:
            artifact = load_model(args.model)
            predictions = score_customers(
                _read_transactions(args.input), artifact, as_of_date=args.as_of
            )
            _write_csv(predictions, args.output)
            print(json.dumps({"output": str(args.output), "customers": len(predictions)}, indent=2))
    except (DataValidationError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
