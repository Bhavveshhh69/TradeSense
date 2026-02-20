"""Phase 11.5 empirical validation runner for TradeSense."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .backtest_engine import run_backtest


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _default_window() -> tuple[str, str]:
    today = datetime.now(tz=UTC).date()
    start = today - timedelta(days=365)
    end = today - timedelta(days=14)
    return start.isoformat(), end.isoformat()


def _build_parser() -> argparse.ArgumentParser:
    default_start, default_end = _default_window()
    parser = argparse.ArgumentParser(
        description="Run empirical backtest validation and save calibration report.",
    )
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol to evaluate.")
    parser.add_argument(
        "--start-date",
        default=default_start,
        help="Backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default=default_end,
        help="Backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument("--interval", default="1d", help="Market data interval.")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon in days.")
    return parser


def _reports_dir() -> Path:
    reports_path = Path(__file__).resolve().parent / "reports"
    reports_path.mkdir(parents=True, exist_ok=True)
    return reports_path


def _validate_calibrated_predictions(predictions: pd.DataFrame) -> None:
    if "probability_calibrated" not in predictions.columns:
        raise RuntimeError("Backtest predictions are missing calibrated probabilities")
    if "probability_raw" not in predictions.columns:
        raise RuntimeError("Backtest predictions are missing raw probabilities")
    # Safety gate: evaluation must use the calibrated probability output only.
    if "probability" in predictions.columns and not predictions["probability"].equals(
        predictions["probability_calibrated"]
    ):
        raise RuntimeError("Validation runner detected non-calibrated probability usage")


def main() -> None:
    args = _build_parser().parse_args()
    symbol = str(args.symbol).upper().strip()

    predictions, result = run_backtest(
        symbol=symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        horizon=args.horizon,
    )
    _validate_calibrated_predictions(predictions)

    metrics = _model_to_dict(result.metrics)
    reliability_curve = [_model_to_dict(bucket) for bucket in result.reliability]

    summary = {
        "accuracy": float(metrics["overall_accuracy"]),
        "ece": float(metrics["expected_calibration_error"]),
        "brier_score": float(metrics["brier_score"]),
        "accuracy_by_confidence": metrics["accuracy_by_confidence_level"],
    }

    print(f"symbol: {result.symbol}")
    print(f"period: {result.start_date} -> {result.end_date}")
    print(f"accuracy: {summary['accuracy']:.6f}")
    print(f"ece: {summary['ece']:.6f}")
    print(f"brier_score: {summary['brier_score']:.6f}")
    print(f"accuracy_by_confidence: {summary['accuracy_by_confidence']}")

    report_payload = {
        "symbol": result.symbol,
        "period": {
            "start_date": result.start_date,
            "end_date": result.end_date,
            "horizon": result.horizon,
            "total_predictions": result.total_predictions,
        },
        "metrics": summary,
        "calibration_metrics": metrics,
        "reliability_curve": reliability_curve,
    }

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = _reports_dir() / f"{result.symbol}_{timestamp}.json"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"report_file: {report_path}")


if __name__ == "__main__":
    main()
