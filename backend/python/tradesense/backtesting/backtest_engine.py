"""Phase 11 backtesting engine for calibrated inference evaluation."""

from __future__ import annotations

from datetime import timedelta
from typing import Callable, Dict, Tuple
from unittest.mock import patch

import pandas as pd

from tradesense.data_provider import get_market_data
from tradesense.features import build_feature_matrix
from tradesense.inference.orchestrator import analyze_symbol
from tradesense.modeling import create_target

from .metrics import compute_backtest_metrics
from .reliability import compute_reliability_curve
from .schemas import BacktestResult, ReliabilityBucket

_LOOKBACK_DAYS = 220
_MIN_MARKET_ROWS = 50


def _normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    return symbol.upper().strip()


def _normalize_timestamp(value: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid date: {value}") from exc


def _provider_for_timestep(
    symbol: str,
    full_market_data: pd.DataFrame,
    as_of: pd.Timestamp,
) -> Callable[[list[str], str, str, str], Dict[str, pd.DataFrame]]:
    symbol = symbol.upper()
    history_slice = full_market_data.loc[full_market_data.index <= as_of].copy()
    empty_frame = full_market_data.iloc[0:0].copy()

    def _patched_get_market_data(
        symbols: list[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        del start_date, end_date, interval
        output: Dict[str, pd.DataFrame] = {}
        for item in symbols:
            key = str(item).upper().strip()
            output[key] = history_slice.copy() if key == symbol else empty_frame.copy()
        return output

    return _patched_get_market_data


def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    horizon: int = 5,
) -> Tuple[pd.DataFrame, BacktestResult]:
    """Run time-stepped backtesting using the existing inference orchestrator."""
    symbol = _normalize_symbol(symbol)
    start_ts = _normalize_timestamp(start_date)
    end_ts = _normalize_timestamp(end_date)

    if start_ts > end_ts:
        raise ValueError("start_date must be on or before end_date")
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    fetch_start = (start_ts - timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    fetch_end = (end_ts + timedelta(days=horizon + 5)).strftime("%Y-%m-%d")

    market_data_map = get_market_data([symbol], fetch_start, fetch_end, interval=interval)
    market_data = market_data_map.get(symbol)
    if market_data is None or market_data.empty:
        raise ValueError(f"No market data available for symbol: {symbol}")

    market_data = market_data.copy()
    market_data.index = pd.to_datetime(market_data.index)
    if market_data.index.tz is not None:
        market_data.index = market_data.index.tz_localize(None)
    market_data = market_data.sort_index()

    features = build_feature_matrix(market_data)
    if features.empty:
        raise ValueError("No features generated for backtesting")

    target = create_target(market_data["close"], horizon=horizon).reindex(features.index)
    evaluation_index = features.index[
        (features.index >= start_ts) & (features.index <= end_ts) & target.notna()
    ]

    rows = []
    for as_of in evaluation_index:
        if len(market_data.loc[market_data.index <= as_of]) < _MIN_MARKET_ROWS:
            continue

        patched_provider = _provider_for_timestep(symbol, market_data, as_of)
        with patch("tradesense.inference.orchestrator.get_market_data", new=patched_provider):
            insight = analyze_symbol(symbol)

        rows.append(
            {
                "date": as_of,
                "symbol": symbol,
                "probability_raw": float(insight["probability_raw"]),
                "probability_calibrated": float(insight["probability_calibrated"]),
                "probability": float(insight["probability_calibrated"]),
                "confidence_level": str(insight["confidence_level"]),
                "actual_outcome": int(target.loc[as_of]),
            }
        )

    if not rows:
        raise ValueError("No backtest rows produced for the requested window")

    predictions = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    predictions["predicted_label"] = (predictions["probability_calibrated"] >= 0.5).astype("int64")
    predictions["is_correct"] = (
        predictions["predicted_label"] == predictions["actual_outcome"]
    ).astype("int64")

    metrics = compute_backtest_metrics(predictions)
    reliability_payload = compute_reliability_curve(predictions)
    reliability = [ReliabilityBucket(**row) for row in reliability_payload["buckets"]]

    result = BacktestResult(
        symbol=symbol,
        start_date=start_ts.strftime("%Y-%m-%d"),
        end_date=end_ts.strftime("%Y-%m-%d"),
        horizon=int(horizon),
        total_predictions=int(len(predictions)),
        metrics=metrics,
        reliability=reliability,
    )

    return predictions, result
