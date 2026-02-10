# tradesense/data_provider.py
"""Market data extraction and indicator enrichment."""

from __future__ import annotations

import warnings
from typing import Dict, Iterable

import pandas as pd
import yfinance as yf

from .indicators import compute_ema, compute_macd, compute_rsi


_REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi",
    "ema_20",
    "ema_50",
    "macd",
    "macd_signal",
    "macd_hist",
]


def _validate_inputs(symbols: Iterable[str], start_date: str, end_date: str, interval: str) -> None:
    if not isinstance(symbols, (list, tuple)):
        raise ValueError("symbols must be a list of strings")
    if len(symbols) == 0:
        raise ValueError("symbols must not be empty")
    if not all(isinstance(sym, str) and sym.strip() for sym in symbols):
        raise ValueError("each symbol must be a non-empty string")
    if not isinstance(interval, str) or not interval.strip():
        raise ValueError("interval must be a non-empty string")

    try:
        start = pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise")
        end = pd.to_datetime(end_date, format="%Y-%m-%d", errors="raise")
    except Exception as exc:  # noqa: BLE001 - enforce clear error for callers
        raise ValueError("start_date and end_date must be in YYYY-MM-DD format") from exc

    if start > end:
        raise ValueError("start_date must be on or before end_date")


def _empty_frame() -> pd.DataFrame:
    df = pd.DataFrame(columns=_REQUIRED_COLUMNS)
    df.index = pd.DatetimeIndex([], name="date")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if "Open" in df.columns.get_level_values(0):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        else:
            df = df.copy()
            df.columns = df.columns.get_level_values(-1)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    return df


def get_market_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical market data and compute technical indicators.

    Returns a dict mapping each symbol to a pandas DataFrame with required columns.
    """

    _validate_inputs(symbols, start_date, end_date, interval)

    results: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            raw = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:  # noqa: BLE001 - per-symbol isolation
            warnings.warn(f"failed to download data for {symbol}: {exc}", RuntimeWarning)
            results[symbol] = _empty_frame()
            continue

        if raw is None or raw.empty:
            warnings.warn(f"no data returned for {symbol}", RuntimeWarning)
            results[symbol] = _empty_frame()
            continue

        df = _normalize_columns(raw)

        missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
        if missing:
            warnings.warn(
                f"missing columns for {symbol}: {sorted(missing)}", RuntimeWarning
            )
            results[symbol] = _empty_frame()
            continue

        df = df[["open", "high", "low", "close", "volume"]].copy()

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "date"

        df["rsi"] = compute_rsi(df["close"], 14)
        df["ema_20"] = compute_ema(df["close"], 20)
        df["ema_50"] = compute_ema(df["close"], 50)
        macd_line, signal_line, hist = compute_macd(df["close"])
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = hist

        df["date"] = df.index

        df = df[_REQUIRED_COLUMNS]

        df = df.dropna(subset=_REQUIRED_COLUMNS)

        results[symbol] = df

    return results
