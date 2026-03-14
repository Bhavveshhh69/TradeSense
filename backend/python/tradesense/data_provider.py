# tradesense/data_provider.py
"""Market data extraction and indicator enrichment."""

from __future__ import annotations

import warnings
from typing import Dict, Iterable

import math
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


class MarketDataError(Exception):
    """Base market data error for provider and payload issues."""


class MarketDataProviderError(MarketDataError):
    """Raised when the upstream provider request fails."""


class MarketDataUnavailableError(MarketDataError):
    """Raised when no rows are available for a symbol."""


class MarketDataResponseError(MarketDataError):
    """Raised when provider payload shape is invalid."""


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


def _resolve_close_column(df: pd.DataFrame) -> str | None:
    if "Close" in df.columns:
        return "Close"
    if "close" in df.columns:
        return "close"
    return None


def _extract_valid_close_series(df: pd.DataFrame) -> pd.Series:
    close_column = _resolve_close_column(df)
    if close_column is None:
        raise MarketDataResponseError("missing close column in market response")

    close_series = pd.to_numeric(df[close_column], errors="coerce").dropna()
    close_series = close_series[close_series > 0]
    return close_series


def _fetch_symbol_history(
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        request_kwargs = {
            "interval": interval,
            "auto_adjust": auto_adjust,
            "actions": False,
        }
        if start is not None:
            request_kwargs["start"] = start
        if end is not None:
            request_kwargs["end"] = end
        if period is not None:
            request_kwargs["period"] = period

        history = ticker.history(**request_kwargs)
    except Exception as exc:  # noqa: BLE001 - normalize provider failures for callers
        raise MarketDataProviderError(f"market provider request failed for {symbol}") from exc

    if history is None or history.empty:
        raise MarketDataUnavailableError(f"no market data for {symbol}")

    return history


def get_market_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    raise_on_error: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical market data and compute technical indicators.

    Returns a dict mapping each symbol to a pandas DataFrame with required columns.
    """

    _validate_inputs(symbols, start_date, end_date, interval)

    results: Dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            raw = _fetch_symbol_history(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
            )
        except MarketDataError as exc:
            if raise_on_error:
                raise
            warnings.warn(f"failed to download data for {symbol}: {exc}", RuntimeWarning)
            results[symbol] = _empty_frame()
            continue

        df = _normalize_columns(raw)

        missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
        if missing:
            if raise_on_error:
                raise MarketDataResponseError(f"missing columns for {symbol}: {sorted(missing)}")
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


def get_latest_price(symbol: str, *, lookback_days: int = 30, interval: str = "1d") -> float:
    normalized_symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not normalized_symbol:
        raise ValueError("symbol must be a non-empty string")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")

    history = _fetch_symbol_history(
        normalized_symbol,
        period=f"{int(lookback_days)}d",
        interval=interval,
        auto_adjust=False,
    )

    closes = _extract_valid_close_series(history)
    if closes.empty:
        raise MarketDataUnavailableError(f"no valid close prices for {normalized_symbol}")

    latest_close = float(closes.iloc[-1])
    if not math.isfinite(latest_close) or latest_close <= 0:
        raise MarketDataResponseError(f"invalid latest close value for {normalized_symbol}")

    return latest_close


def get_historical_prices(
    symbol: str,
    *,
    days: int,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    normalized_symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not normalized_symbol:
        raise ValueError("symbol must be a non-empty string")
    if days <= 0:
        raise ValueError("days must be > 0")

    history = _fetch_symbol_history(
        normalized_symbol,
        period=f"{int(days)}d",
        interval=interval,
        auto_adjust=auto_adjust,
    )
    close_column = _resolve_close_column(history)
    if close_column is None:
        raise MarketDataResponseError(f"missing close column for {normalized_symbol}")

    closes = history[[close_column]].copy()
    closes = closes.rename(columns={close_column: "close"})
    closes["close"] = pd.to_numeric(closes["close"], errors="coerce")
    closes = closes.dropna(subset=["close"])
    closes = closes[closes["close"] > 0]
    if closes.empty:
        raise MarketDataUnavailableError(f"no valid close history for {normalized_symbol}")

    closes.index = pd.to_datetime(closes.index)
    return closes
