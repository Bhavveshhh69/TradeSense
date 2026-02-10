# tradesense/indicators.py
"""Technical indicators computed with pandas and numpy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) using Wilder's smoothing."""
    if period <= 0:
        raise ValueError("period must be a positive integer")
    if close is None:
        raise ValueError("close series is required")

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    alpha = 1 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.clip(lower=0, upper=100)

    return rsi


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute Exponential Moving Average (EMA)."""
    if span <= 0:
        raise ValueError("span must be a positive integer")
    if series is None:
        raise ValueError("series is required")

    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    if close is None:
        raise ValueError("close series is required")

    ema_12 = compute_ema(close, 12)
    ema_26 = compute_ema(close, 26)
    macd_line = ema_12 - ema_26
    signal_line = compute_ema(macd_line, 9)
    hist = macd_line - signal_line

    return macd_line, signal_line, hist
