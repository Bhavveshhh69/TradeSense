# tradesense/features.py
"""Phase 2 feature engineering for TradeSense."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: Iterable[str] = (
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
)


def _validate_input(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a pure feature matrix from Phase 1 indicator outputs.

    The output contains only engineered features, aligned to the input index,
    with warm-up rows dropped to eliminate NaNs.
    """

    _validate_input(df)

    features = pd.DataFrame(index=df.index)

    # Trend Structure
    features["price_vs_ema20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
    features["price_vs_ema50"] = (df["close"] - df["ema_50"]) / df["ema_50"]
    features["ema20_vs_ema50"] = (df["ema_20"] - df["ema_50"]) / df["ema_50"]
    features["ema20_slope"] = df["ema_20"].diff()
    features["ema50_slope"] = df["ema_50"].diff()

    # Momentum Quality
    features["rsi_delta"] = df["rsi"].diff()
    features["rsi_slope_3"] = features["rsi_delta"].rolling(3).mean()
    features["macd_hist_delta"] = df["macd_hist"].diff()
    features["macd_hist_accel"] = features["macd_hist_delta"].diff()

    # Volatility & Range
    features["candle_range"] = (df["high"] - df["low"]) / df["close"]
    features["range_mean_14"] = features["candle_range"].rolling(14).mean()
    features["range_expansion"] = features["candle_range"] / features["range_mean_14"]

    # Quantile-based thresholds for volatility regime.
    vol_q33 = features["range_expansion"].quantile(0.33)
    vol_q67 = features["range_expansion"].quantile(0.67)

    volatility_regime = pd.Series(
        np.select(
            [features["range_expansion"] <= vol_q33, features["range_expansion"] <= vol_q67],
            [0, 1],
            default=2,
        ),
        index=df.index,
        dtype="float64",
    )
    volatility_regime[features["range_expansion"].isna()] = np.nan
    features["volatility_regime"] = volatility_regime

    # Volume Confirmation
    features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    features["price_volume_trend"] = np.sign(df["close"].diff()) * df["volume"]

    # Market State Encoders
    # trend_state: bullish when price and EMA20 are above EMA50, bearish when both are below.
    trend_state = pd.Series(
        np.select(
            [
                (features["price_vs_ema50"] > 0) & (features["ema20_vs_ema50"] > 0),
                (features["price_vs_ema50"] < 0) & (features["ema20_vs_ema50"] < 0),
            ],
            [1, -1],
            default=0,
        ),
        index=df.index,
        dtype="float64",
    )
    trend_state[
        features[["price_vs_ema50", "ema20_vs_ema50"]].isna().any(axis=1)
    ] = np.nan
    features["trend_state"] = trend_state

    # momentum_state: strengthening when combined momentum is non-negative, otherwise weakening.
    combined_momentum = features["rsi_slope_3"] + features["macd_hist_accel"]
    momentum_state = pd.Series(
        np.where(combined_momentum >= 0, 1, -1),
        index=df.index,
        dtype="float64",
    )
    momentum_state[combined_momentum.isna()] = np.nan
    features["momentum_state"] = momentum_state

    # risk_state: mirrors volatility regime (0=low, 1=medium, 2=high).
    features["risk_state"] = features["volatility_regime"]

    features = features.dropna(how="any")

    for col in ["volatility_regime", "trend_state", "momentum_state", "risk_state"]:
        features[col] = features[col].astype("int64")

    return features
