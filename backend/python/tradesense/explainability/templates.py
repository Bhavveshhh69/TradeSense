# tradesense/explainability/templates.py
"""Feature label templates for rule-based explanations."""

from __future__ import annotations

from typing import Dict, Tuple

FEATURE_TEMPLATES: Dict[str, Tuple[str, str, str]] = {
    "price_vs_ema20": ("Price above EMA20", "Price below EMA20", "Price near EMA20"),
    "price_vs_ema50": ("Price above EMA50", "Price below EMA50", "Price near EMA50"),
    "ema20_vs_ema50": ("EMA20 above EMA50", "EMA20 below EMA50", "EMA20 near EMA50"),
    "ema20_slope": ("EMA20 rising", "EMA20 falling", "EMA20 flat"),
    "ema50_slope": ("EMA50 rising", "EMA50 falling", "EMA50 flat"),
    "rsi_delta": ("RSI rising", "RSI falling", "RSI flat"),
    "rsi_slope_3": (
        "RSI momentum strengthening",
        "RSI momentum weakening",
        "RSI momentum flat",
    ),
    "macd_hist_delta": (
        "MACD histogram rising",
        "MACD histogram falling",
        "MACD histogram flat",
    ),
    "macd_hist_accel": (
        "MACD momentum accelerating",
        "MACD momentum decelerating",
        "MACD momentum flat",
    ),
    "candle_range": (
        "Candle range expanding",
        "Candle range contracting",
        "Candle range steady",
    ),
    "range_mean_14": (
        "Average range increasing",
        "Average range decreasing",
        "Average range steady",
    ),
    "range_expansion": (
        "Range expansion above average",
        "Range expansion below average",
        "Range expansion neutral",
    ),
    "volume_ratio": ("Volume above average", "Volume below average", "Volume near average"),
    "price_volume_trend": (
        "Price-volume alignment positive",
        "Price-volume alignment negative",
        "Price-volume alignment neutral",
    ),
}

CATEGORICAL_FEATURE_MAP: Dict[str, Dict[int, str]] = {
    "trend_state": {-1: "Trend regime bearish", 0: "Trend regime sideways", 1: "Trend regime bullish"},
    "momentum_state": {-1: "Momentum regime weakening", 1: "Momentum regime strengthening"},
    "risk_state": {0: "Risk regime low", 1: "Risk regime medium", 2: "Risk regime high"},
    "volatility_regime": {
        0: "Volatility regime low",
        1: "Volatility regime medium",
        2: "Volatility regime high",
    },
}


def explain_feature(name: str, value: float) -> str:
    """Convert a feature name/value into a human-readable statement."""
    if name in CATEGORICAL_FEATURE_MAP:
        mapping = CATEGORICAL_FEATURE_MAP[name]
        key = int(value) if int(value) == value else value
        return mapping.get(key, f"{name} regime unmapped")

    if name in FEATURE_TEMPLATES:
        positive, negative, neutral = FEATURE_TEMPLATES[name]
        if value > 0:
            return positive
        if value < 0:
            return negative
        return neutral

    if value > 0:
        return f"{name} positive"
    if value < 0:
        return f"{name} negative"
    return f"{name} neutral"
