"""Aggregate FinBERT sentiment results into a deterministic sentiment object."""

from __future__ import annotations

from typing import List

_LABEL_SCORE = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}

# Fixed thresholds for bias and strength classification.
_BIAS_THRESHOLD = 0.15  # >= 0.15 bullish, <= -0.15 bearish
_STRENGTH_LOW_MAX = 0.20  # < 0.20 => low
_STRENGTH_MEDIUM_MAX = 0.50  # < 0.50 => medium, else high

# Neutral dampening factor (applied based on neutral weight share).
_NEUTRAL_DAMPENING_MAX = 0.50  # max 50% dampening when all inputs are neutral


def aggregate_sentiment(results: List[dict]) -> dict:
    """Aggregate multiple FinBERT outputs into a single sentiment object.

    Uses confidence-weighted averaging with neutral dampening and fixed
    thresholds for bias/strength classification.
    """
    if results is None:
        raise ValueError("results must be a list of sentiment dicts")
    if not isinstance(results, list):
        raise ValueError("results must be a list of sentiment dicts")
    if not results:
        return {
            "sentiment_score": 0.0,
            "sentiment_bias": "neutral",
            "sentiment_strength": "low",
        }

    total_weight = 0.0
    weighted_sum = 0.0
    neutral_weight = 0.0

    for item in results:
        if not isinstance(item, dict):
            raise ValueError("results must be a list of sentiment dicts")
        label = str(item.get("label", "neutral")).lower()
        score = float(item.get("score", 0.0))
        score = max(0.0, min(1.0, score))

        weight = score
        total_weight += weight
        weighted_sum += _LABEL_SCORE.get(label, 0.0) * weight
        if label == "neutral":
            neutral_weight += weight

    if total_weight <= 0:
        return {
            "sentiment_score": 0.0,
            "sentiment_bias": "neutral",
            "sentiment_strength": "low",
        }

    base_score = weighted_sum / total_weight
    neutral_ratio = neutral_weight / total_weight
    dampening = 1.0 - (_NEUTRAL_DAMPENING_MAX * neutral_ratio)
    sentiment_score = base_score * dampening
    sentiment_score = max(-1.0, min(1.0, sentiment_score))

    if sentiment_score >= _BIAS_THRESHOLD:
        bias = "bullish"
    elif sentiment_score <= -_BIAS_THRESHOLD:
        bias = "bearish"
    else:
        bias = "neutral"

    magnitude = abs(sentiment_score)
    if magnitude < _STRENGTH_LOW_MAX:
        strength = "low"
    elif magnitude < _STRENGTH_MEDIUM_MAX:
        strength = "medium"
    else:
        strength = "high"

    return {
        "sentiment_score": float(sentiment_score),
        "sentiment_bias": bias,
        "sentiment_strength": strength,
    }
