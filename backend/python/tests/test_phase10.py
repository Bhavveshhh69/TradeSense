# tests/test_phase10.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.explainability.attribution import (  # noqa: E402
    compute_attributions_from_importance,
)
from tradesense.explainability.rules import build_structured_explanation  # noqa: E402
from tradesense.reasoning_core import generate_insight  # noqa: E402


def _sample_inputs():
    return {
        "symbol": "TEST",
        "probability": 0.62,
        "feature_importance": {
            "price_vs_ema20": 0.25,
            "rsi_slope_3": 0.2,
            "macd_hist_accel": 0.15,
            "volume_ratio": 0.1,
            "price_vs_ema50": 0.05,
        },
        "feature_values": {
            "price_vs_ema20": 0.02,
            "rsi_slope_3": -0.1,
            "macd_hist_accel": 0.0,
            "volume_ratio": 1.2,
            "price_vs_ema50": -0.01,
        },
        "trend_state": 1,
        "momentum_state": 1,
        "risk_state": 2,
    }


def test_attribution_is_bounded_and_deterministic():
    feature_importance = {"alpha": 0.6, "beta": 0.4}
    feature_values = {"alpha": 1.5, "beta": -0.5}

    first = compute_attributions_from_importance(feature_importance, feature_values)
    second = compute_attributions_from_importance(feature_importance, feature_values)

    assert first == second
    assert all(-1.0 <= value <= 1.0 for value in first.values())
    assert sum(abs(value) for value in first.values()) <= 1.000001


def test_rule_engine_separates_positive_negative_confidence():
    feature_values = {
        "price_vs_ema20": 0.1,
        "price_vs_ema50": -0.2,
        "rsi_slope_3": 0.0,
    }
    attributions = {
        "price_vs_ema20": 0.6,
        "price_vs_ema50": -0.4,
        "rsi_slope_3": 0.0,
    }
    explanation = build_structured_explanation(
        feature_attributions=attributions,
        feature_values=feature_values,
        confidence_reason="High volatility regime reduces certainty.",
        trend_state=1,
        momentum_state=-1,
        risk_state=2,
    )

    assert "Price above EMA20" in explanation["key_drivers"]
    assert "Price below EMA50" in explanation["negative_factors"]
    assert "High volatility regime reduces certainty." in explanation["confidence_modifiers"]
    assert "Elevated risk regime" in explanation["confidence_modifiers"]
    assert "Trend and momentum conflict" in explanation["confidence_modifiers"]


def test_generate_insight_includes_structured_explanation():
    inputs = _sample_inputs()
    insight = generate_insight(**inputs)

    assert "structured_explanation" in insight
    structured = insight["structured_explanation"]
    assert structured["key_drivers"] == insight["key_drivers"]
    assert isinstance(structured["negative_factors"], list)
    assert isinstance(structured["confidence_modifiers"], list)
