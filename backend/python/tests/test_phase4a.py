# tests/test_phase4a.py
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
        "risk_state": 1,
    }


def test_output_schema_correctness():
    inputs = _sample_inputs()
    insight = generate_insight(**inputs)

    expected_keys = {
        "symbol",
        "probability",
        "probability_raw",
        "probability_calibrated",
        "confidence_level",
        "confidence_reason",
        "summary",
        "market_context",
        "key_drivers",
        "structured_explanation",
        "risk_notes",
        "model_honesty",
    }
    assert set(insight.keys()) == expected_keys
    assert isinstance(insight["symbol"], str)
    assert isinstance(insight["probability"], float)
    assert isinstance(insight["probability_raw"], float)
    assert isinstance(insight["probability_calibrated"], float)
    assert isinstance(insight["confidence_level"], str)
    assert isinstance(insight["confidence_reason"], str)
    assert isinstance(insight["summary"], str)
    assert isinstance(insight["model_honesty"], str)

    market_context = insight["market_context"]
    assert isinstance(market_context, dict)
    assert set(market_context.keys()) == {"trend", "momentum", "volatility"}
    assert all(isinstance(value, str) for value in market_context.values())

    assert isinstance(insight["key_drivers"], list)
    assert all(isinstance(item, str) for item in insight["key_drivers"])

    structured = insight["structured_explanation"]
    assert isinstance(structured, dict)
    assert set(structured.keys()) == {
        "key_drivers",
        "negative_factors",
        "confidence_modifiers",
    }
    assert all(isinstance(item, str) for item in structured["key_drivers"])
    assert all(isinstance(item, str) for item in structured["negative_factors"])
    assert all(isinstance(item, str) for item in structured["confidence_modifiers"])

    assert isinstance(insight["risk_notes"], list)
    assert all(isinstance(item, str) for item in insight["risk_notes"])


@pytest.mark.parametrize(
    ("probability", "risk_state", "expected"),
    [
        (0.54, 1, "low"),
        (0.55, 1, "moderate"),
        (0.65, 1, "moderate"),
        (0.66, 0, "high"),
        (0.66, 2, "moderate"),
    ],
)
def test_confidence_mapping(probability, risk_state, expected):
    inputs = _sample_inputs()
    inputs["probability"] = probability
    inputs["risk_state"] = risk_state
    insight = generate_insight(**inputs)
    assert insight["confidence_level"] == expected


def test_deterministic_output_for_same_input():
    inputs = _sample_inputs()
    first = generate_insight(**inputs)
    second = generate_insight(**inputs)
    assert first == second


def test_risk_notes_appear_when_expected():
    inputs = _sample_inputs()
    inputs["trend_state"] = 1
    inputs["momentum_state"] = -1
    inputs["risk_state"] = 2
    insight = generate_insight(**inputs)

    assert "Elevated risk regime" in insight["risk_notes"]
    assert "Trend and momentum conflict" in insight["risk_notes"]


def test_no_empty_explanation_fields():
    inputs = _sample_inputs()
    inputs["trend_state"] = -1
    inputs["momentum_state"] = 1
    inputs["risk_state"] = 2
    insight = generate_insight(**inputs)

    assert insight["summary"].strip()
    assert insight["confidence_reason"].strip()
    assert insight["model_honesty"].strip()
    assert all(value.strip() for value in insight["market_context"].values())
    assert all(item.strip() for item in insight["key_drivers"])
    assert all(item.strip() for item in insight["risk_notes"])
    assert all(
        item.strip() for item in insight["structured_explanation"]["confidence_modifiers"]
    )
