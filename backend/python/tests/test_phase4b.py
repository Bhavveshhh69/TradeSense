# tests/test_phase4b.py
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.api import app  # noqa: E402


client = TestClient(app)


def _sample_payload():
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


def test_reason_endpoint_success():
    response = client.post("/reason", json=_sample_payload())
    assert response.status_code == 200


def test_response_schema_correctness():
    response = client.post("/reason", json=_sample_payload())
    data = response.json()

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
    assert set(data.keys()) == expected_keys
    assert isinstance(data["symbol"], str)
    assert isinstance(data["probability"], float)
    assert isinstance(data["probability_raw"], float)
    assert isinstance(data["probability_calibrated"], float)
    assert isinstance(data["confidence_level"], str)
    assert isinstance(data["confidence_reason"], str)
    assert isinstance(data["summary"], str)
    assert isinstance(data["model_honesty"], str)

    market_context = data["market_context"]
    assert isinstance(market_context, dict)
    assert set(market_context.keys()) == {"trend", "momentum", "volatility"}
    assert all(isinstance(value, str) for value in market_context.values())

    assert isinstance(data["key_drivers"], list)
    assert all(isinstance(item, str) for item in data["key_drivers"])

    structured = data["structured_explanation"]
    assert isinstance(structured, dict)
    assert set(structured.keys()) == {
        "key_drivers",
        "negative_factors",
        "confidence_modifiers",
    }
    assert all(isinstance(item, str) for item in structured["key_drivers"])
    assert all(isinstance(item, str) for item in structured["negative_factors"])
    assert all(isinstance(item, str) for item in structured["confidence_modifiers"])

    assert isinstance(data["risk_notes"], list)
    assert all(isinstance(item, str) for item in data["risk_notes"])


def test_deterministic_output_for_same_input():
    payload = _sample_payload()
    first = client.post("/reason", json=payload).json()
    second = client.post("/reason", json=payload).json()
    assert first == second
