# tests/test_phase6b.py
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tradesense.api as api  # noqa: E402


client = TestClient(api.app)


def _sample_response(symbol: str = "AAPL"):
    return {
        "symbol": symbol,
        "probability": 0.69,
        "probability_raw": 0.71,
        "probability_calibrated": 0.69,
        "confidence_level": "moderate",
        "confidence_reason": "Medium volatility regime reduces certainty; confidence capped at moderate.",
        "summary": "Momentum is improving with moderate risk.",
        "market_context": {
            "trend": "Uptrend",
            "momentum": "Strengthening",
            "volatility": "Moderate",
        },
        "key_drivers": ["ema20_vs_ema50", "rsi_slope_3"],
        "structured_explanation": {
            "key_drivers": ["ema20_vs_ema50", "rsi_slope_3"],
            "negative_factors": ["volume_ratio"],
            "confidence_modifiers": ["Moderate volatility regime caps confidence."],
        },
        "risk_notes": ["volatility elevated vs 90d average"],
        "model_honesty": "Model confidence is limited by short-term volatility.",
    }


def test_analyze_endpoint_success(monkeypatch):
    def _fake_analyze(symbol: str):
        return _sample_response(symbol)

    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: _fake_analyze)
    response = client.post("/analyze", json={"symbol": "AAPL"})
    assert response.status_code == 200
    assert response.json() == _sample_response("AAPL")


def test_analyze_endpoint_empty_symbol():
    response = client.post("/analyze", json={"symbol": ""})
    assert response.status_code == 400
    payload = response.json()
    assert "detail" in payload
    assert "symbol" in payload["detail"]


def test_analyze_response_schema_correctness(monkeypatch):
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol: _sample_response(symbol)))
    response = client.post("/analyze", json={"symbol": "AAPL"})
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
