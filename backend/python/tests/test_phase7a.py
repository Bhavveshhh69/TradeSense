# tests/test_phase7a.py
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tradesense.api as api  # noqa: E402
from tradesense.sentiment.aggregator import aggregate_sentiment  # noqa: E402


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


def test_sentiment_aggregation_deterministic():
    results = [
        {"label": "positive", "score": 0.9},
        {"label": "neutral", "score": 0.4},
        {"label": "negative", "score": 0.2},
    ]
    first = aggregate_sentiment(results)
    second = aggregate_sentiment(results)
    assert first == second


def test_sentiment_output_ranges():
    results = [
        {"label": "positive", "score": 0.75},
        {"label": "neutral", "score": 0.25},
    ]
    aggregated = aggregate_sentiment(results)
    assert -1.0 <= aggregated["sentiment_score"] <= 1.0
    assert aggregated["sentiment_bias"] in {"bullish", "neutral", "bearish"}
    assert aggregated["sentiment_strength"] in {"low", "medium", "high"}


def test_analyze_with_news_includes_sentiment(monkeypatch):
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol: _sample_response(symbol)))

    def _fake_analyze_texts(texts):
        return [
            {"label": "positive", "score": 0.8},
            {"label": "negative", "score": 0.2},
        ]

    def _fake_aggregate(results):
        return {
            "sentiment_score": 0.3,
            "sentiment_bias": "bullish",
            "sentiment_strength": "medium",
        }

    monkeypatch.setattr(api, "_get_sentiment_handlers", lambda: (_fake_analyze_texts, _fake_aggregate))

    response = client.post(
        "/analyze",
        json={"symbol": "AAPL", "news": ["Apple reports strong quarterly earnings"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == {
        "sentiment_score": 0.3,
        "sentiment_bias": "bullish",
        "sentiment_strength": "medium",
    }


def test_analyze_without_news_skips_sentiment(monkeypatch):
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol: _sample_response(symbol)))

    def _should_not_run():
        raise AssertionError("sentiment handlers should not be invoked")

    monkeypatch.setattr(api, "_get_sentiment_handlers", _should_not_run)

    response = client.post("/analyze", json={"symbol": "AAPL"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" not in data
    assert data == _sample_response("AAPL")
