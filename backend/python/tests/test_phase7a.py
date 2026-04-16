import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tradesense.api as api  # noqa: E402
from tradesense.sentiment.aggregator import aggregate_sentiment  # noqa: E402


client = TestClient(api.app)


def _sample_response(symbol: str = "AAPL", **overrides):
    payload = {
        "symbol": symbol,
        "market": "US",
        "exchange": "NASDAQ",
        "timeframe": "15m",
        "strategy_family": "orb_vwap_continuation",
        "prediction": 1,
        "probability": 0.61,
        "confidence": 0.06,
        "decision": "LONG",
        "confidence_level": "moderate",
        "strength": 0.06,
        "context": {"trend_summary": "ORB + VWAP aligned", "risk_summary": "Normal session risk"},
        "model_version": "intraday-xgboost",
        "model_name": "xgboost",
        "model_threshold": 0.55,
        "model_bench_summary": {"xgboost": {"validation": {"net_expectancy": 0.2}, "holdout": {"net_expectancy": 0.12}, "threshold": 0.55}},
        "timestamp": "2026-04-15T14:15:00+00:00",
        "generated_at": "2026-04-15T14:16:00+00:00",
        "setup_side": "LONG",
        "entry_price": 194.25,
        "stop_price": 192.75,
        "take_profit_price": 196.5,
        "forced_exit_time": "2026-04-15T19:45:00+00:00",
        "no_trade_reason": None,
        "data_quality": {
            "missing_bar_count": 0,
            "expected_bar_count": 25,
            "completeness_score": 1.0,
            "stale_data": False,
            "timezone_valid": True,
            "session_valid": True,
            "usable_for_live": True,
            "usable_for_backtest": True,
            "warnings": [],
        },
        "summary": "Long intraday setup detected.",
        "market_context": {"market": "US", "exchange": "NASDAQ", "session_window": {"start": "10:00", "end": "11:00"}},
        "key_drivers": ["breakout_strength"],
        "risk_notes": [],
        "model_honesty": "Setup probability only applies to the defined bracket plan.",
        "current_price": 194.25,
        "trade_window": {"start": "10:00", "end": "11:00"},
        "threshold": 0.55,
        "stock_sentiment_score": 0.0,
        "sector_sentiment_score": 0.0,
        "contextual_sentiment_score": 0.0,
        "sentiment_confidence": 0.0,
        "sentiment_gate_reason": "Sentiment coverage is weak, so the news gate is neutral.",
        "stock_article_count": 0,
        "sector_article_count": 0,
    }
    payload.update(overrides)
    return payload


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


def test_analyze_exposes_explicit_sentiment_fields(monkeypatch):
    monkeypatch.setattr(
        api,
        "_get_analyze_symbol",
        lambda: (
            lambda symbol, news_texts=None: _sample_response(
                symbol,
                stock_sentiment_score=0.5,
                sector_sentiment_score=0.2,
                contextual_sentiment_score=0.41,
                sentiment_confidence=0.74,
                sentiment_gate_reason="Company and Technology news are supportive.",
                stock_article_count=2,
                sector_article_count=3,
            )
        ),
    )

    response = client.post("/analyze", json={"symbol": "AAPL", "news": ["Apple reports strong quarterly earnings"]})
    assert response.status_code == 200
    data = response.json()
    assert data["stock_sentiment_score"] == 0.5
    assert data["sector_sentiment_score"] == 0.2
    assert data["contextual_sentiment_score"] == 0.41
    assert data["sentiment_confidence"] == 0.74
    assert data["stock_article_count"] == 2
    assert data["sector_article_count"] == 3


def test_analyze_without_manual_news_still_returns_sentiment_fields(monkeypatch):
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol, news_texts=None: _sample_response(symbol)))

    response = client.post("/analyze", json={"symbol": "AAPL"})
    assert response.status_code == 200
    data = response.json()
    assert "stock_sentiment_score" in data
    assert "contextual_sentiment_score" in data
    assert data["sentiment_gate_reason"] == "Sentiment coverage is weak, so the news gate is neutral."
