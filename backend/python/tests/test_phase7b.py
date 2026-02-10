# tests/test_phase7b.py
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tradesense.api as api  # noqa: E402
from tradesense.news import fetcher as news_fetcher  # noqa: E402
from tradesense.news.normalizer import MAX_NEWS_CHARS, normalize_news  # noqa: E402


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


def test_normalize_news_rules():
    raw = [
        "  Hello world  ",
        "Hello world",
        "",
        "   ",
        "A" * (MAX_NEWS_CHARS + 10),
    ]
    normalized = normalize_news(raw)
    assert normalized[0] == "Hello world"
    assert normalized[1] == "A" * MAX_NEWS_CHARS
    assert len(normalized) == 2


def test_fetch_news_mocked_httpx(monkeypatch):
    class _DummyResponse:
        status_code = 200

        @staticmethod
        def json():
            return [
                {"headline": "Headline", "summary": "Summary"},
                {"headline": "Only headline", "summary": ""},
            ]

    class _DummyClient:
        def __init__(self, *args, **kwargs):
            self.requests = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None):
            self.requests.append((url, params))
            return _DummyResponse()

    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    monkeypatch.setattr(news_fetcher.httpx, "Client", _DummyClient)

    results = news_fetcher.fetch_news("AAPL", limit=10)
    assert results == ["Headline. Summary", "Only headline"]


def test_analyze_with_manual_news_takes_precedence(monkeypatch):
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol: _sample_response(symbol)))

    def _fake_analyze_texts(texts):
        return [
            {"label": "positive", "score": 0.9},
        ]

    def _fake_aggregate(results):
        return {
            "sentiment_score": 0.6,
            "sentiment_bias": "bullish",
            "sentiment_strength": "high",
        }

    monkeypatch.setattr(api, "_get_sentiment_handlers", lambda: (_fake_analyze_texts, _fake_aggregate))

    def _should_not_call():
        raise AssertionError("news fetcher should not be invoked when manual news is provided")

    monkeypatch.setattr(api, "_get_news_handlers", _should_not_call)

    response = client.post(
        "/analyze",
        json={
            "symbol": "AAPL",
            "news": ["Apple reports strong quarterly earnings"],
            "use_news": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == {
        "sentiment_score": 0.6,
        "sentiment_bias": "bullish",
        "sentiment_strength": "high",
    }


def test_analyze_with_use_news_fetches_and_scores(monkeypatch):
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol: _sample_response(symbol)))

    def _fake_fetch(symbol):
        assert symbol == "AAPL"
        return ["Headline. Summary", "Another item"]

    def _fake_normalize(texts):
        return texts

    monkeypatch.setattr(api, "_get_news_handlers", lambda: (_fake_fetch, _fake_normalize))

    def _fake_analyze_texts(texts):
        return [
            {"label": "neutral", "score": 0.6},
            {"label": "positive", "score": 0.4},
        ]

    def _fake_aggregate(results):
        return {
            "sentiment_score": 0.2,
            "sentiment_bias": "neutral",
            "sentiment_strength": "medium",
        }

    monkeypatch.setattr(api, "_get_sentiment_handlers", lambda: (_fake_analyze_texts, _fake_aggregate))

    response = client.post(
        "/analyze",
        json={"symbol": "AAPL", "use_news": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == {
        "sentiment_score": 0.2,
        "sentiment_bias": "neutral",
        "sentiment_strength": "medium",
    }


def test_analyze_without_news_or_use_news_skips_sentiment(monkeypatch):
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol: _sample_response(symbol)))

    def _should_not_run():
        raise AssertionError("news handlers should not be invoked")

    monkeypatch.setattr(api, "_get_news_handlers", _should_not_run)

    response = client.post("/analyze", json={"symbol": "AAPL"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" not in data
    assert data == _sample_response("AAPL")
