import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tradesense.api as api  # noqa: E402
from tradesense.news import fetcher as news_fetcher  # noqa: E402
from tradesense.news.normalizer import MAX_NEWS_CHARS, normalize_news  # noqa: E402


client = TestClient(api.app)


def _sample_response(symbol: str = "AAPL", **overrides):
    payload = {
        "symbol": symbol,
        "market": "US",
        "exchange": "NASDAQ",
        "timeframe": "15m",
        "strategy_family": "orb_vwap_continuation",
        "prediction": 0,
        "probability": 0.0,
        "confidence": 0.0,
        "decision": "NO_TRADE",
        "decision_reason_type": "hard_blocker",
        "actionability_state": "blocked",
        "confidence_level": "low",
        "strength": 0.0,
        "context": {"trend_summary": "Setup not aligned", "risk_summary": "No trade"},
        "model_version": "intraday-heuristic",
        "model_name": "heuristic",
        "model_threshold": 0.55,
        "model_bench_summary": {},
        "timestamp": "2026-04-15T14:15:00+00:00",
        "generated_at": "2026-04-15T14:16:00+00:00",
        "setup_side": None,
        "entry_price": None,
        "stop_price": None,
        "take_profit_price": None,
        "forced_exit_time": None,
        "no_trade_reason": "Price has not broken the opening range",
        "promotion_gate": {"passed": True, "reason": "Promotion gate passed.", "market": "US", "artifact_timestamp": "2026-04-15T14:00:00+00:00"},
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
        "summary": "No intraday trade is being taken for the current US session.",
        "market_context": {"market": "US", "exchange": "NASDAQ", "session_window": {"start": "10:00", "end": "11:00"}},
        "key_drivers": [],
        "risk_notes": [],
        "model_honesty": "No-trade output is based on setup filters or quality gates before the probability model is allowed to act.",
        "current_price": 194.25,
        "trade_window": {"start": "10:00", "end": "11:00"},
        "threshold": 0.55,
        "base_threshold": 0.55,
        "effective_threshold": 0.55,
        "threshold_adjustment_reason": "Threshold evaluation was skipped because the setup did not advance past the deterministic gate.",
        "threshold_gap": None,
        "stock_sentiment_score": 0.0,
        "sector_sentiment_score": None,
        "contextual_sentiment_score": 0.0,
        "sentiment_confidence": 0.0,
        "sentiment_gate_reason": "Sentiment coverage is weak, so the news gate is neutral.",
        "stock_article_count": 0,
        "sector_article_count": 0,
    }
    payload.update(overrides)
    return payload


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


def test_analyze_with_manual_news_passes_news_into_python_engine(monkeypatch):
    captured = {}

    def _fake_analyze(symbol, news_texts=None):
        captured["symbol"] = symbol
        captured["news_texts"] = news_texts
        return _sample_response(symbol, stock_sentiment_score=0.6, contextual_sentiment_score=0.6, sentiment_confidence=0.8)

    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: _fake_analyze)

    response = client.post(
        "/analyze",
        json={
            "symbol": "AAPL",
            "news": ["Apple reports strong quarterly earnings"],
            "use_news": True,
        },
    )
    assert response.status_code == 200
    assert captured["symbol"] == "AAPL"
    assert captured["news_texts"] == ["Apple reports strong quarterly earnings"]
    assert response.json()["stock_sentiment_score"] == 0.6


def test_analyze_without_manual_news_does_not_require_transport_side_news_fetch(monkeypatch):
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol, news_texts=None: _sample_response(symbol)))

    response = client.post("/analyze", json={"symbol": "AAPL", "use_news": True})
    assert response.status_code == 200
    data = response.json()
    assert data["contextual_sentiment_score"] == 0.0
    assert data["sentiment_gate_reason"] == "Sentiment coverage is weak, so the news gate is neutral."
