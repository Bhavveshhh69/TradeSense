import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tradesense.api as api  # noqa: E402
from tradesense.rag.retriever import retrieve_context  # noqa: E402
from tradesense.rag.store import store_insight  # noqa: E402


client = TestClient(api.app)


def _insight(symbol: str, timestamp: str):
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "probability": 0.62,
        "confidence_level": "moderate",
        "key_drivers": ["breakout_strength"],
        "risk_notes": ["range expansion elevated"],
        "summary": "Intraday setup is live.",
        "market_context": {"market": "US", "exchange": "NASDAQ", "session_window": {"start": "10:00", "end": "11:00"}},
    }


def _sample_response(symbol: str = "AAPL"):
    return {
        "symbol": symbol,
        "market": "US",
        "exchange": "NASDAQ",
        "timeframe": "15m",
        "strategy_family": "orb_vwap_continuation",
        "prediction": 1,
        "probability": 0.62,
        "confidence": 0.07,
        "decision": "LONG",
        "confidence_level": "moderate",
        "strength": 0.07,
        "context": {"trend_summary": "ORB + VWAP aligned", "risk_summary": "Normal session risk"},
        "model_version": "intraday-xgboost",
        "model_name": "xgboost",
        "model_threshold": 0.55,
        "model_bench_summary": {"xgboost": {"validation": {"net_expectancy": 0.21}, "holdout": {"net_expectancy": 0.15}, "threshold": 0.55}},
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
        "stock_sentiment_score": 0.21,
        "sector_sentiment_score": 0.14,
        "contextual_sentiment_score": 0.189,
        "sentiment_confidence": 0.62,
        "sentiment_gate_reason": "Company and Technology news are supportive.",
        "stock_article_count": 2,
        "sector_article_count": 2,
    }


def test_store_retrieve_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADESENSE_RAG_DIR", str(tmp_path))

    store_insight(_insight("AAPL", "2025-01-01T00:00:00+00:00"))
    store_insight(_insight("AAPL", "2025-02-01T00:00:00+00:00"))

    history = retrieve_context("AAPL", limit=5)
    assert len(history) == 2
    assert history[0]["timestamp"] == "2025-02-01T00:00:00+00:00"


def test_empty_history_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADESENSE_RAG_DIR", str(tmp_path))

    history = retrieve_context("MSFT", limit=3)
    assert history == []


def test_deterministic_retrieval(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADESENSE_RAG_DIR", str(tmp_path))

    store_insight(_insight("AAPL", "2025-03-01T00:00:00+00:00"))

    first = retrieve_context("AAPL", limit=5)
    second = retrieve_context("AAPL", limit=5)
    assert first == second


def test_analyze_include_context(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADESENSE_RAG_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol, news_texts=None: _sample_response(symbol)))

    response = client.post("/analyze", json={"symbol": "AAPL"})
    assert response.status_code == 200
    assert "context" in response.json()

    response = client.post("/analyze", json={"symbol": "AAPL", "include_context": True})
    assert response.status_code == 200
    data = response.json()
    assert data["context"]["num_items"] == 1
    assert isinstance(data["context"]["history_summary"], str)
