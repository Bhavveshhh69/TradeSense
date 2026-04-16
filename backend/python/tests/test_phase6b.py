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
        "market": "US",
        "exchange": "NASDAQ",
        "timeframe": "15m",
        "strategy_family": "orb_vwap_continuation",
        "prediction": 1,
        "probability": 0.64,
        "confidence": 0.09,
        "decision": "LONG",
        "confidence_level": "moderate",
        "strength": 0.09,
        "context": {
            "trend_summary": "Intraday setup is evaluated against opening-range direction and session VWAP alignment.",
            "risk_summary": "Quality gates, bracket sizing, and sentiment gates are session-aware and market-aware.",
        },
        "model_version": "intraday-xgboost",
        "model_name": "xgboost",
        "model_threshold": 0.52,
        "model_bench_summary": {
            "xgboost": {"validation": {"net_expectancy": 0.22}, "holdout": {"net_expectancy": 0.18}, "threshold": 0.52},
            "logistic_regression": {"validation": {"net_expectancy": 0.14}, "holdout": {"net_expectancy": 0.1}, "threshold": 0.55},
        },
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
        "summary": "Long intraday setup detected from the long ORB+VWAP family with estimated same-session win probability of 64%.",
        "market_context": {
            "market": "US",
            "exchange": "NASDAQ",
            "session_window": {"start": "10:00", "end": "11:00", "opening_range_bars": 2},
            "sector": "Technology",
        },
        "key_drivers": ["breakout_strength", "vwap_distance", "relative_volume"],
        "risk_notes": [],
        "model_honesty": "The probability estimates a same-session bracket outcome for the detected setup. News sentiment can adjust the gate, but price action remains the primary alpha source.",
        "current_price": 194.25,
        "trade_window": {"start": "10:00", "end": "11:00", "opening_range_bars": 2},
        "threshold": 0.52,
        "stock_sentiment_score": 0.42,
        "sector_sentiment_score": 0.18,
        "contextual_sentiment_score": 0.348,
        "sentiment_confidence": 0.7,
        "sentiment_gate_reason": "Company and Technology news are supportive.",
        "stock_article_count": 3,
        "sector_article_count": 4,
    }


def test_analyze_endpoint_success(monkeypatch):
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol, news_texts=None: _sample_response(symbol)))
    response = client.post("/analyze", json={"symbol": "AAPL"})
    assert response.status_code == 200
    assert response.json()["decision"] == "LONG"
    assert response.json()["timeframe"] == "15m"


def test_analyze_endpoint_empty_symbol():
    response = client.post("/analyze", json={"symbol": ""})
    assert response.status_code == 400
    payload = response.json()
    assert "detail" in payload
    assert "symbol" in payload["detail"]


def test_analyze_response_schema_correctness(monkeypatch):
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol, news_texts=None: _sample_response(symbol)))
    response = client.post("/analyze", json={"symbol": "AAPL"})
    data = response.json()

    expected_keys = {
        "symbol",
        "market",
        "exchange",
        "timeframe",
        "strategy_family",
        "prediction",
        "probability",
        "confidence",
        "decision",
        "confidence_level",
        "strength",
        "context",
        "model_version",
        "model_name",
        "model_threshold",
        "model_bench_summary",
        "timestamp",
        "generated_at",
        "setup_side",
        "entry_price",
        "stop_price",
        "take_profit_price",
        "forced_exit_time",
        "no_trade_reason",
        "data_quality",
        "summary",
        "market_context",
        "key_drivers",
        "risk_notes",
        "model_honesty",
        "current_price",
        "trade_window",
        "threshold",
        "stock_sentiment_score",
        "sector_sentiment_score",
        "contextual_sentiment_score",
        "sentiment_confidence",
        "sentiment_gate_reason",
        "stock_article_count",
        "sector_article_count",
    }
    assert expected_keys.issubset(set(data.keys()))
    assert data["market"] in {"US", "IN"}
    assert data["decision"] in {"LONG", "SHORT", "NO_TRADE"}
    assert isinstance(data["data_quality"], dict)
    assert isinstance(data["key_drivers"], list)
    assert isinstance(data["risk_notes"], list)
