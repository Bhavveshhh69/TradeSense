# tests/test_phase8a.py
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tradesense.api as api  # noqa: E402
from tradesense.rag.retriever import retrieve_context  # noqa: E402
from tradesense.rag.store import store_insight  # noqa: E402


client = TestClient(api.app)


def _insight(
    symbol: str,
    timestamp: str,
    probability: float,
    confidence: str,
    drivers=None,
    risks=None,
    sentiment=None,
    news=None,
):
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "probability": probability,
        "probability_raw": probability,
        "probability_calibrated": probability,
        "confidence_level": confidence,
        "confidence_reason": "Calibration unavailable; using provided probability.",
        "key_drivers": drivers or ["ema20_vs_ema50"],
        "structured_explanation": {
            "key_drivers": drivers or ["ema20_vs_ema50"],
            "negative_factors": ["volume_ratio"],
            "confidence_modifiers": ["Calibration unavailable; using provided probability."],
        },
        "risk_notes": risks or ["volatility elevated"],
        "sentiment": sentiment,
        "news_headlines": news,
        "summary": "Moderate continuation bias",
        "market_context": {
            "trend": "bullish",
            "momentum": "strengthening",
            "volatility": "medium",
        },
    }


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


def test_store_retrieve_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADESENSE_RAG_DIR", str(tmp_path))

    record1 = _insight(
        "AAPL",
        "2025-01-01T00:00:00+00:00",
        0.45,
        "low",
        drivers=["ema20_vs_ema50"],
    )
    record2 = _insight(
        "AAPL",
        "2025-02-01T00:00:00+00:00",
        0.72,
        "high",
        drivers=["ema20_vs_ema50", "rsi_slope_3"],
    )

    store_insight(record1)
    store_insight(record2)

    history = retrieve_context("AAPL", limit=5)
    assert len(history) == 2
    assert history[0]["timestamp"] == record2["timestamp"]
    assert history[1]["timestamp"] == record1["timestamp"]


def test_empty_history_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADESENSE_RAG_DIR", str(tmp_path))

    history = retrieve_context("MSFT", limit=3)
    assert history == []


def test_deterministic_retrieval(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADESENSE_RAG_DIR", str(tmp_path))

    record = _insight(
        "AAPL",
        "2025-03-01T00:00:00+00:00",
        0.55,
        "moderate",
    )
    store_insight(record)

    first = retrieve_context("AAPL", limit=5)
    second = retrieve_context("AAPL", limit=5)
    assert first == second


def test_analyze_include_context(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADESENSE_RAG_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol: _sample_response(symbol)))

    response = client.post("/analyze", json={"symbol": "AAPL"})
    assert response.status_code == 200
    assert "context" not in response.json()

    response = client.post("/analyze", json={"symbol": "AAPL", "include_context": True})
    assert response.status_code == 200
    data = response.json()
    assert data["context"]["num_items"] == 1
    assert isinstance(data["context"]["history_summary"], str)
