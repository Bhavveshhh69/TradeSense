import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tradesense.api as api  # noqa: E402
from tradesense.explainer.prompt_builder import build_explanation_prompt  # noqa: E402


client = TestClient(api.app)


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


def test_prompt_builder_includes_required_rules():
    prompt = build_explanation_prompt(
        {
            "structured_explanation": {
                "key_drivers": ["breakout_strength"],
                "negative_factors": ["relative_volume"],
                "confidence_modifiers": ["Elevated range expansion"],
            },
        },
        "Recent history (1 items).",
    )
    assert "Do NOT predict prices" in prompt
    assert "Do NOT provide financial advice" in prompt
    assert '"structured_explanation"' in prompt


def test_analyze_without_explain_skips_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADESENSE_RAG_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol, news_texts=None: _sample_response(symbol)))

    def _should_not_call():
        raise AssertionError("explainer should not be invoked when explain=false")

    monkeypatch.setattr(api, "_get_explainer_handlers", _should_not_call)

    response = client.post("/analyze", json={"symbol": "AAPL"})
    assert response.status_code == 200
    assert "explanation" not in response.json()


def test_analyze_with_explain_adds_explanation(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADESENSE_RAG_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol, news_texts=None: _sample_response(symbol)))

    base = client.post("/analyze", json={"symbol": "AAPL"}).json()

    def _fake_build_prompt(insight, context_summary):
        assert isinstance(insight, dict)
        assert isinstance(context_summary, str)
        return "PROMPT"

    def _fake_generate(prompt):
        assert prompt == "PROMPT"
        return {
            "summary": "Summary text",
            "narrative": "Narrative text",
            "disclaimer": "Disclaimer text",
        }

    monkeypatch.setattr(api, "_get_explainer_handlers", lambda: (_fake_build_prompt, _fake_generate))

    response = client.post("/analyze", json={"symbol": "AAPL", "explain": True})
    assert response.status_code == 200
    data = response.json()
    assert data["explanation"] == {
        "summary": "Summary text",
        "narrative": "Narrative text",
        "disclaimer": "Disclaimer text",
    }
    for key, value in base.items():
        assert data[key] == value
