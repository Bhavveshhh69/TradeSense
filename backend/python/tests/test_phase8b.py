# tests/test_phase8b.py
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


def test_prompt_builder_includes_required_rules():
    prompt = build_explanation_prompt(
        {
            "structured_explanation": {
                "key_drivers": ["ema20_vs_ema50"],
                "negative_factors": ["volume_ratio"],
                "confidence_modifiers": ["Elevated risk regime"],
            },
        },
        "Recent history (1 items).",
    )
    assert "Do NOT predict prices" in prompt
    assert "Do NOT provide financial advice" in prompt
    assert '"structured_explanation"' in prompt
    assert '"key_drivers"' in prompt
    assert '"negative_factors"' in prompt
    assert '"confidence_modifiers"' in prompt
    assert '"historical_context"' in prompt


def test_analyze_without_explain_skips_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADESENSE_RAG_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol: _sample_response(symbol)))

    def _should_not_call():
        raise AssertionError("explainer should not be invoked when explain=false")

    monkeypatch.setattr(api, "_get_explainer_handlers", _should_not_call)

    response = client.post("/analyze", json={"symbol": "AAPL"})
    assert response.status_code == 200
    assert "explanation" not in response.json()


def test_analyze_with_explain_adds_explanation(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADESENSE_RAG_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_get_analyze_symbol", lambda: (lambda symbol: _sample_response(symbol)))

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
