import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.api import app  # noqa: E402


def test_validation_endpoint_returns_flattened_metrics(monkeypatch):
    monkeypatch.setattr(
        "tradesense.api_predict.ENGINE.validate_symbol",
        lambda **kwargs: {
            "symbol": "AAPL",
            "market": "US",
            "timeframe": "15m",
            "period": {
                "start_date": "2026-03-02",
                "end_date": "2026-04-02",
                "horizon": 1,
            },
            "total_predictions": 243,
            "accuracy": 0.52,
            "ece": 0.2,
            "brier_score": 0.29,
            "accuracy_by_confidence": {"low": 0.48, "moderate": 0.55},
            "reliability_curve": [{"probability_mean": 0.55, "accuracy": 0.5, "count": 30}],
            "trade_metrics": {
                "trade_count": 18,
                "eligible_session_coverage": 0.42,
                "average_r_multiple": 0.14,
                "base_net_expectancy": 0.12,
                "net_expectancy": 0.09,
                "profit_factor": 1.33,
                "win_rate": 0.56,
                "wilson_lower_bound": 0.51,
                "max_drawdown": 1.2,
            },
            "regime_breakdown": {
                "volatility": {"normal": {"sessions": 12, "trade_count": 8, "win_rate": 0.63, "net_expectancy": 0.11, "profit_factor": 1.4}},
                "trend": {"bullish": {"sessions": 9, "trade_count": 6, "win_rate": 0.67, "net_expectancy": 0.13, "profit_factor": 1.5}},
            },
            "cost_assumptions": {
                "market": "US",
                "entry_slippage_bps": 4.0,
                "exit_slippage_bps": 5.0,
                "borrow_bps_short_only": 1.0,
                "round_trip_cost_r": 0.02,
                "stress_cost_multiplier": 1.75,
                "stressed_round_trip_cost_r": 0.035,
            },
            "sample_quality": {
                "total_sessions": 30,
                "eligible_sessions": 20,
                "traded_sessions": 18,
                "skipped_sessions": 12,
                "survivorship_limited_universe": True,
                "survivorship_note": "Bootstrap universe is liquid but survivorship-limited and should not be treated as survivorship-bias-free.",
                "multiple_testing_search_space": {"models_tested": 3, "thresholds_tested": 11, "policy_variants_tested": 8, "total_configurations": 264},
                "execution_assumption": "Signals are generated on bar t and evaluated with next-bar-open entry. Same-bar fills are not allowed.",
                "data_quality": {"usable_for_live": True},
            },
            "promotion_gate": {
                "passed": True,
                "reason": "Promotion gate passed.",
                "market": "US",
                "artifact_timestamp": "2026-04-02T14:00:00+00:00",
            },
        },
    )

    client = TestClient(app)
    response = client.post("/analyze/validate", json={"symbol": "aapl"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["market"] == "US"
    assert payload["timeframe"] == "15m"
    assert payload["period"] == {
        "start_date": "2026-03-02",
        "end_date": "2026-04-02",
        "horizon": 1,
    }
    assert payload["total_predictions"] == 243
    assert payload["accuracy"] == 0.52
    assert payload["ece"] == 0.2
    assert payload["brier_score"] == 0.29
    assert payload["accuracy_by_confidence"] == {"low": 0.48, "moderate": 0.55}
    assert payload["reliability_curve"] == [
        {"probability_mean": 0.55, "accuracy": 0.5, "count": 30}
    ]
    assert payload["trade_metrics"]["trade_count"] == 18
    assert payload["cost_assumptions"]["stress_cost_multiplier"] == 1.75
    assert payload["promotion_gate"]["passed"] is True
