import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.api import app  # noqa: E402


def test_validation_endpoint_returns_flattened_metrics(monkeypatch):
    class _Metrics:
        def model_dump(self):
            return {
                "overall_accuracy": 0.52,
                "accuracy_by_confidence_level": {"low": 0.48, "moderate": 0.55},
                "accuracy_by_probability_bucket": {"0.5-0.6": 0.51},
                "expected_calibration_error": 0.2,
                "brier_score": 0.29,
            }

    class _Bucket:
        def __init__(self, probability_mean, accuracy, count):
            self.probability_mean = probability_mean
            self.accuracy = accuracy
            self.count = count

        def model_dump(self):
            return {
                "probability_mean": self.probability_mean,
                "accuracy": self.accuracy,
                "count": self.count,
            }

    class _Result:
        symbol = "AAPL"
        start_date = "2025-04-02"
        end_date = "2026-04-02"
        horizon = 5
        total_predictions = 243
        metrics = _Metrics()
        reliability = [_Bucket(0.55, 0.5, 30)]

    monkeypatch.setattr("tradesense.api_predict.run_backtest", lambda **kwargs: (None, _Result()))

    client = TestClient(app)
    response = client.post("/analyze/validate", json={"symbol": "aapl"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["period"] == {
        "start_date": "2025-04-02",
        "end_date": "2026-04-02",
        "horizon": 5,
    }
    assert payload["total_predictions"] == 243
    assert payload["accuracy"] == 0.52
    assert payload["ece"] == 0.2
    assert payload["brier_score"] == 0.29
    assert payload["accuracy_by_confidence"] == {"low": 0.48, "moderate": 0.55}
    assert payload["reliability_curve"] == [
        {"probability_mean": 0.55, "accuracy": 0.5, "count": 30}
    ]
