import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.api import app  # noqa: E402


def _mock_price_frame(values):
    index = pd.date_range("2026-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"close": values}, index=index)


def test_latest_price_endpoint_returns_latest_close(monkeypatch):
    def _mock_get_market_data(symbols, start_date, end_date, interval="1d"):
        return {symbols[0]: _mock_price_frame([100.5, 101.75, 103.25])}

    monkeypatch.setattr("tradesense.api_predict.get_market_data", _mock_get_market_data)

    client = TestClient(app)
    response = client.get("/market/latest-price/aapl")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["price"] == 103.25
    assert payload["source"] == "close"
    assert isinstance(payload["timestamp"], str) and payload["timestamp"]


def test_latest_price_endpoint_returns_structured_not_found(monkeypatch):
    def _mock_get_market_data(symbols, start_date, end_date, interval="1d"):
        return {symbols[0]: pd.DataFrame(columns=["close"])}

    monkeypatch.setattr("tradesense.api_predict.get_market_data", _mock_get_market_data)

    client = TestClient(app)
    response = client.get("/market/latest-price/INVALID")

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"]["error"] == "Price unavailable for symbol"
    assert payload["detail"]["symbol"] == "INVALID"


def test_market_history_endpoint_returns_close_series(monkeypatch):
    class _MockTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, interval, auto_adjust):
            assert period == "30d"
            assert interval == "1d"
            assert auto_adjust is True
            return _mock_price_frame([99.25, 101.75, 103.5])

    monkeypatch.setattr("tradesense.api_predict.yf.Ticker", _MockTicker)

    client = TestClient(app)
    response = client.get("/market/history/aapl?days=30")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["history"] == [
        {"date": "2026-01-01", "close": 99.25},
        {"date": "2026-01-02", "close": 101.75},
        {"date": "2026-01-03", "close": 103.5},
    ]


def test_market_history_endpoint_returns_structured_not_found(monkeypatch):
    class _MockTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, interval, auto_adjust):
            return pd.DataFrame(columns=["Close"])

    monkeypatch.setattr("tradesense.api_predict.yf.Ticker", _MockTicker)

    client = TestClient(app)
    response = client.get("/market/history/INVALID?days=30")

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"]["error"] == "Price history unavailable for symbol"
    assert payload["detail"]["symbol"] == "INVALID"


def test_market_history_endpoint_drops_nan_rows(monkeypatch):
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    frame = pd.DataFrame({"Close": [101.0, float("nan"), 103.5]}, index=index)

    class _MockTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, interval, auto_adjust):
            return frame

    monkeypatch.setattr("tradesense.api_predict.yf.Ticker", _MockTicker)

    client = TestClient(app)
    response = client.get("/market/history/aapl?days=30")

    assert response.status_code == 200
    payload = response.json()
    assert payload["history"] == [
        {"date": "2026-01-01", "close": 101.0},
        {"date": "2026-01-03", "close": 103.5},
    ]
