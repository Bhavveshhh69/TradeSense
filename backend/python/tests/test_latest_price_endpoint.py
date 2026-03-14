import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.api import app  # noqa: E402
from tradesense.data_provider import MarketDataProviderError, MarketDataUnavailableError  # noqa: E402


def _mock_price_frame(values):
    index = pd.date_range("2026-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"close": values}, index=index)


def test_latest_price_endpoint_returns_latest_close(monkeypatch):
    captured = {}

    def _mock_get_latest_price(symbol, lookback_days=30, interval="1d"):
        captured["symbol"] = symbol
        captured["lookback_days"] = lookback_days
        captured["interval"] = interval
        return 103.25

    monkeypatch.setattr("tradesense.api_predict.get_latest_price", _mock_get_latest_price)

    client = TestClient(app)
    response = client.get("/market/latest-price/aapl")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["price"] == 103.25
    assert payload["source"] == "close"
    assert isinstance(payload["timestamp"], str) and payload["timestamp"]
    assert captured["symbol"] == "AAPL"
    assert captured["lookback_days"] == 220
    assert captured["interval"] == "1d"


def test_latest_price_endpoint_returns_structured_not_found(monkeypatch):
    def _mock_get_latest_price(symbol, lookback_days=30, interval="1d"):
        raise MarketDataUnavailableError(f"no price for {symbol}")

    monkeypatch.setattr("tradesense.api_predict.get_latest_price", _mock_get_latest_price)

    client = TestClient(app)
    response = client.get("/market/latest-price/INVALID")

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"]["error"] == "Price unavailable for symbol"
    assert payload["detail"]["symbol"] == "INVALID"


def test_latest_price_endpoint_does_not_reuse_previous_symbol_price(monkeypatch):
    calls = []
    prices = {
        "AAPL": 250.0,
        "NVDA": 910.5,
        "TCS.NS": 4102.35,
    }

    def _mock_get_latest_price(symbol, lookback_days=30, interval="1d"):
        calls.append(symbol)
        return prices[symbol]

    monkeypatch.setattr("tradesense.api_predict.get_latest_price", _mock_get_latest_price)

    client = TestClient(app)
    aapl = client.get("/market/latest-price/AAPL")
    nvda = client.get("/market/latest-price/NVDA")
    tcs = client.get("/market/latest-price/TCS.NS")

    assert aapl.status_code == 200
    assert nvda.status_code == 200
    assert tcs.status_code == 200
    assert aapl.json()["price"] == 250.0
    assert nvda.json()["price"] == 910.5
    assert tcs.json()["price"] == 4102.35
    assert calls == ["AAPL", "NVDA", "TCS.NS"]


def test_latest_price_endpoint_returns_structured_provider_failure(monkeypatch):
    def _mock_get_latest_price(symbol, lookback_days=30, interval="1d"):
        raise MarketDataProviderError(f"provider down for {symbol}")

    monkeypatch.setattr("tradesense.api_predict.get_latest_price", _mock_get_latest_price)

    client = TestClient(app)
    response = client.get("/market/latest-price/NVDA")

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"]["error"] == "Market data provider failure"
    assert payload["detail"]["symbol"] == "NVDA"


def test_market_history_endpoint_returns_close_series(monkeypatch):
    def _mock_get_historical_prices(symbol, days, interval="1d", auto_adjust=True):
        assert symbol == "AAPL"
        assert days == 30
        assert interval == "1d"
        assert auto_adjust is True
        return _mock_price_frame([99.25, 101.75, 103.5])

    monkeypatch.setattr("tradesense.api_predict.get_historical_prices", _mock_get_historical_prices)

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
    def _mock_get_historical_prices(symbol, days, interval="1d", auto_adjust=True):
        raise MarketDataUnavailableError(f"no history for {symbol}")

    monkeypatch.setattr("tradesense.api_predict.get_historical_prices", _mock_get_historical_prices)

    client = TestClient(app)
    response = client.get("/market/history/INVALID?days=30")

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"]["error"] == "Price history unavailable for symbol"
    assert payload["detail"]["symbol"] == "INVALID"


def test_market_history_endpoint_handles_provider_error_gracefully(monkeypatch):
    def _mock_get_historical_prices(symbol, days, interval="1d", auto_adjust=True):
        raise MarketDataProviderError(f"provider down for {symbol}")

    monkeypatch.setattr("tradesense.api_predict.get_historical_prices", _mock_get_historical_prices)

    client = TestClient(app)
    response = client.get("/market/history/AAPL?days=30")

    assert response.status_code == 502
    payload = response.json()
    assert payload["detail"]["error"] == "Market data provider failure"
    assert payload["detail"]["symbol"] == "AAPL"


def test_market_history_endpoint_drops_nan_rows(monkeypatch):
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    frame = pd.DataFrame({"close": [101.0, float("nan"), 103.5]}, index=index)

    def _mock_get_historical_prices(symbol, days, interval="1d", auto_adjust=True):
        return frame

    monkeypatch.setattr("tradesense.api_predict.get_historical_prices", _mock_get_historical_prices)

    client = TestClient(app)
    response = client.get("/market/history/aapl?days=30")

    assert response.status_code == 200
    payload = response.json()
    assert payload["history"] == [
        {"date": "2026-01-01", "close": 101.0},
        {"date": "2026-01-03", "close": 103.5},
    ]
