import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.api import app  # noqa: E402
from tradesense.intraday.contracts import Bar  # noqa: E402


def _bar(symbol: str, market: str, exchange: str, timezone: str, timestamp: str, close: float) -> Bar:
    ts = datetime.fromisoformat(timestamp).astimezone(ZoneInfo(timezone))
    return Bar(
        symbol=symbol,
        market=market,
        exchange=exchange,
        timezone=timezone,
        timestamp=ts,
        timeframe_min=15,
        open=close - 1.0,
        high=close + 1.0,
        low=close - 2.0,
        close=close,
        volume=1000.0,
        currency="USD" if market == "US" else "INR",
        source="mock",
        is_regular_session=True,
        session_date=ts.date(),
        vwap=close - 0.2,
        trade_count=None,
    )


def test_latest_price_endpoint_returns_latest_intraday_close(monkeypatch):
    def _mock_fetch_bars(symbol, days):
        return (
            [
                _bar("AAPL", "US", "NASDAQ", "America/New_York", "2026-04-15T10:00:00-04:00", 193.1),
                _bar("AAPL", "US", "NASDAQ", "America/New_York", "2026-04-15T10:15:00-04:00", 194.4),
            ],
            type("Profile", (), {"market": "US"})(),
        )

    monkeypatch.setattr("tradesense.api_predict._fetch_bars", _mock_fetch_bars)

    client = TestClient(app)
    response = client.get("/market/latest-price/aapl")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["market"] == "US"
    assert payload["timeframe"] == "15m"
    assert payload["price"] == 194.4
    assert payload["source"] == "intraday_close"


def test_latest_price_endpoint_supports_india_index_symbols(monkeypatch):
    def _mock_fetch_bars(symbol, days):
        return (
            [
                _bar("^NSEI", "IN", "NSE", "Asia/Kolkata", "2026-04-15T10:00:00+05:30", 23810.0),
                _bar("^NSEI", "IN", "NSE", "Asia/Kolkata", "2026-04-15T10:15:00+05:30", 23850.5),
            ],
            type("Profile", (), {"market": "IN"})(),
        )

    monkeypatch.setattr("tradesense.api_predict._fetch_bars", _mock_fetch_bars)

    client = TestClient(app)
    response = client.get("/market/latest-price/%5ENSEI")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "^NSEI"
    assert payload["market"] == "IN"
    assert payload["price"] == 23850.5


def test_latest_price_endpoint_returns_structured_not_found(monkeypatch):
    monkeypatch.setattr(
        "tradesense.api_predict._fetch_bars",
        lambda symbol, days: ([], type("Profile", (), {"market": "US"})()),
    )

    client = TestClient(app)
    response = client.get("/market/latest-price/INVALID")

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"]["error"] == "Price unavailable for symbol"
    assert payload["detail"]["symbol"] == "INVALID"


def test_market_history_endpoint_returns_intraday_close_series(monkeypatch):
    def _mock_fetch_bars(symbol, days):
        return (
            [
                _bar("AAPL", "US", "NASDAQ", "America/New_York", "2026-04-15T10:00:00-04:00", 193.1),
                _bar("AAPL", "US", "NASDAQ", "America/New_York", "2026-04-15T10:15:00-04:00", 194.4),
            ],
            type("Profile", (), {"market": "US"})(),
        )

    monkeypatch.setattr("tradesense.api_predict._fetch_bars", _mock_fetch_bars)

    client = TestClient(app)
    response = client.get("/market/history/aapl?days=30")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["market"] == "US"
    assert payload["timeframe"] == "15m"
    assert len(payload["history"]) == 2
    assert payload["history"][0]["close"] == 193.1


def test_market_history_endpoint_handles_provider_error_gracefully(monkeypatch):
    def _raise(symbol, days):
        raise RuntimeError("provider down")

    monkeypatch.setattr("tradesense.api_predict._fetch_bars", _raise)

    client = TestClient(app)
    response = client.get("/market/history/AAPL?days=30")

    assert response.status_code == 502
    payload = response.json()
    assert payload["detail"]["error"] == "Market data provider failure"
    assert payload["detail"]["symbol"] == "AAPL"
