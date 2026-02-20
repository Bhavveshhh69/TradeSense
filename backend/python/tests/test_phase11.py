import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.backtesting.backtest_engine import run_backtest  # noqa: E402
from tradesense.backtesting import backtest_engine  # noqa: E402


def _synthetic_market_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=160, freq="B")
    x = np.linspace(0.0, 12.0, len(dates))

    close = 100.0 + (0.2 * np.arange(len(dates))) + 2.0 * np.sin(x)
    open_ = close + 0.1 * np.cos(x)
    high = np.maximum(open_, close) + 0.6
    low = np.minimum(open_, close) - 0.6
    volume = 1_000_000 + (2_000 * np.arange(len(dates)))

    close_s = pd.Series(close, index=dates)
    ema_12 = close_s.ewm(span=12, adjust=False).mean()
    ema_26 = close_s.ewm(span=26, adjust=False).mean()
    ema_20 = close_s.ewm(span=20, adjust=False).mean()
    ema_50 = close_s.ewm(span=50, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    price_delta = close_s.diff()
    gain = price_delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    loss = -price_delta.clip(upper=0.0).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "rsi": rsi.values,
            "ema_20": ema_20.values,
            "ema_50": ema_50.values,
            "macd": macd.values,
            "macd_signal": macd_signal.values,
            "macd_hist": macd_hist.values,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def test_phase11_backtest_metrics_and_bounds(monkeypatch):
    market_data = _synthetic_market_data()

    def _fake_get_market_data(symbols, start_date, end_date, interval="1d"):
        del start_date, end_date, interval
        return {str(symbol).upper(): market_data.copy() for symbol in symbols}

    monkeypatch.setattr(backtest_engine, "get_market_data", _fake_get_market_data)

    predictions, result = run_backtest(
        symbol="AAPL",
        start_date="2024-05-20",
        end_date="2024-06-14",
        horizon=5,
    )

    assert not predictions.empty
    assert {
        "probability_raw",
        "probability_calibrated",
        "confidence_level",
        "actual_outcome",
    }.issubset(predictions.columns)
    assert result.total_predictions == len(predictions)
    assert 0.0 <= result.metrics.overall_accuracy <= 1.0
    assert 0.0 <= result.metrics.expected_calibration_error <= 1.0
    assert 0.0 <= result.metrics.brier_score <= 1.0

    for value in result.metrics.accuracy_by_confidence_level.values():
        assert 0.0 <= value <= 1.0
    for value in result.metrics.accuracy_by_probability_bucket.values():
        assert 0.0 <= value <= 1.0

    assert len(result.reliability) > 0
    assert sum(bucket.count for bucket in result.reliability) == len(predictions)
    for bucket in result.reliability:
        assert 0.0 <= bucket.probability_mean <= 1.0
        assert 0.0 <= bucket.accuracy <= 1.0
