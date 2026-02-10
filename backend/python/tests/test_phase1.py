# tests/test_phase1.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.data_provider import get_market_data  # noqa: E402


def _mock_download(symbol, start=None, end=None, interval=None, auto_adjust=None, progress=None):
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    base = np.linspace(100, 200, num=len(dates))
    data = pd.DataFrame(
        {
            "Open": base,
            "High": base + 1,
            "Low": base - 1,
            "Close": base + 0.5,
            "Volume": np.full(len(dates), 1000, dtype=np.int64),
        },
        index=dates,
    )
    return data


def test_phase1_outputs(monkeypatch):
    monkeypatch.setattr("tradesense.data_provider.yf.download", _mock_download)

    symbols = ["AAA", "BBB"]
    result = get_market_data(symbols, "2020-01-01", "2020-06-01")

    assert set(result.keys()) == set(symbols)

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi",
        "ema_20",
        "ema_50",
        "macd",
        "macd_signal",
        "macd_hist",
    ]

    for symbol in symbols:
        df = result[symbol]
        assert not df.empty
        for col in required:
            assert col in df.columns
        assert df["rsi"].between(0, 100).all()
