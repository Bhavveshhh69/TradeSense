# tests/test_phase2.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.features import build_feature_matrix  # noqa: E402


def _sample_df(rows: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = 100 + np.linspace(0, 10, num=rows)
    data = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000 + (np.arange(rows) % 10),
            "rsi": 50 + np.sin(np.linspace(0, 3.14, num=rows)) * 10,
            "ema_20": close * 0.98,
            "ema_50": close * 0.97,
            "macd": close * 0.001,
            "macd_signal": close * 0.0009,
            "macd_hist": close * 0.001 - close * 0.0009,
        },
        index=dates,
    )
    return data


def test_phase2_feature_matrix():
    df = _sample_df()
    features = build_feature_matrix(df)

    expected_columns = {
        "price_vs_ema20",
        "price_vs_ema50",
        "ema20_vs_ema50",
        "ema20_slope",
        "ema50_slope",
        "rsi_delta",
        "rsi_slope_3",
        "macd_hist_delta",
        "macd_hist_accel",
        "candle_range",
        "range_mean_14",
        "range_expansion",
        "volatility_regime",
        "volume_ratio",
        "price_volume_trend",
        "trend_state",
        "momentum_state",
        "risk_state",
    }

    assert not features.empty
    assert features.isna().sum().sum() == 0
    assert set(features.columns) == expected_columns
    assert features.shape[0] <= df.shape[0]
    assert features.index.isin(df.index).all()

    for col in features.columns:
        assert pd.api.types.is_numeric_dtype(features[col])
