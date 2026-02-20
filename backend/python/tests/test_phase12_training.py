import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.models import train_phase12  # noqa: E402


def _synthetic_market_data(rows: int = 900) -> pd.DataFrame:
    dates = pd.date_range("2022-01-03", periods=rows, freq="B")
    x = np.linspace(0.0, 36.0, rows)
    trend = np.linspace(0.0, 45.0, rows)
    close = 100.0 + trend + 2.0 * np.sin(x) + 1.0 * np.sin(2.7 * x)
    open_ = close + 0.15 * np.cos(x)
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    volume = 1_000_000 + (2_500 * np.arange(rows))

    close_s = pd.Series(close, index=dates)
    ema_12 = close_s.ewm(span=12, adjust=False).mean()
    ema_26 = close_s.ewm(span=26, adjust=False).mean()
    ema_20 = close_s.ewm(span=20, adjust=False).mean()
    ema_50 = close_s.ewm(span=50, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    delta = close_s.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    loss = -delta.clip(upper=0.0).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

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


def test_phase12_training_bundle_contract(monkeypatch, tmp_path):
    market_df = _synthetic_market_data()

    def _fake_get_market_data(symbols, start_date, end_date, interval="1d"):
        del start_date, end_date, interval
        return {str(symbol).upper(): market_df.copy() for symbol in symbols}

    monkeypatch.setattr(train_phase12, "get_market_data", _fake_get_market_data)

    output_path = tmp_path / "xgboost.joblib"
    train_phase12.train_phase12_model(model_path=output_path)

    bundle = joblib.load(output_path)

    assert isinstance(bundle, dict)
    assert bundle.get("model") is not None
    assert bundle.get("calibrator") is not None
    assert hasattr(bundle["calibrator"], "predict_proba")

    meta = bundle.get("calibration_meta", {})
    assert meta.get("version") == "phase12"
    assert meta.get("method") == "platt"

    assert bundle.get("feature_names") == train_phase12.EXPECTED_FEATURE_NAMES
