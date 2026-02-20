"""Phase 12 production training pipeline with walk-forward calibration."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

# Add backend/python to path for direct script execution.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tradesense.data_provider import get_market_data
from tradesense.features import build_feature_matrix
from tradesense.modeling import create_target


SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "AMZN",
    "META",
    "TSLA",
]

EXPECTED_FEATURE_NAMES = [
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
]

MODEL_PATH = Path(__file__).resolve().parent / "xgboost.joblib"


def _prepare_symbol_dataset(symbol: str, market_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if market_df is None or market_df.empty:
        raise RuntimeError(f"No data available for {symbol}")

    features = build_feature_matrix(market_df).sort_index()
    close_prices = market_df["close"].sort_index()
    common_index = features.index.intersection(close_prices.index)
    if common_index.empty:
        raise RuntimeError(f"No common index between features and close prices for {symbol}")

    features = features.loc[common_index]
    close_prices = close_prices.loc[common_index]

    # Phase 12 requirement: use existing create_target exactly as implemented.
    target = create_target(close_prices, horizon=5)
    valid_mask = target.notna() & features.notna().all(axis=1)
    X = features.loc[valid_mask]
    y = target.loc[valid_mask].astype("int64")

    if X.empty:
        raise RuntimeError(f"No valid training rows after target alignment for {symbol}")
    if y.nunique() < 2:
        raise RuntimeError(f"Target for {symbol} must contain both classes")

    columns = list(X.columns)
    if columns != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "Feature contract mismatch. "
            f"Expected {EXPECTED_FEATURE_NAMES}, got {columns}"
        )

    return X, y


def _chronological_split(
    X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    split_index = int(len(X) * train_ratio)
    if split_index <= 0 or split_index >= len(X):
        raise RuntimeError("Not enough rows for 80/20 chronological split")

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_val = X.iloc[split_index:]
    y_val = y.iloc[split_index:]
    return X_train, X_val, y_train, y_val


def _expected_calibration_error(
    probabilities: Iterable[float], targets: Iterable[int], n_bins: int = 10
) -> float:
    probs = np.asarray(list(probabilities), dtype=float).ravel()
    y = np.asarray(list(targets), dtype=int).ravel()

    if probs.size == 0:
        return 0.0
    if probs.shape[0] != y.shape[0]:
        raise ValueError("probabilities and targets length mismatch")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    buckets = np.digitize(probs, edges[1:-1], right=True)
    total = float(probs.size)
    ece = 0.0

    for bucket_id in range(n_bins):
        mask = buckets == bucket_id
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(y[mask].mean())
        ece += (count / total) * abs(accuracy - confidence)

    return float(ece)


def train_phase12_model(
    symbols: list[str] | None = None, model_path: Path | None = None
) -> dict:
    symbols = symbols or SYMBOLS
    output_path = model_path or MODEL_PATH

    end_dt = datetime.now(tz=UTC).date()
    start_dt = end_dt - timedelta(days=(3 * 365 + 30))
    start_date = start_dt.isoformat()
    end_date = end_dt.isoformat()

    print(f"Fetching market data ({start_date} -> {end_date}) for: {symbols}")
    market_data = get_market_data(symbols, start_date, end_date, interval="1d")

    train_frames = []
    train_targets = []
    val_frames = []
    val_targets = []

    for symbol in symbols:
        if symbol not in market_data:
            raise RuntimeError(f"Data provider did not return symbol: {symbol}")
        X_symbol, y_symbol = _prepare_symbol_dataset(symbol, market_data[symbol])
        X_train_s, X_val_s, y_train_s, y_val_s = _chronological_split(
            X_symbol, y_symbol, train_ratio=0.8
        )
        train_frames.append(X_train_s)
        train_targets.append(y_train_s)
        val_frames.append(X_val_s)
        val_targets.append(y_val_s)

    X_train = pd.concat(train_frames, axis=0)
    y_train = pd.concat(train_targets, axis=0)
    X_val = pd.concat(val_frames, axis=0)
    y_val = pd.concat(val_targets, axis=0)

    if y_train.nunique() < 2:
        raise RuntimeError("Combined training target must include both classes")
    if y_val.nunique() < 2:
        raise RuntimeError("Combined validation target must include both classes")

    feature_names = list(X_train.columns)
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise RuntimeError("Combined feature_names contract mismatch")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    validation_probs = model.predict_proba(X_val)[:, 1].reshape(-1, 1)

    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(validation_probs, y_val)
    print("OK Calibration fitted")

    calibrated_probs = calibrator.predict_proba(validation_probs)[:, 1]
    accuracy = float(accuracy_score(y_val, (calibrated_probs >= 0.5).astype(int)))
    brier = float(brier_score_loss(y_val, calibrated_probs))
    ece = _expected_calibration_error(calibrated_probs, y_val.to_numpy())

    print(f"Validation accuracy: {accuracy:.6f}")
    print(f"Validation brier_score: {brier:.6f}")
    print(f"Validation ece: {ece:.6f}")

    model_bundle = {
        "model": model,
        "feature_names": feature_names,
        "calibrator": calibrator,
        "calibration_meta": {
            "method": "platt",
            "created_at": datetime.now(tz=UTC).isoformat(),
            "version": "phase12",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, output_path)

    reloaded = joblib.load(output_path)
    loaded_model = reloaded.get("model")
    loaded_calibrator = reloaded.get("calibrator")
    loaded_feature_names = reloaded.get("feature_names")

    if loaded_model is None:
        raise RuntimeError("Model bundle verification failed: model missing")
    if loaded_calibrator is None:
        raise RuntimeError("Model bundle verification failed: calibrator missing")
    if loaded_feature_names != feature_names:
        raise RuntimeError("Model bundle verification failed: feature_names mismatch")
    if not hasattr(loaded_calibrator, "predict_proba"):
        raise RuntimeError(
            "Model bundle verification failed: calibrator missing predict_proba"
        )

    print("OK Model bundle verified")
    print("OK Phase 12 training complete")
    return model_bundle


if __name__ == "__main__":
    train_phase12_model()
