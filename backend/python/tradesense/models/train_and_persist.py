"""One-time utility to train and persist the XGBoost model.

This script is NOT used at runtime. It trains a global model on SPY data
to use for all subsequent inference operations.

Usage:
    python tradesense/models/train_and_persist.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

# Add parent to path for imports
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from tradesense.data_provider import get_market_data
from tradesense.features import build_feature_matrix
from tradesense.calibration import fit_platt_scaler
from tradesense.modeling import prepare_model_data, train_models


def train_and_persist_model() -> None:
    """Train XGBoost model on SPY data and save to disk."""
    
    print("Fetching market data for SPY...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    market_data_dict = get_market_data(["SPY"], start_date, end_date, interval="1d")
    market_data = market_data_dict["SPY"]
    
    if market_data.empty:
        raise ValueError("Failed to fetch SPY data")
    
    print(f"Fetched {len(market_data)} rows of SPY data")
    
    print("Building feature matrix...")
    features = build_feature_matrix(market_data)
    
    if features.empty:
        raise ValueError("No valid features generated")
    
    print(f"Generated {len(features)} feature rows")
    
    print("Preparing model data...")
    close_prices = market_data["close"]
    X_model, y_model = prepare_model_data(features, close_prices, horizon=5)
    
    print(f"Model will train on {len(X_model)} rows")
    
    print("Splitting train/validation data for calibration...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_model,
        y_model,
        test_size=0.2,
        random_state=42,
        stratify=y_model,
    )

    print("Training XGBoost model...")
    xgb_model, _ = train_models(X_train, y_train)

    print("Fitting Platt scaler on validation data...")
    raw_val_probs = xgb_model.predict_proba(X_val)[:, 1]
    calibrator = fit_platt_scaler(raw_val_probs, y_val)
    
    # Save model bundle (model + ordered feature columns)
    model_path = Path(__file__).parent / "xgboost.joblib"
    print(f"Saving model to {model_path}...")
    calibration_meta = {
        "method": "platt",
        "fitted_on": "validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    model_bundle = {
        "model": xgb_model,
        "feature_names": list(X_model.columns),
        "calibrator": calibrator,
        "calibration_meta": calibration_meta,
    }
    joblib.dump(model_bundle, model_path)
    
    print(f"✓ Model successfully saved to {model_path}")
    print(f"✓ Model feature columns: {list(X_model.columns)}")


if __name__ == "__main__":
    train_and_persist_model()
