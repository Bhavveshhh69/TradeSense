"""Phase 6A inference orchestrator: deterministic pipeline that wires together Phases 1-5."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import logging

from tradesense.calibration import calibrate_probability
from tradesense.data_provider import get_market_data
from tradesense.features import build_feature_matrix
from tradesense.modeling import prepare_model_data
from tradesense.explainability.attribution import (
    compute_attributions_from_importance,
    compute_attributions_from_model,
)
from tradesense.reasoning_core import generate_insight


logger = logging.getLogger(__name__)


class CalibrationArtifactError(ValueError):
    """Raised when required calibration artifacts are missing or invalid."""

# Load pre-trained model once at module load time
_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgboost.joblib"


def _load_model():
    """Load pre-trained XGBoost model bundle from disk."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pre-trained model not found at {_MODEL_PATH}. "
            f"Run: python tradesense/models/train_and_persist.py"
        )
    model_bundle = joblib.load(_MODEL_PATH)
    if isinstance(model_bundle, dict):
        model = model_bundle.get("model")
        feature_names = model_bundle.get("feature_names")
        calibrator = model_bundle.get("calibrator")
        calibration_meta = model_bundle.get("calibration_meta")
        if model is None or feature_names is None:
            raise ValueError(
                "Model bundle is missing required keys. "
                "Re-run: python tradesense/models/train_and_persist.py"
            )
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError(
                "Model bundle feature_names is empty or invalid. "
                "Re-run: python tradesense/models/train_and_persist.py"
            )
        if calibrator is None or calibration_meta is None:
            raise CalibrationArtifactError(
                "Missing calibration artifacts in model bundle: "
                "calibrator and calibration_meta are required. "
                "Re-run: python tradesense/models/train_and_persist.py"
            )
        if not hasattr(calibrator, "predict_proba"):
            raise CalibrationArtifactError(
                "Invalid calibrator in model bundle: missing predict_proba. "
                "Re-run: python tradesense/models/train_and_persist.py"
            )
        if not isinstance(calibration_meta, dict) or calibration_meta.get("method") != "platt":
            raise CalibrationArtifactError(
                "Invalid calibration metadata in model bundle: "
                "expected method=platt. "
                "Re-run: python tradesense/models/train_and_persist.py"
            )
        return model, feature_names, calibrator, calibration_meta

    model = model_bundle
    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        feature_names = list(getattr(model.get_booster(), "feature_names", []))
    if not feature_names:
        raise ValueError(
            "Persisted model does not include feature names. "
            "Re-run: python tradesense/models/train_and_persist.py"
        )
    raise CalibrationArtifactError(
        "Persisted model bundle does not include calibration artifacts. "
        "Re-run: python tradesense/models/train_and_persist.py"
    )


_XGB_MODEL, _MODEL_FEATURES, _CALIBRATOR, _CALIBRATION_META = _load_model()


def analyze_symbol(symbol: str) -> dict:
    """
    Given a stock symbol, run the full inference pipeline and
    return the final reasoning insight object.
    
    Uses a pre-trained XGBoost model for fast, deterministic inference.
    
    Steps:
    1. Validate input
    2. Fetch market data (≈180 days)
    3. Compute indicators (Phase 1)
    4. Build feature matrix (Phase 2)
    5. Select latest valid feature row
    6. Run inference on latest row with pre-trained model (Phase 3)
    7. Calibrate probability (Phase 9)
    8. Derive market states
    9. Generate reasoning (Phase 4A)
    10. Return insight dict
    
    Args:
        symbol: Stock symbol (e.g., "AAPL")
    
    Returns:
        dict: Reasoning insight object with symbol, probability, confidence_level, etc.
    
    Raises:
        ValueError: If symbol is invalid, data fetch fails, or inference fails.
    """
    
    # Step 1: Validate input
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    
    symbol = symbol.upper().strip()
    
    # Step 2: Fetch market data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    
    market_data_dict = get_market_data([symbol], start_date, end_date, interval="1d")
    
    if symbol not in market_data_dict or market_data_dict[symbol].empty:
        raise ValueError(f"No market data available for symbol: {symbol}")
    
    market_data = market_data_dict[symbol]
    
    if len(market_data) < 50:
        raise ValueError(
            f"Insufficient market data for {symbol}: "
            f"got {len(market_data)} rows, need at least 50"
        )
    
    # Step 3 & 4: Build feature matrix (indicators computed in Phase 1)
    features = build_feature_matrix(market_data)
    
    if features.empty:
        raise ValueError(
            f"No valid features generated for {symbol} after dropping NaNs"
        )
    
    # Step 5: Select latest valid feature row (inference row)
    latest_idx = features.index[-1]
    latest_features = features.loc[[latest_idx]]
    
    # Extract feature values for later use
    feature_values_dict = latest_features.iloc[0].to_dict()
    
    # Step 6: Run inference with pre-trained model
    # Use the exact feature set (and order) that the model was trained on.
    expected_features = list(_MODEL_FEATURES)

    print(f"Model expected feature names: {expected_features}")
    print(f"Latest feature row columns: {list(latest_features.columns)}")

    missing_features = [name for name in expected_features if name not in latest_features.columns]
    if missing_features:
        raise ValueError(
            f"Missing required feature columns for {symbol}: {missing_features}"
        )

    inference_row = latest_features.loc[:, expected_features]

    nan_features = [
        name for name in inference_row.columns if pd.isna(inference_row.iloc[0][name])
    ]
    if nan_features:
        raise ValueError(
            f"NaN values in inference row for {symbol}: {nan_features}"
        )

    probability_proba = _XGB_MODEL.predict_proba(inference_row)
    probability_raw = float(probability_proba[0, 1])
    try:
        probability_calibrated = calibrate_probability(_CALIBRATOR, probability_raw)
    except Exception as exc:
        raise ValueError(f"Calibration failed for {symbol}: {exc}") from exc
    # Step 7: Calibrate probability (Phase 9)
    logger.debug(
        "Calibration complete for %s: raw=%.6f calibrated=%.6f",
        symbol,
        probability_raw,
        probability_calibrated,
    )
    
    # Step 8: Derive market states from feature values
    # Extract key features for state derivation
    price_vs_ema50 = feature_values_dict.get("price_vs_ema50", np.nan)
    ema20_vs_ema50 = feature_values_dict.get("ema20_vs_ema50", np.nan)
    rsi_slope_3 = feature_values_dict.get("rsi_slope_3", np.nan)
    macd_hist_accel = feature_values_dict.get("macd_hist_accel", np.nan)
    volatility_regime = feature_values_dict.get("volatility_regime", np.nan)
    
    # Derive trend_state: bullish when price and EMA20 are above EMA50
    if not np.isnan(price_vs_ema50) and not np.isnan(ema20_vs_ema50):
        if price_vs_ema50 > 0 and ema20_vs_ema50 > 0:
            trend_state = 1  # bullish
        elif price_vs_ema50 < 0 and ema20_vs_ema50 < 0:
            trend_state = -1  # bearish
        else:
            trend_state = 0  # sideways
    else:
        trend_state = 0  # default to sideways if unable to compute
    
    # Derive momentum_state: strengthening when combined momentum is non-negative
    if not np.isnan(rsi_slope_3) and not np.isnan(macd_hist_accel):
        combined_momentum = rsi_slope_3 + macd_hist_accel
        momentum_state = 1 if combined_momentum >= 0 else -1
    else:
        momentum_state = 1  # default to strengthening
    
    # Derive risk_state: mirrors volatility regime (0=low, 1=medium, 2=high)
    if not np.isnan(volatility_regime):
        risk_state = int(volatility_regime)
    else:
        risk_state = 1  # default to medium
    
    # Ensure state values are within valid ranges
    trend_state = int(max(-1, min(1, trend_state)))
    momentum_state = 1 if momentum_state >= 0 else -1
    risk_state = int(max(0, min(2, risk_state)))
    
    # Step 9: Prepare feature importance and feature values for generate_insight
    feature_importance = {
        name: float(importance)
        for name, importance in zip(
            _MODEL_FEATURES,
            _XGB_MODEL.feature_importances_,
        )
    }
    
    # Filter feature values to only include those the model uses
    model_features = set(_MODEL_FEATURES)
    filtered_feature_values = {
        k: v for k, v in feature_values_dict.items()
        if k in model_features
    }
    
    feature_attributions = None
    try:
        feature_attributions = compute_attributions_from_model(_XGB_MODEL, inference_row)
    except Exception as exc:
        logger.warning(
            "Model-based attribution failed for %s; falling back to importance: %s",
            symbol,
            exc,
        )
        feature_attributions = compute_attributions_from_importance(
            feature_importance,
            filtered_feature_values,
        )

    # Step 10: Call generate_insight and return
    insight = generate_insight(
        symbol=symbol,
        probability=probability_calibrated,
        probability_raw=probability_raw,
        probability_calibrated=probability_calibrated,
        feature_importance=feature_importance,
        feature_values=filtered_feature_values,
        trend_state=trend_state,
        momentum_state=momentum_state,
        risk_state=risk_state,
        feature_attributions=feature_attributions,
    )
    
    return insight
