# tradesense/reasoning_core.py
"""Phase 4A deterministic reasoning core for TradeSense."""

from __future__ import annotations

import math
from typing import Dict, Tuple

from tradesense.calibration import derive_confidence_level
from tradesense.explainability.attribution import (
    compute_attributions_from_importance,
    normalize_attributions,
)
from tradesense.explainability.rules import (
    build_model_honesty,
    build_risk_notes,
    build_structured_explanation,
    signals_conflict,
)

CONFIDENCE_LOW_THRESHOLD = 0.55
CONFIDENCE_HIGH_THRESHOLD = 0.65

TREND_MAP = {-1: "bearish", 0: "sideways", 1: "bullish"}
MOMENTUM_MAP = {-1: "weakening", 1: "strengthening"}
RISK_MAP = {0: "low", 1: "medium", 2: "high"}


def generate_insight(
    symbol: str,
    probability: float,
    feature_importance: Dict[str, float],
    feature_values: Dict[str, float],
    trend_state: int,
    momentum_state: int,
    risk_state: int,
    probability_raw: float | None = None,
    probability_calibrated: float | None = None,
    feature_attributions: Dict[str, float] | None = None,
) -> dict:
    """Generate a deterministic insight object from model outputs and market state.

    Contract:
      - probability is the calibrated probability returned to clients.
      - probability_raw is the raw model output (pre-calibration).
    """

    probability_raw = float(probability if probability_raw is None else probability_raw)
    probability_calibrated = float(
        probability if probability_calibrated is None else probability_calibrated
    )
    _validate_probability(probability_raw, "probability_raw")
    _validate_probability(probability_calibrated, "probability_calibrated")

    _validate_inputs(
        symbol=symbol,
        probability=probability_calibrated,
        feature_importance=feature_importance,
        feature_values=feature_values,
        trend_state=trend_state,
        momentum_state=momentum_state,
        risk_state=risk_state,
    )

    probability = float(probability_calibrated)
    trend_state = int(trend_state)
    momentum_state = int(momentum_state)
    risk_state = int(risk_state)

    volatility_regime = RISK_MAP[risk_state]
    confidence_level, confidence_reason = derive_confidence_level(
        probability_calibrated,
        volatility_regime,
    )
    summary = _summary_from_confidence(confidence_level)
    market_context = {
        "trend": TREND_MAP[trend_state],
        "momentum": MOMENTUM_MAP[momentum_state],
        "volatility": volatility_regime,
    }

    feature_attributions = _prepare_attributions(
        feature_attributions,
        feature_importance,
        feature_values,
    )
    structured_explanation = build_structured_explanation(
        feature_attributions=feature_attributions,
        feature_values=feature_values,
        confidence_reason=confidence_reason,
        trend_state=trend_state,
        momentum_state=momentum_state,
        risk_state=risk_state,
    )

    key_drivers = structured_explanation["key_drivers"]
    risk_notes = build_risk_notes(trend_state, momentum_state, risk_state)
    conflict = signals_conflict(trend_state, momentum_state)
    model_honesty = build_model_honesty(confidence_level, conflict)

    return {
        "symbol": symbol,
        "probability": probability,
        "probability_raw": probability_raw,
        "probability_calibrated": probability_calibrated,
        "confidence_level": confidence_level,
        "confidence_reason": confidence_reason,
        "summary": summary,
        "market_context": market_context,
        "key_drivers": key_drivers,
        "structured_explanation": structured_explanation,
        "risk_notes": risk_notes,
        "model_honesty": model_honesty,
    }


def _validate_inputs(
    symbol: str,
    probability: float,
    feature_importance: Dict[str, float],
    feature_values: Dict[str, float],
    trend_state: int,
    momentum_state: int,
    risk_state: int,
) -> None:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    if not _is_valid_number(probability) or not (0.0 <= float(probability) <= 1.0):
        raise ValueError("probability must be a float between 0 and 1")

    if not isinstance(feature_importance, dict) or not feature_importance:
        raise ValueError("feature_importance must be a non-empty dict")
    if not isinstance(feature_values, dict) or not feature_values:
        raise ValueError("feature_values must be a non-empty dict")

    for key, value in feature_importance.items():
        if not isinstance(key, str) or not key:
            raise ValueError("feature_importance keys must be non-empty strings")
        if not _is_valid_number(value):
            raise ValueError("feature_importance values must be finite numbers")

    for key, value in feature_values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("feature_values keys must be non-empty strings")
        if not _is_valid_number(value):
            raise ValueError("feature_values values must be finite numbers")

    missing = [key for key in feature_importance if key not in feature_values]
    if missing:
        raise ValueError(f"feature_values missing keys: {missing}")

    _validate_state(trend_state, TREND_MAP, "trend_state")
    _validate_state(momentum_state, MOMENTUM_MAP, "momentum_state")
    _validate_state(risk_state, RISK_MAP, "risk_state")


def _is_valid_number(value: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_probability(value: float, name: str) -> None:
    if not _is_valid_number(value) or not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be a float between 0 and 1")


def _validate_state(value: int, allowed: Dict[int, str], name: str) -> None:
    if not _is_valid_number(value) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    if int(value) not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed.keys())}")


def _confidence_from_probability(probability: float) -> str:
    if probability < CONFIDENCE_LOW_THRESHOLD:
        return "low"
    if probability <= CONFIDENCE_HIGH_THRESHOLD:
        return "moderate"
    return "high"


def _summary_from_confidence(confidence_level: str) -> str:
    if confidence_level == "medium":
        confidence_level = "moderate"
    summaries = {
        "low": "Low continuation bias",
        "moderate": "Moderate continuation bias",
        "high": "Strong continuation bias",
    }
    return summaries[confidence_level]


def _prepare_attributions(
    feature_attributions: Dict[str, float] | None,
    feature_importance: Dict[str, float],
    feature_values: Dict[str, float],
) -> Dict[str, float]:
    if feature_attributions is None:
        return compute_attributions_from_importance(feature_importance, feature_values)

    if not isinstance(feature_attributions, dict) or not feature_attributions:
        return compute_attributions_from_importance(feature_importance, feature_values)

    cleaned = {
        name: float(value)
        for name, value in feature_attributions.items()
        if name in feature_values and _is_valid_number(value)
    }
    if not cleaned:
        return compute_attributions_from_importance(feature_importance, feature_values)

    return normalize_attributions(cleaned)
