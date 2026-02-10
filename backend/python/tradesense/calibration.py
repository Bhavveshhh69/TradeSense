"""Probability calibration and confidence discipline utilities."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Phase 9: Locked — Calibration & Confidence Discipline
PHASE_9_LOCKED = "Phase 9: Locked — Calibration & Confidence Discipline"

# Baseline confidence bands carried forward from Phase 4 thresholds for continuity.
_BASE_CONFIDENCE_LOW_MAX = 0.55
_BASE_CONFIDENCE_MODERATE_MAX = 0.65

_CONFIDENCE_ORDER = {"low": 0, "moderate": 1, "high": 2}

# Volatility caps ensure confidence never increases as volatility rises.
_VOLATILITY_CAP = {
    "low": "high",
    "medium": "moderate",
    "high": "moderate",
}


def fit_platt_scaler(
    raw_probabilities: np.ndarray,
    y_true: np.ndarray,
) -> LogisticRegression:
    """Fit a Platt scaler (logistic regression) on raw model probabilities."""
    raw_probabilities = np.asarray(raw_probabilities, dtype=float).reshape(-1, 1)
    y_true = np.asarray(y_true, dtype=int).ravel()

    if raw_probabilities.size == 0 or y_true.size == 0:
        raise ValueError("Calibration data must be non-empty")
    if raw_probabilities.shape[0] != y_true.shape[0]:
        raise ValueError("Calibration data length mismatch")
    if np.any((raw_probabilities < 0.0) | (raw_probabilities > 1.0)):
        raise ValueError("raw_probabilities must be in [0, 1]")
    if np.unique(y_true).size < 2:
        raise ValueError("Calibration labels must include both classes")

    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(raw_probabilities, y_true)

    coef = float(calibrator.coef_.ravel()[0])
    intercept = float(calibrator.intercept_.ravel()[0])
    if coef <= 0:
        raise ValueError(
            "Calibrator fit produced non-positive slope; "
            "calibration would invert probabilities."
        )

    logger.debug(
        "Platt scaler fitted: coef=%.6f intercept=%.6f", coef, intercept
    )
    return calibrator


def calibrate_probability(calibrator: LogisticRegression, raw_probability: float) -> float:
    """Apply a fitted Platt scaler to a raw model probability."""
    if calibrator is None or not hasattr(calibrator, "predict_proba"):
        raise ValueError("Calibrator is missing or invalid")
    if not _is_valid_probability(raw_probability):
        raise ValueError("raw_probability must be in [0, 1]")

    proba = calibrator.predict_proba(np.array([[float(raw_probability)]]))
    calibrated = float(proba[0, 1])

    if not _is_valid_probability(calibrated):
        raise ValueError("Calibrated probability is out of bounds")

    logger.debug(
        "Calibrated probability: raw=%.6f calibrated=%.6f",
        float(raw_probability),
        calibrated,
    )
    return calibrated


def derive_confidence_level(
    calibrated_probability: float,
    volatility_regime: str,
) -> Tuple[str, str]:
    """Derive confidence level and reason from calibrated probability and volatility."""
    if not _is_valid_probability(calibrated_probability):
        raise ValueError("calibrated_probability must be in [0, 1]")
    if volatility_regime not in _VOLATILITY_CAP:
        raise ValueError(
            "volatility_regime must be one of: low, medium, high"
        )

    base_level = _base_confidence_from_probability(calibrated_probability)
    cap_level = _VOLATILITY_CAP[volatility_regime]
    final_level = _min_confidence(base_level, cap_level)

    if final_level != base_level:
        reason = (
            f"{volatility_regime.title()} volatility regime reduces certainty; "
            f"confidence capped at {final_level}."
        )
    else:
        reason = (
            f"Calibrated probability supports {final_level} confidence "
            f"under {volatility_regime} volatility."
        )

    return final_level, reason


def _base_confidence_from_probability(probability: float) -> str:
    if probability < _BASE_CONFIDENCE_LOW_MAX:
        return "low"
    if probability <= _BASE_CONFIDENCE_MODERATE_MAX:
        return "moderate"
    return "high"


def _min_confidence(left: str, right: str) -> str:
    if _CONFIDENCE_ORDER[left] <= _CONFIDENCE_ORDER[right]:
        return left
    return right


def _is_valid_probability(value: float) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
