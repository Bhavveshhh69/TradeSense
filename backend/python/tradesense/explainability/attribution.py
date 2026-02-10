# tradesense/explainability/attribution.py
"""Deterministic attribution utilities for model outputs."""

from __future__ import annotations

import math
from typing import Dict

import pandas as pd


def compute_attributions_from_importance(
    feature_importance: Dict[str, float],
    feature_values: Dict[str, float],
) -> Dict[str, float]:
    """Compute bounded, signed attributions using feature values and importance."""
    if not isinstance(feature_importance, dict) or not feature_importance:
        raise ValueError("feature_importance must be a non-empty dict")
    if not isinstance(feature_values, dict) or not feature_values:
        raise ValueError("feature_values must be a non-empty dict")

    aligned = {k: float(v) for k, v in feature_importance.items() if k in feature_values}
    if not aligned:
        raise ValueError("feature_importance has no overlap with feature_values")

    total = sum(abs(value) for value in aligned.values())
    if total <= 0:
        weight = 1.0 / len(aligned)
        contributions = {
            name: weight * _bounded_direction(feature_values[name]) for name in aligned
        }
        return normalize_attributions(contributions)

    contributions = {}
    for name, importance in aligned.items():
        direction = _bounded_direction(feature_values[name])
        contributions[name] = (abs(float(importance)) / total) * direction

    return normalize_attributions(contributions)


def compute_attributions_from_model(
    model,
    inference_row: pd.DataFrame,
) -> Dict[str, float]:
    """Compute XGBoost per-feature contributions, normalized and bounded."""
    if inference_row is None or not isinstance(inference_row, pd.DataFrame):
        raise ValueError("inference_row must be a pandas.DataFrame")
    if inference_row.shape[0] != 1:
        raise ValueError("inference_row must contain exactly one row")

    if not hasattr(model, "get_booster"):
        raise ValueError("model does not expose get_booster")

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ValueError("xgboost is required for model-based attributions") from exc

    booster = model.get_booster()
    feature_names = list(inference_row.columns)
    dmatrix = xgb.DMatrix(inference_row, feature_names=feature_names)
    contribs = booster.predict(dmatrix, pred_contribs=True)

    if contribs.ndim != 2 or contribs.shape[0] != 1:
        raise ValueError("Unexpected contribution output shape")

    # Last column is the bias term; exclude it for feature attributions.
    raw = {name: float(contribs[0][idx]) for idx, name in enumerate(feature_names)}
    return normalize_attributions(raw)


def normalize_attributions(attributions: Dict[str, float]) -> Dict[str, float]:
    """Normalize attributions so sum(abs) == 1 and values are bounded."""
    if not isinstance(attributions, dict) or not attributions:
        return {}

    total = sum(abs(value) for value in attributions.values())
    if total <= 0:
        return {name: 0.0 for name in attributions}

    normalized = {
        name: _clamp(value / total)
        for name, value in attributions.items()
    }
    return normalized


def _bounded_direction(value: float) -> float:
    if not _is_valid_number(value):
        return 0.0
    value = float(value)
    return value / (abs(value) + 1.0)


def _clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _is_valid_number(value: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
