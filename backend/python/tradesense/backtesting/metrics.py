"""Backtest metric computations."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .schemas import CalibrationMetrics


_PROBABILITY_BUCKETS: List[Tuple[float, float]] = [
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
]


def _expected_calibration_error(
    probabilities: pd.Series,
    outcomes: pd.Series,
    n_bins: int = 10,
) -> float:
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    if len(probabilities) == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_index = pd.cut(
        probabilities,
        bins=edges,
        include_lowest=True,
        labels=False,
    )

    total = float(len(probabilities))
    ece = 0.0
    for bucket_id in range(n_bins):
        mask = bucket_index == bucket_id
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(outcomes[mask].mean())
        ece += (count / total) * abs(accuracy - confidence)

    return float(max(0.0, min(1.0, ece)))


def _accuracy_by_confidence_level(
    predictions: pd.DataFrame,
    correctness: pd.Series,
    confidence_column: str,
) -> Dict[str, float]:
    output: Dict[str, float] = {}
    grouped = predictions.groupby(confidence_column)
    for level, group in grouped:
        output[str(level)] = float(correctness.loc[group.index].mean())
    return output


def _accuracy_by_probability_bucket(
    probabilities: pd.Series,
    correctness: pd.Series,
) -> Dict[str, float]:
    output: Dict[str, float] = {}

    below_mask = probabilities < 0.5
    if int(below_mask.sum()) > 0:
        output["<0.5"] = float(correctness[below_mask].mean())

    for lower, upper in _PROBABILITY_BUCKETS:
        label = f"{lower:.1f}-{upper:.1f}"
        if upper < 1.0:
            mask = (probabilities >= lower) & (probabilities < upper)
        else:
            mask = (probabilities >= lower) & (probabilities <= upper)

        if int(mask.sum()) == 0:
            continue
        output[label] = float(correctness[mask].mean())

    return output


def compute_backtest_metrics(
    predictions: pd.DataFrame,
    probability_column: str = "probability_calibrated",
    actual_column: str = "actual_outcome",
    confidence_column: str = "confidence_level",
) -> CalibrationMetrics:
    """Compute evaluation metrics from a predictions DataFrame."""
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas.DataFrame")
    if predictions.empty:
        raise ValueError("predictions must not be empty")
    for column in (probability_column, actual_column, confidence_column):
        if column not in predictions.columns:
            raise ValueError(f"Missing required column: {column}")

    probabilities = predictions[probability_column].astype(float).clip(0.0, 1.0)
    outcomes = predictions[actual_column].astype(int).clip(0, 1)

    predicted_labels = (probabilities >= 0.5).astype(int)
    correctness = (predicted_labels == outcomes).astype(float)

    overall_accuracy = float(correctness.mean())
    confidence_accuracy = _accuracy_by_confidence_level(
        predictions,
        correctness,
        confidence_column=confidence_column,
    )
    bucket_accuracy = _accuracy_by_probability_bucket(probabilities, correctness)
    ece = _expected_calibration_error(probabilities, outcomes, n_bins=10)
    brier_score = float(np.mean(np.square(probabilities - outcomes)))

    return CalibrationMetrics(
        overall_accuracy=overall_accuracy,
        accuracy_by_confidence_level=confidence_accuracy,
        accuracy_by_probability_bucket=bucket_accuracy,
        expected_calibration_error=ece,
        brier_score=brier_score,
    )
