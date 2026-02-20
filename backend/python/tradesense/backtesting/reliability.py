"""Reliability curve computation utilities."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .schemas import ReliabilityBucket


def _model_to_dict(model: ReliabilityBucket) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def compute_reliability_curve(
    predictions: pd.DataFrame,
    probability_column: str = "probability_calibrated",
    actual_column: str = "actual_outcome",
    n_bins: int = 10,
) -> Dict[str, List[Dict[str, float]]]:
    """Compute reliability buckets from a predictions DataFrame."""
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas.DataFrame")
    if predictions.empty:
        return {"buckets": []}
    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    if probability_column not in predictions.columns:
        raise ValueError(f"Missing required column: {probability_column}")
    if actual_column not in predictions.columns:
        raise ValueError(f"Missing required column: {actual_column}")

    probabilities = predictions[probability_column].astype(float).clip(0.0, 1.0)
    outcomes = predictions[actual_column].astype(float).clip(0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_index = pd.cut(
        probabilities,
        bins=edges,
        include_lowest=True,
        labels=False,
    )

    buckets: List[Dict[str, float]] = []
    for bucket_id in range(n_bins):
        mask = bucket_index == bucket_id
        if int(mask.sum()) == 0:
            continue

        bucket = ReliabilityBucket(
            probability_mean=float(probabilities[mask].mean()),
            accuracy=float(outcomes[mask].mean()),
            count=int(mask.sum()),
        )
        buckets.append(_model_to_dict(bucket))

    return {"buckets": buckets}
