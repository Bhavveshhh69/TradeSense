# tests/test_phase9.py
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.calibration import (  # noqa: E402
    calibrate_probability,
    derive_confidence_level,
    fit_platt_scaler,
)


def test_platt_scaler_bounds_and_monotonicity():
    raw_probs = np.linspace(0.05, 0.95, 10)
    y_true = np.array([0] * 5 + [1] * 5)

    calibrator = fit_platt_scaler(raw_probs, y_true)
    calibrated = [calibrate_probability(calibrator, p) for p in raw_probs]

    assert all(0.0 <= p <= 1.0 for p in calibrated)
    assert all(calibrated[i] <= calibrated[i + 1] for i in range(len(calibrated) - 1))


def test_confidence_non_increasing_with_volatility():
    probability = 0.82
    low_level, _ = derive_confidence_level(probability, "low")
    medium_level, _ = derive_confidence_level(probability, "medium")
    high_level, _ = derive_confidence_level(probability, "high")

    rank = {"low": 0, "moderate": 1, "high": 2}
    assert rank[low_level] >= rank[medium_level] >= rank[high_level]
    assert high_level != "high"
