"""Schemas for backtesting and calibration evaluation outputs."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field


class ReliabilityBucket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probability_mean: float = Field(..., ge=0.0, le=1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    count: int = Field(..., ge=1)


class CalibrationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_accuracy: float = Field(..., ge=0.0, le=1.0)
    accuracy_by_confidence_level: Dict[str, float]
    accuracy_by_probability_bucket: Dict[str, float]
    expected_calibration_error: float = Field(..., ge=0.0, le=1.0)
    brier_score: float = Field(..., ge=0.0, le=1.0)


class BacktestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., min_length=1)
    start_date: str
    end_date: str
    horizon: int = Field(..., ge=1)
    total_predictions: int = Field(..., ge=0)
    metrics: CalibrationMetrics
    reliability: List[ReliabilityBucket]
