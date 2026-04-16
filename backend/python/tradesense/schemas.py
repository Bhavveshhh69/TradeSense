"""Pydantic schemas for the FastAPI service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator


class ReasonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., min_length=1)
    probability: float = Field(..., ge=0.0, le=1.0)
    feature_importance: dict[str, float]
    feature_values: dict[str, float]
    trend_state: Literal[-1, 0, 1]
    momentum_state: Literal[-1, 1]
    risk_state: Literal[0, 1, 2]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: StrictStr = Field(..., min_length=1)
    news: list[StrictStr] | None = None
    use_news: StrictBool = False
    include_context: StrictBool = False
    explain: StrictBool = False

    @field_validator("symbol")
    @classmethod
    def _strip_and_validate_symbol(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("symbol must be a non-empty string")
        return cleaned

    @field_validator("news", mode="before")
    @classmethod
    def _normalize_news(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("news must be a list of strings")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("news must be a list of strings")
            stripped = item.strip()
            if stripped:
                cleaned.append(stripped)
        return cleaned or None


class MarketContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trend: str
    momentum: str
    volatility: str


class StructuredExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_drivers: list[str]
    negative_factors: list[str]
    confidence_modifiers: list[str]


class ReasonResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    probability: float
    probability_raw: float
    probability_calibrated: float
    confidence_level: str
    confidence_reason: str
    summary: str
    market_context: MarketContext
    key_drivers: list[str]
    structured_explanation: StructuredExplanation
    risk_notes: list[str]
    model_honesty: str


class SentimentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sentiment_bias: Literal["bullish", "neutral", "bearish"]
    sentiment_strength: Literal["low", "medium", "high"]


class ContextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history_summary: str
    num_items: int = Field(..., ge=1)


class ExplanationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    narrative: str
    disclaimer: str


class AnalyzeResponse(ReasonResponse):
    model_config = ConfigDict(extra="forbid")

    sentiment: SentimentResponse | None = None
    context: ContextResponse | None = None
    explanation: ExplanationResponse | None = None
