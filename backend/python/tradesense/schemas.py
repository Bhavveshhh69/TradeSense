# tradesense/schemas.py
"""Pydantic schemas for the Phase 4B FastAPI service."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, StrictBool, StrictStr, validator


class ReasonRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    probability: float = Field(..., ge=0.0, le=1.0)
    feature_importance: Dict[str, float]
    feature_values: Dict[str, float]
    trend_state: Literal[-1, 0, 1]
    momentum_state: Literal[-1, 1]
    risk_state: Literal[0, 1, 2]

    class Config:
        extra = "forbid"


class AnalyzeRequest(BaseModel):
    symbol: StrictStr = Field(..., min_length=1)
    news: Optional[List[StrictStr]] = None
    use_news: StrictBool = False
    include_context: StrictBool = False
    explain: StrictBool = False

    @validator("symbol")
    def _strip_and_validate_symbol(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("symbol must be a non-empty string")
        return value

    @validator("news", pre=True)
    def _normalize_news(cls, value):
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("news must be a list of strings")
        cleaned = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("news must be a list of strings")
            stripped = item.strip()
            if stripped:
                cleaned.append(stripped)
        return cleaned or None

    class Config:
        extra = "forbid"


class MarketContext(BaseModel):
    trend: str
    momentum: str
    volatility: str

    class Config:
        extra = "forbid"


class StructuredExplanation(BaseModel):
    key_drivers: List[str]
    negative_factors: List[str]
    confidence_modifiers: List[str]

    class Config:
        extra = "forbid"


class ReasonResponse(BaseModel):
    symbol: str
    probability: float
    probability_raw: float
    probability_calibrated: float
    confidence_level: str
    confidence_reason: str
    summary: str
    market_context: MarketContext
    key_drivers: List[str]
    structured_explanation: StructuredExplanation
    risk_notes: List[str]
    model_honesty: str

    class Config:
        extra = "forbid"


class SentimentResponse(BaseModel):
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sentiment_bias: Literal["bullish", "neutral", "bearish"]
    sentiment_strength: Literal["low", "medium", "high"]

    class Config:
        extra = "forbid"


class ContextResponse(BaseModel):
    history_summary: str
    num_items: int = Field(..., ge=1)

    class Config:
        extra = "forbid"


class ExplanationResponse(BaseModel):
    summary: str
    narrative: str
    disclaimer: str

    class Config:
        extra = "forbid"


class AnalyzeResponse(ReasonResponse):
    sentiment: Optional[SentimentResponse] = None
    context: Optional[ContextResponse] = None
    explanation: Optional[ExplanationResponse] = None

    class Config:
        extra = "forbid"
