"""Intraday prediction API routes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, field_validator

from tradesense.backtesting.backtest_engine import run_backtest
from tradesense.intraday.engine import ENGINE
from tradesense.intraday.market import DEFAULT_TIMEFRAME_MIN, get_market_profile
from tradesense.intraday.provider import latest_regular_close


logger = logging.getLogger(__name__)
router = APIRouter()


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: StrictStr = Field(..., min_length=1)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("symbol must be a non-empty string")
        return normalized


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    market: str
    exchange: str
    timeframe: str
    strategy_family: str
    prediction: int
    probability: float
    confidence: float
    decision: str
    confidence_level: str
    strength: float
    context: dict[str, Any]
    model_version: str
    model_name: str
    model_threshold: float
    model_bench_summary: dict[str, Any]
    timestamp: str
    generated_at: str
    setup_side: str | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    forced_exit_time: str | None = None
    no_trade_reason: str | None = None
    data_quality: dict[str, Any]
    summary: str
    market_context: dict[str, Any]
    key_drivers: list[str]
    risk_notes: list[str]
    model_honesty: str
    current_price: float | None = None
    trade_window: dict[str, Any] | None = None
    threshold: float | None = None
    stock_sentiment_score: float
    sector_sentiment_score: float | None = None
    contextual_sentiment_score: float
    sentiment_confidence: float
    sentiment_gate_reason: str
    stock_article_count: int
    sector_article_count: int
    sector: str | None = None
    sector_available: bool = False
    stock_articles: list[str] = Field(default_factory=list)
    sector_articles: list[str] = Field(default_factory=list)


class LatestPriceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    market: str
    timeframe: str
    price: float = Field(..., gt=0.0)
    source: str
    timestamp: str


class MarketHistoryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    close: float = Field(..., gt=0.0)


class MarketHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    market: str
    timeframe: str
    history: list[MarketHistoryPoint]


class ValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: StrictStr = Field(..., min_length=1)
    start_date: str | None = None
    end_date: str | None = None
    interval: str = "1d"
    horizon: int = Field(default=5, ge=1, le=30)

    @field_validator("symbol")
    @classmethod
    def _normalize_validation_symbol(cls, value: str) -> str:
        return _validated_symbol(value)


class ValidationPeriod(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_date: str
    end_date: str
    horizon: int


class ValidationReliabilityPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probability_mean: float = Field(..., ge=0.0, le=1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    count: int = Field(..., ge=1)


class ValidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    period: ValidationPeriod
    total_predictions: int = Field(..., ge=0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    ece: float = Field(..., ge=0.0)
    brier_score: float = Field(..., ge=0.0)
    accuracy_by_confidence: dict[str, float]
    reliability_curve: list[ValidationReliabilityPoint]


def _validated_symbol(symbol: str) -> str:
    try:
        return PredictRequest(symbol=symbol).symbol
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="symbol must be a non-empty string") from exc


def _fetch_bars(symbol: str, days: int) -> tuple[list, Any]:
    profile = get_market_profile(symbol, timeframe_min=DEFAULT_TIMEFRAME_MIN)
    request = type(
        "Req",
        (),
        {
            "symbol": symbol,
            "market": profile.market,
            "exchange": profile.exchange,
            "timezone": profile.timezone,
            "currency": profile.currency,
            "timeframe_min": DEFAULT_TIMEFRAME_MIN,
            "lookback_days": days,
            "source": "yfinance",
        },
    )()
    bars = ENGINE.provider.fetch_bars(request)
    return bars, profile


def _default_validation_window() -> tuple[str, str]:
    today = datetime.now(tz=UTC).date()
    start = today - timedelta(days=365)
    end = today - timedelta(days=14)
    return start.isoformat(), end.isoformat()


@router.post("/predict", response_model=PredictResponse)
def predict(payload: dict = Body(...)):
    try:
        request = PredictRequest(**payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="symbol must be a non-empty string") from exc

    try:
        result = ENGINE.predict(request.symbol, timeframe_min=DEFAULT_TIMEFRAME_MIN)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Intraday prediction failed for %s", request.symbol)
        raise HTTPException(status_code=500, detail="Prediction failure") from exc

    return PredictResponse(**result)


@router.get("/market/latest-price/{symbol}", response_model=LatestPriceResponse)
def latest_price(symbol: str):
    normalized_symbol = _validated_symbol(symbol)
    try:
        bars, profile = _fetch_bars(normalized_symbol, days=10)
        latest_close, latest_ts = latest_regular_close(bars)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Latest price fetch failure for %s", normalized_symbol)
        raise HTTPException(
            status_code=500,
            detail={"error": "Market data provider failure", "symbol": normalized_symbol},
        ) from exc

    if latest_close is None or latest_ts is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "Price unavailable for symbol", "symbol": normalized_symbol},
        )

    payload = {
        "symbol": normalized_symbol,
        "market": profile.market,
        "timeframe": f"{DEFAULT_TIMEFRAME_MIN}m",
        "price": float(latest_close),
        "source": "intraday_close",
        "timestamp": latest_ts.astimezone(UTC).isoformat(),
    }
    return LatestPriceResponse(**payload)


@router.get("/market/history/{symbol}", response_model=MarketHistoryResponse)
def market_history(symbol: str, days: int = Query(default=30, ge=1, le=60)):
    normalized_symbol = _validated_symbol(symbol)
    try:
        bars, profile = _fetch_bars(normalized_symbol, days=days)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Price history fetch failure for %s", normalized_symbol)
        raise HTTPException(
            status_code=502,
            detail={"error": "Market data provider failure", "symbol": normalized_symbol},
        ) from exc

    history = [
        {
            "date": bar.timestamp.astimezone(UTC).isoformat(),
            "close": float(bar.close),
        }
        for bar in bars
        if bar.is_regular_session
    ]
    if not history:
        raise HTTPException(
            status_code=404,
            detail={"error": "Price history unavailable for symbol", "symbol": normalized_symbol},
        )

    return MarketHistoryResponse(
        symbol=normalized_symbol,
        market=profile.market,
        timeframe=f"{DEFAULT_TIMEFRAME_MIN}m",
        history=history,
    )


@router.post("/analyze/validate", response_model=ValidationResponse)
def validate_analysis(payload: dict = Body(...)):
    try:
        request = ValidationRequest(**payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="symbol must be a non-empty string") from exc

    start_date, end_date = _default_validation_window()
    if request.start_date:
        start_date = request.start_date
    if request.end_date:
        end_date = request.end_date

    try:
        _, result = run_backtest(
            symbol=request.symbol,
            start_date=start_date,
            end_date=end_date,
            interval=request.interval,
            horizon=request.horizon,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Validation failed for %s", request.symbol)
        raise HTTPException(status_code=500, detail="Validation failure") from exc

    metrics = result.metrics.model_dump()
    reliability_curve = [bucket.model_dump() for bucket in result.reliability]
    return ValidationResponse(
        symbol=result.symbol,
        period=ValidationPeriod(
            start_date=result.start_date,
            end_date=result.end_date,
            horizon=result.horizon,
        ),
        total_predictions=result.total_predictions,
        accuracy=float(metrics["overall_accuracy"]),
        ece=float(metrics["expected_calibration_error"]),
        brier_score=float(metrics["brier_score"]),
        accuracy_by_confidence=metrics["accuracy_by_confidence_level"],
        reliability_curve=[
            ValidationReliabilityPoint(**bucket) for bucket in reliability_curve
        ],
    )
