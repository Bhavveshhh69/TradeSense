"""Production inference API router for Phase 14."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging
from typing import Any, Dict, Tuple

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field, StrictStr, ValidationError, validator
import yfinance as yf

from tradesense.data_provider import get_market_data
from tradesense.features import build_feature_matrix
from tradesense.inference.context_engine import TradeSenseContextEngine
from tradesense.inference.decision_engine import TradeSenseDecisionEngine
from tradesense.inference.predict import TradeSensePredictor


logger = logging.getLogger(__name__)
router = APIRouter()

_LOOKBACK_DAYS = 220
_PRICE_LOOKBACK_DAYS = _LOOKBACK_DAYS


class PredictRequest(BaseModel):
    symbol: StrictStr = Field(..., min_length=1)

    @validator("symbol")
    def _normalize_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("symbol must be a non-empty string")
        return normalized

    class Config:
        extra = "forbid"


class PredictResponse(BaseModel):
    symbol: str = Field(..., min_length=1)
    prediction: int = Field(..., ge=0, le=1)
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., min_length=1)
    confidence_level: str = Field(..., min_length=1)
    strength: float = Field(..., ge=0.0, le=1.0)
    context: Dict[str, Any]
    model_version: str = Field(..., min_length=1)
    timestamp: str = Field(..., min_length=1)
    generated_at: str = Field(..., min_length=1)

    class Config:
        extra = "forbid"


class LatestPriceResponse(BaseModel):
    symbol: str = Field(..., min_length=1)
    price: float = Field(..., gt=0.0)
    source: str = Field(..., min_length=1)
    timestamp: str = Field(..., min_length=1)

    class Config:
        extra = "forbid"


class MarketHistoryPoint(BaseModel):
    date: str = Field(..., min_length=1)
    close: float = Field(..., gt=0.0)

    class Config:
        extra = "forbid"


class MarketHistoryResponse(BaseModel):
    symbol: str = Field(..., min_length=1)
    history: list[MarketHistoryPoint]

    class Config:
        extra = "forbid"


def _initialize_runtime() -> Tuple[TradeSensePredictor | None, Exception | None]:
    try:
        return TradeSensePredictor(), None
    except Exception as exc:  # noqa: BLE001 - allow endpoint-safe fallback
        logger.error("Predictor initialization failed: %s", exc.__class__.__name__)
        return None, exc


_PREDICTOR, _PREDICTOR_INIT_ERROR = _initialize_runtime()
_DECISION_ENGINE = TradeSenseDecisionEngine()
_CONTEXT_ENGINE = TradeSenseContextEngine()


def _load_latest_feature_row(symbol: str):
    end_date = datetime.now(tz=UTC).date()
    start_date = end_date - timedelta(days=_LOOKBACK_DAYS)

    try:
        market_data = get_market_data(
            [symbol],
            start_date.isoformat(),
            end_date.isoformat(),
            interval="1d",
        )
    except Exception as exc:  # noqa: BLE001 - external provider isolation
        logger.error("Market data fetch failed for symbol=%s", symbol)
        raise HTTPException(status_code=500, detail="Market data fetch failure") from exc

    symbol_data = market_data.get(symbol)
    if symbol_data is None or symbol_data.empty:
        raise HTTPException(status_code=400, detail="Invalid symbol")

    try:
        feature_matrix = build_feature_matrix(symbol_data)
    except Exception as exc:  # noqa: BLE001 - feature pipeline isolation
        logger.error("Feature generation failed for symbol=%s", symbol)
        raise HTTPException(status_code=500, detail="Feature generation failure") from exc

    if feature_matrix.empty:
        raise HTTPException(status_code=500, detail="Feature generation failure")

    return feature_matrix.tail(1)


def _load_latest_close_price(symbol: str) -> float:
    end_date = datetime.now(tz=UTC).date()
    start_date = end_date - timedelta(days=_PRICE_LOOKBACK_DAYS)

    try:
        market_data = get_market_data(
            [symbol],
            start_date.isoformat(),
            end_date.isoformat(),
            interval="1d",
        )
    except Exception as exc:  # noqa: BLE001 - external provider isolation
        logger.error("Market data fetch failed for latest price symbol=%s", symbol)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Market data fetch failure",
                "symbol": symbol,
            },
        ) from exc

    symbol_data = market_data.get(symbol)
    if symbol_data is None or symbol_data.empty:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Price unavailable for symbol",
                "symbol": symbol,
            },
        )

    try:
        latest_close = float(symbol_data["close"].iloc[-1])
    except Exception as exc:  # noqa: BLE001 - provider data shape safety
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Invalid market data response",
                "symbol": symbol,
            },
        ) from exc

    if latest_close <= 0.0:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Price unavailable for symbol",
                "symbol": symbol,
            },
        )

    return latest_close


def _load_historical_close_prices(symbol: str, days: int) -> list[dict[str, Any]]:
    try:
        history_df = yf.Ticker(symbol).history(
            period=f"{int(days)}d",
            interval="1d",
            auto_adjust=True,
        )
    except Exception as exc:  # noqa: BLE001 - external provider isolation
        logger.error("Market data fetch failed for history symbol=%s", symbol)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Market data fetch failure",
                "symbol": symbol,
            },
        ) from exc

    if history_df is None or history_df.empty:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Price history unavailable for symbol",
                "symbol": symbol,
            },
        )

    close_column = None
    if "Close" in history_df.columns:
        close_column = "Close"
    elif "close" in history_df.columns:
        close_column = "close"

    if close_column is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Invalid market data response",
                "symbol": symbol,
            },
        )

    closes = history_df[[close_column]].copy()
    closes = closes.rename(columns={close_column: "close"})
    closes = closes.dropna(subset=["close"])

    if closes.empty:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Price history unavailable for symbol",
                "symbol": symbol,
            },
        )

    history: list[dict[str, Any]] = []
    for date_index, row in closes.iterrows():
        try:
            close_price = float(row["close"])
        except Exception:  # noqa: BLE001 - row-level shape safety
            continue
        if close_price <= 0.0:
            continue

        date_value = date_index.to_pydatetime() if hasattr(date_index, "to_pydatetime") else date_index
        date_part = date_value.date() if hasattr(date_value, "date") else None
        date_str = date_part.isoformat() if date_part else str(date_index)[:10]

        history.append(
            {
                "date": date_str,
                "close": close_price,
            }
        )

    if not history:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Price history unavailable for symbol",
                "symbol": symbol,
            },
        )

    return history


@router.post("/predict", response_model=PredictResponse)
def predict(payload: dict = Body(...)):
    try:
        request = PredictRequest(**payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="symbol must be a non-empty string") from exc

    symbol = request.symbol

    if _PREDICTOR is None:
        logger.error(
            "Model bundle unavailable for /predict: %s",
            type(_PREDICTOR_INIT_ERROR).__name__,
        )
        raise HTTPException(status_code=500, detail="Model bundle missing")

    feature_row = _load_latest_feature_row(symbol)

    try:
        prediction = _PREDICTOR.predict_from_features(feature_row)
        decision = _DECISION_ENGINE.decide_from_prediction(prediction)
        context = _CONTEXT_ENGINE.build_context(
            prediction=prediction,
            decision=decision,
            features=feature_row.iloc[0].to_dict(),
        )
    except FileNotFoundError as exc:
        logger.error("Model bundle missing during prediction")
        raise HTTPException(status_code=500, detail="Model bundle missing") from exc
    except Exception as exc:  # noqa: BLE001 - endpoint-safe failure boundary
        logger.error("Prediction failure for symbol=%s", symbol)
        raise HTTPException(status_code=500, detail="Prediction failure") from exc

    generated_at = context.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.strip():
        raise HTTPException(status_code=500, detail="Prediction failure")

    response_payload = {
        "symbol": symbol,
        "prediction": int(prediction["prediction"]),
        "probability": float(prediction["probability"]),
        "confidence": float(prediction["confidence"]),
        "decision": str(decision["decision"]),
        "confidence_level": str(decision["confidence_level"]),
        "strength": float(decision["strength"]),
        "context": dict(context),
        "model_version": str(prediction["model_version"]),
        "timestamp": str(prediction["timestamp"]),
        "generated_at": generated_at,
    }

    try:
        return PredictResponse(**response_payload)
    except ValidationError as exc:
        logger.error("PredictResponse validation failed for symbol=%s", symbol)
        raise HTTPException(status_code=500, detail="Prediction failure") from exc


@router.get("/market/latest-price/{symbol}", response_model=LatestPriceResponse)
def latest_price(symbol: str):
    try:
        request = PredictRequest(symbol=symbol)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="symbol must be a non-empty string") from exc

    normalized_symbol = request.symbol
    latest_close = _load_latest_close_price(normalized_symbol)

    response_payload = {
        "symbol": normalized_symbol,
        "price": float(latest_close),
        "source": "close",
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }

    try:
        return LatestPriceResponse(**response_payload)
    except ValidationError as exc:
        logger.error("LatestPriceResponse validation failed for symbol=%s", normalized_symbol)
        raise HTTPException(status_code=500, detail="Latest price fetch failure") from exc


@router.get("/market/history/{symbol}", response_model=MarketHistoryResponse)
def market_history(symbol: str, days: int = Query(default=30, ge=1, le=365)):
    try:
        request = PredictRequest(symbol=symbol)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="symbol must be a non-empty string") from exc

    normalized_symbol = request.symbol
    prices = _load_historical_close_prices(normalized_symbol, days)

    response_payload = {
        "symbol": normalized_symbol,
        "history": prices,
    }

    try:
        return MarketHistoryResponse(**response_payload)
    except ValidationError as exc:
        logger.error("MarketHistoryResponse validation failed for symbol=%s", normalized_symbol)
        raise HTTPException(status_code=500, detail="Price history fetch failure") from exc
