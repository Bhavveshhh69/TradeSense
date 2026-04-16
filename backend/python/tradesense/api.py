# tradesense/api.py
"""FastAPI transport wrapper for the reasoning and intraday inference stack."""

from __future__ import annotations

import traceback
from datetime import datetime, timezone

from fastapi import Body, FastAPI, HTTPException
from pydantic import ValidationError

from tradesense.api_predict import router as predict_router
from tradesense.reasoning_core import generate_insight
from tradesense.schemas import AnalyzeRequest, ReasonRequest, ReasonResponse

app = FastAPI()
app.include_router(predict_router)


def _get_analyze_symbol():
    from tradesense.intraday import analyze_symbol

    return analyze_symbol


def _get_rag_handlers():
    from tradesense.rag.formatter import format_context
    from tradesense.rag.retriever import retrieve_context
    from tradesense.rag.store import store_insight

    return store_insight, retrieve_context, format_context


def _get_explainer_handlers():
    from tradesense.explainer.llm_client import generate_explanation
    from tradesense.explainer.prompt_builder import build_explanation_prompt

    return build_explanation_prompt, generate_explanation


def _legacy_sentiment_view(result: dict) -> dict | None:
    score = result.get("contextual_sentiment_score")
    if score is None:
        return None
    score = float(score)
    if score >= 0.15:
        bias = "bullish"
    elif score <= -0.15:
        bias = "bearish"
    else:
        bias = "neutral"
    magnitude = abs(score)
    if magnitude < 0.2:
        strength = "low"
    elif magnitude < 0.5:
        strength = "medium"
    else:
        strength = "high"
    return {
        "sentiment_score": score,
        "sentiment_bias": bias,
        "sentiment_strength": strength,
    }


@app.post("/reason", response_model=ReasonResponse)
def reason(payload: ReasonRequest):
    try:
        result = generate_insight(**payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@app.post("/analyze", response_model_exclude_none=True)
def analyze(payload: dict = Body(...)):
    try:
        request = AnalyzeRequest(**payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail="symbol must be a non-empty string",
        ) from exc

    symbol = request.symbol
    if not symbol.strip():
        raise HTTPException(
            status_code=400,
            detail="symbol must be a non-empty string",
        )

    try:
        analyze_symbol = _get_analyze_symbol()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    try:
        result = analyze_symbol(symbol, news_texts=request.news)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    context_summary = None
    try:
        store_insight, retrieve_context, format_context = _get_rag_handlers()
        timestamp = datetime.now(timezone.utc).isoformat()
        insight_record = {
            "symbol": result.get("symbol", symbol),
            "timestamp": timestamp,
            "probability": result.get("probability"),
            "confidence_level": result.get("confidence_level"),
            "sentiment": _legacy_sentiment_view(result),
            "key_drivers": result.get("key_drivers"),
            "risk_notes": result.get("risk_notes"),
            "news_headlines": list(request.news) if request.news else None,
            "summary": result.get("summary"),
            "market_context": result.get("market_context"),
        }
        store_insight(insight_record)

        if request.include_context or request.explain:
            history = retrieve_context(symbol, limit=6)
            history = [item for item in history if item.get("timestamp") != timestamp]
            if history:
                history_summary = format_context(history)
                if history_summary:
                    context_summary = history_summary
                    if request.include_context:
                        result = {
                            **result,
                            "context": {
                                **(result.get("context") or {}),
                                "history_summary": history_summary,
                                "num_items": len(history),
                            },
                        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    if request.explain:
        try:
            build_prompt, generate_explanation = _get_explainer_handlers()
            prompt = build_prompt(result, context_summary)
            explanation = generate_explanation(prompt)
            result = {
                **result,
                "explanation": explanation,
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Internal server error") from exc

    return result
