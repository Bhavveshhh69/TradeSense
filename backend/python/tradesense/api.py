# tradesense/api.py
"""FastAPI transport wrapper for the Phase 4A/6A reasoning core."""

import traceback
from datetime import datetime, timezone

from fastapi import Body, FastAPI, HTTPException
from pydantic import ValidationError
from tradesense.reasoning_core import generate_insight
from tradesense.schemas import AnalyzeRequest, AnalyzeResponse, ReasonRequest, ReasonResponse

app = FastAPI()


def _get_analyze_symbol():
    from tradesense.inference import analyze_symbol

    return analyze_symbol


def _get_sentiment_handlers():
    from tradesense.sentiment.aggregator import aggregate_sentiment
    from tradesense.sentiment.finbert import analyze_texts

    return analyze_texts, aggregate_sentiment


def _get_news_handlers():
    from tradesense.news.fetcher import fetch_news
    from tradesense.news.normalizer import normalize_news

    return fetch_news, normalize_news


def _get_rag_handlers():
    from tradesense.rag.formatter import format_context
    from tradesense.rag.retriever import retrieve_context
    from tradesense.rag.store import store_insight

    return store_insight, retrieve_context, format_context


def _get_explainer_handlers():
    from tradesense.explainer.llm_client import generate_explanation
    from tradesense.explainer.prompt_builder import build_explanation_prompt

    return build_explanation_prompt, generate_explanation


@app.post("/reason", response_model=ReasonResponse)
def reason(payload: ReasonRequest):
    try:
        result = generate_insight(**payload.dict())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@app.post("/analyze", response_model=AnalyzeResponse, response_model_exclude_none=True)
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
        # DEV ONLY: log full traceback for debugging
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from exc

    try:
        result = analyze_symbol(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        # DEV ONLY: log full traceback for debugging
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from exc
    news_texts = request.news
    include_context = request.include_context
    explain = request.explain
    # If news fetch yields nothing, sentiment is omitted to preserve Phase 7A behavior.
    if not news_texts and request.use_news:
        try:
            fetch_news, normalize_news = _get_news_handlers()
            fetched_news = fetch_news(symbol)
            news_texts = normalize_news(fetched_news)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            # DEV ONLY: log full traceback for debugging
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail="Internal server error",
            ) from exc

    if news_texts:
        try:
            analyze_texts, aggregate_sentiment = _get_sentiment_handlers()
            finbert_results = analyze_texts(news_texts)
            sentiment = aggregate_sentiment(finbert_results)
            result = {**result, "sentiment": sentiment}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            # DEV ONLY: log full traceback for debugging
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail="Internal server error",
            ) from exc

    context_summary = None
    try:
        store_insight, retrieve_context, format_context = _get_rag_handlers()
        timestamp = datetime.now(timezone.utc).isoformat()
        insight_record = {
            "symbol": result.get("symbol", symbol),
            "timestamp": timestamp,
            "probability": result.get("probability"),
            "confidence_level": result.get("confidence_level"),
            "sentiment": result.get("sentiment"),
            "key_drivers": result.get("key_drivers"),
            "risk_notes": result.get("risk_notes"),
            "news_headlines": news_texts,
            "summary": result.get("summary"),
            "market_context": result.get("market_context"),
        }
        store_insight(insight_record)

        if include_context or explain:
            history = retrieve_context(symbol, limit=6)
            history = [item for item in history if item.get("timestamp") != timestamp]
            if history:
                history_summary = format_context(history)
                if history_summary:
                    context_summary = history_summary
                    if include_context:
                        result = {
                            **result,
                            "context": {
                                "history_summary": history_summary,
                                "num_items": len(history),
                            },
                        }
    except Exception as exc:
        # DEV ONLY: log full traceback for debugging
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from exc

    if explain:
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
            # DEV ONLY: log full traceback for debugging
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail="Internal server error",
            ) from exc

    return result
