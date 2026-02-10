# Project Overview
TradeSense is a production-grade financial data module with a thin Node.js API and a minimal React frontend. Phases 1 through 8A are complete, spanning market data ingestion, indicators, feature engineering, probabilistic modeling, deterministic reasoning, end-to-end inference, optional sentiment/news enrichment, and a local RAG memory layer for explainability. The implementation remains deterministic and modular, and it does not include trading, execution, or deep learning.

# Architecture Snapshot (Current)
The module includes a data provider, indicator computation, a feature engineering layer, and a probabilistic modeling layer. Market data is fetched via yfinance, and indicators are computed with pandas and numpy. The public entry point returns a symbol-keyed dictionary of DataFrames with a standardized, lowercased schema and datetime index. A separate feature engineering function transforms Phase 1 outputs into an engineered feature matrix. The modeling layer consumes Phase 2 features and close prices to create a binary 5-day continuation label, trains a time-ordered split, and reports probabilistic metrics plus human-readable XGBoost feature importance. A deterministic reasoning core (library) converts Phase 3 outputs into structured insight objects. A FastAPI service exposes `/reason` (reasoning) and `/analyze` (full inference with optional news/sentiment and RAG context), a Node.js Express API fronts Python at `/api/analyze`, and a React frontend calls the Node API. A local RAG store persists structured insight summaries for historical context.

# Phase 1 — Market Data & Indicators (Completed)
Phase 1 delivers multi-symbol historical OHLCV retrieval using yfinance and computes RSI (14), EMA 20, EMA 50, and MACD (line, signal, histogram). The output contract is a per-symbol DataFrame containing date, open, high, low, close, volume, rsi, ema_20, ema_50, macd, macd_signal, and macd_hist. Missing or empty data is handled per symbol without failing the entire request.

# Phase 2 — Feature Engineering (Completed)

## Goals
- Add a dedicated feature engineering layer using only Phase 1 outputs.
- Keep transformations pure with no trading logic, execution, labels, or targets.
- Avoid look-ahead bias and external data sources.

## Feature Categories Implemented
- Trend structure features.
- Momentum change features.
- Volatility and range features.
- Volume confirmation features.
- Encoded market state features.

## Implementation Summary
Feature engineering is implemented in features.py as a pure transformation function. The input contract is strictly Phase 1 outputs with defensive input validation. Warm-up rows are dropped cleanly and the final output contains no NaNs.

## Verification & Testing
Pytest unit tests validate non-empty output, no NaNs, expected feature columns, and numeric feature values. All tests pass successfully.

# Phase 3 — Probabilistic Modeling (Completed)

## Goals
- Introduce probabilistic modeling to estimate 5-day continuation without any trading logic.
- Use Phase 2 engineered features as model inputs.
- Produce explainable outputs and standard classification metrics.

## Target Definition
The binary target variable represents continuation over the next 5 trading days: 1 when close(t + 5) > close(t), otherwise 0. Future close prices are used only for label creation.

## Modeling Approach
- Primary model: XGBoost classifier.
- Baseline model: Logistic Regression.
- Time-based train/test split with no shuffling (first 70% train, last 30% test).
- Outputs are probability estimates only, not actions.

## Evaluation Summary
Precision, recall, F1-score, and ROC-AUC are computed for both models. XGBoost feature importance is reported in a human-readable format.

## Verification & Testing
Pytest unit tests were added for Phase 3 to validate binary target correctness, time-order preservation in the split, metric validity, and the existence of feature importance. All Phase 1, Phase 2, and Phase 3 tests pass successfully.

# Phase 4A — Deterministic Reasoning Core (Completed)

## Goals
- Provide a pure Python reasoning module that converts Phase 3 outputs and feature context into structured insights.
- Keep the logic deterministic and rule-based with no LLMs, training, or external services.

## Reasoning Scope
The reasoning core accepts the Phase 3 probability, feature importance, feature values, and market state inputs, and produces an insight object. It performs strict input validation and uses deterministic rules to interpret trend, momentum, and risk regimes.

## Output Structure
The output includes confidence level, summary statement, market context, key drivers, risk notes, and a model honesty statement.

## Verification & Testing
Pytest unit tests cover schema correctness, confidence mapping, risk notes, non-empty text fields, and deterministic outputs for identical inputs. The module performs no networking, persistence, or API work.

# Phase 4B — Reasoning Service API (Completed)

## Purpose
Expose the Phase 4A deterministic reasoning core via a minimal FastAPI transport layer.

## API Overview
The service provides a single POST endpoint, `/reason`, that accepts a structured request and returns the reasoning core output without transformation.

## Validation & Contracts
Request and response schemas are enforced with strict Pydantic models. The API performs validation only and does not modify reasoning logic.

## Verification & Testing
Pytest tests use the FastAPI test client to validate 200 responses, schema correctness, and deterministic output for identical inputs. No ML, training, databases, authentication, caching, or persistence is introduced.

# Phase 4C — Node.js Product API (Completed)

## Purpose
Provide a frontend-safe API that validates requests, forwards reasoning payloads to Python, and applies short-lived caching.

## API Overview
The service provides a single POST endpoint, `/api/analyze`, that validates `{ symbol, payload }` and forwards `payload` to `/reason`.

## Verification & Testing
Jest/Supertest tests validate valid/invalid request handling and cache reuse.

# Phase 5A — React Frontend Integration (Completed)

## Purpose
Provide a minimal React UI for triggering deterministic market analysis.

## Summary
The React frontend calls `/api/analyze`, renders the response JSON verbatim, and surfaces network errors in the UI.

# Phase 5B — Dynamic Symbol Input (Completed)

## Purpose
Allow users to supply the stock symbol dynamically from the UI.

## Summary
The frontend captures symbol input in state, normalizes it to uppercase, disables submissions on empty input, and highlights the analyzed symbol in results.


# Phase 6A ? Deterministic Inference Orchestrator (Completed)

## Purpose
Run an end-to-end inference pipeline from market data to deterministic insight.

## Summary
The inference orchestrator loads the persisted XGBoost model, computes indicators and features, derives market states, and calls the deterministic reasoning core for a symbol-based insight.

# Phase 6B ? FastAPI /analyze Endpoint (Completed)

## Purpose
Expose the inference pipeline via HTTP without breaking the existing /reason endpoint.

## Summary
FastAPI now provides POST /analyze with strict request validation, returning the same deterministic insight schema used by /reason.

# Phase 7A ? Optional FinBERT Sentiment (Completed)

## Purpose
Provide optional sentiment scoring when news text is supplied.

## Summary
FinBERT sentiment is computed for supplied news text and aggregated into a compact sentiment block appended to /analyze responses.

# Phase 7B ? Optional Finnhub News Ingestion (Completed)

## Purpose
Fetch real news when requested to enrich explainability context.

## Summary
When use_news=true, Finnhub headlines are fetched and normalized, then passed through FinBERT sentiment scoring.

# Phase 8A ? RAG Memory Layer (Completed)

## Purpose
Persist and retrieve structured insight history to enrich explanations without altering model outputs.

## Summary
A local vector store captures structured insight summaries, retrieves symbol-scoped history deterministically, and formats a compact context block when include_context=true.

# Decision Log (Accepted & Rejected Decisions)
Accepted decisions:
- Enforced strict phase gating and testing before moving to later phases.
- Implemented dynamic symbol input with no hard-coded symbols.
- Selected yfinance as the data source for Phase 1.

Rejected or deferred decisions:
- Deferred Alpaca integration to a later paper-trading phase.
- Rejected LSTM for core logic.
- Rejected machine learning and training in Phase 1; sentiment/news were added later as optional explainability layers.

# Known Limitations (As of Now)
- Scope is limited to analysis, deterministic reasoning, optional sentiment/news enrichment, and RAG context (no trading or execution).
- Data access depends on yfinance availability and Finnhub when use_news=true.
- No hyperparameter tuning loops or cross-validation yet.
- No deep learning beyond the optional FinBERT sentiment model.
- No authentication or authorization is implemented.
- No external database integration; persistence is limited to the local RAG store.
- RAG context is explainability-only and does not modify probabilities or signals.

# Next Planned Phase (High-Level Only)
No additional phases are defined beyond Phase 8A.
