# TradeSense Architecture

## Project Overview
TradeSense is a Python-first analytics and reasoning stack with a thin Node.js product API. It ingests market data, computes indicators, builds engineered features, trains probabilistic models, and converts model outputs into deterministic insights. The reasoning core is exposed via a FastAPI service, and the Node.js service forwards validated requests to Python without transformation.

## Current Architecture Snapshot

```text
React frontend
  -> Node.js Express API (POST /api/analyze)
     - validates request shape
     - in-memory cache with TTL (default 300s, configurable)
     - forwards payload to Python FastAPI (/reason)
  -> Python FastAPI Service
     - POST /reason: validates request with Pydantic and calls deterministic reasoning core
     - POST /analyze: runs inference pipeline and optional news/sentiment + RAG context
     - local RAG store for historical insight summaries (file-based)
  <- JSON response returned unchanged to client
```

## Request Flow (Node -> Python /reason)
1. Client sends `POST /api/analyze` to the Node.js service with `{ symbol, payload }`.
2. Node validates the request shape and computes a stable hash of the request body.
3. If a cached response exists, it is returned immediately.
4. On cache miss, Node forwards `payload` to the Python `POST /reason` endpoint.
5. FastAPI validates the payload against Pydantic schemas.
6. The deterministic reasoning core generates the insight response.
7. The Python response is returned to Node and then to the client unchanged.

## Request Flow (Direct Python /analyze)
1. Client sends `POST /analyze` to the Python FastAPI service with `{ symbol, use_news, news, include_context }`.
2. The inference orchestrator runs the Phase 1–6 pipeline for the symbol.
3. If `use_news=true`, Python fetches news from Finnhub and normalizes headlines.
4. If news text exists (manual or fetched), FinBERT sentiment is computed and attached.
5. The insight is stored in the local RAG store as a structured summary.
6. If `include_context=true`, recent insights are retrieved and formatted into a compact history summary.
7. The response returns the core insight plus optional sentiment and context blocks.

## Phase-by-Phase Evolution Summary
- Phase 1: Market Data & Indicators. Fetch OHLCV from yfinance and compute RSI (14), EMA 20, EMA 50, and MACD (line, signal, histogram) per symbol.
- Phase 2: Feature Engineering. Convert Phase 1 outputs into a numeric feature matrix with trend, momentum, volatility/range, and volume features plus encoded market state signals; warm-up rows are removed.
- Phase 3: Probabilistic Modeling. Create a 5-day continuation target, perform time-ordered train/test split, train XGBoost and Logistic Regression models, and report classification metrics and feature importance.
- Phase 4A: Deterministic Reasoning Core. Map probabilities and market state into a structured, rule-based insight with strict input validation.
- Phase 4B: FastAPI Reasoning Service. Expose the reasoning core via a single `POST /reason` endpoint with Pydantic request/response models.
- Phase 4C: Node.js Product API. Provide `POST /api/analyze` with validation, short-lived in-memory caching, and forwarding to Python.
- Phase 4D: Performance & Operational Optimization. Optional uvloop/httptools support, guidance to keep Python hot, configurable cache TTL, optional pre-warm request, and minimal timing logs.
- Phase 5A: React Frontend Integration. Minimal UI wired to `POST /api/analyze`, renders JSON, and surfaces errors.
- Phase 5B: Dynamic Symbol Input. Frontend captures symbol input, normalizes to uppercase, prevents empty submissions, and highlights analyzed symbol.
- Phase 6A: Deterministic Inference Orchestrator. End-to-end pipeline from market data to reasoning output using the persisted XGBoost model.
- Phase 6B: FastAPI `/analyze` Endpoint. Exposes the inference pipeline via `POST /analyze` with a strict response contract.
- Phase 7A: Optional FinBERT Sentiment. When news text is supplied, sentiment is computed and appended to `/analyze` responses.
- Phase 7B: Optional Finnhub News Ingestion. When `use_news=true`, Finnhub headlines are fetched and normalized before sentiment scoring.
- Phase 8A: RAG Memory Layer. Local vector store persists structured insight summaries and returns formatted history context when requested.

## Determinism & Testing Guarantees
- The reasoning core is deterministic for identical inputs and uses fixed thresholds and mappings.
- Request hashing is stable due to sorted-key serialization before hashing.
- Modeling uses fixed random seeds for repeatable training behavior in local runs.
- RAG embeddings use deterministic hashing, and retrieval is stable for identical inputs.
- Pytest validates Phases 1 through 8A (data, features, modeling, reasoning, inference, sentiment/news, and RAG).
- Jest/Supertest validates the Node API (valid/invalid requests and cache reuse behavior).

## Explicit Non-Goals
- Trading, execution, or order management.
- Brokerage integrations or paper trading.
- Advanced UI beyond the minimal dashboard.
- Authentication or authorization.
- External databases or cloud storage.
- Streaming/real-time data pipelines.
- LLM-based generation or chat interfaces.
- Using RAG context to modify probabilities or signals.
- Distributed cache, queues, or background job systems.
