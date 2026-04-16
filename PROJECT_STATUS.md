# TradeSense Project Documentation (Practicality Pass Complete)

## Recovery Ledger (2026-04-16)

### Objective
Recover the interrupted TradeSense practicality pass, verify the real shipped state against the live codebase, finish the remaining integration and verification work end to end, and leave the repo in a clean shippable state.

### Scope In
- Real route-based trader workflow across `Today`, `Analysis`, and `Portfolio`.
- Strict symbol search/normalization for US and India equities plus curated benchmark indices.
- Live market quote/history routes, analysis route, validation/backtest route, command-center route, and ledger-backed paper portfolio routes.
- Frontend shell parity, portfolio workspace, validation panel, and global paper-trade drawer.
- Full verification matrix, live API verification, live browser verification, and documentation refresh.

### Scope Out
- Multi-user auth, production database persistence, and broker connectivity.
- P&L/profitability claims derived from validation metrics.
- Full bundle-splitting/performance hardening beyond correctness and usability.

### Locked Decisions
- TradeSense v1 uses a real route-based React shell instead of a single scrolling dashboard.
- Mobile behavior stays responsive within the same routes, not separate mobile routes.
- Instrument search uses grouped async dropdown results with metadata for India, US, and indices.
- Portfolio state is ledger-backed with explicit `BUY`, `SELL`, `SHORT`, and `COVER` trades plus adjustment entries.
- `Alerts` and `Daily Review` remain visible concept screens with no fake persistence.
- Python intraday inference and validation are part of the product contract and must stay honest about `NO_TRADE` outcomes.
- Fresh shipped state must contain no seeded holdings, no seeded trades, and no seeded recent analyses.

### Completed
- Reconstructed the interrupted state from the worktree, git history, and continuity file without widening scope.
- Confirmed the codebase had already moved beyond the stale ledger: backend market/command-center/portfolio work, frontend routed shell, tests, and Playwright suite were already present in code.
- Reset runtime JSON persistence to a clean shipped state and re-verified clean empty portfolio/transaction/recent-analysis behavior after live testing.
- Backend symbol search and normalization now resolve:
  - US equity: `NVDA`
  - India equity: `RELIANCE.NS`
  - US index: `^GSPC`
  - India index: `^NSEI`
- Backend market quote/history routes, command-center aggregation, recent-analysis persistence, and ledger-backed portfolio routes are implemented and working.
- India benchmark indices are now handled correctly by the Python intraday market resolver.
- Frontend routed shell, pinned active instrument bar, quote strip, analysis workspace, validation panel, recent-symbol reopen flow, and global paper-trade drawer are implemented.
- Full verification passed on 2026-04-16:
  - `cd backend/node && npm test -- --runInBand` -> `9` suites, `48` tests passed
  - `./.venv/bin/pytest -q backend/python/tests` -> `58` tests passed, `1` XGBoost model-serialization warning
  - `cd frontend && npm test` -> `2` files, `5` tests passed
  - `cd frontend && npm run lint` -> passed
  - `cd frontend && npm run build` -> passed
  - `cd frontend && npm run test:e2e` -> `4` Playwright tests passed
- Live API verification passed on 2026-04-16:
  - `GET /api/market/quote/{symbol}` succeeded for `NVDA`, `RELIANCE.NS`, `^GSPC`, and `^NSEI`
  - `POST /api/analyze` returned honest `NO_TRADE` outputs for both `NVDA` and `RELIANCE.NS`
  - `POST /api/analyze/validate` returned live leak-safe classification metrics for `NVDA` and `RELIANCE.NS`
- Live browser verification passed against the real running stack at `http://127.0.0.1:4173/`:
  - selected `NVDA`
  - rendered live quote strip
  - ran live analysis
  - rendered live validation metrics in the real UI

### In Progress
- None for the practicality pass itself. Remaining items are production-hardening follow-ups, not blocked implementation work.

### Next Safe Step
If work continues beyond this pass, the next safe step is production hardening:
- replace JSON persistence with a durable store
- split the large frontend bundle
- add auth and user scoping
- expand live-browser coverage beyond the current operator-critical flows

### Evidence
- `README.md`
- `frontend/src/App.jsx`
- `frontend/src/components/InstrumentPicker.jsx`
- `frontend/src/App.test.jsx`
- `frontend/src/components/InstrumentPicker.test.jsx`
- `frontend/tests/smoke.spec.js`
- `frontend/package.json`
- `frontend/src/App.css`
- `backend/node/server/routes/command-center.js`
- `backend/node/server/routes/market.js`
- `backend/node/server/routes/analyze.js`
- `backend/node/server/services/recent_analysis.service.js`
- `backend/node/portfolio/portfolio.routes.js`
- `backend/python/tradesense/intraday/market.py`
- `backend/python/tests/test_intraday_platform.py`
- `backend/python/tests/test_latest_price_endpoint.py`
- `backend/python/tests/test_validation_endpoint.py`
- 2026-04-16 verification:
  - `cd backend/node && npm test -- --runInBand`
  - `./.venv/bin/pytest -q backend/python/tests`
  - `cd frontend && npm test`
  - `cd frontend && npm run lint`
  - `cd frontend && npm run build`
  - `cd frontend && npm run test:e2e`
  - Live Node/Python/API/browser checks on `http://127.0.0.1:3000`, `http://127.0.0.1:8000`, and `http://127.0.0.1:4173`

### Residual Gaps
- Frontend production build still emits a large-chunk warning (`~700 kB` main JS after minification); correctness is verified, but bundle splitting remains open.
- Portfolio persistence is still JSON-file based and therefore single-user/dev oriented.
- Live news and AI explanation richness still depend on external provider availability and configured keys.

## Project Overview
TradeSense is a modular financial intelligence system implemented as a Python analytics stack with a deterministic reasoning layer, a thin Node.js product API, and a minimal React frontend. The Python codebase covers market data ingestion, indicator computation, feature engineering, probabilistic modeling, deterministic reasoning, end-to-end inference, optional sentiment/news enrichment, and a local RAG memory layer for explainability. The FastAPI service exposes both reasoning and analysis endpoints. The Node.js Express service provides a frontend-safe API that validates requests, forwards them to the Python service, and returns responses without transformation.

## Architecture Snapshot (Current State)
Core components and flow:

```text
React frontend
  -> Node.js Express API (POST /api/analyze)
     - validates request shape
     - in-memory cache (TTL 300s by default) keyed by request body hash
     - forwards payload to Python FastAPI (POST /reason)
  -> Python FastAPI Service
     - POST /reason: deterministic reasoning core
     - POST /analyze: inference pipeline + optional news/sentiment + RAG context
     - local RAG store for historical insight summaries (file-based)
  <- JSON response returned unchanged to client
```

Python services:
- Library modules for data, indicators, features, modeling, and deterministic reasoning.
- Inference orchestrator for end-to-end symbol analysis.
- Sentiment aggregation and FinBERT analysis (optional, `news`).
- News ingestion and normalization (optional, Finnhub).
- RAG store + retriever + formatter for history context.
- FastAPI transport layer exposing `/reason` and `/analyze`.

Node.js services:
- Express API exposing `/api/analyze`.
- Axios-based HTTP client to Python reasoning service.
- In-memory cache with 300-second default TTL (configurable).

## Phase-by-Phase Breakdown

### Phase 1: Market Data & Indicators
Goals:
- Retrieve historical OHLCV data for multiple symbols.
- Compute core technical indicators on top of market data.

What was implemented:
- `get_market_data` in `tradesense/data_provider.py` using `yfinance`.
- Indicators: RSI (14), EMA 20, EMA 50, MACD (line, signal, histogram).
- Per-symbol DataFrame normalization with required columns and date index.
- Defensive validation and per-symbol failure isolation.

What was explicitly NOT implemented:
- Trading, execution, or signal generation.
- Persistence or databases.
- UI, authentication, or external APIs beyond yfinance.

API surface:
- Library function: `tradesense.data_provider.get_market_data`.

### Phase 2: Feature Engineering
Goals:
- Transform Phase 1 outputs into a clean, numeric feature matrix.
- Encode trend, momentum, volatility, and risk states.

What was implemented:
- `build_feature_matrix` in `tradesense/features.py`.
- Trend structure, momentum, volatility/range, volume confirmation features.
- Encoded market state signals: trend, momentum, risk.
- Warm-up row removal with no NaNs in output.

What was explicitly NOT implemented:
- Labels, targets, or model training.
- Any trading logic or execution steps.

API surface:
- Library function: `tradesense.features.build_feature_matrix`.

### Phase 3: Probabilistic Modeling
Goals:
- Create a probabilistic continuation model using Phase 2 features.
- Provide standard classification metrics.

What was implemented:
- Target creation for 5-day continuation in `tradesense/modeling.py`.
- Time-ordered train/test split (no shuffling).
- XGBoost classifier and Logistic Regression baseline.
- Probability outputs and metrics (precision, recall, F1, ROC-AUC).
- Feature importance from XGBoost.

What was explicitly NOT implemented:
- Trading signals or execution logic.
- Hyperparameter tuning, cross-validation, or model serving.

API surface:
- Library functions: `create_target`, `prepare_model_data`, `time_train_test_split`, `train_models`, `run_modeling_pipeline`.

### Phase 4A: Deterministic Reasoning Core
Goals:
- Convert probabilistic model outputs and market state into a structured insight.
- Maintain deterministic, rule-based logic.

What was implemented:
- `generate_insight` in `tradesense/reasoning_core.py`.
- Confidence mapping from probability thresholds.
- Market context mapping (trend, momentum, risk).
- Key driver selection and risk notes.
- Strict validation on all inputs.

What was explicitly NOT implemented:
- Any ML/LLM reasoning or external service calls.
- Persistence or stateful workflow.

API surface:
- Library function: `tradesense.reasoning_core.generate_insight`.

### Phase 4B: Reasoning Service (FastAPI)
Goals:
- Expose deterministic reasoning via a minimal HTTP API.

What was implemented:
- FastAPI app in `tradesense/api.py`.
- POST `/reason` endpoint with Pydantic request/response schemas.
- Strict validation and error reporting (422 on invalid inputs).

What was explicitly NOT implemented:
- Authentication or authorization.
- Caching, persistence, or additional endpoints.

API surface:
- HTTP endpoint: `POST /reason`.

### Phase 4C: Node.js / Express Product API Integration
Goals:
- Provide a frontend-safe API that forwards to the Python reasoning service.
- Add short-lived caching and defensive validation.

What was implemented:
- Express app in `server/index.js` with route `POST /api/analyze`.
- Request validation for `symbol` and `payload` shape.
- In-memory cache keyed by request-body hash (TTL 300 seconds by default).
- Axios client to Python `/reason` with timeout handling.
- Errors propagated with appropriate HTTP status.

What was explicitly NOT implemented:
- Persistence or databases.
- Authentication or authorization.
- Any transformation of the Python response.

API surface:
- HTTP endpoint: `POST /api/analyze`.

### Phase 5A: React Frontend Integration
Goals:
- Provide a minimal React UI for triggering deterministic market analysis.
- Validate end-to-end wiring from React to Node to Python.

What was implemented:
- React (Vite) frontend wired to `POST /api/analyze`.
- UI renders the response JSON and surfaces network errors.

What was explicitly NOT implemented:
- Authentication or authorization.
- Persistent storage or databases.
- Direct frontend calls to the Python service.

### Phase 5B: Dynamic Symbol Input (Frontend)
Goals:
- Allow users to input the stock symbol dynamically in the UI.
- Normalize the symbol to uppercase and prevent empty submissions.

What was implemented:
- Symbol input stored in React state and normalized to uppercase.
- Run Analysis disabled when the symbol input is empty.
- Analyzed symbol highlighted in the results section.

What was explicitly NOT implemented:
- Additional validation beyond non-empty input.
- Backend changes or new endpoints.

### Phase 6A: Deterministic Inference Orchestrator
Goals:
- Run an end-to-end inference pipeline from market data to deterministic insight.
- Keep inference deterministic using a persisted XGBoost model.

What was implemented:
- `analyze_symbol` in `tradesense/inference/orchestrator.py`.
- Model load from `tradesense/models/xgboost.joblib` at module import time.
- End-to-end pipeline: data fetch → indicators → features → inference → reasoning core.
- Explicit market state derivation (trend, momentum, risk) from feature values.

What was explicitly NOT implemented:
- Any alteration of the deterministic reasoning rules.
- Live training or re-fitting of models during inference.

API surface:
- Library function: `tradesense.inference.analyze_symbol`.

### Phase 6B: FastAPI `/analyze` Endpoint
Goals:
- Expose the inference pipeline via HTTP without breaking existing `/reason`.

What was implemented:
- `POST /analyze` in `tradesense/api.py`.
- Request schema validation via `AnalyzeRequest`.
- Response schema compatibility with `AnalyzeResponse`.

What was explicitly NOT implemented:
- Replacing or removing `/reason`.
- Any predictive changes beyond returning deterministic inference outputs.

API surface:
- HTTP endpoint: `POST /analyze`.

### Phase 7A: Optional FinBERT Sentiment
Goals:
- Add optional sentiment scoring for provided news text.

What was implemented:
- FinBERT-based sentiment in `tradesense/sentiment/finbert.py`.
- Aggregation logic in `tradesense/sentiment/aggregator.py`.
- `/analyze` attaches sentiment when `news` is provided.

What was explicitly NOT implemented:
- Sentiment affecting probabilities or signals.
- Any external LLM or cloud inference dependency.

API surface:
- Optional `sentiment` field on `/analyze` responses.

### Phase 7B: Optional Finnhub News Ingestion
Goals:
- Allow `/analyze` to fetch recent news when `use_news=true`.

What was implemented:
- Finnhub fetcher in `tradesense/news/fetcher.py` (requires `FINNHUB_API_KEY`).
- News normalization and truncation in `tradesense/news/normalizer.py`.
- `/analyze` fetches + normalizes news before sentiment scoring.

What was explicitly NOT implemented:
- Persistent storage of raw news or full articles.
- Any data enrichment beyond headline + summary normalization.

API surface:
- Request flag: `use_news` on `/analyze`.

### Phase 8A: RAG Memory Layer
Goals:
- Store and retrieve structured historical insights for explainability.
- Provide a compact context summary without altering model outputs.

What was implemented:
- New `tradesense/rag/` package with store, retriever, and formatter.
- Deterministic embeddings via `HashingVectorizer`.
- Local vector storage (FAISS when available, NumPy fallback).
- `/analyze` stores insights and returns formatted context when `include_context=true`.

What was explicitly NOT implemented:
- LLM-based generation or chat interfaces.
- Using historical context to alter probabilities or model signals.
- External databases or cloud storage.

API surface:
- Request flag: `include_context` on `/analyze`.

## API Surface

### Python FastAPI: `POST /reason`
Input (high-level):
- Symbol identifier.
- Probability score (0 to 1).
- Feature importance map and feature value map.
- Discrete market state indicators (trend, momentum, risk).

Output (high-level):
- Echoed symbol and probability.
- Confidence level and summary.
- Market context (trend, momentum, volatility).
- Key drivers, risk notes, and model honesty statement.

### Python FastAPI: `POST /analyze`
Input (high-level):
- `symbol` string.
- Optional `news` list of strings (for sentiment).
- Optional `use_news` boolean (fetch Finnhub news when true).
- Optional `include_context` boolean (return RAG history summary).

Output (high-level):
- Deterministic insight payload (same shape as `/reason` output).
- Optional `sentiment` block when news is present.
- Optional `context` block with `history_summary` and `num_items` when history is available.

### Node Express: `POST /api/analyze`
Input (high-level):
- `symbol` string.
- `payload` object forwarded verbatim to `/reason`.

Output (high-level):
- The JSON response from the Python `/reason` endpoint, returned unchanged.

## Verification & Testing
Pytest coverage:
- Phase 1: indicator outputs and required columns (`tests/test_phase1.py`).
- Phase 2: feature matrix correctness and NaN removal (`tests/test_phase2.py`).
- Phase 3: target creation, time-split behavior, metrics, and feature importance (`tests/test_phase3.py`).
- Phase 4A: reasoning schema, confidence thresholds, deterministic outputs (`tests/test_phase4a.py`).
- Phase 4B: FastAPI response shape and determinism (`tests/test_phase4b.py`).
- Phase 6B: `/analyze` endpoint contract and behavior (`tests/test_phase6b.py`).
- Phase 7A: sentiment aggregation and `/analyze` sentiment attachment (`tests/test_phase7a.py`).
- Phase 7B: Finnhub news ingestion and precedence rules (`tests/test_phase7b.py`).
- Phase 8A: RAG store/retrieve roundtrip and `/analyze` context (`tests/test_phase8a.py`).

Jest/Supertest coverage:
- Phase 4C: valid requests return 200, Python service call is executed, cache reuse, and invalid requests return 400 (`server/tests/analyze.test.js`).

Determinism guarantees:
- The reasoning core is deterministic for identical inputs.
- Tests explicitly check for identical outputs on repeated calls.
- RAG retrieval is deterministic for identical inputs.

## Operational Notes
Startup order:
- Start the Python FastAPI service first (port 8000).
- Start the Node.js Express service second (port 3000).

Required dependencies:
- Python: pandas, numpy, yfinance, scikit-learn, xgboost, fastapi, uvicorn, pydantic, transformers, torch, httpx, faiss-cpu.
- Node.js: express, axios (and jest/supertest for tests).

Known runtime pitfalls:
- Port conflicts on 8000 or 3000 will prevent services from starting.
- Node service defaults to `http://localhost:8000/reason`; override with `REASONING_URL` if needed.
- Python service requires network access to yfinance.
- Finnhub news ingestion requires `FINNHUB_API_KEY` to be set.
- RAG store path can be overridden with `TRADESENSE_RAG_DIR`.
- Node request timeout is controlled by `REASONING_TIMEOUT_MS`.
- In-memory cache is per-process and clears on restart.

## Decision Log
Accepted decisions:
- Use yfinance as the Phase 1 data source.
- Use deterministic, rule-based reasoning in Phase 4A.
- Expose reasoning via a minimal FastAPI transport in Phase 4B.
- Provide a thin Express API in Phase 4C that forwards payloads without transformation.
- Add an in-memory cache with a 300-second default TTL in the Node service.

Explicitly rejected or deferred decisions:
- Defer Alpaca integration to a later phase.
- Reject LSTM-based modeling for the core logic.
- Reject sentiment analysis and other external data sources in Phase 1 (later phases add optional sentiment/news without altering core inference).

## Known Limitations (As of Now)
- No trading or execution logic exists.
- No authentication or authorization.
- No external database integration; persistence is limited to the local RAG store.
- No model serving beyond the reasoning endpoint.
- No deployment or infrastructure automation.
- RAG context is explainability-only and does not modify probabilities or signals.

## Next Planned Phases (High-Level Only)
No additional phases are defined in the current codebase beyond Phase 8A.
