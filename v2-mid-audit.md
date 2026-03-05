# v2 Mid Audit — TradeSense

Audit date: 2026-03-04  
Repository snapshot: current workspace state (`e:\Bhavesh Files\TradeSense`)  
Audit method: static code-path inspection across Node, Python, React, configs, and tests; plus test/build execution.

---

## SECTION 1 — Project Overview

TradeSense is an AI-assisted portfolio analysis system with three active layers:

- Frontend: React 19 + Vite + Axios + Recharts (`frontend/`)
- Backend API Gateway: Node.js (Express + Axios) (`backend/node/`)
- ML/Market Service: Python FastAPI + pandas/numpy + scikit-learn/XGBoost + yfinance (`backend/python/`)

Market data source in production code:

- Primary price/market source: `yfinance` (US + NSE/BSE symbols via Yahoo ticker format)
- Optional news source: Finnhub (`FINNHUB_API_KEY` required)

Subsystem purpose:

- React frontend: user interaction for analysis + portfolio tracking, chart rendering, portfolio insights/advisor UI.
- Node backend: product-facing API (`/api/*`), symbol normalization, holdings persistence (JSON file), portfolio analytics aggregation, and proxying to Python market/prediction endpoints.
- Python service: prediction pipeline, latest/historical market endpoints, deterministic reasoning, optional sentiment/news/RAG/explainer paths, model training/backtesting utilities.

---

## SECTION 2 — Architecture Map

### 2.1 High-Level Interaction

```text
User
  -> React UI
    -> Node API (/api/analyze, /api/portfolio/*, /api/symbols/*)
      -> Python API (/predict, /market/latest-price, /market/history)
        -> yfinance market data
      <- Python responses
    <- Node-enriched responses
  <- UI render
```

Optional Python-only path (implemented, not used by current React screens):

```text
Client -> Python /analyze -> orchestrator + sentiment/news + RAG + optional LLM explain
```

### 2.2 Request Flows Required by Audit

#### Portfolio data flow

1. React `PortfolioPage` loads `/api/portfolio`.
2. Node `portfolio.service.getHoldings()` reads holdings from `backend/node/data/portfolio.json`.
3. For each holding, Node calls Python `/market/latest-price/{symbol}`.
4. Node computes current value/P&L/summary and returns payload.
5. React renders summary + table + allocation chart.

#### Stock analysis flow

1. React `AnalysisPage` sends `{ symbol }` to Node `/api/analyze`.
2. Node validates, normalizes symbol, checks in-memory cache.
3. Node calls Python `/market/latest-price/{symbol}` and `/predict`.
4. Python `/predict` runs model pipeline and returns prediction + decision + context.
5. Node enriches with current price and computes signal/recommendation fields.
6. React renders decision cards + chart (chart calls `/market/history/{symbol}` directly via Vite proxy).

#### Market price retrieval flow

- Node analysis/portfolio endpoints call Python `/market/latest-price/{symbol}`.
- Python endpoint fetches via `get_market_data()` (yfinance) and returns latest close.

#### Historical data flow

- Analysis chart: React -> Python `/market/history/{symbol}` directly.
- Portfolio equity curve: React -> Node `/api/portfolio/history` -> Python `/market/history/{symbol}` per symbol -> Node aggregates curve.

#### Prediction pipeline flow (`/predict`)

1. Validate symbol.
2. Fetch ~220 days market data via `get_market_data()`.
3. Build feature matrix.
4. Select latest row.
5. `TradeSensePredictor.predict_from_features()` (calibrated probability).
6. `TradeSenseDecisionEngine` maps to BUY/SELL/HOLD + confidence level.
7. `TradeSenseContextEngine` builds trend/risk/context summaries.
8. Return structured response.

---

## SECTION 3 — Implemented Feature Inventory

| Feature | Purpose | Key files | Internal behavior |
|---|---|---|---|
| Portfolio tracker | Add/list/delete holdings and compute valuation | `backend/node/portfolio/*`, `frontend/src/pages/Portfolio/*` | Holdings persisted in JSON; Node enriches each row with live price; frontend renders table/summary. |
| Holdings storage | Durable local holding records | `backend/node/portfolio/portfolio.repository.js`, `backend/node/data/portfolio.json` | Atomic temp-file write + rename, serialized write chain, sanitation on read. |
| Market price retrieval | Current price for any ticker | `backend/python/tradesense/api_predict.py` (`/market/latest-price`) | Pulls data via yfinance-backed provider, returns latest close + timestamp. |
| Portfolio equity curve | Time-series portfolio value | `backend/node/portfolio/portfolio.service.js`, `frontend/src/components/portfolio/PortfolioEquityChart.jsx` | Node fetches each symbol history and carries forward last known prices by date. |
| Portfolio intelligence metrics | Concentration/diversification/volatility/performance insights | `backend/node/portfolio/portfolio.service.js`, `frontend/src/components/portfolio/PortfolioInsights.jsx` | Computes risk bands, best/worst performer, diversification score, volatility level, insight text. |
| Portfolio advisor | Actionable recommendation strings | `backend/node/portfolio/portfolio.service.js`, `frontend/src/components/portfolio/PortfolioAdvisor.jsx` | Rule-based recommendations from insights (concentration/diversification/volatility/profit booking). |
| Portfolio allocation chart | Weight visualization | `frontend/src/components/portfolio/PortfolioAllocationChart.jsx` | Uses current values to compute percentages and render Recharts pie. |
| Stock analysis engine | Per-symbol prediction + context | `backend/node/server/routes/analyze.js`, `backend/python/tradesense/api_predict.py` | Node calls `/predict` and price endpoint, adds signal/recommendation metadata. |
| Trading signal engine | Surface strength/market condition/recommendation | `backend/node/server/routes/analyze.js` | Derived from probability thresholds + trend text parser. |
| Symbol normalization | Map user symbol to exchange suffix format | `backend/node/symbols/*`, `frontend/src/pages/Portfolio/AddHoldingForm.jsx` | Static symbol registries resolve NSE (`.NS`) / BSE (`.BO`) / US tickers. |
| Historical charting | 30-day close chart in analysis UI | `frontend/src/components/analysis/StockPriceChart.jsx`, `backend/python/tradesense/api_predict.py` | Frontend calls Python market history endpoint and plots close series. |
| Deterministic reasoning core | Structured explanation from model/state inputs | `backend/python/tradesense/reasoning_core.py`, `tradesense/explainability/*` | Validates inputs, applies confidence/risk rules, generates drivers/risk notes/honesty string. |
| Optional sentiment/news | News fetch + FinBERT sentiment | `tradesense/news/*`, `tradesense/sentiment/*`, `tradesense/api.py` | Optional via `/analyze`; manual news or Finnhub fetch -> FinBERT aggregation. |
| Optional RAG context + explainer | Historical context and optional LLM narrative | `tradesense/rag/*`, `tradesense/explainer/*`, `tradesense/api.py` | Stores insights in local vector store, retrieves summaries, optional OpenAI explanation JSON. |

---

## SECTION 4 — Data Flow Audit

### 4.1 Symbol input propagation

- Analysis input is uppercased in React and validated in Node middleware.
- Node attempts normalization via symbol service; on failure it falls back to raw symbol.
- Portfolio add flow normalizes on frontend (`/api/symbols/normalize`) and again in Node service before persistence.

Contract quality:

- Strong for portfolio ticker format (`portfolio.model.js` regex).
- Weaker for `/api/analyze`/`/predict` (non-empty string only; no max length/pattern).

### 4.2 Market price data propagation

- Current prices: Python `/market/latest-price` -> Node enrichment -> frontend cards/table.
- History prices: Python `/market/history` -> analysis chart directly or Node portfolio history aggregator.

Transformation correctness:

- Numeric validation exists at each stage.
- Symbol echo checks exist in Node before assigning prices.

Fragility:

- `latest-price` and `history` use different acquisition logic (provider vs ticker history), which can produce inconsistencies.

### 4.3 Prediction result propagation

- `/predict` returns prediction/decision/context fields.
- Node `/api/analyze` preserves these fields and adds `current_price`, `signal_strength`, `recommendation`, etc.
- Frontend consumes both original prediction fields and Node-derived recommendation fields.

Fragility:

- Node recommendation logic can diverge from model decision logic (details in Section 6).

### 4.4 Portfolio metrics propagation

- Holdings are enriched with current prices in Node service.
- Summary + insights + advisor are all recomputed server-side from enriched data.
- React stores base portfolio payload in parent state, but insights/equity/advisor components fetch separately.

State propagation issue:

- After add/delete, `PortfolioInsights` and `PortfolioEquityChart` are not explicitly refreshed (stale UI risk).

### 4.5 API contract checks

Contracts are generally explicit and consistent for:

- `/predict`, `/market/latest-price`, `/market/history`
- `/api/portfolio*`, `/api/analyze`, `/api/symbols/*`

Main contract gaps:

- No currency metadata in holdings or valuation responses.
- `/analyze` validation errors in Python often collapse to generic symbol error message.

---

## SECTION 5 — Currency Handling Audit (CRITICAL)

### 5.1 Current implementation status

- Holdings do **not** store `exchange`, `currency`, `fx_rate`, or base-currency value.
- Symbol normalization detects exchange suffixes (`.NS`, `.BO`) only for a small static registry.
- No FX conversion pipeline exists in Node or Python valuation paths.

### 5.2 Impact of mixed-currency portfolios

If portfolio includes symbols like `AAPL` (USD) and `RELIANCE.NS` (INR):

- `total_portfolio_value` is a raw sum of USD + INR numbers.
- `total_invested_value` is also raw mixed-unit sum.
- Portfolio P&L, allocation weights, concentration risk, equity curve, volatility metrics, and advisor recommendations become distorted.

Affected calculations:

- `getHoldings()` summary totals
- `getPortfolioHistory()` equity curve values
- `getPortfolioInsights()` concentration/diversification/volatility/performance
- `getPortfolioAdvisor()` recommendation output
- Frontend allocation chart and all portfolio-level figures

### 5.3 Correct architecture recommendation

Adopt canonical holding/value schema:

- `symbol`
- `exchange`
- `instrument_currency`
- `price_native`
- `fx_rate_to_base`
- `price_base`
- `market_value_base`
- `cost_basis_base`
- `base_currency` (portfolio-level)

Required flow:

1. Resolve symbol -> exchange -> currency.
2. Fetch native price.
3. Fetch FX rate (`instrument_currency -> base_currency`).
4. Convert all valuation/risk/advisor metrics to base currency.
5. Preserve native and base values in API for transparency.

Without this, mixed-market support remains analytically incorrect.

---

## SECTION 6 — Logic and Bug Detection

Severity scale: Critical / High / Medium / Low.

1. **Critical — Multi-currency portfolio distortion**
- Location: `backend/node/portfolio/portfolio.service.js` + frontend portfolio views.
- Issue: Mixed currency values are summed directly.
- Impact: Total value, P&L, allocation, risk metrics, advisor guidance are invalid for mixed US/India portfolios.

2. **High — Signal direction bug in Node trading signal layer**
- Location: `backend/node/server/routes/analyze.js`.
- Issue: `signal_strength` uses only absolute bullish-side thresholds (`probability < 0.55 => NO_EDGE`), so bearish edge probabilities are not represented correctly.
- Impact: Strong bearish probabilities can be labeled as weak/no-edge; recommendation quality degrades.

3. **High — Recommendation can conflict with model decision**
- Location: `backend/node/server/routes/analyze.js`.
- Issue: Recommendation is driven by trend text + strength, not model direction semantics.
- Impact: Potential mismatch between `decision` from `/predict` and Node-generated `recommendation`.

4. **High — Portfolio subviews can be stale after mutations**
- Location: `frontend/src/pages/Portfolio/PortfolioPage.jsx`, `PortfolioEquityChart.jsx`, `PortfolioInsights.jsx`.
- Issue: Equity and insights components fetch only on mount/`days` change; add/delete does not trigger their reload.
- Impact: User sees updated holdings table but stale curve/insights until full refresh.

5. **Medium — Adjusted/unadjusted price inconsistency**
- Location: Python `/market/latest-price` vs `/market/history` implementations.
- Issue: Different data paths/settings can produce non-identical latest values vs chart endpoints.
- Impact: Current price shown may not match latest plotted close.

6. **Medium — Latest price endpoint over-coupled to indicator pipeline**
- Location: `tradesense/api_predict.py` -> `_load_latest_close_price()` via `get_market_data()`.
- Issue: Current-price fetch depends on indicator-enriched dataset pipeline.
- Impact: Short-history/new listings may fail availability checks unnecessarily.

7. **Medium — Optional RAG acts as mandatory failure point for `/analyze`**
- Location: `tradesense/api.py`.
- Issue: `store_insight` block executes unconditionally; errors return 500.
- Impact: Context feature instability can break core analyze path.

8. **Medium — Frontend icon rendering mojibake**
- Location: `PortfolioInsights.jsx`, `PortfolioAdvisor.jsx`.
- Issue: emoji/icon literals appear encoding-corrupted.
- Impact: UI quality/clarity degradation.

9. **Low — Monolithic service file complexity**
- Location: `backend/node/portfolio/portfolio.service.js` (~750 lines).
- Issue: data access, enrichment, history logic, analytics, and advisor logic tightly coupled.
- Impact: higher regression risk and harder maintenance.

---

## SECTION 7 — Code Quality Assessment

### Modularity

Strengths:

- Python layer is well segmented: provider, features, inference, calibration, explainability, backtesting, API adapters.
- Node has separated route/controller/service/repository modules.

Weak areas:

- Node portfolio service is too broad and should be split into pricing, history, analytics, recommendations modules.
- Frontend contains duplicate/legacy API/page modules (`api/analyze.js`, `pages/Dashboard.jsx`) not in active flow.

### Service boundaries

- Boundaries are clear between Node gateway and Python ML service.
- React analysis chart bypasses Node and calls Python `/market/*` directly (works in dev proxy, weakens single-backend boundary).

### API design

- Overall clean JSON contracts and explicit endpoints.
- Missing cross-currency schema fields is the biggest contract hole.

### Error handling

- Node generally returns user-friendly errors.
- Python uses broad exception handling in places and emits tracebacks; client detail is sometimes too generic.

### Test coverage

Executed in this audit:

- Node: 33 tests passed (Jest/Supertest).
- Python: 49 tests passed (pytest).
- Frontend: build succeeds; bundle warning about large chunk.

Coverage gaps:

- No pytest assertions directly validating `tradesense/inference/test_*.py` scripts (they are script-style, not test functions).
- Limited end-to-end tests spanning React -> Node -> Python with live data contracts.

---

## SECTION 8 — Performance Considerations

1. Duplicate expensive calls on portfolio page load.
- Frontend loads `/api/portfolio`, `/api/portfolio/history`, `/api/portfolio/insights`, `/api/portfolio/advisor` separately.
- Advisor endpoint internally calls insights pipeline again.
- Result: repeated price/history fetching for same holdings.

2. Sequential per-holding latest-price fetch.
- `getHoldings()` fetches prices one-by-one, not in parallel.
- Latency scales poorly with portfolio size.

3. Repeated symbol registry file reads.
- Symbol normalization/search reads JSON registries each call.
- In-memory registry cache would reduce I/O.

4. Analyze endpoint calls two Python requests serially.
- Node `/api/analyze` fetches `/market/latest-price` then `/predict` sequentially.

5. Frontend bundle size warning.
- Build output JS chunk ~622 KB minified.
- Likely from charting + app code consolidated into one chunk.

6. RAG writes on every `/analyze` in Python path.
- Adds synchronous filesystem/embedding overhead even when context is not requested.

---

## SECTION 9 — Security and Stability

### Input validation / sanitization

- Portfolio add endpoint has strong ticker pattern checks.
- Analyze/predict endpoints only enforce non-empty string symbols (no strict pattern/length guard).

### Network failure handling

- Node handles price/history failures gracefully and tags `price_error`.
- Python market/news providers handle not-found and provider errors, but optional service failures can still bubble as 500.

### Stability risks

- JSON-file storage is single-node/local; no multi-process lock semantics.
- Runtime mutable data (`portfolio.json`, `rag_store/*`) exists in repository tree, increasing accidental commit risk.
- No authentication/authorization/rate limiting on APIs.

### Secrets and external dependencies

- Uses `FINNHUB_API_KEY` and `OPENAI_API_KEY` env vars.
- Optional services depend on external network availability.

---

## SECTION 10 — Development Progress Summary

Proposed phase status against implemented system:

1. **Phase 1 — Portfolio Foundation**: **Completed**  
Implemented add/list/delete holdings, file persistence, valuation summary.

2. **Phase 2 — Portfolio Intelligence**: **Completed (with currency limitation)**  
Implemented concentration/diversification/volatility/performance insights and equity curve.

3. **Phase 3 — Analysis System**: **Completed**  
Implemented analysis UI + Node gateway + Python `/predict` pipeline + charting.

4. **Phase 4 — Signal Engine**: **Partially completed**  
Signal/recommendation fields are implemented, but directionality logic has defects.

5. **Phase 5 — Portfolio Advisor**: **Completed (quality limited by upstream metrics)**  
Rule-based advisor is live and integrated.

6. **Phase 6 — Allocation Chart**: **Completed**  
Recharts-based allocation visualization implemented.

Additional implemented platform phases:

- Calibration, explainability, backtesting, production training, and optional news/sentiment/RAG/explainer paths are all present in backend code.

---

## SECTION 11 — Next Development Roadmap

Recommended sequence for next features:

1. **Currency-aware portfolio calculations (foundation first)**
- Introduce instrument currency + FX conversion + base currency accounting.
- Make all portfolio analytics and advisor logic base-currency normalized.

2. **AI Portfolio Assistant**
- Conversational assistant over normalized portfolio + insights.
- Must be grounded in deterministic metrics to prevent hallucinated advice.

3. **Portfolio Screenshot Import (OCR)**
- Parse broker screenshots into structured holdings.
- Add symbol normalization + exchange/currency resolution + user confirmation workflow.

4. **Sector Exposure Analysis**
- Map holdings to sectors/industries and compute base-currency-weighted exposure.
- Add concentration alerts at sector level.

5. **Portfolio Risk Scoring**
- Composite risk score combining concentration, volatility, drawdown, and beta-like proxies.
- Calibrate score bands and explanation text.

6. **Smart Rebalancing Engine**
- Suggest target allocation shifts using risk caps, sector limits, and tax/friction constraints.
- Keep as recommendation-only (no execution) unless explicitly expanded later.

---

## SECTION 12 — Immediate Fix Priorities

Ranked by urgency before major new feature work:

1. **Implement multi-currency data model and FX conversion** (Critical)
2. **Correct Node signal/recommendation direction logic** (High)
3. **Fix stale portfolio insights/equity refresh behavior after add/delete** (High)
4. **Remove duplicated portfolio fetch computations and parallelize price enrichment** (High)
5. **Unify latest-price/history pricing methodology for consistency** (Medium)
6. **Decouple optional RAG failures from core `/analyze` success path** (Medium)
7. **Harden symbol validation contracts for analyze/predict endpoints** (Medium)
8. **Migrate Pydantic v1 validators/config to v2 patterns** (Medium)
9. **Resolve frontend text/icon encoding corruption** (Low)
10. **Refactor monolithic portfolio service into smaller modules** (Low/Medium)

---

## SECTION 13 — Final Audit Summary

### System strengths

- Strong end-to-end architecture separation (UI, Node gateway, Python ML service).
- Clear prediction pipeline with explicit decision/context layering.
- Broad test coverage across phases and stable local build/test status.
- Useful portfolio feature breadth already implemented.

### Architectural soundness

- Sound for a local/service-oriented analysis platform.
- Major architectural gap is currency-awareness in cross-market portfolio math.
- Secondary gap is duplicated computation paths causing avoidable latency/staleness.

### Major risks

- Cross-currency valuation is currently incorrect for mixed US/India portfolios.
- Signal engine logic can misrepresent bearish edges and conflict with model outputs.
- Portfolio dashboard can display stale sub-panels after portfolio mutations.

### Readiness for next phase development

- **Ready for next phase only after critical fixes** (especially currency normalization and signal correctness).
- For single-currency demo usage, system is functional.
- For real mixed-market decision support, current state is not yet reliable.

---

## Verification Run During Audit

- Node tests: `33 passed`
- Python tests: `49 passed` (with Pydantic deprecation warnings)
- Frontend build: success (bundle-size warning)

