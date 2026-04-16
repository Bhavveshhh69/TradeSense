# TradeSense

TradeSense is an active-trader workspace for US and India markets. It combines a React operator shell, a Node.js product API, and a Python intraday inference/backtesting service so a trader can lock one instrument, inspect live quote context, run an honest analysis, validate the model on a leak-safe historical window, and manage a paper portfolio from the same workspace.

## Current Status

Verified on `2026-04-16` (Asia/Kolkata / IST):

- The practical product surface is working end to end.
- The app ships with a clean empty state:
  - no seeded holdings
  - no seeded paper trades
  - no seeded recent analyses
- The routed shell is live across:
  - `Today`
  - `Analysis`
  - `Portfolio`
- Supported instrument coverage is live for:
  - US equities
  - India equities
  - curated US benchmark indices
  - curated India benchmark indices

## What The Product Does Now

### 1. Active instrument workflow

- Search and lock a supported symbol from the top bar.
- Keep the selected instrument pinned across every route.
- Show a persistent live quote strip with:
  - current price
  - day change
  - 5-day trend
  - 30-day trend

### 2. Analysis workspace

- Run the live analysis flow for the locked instrument.
- Render the product-safe decision state in the UI:
  - signal
  - confidence
  - execution blockers
  - observed context
  - model honesty note
- Reopen recent analyses directly from the UI.

### 3. Validation workspace

- Run a leak-safe backtest/validation window directly from the product UI.
- Show classification-quality metrics instead of pretending they are P&L:
  - total predictions
  - accuracy
  - expected calibration error
  - Brier score
  - reliability buckets

### 4. Portfolio workspace

- Maintain a ledger-backed paper portfolio.
- Support `BUY`, `SELL`, `SHORT`, `COVER`, and position adjustments.
- Surface:
  - portfolio summary
  - allocation
  - history
  - portfolio insights
  - portfolio advisor
- Keep the paper-trade drawer reachable from every screen.

### 5. Command center surface

- Show market sessions for India and the US.
- Show portfolio risk framing and operator brief.
- Surface recent signals and market headline context when available.

## What Was Completed In This Practicality Pass

- Replaced the generic dashboard flow with a real trader workflow.
- Added strict market-master driven symbol search and normalization.
- Added live Node routes for quote snapshots and market history.
- Added a Node command-center route and recent-analysis persistence.
- Migrated portfolio handling to a ledger-backed trade model.
- Added Python validation/backtest exposure at `/analyze/validate`.
- Fixed India benchmark index resolution in the intraday stack.
- Added frontend unit tests and Playwright browser coverage.
- Fixed top-bar layout issues so:
  - the instrument picker panel opens fully without clipping
  - the primary navigation remains visible instead of collapsing out

## Architecture

```text
React frontend
  -> Node.js Express product API
     -> Python FastAPI intraday + validation service
        -> yfinance market data
        -> trained intraday models
        -> leak-safe validation/backtesting engine
```

### Frontend

Location: `frontend/`

Responsibilities:

- routed operator shell
- active-instrument state
- quote strip and analysis UI
- validation panel
- paper-trade drawer
- portfolio charts and portfolio insight surfaces

Key files:

- `frontend/src/App.jsx`
- `frontend/src/App.css`
- `frontend/src/components/InstrumentPicker.jsx`
- `frontend/src/api/analysis.js`
- `frontend/src/api/symbols.js`
- `frontend/src/api/portfolio.js`
- `frontend/src/api/commandCenter.js`

### Node product API

Location: `backend/node/`

Responsibilities:

- symbol search and normalization
- command-center aggregation
- recent-analysis persistence
- portfolio ledger routes
- market quote/history composition
- frontend-safe orchestration of Python routes

Key routes:

- `GET /api/symbols/search`
- `GET /api/symbols/normalize/:symbol`
- `GET /api/market/quote/:symbol`
- `GET /api/market/history/:symbol`
- `POST /api/analyze`
- `GET /api/analyze/recent`
- `POST /api/analyze/validate`
- `GET /api/command-center`
- `GET /api/portfolio`
- `POST /api/portfolio/trades`
- `GET /api/portfolio/transactions`
- `POST /api/portfolio/positions/:symbol/adjust`

Key files:

- `backend/node/server/index.js`
- `backend/node/server/routes/analyze.js`
- `backend/node/server/routes/market.js`
- `backend/node/server/routes/command-center.js`
- `backend/node/server/services/recent_analysis.service.js`
- `backend/node/portfolio/portfolio.routes.js`

### Python intraday + validation service

Location: `backend/python/`

Responsibilities:

- intraday inference
- latest-price and price-history access
- model-backed prediction outputs
- leak-safe validation/backtests
- market resolution for US and India sessions

Key routes:

- `POST /predict`
- `GET /market/latest-price/{symbol}`
- `GET /market/history/{symbol}`
- `POST /analyze`
- `POST /analyze/validate`
- `POST /reason`

Key files:

- `backend/python/tradesense/api.py`
- `backend/python/tradesense/api_predict.py`
- `backend/python/tradesense/intraday/market.py`
- `backend/python/tradesense/intraday/engine.py`

## Supported Symbol Coverage

The symbol picker and normalization layer are now grounded in a compiled market master:

- US equities from exchange symbol lists
- India equities from NSE symbol lists
- curated benchmark indices for both markets

Verified live examples:

| Query | Normalized | Market | Exchange | Type |
| --- | --- | --- | --- | --- |
| `NVDA` | `NVDA` | `US` | `NASDAQ` | `Equity` |
| `RELIANCE` | `RELIANCE.NS` | `IN` | `NSE` | `Equity` |
| `^GSPC` | `^GSPC` | `US` | `INDEX` | `Index` |
| `^NSEI` | `^NSEI` | `IN` | `NSE` | `Index` |

## Run Locally

### 1. Python service

```bash
cd backend/python
../../.venv/bin/python -m uvicorn tradesense.api:app --host 127.0.0.1 --port 8000
```

### 2. Node service

```bash
cd backend/node
node server/index.js
```

### 3. Frontend

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 4173
```

Frontend dev proxy:

- `/api` -> `http://localhost:3000`
- `/market` -> `http://localhost:8000`

## Verification Results

These are the exact results from the final verification pass on `2026-04-16`.

### Automated verification matrix

| Layer | Command | Result |
| --- | --- | --- |
| Node API | `cd backend/node && npm test -- --runInBand` | `9` suites, `48` tests passed |
| Python | `./.venv/bin/pytest -q backend/python/tests` | `58` tests passed, `1` XGBoost serialization warning |
| Frontend unit | `cd frontend && npm test` | `2` files, `5` tests passed |
| Frontend lint | `cd frontend && npm run lint` | passed |
| Frontend build | `cd frontend && npm run build` | passed |
| Browser suite | `cd frontend && npm run test:e2e` | `4` Playwright tests passed |

Playwright suite coverage includes:

- instrument picker coverage for US equity, India equity, US index, and India index
- analysis + validation UI flow
- global paper-trade drawer workflow
- responsive layout checks

Running `npm run test:e2e` also generates responsive screenshots under `frontend/test-results/`.

### Live browser verification

Verified against the real running stack at `http://127.0.0.1:4173`:

- selected `NVDA` from the real search box
- rendered the live quote strip
- ran the live analysis flow
- ran the live validation flow
- confirmed the top-bar picker no longer clips and the primary nav remains visible

### Live quote verification

Observed live quote snapshots through `GET /api/market/quote/{symbol}` during the final pass:

| Symbol | Market | Price | Day % | 5D % | 30D % | Currency | As Of |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `NVDA` | US | `198.8678` | `0.3217%` | `0.7921%` | `9.5147%` | `USD` | `2026-04-15T19:45:00+00:00` |
| `RELIANCE.NS` | IN | `1344.1000` | `0.0000%` | `0.2686%` | `-3.4688%` | `INR` | `2026-04-16T03:45:00+00:00` |
| `^GSPC` | US | `7022.1001` | `0.0472%` | `0.1379%` | `3.0150%` | `USD` | `2026-04-15T19:45:00+00:00` |
| `^NSEI` | IN | `24394.6992` | `0.7550%` | `0.6658%` | `-3.8236%` | `INR` | `2026-04-16T03:45:00+00:00` |

### Live analysis verification

Observed live `POST /api/analyze` results:

| Symbol | Signal | Decision Label | Confidence | Trade Actionable | Result |
| --- | --- | --- | --- | --- | --- |
| `NVDA` | `NO_TRADE` | `No Trade` | `Strong` | `false` | Entry window is closed |
| `RELIANCE.NS` | `NO_TRADE` | `No Trade` | `Strong` | `false` | Entry window has not opened yet |

This is intentional. The system is now honest about when there is no actionable setup instead of forcing a fake trade call.

### Live validation / backtest results

Observed live `POST /api/analyze/validate` results on the final pass:

Window used by the product:

- start date: `2025-04-16`
- end date: `2026-04-02`
- horizon: `5` trading days

| Symbol | Predictions | Accuracy | ECE | Brier Score | Reliability Points |
| --- | ---: | ---: | ---: | ---: | ---: |
| `NVDA` | `242` | `0.6157024793` | `0.0842813999` | `0.2433023972` | `1` |
| `RELIANCE.NS` | `240` | `0.5250000000` | `0.0051776711` | `0.2492887428` | `1` |

Important interpretation:

- these are directional classification and calibration metrics
- they are not profitability claims
- the product UI explicitly states that validation does not claim P&L

### Clean shipped-state verification

After live verification, the persistence files were reset and rechecked.

Final shipped state:

- `GET /api/portfolio` -> `0` holdings / `0` active positions
- `GET /api/portfolio/transactions` -> `0` transactions
- `GET /api/command-center` -> `0` recent signals

## Data And Persistence Behavior

Current persistence is file-backed for practical local use:

- `backend/node/data/portfolio.json`
- `backend/node/data/portfolio_trades.json`
- `backend/node/data/analysis_recent.json`

These now ship empty by design. Running live analysis or live paper trades will repopulate them during use.

## Environment Notes

Important Node environment variables live in `backend/node/.env`.

Common ones:

- `PORT`
- `REASONING_URL`
- `PYTHON_API_BASE_URL`
- `FINNHUB_API_KEY`
- `ALPHA_VANTAGE_API_KEY`
- `GROQ_API_KEY`

## Known Limitations

- Persistence is still JSON-file based, so this is a local/single-user setup rather than a production multi-user platform.
- Frontend production build still emits a large chunk warning for the main JS bundle.
- Live news context and AI explanation richness still depend on external provider availability and configured API keys.
- Validation is an empirical classification check, not a brokerage-grade execution simulator.

## Technical Intent

The technical goal of this pass was to turn TradeSense from a partially connected prototype into a product surface that behaves like one consistent trader workflow:

- one pinned instrument
- one quote context
- one honest analysis surface
- one empirical validation surface
- one ledger-backed paper portfolio

That intent is now delivered and verified end to end.

## Business Intent And What This Unlocks

This pass changes TradeSense from “an ML demo with scattered screens” into “an operator-ready trading workspace.”

What this unlocks:

- A trader can move from symbol selection to analysis to validation to paper execution without losing context.
- The product can now support both US and India workflows in the same shell instead of being biased toward one market.
- Validation can be shown to users, stakeholders, or pilots as evidence of model behavior without overclaiming profitability.
- Portfolio workflows can start from a clean slate, which makes demos, testing, and onboarding credible.
- The app now behaves like a product that can be evaluated for real usability, not just backend correctness.
