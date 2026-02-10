# TradeSense

## What TradeSense Is
- A Python-based analytics stack for market data, indicator computation, feature engineering, probabilistic modeling, and deterministic reasoning.
- A FastAPI service that exposes the reasoning core via `POST /reason` and the full inference pipeline via `POST /analyze` on port 8000.
- Optional news ingestion (Finnhub) and FinBERT sentiment scoring through `/analyze`.
- A local, file-based RAG memory store for historical insight context (no external database).
- A Node.js Express API that exposes `POST /api/analyze` on port 3000, validates input, caches responses briefly, and forwards payloads to Python.
- A React (Vite) frontend on port 5173 that accepts a symbol input, calls the Node API, and renders the reasoning output.

## What TradeSense Is Not
- A trading or execution system.
- A brokerage integration or order manager.
- An externally managed database-backed service (persistence is limited to the local RAG store).
- An authenticated or multi-tenant service.
- An LLM-based generation or chat system.

## Architecture (Phase 8A)

```
React (Vite, http://localhost:5173)
  POST /api/analyze
Node.js Express (http://localhost:3000)
  POST /reason
Python FastAPI (http://localhost:8000)
  POST /reason (reasoning core)
  POST /analyze (inference + optional news/sentiment + RAG context)
  Local RAG store (file-based)
```

## Phase 9: Calibration & Confidence Discipline (Locked)
This exists to prevent overconfident outputs. Raw model probabilities are often miscalibrated, so Phase 9 adds a post-model calibration step and a volatility-aware confidence cap.

**Probability contract (mandatory calibration)**  
- `probability` = calibrated probability  
- `probability_raw` = raw model output  
Calibration is required; missing artifacts fail fast.

Example response excerpt:
```json
{
  "probability": 0.62,
  "probability_raw": 0.66,
  "probability_calibrated": 0.62,
  "confidence_level": "moderate",
  "confidence_reason": "Medium volatility regime reduces certainty; confidence capped at moderate."
}
```

For full design rationale and failure modes, see `docs/phase-9-calibration.md`.

## Backend vs Frontend Responsibilities
- Python FastAPI: Exposes `/reason` (deterministic reasoning core) and `/analyze` (full pipeline with optional news/sentiment and RAG context); persists insight summaries to the local RAG store.
- Node.js Express: Validates requests, forwards `payload` to Python, applies a short-lived in-memory cache, and returns the Python response as-is.
- React frontend: Accepts a stock symbol input (normalized to uppercase), disables analysis when empty, sends requests to `/api/analyze`, renders the response JSON verbatim, and surfaces network errors in the UI.

## Folder Structure

```
backend/
  python/
  node/
frontend/
docs/
```

## Run Locally (Python + Node + React)

## Startup Order
1. Start the Python FastAPI service first (port 8000).
2. Start the Node.js Express service second (port 3000).
3. Start the React frontend last (port 5173).

Python service startup (Windows)

Run these commands from the repository root. They assume the project Python virtual environment is at `backend/python/.venv`.

PowerShell (recommended on Windows)

```powershell
cd backend/python
. .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Start FastAPI with reload; scope reload to the `tradesense` package so edits there trigger reload
python -m uvicorn tradesense.api:app --host 127.0.0.1 --port 8000 --reload --reload-dir tradesense
```

Command Prompt (cmd.exe)

```cmd
cd backend\python
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m uvicorn tradesense.api:app --host 127.0.0.1 --port 8000 --reload --reload-dir tradesense
```

Notes:
- Always run the `python -m uvicorn tradesense.api:app` command from `backend/python` so the local `tradesense` package is importable without setting PYTHONPATH.
- Use `--reload --reload-dir tradesense` to limit auto-reload watching to the package directory (helps reliable reload on Windows).
- Do not run plain `uvicorn` (use `python -m uvicorn` as shown).

Optional (lower latency hot loop + HTTP parser)

```powershell
cd backend/python
python -m pip install uvloop httptools
python -m uvicorn tradesense.api:app --host 127.0.0.1 --port 8000 --loop uvloop --http httptools
```

Node service startup

```powershell
cd backend/node
npm install
node server/index.js
```

React frontend startup

```powershell
cd frontend
npm install
npm run dev
```

## Port Usage

| Service | Port | Purpose |
| --- | --- | --- |
| Python FastAPI | 8000 | `POST /reason` reasoning endpoint, `POST /analyze` full inference endpoint |
| Node.js Express | 3000 | `POST /api/analyze` API gateway |
| React (Vite) | 5173 | Frontend UI and `/api` proxy |

## One End-to-End curl Example

Replace `SYMBOL` with the stock symbol you want to analyze.

```powershell
curl.exe -s -X POST http://127.0.0.1:3000/api/analyze -H "Content-Type: application/json" --data-raw '{"symbol":"SYMBOL","payload":{"symbol":"SYMBOL","probability":0.62,"feature_importance":{"price_vs_ema20":0.25,"rsi_slope_3":0.2,"macd_hist_accel":0.15,"volume_ratio":0.1,"price_vs_ema50":0.05},"feature_values":{"price_vs_ema20":0.02,"rsi_slope_3":-0.1,"macd_hist_accel":0,"volume_ratio":1.2,"price_vs_ema50":-0.01},"trend_state":1,"momentum_state":1,"risk_state":1}}'
```

Example response for the payload above

Note: `probability` is the calibrated probability. `probability_raw` is the raw model output.

```json
{
  "symbol": "SYMBOL",
  "probability": 0.62,
  "probability_raw": 0.66,
  "probability_calibrated": 0.62,
  "confidence_level": "moderate",
  "confidence_reason": "Medium volatility regime reduces certainty; confidence capped at moderate.",
  "summary": "Moderate continuation bias",
  "market_context": {
    "trend": "bullish",
    "momentum": "strengthening",
    "volatility": "medium"
  },
  "key_drivers": [
    "Price above EMA20",
    "RSI momentum weakening",
    "MACD momentum flat"
  ],
  "risk_notes": [],
  "model_honesty": "Confidence is aligned with probability strength."
}
```

## Direct Python /analyze Example (Optional Context)

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" --data-raw '{"symbol":"SYMBOL","include_context":true}'
```

## Common Runtime Errors
- Network error in frontend: Ensure the Node service is running on port 3000 and the Vite dev server is running; the frontend posts to `/api/analyze` through the Vite proxy.
- Port already in use: Stop the process using the port or change the startup port (Python `--port`, Node `PORT`, Vite `--port`) and update any dependent proxy settings.
- Backend not running: Start the Python service first and the Node service second; the Node API depends on the Python `/reason` endpoint.
- News ingestion failing: Set `FINNHUB_API_KEY` before calling `/analyze` with `use_news`.
- RAG store errors: Ensure `TRADESENSE_RAG_DIR` is writable if you override the default location.

## Performance Notes (Phase 4D)
- Optional uvloop and httptools can reduce Python event-loop and HTTP parsing latency.
- The Node in-memory cache TTL is configurable via `CACHE_TTL_SECONDS` (default 300 seconds).
- Optional pre-warm on Node startup: set `REASONING_PREWARM=true` to send a single dummy request to `/reason`.
- Node logs minimal timings per request: `timing total_ms=... python_ms=...`.
- Node uses `REASONING_URL` and `REASONING_TIMEOUT_MS` for Python connectivity and timeouts.

## Known Limitations
- No trading, execution, or brokerage integration.
- No authentication or authorization.
- No external database; persistence is limited to the local RAG store.
- In-memory cache is per-process and clears on restart.
- Market data retrieval depends on yfinance availability.
- RAG context is explainability-only and does not alter probabilities or signals.

## Current Project Status
Backend complete through Phase 8A (RAG memory and context). Optional FinBERT sentiment (Phase 7A) and Finnhub news ingestion (Phase 7B) are available via `/analyze`. Phase 5B React frontend symbol input remains wired to the Node API.
