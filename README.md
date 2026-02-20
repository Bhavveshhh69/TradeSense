# TradeSense

TradeSense is a multi-service market analysis system with calibrated ML inference, deterministic decision/context generation, explainability, and optional sentiment/news/context layers.

## Project Overview

TradeSense combines:
- Python FastAPI backend for inference and reasoning.
- Node.js Express gateway for `/api/analyze` routing/caching.
- React frontend dashboard for symbol-driven analysis UI.

## Current System Capabilities

- Calibrated probability inference from engineered technical features.
- Deterministic trade decision mapping (`BUY` / `SELL` / `HOLD`) with confidence and strength.
- Deterministic context summaries for trend/risk interpretation.
- Full analysis endpoint with optional sentiment/news/RAG/LLM explanation flags.
- Backtesting and calibration evaluation (accuracy, ECE, Brier, reliability buckets).

## Architecture Diagram

```text
Frontend (React + Vite, http://localhost:5173)
  -> POST /api/analyze
Node Gateway (Express, http://localhost:3000)
  -> validate/normalize symbol + cache + POST /predict
Python FastAPI (http://127.0.0.1:8000)
  -> POST /predict  (predictor -> decision -> context)
  -> POST /analyze  (orchestrator + calibration + reasoning + optional extras)
  -> POST /reason   (deterministic reasoning-only)
Artifacts:
  -> backend/python/tradesense/models/xgboost.joblib
  -> backend/python/tradesense/rag_store/
```

## Component Descriptions

- `backend/python/tradesense/api.py`: FastAPI app with `/analyze` and `/reason`.
- `backend/python/tradesense/api_predict.py`: `/predict` route and runtime singletons.
- `backend/python/tradesense/inference/predict.py`: strict feature-schema predictor with calibrated output.
- `backend/python/tradesense/inference/decision_engine.py`: probability/confidence validation and decision mapping.
- `backend/python/tradesense/inference/context_engine.py`: deterministic context generation.
- `backend/python/tradesense/inference/orchestrator.py`: full Phase 6A analysis path.
- `backend/python/tradesense/backtesting/*`: empirical validation and reliability reporting.
- `backend/node/server/*`: gateway route, validation middleware, cache, Python forwarding.
- `frontend/src/pages/Dashboard.jsx`: UI rendering backend response fields.

## Phase Completion Status

| Area | Status |
| --- | --- |
| Data + Features + Modeling (Phases 1-3) | Complete |
| Deterministic Reasoning API (Phase 4A/4B) | Complete |
| Inference + Analyze API (Phase 6A/6B) | Complete |
| Sentiment + News (Phase 7A/7B) | Complete (optional dependencies/keys) |
| RAG Context + LLM Explain (Phase 8A/8B) | Complete (optional usage) |
| Calibration Discipline (Phase 9) | Complete |
| Explainability Package (Phase 10) | Complete |
| Backtesting + Empirical Validation (Phase 11/11.5) | Complete |
| Production Training Bundle (Phase 12) | Complete |
| Predictor/Decision/Context + `/predict` (Phase 13/14) | Complete |

## Installation Instructions

### Python backend

```powershell
cd backend/python
python -m pip install -r requirements.txt
```

### Node gateway

```powershell
cd backend/node
npm install
```

### Frontend

```powershell
cd frontend
npm install
```

## Running TradeSense Locally

### 1. Start Python backend (FastAPI)

```powershell
cd backend/python
. .venv\Scripts\Activate.ps1
python -m uvicorn tradesense.api:app --host 127.0.0.1 --port 8000 --reload --reload-dir tradesense
```

Expected URL: `http://127.0.0.1:8000`

### 2. Start Node gateway

```powershell
cd backend/node
node server/index.js
```

Expected URL: `http://localhost:3000`

### 3. Start frontend

```powershell
cd frontend
npm run dev
```

Expected URL: `http://localhost:5173`

### 4. Test `/predict` endpoint

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-raw '{"symbol":"AAPL"}'
```

You should receive JSON with `decision`, `probability`, `confidence`, `strength`, and `context` fields.

## API Documentation

### `POST /predict`

Request:

```json
{
  "symbol": "AAPL"
}
```

Response (example):

```json
{
  "symbol": "AAPL",
  "prediction": 1,
  "probability": 0.53,
  "confidence": 0.53,
  "decision": "HOLD",
  "confidence_level": "very_low",
  "strength": 0.07,
  "context": {
    "decision_summary": "Model prediction is within neutral zone. No directional edge detected.",
    "confidence_summary": "Prediction confidence is extremely weak. Signal reliability is poor.",
    "strength_summary": "Signal strength is very weak.",
    "trend_summary": "Market trend is mixed or transitional.",
    "risk_summary": "Market risk conditions are normal.",
    "model_summary": "Prediction generated using calibrated XGBoost model version phase12.",
    "generated_at": "2026-02-20T17:15:57.391162+00:00"
  },
  "model_version": "phase12",
  "timestamp": "2026-02-20T17:15:57.390587+00:00",
  "generated_at": "2026-02-20T17:15:57.391162+00:00"
}
```

### `POST /analyze`

Runs data fetch, feature build, model inference, calibration, deterministic reasoning, and optional sentiment/news/context/explain branches.

Base request:

```json
{
  "symbol": "AAPL"
}
```

Optional flags:
- `use_news` (bool)
- `include_context` (bool)
- `explain` (bool)
- `news` (list of strings)

### `POST /reason`

Reasoning-only endpoint. Caller provides probability/states/feature info and receives deterministic structured insight.

## Model and Calibration Description

- Persisted model bundle: `backend/python/tradesense/models/xgboost.joblib`.
- Bundle keys: `model`, `feature_names`, `calibrator`, `calibration_meta`.
- Calibration method: Platt scaling (`calibration_meta.method = "platt"`).
- Inference enforces strict feature schema (exact names + order + no NaN).

## Backtesting Description

Backtesting modules are under `backend/python/tradesense/backtesting/`.

Included outputs:
- `overall_accuracy`
- `expected_calibration_error` (ECE)
- `brier_score`
- `accuracy_by_probability_bucket`
- Reliability curve buckets (`probability_mean`, `accuracy`, `count`)

Runner:

```powershell
cd backend/python
python -m tradesense.backtesting.run_validation --symbol AAPL
```

## Explainability and Reasoning Description

- Deterministic explainability: attribution normalization + rule templates (`backend/python/tradesense/explainability/*`).
- Deterministic reasoning output: `symbol`, calibrated probability fields, confidence reason, market context, key drivers, risk notes.
- Optional LLM explanation branch (`explain=true`) with strict prompt guardrails.

## Frontend Integration Description

- Frontend API call path: `frontend/src/api/analyze.js` -> `POST /api/analyze`.
- Frontend request body for gateway: `{"symbol":"AAPL"}`.
- Dashboard renders decision/probability/confidence/strength/context from backend response (`frontend/src/pages/Dashboard.jsx`).
- Gateway request contract is `{"symbol":"AAPL"}` (symbol-only); gateway trims/uppercases and forwards to FastAPI `POST /predict`.

## Project Structure Overview

```text
backend/
  node/
    server/
  python/
    tests/
    tradesense/
      backtesting/
      explainability/
      explainer/
      inference/
      models/
      news/
      rag/
      rag_store/
      sentiment/
frontend/
  src/
docs/
```

## Known Limitations

- Optional features depend on environment setup:
  - `FINNHUB_API_KEY` for news ingestion.
  - `OPENAI_API_KEY` for `explain=true` LLM explanations.
- Timestamp fields make responses non-identical across repeated calls.

## Future Roadmap

1. Migrate Pydantic v1 validators/config to v2 style to remove deprecation warnings.
2. Improve `/analyze` server-side error logging strategy (structured logs instead of tracebacks).
3. Expand gateway integration tests to include live FastAPI `/predict` smoke coverage.
