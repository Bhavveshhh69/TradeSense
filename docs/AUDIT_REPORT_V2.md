# TradeSense Audit Report v2

Audit date: 2026-02-20  
Repository: `E:\Bhavesh Files\TradeSense`  
Audit mode: documentation and verification only (no logic changes)

## System Architecture Overview

Implemented runtime architecture:

```text
Frontend (React/Vite, :5173)
  -> POST /api/analyze
Node Gateway (Express, :3000)
  -> validates symbol-only request
  -> trims + uppercases symbol
  -> POST FastAPI /predict
Python API (FastAPI, :8000)
  -> /predict (Phase 14 predictor + decision + context)
  -> /analyze (Phase 6A orchestrator + calibration + reasoning + optional sentiment/RAG/LLM)
  -> /reason (deterministic reasoning-only endpoint)
Model/Artifacts
  -> backend/python/tradesense/models/xgboost.joblib
RAG Store
  -> backend/python/tradesense/rag_store/<SYMBOL>/
```

## Implemented Components Inventory

- Python API: `backend/python/tradesense/api.py`, `backend/python/tradesense/api_predict.py`
- Inference predictor: `backend/python/tradesense/inference/predict.py`
- Decision engine: `backend/python/tradesense/inference/decision_engine.py`
- Context engine: `backend/python/tradesense/inference/context_engine.py`
- Full orchestrator path: `backend/python/tradesense/inference/orchestrator.py`
- Calibration utilities: `backend/python/tradesense/calibration.py`
- Explainability/rules: `backend/python/tradesense/explainability/*`
- Backtesting: `backend/python/tradesense/backtesting/*`
- Node gateway: `backend/node/server/*`
- Frontend: `frontend/src/*`

Repository scope presence check requested by audit:

| Scope item | Status | Notes |
| --- | --- | --- |
| `backend/python` | Present | Core implementation lives here |
| `backend/node` | Present | Express gateway |
| `frontend` | Present | React/Vite client |
| `models` (repo root) | Missing | Model artifacts are under `backend/python/tradesense/models` |
| `rag_store` (repo root) | Missing | RAG store is under `backend/python/tradesense/rag_store` |
| `docs` | Present | Existing docs and prior audit |
| `tests` (repo root) | Missing | Tests exist under `backend/python/tests` and `backend/node/server/tests` |

## Phase Completion Status

| Phase | Status | Evidence |
| --- | --- | --- |
| Phase 1 Data/Indicators | Complete | `test_phase1.py` |
| Phase 2 Features | Complete | `test_phase2.py` |
| Phase 3 Modeling | Complete | `test_phase3.py` |
| Phase 4A Reasoning core | Complete | `reasoning_core.py`, `test_phase4a.py` |
| Phase 4B FastAPI /reason | Complete | `api.py`, `test_phase4b.py` |
| Phase 6A Inference orchestrator | Complete | `inference/orchestrator.py` |
| Phase 6B /analyze endpoint | Complete | `api.py`, `test_phase6b.py` |
| Phase 7A Sentiment | Complete (optional runtime deps) | `sentiment/*`, `test_phase7a.py` |
| Phase 7B News ingestion | Complete (API-key dependent) | `news/*`, `test_phase7b.py` |
| Phase 8A RAG context | Complete | `rag/*`, `test_phase8a.py` |
| Phase 8B LLM explanation | Complete (optional) | `explainer/*`, `test_phase8b.py` |
| Phase 9 Calibration discipline | Complete | `calibration.py`, `test_phase9.py` |
| Phase 10 Explainability package | Complete | `explainability/*`, `test_phase10.py` |
| Phase 11 Backtesting | Complete | `backtesting/*`, `test_phase11.py` |
| Phase 12 Training pipeline | Complete | `models/train_phase12.py`, `test_phase12_training.py` |
| Phase 13/14 predictor/decision/context + /predict | Complete | `inference/predict.py`, `decision_engine.py`, `context_engine.py`, `api_predict.py` |

## Artifact Validation Results

### Model Bundle Integrity

Artifact inspected: `backend/python/tradesense/models/xgboost.joblib`

| Validation requirement | Result | Evidence |
| --- | --- | --- |
| Contains `model` | PASS | Runtime inspection via joblib |
| Contains `feature_names` | PASS | Runtime inspection via joblib |
| Contains `calibrator` | PASS | Runtime inspection via joblib |
| Contains `calibration_meta` | PASS | Runtime inspection via joblib |
| Calibrator supports `predict_proba` | PASS | `hasattr(calibrator, "predict_proba") == True` |
| `calibration_meta.method` exists | PASS | `platt` |
| `calibration_meta.created_at` exists | PASS | ISO timestamp present |
| `calibration_meta.version` exists | PASS | `phase12` |

## API Validation Results

### Endpoint Integrity

FastAPI routes discovered:
- `POST /predict`
- `POST /analyze`
- `POST /reason`

Result: PASS

### `/predict` Full Pipeline Validation

Required flow: data fetch -> feature build -> predictor -> decision engine -> context engine -> structured response.

Result: PASS

Validation evidence:
- Static code path in `api_predict.py` follows exact sequence.
- Runtime monkeypatch check captured call order: `data_fetch_feature_build -> predictor -> decision -> context`.
- Live request returned full schema with required fields.

### Response Schema Integrity

`/predict` response model requires:
- `symbol`, `prediction`, `probability`, `confidence`, `decision`, `confidence_level`, `strength`, `context`, `model_version`, `timestamp`, `generated_at`

Result: PASS (validated via `PredictResponse`).

### Error Handling Discipline

- Client responses for internal failures are generic (`"Prediction failure"`, `"Internal server error"`): PASS for non-leakage.
- Internal tracebacks are printed to server logs in `/analyze`: PARTIAL (no client leak, but noisy operational behavior).

## Frontend Validation Results

### Required checks

| Requirement | Result | Evidence |
| --- | --- | --- |
| `frontend/src/api/analyze.js` uses live API call path | PASS | Calls `POST /api/analyze` |
| `Dashboard.jsx` uses real response values (no hardcoded prediction values) | PASS | Renders from `predictionData.*` and `predictionData.context.*` |
| decision/probability/confidence/strength/context originate from backend response | PASS | All displayed fields come from response object |

### Integration reality check

- Frontend sends `{ symbol }`.
- Node gateway now correctly accepts symbol-only `PredictRequest`, validates/normalizes input, and forwards to FastAPI `POST /predict`.
- Contract mismatch resolved; frontend integration is fully functional.
- End-to-end pipeline (`/api/analyze` -> Node gateway -> `/predict`) verified operational.

Status: PASS.

## Backtesting Validation Results

| Requirement | Result | Evidence |
| --- | --- | --- |
| `backend/python/tradesense/backtesting/` present | PASS | Directory and modules exist |
| Backtest engine uses existing inference + calibration logic | PASS | `run_backtest` calls `analyze_symbol` (orchestrator applies calibration) |
| Includes accuracy metric | PASS | `overall_accuracy` |
| Includes ECE metric | PASS | `expected_calibration_error` |
| Includes Brier score | PASS | `brier_score` |
| Includes probability bucket reliability | PASS | `accuracy_by_probability_bucket` + reliability buckets |

## Calibration Validation Results

- Calibration is enforced in `/analyze` via `calibrate_probability` and persisted calibrator artifact.
- Calibration is enforced in `/predict` via legacy adapter wrapping `model + calibrator` into calibrated `predict_proba` output.
- Missing/invalid calibration artifacts trigger explicit errors in load/validation paths.

Result: PASS

## Determinism and Safety Assessment

| Check | Result | Notes |
| --- | --- | --- |
| Inference path deterministic for fixed numerical inputs | PARTIAL | Core decision math deterministic; timestamps vary per call |
| No randomness in predictor/decision/context | PASS | No random API usage in these modules |
| Model bundle loaded once and reused | PASS | Global singletons in `api_predict.py`; module-level load in orchestrator |
| Calibration always applied | PASS | Both `/predict` and `/analyze` apply calibration path |
| No silent fallback behavior | PARTIAL | Core inference fails fast; controlled fallbacks exist in other layers (e.g., attribution fallback, RAG FAISS->NumPy fallback) |

## Documentation Alignment Assessment

Current top-level docs are not fully aligned with implemented state.

Key misalignments:
- Existing `README.md` described old frontend payload contract and outdated gateway forwarding behavior.
- Existing `docs/FULL_SYSTEM_AUDIT.md` reports stale findings (e.g., earlier bundle mismatch) that no longer match current artifact.
- `README.md` updated in v1.0 stabilization to document symbol-only gateway forwarding to `/predict`.

Assessment: RESOLVED FOR GATEWAY CONTRACT ALIGNMENT.

## Risk and Weakness Assessment

1. Previous frontend/gateway request contract mismatch was resolved in v1.0 stabilization (symbol-only request forwarding to `/predict`).
2. `/analyze` path logs traceback details server-side for internal exceptions.
3. Non-deterministic timestamp fields (`timestamp`, `generated_at`) prevent byte-for-byte deterministic outputs.
4. Optional subsystems (FinBERT, Finnhub, LLM explain) require external dependencies/keys and are not guaranteed in all environments.

## Production Readiness Assessment

Current readiness: **Moderate**

Ready:
- Python inference/calibration APIs operational.
- Model artifact contract is valid.
- Backtesting and calibration metrics present.
- Python test suite passes (`44 passed`).

Not ready without fixes:
- Integration checks should be rerun whenever gateway or API contracts change.

## Final System Maturity Rating

**Maturity rating: 8.2 / 10 (Functionally strong core with gateway/frontend contract alignment fixed).**

Rationale:
- Core ML inference, calibration discipline, decision/context generation, and backtesting are implemented and verifiable.
- Remaining maturity drag is operational hardening and optional dependency reliability, not core architecture.

## Verification Commands Executed

- Python tests: `backend/python/.venv/Scripts/python.exe -m pytest -q`
- Node tests: `npm test -- --runInBand` (in `backend/node`)
- Frontend build: `npm run build` (in `frontend`)
- Bundle inspection: joblib load/contract checks on `xgboost.joblib`
- FastAPI route introspection for `/predict`, `/analyze`, `/reason`
- Runtime endpoint checks for `/predict`, `/analyze`, `/reason`
- Gateway smoke check: live `POST /api/analyze` with `{"symbol":"AAPL"}` returned HTTP 200 and full prediction payload via Node forwarding to `/predict`.
