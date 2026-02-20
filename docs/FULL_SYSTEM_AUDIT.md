# TradeSense Full System Audit

Audit date: 2026-02-20  
Repository root: `e:\Bhavesh Files\TradeSense`  
Audit mode: code and contract audit only (no behavior changes)

## Scope And Method

This audit reviewed:

- `backend/python`
- `backend/node`
- `frontend`
- `docs`
- `models`
- `rag_store`
- calibration layer
- explainability layer
- inference orchestrator
- reasoning core
- API contracts

Validation performed:

- Static code inspection across all modules listed above.
- Python tests: `python -m pytest -q` (42 passed).
- Node tests: `npm test -- --runInBand` (3 passed).
- Runtime sanity call to FastAPI `/analyze` with live code path.
- Model artifact introspection (`xgboost.joblib` keys and structure).

## 1. System Architecture Overview

## Target Pipeline (Requested)

`Frontend -> Node -> FastAPI -> inference -> calibration -> explainability -> response`

## Actual Implemented Runtime Paths

### Path A: Current frontend product path

`Frontend (Dashboard) -> Node POST /api/analyze -> Python POST /reason -> reasoning_core.generate_insight -> response`

Evidence:

- Frontend posts to `/api/analyze` in `frontend/src/api/analyze.js:8`.
- Frontend sends a static `payload` block in `frontend/src/pages/Dashboard.jsx:35`.
- Node route forwards `req.body.payload` to Python `/reason` in `backend/node/server/routes/analyze.js:26` and `backend/node/server/services/reasoning.js:3`.
- Python `/reason` calls deterministic reasoning core in `backend/python/tradesense/api.py:50`.

### Path B: Direct Python full pipeline path

`Client -> Python POST /analyze -> inference.orchestrator.analyze_symbol -> calibration -> deterministic explainability -> optional sentiment/news -> optional RAG context -> optional LLM explanation -> response`

Evidence:

- `/analyze` entry in `backend/python/tradesense/api.py:59`.
- Inference + calibration in `backend/python/tradesense/inference/orchestrator.py:185` and `backend/python/tradesense/inference/orchestrator.py:188`.
- Deterministic explainability in `backend/python/tradesense/reasoning_core.py:87`.
- Optional sentiment/news in `backend/python/tradesense/api.py:105` and `backend/python/tradesense/api.py:123`.
- Optional RAG context in `backend/python/tradesense/api.py:142`.
- Optional LLM explanation in `backend/python/tradesense/api.py:181`.

### Critical architecture reality

The current frontend and Node gateway do not traverse the inference/calibration path. They use `/reason`, not `/analyze`. So the requested end-to-end pipeline is implemented in Python, but not exposed through the active frontend route.

## 2. Phase Completion Analysis

| Phase | Status | Assessment |
| --- | --- | --- |
| Phase 1 (data + indicators) | fully complete | Implemented and tested (`tests/test_phase1.py`). |
| Phase 2 (feature engineering) | fully complete | Implemented and tested (`tests/test_phase2.py`). |
| Phase 3 (modeling) | fully complete | Implemented and tested (`tests/test_phase3.py`). |
| Phase 4A (reasoning core) | fully complete | Deterministic rule engine implemented and tested. |
| Phase 4B (FastAPI /reason) | fully complete | Implemented with schema contract and tests. |
| Phase 4C (Node gateway) | fully complete | Validation, forwarding, caching implemented and tested. |
| Phase 5A/5B (frontend wiring + symbol input) | fully complete | Implemented; dynamic symbol exists. |
| Phase 6A (inference orchestrator) | partially complete | Code exists, but bundled model artifact currently breaks calibration-required startup for `/analyze`. |
| Phase 6B (FastAPI /analyze) | partially complete | Endpoint exists, but current repo artifact causes runtime 500 on live path. |
| Phase 7A (FinBERT sentiment) | partially complete | Code and mocked tests exist; runtime depends on heavy external deps and model download. |
| Phase 7B (Finnhub news ingestion) | partially complete | Code exists and mocked tests exist; production behavior depends on API key/network. |
| Phase 8A (RAG memory/context) | fully complete | File-based store/retrieve/format implemented and tested. |
| Phase 8B (LLM explanation) | undocumented but implemented | `explain` request path and explainer modules exist with tests, but not reflected in primary architecture docs. |
| Phase 9 (calibration discipline) | partially complete | Calibration logic and tests exist, but checked-in model bundle lacks required calibration artifacts. |
| Phase 10 (deterministic explainability package) | undocumented but implemented | Explainability package and tests exist but top-level docs still frame project as through Phase 8A. |

## 3. Contract Validation

## Feature Contract Enforcement

Result: partially enforced.

- Stronger enforcement on `/analyze` path:
  - Exact model feature set required (`missing_features` check) in `backend/python/tradesense/inference/orchestrator.py:169`.
  - NaN rejection in inference row in `backend/python/tradesense/inference/orchestrator.py:176`.
- Weaker enforcement on `/reason` path:
  - Requires non-empty `feature_importance` and `feature_values`, numeric values, and key overlap (`feature_values missing keys`) in `backend/python/tradesense/reasoning_core.py:151`.
  - Does not require canonical model feature names.

## Calibration Contract Enforcement

Result: contract defined and enforced in code, but artifact in repo is currently non-compliant.

- Orchestrator hard-requires `calibrator` and `calibration_meta` in model bundle in `backend/python/tradesense/inference/orchestrator.py:58`.
- Current artifact (`backend/python/tradesense/models/xgboost.joblib`) contains only `model` and `feature_names` (observed via runtime inspection).
- Live `/analyze` call currently returns 500 due missing calibration artifacts.

## Probability Contract

Result: structurally enforced, operationally blocked on `/analyze`.

- Response schema requires `probability`, `probability_raw`, `probability_calibrated` in `backend/python/tradesense/schemas.py:73`.
- Reasoning core treats `probability` as calibrated output and validates bounds for raw/calibrated fields in `backend/python/tradesense/reasoning_core.py:48`.
- On `/reason`, raw and calibrated default to same value if not explicitly provided.
- On `/analyze`, intended behavior is raw then calibrated, but currently blocked by missing calibration artifacts.

## Explainability Contract

Result: largely enforced.

- Deterministic structured explanation contract (`key_drivers`, `negative_factors`, `confidence_modifiers`) built in `backend/python/tradesense/explainability/rules.py:11`.
- Top-level response includes `structured_explanation` and `key_drivers` in `backend/python/tradesense/reasoning_core.py:111`.
- Optional LLM explanation gated by `explain` flag in `backend/python/tradesense/api.py:181` and schema field in `backend/python/tradesense/schemas.py:117`.

## API Response Contract

Result: strong schema contracts, mixed error-surface quality.

- `/reason` and `/analyze` both use strict response models (`ReasonResponse`, `AnalyzeResponse`) in `backend/python/tradesense/api.py:50` and `backend/python/tradesense/api.py:59`.
- Node returns Python response unchanged on success (`res.json(data)`) in `backend/node/server/routes/analyze.js:30`.
- Analyze request validation is strict at schema level, but the API maps all `AnalyzeRequest` validation failures to a generic `400 symbol must be a non-empty string` in `backend/python/tradesense/api.py:64`, masking non-symbol contract violations.

## 4. Storage And Artifact Analysis

| Storage Type | Location | Implementation | Audit Result |
| --- | --- | --- | --- |
| Model storage | `backend/python/tradesense/models/xgboost.joblib` | Joblib bundle loaded at import | Bundle exists, but missing calibration keys required by runtime. |
| Calibration storage | Same joblib bundle (`calibrator`, `calibration_meta`) | Required by orchestrator | Incorrect in current artifact; calibration contract fails at runtime. |
| RAG storage | `backend/python/tradesense/rag_store/<SYMBOL>/` | `metadata.json`, `vectors.npy`, optional `index.faiss` | Correct format and retrieval behavior; mutable runtime artifacts are tracked in git. |
| Cache storage (Node) | In-memory `Map` only | SHA256 hash key + TTL | Correct for process-local cache; non-persistent, cleared on restart. |
| FinBERT storage | Not vendored in repo; downloaded by `transformers` at runtime | `AutoTokenizer.from_pretrained`, `AutoModel...from_pretrained` | External runtime dependency; no pinned local artifact in repository. |

Additional observations:

- RAG store path is configurable via `TRADESENSE_RAG_DIR` in `backend/python/tradesense/rag/store.py:199`.
- Current environment check during audit showed missing `transformers`, `torch`, and `faiss` modules; this does not invalidate repository code, but it confirms optional runtime dependencies are not guaranteed by default environment state.

## 5. Inference Integrity Analysis

## Deterministic

Status: partially deterministic.

- Reasoning and explainability layers are deterministic for identical inputs.
- End-to-end inference depends on live yfinance data and current date window (`datetime.now()`), so outputs vary over time.
- Within a fixed input row/model, inference logic is deterministic.

## Calibrated

Status: not currently operational in this repository snapshot.

- Calibration step is mandatory in code.
- Bundled model artifact is missing calibration objects.
- Verified runtime result: `/analyze` currently returns `500 Internal server error` on live path.

## Explainable

Status: yes (deterministic explainability), plus optional LLM narrative.

- Deterministic attribution and rule rendering are implemented.
- Optional LLM explanation path exists and is off by default.

## Fail-fast Safe

Status: mixed.

- Strong fail-fast validation in orchestrator (missing model/features/NaNs/calibration).
- API converts many internal failures to generic 500 and prints traceback (`traceback.print_exc`), which is safe for not returning internals to client but weak for actionable client diagnostics and noisy in logs.

## 6. Explainability Integrity Analysis

## Deterministic Attribution Exists

Yes.

- Model-based attributions via XGBoost contributions in `backend/python/tradesense/explainability/attribution.py:39`.
- Deterministic fallback attribution from feature importance and values in `backend/python/tradesense/explainability/attribution.py:12`.
- Normalization and bounded outputs enforced.

## Explanations Are Grounded

Mostly yes.

- Rule-based explanations are derived from provided feature attributions/values, confidence reason, and market states.
- No hidden signal generation in deterministic path.
- LLM path prompt explicitly restricts new predictions and advice in `backend/python/tradesense/explainer/prompt_builder.py:28`.

## LLM Explanations Optional

Yes.

- Controlled by `explain: false` default in request schema (`backend/python/tradesense/schemas.py:27`).
- No LLM call when `explain` is false (covered by `tests/test_phase8b.py:62`).

## 7. Risk And Weakness Analysis

## High Risk

1. Calibration artifact mismatch breaks live `/analyze` inference.
   - Code requires calibration artifacts, but repository model bundle lacks them.
   - Impact: full inference endpoint currently unusable in default checked-in state.

2. Active product path bypasses inference/calibration pipeline.
   - Frontend -> Node route calls Python `/reason` with static payload, not `/analyze`.
   - Impact: users do not traverse the full inference/calibration/explainability pipeline through main UI/API gateway.

3. Integration tests do not exercise real `/analyze` inference path.
   - Phase 6B/7A/7B/8A/8B tests monkeypatch `_get_analyze_symbol` and related handlers.
   - Impact: critical runtime failures (like missing calibration artifacts) are not caught by CI tests.

## Medium Risk

1. Documentation drift across phases and capabilities.
   - Docs still describe project as through Phase 8A and often state no LLM layer, while code includes Phase 8B and Phase 10 components.

2. Analyze validation error messages are overly generic.
   - Different schema violations are collapsed into `symbol must be a non-empty string`.

3. Runtime artifacts in tracked repo paths.
   - `rag_store` files are tracked and mutate during tests/runtime, creating noisy diffs and non-hermetic repository state.

4. Training script uses random split for calibration (`train_test_split`), not explicit time-ordered validation.
   - This can leak temporal structure in financial time series.

5. Pydantic v1-style config/validators are deprecated under Pydantic v2.
   - Current test run emits deprecation warnings.

## Low Risk

1. Debug `print` calls in inference orchestrator (`backend/python/tradesense/inference/orchestrator.py:166`).
2. Mixed casing/style in stored RAG records from different call sites can reduce summary consistency.

## 8. Documentation Alignment (Code vs Docs)

| Area | Documentation Says | Code Reality | Alignment |
| --- | --- | --- | --- |
| Phase coverage | Through Phase 8A (`PROJECT_STATUS.md:1`, `README.md:199`) | Code and tests include Phase 8B, 9, 10 (`tests/test_phase8b.py`, `tests/test_phase9.py`, `tests/test_phase10.py`) | mismatched |
| LLM usage | Non-goals include no LLM generation/chat (`README.md:16`, `ARCHITECTURE.md:70`) | Optional LLM explanation layer exists (`backend/python/tradesense/explainer/*`, `api.py` `explain` path) | mismatched |
| End-to-end path expectation | Docs imply full inference available in backend | Frontend+Node default path hits `/reason`, bypassing `/analyze` inference | partially aligned but operationally divergent |
| Calibration failure behavior | Phase 9 doc says explicit missing-artifact failure guidance (`docs/phase-9-calibration.md:30`) | API currently returns generic 500 for missing calibration artifact during import path | mismatched |
| Explain API surface | Not prominently documented in top-level docs | `AnalyzeRequest.explain` and `AnalyzeResponse.explanation` are implemented | undocumented but implemented |

## Documentation Fixes Recommended

1. Add explicit architecture section separating:
   - current frontend path (`/api/analyze -> /reason`)
   - direct full inference path (`/analyze`).
2. Update phase tracking docs to include 8B/9/10 status and readiness.
3. Document `explain` request flag and `explanation` response contract.
4. Document calibration artifact requirement with concrete operational check (`joblib` keys expected).
5. Document RAG store mutability and recommended git ignore policy for runtime artifacts.

## 9. Safe Fix Recommendations (Do Not Auto-Apply)

1. Regenerate and verify model bundle calibration artifacts.
   - Why: `/analyze` currently fails due missing `calibrator` and `calibration_meta`.
   - Safe action: run `python tradesense/models/train_and_persist.py`, then verify bundle keys include `model`, `feature_names`, `calibrator`, `calibration_meta`.

2. Add a non-mocked integration test for live `/analyze` path.
   - Why: current tests miss critical runtime failures.
   - Safe action: include one deterministic fixture/model bundle test that imports real orchestrator.

3. Decide product path intentionally: `/reason` gateway or `/analyze` gateway.
   - Why: current frontend route bypasses requested full pipeline.
   - Safe action: either keep current path and document clearly, or add separate Node route for `/analyze` passthrough.

4. Improve calibration-related error surfacing in `/analyze`.
   - Why: current generic 500 obscures actionable fix guidance.
   - Safe action: map known calibration artifact errors to a clear 422/500 detail consistent with phase docs.

5. Align docs with implemented LLM and Phase 10 explainability components.
   - Why: current docs under-report implemented behavior.

6. Stop tracking mutable RAG runtime artifacts in git.
   - Why: repeated runtime/test writes create noisy diffs and state coupling.
   - Safe action: move seeded fixtures to test assets and ignore runtime store paths.

7. Migrate Pydantic v1-style validators/config to v2 equivalents.
   - Why: deprecation warnings indicate future break risk.

## 10. Final System Maturity Assessment

## Engineering Quality

Rating: medium.

Strengths:

- Clear modular decomposition (data, features, modeling, reasoning, inference, explainability, API layers).
- Strong deterministic reasoning and explainability primitives.
- Good unit test coverage breadth across phases.

Weaknesses:

- Critical artifact/runtime mismatch for calibration.
- Heavy reliance on monkeypatched tests for key integration phases.
- Documentation lag behind implemented system.

## Architectural Correctness

Rating: medium-low.

- Internal Python architecture for full pipeline is coherent.
- Product-facing path currently does not use that full pipeline.
- Calibration contract is architecturally sound but currently violated by checked-in artifact state.

## Production Readiness

Rating: low (current snapshot).

Blocking reasons:

1. `/analyze` is not operational with current model artifact.
2. Main frontend route bypasses full inference/calibration pipeline.
3. Integration test gap allows critical regressions to pass CI.

Once the calibration artifact and integration-path/test gaps are fixed, readiness could move to medium.

