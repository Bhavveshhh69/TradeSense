# TradeSense Python Backend

This folder contains the FastAPI service and Python analytics stack.

## Run

```powershell
python -m pip install -r requirements.txt
python -m uvicorn tradesense.api:app --host 127.0.0.1 --port 8000
```

Optional (lower latency hot loop + HTTP parser)

```powershell
python -m pip install uvloop httptools
python -m uvicorn tradesense.api:app --host 127.0.0.1 --port 8000 --loop uvloop --http httptools
```

## Endpoints
- `POST /reason`: deterministic reasoning core (Phase 4B).
- `POST /analyze`: full inference pipeline with optional sentiment/news and RAG context.

## Optional Environment Variables
- `FINNHUB_API_KEY`: required when calling `/analyze` with `use_news=true`.
- `TRADESENSE_RAG_DIR`: override the local RAG store directory (default: `backend/python/tradesense/rag_store`).

## Tests

```powershell
python -m pytest
```
