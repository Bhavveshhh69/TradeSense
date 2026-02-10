# Phase 5B: Dynamic Symbol Input (Frontend)

## Purpose
- Provide a minimal React UI for triggering deterministic market analysis.
- Allow users to input the symbol dynamically from the UI.
- Validate end-to-end wiring from React to Node to Python.

## What the React Frontend Does
- Renders the TradeSense dashboard with a symbol input and "Run Analysis" action.
- Normalizes the symbol to uppercase and disables analysis when the input is empty.
- Sends the user-provided payload to `/api/analyze` using `axios`.
- Displays the API response JSON verbatim and highlights the analyzed symbol.
- Surfaces network or backend errors in the UI.

## API Flow

```
React (Vite) -> POST /api/analyze -> Node (Express) -> POST /reason -> Python (FastAPI)
```

Note: The Python service also exposes `POST /analyze` for direct inference, optional sentiment/news, and RAG context. The Phase 5B frontend continues to call the Node `/api/analyze` gateway.

## Request Example (React -> Node)

Replace `SYMBOL` with the stock symbol you want to analyze.

```json
{
  "symbol": "SYMBOL",
  "payload": {
    "symbol": "SYMBOL",
    "probability": 0.62,
    "feature_importance": {
      "price_vs_ema20": 0.25,
      "rsi_slope_3": 0.2,
      "macd_hist_accel": 0.15,
      "volume_ratio": 0.1,
      "price_vs_ema50": 0.05
    },
    "feature_values": {
      "price_vs_ema20": 0.02,
      "rsi_slope_3": -0.1,
      "macd_hist_accel": 0.0,
      "volume_ratio": 1.2,
      "price_vs_ema50": -0.01
    },
    "trend_state": 1,
    "momentum_state": 1,
    "risk_state": 1
  }
}
```

## Response Example (Python -> Node -> React)

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

## Environment Assumptions
- Localhost only.
- Python FastAPI runs on http://127.0.0.1:8000.
- Node.js Express runs on http://127.0.0.1:3000.
- Vite dev server runs on http://127.0.0.1:5173 and proxies `/api` to the Node server.

## What Phase 5B Does Not Include
- Authentication or authorization.
- Frontend-managed persistence (backend-only local RAG storage exists).
- Trading execution or brokerage integrations.
- Direct frontend calls to the Python service.
- Mock or hardcoded response data in the UI.
