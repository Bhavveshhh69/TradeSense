"""Validation script for POST /predict."""

from __future__ import annotations

import json
import os
import sys

import httpx


API_URL = os.getenv("TRADESENSE_PREDICT_URL", "http://127.0.0.1:8000/predict")
REQUIRED_FIELDS = (
    "symbol",
    "prediction",
    "probability",
    "confidence",
    "decision",
    "confidence_level",
    "strength",
    "context",
    "model_version",
    "timestamp",
    "generated_at",
)


def _validate_response(payload: dict) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise RuntimeError(f"Missing required fields: {missing}")

    for field in ("probability", "confidence", "strength"):
        value = payload[field]
        if not isinstance(value, (int, float)):
            raise RuntimeError(f"{field} must be numeric")
        if not 0.0 <= float(value) <= 1.0:
            raise RuntimeError(f"{field} must be within [0, 1]")


def main() -> None:
    symbol = sys.argv[1].strip().upper() if len(sys.argv) > 1 else "AAPL"
    if not symbol:
        raise RuntimeError("symbol must be a non-empty string")

    response = httpx.post(API_URL, json={"symbol": symbol}, timeout=30.0)
    if response.status_code >= 400:
        raise RuntimeError(
            f"/predict request failed: status={response.status_code} body={response.text}"
        )

    payload = response.json()
    print(json.dumps(payload, indent=2))
    _validate_response(payload)


if __name__ == "__main__":
    main()
