"""Finnhub news fetcher for TradeSense (Phase 7B)."""

from __future__ import annotations

from datetime import date, timedelta
import os
from typing import List

import httpx

_FINNHUB_ENDPOINT = "https://finnhub.io/api/v1/company-news"
_LOOKBACK_DAYS = 7
_TIMEOUT_SECONDS = 10.0


def _date_range() -> tuple[str, str]:
    end_date = date.today()
    start_date = end_date - timedelta(days=_LOOKBACK_DAYS)
    return start_date.isoformat(), end_date.isoformat()


def fetch_news(symbol: str, limit: int = 10) -> List[str]:
    """Fetch recent company news from Finnhub.

    Returns a list of raw text items (headline + summary/body). Network or API
    failures return an empty list; the API key is never logged.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        return []

    if limit is None:
        limit = 10
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10
    if limit <= 0:
        return []

    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        return []

    start_date, end_date = _date_range()
    params = {
        "symbol": symbol.upper().strip(),
        "from": start_date,
        "to": end_date,
        "token": api_key,
    }

    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            response = client.get(_FINNHUB_ENDPOINT, params=params)
            if response.status_code != 200:
                return []
            payload = response.json()
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    texts: List[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        headline = str(item.get("headline", "") or "").strip()
        summary = str(item.get("summary", "") or item.get("description", "") or "").strip()

        if headline and summary:
            text = f"{headline}. {summary}"
        elif headline:
            text = headline
        elif summary:
            text = summary
        else:
            continue

        texts.append(text)
        if len(texts) >= limit:
            break

    return texts
