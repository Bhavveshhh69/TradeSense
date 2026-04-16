"""Finnhub news fetcher for TradeSense."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import os
from typing import List

import httpx

_FINNHUB_ENDPOINT = "https://finnhub.io/api/v1/company-news"
_LOOKBACK_DAYS = 7
_TIMEOUT_SECONDS = 10.0


def _to_iso_date(value: date | datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _date_range(
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
) -> tuple[str, str]:
    end_text = _to_iso_date(end_date) or date.today().isoformat()
    start_text = _to_iso_date(start_date)
    if start_text is None:
        parsed_end = date.fromisoformat(end_text)
        start_text = (parsed_end - timedelta(days=_LOOKBACK_DAYS)).isoformat()
    return start_text, end_text


def fetch_news(
    symbol: str,
    limit: int = 10,
    *,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
) -> List[str]:
    """Fetch company news from Finnhub.

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

    start_date_text, end_date_text = _date_range(start_date, end_date)
    params = {
        "symbol": symbol.upper().strip(),
        "from": start_date_text,
        "to": end_date_text,
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
