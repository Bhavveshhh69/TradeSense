"""Normalize news text before sentiment analysis."""

from __future__ import annotations

from typing import List

MAX_NEWS_CHARS = 1000  # Maximum characters per news item after normalization.


def normalize_news(texts: List[str]) -> List[str]:
    """Normalize raw news text for FinBERT.

    Rules:
    - Strip leading/trailing whitespace
    - Remove duplicates
    - Drop empty strings
    - Enforce max length per item (MAX_NEWS_CHARS)
    """
    if texts is None:
        return []
    if not isinstance(texts, list):
        raise ValueError("texts must be a list of strings")

    normalized: List[str] = []
    seen = set()

    for text in texts:
        if not isinstance(text, str):
            raise ValueError("texts must be a list of strings")
        cleaned = text.strip()
        if not cleaned:
            continue
        if len(cleaned) > MAX_NEWS_CHARS:
            cleaned = cleaned[:MAX_NEWS_CHARS]
        if cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)

    return normalized
