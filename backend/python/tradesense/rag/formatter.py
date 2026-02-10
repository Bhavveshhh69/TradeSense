# tradesense/rag/formatter.py
"""Formatter for RAG context summaries.

Output is capped to MAX_CONTEXT_CHARS to keep responses compact.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import List


MAX_CONTEXT_CHARS = 900  # Fixed budget to keep response payloads small.


def format_context(context: List[dict]) -> str:
    """Format retrieved context into a compact narrative block."""
    if not context:
        return ""

    items = sorted(context, key=_timestamp_key, reverse=True)
    latest = items[0]
    earliest = items[-1]

    parts: List[str] = [f"Recent history ({len(items)} items)."]

    latest_prob = _format_prob(latest.get("probability"))
    earliest_prob = _format_prob(earliest.get("probability"))
    latest_conf = _safe_str(latest.get("confidence_level"))
    earliest_conf = _safe_str(earliest.get("confidence_level"))

    if len(items) > 1:
        parts.append(
            "What changed: probability "
            f"{earliest_prob} ({earliest_conf}) -> {latest_prob} ({latest_conf})."
        )
    else:
        parts.append(
            f"What changed: latest probability {latest_prob} ({latest_conf})."
        )

    drivers = _repeated_items(items, "key_drivers")
    if drivers:
        parts.append(f"Repeated drivers: {', '.join(drivers)}.")
    else:
        parts.append("Repeated drivers: none detected.")

    sentiment_line = _sentiment_shift(items)
    if sentiment_line:
        parts.append(sentiment_line)

    summary = " ".join(parts)
    return _truncate(summary, MAX_CONTEXT_CHARS)


def _repeated_items(items: List[dict], field: str, limit: int = 3) -> List[str]:
    counter: Counter[str] = Counter()
    for item in items:
        values = item.get(field) or []
        if not isinstance(values, list):
            continue
        for entry in values:
            if not isinstance(entry, str):
                entry = str(entry)
            entry = entry.strip()
            if entry:
                counter[entry] += 1
    repeated = [entry for entry, count in counter.items() if count > 1]
    repeated.sort(key=lambda entry: (-counter[entry], entry))
    return repeated[:limit]


def _sentiment_shift(items: List[dict]) -> str:
    sentiments = []
    for item in items:
        sentiment = item.get("sentiment")
        if not isinstance(sentiment, dict):
            continue
        bias = sentiment.get("sentiment_bias")
        score = sentiment.get("sentiment_score")
        if bias is None and score is None:
            continue
        sentiments.append((bias, score))

    if not sentiments:
        return "Sentiment shift: no sentiment history available."

    earliest_bias, earliest_score = sentiments[-1]
    latest_bias, latest_score = sentiments[0]

    earliest_label = _sentiment_label(earliest_bias, earliest_score)
    latest_label = _sentiment_label(latest_bias, latest_score)

    if earliest_label == latest_label:
        return f"Sentiment shift: remained {latest_label}."

    return f"Sentiment shift: {earliest_label} -> {latest_label}."


def _sentiment_label(bias, score) -> str:
    if bias:
        return str(bias)
    if score is None:
        return "neutral"
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "neutral"
    if score > 0.1:
        return "bullish"
    if score < -0.1:
        return "bearish"
    return "neutral"


def _format_prob(value) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _safe_str(value) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def _timestamp_key(record: dict) -> float:
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        return 0.0
    cleaned = timestamp.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."
