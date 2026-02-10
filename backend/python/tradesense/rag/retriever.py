# tradesense/rag/retriever.py
"""Context retrieval for stored RAG insights."""

from __future__ import annotations

from typing import List

from tradesense.rag.store import _get_store


def retrieve_context(symbol: str, limit: int = 5) -> List[dict]:
    """Retrieve past insights for a symbol ordered by relevance and recency."""
    return _get_store().retrieve_context(symbol, limit=limit)
