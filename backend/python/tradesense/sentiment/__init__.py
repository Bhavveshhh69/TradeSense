"""Sentiment analysis utilities for TradeSense (Phase 7A)."""

from tradesense.sentiment.aggregator import aggregate_sentiment


def analyze_texts(texts):
    """Lazy import wrapper to avoid loading FinBERT when unused."""
    from tradesense.sentiment.finbert import analyze_texts as _analyze_texts

    return _analyze_texts(texts)


__all__ = ["aggregate_sentiment", "analyze_texts"]
