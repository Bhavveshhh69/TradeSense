"""News ingestion utilities for TradeSense (Phase 7B)."""

from tradesense.news.fetcher import fetch_news
from tradesense.news.normalizer import normalize_news

__all__ = ["fetch_news", "normalize_news"]
