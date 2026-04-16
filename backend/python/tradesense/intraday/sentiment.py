from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from .contracts import MarketProfile, SectorResolution, SentimentSnapshot
from .sector import SectorResolver


SentimentFetcher = Callable[..., list[str]]


def _previous_regular_close(decision_time: datetime, profile: MarketProfile) -> datetime:
    prior_session = decision_time.date() - timedelta(days=1)
    return datetime.combine(prior_session, profile.regular_close, tzinfo=decision_time.tzinfo)


def _neutral_snapshot(resolution: SectorResolution, reason: str) -> SentimentSnapshot:
    return SentimentSnapshot(
        stock_sentiment_score=0.0,
        sector_sentiment_score=None if not resolution.sector_available else 0.0,
        contextual_sentiment_score=0.0,
        sentiment_confidence=0.0,
        sentiment_gate_reason=reason,
        stock_article_count=0,
        sector_article_count=0,
        sector=resolution.sector,
        sector_available=resolution.sector_available,
        stock_articles=(),
        sector_articles=(),
    )


@dataclass
class IntradaySentimentEngine:
    sector_resolver: SectorResolver | None = None
    news_fetcher: SentimentFetcher | None = None

    def __post_init__(self) -> None:
        self.sector_resolver = self.sector_resolver or SectorResolver()
        if self.news_fetcher is None:
            from tradesense.news.fetcher import fetch_news

            self.news_fetcher = fetch_news

    def snapshot(
        self,
        symbol: str,
        profile: MarketProfile,
        decision_time: datetime,
        manual_news: list[str] | None = None,
    ) -> SentimentSnapshot:
        resolution = self.sector_resolver.resolve(symbol)
        stock_articles = tuple(text for text in (manual_news or []) if isinstance(text, str) and text.strip())
        if not stock_articles:
            stock_articles = tuple(self._fetch_articles(symbol, profile, decision_time))

        sector_articles: tuple[str, ...] = ()
        if resolution.sector_available:
            sector_bucket: list[str] = []
            for peer_symbol in resolution.peer_symbols[:3]:
                sector_bucket.extend(self._fetch_articles(peer_symbol, profile, decision_time, limit=3))
            seen: set[str] = set()
            deduped: list[str] = []
            for text in sector_bucket:
                if text not in seen:
                    seen.add(text)
                    deduped.append(text)
            sector_articles = tuple(deduped[:9])

        if not stock_articles and not sector_articles:
            return _neutral_snapshot(resolution, "Sentiment coverage is weak, so the news gate is neutral.")

        stock_score, stock_strength = self._score_articles(stock_articles)
        if sector_articles:
            sector_score, sector_strength = self._score_articles(sector_articles)
        else:
            sector_score, sector_strength = None, 0.0

        contextual = (0.7 * stock_score) + (0.3 * sector_score if sector_score is not None else 0.0)
        if sector_score is None:
            contextual = stock_score
        contextual = max(-1.0, min(1.0, contextual))
        coverage = min((len(stock_articles) + len(sector_articles)) / 6.0, 1.0)
        confidence = max(0.0, min(1.0, coverage * max(stock_strength, sector_strength or 0.0, abs(contextual))))
        gate_reason = self._describe_snapshot(contextual, confidence, resolution)
        return SentimentSnapshot(
            stock_sentiment_score=round(stock_score, 4),
            sector_sentiment_score=round(sector_score, 4) if sector_score is not None else None,
            contextual_sentiment_score=round(contextual, 4),
            sentiment_confidence=round(confidence, 4),
            sentiment_gate_reason=gate_reason,
            stock_article_count=len(stock_articles),
            sector_article_count=len(sector_articles),
            sector=resolution.sector,
            sector_available=resolution.sector_available,
            stock_articles=stock_articles,
            sector_articles=sector_articles,
        )

    def gate_threshold(self, base_threshold: float, side: str, snapshot: SentimentSnapshot) -> tuple[float, str | None]:
        score = snapshot.contextual_sentiment_score
        confidence = snapshot.sentiment_confidence
        if confidence < 0.35:
            return base_threshold, "Sentiment coverage is too weak to move the live threshold."

        if side == "LONG":
            if score <= -0.60 and confidence >= 0.65:
                return 1.01, "Strong adverse sentiment vetoed the long setup."
            if score <= -0.35:
                return round(base_threshold + 0.08, 4), "Adverse sentiment raised the long-entry threshold."
            if score >= 0.35:
                return round(max(0.0, base_threshold - 0.03), 4), "Supportive sentiment slightly lowered the long-entry threshold."
            return base_threshold, "Sentiment is mixed, so the long threshold stays unchanged."

        if score >= 0.60 and confidence >= 0.65:
            return 1.01, "Strong adverse sentiment vetoed the short setup."
        if score >= 0.35:
            return round(base_threshold + 0.08, 4), "Adverse sentiment raised the short-entry threshold."
        if score <= -0.35:
            return round(max(0.0, base_threshold - 0.03), 4), "Supportive sentiment slightly lowered the short-entry threshold."
        return base_threshold, "Sentiment is mixed, so the short threshold stays unchanged."

    def _fetch_articles(self, symbol: str, profile: MarketProfile, decision_time: datetime, limit: int = 5) -> list[str]:
        if self.news_fetcher is None:
            return []
        try:
            return list(
                self.news_fetcher(
                    symbol,
                    limit=limit,
                    start_date=_previous_regular_close(decision_time, profile),
                    end_date=decision_time,
                )
            )
        except Exception:
            return []

    def _score_articles(self, articles: tuple[str, ...]) -> tuple[float, float]:
        if not articles:
            return 0.0, 0.0
        try:
            from tradesense.sentiment.aggregator import aggregate_sentiment
            from tradesense.sentiment.finbert import analyze_texts

            aggregated = aggregate_sentiment(analyze_texts(list(articles)))
            score = float(aggregated["sentiment_score"])
            magnitude = abs(score)
            return score, magnitude
        except Exception:
            return 0.0, 0.0

    def _describe_snapshot(self, score: float, confidence: float, resolution: SectorResolution) -> str:
        if confidence < 0.35:
            return "Sentiment coverage is weak, so the news gate is neutral."
        if score >= 0.35:
            if resolution.sector_available:
                return f"Company and {resolution.sector} news are supportive."
            return "Company news is supportive."
        if score <= -0.35:
            if resolution.sector_available:
                return f"Company and {resolution.sector} news are adverse."
            return "Company news is adverse."
        return "News context is mixed and does not materially change the setup."
