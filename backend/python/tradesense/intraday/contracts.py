from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any


@dataclass(frozen=True)
class MarketDataRequest:
    symbol: str
    market: str
    exchange: str
    timezone: str
    currency: str
    timeframe_min: int = 15
    lookback_days: int = 45
    source: str = "yfinance"


@dataclass(frozen=True)
class Bar:
    symbol: str
    market: str
    exchange: str
    timezone: str
    timestamp: datetime
    timeframe_min: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    currency: str
    source: str
    is_regular_session: bool
    session_date: date
    vwap: float | None = None
    trade_count: int | None = None


@dataclass(frozen=True)
class ProviderHealth:
    provider: str
    status: str
    supports_intraday: bool
    supports_markets: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketProfile:
    market: str
    exchange: str
    timezone: str
    currency: str
    regular_open: time
    regular_close: time
    calendar_id: str
    symbol_rules: dict[str, Any]
    entry_window_policy: dict[str, Any]
    forced_exit_policy: dict[str, Any]
    bar_expectation_policy: dict[str, Any]
    holiday_policy: dict[str, Any]


@dataclass(frozen=True)
class SectorResolution:
    symbol: str
    market: str
    sector: str | None
    peer_symbols: tuple[str, ...] = ()
    sector_available: bool = False


@dataclass(frozen=True)
class DataQualityReport:
    missing_bar_count: int
    expected_bar_count: int
    completeness_score: float
    stale_data: bool
    timezone_valid: bool
    session_valid: bool
    usable_for_live: bool
    usable_for_backtest: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_bar_count": self.missing_bar_count,
            "expected_bar_count": self.expected_bar_count,
            "completeness_score": self.completeness_score,
            "stale_data": self.stale_data,
            "timezone_valid": self.timezone_valid,
            "session_valid": self.session_valid,
            "usable_for_live": self.usable_for_live,
            "usable_for_backtest": self.usable_for_backtest,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class BracketSpec:
    entry_price: float
    stop_price: float
    take_profit_price: float
    risk_unit: float


@dataclass(frozen=True)
class TradeProposal:
    symbol: str
    market: str
    strategy_family: str
    side: str
    entry_timestamp: datetime
    features: dict[str, float]
    rationale: tuple[str, ...]
    brackets: BracketSpec


@dataclass(frozen=True)
class StrategyContext:
    symbol: str
    market_profile: MarketProfile
    data_quality: DataQualityReport
    bars: list[Bar]
    latest_timestamp: datetime | None
    latest_session_date: date | None
    feature_frame: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NoTrade:
    reason: str
    decision: str = "NO_TRADE"
    reason_type: str = "hard_blocker"
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class SentimentSnapshot:
    stock_sentiment_score: float
    sector_sentiment_score: float | None
    contextual_sentiment_score: float
    sentiment_confidence: float
    sentiment_gate_reason: str
    stock_article_count: int
    sector_article_count: int
    sector: str | None = None
    sector_available: bool = False
    stock_articles: tuple[str, ...] = ()
    sector_articles: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stock_sentiment_score": self.stock_sentiment_score,
            "sector_sentiment_score": self.sector_sentiment_score,
            "contextual_sentiment_score": self.contextual_sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "sentiment_gate_reason": self.sentiment_gate_reason,
            "stock_article_count": self.stock_article_count,
            "sector_article_count": self.sector_article_count,
            "sector": self.sector,
            "sector_available": self.sector_available,
            "stock_articles": list(self.stock_articles),
            "sector_articles": list(self.sector_articles),
        }
