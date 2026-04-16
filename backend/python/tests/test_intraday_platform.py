import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradesense.intraday.contracts import Bar, BracketSpec, DataQualityReport, NoTrade, StrategyContext, TradeProposal  # noqa: E402
from tradesense.intraday.engine import IntradayEngine  # noqa: E402
from tradesense.intraday.market import get_market_profile, resolve_market  # noqa: E402
from tradesense.intraday.provider import bars_to_frame  # noqa: E402
from tradesense.intraday.quality import DataQualityValidator  # noqa: E402
from tradesense.intraday.registry import ModelRegistry  # noqa: E402
from tradesense.intraday.sector import SectorResolver  # noqa: E402
from tradesense.intraday.sentiment import IntradaySentimentEngine  # noqa: E402
from tradesense.intraday.strategy import FEATURE_COLUMNS, ORBSessionVWAPStrategy  # noqa: E402


def _bar(symbol: str, timestamp: str, close: float, *, market: str, exchange: str, timezone: str) -> Bar:
    ts = datetime.fromisoformat(timestamp).astimezone(ZoneInfo(timezone))
    return Bar(
        symbol=symbol,
        market=market,
        exchange=exchange,
        timezone=timezone,
        timestamp=ts,
        timeframe_min=15,
        open=close - 0.5,
        high=close + 0.5,
        low=close - 1.0,
        close=close,
        volume=1000.0,
        currency="USD" if market == "US" else "INR",
        source="mock",
        is_regular_session=True,
        session_date=ts.date(),
        vwap=close - 0.2,
        trade_count=None,
    )


def test_market_profile_resolves_us_and_india():
    assert resolve_market("AAPL") == ("US", "NASDAQ")
    assert resolve_market("RELIANCE.NS") == ("IN", "NSE")
    assert resolve_market("^NSEI") == ("IN", "NSE")
    assert resolve_market("^BSESN") == ("IN", "BSE")
    assert get_market_profile("AAPL").timezone == "America/New_York"
    assert get_market_profile("RELIANCE.NS").timezone == "Asia/Kolkata"
    assert get_market_profile("^NSEI").timezone == "Asia/Kolkata"


def test_data_quality_validator_detects_missing_bars():
    profile = get_market_profile("AAPL")
    validator = DataQualityValidator()
    bars = [
        _bar("AAPL", "2026-04-15T09:30:00-04:00", 100.0, market="US", exchange="NASDAQ", timezone=profile.timezone),
        _bar("AAPL", "2026-04-15T09:45:00-04:00", 101.0, market="US", exchange="NASDAQ", timezone=profile.timezone),
    ]
    report = validator.validate(bars, profile, timeframe_min=15)
    assert isinstance(report, DataQualityReport)
    assert report.missing_bar_count > 0
    assert report.expected_bar_count >= len(bars)


def test_strategy_builds_intraday_feature_frame():
    profile = get_market_profile("AAPL")
    strategy = ORBSessionVWAPStrategy()
    bars = []
    closes = [100, 101, 102, 103, 104]
    timestamps = [
        "2026-04-15T09:30:00-04:00",
        "2026-04-15T09:45:00-04:00",
        "2026-04-15T10:00:00-04:00",
        "2026-04-15T10:15:00-04:00",
        "2026-04-15T10:30:00-04:00",
    ]
    for timestamp, close in zip(timestamps, closes):
        bars.append(_bar("AAPL", timestamp, close, market="US", exchange="NASDAQ", timezone=profile.timezone))
    report = DataQualityReport(0, 5, 1.0, False, True, True, True, True, ())
    context = strategy.build_context("AAPL", bars, profile, report)
    frame = bars_to_frame(bars)
    assert not frame.empty
    assert not context.feature_frame.empty
    assert "session_vwap" in context.feature_frame.columns
    assert "breakout_strength" in context.feature_frame.columns


def test_model_registry_uses_separate_market_artifact_paths():
    registry = ModelRegistry()
    us_path = registry.artifact_path("US")
    in_path = registry.artifact_path("IN")
    assert us_path != in_path
    assert "intraday_US" in us_path.name
    assert "intraday_IN" in in_path.name


def test_sector_resolver_handles_us_india_and_unmapped():
    resolver = SectorResolver()
    us = resolver.resolve("AAPL")
    india = resolver.resolve("RELIANCE.NS")
    missing = resolver.resolve("UNKNOWN")
    assert us.sector == "Technology"
    assert india.sector == "Energy"
    assert missing.sector is None
    assert missing.sector_available is False


def test_sentiment_engine_combines_stock_and_sector_scores():
    profile = get_market_profile("AAPL")

    def _fake_fetch(symbol, limit=10, **kwargs):
        mapping = {
            "AAPL": ["Apple strong demand"],
            "MSFT": ["Microsoft cloud strength"],
            "NVDA": ["Nvidia AI demand"],
            "META": ["Meta ad growth"],
        }
        return mapping.get(symbol, [])

    engine = IntradaySentimentEngine(news_fetcher=_fake_fetch)
    engine._score_articles = lambda articles: (0.5 if "Apple strong demand" in articles else 0.2, 0.8)  # type: ignore[attr-defined]
    snapshot = engine.snapshot("AAPL", profile, datetime.fromisoformat("2026-04-15T10:15:00-04:00"))
    assert snapshot.stock_sentiment_score == 0.5
    assert snapshot.sector_sentiment_score == 0.2
    assert snapshot.contextual_sentiment_score == 0.41


def test_sentiment_engine_uses_neutral_fallback_on_low_coverage():
    profile = get_market_profile("AAPL")
    engine = IntradaySentimentEngine(news_fetcher=lambda *args, **kwargs: [])
    snapshot = engine.snapshot("AAPL", profile, datetime.fromisoformat("2026-04-15T10:15:00-04:00"))
    assert snapshot.contextual_sentiment_score == 0.0
    assert snapshot.sentiment_confidence == 0.0
    assert "neutral" in snapshot.sentiment_gate_reason.lower()


def test_sentiment_gate_adjusts_thresholds_and_hard_rejects():
    engine = IntradaySentimentEngine(news_fetcher=lambda *args, **kwargs: [])
    supportive = type("Snapshot", (), {"contextual_sentiment_score": 0.4, "sentiment_confidence": 0.8})()
    adverse = type("Snapshot", (), {"contextual_sentiment_score": -0.7, "sentiment_confidence": 0.8})()
    threshold, _ = engine.gate_threshold(0.55, "LONG", supportive)
    hard_threshold, reason = engine.gate_threshold(0.55, "LONG", adverse)
    assert threshold == 0.52
    assert hard_threshold > 1.0
    assert "vetoed" in reason.lower()


def test_walk_forward_metrics_report_sentiment_uplift(monkeypatch):
    registry = ModelRegistry()
    dataset = pd.DataFrame(
        [
            {
                "session_date": f"2026-04-{day:02d}",
                "target": 1 if day % 2 == 0 else 0,
                "r_multiple": 1.5 if day % 2 == 0 else -1.0,
                "setup_side": "LONG",
                "contextual_sentiment_score": 0.5 if day % 2 == 0 else -0.5,
                "sentiment_confidence": 0.8,
                **{name: 0.1 for name in FEATURE_COLUMNS},
            }
            for day in range(1, 25)
        ]
    )

    monkeypatch.setattr(registry, "_build_training_dataset", lambda market, timeframe_min: dataset)
    report = registry.grouped_walk_forward_report("US")
    assert report["market"] == "US"
    assert "xgboost" in report["models"]
    assert "sentiment_uplift" in report["models"]["xgboost"]


def test_strategy_marks_pre_window_as_watchlist():
    profile = get_market_profile("AAPL")
    strategy = ORBSessionVWAPStrategy()
    context = StrategyContext(
        symbol="AAPL",
        market_profile=profile,
        data_quality=DataQualityReport(0, 2, 1.0, False, True, True, True, True, ()),
        bars=[],
        latest_timestamp=datetime.fromisoformat("2026-04-15T09:45:00-04:00"),
        latest_session_date=datetime.fromisoformat("2026-04-15T09:45:00-04:00").date(),
        feature_frame=pd.DataFrame(
            [
                {
                    "session_date": datetime.fromisoformat("2026-04-15T09:45:00-04:00").date(),
                    "is_regular_session": True,
                    "timestamp": pd.Timestamp("2026-04-15T09:45:00-04:00"),
                    "close": 101.0,
                    "opening_range_high": 102.0,
                    "opening_range_low": 99.0,
                    "session_vwap": 100.5,
                    "vwap_slope": 0.1,
                    "breakout_strength": 0.0,
                    "opening_range_width": 3.0,
                }
            ]
        ),
    )

    proposal = strategy.propose_trade(context)
    assert isinstance(proposal, NoTrade)
    assert proposal.decision == "WATCHLIST"
    assert proposal.reason_type == "pending_setup"


def test_engine_returns_watchlist_for_threshold_miss_without_sentiment_blocker():
    profile = get_market_profile("AAPL")
    timestamp = datetime.fromisoformat("2026-04-15T10:15:00-04:00")
    bars = [_bar("AAPL", "2026-04-15T10:15:00-04:00", 100.0, market="US", exchange="NASDAQ", timezone=profile.timezone)]

    class _Provider:
        def fetch_bars(self, request):
            return bars

    class _Validator:
        def validate(self, bars_arg, profile_arg, timeframe_min):
            return DataQualityReport(0, 1, 1.0, False, True, True, True, True, ())

    class _Strategy:
        def build_context(self, symbol, bars_arg, profile_arg, quality):
            return StrategyContext(
                symbol=symbol,
                market_profile=profile_arg,
                data_quality=quality,
                bars=bars_arg,
                latest_timestamp=timestamp,
                latest_session_date=timestamp.date(),
                feature_frame=pd.DataFrame(),
            )

        def propose_trade(self, context):
            return TradeProposal(
                symbol="AAPL",
                market="US",
                strategy_family="orb_vwap_continuation",
                side="LONG",
                entry_timestamp=timestamp,
                features={name: 0.1 for name in FEATURE_COLUMNS},
                rationale=("opening-range breakout",),
                brackets=BracketSpec(
                    entry_price=100.0,
                    stop_price=99.0,
                    take_profit_price=101.5,
                    risk_unit=1.0,
                ),
            )

    class _Registry:
        def load_or_train(self, market, timeframe_min=15):
            return type(
                "Artifact",
                (),
                {
                    "threshold": 0.62,
                    "feature_names": tuple(FEATURE_COLUMNS),
                    "model_name": "xgboost",
                    "metadata": {"market": market},
                    "model_bench_summary": {},
                    "promotion_gate": {"passed": True, "reason": "Promotion gate passed.", "market": market, "artifact_timestamp": "2026-04-15T14:00:00+00:00"},
                },
            )()

        def predict_probability(self, artifact, feature_frame):
            return 0.60

    class _Sentiment:
        def gate_threshold(self, base_threshold, side, snapshot):
            return base_threshold, "Sentiment is mixed, so the long threshold stays unchanged."

        def snapshot(self, symbol, profile_arg, decision_time, manual_news=None):
            return type(
                "Snapshot",
                (),
                {
                    "stock_sentiment_score": 0.0,
                    "sector_sentiment_score": None,
                    "contextual_sentiment_score": 0.0,
                    "sentiment_confidence": 0.0,
                    "sentiment_gate_reason": "Sentiment coverage is weak, so the news gate is neutral.",
                    "stock_article_count": 0,
                    "sector_article_count": 0,
                    "sector": None,
                    "sector_available": False,
                    "stock_articles": (),
                    "sector_articles": (),
                    "to_dict": lambda self=None: {
                        "stock_sentiment_score": 0.0,
                        "sector_sentiment_score": None,
                        "contextual_sentiment_score": 0.0,
                        "sentiment_confidence": 0.0,
                        "sentiment_gate_reason": "Sentiment coverage is weak, so the news gate is neutral.",
                        "stock_article_count": 0,
                        "sector_article_count": 0,
                        "sector": None,
                        "sector_available": False,
                        "stock_articles": [],
                        "sector_articles": [],
                    },
                },
            )()

    engine = IntradayEngine(
        provider=_Provider(),
        validator=_Validator(),
        strategy=_Strategy(),
        registry=_Registry(),
        sentiment_engine=_Sentiment(),
    )

    result = engine.predict("AAPL")
    assert result["decision"] == "WATCHLIST"
    assert result["decision_reason_type"] == "threshold_miss"
    assert result["no_trade_reason"] == "Model probability did not clear the live expectancy threshold."


def test_engine_blocks_actionable_setup_when_promotion_gate_fails():
    profile = get_market_profile("AAPL")
    timestamp = datetime.fromisoformat("2026-04-15T10:15:00-04:00")
    bars = [_bar("AAPL", "2026-04-15T10:15:00-04:00", 100.0, market="US", exchange="NASDAQ", timezone=profile.timezone)]

    class _Provider:
        def fetch_bars(self, request):
            return bars

    class _Validator:
        def validate(self, bars_arg, profile_arg, timeframe_min):
            return DataQualityReport(0, 1, 1.0, False, True, True, True, True, ())

    class _Strategy:
        def build_context(self, symbol, bars_arg, profile_arg, quality):
            return StrategyContext(
                symbol=symbol,
                market_profile=profile_arg,
                data_quality=quality,
                bars=bars_arg,
                latest_timestamp=timestamp,
                latest_session_date=timestamp.date(),
                feature_frame=pd.DataFrame(),
            )

        def propose_trade(self, context):
            return TradeProposal(
                symbol="AAPL",
                market="US",
                strategy_family="orb_vwap_continuation",
                side="LONG",
                entry_timestamp=timestamp,
                features={name: 0.2 for name in FEATURE_COLUMNS},
                rationale=("opening-range breakout",),
                brackets=BracketSpec(entry_price=100.0, stop_price=99.0, take_profit_price=101.5, risk_unit=1.0),
            )

    class _Registry:
        def load_or_train(self, market, timeframe_min=15):
            return type(
                "Artifact",
                (),
                {
                    "threshold": 0.55,
                    "feature_names": tuple(FEATURE_COLUMNS),
                    "model_name": "xgboost",
                    "metadata": {"market": market},
                    "model_bench_summary": {},
                    "promotion_gate": {
                        "passed": False,
                        "reason": "Promotion blocked because holdout net expectancy is not positive.",
                        "market": market,
                        "artifact_timestamp": "2026-04-15T14:00:00+00:00",
                    },
                },
            )()

        def predict_probability(self, artifact, feature_frame):
            return 0.72

    class _Sentiment:
        def gate_threshold(self, base_threshold, side, snapshot):
            return base_threshold, "Sentiment is mixed, so the long threshold stays unchanged."

        def snapshot(self, symbol, profile_arg, decision_time, manual_news=None):
            return type(
                "Snapshot",
                (),
                {
                    "stock_sentiment_score": 0.0,
                    "sector_sentiment_score": None,
                    "contextual_sentiment_score": 0.0,
                    "sentiment_confidence": 0.0,
                    "sentiment_gate_reason": "Sentiment coverage is weak, so the news gate is neutral.",
                    "stock_article_count": 0,
                    "sector_article_count": 0,
                    "sector": None,
                    "sector_available": False,
                    "stock_articles": (),
                    "sector_articles": (),
                    "to_dict": lambda self=None: {
                        "stock_sentiment_score": 0.0,
                        "sector_sentiment_score": None,
                        "contextual_sentiment_score": 0.0,
                        "sentiment_confidence": 0.0,
                        "sentiment_gate_reason": "Sentiment coverage is weak, so the news gate is neutral.",
                        "stock_article_count": 0,
                        "sector_article_count": 0,
                        "sector": None,
                        "sector_available": False,
                        "stock_articles": [],
                        "sector_articles": [],
                    },
                },
            )()

    engine = IntradayEngine(
        provider=_Provider(),
        validator=_Validator(),
        strategy=_Strategy(),
        registry=_Registry(),
        sentiment_engine=_Sentiment(),
    )

    result = engine.predict("AAPL")
    assert result["decision"] == "WATCHLIST"
    assert result["decision_reason_type"] == "promotion_blocked"
    assert "Promotion blocked" in result["no_trade_reason"]
