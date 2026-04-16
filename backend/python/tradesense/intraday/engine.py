from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .contracts import NoTrade, SentimentSnapshot
from .market import DEFAULT_TIMEFRAME_MIN, STRATEGY_FAMILY, combine_local, get_market_profile, parse_hhmm
from .provider import YahooIntradayProvider, latest_regular_close
from .quality import DataQualityValidator
from .registry import ModelRegistry
from .sentiment import IntradaySentimentEngine
from .strategy import ORBSessionVWAPStrategy, proposal_features


def _neutral_sentiment(reason: str) -> SentimentSnapshot:
    return SentimentSnapshot(
        stock_sentiment_score=0.0,
        sector_sentiment_score=None,
        contextual_sentiment_score=0.0,
        sentiment_confidence=0.0,
        sentiment_gate_reason=reason,
        stock_article_count=0,
        sector_article_count=0,
        sector=None,
        sector_available=False,
        stock_articles=(),
        sector_articles=(),
    )


@dataclass
class IntradayEngine:
    provider: YahooIntradayProvider | None = None
    validator: DataQualityValidator | None = None
    strategy: ORBSessionVWAPStrategy | None = None
    registry: ModelRegistry | None = None
    sentiment_engine: IntradaySentimentEngine | None = None

    def __post_init__(self) -> None:
        self.provider = self.provider or YahooIntradayProvider()
        self.validator = self.validator or DataQualityValidator()
        self.strategy = self.strategy or ORBSessionVWAPStrategy()
        self.sentiment_engine = self.sentiment_engine or IntradaySentimentEngine()
        self.registry = self.registry or ModelRegistry(provider=self.provider, sentiment_engine=self.sentiment_engine)

    def predict(
        self,
        symbol: str,
        timeframe_min: int = DEFAULT_TIMEFRAME_MIN,
        news_texts: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_symbol = symbol.strip().upper()
        profile = get_market_profile(normalized_symbol, timeframe_min=timeframe_min)
        request = type(
            "Req",
            (),
            {
                "symbol": normalized_symbol,
                "market": profile.market,
                "exchange": profile.exchange,
                "timezone": profile.timezone,
                "currency": profile.currency,
                "timeframe_min": timeframe_min,
                "lookback_days": 45,
                "source": "yfinance",
            },
        )()
        bars = self.provider.fetch_bars(request)
        quality = self.validator.validate(bars, profile, timeframe_min)
        context = self.strategy.build_context(normalized_symbol, bars, profile, quality)
        artifact = self.registry.load_or_train(profile.market, timeframe_min=timeframe_min)
        latest_close, latest_close_ts = latest_regular_close(bars)
        generated_at = datetime.now(tz=UTC).isoformat()
        sentiment_snapshot = self._build_sentiment_snapshot(profile, context.latest_timestamp or latest_close_ts, normalized_symbol, news_texts)
        proposal = self.strategy.propose_trade(context)

        if isinstance(proposal, NoTrade):
            no_trade_reason = proposal.reason
            summary = self._build_summary(
                decision="NO_TRADE",
                setup_side=None,
                probability=0.0,
                no_trade_reason=no_trade_reason,
                profile=profile,
            )
            return {
                "symbol": normalized_symbol,
                "market": profile.market,
                "exchange": profile.exchange,
                "timeframe": f"{timeframe_min}m",
                "strategy_family": STRATEGY_FAMILY,
                "prediction": 0,
                "probability": 0.0,
                "confidence": 0.0,
                "decision": "NO_TRADE",
                "confidence_level": self._confidence_level(0.0, artifact.threshold),
                "strength": 0.0,
                "context": self._build_context_payload(profile, quality, None, artifact, sentiment_snapshot),
                "model_version": f"intraday-{artifact.model_name}",
                "model_name": artifact.model_name,
                "model_threshold": artifact.threshold,
                "model_bench_summary": artifact.model_bench_summary,
                "timestamp": latest_close_ts.isoformat() if latest_close_ts else generated_at,
                "generated_at": generated_at,
                "setup_side": None,
                "entry_price": None,
                "stop_price": None,
                "take_profit_price": None,
                "forced_exit_time": None,
                "no_trade_reason": no_trade_reason,
                "data_quality": quality.to_dict(),
                "summary": summary,
                "market_context": {
                    "market": profile.market,
                    "exchange": profile.exchange,
                    "session_window": profile.entry_window_policy,
                    "sector": sentiment_snapshot.sector,
                },
                "key_drivers": [],
                "risk_notes": list(quality.warnings),
                "model_honesty": "No-trade output is based on setup filters or quality gates before the probability model is allowed to act.",
                "current_price": latest_close,
                "trade_window": profile.entry_window_policy,
                "threshold": artifact.threshold,
                **sentiment_snapshot.to_dict(),
            }

        feature_frame = proposal_features(proposal)
        probability = self.registry.predict_probability(artifact, feature_frame)
        adjusted_threshold, gate_reason = self.sentiment_engine.gate_threshold(
            artifact.threshold,
            proposal.side,
            sentiment_snapshot,
        )
        decision = proposal.side if probability >= adjusted_threshold else "NO_TRADE"
        if adjusted_threshold > 1.0:
            decision = "NO_TRADE"
            no_trade_reason = gate_reason
        elif decision == "NO_TRADE":
            no_trade_reason = gate_reason or "Model probability did not clear the live expectancy threshold"
        else:
            no_trade_reason = None

        forced_exit_dt = combine_local(
            context.latest_session_date,
            parse_hhmm(profile.forced_exit_policy["time"]),
            profile,
        )
        summary = self._build_summary(
            decision=decision,
            setup_side=proposal.side,
            probability=probability,
            no_trade_reason=no_trade_reason,
            profile=profile,
        )
        return {
            "symbol": normalized_symbol,
            "market": profile.market,
            "exchange": profile.exchange,
            "timeframe": f"{timeframe_min}m",
            "strategy_family": STRATEGY_FAMILY,
            "prediction": 1 if decision in {"LONG", "SHORT"} else 0,
            "probability": float(round(probability, 4)),
            "confidence": float(round(abs(probability - adjusted_threshold), 4)),
            "decision": decision,
            "confidence_level": self._confidence_level(probability, adjusted_threshold),
            "strength": float(round(max(0.0, probability - adjusted_threshold), 4)),
            "context": self._build_context_payload(profile, quality, proposal, artifact, sentiment_snapshot),
            "model_version": f"intraday-{artifact.model_name}",
            "model_name": artifact.model_name,
            "model_threshold": adjusted_threshold,
            "model_bench_summary": artifact.model_bench_summary,
            "timestamp": proposal.entry_timestamp.isoformat(),
            "generated_at": generated_at,
            "setup_side": proposal.side,
            "entry_price": round(proposal.brackets.entry_price, 4),
            "stop_price": round(proposal.brackets.stop_price, 4),
            "take_profit_price": round(proposal.brackets.take_profit_price, 4),
            "forced_exit_time": forced_exit_dt.isoformat(),
            "no_trade_reason": no_trade_reason,
            "data_quality": quality.to_dict(),
            "summary": summary,
            "market_context": {
                "market": profile.market,
                "exchange": profile.exchange,
                "session_window": profile.entry_window_policy,
                "sector": sentiment_snapshot.sector,
            },
            "key_drivers": self._rank_drivers(proposal.features),
            "risk_notes": self._build_risk_notes(quality, proposal.features, sentiment_snapshot),
            "model_honesty": "The probability estimates a same-session bracket outcome for the detected setup. News sentiment can adjust the gate, but price action remains the primary alpha source.",
            "current_price": latest_close,
            "trade_window": profile.entry_window_policy,
            "threshold": adjusted_threshold,
            **sentiment_snapshot.to_dict(),
        }

    def backtest_market(self, market: str, timeframe_min: int = DEFAULT_TIMEFRAME_MIN) -> dict[str, Any]:
        return self.registry.grouped_walk_forward_report(market, timeframe_min=timeframe_min)

    def _build_sentiment_snapshot(
        self,
        profile,
        decision_time: datetime | None,
        symbol: str,
        news_texts: list[str] | None,
    ) -> SentimentSnapshot:
        if decision_time is None:
            return _neutral_sentiment("No recent regular-session timestamp was available for sentiment analysis.")
        return self.sentiment_engine.snapshot(symbol, profile, decision_time, manual_news=news_texts)

    def _build_context_payload(self, profile, quality, proposal, artifact, sentiment_snapshot) -> dict[str, Any]:
        payload = {
            "trend_summary": "Intraday setup is evaluated against opening-range direction and session VWAP alignment.",
            "risk_summary": "Quality gates, bracket sizing, and sentiment gates are session-aware and market-aware.",
            "market": profile.market,
            "exchange": profile.exchange,
            "entry_window": profile.entry_window_policy,
            "data_quality": quality.to_dict(),
            "sentiment": {
                "contextual_score": sentiment_snapshot.contextual_sentiment_score,
                "confidence": sentiment_snapshot.sentiment_confidence,
                "gate_reason": sentiment_snapshot.sentiment_gate_reason,
                "sector": sentiment_snapshot.sector,
            },
        }
        if proposal is not None:
            payload["setup_rationale"] = list(proposal.rationale)
        if artifact is not None:
            payload["model_metadata"] = artifact.metadata
        return payload

    def _build_summary(self, decision: str, setup_side: str | None, probability: float, no_trade_reason: str | None, profile) -> str:
        if decision == "NO_TRADE":
            return f"No intraday trade is being taken for the current {profile.market} session. {no_trade_reason or 'The setup did not clear the entry policy.'}"
        return (
            f"{decision.title()} intraday setup detected from the {setup_side.lower()} ORB+VWAP family "
            f"with estimated same-session win probability of {probability:.0%}."
        )

    def _confidence_level(self, probability: float, threshold: float) -> str:
        distance = abs(probability - threshold)
        if distance >= 0.2:
            return "strong"
        if distance >= 0.1:
            return "high"
        if distance >= 0.05:
            return "moderate"
        return "low"

    def _rank_drivers(self, features: dict[str, float]) -> list[str]:
        ordered = sorted(features.items(), key=lambda item: abs(item[1]), reverse=True)
        return [name for name, _ in ordered[:4]]

    def _build_risk_notes(self, quality, features: dict[str, float], sentiment_snapshot: SentimentSnapshot) -> list[str]:
        notes = list(quality.warnings)
        if features.get("range_expansion", 0.0) > 1.5:
            notes.append("Range expansion is elevated, which can increase stop-out risk.")
        if abs(features.get("gap_vs_prior_close", 0.0)) > 0.02:
            notes.append("Opening gap is large relative to recent intraday structure.")
        if sentiment_snapshot.sentiment_confidence >= 0.35 and abs(sentiment_snapshot.contextual_sentiment_score) >= 0.35:
            notes.append(sentiment_snapshot.sentiment_gate_reason)
        return notes


ENGINE = IntradayEngine()


def analyze_symbol(symbol: str, news_texts: list[str] | None = None) -> dict[str, Any]:
    return ENGINE.predict(symbol, news_texts=news_texts)
