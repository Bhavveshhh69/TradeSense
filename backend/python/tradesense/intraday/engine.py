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
            decision = proposal.decision
            no_trade_reason = proposal.reason
            reason_type = proposal.reason_type
            summary = self._build_summary(
                decision=decision,
                setup_side=None,
                probability=None,
                no_trade_reason=no_trade_reason,
                profile=profile,
                reason_type=reason_type,
                threshold_gap=None,
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
                "decision": decision,
                "decision_reason_type": reason_type,
                "actionability_state": "monitor" if decision == "WATCHLIST" else "blocked",
                "confidence_level": self._confidence_level(
                    probability=None,
                    threshold=artifact.threshold,
                    decision=decision,
                    reason_type=reason_type,
                ),
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
                "promotion_gate": artifact.promotion_gate,
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
                "base_threshold": artifact.threshold,
                "effective_threshold": artifact.threshold,
                "threshold_adjustment_reason": "Threshold evaluation was skipped because the setup did not advance past the deterministic gate.",
                "threshold_gap": None,
                **sentiment_snapshot.to_dict(),
            }

        feature_frame = proposal_features(proposal)
        probability = self.registry.predict_probability(artifact, feature_frame)
        base_threshold = artifact.threshold
        adjusted_threshold, gate_reason = self.sentiment_engine.gate_threshold(
            base_threshold,
            proposal.side,
            sentiment_snapshot,
        )
        threshold_gap = float(round(probability - adjusted_threshold, 4))
        if not artifact.promotion_gate.get("passed", False):
            decision = "WATCHLIST"
            reason_type = "promotion_blocked"
            no_trade_reason = str(artifact.promotion_gate.get("reason", "Live promotion gate blocked the setup."))
        elif adjusted_threshold > 1.0:
            decision = "NO_TRADE"
            reason_type = "sentiment_veto"
            no_trade_reason = gate_reason or "Strong adverse sentiment vetoed the setup."
        elif probability >= adjusted_threshold:
            decision = proposal.side
            reason_type = None
            no_trade_reason = None
        else:
            decision = "WATCHLIST"
            reason_type = "threshold_miss"
            no_trade_reason = "Model probability did not clear the live expectancy threshold."

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
            reason_type=reason_type,
            threshold_gap=threshold_gap,
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
            "decision_reason_type": reason_type,
            "actionability_state": "actionable" if decision in {"LONG", "SHORT"} else "monitor" if decision == "WATCHLIST" else "blocked",
            "confidence_level": self._confidence_level(
                probability=probability,
                threshold=adjusted_threshold,
                decision=decision,
                reason_type=reason_type,
            ),
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
            "promotion_gate": artifact.promotion_gate,
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
            "base_threshold": base_threshold,
            "effective_threshold": adjusted_threshold,
            "threshold_adjustment_reason": gate_reason,
            "threshold_gap": threshold_gap,
            **sentiment_snapshot.to_dict(),
        }

    def backtest_market(self, market: str, timeframe_min: int = DEFAULT_TIMEFRAME_MIN) -> dict[str, Any]:
        return self.registry.grouped_walk_forward_report(market, timeframe_min=timeframe_min)

    def validate_symbol(
        self,
        symbol: str,
        timeframe_min: int = DEFAULT_TIMEFRAME_MIN,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        return self.registry.validate_symbol_intraday(
            symbol,
            timeframe_min=timeframe_min,
            start_date=start_date,
            end_date=end_date,
        )

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

    def _build_summary(
        self,
        decision: str,
        setup_side: str | None,
        probability: float | None,
        no_trade_reason: str | None,
        profile,
        reason_type: str | None,
        threshold_gap: float | None,
    ) -> str:
        if decision == "WATCHLIST":
            if reason_type == "threshold_miss" and probability is not None and threshold_gap is not None:
                return (
                    f"Watchlist only for the current {profile.market} session. The {setup_side.lower()} setup is valid, "
                    f"but the estimated win probability is {probability:.0%} and remains {abs(threshold_gap):.0%} below the live threshold."
                )
            if reason_type == "promotion_blocked":
                return (
                    f"Watchlist only for the current {profile.market} session. The price-action setup exists, "
                    f"but live execution is blocked until the market artifact clears the promotion gate. {no_trade_reason or ''}".strip()
                )
            return (
                f"Watchlist for the current {profile.market} session. {no_trade_reason or 'The setup is forming but has not completed its entry conditions yet.'}"
            )
        if decision == "NO_TRADE":
            return f"No intraday trade is being taken for the current {profile.market} session. {no_trade_reason or 'A hard blocker prevented execution.'}"
        return (
            f"{decision.title()} intraday setup detected from the {setup_side.lower()} ORB+VWAP family "
            f"with estimated same-session win probability of {probability:.0%}."
        )

    def _confidence_level(self, probability: float | None, threshold: float, decision: str, reason_type: str | None) -> str:
        if decision == "NO_TRADE" and reason_type in {"hard_blocker", "window_closed", "sentiment_veto"}:
            return "high"
        if decision == "WATCHLIST" and reason_type == "promotion_blocked":
            return "high"
        if decision == "WATCHLIST" and reason_type == "pending_setup":
            return "moderate"
        if probability is None:
            return "low"
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
