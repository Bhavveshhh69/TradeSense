from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import BracketSpec, DataQualityReport, MarketProfile, NoTrade, StrategyContext, TradeProposal
from .market import STRATEGY_FAMILY, combine_local, parse_hhmm
from .provider import bars_to_frame


FEATURE_COLUMNS = [
    "side_bias",
    "vwap_distance",
    "vwap_slope",
    "opening_range_width",
    "breakout_strength",
    "gap_vs_prior_close",
    "relative_volume",
    "body_wick_imbalance",
    "distance_to_session_high",
    "distance_to_session_low",
    "continuation_2",
    "range_expansion",
    "compression_ratio",
    "session_progress",
]


class StrategyFamily(ABC):
    @abstractmethod
    def build_context(
        self,
        symbol: str,
        bars: list,
        market_profile: MarketProfile,
        data_quality: DataQualityReport,
    ) -> StrategyContext:
        raise NotImplementedError

    @abstractmethod
    def propose_trade(self, context: StrategyContext) -> TradeProposal | NoTrade:
        raise NotImplementedError

    @abstractmethod
    def build_brackets(self, proposal: TradeProposal) -> BracketSpec:
        raise NotImplementedError

    @abstractmethod
    def describe_no_trade(self, context: StrategyContext) -> str:
        raise NotImplementedError


class ORBSessionVWAPStrategy(StrategyFamily):
    strategy_family = STRATEGY_FAMILY

    def build_context(
        self,
        symbol: str,
        bars: list,
        market_profile: MarketProfile,
        data_quality: DataQualityReport,
    ) -> StrategyContext:
        frame = bars_to_frame(bars)
        if frame.empty:
            return StrategyContext(
                symbol=symbol,
                market_profile=market_profile,
                data_quality=data_quality,
                bars=bars,
                latest_timestamp=None,
                latest_session_date=None,
                feature_frame=frame,
                metadata={"reason": "no bars"},
            )
        frame = self._build_features(frame, market_profile)
        latest_session_date = frame["session_date"].iloc[-1]
        latest_timestamp = frame["timestamp"].iloc[-1].to_pydatetime()
        return StrategyContext(
            symbol=symbol,
            market_profile=market_profile,
            data_quality=data_quality,
            bars=bars,
            latest_timestamp=latest_timestamp,
            latest_session_date=latest_session_date,
            feature_frame=frame,
            metadata={},
        )

    def propose_trade(self, context: StrategyContext) -> TradeProposal | NoTrade:
        frame = context.feature_frame
        if frame.empty:
            return NoTrade("No market bars available")
        session_frame = frame[frame["session_date"] == context.latest_session_date].copy()
        session_frame = session_frame[session_frame["is_regular_session"]].copy()
        if session_frame.empty:
            return NoTrade("No regular-session bars available")

        latest = session_frame.iloc[-1]
        profile = context.market_profile
        entry_start = combine_local(context.latest_session_date, parse_hhmm(profile.entry_window_policy["start"]), profile)
        entry_end = combine_local(context.latest_session_date, parse_hhmm(profile.entry_window_policy["end"]), profile)
        latest_ts = latest["timestamp"].to_pydatetime()
        if latest_ts < entry_start:
            return NoTrade(
                "Entry window has not opened yet",
                decision="WATCHLIST",
                reason_type="pending_setup",
            )
        if latest_ts > entry_end:
            return NoTrade("Entry window is closed", reason_type="window_closed")
        if not context.data_quality.usable_for_live:
            return NoTrade(
                "Data quality gate failed",
                reason_type="hard_blocker",
                details=tuple(context.data_quality.warnings),
            )

        long_ok = bool(
            latest["close"] > latest["opening_range_high"]
            and latest["close"] > latest["session_vwap"]
            and latest["vwap_slope"] > 0
            and latest["breakout_strength"] > 0
        )
        short_ok = bool(
            latest["close"] < latest["opening_range_low"]
            and latest["close"] < latest["session_vwap"]
            and latest["vwap_slope"] < 0
            and latest["breakout_strength"] < 0
        )
        if not long_ok and not short_ok:
            return NoTrade(
                self.describe_no_trade(context),
                decision="WATCHLIST",
                reason_type="pending_setup",
            )

        side = "LONG" if long_ok else "SHORT"
        risk_unit = max(float(latest["opening_range_width"]), float(latest["close"]) * 0.0035)
        brackets = self._build_brackets_from_side(side, float(latest["close"]), risk_unit)
        features = {name: float(latest[name]) for name in FEATURE_COLUMNS}
        return TradeProposal(
            symbol=context.symbol,
            market=context.market_profile.market,
            strategy_family=self.strategy_family,
            side=side,
            entry_timestamp=latest_ts,
            features=features,
            rationale=(
                "opening-range breakout",
                "session VWAP confirmation",
                "intraday single-window entry",
            ),
            brackets=brackets,
        )

    def build_brackets(self, proposal: TradeProposal) -> BracketSpec:
        return proposal.brackets

    def describe_no_trade(self, context: StrategyContext) -> str:
        if context.feature_frame.empty:
            return "No bars available for feature generation"
        latest = context.feature_frame.iloc[-1]
        if latest["close"] <= latest["opening_range_high"] and latest["close"] >= latest["opening_range_low"]:
            return "Price has not broken the opening range"
        if latest["close"] > latest["opening_range_high"] and latest["close"] <= latest["session_vwap"]:
            return "Long breakout is not confirmed by session VWAP"
        if latest["close"] < latest["opening_range_low"] and latest["close"] >= latest["session_vwap"]:
            return "Short breakdown is not confirmed by session VWAP"
        return "Current bar does not satisfy the intraday setup filters"

    def _build_features(self, frame: pd.DataFrame, market_profile: MarketProfile) -> pd.DataFrame:
        working = frame.copy()
        working = working[working["is_regular_session"]].copy()
        if working.empty:
            return working
        working["typical_price"] = (working["high"] + working["low"] + working["close"]) / 3.0
        working["tpv"] = working["typical_price"] * working["volume"]
        working["session_vwap"] = working.groupby("session_date")["tpv"].cumsum() / working.groupby("session_date")["volume"].cumsum().replace(0, np.nan)
        working["vwap_distance"] = (working["close"] - working["session_vwap"]) / working["session_vwap"].replace(0, np.nan)
        working["vwap_slope"] = working.groupby("session_date")["session_vwap"].diff().fillna(0.0)
        working["bar_index"] = working.groupby("session_date").cumcount()
        opening_bars = int(market_profile.entry_window_policy.get("opening_range_bars", 2))
        or_stats = (
            working[working["bar_index"] < opening_bars]
            .groupby("session_date")
            .agg(opening_range_high=("high", "max"), opening_range_low=("low", "min"))
        )
        or_stats["opening_range_width"] = (or_stats["opening_range_high"] - or_stats["opening_range_low"]).clip(lower=1e-6)
        working = working.merge(or_stats, on="session_date", how="left")
        working["breakout_strength"] = np.where(
            working["close"] > working["opening_range_high"],
            (working["close"] - working["opening_range_high"]) / working["opening_range_width"],
            np.where(
                working["close"] < working["opening_range_low"],
                -((working["opening_range_low"] - working["close"]) / working["opening_range_width"]),
                0.0,
            ),
        )
        session_first = working.groupby("session_date")["open"].transform("first")
        prior_close = working.groupby("session_date")["close"].transform("last").shift(1)
        working["gap_vs_prior_close"] = (session_first - prior_close) / prior_close.replace(0, np.nan)
        working["relative_volume"] = working["volume"] / working["volume"].rolling(20, min_periods=3).mean()
        body = (working["close"] - working["open"]).abs()
        wick = (working["high"] - working["low"]).replace(0, np.nan)
        working["body_wick_imbalance"] = body / wick
        working["distance_to_session_high"] = (
            working.groupby("session_date")["high"].cummax() - working["close"]
        ) / working["close"].replace(0, np.nan)
        working["distance_to_session_low"] = (
            working["close"] - working.groupby("session_date")["low"].cummin()
        ) / working["close"].replace(0, np.nan)
        working["continuation_2"] = working.groupby("session_date")["close"].pct_change(2).fillna(0.0)
        bar_range = (working["high"] - working["low"]).clip(lower=1e-6)
        working["range_expansion"] = bar_range / bar_range.rolling(20, min_periods=3).mean()
        working["compression_ratio"] = working["opening_range_width"] / working.groupby("session_date")["opening_range_width"].transform("first").replace(0, np.nan)
        bars_per_session = max(int(market_profile.bar_expectation_policy["bars_per_session"]), 1)
        working["session_progress"] = (working["bar_index"] + 1) / bars_per_session
        breakout_sign = np.sign(working["breakout_strength"])
        vwap_sign = np.sign(working["vwap_distance"])
        working["side_bias"] = np.where(breakout_sign != 0, breakout_sign, vwap_sign)
        working[FEATURE_COLUMNS] = working[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return working

    def _build_brackets_from_side(self, side: str, entry_price: float, risk_unit: float) -> BracketSpec:
        if side == "LONG":
            return BracketSpec(
                entry_price=entry_price,
                stop_price=entry_price - risk_unit,
                take_profit_price=entry_price + risk_unit * 1.5,
                risk_unit=risk_unit,
            )
        return BracketSpec(
            entry_price=entry_price,
            stop_price=entry_price + risk_unit,
            take_profit_price=entry_price - risk_unit * 1.5,
            risk_unit=risk_unit,
        )


def proposal_features(proposal: TradeProposal) -> pd.DataFrame:
    return pd.DataFrame([{name: proposal.features.get(name, 0.0) for name in FEATURE_COLUMNS}])
