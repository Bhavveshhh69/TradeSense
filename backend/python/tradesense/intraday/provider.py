from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf

from .contracts import Bar, MarketDataRequest, ProviderHealth
from .market import get_market_profile, market_tz, normalize_session_date


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_bars(self, request: MarketDataRequest) -> list[Bar]:
        raise NotImplementedError

    @abstractmethod
    def get_provider_health(self) -> ProviderHealth:
        raise NotImplementedError


class YahooIntradayProvider(MarketDataProvider):
    def __init__(self) -> None:
        self.provider_name = "yfinance"

    def get_provider_health(self) -> ProviderHealth:
        return ProviderHealth(
            provider=self.provider_name,
            status="ok",
            supports_intraday=True,
            supports_markets=("US", "IN"),
            notes=("bootstrap adapter", "provider-swappable"),
        )

    def fetch_bars(self, request: MarketDataRequest) -> list[Bar]:
        interval = f"{int(request.timeframe_min)}m"
        lookback_days = min(max(int(request.lookback_days), 5), 59)
        profile = get_market_profile(request.symbol, timeframe_min=request.timeframe_min)
        ticker = yf.Ticker(request.symbol)
        frame = ticker.history(
            period=f"{lookback_days}d",
            interval=interval,
            auto_adjust=False,
            actions=False,
            prepost=False,
        )
        if frame is None or frame.empty:
            return []
        return self._normalize_frame(frame, request, profile)

    def _normalize_frame(
        self,
        frame: pd.DataFrame,
        request: MarketDataRequest,
        profile,
    ) -> list[Bar]:
        normalized = frame.copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            normalized.columns = normalized.columns.get_level_values(0)
        normalized = normalized.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "VWAP": "vwap",
            }
        )
        missing = {"open", "high", "low", "close", "volume"} - set(normalized.columns)
        if missing:
            return []
        normalized = normalized.reset_index()
        ts_col = normalized.columns[0]
        normalized = normalized.rename(columns={ts_col: "timestamp"})
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=False)
        if getattr(normalized["timestamp"].dt, "tz", None) is None:
            normalized["timestamp"] = normalized["timestamp"].dt.tz_localize("UTC")
        normalized["timestamp"] = normalized["timestamp"].dt.tz_convert(market_tz(profile))
        numeric_cols = ["open", "high", "low", "close", "volume"]
        if "vwap" in normalized.columns:
            numeric_cols.append("vwap")
        for col in numeric_cols:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        normalized = normalized.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
        bars: list[Bar] = []
        for row in normalized.itertuples(index=False):
            timestamp = getattr(row, "timestamp")
            session_date = normalize_session_date(timestamp, profile)
            local_time = timestamp.timetz().replace(tzinfo=None)
            is_regular = profile.regular_open <= local_time < profile.regular_close
            bars.append(
                Bar(
                    symbol=request.symbol,
                    market=request.market,
                    exchange=request.exchange,
                    timezone=request.timezone,
                    timestamp=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp,
                    timeframe_min=request.timeframe_min,
                    open=float(getattr(row, "open")),
                    high=float(getattr(row, "high")),
                    low=float(getattr(row, "low")),
                    close=float(getattr(row, "close")),
                    volume=float(getattr(row, "volume")),
                    currency=request.currency,
                    source=request.source,
                    is_regular_session=bool(is_regular),
                    session_date=session_date,
                    vwap=float(getattr(row, "vwap")) if hasattr(row, "vwap") and pd.notna(getattr(row, "vwap")) else None,
                    trade_count=None,
                )
            )
        return bars


def bars_to_frame(bars: Iterable[Bar]) -> pd.DataFrame:
    rows = [
        {
            "symbol": bar.symbol,
            "market": bar.market,
            "exchange": bar.exchange,
            "timezone": bar.timezone,
            "timestamp": bar.timestamp,
            "timeframe_min": bar.timeframe_min,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "currency": bar.currency,
            "source": bar.source,
            "is_regular_session": bar.is_regular_session,
            "session_date": bar.session_date,
            "vwap": bar.vwap,
            "trade_count": bar.trade_count,
        }
        for bar in bars
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def latest_regular_close(bars: Iterable[Bar]) -> tuple[float | None, datetime | None]:
    regular = [bar for bar in bars if bar.is_regular_session]
    if not regular:
        return None, None
    latest = max(regular, key=lambda bar: bar.timestamp)
    return latest.close, latest.timestamp

