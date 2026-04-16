from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .contracts import MarketProfile


DEFAULT_TIMEFRAME_MIN = 15
STRATEGY_FAMILY = "orb_vwap_continuation"
INDIA_INDEX_EXCHANGES = {
    "^NSEI": "NSE",
    "^NSEBANK": "NSE",
    "^CNXIT": "NSE",
    "^BSESN": "BSE",
}


def resolve_market(symbol: str) -> tuple[str, str]:
    normalized = symbol.strip().upper()
    if normalized in INDIA_INDEX_EXCHANGES:
        return "IN", INDIA_INDEX_EXCHANGES[normalized]
    if normalized.endswith(".NS"):
        return "IN", "NSE"
    if normalized.endswith(".BO"):
        return "IN", "BSE"
    return "US", "NASDAQ"


def get_market_profile(symbol: str, timeframe_min: int = DEFAULT_TIMEFRAME_MIN) -> MarketProfile:
    market, exchange = resolve_market(symbol)
    if market == "IN":
        return MarketProfile(
            market="IN",
            exchange=exchange,
            timezone="Asia/Kolkata",
            currency="INR",
            regular_open=time(9, 15),
            regular_close=time(15, 30),
            calendar_id="NSE",
            symbol_rules={"suffixes": [".NS", ".BO"], "default_exchange": "NSE"},
            entry_window_policy={
                "mode": "single_window",
                "start": "09:45",
                "end": "10:45",
                "opening_range_bars": 2,
            },
            forced_exit_policy={"mode": "session_close", "time": "15:15"},
            bar_expectation_policy={"timeframe_min": timeframe_min, "bars_per_session": 25},
            holiday_policy={"mode": "provider_filtered_weekday_sessions"},
        )
    return MarketProfile(
        market="US",
        exchange=exchange,
        timezone="America/New_York",
        currency="USD",
        regular_open=time(9, 30),
        regular_close=time(16, 0),
        calendar_id="NYSE",
        symbol_rules={"suffixes": [], "default_exchange": "NASDAQ"},
        entry_window_policy={
            "mode": "single_window",
            "start": "10:00",
            "end": "11:00",
            "opening_range_bars": 2,
        },
        forced_exit_policy={"mode": "session_close", "time": "15:45"},
        bar_expectation_policy={"timeframe_min": timeframe_min, "bars_per_session": 26},
        holiday_policy={"mode": "provider_filtered_weekday_sessions"},
    )


def market_tz(profile: MarketProfile) -> ZoneInfo:
    return ZoneInfo(profile.timezone)


def parse_hhmm(value: str) -> time:
    hour, minute = value.split(":", 1)
    return time(int(hour), int(minute))


def combine_local(session_date: date, session_time: time, profile: MarketProfile) -> datetime:
    return datetime.combine(session_date, session_time, tzinfo=market_tz(profile))


def session_bar_starts(session_date: date, profile: MarketProfile, timeframe_min: int) -> list[datetime]:
    current = combine_local(session_date, profile.regular_open, profile)
    last = combine_local(session_date, profile.regular_close, profile) - timedelta(minutes=timeframe_min)
    output: list[datetime] = []
    while current <= last:
        output.append(current)
        current = current + timedelta(minutes=timeframe_min)
    return output


def normalize_session_date(timestamp: datetime, profile: MarketProfile) -> date:
    return timestamp.astimezone(market_tz(profile)).date()
