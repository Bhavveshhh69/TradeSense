from __future__ import annotations

from datetime import datetime, timedelta

from .contracts import Bar, DataQualityReport, MarketProfile
from .market import market_tz, session_bar_starts


class DataQualityValidator:
    def validate(
        self,
        bars: list[Bar],
        market_profile: MarketProfile,
        timeframe_min: int,
    ) -> DataQualityReport:
        if not bars:
            return DataQualityReport(
                missing_bar_count=0,
                expected_bar_count=0,
                completeness_score=0.0,
                stale_data=True,
                timezone_valid=False,
                session_valid=False,
                usable_for_live=False,
                usable_for_backtest=False,
                warnings=("no bars returned by provider",),
            )

        regular = [bar for bar in bars if bar.is_regular_session]
        session_dates = sorted({bar.session_date for bar in regular})
        expected = 0
        missing = 0
        for session_date in session_dates:
            expected_starts = session_bar_starts(session_date, market_profile, timeframe_min)
            expected += len(expected_starts)
            actual_starts = {
                bar.timestamp.astimezone(market_tz(market_profile)).replace(second=0, microsecond=0)
                for bar in regular
                if bar.session_date == session_date
            }
            missing += max(0, len(expected_starts) - len(actual_starts))

        completeness = 0.0 if expected == 0 else max(0.0, min(1.0, (expected - missing) / expected))
        timezone_valid = all(bar.timezone == market_profile.timezone for bar in bars)
        session_valid = all(
            (not bar.is_regular_session)
            or (
                market_profile.regular_open
                <= bar.timestamp.astimezone(market_tz(market_profile)).timetz().replace(tzinfo=None)
                < market_profile.regular_close
            )
            for bar in bars
        )
        latest_timestamp = max(bar.timestamp for bar in bars)
        now_local = datetime.now(tz=market_tz(market_profile))
        stale = latest_timestamp.astimezone(market_tz(market_profile)) < now_local - timedelta(days=3)
        warnings: list[str] = []
        if missing:
            warnings.append(f"missing {missing} regular-session bars")
        if stale:
            warnings.append("latest bar is stale")
        if not timezone_valid:
            warnings.append("bar timezone does not match market profile")
        if not session_valid:
            warnings.append("regular-session validation failed")
        return DataQualityReport(
            missing_bar_count=missing,
            expected_bar_count=expected,
            completeness_score=round(completeness, 4),
            stale_data=stale,
            timezone_valid=timezone_valid,
            session_valid=session_valid,
            usable_for_live=bool(expected and completeness >= 0.9 and timezone_valid and session_valid),
            usable_for_backtest=bool(expected and completeness >= 0.8 and timezone_valid and session_valid),
            warnings=tuple(warnings),
        )

