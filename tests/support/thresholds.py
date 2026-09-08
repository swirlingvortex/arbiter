"""Structured threshold metadata factories for deterministic discovery tests."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from arbiter.models.event import Event
from arbiter.models.market import Market
from arbiter.models.series import Series, SettlementSource

TIMESTAMP = datetime(2026, 9, 3, 12, tzinfo=UTC)


def make_threshold_market(
    ticker: str,
    threshold: Decimal,
    *,
    direction: str = "greater",
    event_ticker: str = "EVENT",
    series_ticker: str = "SERIES",
    occurrence_datetime: datetime | None = TIMESTAMP,
    close_time: datetime | None = TIMESTAMP,
    expected_expiration_time: datetime | None = TIMESTAMP,
    expiration_time: datetime | None = TIMESTAMP,
    rules_suffix: str = "The source is final.",
    custom_strike: dict[str, object] | None = None,
    both_bounds: bool = False,
) -> Market:
    """Create a fully compatible greater/less threshold market by default."""

    floor = threshold if direction == "greater" or both_bounds else None
    cap = threshold if direction == "less" or both_bounds else None
    return Market(
        ticker=ticker,
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        title=f"Value {direction} {threshold}",
        status="active",
        close_time=close_time,
        expiration_time=expiration_time,
        expected_expiration_time=expected_expiration_time,
        occurrence_datetime=occurrence_datetime,
        strike_type=direction,
        floor_strike=floor,
        cap_strike=cap,
        functional_strike=direction,
        custom_strike=custom_strike,
        rules_primary=f"Resolves YES when the value is {direction} {threshold}.",
        rules_secondary=rules_suffix,
        raw={"ticker": ticker},
    )


def make_event(
    *,
    ticker: str = "EVENT",
    series_ticker: str = "SERIES",
    markets: tuple[str, ...] = (),
    mutually_exclusive: bool | None = None,
) -> Event:
    return Event(
        ticker=ticker,
        series_ticker=series_ticker,
        title=ticker,
        mutually_exclusive=mutually_exclusive,
        market_tickers=markets,
        raw={"event_ticker": ticker},
    )


def make_series(
    *,
    ticker: str = "SERIES",
    source_name: str = "Authority",
    source_url: str = "https://example.invalid/source",
) -> Series:
    return Series(
        ticker=ticker,
        title=ticker,
        settlement_sources=(SettlementSource(name=source_name, url=source_url),),
        raw={"ticker": ticker},
    )
