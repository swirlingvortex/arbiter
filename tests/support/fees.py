"""Concise dated Event/Series/fee-policy fixtures for Milestone 6 tests."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from arbiter.models.event import Event, EventFeeChange
from arbiter.models.opportunity import EffectiveFeePolicy
from arbiter.models.series import Series
from arbiter.solver.fees import KALSHI_FEE_POLICY_VERSION

POLICY_TIME = datetime(2026, 9, 3, 12, tzinfo=UTC)


def series(
    *,
    fee_type: str | None = "quadratic",
    multiplier: Decimal | None = Decimal("1"),
) -> Series:
    return Series(
        ticker="KXARBITER",
        title="Arbiter fee fixture",
        fee_type=fee_type,
        fee_multiplier=multiplier,
        raw={},
    )


def fee_change(
    change_id: str,
    scheduled_ts: datetime,
    *,
    fee_type: str | None,
    multiplier: Decimal | None,
) -> EventFeeChange:
    return EventFeeChange(
        change_id=change_id,
        event_ticker="KXARBITER-EVENT",
        series_ticker="KXARBITER",
        fee_type_override=fee_type,
        fee_multiplier_override=multiplier,
        scheduled_ts=scheduled_ts,
        raw={},
    )


def event(
    *,
    changes: tuple[EventFeeChange, ...] = (),
    fee_type_override: str | None = None,
    fee_multiplier_override: Decimal | None = None,
) -> Event:
    return Event(
        ticker="KXARBITER-EVENT",
        series_ticker="KXARBITER",
        title="Arbiter fee event",
        fee_type_override=fee_type_override,
        fee_multiplier_override=fee_multiplier_override,
        fee_changes=changes,
        raw={},
    )


def policy(
    ticker: str,
    *,
    fee_type: str = "quadratic",
    multiplier: Decimal = Decimal("1"),
) -> EffectiveFeePolicy:
    return EffectiveFeePolicy(
        market_ticker=ticker,
        event_ticker="KXARBITER-EVENT",
        series_ticker="KXARBITER",
        fee_type=fee_type,
        fee_multiplier=multiplier,
        source="series_default",
        policy_version=KALSHI_FEE_POLICY_VERSION,
    )
