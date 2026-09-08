"""Canonical ordering helpers for exchange-independent instruments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import ROUND_FLOOR, Decimal

from arbiter.models.orderbook import BookStatus, OrderBook
from arbiter.models.portfolio import Instrument, InstrumentBuildResult, InstrumentBuildStatus
from arbiter.models.relation import LogicalComponent

_SIDE_RANK = {"yes": 0, "no": 1}


def instrument_sort_key(instrument: Instrument) -> tuple[str, int, Decimal, Decimal, Decimal]:
    """Return the stable order shared by optimization, diagnostics, and persistence."""

    return (
        instrument.ticker,
        _SIDE_RANK[instrument.side],
        instrument.price,
        instrument.source_price,
        instrument.max_quantity,
    )


def sort_instruments(instruments: Iterable[Instrument]) -> tuple[Instrument, ...]:
    """Materialize instruments in canonical deterministic order."""

    return tuple(sorted(instruments, key=instrument_sort_key))


def build_executable_instruments(
    component: LogicalComponent,
    orderbooks: Mapping[str, OrderBook],
    *,
    as_of: datetime,
    stale_after: timedelta,
) -> InstrumentBuildResult:
    """Create one bounded complementary buy instrument per real component bid level."""

    if stale_after < timedelta(0):
        raise ValueError("stale_after must be nonnegative")
    instruments: list[Instrument] = []
    for ticker in component.market_tickers:
        book = orderbooks.get(ticker)
        if book is None:
            return _build_failure("missing_book", ticker, "no order book is available")
        if book.ticker != ticker:
            return _build_failure(
                "invalid_book",
                ticker,
                f"book payload identifies ticker {book.ticker!r}",
            )
        if book.status is BookStatus.RESYNC_REQUIRED:
            return _build_failure(
                "invalid_book",
                ticker,
                f"book requires resynchronization: {book.status_reason}",
            )
        if book.status is BookStatus.STALE:
            return _build_failure(
                "stale_book",
                ticker,
                f"book is explicitly stale: {book.status_reason}",
            )
        if book.local_timestamp > as_of:
            return _build_failure(
                "invalid_book",
                ticker,
                "book timestamp is later than the evaluation time",
            )
        if not book.is_fresh(as_of=as_of, stale_after=stale_after):
            return _build_failure(
                "stale_book",
                ticker,
                "book age exceeds the configured freshness window",
            )
        if not book.yes_bids and not book.no_bids:
            return _build_failure(
                "empty_book",
                ticker,
                "book has no positive displayed liquidity on either bid side",
            )
        for level in book.no_bids:
            instruments.append(
                Instrument(
                    ticker=ticker,
                    side="yes",
                    price=Decimal("1") - level.price,
                    max_quantity=level.quantity,
                    source_side="no_bid",
                    source_price=level.price,
                )
            )
        for level in book.yes_bids:
            instruments.append(
                Instrument(
                    ticker=ticker,
                    side="no",
                    price=Decimal("1") - level.price,
                    max_quantity=level.quantity,
                    source_side="yes_bid",
                    source_price=level.price,
                )
            )
    return InstrumentBuildResult(status="ok", instruments=sort_instruments(instruments))


def _build_failure(
    status: InstrumentBuildStatus,
    ticker: str,
    reason: str,
) -> InstrumentBuildResult:
    return InstrumentBuildResult(
        status=status,
        market_ticker=ticker,
        reason=reason,
    )


def snap_quantity_down(
    quantity: Decimal,
    *,
    quantum: Decimal = Decimal("0.01"),
    numeric_tolerance: Decimal,
) -> Decimal:
    """Normalize tiny numerical noise, then floor a quantity toward zero to its grid."""

    if not quantity.is_finite():
        raise ValueError("quantity must be finite")
    if not quantum.is_finite() or quantum <= 0:
        raise ValueError("quantity quantum must be finite and positive")
    if not numeric_tolerance.is_finite() or numeric_tolerance < 0:
        raise ValueError("numeric tolerance must be finite and nonnegative")
    if quantity < -numeric_tolerance:
        raise ValueError("quantity cannot be negative beyond numerical tolerance")
    if quantity <= 0:
        return Decimal("0")

    units = quantity / quantum
    nearest_units = units.to_integral_value()
    nearest = nearest_units * quantum
    if abs(quantity - nearest) <= numeric_tolerance:
        quantity = nearest
    floored_units = (quantity / quantum).to_integral_value(rounding=ROUND_FLOOR)
    return floored_units * quantum


def price_priority_violation(
    instruments: Sequence[Instrument],
    quantities: Sequence[Decimal],
) -> str | None:
    """Return a reason when a worse same-payoff level is used before a cheaper one."""

    if len(instruments) != len(quantities):
        return "instrument and quantity counts differ"
    for index, (instrument, quantity) in enumerate(zip(instruments, quantities, strict=True)):
        if quantity <= 0:
            continue
        for better, better_quantity in zip(
            instruments[:index],
            quantities[:index],
            strict=True,
        ):
            if (
                better.ticker == instrument.ticker
                and better.side == instrument.side
                and better.price < instrument.price
                and better_quantity != better.max_quantity
            ):
                return (
                    f"used {instrument.ticker} {instrument.side} at {instrument.price} "
                    f"before exhausting {better.price}"
                )
    return None
