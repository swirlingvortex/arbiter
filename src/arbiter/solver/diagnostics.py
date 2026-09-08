"""Exact solver verification and human-readable economic diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from arbiter.models.portfolio import Instrument
from arbiter.models.world import WorldSet


class PortfolioVerificationError(ValueError):
    """Raised when numerical output violates exact domain constraints."""


@dataclass(frozen=True, slots=True)
class PortfolioVerification:
    """Economics recomputed with Decimal and binary world values outside SciPy."""

    capital_required: Decimal
    state_payouts: tuple[Decimal, ...]
    state_profits: tuple[Decimal, ...]
    min_state_profit: Decimal


def recompute_portfolio(
    world_set: WorldSet,
    instruments: Sequence[Instrument],
    quantities: Sequence[Decimal],
) -> PortfolioVerification:
    """Independently recompute cost, payouts, and profits using exact domain values."""

    if world_set.states.shape[0] == 0:
        raise PortfolioVerificationError("a portfolio cannot be verified without feasible worlds")
    if len(instruments) != len(quantities):
        raise PortfolioVerificationError("instrument and quantity counts differ")

    ticker_columns = {ticker: index for index, ticker in enumerate(world_set.tickers)}
    for instrument, quantity in zip(instruments, quantities, strict=True):
        if instrument.ticker not in ticker_columns:
            raise PortfolioVerificationError(
                f"instrument ticker {instrument.ticker!r} is absent from the world set"
            )
        if not quantity.is_finite():
            raise PortfolioVerificationError("instrument quantity must be finite")
        if quantity < 0:
            raise PortfolioVerificationError("instrument quantity cannot be negative")
        if quantity > instrument.max_quantity:
            raise PortfolioVerificationError("instrument quantity exceeds displayed depth")

    capital = sum(
        (
            instrument.price * quantity
            for instrument, quantity in zip(instruments, quantities, strict=True)
        ),
        start=Decimal("0"),
    )
    payouts: list[Decimal] = []
    for state in world_set.states:
        payout = Decimal("0")
        for instrument, quantity in zip(instruments, quantities, strict=True):
            outcome = int(state[ticker_columns[instrument.ticker]])
            pays = outcome if instrument.side == "yes" else 1 - outcome
            payout += Decimal(pays) * quantity
        payouts.append(payout)
    profits = tuple(payout - capital for payout in payouts)
    return PortfolioVerification(
        capital_required=capital,
        state_payouts=tuple(payouts),
        state_profits=profits,
        min_state_profit=min(profits),
    )


def decimal_text(value: Decimal, *, places: int = 2) -> str:
    """Format a finite Decimal with a fixed number of display places."""

    quantum = Decimal(1).scaleb(-places)
    return f"{value.quantize(quantum):.{places}f}"
