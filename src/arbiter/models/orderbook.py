"""Normalized bid books, derived asks, and explicit freshness state."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BookStatus(StrEnum):
    """Whether a reconstructed book may be used by executable solving."""

    FRESH = "fresh"
    STALE = "stale"
    RESYNC_REQUIRED = "resync_required"


class OrderBookUpdateError(ValueError):
    """Raised when an incremental update cannot preserve a trustworthy local book."""


class PriceLevel(BaseModel):
    """One positive bid or derived-ask level."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    price: Decimal = Field(ge=0, le=1)
    quantity: Decimal = Field(gt=0)


class OrderBook(BaseModel):
    """YES/NO bids in best-first order, with asks derived only by complement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str = Field(min_length=1)
    yes_bids: tuple[PriceLevel, ...] = ()
    no_bids: tuple[PriceLevel, ...] = ()
    sequence: int | None = None
    exchange_timestamp: datetime | None = None
    local_timestamp: datetime
    status: BookStatus = BookStatus.FRESH
    status_reason: str | None = None
    raw: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_book(self) -> OrderBook:
        if self.local_timestamp.tzinfo is None or self.local_timestamp.utcoffset() is None:
            raise ValueError("order-book local timestamp must be timezone-aware")
        if self.exchange_timestamp is not None and (
            self.exchange_timestamp.tzinfo is None or self.exchange_timestamp.utcoffset() is None
        ):
            raise ValueError("order-book exchange timestamp must be timezone-aware")
        for label, levels in (("YES", self.yes_bids), ("NO", self.no_bids)):
            prices = tuple(level.price for level in levels)
            if prices != tuple(sorted(prices, reverse=True)):
                raise ValueError(f"{label} bids must be sorted by descending price")
            if len(set(prices)) != len(prices):
                raise ValueError(f"{label} bid prices must be unique")
        if self.status is BookStatus.FRESH and self.status_reason is not None:
            raise ValueError("a fresh book cannot have a stale/resync reason")
        if self.status is not BookStatus.FRESH and not self.status_reason:
            raise ValueError("a non-fresh book requires a status reason")
        return self

    @property
    def yes_asks(self) -> tuple[PriceLevel, ...]:
        """Derive all YES asks from real NO bids, preserving exact quantity."""

        return tuple(
            PriceLevel(price=Decimal("1") - level.price, quantity=level.quantity)
            for level in self.no_bids
        )

    @property
    def no_asks(self) -> tuple[PriceLevel, ...]:
        """Derive all NO asks from real YES bids, preserving exact quantity."""

        return tuple(
            PriceLevel(price=Decimal("1") - level.price, quantity=level.quantity)
            for level in self.yes_bids
        )

    def best_yes_bid(self) -> PriceLevel | None:
        return self.yes_bids[0] if self.yes_bids else None

    def best_no_bid(self) -> PriceLevel | None:
        return self.no_bids[0] if self.no_bids else None

    def best_yes_ask(self) -> PriceLevel | None:
        asks = self.yes_asks
        return asks[0] if asks else None

    def best_no_ask(self) -> PriceLevel | None:
        asks = self.no_asks
        return asks[0] if asks else None

    def is_fresh(self, *, as_of: datetime, stale_after: timedelta) -> bool:
        """Evaluate freshness using an explicit clock suitable for live or replay."""

        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("freshness as_of must be timezone-aware")
        age = as_of - self.local_timestamp
        return (
            self.status is BookStatus.FRESH
            and stale_after >= timedelta(0)
            and timedelta(0) <= age <= stale_after
        )

    def with_status(self, status: BookStatus, *, reason: str | None = None) -> OrderBook:
        """Return a validated status transition without mutating the prior book."""

        values = self.model_dump()
        values.update(status=status, status_reason=reason)
        return OrderBook.model_validate(values)

    def apply_delta(
        self,
        *,
        side: Literal["yes", "no"],
        price: Decimal,
        quantity_delta: Decimal,
        sequence: int,
        local_timestamp: datetime,
        exchange_timestamp: datetime | None,
    ) -> OrderBook:
        """Apply one already-normalized delta and return a new exact book.

        Sequence continuity belongs to the WebSocket subscription tracker. This method owns
        only market-local depth invariants and therefore refuses updates to a non-fresh book.
        """

        if self.status is not BookStatus.FRESH:
            raise OrderBookUpdateError("cannot apply a delta to a non-fresh order book")
        if sequence < 1:
            raise OrderBookUpdateError("delta sequence must be positive")
        if local_timestamp.tzinfo is None or local_timestamp.utcoffset() is None:
            raise OrderBookUpdateError("delta local timestamp must be timezone-aware")
        if exchange_timestamp is not None and (
            exchange_timestamp.tzinfo is None or exchange_timestamp.utcoffset() is None
        ):
            raise OrderBookUpdateError("delta exchange timestamp must be timezone-aware")
        if not price.is_finite() or not Decimal("0") <= price <= Decimal("1"):
            raise OrderBookUpdateError("delta price must be finite and between zero and one")
        if not quantity_delta.is_finite() or quantity_delta == 0:
            raise OrderBookUpdateError("delta quantity must be finite and nonzero")

        levels = self.yes_bids if side == "yes" else self.no_bids
        by_price = {level.price: level.quantity for level in levels}
        updated_quantity = by_price.get(price, Decimal("0")) + quantity_delta
        if updated_quantity < 0:
            raise OrderBookUpdateError("delta would create negative displayed depth")
        if updated_quantity == 0:
            by_price.pop(price, None)
        else:
            by_price[price] = updated_quantity
        updated_levels = tuple(
            PriceLevel(price=level_price, quantity=quantity)
            for level_price, quantity in sorted(by_price.items(), reverse=True)
        )

        values = self.model_dump()
        values.update(
            yes_bids=updated_levels if side == "yes" else self.yes_bids,
            no_bids=updated_levels if side == "no" else self.no_bids,
            sequence=sequence,
            exchange_timestamp=exchange_timestamp,
            local_timestamp=local_timestamp,
            raw=None,
        )
        return OrderBook.model_validate(values)
