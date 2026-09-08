"""Normalized prediction-market metadata."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PriceRange(BaseModel):
    """One inclusive interval and valid price increment advertised by the exchange."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: Decimal = Field(ge=0, le=1)
    end: Decimal = Field(ge=0, le=1)
    step: Decimal = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_interval(self) -> PriceRange:
        if self.end < self.start:
            raise ValueError("price-range end cannot precede its start")
        return self

    def contains(self, price: Decimal) -> bool:
        """Whether ``price`` is an exact tick inside this interval."""

        return self.start <= price <= self.end and (price - self.start) % self.step == 0


class Market(BaseModel):
    """Exchange-neutral market metadata used by relation and order-book logic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str = Field(min_length=1)
    event_ticker: str = Field(min_length=1)
    series_ticker: str | None = None
    market_type: str | None = None

    title: str = Field(min_length=1)
    subtitle: str | None = None
    yes_sub_title: str | None = None
    no_sub_title: str | None = None

    status: str = Field(min_length=1)
    created_time: datetime | None = None
    updated_time: datetime | None = None
    open_time: datetime | None = None
    close_time: datetime | None = None
    expiration_time: datetime | None = None
    latest_expiration_time: datetime | None = None
    expected_expiration_time: datetime | None = None
    settlement_ts: datetime | None = None
    occurrence_datetime: datetime | None = None

    strike_type: str | None = None
    floor_strike: Decimal | None = None
    cap_strike: Decimal | None = None
    functional_strike: str | None = None
    custom_strike: dict[str, Any] | None = None

    rules_primary: str | None = None
    rules_secondary: str | None = None
    early_close_condition: str | None = None
    price_level_structure: str | None = None
    price_ranges: tuple[PriceRange, ...] = ()
    result: str | None = None
    raw: dict[str, Any]

    def accepts_price(self, price: Decimal) -> bool:
        """Validate a price against at least one exchange-advertised tick interval."""

        return any(price_range.contains(price) for price_range in self.price_ranges)
