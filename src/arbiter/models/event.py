"""Normalized event metadata."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EventFeeChange(BaseModel):
    """One scheduled event-level fee-policy override from Kalshi."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    change_id: str = Field(min_length=1)
    event_ticker: str = Field(min_length=1)
    series_ticker: str = Field(min_length=1)
    scheduled_ts: datetime
    fee_type_override: str | None = Field(default=None, min_length=1)
    fee_multiplier_override: Decimal | None = None
    raw: dict[str, Any]

    @field_validator("fee_multiplier_override")
    @classmethod
    def validate_fee_multiplier(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and (not value.is_finite() or value < 0):
            raise ValueError("event fee multiplier override must be finite and nonnegative")
        return value

    @model_validator(mode="after")
    def validate_change(self) -> EventFeeChange:
        if self.scheduled_ts.tzinfo is None or self.scheduled_ts.utcoffset() is None:
            raise ValueError("event fee change scheduled_ts must be timezone-aware")
        if (self.fee_type_override is None) != (self.fee_multiplier_override is None):
            raise ValueError("event fee override values must both be present or both be null")
        return self


class Event(BaseModel):
    """One exchange event and the markets currently associated with it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str = Field(min_length=1)
    series_ticker: str | None = None
    title: str = Field(min_length=1)
    subtitle: str | None = None
    category: str | None = None
    mutually_exclusive: bool | None = None
    available_on_brokers: bool | None = None
    market_tickers: tuple[str, ...] = ()
    last_updated_ts: datetime | None = None
    fee_type_override: str | None = Field(default=None, min_length=1)
    fee_multiplier_override: Decimal | None = None
    fee_changes: tuple[EventFeeChange, ...] = ()
    raw: dict[str, Any]

    @field_validator("fee_multiplier_override")
    @classmethod
    def validate_fee_multiplier(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and (not value.is_finite() or value < 0):
            raise ValueError("event fee multiplier override must be finite and nonnegative")
        return value

    @model_validator(mode="after")
    def validate_overrides(self) -> Event:
        if (self.fee_type_override is None) != (self.fee_multiplier_override is None):
            raise ValueError("event fee override values must both be present or both be null")
        return self
