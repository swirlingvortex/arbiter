"""Normalized recurring-series and settlement-source metadata."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SettlementSource(BaseModel):
    """Authoritative source named in a series settlement definition."""

    model_config = ConfigDict(extra="allow", frozen=True)

    name: str = ""
    url: str = ""


class Series(BaseModel):
    """Recurring event template including current fee and settlement metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str = Field(min_length=1)
    title: str = Field(min_length=1)
    frequency: str | None = None
    category: str | None = None
    tags: tuple[str, ...] = ()
    fee_type: str | None = None
    fee_multiplier: Decimal | None = None
    settlement_sources: tuple[SettlementSource, ...] = ()
    contract_url: str | None = None
    contract_terms_url: str | None = None
    last_updated_ts: datetime | None = None
    raw: dict[str, Any]

    @field_validator("fee_multiplier")
    @classmethod
    def validate_fee_multiplier(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and (not value.is_finite() or value < 0):
            raise ValueError("series fee multiplier must be finite and nonnegative")
        return value
