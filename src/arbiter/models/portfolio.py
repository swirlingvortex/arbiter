"""Exchange-independent executable instrument and solver-result models."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

InstrumentSide = Literal["yes", "no"]
SourceSide = Literal["yes_bid", "no_bid"]
InstrumentBuildStatus = Literal[
    "ok",
    "missing_book",
    "empty_book",
    "stale_book",
    "invalid_book",
]


class Instrument(BaseModel):
    """One bounded buy-side instrument at a specific executable price level."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str = Field(min_length=1)
    side: InstrumentSide
    price: Decimal = Field(ge=0, le=1)
    max_quantity: Decimal = Field(gt=0)
    source_side: SourceSide
    source_price: Decimal = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_complement(self) -> Instrument:
        """Ensure every buy instrument identifies the real complementary bid."""

        expected_source = "no_bid" if self.side == "yes" else "yes_bid"
        if self.source_side != expected_source:
            raise ValueError(f"a {self.side.upper()} buy must originate from {expected_source}")
        if self.price + self.source_price != Decimal("1"):
            raise ValueError("instrument price must exactly complement its source bid")
        return self


class InstrumentAllocation(BaseModel):
    """An exact quantity and cost assigned to one instrument."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instrument: Instrument
    quantity: Decimal = Field(gt=0)
    cost: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def validate_cost_and_depth(self) -> InstrumentAllocation:
        """Make stored allocations internally self-verifying."""

        if self.quantity > self.instrument.max_quantity:
            raise ValueError("allocation exceeds instrument depth")
        if self.cost != self.instrument.price * self.quantity:
            raise ValueError("allocation cost must equal price times quantity")
        return self


class InstrumentBuildResult(BaseModel):
    """Fail-closed outcome of converting component books into executable levels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: InstrumentBuildStatus
    instruments: tuple[Instrument, ...] = ()
    market_ticker: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> InstrumentBuildResult:
        if self.status == "ok":
            if not self.instruments:
                raise ValueError("an ok instrument build requires executable instruments")
            if self.reason is not None or self.market_ticker is not None:
                raise ValueError("an ok instrument build cannot contain a failure reason")
        else:
            if self.instruments:
                raise ValueError("a failed instrument build cannot contain instruments")
            if self.reason is None or self.market_ticker is None:
                raise ValueError("a failed instrument build requires market and reason")
        return self


class SolverResult(BaseModel):
    """A post-verified worst-case-profit solve result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    guaranteed_gross_profit: Decimal
    capital_required: Decimal = Field(ge=0)
    gross_edge: Decimal | None
    quantities: tuple[InstrumentAllocation, ...]
    state_payouts: tuple[Decimal, ...]
    state_profits: tuple[Decimal, ...]
    min_state_profit: Decimal
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_arbitrage(self) -> bool:
        """Whether this result passed independent positive-profit verification."""

        return self.status == "optimal" and self.guaranteed_gross_profit > 0
