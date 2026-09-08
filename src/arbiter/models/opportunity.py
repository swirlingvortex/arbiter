"""Fee-policy snapshots and staged structural-arbitrage opportunities."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arbiter.models.orderbook import OrderBook
from arbiter.models.portfolio import InstrumentAllocation, SourceSide
from arbiter.models.relation import RelationType

InstrumentSide = Literal["yes", "no"]
LiquidityRole = Literal["taker", "maker"]
FeePolicySource = Literal["event_override", "series_default", "documented_standard"]
PaperExecutionStatus = Literal["not_evaluated", "survived", "failed"]


class OpportunityStage(StrEnum):
    """Evidence stages; Stage 0 is independent and Stages 1 through 3 are cumulative."""

    LOGICAL = "stage_0"
    GROSS_EXECUTABLE = "stage_1"
    NET_EXECUTABLE = "stage_2"
    PAPER_SURVIVED = "stage_3"


class FeeValidationStatus(StrEnum):
    """Whether deterministic entry fees were applied successfully."""

    NOT_EVALUATED = "not_evaluated"
    APPLIED = "applied"
    UNSUPPORTED = "unsupported_fee_model"


class OpportunityTransition(StrEnum):
    """Deterministic states emitted by the opportunity lifecycle."""

    NOT_PRESENT = "not_present"
    OPEN = "open"
    UPDATED = "updated"
    CLOSED = "closed"
    RIGHT_CENSORED = "right_censored"


class MarketResearchContext(BaseModel):
    """Immutable market dimensions captured from metadata at scan time."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ticker: str = Field(min_length=1)
    event_ticker: str = Field(min_length=1)
    category: str | None = Field(default=None, min_length=1)
    settlement_at: datetime | None = None

    @field_validator("ticker", "event_ticker", "category")
    @classmethod
    def require_nonblank_context_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("market research context text cannot be blank")
        return value

    @field_validator("settlement_at")
    @classmethod
    def require_aware_settlement_time(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("market research settlement_at must be timezone-aware")
        return value


class EffectiveFeePolicy(BaseModel):
    """The dated fee terms actually used for one event/series decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    market_ticker: str | None = None
    event_ticker: str = Field(min_length=1)
    series_ticker: str = Field(min_length=1)
    fee_type: str = Field(min_length=1)
    fee_multiplier: Decimal = Field(ge=0)
    source: FeePolicySource
    effective_at: datetime | None = None
    change_id: str | None = None
    next_scheduled_at: datetime | None = None
    next_change_id: str | None = None
    policy_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_policy_times(self) -> EffectiveFeePolicy:
        for label, value in (
            ("effective_at", self.effective_at),
            ("next_scheduled_at", self.next_scheduled_at),
        ):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{label} must be timezone-aware")
        if (self.change_id is None) != (self.effective_at is None):
            raise ValueError("an event change ID and effective time must be supplied together")
        if (self.next_change_id is None) != (self.next_scheduled_at is None):
            raise ValueError("a next change ID and scheduled time must be supplied together")
        return self


class FeeFill(BaseModel):
    """One displayed-liquidity fill belonging to a synthetic order leg."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    order_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    side: InstrumentSide
    price: Decimal = Field(ge=0, le=1)
    quantity: Decimal = Field(gt=0)
    liquidity_role: LiquidityRole = "taker"

    @property
    def signed_revenue(self) -> Decimal:
        """Return the exchange convention for a purchase's balance revenue."""

        return -(self.price * self.quantity)


class FeeFillQuote(BaseModel):
    """Exact model, balance-alignment, and ledger result for one fill."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fill: FeeFill
    signed_revenue: Decimal
    model_fee: Decimal = Field(ge=0)
    trade_fee: Decimal = Field(ge=0)
    aligned_change_before_rebate: Decimal
    rounding_fee: Decimal = Field(ge=0)
    rebate: Decimal = Field(ge=0)
    net_fee: Decimal = Field(ge=0)
    balance_change: Decimal
    accumulator_before: Decimal = Field(ge=0)
    accumulator_after: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def validate_accounting(self) -> FeeFillQuote:
        if self.signed_revenue != self.fill.signed_revenue:
            raise ValueError("signed revenue must equal negative purchase cost")
        if self.net_fee != self.trade_fee + self.rounding_fee - self.rebate:
            raise ValueError("net fee components do not balance")
        if self.balance_change != self.signed_revenue - self.net_fee:
            raise ValueError("balance change must include the exact net fee")
        if self.accumulator_after != (self.accumulator_before + self.rounding_fee - self.rebate):
            raise ValueError("rounding accumulator components do not balance")
        if self.aligned_change_before_rebate != (
            self.signed_revenue - self.trade_fee - self.rounding_fee
        ):
            raise ValueError("pre-rebate aligned balance change is inconsistent")
        return self


class FeeQuote(BaseModel):
    """Aggregate fee quote for every fill belonging to one order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    order_id: str = Field(min_length=1)
    fills: tuple[FeeFillQuote, ...]
    model_fee: Decimal = Field(ge=0)
    trade_fee: Decimal = Field(ge=0)
    rounding_fee: Decimal = Field(ge=0)
    rebate: Decimal = Field(ge=0)
    net_fee: Decimal = Field(ge=0)
    balance_change: Decimal
    ending_accumulator: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def validate_totals(self) -> FeeQuote:
        if not self.fills:
            raise ValueError("an order fee quote requires at least one fill")
        if any(item.fill.order_id != self.order_id for item in self.fills):
            raise ValueError("every quoted fill must belong to the quoted order")
        expected = {
            "model_fee": sum((item.model_fee for item in self.fills), Decimal("0")),
            "trade_fee": sum((item.trade_fee for item in self.fills), Decimal("0")),
            "rounding_fee": sum((item.rounding_fee for item in self.fills), Decimal("0")),
            "rebate": sum((item.rebate for item in self.fills), Decimal("0")),
            "net_fee": sum((item.net_fee for item in self.fills), Decimal("0")),
            "balance_change": sum((item.balance_change for item in self.fills), Decimal("0")),
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"aggregate {field} does not equal its fill total")
        if self.ending_accumulator != self.fills[-1].accumulator_after:
            raise ValueError("ending accumulator must equal the final fill state")
        return self


class Opportunity(BaseModel):
    """One evidence-graded opportunity without any real-order execution semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: OpportunityStage
    quantities: tuple[InstrumentAllocation, ...] = ()
    capital_required: Decimal | None = Field(default=None, ge=0)
    gross_profit: Decimal | None = None
    gross_edge: Decimal | None = None
    gross_state_profits: tuple[Decimal, ...] | None = None
    fee_status: FeeValidationStatus = FeeValidationStatus.NOT_EVALUATED
    fees: Decimal | None = Field(default=None, ge=0)
    net_profit: Decimal | None = None
    net_edge: Decimal | None = None
    net_state_profits: tuple[Decimal, ...] | None = None
    fee_policies: tuple[EffectiveFeePolicy, ...] = ()
    fee_quotes: tuple[FeeQuote, ...] = ()
    reason: str | None = None
    paper_execution_status: PaperExecutionStatus = "not_evaluated"

    @model_validator(mode="after")
    def validate_stage_evidence(self) -> Opportunity:
        gross_fields = (self.capital_required, self.gross_profit, self.gross_state_profits)
        if self.stage is OpportunityStage.LOGICAL:
            if self.quantities:
                raise ValueError("Stage 0 cannot contain executable portfolio legs")
            if self.gross_edge is not None or any(value is not None for value in gross_fields):
                raise ValueError("Stage 0 cannot claim executable gross economics")
            if any(
                value is not None
                for value in (
                    self.fees,
                    self.net_profit,
                    self.net_edge,
                    self.net_state_profits,
                )
            ):
                raise ValueError("Stage 0 cannot claim fee-adjusted economics")
            if self.fee_status is not FeeValidationStatus.NOT_EVALUATED:
                raise ValueError("Stage 0 fees cannot be evaluated")
            if self.fee_policies or self.fee_quotes:
                raise ValueError("Stage 0 cannot contain executable fee evidence")
            if self.paper_execution_status != "not_evaluated":
                raise ValueError("Stage 0 cannot contain paper execution evidence")
            return self

        if any(value is None for value in gross_fields):
            raise ValueError("Stage 1 and above require complete gross economics")
        if not self.quantities:
            raise ValueError("Stage 1 and above require executable portfolio legs")
        assert self.gross_state_profits is not None
        assert self.gross_profit is not None
        assert self.capital_required is not None
        if not self.gross_state_profits or self.gross_profit != min(self.gross_state_profits):
            raise ValueError("gross profit must equal the minimum gross state profit")
        if self.gross_profit <= 0:
            raise ValueError("Stage 1 and above require positive gross profit")
        if self.capital_required > 0:
            if self.gross_edge != self.gross_profit / self.capital_required:
                raise ValueError("gross edge must equal gross profit divided by capital")
        elif self.gross_edge is not None:
            raise ValueError("gross edge is undefined when capital is zero")

        has_net = all(
            value is not None for value in (self.fees, self.net_profit, self.net_state_profits)
        )
        if has_net:
            assert self.fees is not None
            assert self.net_profit is not None
            assert self.net_state_profits is not None
            if not self.net_state_profits or self.net_profit != min(self.net_state_profits):
                raise ValueError("net profit must equal the minimum net state profit")
            if any(
                net != gross - self.fees
                for gross, net in zip(
                    self.gross_state_profits,
                    self.net_state_profits,
                    strict=True,
                )
            ):
                raise ValueError("each net state profit must subtract the same entry fees")
            if self.capital_required > 0:
                if self.net_edge != self.net_profit / self.capital_required:
                    raise ValueError("net edge must equal net profit divided by capital")
            elif self.net_edge is not None:
                raise ValueError("net edge is undefined when capital is zero")
        elif any(
            value is not None
            for value in (self.fees, self.net_profit, self.net_edge, self.net_state_profits)
        ):
            raise ValueError("fee-adjusted economics must be supplied together")

        if has_net and self.fee_status is not FeeValidationStatus.APPLIED:
            raise ValueError("fee-adjusted economics require applied fee evidence")
        if self.fee_status is FeeValidationStatus.APPLIED and not has_net:
            raise ValueError("applied fee evidence requires complete net economics")
        if has_net and self.fee_quotes:
            assert self.fees is not None
            quoted_fees = sum((quote.net_fee for quote in self.fee_quotes), Decimal("0"))
            if self.fees != quoted_fees:
                raise ValueError("fees must equal the exact quoted order-fee total")

        if self.stage in {
            OpportunityStage.NET_EXECUTABLE,
            OpportunityStage.PAPER_SURVIVED,
        }:
            if self.fee_status is not FeeValidationStatus.APPLIED or not has_net:
                raise ValueError("Stage 2 and above require applied fee evidence")
            assert self.net_profit is not None
            if self.net_profit <= 0:
                raise ValueError("Stage 2 and above require positive net profit")
        if self.stage is OpportunityStage.PAPER_SURVIVED:
            if self.paper_execution_status != "survived":
                raise ValueError("Stage 3 requires paper execution survival")
        elif self.paper_execution_status == "survived":
            raise ValueError("paper survival must promote an opportunity to Stage 3")
        if self.fee_status is FeeValidationStatus.UNSUPPORTED and has_net:
            raise ValueError("unsupported fee policies cannot claim net economics")
        return self


class PortfolioLegSnapshot(BaseModel):
    """Immutable persistence projection of one executable portfolio allocation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ticker: str = Field(min_length=1)
    side: InstrumentSide
    price: Decimal = Field(ge=0, le=1)
    quantity: Decimal = Field(gt=0)
    source_side: SourceSide
    source_price: Decimal = Field(ge=0, le=1)
    fee: Decimal | None = Field(default=None, ge=0)

    @field_validator("ticker")
    @classmethod
    def require_nonblank_ticker(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("portfolio-leg ticker cannot be blank")
        return value

    @field_validator("price", "quantity", "source_price", "fee")
    @classmethod
    def require_finite_decimal(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and not value.is_finite():
            raise ValueError("portfolio-leg decimals must be finite")
        return value

    @model_validator(mode="after")
    def validate_source(self) -> PortfolioLegSnapshot:
        expected_source = "no_bid" if self.side == "yes" else "yes_bid"
        if self.source_side != expected_source:
            raise ValueError(f"a {self.side.upper()} leg must originate from {expected_source}")
        if self.price + self.source_price != Decimal("1"):
            raise ValueError("portfolio-leg price must exactly complement its source bid")
        return self

    @classmethod
    def from_allocation(
        cls,
        allocation: InstrumentAllocation,
        *,
        fee: Decimal | None = None,
    ) -> PortfolioLegSnapshot:
        """Project a verified allocation into the exact row persisted by the scanner."""

        instrument = allocation.instrument
        return cls(
            ticker=instrument.ticker,
            side=instrument.side,
            price=instrument.price,
            quantity=allocation.quantity,
            source_side=instrument.source_side,
            source_price=instrument.source_price,
            fee=fee,
        )

    @property
    def signature(self) -> str:
        """Return the canonical signature of this one-leg portfolio."""

        return portfolio_signature((self,))


def _canonical_decimal(value: Decimal | None) -> str | None:
    if value is None:
        return None
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def portfolio_signature(legs: Sequence[PortfolioLegSnapshot]) -> str:
    """Hash portfolio economics independent of input ordering or Decimal formatting."""

    rows = [
        (
            leg.ticker,
            leg.side,
            _canonical_decimal(leg.price),
            _canonical_decimal(leg.quantity),
            leg.source_side,
            _canonical_decimal(leg.source_price),
            _canonical_decimal(leg.fee),
        )
        for leg in legs
    ]
    rows.sort(key=lambda row: json.dumps(row, separators=(",", ":"), ensure_ascii=True))
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode()).hexdigest()


def _require_identity_part(label: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} cannot be blank")


def opportunity_episode_id(*, run_id: str, component_id: str, opened_event_index: int) -> str:
    """Return the stable episode identity for one component opening within one run."""

    _require_identity_part("run_id", run_id)
    _require_identity_part("component_id", component_id)
    if opened_event_index < 0:
        raise ValueError("opened_event_index cannot be negative")
    payload = json.dumps(
        [run_id, component_id, opened_event_index],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"opportunity:{sha256(payload.encode()).hexdigest()}"


def opportunity_observation_id(
    *,
    run_id: str,
    component_id: str,
    event_index: int,
    transition: OpportunityTransition,
    opportunity_id: str | None,
) -> str:
    """Return a retry-safe identity for one component decision at one event index."""

    _require_identity_part("run_id", run_id)
    _require_identity_part("component_id", component_id)
    if opportunity_id is not None:
        _require_identity_part("opportunity_id", opportunity_id)
    if event_index < 0:
        raise ValueError("event_index cannot be negative")
    payload = json.dumps(
        [run_id, component_id, event_index, transition.value, opportunity_id],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"observation:{sha256(payload.encode()).hexdigest()}"


class OpportunityObservation(BaseModel):
    """One deterministic scanner decision and its persistence-ready evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    observation_id: str = Field(min_length=1)
    opportunity_id: str | None = Field(default=None, min_length=1)
    run_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    observed_at: datetime
    event_index: int = Field(ge=0)
    transition: OpportunityTransition
    market_tickers: tuple[str, ...]
    relation_types: tuple[RelationType, ...]
    relation_sources: tuple[str, ...] = ()
    opportunity: Opportunity | None = None
    solver_status: str = Field(min_length=1)
    solver_reason: str | None = Field(default=None, min_length=1)
    solve_duration_ms: Decimal = Field(ge=0)
    num_states: int = Field(ge=0)
    num_instruments: int = Field(ge=0)
    num_legs: int = Field(ge=0)
    market_contexts: tuple[MarketResearchContext, ...] = ()
    portfolio_legs: tuple[PortfolioLegSnapshot, ...] = ()
    portfolio_signature: str | None = Field(default=None, min_length=1)
    close_reason: str | None = Field(default=None, min_length=1)
    censor_reason: str | None = Field(default=None, min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("observation_id", "opportunity_id", "run_id", "component_id")
    @classmethod
    def require_nonblank_identifier(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("opportunity lifecycle identifiers cannot be blank")
        return value

    @field_validator("observed_at")
    @classmethod
    def require_aware_observation_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("opportunity observation time must be timezone-aware")
        return value

    @field_validator("solve_duration_ms")
    @classmethod
    def require_finite_duration(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("solve duration must be finite")
        return value

    @field_validator("solver_status", "solver_reason", "close_reason", "censor_reason")
    @classmethod
    def require_nonblank_reason(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("solver status and lifecycle reasons cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_observation(self) -> OpportunityObservation:
        if self.market_tickers != tuple(sorted(set(self.market_tickers))):
            raise ValueError("observation market tickers must be nonempty and canonically sorted")
        if not self.market_tickers:
            raise ValueError("an opportunity observation requires component markets")
        if self.relation_types != tuple(
            sorted(set(self.relation_types), key=lambda item: item.value)
        ):
            raise ValueError("observation relation types must be unique and canonically sorted")
        if self.relation_sources != tuple(sorted(set(self.relation_sources))) or any(
            not source.strip() for source in self.relation_sources
        ):
            raise ValueError("observation relation sources must be unique and canonically sorted")
        context_tickers = tuple(context.ticker for context in self.market_contexts)
        if context_tickers and context_tickers != self.market_tickers:
            raise ValueError(
                "observation market contexts must match canonically sorted component markets"
            )
        if self.num_legs != len(self.portfolio_legs):
            raise ValueError("num_legs must equal the persisted portfolio-leg count")
        expected_observation_id = opportunity_observation_id(
            run_id=self.run_id,
            component_id=self.component_id,
            event_index=self.event_index,
            transition=self.transition,
            opportunity_id=self.opportunity_id,
        )
        if self.observation_id != expected_observation_id:
            raise ValueError("observation_id does not match its deterministic identity fields")

        if self.transition is OpportunityTransition.NOT_PRESENT:
            if self.opportunity_id is not None or self.opportunity is not None:
                raise ValueError("NOT_PRESENT cannot reference an opportunity episode")
            if self.portfolio_legs or self.portfolio_signature is not None or self.num_legs:
                raise ValueError("NOT_PRESENT cannot contain executable portfolio evidence")
            if self.close_reason is not None or self.censor_reason is not None:
                raise ValueError("NOT_PRESENT is not a terminal episode transition")
            return self

        if self.opportunity_id is None:
            raise ValueError("episode transitions require an opportunity_id")
        if self.transition is OpportunityTransition.CLOSED:
            if self.opportunity is not None:
                raise ValueError(
                    "CLOSED describes absence and cannot contain a current opportunity"
                )
            if self.portfolio_legs or self.portfolio_signature is not None or self.num_legs:
                raise ValueError("CLOSED cannot claim a currently executable portfolio")
            if self.close_reason is None:
                raise ValueError("CLOSED requires a close reason")
            if self.censor_reason is not None:
                raise ValueError("CLOSED cannot contain a censor reason")
            return self

        if self.close_reason is not None:
            raise ValueError("only CLOSED may contain a close reason")
        if self.transition is OpportunityTransition.RIGHT_CENSORED:
            if self.censor_reason is None:
                raise ValueError("RIGHT_CENSORED requires a censor reason")
        elif self.censor_reason is not None:
            raise ValueError("only RIGHT_CENSORED may contain a censor reason")
        if self.opportunity is None:
            raise ValueError(
                "OPEN, UPDATED, and RIGHT_CENSORED require current opportunity evidence"
            )
        if self.num_legs != len(self.opportunity.quantities):
            raise ValueError("persisted legs must match the opportunity allocation count")
        expected_legs = tuple(
            PortfolioLegSnapshot.from_allocation(allocation, fee=leg.fee)
            for allocation, leg in zip(
                self.opportunity.quantities,
                self.portfolio_legs,
                strict=True,
            )
        )
        if self.portfolio_legs != expected_legs:
            raise ValueError("persisted legs must match the opportunity allocations")
        expected_signature = portfolio_signature(self.portfolio_legs)
        if self.portfolio_signature != expected_signature:
            raise ValueError("portfolio_signature does not match the persisted legs")
        return self

    @property
    def num_markets(self) -> int:
        """Return the component market count without persisting redundant state."""

        return len(self.market_tickers)


class OpportunityEpisode(BaseModel):
    """Current immutable state of one open-to-terminal opportunity episode."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_id: str = Field(min_length=1)
    opportunity_id: str = Field(min_length=1)
    component_id: str = Field(min_length=1)
    transition: OpportunityTransition
    opened_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    censored_at: datetime | None = None
    opened_event_index: int = Field(ge=0)
    last_event_index: int = Field(ge=0)
    closed_event_index: int | None = Field(default=None, ge=0)
    censored_event_index: int | None = Field(default=None, ge=0)
    observation_count: int = Field(ge=1)
    market_tickers: tuple[str, ...]
    relation_types: tuple[RelationType, ...]
    relation_sources: tuple[str, ...] = ()
    market_contexts: tuple[MarketResearchContext, ...] = ()
    opportunity: Opportunity
    portfolio_legs: tuple[PortfolioLegSnapshot, ...]
    portfolio_signature: str = Field(min_length=1)
    close_reason: str | None = Field(default=None, min_length=1)
    censor_reason: str | None = Field(default=None, min_length=1)

    @field_validator(
        "run_id",
        "opportunity_id",
        "component_id",
        "close_reason",
        "censor_reason",
    )
    @classmethod
    def require_nonblank_value(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("episode identifiers and close reasons cannot be blank")
        return value

    @field_validator("opened_at", "updated_at", "closed_at", "censored_at")
    @classmethod
    def require_aware_episode_time(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("opportunity episode times must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_episode(self) -> OpportunityEpisode:
        if self.transition is OpportunityTransition.NOT_PRESENT:
            raise ValueError("NOT_PRESENT has no opportunity episode")
        expected_id = opportunity_episode_id(
            run_id=self.run_id,
            component_id=self.component_id,
            opened_event_index=self.opened_event_index,
        )
        if self.opportunity_id != expected_id:
            raise ValueError("opportunity_id does not match its deterministic episode identity")
        if self.updated_at < self.opened_at:
            raise ValueError("episode updated_at cannot precede opened_at")
        if self.last_event_index < self.opened_event_index:
            raise ValueError("last_event_index cannot precede opened_event_index")
        if self.market_tickers != tuple(sorted(set(self.market_tickers))):
            raise ValueError("episode market tickers must be nonempty and canonically sorted")
        if not self.market_tickers:
            raise ValueError("an opportunity episode requires component markets")
        if self.relation_types != tuple(
            sorted(set(self.relation_types), key=lambda item: item.value)
        ):
            raise ValueError("episode relation types must be unique and canonically sorted")
        if self.relation_sources != tuple(sorted(set(self.relation_sources))) or any(
            not source.strip() for source in self.relation_sources
        ):
            raise ValueError("episode relation sources must be unique and canonically sorted")
        context_tickers = tuple(context.ticker for context in self.market_contexts)
        if context_tickers and context_tickers != self.market_tickers:
            raise ValueError("episode market contexts must match component markets")
        if len(self.portfolio_legs) != len(self.opportunity.quantities):
            raise ValueError("episode legs must match the opportunity allocation count")
        expected_legs = tuple(
            PortfolioLegSnapshot.from_allocation(allocation, fee=leg.fee)
            for allocation, leg in zip(
                self.opportunity.quantities,
                self.portfolio_legs,
                strict=True,
            )
        )
        if self.portfolio_legs != expected_legs:
            raise ValueError("episode legs must match the opportunity allocations")
        if self.portfolio_signature != portfolio_signature(self.portfolio_legs):
            raise ValueError("episode portfolio_signature does not match its legs")

        if self.transition is OpportunityTransition.OPEN:
            if self.observation_count != 1:
                raise ValueError("an OPEN episode must contain exactly one observation")
            if self.updated_at != self.opened_at:
                raise ValueError("an OPEN episode cannot have a later update time")
            if self.last_event_index != self.opened_event_index:
                raise ValueError("an OPEN episode cannot have a later event index")
        elif self.transition is OpportunityTransition.UPDATED:
            if self.observation_count < 2 or self.last_event_index <= self.opened_event_index:
                raise ValueError("an UPDATED episode requires a later observation")

        if self.transition is OpportunityTransition.CLOSED:
            if self.closed_at is None or self.closed_event_index is None:
                raise ValueError("a CLOSED episode requires close time and event index")
            if self.close_reason is None:
                raise ValueError("a CLOSED episode requires a close reason")
            if self.closed_at != self.updated_at:
                raise ValueError("closed_at must equal the final observation time")
            if self.closed_event_index != self.last_event_index:
                raise ValueError("closed_event_index must equal the final event index")
            if self.closed_event_index <= self.opened_event_index:
                raise ValueError("an episode must close after its opening event")
            if self.observation_count < 2:
                raise ValueError("a CLOSED episode requires an opening and closing observation")
        elif (
            self.closed_at is not None
            or self.closed_event_index is not None
            or self.close_reason is not None
        ):
            raise ValueError("only a CLOSED episode may contain closing evidence")

        if self.transition is OpportunityTransition.RIGHT_CENSORED:
            if self.censored_at is None or self.censored_event_index is None:
                raise ValueError("a RIGHT_CENSORED episode requires censor time and event index")
            if self.censor_reason is None:
                raise ValueError("a RIGHT_CENSORED episode requires a censor reason")
            if self.censored_at != self.updated_at:
                raise ValueError("censored_at must equal the final observation time")
            if self.censored_event_index != self.last_event_index:
                raise ValueError("censored_event_index must equal the final event index")
            if self.censored_event_index <= self.opened_event_index:
                raise ValueError("an episode must be censored after its opening event")
            if self.observation_count < 2:
                raise ValueError(
                    "a RIGHT_CENSORED episode requires an opening and censor observation"
                )
        elif (
            self.censored_at is not None
            or self.censored_event_index is not None
            or self.censor_reason is not None
        ):
            raise ValueError("only a RIGHT_CENSORED episode may contain censor evidence")
        return self

    @classmethod
    def from_open_observation(cls, observation: OpportunityObservation) -> OpportunityEpisode:
        """Create an episode from its one valid OPEN observation."""

        if observation.transition is not OpportunityTransition.OPEN:
            raise ValueError("an episode must begin with an OPEN observation")
        assert observation.opportunity_id is not None
        assert observation.opportunity is not None
        assert observation.portfolio_signature is not None
        return cls(
            run_id=observation.run_id,
            opportunity_id=observation.opportunity_id,
            component_id=observation.component_id,
            transition=OpportunityTransition.OPEN,
            opened_at=observation.observed_at,
            updated_at=observation.observed_at,
            opened_event_index=observation.event_index,
            last_event_index=observation.event_index,
            observation_count=1,
            market_tickers=observation.market_tickers,
            relation_types=observation.relation_types,
            relation_sources=observation.relation_sources,
            market_contexts=observation.market_contexts,
            opportunity=observation.opportunity,
            portfolio_legs=observation.portfolio_legs,
            portfolio_signature=observation.portfolio_signature,
        )

    def apply_observation(self, observation: OpportunityObservation) -> Self:
        """Apply one later update or terminal observation without consulting wall time."""

        if self.transition in {
            OpportunityTransition.CLOSED,
            OpportunityTransition.RIGHT_CENSORED,
        }:
            raise ValueError("a terminal opportunity episode cannot transition again")
        if observation.transition not in {
            OpportunityTransition.UPDATED,
            OpportunityTransition.CLOSED,
            OpportunityTransition.RIGHT_CENSORED,
        }:
            raise ValueError(
                "an active episode accepts only UPDATED, CLOSED, or RIGHT_CENSORED observations"
            )
        if (
            observation.run_id != self.run_id
            or observation.component_id != self.component_id
            or observation.opportunity_id != self.opportunity_id
        ):
            raise ValueError("observation identity does not match the active episode")
        if observation.market_tickers != self.market_tickers:
            raise ValueError("component markets cannot change within an opportunity episode")
        if observation.relation_types != self.relation_types:
            raise ValueError("relation types cannot change within an opportunity episode")
        if observation.relation_sources != self.relation_sources:
            raise ValueError("relation sources cannot change within an opportunity episode")
        if observation.event_index <= self.last_event_index:
            raise ValueError("episode observations require strictly increasing event indices")
        if observation.observed_at < self.updated_at:
            raise ValueError("episode event time cannot move backwards")

        values = self.model_dump()
        values.update(
            transition=observation.transition,
            updated_at=observation.observed_at,
            last_event_index=observation.event_index,
            observation_count=self.observation_count + 1,
            market_contexts=observation.market_contexts,
        )
        if observation.transition is OpportunityTransition.CLOSED:
            values.update(
                closed_at=observation.observed_at,
                closed_event_index=observation.event_index,
                close_reason=observation.close_reason,
            )
        else:
            assert observation.opportunity is not None
            assert observation.portfolio_signature is not None
            values.update(
                opportunity=observation.opportunity,
                portfolio_legs=observation.portfolio_legs,
                portfolio_signature=observation.portfolio_signature,
            )
            if observation.transition is OpportunityTransition.RIGHT_CENSORED:
                values.update(
                    censored_at=observation.observed_at,
                    censored_event_index=observation.event_index,
                    censor_reason=observation.censor_reason,
                )
        return type(self).model_validate(values)

    @property
    def duration(self) -> timedelta:
        """Return event-time duration through the latest persisted observation."""

        return self.updated_at - self.opened_at


def classify_opportunity_stage(
    *,
    logical_violation: bool,
    gross_executable: bool = False,
    net_executable: bool = False,
    paper_survives: bool = False,
) -> OpportunityStage | None:
    """Return the highest evidenced stage while enforcing prerequisite ordering."""

    if gross_executable and not logical_violation:
        raise ValueError("gross execution evidence requires a logical violation")
    if net_executable and not gross_executable:
        raise ValueError("net execution evidence requires gross execution evidence")
    if paper_survives and not net_executable:
        raise ValueError("paper survival requires net execution evidence")
    if paper_survives:
        return OpportunityStage.PAPER_SURVIVED
    if net_executable:
        return OpportunityStage.NET_EXECUTABLE
    if gross_executable:
        return OpportunityStage.GROSS_EXECUTABLE
    if logical_violation:
        return OpportunityStage.LOGICAL
    return None


def fresh_yes_midpoint(
    book: OrderBook,
    *,
    as_of: datetime,
    stale_after: timedelta,
) -> Decimal | None:
    """Return a Stage 0 reference midpoint only from a fresh two-sided YES market."""

    if not book.is_fresh(as_of=as_of, stale_after=stale_after):
        return None
    bid = book.best_yes_bid()
    ask = book.best_yes_ask()
    if bid is None or ask is None:
        return None
    return (bid.price + ask.price) / Decimal("2")
