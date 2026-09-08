"""Deterministic, all-or-none paper execution against reconstructed books.

The executor never submits an exchange order.  It preflights every synthetic
``(ticker, side)`` order at the configured latency deadline and records fills
only when the complete portfolio remains available at its detection-time price
limits.  This deliberately fail-closed policy keeps a failed multi-leg attempt
from looking like a partially executed arbitrage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arbiter.models.opportunity import (
    EffectiveFeePolicy,
    FeeFill,
    FeeQuote,
    FeeValidationStatus,
    InstrumentSide,
    Opportunity,
    OpportunityStage,
)
from arbiter.models.orderbook import BookStatus, OrderBook, PriceLevel
from arbiter.models.portfolio import InstrumentAllocation
from arbiter.models.series import Series
from arbiter.solver.fees import (
    DIRECT_ACCOUNT_PRECISION,
    FeeModel,
    FeeRoundingLedger,
    KalshiFeeModel,
    UnsupportedFeeModel,
    quote_order,
)

PaperFillStatus = Literal["filled", "not_filled"]
_SIDE_RANK: dict[InstrumentSide, int] = {"yes": 0, "no": 1}


class PaperExecutionStatus(StrEnum):
    """Terminal outcome of one scheduled paper attempt."""

    SURVIVED = "survived"
    FAILED = "failed"
    INSUFFICIENT_FUTURE_DATA = "insufficient_future_data"


class PaperPriceFill(BaseModel):
    """An exact expected or simulated fill at one executable price level."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    price: Decimal = Field(ge=0, le=1)
    quantity: Decimal = Field(gt=0)
    cost: Decimal = Field(ge=0)

    @field_validator("price", "quantity", "cost")
    @classmethod
    def require_finite_decimal(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("paper-fill decimals must be finite")
        return value

    @model_validator(mode="after")
    def validate_cost(self) -> Self:
        if self.cost != self.price * self.quantity:
            raise ValueError("paper-fill cost must equal price times quantity")
        return self


class PaperLegResult(BaseModel):
    """One synthetic taker order, grouped canonically by market and side."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ticker: str = Field(min_length=1)
    side: InstrumentSide
    quantity: Decimal = Field(gt=0)
    expected_prices: tuple[PaperPriceFill, ...]
    actual_prices: tuple[PaperPriceFill, ...] = ()
    expected_average_price: Decimal = Field(ge=0, le=1)
    actual_average_price: Decimal | None = Field(default=None, ge=0, le=1)
    expected_cost: Decimal = Field(ge=0)
    actual_cost: Decimal | None = Field(default=None, ge=0)
    fill_status: PaperFillStatus
    failure_reason: str | None = Field(default=None, min_length=1)

    @field_validator(
        "quantity",
        "expected_average_price",
        "actual_average_price",
        "expected_cost",
        "actual_cost",
    )
    @classmethod
    def require_finite_optional_decimal(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and not value.is_finite():
            raise ValueError("paper-leg decimals must be finite")
        return value

    @model_validator(mode="after")
    def validate_leg(self) -> Self:
        if not self.ticker.strip():
            raise ValueError("paper-leg ticker cannot be blank")
        if not self.expected_prices:
            raise ValueError("a paper leg requires at least one expected price")
        if tuple(fill.price for fill in self.expected_prices) != tuple(
            sorted(fill.price for fill in self.expected_prices)
        ):
            raise ValueError("expected paper prices must be in ascending execution order")
        expected_quantity = sum((fill.quantity for fill in self.expected_prices), Decimal("0"))
        expected_cost = sum((fill.cost for fill in self.expected_prices), Decimal("0"))
        if expected_quantity != self.quantity or expected_cost != self.expected_cost:
            raise ValueError("expected paper fills do not match leg totals")
        if self.expected_average_price != self.expected_cost / self.quantity:
            raise ValueError("expected average price does not match expected cost")

        if self.fill_status == "filled":
            if self.failure_reason is not None:
                raise ValueError("a filled paper leg cannot have a failure reason")
            if not self.actual_prices or self.actual_cost is None:
                raise ValueError("a filled paper leg requires actual prices and cost")
            if self.actual_average_price is None:
                raise ValueError("a filled paper leg requires an actual average price")
            actual_quantity = sum((fill.quantity for fill in self.actual_prices), Decimal("0"))
            actual_cost = sum((fill.cost for fill in self.actual_prices), Decimal("0"))
            if actual_quantity != self.quantity or actual_cost != self.actual_cost:
                raise ValueError("actual paper fills do not match leg totals")
            if self.actual_average_price != self.actual_cost / self.quantity:
                raise ValueError("actual average price does not match actual cost")
            if tuple(fill.price for fill in self.actual_prices) != tuple(
                sorted(fill.price for fill in self.actual_prices)
            ):
                raise ValueError("actual paper prices must be in ascending execution order")
            expected_cumulative = Decimal("0")
            for expected_fill in self.expected_prices:
                expected_cumulative += expected_fill.quantity
                actual_within_limit = sum(
                    (
                        actual_fill.quantity
                        for actual_fill in self.actual_prices
                        if actual_fill.price <= expected_fill.price
                    ),
                    Decimal("0"),
                )
                if actual_within_limit < expected_cumulative:
                    raise ValueError(
                        "actual paper fills cannot exceed a detection-time price limit"
                    )
        elif (
            self.actual_prices
            or self.actual_average_price is not None
            or self.actual_cost is not None
            or self.failure_reason is None
        ):
            raise ValueError("an unfilled paper leg has no actual fills and requires a reason")
        return self


class PaperExecutionRequest(BaseModel):
    """A net-executable opportunity scheduled for one event-time deadline."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    opportunity_id: str = Field(min_length=1)
    opportunity: Opportunity
    detected_at: datetime
    execute_at: datetime
    latency_ms: int = Field(ge=0)

    @field_validator("opportunity_id")
    @classmethod
    def require_nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("paper opportunity_id cannot be blank")
        return value

    @field_validator("detected_at", "execute_at")
    @classmethod
    def require_aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("paper-execution times must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        expected_execute_at = self.detected_at + timedelta(milliseconds=self.latency_ms)
        if self.execute_at != expected_execute_at:
            raise ValueError("paper execution time must equal detection time plus latency")
        if self.opportunity.stage is not OpportunityStage.NET_EXECUTABLE:
            raise ValueError("paper execution requires a Stage 2 net-executable opportunity")
        if self.opportunity.fee_status is not FeeValidationStatus.APPLIED:
            raise ValueError("paper execution requires applied entry-fee evidence")
        if self.opportunity.paper_execution_status != "not_evaluated":
            raise ValueError("paper execution requires an unevaluated opportunity")
        assert self.opportunity.capital_required is not None
        allocation_cost = sum(
            (allocation.cost for allocation in self.opportunity.quantities),
            Decimal("0"),
        )
        if allocation_cost != self.opportunity.capital_required:
            raise ValueError("paper opportunity capital must equal its allocation costs")
        return self


class PaperExecutionResult(BaseModel):
    """Complete, persistence-ready evidence for one paper attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    opportunity_id: str = Field(min_length=1)
    detected_at: datetime
    simulated_execution_at: datetime
    latency_ms: int = Field(ge=0)
    status: PaperExecutionStatus
    legs: tuple[PaperLegResult, ...]
    minimum_terminal_payout: Decimal = Field(ge=0)
    expected_profit: Decimal
    simulated_locked_profit: Decimal | None = None
    expected_cost: Decimal = Field(ge=0)
    actual_cost: Decimal | None = Field(default=None, ge=0)
    expected_fees: Decimal = Field(ge=0)
    actual_fees: Decimal | None = Field(default=None, ge=0)
    execution_fee_policies: tuple[EffectiveFeePolicy, ...] = ()
    execution_fee_quotes: tuple[FeeQuote, ...] = ()
    failure_reason: str | None = Field(default=None, min_length=1)

    @field_validator("detected_at", "simulated_execution_at")
    @classmethod
    def require_aware_result_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("paper-execution result times must be timezone-aware")
        return value

    @field_validator(
        "expected_profit",
        "minimum_terminal_payout",
        "simulated_locked_profit",
        "expected_cost",
        "actual_cost",
        "expected_fees",
        "actual_fees",
    )
    @classmethod
    def require_finite_result_decimal(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and not value.is_finite():
            raise ValueError("paper-execution result decimals must be finite")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.simulated_execution_at != self.detected_at + timedelta(
            milliseconds=self.latency_ms
        ):
            raise ValueError("result execution time must equal detection time plus latency")
        keys = tuple((leg.ticker, _SIDE_RANK[leg.side]) for leg in self.legs)
        if keys != tuple(sorted(keys)):
            raise ValueError("paper legs must use deterministic ticker/side ordering")
        policy_keys = tuple(
            (
                "" if policy.market_ticker is None else policy.market_ticker,
                policy.policy_version,
            )
            for policy in self.execution_fee_policies
        )
        if policy_keys != tuple(sorted(policy_keys)) or len(policy_keys) != len(set(policy_keys)):
            raise ValueError("execution fee policies must use deterministic unique ordering")
        quote_ids = tuple(quote.order_id for quote in self.execution_fee_quotes)
        if quote_ids != tuple(sorted(set(quote_ids))):
            raise ValueError("execution fee quotes must use deterministic unique ordering")
        if sum((leg.expected_cost for leg in self.legs), Decimal("0")) != self.expected_cost:
            raise ValueError("paper result expected cost must equal its leg total")
        if self.expected_profit != (
            self.minimum_terminal_payout - self.expected_cost - self.expected_fees
        ):
            raise ValueError("expected paper profit does not match payout, cost, and fees")
        if self.status is PaperExecutionStatus.SURVIVED:
            if self.failure_reason is not None:
                raise ValueError("a surviving paper execution cannot have a failure reason")
            if not self.legs or any(leg.fill_status != "filled" for leg in self.legs):
                raise ValueError("a surviving paper execution requires every leg to fill")
            if (
                self.simulated_locked_profit is None
                or self.actual_cost is None
                or self.actual_fees is None
            ):
                raise ValueError("a surviving paper execution requires complete actual economics")
            if (
                sum(
                    (leg.actual_cost for leg in self.legs if leg.actual_cost is not None),
                    Decimal("0"),
                )
                != self.actual_cost
            ):
                raise ValueError("paper result actual cost must equal its leg total")
            assert self.simulated_locked_profit is not None
            assert self.actual_cost is not None
            assert self.actual_fees is not None
            if self.simulated_locked_profit <= 0:
                raise ValueError("a surviving paper execution requires positive locked profit")
            if self.simulated_locked_profit != (
                self.minimum_terminal_payout - self.actual_cost - self.actual_fees
            ):
                raise ValueError("locked paper profit does not match payout, cost, and fees")
            if not self.execution_fee_policies or not self.execution_fee_quotes:
                raise ValueError("a surviving paper execution requires execution-time fee evidence")
            if self.actual_fees != sum(
                (quote.net_fee for quote in self.execution_fee_quotes),
                Decimal("0"),
            ):
                raise ValueError("actual fees must equal execution-time fee quotes")
        else:
            if self.failure_reason is None:
                raise ValueError("a non-surviving paper execution requires a reason")
            if any(leg.fill_status != "not_filled" for leg in self.legs):
                raise ValueError("all-or-none failure cannot record a partial fill")
            if (
                self.simulated_locked_profit is not None
                or self.actual_cost is not None
                or self.actual_fees is not None
            ):
                raise ValueError("an unexecuted portfolio cannot claim actual economics")
            if self.execution_fee_quotes:
                raise ValueError("an unexecuted portfolio cannot claim execution fee quotes")
            if (
                self.status is PaperExecutionStatus.INSUFFICIENT_FUTURE_DATA
                and self.execution_fee_policies
            ):
                raise ValueError("future-data censoring has no execution-time fee policy")
        return self

    @property
    def survived(self) -> bool:
        """Whether every required leg filled and retained positive locked profit."""

        return self.status is PaperExecutionStatus.SURVIVED


class PaperExecutor:
    """Schedule and atomically simulate latency-aware portfolio execution."""

    def __init__(
        self,
        *,
        latency_ms: int,
        stale_after: timedelta,
        fee_model: FeeModel | None = None,
        account_precision: Decimal = DIRECT_ACCOUNT_PRECISION,
        allow_partial_fill: bool = False,
    ) -> None:
        if isinstance(latency_ms, bool) or not isinstance(latency_ms, int) or latency_ms < 0:
            raise ValueError("paper latency_ms must be a nonnegative integer")
        if stale_after < timedelta(0):
            raise ValueError("paper stale_after must be nonnegative")
        if allow_partial_fill:
            raise ValueError("partial paper fills are unsupported; all-or-none is required")
        FeeRoundingLedger(order_id="paper-config-validation", account_precision=account_precision)
        self.latency_ms = latency_ms
        self.stale_after = stale_after
        self.fee_model = KalshiFeeModel() if fee_model is None else fee_model
        self.account_precision = account_precision

    def schedule(
        self,
        *,
        opportunity_id: str,
        opportunity: Opportunity,
        detected_at: datetime,
    ) -> PaperExecutionRequest:
        """Create the immutable timer request for ``detection + configured latency``."""

        return PaperExecutionRequest(
            opportunity_id=opportunity_id,
            opportunity=opportunity,
            detected_at=detected_at,
            execute_at=detected_at + timedelta(milliseconds=self.latency_ms),
            latency_ms=self.latency_ms,
        )

    def execute(
        self,
        request: PaperExecutionRequest,
        *,
        orderbooks: Mapping[str, OrderBook],
        fee_policies: Sequence[EffectiveFeePolicy] | None = None,
    ) -> PaperExecutionResult:
        """Preflight all legs at the deadline, then atomically record fills or failure."""

        expected_legs = _expected_legs(request.opportunity.quantities)
        execution_fee_policies = _canonical_fee_policies(
            request.opportunity.fee_policies if fee_policies is None else fee_policies
        )
        candidate_fills: dict[tuple[str, InstrumentSide], tuple[PaperPriceFill, ...]] = {}
        failed_key: tuple[str, InstrumentSide] | None = None
        failure_reason: str | None = None
        for leg in expected_legs:
            key = (leg.ticker, leg.side)
            book = orderbooks.get(leg.ticker)
            reason = self._book_failure(book, ticker=leg.ticker, as_of=request.execute_at)
            if reason is None:
                assert book is not None
                fills = _walk_leg(leg, book)
                if fills is None:
                    reason = "insufficient_liquidity_at_or_better"
                else:
                    candidate_fills[key] = fills
            if reason is not None:
                failed_key = key
                failure_reason = f"{leg.ticker}:{leg.side}:{reason}"
                break

        if failure_reason is not None:
            assert failed_key is not None
            return _failed_result(
                request,
                expected_legs,
                failure_reason=failure_reason,
                failed_key=failed_key,
                execution_fee_policies=execution_fee_policies,
            )

        actual_cost = sum(
            (fill.cost for fills in candidate_fills.values() for fill in fills),
            Decimal("0"),
        )
        try:
            actual_fees, execution_fee_quotes = self._actual_fees(
                request,
                candidate_fills,
                fee_policies=execution_fee_policies,
            )
        except UnsupportedFeeModel:
            return _failed_result(
                request,
                expected_legs,
                failure_reason="unsupported_fee_model",
                execution_fee_policies=execution_fee_policies,
            )

        opportunity = request.opportunity
        assert opportunity.capital_required is not None
        assert opportunity.gross_state_profits is not None
        minimum_terminal_payout = _minimum_terminal_payout(opportunity)
        locked_profit = minimum_terminal_payout - actual_cost - actual_fees
        if locked_profit <= 0:
            return _failed_result(
                request,
                expected_legs,
                failure_reason="non_positive_simulated_locked_profit",
                execution_fee_policies=execution_fee_policies,
            )

        filled_legs = tuple(
            _filled_leg(
                leg,
                candidate_fills[(leg.ticker, leg.side)],
            )
            for leg in expected_legs
        )
        assert opportunity.net_profit is not None
        assert opportunity.fees is not None
        return PaperExecutionResult(
            opportunity_id=request.opportunity_id,
            detected_at=request.detected_at,
            simulated_execution_at=request.execute_at,
            latency_ms=request.latency_ms,
            status=PaperExecutionStatus.SURVIVED,
            legs=filled_legs,
            minimum_terminal_payout=minimum_terminal_payout,
            expected_profit=opportunity.net_profit,
            simulated_locked_profit=locked_profit,
            expected_cost=opportunity.capital_required,
            actual_cost=actual_cost,
            expected_fees=opportunity.fees,
            actual_fees=actual_fees,
            execution_fee_policies=execution_fee_policies,
            execution_fee_quotes=execution_fee_quotes,
        )

    def insufficient_future_data(
        self,
        request: PaperExecutionRequest,
        *,
        recorded_through: datetime,
    ) -> PaperExecutionResult:
        """Resolve a pending attempt whose deadline lies beyond recorded coverage."""

        if recorded_through.tzinfo is None or recorded_through.utcoffset() is None:
            raise ValueError("recorded_through must be timezone-aware")
        if recorded_through >= request.execute_at:
            raise ValueError("recorded coverage reaches the paper-execution deadline")
        if recorded_through < request.detected_at:
            raise ValueError("recorded coverage cannot precede the detected opportunity")
        expected_legs = _expected_legs(request.opportunity.quantities)
        assert request.opportunity.net_profit is not None
        assert request.opportunity.capital_required is not None
        assert request.opportunity.fees is not None
        return PaperExecutionResult(
            opportunity_id=request.opportunity_id,
            detected_at=request.detected_at,
            simulated_execution_at=request.execute_at,
            latency_ms=request.latency_ms,
            status=PaperExecutionStatus.INSUFFICIENT_FUTURE_DATA,
            legs=tuple(
                _unfilled_leg(leg, failure_reason="insufficient_future_data")
                for leg in expected_legs
            ),
            minimum_terminal_payout=_minimum_terminal_payout(request.opportunity),
            expected_profit=request.opportunity.net_profit,
            expected_cost=request.opportunity.capital_required,
            expected_fees=request.opportunity.fees,
            failure_reason="insufficient_future_data",
        )

    def _book_failure(
        self,
        book: OrderBook | None,
        *,
        ticker: str,
        as_of: datetime,
    ) -> str | None:
        if book is None:
            return "missing_book"
        if book.ticker != ticker:
            return "mismatched_book_ticker"
        if book.local_timestamp > as_of:
            return "book_from_future"
        if book.status is BookStatus.RESYNC_REQUIRED:
            return "resync_required"
        if book.status is BookStatus.STALE:
            return "stale_book"
        if not book.is_fresh(as_of=as_of, stale_after=self.stale_after):
            return "stale_book"
        return None

    def _actual_fees(
        self,
        request: PaperExecutionRequest,
        candidate_fills: Mapping[
            tuple[str, InstrumentSide],
            tuple[PaperPriceFill, ...],
        ],
        *,
        fee_policies: Sequence[EffectiveFeePolicy],
    ) -> tuple[Decimal, tuple[FeeQuote, ...]]:
        policies = _policies_by_ticker(fee_policies)
        total = Decimal("0")
        quotes: list[FeeQuote] = []
        for ticker, side in sorted(
            candidate_fills,
            key=lambda key: (key[0], _SIDE_RANK[key[1]]),
        ):
            try:
                policy = policies[ticker]
            except KeyError:
                raise UnsupportedFeeModel(f"missing fee policy for paper leg {ticker}") from None
            order_id = f"paper:{request.opportunity_id}:{ticker}:{side}"
            ledger = FeeRoundingLedger(
                order_id=order_id,
                account_precision=self.account_precision,
            )
            quote = quote_order(
                (
                    FeeFill(
                        order_id=order_id,
                        ticker=ticker,
                        side=side,
                        price=fill.price,
                        quantity=fill.quantity,
                        liquidity_role="taker",
                    )
                    for fill in candidate_fills[(ticker, side)]
                ),
                ledger,
                series=_series_for_policy(policy),
                fee_model=self.fee_model,
            )
            total += quote.net_fee
            quotes.append(quote)
        return total, tuple(quotes)


def _expected_legs(
    allocations: Sequence[InstrumentAllocation],
) -> tuple[PaperLegResult, ...]:
    grouped: dict[tuple[str, InstrumentSide], list[InstrumentAllocation]] = {}
    for allocation in allocations:
        instrument = allocation.instrument
        grouped.setdefault((instrument.ticker, instrument.side), []).append(allocation)

    legs: list[PaperLegResult] = []
    for ticker, side in sorted(grouped, key=lambda key: (key[0], _SIDE_RANK[key[1]])):
        by_price: dict[Decimal, Decimal] = {}
        for allocation in grouped[(ticker, side)]:
            price = allocation.instrument.price
            by_price[price] = by_price.get(price, Decimal("0")) + allocation.quantity
        expected = tuple(
            PaperPriceFill(price=price, quantity=quantity, cost=price * quantity)
            for price, quantity in sorted(by_price.items())
        )
        quantity = sum((fill.quantity for fill in expected), Decimal("0"))
        cost = sum((fill.cost for fill in expected), Decimal("0"))
        legs.append(
            PaperLegResult(
                ticker=ticker,
                side=side,
                quantity=quantity,
                expected_prices=expected,
                expected_average_price=cost / quantity,
                expected_cost=cost,
                fill_status="not_filled",
                failure_reason="pending_preflight",
            )
        )
    if not legs:
        raise ValueError("paper execution requires at least one portfolio leg")
    return tuple(legs)


def _available_asks(leg: PaperLegResult, book: OrderBook) -> tuple[PriceLevel, ...]:
    return book.yes_asks if leg.side == "yes" else book.no_asks


def _walk_leg(
    leg: PaperLegResult,
    book: OrderBook,
) -> tuple[PaperPriceFill, ...] | None:
    available = [[level.price, level.quantity] for level in _available_asks(leg, book)]
    level_index = 0
    actual_by_price: dict[Decimal, Decimal] = {}
    for expected in leg.expected_prices:
        remaining = expected.quantity
        while remaining > 0:
            while level_index < len(available) and available[level_index][1] == 0:
                level_index += 1
            if level_index >= len(available):
                return None
            price, quantity = available[level_index]
            if price > expected.price:
                return None
            consumed = min(remaining, quantity)
            actual_by_price[price] = actual_by_price.get(price, Decimal("0")) + consumed
            available[level_index][1] -= consumed
            remaining -= consumed
    return tuple(
        PaperPriceFill(price=price, quantity=quantity, cost=price * quantity)
        for price, quantity in sorted(actual_by_price.items())
    )


def _filled_leg(
    expected: PaperLegResult,
    actual_prices: tuple[PaperPriceFill, ...],
) -> PaperLegResult:
    actual_cost = sum((fill.cost for fill in actual_prices), Decimal("0"))
    return PaperLegResult(
        ticker=expected.ticker,
        side=expected.side,
        quantity=expected.quantity,
        expected_prices=expected.expected_prices,
        actual_prices=actual_prices,
        expected_average_price=expected.expected_average_price,
        actual_average_price=actual_cost / expected.quantity,
        expected_cost=expected.expected_cost,
        actual_cost=actual_cost,
        fill_status="filled",
    )


def _unfilled_leg(
    expected: PaperLegResult,
    *,
    failure_reason: str,
) -> PaperLegResult:
    return PaperLegResult(
        ticker=expected.ticker,
        side=expected.side,
        quantity=expected.quantity,
        expected_prices=expected.expected_prices,
        expected_average_price=expected.expected_average_price,
        expected_cost=expected.expected_cost,
        fill_status="not_filled",
        failure_reason=failure_reason,
    )


def _failed_result(
    request: PaperExecutionRequest,
    expected_legs: tuple[PaperLegResult, ...],
    *,
    failure_reason: str,
    failed_key: tuple[str, InstrumentSide] | None = None,
    execution_fee_policies: tuple[EffectiveFeePolicy, ...],
) -> PaperExecutionResult:
    opportunity = request.opportunity
    assert opportunity.net_profit is not None
    assert opportunity.capital_required is not None
    assert opportunity.fees is not None
    return PaperExecutionResult(
        opportunity_id=request.opportunity_id,
        detected_at=request.detected_at,
        simulated_execution_at=request.execute_at,
        latency_ms=request.latency_ms,
        status=PaperExecutionStatus.FAILED,
        legs=tuple(
            _unfilled_leg(
                leg,
                failure_reason=(
                    failure_reason
                    if failed_key == (leg.ticker, leg.side)
                    else "all_or_none_aborted"
                ),
            )
            for leg in expected_legs
        ),
        minimum_terminal_payout=_minimum_terminal_payout(opportunity),
        expected_profit=opportunity.net_profit,
        expected_cost=opportunity.capital_required,
        expected_fees=opportunity.fees,
        execution_fee_policies=execution_fee_policies,
        failure_reason=failure_reason,
    )


def _canonical_fee_policies(
    policies: Sequence[EffectiveFeePolicy],
) -> tuple[EffectiveFeePolicy, ...]:
    return tuple(
        sorted(
            policies,
            key=lambda policy: (
                "" if policy.market_ticker is None else policy.market_ticker,
                policy.policy_version,
            ),
        )
    )


def _policies_by_ticker(
    policies: Sequence[EffectiveFeePolicy],
) -> dict[str, EffectiveFeePolicy]:
    result: dict[str, EffectiveFeePolicy] = {}
    for policy in policies:
        ticker = policy.market_ticker
        if ticker is None:
            continue
        previous = result.get(ticker)
        if previous is not None and previous != policy:
            raise UnsupportedFeeModel(f"conflicting fee policies for paper leg {ticker}")
        result[ticker] = policy
    return result


def _minimum_terminal_payout(opportunity: Opportunity) -> Decimal:
    assert opportunity.capital_required is not None
    assert opportunity.gross_state_profits is not None
    return min(opportunity.gross_state_profits) + opportunity.capital_required


def _series_for_policy(policy: EffectiveFeePolicy) -> Series:
    return Series(
        ticker=policy.series_ticker,
        title=policy.series_ticker,
        fee_type=policy.fee_type,
        fee_multiplier=policy.fee_multiplier,
        raw={},
    )
