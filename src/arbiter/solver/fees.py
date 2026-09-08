"""Current Kalshi prediction-contract fees and exact order-level rounding."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Protocol

from arbiter.models.event import Event, EventFeeChange
from arbiter.models.opportunity import (
    EffectiveFeePolicy,
    FeeFill,
    FeeFillQuote,
    FeePolicySource,
    FeeQuote,
    FeeValidationStatus,
    InstrumentSide,
    Opportunity,
    OpportunityStage,
)
from arbiter.models.portfolio import SolverResult
from arbiter.models.series import Series

KALSHI_FEE_POLICY_VERSION = "kalshi-prediction-fees-2026-07-07+rounding-2026-09-03"
MODEL_FEE_QUANTUM = Decimal("0.000001")
DIRECT_ACCOUNT_PRECISION = Decimal("0.0001")
NON_DIRECT_ACCOUNT_PRECISION = Decimal("0.01")
_ACCOUNT_PRECISIONS = frozenset({DIRECT_ACCOUNT_PRECISION, NON_DIRECT_ACCOUNT_PRECISION})
_TAKER_RATE = Decimal("0.07")
_MAKER_RATE = Decimal("0.0175")
_COMBO_MAKER_RATE = Decimal("0.035")
_QUADRATIC_TYPES = frozenset(
    {
        "quadratic",
        "quadratic_with_maker_fees",
        "quadratic_with_combo_maker_fees",
    }
)


class UnsupportedFeeModel(ValueError):
    """Raised when current official behavior is insufficient for a net claim."""


class FeePolicyResolutionError(ValueError):
    """Raised when event and series metadata cannot determine one policy safely."""


class FeeModel(Protocol):
    """Pure unrounded model-fee calculation independent of account state."""

    def fee_for_fill(
        self,
        *,
        series: Series,
        side: str,
        price: Decimal,
        quantity: Decimal,
        liquidity_role: str = "taker",
    ) -> Decimal: ...


class ZeroFeeModel:
    """Explicit zero model for mathematical and pipeline-isolation tests."""

    def fee_for_fill(
        self,
        *,
        series: Series,
        side: str,
        price: Decimal,
        quantity: Decimal,
        liquidity_role: str = "taker",
    ) -> Decimal:
        _validate_fee_inputs(
            side=side,
            price=price,
            quantity=quantity,
            liquidity_role=liquidity_role,
        )
        del series
        return Decimal("0")


class KalshiFeeModel:
    """Verified July 7, 2026 quadratic prediction-contract fee models."""

    def fee_for_fill(
        self,
        *,
        series: Series,
        side: str,
        price: Decimal,
        quantity: Decimal,
        liquidity_role: str = "taker",
    ) -> Decimal:
        _validate_fee_inputs(
            side=side,
            price=price,
            quantity=quantity,
            liquidity_role=liquidity_role,
        )
        fee_type, multiplier = _series_fee_terms(series)
        if fee_type not in _QUADRATIC_TYPES:
            raise UnsupportedFeeModel(f"unsupported Kalshi fee type: {fee_type}")
        if liquidity_role == "taker":
            rate = _TAKER_RATE
        elif fee_type == "quadratic":
            rate = Decimal("0")
        elif fee_type == "quadratic_with_maker_fees":
            rate = _MAKER_RATE
        else:
            rate = _COMBO_MAKER_RATE
        return multiplier * rate * quantity * price * (Decimal("1") - price)


@dataclass(slots=True)
class FeeRoundingLedger:
    """Explicit mutable rounding carry for exactly one exchange order."""

    order_id: str
    account_precision: Decimal = DIRECT_ACCOUNT_PRECISION
    accumulated_rounding: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("fee ledger order_id cannot be empty")
        if self.account_precision not in _ACCOUNT_PRECISIONS:
            raise ValueError("account precision must be 0.0001 (direct) or 0.01 (non-direct)")
        if not self.accumulated_rounding.is_finite() or self.accumulated_rounding < Decimal("0"):
            raise ValueError("accumulated rounding must be finite and nonnegative")


def _validate_fee_inputs(
    *,
    side: str,
    price: Decimal,
    quantity: Decimal,
    liquidity_role: str,
) -> None:
    if side not in {"yes", "no"}:
        raise ValueError("fee side must be yes or no")
    if not price.is_finite() or not Decimal("0") <= price <= Decimal("1"):
        raise ValueError("fee price must be finite and between zero and one")
    if not quantity.is_finite() or quantity < Decimal("0"):
        raise ValueError("fee quantity must be finite and nonnegative")
    if liquidity_role not in {"taker", "maker"}:
        raise ValueError("liquidity role must be taker or maker")


def _series_fee_terms(series: Series) -> tuple[str, Decimal]:
    fee_type = series.fee_type
    multiplier = series.fee_multiplier
    if fee_type is None and multiplier is not None:
        raise UnsupportedFeeModel("fee multiplier is present without a fee type")
    if fee_type is None:
        return "quadratic", Decimal("1")
    if multiplier is None:
        return fee_type, Decimal("1")
    if not multiplier.is_finite() or multiplier < 0:
        raise UnsupportedFeeModel("fee multiplier must be finite and nonnegative")
    return fee_type, multiplier


def _series_base_policy(
    *,
    event: Event,
    series: Series,
    market_ticker: str | None,
) -> EffectiveFeePolicy:
    if series.fee_type is None and series.fee_multiplier is not None:
        raise FeePolicyResolutionError("series fee multiplier is present without a fee type")
    if series.fee_type is None or series.fee_multiplier is None:
        fee_type = "quadratic" if series.fee_type is None else series.fee_type
        multiplier = Decimal("1")
        source: FeePolicySource = "documented_standard"
    else:
        fee_type = series.fee_type
        multiplier = series.fee_multiplier
        source = "series_default"
    return EffectiveFeePolicy(
        market_ticker=market_ticker,
        event_ticker=event.ticker,
        series_ticker=series.ticker,
        fee_type=fee_type,
        fee_multiplier=multiplier,
        source=source,
        policy_version=KALSHI_FEE_POLICY_VERSION,
    )


def _validate_change(change: EventFeeChange, *, event: Event, series: Series) -> None:
    if change.event_ticker != event.ticker:
        raise FeePolicyResolutionError("attached fee change has the wrong event ticker")
    if change.series_ticker != series.ticker:
        raise FeePolicyResolutionError("attached fee change has the wrong series ticker")
    if change.scheduled_ts.tzinfo is None or change.scheduled_ts.utcoffset() is None:
        raise FeePolicyResolutionError("fee change scheduled_ts must be timezone-aware")


def _reject_conflicting_timestamps(changes: Sequence[EventFeeChange]) -> None:
    grouped: dict[datetime, set[tuple[str | None, Decimal | None]]] = defaultdict(set)
    for change in changes:
        grouped[change.scheduled_ts].add((change.fee_type_override, change.fee_multiplier_override))
    if any(len(terms) > 1 for terms in grouped.values()):
        raise FeePolicyResolutionError("conflicting fee changes share one scheduled timestamp")


def resolve_fee_policy(
    *,
    event: Event,
    series: Series,
    as_of: datetime,
    market_ticker: str | None = None,
) -> EffectiveFeePolicy:
    """Resolve current event override then dated changes over the series default."""

    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise FeePolicyResolutionError("fee-policy as_of must be timezone-aware")
    if event.series_ticker is not None and event.series_ticker != series.ticker:
        raise FeePolicyResolutionError("event and series tickers do not match")

    changes = tuple(event.fee_changes)
    for change in changes:
        _validate_change(change, event=event, series=series)
    _reject_conflicting_timestamps(changes)
    ordered = tuple(sorted(changes, key=lambda item: (item.scheduled_ts, item.change_id)))
    future = tuple(change for change in ordered if change.scheduled_ts > as_of)
    next_change = future[0] if future else None

    series_policy = _series_base_policy(
        event=event,
        series=series,
        market_ticker=market_ticker,
    )
    current_type = event.fee_type_override
    current_multiplier = event.fee_multiplier_override
    if (current_type is None) != (current_multiplier is None):
        raise FeePolicyResolutionError("event fee override is only partially populated")
    if current_type is not None and current_multiplier is not None:
        policy = series_policy.model_copy(
            update={
                "fee_type": current_type,
                "fee_multiplier": current_multiplier,
                "source": "event_override",
            }
        )
    else:
        policy = series_policy

    applicable = tuple(change for change in ordered if change.scheduled_ts <= as_of)
    if applicable:
        selected = applicable[-1]
        if selected.fee_type_override is None:
            policy = series_policy.model_copy(
                update={
                    "effective_at": selected.scheduled_ts,
                    "change_id": selected.change_id,
                }
            )
        else:
            assert selected.fee_multiplier_override is not None
            policy = series_policy.model_copy(
                update={
                    "fee_type": selected.fee_type_override,
                    "fee_multiplier": selected.fee_multiplier_override,
                    "source": "event_override",
                    "effective_at": selected.scheduled_ts,
                    "change_id": selected.change_id,
                }
            )
    if next_change is not None:
        policy = policy.model_copy(
            update={
                "next_scheduled_at": next_change.scheduled_ts,
                "next_change_id": next_change.change_id,
            }
        )
    return policy


def ceil_model_fee(value: Decimal) -> Decimal:
    """Round a nonnegative raw model fee upward to six dollar decimals."""

    if not value.is_finite() or value < 0:
        raise ValueError("model fee must be finite and nonnegative")
    return value.quantize(MODEL_FEE_QUANTUM, rounding=ROUND_CEILING)


def _floor_to_grid(value: Decimal, quantum: Decimal) -> Decimal:
    units = (value / quantum).to_integral_value(rounding=ROUND_FLOOR)
    return units * quantum


def quote_order(
    fills: Iterable[FeeFill],
    ledger: FeeRoundingLedger,
    *,
    series: Series,
    fee_model: FeeModel,
) -> FeeQuote:
    """Quote fills in order and atomically advance their explicit order ledger."""

    ordered = tuple(fills)
    if not ordered:
        raise ValueError("cannot quote an order without fills")
    if any(fill.order_id != ledger.order_id for fill in ordered):
        raise ValueError("every fill must match the ledger order_id")
    first = ordered[0]
    if any(fill.ticker != first.ticker or fill.side != first.side for fill in ordered):
        raise ValueError("one fee ledger may cover only one ticker and purchased side")

    accumulator = ledger.accumulated_rounding
    quoted: list[FeeFillQuote] = []
    for fill in ordered:
        raw_fee = fee_model.fee_for_fill(
            series=series,
            side=fill.side,
            price=fill.price,
            quantity=fill.quantity,
            liquidity_role=fill.liquidity_role,
        )
        trade_fee = ceil_model_fee(raw_fee)
        unaligned_change = fill.signed_revenue - trade_fee
        aligned_change = _floor_to_grid(unaligned_change, ledger.account_precision)
        rounding_fee = unaligned_change - aligned_change
        available = accumulator + rounding_fee
        eligible_rebate = _floor_to_grid(available, ledger.account_precision)
        affordable_rebate = _floor_to_grid(
            trade_fee + rounding_fee,
            ledger.account_precision,
        )
        rebate = min(eligible_rebate, affordable_rebate)
        net_fee = trade_fee + rounding_fee - rebate
        next_accumulator = available - rebate
        fill_quote = FeeFillQuote(
            fill=fill,
            signed_revenue=fill.signed_revenue,
            model_fee=raw_fee,
            trade_fee=trade_fee,
            aligned_change_before_rebate=aligned_change,
            rounding_fee=rounding_fee,
            rebate=rebate,
            net_fee=net_fee,
            balance_change=fill.signed_revenue - net_fee,
            accumulator_before=accumulator,
            accumulator_after=next_accumulator,
        )
        quoted.append(fill_quote)
        accumulator = next_accumulator

    ledger.accumulated_rounding = accumulator
    return FeeQuote(
        order_id=ledger.order_id,
        fills=tuple(quoted),
        model_fee=sum((item.model_fee for item in quoted), Decimal("0")),
        trade_fee=sum((item.trade_fee for item in quoted), Decimal("0")),
        rounding_fee=sum((item.rounding_fee for item in quoted), Decimal("0")),
        rebate=sum((item.rebate for item in quoted), Decimal("0")),
        net_fee=sum((item.net_fee for item in quoted), Decimal("0")),
        balance_change=sum((item.balance_change for item in quoted), Decimal("0")),
        ending_accumulator=accumulator,
    )


def _policy_series(policy: EffectiveFeePolicy) -> Series:
    return Series(
        ticker=policy.series_ticker,
        title=policy.series_ticker,
        fee_type=policy.fee_type,
        fee_multiplier=policy.fee_multiplier,
        raw={},
    )


def _stage_one(
    result: SolverResult,
    *,
    policies: tuple[EffectiveFeePolicy, ...],
    reason: str,
) -> Opportunity:
    return Opportunity(
        stage=OpportunityStage.GROSS_EXECUTABLE,
        quantities=result.quantities,
        capital_required=result.capital_required,
        gross_profit=result.guaranteed_gross_profit,
        gross_edge=result.gross_edge,
        gross_state_profits=result.state_profits,
        fee_status=FeeValidationStatus.UNSUPPORTED,
        fee_policies=policies,
        reason=reason,
    )


def validate_solver_fees(
    result: SolverResult,
    *,
    policies_by_ticker: Mapping[str, EffectiveFeePolicy],
    fee_model: FeeModel | None = None,
    account_precision: Decimal = DIRECT_ACCOUNT_PRECISION,
    minimum_net_profit: Decimal = Decimal("0.01"),
    minimum_net_edge_bps: Decimal = Decimal("1"),
) -> Opportunity:
    """Apply per-order entry fees to a gross result without changing its LP portfolio."""

    if result.status != "optimal" or not result.quantities:
        raise ValueError("fee validation requires a positive executable solver result")
    for label, value in (
        ("minimum_net_profit", minimum_net_profit),
        ("minimum_net_edge_bps", minimum_net_edge_bps),
    ):
        if not value.is_finite() or value < 0:
            raise ValueError(f"{label} must be finite and nonnegative")
    model = KalshiFeeModel() if fee_model is None else fee_model

    grouped: dict[tuple[str, InstrumentSide], list[FeeFill]] = {}
    applied_policies: dict[str, EffectiveFeePolicy] = {}
    for allocation in result.quantities:
        ticker = allocation.instrument.ticker
        side = allocation.instrument.side
        policy = policies_by_ticker.get(ticker)
        if policy is None:
            if isinstance(model, ZeroFeeModel):
                policy = EffectiveFeePolicy(
                    market_ticker=ticker,
                    event_ticker=ticker,
                    series_ticker=ticker,
                    fee_type="quadratic",
                    fee_multiplier=Decimal("0"),
                    source="documented_standard",
                    policy_version=KALSHI_FEE_POLICY_VERSION,
                )
            else:
                return _stage_one(
                    result,
                    policies=tuple(applied_policies.values()),
                    reason="unsupported_fee_model",
                )
        if policy.market_ticker not in {None, ticker}:
            raise FeePolicyResolutionError("fee policy is mapped to the wrong market ticker")
        policy = policy.model_copy(update={"market_ticker": ticker})
        applied_policies[ticker] = policy
        order_id = f"{ticker}:{side}"
        grouped.setdefault((ticker, side), []).append(
            FeeFill(
                order_id=order_id,
                ticker=ticker,
                side=side,
                price=allocation.instrument.price,
                quantity=allocation.quantity,
                liquidity_role="taker",
            )
        )

    quotes: list[FeeQuote] = []
    try:
        for ticker, side in sorted(grouped):
            policy = applied_policies[ticker]
            ledger = FeeRoundingLedger(
                order_id=f"{ticker}:{side}",
                account_precision=account_precision,
            )
            quotes.append(
                quote_order(
                    grouped[(ticker, side)],
                    ledger,
                    series=_policy_series(policy),
                    fee_model=model,
                )
            )
    except UnsupportedFeeModel:
        return _stage_one(
            result,
            policies=tuple(applied_policies[ticker] for ticker in sorted(applied_policies)),
            reason="unsupported_fee_model",
        )

    total_fees = sum((quote.net_fee for quote in quotes), Decimal("0"))
    net_state_profits = tuple(profit - total_fees for profit in result.state_profits)
    net_profit = min(net_state_profits)
    net_edge = net_profit / result.capital_required if result.capital_required > 0 else None
    edge_survives = net_edge is not None and net_edge * Decimal("10000") > minimum_net_edge_bps
    net_survives = net_profit > minimum_net_profit and edge_survives
    return Opportunity(
        stage=(
            OpportunityStage.NET_EXECUTABLE if net_survives else OpportunityStage.GROSS_EXECUTABLE
        ),
        quantities=result.quantities,
        capital_required=result.capital_required,
        gross_profit=result.guaranteed_gross_profit,
        gross_edge=result.gross_edge,
        gross_state_profits=result.state_profits,
        fee_status=FeeValidationStatus.APPLIED,
        fees=total_fees,
        net_profit=net_profit,
        net_edge=net_edge,
        net_state_profits=net_state_profits,
        fee_policies=tuple(applied_policies[ticker] for ticker in sorted(applied_policies)),
        fee_quotes=tuple(quotes),
        reason=None if net_survives else "fees_below_threshold",
    )
