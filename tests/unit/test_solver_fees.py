"""Post-LP fee validation, grouped fills, and gross-to-net stage tests."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from tests.support.executable import implication_component_and_worlds
from tests.support.fees import POLICY_TIME, event, fee_change, policy, series

from arbiter.models.opportunity import FeeValidationStatus, OpportunityStage
from arbiter.models.portfolio import Instrument
from arbiter.models.series import Series
from arbiter.solver.fees import (
    NON_DIRECT_ACCOUNT_PRECISION,
    ZeroFeeModel,
    resolve_fee_policy,
    validate_solver_fees,
)
from arbiter.solver.lp import solve_worst_case_profit


def _instrument(
    ticker: str,
    side: str,
    price: str,
    depth: str = "1",
) -> Instrument:
    source_side = "no_bid" if side == "yes" else "yes_bid"
    return Instrument(
        ticker=ticker,
        side=side,
        price=Decimal(price),
        max_quantity=Decimal(depth),
        source_side=source_side,
        source_price=Decimal("1") - Decimal(price),
    )


def _solve_pair(argentina_price: str = "0.62", *, depth: str = "1"):
    _, worlds = implication_component_and_worlds()
    return solve_worst_case_profit(
        worlds,
        (
            _instrument("A", "yes", argentina_price, depth),
            _instrument("M", "no", "0.30", depth),
        ),
    )


def _policies(
    *,
    fee_type: str = "quadratic",
    multiplier: Decimal = Decimal("1"),
):
    return {
        ticker: policy(ticker, fee_type=fee_type, multiplier=multiplier) for ticker in ("A", "M")
    }


def test_zero_model_preserves_canonical_gross_result_as_net() -> None:
    result = _solve_pair()
    before = result.model_dump()

    opportunity = validate_solver_fees(
        result,
        policies_by_ticker={},
        fee_model=ZeroFeeModel(),
    )

    assert result.model_dump() == before
    assert opportunity.stage is OpportunityStage.NET_EXECUTABLE
    assert opportunity.fees == 0
    assert opportunity.net_state_profits == result.state_profits
    assert opportunity.net_profit == Decimal("0.08")


def test_canonical_standard_direct_fees_match_exact_expected_net() -> None:
    result = _solve_pair()
    opportunity = validate_solver_fees(result, policies_by_ticker=_policies())

    assert result.capital_required == Decimal("0.92")
    assert result.guaranteed_gross_profit == Decimal("0.08")
    assert [quote.net_fee for quote in opportunity.fee_quotes] == [
        Decimal("0.016500"),
        Decimal("0.014700"),
    ]
    assert opportunity.fees == Decimal("0.031200")
    assert opportunity.net_state_profits == (
        Decimal("0.048800"),
        Decimal("1.048800"),
        Decimal("0.048800"),
    )
    assert opportunity.net_profit == Decimal("0.048800")
    assert opportunity.net_edge == Decimal("0.048800") / Decimal("0.92")
    assert opportunity.stage is OpportunityStage.NET_EXECUTABLE


def test_canonical_non_direct_precision_charges_two_cent_aligned_legs() -> None:
    opportunity = validate_solver_fees(
        _solve_pair(),
        policies_by_ticker=_policies(),
        account_precision=NON_DIRECT_ACCOUNT_PRECISION,
    )

    assert [quote.net_fee for quote in opportunity.fee_quotes] == [
        Decimal("0.020000"),
        Decimal("0.020000"),
    ]
    assert opportunity.fees == Decimal("0.040000")
    assert opportunity.net_profit == Decimal("0.040000")


def test_fee_eliminated_gross_candidate_remains_stage_one() -> None:
    result = _solve_pair("0.67")
    opportunity = validate_solver_fees(
        result,
        policies_by_ticker=_policies(),
        minimum_net_profit=Decimal("0"),
    )

    assert result.status == "optimal"
    assert result.guaranteed_gross_profit == Decimal("0.03")
    assert opportunity.fees == Decimal("0.030200")
    assert opportunity.net_profit == Decimal("-0.000200")
    assert opportunity.stage is OpportunityStage.GROSS_EXECUTABLE
    assert opportunity.reason == "fees_below_threshold"


def test_effective_multiplier_changes_only_when_schedule_becomes_active() -> None:
    scheduled = POLICY_TIME + timedelta(days=1)
    market_event = event(
        changes=(
            fee_change(
                "override",
                scheduled,
                fee_type="quadratic_with_maker_fees",
                multiplier=Decimal("1"),
            ),
        )
    )
    parent = series(
        fee_type="quadratic_with_maker_fees",
        multiplier=Decimal("0.5"),
    )
    before_policies = {
        ticker: resolve_fee_policy(
            event=market_event,
            series=parent,
            as_of=POLICY_TIME,
            market_ticker=ticker,
        )
        for ticker in ("A", "M")
    }
    active_policies = {
        ticker: resolve_fee_policy(
            event=market_event,
            series=parent,
            as_of=scheduled,
            market_ticker=ticker,
        )
        for ticker in ("A", "M")
    }

    before = validate_solver_fees(_solve_pair(), policies_by_ticker=before_policies)
    active = validate_solver_fees(_solve_pair(), policies_by_ticker=active_policies)

    assert before.fees == Decimal("0.015700")
    assert before.net_profit == Decimal("0.064300")
    assert active.fees == Decimal("0.031200")
    assert active.net_profit == Decimal("0.048800")
    assert all(item.source == "event_override" for item in active.fee_policies)


def test_verified_zero_and_missing_multiplier_are_not_conflated() -> None:
    zero = validate_solver_fees(
        _solve_pair(),
        policies_by_ticker=_policies(multiplier=Decimal("0")),
    )
    missing_parent = series(multiplier=None)
    conservative_policy = resolve_fee_policy(
        event=event(),
        series=missing_parent,
        as_of=POLICY_TIME,
    )
    conservative = validate_solver_fees(
        _solve_pair(),
        policies_by_ticker={
            ticker: conservative_policy.model_copy(update={"market_ticker": ticker})
            for ticker in ("A", "M")
        },
    )

    assert zero.fees == 0
    assert conservative.fees == Decimal("0.031200")
    assert all(item.source == "documented_standard" for item in conservative.fee_policies)


@pytest.mark.parametrize("fee_type", ["flat", "unknown_live_type"])
def test_unsupported_policy_preserves_gross_candidate_without_numeric_net(
    fee_type: str,
) -> None:
    result = _solve_pair()
    opportunity = validate_solver_fees(
        result,
        policies_by_ticker=_policies(fee_type=fee_type),
    )

    assert opportunity.stage is OpportunityStage.GROSS_EXECUTABLE
    assert opportunity.quantities == result.quantities
    assert opportunity.gross_profit == result.guaranteed_gross_profit
    assert opportunity.fee_status is FeeValidationStatus.UNSUPPORTED
    assert opportunity.fees is None and opportunity.net_profit is None
    assert opportunity.reason == "unsupported_fee_model"


def test_levels_of_one_leg_share_a_ledger_but_other_leg_is_independent() -> None:
    _, worlds = implication_component_and_worlds()
    result = solve_worst_case_profit(
        worlds,
        (
            _instrument("A", "yes", "0.10"),
            _instrument("A", "yes", "0.11"),
            _instrument("A", "yes", "0.40"),
            _instrument("M", "no", "0.10", "3"),
        ),
    )
    opportunity = validate_solver_fees(
        result,
        policies_by_ticker=_policies(),
        account_precision=NON_DIRECT_ACCOUNT_PRECISION,
    )

    assert result.capital_required == Decimal("0.91")
    assert result.guaranteed_gross_profit == Decimal("2.09")
    assert [len(quote.fills) for quote in opportunity.fee_quotes] == [3, 1]
    assert [quote.net_fee for quote in opportunity.fee_quotes] == [
        Decimal("0.030000"),
        Decimal("0.020000"),
    ]
    assert opportunity.fees == Decimal("0.050000")
    assert opportunity.net_profit == Decimal("2.040000")


def test_displayed_liquidity_is_always_quoted_as_taker() -> None:
    class SpyFeeModel:
        roles: list[str]

        def __init__(self) -> None:
            self.roles = []

        def fee_for_fill(
            self,
            *,
            series: Series,
            side: str,
            price: Decimal,
            quantity: Decimal,
            liquidity_role: str = "taker",
        ) -> Decimal:
            del series, side, price, quantity
            self.roles.append(liquidity_role)
            return Decimal("0")

    spy = SpyFeeModel()
    validate_solver_fees(
        _solve_pair(),
        policies_by_ticker=_policies(),
        fee_model=spy,
    )

    assert spy.roles == ["taker", "taker"]


def test_strict_profit_and_edge_thresholds_do_not_accept_equality() -> None:
    baseline = validate_solver_fees(_solve_pair(), policies_by_ticker=_policies())
    assert baseline.net_profit is not None and baseline.net_edge is not None

    profit_equal = validate_solver_fees(
        _solve_pair(),
        policies_by_ticker=_policies(),
        minimum_net_profit=baseline.net_profit,
        minimum_net_edge_bps=Decimal("0"),
    )
    edge_equal = validate_solver_fees(
        _solve_pair(),
        policies_by_ticker=_policies(),
        minimum_net_profit=Decimal("0"),
        minimum_net_edge_bps=baseline.net_edge * Decimal("10000"),
    )

    assert profit_equal.stage is OpportunityStage.GROSS_EXECUTABLE
    assert edge_equal.stage is OpportunityStage.GROSS_EXECUTABLE


def test_nonoptimal_solver_result_cannot_be_upgraded_by_fee_validation() -> None:
    result = _solve_pair("0.80")
    assert result.status == "no_arbitrage"

    with pytest.raises(ValueError, match="positive executable"):
        validate_solver_fees(result, policies_by_ticker=_policies())
