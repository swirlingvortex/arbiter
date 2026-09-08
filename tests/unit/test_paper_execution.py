"""Latency-aware all-or-none paper execution tests."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError
from tests.support.executable import NOW, book
from tests.support.fees import policy

from arbiter.engine.paper_execution import (
    PaperExecutionResult,
    PaperExecutionStatus,
    PaperExecutor,
    PaperLegResult,
    PaperPriceFill,
)
from arbiter.models.opportunity import FeeValidationStatus, Opportunity, OpportunityStage
from arbiter.models.orderbook import BookStatus
from arbiter.models.portfolio import Instrument, InstrumentAllocation, SolverResult
from arbiter.solver.fees import ZeroFeeModel, validate_solver_fees


def _allocation(
    ticker: str,
    side: str,
    price: str,
    quantity: str,
) -> InstrumentAllocation:
    decimal_price = Decimal(price)
    decimal_quantity = Decimal(quantity)
    source_side = "no_bid" if side == "yes" else "yes_bid"
    return InstrumentAllocation(
        instrument=Instrument(
            ticker=ticker,
            side=side,  # type: ignore[arg-type]
            price=decimal_price,
            max_quantity=decimal_quantity,
            source_side=source_side,  # type: ignore[arg-type]
            source_price=Decimal("1") - decimal_price,
        ),
        quantity=decimal_quantity,
        cost=decimal_price * decimal_quantity,
    )


def _opportunity(
    allocations: tuple[InstrumentAllocation, ...] | None = None,
    *,
    gross_state_profits: tuple[Decimal, ...] | None = None,
) -> Opportunity:
    legs = allocations or (
        _allocation("M", "no", "0.30", "1.00"),
        _allocation("A", "yes", "0.62", "1.00"),
    )
    capital = sum((allocation.cost for allocation in legs), Decimal("0"))
    state_profits = gross_state_profits or (
        Decimal("0.08"),
        Decimal("1.08"),
        Decimal("0.08"),
    )
    profit = min(state_profits)
    return Opportunity(
        stage=OpportunityStage.NET_EXECUTABLE,
        quantities=legs,
        capital_required=capital,
        gross_profit=profit,
        gross_edge=profit / capital,
        gross_state_profits=state_profits,
        fee_status=FeeValidationStatus.APPLIED,
        fees=Decimal("0"),
        net_profit=profit,
        net_edge=profit / capital,
        net_state_profits=state_profits,
        fee_policies=(policy("M", multiplier=Decimal("0")), policy("A", multiplier=Decimal("0"))),
    )


def _executor(*, latency_ms: int = 100) -> PaperExecutor:
    return PaperExecutor(
        latency_ms=latency_ms,
        stale_after=timedelta(seconds=2),
        fee_model=ZeroFeeModel(),
    )


def test_schedules_exact_configured_latency_and_fills_in_canonical_leg_order() -> None:
    executor = _executor(latency_ms=125)
    request = executor.schedule(
        opportunity_id="opportunity-1",
        opportunity=_opportunity(),
        detected_at=NOW,
    )

    result = executor.execute(
        request,
        orderbooks={
            "M": book("M", yes=(("0.71", "1.00"),)),
            "A": book("A", no=(("0.39", "1.00"),)),
        },
    )

    assert request.execute_at == NOW + timedelta(milliseconds=125)
    assert result.status is PaperExecutionStatus.SURVIVED
    assert result.survived is True
    assert [(leg.ticker, leg.side) for leg in result.legs] == [("A", "yes"), ("M", "no")]
    assert [leg.expected_average_price for leg in result.legs] == [
        Decimal("0.62"),
        Decimal("0.30"),
    ]
    assert [leg.actual_average_price for leg in result.legs] == [
        Decimal("0.61"),
        Decimal("0.29"),
    ]
    assert all(leg.fill_status == "filled" for leg in result.legs)
    assert result.expected_profit == Decimal("0.08")
    assert result.expected_cost == Decimal("0.9200")
    assert result.actual_cost == Decimal("0.9000")
    assert result.actual_fees == Decimal("0")
    assert result.simulated_locked_profit == Decimal("0.1000")


def test_persistence_models_reject_nonpositive_survival_and_worse_actual_tranches() -> None:
    executor = _executor()
    request = executor.schedule(
        opportunity_id="opportunity-validation",
        opportunity=_opportunity(),
        detected_at=NOW,
    )
    result = executor.execute(
        request,
        orderbooks={
            "M": book("M", yes=(("0.71", "1.00"),)),
            "A": book("A", no=(("0.39", "1.00"),)),
        },
    )
    invalid_result = result.model_dump(mode="python")
    invalid_result["simulated_locked_profit"] = Decimal("0")
    with pytest.raises(ValidationError, match="positive locked profit"):
        PaperExecutionResult.model_validate(invalid_result)

    with pytest.raises(ValidationError, match="detection-time price limit"):
        PaperLegResult(
            ticker="A",
            side="yes",
            quantity=Decimal("3"),
            expected_prices=(
                PaperPriceFill(
                    price=Decimal("0.60"),
                    quantity=Decimal("1"),
                    cost=Decimal("0.60"),
                ),
                PaperPriceFill(
                    price=Decimal("0.65"),
                    quantity=Decimal("2"),
                    cost=Decimal("1.30"),
                ),
            ),
            actual_prices=(
                PaperPriceFill(
                    price=Decimal("0.64"),
                    quantity=Decimal("3"),
                    cost=Decimal("1.92"),
                ),
            ),
            expected_average_price=Decimal("1.90") / Decimal("3"),
            actual_average_price=Decimal("0.64"),
            expected_cost=Decimal("1.90"),
            actual_cost=Decimal("1.92"),
            fill_status="filled",
        )


def test_walks_current_depth_without_violating_each_detection_price_limit() -> None:
    allocations = (
        _allocation("M", "no", "0.30", "3.00"),
        _allocation("A", "yes", "0.65", "2.00"),
        _allocation("A", "yes", "0.60", "1.00"),
    )
    opportunity = _opportunity(
        allocations,
        gross_state_profits=(Decimal("0.20"), Decimal("3.20"), Decimal("0.20")),
    )
    request = _executor().schedule(
        opportunity_id="opportunity-depth",
        opportunity=opportunity,
        detected_at=NOW,
    )

    result = _executor().execute(
        request,
        orderbooks={
            "A": book("A", no=(("0.41", "1.50"), ("0.36", "1.50"))),
            "M": book("M", yes=(("0.71", "3.00"),)),
        },
    )

    assert result.status is PaperExecutionStatus.SURVIVED
    a_leg = result.legs[0]
    assert [(fill.price, fill.quantity) for fill in a_leg.expected_prices] == [
        (Decimal("0.60"), Decimal("1.00")),
        (Decimal("0.65"), Decimal("2.00")),
    ]
    assert [(fill.price, fill.quantity) for fill in a_leg.actual_prices] == [
        (Decimal("0.59"), Decimal("1.50")),
        (Decimal("0.64"), Decimal("1.50")),
    ]
    assert result.simulated_locked_profit == Decimal("0.2850")


def test_reprices_exact_entry_fees_at_actual_fill_prices() -> None:
    allocations = (
        _allocation("M", "no", "0.30", "1.00"),
        _allocation("A", "yes", "0.62", "1.00"),
    )
    gross = SolverResult(
        status="optimal",
        guaranteed_gross_profit=Decimal("0.08"),
        capital_required=Decimal("0.92"),
        gross_edge=Decimal("0.08") / Decimal("0.92"),
        quantities=allocations,
        state_payouts=(Decimal("1"), Decimal("2"), Decimal("1")),
        state_profits=(Decimal("0.08"), Decimal("1.08"), Decimal("0.08")),
        min_state_profit=Decimal("0.08"),
    )
    opportunity = validate_solver_fees(
        gross,
        policies_by_ticker={"A": policy("A"), "M": policy("M")},
        minimum_net_profit=Decimal("0"),
        minimum_net_edge_bps=Decimal("0"),
    )
    executor = PaperExecutor(latency_ms=100, stale_after=timedelta(seconds=2))
    request = executor.schedule(
        opportunity_id="opportunity-fees",
        opportunity=opportunity,
        detected_at=NOW,
    )

    result = executor.execute(
        request,
        orderbooks={
            "A": book("A", no=(("0.39", "1.00"),)),
            "M": book("M", yes=(("0.71", "1.00"),)),
        },
    )

    assert result.status is PaperExecutionStatus.SURVIVED
    assert result.expected_fees == Decimal("0.031200")
    assert result.actual_fees == Decimal("0.031200")
    assert result.expected_profit == Decimal("0.048800")
    assert result.simulated_locked_profit == Decimal("0.068800")
    assert result.simulated_locked_profit == (
        result.minimum_terminal_payout - result.actual_cost - result.actual_fees
    )


def test_disappearing_leg_fails_the_entire_attempt_without_partial_fills() -> None:
    executor = _executor()
    request = executor.schedule(
        opportunity_id="opportunity-disappeared",
        opportunity=_opportunity(),
        detected_at=NOW,
    )

    result = executor.execute(
        request,
        orderbooks={
            "A": book("A", no=(("0.39", "1.00"),)),
            "M": book("M", no=(("0.20", "10.00"),)),
        },
    )

    assert result.status is PaperExecutionStatus.FAILED
    assert result.failure_reason == "M:no:insufficient_liquidity_at_or_better"
    assert all(leg.fill_status == "not_filled" for leg in result.legs)
    assert all(leg.actual_prices == () for leg in result.legs)
    assert result.actual_cost is None
    assert result.actual_fees is None
    assert result.simulated_locked_profit is None
    assert result.legs[0].failure_reason == "all_or_none_aborted"
    assert result.legs[1].failure_reason == result.failure_reason


def test_a_cheaper_average_cannot_hide_failure_of_the_first_price_tranche() -> None:
    allocations = (
        _allocation("A", "yes", "0.60", "1.00"),
        _allocation("A", "yes", "0.90", "1.00"),
        _allocation("M", "no", "0.05", "2.00"),
    )
    opportunity = _opportunity(
        allocations,
        gross_state_profits=(Decimal("0.40"), Decimal("2.40"), Decimal("0.40")),
    )
    executor = _executor()
    request = executor.schedule(
        opportunity_id="opportunity-limit",
        opportunity=opportunity,
        detected_at=NOW,
    )

    result = executor.execute(
        request,
        orderbooks={
            # Two units at .70 would beat the total .75 average budget, but the first
            # unit's detection-time limit was .60 and therefore must fail closed.
            "A": book("A", no=(("0.30", "2.00"),)),
            "M": book("M", yes=(("0.95", "2.00"),)),
        },
    )

    assert result.status is PaperExecutionStatus.FAILED
    assert result.failure_reason == "A:yes:insufficient_liquidity_at_or_better"


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        (None, "A:yes:missing_book"),
        (
            book(
                "A",
                no=(("0.39", "1.00"),),
                status=BookStatus.STALE,
                status_reason="fixture stale",
            ),
            "A:yes:stale_book",
        ),
        (
            book(
                "A",
                no=(("0.39", "1.00"),),
                status=BookStatus.RESYNC_REQUIRED,
                status_reason="sequence gap",
            ),
            "A:yes:resync_required",
        ),
        (
            book(
                "A",
                no=(("0.39", "1.00"),),
                local_timestamp=NOW + timedelta(seconds=1),
            ),
            "A:yes:book_from_future",
        ),
    ],
)
def test_missing_ambiguous_or_nonfresh_books_fail_closed(
    replacement: object,
    reason: str,
) -> None:
    executor = _executor()
    request = executor.schedule(
        opportunity_id="opportunity-bad-book",
        opportunity=_opportunity(),
        detected_at=NOW,
    )
    books = {
        "M": book("M", yes=(("0.71", "1.00"),)),
    }
    if replacement is not None:
        books["A"] = replacement  # type: ignore[assignment]

    result = executor.execute(request, orderbooks=books)

    assert result.status is PaperExecutionStatus.FAILED
    assert result.failure_reason == reason
    assert not any(leg.actual_prices for leg in result.legs)


def test_eof_before_latency_deadline_is_distinct_from_execution_failure() -> None:
    executor = _executor(latency_ms=100)
    request = executor.schedule(
        opportunity_id="opportunity-eof",
        opportunity=_opportunity(),
        detected_at=NOW,
    )

    result = executor.insufficient_future_data(
        request,
        recorded_through=NOW + timedelta(milliseconds=99),
    )

    assert result.status is PaperExecutionStatus.INSUFFICIENT_FUTURE_DATA
    assert result.failure_reason == "insufficient_future_data"
    assert all(leg.failure_reason == "insufficient_future_data" for leg in result.legs)
    assert all(leg.fill_status == "not_filled" for leg in result.legs)
    assert result.simulated_locked_profit is None

    with pytest.raises(ValueError, match="reaches"):
        executor.insufficient_future_data(
            request,
            recorded_through=request.execute_at,
        )


def test_partial_fill_mode_and_non_stage_two_requests_are_rejected() -> None:
    with pytest.raises(ValueError, match="all-or-none"):
        PaperExecutor(
            latency_ms=100,
            stale_after=timedelta(seconds=2),
            allow_partial_fill=True,
        )

    stage_one = _opportunity().model_copy(
        update={
            "stage": OpportunityStage.GROSS_EXECUTABLE,
            "fee_status": FeeValidationStatus.NOT_EVALUATED,
            "fees": None,
            "net_profit": None,
            "net_edge": None,
            "net_state_profits": None,
            "fee_policies": (),
        }
    )
    with pytest.raises(ValidationError, match="Stage 2"):
        _executor().schedule(
            opportunity_id="opportunity-stage-one",
            opportunity=stage_one,
            detected_at=NOW,
        )


def test_persistable_leg_rejects_an_average_that_hides_a_tranche_violation() -> None:
    with pytest.raises(ValidationError, match="detection-time price limit"):
        PaperLegResult(
            ticker="A",
            side="yes",
            quantity=Decimal("2"),
            expected_prices=(
                PaperPriceFill(
                    price=Decimal("0.60"),
                    quantity=Decimal("1"),
                    cost=Decimal("0.60"),
                ),
                PaperPriceFill(
                    price=Decimal("0.90"),
                    quantity=Decimal("1"),
                    cost=Decimal("0.90"),
                ),
            ),
            actual_prices=(
                PaperPriceFill(
                    price=Decimal("0.70"),
                    quantity=Decimal("1"),
                    cost=Decimal("0.70"),
                ),
                PaperPriceFill(
                    price=Decimal("0.80"),
                    quantity=Decimal("1"),
                    cost=Decimal("0.80"),
                ),
            ),
            expected_average_price=Decimal("0.75"),
            actual_average_price=Decimal("0.75"),
            expected_cost=Decimal("1.50"),
            actual_cost=Decimal("1.50"),
            fill_status="filled",
        )


def test_persistable_survivor_requires_strictly_positive_locked_profit() -> None:
    executor = _executor()
    request = executor.schedule(
        opportunity_id="opportunity-positive",
        opportunity=_opportunity(),
        detected_at=NOW,
    )
    survived = executor.execute(
        request,
        orderbooks={
            "A": book("A", no=(("0.39", "1.00"),)),
            "M": book("M", yes=(("0.71", "1.00"),)),
        },
    )
    payload = survived.model_dump()
    payload["simulated_locked_profit"] = Decimal("0")

    with pytest.raises(ValidationError, match="positive locked profit"):
        PaperExecutionResult.model_validate(payload)


def test_execution_uses_current_fee_policies_not_detection_snapshots() -> None:
    allocations = (
        _allocation("M", "no", "0.30", "1.00"),
        _allocation("A", "yes", "0.62", "1.00"),
    )
    gross = SolverResult(
        status="optimal",
        guaranteed_gross_profit=Decimal("0.08"),
        capital_required=Decimal("0.92"),
        gross_edge=Decimal("0.08") / Decimal("0.92"),
        quantities=allocations,
        state_payouts=(Decimal("1"), Decimal("2"), Decimal("1")),
        state_profits=(Decimal("0.08"), Decimal("1.08"), Decimal("0.08")),
        min_state_profit=Decimal("0.08"),
    )
    opportunity = validate_solver_fees(
        gross,
        policies_by_ticker={"A": policy("A"), "M": policy("M")},
        minimum_net_profit=Decimal("0"),
        minimum_net_edge_bps=Decimal("0"),
    )
    executor = PaperExecutor(latency_ms=100, stale_after=timedelta(seconds=2))
    request = executor.schedule(
        opportunity_id="opportunity-fee-change",
        opportunity=opportunity,
        detected_at=NOW,
    )
    current_policies = (
        policy("A", multiplier=Decimal("0.14")),
        policy("M", multiplier=Decimal("0.14")),
    )

    result = executor.execute(
        request,
        orderbooks={
            "A": book("A", no=(("0.39", "1.00"),)),
            "M": book("M", yes=(("0.71", "1.00"),)),
        },
        fee_policies=current_policies,
    )

    assert result.execution_fee_policies == current_policies
    assert result.actual_fees == Decimal("0.004500")
    assert result.actual_fees != result.expected_fees
