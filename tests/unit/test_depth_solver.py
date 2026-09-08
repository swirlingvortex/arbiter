"""Full-depth executable solving, level priority, and exact capital tests."""

from datetime import timedelta
from decimal import Decimal

from tests.support.executable import NOW, book, implication_component_and_worlds

from arbiter.models.orderbook import BookStatus
from arbiter.solver.lp import solve_executable_component


def test_better_level_is_exhausted_before_profitable_worse_level_is_used() -> None:
    component, worlds = implication_component_and_worlds()
    result = solve_executable_component(
        worlds,
        component,
        {
            "A": book("A", no=(("0.40", "1.00"), ("0.35", "2.00"))),
            "M": book("M", yes=(("0.70", "3.00"),)),
        },
        as_of=NOW,
        stale_after=timedelta(seconds=2),
    )

    assert result.status == "optimal"
    a_yes = [
        allocation
        for allocation in result.quantities
        if allocation.instrument.ticker == "A" and allocation.instrument.side == "yes"
    ]
    assert [(item.instrument.price, item.quantity) for item in a_yes] == [
        (Decimal("0.60"), Decimal("1.00")),
        (Decimal("0.65"), Decimal("2.00")),
    ]
    assert result.guaranteed_gross_profit == Decimal("0.20")


def test_unprofitable_worse_level_remains_unused() -> None:
    component, worlds = implication_component_and_worlds()
    result = solve_executable_component(
        worlds,
        component,
        {
            "A": book("A", no=(("0.40", "1.00"), ("0.20", "2.00"))),
            "M": book("M", yes=(("0.70", "3.00"),)),
        },
        as_of=NOW,
        stale_after=timedelta(seconds=2),
    )

    assert result.status == "optimal"
    assert all(allocation.instrument.price != Decimal("0.80") for allocation in result.quantities)
    assert result.capital_required == Decimal("0.90")
    assert result.guaranteed_gross_profit == Decimal("0.10")


def test_exact_capital_cap_holds_after_quantity_snapping() -> None:
    component, worlds = implication_component_and_worlds()
    result = solve_executable_component(
        worlds,
        component,
        {
            "A": book("A", no=(("0.38", "10.00"),)),
            "M": book("M", yes=(("0.70", "10.00"),)),
        },
        as_of=NOW,
        stale_after=timedelta(seconds=2),
        capital_limit=Decimal("0.46"),
    )

    assert result.status == "optimal"
    assert result.capital_required == Decimal("0.4600")
    assert [allocation.quantity for allocation in result.quantities] == [
        Decimal("0.50"),
        Decimal("0.50"),
    ]
    assert all(
        allocation.quantity <= allocation.instrument.max_quantity
        for allocation in result.quantities
    )


def test_high_level_solver_propagates_nonfresh_book_reason_without_solving() -> None:
    component, worlds = implication_component_and_worlds()
    result = solve_executable_component(
        worlds,
        component,
        {
            "A": book(
                "A",
                no=(("0.40", "1.00"),),
                status=BookStatus.RESYNC_REQUIRED,
                status_reason="gap",
            ),
            "M": book("M", yes=(("0.70", "1.00"),)),
        },
        as_of=NOW,
        stale_after=timedelta(seconds=2),
    )

    assert result.status == "invalid_book"
    assert result.quantities == ()
    assert result.diagnostics["market_ticker"] == "A"
    assert "resynchronization" in result.diagnostics["reason"]


def test_reported_guarantee_is_minimum_recomputed_state_profit() -> None:
    component, worlds = implication_component_and_worlds()
    result = solve_executable_component(
        worlds,
        component,
        {
            "A": book("A", no=(("0.40", "1.37"),)),
            "M": book("M", yes=(("0.70", "1.37"),)),
        },
        as_of=NOW,
        stale_after=timedelta(seconds=2),
    )

    assert result.status == "optimal"
    assert result.guaranteed_gross_profit == result.min_state_profit == min(result.state_profits)
