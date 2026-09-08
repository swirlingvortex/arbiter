"""Worst-case-profit LP and independent verification tests."""

from decimal import Decimal

import numpy as np

from arbiter.logic.constraints import ImplicationConstraint
from arbiter.logic.worlds import enumerate_feasible_worlds
from arbiter.models.portfolio import Instrument
from arbiter.models.world import WorldSet
from arbiter.solver.lp import solve_worst_case_profit


def _worlds() -> WorldSet:
    return enumerate_feasible_worlds(("M", "A"), (ImplicationConstraint("M", "A"),))


def _instruments(
    *,
    argentina_price: Decimal = Decimal("0.62"),
    messi_no_price: Decimal = Decimal("0.30"),
    depth: Decimal = Decimal("1"),
) -> tuple[Instrument, Instrument]:
    return (
        Instrument(
            ticker="A",
            side="yes",
            price=argentina_price,
            max_quantity=depth,
            source_side="no_bid",
            source_price=Decimal("1") - argentina_price,
        ),
        Instrument(
            ticker="M",
            side="no",
            price=messi_no_price,
            max_quantity=depth,
            source_side="yes_bid",
            source_price=Decimal("1") - messi_no_price,
        ),
    )


def test_canonical_implication_arbitrage_is_exactly_eight_cents() -> None:
    result = solve_worst_case_profit(_worlds(), reversed(_instruments()))

    assert result.status == "optimal"
    assert result.is_arbitrage
    assert tuple(
        (allocation.instrument.ticker, allocation.instrument.side, allocation.quantity)
        for allocation in result.quantities
    ) == (("A", "yes", Decimal("1")), ("M", "no", Decimal("1")))
    assert result.capital_required == Decimal("0.92")
    assert result.state_payouts == (Decimal("1"), Decimal("2"), Decimal("1"))
    assert result.state_profits == (
        Decimal("0.08"),
        Decimal("1.08"),
        Decimal("0.08"),
    )
    assert result.guaranteed_gross_profit == Decimal("0.08")
    assert result.min_state_profit == min(result.state_profits)


def test_coherent_prices_return_no_arbitrage() -> None:
    result = solve_worst_case_profit(
        _worlds(),
        _instruments(argentina_price=Decimal("0.70"), messi_no_price=Decimal("0.32")),
    )

    assert result.status == "no_arbitrage"
    assert not result.is_arbitrage
    assert result.quantities == ()
    assert result.guaranteed_gross_profit == 0


def test_depth_and_capital_limit_bound_the_exact_result() -> None:
    result = solve_worst_case_profit(
        _worlds(),
        _instruments(depth=Decimal("10")),
        capital_limit=Decimal("0.46"),
    )

    assert result.status == "optimal"
    assert result.capital_required <= Decimal("0.46")
    assert Decimal("0.0399999") <= result.min_state_profit <= Decimal("0.04")
    assert all(
        allocation.quantity <= allocation.instrument.max_quantity
        for allocation in result.quantities
    )


def test_tiny_float_positive_is_rejected_by_decimal_post_verification_threshold() -> None:
    result = solve_worst_case_profit(
        _worlds(),
        _instruments(
            argentina_price=Decimal("0.699999999"),
            messi_no_price=Decimal("0.30"),
        ),
    )

    assert result.status == "no_arbitrage"
    assert result.guaranteed_gross_profit == 0
    assert Decimal(result.diagnostics["recomputed_min_profit"]) <= Decimal("0.00000001")


def test_zero_instruments_and_zero_worlds_are_explicit_statuses() -> None:
    no_instruments = solve_worst_case_profit(_worlds(), ())
    no_worlds = solve_worst_case_profit(
        WorldSet(tickers=("A",), states=np.empty((0, 1), dtype=np.int8)),
        (_instruments()[0],),
    )

    assert no_instruments.status == "no_instruments"
    assert no_instruments.state_profits == (Decimal("0"),) * 3
    assert no_worlds.status == "no_feasible_worlds"


def test_unknown_instrument_ticker_fails_closed() -> None:
    unknown = Instrument(
        ticker="UNKNOWN",
        side="yes",
        price=Decimal("0.40"),
        max_quantity=Decimal("1"),
        source_side="no_bid",
        source_price=Decimal("0.60"),
    )

    result = solve_worst_case_profit(_worlds(), (unknown,))

    assert result.status == "invalid_instrument"
    assert "UNKNOWN" in result.diagnostics["reason"]
