"""Quantity-grid normalization and post-snap false-positive tests."""

from decimal import Decimal

import pytest
from tests.support.executable import implication_component_and_worlds

from arbiter.models.portfolio import Instrument
from arbiter.solver.instruments import price_priority_violation, snap_quantity_down
from arbiter.solver.lp import solve_worst_case_profit


@pytest.mark.parametrize(
    ("quantity", "expected"),
    [
        (Decimal("0.019"), Decimal("0.01")),
        (Decimal("0.009"), Decimal("0.00")),
        (Decimal("0.999999999"), Decimal("1.00")),
        (Decimal("1.000000001"), Decimal("1.00")),
        (Decimal("-0.000000001"), Decimal("0")),
    ],
)
def test_snap_quantity_down_normalizes_then_floors(
    quantity: Decimal,
    expected: Decimal,
) -> None:
    assert (
        snap_quantity_down(
            quantity,
            numeric_tolerance=Decimal("0.00000001"),
        )
        == expected
    )


def test_snap_quantity_rejects_material_negative_and_invalid_quantum() -> None:
    with pytest.raises(ValueError, match="negative"):
        snap_quantity_down(
            Decimal("-0.01"),
            numeric_tolerance=Decimal("0.00000001"),
        )
    with pytest.raises(ValueError, match="quantum"):
        snap_quantity_down(
            Decimal("1"),
            quantum=Decimal("0"),
            numeric_tolerance=Decimal("0.00000001"),
        )


def _instruments(depth: Decimal = Decimal("1")) -> tuple[Instrument, Instrument]:
    return (
        Instrument(
            ticker="A",
            side="yes",
            price=Decimal("0.62"),
            max_quantity=depth,
            source_side="no_bid",
            source_price=Decimal("0.38"),
        ),
        Instrument(
            ticker="M",
            side="no",
            price=Decimal("0.30"),
            max_quantity=depth,
            source_side="yes_bid",
            source_price=Decimal("0.70"),
        ),
    )


def test_non_grid_continuous_solution_is_floored_before_reporting() -> None:
    _, worlds = implication_component_and_worlds()
    result = solve_worst_case_profit(
        worlds,
        _instruments(),
        capital_limit=Decimal("0.01748"),
    )

    assert result.status == "optimal"
    assert [allocation.quantity for allocation in result.quantities] == [
        Decimal("0.01"),
        Decimal("0.01"),
    ]
    assert result.capital_required == Decimal("0.0092")
    assert result.guaranteed_gross_profit == Decimal("0.0008")
    assert result.diagnostics["continuous_quantities"] != ("0.01", "0.01")


def test_sub_grid_raw_opportunity_disappears_after_snapping() -> None:
    _, worlds = implication_component_and_worlds()
    result = solve_worst_case_profit(
        worlds,
        _instruments(),
        capital_limit=Decimal("0.00828"),
    )

    assert result.status == "no_arbitrage"
    assert result.quantities == ()
    assert Decimal(result.diagnostics["continuous_recomputed_min_profit"]) > 0
    assert Decimal(result.diagnostics["recomputed_min_profit"]) == 0


def test_exact_recomputed_capital_cannot_exceed_limit_within_numeric_tolerance() -> None:
    _, worlds = implication_component_and_worlds()
    result = solve_worst_case_profit(
        worlds,
        _instruments(),
        capital_limit=Decimal("0.919999999"),
    )

    assert result.status == "post_verification_failed"
    assert result.capital_required == 0
    assert result.diagnostics["reason"] == "recomputed capital exceeds the configured limit"


def test_price_priority_postcondition_detects_skipped_cheaper_level() -> None:
    better, worse = (
        Instrument(
            ticker="A",
            side="yes",
            price=price,
            max_quantity=Decimal("1"),
            source_side="no_bid",
            source_price=Decimal("1") - price,
        )
        for price in (Decimal("0.60"), Decimal("0.65"))
    )

    assert (
        price_priority_violation(
            (better, worse),
            (Decimal("0.50"), Decimal("0.10")),
        )
        is not None
    )
    assert (
        price_priority_violation(
            (better, worse),
            (Decimal("1.00"), Decimal("0.10")),
        )
        is None
    )
