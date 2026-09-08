"""Property comparison between continuous LP and an exhaustive exact grid."""

from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support.bruteforce import solve_on_grid

from arbiter.logic.constraints import ImplicationConstraint
from arbiter.logic.worlds import enumerate_feasible_worlds
from arbiter.models.portfolio import Instrument
from arbiter.solver.lp import solve_worst_case_profit


@settings(max_examples=40, deadline=None)
@given(
    argentina_cents=st.integers(min_value=20, max_value=80),
    messi_no_cents=st.integers(min_value=10, max_value=70),
    depth=st.integers(min_value=1, max_value=3),
)
def test_lp_matches_exhaustive_solver_when_optimum_lies_on_integer_grid(
    argentina_cents: int,
    messi_no_cents: int,
    depth: int,
) -> None:
    worlds = enumerate_feasible_worlds(("M", "A"), (ImplicationConstraint("M", "A"),))
    argentina_price = Decimal(argentina_cents) / 100
    messi_no_price = Decimal(messi_no_cents) / 100
    instruments = (
        Instrument(
            ticker="A",
            side="yes",
            price=argentina_price,
            max_quantity=Decimal(depth),
            source_side="no_bid",
            source_price=Decimal("1") - argentina_price,
        ),
        Instrument(
            ticker="M",
            side="no",
            price=messi_no_price,
            max_quantity=Decimal(depth),
            source_side="yes_bid",
            source_price=Decimal("1") - messi_no_price,
        ),
    )

    exhaustive = solve_on_grid(worlds, instruments, quantum=Decimal("1"))
    solved = solve_worst_case_profit(worlds, instruments)

    expected = max(exhaustive.guaranteed_profit, Decimal("0"))
    assert solved.guaranteed_gross_profit == expected
    if expected > 0:
        assert solved.status == "optimal"
        assert solved.capital_required == exhaustive.capital_required
    else:
        assert solved.status == "no_arbitrage"
