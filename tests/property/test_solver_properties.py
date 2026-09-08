"""Property checks for snapped depth, capital, payoff, and guarantee consistency."""

from decimal import Decimal

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from tests.support.executable import implication_component_and_worlds

from arbiter.models.portfolio import Instrument
from arbiter.solver.diagnostics import recompute_portfolio
from arbiter.solver.instruments import sort_instruments
from arbiter.solver.lp import solve_worst_case_profit


@settings(max_examples=50, deadline=None)
@given(
    argentina_cents=st.integers(min_value=20, max_value=80),
    messi_no_cents=st.integers(min_value=10, max_value=70),
    depth_units=st.integers(min_value=1, max_value=200),
    capital_cents=st.integers(min_value=1, max_value=300),
)
def test_every_accepted_solver_result_is_exactly_executable(
    argentina_cents: int,
    messi_no_cents: int,
    depth_units: int,
    capital_cents: int,
) -> None:
    assume(argentina_cents + messi_no_cents < 100)
    _, worlds = implication_component_and_worlds()
    argentina_price = Decimal(argentina_cents) / 100
    messi_price = Decimal(messi_no_cents) / 100
    depth = Decimal(depth_units) / 100
    instruments = (
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
            price=messi_price,
            max_quantity=depth,
            source_side="yes_bid",
            source_price=Decimal("1") - messi_price,
        ),
    )
    capital_limit = Decimal(capital_cents) / 100
    result = solve_worst_case_profit(
        worlds,
        instruments,
        capital_limit=capital_limit,
    )

    if result.status != "optimal":
        assert result.quantities == ()
        return
    allocations = {
        (allocation.instrument.ticker, allocation.instrument.side): allocation
        for allocation in result.quantities
    }
    ordered = sort_instruments(instruments)
    quantities = tuple(
        allocations.get((instrument.ticker, instrument.side)).quantity
        if (instrument.ticker, instrument.side) in allocations
        else Decimal("0")
        for instrument in ordered
    )
    assert all(quantity % Decimal("0.01") == 0 for quantity in quantities)
    assert all(
        quantity <= instrument.max_quantity
        for quantity, instrument in zip(quantities, ordered, strict=True)
    )
    verified = recompute_portfolio(worlds, ordered, quantities)
    assert verified.capital_required <= capital_limit
    assert all(payout >= 0 for payout in verified.state_payouts)
    assert result.capital_required == verified.capital_required
    assert result.state_payouts == verified.state_payouts
    assert result.state_profits == verified.state_profits
    assert result.guaranteed_gross_profit == min(verified.state_profits)
