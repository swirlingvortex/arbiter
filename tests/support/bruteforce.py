"""Deliberately slow exhaustive portfolio validation for tiny exact fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import product

from arbiter.models.portfolio import Instrument
from arbiter.models.world import WorldSet
from arbiter.solver.diagnostics import recompute_portfolio
from arbiter.solver.instruments import sort_instruments


@dataclass(frozen=True, slots=True)
class BruteForceResult:
    """Best grid portfolio, with minimum capital breaking profit ties."""

    guaranteed_profit: Decimal
    capital_required: Decimal
    quantities: tuple[Decimal, ...]


def solve_on_grid(
    world_set: WorldSet,
    instruments: tuple[Instrument, ...],
    *,
    quantum: Decimal,
    capital_limit: Decimal | None = None,
) -> BruteForceResult:
    """Enumerate every quantity vector; suitable only for tiny test problems."""

    if quantum <= 0:
        raise ValueError("grid quantum must be positive")
    ordered = sort_instruments(instruments)
    choices: list[tuple[Decimal, ...]] = []
    for instrument in ordered:
        quotient = instrument.max_quantity / quantum
        if quotient != quotient.to_integral_value():
            raise ValueError("instrument depth must lie on the brute-force grid")
        choices.append(tuple(quantum * index for index in range(int(quotient) + 1)))

    best = BruteForceResult(
        guaranteed_profit=Decimal("-Infinity"),
        capital_required=Decimal("Infinity"),
        quantities=tuple(Decimal("0") for _ in ordered),
    )
    for quantities in product(*choices):
        verified = recompute_portfolio(world_set, ordered, quantities)
        if capital_limit is not None and verified.capital_required > capital_limit:
            continue
        if verified.min_state_profit > best.guaranteed_profit or (
            verified.min_state_profit == best.guaranteed_profit
            and verified.capital_required < best.capital_required
        ):
            best = BruteForceResult(
                guaranteed_profit=verified.min_state_profit,
                capital_required=verified.capital_required,
                quantities=quantities,
            )
    return best
