"""Entirely offline canonical implication-arbitrage demonstration."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from rich.console import Console
from rich.table import Table

from arbiter.logic.compiler import compile_relation
from arbiter.logic.worlds import enumerate_feasible_worlds
from arbiter.models.portfolio import Instrument, SolverResult
from arbiter.models.relation import Relation, RelationType
from arbiter.solver.diagnostics import decimal_text
from arbiter.solver.lp import solve_worst_case_profit

MESSI = "MESSI_SCORES"
ARGENTINA = "ARGENTINA_SCORES"


def canonical_demo_result() -> tuple[Relation, SolverResult, tuple[dict[str, int], ...]]:
    """Build and solve the generic synthetic fixture required by the specification."""

    relation = Relation(
        relation_id="demo:messi-implies-argentina",
        market_tickers=(MESSI, ARGENTINA),
        relation_type=RelationType.IMPLIES,
        antecedent=MESSI,
        consequent=ARGENTINA,
        source="manual",
        confidence=1.0,
        verified=True,
        rationale="If Messi scores, Argentina necessarily scores.",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    worlds = enumerate_feasible_worlds(
        relation.market_tickers,
        (compile_relation(relation),),
    )
    instruments = (
        Instrument(
            ticker=ARGENTINA,
            side="yes",
            price=Decimal("0.62"),
            max_quantity=Decimal("1"),
            source_side="no_bid",
            source_price=Decimal("0.38"),
        ),
        Instrument(
            ticker=MESSI,
            side="no",
            price=Decimal("0.30"),
            max_quantity=Decimal("1"),
            source_side="yes_bid",
            source_price=Decimal("0.70"),
        ),
    )
    result = solve_worst_case_profit(worlds, instruments)
    return relation, result, tuple(worlds.assignments())


def render_demo(console: Console | None = None) -> SolverResult:
    """Print the relation, worlds, chosen portfolio, and exact canonical economics."""

    output = Console() if console is None else console
    relation, result, assignments = canonical_demo_result()
    if not result.is_arbitrage:
        raise RuntimeError(f"canonical demo failed with solver status {result.status!r}")

    output.print("[bold]Canonical implication arbitrage[/bold]")
    output.print(f"Logical relation: {relation.antecedent} => {relation.consequent}")

    world_table = Table(title="Feasible worlds")
    world_table.add_column("State")
    world_table.add_column("Messi scores", justify="right")
    world_table.add_column("Argentina scores", justify="right")
    for index, assignment in enumerate(assignments, start=1):
        world_table.add_row(
            f"S{index}",
            str(assignment[MESSI]),
            str(assignment[ARGENTINA]),
        )
    output.print(world_table)

    portfolio_table = Table(title="Chosen portfolio")
    portfolio_table.add_column("Instrument")
    portfolio_table.add_column("Quantity", justify="right")
    portfolio_table.add_column("Price", justify="right")
    portfolio_table.add_column("Cost", justify="right")
    for allocation in result.quantities:
        portfolio_table.add_row(
            f"{allocation.instrument.ticker} {allocation.instrument.side.upper()}",
            decimal_text(allocation.quantity),
            decimal_text(allocation.instrument.price),
            decimal_text(allocation.cost),
        )
    output.print(portfolio_table)

    payout_text = ", ".join(decimal_text(value) for value in result.state_payouts)
    output.print(f"Cost: {decimal_text(result.capital_required)}")
    output.print(f"State payouts: {payout_text}")
    output.print(f"Worst-case payout: {decimal_text(min(result.state_payouts))}")
    output.print(f"Gross guaranteed profit: {decimal_text(result.guaranteed_gross_profit)}")
    return result
