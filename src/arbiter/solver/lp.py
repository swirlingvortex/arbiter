"""Two-phase liquidity-constrained maximum worst-case-profit solver."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from math import isfinite
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog  # type: ignore[import-untyped]

from arbiter.models.orderbook import OrderBook
from arbiter.models.portfolio import Instrument, InstrumentAllocation, SolverResult
from arbiter.models.relation import LogicalComponent
from arbiter.models.world import WorldSet
from arbiter.solver.diagnostics import PortfolioVerificationError, recompute_portfolio
from arbiter.solver.instruments import (
    build_executable_instruments,
    price_priority_violation,
    snap_quantity_down,
    sort_instruments,
)
from arbiter.solver.payoff import UnknownInstrumentTickerError, build_payoff_matrix

DEFAULT_NUMERIC_TOLERANCE = Decimal("0.00000001")
DEFAULT_MINIMUM_GROSS_PROFIT = Decimal("0")
_PHASE_EPSILON = 1e-12
_HIGHS_TOLERANCE = 1e-10


class SolverInputError(ValueError):
    """Raised when a solve request is malformed before numerical optimization."""


class _OptimizeResult(Protocol):
    """Typed subset of SciPy's otherwise untyped optimization result."""

    success: bool
    status: int
    message: str
    nit: int
    x: NDArray[np.float64] | None


def _zero_result(
    status: str,
    state_count: int,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> SolverResult:
    zeros = tuple(Decimal("0") for _ in range(state_count))
    return SolverResult(
        status=status,
        guaranteed_gross_profit=Decimal("0"),
        capital_required=Decimal("0"),
        gross_edge=None,
        quantities=(),
        state_payouts=zeros,
        state_profits=zeros,
        min_state_profit=Decimal("0"),
        diagnostics={} if diagnostics is None else diagnostics,
    )


def _status_name(result: _OptimizeResult, phase: str) -> str:
    names = {
        1: "iteration_limit",
        2: "infeasible",
        3: "unbounded",
        4: "numerical_failure",
    }
    return f"{phase}_{names.get(int(result.status), 'solver_failure')}"


def _result_diagnostics(result: _OptimizeResult) -> dict[str, Any]:
    return {
        "scipy_status": int(result.status),
        "scipy_message": str(result.message),
        "scipy_iterations": int(result.nit),
    }


def _normalize_quantity(
    value: float,
    instrument: Instrument,
    numeric_tolerance: Decimal,
) -> Decimal:
    if not isfinite(value):
        raise PortfolioVerificationError("SciPy returned a nonfinite quantity")
    tolerance = float(numeric_tolerance)
    depth = float(instrument.max_quantity)
    if value < -tolerance or value > depth + tolerance:
        raise PortfolioVerificationError("SciPy quantity violates an instrument bound")
    if abs(value) <= tolerance:
        return Decimal("0")
    # Account for one final binary-float ULP at the configured absolute boundary.
    if abs(value - depth) <= tolerance * 2:
        return instrument.max_quantity
    quantity = Decimal(str(value))
    if not quantity.is_finite():
        raise PortfolioVerificationError("normalized quantity is nonfinite")
    return quantity


def _linear_constraints(
    payoff: NDArray[np.float64],
    costs: NDArray[np.float64],
    capital_limit: Decimal | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # A q - c q >= g  becomes  (c - A) q + g <= 0.
    a_ub = np.column_stack((costs[np.newaxis, :] - payoff, np.ones(payoff.shape[0])))
    b_ub = np.zeros(payoff.shape[0], dtype=np.float64)
    if capital_limit is not None:
        capital_row = np.concatenate((costs, np.asarray([0.0], dtype=np.float64)))
        a_ub = np.vstack((a_ub, capital_row))
        b_ub = np.append(b_ub, float(capital_limit))
    return a_ub, b_ub


def _run_linprog(
    objective: NDArray[np.float64],
    *,
    a_ub: NDArray[np.float64],
    b_ub: NDArray[np.float64],
    bounds: Sequence[tuple[float | None, float | None]],
) -> _OptimizeResult:
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options={
            "primal_feasibility_tolerance": _HIGHS_TOLERANCE,
            "dual_feasibility_tolerance": _HIGHS_TOLERANCE,
        },
    )
    return cast(_OptimizeResult, result)


def solve_worst_case_profit(
    world_set: WorldSet,
    instruments: Iterable[Instrument],
    *,
    capital_limit: Decimal | None = None,
    minimum_guaranteed_profit: Decimal = DEFAULT_MINIMUM_GROSS_PROFIT,
    numeric_tolerance: Decimal = DEFAULT_NUMERIC_TOLERANCE,
    quantity_quantum: Decimal = Decimal("0.01"),
) -> SolverResult:
    """Maximize minimum gross profit, then minimize capital at that optimum.

    SciPy receives float arrays only inside this function. Every economic field returned
    for an accepted portfolio is independently reconstructed with Decimal afterward.
    """

    if numeric_tolerance <= 0 or not numeric_tolerance.is_finite():
        raise SolverInputError("numeric tolerance must be finite and positive")
    if minimum_guaranteed_profit < 0 or not minimum_guaranteed_profit.is_finite():
        raise SolverInputError("minimum guaranteed profit must be finite and nonnegative")
    if capital_limit is not None and (capital_limit < 0 or not capital_limit.is_finite()):
        raise SolverInputError("capital limit must be finite and nonnegative")
    if not quantity_quantum.is_finite() or quantity_quantum <= 0:
        raise SolverInputError("quantity quantum must be finite and positive")
    state_count = int(world_set.states.shape[0])
    if state_count == 0:
        return _zero_result("no_feasible_worlds", 0)

    ordered = sort_instruments(instruments)
    if not ordered:
        return _zero_result("no_instruments", state_count)
    try:
        payoff = build_payoff_matrix(world_set, ordered)
    except UnknownInstrumentTickerError as exc:
        return _zero_result(
            "invalid_instrument",
            state_count,
            diagnostics={"reason": str(exc)},
        )

    costs = np.asarray([float(instrument.price) for instrument in ordered], dtype=np.float64)
    if not np.all(np.isfinite(costs)):
        return _zero_result("invalid_instrument", state_count, diagnostics={"reason": "cost"})
    a_ub, b_ub = _linear_constraints(payoff, costs, capital_limit)
    bounds: list[tuple[float | None, float | None]] = [
        (0.0, float(instrument.max_quantity)) for instrument in ordered
    ]
    bounds.append((None, None))

    phase_one_objective = np.concatenate(
        (np.zeros(len(ordered), dtype=np.float64), np.asarray([-1.0]))
    )
    phase_one = _run_linprog(
        phase_one_objective,
        a_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
    )
    if not phase_one.success or phase_one.x is None:
        return _zero_result(
            _status_name(phase_one, "phase_one"),
            state_count,
            diagnostics=_result_diagnostics(phase_one),
        )
    raw_guarantee = float(phase_one.x[-1])
    if not isfinite(raw_guarantee):
        return _zero_result(
            "phase_one_nonfinite",
            state_count,
            diagnostics=_result_diagnostics(phase_one),
        )

    # Keep the primary optimum within a tiny explicit epsilon while selecting the
    # least-capital portfolio among economically equivalent solutions.
    guarantee_row = np.concatenate((np.zeros(len(ordered), dtype=np.float64), np.asarray([-1.0])))
    phase_two_a = np.vstack((a_ub, guarantee_row))
    phase_two_b = np.append(b_ub, -(raw_guarantee - _PHASE_EPSILON))
    phase_two_objective = np.concatenate((costs, np.asarray([0.0])))
    phase_two = _run_linprog(
        phase_two_objective,
        a_ub=phase_two_a,
        b_ub=phase_two_b,
        bounds=bounds,
    )
    if not phase_two.success or phase_two.x is None:
        diagnostics = _result_diagnostics(phase_two)
        diagnostics["phase_one_guarantee_float"] = raw_guarantee
        return _zero_result(
            _status_name(phase_two, "phase_two"),
            state_count,
            diagnostics=diagnostics,
        )

    try:
        continuous_quantities = tuple(
            _normalize_quantity(float(value), instrument, numeric_tolerance)
            for value, instrument in zip(phase_two.x[:-1], ordered, strict=True)
        )
        continuous_verified = recompute_portfolio(
            world_set,
            ordered,
            continuous_quantities,
        )
        quantities = tuple(
            snap_quantity_down(
                quantity,
                quantum=quantity_quantum,
                numeric_tolerance=numeric_tolerance,
            )
            for quantity in continuous_quantities
        )
        verified = recompute_portfolio(world_set, ordered, quantities)
    except PortfolioVerificationError as exc:
        return _zero_result(
            "post_verification_failed",
            state_count,
            diagnostics={
                "reason": str(exc),
                "phase_one_guarantee_float": raw_guarantee,
            },
        )

    if capital_limit is not None and verified.capital_required > capital_limit:
        return _zero_result(
            "post_verification_failed",
            state_count,
            diagnostics={"reason": "recomputed capital exceeds the configured limit"},
        )

    acceptance_threshold = max(minimum_guaranteed_profit, numeric_tolerance)
    priority_error = price_priority_violation(ordered, quantities)
    if priority_error is not None:
        return _zero_result(
            "post_verification_failed",
            state_count,
            diagnostics={"reason": priority_error},
        )
    common_diagnostics: dict[str, Any] = {
        "phase_one_guarantee_float": raw_guarantee,
        "phase_two_guarantee_float": float(phase_two.x[-1]),
        "phase_one_iterations": int(phase_one.nit),
        "phase_two_iterations": int(phase_two.nit),
        "numeric_tolerance": str(numeric_tolerance),
        "minimum_guaranteed_profit": str(minimum_guaranteed_profit),
        "quantity_quantum": str(quantity_quantum),
        "continuous_quantities": tuple(str(quantity) for quantity in continuous_quantities),
        "continuous_recomputed_min_profit": str(continuous_verified.min_state_profit),
        "recomputed_min_profit": str(verified.min_state_profit),
    }
    if verified.min_state_profit <= acceptance_threshold:
        return _zero_result("no_arbitrage", state_count, diagnostics=common_diagnostics)

    allocations = tuple(
        InstrumentAllocation(
            instrument=instrument,
            quantity=quantity,
            cost=instrument.price * quantity,
        )
        for instrument, quantity in zip(ordered, quantities, strict=True)
        if quantity > 0
    )
    edge = (
        verified.min_state_profit / verified.capital_required
        if verified.capital_required > 0
        else None
    )
    return SolverResult(
        status="optimal",
        guaranteed_gross_profit=verified.min_state_profit,
        capital_required=verified.capital_required,
        gross_edge=edge,
        quantities=allocations,
        state_payouts=verified.state_payouts,
        state_profits=verified.state_profits,
        min_state_profit=verified.min_state_profit,
        diagnostics=common_diagnostics,
    )


def solve_arbitrage(
    world_set: WorldSet,
    instruments: Iterable[Instrument],
    **kwargs: Any,
) -> SolverResult:
    """Compatibility name for the public worst-case-profit solver."""

    return solve_worst_case_profit(world_set, instruments, **kwargs)


def solve_executable_component(
    world_set: WorldSet,
    component: LogicalComponent,
    orderbooks: Mapping[str, OrderBook],
    *,
    as_of: datetime,
    stale_after: timedelta,
    capital_limit: Decimal | None = None,
    minimum_guaranteed_profit: Decimal = DEFAULT_MINIMUM_GROSS_PROFIT,
    numeric_tolerance: Decimal = DEFAULT_NUMERIC_TOLERANCE,
    quantity_quantum: Decimal = Decimal("0.01"),
) -> SolverResult:
    """Fail closed on component book state, then solve and verify executable depth."""

    built = build_executable_instruments(
        component,
        orderbooks,
        as_of=as_of,
        stale_after=stale_after,
    )
    if built.status != "ok":
        return _zero_result(
            built.status,
            int(world_set.states.shape[0]),
            diagnostics={
                "market_ticker": built.market_ticker,
                "reason": built.reason,
            },
        )
    return solve_worst_case_profit(
        world_set,
        built.instruments,
        capital_limit=capital_limit,
        minimum_guaranteed_profit=minimum_guaranteed_profit,
        numeric_tolerance=numeric_tolerance,
        quantity_quantum=quantity_quantum,
    )
