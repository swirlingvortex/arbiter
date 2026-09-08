"""Exchange-independent payoff construction and worst-case-profit optimization."""

from arbiter.solver.fees import (
    FeeRoundingLedger,
    KalshiFeeModel,
    ZeroFeeModel,
    quote_order,
    resolve_fee_policy,
    validate_solver_fees,
)
from arbiter.solver.instruments import (
    build_executable_instruments,
    snap_quantity_down,
    sort_instruments,
)
from arbiter.solver.lp import solve_arbitrage, solve_executable_component, solve_worst_case_profit
from arbiter.solver.payoff import build_payoff_matrix

__all__ = [
    "build_executable_instruments",
    "build_payoff_matrix",
    "FeeRoundingLedger",
    "KalshiFeeModel",
    "quote_order",
    "resolve_fee_policy",
    "solve_arbitrage",
    "solve_executable_component",
    "solve_worst_case_profit",
    "sort_instruments",
    "snap_quantity_down",
    "validate_solver_fees",
    "ZeroFeeModel",
]
