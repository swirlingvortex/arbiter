"""State-contingent YES/NO payoff-matrix tests."""

from decimal import Decimal

import numpy as np
import pytest

from arbiter.models.portfolio import Instrument
from arbiter.models.world import WorldSet
from arbiter.solver.payoff import UnknownInstrumentTickerError, build_payoff_matrix, payoff_column


def _instrument(ticker: str, side: str) -> Instrument:
    if side == "yes":
        return Instrument(
            ticker=ticker,
            side="yes",
            price=Decimal("0.60"),
            max_quantity=Decimal("2"),
            source_side="no_bid",
            source_price=Decimal("0.40"),
        )
    return Instrument(
        ticker=ticker,
        side="no",
        price=Decimal("0.40"),
        max_quantity=Decimal("2"),
        source_side="yes_bid",
        source_price=Decimal("0.60"),
    )


def test_yes_and_no_payoffs_follow_the_same_world_column() -> None:
    worlds = WorldSet(tickers=("A",), states=np.asarray([[0], [1]], dtype=np.int8))

    np.testing.assert_array_equal(payoff_column(worlds, _instrument("A", "yes")), [0.0, 1.0])
    np.testing.assert_array_equal(payoff_column(worlds, _instrument("A", "no")), [1.0, 0.0])


def test_payoff_matrix_preserves_state_and_instrument_order() -> None:
    worlds = WorldSet(
        tickers=("M", "A"),
        states=np.asarray(((0, 0), (0, 1), (1, 1)), dtype=np.int8),
    )
    matrix = build_payoff_matrix(worlds, (_instrument("A", "yes"), _instrument("M", "no")))

    np.testing.assert_array_equal(
        matrix,
        np.asarray(((0, 1), (1, 1), (1, 0)), dtype=np.float64),
    )
    assert matrix.shape == (3, 2)


def test_payoff_matrix_rejects_unknown_ticker() -> None:
    worlds = WorldSet(tickers=("A",), states=np.asarray([[0], [1]], dtype=np.int8))

    with pytest.raises(UnknownInstrumentTickerError, match="UNKNOWN"):
        build_payoff_matrix(worlds, (_instrument("UNKNOWN", "yes"),))


def test_empty_instrument_matrix_retains_state_dimension() -> None:
    worlds = WorldSet(tickers=("A",), states=np.asarray([[0], [1]], dtype=np.int8))

    assert build_payoff_matrix(worlds, ()).shape == (2, 0)
