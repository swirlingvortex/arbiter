"""State-contingent terminal payoff matrix construction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from arbiter.models.portfolio import Instrument
from arbiter.models.world import WorldSet


class UnknownInstrumentTickerError(ValueError):
    """Raised when an instrument cannot be aligned to a world-set column."""


def payoff_column(world_set: WorldSet, instrument: Instrument) -> NDArray[np.float64]:
    """Build one YES or NO payout vector without changing world order."""

    try:
        column_index = world_set.tickers.index(instrument.ticker)
    except ValueError as exc:
        raise UnknownInstrumentTickerError(
            f"instrument ticker {instrument.ticker!r} is absent from the world set"
        ) from exc
    outcomes = world_set.states[:, column_index].astype(np.float64, copy=True)
    if instrument.side == "yes":
        return outcomes
    return np.subtract(1.0, outcomes)


def build_payoff_matrix(
    world_set: WorldSet,
    instruments: Sequence[Instrument],
) -> NDArray[np.float64]:
    """Construct ``A[state, instrument]`` in the supplied instrument order."""

    if not instruments:
        return np.empty((world_set.states.shape[0], 0), dtype=np.float64)
    return np.column_stack(
        tuple(payoff_column(world_set, instrument) for instrument in instruments)
    ).astype(np.float64, copy=False)
