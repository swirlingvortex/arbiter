"""Feasible binary settlement-world models."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class WorldSet(BaseModel):
    """Binary settlement states whose columns align exactly with ``tickers``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    tickers: tuple[str, ...]
    states: NDArray[np.int8]

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require a stable, unambiguous column key for every state."""

        if any(not ticker.strip() for ticker in value):
            raise ValueError("world-set tickers must be nonempty")
        if len(set(value)) != len(value):
            raise ValueError("world-set tickers must be unique")
        return value

    @field_validator("states", mode="before")
    @classmethod
    def coerce_states(cls, value: object) -> NDArray[np.int8]:
        """Validate exact binary inputs before copying them into an integer array."""

        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError("world-set states must be a two-dimensional array")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError("world-set states must contain only binary values")
        try:
            return np.array(array, dtype=np.int8, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("world-set states must contain only binary values") from exc

    @model_validator(mode="after")
    def validate_states(self) -> WorldSet:
        """Verify shape and binary values, then prevent accidental mutation."""

        if self.states.ndim != 2:
            raise ValueError("world-set states must be a two-dimensional array")
        if self.states.shape[1] != len(self.tickers):
            raise ValueError("world-set state columns must match ticker count")
        if not np.all((self.states == 0) | (self.states == 1)):
            raise ValueError("world-set states must contain only binary values")
        self.states.setflags(write=False)
        return self

    def assignments(self) -> Iterator[dict[str, int]]:
        """Yield each state as a ticker-to-outcome mapping in stored row order."""

        for state in self.states:
            yield {ticker: int(value) for ticker, value in zip(self.tickers, state, strict=True)}


class WorldGenerationResult(BaseModel):
    """Explicit outcome for bounded component world generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    status: Literal["ok", "impossible", "oversized"]
    world_set: WorldSet | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> WorldGenerationResult:
        if self.status == "ok" and self.world_set is None:
            raise ValueError("an ok result requires a world set")
        if self.status != "ok" and self.world_set is not None:
            raise ValueError("non-ok results cannot contain a world set")
        if self.status != "ok" and not self.reason:
            raise ValueError("non-ok results require a reason")
        return self
