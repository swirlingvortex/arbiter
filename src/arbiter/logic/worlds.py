"""Transparent exhaustive enumeration of small feasible settlement spaces."""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import numpy as np

from arbiter.logic.compiler import compile_relations
from arbiter.logic.constraints import LogicalConstraint
from arbiter.models.relation import LogicalComponent, Relation
from arbiter.models.world import WorldGenerationResult, WorldSet


def enumerate_feasible_worlds(
    market_tickers: Iterable[str],
    constraints: Iterable[LogicalConstraint],
) -> WorldSet:
    """Enumerate binary states in caller-specified ticker order and lexicographic row order.

    The caller owns column order explicitly. Logical components introduced in Milestone 2
    provide their canonical order; preserving it here keeps the matrix-to-ticker mapping
    visible and avoids silently changing directional examples such as ``M => A``.
    """

    tickers = tuple(market_tickers)
    if len(set(tickers)) != len(tickers):
        raise ValueError("market tickers must be unique")
    compiled = tuple(constraints)
    unknown = sorted(
        {ticker for constraint in compiled for ticker in constraint.tickers} - set(tickers)
    )
    if unknown:
        raise ValueError(f"constraints reference unknown market tickers: {', '.join(unknown)}")

    rows: list[tuple[int, ...]] = []
    for bits in itertools.product((0, 1), repeat=len(tickers)):
        assignment = dict(zip(tickers, bits, strict=True))
        if all(constraint.is_satisfied(assignment) for constraint in compiled):
            rows.append(bits)
    states = np.asarray(rows, dtype=np.int8).reshape((-1, len(tickers)))
    return WorldSet(tickers=tickers, states=states)


def generate_worlds(market_tickers: Iterable[str], relations: Iterable[Relation]) -> WorldSet:
    """Compile relations and enumerate their feasible worlds."""

    return enumerate_feasible_worlds(market_tickers, compile_relations(relations))


# A concise alias for callers that already operate in the logic package.
enumerate_worlds = enumerate_feasible_worlds


class WorldCache:
    """Cache successful and impossible worlds by mathematical relation fingerprint."""

    def __init__(self) -> None:
        self._results: dict[str, WorldGenerationResult] = {}
        self.hits = 0
        self.misses = 0

    def get(self, relation_fingerprint: str) -> WorldGenerationResult | None:
        result = self._results.get(relation_fingerprint)
        if result is None:
            self.misses += 1
        else:
            self.hits += 1
        return result

    def put(self, relation_fingerprint: str, result: WorldGenerationResult) -> None:
        if result.status not in {"ok", "impossible"}:
            raise ValueError("only successful or impossible world results may be cached")
        self._results[relation_fingerprint] = result

    def __len__(self) -> int:
        return len(self._results)


def generate_component_worlds(
    component: LogicalComponent,
    *,
    max_markets: int = 12,
    cache: WorldCache | None = None,
) -> WorldGenerationResult:
    """Generate or reuse worlds while failing explicitly on unsafe component shapes."""

    if max_markets < 1:
        raise ValueError("maximum market count must be positive")
    market_count = len(component.market_tickers)
    if market_count > max_markets:
        return WorldGenerationResult(
            status="oversized",
            reason=f"component has {market_count} markets; configured maximum is {max_markets}",
        )
    if cache is not None:
        cached = cache.get(component.relation_fingerprint)
        if cached is not None:
            return cached

    worlds = generate_worlds(component.market_tickers, component.relations)
    if worlds.states.shape[0] == 0:
        result = WorldGenerationResult(
            status="impossible",
            reason="trusted constraints admit no feasible settlement world",
        )
    else:
        result = WorldGenerationResult(status="ok", world_set=worlds)
    if cache is not None:
        cache.put(component.relation_fingerprint, result)
    return result
