"""Reusable implication component, world, and book fixtures for executable solving."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from arbiter.logic.worlds import generate_component_worlds
from arbiter.models.orderbook import OrderBook, PriceLevel
from arbiter.models.relation import LogicalComponent, RelationType
from arbiter.models.world import WorldSet
from arbiter.relations.graph import RelationGraph
from tests.support.relations import make_relation

NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)


def implication_component_and_worlds() -> tuple[LogicalComponent, WorldSet]:
    relation = make_relation(RelationType.IMPLIES, ("M", "A"))
    component = RelationGraph((relation,)).components[0]
    generated = generate_component_worlds(component)
    assert generated.status == "ok" and generated.world_set is not None
    return component, generated.world_set


def book(
    ticker: str,
    *,
    yes: tuple[tuple[str, str], ...] = (),
    no: tuple[tuple[str, str], ...] = (),
    local_timestamp: datetime = NOW,
    **kwargs: object,
) -> OrderBook:
    """Build a valid best-first domain book from concise decimal strings."""

    return OrderBook(
        ticker=ticker,
        yes_bids=tuple(
            PriceLevel(price=Decimal(price), quantity=Decimal(quantity)) for price, quantity in yes
        ),
        no_bids=tuple(
            PriceLevel(price=Decimal(price), quantity=Decimal(quantity)) for price, quantity in no
        ),
        local_timestamp=local_timestamp,
        **kwargs,
    )
