"""Factories for concise, deterministic relation fixtures."""

from __future__ import annotations

from datetime import UTC, datetime

from arbiter.models.relation import Relation, RelationType


def make_relation(
    relation_type: RelationType,
    market_tickers: tuple[str, ...],
    *,
    relation_id: str | None = None,
    verified: bool = True,
    source: str = "manual",
    rationale: str = "fixture",
    confidence: float | None = 1.0,
    created_at: datetime | None = None,
) -> Relation:
    """Create one fully sourced relation with explicit implication direction."""

    directional: dict[str, str] = {}
    if relation_type is RelationType.IMPLIES:
        directional = {
            "antecedent": market_tickers[0],
            "consequent": market_tickers[1],
        }
    identifier = relation_id or f"test:{relation_type.value}:{'-'.join(market_tickers)}"
    return Relation(
        relation_id=identifier,
        market_tickers=market_tickers,
        relation_type=relation_type,
        source=source,
        confidence=confidence,
        verified=verified,
        rationale=rationale,
        created_at=created_at or datetime(2026, 1, 1, tzinfo=UTC),
        **directional,
    )
