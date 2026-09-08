"""Shared deterministic relation identities and exchange-declared discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256

from arbiter.models.event import Event
from arbiter.models.relation import Relation, RelationType


@dataclass(frozen=True, slots=True)
class DiscoveryDiagnostic:
    """A market/event deliberately skipped instead of becoming a trusted relation."""

    source: str
    item_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class DiscoveryBatch:
    """Trusted deterministic relations plus transparent skip diagnostics."""

    relations: tuple[Relation, ...]
    skipped: tuple[DiscoveryDiagnostic, ...] = ()


def canonical_relation_id(
    prefix: str,
    relation_type: RelationType,
    market_tickers: tuple[str, ...],
    *,
    antecedent: str | None = None,
    consequent: str | None = None,
) -> str:
    """Build a stable ID from mathematical semantics, preserving implication direction."""

    members: tuple[str, ...]
    if relation_type is RelationType.IMPLIES:
        if antecedent is None or consequent is None:
            raise ValueError("implication identity requires explicit endpoints")
        members = (antecedent, consequent)
    else:
        members = tuple(sorted(market_tickers))
    canonical = json.dumps(
        [relation_type.value, list(members)],
        separators=(",", ":"),
    )
    return f"{prefix}:{sha256(canonical.encode()).hexdigest()}"


def discover_exchange_relations(
    events: tuple[Event, ...],
    *,
    known_market_tickers: set[str],
    created_at: datetime | None = None,
) -> DiscoveryBatch:
    """Trust only exchange-declared at-most-one groups with at least two known markets."""

    timestamp = datetime.now(UTC) if created_at is None else created_at
    relations: list[Relation] = []
    skipped: list[DiscoveryDiagnostic] = []
    for event in sorted(events, key=lambda item: item.ticker):
        if event.mutually_exclusive is not True:
            continue
        members = tuple(sorted(set(event.market_tickers) & known_market_tickers))
        if len(members) < 2:
            skipped.append(
                DiscoveryDiagnostic(
                    source="exchange_declared",
                    item_id=event.ticker,
                    reason="mutually exclusive event has fewer than two known markets",
                )
            )
            continue
        relations.append(
            Relation(
                relation_id=canonical_relation_id(
                    "exchange",
                    RelationType.MUTUALLY_EXCLUSIVE,
                    members,
                ),
                market_tickers=members,
                relation_type=RelationType.MUTUALLY_EXCLUSIVE,
                source="exchange_declared",
                confidence=1.0,
                verified=True,
                rationale=(
                    f"Kalshi event {event.ticker} explicitly declares its markets mutually "
                    "exclusive; this establishes at most one YES, not exhaustiveness."
                ),
                created_at=timestamp,
            )
        )
    return DiscoveryBatch(relations=tuple(relations), skipped=tuple(skipped))
