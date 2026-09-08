"""Logical relation records used to constrain feasible settlements."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RelationType(StrEnum):
    """Logical relations supported by the approved Arbiter architecture."""

    IMPLIES = "implies"
    EQUIVALENT = "equivalent"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    EXACTLY_ONE = "exactly_one"


class Relation(BaseModel):
    """A sourced assertion about the settlement outcomes of related markets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_id: str = Field(min_length=1)
    market_tickers: tuple[str, ...]
    relation_type: RelationType
    source: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    verified: bool
    rationale: str
    created_at: datetime
    antecedent: str | None = None
    consequent: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Relation:
        """Reject ambiguous or malformed relation shapes at the domain boundary."""

        if not self.market_tickers:
            raise ValueError("a relation must contain at least one market ticker")
        if any(not ticker.strip() for ticker in self.market_tickers):
            raise ValueError("market tickers must be nonempty")

        if self.relation_type is RelationType.IMPLIES:
            if len(self.market_tickers) != 2:
                raise ValueError("an implication must contain exactly two markets")
            if self.antecedent is None or self.consequent is None:
                raise ValueError("an implication requires explicit antecedent and consequent")
            if self.antecedent == self.consequent:
                raise ValueError("implication antecedent and consequent must differ")
            if {self.antecedent, self.consequent} != set(self.market_tickers):
                raise ValueError("implication endpoints must match its market tickers")
        elif self.antecedent is not None or self.consequent is not None:
            raise ValueError("only implications may define antecedent and consequent")
        if len(set(self.market_tickers)) != len(self.market_tickers):
            raise ValueError("market tickers must be unique")

        if self.relation_type is RelationType.EQUIVALENT and len(self.market_tickers) != 2:
            raise ValueError("an equivalence must contain exactly two markets")
        if (
            self.relation_type
            in {
                RelationType.MUTUALLY_EXCLUSIVE,
                RelationType.EXACTLY_ONE,
            }
            and len(self.market_tickers) < 2
        ):
            raise ValueError("group relations must contain at least two markets")
        return self

    @property
    def semantic_key(self) -> tuple[str, tuple[str, ...]]:
        """Return only the fields that alter the relation's mathematical meaning."""

        members: tuple[str, ...]
        if self.relation_type is RelationType.IMPLIES:
            if self.antecedent is None or self.consequent is None:
                raise ValueError("validated implication is missing directional endpoints")
            members = (self.antecedent, self.consequent)
        else:
            members = tuple(sorted(self.market_tickers))
        return (self.relation_type.value, members)


class LogicalComponent(BaseModel):
    """A deterministically identified connected set of trusted logical relations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str = Field(min_length=1)
    market_tickers: tuple[str, ...]
    relations: tuple[Relation, ...]
    relation_fingerprint: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_component(self) -> LogicalComponent:
        if not self.market_tickers:
            raise ValueError("a logical component must contain at least one market")
        if self.market_tickers != tuple(sorted(set(self.market_tickers))):
            raise ValueError("component market tickers must be unique and canonically sorted")
        members = set(self.market_tickers)
        if any(not set(relation.market_tickers) <= members for relation in self.relations):
            raise ValueError("component relations must reference only component markets")
        if any(not relation.verified for relation in self.relations):
            raise ValueError("component relations must be verified")
        return self


def component_identifier(market_tickers: tuple[str, ...]) -> str:
    """Hash only sorted member tickers so metadata changes preserve episode identity."""

    canonical = json.dumps(sorted(market_tickers), separators=(",", ":"))
    return f"component:{sha256(canonical.encode()).hexdigest()}"


def relations_fingerprint(relations: tuple[Relation, ...]) -> str:
    """Hash canonical mathematical semantics while excluding mutable metadata."""

    semantic_rows = sorted(
        (kind, list(members)) for kind, members in (r.semantic_key for r in relations)
    )
    canonical = json.dumps(semantic_rows, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()
