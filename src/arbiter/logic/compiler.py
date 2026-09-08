"""Compile validated relation records into pure logical constraints."""

from __future__ import annotations

from collections.abc import Iterable

from arbiter.logic.constraints import (
    EquivalenceConstraint,
    ExactlyOneConstraint,
    ImplicationConstraint,
    LogicalConstraint,
    MutualExclusionConstraint,
)
from arbiter.models.relation import Relation, RelationType


def compile_relation(relation: Relation) -> LogicalConstraint:
    """Compile one relation without embedding market-specific special cases."""

    if relation.relation_type is RelationType.IMPLIES:
        if relation.antecedent is None or relation.consequent is None:
            raise ValueError("validated implication is missing directional endpoints")
        return ImplicationConstraint(relation.antecedent, relation.consequent)
    members = tuple(sorted(relation.market_tickers))
    if relation.relation_type is RelationType.EQUIVALENT:
        return EquivalenceConstraint(*members)
    if relation.relation_type is RelationType.MUTUALLY_EXCLUSIVE:
        return MutualExclusionConstraint(members)
    if relation.relation_type is RelationType.EXACTLY_ONE:
        return ExactlyOneConstraint(members)
    raise AssertionError(f"unhandled relation type: {relation.relation_type}")


def compile_relations(relations: Iterable[Relation]) -> tuple[LogicalConstraint, ...]:
    """Compile relations in stable relation-ID order."""

    return tuple(compile_relation(relation) for relation in sorted(relations, key=_relation_key))


def _relation_key(relation: Relation) -> tuple[str, str]:
    return (relation.relation_id, relation.relation_type.value)
