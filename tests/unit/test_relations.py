"""Relation-shape, identity, and mathematical fingerprint tests."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from tests.support.relations import make_relation

from arbiter.models.relation import (
    Relation,
    RelationType,
    component_identifier,
    relations_fingerprint,
)


def test_implication_requires_explicit_distinct_endpoints() -> None:
    base = make_relation(RelationType.IMPLIES, ("A", "B")).model_dump()

    with pytest.raises(ValidationError, match="explicit antecedent"):
        Relation(**{**base, "antecedent": None})
    with pytest.raises(ValidationError, match="must differ"):
        Relation(**{**base, "antecedent": "A", "consequent": "A"})
    with pytest.raises(ValidationError, match="must match"):
        Relation(**{**base, "antecedent": "A", "consequent": "C"})


@pytest.mark.parametrize(
    ("relation_type", "members", "message"),
    [
        (RelationType.EQUIVALENT, ("A", "B", "C"), "exactly two"),
        (RelationType.MUTUALLY_EXCLUSIVE, ("A",), "at least two"),
        (RelationType.EXACTLY_ONE, ("A",), "at least two"),
    ],
)
def test_relation_shapes_reject_invalid_member_counts(
    relation_type: RelationType,
    members: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        make_relation(relation_type, members)


def test_symmetric_semantics_ignore_member_and_relation_input_order() -> None:
    first = make_relation(
        RelationType.MUTUALLY_EXCLUSIVE,
        ("C", "A", "B"),
        relation_id="first",
    )
    second = make_relation(RelationType.EQUIVALENT, ("Y", "X"), relation_id="second")
    reordered_first = make_relation(
        RelationType.MUTUALLY_EXCLUSIVE,
        ("B", "C", "A"),
        relation_id="different-id",
        rationale="different metadata",
    )
    reordered_second = make_relation(
        RelationType.EQUIVALENT,
        ("X", "Y"),
        relation_id="also-different",
        confidence=0.25,
    )

    assert relations_fingerprint((first, second)) == relations_fingerprint(
        (reordered_second, reordered_first)
    )


def test_fingerprint_ignores_metadata_but_changes_with_implication_direction() -> None:
    forward = make_relation(RelationType.IMPLIES, ("A", "B"), rationale="original")
    metadata_change = make_relation(
        RelationType.IMPLIES,
        ("A", "B"),
        relation_id="renamed",
        rationale="new explanation",
        confidence=0.5,
        created_at=datetime(2026, 1, 2, tzinfo=UTC) + timedelta(days=4),
    )
    reverse = make_relation(RelationType.IMPLIES, ("B", "A"))

    assert relations_fingerprint((forward,)) == relations_fingerprint((metadata_change,))
    assert relations_fingerprint((forward,)) != relations_fingerprint((reverse,))


def test_component_identifier_depends_only_on_canonical_member_set() -> None:
    assert component_identifier(("B", "A")) == component_identifier(("A", "B"))
    assert component_identifier(("A", "B")) != component_identifier(("A", "C"))
