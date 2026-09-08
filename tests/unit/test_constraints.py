"""Truth-table and compiler tests for pure logical constraints."""

from datetime import UTC, datetime

import pytest

from arbiter.logic.compiler import compile_relation, compile_relations
from arbiter.logic.constraints import (
    EquivalenceConstraint,
    ExactlyOneConstraint,
    ImplicationConstraint,
    MutualExclusionConstraint,
)
from arbiter.models.relation import Relation, RelationType


def _relation(
    relation_id: str = "relation:m-implies-a",
    *,
    relation_type: RelationType = RelationType.IMPLIES,
) -> Relation:
    kwargs: dict[str, str] = {}
    if relation_type is RelationType.IMPLIES:
        kwargs = {"antecedent": "M", "consequent": "A"}
    return Relation(
        relation_id=relation_id,
        market_tickers=("M", "A"),
        relation_type=relation_type,
        source="manual",
        confidence=1.0,
        verified=True,
        rationale="fixture",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("messi", "argentina", "expected"),
    [(0, 0, True), (0, 1, True), (1, 0, False), (1, 1, True)],
)
def test_implication_truth_table(messi: int, argentina: int, expected: bool) -> None:
    constraint = ImplicationConstraint("M", "A")

    assert constraint.is_satisfied({"M": messi, "A": argentina}) is expected


def test_implication_rejects_incomplete_or_nonbinary_assignment() -> None:
    constraint = ImplicationConstraint("M", "A")

    with pytest.raises(KeyError, match="missing ticker"):
        constraint.is_satisfied({"M": 1})
    with pytest.raises(ValueError, match="binary"):
        constraint.is_satisfied({"M": 2, "A": 1})


def test_compiler_preserves_explicit_implication_direction() -> None:
    compiled = compile_relation(_relation())

    assert isinstance(compiled, ImplicationConstraint)
    assert compiled.antecedent == "M"
    assert compiled.consequent == "A"


def test_compile_relations_uses_stable_relation_id_order() -> None:
    later = _relation("z-last")
    earlier = Relation(
        **{
            **_relation("a-first").model_dump(),
            "market_tickers": ("X", "Y"),
            "antecedent": "X",
            "consequent": "Y",
        }
    )

    compiled = compile_relations((later, earlier))

    assert tuple(constraint.tickers for constraint in compiled) == (("X", "Y"), ("M", "A"))


@pytest.mark.parametrize(
    ("relation_type", "constraint_type"),
    [
        (RelationType.EQUIVALENT, EquivalenceConstraint),
        (RelationType.MUTUALLY_EXCLUSIVE, MutualExclusionConstraint),
        (RelationType.EXACTLY_ONE, ExactlyOneConstraint),
    ],
)
def test_compiler_supports_every_nondirectional_relation_type(
    relation_type: RelationType,
    constraint_type: type[object],
) -> None:
    assert isinstance(compile_relation(_relation(relation_type=relation_type)), constraint_type)


@pytest.mark.parametrize(
    ("constraint", "accepted"),
    [
        (EquivalenceConstraint("A", "B"), {(0, 0), (1, 1)}),
        (MutualExclusionConstraint(("A", "B")), {(0, 0), (0, 1), (1, 0)}),
        (ExactlyOneConstraint(("A", "B")), {(0, 1), (1, 0)}),
    ],
)
def test_group_constraint_truth_tables(
    constraint: EquivalenceConstraint | MutualExclusionConstraint | ExactlyOneConstraint,
    accepted: set[tuple[int, int]],
) -> None:
    actual = {
        (left, right)
        for left in (0, 1)
        for right in (0, 1)
        if constraint.is_satisfied({"A": left, "B": right})
    }

    assert actual == accepted
