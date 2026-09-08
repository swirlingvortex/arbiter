"""Feasible-world generation and WorldSet invariant tests."""

import numpy as np
import pytest
from pydantic import ValidationError
from tests.support.relations import make_relation

from arbiter.logic.constraints import ImplicationConstraint
from arbiter.logic.worlds import WorldCache, enumerate_feasible_worlds, generate_component_worlds
from arbiter.models.relation import RelationType
from arbiter.models.world import WorldSet
from arbiter.relations.graph import RelationGraph


def test_canonical_implication_worlds_are_exact_and_ordered() -> None:
    worlds = enumerate_feasible_worlds(("M", "A"), (ImplicationConstraint("M", "A"),))

    assert worlds.tickers == ("M", "A")
    np.testing.assert_array_equal(
        worlds.states,
        np.asarray(((0, 0), (0, 1), (1, 1)), dtype=np.int8),
    )
    assert tuple(worlds.assignments()) == (
        {"M": 0, "A": 0},
        {"M": 0, "A": 1},
        {"M": 1, "A": 1},
    )


def test_world_enumerator_rejects_unknown_constraint_ticker() -> None:
    with pytest.raises(ValueError, match="unknown market tickers: B"):
        enumerate_feasible_worlds(("M", "A"), (ImplicationConstraint("M", "B"),))


def test_world_set_rejects_wrong_shape_and_nonbinary_values() -> None:
    with pytest.raises(ValidationError, match="two-dimensional"):
        WorldSet(tickers=("A",), states=np.asarray([0, 1], dtype=np.int8))
    with pytest.raises(ValidationError, match="binary"):
        WorldSet(tickers=("A",), states=np.asarray([[2]], dtype=np.int8))


@pytest.mark.parametrize("value", [0.9, 1.9, -0.1])
def test_world_set_rejects_values_that_would_become_binary_only_after_integer_cast(
    value: float,
) -> None:
    with pytest.raises(ValidationError, match="binary"):
        WorldSet(tickers=("A",), states=np.asarray([[value]]))


def test_world_set_owns_immutable_state_copy() -> None:
    source = np.asarray([[0], [1]], dtype=np.int8)
    worlds = WorldSet(tickers=("A",), states=source)
    source[0, 0] = 1

    assert worlds.states[0, 0] == 0
    with pytest.raises(ValueError, match="read-only"):
        worlds.states[0, 0] = 1


def test_exactly_one_of_three_has_exactly_three_worlds() -> None:
    relation = make_relation(RelationType.EXACTLY_ONE, ("C", "A", "B"))
    component = RelationGraph((relation,)).components[0]

    result = generate_component_worlds(component)

    assert result.status == "ok"
    assert result.world_set is not None
    assert result.world_set.tickers == ("A", "B", "C")
    assert result.world_set.states.tolist() == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]


def test_equivalence_has_only_equal_worlds() -> None:
    relation = make_relation(RelationType.EQUIVALENT, ("B", "A"))
    component = RelationGraph((relation,)).components[0]

    result = generate_component_worlds(component)

    assert result.status == "ok"
    assert result.world_set is not None
    assert result.world_set.states.tolist() == [[0, 0], [1, 1]]


def test_contradictory_component_is_explicitly_impossible_and_cached() -> None:
    relations = (
        make_relation(RelationType.EXACTLY_ONE, ("A", "B")),
        make_relation(RelationType.EQUIVALENT, ("A", "B")),
    )
    component = RelationGraph(relations).components[0]
    cache = WorldCache()

    first = generate_component_worlds(component, cache=cache)
    second = generate_component_worlds(component, cache=cache)

    assert first.status == "impossible"
    assert first.world_set is None
    assert second is first
    assert len(cache) == 1
    assert cache.hits == 1


def test_oversized_component_is_rejected_before_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = tuple(f"M{index:02d}" for index in range(13))
    relation = make_relation(RelationType.MUTUALLY_EXCLUSIVE, members)
    component = RelationGraph((relation,)).components[0]

    def fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("itertools.product must not run for an oversized component")

    monkeypatch.setattr("arbiter.logic.worlds.itertools.product", fail_if_called)
    result = generate_component_worlds(component, max_markets=12)

    assert result.status == "oversized"
    assert result.world_set is None
    assert "13 markets" in (result.reason or "")


def test_metadata_only_relation_change_reuses_world_cache() -> None:
    original = RelationGraph(
        (make_relation(RelationType.IMPLIES, ("A", "B"), rationale="first"),)
    ).components[0]
    changed = RelationGraph(
        (
            make_relation(
                RelationType.IMPLIES,
                ("A", "B"),
                relation_id="different",
                rationale="second",
            ),
        )
    ).components[0]
    cache = WorldCache()

    first = generate_component_worlds(original, cache=cache)
    second = generate_component_worlds(changed, cache=cache)

    assert first.status == "ok"
    assert second is first
    assert cache.hits == 1
