"""Property checks that every emitted world satisfies every trusted relation."""

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from tests.support.relations import make_relation

from arbiter.logic.compiler import compile_relations
from arbiter.logic.worlds import generate_component_worlds
from arbiter.models.relation import RelationType
from arbiter.relations.graph import RelationGraph


@settings(max_examples=40, deadline=None)
@given(
    implication=st.booleans(),
    equivalence=st.booleans(),
    mutually_exclusive=st.booleans(),
    exactly_one=st.booleans(),
)
def test_generated_worlds_are_binary_unique_and_satisfy_all_constraints(
    implication: bool,
    equivalence: bool,
    mutually_exclusive: bool,
    exactly_one: bool,
) -> None:
    assume(implication or equivalence or mutually_exclusive or exactly_one)
    relations = []
    if implication:
        relations.append(make_relation(RelationType.IMPLIES, ("A", "B"), relation_id="r1"))
    if equivalence:
        relations.append(make_relation(RelationType.EQUIVALENT, ("B", "C"), relation_id="r2"))
    if mutually_exclusive:
        relations.append(
            make_relation(RelationType.MUTUALLY_EXCLUSIVE, ("A", "C"), relation_id="r3")
        )
    if exactly_one:
        relations.append(make_relation(RelationType.EXACTLY_ONE, ("A", "B", "C"), relation_id="r4"))

    graph = RelationGraph(relations)
    for component in graph.components:
        result = generate_component_worlds(component)
        if result.status == "impossible":
            continue
        assert result.status == "ok"
        assert result.world_set is not None
        constraints = compile_relations(component.relations)
        assignments = tuple(result.world_set.assignments())
        assert len({tuple(assignment.items()) for assignment in assignments}) == len(assignments)
        assert all(value in (0, 1) for assignment in assignments for value in assignment.values())
        assert all(
            constraint.is_satisfied(assignment)
            for assignment in assignments
            for constraint in constraints
        )
