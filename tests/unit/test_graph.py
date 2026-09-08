"""Verified relation graph and affected-component indexing tests."""

from tests.support.relations import make_relation

from arbiter.models.relation import RelationType
from arbiter.relations.graph import RelationGraph


def test_disconnected_relations_form_two_deterministic_components() -> None:
    implication = make_relation(RelationType.IMPLIES, ("A", "B"), relation_id="z")
    equivalence = make_relation(RelationType.EQUIVALENT, ("D", "C"), relation_id="a")

    graph = RelationGraph((implication, equivalence))

    assert {component.market_tickers for component in graph.components} == {
        ("A", "B"),
        ("C", "D"),
    }
    assert all(
        component.market_tickers == tuple(sorted(component.market_tickers))
        for component in graph.components
    )


def test_multi_market_group_is_connected_as_one_component() -> None:
    relation = make_relation(RelationType.EXACTLY_ONE, ("C", "A", "B"))

    graph = RelationGraph((relation,))

    assert len(graph.components) == 1
    assert graph.components[0].market_tickers == ("A", "B", "C")
    assert graph.graph.number_of_edges() == 3


def test_unverified_relations_never_enter_trusted_graph() -> None:
    trusted = make_relation(RelationType.IMPLIES, ("A", "B"))
    suggestion = make_relation(
        RelationType.IMPLIES,
        ("X", "Y"),
        relation_id="semantic-suggestion",
        verified=False,
        source="semantic",
    )

    graph = RelationGraph((suggestion, trusted))

    assert tuple(component.market_tickers for component in graph.components) == (("A", "B"),)
    assert graph.components_for_markets({"X", "Y"}) == []


def test_semantic_confidence_cannot_bypass_source_review_boundary() -> None:
    forged = make_relation(
        RelationType.IMPLIES,
        ("A", "B"),
        relation_id="semantic-high-confidence",
        verified=True,
        source="semantic",
        confidence=1.0,
    )
    reviewed = make_relation(
        RelationType.IMPLIES,
        ("X", "Y"),
        relation_id="semantic-reviewed",
        verified=True,
        source="semantic_verified",
        confidence=0.1,
    )

    graph = RelationGraph((forged, reviewed))

    assert tuple(component.market_tickers for component in graph.components) == (("X", "Y"),)
    assert graph.components_for_markets({"A", "B"}) == []
    assert graph.components_for_markets({"X"})[0].relations == (reviewed,)


def test_components_for_markets_is_unique_sorted_and_ignores_unknowns() -> None:
    first = make_relation(RelationType.IMPLIES, ("A", "B"))
    second = make_relation(RelationType.EQUIVALENT, ("C", "D"))
    graph = RelationGraph((first, second))

    affected = graph.components_for_markets({"UNKNOWN", "B", "A", "D"})

    assert len(affected) == 2
    assert [component.component_id for component in affected] == sorted(
        component.component_id for component in affected
    )


def test_metadata_only_changes_preserve_component_and_relation_identity() -> None:
    original = RelationGraph((make_relation(RelationType.IMPLIES, ("A", "B")),)).components[0]
    changed = RelationGraph(
        (
            make_relation(
                RelationType.IMPLIES,
                ("A", "B"),
                relation_id="new-record-id",
                rationale="new rationale",
                confidence=0.1,
            ),
        )
    ).components[0]

    assert changed.component_id == original.component_id
    assert changed.relation_fingerprint == original.relation_fingerprint
