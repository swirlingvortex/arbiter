"""Verified logical-relation graph and deterministic connected components."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import networkx as nx

from arbiter.models.relation import (
    LogicalComponent,
    Relation,
    component_identifier,
    relations_fingerprint,
)
from arbiter.relations.trust import is_trusted_relation


class RelationGraph:
    """Connectivity index containing only verified, allowlisted logical relations."""

    def __init__(self, relations: Iterable[Relation]) -> None:
        trusted = tuple(sorted(filter(is_trusted_relation, relations), key=_key))
        graph: nx.Graph[str] = nx.Graph()
        for relation in trusted:
            members = tuple(sorted(relation.market_tickers))
            graph.add_nodes_from(members)
            graph.add_edges_from(combinations(members, 2))

        components: list[LogicalComponent] = []
        for member_set in nx.connected_components(graph):
            members = tuple(sorted(member_set))
            member_lookup = set(members)
            component_relations = tuple(
                relation
                for relation in trusted
                if set(relation.market_tickers).issubset(member_lookup)
            )
            components.append(
                LogicalComponent(
                    component_id=component_identifier(members),
                    market_tickers=members,
                    relations=component_relations,
                    relation_fingerprint=relations_fingerprint(component_relations),
                )
            )

        self._graph = graph
        self._components = tuple(sorted(components, key=lambda component: component.component_id))
        self._ticker_index = {
            ticker: component
            for component in self._components
            for ticker in component.market_tickers
        }

    @property
    def graph(self) -> nx.Graph[str]:
        """Return a defensive copy of the underlying connectivity graph."""

        return self._graph.copy()

    @property
    def components(self) -> tuple[LogicalComponent, ...]:
        """Return all components in deterministic component-ID order."""

        return self._components

    def components_for_markets(self, changed_tickers: set[str]) -> list[LogicalComponent]:
        """Return unique affected components; unknown market tickers are ignored."""

        selected = {
            component.component_id: component
            for ticker in changed_tickers
            if (component := self._ticker_index.get(ticker)) is not None
        }
        return [selected[component_id] for component_id in sorted(selected)]


def _key(relation: Relation) -> tuple[str, tuple[str, ...], str]:
    kind, members = relation.semantic_key
    return (kind, members, relation.relation_id)


def build_relation_graph(relations: Iterable[Relation]) -> RelationGraph:
    """Build the verified-and-allowlisted relation connectivity index."""

    return RelationGraph(relations)


LogicalRelationGraph = RelationGraph
TrustedRelationGraph = RelationGraph
