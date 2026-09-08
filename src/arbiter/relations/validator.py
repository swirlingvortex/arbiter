"""Fail-closed validation for persisted trusted logical relations."""

from __future__ import annotations

from dataclasses import dataclass

from arbiter.logic.worlds import WorldCache, generate_component_worlds
from arbiter.models.relation import Relation
from arbiter.relations.graph import RelationGraph
from arbiter.relations.trust import is_allowed_relation_source


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    item_id: str
    detail: str


@dataclass(frozen=True, slots=True)
class RelationValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def validate_relations(
    relations: tuple[Relation, ...],
    *,
    known_market_tickers: set[str],
    max_component_markets: int = 12,
) -> RelationValidationReport:
    """Check references, duplicates, impossible worlds, and configured component size."""

    issues: list[ValidationIssue] = []
    seen_semantics: dict[tuple[str, tuple[str, ...]], str] = {}
    graph_candidates: list[Relation] = []
    for relation in sorted(relations, key=lambda item: item.relation_id):
        if relation.verified and not is_allowed_relation_source(relation.source):
            issues.append(
                ValidationIssue(
                    "untrusted_source",
                    relation.relation_id,
                    f"verified relation source is not allowlisted: {relation.source}",
                )
            )
        missing = sorted(set(relation.market_tickers) - known_market_tickers)
        if missing:
            issues.append(
                ValidationIssue(
                    "missing_market",
                    relation.relation_id,
                    f"unknown markets: {', '.join(missing)}",
                )
            )
        else:
            graph_candidates.append(relation)
        previous = seen_semantics.get(relation.semantic_key)
        if previous is not None and previous != relation.relation_id:
            issues.append(
                ValidationIssue(
                    "duplicate_relation",
                    relation.relation_id,
                    f"duplicates mathematical semantics of {previous}",
                )
            )
        else:
            seen_semantics[relation.semantic_key] = relation.relation_id

    cache = WorldCache()
    for component in RelationGraph(graph_candidates).components:
        result = generate_component_worlds(
            component,
            max_markets=max_component_markets,
            cache=cache,
        )
        if result.status == "impossible":
            issues.append(
                ValidationIssue(
                    "impossible_component",
                    component.component_id,
                    result.reason or "component admits no feasible worlds",
                )
            )
        elif result.status == "oversized":
            issues.append(
                ValidationIssue(
                    "oversized_component",
                    component.component_id,
                    result.reason or "component exceeds configured limit",
                )
            )
    return RelationValidationReport(issues=tuple(issues))
