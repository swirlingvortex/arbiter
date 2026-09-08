"""Conservative structured threshold nesting discovery."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Literal, cast

from arbiter.models.event import Event
from arbiter.models.market import Market
from arbiter.models.relation import Relation, RelationType
from arbiter.models.series import Series
from arbiter.relations.deterministic import (
    DiscoveryBatch,
    DiscoveryDiagnostic,
    canonical_relation_id,
)

ThresholdDirection = Literal["greater", "less"]
_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?\$?[0-9][0-9,]*(?:\.[0-9]+)?(?![A-Za-z0-9])")


@dataclass(frozen=True, slots=True)
class ThresholdDescriptor:
    """One safe one-sided threshold and its full structured compatibility signature."""

    ticker: str
    direction: ThresholdDirection
    threshold: Decimal
    compatibility_key: str


@dataclass(frozen=True, slots=True)
class ThresholdDescription:
    """Either a usable descriptor or an explicit conservative skip reason."""

    descriptor: ThresholdDescriptor | None
    reason: str | None


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    return format(normalized, "f")


def _replace_strike(rules: tuple[str | None, str | None], strike: Decimal) -> tuple[str, str]:
    normalized_rules = tuple((rule or "").strip() for rule in rules)
    if not all(normalized_rules):
        raise ValueError("primary and secondary resolution rules are required")
    matches: list[tuple[int, int, int]] = []
    for rule_index, rule in enumerate(normalized_rules):
        for match in _NUMBER.finditer(rule):
            token = match.group().replace("$", "").replace(",", "")
            try:
                value = Decimal(token)
            except InvalidOperation:
                continue
            if value == strike:
                matches.append((rule_index, match.start(), match.end()))
    if len(matches) != 1:
        raise ValueError("structured strike must occur exactly once across the resolution rules")
    rule_index, start, end = matches[0]
    output = list(normalized_rules)
    output[rule_index] = f"{output[rule_index][:start]}<STRIKE>{output[rule_index][end:]}"
    return (" ".join(output[0].split()).casefold(), " ".join(output[1].split()).casefold())


def _settlement_signature(series: Series) -> tuple[tuple[str, str], ...]:
    sources = tuple(
        sorted(
            (source.name.strip().casefold(), source.url.strip().casefold())
            for source in series.settlement_sources
        )
    )
    if not sources or any(not name or not url for name, url in sources):
        raise ValueError("complete settlement-source names and URLs are required")
    return sources


def _time_text(value: datetime | None, label: str) -> str:
    if value is None:
        raise ValueError(f"{label} is required")
    return value.isoformat()


def describe_threshold(market: Market, event: Event, series: Series) -> ThresholdDescription:
    """Build a descriptor only for an unambiguous, fully compatible one-sided market."""

    if market.event_ticker != event.ticker:
        return ThresholdDescription(None, "market/event identity mismatch")
    if market.series_ticker is None or market.series_ticker != event.series_ticker:
        return ThresholdDescription(None, "market/event series identity mismatch")
    if event.series_ticker != series.ticker:
        return ThresholdDescription(None, "event/series identity mismatch")
    if market.strike_type not in {"greater", "less"}:
        return ThresholdDescription(None, "unsupported non-one-sided strike type")
    direction = cast(ThresholdDirection, market.strike_type)
    if direction == "greater":
        if market.floor_strike is None or market.cap_strike is not None:
            return ThresholdDescription(None, "greater market needs only one floor strike")
        threshold = market.floor_strike
    else:
        if market.cap_strike is None or market.floor_strike is not None:
            return ThresholdDescription(None, "less market needs only one cap strike")
        threshold = market.cap_strike
    if not threshold.is_finite():
        return ThresholdDescription(None, "threshold must be finite")
    if market.custom_strike:
        return ThresholdDescription(None, "custom/range strike metadata is unsupported")
    if market.functional_strike not in {None, "", direction}:
        return ThresholdDescription(None, "functional strike conflicts with one-sided direction")
    try:
        normalized_rules = _replace_strike(
            (market.rules_primary, market.rules_secondary),
            threshold,
        )
        payload = {
            "series_ticker": series.ticker,
            "event_ticker": event.ticker,
            "occurrence_datetime": _time_text(
                market.occurrence_datetime,
                "occurrence_datetime",
            ),
            "close_time": _time_text(market.close_time, "close_time"),
            "expected_expiration_time": _time_text(
                market.expected_expiration_time,
                "expected_expiration_time",
            ),
            "expiration_time": _time_text(market.expiration_time, "expiration_time"),
            "settlement_sources": _settlement_signature(series),
            "direction": direction,
            "rules": normalized_rules,
        }
    except ValueError as exc:
        return ThresholdDescription(None, str(exc))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return ThresholdDescription(
        ThresholdDescriptor(
            ticker=market.ticker,
            direction=direction,
            threshold=threshold,
            compatibility_key=sha256(canonical.encode()).hexdigest(),
        ),
        None,
    )


def discover_threshold_relations(
    markets: tuple[Market, ...],
    *,
    events: dict[str, Event],
    series: dict[str, Series],
    created_at: datetime,
) -> DiscoveryBatch:
    """Discover only adjacent nested implications and equal-strike equivalence chains."""

    groups: dict[str, list[ThresholdDescriptor]] = defaultdict(list)
    skipped: list[DiscoveryDiagnostic] = []
    for market in sorted(markets, key=lambda item: item.ticker):
        event = events.get(market.event_ticker)
        if event is None:
            skipped.append(
                DiscoveryDiagnostic("threshold", market.ticker, "missing parent event metadata")
            )
            continue
        if (
            event.series_ticker is None
            or (parent_series := series.get(event.series_ticker)) is None
        ):
            skipped.append(
                DiscoveryDiagnostic("threshold", market.ticker, "missing parent series metadata")
            )
            continue
        description = describe_threshold(market, event, parent_series)
        if description.descriptor is None:
            skipped.append(
                DiscoveryDiagnostic(
                    "threshold",
                    market.ticker,
                    description.reason or "unsupported threshold",
                )
            )
            continue
        groups[description.descriptor.compatibility_key].append(description.descriptor)

    relations: list[Relation] = []
    for descriptors in groups.values():
        by_threshold: dict[Decimal, list[ThresholdDescriptor]] = defaultdict(list)
        for descriptor in descriptors:
            by_threshold[descriptor.threshold].append(descriptor)
        thresholds = sorted(by_threshold)
        representatives: list[ThresholdDescriptor] = []
        for threshold in thresholds:
            equal_group = sorted(by_threshold[threshold], key=lambda item: item.ticker)
            representatives.append(equal_group[0])
            for left, right in zip(equal_group, equal_group[1:], strict=False):
                members = (left.ticker, right.ticker)
                relations.append(
                    Relation(
                        relation_id=canonical_relation_id(
                            "threshold",
                            RelationType.EQUIVALENT,
                            members,
                        ),
                        market_tickers=members,
                        relation_type=RelationType.EQUIVALENT,
                        source="threshold_deterministic",
                        confidence=1.0,
                        verified=True,
                        rationale=(
                            f"Compatible {left.direction} markets share threshold "
                            f"{_decimal_text(threshold)}."
                        ),
                        created_at=created_at,
                    )
                )
        for lower, higher in zip(representatives, representatives[1:], strict=False):
            antecedent, consequent = (
                (higher, lower) if lower.direction == "greater" else (lower, higher)
            )
            members = (antecedent.ticker, consequent.ticker)
            relations.append(
                Relation(
                    relation_id=canonical_relation_id(
                        "threshold",
                        RelationType.IMPLIES,
                        members,
                        antecedent=antecedent.ticker,
                        consequent=consequent.ticker,
                    ),
                    market_tickers=members,
                    relation_type=RelationType.IMPLIES,
                    antecedent=antecedent.ticker,
                    consequent=consequent.ticker,
                    source="threshold_deterministic",
                    confidence=1.0,
                    verified=True,
                    rationale=(
                        f"Compatible {lower.direction} threshold nesting: "
                        f"{_decimal_text(antecedent.threshold)} implies "
                        f"{_decimal_text(consequent.threshold)}."
                    ),
                    created_at=created_at,
                )
            )
    return DiscoveryBatch(
        relations=tuple(sorted(relations, key=lambda relation: relation.relation_id)),
        skipped=tuple(skipped),
    )
