"""Positive nesting and exhaustive structured-compatibility rejection tests."""

from datetime import timedelta
from decimal import Decimal

import pytest
from tests.support.thresholds import TIMESTAMP, make_event, make_series, make_threshold_market

from arbiter.models.market import Market
from arbiter.models.relation import RelationType
from arbiter.relations.thresholds import describe_threshold, discover_threshold_relations


def test_greater_thresholds_emit_only_adjacent_high_to_low_implications() -> None:
    markets = tuple(
        make_threshold_market(f"GT-{value}", Decimal(value)) for value in (100, 110, 120)
    )

    batch = discover_threshold_relations(
        markets,
        events={"EVENT": make_event()},
        series={"SERIES": make_series()},
        created_at=TIMESTAMP,
    )

    assert {(relation.antecedent, relation.consequent) for relation in batch.relations} == {
        ("GT-120", "GT-110"),
        ("GT-110", "GT-100"),
    }
    assert all(relation.relation_type is RelationType.IMPLIES for relation in batch.relations)
    assert batch.skipped == ()


def test_less_thresholds_emit_only_adjacent_low_to_high_implications() -> None:
    markets = tuple(
        make_threshold_market(
            f"LT-{value}",
            Decimal(value),
            direction="less",
        )
        for value in (100, 110, 120)
    )

    batch = discover_threshold_relations(
        markets,
        events={"EVENT": make_event()},
        series={"SERIES": make_series()},
        created_at=TIMESTAMP,
    )

    assert {(relation.antecedent, relation.consequent) for relation in batch.relations} == {
        ("LT-100", "LT-110"),
        ("LT-110", "LT-120"),
    }


def test_equal_strikes_emit_a_deterministic_equivalence_chain() -> None:
    markets = tuple(make_threshold_market(ticker, Decimal("100")) for ticker in ("C", "A", "B"))

    batch = discover_threshold_relations(
        markets,
        events={"EVENT": make_event()},
        series={"SERIES": make_series()},
        created_at=TIMESTAMP,
    )

    assert len(batch.relations) == 2
    assert all(relation.relation_type is RelationType.EQUIVALENT for relation in batch.relations)
    assert {frozenset(relation.market_tickers) for relation in batch.relations} == {
        frozenset(("A", "B")),
        frozenset(("B", "C")),
    }


@pytest.mark.parametrize(
    "dimension",
    [
        "event",
        "series",
        "occurrence",
        "close",
        "expected_expiration",
        "expiration",
        "settlement_source_name",
        "settlement_source_url",
        "direction",
        "rules",
    ],
)
def test_each_compatibility_mismatch_prevents_relation(dimension: str) -> None:
    first = make_threshold_market("A", Decimal("100"))
    second_kwargs: dict[str, object] = {}
    events = {"EVENT": make_event()}
    series = {"SERIES": make_series()}
    if dimension == "event":
        second_kwargs["event_ticker"] = "OTHER_EVENT"
        events["OTHER_EVENT"] = make_event(ticker="OTHER_EVENT")
    elif dimension == "series":
        second_kwargs["event_ticker"] = "OTHER_EVENT"
        second_kwargs["series_ticker"] = "OTHER_SERIES"
        events["OTHER_EVENT"] = make_event(
            ticker="OTHER_EVENT",
            series_ticker="OTHER_SERIES",
        )
        series["OTHER_SERIES"] = make_series(ticker="OTHER_SERIES")
    elif dimension == "occurrence":
        second_kwargs["occurrence_datetime"] = TIMESTAMP + timedelta(seconds=1)
    elif dimension == "close":
        second_kwargs["close_time"] = TIMESTAMP + timedelta(seconds=1)
    elif dimension == "expected_expiration":
        second_kwargs["expected_expiration_time"] = TIMESTAMP + timedelta(seconds=1)
    elif dimension == "expiration":
        second_kwargs["expiration_time"] = TIMESTAMP + timedelta(seconds=1)
    elif dimension == "settlement_source_name":
        second_kwargs["event_ticker"] = "OTHER_EVENT"
        second_kwargs["series_ticker"] = "OTHER_SERIES"
        events["OTHER_EVENT"] = make_event(
            ticker="OTHER_EVENT",
            series_ticker="OTHER_SERIES",
        )
        series["OTHER_SERIES"] = make_series(
            ticker="OTHER_SERIES",
            source_name="Different Authority",
        )
    elif dimension == "settlement_source_url":
        second_kwargs["event_ticker"] = "OTHER_EVENT"
        second_kwargs["series_ticker"] = "OTHER_SERIES"
        events["OTHER_EVENT"] = make_event(
            ticker="OTHER_EVENT",
            series_ticker="OTHER_SERIES",
        )
        series["OTHER_SERIES"] = make_series(
            ticker="OTHER_SERIES",
            source_url="https://example.invalid/other",
        )
    elif dimension == "direction":
        second_kwargs["direction"] = "less"
    elif dimension == "rules":
        second_kwargs["rules_suffix"] = "A materially different settlement definition."
    second = make_threshold_market("B", Decimal("110"), **second_kwargs)

    batch = discover_threshold_relations(
        (first, second),
        events=events,
        series=series,
        created_at=TIMESTAMP,
    )

    assert batch.relations == ()


def test_settlement_source_name_and_url_are_part_of_compatibility_key() -> None:
    market = make_threshold_market("A", Decimal("100"))
    event = make_event()
    baseline = describe_threshold(market, event, make_series())
    changed_name = describe_threshold(
        market,
        event,
        make_series(source_name="Other Authority"),
    )
    changed_url = describe_threshold(
        market,
        event,
        make_series(source_url="https://example.invalid/other"),
    )

    assert baseline.descriptor is not None
    assert changed_name.descriptor is not None
    assert changed_url.descriptor is not None
    assert baseline.descriptor.compatibility_key != changed_name.descriptor.compatibility_key
    assert baseline.descriptor.compatibility_key != changed_url.descriptor.compatibility_key


@pytest.mark.parametrize(
    ("market", "reason"),
    [
        (make_threshold_market("RANGE", Decimal("100"), direction="between"), "unsupported"),
        (
            make_threshold_market("BOTH", Decimal("100"), both_bounds=True),
            "only one floor",
        ),
        (
            make_threshold_market(
                "CUSTOM",
                Decimal("100"),
                custom_strike={"range": "custom"},
            ),
            "custom/range",
        ),
        (
            make_threshold_market(
                "AMBIGUOUS",
                Decimal("100"),
                rules_suffix="The value 100 is repeated.",
            ),
            "exactly once",
        ),
        (
            make_threshold_market("MISSING_TIME", Decimal("100"), close_time=None),
            "close_time is required",
        ),
    ],
)
def test_unsafe_threshold_shapes_are_skipped(market: Market, reason: str) -> None:
    description = describe_threshold(
        market,
        make_event(),
        make_series(),
    )

    assert description.descriptor is None
    assert reason in (description.reason or "")


def test_missing_event_or_series_is_reported_as_skip() -> None:
    market = make_threshold_market("A", Decimal("100"))

    missing_event = discover_threshold_relations(
        (market,),
        events={},
        series={},
        created_at=TIMESTAMP,
    )
    missing_series = discover_threshold_relations(
        (market,),
        events={"EVENT": make_event()},
        series={},
        created_at=TIMESTAMP,
    )

    assert "missing parent event" in missing_event.skipped[0].reason
    assert "missing parent series" in missing_series.skipped[0].reason
