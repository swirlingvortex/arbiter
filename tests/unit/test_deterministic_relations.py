"""Exchange discovery and full trusted-relation validator tests."""

from datetime import UTC, datetime

from tests.support.relations import make_relation
from tests.support.thresholds import make_event

from arbiter.models.relation import RelationType
from arbiter.relations.deterministic import discover_exchange_relations
from arbiter.relations.validator import validate_relations

NOW = datetime(2026, 9, 3, tzinfo=UTC)


def test_exchange_mutual_exclusion_never_claims_exactly_one() -> None:
    event = make_event(
        markets=("C", "A", "B", "UNKNOWN"),
        mutually_exclusive=True,
    )

    batch = discover_exchange_relations(
        (event,),
        known_market_tickers={"A", "B", "C"},
        created_at=NOW,
    )

    assert len(batch.relations) == 1
    relation = batch.relations[0]
    assert relation.relation_type is RelationType.MUTUALLY_EXCLUSIVE
    assert relation.market_tickers == ("A", "B", "C")
    assert relation.verified and relation.source == "exchange_declared"
    assert "not exhaustiveness" in relation.rationale


def test_exchange_discovery_skips_true_flag_with_fewer_than_two_known_markets() -> None:
    event = make_event(markets=("A", "UNKNOWN"), mutually_exclusive=True)

    batch = discover_exchange_relations(
        (event,),
        known_market_tickers={"A"},
        created_at=NOW,
    )

    assert batch.relations == ()
    assert len(batch.skipped) == 1
    assert "fewer than two" in batch.skipped[0].reason


def test_exchange_discovery_ignores_false_or_unknown_flag() -> None:
    events = (
        make_event(ticker="FALSE", markets=("A", "B"), mutually_exclusive=False),
        make_event(ticker="NONE", markets=("A", "B"), mutually_exclusive=None),
    )

    assert (
        discover_exchange_relations(
            events,
            known_market_tickers={"A", "B"},
            created_at=NOW,
        ).relations
        == ()
    )


def test_validator_reports_missing_and_duplicate_relations() -> None:
    original = make_relation(RelationType.IMPLIES, ("A", "B"), relation_id="first")
    duplicate = make_relation(RelationType.IMPLIES, ("A", "B"), relation_id="second")
    missing = make_relation(RelationType.IMPLIES, ("A", "UNKNOWN"), relation_id="missing")

    report = validate_relations(
        (original, duplicate, missing),
        known_market_tickers={"A", "B"},
    )

    assert {issue.code for issue in report.issues} == {
        "duplicate_relation",
        "missing_market",
    }


def test_validator_reports_impossible_component() -> None:
    relations = (
        make_relation(RelationType.EXACTLY_ONE, ("A", "B")),
        make_relation(RelationType.EQUIVALENT, ("A", "B")),
    )

    report = validate_relations(relations, known_market_tickers={"A", "B"})

    assert not report.valid
    assert [issue.code for issue in report.issues] == ["impossible_component"]


def test_validator_reports_oversized_component_without_enumerating() -> None:
    members = tuple(f"M{index:02d}" for index in range(13))
    relation = make_relation(RelationType.MUTUALLY_EXCLUSIVE, members)

    report = validate_relations((relation,), known_market_tickers=set(members))

    assert [issue.code for issue in report.issues] == ["oversized_component"]


def test_validator_reports_verified_relation_from_untrusted_source() -> None:
    forged = make_relation(
        RelationType.IMPLIES,
        ("A", "B"),
        relation_id="forged-semantic",
        verified=True,
        source="semantic",
        confidence=1.0,
    )
    unreviewed = make_relation(
        RelationType.IMPLIES,
        ("C", "D"),
        relation_id="unreviewed-semantic",
        verified=False,
        source="semantic",
        confidence=1.0,
    )

    report = validate_relations(
        (forged, unreviewed),
        known_market_tickers={"A", "B", "C", "D"},
    )

    assert [(issue.code, issue.item_id) for issue in report.issues] == [
        ("untrusted_source", "forged-semantic")
    ]
