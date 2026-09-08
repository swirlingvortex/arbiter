"""Strict manual YAML and canonical trusted-ID tests."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from arbiter.models.relation import RelationType
from arbiter.relations.manual import ManualRelationError, load_manual_relations

NOW = datetime(2026, 9, 3, tzinfo=UTC)


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_manual_file_parses_all_documented_shapes_as_verified(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "relations.yaml",
        """
relations:
  - type: implies
    antecedent: A
    consequent: B
    rationale: A always entails B.
  - type: equivalent
    markets: [C, D]
    rationale: C and D settle together.
  - type: mutually_exclusive
    markets: [E, F, G]
    rationale: At most one settles YES.
  - type: exactly_one
    relation_id: supplied-id
    markets: [H, I]
    rationale: Exactly one settles YES.
""",
    )

    relations = load_manual_relations(
        path,
        known_market_tickers=set("ABCDEFGHI"),
        created_at=NOW,
    )

    assert {relation.relation_type for relation in relations} == set(RelationType)
    assert all(relation.verified and relation.source == "manual" for relation in relations)
    assert all(relation.created_at == NOW for relation in relations)
    implication = next(
        relation for relation in relations if relation.relation_type is RelationType.IMPLIES
    )
    assert implication.antecedent == "A" and implication.consequent == "B"
    assert any(relation.relation_id == "supplied-id" for relation in relations)


def test_manual_symmetric_id_is_stable_under_member_reordering(tmp_path: Path) -> None:
    first = _write(
        tmp_path / "first.yaml",
        "relations:\n"
        "  - type: mutually_exclusive\n"
        "    markets: [C, A, B]\n"
        "    rationale: First text.\n",
    )
    second = _write(
        tmp_path / "second.yaml",
        "relations:\n"
        "  - type: mutually_exclusive\n"
        "    markets: [B, C, A]\n"
        "    rationale: Changed text.\n",
    )

    first_id = load_manual_relations(
        first,
        known_market_tickers={"A", "B", "C"},
        created_at=NOW,
    )[0].relation_id
    second_id = load_manual_relations(
        second,
        known_market_tickers={"A", "B", "C"},
        created_at=NOW,
    )[0].relation_id

    assert first_id == second_id
    assert first_id.startswith("manual:")


def test_manual_implication_id_changes_when_direction_reverses(tmp_path: Path) -> None:
    forward = _write(
        tmp_path / "forward.yaml",
        "relations:\n"
        "  - type: implies\n"
        "    antecedent: A\n"
        "    consequent: B\n"
        "    rationale: Forward.\n",
    )
    reverse = _write(
        tmp_path / "reverse.yaml",
        "relations:\n"
        "  - type: implies\n"
        "    antecedent: B\n"
        "    consequent: A\n"
        "    rationale: Reverse.\n",
    )

    assert (
        load_manual_relations(
            forward,
            known_market_tickers={"A", "B"},
            created_at=NOW,
        )[0].relation_id
        != load_manual_relations(
            reverse,
            known_market_tickers={"A", "B"},
            created_at=NOW,
        )[0].relation_id
    )


def test_manual_file_rejects_unknown_fields_and_markets(tmp_path: Path) -> None:
    extra = _write(
        tmp_path / "extra.yaml",
        "relations:\n"
        "  - type: implies\n"
        "    antecedent: A\n"
        "    consequent: B\n"
        "    rationale: Fixture.\n"
        "    guessed_probability: 0.9\n",
    )
    missing = _write(
        tmp_path / "missing.yaml",
        "relations:\n"
        "  - type: implies\n"
        "    antecedent: A\n"
        "    consequent: UNKNOWN\n"
        "    rationale: Fixture.\n",
    )

    with pytest.raises(ManualRelationError, match="extra_forbidden"):
        load_manual_relations(extra, known_market_tickers={"A", "B"})
    with pytest.raises(ManualRelationError, match="unknown markets: UNKNOWN"):
        load_manual_relations(missing, known_market_tickers={"A", "B"})


def test_manual_file_rejects_ambiguous_relation_shape(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "ambiguous.yaml",
        "relations:\n"
        "  - type: implies\n"
        "    antecedent: A\n"
        "    consequent: A\n"
        "    rationale: Invalid self implication.\n",
    )

    with pytest.raises(ManualRelationError, match="must differ"):
        load_manual_relations(path, known_market_tickers={"A"})
