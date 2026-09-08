"""Relation discovery, persistence, listing, and validation CLI integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from arbiter.cli import app
from arbiter.kalshi.normalize import normalize_event, normalize_market, normalize_series
from arbiter.storage.duckdb import DuckDBRepository

runner = CliRunner()


def _config(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    db_path = data_dir / "arbiter.duckdb"
    path = tmp_path / "config.yaml"
    path.write_text(
        "relations:\n"
        "  discover_exchange: false\n"
        "  discover_thresholds: false\n"
        "storage:\n"
        f"  data_dir: {data_dir}\n"
        f"  db_path: {db_path}\n",
        encoding="utf-8",
    )
    return path, db_path


def _seed_metadata(db_path: Path) -> None:
    market_payloads: tuple[dict[str, Any], ...] = (
        {"ticker": "A", "event_ticker": "E", "title": "A", "status": "active"},
        {"ticker": "B", "event_ticker": "E", "title": "B", "status": "active"},
    )
    markets = tuple(normalize_market(payload, series_ticker="S") for payload in market_payloads)
    event = normalize_event(
        {
            "event_ticker": "E",
            "series_ticker": "S",
            "title": "Event",
            "markets": market_payloads,
        }
    )
    series = normalize_series({"ticker": "S", "title": "Series"})
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with DuckDBRepository(db_path) as repository:
        repository.sync_metadata(markets, (event,), (series,), run_id="seed")


def _base_args(config: Path, tmp_path: Path) -> list[str]:
    return ["--config", str(config), "--env-file", str(tmp_path / "missing.env")]


def test_manual_discover_is_idempotent_then_lists_and_validates(tmp_path: Path) -> None:
    config, db_path = _config(tmp_path)
    _seed_metadata(db_path)
    manual = tmp_path / "relations.yaml"
    manual.write_text(
        "relations:\n"
        "  - type: implies\n"
        "    antecedent: A\n"
        "    consequent: B\n"
        "    rationale: A always entails B.\n",
        encoding="utf-8",
    )
    discover_args = [
        "relations",
        "discover",
        "--manual",
        str(manual),
        *_base_args(config, tmp_path),
    ]

    first = runner.invoke(app, discover_args)
    with DuckDBRepository(db_path) as repository:
        first_relation = repository.list_relations()[0]
    second = runner.invoke(app, discover_args)
    listed = runner.invoke(app, ["relations", "list", *_base_args(config, tmp_path)])
    validated = runner.invoke(app, ["relations", "validate", *_base_args(config, tmp_path)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "upserted 1 trusted relations" in second.output
    assert listed.exit_code == 0, listed.output
    assert "implies" in listed.output and "manual" in listed.output
    assert validated.exit_code == 0, validated.output
    assert "Relation validation passed" in validated.output
    with DuckDBRepository(db_path) as repository:
        relations = repository.list_relations()
        assert len(relations) == 1
        assert relations[0].created_at == first_relation.created_at


def test_impossible_manual_set_is_rejected_before_persistence(tmp_path: Path) -> None:
    config, db_path = _config(tmp_path)
    _seed_metadata(db_path)
    manual = tmp_path / "impossible.yaml"
    manual.write_text(
        "relations:\n"
        "  - type: exactly_one\n"
        "    markets: [A, B]\n"
        "    rationale: Exactly one.\n"
        "  - type: equivalent\n"
        "    markets: [A, B]\n"
        "    rationale: Equal.\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "relations",
            "discover",
            "--manual",
            str(manual),
            *_base_args(config, tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "impossible_component" in result.output
    with DuckDBRepository(db_path) as repository:
        assert repository.table_count("relations") == 0
