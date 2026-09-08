"""CLI and transactional DuckDB metadata integration tests."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from arbiter.cli import app
from arbiter.kalshi.normalize import normalize_event, normalize_market, normalize_series
from arbiter.models.event import Event, EventFeeChange
from arbiter.models.market import Market
from arbiter.models.series import Series
from arbiter.storage.duckdb import DuckDBRepository, StorageError
from arbiter.storage.migrations import MIGRATIONS

runner = CliRunner()


def _domain_records() -> tuple[Market, Event, Series]:
    market_payload: dict[str, Any] = {
        "ticker": "MARKET-A",
        "event_ticker": "EVENT-A",
        "title": "Original title",
        "status": "active",
        "close_time": "2026-09-04T12:00:00Z",
        "price_ranges": [{"start": "0.0000", "end": "1.0000", "step": "0.0100"}],
    }
    event_payload: dict[str, Any] = {
        "event_ticker": "EVENT-A",
        "series_ticker": "SERIES-A",
        "title": "Event A",
        "mutually_exclusive": False,
    }
    series_payload: dict[str, Any] = {
        "ticker": "SERIES-A",
        "title": "Series A",
        "fee_type": "quadratic",
        "fee_multiplier": "1.0",
        "settlement_sources": [{"name": "Source", "url": "https://example.invalid"}],
    }
    return (
        normalize_market(market_payload, series_ticker="SERIES-A"),
        normalize_event(event_payload),
        normalize_series(series_payload),
    )


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    data_dir = tmp_path / "data"
    db_path = data_dir / "arbiter.duckdb"
    config.write_text(
        f"storage:\n  data_dir: {data_dir}\n  db_path: {db_path}\n",
        encoding="utf-8",
    )
    return config


class _FakeRestClient:
    def __init__(self, base_url: str, **kwargs: object) -> None:
        self.base_url = base_url
        self.market, self.event, self.series = _domain_records()
        self.fee_change_calls = 0

    def __enter__(self) -> _FakeRestClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def iter_markets(self, **kwargs: object) -> tuple[Market, ...]:
        assert kwargs["status"] == "open"
        return (self.market,)

    def get_event(self, event_ticker: str, **kwargs: object) -> Event:
        assert event_ticker == self.event.ticker
        return self.event

    def get_series(self, series_ticker: str) -> Series:
        assert series_ticker == self.series.ticker
        return self.series

    def iter_event_fee_changes(self, **kwargs: object) -> tuple[EventFeeChange, ...]:
        assert kwargs == {}
        self.fee_change_calls += 1
        return (
            EventFeeChange(
                change_id="later",
                event_ticker=self.event.ticker,
                series_ticker=self.series.ticker,
                scheduled_ts=datetime(2026, 9, 5, tzinfo=UTC),
                raw={"id": "later"},
            ),
            EventFeeChange(
                change_id="earlier",
                event_ticker=self.event.ticker,
                series_ticker=self.series.ticker,
                scheduled_ts=datetime(2026, 9, 3, tzinfo=UTC),
                fee_type_override="quadratic",
                fee_multiplier_override=Decimal("1.5"),
                raw={"id": "earlier"},
            ),
            EventFeeChange(
                change_id="irrelevant",
                event_ticker="OTHER-EVENT",
                series_ticker="OTHER-SERIES",
                scheduled_ts=datetime(2026, 9, 3, tzinfo=UTC),
                raw={"id": "irrelevant"},
            ),
        )


def test_markets_sync_and_list_round_trip_through_duckdb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path)
    monkeypatch.setattr("arbiter.cli.KalshiRestClient", _FakeRestClient)

    sync = runner.invoke(
        app,
        [
            "markets",
            "sync",
            "--max-pages",
            "1",
            "--config",
            str(config),
            "--env-file",
            str(tmp_path / "missing.env"),
        ],
    )
    listed = runner.invoke(
        app,
        [
            "markets",
            "list",
            "--config",
            str(config),
            "--env-file",
            str(tmp_path / "missing.env"),
        ],
    )

    assert sync.exit_code == 0, sync.output
    assert "Synchronized 1 markets, 1 events, and 1 series" in sync.output
    assert listed.exit_code == 0, listed.output
    assert "MARKET-A" in listed.output
    assert "SERIES-A" in listed.output
    assert "Original title" in listed.output
    with DuckDBRepository(tmp_path / "data/arbiter.duckdb") as repository:
        stored_event = repository.load_events()[0]
        assert [change.change_id for change in stored_event.fee_changes] == [
            "earlier",
            "later",
        ]


def test_migration_three_creates_fee_schedule_and_opportunity_schema(tmp_path: Path) -> None:
    market, event, series = _domain_records()
    change = EventFeeChange(
        change_id="change",
        event_ticker=event.ticker,
        series_ticker=series.ticker,
        scheduled_ts=datetime(2026, 9, 3, 12, tzinfo=UTC),
        fee_type_override="quadratic",
        fee_multiplier_override=Decimal("1.25"),
        raw={"source": "fixture"},
    )
    enriched = event.model_copy(update={"fee_changes": (change,)})

    with DuckDBRepository(tmp_path / "fee-schema.duckdb") as repository:
        repository.sync_metadata((market,), (enriched,), (series,), run_id="fee-schema")

        restored = repository.load_events()[0]
        migrations = repository.connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()
        event_columns = {
            str(row[1])
            for row in repository.connection.execute("PRAGMA table_info('events')").fetchall()
        }
        opportunity_columns = {
            str(row[1])
            for row in repository.connection.execute(
                "PRAGMA table_info('opportunities')"
            ).fetchall()
        }

        assert repository.table_count("opportunities") == 0
        assert repository.table_count("portfolio_legs") == 0

    assert (3, "fee_schedules_and_opportunities") in migrations
    assert (4, "scanner_runs_and_observations") in migrations
    assert (5, "replay_paper_evidence") in migrations
    assert (6, "semantic_discovery_review") in migrations
    assert migrations[-1] == (7, "research_observation_context")
    assert "fee_changes_json" in event_columns
    assert {
        "fee_policy_version",
        "fee_policies_json",
        "run_id",
        "opened_at",
        "closed_at",
        "updated_at",
    } <= opportunity_columns
    assert restored.fee_changes == (change,)


def test_migration_three_backfills_existing_event_schedules(tmp_path: Path) -> None:
    database = tmp_path / "version-two.duckdb"
    applied_at = datetime(2026, 9, 3, 12, tzinfo=UTC)
    with DuckDBRepository(database) as repository:
        repository.connection.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        for migration in MIGRATIONS[:2]:
            repository.connection.execute(migration.sql)
            repository.connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?)",
                (migration.version, migration.name, applied_at),
            )
        repository.connection.execute(
            """
            INSERT INTO events (
                ticker, series_ticker, title, subtitle, category,
                mutually_exclusive, available_on_brokers, market_tickers_json,
                last_updated_ts, raw_json, updated_at
            ) VALUES (
                'EVENT-OLD', 'SERIES-OLD', 'Old event', NULL, NULL,
                NULL, NULL, '[]', NULL,
                '{"event_ticker":"EVENT-OLD","series_ticker":"SERIES-OLD","title":"Old event"}',
                ?
            )
            """,
            (applied_at,),
        )

        repository.migrate()
        restored = repository.load_events()[0]
        stored_json = repository.connection.execute(
            "SELECT fee_changes_json FROM events WHERE ticker = 'EVENT-OLD'"
        ).fetchone()

    assert restored.fee_changes == ()
    assert stored_json is not None and str(stored_json[0]) == "[]"


def test_metadata_upserts_are_idempotent_and_update_existing_rows(tmp_path: Path) -> None:
    market, event, series = _domain_records()
    database = tmp_path / "metadata.duckdb"
    with DuckDBRepository(database) as repository:
        repository.sync_metadata((market,), (event,), (series,), run_id="first")
        changed = market.model_copy(update={"title": "Updated title"})
        repository.sync_metadata((changed,), (event,), (series,), run_id="second")

        assert repository.table_count("markets") == 1
        assert repository.table_count("events") == 1
        assert repository.table_count("series") == 1
        assert repository.table_count("metadata_sync_runs") == 2
        assert repository.list_markets()[0].title == "Updated title"


def test_failed_sync_rolls_back_partial_upserts_and_records_failure(tmp_path: Path) -> None:
    market, event, series = _domain_records()

    class FailingRepository(DuckDBRepository):
        def _upsert(
            self,
            table: str,
            columns: tuple[str, ...],
            rows: tuple[tuple[object, ...], ...],
        ) -> None:
            super()._upsert(table, columns, rows)
            if table == "markets":
                raise RuntimeError("synthetic interruption")

    with FailingRepository(tmp_path / "failed.duckdb") as repository:
        with pytest.raises(StorageError, match="synthetic interruption"):
            repository.sync_metadata((market,), (event,), (series,), run_id="failed-run")

        assert repository.table_count("markets") == 0
        row = repository.connection.execute(
            "SELECT status, error FROM metadata_sync_runs WHERE run_id = 'failed-run'"
        ).fetchone()
        assert row is not None
        assert row[0] == "failed"
        assert "synthetic interruption" in str(row[1])


def test_markets_list_on_new_database_is_safe(tmp_path: Path) -> None:
    config = _write_config(tmp_path)

    result = runner.invoke(
        app,
        [
            "markets",
            "list",
            "--config",
            str(config),
            "--env-file",
            str(tmp_path / "missing.env"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "No synchronized markets found" in result.output
