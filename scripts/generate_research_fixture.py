#!/usr/bin/env python3
"""Build the deterministic DuckDB research fixture from the canonical replay stream."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from arbiter.engine.paper_execution import PaperExecutionStatus
from arbiter.engine.state import EngineState
from arbiter.models.opportunity import OpportunityTransition
from arbiter.replay.engine import ReplayResult, replay_to_repository
from arbiter.replay.events import RecordedEvent, RunStartedEvent
from arbiter.storage.duckdb import DuckDBRepository
from arbiter.storage.parquet import read_recorded_events

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPOSITORY_ROOT / "tests/fixtures/orderbooks/canonical_lifecycle.parquet"
FIXTURE_PATH = REPOSITORY_ROOT / "tests/fixtures/research/arbiter.duckdb"
METADATA_SYNC_ID = "fixture-research-metadata-v1"
REPLAY_RUN_ID = "fixture-research-replay-v1"
FIXED_CLOCK = datetime(2026, 9, 3, 15, tzinfo=UTC)
SNAPSHOT_COLUMNS = (
    "market_tickers_json",
    "relation_types_json",
    "relation_sources_json",
    "market_contexts_json",
)


def _started_record() -> tuple[tuple[RecordedEvent, ...], RunStartedEvent]:
    records = read_recorded_events(SOURCE_PATH)
    if not records or not isinstance(records[0], RunStartedEvent):
        raise RuntimeError("canonical lifecycle fixture lacks its immutable run-start inputs")
    return records, records[0]


def _expected_snapshots(
    started: RunStartedEvent,
) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
    market_tickers = sorted(market.ticker for market in started.inputs.markets)
    relation_types = sorted({relation.relation_type.value for relation in started.inputs.relations})
    relation_sources = sorted({relation.source for relation in started.inputs.relations})
    state = EngineState(
        markets=started.inputs.markets,
        events=started.inputs.events,
        series=started.inputs.series,
        relations=started.inputs.relations,
    )
    market_contexts = [
        context.model_dump(mode="json")
        for context in state.market_research_contexts(market_tickers)
    ]
    return market_tickers, relation_types, relation_sources, market_contexts


def _decode_json(value: object) -> object:
    if not isinstance(value, str):
        raise RuntimeError("research fixture contains a non-text JSON value")
    return json.loads(value)


def _validate_replay_result(result: ReplayResult) -> None:
    transitions = tuple(observation.transition for observation in result.observations)
    if transitions != (OpportunityTransition.OPEN, OpportunityTransition.CLOSED):
        raise RuntimeError("research replay must produce exactly OPEN then CLOSED")
    if len(result.paper_executions) != 1:
        raise RuntimeError("research replay must produce exactly one paper execution")
    paper = result.paper_executions[0].result
    if paper.status is not PaperExecutionStatus.FAILED or not paper.failure_reason:
        raise RuntimeError("research replay must preserve failed paper-execution evidence")


def _validate_database(
    repository: DuckDBRepository,
    *,
    started: RunStartedEvent,
) -> dict[str, object]:
    manifest = repository.get_run_manifest(REPLAY_RUN_ID)
    if manifest is None or manifest.status != "succeeded" or manifest.run_type != "replay":
        raise RuntimeError("research fixture lacks its successful replay manifest")
    if manifest.source_run_id != started.run_id:
        raise RuntimeError("research replay manifest has incorrect source provenance")

    inputs = started.inputs
    market_rows = repository.connection.execute(
        """
        SELECT ticker, event_ticker, series_ticker, title, status, settlement_ts
        FROM markets
        ORDER BY ticker
        """
    ).fetchall()
    expected_markets = tuple(
        (
            market.ticker,
            market.event_ticker,
            market.series_ticker,
            market.title,
            market.status,
            market.settlement_ts,
        )
        for market in inputs.markets
    )
    if tuple(market_rows) != expected_markets:
        raise RuntimeError("research fixture current markets differ from run-start inputs")
    event_rows = repository.connection.execute(
        """
        SELECT ticker, series_ticker, title, category, market_tickers_json
        FROM events
        ORDER BY ticker
        """
    ).fetchall()
    expected_events = tuple(
        (
            event.ticker,
            event.series_ticker,
            event.title,
            event.category,
            json.dumps(list(event.market_tickers), separators=(",", ":")),
        )
        for event in inputs.events
    )
    if tuple(event_rows) != expected_events:
        raise RuntimeError("research fixture current events differ from run-start inputs")
    series_rows = repository.connection.execute(
        """
        SELECT ticker, title, category, fee_type, fee_multiplier
        FROM series
        ORDER BY ticker
        """
    ).fetchall()
    expected_series = tuple(
        (
            series.ticker,
            series.title,
            series.category,
            series.fee_type,
            series.fee_multiplier,
        )
        for series in inputs.series
    )
    if tuple(series_rows) != expected_series:
        raise RuntimeError("research fixture current series differ from run-start inputs")
    if repository.list_relations() != inputs.relations:
        raise RuntimeError("research fixture current relations differ from run-start inputs")

    columns = {
        str(row[1])
        for row in repository.connection.execute(
            "PRAGMA table_info('opportunity_observations')"
        ).fetchall()
    }
    if not set(SNAPSHOT_COLUMNS) <= columns:
        raise RuntimeError("research fixture lacks immutable observation snapshot columns")

    rows = repository.connection.execute(
        """
        SELECT transition, event_index, market_tickers_json, relation_types_json,
               relation_sources_json, market_contexts_json
        FROM opportunity_observations
        WHERE run_id = ?
        ORDER BY event_index
        """,
        (REPLAY_RUN_ID,),
    ).fetchall()
    transitions = tuple(str(row[0]) for row in rows)
    if transitions != ("open", "closed"):
        raise RuntimeError("research fixture observations are not OPEN then CLOSED")
    expected = _expected_snapshots(started)
    for row in rows:
        actual = tuple(_decode_json(value) for value in row[2:])
        if actual != expected:
            raise RuntimeError("research fixture observation snapshots changed")

    paper_rows = repository.connection.execute(
        """
        SELECT status, failure_reason
        FROM paper_executions
        WHERE run_id = ?
        ORDER BY attempt_id
        """,
        (REPLAY_RUN_ID,),
    ).fetchall()
    if len(paper_rows) != 1 or paper_rows[0][0] != "failed" or not paper_rows[0][1]:
        raise RuntimeError("research fixture lacks one failed paper execution")
    paper_legs = repository.connection.execute(
        """
        SELECT fill_status
        FROM paper_execution_legs
        ORDER BY attempt_id, leg_index
        """
    ).fetchall()
    if not paper_legs or any(str(row[0]) != "not_filled" for row in paper_legs):
        raise RuntimeError("research fixture paper legs lack all-or-none failure evidence")

    return {
        "markets": repository.table_count("markets"),
        "events": repository.table_count("events"),
        "series": repository.table_count("series"),
        "relations": repository.table_count("relations"),
        "observations": len(rows),
        "transitions": list(transitions),
        "paper_executions": len(paper_rows),
        "paper_status": str(paper_rows[0][0]),
    }


def _logical_hash(summary: dict[str, object], started: RunStartedEvent) -> str:
    payload = {
        "source_run_id": started.run_id,
        "replay_run_id": REPLAY_RUN_ID,
        "fixed_clock": FIXED_CLOCK.isoformat(),
        "source_hashes": {
            "config": started.config_hash,
            "metadata": started.metadata_hash,
            "relations": started.relations_hash,
            "fees": started.fee_policy_hash,
        },
        "summary": summary,
        "snapshots": _expected_snapshots(started),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_fixture() -> tuple[Path, dict[str, object], str]:
    """Build, validate, and atomically replace the tracked research database."""

    records, started = _started_record()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".arbiter-research-", dir=FIXTURE_PATH.parent) as temporary:
        generated = Path(temporary) / FIXTURE_PATH.name
        with DuckDBRepository(generated, clock=lambda: FIXED_CLOCK) as repository:
            repository.sync_metadata(
                started.inputs.markets,
                started.inputs.events,
                started.inputs.series,
                run_id=METADATA_SYNC_ID,
            )
            repository.upsert_relations(started.inputs.relations)
            result = replay_to_repository(
                records,
                replay_run_id=REPLAY_RUN_ID,
                repository=repository,
                speed="max",
                sleeper=lambda _: None,
                clock=lambda: FIXED_CLOCK,
            )
            _validate_replay_result(result)
            summary = _validate_database(repository, started=started)
        os.replace(generated, FIXTURE_PATH)

    with DuckDBRepository(FIXTURE_PATH, clock=lambda: FIXED_CLOCK) as repository:
        restored_summary = _validate_database(repository, started=started)
    if restored_summary != summary:
        raise RuntimeError("atomic research fixture replacement changed database content")
    return FIXTURE_PATH, summary, _logical_hash(summary, started)


def main() -> None:
    fixture, summary, logical_hash = generate_fixture()
    payload = fixture.read_bytes()
    print(f"wrote {fixture.relative_to(REPOSITORY_ROOT)}")
    print(f"bytes={len(payload)}")
    print(f"sha256={hashlib.sha256(payload).hexdigest()}")
    print(f"logical_sha256={logical_hash}")
    print(f"content={json.dumps(summary, sort_keys=True, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
