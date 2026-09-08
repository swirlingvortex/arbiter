"""Offline CLI integration tests for deterministic replay source selection."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from arbiter.cli import app
from arbiter.config import (
    CollectorSettings,
    EngineSettings,
    FeeSettings,
    OrderBookSettings,
    PaperExecutionSettings,
)
from arbiter.replay.engine import recorded_event_stream_hash
from arbiter.replay.events import (
    OrderBookSnapshotEvent,
    RecordedEvent,
    RunEndedEvent,
    RunInputPayload,
    RunStartedEvent,
)
from arbiter.solver.fees import KALSHI_FEE_POLICY_VERSION
from arbiter.storage.duckdb import DuckDBRepository
from arbiter.storage.parquet import ParquetEventWriter

runner = CliRunner()
NOW = datetime(2026, 9, 3, 23, 59, 59, tzinfo=UTC)


def _inputs() -> RunInputPayload:
    return RunInputPayload.build(
        config={
            "engine": EngineSettings().model_dump(mode="json"),
            "orderbook": OrderBookSettings().model_dump(mode="json"),
            "fees": FeeSettings().model_dump(mode="json"),
            "paper_execution": PaperExecutionSettings().model_dump(mode="json"),
            "collector": CollectorSettings().model_dump(mode="json"),
            "kalshi_environment": "demo",
        },
        markets=(),
        events=(),
        series=(),
        relations=(),
        fee_policy={"policy_version": KALSHI_FEE_POLICY_VERSION},
    )


def _run_records(
    *,
    run_id: str = "source-run",
    first_event_index: int = 0,
    started_at: datetime = NOW,
    ended_at: datetime | None = None,
) -> tuple[RecordedEvent, ...]:
    return (
        RunStartedEvent.build(
            event_index=first_event_index,
            local_received_ts=started_at,
            run_id=run_id,
            inputs=_inputs(),
        ),
        RunEndedEvent(
            event_index=first_event_index + 1,
            local_received_ts=ended_at or started_at,
            run_id=run_id,
            status="succeeded",
        ),
    )


def _write_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "data"
    db_path = data_dir / "arbiter.duckdb"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"storage:\n  data_dir: {data_dir}\n  db_path: {db_path}\n",
        encoding="utf-8",
    )
    return config, data_dir, db_path


def _replay_id(stdout: str) -> str:
    match = re.search(r"Replay (replay-[^ ]+) from source", stdout)
    assert match is not None
    return match.group(1)


def test_file_replay_persists_unique_manifests_and_accepts_numeric_speed(
    tmp_path: Path,
) -> None:
    config, _, db_path = _write_config(tmp_path)
    records = _run_records()
    recording_root = tmp_path / "recording"
    with ParquetEventWriter(recording_root) as writer:
        writer.extend(records)
        files = writer.flush()
    assert len(files) == 1

    first = runner.invoke(
        app,
        [
            "replay",
            "--file",
            str(files[0]),
            "--speed",
            "max",
            "--config",
            str(config),
            "--env-file",
            str(tmp_path / "missing.env"),
        ],
    )
    second = runner.invoke(
        app,
        [
            "replay",
            "--file",
            str(files[0]),
            "--speed",
            "100",
            "--config",
            str(config),
            "--env-file",
            str(tmp_path / "missing.env"),
        ],
    )

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    first_id = _replay_id(first.stdout)
    second_id = _replay_id(second.stdout)
    assert first_id != second_id
    for output, replay_id in ((first.stdout, first_id), (second.stdout, second_id)):
        normalized_output = " ".join(output.split())
        assert "from source source-run" in normalized_output
        assert "2 event(s)" in normalized_output
        assert "0 observation(s)" in normalized_output
        assert "0 paper execution(s)" in normalized_output
        assert recorded_event_stream_hash(records) in normalized_output
        with DuckDBRepository(db_path) as repository:
            manifest = repository.get_run_manifest(replay_id)
        assert manifest is not None
        assert manifest.status == "succeeded"
        assert manifest.source_run_id == "source-run"
    with DuckDBRepository(db_path) as repository:
        numeric_manifest = repository.get_run_manifest(second_id)
    assert numeric_manifest is not None
    assert numeric_manifest.metadata["replay_speed"] == 100.0


def test_date_replay_reads_full_root_selects_start_date_and_allows_raw_records(
    tmp_path: Path,
) -> None:
    config, data_dir, db_path = _write_config(tmp_path)
    orderbooks_root = data_dir / "parquet" / "orderbooks"
    raw_before = OrderBookSnapshotEvent(
        event_index=0,
        local_received_ts=NOW - timedelta(seconds=1),
        exchange_ts=NOW - timedelta(seconds=1),
        ticker="RAW",
        sequence=1,
        sid=1,
        connection_id="raw-connection",
        snapshot_id="raw-before",
    )
    bounded = _run_records(
        run_id="cross-midnight",
        first_event_index=1,
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=2),
    )
    raw_after = OrderBookSnapshotEvent(
        event_index=3,
        local_received_ts=NOW + timedelta(seconds=3),
        exchange_ts=NOW + timedelta(seconds=3),
        ticker="RAW",
        sequence=2,
        sid=1,
        connection_id="raw-connection",
        snapshot_id="raw-after",
    )
    with ParquetEventWriter(orderbooks_root) as writer:
        writer.extend((raw_before, *bounded, raw_after))

    strict_file = runner.invoke(
        app,
        ["replay", "--file", str(orderbooks_root), "--config", str(config)],
    )
    selected = runner.invoke(
        app,
        [
            "replay",
            "--date",
            "2026-09-03",
            "--speed",
            "max",
            "--config",
            str(config),
        ],
    )
    wrong_date = runner.invoke(
        app,
        ["replay", "--date", "2026-09-04", "--config", str(config)],
    )

    assert strict_file.exit_code == 1
    assert "outside a self-contained run boundary" in strict_file.stdout
    assert selected.exit_code == 0, selected.stdout
    assert "from source cross-midnight" in selected.stdout
    assert wrong_date.exit_code == 1
    assert "no complete replay runs started on UTC date 2026-09-04" in wrong_date.stdout
    with DuckDBRepository(db_path) as repository:
        manifest = repository.get_run_manifest(_replay_id(selected.stdout))
    assert manifest is not None
    assert manifest.source_run_id == "cross-midnight"
