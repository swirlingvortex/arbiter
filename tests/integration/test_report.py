"""Read-only report generation and historical-dimension integration tests."""

from __future__ import annotations

import csv
import shutil
from hashlib import sha256
from pathlib import Path

import duckdb
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from typer.testing import CliRunner

from arbiter.analytics.report import generate_report
from arbiter.cli import app

runner = CliRunner()
EXPECTED_EPISODE_COLUMNS = (
    "fact_id",
    "cohort_id",
    "run_id",
    "opportunity_id",
    "component_id",
    "opened_at",
    "terminal_status",
    "ended_at",
    "duration_seconds",
    "relation_types",
    "relation_sources",
    "categories",
    "settlement_bucket",
    "peak_stage",
    "peak_gross_edge",
    "peak_net_edge",
    "peak_capacity",
    "peak_net_guarantee",
    "midpoint_available",
    "reference_violation",
    "one_contract_gross_survived",
    "one_contract_fee_survived",
    "depth_executable",
    "paper_status",
    "paper_locked_profit",
)


def _fixture(project_root: Path) -> Path:
    return project_root / "tests/fixtures/research/arbiter.duckdb"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _artifact_bytes(directory: Path) -> dict[str, bytes]:
    return {
        name: (directory / name).read_bytes()
        for name in (
            "report.md",
            "summary.csv",
            "funnel.csv",
            "breakdowns.csv",
            "episodes.parquet",
        )
    }


def test_fixture_report_writes_required_tables_without_mutating_database(
    project_root: Path,
    tmp_path: Path,
) -> None:
    database = _fixture(project_root)
    before = sha256(database.read_bytes()).hexdigest()

    artifacts = generate_report(database, tmp_path / "report")

    assert sha256(database.read_bytes()).hexdigest() == before
    assert artifacts.exposure_method == "recording_coverage_fallback"
    assert artifacts.warnings == ()
    assert len(artifacts.analytics.episodes) == 1
    funnel_summary = artifacts.analytics.summary.funnel
    assert funnel_summary.logical_violation_count == 1
    assert funnel_summary.gross_executable_count == 1
    assert funnel_summary.fee_adjusted_executable_count == 1
    assert funnel_summary.depth_executable_count == 1
    assert funnel_summary.paper_surviving_count == 0
    assert funnel_summary.paper_failed_count == 1

    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "append-only observation snapshots" in markdown
    assert "Legacy NULL snapshots remain unknown" in markdown
    assert "overlap" in markdown.casefold()
    assert "non-atomic" in markdown

    summary = {row["metric"]: row for row in _csv_rows(artifacts.summary_csv_path)}
    assert summary["unique_episodes"]["value"] == "1"
    assert summary["maximum_capital"]["median"] == "0.920000000000000000"
    assert summary["guaranteed_dollars"]["median"] == "0.080000000000000000"
    funnel = {row["stage"]: row for row in _csv_rows(artifacts.funnel_csv_path)}
    assert funnel["midpoint_logical_violation"]["surviving_count"] == "1"
    assert funnel["paper_execution_survival"]["surviving_count"] == "0"

    table = pq.read_table(artifacts.episodes_parquet_path)
    assert tuple(table.column_names) == EXPECTED_EPISODE_COLUMNS
    assert table.num_rows == 1
    episode = table.to_pylist()[0]
    assert episode["categories"] == ["Sports"]
    assert episode["relation_types"] == ["implies"]
    assert episode["relation_sources"] == ["manual"]
    assert episode["settlement_bucket"] == "1–6h"
    assert episode["paper_status"] == "failed"


def test_current_metadata_mutation_cannot_rewrite_historical_report(
    project_root: Path,
    tmp_path: Path,
) -> None:
    database = tmp_path / "mutable-copy.duckdb"
    shutil.copyfile(_fixture(project_root), database)
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    generate_report(database, before_dir)

    connection = duckdb.connect(str(database))
    try:
        connection.execute("UPDATE events SET category = 'Rewritten category'")
        connection.execute("UPDATE series SET category = 'Rewritten series'")
        connection.execute(
            "UPDATE markets SET settlement_ts = TIMESTAMPTZ '2035-01-01 00:00:00+00'"
        )
        connection.execute(
            "UPDATE relations SET relation_type = 'equivalent', source = 'semantic_verified'"
        )
    finally:
        connection.close()

    generate_report(database, after_dir)

    assert _artifact_bytes(after_dir) == _artifact_bytes(before_dir)


def test_legacy_null_snapshots_remain_unknown_instead_of_using_current_tables(
    project_root: Path,
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-copy.duckdb"
    shutil.copyfile(_fixture(project_root), database)
    connection = duckdb.connect(str(database))
    try:
        connection.execute(
            """
            UPDATE opportunity_observations
            SET market_tickers_json = NULL,
                relation_types_json = NULL,
                relation_sources_json = NULL,
                market_contexts_json = NULL
            """
        )
        connection.execute("UPDATE events SET category = 'Fabricated current category'")
        connection.execute(
            "UPDATE markets SET settlement_ts = TIMESTAMPTZ '2026-09-03 12:30:00+00'"
        )
        connection.execute(
            "UPDATE relations SET relation_type = 'equivalent', source = 'semantic_verified'"
        )
    finally:
        connection.close()

    artifacts = generate_report(database, tmp_path / "legacy-report")
    episode = pq.read_table(artifacts.episodes_parquet_path).to_pylist()[0]

    assert episode["categories"] == []
    assert episode["relation_types"] == []
    assert episode["relation_sources"] == []
    assert episode["settlement_bucket"] == "unknown"
    breakdowns = _csv_rows(artifacts.breakdowns_csv_path)
    assert {
        (row["dimension"], row["label"], row["episode_count"])
        for row in breakdowns
        if row["episode_count"] == "1"
    } >= {
        ("category", "unknown", "1"),
        ("relation_type", "unknown", "1"),
        ("relation_source", "unknown", "1"),
        ("time_to_settlement", "unknown", "1"),
    }


def test_report_cli_exposes_required_interface_and_generates_artifacts(
    project_root: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "cli-report"
    result = runner.invoke(
        app,
        ["report", "--db", str(_fixture(project_root)), "--output-dir", str(output)],
    )

    assert result.exit_code == 0
    assert "Wrote research report" in result.stdout
    assert (output / "report.md").is_file()

    help_result = runner.invoke(app, ["report", "--help"])
    assert help_result.exit_code == 0
    assert "--db" in help_result.stdout
    assert "--output-dir" in help_result.stdout


def test_report_cli_fails_cleanly_for_missing_database(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "report",
            "--db",
            str(tmp_path / "missing.duckdb"),
            "--output-dir",
            str(tmp_path / "report"),
        ],
    )

    assert result.exit_code == 1
    assert "Report failed" in result.stdout
    assert "Traceback" not in result.stdout
