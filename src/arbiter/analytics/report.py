"""Read-only DuckDB adapter and deterministic research-report renderers."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import duckdb
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from arbiter.analytics.metrics import (
    AnalyticsResult,
    AnalyticsSummary,
    Breakdown,
    MarketContextMetricRow,
    MarketObservationWindowRow,
    ObservationMetricRow,
    PaperExecutionMetricRow,
    RunCohortRow,
    RunCohortSelection,
    compute_analytics,
    select_run_cohorts,
)

REPORT_SCHEMA_VERSION = 1
REQUIRED_DATABASE_MIGRATION = 7
FRESH_WINDOW_METHOD = "fresh_observation_windows"
REPLAY_WINDOW_METHOD = "selected_replay_event_time_windows"
FALLBACK_WINDOW_METHOD = "recording_coverage_fallback"


class ReportError(RuntimeError):
    """A safe, user-facing report input or output failure."""


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    """Paths and computed values produced by one successful report run."""

    database_path: Path
    output_dir: Path
    markdown_path: Path
    summary_csv_path: Path
    funnel_csv_path: Path
    breakdowns_csv_path: Path
    episodes_parquet_path: Path
    analytics: AnalyticsResult
    exposure_method: str
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ManifestContext:
    """Immutable run-manifest material used only for denominator fallback."""

    run: RunCohortRow
    input_payload: Mapping[str, object]
    metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _LoadedRows:
    """Normalized database rows before pure analytics reduction."""

    runs: tuple[RunCohortRow, ...]
    manifests: Mapping[str, _ManifestContext]
    observations: tuple[ObservationMetricRow, ...]
    paper_executions: tuple[PaperExecutionMetricRow, ...]
    market_windows: tuple[MarketObservationWindowRow, ...]


def generate_report(database_path: Path, output_dir: Path) -> ReportArtifacts:
    """Load immutable evidence read-only and atomically publish all report artifacts."""

    database_path = database_path.resolve()
    output_dir = output_dir.resolve()
    if not database_path.is_file():
        raise ReportError(f"report database does not exist or is not a file: {database_path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ReportError(f"report output path is not a directory: {output_dir}")

    try:
        connection = duckdb.connect(str(database_path), read_only=True)
    except duckdb.Error as exc:
        raise ReportError("report database could not be opened read-only") from exc
    try:
        _require_report_schema(connection)
        loaded = _load_rows(connection)
    except (duckdb.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ReportError(f"report database contains invalid research evidence: {exc}") from exc
    finally:
        connection.close()

    selection = select_run_cohorts(loaded.runs)
    windows, exposure_method, warnings = _market_windows_with_fallback(
        loaded.market_windows,
        selection=selection,
        manifests=loaded.manifests,
    )
    try:
        analytics = compute_analytics(
            runs=loaded.runs,
            observations=loaded.observations,
            paper_executions=loaded.paper_executions,
            market_windows=windows,
        )
    except ValueError as exc:
        raise ReportError(f"research evidence is internally inconsistent: {exc}") from exc
    analytics = _label_exposure_method(analytics, exposure_method)

    summary_rows = _summary_rows(analytics.summary, exposure_method)
    funnel_rows = _funnel_rows(analytics.summary)
    breakdown_rows = _breakdown_rows(analytics.summary)
    episode_rows = _episode_rows(analytics)
    markdown = _render_markdown(
        analytics,
        summary_rows=summary_rows,
        funnel_rows=funnel_rows,
        breakdown_rows=breakdown_rows,
        exposure_method=exposure_method,
        warnings=warnings,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}-",
            dir=output_dir.parent,
        ) as temporary:
            staging = Path(temporary)
            _write_text(staging / "report.md", markdown)
            _write_csv(staging / "summary.csv", summary_rows)
            _write_csv(staging / "funnel.csv", funnel_rows)
            _write_csv(staging / "breakdowns.csv", breakdown_rows)
            _write_episodes_parquet(staging / "episodes.parquet", episode_rows)
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "report.md",
                "summary.csv",
                "funnel.csv",
                "breakdowns.csv",
                "episodes.parquet",
            ):
                os.replace(staging / name, output_dir / name)
    except OSError as exc:
        raise ReportError(f"could not publish report artifacts: {exc}") from exc

    return ReportArtifacts(
        database_path=database_path,
        output_dir=output_dir,
        markdown_path=output_dir / "report.md",
        summary_csv_path=output_dir / "summary.csv",
        funnel_csv_path=output_dir / "funnel.csv",
        breakdowns_csv_path=output_dir / "breakdowns.csv",
        episodes_parquet_path=output_dir / "episodes.parquet",
        analytics=analytics,
        exposure_method=exposure_method,
        warnings=warnings,
    )


def _require_report_schema(connection: duckdb.DuckDBPyConnection) -> None:
    applied = {
        int(row[0])
        for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
    }
    if REQUIRED_DATABASE_MIGRATION not in applied:
        raise ValueError(
            f"migration {REQUIRED_DATABASE_MIGRATION} is required for historical analytics"
        )
    columns = {
        str(row[0])
        for row in connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'main' AND table_name = 'opportunity_observations'
            """
        ).fetchall()
    }
    required = {
        "market_tickers_json",
        "relation_types_json",
        "relation_sources_json",
        "market_contexts_json",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"historical observation columns are missing: {', '.join(missing)}")


def _load_rows(connection: duckdb.DuckDBPyConnection) -> _LoadedRows:
    runs: list[RunCohortRow] = []
    manifests: dict[str, _ManifestContext] = {}
    for row in connection.execute(
        """
        SELECT run_id, run_type, started_at, ended_at, status, recording_id,
               source_run_id, input_payload_json, metadata_json
        FROM run_manifests
        ORDER BY run_id
        """
    ).fetchall():
        run = RunCohortRow(
            run_id=str(row[0]),
            run_type=str(row[1]),
            started_at=_datetime(row[2], "run started_at"),
            ended_at=_optional_datetime(row[3], "run ended_at"),
            status=str(row[4]),
            recording_id=None if row[5] is None else str(row[5]),
            source_run_id=None if row[6] is None else str(row[6]),
        )
        runs.append(run)
        manifests[run.run_id] = _ManifestContext(
            run=run,
            input_payload=_json_object(row[7], "run input payload"),
            metadata=_json_object(row[8], "run metadata"),
        )

    observations = tuple(
        _observation_row(row)
        for row in connection.execute(
            """
            SELECT observation_id, opportunity_id, run_id, component_id, observed_at,
                   event_index, transition, stage, gross_edge, net_edge, capacity,
                   net_profit, market_tickers_json, relation_types_json,
                   relation_sources_json, market_contexts_json, evidence_json
            FROM opportunity_observations
            ORDER BY run_id, event_index, observation_id
            """
        ).fetchall()
    )
    paper = tuple(
        PaperExecutionMetricRow(
            attempt_id=str(row[0]),
            run_id=str(row[1]),
            opportunity_id=str(row[2]),
            status=cast(Any, str(row[3])),
            simulated_locked_profit=_optional_decimal(row[4], "paper locked profit"),
        )
        for row in connection.execute(
            """
            SELECT attempt_id, run_id, opportunity_id, status, simulated_locked_profit
            FROM paper_executions
            ORDER BY run_id, opportunity_id, attempt_id
            """
        ).fetchall()
    )
    windows = tuple(
        MarketObservationWindowRow(
            observation_id=str(row[0]),
            run_id=str(row[1]),
            market_ticker=str(row[2]),
            opened_at=_datetime(row[3], "market window opened_at"),
            updated_at=_datetime(row[4], "market window updated_at"),
            closed_at=_optional_datetime(row[5], "market window closed_at"),
        )
        for row in connection.execute(
            """
            SELECT observation_id, run_id, market_ticker, opened_at, updated_at, closed_at
            FROM market_observation_windows
            ORDER BY run_id, market_ticker, opened_at, observation_id
            """
        ).fetchall()
    )
    return _LoadedRows(
        runs=tuple(runs),
        manifests=manifests,
        observations=observations,
        paper_executions=paper,
        market_windows=windows,
    )


def _observation_row(row: Sequence[object]) -> ObservationMetricRow:
    evidence = _json_object(row[16], "opportunity evidence")
    reference = _optional_bool(evidence.get("reference_violation"), "reference_violation")
    explicit_midpoint = _optional_bool(evidence.get("midpoint_available"), "midpoint_available")
    # Before explicit availability was recorded, only a positive violation proves that
    # a complete midpoint existed. False remains unknown because the old producer also
    # used it when one side of a book was absent.
    midpoint_evaluable = explicit_midpoint
    if midpoint_evaluable is None and reference is True:
        midpoint_evaluable = True
    if midpoint_evaluable is not True:
        reference = None

    market_tickers = _optional_string_tuple(row[12], "historical market tickers")
    relation_types = _optional_string_tuple(row[13], "historical relation types")
    relation_sources = _optional_string_tuple(row[14], "historical relation sources")
    contexts_payload = _json_array(row[15], "historical market contexts")
    contexts: list[MarketContextMetricRow] = []
    for item in contexts_payload or ():
        if not isinstance(item, dict):
            raise ValueError("historical market contexts must contain objects")
        contexts.append(
            MarketContextMetricRow(
                ticker=_required_string(item.get("ticker"), "context ticker"),
                event_ticker=_required_string(
                    item.get("event_ticker"),
                    "context event ticker",
                ),
                category=_optional_string(item.get("category"), "context category"),
                settlement_at=_optional_datetime(
                    item.get("settlement_at"),
                    "context settlement_at",
                ),
            )
        )
    return ObservationMetricRow(
        observation_id=str(row[0]),
        opportunity_id=None if row[1] is None else str(row[1]),
        run_id=str(row[2]),
        component_id=str(row[3]),
        observed_at=_datetime(row[4], "opportunity observed_at"),
        event_index=int(cast(int, row[5])),
        transition=cast(Any, str(row[6])),
        stage=None if row[7] is None else cast(Any, str(row[7])),
        gross_edge=_optional_decimal(row[8], "gross edge"),
        net_edge=_optional_decimal(row[9], "net edge"),
        capacity=_optional_decimal(row[10], "capacity"),
        net_profit=_optional_decimal(row[11], "net profit"),
        market_tickers=market_tickers,
        relation_types=relation_types,
        relation_sources=relation_sources,
        market_contexts=tuple(contexts),
        midpoint_evaluable=midpoint_evaluable,
        reference_violation=reference,
        one_contract_gross_survived=_optional_bool(
            evidence.get("one_contract_gross_survived"),
            "one-contract gross result",
        ),
        one_contract_fee_survived=_optional_bool(
            evidence.get("one_contract_fee_survived"),
            "one-contract fee result",
        ),
        depth_executable=_optional_bool(
            evidence.get("depth_executable"),
            "depth result",
        ),
    )


def _market_windows_with_fallback(
    windows: Sequence[MarketObservationWindowRow],
    *,
    selection: RunCohortSelection,
    manifests: Mapping[str, _ManifestContext],
) -> tuple[tuple[MarketObservationWindowRow, ...], str, tuple[str, ...]]:
    selected_ids = set(selection.coverage_run_ids)
    selected = [item for item in windows if item.run_id in selected_ids]
    runs_with_windows = {item.run_id for item in selected}
    methods: set[str] = set()
    warnings: list[str] = []
    for cohort in selection.cohorts:
        manifest = manifests[cohort.coverage_run_id]
        if cohort.coverage_run_id in runs_with_windows:
            methods.add(
                REPLAY_WINDOW_METHOD if manifest.run.run_type == "replay" else FRESH_WINDOW_METHOD
            )
            continue
        fallback = _fallback_windows(cohort.cohort_id, manifest)
        if fallback:
            selected.extend(fallback)
            methods.add(FALLBACK_WINDOW_METHOD)
        else:
            warnings.append(f"cohort {cohort.cohort_id}: market-hour exposure is unavailable")
    if not methods:
        method = "unavailable"
    elif len(methods) == 1:
        method = next(iter(methods))
    else:
        method = "+".join(sorted(methods))
    return tuple(selected), method, tuple(warnings)


def _fallback_windows(
    cohort_id: str,
    manifest: _ManifestContext,
) -> tuple[MarketObservationWindowRow, ...]:
    markets = manifest.input_payload.get("markets")
    if not isinstance(markets, list):
        return ()
    tickers = tuple(
        sorted(
            {
                item["ticker"]
                for item in markets
                if isinstance(item, dict)
                and isinstance(item.get("ticker"), str)
                and item["ticker"].strip()
            }
        )
    )
    if not tickers:
        return ()
    if manifest.run.run_type == "replay":
        opened_at = _optional_datetime(
            manifest.metadata.get("source_started_at"),
            "source recording start",
        )
        closed_at = _optional_datetime(
            manifest.metadata.get("source_ended_at"),
            "source recording end",
        )
    else:
        opened_at = manifest.run.started_at
        closed_at = manifest.run.ended_at
    if opened_at is None or closed_at is None or closed_at < opened_at:
        return ()
    return tuple(
        MarketObservationWindowRow(
            observation_id=f"fallback:{cohort_id}:{ticker}",
            run_id=manifest.run.run_id,
            market_ticker=ticker,
            opened_at=opened_at,
            updated_at=closed_at,
            closed_at=closed_at,
        )
        for ticker in tickers
    )


def _label_exposure_method(result: AnalyticsResult, method: str) -> AnalyticsResult:
    explanation = (
        f"{method}; deduplicated by source recording; event-time intervals are unioned "
        "within each market"
    )
    summary = replace(
        result.summary,
        market_hour_rates=tuple(
            replace(item, denominator_method=explanation)
            for item in result.summary.market_hour_rates
        ),
        relation_type_breakdown=replace(
            result.summary.relation_type_breakdown,
            rate_denominator_method=explanation,
        ),
        relation_source_breakdown=replace(
            result.summary.relation_source_breakdown,
            rate_denominator_method=explanation,
        ),
        category_breakdown=replace(
            result.summary.category_breakdown,
            rate_denominator_method=explanation,
        ),
        settlement_breakdown=replace(
            result.summary.settlement_breakdown,
            rate_denominator_method=explanation,
        ),
    )
    return replace(result, summary=summary)


def _summary_rows(summary: AnalyticsSummary, exposure_method: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "metric": "unique_episodes",
            "count": summary.episode_count,
            "median": "",
            "p90": "",
            "maximum": "",
            "value": summary.episode_count,
            "note": "one OPEN-led episode after recording/replay cohort deduplication",
        },
        {
            "metric": "market_hours",
            "count": "",
            "median": "",
            "p90": "",
            "maximum": "",
            "value": _decimal(summary.market_hour_rates[0].market_hours)
            if summary.market_hour_rates
            else "0",
            "note": exposure_method,
        },
    ]
    distributions = (
        ("gross_edge", summary.gross_edge),
        ("net_edge", summary.net_edge),
        ("closed_duration_seconds", summary.closed_duration_seconds),
        (
            "right_censored_lower_bound_seconds",
            summary.right_censored_lower_bound_seconds,
        ),
        ("maximum_capital", summary.maximum_capital),
        ("guaranteed_dollars", summary.guaranteed_dollars),
        ("paper_locked_profit", summary.paper_locked_profit),
    )
    for name, distribution in distributions:
        rows.append(
            {
                "metric": name,
                "count": distribution.count,
                "median": _decimal(distribution.median),
                "p90": _decimal(distribution.p90),
                "maximum": _decimal(distribution.maximum),
                "value": "",
                "note": "Decimal type-7 interpolation; missing values are excluded",
            }
        )
    rows.append(
        {
            "metric": "total_hypothetical_guaranteed_dollars",
            "count": summary.guaranteed_dollars.count,
            "median": "",
            "p90": "",
            "maximum": "",
            "value": _decimal(summary.total_hypothetical_guaranteed_dollars),
            "note": summary.hypothetical_profit_note,
        }
    )
    return rows


def _funnel_rows(summary: AnalyticsSummary) -> list[dict[str, object]]:
    funnel = summary.funnel
    rates = {item.metric: item.rate_per_market_hour for item in summary.market_hour_rates}
    return [
        {
            "stage": "midpoint_logical_violation",
            "input_count": funnel.unique_episode_count,
            "evaluable_count": funnel.midpoint_evaluable_count,
            "surviving_count": funnel.logical_violation_count,
            "rate_per_market_hour": _decimal(rates.get("logical_violations")),
            "note": (
                f"midpoint unavailable={funnel.midpoint_unavailable_count}; "
                f"legacy availability unknown={funnel.midpoint_unknown_count}"
            ),
        },
        {
            "stage": "one_contract_spread_survival",
            "input_count": funnel.logical_violation_count,
            "evaluable_count": funnel.spread_evaluable_count,
            "surviving_count": funnel.gross_executable_count,
            "rate_per_market_hour": _decimal(rates.get("gross_executable")),
            "note": (
                "fixed one-contract probe; executable signals without a midpoint reference="
                f"{funnel.executable_without_midpoint_count}"
            ),
        },
        {
            "stage": "modeled_fee_survival",
            "input_count": funnel.gross_executable_count,
            "evaluable_count": funnel.fee_evaluable_count,
            "surviving_count": funnel.fee_adjusted_executable_count,
            "rate_per_market_hour": _decimal(rates.get("fee_adjusted_executable")),
            "note": "run-specific fee policy and account rounding",
        },
        {
            "stage": "actual_depth_survival",
            "input_count": funnel.fee_adjusted_executable_count,
            "evaluable_count": funnel.depth_evaluable_count,
            "surviving_count": funnel.depth_executable_count,
            "rate_per_market_hour": _decimal(rates.get("depth_executable")),
            "note": f"all net-executable episodes={funnel.net_executable_count}",
        },
        {
            "stage": "paper_execution_survival",
            "input_count": funnel.paper_eligible_count,
            "evaluable_count": funnel.paper_evaluable_count,
            "surviving_count": funnel.paper_surviving_count,
            "rate_per_market_hour": _decimal(rates.get("paper_surviving")),
            "note": (
                f"failed={funnel.paper_failed_count}; "
                f"insufficient future data={funnel.paper_unevaluable_count}; "
                f"not evaluated={funnel.paper_not_evaluated_count}"
            ),
        },
    ]


def _breakdown_rows(summary: AnalyticsSummary) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for breakdown in (
        summary.relation_type_breakdown,
        summary.relation_source_breakdown,
        summary.category_breakdown,
        summary.settlement_breakdown,
    ):
        rows.extend(_one_breakdown_rows(breakdown))
    return rows


def _one_breakdown_rows(breakdown: Breakdown) -> list[dict[str, object]]:
    return [
        {
            "dimension": breakdown.dimension,
            "label": row.label,
            "episode_count": row.episode_count,
            "share_of_episodes": _decimal(row.share_of_episodes),
            "rate_per_market_hour": _decimal(row.rate_per_market_hour),
            "additive": breakdown.additive,
            "method": breakdown.method,
            "rate_denominator_method": breakdown.rate_denominator_method,
        }
        for row in breakdown.rows
    ]


def _episode_rows(result: AnalyticsResult) -> list[dict[str, object]]:
    return [
        {
            "fact_id": item.fact_id,
            "cohort_id": item.cohort_id,
            "run_id": item.run_id,
            "opportunity_id": item.opportunity_id,
            "component_id": item.component_id,
            "opened_at": item.opened_at.astimezone(UTC),
            "terminal_status": item.terminal_status,
            "ended_at": None if item.ended_at is None else item.ended_at.astimezone(UTC),
            "duration_seconds": item.duration_seconds,
            "relation_types": list(item.relation_types),
            "relation_sources": list(item.relation_sources),
            "categories": list(item.categories),
            "settlement_bucket": item.settlement_bucket.value,
            "peak_stage": item.peak_stage,
            "peak_gross_edge": item.peak_gross_edge,
            "peak_net_edge": item.peak_net_edge,
            "peak_capacity": item.peak_capacity,
            "peak_net_guarantee": item.peak_net_guarantee,
            "midpoint_available": item.midpoint_evaluable,
            "reference_violation": item.reference_violation,
            "one_contract_gross_survived": item.one_contract_gross_survived,
            "one_contract_fee_survived": item.one_contract_fee_survived,
            "depth_executable": item.depth_executable,
            "paper_status": item.paper_status,
            "paper_locked_profit": item.paper_locked_profit,
        }
        for item in result.episodes
    ]


def _write_episodes_parquet(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    decimal_type = pa.decimal128(38, 18)
    schema = pa.schema(
        (
            pa.field("fact_id", pa.string(), nullable=False),
            pa.field("cohort_id", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("opportunity_id", pa.string(), nullable=False),
            pa.field("component_id", pa.string(), nullable=False),
            pa.field("opened_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("terminal_status", pa.string(), nullable=False),
            pa.field("ended_at", pa.timestamp("us", tz="UTC")),
            pa.field("duration_seconds", decimal_type),
            pa.field("relation_types", pa.list_(pa.string()), nullable=False),
            pa.field("relation_sources", pa.list_(pa.string()), nullable=False),
            pa.field("categories", pa.list_(pa.string()), nullable=False),
            pa.field("settlement_bucket", pa.string(), nullable=False),
            pa.field("peak_stage", pa.string(), nullable=False),
            pa.field("peak_gross_edge", decimal_type),
            pa.field("peak_net_edge", decimal_type),
            pa.field("peak_capacity", decimal_type),
            pa.field("peak_net_guarantee", decimal_type),
            pa.field("midpoint_available", pa.bool_()),
            pa.field("reference_violation", pa.bool_()),
            pa.field("one_contract_gross_survived", pa.bool_()),
            pa.field("one_contract_fee_survived", pa.bool_()),
            pa.field("depth_executable", pa.bool_()),
            pa.field("paper_status", pa.string()),
            pa.field("paper_locked_profit", decimal_type),
        ),
        metadata={
            b"arbiter_schema": b"arbiter.research.episodes",
            b"arbiter_schema_version": str(REPORT_SCHEMA_VERSION).encode("ascii"),
        },
    )
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(table, path, compression="zstd", version="2.6")


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("report CSV tables require a declared row schema")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(
    result: AnalyticsResult,
    *,
    summary_rows: Sequence[Mapping[str, object]],
    funnel_rows: Sequence[Mapping[str, object]],
    breakdown_rows: Sequence[Mapping[str, object]],
    exposure_method: str,
    warnings: Sequence[str],
) -> str:
    summary = result.summary
    lines = [
        "# Arbiter research report",
        "",
        "> This report is descriptive research evidence, not realized P&L or a claim of "
        "atomic execution. Multi-leg fills are non-atomic.",
        "",
        "## Scope and cohort",
        "",
        f"- Report schema: `{REPORT_SCHEMA_VERSION}`",
        f"- Selected source-recording cohorts: `{len(result.selection.cohorts)}`",
        f"- Unique OPEN-led episodes: `{summary.episode_count}`",
        f"- Excluded duplicate projections: `{len(result.selection.excluded_projection_run_ids)}`",
        f"- Omitted unsuccessful runs: `{len(result.selection.omitted_run_ids)}`",
        f"- Market-hour method: `{exposure_method}`",
        "",
    ]
    if warnings:
        lines.extend(("### Data-quality warnings", ""))
        lines.extend(f"- {_markdown(value)}" for value in warnings)
        lines.append("")
    lines.extend(
        (
            "Historical category, settlement, relation-type, and relation-source dimensions "
            "come only from append-only observation snapshots. Legacy NULL snapshots remain "
            "unknown; current mutable metadata is never used to rewrite them.",
            "",
            "## Diagnostic funnel",
            "",
            _markdown_table(funnel_rows),
            "",
            "Each stage uses its displayed evaluable denominator. Missing midpoint evidence and "
            "`insufficient_future_data` are unavailable/unevaluable, not failures.",
            "",
            "## Episode-level summaries",
            "",
            _markdown_table(summary_rows),
            "",
            f"**Overlap caveat:** {_markdown(summary.hypothetical_profit_note)}",
            "",
            "Closed durations and right-censored lower bounds are summarized separately. "
            "Open episodes without a terminal boundary: "
            f"`{summary.funnel.incomplete_open_duration_count}`.",
            "",
            "## Breakdowns",
            "",
            _markdown_table(breakdown_rows),
            "",
            "Relation-type, relation-source, and category tables are multi-label and therefore "
            "non-additive. Settlement buckets are single-label and additive. `unknown` means the "
            "required observation-time metadata was missing, incomplete, contradictory, or past.",
            "",
            "## Interpretation boundary",
            "",
            "The report counts recorded, deduplicated research episodes. It does not establish a "
            "venue-wide frequency without a declared collection cohort, and it does not model "
            "queue priority, hidden liquidity, market impact, custody, or guaranteed "
            "real-world fills.",
            "",
        )
    )
    return "\n".join(lines)


def _markdown_table(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return "_No rows._"
    headers = list(rows[0])
    rendered = [
        "| " + " | ".join(_markdown(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        rendered.append(
            "| " + " | ".join(_markdown(row.get(header, "")) for header in headers) + " |"
        )
    return "\n".join(rendered)


def _markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _json_value(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else value


def _json_object(value: object, label: str) -> dict[str, object]:
    decoded = _json_value(value)
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], decoded)


def _json_array(value: object, label: str) -> list[object] | None:
    if value is None:
        return None
    decoded = _json_value(value)
    if not isinstance(decoded, list):
        raise ValueError(f"{label} must be a JSON array")
    return cast(list[object], decoded)


def _optional_string_tuple(value: object, label: str) -> tuple[str, ...]:
    decoded = _json_array(value, label)
    if decoded is None:
        return ()
    if any(not isinstance(item, str) or not item.strip() for item in decoded):
        raise ValueError(f"{label} must contain nonblank strings")
    return tuple(sorted(set(cast(list[str], decoded))))


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, label)


def _optional_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _datetime(value: object, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError(f"{label} must be a datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def _optional_datetime(value: object, label: str) -> datetime | None:
    return None if value is None else _datetime(value, label)


def _optional_decimal(value: object, label: str) -> Decimal | None:
    if value is None:
        return None
    decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError(f"{label} must be finite")
    return decimal


def _decimal(value: Decimal | None) -> str:
    return "" if value is None else format(value, "f")


__all__ = [
    "ReportArtifacts",
    "ReportError",
    "generate_report",
]
