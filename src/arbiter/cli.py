"""Command-line entrypoint for Arbiter."""

from __future__ import annotations

import asyncio
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import date as Date
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from uuid import uuid4

import duckdb
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from arbiter import __version__
from arbiter.config import DEFAULT_CONFIG_PATH, DEFAULT_ENV_FILE, ArbiterSettings, ConfigLoadError
from arbiter.config import load_settings as load_arbiter_settings
from arbiter.engine.state import EngineState, SubscriptionStateError
from arbiter.kalshi.auth import KalshiAuthError, KalshiWebSocketAuthenticator
from arbiter.kalshi.client import KalshiRestClient, KalshiRestError
from arbiter.kalshi.websocket import (
    CollectorResult,
    KalshiWebSocketError,
    KalshiWebSocketSession,
    MarketDataCollector,
)
from arbiter.logging import configure_logging
from arbiter.models.event import Event, EventFeeChange
from arbiter.models.market import Market
from arbiter.models.relation import Relation
from arbiter.models.series import Series
from arbiter.relations.deterministic import DiscoveryDiagnostic, discover_exchange_relations
from arbiter.relations.manual import ManualRelationError, load_manual_relations
from arbiter.relations.semantic import (
    EmbeddingProvider,
    OpenAICompatibleSemanticClassifier,
    SemanticClassifier,
    SemanticError,
    SemanticMarketDocument,
    SemanticProviderUnavailable,
    SemanticReviewState,
    SemanticSuggestion,
    SentenceTransformerEmbeddingProvider,
    build_classifier_prompt,
    build_semantic_document,
    embed_semantic_documents,
    parse_classifier_output,
    retrieve_semantic_candidates,
    semantic_suggestion_id,
)
from arbiter.relations.thresholds import discover_threshold_relations
from arbiter.relations.validator import RelationValidationReport, validate_relations
from arbiter.replay.engine import (
    ReplayResult,
    ReplaySpeed,
    ReplayValidationError,
    replay_to_repository,
    split_recorded_runs,
)
from arbiter.replay.events import (
    EventIndexAllocator,
    FeeRefreshStartedEvent,
    RecordedEvent,
    RunStartedEvent,
)
from arbiter.storage.duckdb import DuckDBRepository, StorageError
from arbiter.storage.parquet import (
    ParquetEventWriter,
    ParquetStorageError,
    read_recorded_events,
)

if TYPE_CHECKING:
    from arbiter.engine.live import LiveScanResult

app = typer.Typer(
    name="arbiter",
    help="Research structural arbitrage in logically related prediction markets.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
markets_app = typer.Typer(help="Synchronize and inspect market metadata.")
relations_app = typer.Typer(help="Discover, inspect, and review logical relations.")
app.add_typer(markets_app, name="markets")
app.add_typer(relations_app, name="relations")

console = Console()


class LiveScanCommandError(RuntimeError):
    """Secret-safe live-scanner failure translated for the CLI boundary."""


@markets_app.callback(invoke_without_command=True)
def markets(ctx: typer.Context) -> None:
    """Market-data commands are implemented beginning in Milestone 3."""

    if ctx.invoked_subcommand is None:
        console.print("Market commands are not available until Milestone 3.")


@relations_app.callback(invoke_without_command=True)
def relations(ctx: typer.Context) -> None:
    """Relationship commands are implemented beginning in Milestone 4."""

    if ctx.invoked_subcommand is None:
        console.print("Relation commands are not available until Milestone 4.")


class CheckStatus(StrEnum):
    """Result severity for a local doctor check."""

    PASS = "PASS"
    INFO = "INFO"
    FAIL = "FAIL"


class SemanticReviewAction(StrEnum):
    """Human actions exposed by the semantic review CLI."""

    APPROVE = "approve"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """One safe, user-facing environment diagnostic."""

    name: str
    status: CheckStatus
    detail: str


def _directory_check(label: str, path: Path) -> DoctorCheck:
    if not path.exists():
        return DoctorCheck(label, CheckStatus.FAIL, f"Missing directory: {path}")
    if not path.is_dir():
        return DoctorCheck(label, CheckStatus.FAIL, f"Not a directory: {path}")
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        return DoctorCheck(label, CheckStatus.FAIL, f"Directory is not writable: {path}")
    return DoctorCheck(label, CheckStatus.PASS, str(path))


def _db_parent_check(path: Path) -> DoctorCheck:
    parent = path.parent
    if not parent.is_dir():
        return DoctorCheck("Database path", CheckStatus.FAIL, f"Missing parent: {parent}")
    if path.exists() and not path.is_file():
        return DoctorCheck("Database path", CheckStatus.FAIL, "Configured path is not a file.")
    try:
        if path.exists():
            probe_path = path
            temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        else:
            temporary_directory = tempfile.TemporaryDirectory(
                prefix=".arbiter-doctor-",
                dir=parent,
            )
            probe_path = Path(temporary_directory.name) / "probe.duckdb"
        try:
            connection = duckdb.connect(str(probe_path), read_only=False)
            try:
                connection.execute("BEGIN TRANSACTION")
                connection.execute("CREATE TABLE __arbiter_doctor_write_probe (value INTEGER)")
                connection.execute("ROLLBACK")
            finally:
                connection.close()
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()
    except (duckdb.Error, OSError):
        return DoctorCheck(
            "Database path",
            CheckStatus.FAIL,
            "Configured database cannot be opened for safe transactional writes.",
        )
    return DoctorCheck("Database path", CheckStatus.PASS, str(path))


def _auth_check(settings: ArbiterSettings, *, required: bool = False) -> DoctorCheck:
    key_id_present = settings.kalshi.api_key_id is not None
    key_path = settings.kalshi.private_key_path
    if not key_id_present and key_path is None:
        return DoctorCheck(
            "Kalshi WebSocket auth",
            CheckStatus.FAIL if required else CheckStatus.INFO,
            (
                "Required but not configured; both key ID and private-key path are required."
                if required
                else "Not configured; credential-dependent live checks remain unavailable."
            ),
        )
    if not key_id_present or key_path is None:
        return DoctorCheck(
            "Kalshi WebSocket auth",
            CheckStatus.FAIL,
            "Incomplete configuration; both key ID and private-key path are required.",
        )
    if not key_path.is_file() or not os.access(key_path, os.R_OK):
        return DoctorCheck(
            "Kalshi WebSocket auth",
            CheckStatus.FAIL,
            "Configured private-key file is missing or unreadable; path was not printed.",
        )
    try:
        mode = key_path.stat().st_mode
    except OSError:
        return DoctorCheck(
            "Kalshi WebSocket auth",
            CheckStatus.FAIL,
            "Could not inspect private-key permissions; path was not printed.",
        )
    if os.name == "posix" and mode & 0o077:
        return DoctorCheck(
            "Kalshi WebSocket auth",
            CheckStatus.FAIL,
            "Private-key permissions allow group/other access; restrict the file to its owner.",
        )
    return DoctorCheck(
        "Kalshi WebSocket auth",
        CheckStatus.PASS,
        "Credentials are configured and readable with owner-only key permissions; "
        "secret values and paths were not printed.",
    )


def _network_check(settings: ArbiterSettings) -> DoctorCheck:
    try:
        with KalshiRestClient(
            settings.kalshi.rest_base_url,
            timeout_seconds=5,
            max_attempts=2,
        ) as client:
            client.healthcheck()
    except KalshiRestError as exc:
        return DoctorCheck("Kalshi public REST", CheckStatus.FAIL, str(exc))
    return DoctorCheck(
        "Kalshi public REST",
        CheckStatus.PASS,
        settings.kalshi.rest_base_url,
    )


def local_doctor_checks(
    settings: ArbiterSettings,
    *,
    require_auth: bool = False,
) -> list[DoctorCheck]:
    """Run only non-network checks; missing optional credentials are informational."""

    python_ok = sys.version_info >= (3, 12)
    checks = [
        DoctorCheck(
            "Python runtime",
            CheckStatus.PASS if python_ok else CheckStatus.FAIL,
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        ),
        DoctorCheck("Configuration", CheckStatus.PASS, "YAML and environment values validated."),
        DoctorCheck(
            "Kalshi WebSocket endpoint",
            CheckStatus.PASS,
            settings.kalshi.websocket_url,
        ),
        DoctorCheck(
            "Collector bounds",
            CheckStatus.PASS,
            (
                f"queues={settings.collector.inbound_queue_capacity}/"
                f"{settings.collector.writer_queue_capacity}; "
                f"reconnect attempts={settings.collector.reconnect_max_attempts}"
            ),
        ),
        _directory_check("Data directory", settings.storage.data_dir),
        _directory_check("Raw-data directory", settings.storage.data_dir / "raw"),
        _directory_check("Parquet directory", settings.storage.data_dir / "parquet"),
        _db_parent_check(settings.storage.db_path),
        _auth_check(settings, required=require_auth),
    ]
    return checks


@app.command()
def doctor(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the normal YAML settings file.",
            exists=False,
            dir_okay=False,
        ),
    ] = DEFAULT_CONFIG_PATH,
    env_file: Annotated[
        Path | None,
        typer.Option(
            "--env-file",
            help="Optional dotenv file for secrets and environment-specific paths.",
            dir_okay=False,
        ),
    ] = DEFAULT_ENV_FILE,
    network: Annotated[
        bool,
        typer.Option(
            "--network",
            help="Perform one bounded public Kalshi REST reachability check.",
        ),
    ] = False,
    require_auth: Annotated[
        bool,
        typer.Option(
            "--require-auth",
            help="Fail when Kalshi WebSocket credentials are not fully configured.",
        ),
    ] = False,
) -> None:
    """Validate the local runtime, configuration, storage, and auth presence."""

    try:
        settings = load_arbiter_settings(config, env_file=env_file)
    except ConfigLoadError as exc:
        console.print(f"[red]Configuration failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Arbiter doctor", show_header=True)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    checks = local_doctor_checks(settings, require_auth=require_auth)
    if network:
        checks.append(_network_check(settings))
    for check in checks:
        style = {
            CheckStatus.PASS: "green",
            CheckStatus.INFO: "yellow",
            CheckStatus.FAIL: "red",
        }[check.status]
        table.add_row(check.name, f"[{style}]{check.status.value}[/{style}]", check.detail)
    console.print(table)
    console.print(
        f"Environment: {settings.kalshi.environment.value}; "
        f"semantic provider: {settings.semantic.provider}"
    )
    if any(check.status is CheckStatus.FAIL for check in checks):
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the installed Arbiter package version."""

    console.print(__version__)


@app.command()
def demo() -> None:
    """Run the offline canonical implication-arbitrage demonstration."""

    from arbiter.demo import render_demo

    render_demo(console)


async def _run_market_data_collection(
    *,
    settings: ArbiterSettings,
    tickers: tuple[str, ...],
    duration_seconds: float,
    output_dir: Path,
) -> CollectorResult:
    """Build the authenticated collector while keeping CLI validation separately testable."""

    api_key = settings.kalshi.api_key_id
    private_key_path = settings.kalshi.private_key_path
    if api_key is None or private_key_path is None:
        raise KalshiAuthError("both Kalshi WebSocket credentials are required")
    authenticator = KalshiWebSocketAuthenticator(
        api_key_id=api_key.get_secret_value(),
        private_key_path=private_key_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with KalshiRestClient(settings.kalshi.rest_base_url) as client:
        markets: dict[str, Market] = {}
        for ticker in tickers:
            market = await asyncio.to_thread(client.get_market, ticker)
            markets[ticker] = market

        async def refresh_market(ticker: str) -> Market:
            return await asyncio.to_thread(partial(client.get_market, ticker, refresh=True))

        async def refresh_event_fee(item: FeeRefreshStartedEvent) -> Event:
            def load_current_fee_state() -> Event:
                event = client.get_event(
                    item.event_ticker,
                    with_nested_markets=True,
                    refresh=True,
                )
                changes = tuple(client.iter_event_fee_changes(event_ticker=item.event_ticker))
                return event.model_copy(
                    update={
                        # Preserve the authoritative nested REST membership while
                        # attaching the separately paginated complete fee schedule.
                        "fee_changes": changes,
                    }
                )

            refreshed = await asyncio.to_thread(load_current_fee_state)
            if refreshed.ticker != item.event_ticker:
                raise KalshiRestError("event fee refresh returned the wrong event")
            return refreshed

        with ParquetEventWriter(
            output_dir,
            max_queue_size=settings.collector.writer_queue_capacity,
        ) as writer:
            event_index_allocator = EventIndexAllocator(writer.next_event_index)
            session = KalshiWebSocketSession(
                url=settings.kalshi.websocket_url,
                authenticator=authenticator,
                markets=markets,
                event_index_allocator=event_index_allocator,
                inbound_queue_capacity=settings.collector.inbound_queue_capacity,
                reconnect_max_attempts=settings.collector.reconnect_max_attempts,
                reconnect_initial_backoff_seconds=(
                    settings.collector.reconnect_initial_backoff_seconds
                ),
                reconnect_max_backoff_seconds=settings.collector.reconnect_max_backoff_seconds,
                open_timeout_seconds=settings.collector.open_timeout_seconds,
                close_timeout_seconds=settings.collector.close_timeout_seconds,
            )
            collector = MarketDataCollector(
                session=session,
                state=EngineState(),
                writer=writer,
                writer_batch_size=settings.collector.writer_batch_size,
                writer_flush_interval_seconds=(settings.collector.writer_flush_interval_seconds),
                metadata_refresher=refresh_market,
                event_fee_refresher=refresh_event_fee,
                event_index_allocator=event_index_allocator,
            )
            return await collector.run(duration_seconds=duration_seconds)


@app.command()
def collect(
    market_ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--market-ticker",
            help="Open Kalshi ticker to record; repeat for multiple related markets.",
        ),
    ] = None,
    duration_seconds: Annotated[
        float,
        typer.Option(
            "--duration-seconds",
            min=0.001,
            max=86_400,
            help="Finite collection duration.",
        ),
    ] = 30.0,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Order-book Parquet dataset root (defaults under the configured data dir).",
        ),
    ] = None,
    config: Annotated[
        Path,
        typer.Option("--config", exists=False, dir_okay=False),
    ] = DEFAULT_CONFIG_PATH,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", dir_okay=False),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """Record authenticated order-book events to replayable Parquet; never place orders."""

    tickers = tuple(sorted(market_ticker or ()))
    if not tickers or any(not ticker or ticker.strip() != ticker for ticker in tickers):
        console.print("[red]Collection requires at least one nonblank --market-ticker.[/red]")
        raise typer.Exit(code=2)
    if len(tickers) != len(set(tickers)):
        console.print("[red]Each --market-ticker must be unique.[/red]")
        raise typer.Exit(code=2)

    settings = _load_cli_settings(config, env_file)
    auth_check = _auth_check(settings)
    if auth_check.status is not CheckStatus.PASS:
        console.print(f"[red]Collection authentication failed:[/red] {auth_check.detail}")
        raise typer.Exit(code=1)
    destination = output_dir or settings.storage.data_dir / "parquet" / "orderbooks"
    try:
        result = asyncio.run(
            _run_market_data_collection(
                settings=settings,
                tickers=tickers,
                duration_seconds=duration_seconds,
                output_dir=destination,
            )
        )
    except (
        KalshiAuthError,
        KalshiRestError,
        KalshiWebSocketError,
        ParquetStorageError,
        SubscriptionStateError,
        OSError,
    ) as exc:
        console.print(f"[red]Collection failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Recorded {result.events_written} market-data events to "
        f"{len(result.files_written)} Parquet file(s); "
        f"requested {result.resync_requests} resynchronization(s); "
        f"refreshed {result.event_fee_refreshes} event fee state(s)."
    )


async def _run_live_scan_command(
    settings: ArbiterSettings,
    *,
    duration_seconds: float,
    output_dir: Path,
) -> LiveScanResult:
    """Load and invoke the live engine while preserving its safe error boundary."""

    from arbiter.engine.live import LiveScannerError, run_live_scan

    try:
        return await run_live_scan(
            settings,
            duration_seconds=duration_seconds,
            output_dir=output_dir,
        )
    except (
        LiveScannerError,
        KalshiRestError,
        KalshiWebSocketError,
        ParquetStorageError,
        StorageError,
        OSError,
        ValueError,
    ) as exc:
        detail = str(exc) or type(exc).__name__
        sensitive_values = (
            settings.kalshi.api_key_id.get_secret_value()
            if settings.kalshi.api_key_id is not None
            else None,
            str(settings.kalshi.private_key_path)
            if settings.kalshi.private_key_path is not None
            else None,
        )
        for sensitive_value in sensitive_values:
            if sensitive_value:
                detail = detail.replace(sensitive_value, "[REDACTED]")
        raise LiveScanCommandError(detail) from exc


@app.command()
def scan(
    duration_seconds: Annotated[
        float,
        typer.Option(
            "--duration-seconds",
            min=0.001,
            max=86_400,
            help="Finite live-scanning duration.",
        ),
    ] = 30.0,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            file_okay=False,
            help="Order-book Parquet dataset root (defaults under the configured data dir).",
        ),
    ] = None,
    config: Annotated[
        Path,
        typer.Option("--config", exists=False, dir_okay=False),
    ] = DEFAULT_CONFIG_PATH,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", dir_okay=False),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """Scan trusted live components and persist research observations; never trade."""

    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        console.print("[red]--duration-seconds must be finite and positive.[/red]")
        raise typer.Exit(code=2)
    settings = _load_cli_settings(config, env_file)
    auth_check = _auth_check(settings)
    if auth_check.status is not CheckStatus.PASS:
        console.print(f"[red]Scanner authentication failed:[/red] {auth_check.detail}")
        raise typer.Exit(code=1)

    destination = output_dir or settings.storage.data_dir / "parquet" / "orderbooks"
    configure_logging()
    try:
        result = asyncio.run(
            _run_live_scan_command(
                settings,
                duration_seconds=duration_seconds,
                output_dir=destination,
            )
        )
    except LiveScanCommandError as exc:
        console.print(f"[red]Live scan failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"Live scan {result.run_id} recorded {result.events_written} ordered event(s), "
        f"completed {result.scans_completed} component scan(s), and persisted "
        f"{result.opportunity_observations} opportunity observation(s) to "
        f"{len(result.files_written)} Parquet file(s)."
    )
    console.print(
        f"Requested {result.resync_requests} resynchronization(s); refreshed "
        f"{result.metadata_refreshes} market metadata record(s) and "
        f"{result.event_fee_refreshes} event fee state(s)."
    )


def _parse_replay_speed(value: str) -> ReplaySpeed:
    """Parse the CLI's human-friendly replay speed without accepting NaN or infinity."""

    if value == "max":
        return "max"
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ReplayValidationError("--speed must be 'max' or a finite positive number") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ReplayValidationError("--speed must be 'max' or a finite positive number")
    return numeric


def _select_replay_runs(
    *,
    recording_file: Path | None,
    replay_date: Date | None,
    orderbooks_root: Path,
) -> tuple[tuple[RecordedEvent, ...], ...]:
    """Load strict file input or select UTC-started runs from the full dataset root."""

    if (recording_file is None) == (replay_date is None):
        raise ReplayValidationError("provide exactly one of --file or --date")

    source = recording_file if recording_file is not None else orderbooks_root
    records = read_recorded_events(source)
    runs = split_recorded_runs(
        records,
        allow_unbounded_records=replay_date is not None,
    )
    if replay_date is None:
        return runs

    selected = tuple(
        run
        for run in runs
        if isinstance(run[0], RunStartedEvent)
        and run[0].local_received_ts.astimezone(UTC).date() == replay_date
    )
    if not selected:
        raise ReplayValidationError(
            f"no complete replay runs started on UTC date {replay_date.isoformat()}"
        )
    return selected


def _new_replay_run_id(source_run_id: str) -> str:
    """Return a fresh persistence identity that cannot equal the source identity."""

    del source_run_id
    return f"replay-{uuid4()}"


def _print_replay_result(result: ReplayResult) -> None:
    console.print(
        f"Replay {result.replay_run_id} from source {result.recorded_run_id}: "
        f"{result.event_count} event(s), {len(result.observations)} observation(s), "
        f"{len(result.paper_executions)} paper execution(s), "
        f"stream SHA-256 {result.event_stream_hash}."
    )


@app.command()
def replay(
    recording_file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            help="Self-contained recorded-event Parquet file or dataset.",
        ),
    ] = None,
    replay_date: Annotated[
        str | None,
        typer.Option(
            "--date",
            help="UTC start date (YYYY-MM-DD) in the configured order-book dataset.",
        ),
    ] = None,
    speed: Annotated[
        str,
        typer.Option(
            "--speed",
            help="Playback multiplier, or 'max' to process without sleeping.",
        ),
    ] = "max",
    config: Annotated[
        Path,
        typer.Option("--config", exists=False, dir_okay=False),
    ] = DEFAULT_CONFIG_PATH,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", dir_okay=False),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """Replay complete recorded runs through the production scanner without network access."""

    try:
        normalized_speed = _parse_replay_speed(speed)
        if (recording_file is None) == (replay_date is None):
            raise ReplayValidationError("provide exactly one of --file or --date")
        selected_date: Date | None = None
        if replay_date is not None:
            try:
                selected_date = Date.fromisoformat(replay_date)
            except ValueError as exc:
                raise ReplayValidationError("--date must use YYYY-MM-DD") from exc
            if selected_date.isoformat() != replay_date:
                raise ReplayValidationError("--date must use YYYY-MM-DD")
    except ReplayValidationError as exc:
        console.print(f"[red]Replay failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    settings = _load_cli_settings(config, env_file)
    orderbooks_root = settings.storage.data_dir / "parquet" / "orderbooks"
    settings.storage.db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        runs = _select_replay_runs(
            recording_file=recording_file,
            replay_date=selected_date,
            orderbooks_root=orderbooks_root,
        )
        with DuckDBRepository(
            settings.storage.db_path,
            write_max_attempts=settings.storage.write_max_attempts,
            write_initial_backoff_seconds=(settings.storage.write_initial_backoff_seconds),
            write_max_backoff_seconds=settings.storage.write_max_backoff_seconds,
        ) as repository:
            for records in runs:
                started = records[0]
                if not isinstance(started, RunStartedEvent):
                    raise ReplayValidationError(
                        "selected replay run does not start with a run-start record"
                    )
                result = replay_to_repository(
                    records,
                    replay_run_id=_new_replay_run_id(started.run_id),
                    repository=repository,
                    speed=normalized_speed,
                )
                _print_replay_result(result)
    except (OSError, ParquetStorageError, ReplayValidationError, StorageError) as exc:
        console.print(f"[red]Replay failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def report(
    db: Annotated[
        Path,
        typer.Option(
            "--db",
            help="Migrated Arbiter DuckDB database containing immutable run evidence.",
            dir_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for report.md plus CSV and Parquet research tables.",
            file_okay=False,
        ),
    ],
) -> None:
    """Generate a read-only, episode-deduplicated research report; never trade."""

    from arbiter.analytics.report import ReportError, generate_report

    try:
        artifacts = generate_report(db, output_dir)
    except ReportError as exc:
        console.print(f"[red]Report failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Wrote research report for {len(artifacts.analytics.episodes)} unique episode(s) "
        f"to {artifacts.output_dir}."
    )
    console.print(f"Market-hour method: {artifacts.exposure_method}.")
    for warning in artifacts.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")


def _load_cli_settings(config: Path, env_file: Path | None) -> ArbiterSettings:
    try:
        return load_arbiter_settings(config, env_file=env_file)
    except ConfigLoadError as exc:
        console.print(f"[red]Configuration failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _fetch_metadata(
    client: KalshiRestClient,
    *,
    status: str,
    max_pages: int | None,
) -> tuple[tuple[Market, ...], tuple[Event, ...], tuple[Series, ...]]:
    markets = tuple(client.iter_markets(status=status, max_pages=max_pages, enrich_series=True))
    fee_changes = tuple(client.iter_event_fee_changes())
    fee_changes_by_event: dict[str, list[EventFeeChange]] = {}
    for change in fee_changes:
        fee_changes_by_event.setdefault(change.event_ticker, []).append(change)
    markets_by_event: dict[str, list[str]] = {}
    for market in markets:
        markets_by_event.setdefault(market.event_ticker, []).append(market.ticker)

    events: list[Event] = []
    for event_ticker in sorted(markets_by_event):
        event = client.get_event(event_ticker, with_nested_markets=False)
        relevant_changes = tuple(
            sorted(
                fee_changes_by_event.get(event_ticker, ()),
                key=lambda change: (change.scheduled_ts, change.change_id),
            )
        )
        events.append(
            event.model_copy(
                update={
                    "market_tickers": tuple(sorted(markets_by_event[event_ticker])),
                    "fee_changes": relevant_changes,
                }
            )
        )
    series_tickers = sorted(
        {event.series_ticker for event in events if event.series_ticker is not None}
    )
    series = tuple(client.get_series(ticker) for ticker in series_tickers)
    return markets, tuple(events), series


@markets_app.command("sync")
def markets_sync(
    status: Annotated[
        str,
        typer.Option(help="Kalshi status: unopened, open, paused, closed, or settled."),
    ] = "open",
    max_pages: Annotated[
        int | None,
        typer.Option(min=1, help="Optional finite page limit for a bounded sync."),
    ] = None,
    config: Annotated[
        Path,
        typer.Option("--config", exists=False, dir_okay=False),
    ] = DEFAULT_CONFIG_PATH,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", dir_okay=False),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """Synchronize current public market, event, and referenced-series metadata."""

    allowed_statuses = {"unopened", "open", "paused", "closed", "settled"}
    if status not in allowed_statuses:
        console.print(f"[red]Unsupported market status:[/red] {status}")
        raise typer.Exit(code=2)
    settings = _load_cli_settings(config, env_file)
    settings.storage.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.storage.data_dir / "raw").mkdir(parents=True, exist_ok=True)
    (settings.storage.data_dir / "parquet").mkdir(parents=True, exist_ok=True)
    settings.storage.db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with KalshiRestClient(settings.kalshi.rest_base_url) as client:
            markets, events, series = _fetch_metadata(
                client,
                status=status,
                max_pages=max_pages,
            )
        with DuckDBRepository(settings.storage.db_path) as repository:
            run_id = repository.sync_metadata(markets, events, series)
    except (KalshiRestError, StorageError) as exc:
        console.print(f"[red]Market sync failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Synchronized {len(markets)} markets, {len(events)} events, and "
        f"{len(series)} series (run {run_id})."
    )


@markets_app.command("list")
def markets_list(
    limit: Annotated[int, typer.Option(min=1, max=1000)] = 100,
    config: Annotated[
        Path,
        typer.Option("--config", exists=False, dir_okay=False),
    ] = DEFAULT_CONFIG_PATH,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", dir_okay=False),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """List synchronized markets from the local research database."""

    settings = _load_cli_settings(config, env_file)
    settings.storage.db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with DuckDBRepository(settings.storage.db_path) as repository:
            rows = repository.list_markets(limit=limit)
    except StorageError as exc:
        console.print(f"[red]Could not list markets:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Synchronized markets")
    table.add_column("Ticker")
    table.add_column("Event")
    table.add_column("Series")
    table.add_column("Status")
    table.add_column("Close time")
    table.add_column("Title")
    for row in rows:
        table.add_row(
            row.ticker,
            row.event_ticker,
            row.series_ticker or "—",
            row.status,
            "—" if row.close_time is None else row.close_time.isoformat(),
            row.title,
        )
    console.print(table)
    if not rows:
        console.print("No synchronized markets found. Run `arbiter markets sync` first.")


def _print_validation(report: RelationValidationReport) -> None:
    if report.valid:
        console.print("Relation validation passed.")
        return
    table = Table(title="Relation validation issues")
    table.add_column("Code")
    table.add_column("Item")
    table.add_column("Detail")
    for issue in report.issues:
        table.add_row(issue.code, issue.item_id, issue.detail)
    console.print(table)


def _print_skips(skipped: list[DiscoveryDiagnostic]) -> None:
    if not skipped:
        return
    table = Table(title="Deterministic discovery skips")
    table.add_column("Source")
    table.add_column("Item")
    table.add_column("Reason")
    for item in skipped:
        table.add_row(item.source, item.item_id, item.reason)
    console.print(table)


@dataclass(frozen=True, slots=True)
class SemanticDiscoverySummary:
    """Counts from one optional semantic discovery pass."""

    documents: int
    document_skips: int
    candidates: int
    suggestions: int


def _new_semantic_embedding_provider(settings: ArbiterSettings) -> EmbeddingProvider:
    """Construct the lazy local adapter without importing its optional dependency."""

    return SentenceTransformerEmbeddingProvider(settings.semantic.embedding_model)


def _new_semantic_classifier(settings: ArbiterSettings) -> SemanticClassifier:
    """Validate optional provider configuration before constructing its lazy adapter."""

    provider = settings.semantic.provider
    if not settings.semantic.enabled:
        raise SemanticProviderUnavailable("the semantic classifier provider is disabled")
    model = settings.semantic.model
    if model is None:
        raise SemanticProviderUnavailable("the semantic classifier model is not configured")
    api_key = settings.semantic.api_key
    if api_key is None:
        raise SemanticProviderUnavailable("the semantic classifier API key is not configured")
    return OpenAICompatibleSemanticClassifier(
        provider=provider,
        model=model,
        api_key=api_key.get_secret_value(),
        base_url=settings.semantic.base_url,
    )


def _build_semantic_documents(
    markets: tuple[Market, ...],
    events: tuple[Event, ...],
    series: tuple[Series, ...],
) -> tuple[dict[str, SemanticMarketDocument], tuple[str, ...]]:
    """Build only ancestry-complete documents and explain every conservative skip."""

    events_by_ticker = {event.ticker: event for event in events}
    series_by_ticker = {item.ticker: item for item in series}
    documents: dict[str, SemanticMarketDocument] = {}
    skipped: list[str] = []
    for market in sorted(markets, key=lambda item: item.ticker):
        event = events_by_ticker.get(market.event_ticker)
        if event is None:
            skipped.append(f"{market.ticker}: missing parent event metadata")
            continue
        series_ticker = market.series_ticker or event.series_ticker
        parent_series = None if series_ticker is None else series_by_ticker.get(series_ticker)
        if series_ticker is not None and parent_series is None:
            skipped.append(f"{market.ticker}: missing parent series metadata {series_ticker}")
            continue
        try:
            documents[market.ticker] = build_semantic_document(
                market,
                event=event,
                series=parent_series,
            )
        except ValueError as exc:
            skipped.append(f"{market.ticker}: {exc}")
    return documents, tuple(skipped)


def _run_semantic_discovery(
    *,
    repository: DuckDBRepository,
    settings: ArbiterSettings,
    markets: tuple[Market, ...],
    events: tuple[Event, ...],
    series: tuple[Series, ...],
    limit: int,
    timestamp: datetime,
) -> SemanticDiscoverySummary:
    """Run optional retrieval/classification while leaving approval to review."""

    classifier = _new_semantic_classifier(settings)
    embedding_provider = _new_semantic_embedding_provider(settings)
    documents, document_skips = _build_semantic_documents(markets, events, series)
    if len(documents) < 2:
        return SemanticDiscoverySummary(
            documents=len(documents),
            document_skips=len(document_skips),
            candidates=0,
            suggestions=0,
        )

    cached_embeddings = repository.list_semantic_embeddings(
        provider=embedding_provider.provider,
        model=embedding_provider.model,
    )
    embeddings = embed_semantic_documents(
        tuple(documents.values()),
        embedding_provider,
        cached_embeddings=cached_embeddings,
        clock=lambda: timestamp,
    )
    repository.upsert_semantic_embeddings(embeddings)
    candidates = retrieve_semantic_candidates(
        tuple(documents.values()),
        {embedding.market_ticker: embedding.vector for embedding in embeddings},
        top_k=settings.relations.nearest_neighbors,
    )

    suggestions: list[SemanticSuggestion] = []
    for candidate in candidates[:limit]:
        market_a = documents[candidate.market_a_ticker]
        market_b = documents[candidate.market_b_ticker]
        prompt = build_classifier_prompt(
            market_a,
            market_b,
            prompt_version=settings.semantic.prompt_version,
        )
        raw_response = classifier.classify(prompt)
        parsed = parse_classifier_output(raw_response)
        suggestions.append(
            SemanticSuggestion(
                suggestion_id=semantic_suggestion_id(market_a.ticker, market_b.ticker),
                market_a_ticker=market_a.ticker,
                market_b_ticker=market_b.ticker,
                market_a_text_hash=market_a.canonical_text_hash,
                market_b_text_hash=market_b.canonical_text_hash,
                market_a_rules_hash=market_a.rules_hash,
                market_b_rules_hash=market_b.rules_hash,
                market_a_timing_hash=market_a.timing_hash,
                market_b_timing_hash=market_b.timing_hash,
                market_a_title=market_a.title,
                market_b_title=market_b.title,
                market_a_rules_text=market_a.rules_text,
                market_b_rules_text=market_b.rules_text,
                market_a_timing_text=market_a.timing_text,
                market_b_timing_text=market_b.timing_text,
                embedding_provider=embedding_provider.provider,
                embedding_model=embedding_provider.model,
                cosine_similarity=candidate.cosine_similarity,
                classifier_provider=classifier.provider,
                classifier_model=classifier.model,
                prompt_version=settings.semantic.prompt_version,
                prompt=prompt,
                raw_response=raw_response,
                relation=parsed.relation,
                confidence=parsed.confidence,
                rationale=parsed.rationale,
                requires_review=parsed.requires_review,
                review_state=SemanticReviewState.PENDING,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
    repository.upsert_semantic_suggestions(tuple(suggestions))
    return SemanticDiscoverySummary(
        documents=len(documents),
        document_skips=len(document_skips),
        candidates=min(len(candidates), limit),
        suggestions=len(suggestions),
    )


@relations_app.command("discover")
def relations_discover(
    manual: Annotated[
        Path | None,
        typer.Option(
            "--manual",
            dir_okay=False,
            help="Optional strict manual relation YAML file.",
        ),
    ] = None,
    no_exchange: Annotated[
        bool,
        typer.Option("--no-exchange", help="Skip exchange-declared mutual exclusion."),
    ] = False,
    no_thresholds: Annotated[
        bool,
        typer.Option("--no-thresholds", help="Skip structured threshold discovery."),
    ] = False,
    semantic: Annotated[
        bool,
        typer.Option(
            "--semantic",
            help="Run optional embedding retrieval and semantic proposal generation.",
        ),
    ] = False,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            max=10_000,
            help="Maximum semantic candidates to classify and persist.",
        ),
    ] = 10,
    config: Annotated[
        Path,
        typer.Option("--config", exists=False, dir_okay=False),
    ] = DEFAULT_CONFIG_PATH,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", dir_okay=False),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """Persist trusted deterministic relations, then optionally propose semantic ones."""

    settings = _load_cli_settings(config, env_file)
    manual_path = manual or settings.relations.manual_path
    settings.storage.db_path.parent.mkdir(parents=True, exist_ok=True)
    discovered: list[Relation] = []
    skipped: list[DiscoveryDiagnostic] = []
    semantic_summary: SemanticDiscoverySummary | None = None
    semantic_skip: str | None = None
    try:
        with DuckDBRepository(settings.storage.db_path) as repository:
            markets = repository.load_markets()
            events = repository.load_events()
            series = repository.load_series()
            if not markets:
                console.print(
                    "[red]No synchronized markets found; run `arbiter markets sync` first.[/red]"
                )
                raise typer.Exit(code=1)
            known_tickers = {market.ticker for market in markets}
            timestamp = datetime.now(UTC)
            if manual_path is not None:
                discovered.extend(
                    load_manual_relations(
                        manual_path,
                        known_market_tickers=known_tickers,
                        created_at=timestamp,
                    )
                )
            if settings.relations.discover_exchange and not no_exchange:
                batch = discover_exchange_relations(
                    events,
                    known_market_tickers=known_tickers,
                    created_at=timestamp,
                )
                discovered.extend(batch.relations)
                skipped.extend(batch.skipped)
            if settings.relations.discover_thresholds and not no_thresholds:
                batch = discover_threshold_relations(
                    markets,
                    events={event.ticker: event for event in events},
                    series={item.ticker: item for item in series},
                    created_at=timestamp,
                )
                discovered.extend(batch.relations)
                skipped.extend(batch.skipped)

            existing = {relation.relation_id: relation for relation in repository.list_relations()}
            for relation in discovered:
                existing[relation.relation_id] = relation
            report = validate_relations(
                tuple(existing.values()),
                known_market_tickers=known_tickers,
                max_component_markets=settings.engine.max_component_markets,
            )
            if not report.valid:
                _print_validation(report)
                raise typer.Exit(code=1)
            repository.upsert_relations(tuple(discovered))
            if semantic or settings.relations.semantic_enabled:
                try:
                    semantic_summary = _run_semantic_discovery(
                        repository=repository,
                        settings=settings,
                        markets=markets,
                        events=events,
                        series=series,
                        limit=limit,
                        timestamp=timestamp,
                    )
                except (SemanticError, ValueError) as exc:
                    semantic_skip = str(exc) or type(exc).__name__
    except (ManualRelationError, StorageError) as exc:
        console.print(f"[red]Relation discovery failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    _print_skips(skipped)
    console.print(
        f"Discovered and upserted {len(discovered)} trusted relations; "
        f"skipped {len(skipped)} uncertain items."
    )
    if semantic_skip is not None:
        console.print(f"[yellow]Semantic discovery skipped:[/yellow] {semantic_skip}")
    elif semantic_summary is not None:
        console.print(
            f"Semantic discovery built {semantic_summary.documents} document(s), skipped "
            f"{semantic_summary.document_skips} incomplete document(s), selected "
            f"{semantic_summary.candidates} candidate(s), and persisted "
            f"{semantic_summary.suggestions} pending suggestion(s)."
        )


def _print_semantic_suggestions(suggestions: tuple[SemanticSuggestion, ...]) -> None:
    """Display the exact evidence required for a human trust decision."""

    table = Table(title="Semantic relation review", show_lines=True)
    table.add_column("Suggestion")
    table.add_column("State")
    table.add_column("Proposal")
    table.add_column("Confidence")
    table.add_column("Rationale")
    for suggestion in suggestions:
        table.add_row(
            suggestion.suggestion_id,
            suggestion.review_state.value,
            suggestion.relation.value,
            f"{suggestion.confidence:.6f}",
            Text(suggestion.rationale),
        )
    console.print(table)
    for suggestion in suggestions:
        console.print(Text(f"{suggestion.suggestion_id} market A:"), soft_wrap=True)
        console.print(
            Text(f"{suggestion.market_a_ticker} — {suggestion.market_a_title}"),
            soft_wrap=True,
        )
        console.print(Text(suggestion.market_a_rules_text), soft_wrap=True)
        console.print(Text(f"{suggestion.suggestion_id} market B:"), soft_wrap=True)
        console.print(
            Text(f"{suggestion.market_b_ticker} — {suggestion.market_b_title}"),
            soft_wrap=True,
        )
        console.print(Text(suggestion.market_b_rules_text), soft_wrap=True)


def _semantic_review_state(action: SemanticReviewAction) -> SemanticReviewState:
    return {
        SemanticReviewAction.APPROVE: SemanticReviewState.APPROVED,
        SemanticReviewAction.REJECT: SemanticReviewState.REJECTED,
        SemanticReviewAction.UNCERTAIN: SemanticReviewState.UNCERTAIN,
    }[action]


@relations_app.command("review")
def relations_review(
    suggestion_id: Annotated[
        str | None,
        typer.Option(
            "--suggestion-id",
            help="Suggestion to review; omit both review options to list the queue.",
        ),
    ] = None,
    action: Annotated[
        SemanticReviewAction | None,
        typer.Option(
            "--action",
            case_sensitive=False,
            help="Human decision: approve, reject, or uncertain.",
        ),
    ] = None,
    config: Annotated[
        Path,
        typer.Option("--config", exists=False, dir_okay=False),
    ] = DEFAULT_CONFIG_PATH,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", dir_okay=False),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """List pending evidence or atomically apply one explicit human review action."""

    if (suggestion_id is None) != (action is None):
        console.print("[red]--suggestion-id and --action must be provided together.[/red]")
        raise typer.Exit(code=2)
    settings = _load_cli_settings(config, env_file)
    settings.storage.db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with DuckDBRepository(settings.storage.db_path) as repository:
            if suggestion_id is None:
                queued = repository.list_semantic_suggestions(
                    states={SemanticReviewState.PENDING, SemanticReviewState.STALE}
                )
                if queued:
                    _print_semantic_suggestions(queued)
                else:
                    console.print("No pending or stale semantic suggestions require review.")
                return

            assert action is not None
            suggestion = repository.get_semantic_suggestion(suggestion_id)
            if suggestion is None:
                raise ValueError(f"semantic suggestion does not exist: {suggestion_id}")
            _print_semantic_suggestions((suggestion,))
            documents, _ = _build_semantic_documents(
                repository.load_markets(),
                repository.load_events(),
                repository.load_series(),
            )
            reviewed = repository.review_semantic_suggestion(
                suggestion_id,
                action=_semantic_review_state(action),
                documents=documents,
                max_component_markets=settings.engine.max_component_markets,
            )
    except (StorageError, ValueError) as exc:
        console.print(f"[red]Semantic review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if reviewed.review_state is SemanticReviewState.STALE:
        console.print(
            "[yellow]Semantic suggestion is stale and was not approved:[/yellow] "
            f"{reviewed.stale_reason or 'current evidence changed'}"
        )
        raise typer.Exit(code=1)
    console.print(
        f"Semantic suggestion {reviewed.suggestion_id} is now {reviewed.review_state.value}."
    )


@relations_app.command("list")
def relations_list(
    config: Annotated[
        Path,
        typer.Option("--config", exists=False, dir_okay=False),
    ] = DEFAULT_CONFIG_PATH,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", dir_okay=False),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """List persisted trusted relations."""

    settings = _load_cli_settings(config, env_file)
    settings.storage.db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with DuckDBRepository(settings.storage.db_path) as repository:
            rows = repository.list_relations()
    except StorageError as exc:
        console.print(f"[red]Could not list relations:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Trusted relations")
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Markets")
    table.add_column("Source")
    table.add_column("Verified")
    for relation in rows:
        table.add_row(
            relation.relation_id,
            relation.relation_type.value,
            ", ".join(relation.market_tickers),
            relation.source,
            "yes" if relation.verified else "no",
        )
    console.print(table)
    if not rows:
        console.print("No relations found. Run `arbiter relations discover` first.")


@relations_app.command("validate")
def relations_validate(
    config: Annotated[
        Path,
        typer.Option("--config", exists=False, dir_okay=False),
    ] = DEFAULT_CONFIG_PATH,
    env_file: Annotated[
        Path | None,
        typer.Option("--env-file", dir_okay=False),
    ] = DEFAULT_ENV_FILE,
) -> None:
    """Validate relation references, duplicates, component size, and satisfiability."""

    settings = _load_cli_settings(config, env_file)
    settings.storage.db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with DuckDBRepository(settings.storage.db_path) as repository:
            report = validate_relations(
                repository.list_relations(),
                known_market_tickers=repository.known_market_tickers(),
                max_component_markets=settings.engine.max_component_markets,
            )
    except StorageError as exc:
        console.print(f"[red]Relation validation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _print_validation(report)
    if not report.valid:
        raise typer.Exit(code=1)


def main() -> None:
    """Invoke the Typer application."""

    app()
