"""Bounded live scanning composition over the shared deterministic engine."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

from arbiter.config import ArbiterSettings
from arbiter.engine.scanner import (
    ArbitrageEngine,
    ComponentScanner,
    EngineRecordResult,
    EngineScanDecision,
    OpportunityLifecycle,
)
from arbiter.engine.state import EngineState, StateAction, StateUpdate
from arbiter.kalshi.auth import KalshiWebSocketAuthenticator
from arbiter.kalshi.client import KalshiRestClient
from arbiter.kalshi.websocket import (
    ConnectionInterrupted,
    EventFeeRefresh,
    KalshiWebSocketSession,
    MarketDataSession,
    MetadataRefresh,
    SessionItem,
    SubscriptionStarted,
    WebSocketProtocolError,
)
from arbiter.logging import log_event
from arbiter.models.event import Event
from arbiter.models.market import Market
from arbiter.models.orderbook import BookStatus, OrderBook
from arbiter.models.relation import Relation
from arbiter.models.series import Series
from arbiter.relations.trust import is_trusted_relation
from arbiter.replay.events import (
    BookStaleEvent,
    ConnectionInterruptedEvent,
    EventIndexAllocator,
    FeeRefreshAppliedEvent,
    FeeRefreshStartedEvent,
    MarketDataEvent,
    MarketRefreshAppliedEvent,
    MarketRefreshStartedEvent,
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
    RecordedEvent,
    RunEndedEvent,
    RunInputPayload,
    RunStartedEvent,
    SubscriptionStartedEvent,
    dump_recorded_event_json,
)
from arbiter.solver.fees import KALSHI_FEE_POLICY_VERSION
from arbiter.storage.duckdb import (
    DuckDBRepository,
    MarketObservationRecord,
    RunManifestRecord,
    RunStreamEvidence,
    market_observation_id,
)
from arbiter.storage.parquet import ParquetEventWriter

MetadataRefresher = Callable[[str], Awaitable[Market]]
EventFeeStateRefresher = Callable[[FeeRefreshStartedEvent], Awaitable[Event]]
_SCANNABLE_MARKET_STATUSES = frozenset({"active", "open"})


async def _next_session_item(iterator: AsyncIterator[SessionItem]) -> SessionItem:
    return await iterator.__anext__()


class LiveScannerError(RuntimeError):
    """Raised when local live-scan prerequisites or orchestration are unsafe."""


@dataclass(frozen=True, slots=True)
class LiveCatalog:
    """One coherent active metadata snapshot restricted to trusted components."""

    markets: tuple[Market, ...]
    events: tuple[Event, ...]
    series: tuple[Series, ...]
    relations: tuple[Relation, ...]

    @property
    def market_tickers(self) -> tuple[str, ...]:
        return tuple(market.ticker for market in self.markets)


@dataclass(frozen=True, slots=True)
class LiveScanResult:
    """Finite live outcome returned after raw and relational persistence succeeds."""

    run_id: str
    events_written: int
    scans_completed: int
    opportunity_observations: int
    files_written: tuple[Path, ...]
    resync_requests: int
    metadata_refreshes: int
    event_fee_refreshes: int


def load_live_catalog(repository: DuckDBRepository) -> LiveCatalog:
    """Build a coherent open-market view despite historical metadata upserts."""

    stored_markets = {market.ticker: market for market in repository.load_markets()}
    active_tickers = {
        ticker
        for ticker, market in stored_markets.items()
        if market.status.casefold() in _SCANNABLE_MARKET_STATUSES
    }
    active_verified_relations = tuple(
        relation
        for relation in repository.list_relations()
        if relation.verified and set(relation.market_tickers) <= active_tickers
    )
    untrusted = tuple(
        relation for relation in active_verified_relations if not is_trusted_relation(relation)
    )
    if untrusted:
        details = ", ".join(f"{relation.relation_id} ({relation.source})" for relation in untrusted)
        raise LiveScannerError(f"verified live relations have untrusted sources: {details}")
    relations = tuple(filter(is_trusted_relation, active_verified_relations))
    participant_tickers = {ticker for relation in relations for ticker in relation.market_tickers}
    markets = tuple(stored_markets[ticker] for ticker in sorted(participant_tickers))
    if not markets:
        return LiveCatalog(markets=(), events=(), series=(), relations=())

    stored_events = {event.ticker: event for event in repository.load_events()}
    members_by_event: dict[str, list[str]] = {}
    for market in markets:
        members_by_event.setdefault(market.event_ticker, []).append(market.ticker)
    missing_events = set(members_by_event) - set(stored_events)
    if missing_events:
        raise LiveScannerError(
            "trusted live markets reference missing event metadata: "
            + ", ".join(sorted(missing_events))
        )
    events = tuple(
        stored_events[event_ticker].model_copy(
            update={"market_tickers": tuple(sorted(members_by_event[event_ticker]))}
        )
        for event_ticker in sorted(members_by_event)
    )

    required_series = {
        ticker
        for ticker in (
            *(market.series_ticker for market in markets),
            *(event.series_ticker for event in events),
        )
        if ticker is not None
    }
    stored_series = {item.ticker: item for item in repository.load_series()}
    missing_series = required_series - set(stored_series)
    if missing_series:
        raise LiveScannerError(
            "trusted live markets reference missing series metadata: "
            + ", ".join(sorted(missing_series))
        )
    series = tuple(stored_series[ticker] for ticker in sorted(required_series))
    return LiveCatalog(
        markets=markets,
        events=events,
        series=series,
        relations=relations,
    )


def build_live_run_inputs(
    *,
    settings: ArbiterSettings,
    catalog: LiveCatalog,
) -> RunInputPayload:
    """Build the exact non-secret inputs shared by the run record and manifest."""

    config_payload = {
        "engine": settings.engine.model_dump(mode="json"),
        "orderbook": settings.orderbook.model_dump(mode="json"),
        "fees": settings.fees.model_dump(mode="json"),
        "paper_execution": settings.paper_execution.model_dump(mode="json"),
        "collector": settings.collector.model_dump(mode="json"),
        "kalshi_environment": settings.kalshi.environment.value,
    }
    fee_payload = {
        "policy_version": KALSHI_FEE_POLICY_VERSION,
        "events": [
            {
                "ticker": event.ticker,
                "fee_type_override": event.fee_type_override,
                "fee_multiplier_override": (
                    None
                    if event.fee_multiplier_override is None
                    else str(event.fee_multiplier_override)
                ),
                "fee_changes": [change.model_dump(mode="json") for change in event.fee_changes],
            }
            for event in catalog.events
        ],
        "series": [
            {
                "ticker": item.ticker,
                "fee_type": item.fee_type,
                "fee_multiplier": (
                    None if item.fee_multiplier is None else str(item.fee_multiplier)
                ),
            }
            for item in catalog.series
        ],
    }
    return RunInputPayload.build(
        config=config_payload,
        markets=catalog.markets,
        events=catalog.events,
        series=catalog.series,
        relations=catalog.relations,
        fee_policy=fee_payload,
    )


def build_live_manifest(
    *,
    settings: ArbiterSettings,
    catalog: LiveCatalog,
    run_id: str,
    started_at: datetime,
    start_event_index: int,
) -> RunManifestRecord:
    """Hash every deterministic live input without serializing credentials or key paths."""

    inputs = build_live_run_inputs(settings=settings, catalog=catalog)
    hashes = inputs.computed_hashes()
    return RunManifestRecord(
        run_id=run_id,
        run_type="live",
        started_at=started_at,
        manifest_version=2,
        recording_format_version=2,
        event_schema_version=2,
        recording_id=run_id,
        input_payload=inputs.model_dump(mode="json"),
        config_hash=hashes["config_hash"],
        metadata_hash=hashes["metadata_hash"],
        relations_hash=hashes["relations_hash"],
        fee_policy_hash=hashes["fee_policy_hash"],
        metadata={
            "kalshi_environment": settings.kalshi.environment.value,
            "market_tickers": list(catalog.market_tickers),
            "start_event_index": start_event_index,
        },
    )


@dataclass(frozen=True, slots=True)
class _ActiveMarketWindow:
    observation_id: str
    run_id: str
    market_ticker: str
    opened_at: datetime
    updated_at: datetime
    opened_event_index: int
    last_event_index: int
    source_event_index: int
    start_sequence: int | None
    end_sequence: int | None
    connection_id: str
    sid: int
    snapshot_id: str
    fresh_until: datetime

    def record(
        self,
        *,
        status: str = "fresh",
        closed_at: datetime | None = None,
        closed_event_index: int | None = None,
        stale_reason: str | None = None,
        resync_reason: str | None = None,
    ) -> MarketObservationRecord:
        return MarketObservationRecord(
            observation_id=self.observation_id,
            run_id=self.run_id,
            market_ticker=self.market_ticker,
            opened_at=self.opened_at,
            updated_at=self.updated_at,
            opened_event_index=self.opened_event_index,
            last_event_index=self.last_event_index,
            status=status,
            closed_at=closed_at,
            closed_event_index=closed_event_index,
            start_sequence=self.start_sequence,
            end_sequence=self.end_sequence,
            connection_id=self.connection_id,
            stale_reason=stale_reason,
            resync_reason=resync_reason,
        )


class _MarketWindowTracker:
    """Persist maximal intervals during which each reconstructed book is fresh."""

    def __init__(
        self,
        *,
        run_id: str,
        repository: DuckDBRepository,
        stale_after: timedelta,
    ) -> None:
        self.run_id = run_id
        self.repository = repository
        self.stale_after = stale_after
        self._active: dict[str, _ActiveMarketWindow] = {}
        self._sources: dict[str, MarketDataEvent] = {}

    @property
    def next_expiry(self) -> datetime | None:
        if not self._active:
            return None
        # OrderBook.is_fresh treats equality with stale_after as fresh. Datetime
        # resolution supplies the first representable instant after that boundary.
        return min(window.fresh_until + timedelta.resolution for window in self._active.values())

    def apply_event(
        self,
        event: MarketDataEvent,
        update: StateUpdate,
        books: Mapping[str, OrderBook],
    ) -> None:
        if update.action is StateAction.RESYNC_REQUIRED:
            for ticker in update.affected_tickers:
                self._sources.pop(ticker, None)
            self.close_tickers(
                update.affected_tickers,
                closed_at=event.local_received_ts,
                closed_event_index=event.event_index,
                status="resync_required",
                stale_reason=update.reason,
                resync_reason=update.reason,
            )
            return
        if update.action in {
            StateAction.SNAPSHOT_STAGED,
            StateAction.SNAPSHOT_APPLIED,
            StateAction.DELTA_APPLIED,
            StateAction.RESYNC_COMPLETE,
        }:
            self._sources[event.ticker] = event
        if update.action not in {
            StateAction.SNAPSHOT_APPLIED,
            StateAction.DELTA_APPLIED,
            StateAction.RESYNC_COMPLETE,
        }:
            return
        for ticker in update.affected_tickers:
            book = books.get(ticker)
            source = self._sources.get(ticker)
            if (
                book is not None
                and source is not None
                and book.status is BookStatus.FRESH
                and book.is_fresh(as_of=event.local_received_ts, stale_after=self.stale_after)
            ):
                self._open_or_update(
                    ticker=ticker,
                    event=event,
                    source=source,
                    book=book,
                )

    def _open_or_update(
        self,
        *,
        ticker: str,
        event: MarketDataEvent,
        source: MarketDataEvent,
        book: OrderBook,
    ) -> None:
        current = self._active.get(ticker)
        if current is None:
            current = _ActiveMarketWindow(
                observation_id=market_observation_id(
                    run_id=self.run_id,
                    market_ticker=ticker,
                    opened_event_index=event.event_index,
                ),
                run_id=self.run_id,
                market_ticker=ticker,
                opened_at=event.local_received_ts,
                updated_at=event.local_received_ts,
                opened_event_index=event.event_index,
                last_event_index=event.event_index,
                source_event_index=source.event_index,
                start_sequence=book.sequence,
                end_sequence=book.sequence,
                connection_id=source.connection_id,
                sid=source.sid,
                snapshot_id=source.snapshot_id,
                fresh_until=book.local_timestamp + self.stale_after,
            )
        else:
            current = replace(
                current,
                updated_at=event.local_received_ts,
                last_event_index=event.event_index,
                source_event_index=source.event_index,
                end_sequence=book.sequence,
                connection_id=source.connection_id,
                sid=source.sid,
                snapshot_id=source.snapshot_id,
                fresh_until=book.local_timestamp + self.stale_after,
            )
        self.repository.record_market_observation(current.record())
        self._active[ticker] = current

    def next_due_window(self, as_of: datetime) -> _ActiveMarketWindow | None:
        """Return the next expired source without mutating its durable window."""

        due = tuple(window for window in self._active.values() if window.fresh_until < as_of)
        if not due:
            return None
        return min(due, key=lambda item: (item.fresh_until, item.market_ticker))

    def close_stale(self, event: BookStaleEvent) -> None:
        """Close a window only when the persisted timer names its exact source."""

        window = self._active.get(event.ticker)
        if window is None:
            return
        source = (
            window.source_event_index,
            window.connection_id,
            window.sid,
            window.snapshot_id,
            window.end_sequence,
        )
        expected = (
            event.source_event_index,
            event.source_connection_id,
            event.source_sid,
            event.source_snapshot_id,
            event.source_sequence,
        )
        if source != expected:
            return
        self._close(
            event.ticker,
            closed_at=event.local_received_ts,
            closed_event_index=event.event_index,
            status="stale",
            stale_reason="book age reached the configured freshness window",
            resync_reason=None,
        )

    def close_tickers(
        self,
        tickers: Sequence[str],
        *,
        closed_at: datetime,
        closed_event_index: int | None,
        status: str,
        stale_reason: str | None,
        resync_reason: str | None,
    ) -> None:
        for ticker in tickers:
            window = self._active.get(ticker)
            if window is None:
                continue
            event_index = (
                window.last_event_index
                if closed_event_index is None
                else max(window.last_event_index, closed_event_index)
            )
            self._close(
                ticker,
                closed_at=max(window.updated_at, closed_at),
                closed_event_index=event_index,
                status=status,
                stale_reason=stale_reason,
                resync_reason=resync_reason,
            )

    def close_all(
        self,
        *,
        closed_at: datetime,
        closed_event_index: int,
        status: str,
    ) -> None:
        self.close_tickers(
            tuple(self._active),
            closed_at=closed_at,
            closed_event_index=closed_event_index,
            status=status,
            stale_reason=None,
            resync_reason=None,
        )

    def _close(
        self,
        ticker: str,
        *,
        closed_at: datetime,
        closed_event_index: int,
        status: str,
        stale_reason: str | None,
        resync_reason: str | None,
    ) -> None:
        window = self._active[ticker]
        self.repository.record_market_observation(
            window.record(
                status=status,
                closed_at=closed_at,
                closed_event_index=closed_event_index,
                stale_reason=stale_reason,
                resync_reason=resync_reason,
            )
        )
        del self._active[ticker]


class LiveScanner:
    """Consume one bounded session without bypassing the shared engine or persistence."""

    def __init__(
        self,
        *,
        session: MarketDataSession,
        state: EngineState,
        engine: ArbitrageEngine,
        repository: DuckDBRepository,
        writer: ParquetEventWriter,
        manifest: RunManifestRecord,
        event_index_allocator: EventIndexAllocator,
        stale_after: timedelta,
        writer_batch_size: int,
        writer_flush_interval_seconds: float,
        metadata_refresher: MetadataRefresher | None,
        event_fee_refresher: EventFeeStateRefresher | None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if engine.state is not state:
            raise ValueError("live scanner and engine must share one EngineState")
        if engine.lifecycle.store is not repository:
            raise ValueError("live lifecycle must persist through the supplied repository")
        if engine.lifecycle.run_id != manifest.run_id:
            raise ValueError("live lifecycle and manifest run IDs must match")
        if writer.next_event_index != event_index_allocator.next_event_index:
            raise ValueError("live writer and event-index allocator must start together")
        session_allocator = getattr(session, "event_index_allocator", None)
        if session_allocator is not None and session_allocator is not event_index_allocator:
            raise ValueError("live scanner and session must share one event-index allocator")
        run_inputs = RunInputPayload.model_validate_json(
            json.dumps(
                manifest.input_payload,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        hashes = run_inputs.computed_hashes()
        if any(getattr(manifest, name) != value for name, value in hashes.items()):
            raise ValueError("live manifest hashes must derive from its embedded run inputs")
        if manifest.manifest_version != 2 or manifest.event_schema_version != 2:
            raise ValueError("live scanning requires the replay-complete manifest contract")
        if stale_after < timedelta(0):
            raise ValueError("stale_after must be nonnegative")
        if writer_batch_size < 1:
            raise ValueError("writer_batch_size must be positive")
        if not math.isfinite(writer_flush_interval_seconds) or writer_flush_interval_seconds <= 0:
            raise ValueError("writer flush interval must be finite and positive")
        self.session = session
        self.state = state
        self.engine = engine
        self.repository = repository
        self.writer = writer
        self.manifest = manifest
        self.event_index_allocator = event_index_allocator
        self.run_inputs = run_inputs
        self.writer_batch_size = writer_batch_size
        self.writer_flush_interval_seconds = writer_flush_interval_seconds
        self.metadata_refresher = metadata_refresher
        self.event_fee_refresher = event_fee_refresher
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._logger = logger or logging.getLogger("arbiter.live")
        self._windows = _MarketWindowTracker(
            run_id=manifest.run_id,
            repository=repository,
            stale_after=stale_after,
        )
        self._active_connections: set[str] = set()
        self._files: list[Path] = []
        self._events_written = 0
        self._scans_completed = 0
        self._resync_requests = 0
        self._metadata_refreshes = 0
        self._event_fee_refreshes = 0
        self._latest_time = manifest.started_at
        self._last_flush = self._monotonic()
        self._stream_hasher = sha256()
        self._first_event_index: int | None = None
        self._last_event_index: int | None = None
        self._run_end_recorded = False
        self._run_end_event: RunEndedEvent | None = None

    async def run(self, *, duration_seconds: float) -> LiveScanResult:
        """Run finitely, finalizing the manifest on success, failure, or cancellation."""

        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("duration_seconds must be finite and positive")
        started = False
        next_item: asyncio.Task[SessionItem] | None = None
        self.repository.start_run(self.manifest)
        started = True
        log_event(
            self._logger,
            logging.INFO,
            "live scanner started",
            run_id=self.manifest.run_id,
            duration_seconds=duration_seconds,
        )
        try:
            await self._accept_record(
                RunStartedEvent.build(
                    event_index=self.event_index_allocator.allocate(),
                    local_received_ts=self.manifest.started_at,
                    run_id=self.manifest.run_id,
                    inputs=self.run_inputs,
                )
            )
            iterator = self.session.iter_items(duration_seconds=duration_seconds).__aiter__()
            next_item = asyncio.create_task(_next_session_item(iterator))
            while True:
                timeout = self._next_wait_timeout()
                done, _ = await asyncio.wait({next_item}, timeout=timeout)
                if next_item in done:
                    try:
                        item = next_item.result()
                    except StopAsyncIteration:
                        next_item = None
                        break
                    await self._handle_item(item)
                    self._flush_if_due()
                    # Do not prefetch across metadata/fee refreshes. The next wire
                    # message must be normalized against the refreshed boundary state.
                    next_item = asyncio.create_task(_next_session_item(iterator))
                else:
                    await self._service_due_work()
                    self._flush_if_due()

            self._record_scans(self.engine.flush())
            terminal_at = self._terminal_at()
            await self._record_run_end(status="succeeded", reason=None, terminal_at=terminal_at)
            self._flush_writer()
            self.repository.finalize_run(
                self.manifest.run_id,
                status="succeeded",
                ended_at=terminal_at,
                stream=self._stream_evidence(),
            )
            result = LiveScanResult(
                run_id=self.manifest.run_id,
                events_written=self._events_written,
                scans_completed=self._scans_completed,
                opportunity_observations=len(self.engine.lifecycle.observations),
                files_written=tuple(self._files),
                resync_requests=self._resync_requests,
                metadata_refreshes=self._metadata_refreshes,
                event_fee_refreshes=self._event_fee_refreshes,
            )
            log_event(
                self._logger,
                logging.INFO,
                "live scanner completed",
                run_id=result.run_id,
                events_written=result.events_written,
                scans_completed=result.scans_completed,
                opportunity_observations=result.opportunity_observations,
            )
            return result
        except BaseException as exc:
            await self._best_effort_failure_cleanup(exc, started=started)
            raise
        finally:
            if next_item is not None and not next_item.done():
                next_item.cancel()
                with suppress(asyncio.CancelledError):
                    await next_item

    async def _handle_item(self, item: SessionItem) -> None:
        if isinstance(
            item,
            (SubscriptionStarted, ConnectionInterrupted, MetadataRefresh, EventFeeRefresh),
        ):
            raise WebSocketProtocolError(
                "live scanning requires indexed schema-v2 session controls"
            )
        try:
            processed = await self._accept_record(item)
        except ValueError as exc:
            if isinstance(item, SubscriptionStartedEvent):
                raise WebSocketProtocolError(
                    "live order-book subscription membership does not match trusted components"
                ) from exc
            raise

        if isinstance(item, SubscriptionStartedEvent):
            self._active_connections.add(item.connection_id)
            return
        if isinstance(item, ConnectionInterruptedEvent):
            self._active_connections.discard(item.connection_id)
            log_event(
                self._logger,
                logging.WARNING,
                "market-data connection interrupted",
                connection_id=item.connection_id,
                affected_tickers=processed.state_update.affected_tickers,
            )
            return
        if isinstance(item, MarketRefreshStartedEvent):
            await self._refresh_market(item)
            return
        if isinstance(item, FeeRefreshStartedEvent):
            await self._refresh_event_fee(item)
            return
        if isinstance(item, (OrderBookSnapshotEvent, OrderBookDeltaEvent)):
            return
        raise WebSocketProtocolError(f"unsupported live session record {type(item).__name__}")

    async def _accept_record(self, record: RecordedEvent) -> EngineRecordResult:
        """Append exactly once before applying the same record to the engine."""

        self.writer.append(record)
        self._events_written += 1
        self._latest_time = max(self._latest_time, record.local_received_ts)
        self._record_stream_identity(record)
        if isinstance(record, RunEndedEvent):
            # The durable stream boundary exists even if a later relational
            # right-censor write fails. Never append a second terminal record.
            self._run_end_recorded = True
            self._run_end_event = record
        processed = self.engine.process_record(record)
        self._record_scans(processed.scans_before_record)
        self._record_scans(processed.scans_after_record)

        if isinstance(record, (RunStartedEvent, SubscriptionStartedEvent)):
            return processed
        if isinstance(record, (MarketRefreshAppliedEvent, FeeRefreshAppliedEvent)):
            return processed
        if isinstance(record, ConnectionInterruptedEvent):
            self._windows.close_tickers(
                processed.state_update.affected_tickers,
                closed_at=record.local_received_ts,
                closed_event_index=record.event_index,
                status="disconnected",
                stale_reason=record.reason,
                resync_reason="replacement snapshots required after reconnect",
            )
            return processed
        if isinstance(record, BookStaleEvent):
            self._windows.close_stale(record)
            return processed
        if isinstance(record, MarketRefreshStartedEvent):
            self._windows.close_tickers(
                processed.state_update.affected_tickers,
                closed_at=record.local_received_ts,
                closed_event_index=record.event_index,
                status="resync_required",
                stale_reason=processed.state_update.reason,
                resync_reason=processed.state_update.reason,
            )
            return processed
        if isinstance(record, FeeRefreshStartedEvent):
            return processed
        if isinstance(record, RunEndedEvent):
            censor_reason = record.reason or f"run_{record.status}"
            censored = self.engine.lifecycle.right_censor_all(
                record.local_received_ts,
                record.event_index,
                censor_reason,
            )
            for observation in censored:
                self._latest_time = max(self._latest_time, observation.observed_at)
                log_event(
                    self._logger,
                    logging.INFO,
                    "opportunity right-censored at live run boundary",
                    opportunity_id=observation.opportunity_id,
                    component_id=observation.component_id,
                    event_index=observation.event_index,
                    reason=censor_reason,
                )
            self._windows.close_all(
                closed_at=record.local_received_ts,
                closed_event_index=record.event_index,
                status="run_ended" if record.status == "succeeded" else "run_failed",
            )
            self._active_connections.clear()
            return processed

        self._windows.apply_event(record, processed.state_update, self.state.orderbooks)
        if processed.state_update.action is StateAction.RESYNC_REQUIRED:
            await self.session.request_snapshots(processed.state_update.affected_tickers)
            self._resync_requests += 1
            log_event(
                self._logger,
                logging.WARNING,
                "order-book resynchronization requested",
                ticker=record.ticker,
                update_sequence=record.sequence,
                affected_tickers=processed.state_update.affected_tickers,
                reason=processed.state_update.reason,
            )
        return processed

    async def _refresh_market(self, item: MarketRefreshStartedEvent) -> None:
        if self.metadata_refresher is None:
            raise WebSocketProtocolError(
                "market metadata changed but no bounded metadata refresher is configured"
            )
        refreshed = await self.metadata_refresher(item.ticker)
        if refreshed.ticker != item.ticker:
            raise WebSocketProtocolError("metadata refresh returned the wrong market")
        applied = MarketRefreshAppliedEvent(
            event_index=self.event_index_allocator.allocate(),
            local_received_ts=self._control_time(),
            refresh_started_event_index=item.event_index,
            market=refreshed,
        )
        await self._accept_record(applied)
        self.repository.upsert_market_state(refreshed)
        self.session.update_market_metadata(refreshed)
        if refreshed.status.casefold() not in _SCANNABLE_MARKET_STATUSES:
            raise LiveScannerError(
                f"market {item.ticker} is no longer open; synchronize metadata and restart"
            )
        await self.session.request_snapshots((item.ticker,))
        self._metadata_refreshes += 1
        self._resync_requests += 1
        log_event(
            self._logger,
            logging.INFO,
            "market metadata refreshed",
            ticker=item.ticker,
            event_type=item.reason,
        )

    async def _refresh_event_fee(self, item: FeeRefreshStartedEvent) -> None:
        if self.event_fee_refresher is None:
            raise WebSocketProtocolError(
                "event fee metadata changed but no bounded fee refresher is configured"
            )
        refreshed = await self.event_fee_refresher(item)
        if refreshed.ticker != item.event_ticker:
            raise WebSocketProtocolError("event fee refresh returned the wrong event")
        current = self.state.events[item.event_ticker]
        if (
            refreshed.series_ticker is not None
            and current.series_ticker is not None
            and refreshed.series_ticker != current.series_ticker
        ):
            raise WebSocketProtocolError("event fee refresh changed series ancestry")
        merged = refreshed.model_copy(update={"market_tickers": current.market_tickers})
        applied = FeeRefreshAppliedEvent(
            event_index=self.event_index_allocator.allocate(),
            local_received_ts=self._control_time(),
            refresh_started_event_index=item.event_index,
            event=merged,
        )
        await self._accept_record(applied)
        self.repository.upsert_event_fee_state(merged)
        self._event_fee_refreshes += 1
        log_event(
            self._logger,
            logging.INFO,
            "event fee state refreshed",
            event_ticker=item.event_ticker,
            affected_tickers=item.affected_tickers,
        )

    def _record_scans(self, decisions: Sequence[EngineScanDecision]) -> None:
        for decision in decisions:
            result = decision.result
            observation = decision.observation
            self._scans_completed += 1
            self._latest_time = max(self._latest_time, observation.observed_at)
            log_event(
                self._logger,
                logging.INFO,
                "component scan completed",
                component_id=result.component_id,
                market_tickers=result.market_tickers,
                solve_duration_ms=str(result.solve_duration_ms),
                num_states=result.num_states,
                num_instruments=result.num_instruments,
                solver_status=result.solver_status,
                scan_status=result.status.value,
                opportunity_stage=(
                    None if result.opportunity is None else result.opportunity.stage.value
                ),
                gross_profit=(
                    None if result.opportunity is None else result.opportunity.gross_profit
                ),
                net_profit=(None if result.opportunity is None else result.opportunity.net_profit),
                reason=result.reason,
            )

    def _next_wait_timeout(self) -> float | None:
        timeouts: list[float] = []
        now = self._aware_now()
        for deadline in (self.engine.next_due_at, self._windows.next_expiry):
            if deadline is not None:
                timeouts.append(max(0.0, (deadline - now).total_seconds()))
        if self.writer.pending_count:
            flush_remaining = self.writer_flush_interval_seconds - (
                self._monotonic() - self._last_flush
            )
            timeouts.append(max(0.0, flush_remaining))
        return min(timeouts) if timeouts else None

    async def _service_due_work(self) -> None:
        now = self._aware_now()
        while (window := self._windows.next_due_window(now)) is not None:
            if window.end_sequence is None:
                raise LiveScannerError("active market window has no source sequence")
            stale_at = max(
                window.fresh_until + timedelta.resolution,
                self._latest_time,
            )
            await self._accept_record(
                BookStaleEvent(
                    event_index=self.event_index_allocator.allocate(),
                    local_received_ts=stale_at,
                    ticker=window.market_ticker,
                    source_event_index=window.source_event_index,
                    source_connection_id=window.connection_id,
                    source_sid=window.sid,
                    source_snapshot_id=window.snapshot_id,
                    source_sequence=window.end_sequence,
                )
            )
        due_at = self.engine.next_due_at
        if due_at is not None and due_at <= now:
            self._record_scans(self.engine.advance_time(now))

    def _flush_if_due(self) -> None:
        if self.writer.pending_count >= self.writer_batch_size or (
            self.writer.pending_count
            and self._monotonic() - self._last_flush >= self.writer_flush_interval_seconds
        ):
            self._flush_writer()

    def _flush_writer(self) -> None:
        if self.writer.pending_count:
            self._files.extend(self.writer.flush())
        self._last_flush = self._monotonic()

    def _terminal_at(self) -> datetime:
        return max(self.manifest.started_at, self._latest_time, self._aware_now())

    def _control_time(self) -> datetime:
        """Return an aware control time that cannot move the total order backwards."""

        return max(self._latest_time, self._aware_now())

    def _aware_now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("live scanner clock must be timezone-aware")
        return value

    def _record_stream_identity(self, record: RecordedEvent) -> None:
        """Hash length-framed canonical records in their authoritative order."""

        payload = dump_recorded_event_json(record).encode()
        self._stream_hasher.update(len(payload).to_bytes(8, byteorder="big"))
        self._stream_hasher.update(payload)
        if self._first_event_index is None:
            self._first_event_index = record.event_index
        self._last_event_index = record.event_index

    def _stream_evidence(self) -> RunStreamEvidence:
        if self._first_event_index is None or self._last_event_index is None:
            raise LiveScannerError("live run has no recorded stream evidence")
        return RunStreamEvidence(
            first_event_index=self._first_event_index,
            last_event_index=self._last_event_index,
            event_count=self._events_written,
            event_stream_hash=self._stream_hasher.hexdigest(),
        )

    async def _record_run_end(
        self,
        *,
        status: str,
        reason: str | None,
        terminal_at: datetime,
    ) -> None:
        if self._run_end_recorded:
            return
        await self._accept_record(
            RunEndedEvent.model_validate(
                {
                    "event_index": self.event_index_allocator.allocate(),
                    "local_received_ts": terminal_at,
                    "run_id": self.manifest.run_id,
                    "status": status,
                    "reason": reason,
                }
            )
        )

    async def _best_effort_failure_cleanup(self, exc: BaseException, *, started: bool) -> None:
        terminal_at = self._terminal_at()
        terminal_status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        reason = f"{type(exc).__name__}: live scan terminated"
        try:
            if self.event_index_allocator.next_event_index == self.writer.next_event_index:
                terminal_index = self.event_index_allocator.next_event_index
                try:
                    await self._record_run_end(
                        status=terminal_status,
                        reason=reason,
                        terminal_at=terminal_at,
                    )
                except Exception:
                    # If append succeeded but processing failed, the persisted run end
                    # still supplies an indexed terminal boundary for observation windows.
                    if self._last_event_index == terminal_index:
                        self._windows.close_all(
                            closed_at=terminal_at,
                            closed_event_index=terminal_index,
                            status="run_failed",
                        )
                        self._run_end_recorded = True
                    raise
            if self._run_end_event is not None:
                self._windows.close_all(
                    closed_at=self._run_end_event.local_received_ts,
                    closed_event_index=self._run_end_event.event_index,
                    status="run_failed",
                )
        except Exception as cleanup_exc:
            log_event(
                self._logger,
                logging.ERROR,
                "could not record failed live run boundary",
                run_id=self.manifest.run_id,
                error_type=type(cleanup_exc).__name__,
            )
        flushed = False
        try:
            self._flush_writer()
            flushed = True
        except Exception as cleanup_exc:
            log_event(
                self._logger,
                logging.ERROR,
                "could not flush raw market data after failure",
                run_id=self.manifest.run_id,
                error_type=type(cleanup_exc).__name__,
            )
        if started:
            try:
                self.repository.finalize_run(
                    self.manifest.run_id,
                    status=terminal_status,
                    ended_at=terminal_at,
                    error=reason,
                    stream=self._stream_evidence() if flushed and self._events_written else None,
                )
            except Exception as cleanup_exc:
                log_event(
                    self._logger,
                    logging.ERROR,
                    "could not finalize failed live run",
                    run_id=self.manifest.run_id,
                    error_type=type(cleanup_exc).__name__,
                )
        log_event(
            self._logger,
            logging.ERROR,
            "live scanner terminated",
            run_id=self.manifest.run_id,
            error_type=type(exc).__name__,
        )


async def run_live_scan(
    settings: ArbiterSettings,
    *,
    duration_seconds: float,
    output_dir: Path,
) -> LiveScanResult:
    """Compose authenticated Kalshi I/O with the repository-backed live engine."""

    api_key = settings.kalshi.api_key_id
    private_key_path = settings.kalshi.private_key_path
    if api_key is None or private_key_path is None:
        raise LiveScannerError("both Kalshi WebSocket credentials are required")
    settings.storage.db_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    run_id = f"live:{started_at.strftime('%Y%m%dT%H%M%S.%fZ')}:{uuid.uuid4().hex}"

    with DuckDBRepository(
        settings.storage.db_path,
        write_max_attempts=settings.storage.write_max_attempts,
        write_initial_backoff_seconds=settings.storage.write_initial_backoff_seconds,
        write_max_backoff_seconds=settings.storage.write_max_backoff_seconds,
    ) as repository:
        catalog = load_live_catalog(repository)
        if not catalog.markets or not catalog.relations:
            raise LiveScannerError(
                "no active verified relation components are available; sync markets and "
                "discover or approve relations first"
            )
        state = EngineState(
            markets=catalog.markets,
            events=catalog.events,
            series=catalog.series,
            relations=catalog.relations,
            max_component_markets=settings.engine.max_component_markets,
        )
        if not state.subscription_tickers:
            raise LiveScannerError("trusted relation graph contains no subscribable markets")

        authenticator = KalshiWebSocketAuthenticator(
            api_key_id=api_key.get_secret_value(),
            private_key_path=private_key_path,
        )
        with (
            KalshiRestClient(settings.kalshi.rest_base_url) as client,
            ParquetEventWriter(
                output_dir,
                max_queue_size=settings.collector.writer_queue_capacity,
            ) as writer,
        ):
            event_indices = EventIndexAllocator(writer.next_event_index)
            manifest = build_live_manifest(
                settings=settings,
                catalog=catalog,
                run_id=run_id,
                started_at=started_at,
                start_event_index=writer.next_event_index,
            )
            session = KalshiWebSocketSession(
                url=settings.kalshi.websocket_url,
                authenticator=authenticator,
                markets={ticker: state.markets[ticker] for ticker in state.subscription_tickers},
                event_index_allocator=event_indices,
                inbound_queue_capacity=settings.collector.inbound_queue_capacity,
                reconnect_max_attempts=settings.collector.reconnect_max_attempts,
                reconnect_initial_backoff_seconds=(
                    settings.collector.reconnect_initial_backoff_seconds
                ),
                reconnect_max_backoff_seconds=(settings.collector.reconnect_max_backoff_seconds),
                open_timeout_seconds=settings.collector.open_timeout_seconds,
                close_timeout_seconds=settings.collector.close_timeout_seconds,
            )
            scanner = ComponentScanner(
                state,
                stale_after=timedelta(milliseconds=settings.orderbook.stale_after_ms),
                account_precision=settings.fees.account_precision,
                minimum_net_profit=settings.engine.min_net_profit_dollars,
                minimum_net_edge_bps=settings.engine.min_net_edge_bps,
            )
            lifecycle = OpportunityLifecycle(run_id=run_id, store=repository)
            engine = ArbitrageEngine(
                state=state,
                scanner=scanner,
                lifecycle=lifecycle,
                solve_debounce=timedelta(milliseconds=settings.engine.solve_debounce_ms),
            )

            async def refresh_market(ticker: str) -> Market:
                return await asyncio.to_thread(
                    client.get_market,
                    ticker,
                    refresh=True,
                )

            async def refresh_event_fee(item: FeeRefreshStartedEvent) -> Event:
                def load() -> Event:
                    event = client.get_event(
                        item.event_ticker,
                        with_nested_markets=False,
                        refresh=True,
                    )
                    changes = tuple(client.iter_event_fee_changes(event_ticker=item.event_ticker))
                    return event.model_copy(update={"fee_changes": changes})

                return await asyncio.to_thread(load)

            runner = LiveScanner(
                session=session,
                state=state,
                engine=engine,
                repository=repository,
                writer=writer,
                manifest=manifest,
                event_index_allocator=event_indices,
                stale_after=timedelta(milliseconds=settings.orderbook.stale_after_ms),
                writer_batch_size=settings.collector.writer_batch_size,
                writer_flush_interval_seconds=(settings.collector.writer_flush_interval_seconds),
                metadata_refresher=refresh_market,
                event_fee_refresher=refresh_event_fee,
            )
            return await runner.run(duration_seconds=duration_seconds)


__all__ = [
    "EventFeeStateRefresher",
    "LiveCatalog",
    "LiveScanResult",
    "LiveScanner",
    "LiveScannerError",
    "MetadataRefresher",
    "build_live_run_inputs",
    "build_live_manifest",
    "load_live_catalog",
    "run_live_scan",
]
