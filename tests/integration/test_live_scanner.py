"""End-to-end live scanner orchestration over deterministic session items."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.config import ArbiterSettings
from arbiter.engine.live import (
    EventFeeStateRefresher,
    LiveCatalog,
    LiveScanner,
    LiveScannerError,
    MetadataRefresher,
    build_live_manifest,
    load_live_catalog,
)
from arbiter.engine.scanner import ArbitrageEngine, ComponentScanner, OpportunityLifecycle
from arbiter.engine.state import EngineState
from arbiter.kalshi.websocket import (
    SessionItem,
    WebSocketProtocolError,
)
from arbiter.models.event import Event, EventFeeChange
from arbiter.models.market import Market, PriceRange
from arbiter.models.opportunity import OpportunityObservation
from arbiter.models.orderbook import BookStatus, PriceLevel
from arbiter.models.relation import Relation, RelationType
from arbiter.models.series import Series
from arbiter.replay.engine import recorded_event_stream_hash
from arbiter.replay.events import (
    BookStaleEvent,
    ConnectionInterruptedEvent,
    EventIndexAllocator,
    FeeRefreshAppliedEvent,
    FeeRefreshStartedEvent,
    MarketRefreshAppliedEvent,
    MarketRefreshStartedEvent,
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
    RunEndedEvent,
    RunStartedEvent,
    SubscriptionStartedEvent,
)
from arbiter.solver.fees import ZeroFeeModel
from arbiter.storage.duckdb import DuckDBRepository, StorageError
from arbiter.storage.parquet import (
    ParquetEventWriter,
    read_market_data_events,
    read_recorded_events,
)

BASE = datetime(2026, 9, 3, 12, tzinfo=UTC)
STALE_AFTER = timedelta(seconds=2)
DEBOUNCE = timedelta(milliseconds=5)


class _RealtimeEventClock:
    """Map a local monotonic clock onto a deterministic aware event-time epoch."""

    def __init__(self) -> None:
        self._started = time.perf_counter()

    def __call__(self) -> datetime:
        return BASE + timedelta(seconds=time.perf_counter() - self._started)


class _FakeSession:
    """Yield phases immediately, pausing at explicit gates between them."""

    def __init__(
        self,
        phases: Sequence[Sequence[SessionItem]],
        *,
        terminal_error: BaseException | None = None,
    ) -> None:
        if not phases:
            raise ValueError("a fake session requires at least one phase")
        self.phases = tuple(tuple(phase) for phase in phases)
        self.gates = tuple(asyncio.Event() for _ in self.phases[:-1])
        self.terminal_error = terminal_error
        self.requested_snapshots: list[tuple[str, ...]] = []
        self.updated_markets: list[Market] = []
        self.yielded_items: list[SessionItem] = []
        self.requested_duration: float | None = None
        self._event_indices: EventIndexAllocator | None = None

    @property
    def event_index_allocator(self) -> EventIndexAllocator | None:
        return self._event_indices

    def bind_event_index_allocator(self, allocator: EventIndexAllocator) -> None:
        if self._event_indices is not None:
            raise ValueError("fake session allocator is already bound")
        self._event_indices = allocator

    async def iter_items(self, *, duration_seconds: float) -> AsyncIterator[SessionItem]:
        if self._event_indices is None:
            raise AssertionError("fake session requires a shared event-index allocator")
        self.requested_duration = duration_seconds
        for index, phase in enumerate(self.phases):
            for item in phase:
                if not hasattr(item, "model_copy"):
                    raise AssertionError("fake live sessions must contain recorded events")
                recorded = item.model_copy(update={"event_index": self._event_indices.allocate()})
                self.yielded_items.append(recorded)
                yield recorded
            if index < len(self.gates):
                await self.gates[index].wait()
        if self.terminal_error is not None:
            raise self.terminal_error

    async def request_snapshots(self, tickers: Sequence[str]) -> None:
        self.requested_snapshots.append(tuple(tickers))

    def update_market_metadata(self, market: Market) -> None:
        self.updated_markets.append(market)


class _FailingTransitionRepository(DuckDBRepository):
    """Use real DuckDB for every write except the injected persistent transition failure."""

    def persist_transition(self, observation: OpportunityObservation) -> None:
        del observation
        raise StorageError("synthetic persistent opportunity write failure")


def _catalog() -> LiveCatalog:
    price_ranges = (
        PriceRange(
            start=Decimal("0.0000"),
            end=Decimal("1.0000"),
            step=Decimal("0.0100"),
        ),
    )
    markets = tuple(
        Market(
            ticker=ticker,
            event_ticker="EVENT",
            series_ticker="SERIES",
            title=ticker,
            status="active",
            price_ranges=price_ranges,
            raw={"ticker": ticker, "event_ticker": "EVENT", "status": "active"},
        )
        for ticker in ("A", "M")
    )
    event = Event(
        ticker="EVENT",
        series_ticker="SERIES",
        title="Fixture event",
        market_tickers=("A", "M"),
        raw={"event_ticker": "EVENT", "series_ticker": "SERIES"},
    )
    series = Series(
        ticker="SERIES",
        title="Fixture series",
        fee_type="quadratic",
        fee_multiplier=Decimal("1"),
        raw={"ticker": "SERIES", "title": "Fixture series"},
    )
    relation = Relation(
        relation_id="m-implies-a",
        market_tickers=("M", "A"),
        relation_type=RelationType.IMPLIES,
        source="manual",
        verified=True,
        rationale="M implies A",
        created_at=BASE,
        antecedent="M",
        consequent="A",
    )
    return LiveCatalog(
        markets=markets,
        events=(event,),
        series=(series,),
        relations=(relation,),
    )


def _snapshot(
    ticker: str,
    *,
    event_index: int,
    sequence: int,
    yes_bid: str,
    no_bid: str,
    received_at: datetime = BASE,
) -> OrderBookSnapshotEvent:
    return OrderBookSnapshotEvent(
        event_index=event_index,
        local_received_ts=received_at,
        exchange_ts=received_at,
        ticker=ticker,
        sequence=sequence,
        sid=7,
        connection_id="connection-1",
        snapshot_id=f"snapshot:{ticker}",
        yes_bids=(PriceLevel(price=Decimal(yes_bid), quantity=Decimal("5.00")),),
        no_bids=(PriceLevel(price=Decimal(no_bid), quantity=Decimal("5.00")),),
    )


def _delta(
    ticker: str,
    *,
    event_index: int,
    sequence: int,
    side: str,
    price: str,
    quantity_delta: str,
    received_at: datetime,
) -> OrderBookDeltaEvent:
    return OrderBookDeltaEvent.model_validate(
        {
            "event_index": event_index,
            "local_received_ts": received_at,
            "exchange_ts": received_at,
            "ticker": ticker,
            "sequence": sequence,
            "sid": 7,
            "connection_id": "connection-1",
            "snapshot_id": f"snapshot:{ticker}",
            "side": side,
            "price": Decimal(price),
            "quantity_delta": Decimal(quantity_delta),
        }
    )


def _subscription(*, tickers: tuple[str, ...] = ("A", "M")) -> SubscriptionStartedEvent:
    return SubscriptionStartedEvent(
        event_index=0,
        local_received_ts=BASE,
        connection_id="connection-1",
        sid=7,
        tickers=tickers,
    )


def _seed(repository: DuckDBRepository, catalog: LiveCatalog, *, suffix: str) -> None:
    repository.sync_metadata(
        catalog.markets,
        catalog.events,
        catalog.series,
        run_id=f"metadata-{suffix}",
    )
    repository.upsert_relations(catalog.relations)


def _build_runner(
    *,
    session: _FakeSession,
    repository: DuckDBRepository,
    writer: ParquetEventWriter,
    run_id: str,
    clock: Callable[[], datetime],
    metadata_refresher: MetadataRefresher | None = None,
    event_fee_refresher: EventFeeStateRefresher | None = None,
    stale_after: timedelta = STALE_AFTER,
    solve_debounce: timedelta = DEBOUNCE,
) -> tuple[LiveScanner, EngineState, OpportunityLifecycle]:
    catalog = _catalog()
    state = EngineState(
        markets=catalog.markets,
        events=catalog.events,
        series=catalog.series,
        relations=catalog.relations,
    )
    scanner = ComponentScanner(
        state,
        stale_after=stale_after,
        fee_model=ZeroFeeModel(),
        minimum_net_profit=Decimal("0"),
        minimum_net_edge_bps=Decimal("0"),
    )
    lifecycle = OpportunityLifecycle(run_id=run_id, store=repository)
    engine = ArbitrageEngine(
        state=state,
        scanner=scanner,
        lifecycle=lifecycle,
        solve_debounce=solve_debounce,
    )
    event_indices = EventIndexAllocator(writer.next_event_index)
    session.bind_event_index_allocator(event_indices)
    manifest = build_live_manifest(
        settings=ArbiterSettings(),
        catalog=catalog,
        run_id=run_id,
        started_at=BASE,
        start_event_index=writer.next_event_index,
    )

    async def no_market_refresh(ticker: str) -> Market:
        raise AssertionError(f"unexpected market refresh for {ticker}")

    return (
        LiveScanner(
            session=session,
            state=state,
            engine=engine,
            repository=repository,
            writer=writer,
            manifest=manifest,
            event_index_allocator=event_indices,
            stale_after=stale_after,
            writer_batch_size=100,
            writer_flush_interval_seconds=60,
            metadata_refresher=metadata_refresher or no_market_refresh,
            event_fee_refresher=event_fee_refresher,
            clock=clock,
        ),
        state,
        lifecycle,
    )


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for deterministic live-runner state")
        await asyncio.sleep(0.002)


def test_live_catalog_excludes_historical_closed_components_and_rebuilds_membership(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    closed_markets = tuple(
        catalog.markets[0].model_copy(
            update={
                "ticker": ticker,
                "title": ticker,
                "status": "closed",
                "raw": {
                    **catalog.markets[0].raw,
                    "ticker": ticker,
                    "status": "closed",
                },
            }
        )
        for ticker in ("C", "D")
    )
    event = catalog.events[0].model_copy(update={"market_tickers": ("A", "C", "D", "M")})
    historical_relation = Relation(
        relation_id="c-implies-d",
        market_tickers=("C", "D"),
        relation_type=RelationType.IMPLIES,
        source="manual",
        verified=True,
        rationale="historical closed component",
        created_at=BASE,
        antecedent="C",
        consequent="D",
    )

    with DuckDBRepository(tmp_path / "catalog.duckdb") as repository:
        repository.sync_metadata(
            catalog.markets + closed_markets,
            (event,),
            catalog.series,
            run_id="metadata-catalog",
        )
        repository.upsert_relations(catalog.relations + (historical_relation,))

        loaded = load_live_catalog(repository)

    assert loaded.market_tickers == ("A", "M")
    assert tuple(relation.relation_id for relation in loaded.relations) == ("m-implies-a",)
    assert loaded.events[0].market_tickers == ("A", "M")
    EngineState(
        markets=loaded.markets,
        events=loaded.events,
        series=loaded.series,
        relations=loaded.relations,
    )


def test_live_catalog_rejects_forged_verified_relation_source(tmp_path: Path) -> None:
    catalog = _catalog()
    forged = catalog.relations[0].model_copy(
        update={
            "relation_id": "forged-semantic",
            "source": "semantic",
            "confidence": 1.0,
        }
    )

    with DuckDBRepository(tmp_path / "forged-catalog.duckdb") as repository:
        repository.sync_metadata(
            catalog.markets,
            catalog.events,
            catalog.series,
            run_id="metadata-catalog",
        )
        repository.upsert_relations((forged,))

        with pytest.raises(LiveScannerError, match="untrusted sources.*forged-semantic"):
            load_live_catalog(repository)


@pytest.mark.asyncio
async def test_live_stream_idles_between_open_update_and_close_and_persists_everything(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    subscription = _subscription()
    session = _FakeSession(
        (
            (
                subscription,
                _snapshot("A", event_index=0, sequence=1, yes_bid="0.60", no_bid="0.38"),
                _snapshot("M", event_index=1, sequence=2, yes_bid="0.70", no_bid="0.28"),
            ),
            (
                _delta(
                    "A",
                    event_index=2,
                    sequence=3,
                    side="no",
                    price="0.40",
                    quantity_delta="5.00",
                    received_at=BASE + timedelta(milliseconds=50),
                ),
            ),
            (
                _delta(
                    "M",
                    event_index=3,
                    sequence=4,
                    side="yes",
                    price="0.70",
                    quantity_delta="-5.00",
                    received_at=BASE + timedelta(milliseconds=100),
                ),
            ),
        )
    )
    raw_root = tmp_path / "raw"

    with (
        DuckDBRepository(tmp_path / "live.duckdb") as repository,
        ParquetEventWriter(raw_root) as writer,
    ):
        _seed(repository, catalog, suffix="normal")
        clock = _RealtimeEventClock()
        runner, _, lifecycle = _build_runner(
            session=session,
            repository=repository,
            writer=writer,
            run_id="live-normal",
            clock=clock,
        )
        task = asyncio.create_task(runner.run(duration_seconds=1))

        await _wait_until(lambda: len(lifecycle.observations) == 1)
        assert not task.done(), "OPEN must be emitted during idle time, before EOF flush"
        session.gates[0].set()
        await _wait_until(lambda: len(lifecycle.observations) == 2)
        assert not task.done(), "UPDATED must be emitted during idle time"
        session.gates[1].set()
        result = await task

        manifest = repository.get_run_manifest("live-normal")
        transitions = repository.connection.execute(
            "SELECT transition, event_index FROM opportunity_observations ORDER BY event_index"
        ).fetchall()
        summary = repository.connection.execute(
            "SELECT detected_at, ended_at FROM opportunities"
        ).fetchone()
        windows = repository.connection.execute(
            """
            SELECT market_ticker, status, closed_at, last_event_index
            FROM market_observation_windows
            ORDER BY market_ticker
            """
        ).fetchall()

        assert result.events_written == 7
        assert result.scans_completed == 3
        assert result.opportunity_observations == 3
        assert manifest is not None and manifest.status == "succeeded"
        assert manifest.manifest_version == 2
        assert manifest.recording_format_version == 2
        assert manifest.event_schema_version == 2
        assert manifest.first_event_index == 0
        assert manifest.last_event_index == 6
        assert manifest.event_count == 7
        assert manifest.event_stream_hash is not None
        assert transitions == [("open", 3), ("updated", 4), ("closed", 5)]
        assert summary == (
            BASE + DEBOUNCE,
            BASE + timedelta(milliseconds=100) + DEBOUNCE,
        )
        assert [(row[0], row[1], row[3]) for row in windows] == [
            ("A", "run_ended", 4),
            ("M", "run_ended", 5),
        ]
        assert all(row[2] is not None for row in windows)

    restored = read_recorded_events(raw_root)
    assert [event.event_index for event in restored] == list(range(7))
    assert [event.event_type for event in restored] == [
        "run_started",
        "subscription_started",
        "snapshot",
        "snapshot",
        "delta",
        "delta",
        "run_ended",
    ]
    assert isinstance(restored[0], RunStartedEvent)
    assert isinstance(restored[-1], RunEndedEvent)
    assert restored[0].inputs.model_dump(mode="json") == manifest.input_payload
    assert restored[0].config_hash == manifest.config_hash
    assert restored[0].metadata_hash == manifest.metadata_hash
    assert restored[0].relations_hash == manifest.relations_hash
    assert restored[0].fee_policy_hash == manifest.fee_policy_hash
    assert manifest.event_stream_hash == recorded_event_stream_hash(restored)


@pytest.mark.asyncio
async def test_live_staleness_is_indexed_before_equal_time_debounce(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    stale_after = timedelta(milliseconds=20)
    # OrderBook freshness includes the exact age boundary, so the first stale
    # instant is one microsecond later. Make the debounce exactly that instant.
    debounce = stale_after + timedelta.resolution
    session = _FakeSession(
        (
            (
                _subscription(),
                _snapshot("A", event_index=0, sequence=1, yes_bid="0.60", no_bid="0.38"),
                _snapshot("M", event_index=0, sequence=2, yes_bid="0.70", no_bid="0.28"),
            ),
            (),
        )
    )
    raw_root = tmp_path / "stale-raw"

    with (
        DuckDBRepository(tmp_path / "stale.duckdb") as repository,
        ParquetEventWriter(raw_root) as writer,
    ):
        _seed(repository, catalog, suffix="stale")
        runner, _, lifecycle = _build_runner(
            session=session,
            repository=repository,
            writer=writer,
            run_id="live-stale",
            clock=_RealtimeEventClock(),
            stale_after=stale_after,
            solve_debounce=debounce,
        )
        task = asyncio.create_task(runner.run(duration_seconds=1))
        await _wait_until(lambda: len(lifecycle.observations) >= 2)
        assert all(item.transition.value != "open" for item in lifecycle.observations)
        session.gates[0].set()
        result = await task
        windows = repository.connection.execute(
            """
            SELECT market_ticker, status, closed_event_index
            FROM market_observation_windows
            ORDER BY market_ticker
            """
        ).fetchall()

        assert result.events_written == 7
        assert windows == [("A", "stale", 4), ("M", "stale", 5)]

    records = read_recorded_events(raw_root)
    stale_records = tuple(record for record in records if isinstance(record, BookStaleEvent))
    assert [record.event_index for record in stale_records] == [4, 5]
    assert [record.source_event_index for record in stale_records] == [2, 3]
    assert [record.source_sequence for record in stale_records] == [1, 2]
    assert [record.source_connection_id for record in stale_records] == [
        "connection-1",
        "connection-1",
    ]
    assert [record.source_sid for record in stale_records] == [7, 7]
    assert [record.source_snapshot_id for record in stale_records] == [
        "snapshot:A",
        "snapshot:M",
    ]


@pytest.mark.asyncio
async def test_indexed_disconnect_closes_windows_at_its_control_record(tmp_path: Path) -> None:
    catalog = _catalog()
    interrupted = ConnectionInterruptedEvent(
        event_index=0,
        local_received_ts=BASE + timedelta(milliseconds=10),
        connection_id="connection-1",
        tickers=("A", "M"),
        reason="synthetic disconnect",
    )
    session = _FakeSession(
        (
            (
                _subscription(),
                _snapshot("A", event_index=0, sequence=1, yes_bid="0.60", no_bid="0.38"),
                _snapshot("M", event_index=0, sequence=2, yes_bid="0.70", no_bid="0.28"),
                interrupted,
            ),
        )
    )
    raw_root = tmp_path / "disconnect-raw"

    with (
        DuckDBRepository(tmp_path / "disconnect.duckdb") as repository,
        ParquetEventWriter(raw_root) as writer,
    ):
        _seed(repository, catalog, suffix="disconnect")
        runner, state, _ = _build_runner(
            session=session,
            repository=repository,
            writer=writer,
            run_id="live-disconnect",
            clock=_RealtimeEventClock(),
        )
        result = await runner.run(duration_seconds=1)
        windows = repository.connection.execute(
            """
            SELECT market_ticker, status, closed_event_index
            FROM market_observation_windows
            ORDER BY market_ticker
            """
        ).fetchall()

        assert result.events_written == 6
        assert windows == [("A", "disconnected", 4), ("M", "disconnected", 4)]
        assert all(book.status is BookStatus.RESYNC_REQUIRED for book in state.orderbooks.values())

    records = read_recorded_events(raw_root)
    assert [record.event_type for record in records] == [
        "run_started",
        "subscription_started",
        "snapshot",
        "snapshot",
        "connection_interrupted",
        "run_ended",
    ]


@pytest.mark.asyncio
async def test_successful_run_end_right_censors_an_active_opportunity(tmp_path: Path) -> None:
    catalog = _catalog()
    session = _FakeSession(
        (
            (
                _subscription(),
                _snapshot("A", event_index=0, sequence=1, yes_bid="0.60", no_bid="0.38"),
                _snapshot("M", event_index=0, sequence=2, yes_bid="0.70", no_bid="0.28"),
            ),
        )
    )
    raw_root = tmp_path / "censor-raw"

    with (
        DuckDBRepository(tmp_path / "censor.duckdb") as repository,
        ParquetEventWriter(raw_root) as writer,
    ):
        _seed(repository, catalog, suffix="censor")
        runner, _, lifecycle = _build_runner(
            session=session,
            repository=repository,
            writer=writer,
            run_id="live-censor",
            clock=_RealtimeEventClock(),
        )
        result = await runner.run(duration_seconds=1)
        transitions = repository.connection.execute(
            """
            SELECT transition, event_index
            FROM opportunity_observations
            ORDER BY event_index, transition
            """
        ).fetchall()
        summary = repository.connection.execute(
            """
            SELECT ended_at, closed_at, censored_at, censored_event_index, censor_reason
            FROM opportunities
            """
        ).fetchone()

        assert result.events_written == 5
        assert result.opportunity_observations == 2
        assert transitions == [("open", 3), ("right_censored", 4)]
        assert summary is not None
        assert summary[0] is None and summary[1] is None
        assert summary[2] is not None
        assert summary[3:] == (4, "run_succeeded")
        assert not lifecycle.active_episodes
        assert len(lifecycle.censored_episodes) == 1

    records = read_recorded_events(raw_root)
    assert isinstance(records[-1], RunEndedEvent)
    assert records[-1].event_index == 4
    assert records[-1].status == "succeeded"


@pytest.mark.asyncio
async def test_live_scanner_rejects_subscription_members_outside_trusted_components(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    session = _FakeSession(((_subscription(tickers=("A", "M", "Z")),),))

    with (
        DuckDBRepository(tmp_path / "membership.duckdb") as repository,
        ParquetEventWriter(tmp_path / "membership-raw") as writer,
    ):
        _seed(repository, catalog, suffix="membership")
        runner, _, _ = _build_runner(
            session=session,
            repository=repository,
            writer=writer,
            run_id="live-membership",
            clock=_RealtimeEventClock(),
        )

        with pytest.raises(WebSocketProtocolError, match="trusted|membership"):
            await runner.run(duration_seconds=1)

        manifest = repository.get_run_manifest("live-membership")
        assert manifest is not None and manifest.status == "failed"


@pytest.mark.asyncio
async def test_sequence_gap_requests_resync_and_closes_market_windows_with_reason(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    session = _FakeSession(
        (
            (
                _subscription(),
                _snapshot("A", event_index=0, sequence=1, yes_bid="0.60", no_bid="0.38"),
                _snapshot("M", event_index=1, sequence=2, yes_bid="0.70", no_bid="0.28"),
                _delta(
                    "A",
                    event_index=2,
                    sequence=4,
                    side="no",
                    price="0.40",
                    quantity_delta="1.00",
                    received_at=BASE + timedelta(milliseconds=10),
                ),
            ),
        )
    )
    raw_root = tmp_path / "gap-raw"

    with (
        DuckDBRepository(tmp_path / "gap.duckdb") as repository,
        ParquetEventWriter(raw_root) as writer,
    ):
        _seed(repository, catalog, suffix="gap")
        runner, _, _ = _build_runner(
            session=session,
            repository=repository,
            writer=writer,
            run_id="live-gap",
            clock=_RealtimeEventClock(),
        )
        result = await runner.run(duration_seconds=1)
        windows = repository.connection.execute(
            """
            SELECT market_ticker, status, stale_reason, resync_reason, closed_event_index
            FROM market_observation_windows
            ORDER BY market_ticker
            """
        ).fetchall()

        assert result.resync_requests == 1
        assert session.requested_snapshots == [("A", "M")]
        assert windows == [
            ("A", "resync_required", "sequence gap", "sequence gap", 4),
            ("M", "resync_required", "sequence gap", "sequence gap", 4),
        ]

    assert [event.event_index for event in read_market_data_events(raw_root)] == [2, 3, 4]


@pytest.mark.asyncio
async def test_closed_market_refresh_stays_ineligible_and_stops_without_resnapshot(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    signal = MarketRefreshStartedEvent(
        event_index=0,
        local_received_ts=BASE + timedelta(milliseconds=10),
        ticker="A",
        reason="market_close",
        connection_id="connection-1",
        sid=9,
    )
    session = _FakeSession(
        (
            (
                _subscription(),
                _snapshot("A", event_index=0, sequence=1, yes_bid="0.60", no_bid="0.38"),
                _snapshot("M", event_index=1, sequence=2, yes_bid="0.70", no_bid="0.28"),
                signal,
            ),
        )
    )
    closed = catalog.markets[0].model_copy(
        update={
            "status": "closed",
            "raw": {**catalog.markets[0].raw, "status": "closed"},
        }
    )

    async def refresh_market(ticker: str) -> Market:
        assert ticker == "A"
        return closed

    with (
        DuckDBRepository(tmp_path / "closed.duckdb") as repository,
        ParquetEventWriter(tmp_path / "closed-raw") as writer,
    ):
        _seed(repository, catalog, suffix="closed")
        runner, state, _ = _build_runner(
            session=session,
            repository=repository,
            writer=writer,
            run_id="live-closed",
            clock=_RealtimeEventClock(),
            metadata_refresher=refresh_market,
        )

        with pytest.raises(LiveScannerError, match="no longer open"):
            await runner.run(duration_seconds=1)

        manifest = repository.get_run_manifest("live-closed")
        assert manifest is not None and manifest.status == "failed"
        assert state.markets["A"].status == "closed"
        assert state.orderbooks["A"].status is BookStatus.RESYNC_REQUIRED
        assert session.updated_markets == [closed]
        assert session.requested_snapshots == []
        assert {market.ticker: market.status for market in repository.load_markets()}[
            "A"
        ] == "closed"

    records = read_recorded_events(tmp_path / "closed-raw")
    assert [record.event_type for record in records] == [
        "run_started",
        "subscription_started",
        "snapshot",
        "snapshot",
        "market_refresh_started",
        "market_refresh_applied",
        "run_ended",
    ]
    assert isinstance(records[4], MarketRefreshStartedEvent)
    assert isinstance(records[5], MarketRefreshAppliedEvent)
    assert records[5].refresh_started_event_index == records[4].event_index
    assert records[5].market.status == "closed"


@pytest.mark.asyncio
async def test_fee_refresh_preserves_membership_and_persists_authoritative_rest_state(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    change = EventFeeChange(
        change_id="scheduled-change",
        event_ticker="EVENT",
        series_ticker="SERIES",
        scheduled_ts=BASE + timedelta(days=1),
        fee_type_override="quadratic",
        fee_multiplier_override=Decimal("1.50"),
        raw={"id": "scheduled-change"},
    )
    signal = FeeRefreshStartedEvent(
        event_index=0,
        event_ticker="EVENT",
        affected_tickers=("A", "M"),
        local_received_ts=BASE + timedelta(milliseconds=10),
        connection_id="connection-1",
        sid=8,
    )
    session = _FakeSession(((_subscription(), signal),))

    async def refresh_fee(item: FeeRefreshStartedEvent) -> Event:
        assert item.event_ticker == signal.event_ticker
        assert item.affected_tickers == signal.affected_tickers
        return Event(
            ticker="EVENT",
            series_ticker="SERIES",
            title="REST refresh without nested markets",
            market_tickers=(),
            fee_type_override="quadratic",
            fee_multiplier_override=Decimal("1.25"),
            fee_changes=(change,),
            raw={"event_ticker": "EVENT", "series_ticker": "SERIES"},
        )

    with (
        DuckDBRepository(tmp_path / "fee.duckdb") as repository,
        ParquetEventWriter(tmp_path / "fee-raw") as writer,
    ):
        _seed(repository, catalog, suffix="fee")
        runner, state, _ = _build_runner(
            session=session,
            repository=repository,
            writer=writer,
            run_id="live-fee",
            clock=_RealtimeEventClock(),
            event_fee_refresher=refresh_fee,
        )
        result = await runner.run(duration_seconds=1)
        persisted = repository.get_event_fee_state("EVENT")

        assert result.event_fee_refreshes == 1
        assert result.events_written == 5
        assert state.events["EVENT"].market_tickers == ("A", "M")
        assert state.events["EVENT"].fee_type_override == "quadratic"
        assert state.events["EVENT"].fee_multiplier_override == Decimal("1.25")
        assert state.events["EVENT"].fee_changes == (change,)
        assert persisted is not None
        assert persisted.fee_type_override == "quadratic"
        assert persisted.fee_multiplier_override == Decimal("1.25")
        assert persisted.fee_changes == (change,)

    records = read_recorded_events(tmp_path / "fee-raw")
    assert [record.event_type for record in records] == [
        "run_started",
        "subscription_started",
        "fee_refresh_started",
        "fee_refresh_applied",
        "run_ended",
    ]
    assert isinstance(records[2], FeeRefreshStartedEvent)
    assert isinstance(records[3], FeeRefreshAppliedEvent)
    assert records[3].refresh_started_event_index == records[2].event_index
    assert records[3].event == state.events["EVENT"]


@pytest.mark.asyncio
async def test_upstream_failure_finalizes_failed_and_flushes_accepted_raw_event(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    accepted = _snapshot(
        "A",
        event_index=0,
        sequence=1,
        yes_bid="0.60",
        no_bid="0.38",
    )
    session = _FakeSession(
        ((_subscription(), accepted),),
        terminal_error=RuntimeError("synthetic upstream failure"),
    )
    raw_root = tmp_path / "upstream-raw"

    with (
        DuckDBRepository(tmp_path / "upstream.duckdb") as repository,
        ParquetEventWriter(raw_root) as writer,
    ):
        _seed(repository, catalog, suffix="upstream")
        runner, _, _ = _build_runner(
            session=session,
            repository=repository,
            writer=writer,
            run_id="live-upstream-failure",
            clock=_RealtimeEventClock(),
        )

        with pytest.raises(RuntimeError, match="synthetic upstream failure"):
            await runner.run(duration_seconds=1)

        manifest = repository.get_run_manifest("live-upstream-failure")
        assert manifest is not None and manifest.status == "failed"
        assert manifest.error == "RuntimeError: live scan terminated"

    records = read_recorded_events(raw_root)
    assert [record.event_type for record in records] == [
        "run_started",
        "subscription_started",
        "snapshot",
        "run_ended",
    ]
    assert [event.event_index for event in read_market_data_events(raw_root)] == [2]


@pytest.mark.asyncio
async def test_storage_failure_finalizes_failed_and_flushes_all_accepted_raw_events(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    first = _snapshot("A", event_index=0, sequence=1, yes_bid="0.60", no_bid="0.38")
    second = _snapshot("M", event_index=1, sequence=2, yes_bid="0.70", no_bid="0.28")
    session = _FakeSession(((_subscription(), first, second),))
    raw_root = tmp_path / "storage-raw"

    with (
        _FailingTransitionRepository(tmp_path / "storage.duckdb") as repository,
        ParquetEventWriter(raw_root) as writer,
    ):
        _seed(repository, catalog, suffix="storage")
        runner, _, _ = _build_runner(
            session=session,
            repository=repository,
            writer=writer,
            run_id="live-storage-failure",
            clock=_RealtimeEventClock(),
        )

        with pytest.raises(StorageError, match="persistent opportunity write failure"):
            await runner.run(duration_seconds=1)

        manifest = repository.get_run_manifest("live-storage-failure")
        assert manifest is not None and manifest.status == "failed"
        assert manifest.error == "StorageError: live scan terminated"
        assert repository.table_count("opportunities") == 0

    records = read_recorded_events(raw_root)
    assert [record.event_type for record in records] == [
        "run_started",
        "subscription_started",
        "snapshot",
        "snapshot",
        "run_ended",
    ]
    assert [event.event_index for event in read_market_data_events(raw_root)] == [2, 3]
