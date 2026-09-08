"""Collector integration across session signals, shared state, and Parquet storage."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.engine.state import EngineState
from arbiter.kalshi.websocket import (
    ConnectionInterrupted,
    EventFeeRefresh,
    EventFeeRefresher,
    MarketDataCollector,
    MetadataRefresh,
    MetadataRefresher,
    SessionItem,
    SubscriptionStarted,
    WebSocketProtocolError,
)
from arbiter.models.event import Event
from arbiter.models.market import Market, PriceRange
from arbiter.models.orderbook import PriceLevel
from arbiter.replay.events import (
    ConnectionInterruptedEvent,
    EventIndexAllocator,
    FeeRefreshAppliedEvent,
    FeeRefreshStartedEvent,
    MarketRefreshAppliedEvent,
    MarketRefreshStartedEvent,
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
    SubscriptionStartedEvent,
)
from arbiter.storage.parquet import (
    ParquetBackpressureError,
    ParquetEventWriter,
    read_market_data_events,
    read_recorded_events,
)

NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)


def _market(ticker: str, *, step: str = "0.0100") -> Market:
    return Market(
        ticker=ticker,
        event_ticker="EVENT",
        title=ticker,
        status="active",
        price_ranges=(
            PriceRange(
                start=Decimal("0.0000"),
                end=Decimal("1.0000"),
                step=Decimal(step),
            ),
        ),
        raw={},
    )


def _snapshot(
    ticker: str,
    event_index: int,
    sequence: int,
    *,
    connection_id: str = "connection-1",
    sid: int = 7,
    snapshot_id: str | None = None,
) -> OrderBookSnapshotEvent:
    return OrderBookSnapshotEvent(
        event_index=event_index,
        local_received_ts=NOW + timedelta(milliseconds=event_index),
        exchange_ts=None,
        ticker=ticker,
        sequence=sequence,
        sid=sid,
        connection_id=connection_id,
        snapshot_id=snapshot_id or f"{connection_id}:{sid}:{ticker}:{sequence}",
        yes_bids=(PriceLevel(price=Decimal("0.50"), quantity=Decimal("2.00")),),
        no_bids=(PriceLevel(price=Decimal("0.40"), quantity=Decimal("3.00")),),
    )


def _delta(
    ticker: str,
    event_index: int,
    sequence: int,
    *,
    snapshot_id: str,
    connection_id: str = "connection-1",
    sid: int = 7,
) -> OrderBookDeltaEvent:
    return OrderBookDeltaEvent(
        event_index=event_index,
        local_received_ts=NOW + timedelta(milliseconds=event_index),
        exchange_ts=NOW + timedelta(milliseconds=event_index),
        ticker=ticker,
        sequence=sequence,
        sid=sid,
        connection_id=connection_id,
        snapshot_id=snapshot_id,
        side="yes",
        price=Decimal("0.50"),
        quantity_delta=Decimal("1.00"),
    )


class _FakeSession:
    """Finite deterministic session with observations taken after collector handling."""

    def __init__(
        self,
        items: Sequence[SessionItem | BaseException],
        *,
        state: EngineState,
        tickers: Sequence[str],
    ) -> None:
        self.items = tuple(items)
        self.state = state
        self.tickers = tuple(tickers)
        self.duration_seconds: float | None = None
        self.snapshot_requests: list[tuple[str, ...]] = []
        self.metadata_updates: list[Market] = []
        self.operation_log: list[str] = []
        self.readiness_after_items: list[dict[str, bool]] = []

    async def iter_items(self, *, duration_seconds: float) -> AsyncIterator[SessionItem]:
        self.duration_seconds = duration_seconds
        for item in self.items:
            if isinstance(item, BaseException):
                raise item
            yield item
            self.readiness_after_items.append(
                {ticker: self.state.is_solve_ready(ticker) for ticker in self.tickers}
            )

    async def request_snapshots(self, tickers: Sequence[str]) -> None:
        requested = tuple(tickers)
        self.snapshot_requests.append(requested)
        self.operation_log.append(f"request:{','.join(requested)}")

    def update_market_metadata(self, market: Market) -> None:
        self.metadata_updates.append(market)
        self.operation_log.append(f"update:{market.ticker}")


class _RecordedFakeSession:
    """Yield controls lazily so refresh-applied records share the total order."""

    def __init__(self, *, state: EngineState) -> None:
        self.state = state
        self.event_index_allocator = EventIndexAllocator()
        self.snapshot_requests: list[tuple[str, ...]] = []
        self.metadata_updates: list[Market] = []

    async def iter_items(self, *, duration_seconds: float) -> AsyncIterator[SessionItem]:
        del duration_seconds
        yield SubscriptionStartedEvent(
            event_index=self.event_index_allocator.allocate(),
            local_received_ts=NOW,
            connection_id="connection-1",
            sid=7,
            tickers=("A",),
        )
        initial_index = self.event_index_allocator.allocate()
        initial = _snapshot("A", initial_index, 1)
        yield initial
        yield MarketRefreshStartedEvent(
            event_index=self.event_index_allocator.allocate(),
            local_received_ts=NOW + timedelta(milliseconds=2),
            ticker="A",
            reason="metadata_updated",
            connection_id="connection-1",
            sid=8,
        )
        replacement_index = self.event_index_allocator.allocate()
        yield _snapshot(
            "A",
            replacement_index,
            2,
            snapshot_id="replacement-a",
        )
        yield FeeRefreshStartedEvent(
            event_index=self.event_index_allocator.allocate(),
            local_received_ts=NOW + timedelta(milliseconds=5),
            event_ticker="EVENT",
            affected_tickers=("A",),
            connection_id="connection-1",
            sid=8,
        )
        yield ConnectionInterruptedEvent(
            event_index=self.event_index_allocator.allocate(),
            local_received_ts=NOW + timedelta(milliseconds=7),
            connection_id="connection-1",
            tickers=("A",),
            reason="synthetic disconnect",
        )

    async def request_snapshots(self, tickers: Sequence[str]) -> None:
        self.snapshot_requests.append(tuple(tickers))

    def update_market_metadata(self, market: Market) -> None:
        self.metadata_updates.append(market)


def _collector(
    session: _FakeSession,
    state: EngineState,
    writer: ParquetEventWriter,
    *,
    metadata_refresher: MetadataRefresher | None = None,
    event_fee_refresher: EventFeeRefresher | None = None,
    writer_batch_size: int = 100,
) -> MarketDataCollector:
    return MarketDataCollector(
        session=session,
        state=state,
        writer=writer,
        writer_batch_size=writer_batch_size,
        writer_flush_interval_seconds=60,
        metadata_refresher=metadata_refresher,
        event_fee_refresher=event_fee_refresher,
        monotonic=lambda: 0.0,
    )


@pytest.mark.asyncio
async def test_graceful_collection_writes_replayable_stream_and_invalidates_books(
    tmp_path: Path,
) -> None:
    state = EngineState()
    snapshot = _snapshot("A", 0, 1)
    delta = _delta("A", 1, 2, snapshot_id=snapshot.snapshot_id)
    session = _FakeSession(
        (
            SubscriptionStarted("connection-1", 7, ("A",)),
            snapshot,
            delta,
        ),
        state=state,
        tickers=("A",),
    )
    writer = ParquetEventWriter(tmp_path / "orderbooks")

    result = await _collector(session, state, writer, writer_batch_size=2).run(duration_seconds=30)

    assert result.events_written == 2
    assert len(result.files_written) == 1
    assert result.resync_requests == 0
    assert session.duration_seconds == 30
    assert session.readiness_after_items[-1] == {"A": True}
    assert not state.is_solve_ready("A")
    book = state.get_book("A")
    assert book is not None
    assert book.yes_bids[0].quantity == Decimal("3.00")
    assert read_market_data_events(tmp_path / "orderbooks") == (snapshot, delta)


@pytest.mark.asyncio
async def test_sequence_gap_requests_all_subscription_snapshots_and_recovers_atomically(
    tmp_path: Path,
) -> None:
    state = EngineState()
    initial_a = _snapshot("A", 0, 1)
    initial_b = _snapshot("B", 1, 2)
    gap = _delta("A", 2, 4, snapshot_id=initial_a.snapshot_id)
    replacement_a = _snapshot("A", 3, 10, snapshot_id="replacement-a")
    replacement_b = _snapshot("B", 4, 11, snapshot_id="replacement-b")
    events = (initial_a, initial_b, gap, replacement_a, replacement_b)
    session = _FakeSession(
        (SubscriptionStarted("connection-1", 7, ("A", "B")), *events),
        state=state,
        tickers=("A", "B"),
    )
    writer = ParquetEventWriter(tmp_path / "orderbooks")

    result = await _collector(session, state, writer).run(duration_seconds=30)

    assert result.resync_requests == 1
    assert session.snapshot_requests == [("A", "B")]
    assert session.readiness_after_items[-3:] == [
        {"A": False, "B": False},
        {"A": False, "B": False},
        {"A": True, "B": True},
    ]
    assert all(not state.is_solve_ready(ticker) for ticker in ("A", "B"))
    assert read_market_data_events(tmp_path / "orderbooks") == events


@pytest.mark.asyncio
async def test_lifecycle_refresh_completes_before_replacement_snapshot_is_consumed(
    tmp_path: Path,
) -> None:
    state = EngineState()
    initial = _snapshot("A", 0, 1)
    replacement = _snapshot("A", 1, 2, snapshot_id="replacement-a")
    lifecycle = MetadataRefresh(
        ticker="A",
        event_type="price_level_structure_updated",
        connection_id="connection-1",
        sid=8,
    )
    session = _FakeSession(
        (SubscriptionStarted("connection-1", 7, ("A",)), initial, lifecycle, replacement),
        state=state,
        tickers=("A",),
    )
    refreshed_market = _market("A", step="0.0050")

    async def refresh(ticker: str) -> Market:
        session.operation_log.append(f"refresh:{ticker}")
        return refreshed_market

    writer = ParquetEventWriter(tmp_path / "orderbooks")

    result = await _collector(session, state, writer, metadata_refresher=refresh).run(
        duration_seconds=30
    )

    assert result.metadata_refreshes == 1
    assert result.resync_requests == 1
    assert session.operation_log == ["refresh:A", "update:A", "request:A"]
    assert session.metadata_updates == [refreshed_market]
    assert session.readiness_after_items[-1] == {"A": True}
    assert not state.is_solve_ready("A")
    assert read_market_data_events(tmp_path / "orderbooks") == (initial, replacement)


@pytest.mark.asyncio
async def test_event_fee_refresh_is_explicit_and_does_not_request_book_snapshot(
    tmp_path: Path,
) -> None:
    state = EngineState()
    snapshot = _snapshot("A", 0, 1)
    fee_update = EventFeeRefresh(
        event_ticker="EVENT",
        affected_tickers=("A",),
        fee_type_override="quadratic",
        fee_multiplier_override=Decimal("2"),
        local_received_ts=NOW,
        connection_id="connection-1",
        sid=8,
    )
    session = _FakeSession(
        (SubscriptionStarted("connection-1", 7, ("A",)), snapshot, fee_update),
        state=state,
        tickers=("A",),
    )

    async def refresh(item: EventFeeRefresh) -> None:
        session.operation_log.append(f"fee:{item.event_ticker}")

    writer = ParquetEventWriter(tmp_path / "orderbooks")
    result = await _collector(
        session,
        state,
        writer,
        event_fee_refresher=refresh,
    ).run(duration_seconds=30)

    assert result.event_fee_refreshes == 1
    assert result.resync_requests == 0
    assert session.operation_log == ["fee:EVENT"]
    assert session.snapshot_requests == []
    assert session.readiness_after_items[-1] == {"A": True}
    assert read_market_data_events(tmp_path / "orderbooks") == (snapshot,)


@pytest.mark.asyncio
async def test_recorded_controls_and_authoritative_refreshes_share_one_total_order(
    tmp_path: Path,
) -> None:
    state = EngineState()
    session = _RecordedFakeSession(state=state)
    refreshed_market = _market("A", step="0.0050")
    refreshed_event = Event(
        ticker="EVENT",
        title="Event",
        market_tickers=("A", "B"),
        fee_type_override="quadratic",
        fee_multiplier_override=Decimal("1.5"),
        raw={},
    )

    async def refresh_market(ticker: str) -> Market:
        assert ticker == "A"
        return refreshed_market

    async def refresh_event_fee(item: FeeRefreshStartedEvent) -> Event:
        assert item.event_ticker == "EVENT"
        return refreshed_event

    writer = ParquetEventWriter(tmp_path / "orderbooks")
    result = await MarketDataCollector(
        session=session,
        state=state,
        writer=writer,
        writer_batch_size=100,
        writer_flush_interval_seconds=60,
        metadata_refresher=refresh_market,
        event_fee_refresher=refresh_event_fee,
        event_index_allocator=session.event_index_allocator,
        monotonic=lambda: 0.0,
        local_clock=lambda: NOW + timedelta(milliseconds=3),
    ).run(duration_seconds=30)

    records = read_recorded_events(tmp_path / "orderbooks")
    assert [record.event_index for record in records] == list(range(8))
    assert [record.event_type for record in records] == [
        "subscription_started",
        "snapshot",
        "market_refresh_started",
        "market_refresh_applied",
        "snapshot",
        "fee_refresh_started",
        "fee_refresh_applied",
        "connection_interrupted",
    ]
    assert isinstance(records[3], MarketRefreshAppliedEvent)
    assert records[3].market == refreshed_market
    assert isinstance(records[6], FeeRefreshAppliedEvent)
    assert records[6].event == refreshed_event
    assert records[6].event.market_tickers == ("A", "B")
    assert result.events_written == 2
    assert result.metadata_refreshes == 1
    assert result.event_fee_refreshes == 1
    assert result.resync_requests == 1
    assert session.metadata_updates == [refreshed_market]
    assert session.snapshot_requests == [("A",)]
    assert len(result.files_written) == 1
    assert len(read_market_data_events(tmp_path / "orderbooks")) == 2
    assert not state.is_solve_ready("A")


@pytest.mark.asyncio
async def test_recorded_control_without_shared_allocator_fails_closed(tmp_path: Path) -> None:
    state = EngineState()
    session = _FakeSession(
        (
            SubscriptionStartedEvent(
                event_index=0,
                local_received_ts=NOW,
                connection_id="connection-1",
                sid=7,
                tickers=("A",),
            ),
        ),
        state=state,
        tickers=("A",),
    )
    writer = ParquetEventWriter(tmp_path / "orderbooks")

    with pytest.raises(WebSocketProtocolError, match="share the session event-index allocator"):
        await _collector(session, state, writer).run(duration_seconds=30)

    assert writer.pending_count == 0
    writer.close()


@pytest.mark.asyncio
async def test_recorded_refresh_without_authoritative_refresher_fails_closed(
    tmp_path: Path,
) -> None:
    state = EngineState()
    session = _RecordedFakeSession(state=state)
    writer = ParquetEventWriter(tmp_path / "orderbooks")
    collector = MarketDataCollector(
        session=session,
        state=state,
        writer=writer,
        writer_batch_size=100,
        writer_flush_interval_seconds=60,
        metadata_refresher=None,
        event_fee_refresher=None,
        event_index_allocator=session.event_index_allocator,
        monotonic=lambda: 0.0,
    )

    with pytest.raises(WebSocketProtocolError, match="no bounded metadata refresher"):
        await collector.run(duration_seconds=30)

    records = read_recorded_events(tmp_path / "orderbooks")
    assert [record.event_type for record in records] == [
        "subscription_started",
        "snapshot",
        "market_refresh_started",
    ]
    assert not state.is_solve_ready("A")
    writer.close()


@pytest.mark.asyncio
async def test_connection_interruption_immediately_invalidates_active_books(tmp_path: Path) -> None:
    state = EngineState()
    snapshot = _snapshot("A", 0, 1)
    session = _FakeSession(
        (
            SubscriptionStarted("connection-1", 7, ("A",)),
            snapshot,
            ConnectionInterrupted("connection-1", "network disconnected"),
        ),
        state=state,
        tickers=("A",),
    )
    writer = ParquetEventWriter(tmp_path / "orderbooks")

    result = await _collector(session, state, writer).run(duration_seconds=30)

    assert result.events_written == 1
    assert session.readiness_after_items[-2:] == [{"A": True}, {"A": False}]
    assert not state.is_solve_ready("A")
    assert read_market_data_events(tmp_path / "orderbooks") == (snapshot,)


@pytest.mark.asyncio
async def test_failure_flushes_accepted_events_and_invalidates_books(tmp_path: Path) -> None:
    state = EngineState()
    snapshot = _snapshot("A", 0, 1)
    delta = _delta("A", 1, 2, snapshot_id=snapshot.snapshot_id)
    session = _FakeSession(
        (
            SubscriptionStarted("connection-1", 7, ("A",)),
            snapshot,
            delta,
            WebSocketProtocolError("synthetic session failure"),
        ),
        state=state,
        tickers=("A",),
    )
    writer = ParquetEventWriter(tmp_path / "orderbooks")

    with pytest.raises(WebSocketProtocolError, match="synthetic session failure"):
        await _collector(session, state, writer).run(duration_seconds=30)

    assert writer.pending_count == 0
    assert not state.is_solve_ready("A")
    assert read_market_data_events(tmp_path / "orderbooks") == (snapshot, delta)
    writer.close()


@pytest.mark.asyncio
async def test_writer_overflow_is_visible_and_flushes_only_previously_accepted_events(
    tmp_path: Path,
) -> None:
    state = EngineState()
    snapshot = _snapshot("A", 0, 1)
    delta = _delta("A", 1, 2, snapshot_id=snapshot.snapshot_id)
    session = _FakeSession(
        (SubscriptionStarted("connection-1", 7, ("A",)), snapshot, delta),
        state=state,
        tickers=("A",),
    )
    writer = ParquetEventWriter(tmp_path / "orderbooks", max_queue_size=1)

    with pytest.raises(ParquetBackpressureError, match="1-event limit"):
        await _collector(session, state, writer).run(duration_seconds=30)

    assert writer.pending_count == 0
    assert not state.is_solve_ready("A")
    assert read_market_data_events(tmp_path / "orderbooks") == (snapshot,)
    writer.close()
