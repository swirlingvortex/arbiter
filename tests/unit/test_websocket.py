"""Kalshi WebSocket commands, normalization, state transitions, and reconnect tests."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

import pytest

from arbiter.kalshi.auth import KalshiWebSocketAuthenticator
from arbiter.kalshi.normalize import (
    NormalizationError,
    normalize_websocket_delta,
    normalize_websocket_snapshot,
)
from arbiter.kalshi.websocket import (
    ConnectionOpener,
    ConnectionOptions,
    KalshiWebSocketSession,
    RecordedSessionItem,
    SessionPhase,
    WebSocketConnection,
    WebSocketProtocolError,
    WebSocketReconnectError,
    WebSocketStateMachine,
    lifecycle_subscription_command,
    orderbook_subscription_command,
    snapshot_request_command,
)
from arbiter.models.market import Market, PriceRange
from arbiter.replay.events import (
    ConnectionInterruptedEvent,
    EventIndexAllocator,
    FeeRefreshStartedEvent,
    MarketRefreshStartedEvent,
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
    SubscriptionStartedEvent,
)

NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)


def _market(ticker: str = "A") -> Market:
    return Market(
        ticker=ticker,
        event_ticker="EVENT",
        title=ticker,
        status="active",
        price_ranges=(
            PriceRange(
                start=Decimal("0.0000"),
                end=Decimal("1.0000"),
                step=Decimal("0.0100"),
            ),
        ),
        raw={},
    )


def _snapshot(ticker: str, *, sequence: int, sid: int = 7) -> dict[str, object]:
    return {
        "type": "orderbook_snapshot",
        "sid": sid,
        "seq": sequence,
        "msg": {
            "market_ticker": ticker,
            "market_id": "00000000-0000-0000-0000-000000000001",
            "yes_dollars_fp": [["0.6000", "2.00"]],
            "no_dollars_fp": [["0.3800", "3.00"]],
        },
    }


def _delta(ticker: str, *, sequence: int, sid: int = 7) -> dict[str, object]:
    return {
        "type": "orderbook_delta",
        "sid": sid,
        "seq": sequence,
        "msg": {
            "market_ticker": ticker,
            "market_id": "00000000-0000-0000-0000-000000000001",
            "price_dollars": "0.3900",
            "delta_fp": "0.25",
            "side": "no",
            "ts_ms": 1_788_444_000_123,
        },
    }


class _FakeAuthenticator:
    def handshake_headers(self) -> dict[str, str]:
        return {
            "KALSHI-ACCESS-KEY": "test-id",
            "KALSHI-ACCESS-TIMESTAMP": "1",
            "KALSHI-ACCESS-SIGNATURE": "test-signature",
        }


class _FakeConnection:
    def __init__(self, messages: list[dict[str, object] | BaseException]) -> None:
        self.messages = deque(messages)
        self.sent: list[str] = []

    async def send(self, message: str | bytes) -> None:
        assert isinstance(message, str)
        self.sent.append(message)

    async def recv(self, decode: bool | None = None) -> str | bytes:
        if self.messages:
            item = self.messages.popleft()
            if isinstance(item, BaseException):
                raise item
            return json.dumps(item)
        await asyncio.Future()
        raise AssertionError("unreachable")


def _opener_for(*connections: _FakeConnection) -> ConnectionOpener:
    pending = deque(connections)

    @asynccontextmanager
    async def opener(
        url: str,
        headers: Mapping[str, str],
        options: ConnectionOptions,
    ) -> AsyncIterator[WebSocketConnection]:
        assert url == "wss://example.invalid/trade-api/ws/v2"
        assert headers["KALSHI-ACCESS-KEY"] == "test-id"
        assert options.inbound_queue_capacity == 8
        yield pending.popleft()

    return opener


def _session(
    markets: Mapping[str, Market],
    *,
    opener: ConnectionOpener,
    reconnect_max_attempts: int = 0,
    start_event_index: int | None = None,
    event_index_allocator: EventIndexAllocator | None = None,
) -> KalshiWebSocketSession:
    return KalshiWebSocketSession(
        url="wss://example.invalid/trade-api/ws/v2",
        authenticator=cast(KalshiWebSocketAuthenticator, _FakeAuthenticator()),
        markets=markets,
        inbound_queue_capacity=8,
        reconnect_max_attempts=reconnect_max_attempts,
        reconnect_initial_backoff_seconds=0.001,
        reconnect_max_backoff_seconds=0.002,
        open_timeout_seconds=1,
        close_timeout_seconds=1,
        opener=opener,
        local_clock=lambda: NOW,
        connection_id_factory=lambda: "connection-1",
        start_event_index=start_event_index,
        event_index_allocator=event_index_allocator,
    )


def test_commands_are_explicit_deterministic_and_use_yes_price() -> None:
    assert orderbook_subscription_command(1, ("B", "A")) == {
        "id": 1,
        "cmd": "subscribe",
        "params": {
            "channels": ["orderbook_delta"],
            "market_tickers": ["A", "B"],
            "use_yes_price": True,
        },
    }
    assert lifecycle_subscription_command(2, "market_lifecycle_v2") == {
        "id": 2,
        "cmd": "subscribe",
        "params": {"channels": ["market_lifecycle_v2"]},
    }
    assert snapshot_request_command(3, sid=7, tickers=("B", "A")) == {
        "id": 3,
        "cmd": "update_subscription",
        "params": {
            "sids": [7],
            "market_tickers": ["A", "B"],
            "action": "get_snapshot",
        },
    }


@pytest.mark.parametrize("command_id", [0, -1, True])
def test_command_ids_must_be_positive_integers(command_id: int) -> None:
    with pytest.raises(ValueError, match="command ID"):
        orderbook_subscription_command(command_id, ("A",))


def test_normalization_complements_no_price_exactly_once_and_preserves_signed_delta() -> None:
    market = _market()
    snapshot = normalize_websocket_snapshot(
        _snapshot("A", sequence=10),
        event_index=0,
        connection_id="connection-1",
        local_received_ts=NOW,
        market=market,
    )
    delta = normalize_websocket_delta(
        _delta("A", sequence=11),
        event_index=1,
        connection_id="connection-1",
        local_received_ts=NOW,
        market=market,
        snapshot_id=snapshot.snapshot_id,
    )

    assert snapshot.yes_bids[0].price == Decimal("0.6000")
    assert snapshot.no_bids[0].price == Decimal("0.6200")
    assert delta.side == "no"
    assert delta.price == Decimal("0.6100")
    assert delta.quantity_delta == Decimal("0.25")
    assert delta.exchange_ts is not None
    assert delta.exchange_ts.microsecond == 123_000
    assert snapshot.price_convention == delta.price_convention == "yes_price"


def test_normalization_rejects_bad_ticks_zero_delta_and_conflicting_timestamps() -> None:
    market = _market()
    bad_tick = _delta("A", sequence=2)
    cast(dict[str, object], bad_tick["msg"])["price_dollars"] = "0.3950"
    with pytest.raises(NormalizationError, match="not valid"):
        normalize_websocket_delta(
            bad_tick,
            event_index=1,
            connection_id="connection-1",
            local_received_ts=NOW,
            market=market,
            snapshot_id="snapshot",
        )

    zero = _delta("A", sequence=2)
    cast(dict[str, object], zero["msg"])["delta_fp"] = "0.00"
    with pytest.raises(NormalizationError, match="nonzero"):
        normalize_websocket_delta(
            zero,
            event_index=1,
            connection_id="connection-1",
            local_received_ts=NOW,
            market=market,
            snapshot_id="snapshot",
        )

    conflict = _delta("A", sequence=2)
    cast(dict[str, object], conflict["msg"])["ts"] = "2026-09-03T12:00:00Z"
    with pytest.raises(NormalizationError, match="different timestamps"):
        normalize_websocket_delta(
            conflict,
            event_index=1,
            connection_id="connection-1",
            local_received_ts=NOW,
            market=market,
            snapshot_id="snapshot",
        )


def test_connection_state_requires_all_initial_and_replacement_snapshots() -> None:
    state = WebSocketStateMachine(("A", "B"))
    state.connecting("connection-1")
    state.authenticated()
    state.subscribed(channel="orderbook_delta", sid=7)
    state.snapshot_received(ticker="A", sid=7)

    assert state.phase is SessionPhase.SNAPSHOT_RECEIVED
    assert state.pending_snapshots == {"B"}
    state.snapshot_received(ticker="B", sid=7)
    assert state.phase is SessionPhase.LIVE
    state.require_resync()
    assert state.phase is SessionPhase.RESYNC_REQUIRED
    assert state.pending_snapshots == {"A", "B"}


@pytest.mark.asyncio
async def test_session_normalizes_stream_and_requests_metadata_resync() -> None:
    connection = _FakeConnection(
        [
            {
                "type": "subscribed",
                "id": 1,
                "msg": {"channel": "orderbook_delta", "sid": 7},
            },
            _snapshot("A", sequence=1),
            _snapshot("B", sequence=2),
            {
                "type": "subscribed",
                "id": 2,
                "msg": {"channel": "market_lifecycle_v2", "sid": 8},
            },
            _delta("A", sequence=3),
            {
                "type": "market_lifecycle_v2",
                "sid": 8,
                "msg": {
                    "market_ticker": "A",
                    "event_type": "price_level_structure_updated",
                    "price_level_structure": "deci_cent",
                },
            },
            {"type": "ok", "id": 4, "sid": 7, "msg": {"market_tickers": ["A", "B"]}},
        ]
    )
    session = _session(
        {"A": _market("A"), "B": _market("B")},
        opener=_opener_for(connection),
    )
    items: list[RecordedSessionItem] = []

    async for item in session.iter_items(duration_seconds=0.03):
        items.append(item)
        if isinstance(item, MarketRefreshStartedEvent):
            assert session.state.phase is SessionPhase.RESYNC_REQUIRED
            await session.request_snapshots((item.ticker,))

    events = [
        item for item in items if isinstance(item, (OrderBookSnapshotEvent, OrderBookDeltaEvent))
    ]
    assert [item.event_index for item in items] == [0, 1, 2, 3, 4]
    assert all(item.local_received_ts.tzinfo is not None for item in items)
    assert [event.event_index for event in events] == [1, 2, 3]
    assert [event.ticker for event in events] == ["A", "B", "A"]
    assert isinstance(items[0], SubscriptionStartedEvent)
    assert isinstance(items[-1], MarketRefreshStartedEvent)
    assert items[-1].reason == "price_level_structure_updated"
    sent = [json.loads(message) for message in connection.sent]
    assert [command["id"] for command in sent] == [1, 2, 3, 4]
    assert sent[0]["params"]["use_yes_price"] is True
    assert sent[3]["params"] == {
        "action": "get_snapshot",
        "market_tickers": ["A"],
        "sids": [7],
    }
    assert session.state.phase is SessionPhase.CLOSED


@pytest.mark.asyncio
async def test_protocol_failure_reconnects_boundedly_with_monotonic_command_ids() -> None:
    first = _FakeConnection([OSError("connection lost")])
    second = _FakeConnection(
        [
            {
                "type": "subscribed",
                "id": 4,
                "msg": {"channel": "orderbook_delta", "sid": 9},
            },
            _snapshot("A", sequence=40, sid=9),
        ]
    )
    ids = iter(("connection-1", "connection-2"))
    session = KalshiWebSocketSession(
        url="wss://example.invalid/trade-api/ws/v2",
        authenticator=cast(KalshiWebSocketAuthenticator, _FakeAuthenticator()),
        markets={"A": _market()},
        inbound_queue_capacity=8,
        reconnect_max_attempts=1,
        reconnect_initial_backoff_seconds=0.001,
        reconnect_max_backoff_seconds=0.002,
        open_timeout_seconds=1,
        close_timeout_seconds=1,
        opener=_opener_for(first, second),
        local_clock=lambda: NOW,
        connection_id_factory=lambda: next(ids),
    )

    items = [item async for item in session.iter_items(duration_seconds=0.03)]
    events = [item for item in items if isinstance(item, OrderBookSnapshotEvent)]
    interruptions = [item for item in items if isinstance(item, ConnectionInterruptedEvent)]
    subscriptions = [item for item in items if isinstance(item, SubscriptionStartedEvent)]

    assert len(events) == 1
    assert [item.event_index for item in items] == [0, 1, 2]
    assert interruptions[0].tickers == ("A",)
    assert subscriptions[0].connection_id == "connection-2"
    assert events[0].connection_id == "connection-2"
    assert [json.loads(item)["id"] for item in first.sent] == [1, 2, 3]
    assert [json.loads(item)["id"] for item in second.sent] == [4, 5, 6]


@pytest.mark.asyncio
async def test_delta_before_snapshot_fails_closed_after_attempt_budget() -> None:
    connection = _FakeConnection(
        [
            {
                "type": "subscribed",
                "id": 1,
                "msg": {"channel": "orderbook_delta", "sid": 7},
            },
            _delta("A", sequence=2),
        ]
    )
    session = _session({"A": _market()}, opener=_opener_for(connection))

    with pytest.raises(WebSocketReconnectError, match="1 bounded attempt") as raised:
        _ = [item async for item in session.iter_items(duration_seconds=1)]

    assert isinstance(raised.value.__cause__, WebSocketProtocolError)


@pytest.mark.asyncio
async def test_documented_event_lifecycle_and_relevant_fee_update_are_handled() -> None:
    connection = _FakeConnection(
        [
            {
                "type": "subscribed",
                "id": 1,
                "msg": {"channel": "orderbook_delta", "sid": 7},
            },
            {
                "type": "subscribed",
                "id": 2,
                "msg": {"channel": "market_lifecycle_v2", "sid": 8},
            },
            {
                "type": "subscribed",
                "id": 3,
                "msg": {"channel": "multivariate_market_lifecycle", "sid": 9},
            },
            {
                "type": "event_lifecycle",
                "sid": 9,
                "msg": {
                    "event_ticker": "NEW-EVENT",
                    "exchange_index": 0,
                    "title": "New event",
                    "subtitle": "New subtitle",
                    "collateral_return_type": "MECNET",
                    "series_ticker": "NEW",
                },
            },
            {
                "type": "event_fee_update",
                "sid": 8,
                "msg": {
                    "event_ticker": "EVENT",
                    "fee_type_override": "quadratic",
                    "fee_multiplier_override": 2,
                },
            },
            _snapshot("A", sequence=1),
        ]
    )
    session = _session(
        {"A": _market("A"), "B": _market("B")},
        opener=_opener_for(connection),
        start_event_index=41,
    )

    items = [item async for item in session.iter_items(duration_seconds=0.03)]

    fee_updates = [item for item in items if isinstance(item, FeeRefreshStartedEvent)]
    snapshots = [item for item in items if isinstance(item, OrderBookSnapshotEvent)]
    assert len(fee_updates) == 1
    assert fee_updates[0].event_ticker == "EVENT"
    assert fee_updates[0].affected_tickers == ("A", "B")
    assert fee_updates[0].local_received_ts == NOW
    assert [item.event_index for item in items] == [41, 42, 43]
    assert [event.event_index for event in snapshots] == [43]
    assert len(items) == 3  # order-book subscription, fee refresh, and snapshot


@pytest.mark.asyncio
async def test_irrelevant_fee_update_is_ignored_after_sid_validation() -> None:
    connection = _FakeConnection(
        [
            {
                "type": "subscribed",
                "msg": {"channel": "orderbook_delta", "sid": 7},
            },
            {
                "type": "subscribed",
                "id": 2,
                "msg": {"channel": "market_lifecycle_v2", "sid": 8},
            },
            {
                "type": "event_fee_update",
                "sid": 8,
                "msg": {
                    "event_ticker": "UNRELATED",
                    "fee_type_override": None,
                    "fee_multiplier_override": None,
                },
            },
        ]
    )
    session = _session({"A": _market()}, opener=_opener_for(connection))

    items = [item async for item in session.iter_items(duration_seconds=0.02)]

    assert len(items) == 1
    assert isinstance(items[0], SubscriptionStartedEvent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "cause_match"),
    [
        (
            {
                "type": "event_fee_update",
                "sid": 8,
                "msg": {
                    "event_ticker": "EVENT",
                    "fee_type_override": "quadratic",
                    "fee_multiplier_override": None,
                },
            },
            "strict normalization",
        ),
        (
            {
                "type": "event_fee_update",
                "sid": 99,
                "msg": {
                    "event_ticker": "EVENT",
                    "fee_type_override": None,
                    "fee_multiplier_override": None,
                },
            },
            "unknown subscription",
        ),
    ],
)
async def test_malformed_or_unknown_subscription_fee_update_fails_closed(
    message: dict[str, object],
    cause_match: str,
) -> None:
    connection = _FakeConnection(
        [
            {
                "type": "subscribed",
                "id": 1,
                "msg": {"channel": "orderbook_delta", "sid": 7},
            },
            {
                "type": "subscribed",
                "id": 2,
                "msg": {"channel": "market_lifecycle_v2", "sid": 8},
            },
            message,
        ]
    )
    session = _session({"A": _market()}, opener=_opener_for(connection))

    with pytest.raises(WebSocketReconnectError) as raised:
        _ = [item async for item in session.iter_items(duration_seconds=1)]

    assert isinstance(raised.value.__cause__, WebSocketProtocolError)
    assert cause_match in str(raised.value.__cause__)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_ack",
    [
        {"type": "subscribed", "id": 99, "msg": {"channel": "orderbook_delta", "sid": 7}},
        {
            "type": "subscribed",
            "id": 1,
            "msg": {"channel": "market_lifecycle_v2", "sid": 8},
        },
    ],
)
async def test_subscription_acknowledgements_must_correlate_to_commands(
    bad_ack: dict[str, object],
) -> None:
    session = _session(
        {"A": _market()},
        opener=_opener_for(_FakeConnection([bad_ack])),
    )

    with pytest.raises(WebSocketReconnectError) as raised:
        _ = [item async for item in session.iter_items(duration_seconds=1)]

    assert isinstance(raised.value.__cause__, WebSocketProtocolError)
    assert "acknowledgement" in str(raised.value.__cause__)


@pytest.mark.asyncio
async def test_duplicate_subscription_acknowledgement_fails_closed() -> None:
    acknowledgement = {
        "type": "subscribed",
        "id": 1,
        "msg": {"channel": "orderbook_delta", "sid": 7},
    }
    session = _session(
        {"A": _market()},
        opener=_opener_for(_FakeConnection([acknowledgement, acknowledgement])),
    )

    with pytest.raises(WebSocketReconnectError) as raised:
        _ = [item async for item in session.iter_items(duration_seconds=1)]

    assert isinstance(raised.value.__cause__, WebSocketProtocolError)
    assert "unknown command ID" in str(raised.value.__cause__)


@pytest.mark.asyncio
async def test_snapshot_acknowledgement_must_match_pending_command_and_sid() -> None:
    connection = _FakeConnection(
        [
            {
                "type": "subscribed",
                "id": 1,
                "msg": {"channel": "orderbook_delta", "sid": 7},
            },
            _snapshot("A", sequence=1),
            {
                "type": "subscribed",
                "id": 2,
                "msg": {"channel": "market_lifecycle_v2", "sid": 8},
            },
            {
                "type": "market_lifecycle_v2",
                "sid": 8,
                "msg": {"market_ticker": "A", "event_type": "metadata_updated"},
            },
            {"type": "ok", "id": 4, "sid": 99},
        ]
    )
    session = _session({"A": _market()}, opener=_opener_for(connection))

    with pytest.raises(WebSocketReconnectError) as raised:
        async for item in session.iter_items(duration_seconds=1):
            if isinstance(item, MarketRefreshStartedEvent):
                await session.request_snapshots((item.ticker,))

    assert isinstance(raised.value.__cause__, WebSocketProtocolError)
    assert "wrong subscription ID" in str(raised.value.__cause__)


def test_start_event_index_and_subscription_ids_are_validated() -> None:
    with pytest.raises(ValueError, match="start_event_index"):
        _session(
            {"A": _market()},
            opener=_opener_for(_FakeConnection([])),
            start_event_index=-1,
        )

    with pytest.raises(ValueError, match="cannot both be supplied"):
        _session(
            {"A": _market()},
            opener=_opener_for(_FakeConnection([])),
            start_event_index=0,
            event_index_allocator=EventIndexAllocator(),
        )

    state = WebSocketStateMachine(("A",))
    state.connecting("connection-1")
    state.authenticated()
    state.subscribed(channel="market_lifecycle_v2", sid=8)
    with pytest.raises(WebSocketProtocolError, match="unknown subscription"):
        state.require_lifecycle_sid(channel="multivariate_market_lifecycle", sid=8)


@pytest.mark.asyncio
async def test_shared_event_index_allocator_orders_external_and_session_records() -> None:
    connection = _FakeConnection(
        [
            {
                "type": "subscribed",
                "id": 1,
                "msg": {"channel": "orderbook_delta", "sid": 7},
            },
            _snapshot("A", sequence=1),
        ]
    )
    allocator = EventIndexAllocator(start_event_index=10)
    assert allocator.allocate() == 10  # run-start record from the live composition root
    session = _session(
        {"A": _market()},
        opener=_opener_for(connection),
        event_index_allocator=allocator,
    )

    iterator = session.iter_items(duration_seconds=0.03)
    subscribed = await anext(iterator)
    assert isinstance(subscribed, SubscriptionStartedEvent)
    assert subscribed.event_index == 11
    assert allocator.allocate() == 12  # another producer between WebSocket receipts
    snapshot = await anext(iterator)
    assert isinstance(snapshot, OrderBookSnapshotEvent)
    assert snapshot.event_index == 13
    await iterator.aclose()

    assert session.event_index_allocator is allocator
    assert allocator.next_event_index == 14


@pytest.mark.asyncio
async def test_rejected_wire_message_does_not_consume_an_event_index() -> None:
    malformed = _delta("A", sequence=2)
    cast(dict[str, object], malformed["msg"])["delta_fp"] = "0.00"
    allocator = EventIndexAllocator()
    session = _session(
        {"A": _market()},
        opener=_opener_for(
            _FakeConnection(
                [
                    {
                        "type": "subscribed",
                        "id": 1,
                        "msg": {"channel": "orderbook_delta", "sid": 7},
                    },
                    _snapshot("A", sequence=1),
                    malformed,
                ]
            )
        ),
        event_index_allocator=allocator,
    )

    seen: list[RecordedSessionItem] = []
    with pytest.raises(WebSocketReconnectError):
        async for item in session.iter_items(duration_seconds=1):
            seen.append(item)

    assert [item.event_index for item in seen] == [0, 1, 2]
    assert isinstance(seen[-1], ConnectionInterruptedEvent)
    assert allocator.next_event_index == 3
