"""Authenticated Kalshi WebSocket protocol and bounded connection session.

This module owns exchange-specific commands, envelopes, and the unified YES-price
wire convention. Sequence continuity and local book mutation intentionally live in
``arbiter.engine.state`` so live collection and replay share the same decisions.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import ValidationError
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from arbiter.engine.state import EngineState, StateAction
from arbiter.kalshi.auth import KalshiWebSocketAuthenticator
from arbiter.kalshi.normalize import (
    NormalizationError,
    normalize_websocket_delta,
    normalize_websocket_snapshot,
)
from arbiter.kalshi.schemas import (
    WebSocketErrorWire,
    WebSocketEventFeeUpdateWire,
    WebSocketEventLifecycleWire,
    WebSocketLifecycleWire,
    WebSocketOkWire,
    WebSocketSubscribedWire,
)
from arbiter.models.event import Event
from arbiter.models.market import Market
from arbiter.replay.events import (
    ConnectionInterruptedEvent,
    EventIndexAllocator,
    FeeRefreshAppliedEvent,
    FeeRefreshStartedEvent,
    MarketDataEvent,
    MarketRefreshAppliedEvent,
    MarketRefreshStartedEvent,
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
    SubscriptionStartedEvent,
)
from arbiter.storage.parquet import ParquetEventWriter

ORDERBOOK_CHANNEL = "orderbook_delta"
LIFECYCLE_CHANNELS = ("market_lifecycle_v2", "multivariate_market_lifecycle")
METADATA_REFRESH_EVENTS = frozenset(
    {"created", "price_level_structure_updated", "metadata_updated"}
)


class KalshiWebSocketError(RuntimeError):
    """Base for safe, user-facing WebSocket failures."""


class WebSocketProtocolError(KalshiWebSocketError):
    """Raised when a message or state transition cannot be trusted."""


class WebSocketReconnectError(KalshiWebSocketError):
    """Raised after the configured finite connection-attempt budget is exhausted."""


class SessionPhase(StrEnum):
    """Externally inspectable connection/subscription state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    AUTHENTICATED = "authenticated"
    SUBSCRIBED = "subscribed"
    SNAPSHOT_RECEIVED = "snapshot_received"
    LIVE = "live"
    RESYNC_REQUIRED = "resync_required"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ConnectionOptions:
    """Network bounds supplied to an injectable connection opener."""

    open_timeout_seconds: float
    close_timeout_seconds: float
    inbound_queue_capacity: int


@dataclass(frozen=True, slots=True)
class MetadataRefresh:
    """A lifecycle message requiring bounded REST metadata refresh before reuse."""

    ticker: str
    event_type: str
    connection_id: str
    sid: int


@dataclass(frozen=True, slots=True)
class EventFeeRefresh:
    """A relevant event fee update that must be applied before another net claim."""

    event_ticker: str
    affected_tickers: tuple[str, ...]
    fee_type_override: str | None
    fee_multiplier_override: Decimal | None
    local_received_ts: datetime
    connection_id: str
    sid: int


@dataclass(frozen=True, slots=True)
class SubscriptionStarted:
    """An acknowledged order-book subscription ready for state registration."""

    connection_id: str
    sid: int
    tickers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConnectionInterrupted:
    """A fail-closed signal emitted before a bounded reconnect attempt."""

    connection_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class CollectorResult:
    """Finite collection outcome returned only after all accepted events are flushed."""

    events_written: int
    files_written: tuple[Path, ...]
    resync_requests: int
    metadata_refreshes: int
    event_fee_refreshes: int = 0


LegacySessionItem = (
    MarketDataEvent
    | MetadataRefresh
    | EventFeeRefresh
    | SubscriptionStarted
    | ConnectionInterrupted
)
RecordedSessionItem = (
    MarketDataEvent
    | SubscriptionStartedEvent
    | ConnectionInterruptedEvent
    | MarketRefreshStartedEvent
    | FeeRefreshStartedEvent
)
# Transitional compatibility while the Milestone 9 consumers move from the
# pre-recording dataclass signals to the schema-v2 controls emitted by the real
# session. New production code should consume ``RecordedSessionItem``.
SessionItem = LegacySessionItem | RecordedSessionItem
MetadataRefresher = Callable[[str], Awaitable[Market]]
EventFeeRefresher = Callable[[EventFeeRefresh], Awaitable[None]]
RecordedEventFeeRefresher = Callable[[FeeRefreshStartedEvent], Awaitable[Event]]


class WebSocketConnection(Protocol):
    """Small subset shared by the real client and deterministic test doubles."""

    async def send(self, message: str | bytes) -> None: ...

    async def recv(self, decode: bool | None = None) -> str | bytes: ...


class MarketDataSession(Protocol):
    """Injectable live-session boundary consumed by the collector."""

    def iter_items(self, *, duration_seconds: float) -> AsyncIterator[SessionItem]: ...

    async def request_snapshots(self, tickers: Sequence[str]) -> None: ...

    def update_market_metadata(self, market: Market) -> None: ...


ConnectionOpener = Callable[
    [str, Mapping[str, str], ConnectionOptions],
    AbstractAsyncContextManager[WebSocketConnection],
]


@asynccontextmanager
async def _open_connection(
    url: str,
    headers: Mapping[str, str],
    options: ConnectionOptions,
) -> AsyncIterator[WebSocketConnection]:
    async with connect(
        url,
        additional_headers=headers,
        open_timeout=options.open_timeout_seconds,
        close_timeout=options.close_timeout_seconds,
        max_queue=options.inbound_queue_capacity,
    ) as connection:
        yield cast(WebSocketConnection, connection)


def _validated_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(sorted(set(tickers)))
    if not normalized or any(not ticker or ticker.strip() != ticker for ticker in normalized):
        raise ValueError("at least one nonblank market ticker is required")
    if len(normalized) != len(tickers):
        raise ValueError("market tickers must be unique")
    return normalized


def orderbook_subscription_command(command_id: int, tickers: Sequence[str]) -> dict[str, Any]:
    """Build the current explicit-ticker, unified-YES-price subscription command."""

    _validate_command_id(command_id)
    return {
        "id": command_id,
        "cmd": "subscribe",
        "params": {
            "channels": [ORDERBOOK_CHANNEL],
            "market_tickers": list(_validated_tickers(tickers)),
            "use_yes_price": True,
        },
    }


def lifecycle_subscription_command(command_id: int, channel: str) -> dict[str, Any]:
    """Build a separate global lifecycle subscription command."""

    _validate_command_id(command_id)
    if channel not in LIFECYCLE_CHANNELS:
        raise ValueError("unsupported lifecycle channel")
    return {
        "id": command_id,
        "cmd": "subscribe",
        "params": {"channels": [channel]},
    }


def snapshot_request_command(
    command_id: int,
    *,
    sid: int,
    tickers: Sequence[str],
) -> dict[str, Any]:
    """Request replacement snapshots without changing subscription membership."""

    _validate_command_id(command_id)
    if isinstance(sid, bool) or sid < 1:
        raise ValueError("subscription ID must be positive")
    return {
        "id": command_id,
        "cmd": "update_subscription",
        "params": {
            "sids": [sid],
            "market_tickers": list(_validated_tickers(tickers)),
            "action": "get_snapshot",
        },
    }


def _validate_command_id(command_id: int) -> None:
    if isinstance(command_id, bool) or not isinstance(command_id, int) or command_id < 1:
        raise ValueError("command ID must be a positive integer")


class CommandSequencer:
    """Generate unique, monotonically increasing command IDs within one session."""

    def __init__(self) -> None:
        self._last = 0

    def next(self) -> int:
        self._last += 1
        return self._last


class WebSocketStateMachine:
    """Validate the documented connection lifecycle and replacement-snapshot barrier."""

    def __init__(self, tickers: Sequence[str]) -> None:
        self.tickers = _validated_tickers(tickers)
        self.phase = SessionPhase.DISCONNECTED
        self.connection_id: str | None = None
        self.orderbook_sid: int | None = None
        self.lifecycle_sids: dict[str, int] = {}
        self.pending_snapshots: set[str] = set(self.tickers)

    def connecting(self, connection_id: str) -> None:
        if self.phase not in {SessionPhase.DISCONNECTED, SessionPhase.CLOSED}:
            raise WebSocketProtocolError("connection cannot start from the current state")
        if not connection_id or connection_id.strip() != connection_id:
            raise WebSocketProtocolError("connection ID must be nonblank")
        self.connection_id = connection_id
        self.orderbook_sid = None
        self.lifecycle_sids.clear()
        self.pending_snapshots = set(self.tickers)
        self.phase = SessionPhase.CONNECTING

    def authenticated(self) -> None:
        if self.phase is not SessionPhase.CONNECTING:
            raise WebSocketProtocolError("authentication completed from an invalid state")
        self.phase = SessionPhase.AUTHENTICATED

    def subscribed(self, *, channel: str, sid: int) -> None:
        active_phases = {
            SessionPhase.AUTHENTICATED,
            SessionPhase.SUBSCRIBED,
            SessionPhase.SNAPSHOT_RECEIVED,
            SessionPhase.LIVE,
            SessionPhase.RESYNC_REQUIRED,
        }
        if self.phase not in active_phases:
            raise WebSocketProtocolError("subscription acknowledgement arrived out of state")
        if channel == ORDERBOOK_CHANNEL:
            if self.phase not in {SessionPhase.AUTHENTICATED, SessionPhase.SUBSCRIBED}:
                raise WebSocketProtocolError(
                    "order-book subscription acknowledgement arrived after market data"
                )
            if self.orderbook_sid is not None and self.orderbook_sid != sid:
                raise WebSocketProtocolError("order-book subscription ID changed unexpectedly")
            if sid in self.lifecycle_sids.values():
                raise WebSocketProtocolError("subscription IDs must identify one channel")
            self.orderbook_sid = sid
            self.phase = SessionPhase.SUBSCRIBED
        elif channel in LIFECYCLE_CHANNELS:
            existing = self.lifecycle_sids.get(channel)
            if existing is not None and existing != sid:
                raise WebSocketProtocolError("lifecycle subscription ID changed unexpectedly")
            other_sids = {
                value
                for existing_channel, value in self.lifecycle_sids.items()
                if existing_channel != channel
            }
            if sid == self.orderbook_sid or sid in other_sids:
                raise WebSocketProtocolError("subscription IDs must identify one channel")
            self.lifecycle_sids[channel] = sid
        else:
            raise WebSocketProtocolError("subscription acknowledgement has an unknown channel")

    def require_lifecycle_sid(self, *, channel: str, sid: int) -> None:
        """Prove a lifecycle message belongs to the acknowledged channel subscription."""

        if self.lifecycle_sids.get(channel) != sid:
            raise WebSocketProtocolError("lifecycle message belongs to an unknown subscription")

    def require_event_lifecycle_sid(self, sid: int) -> None:
        """Accept the shared event envelope only from either acknowledged lifecycle channel."""

        if sid not in self.lifecycle_sids.values():
            raise WebSocketProtocolError("event lifecycle message has an unknown subscription")

    def snapshot_received(self, *, ticker: str, sid: int) -> None:
        if self.orderbook_sid is None or sid != self.orderbook_sid:
            raise WebSocketProtocolError("snapshot belongs to an unknown subscription")
        if ticker not in self.tickers:
            raise WebSocketProtocolError("snapshot belongs to an unsubscribed market")
        self.pending_snapshots.discard(ticker)
        self.phase = (
            SessionPhase.LIVE if not self.pending_snapshots else SessionPhase.SNAPSHOT_RECEIVED
        )

    def require_resync(self, tickers: Sequence[str] | None = None) -> None:
        selected = set(self.tickers if tickers is None else _validated_tickers(tickers))
        if not selected.issubset(self.tickers):
            raise WebSocketProtocolError("cannot resynchronize an unsubscribed market")
        self.pending_snapshots.update(selected)
        self.phase = SessionPhase.RESYNC_REQUIRED

    def disconnected(self) -> None:
        self.orderbook_sid = None
        self.lifecycle_sids.clear()
        self.pending_snapshots = set(self.tickers)
        self.phase = SessionPhase.DISCONNECTED

    def close(self) -> None:
        self.orderbook_sid = None
        self.lifecycle_sids.clear()
        self.phase = SessionPhase.CLOSED


class KalshiWebSocketSession:
    """Yield normalized stream items across finitely many reconnect attempts."""

    def __init__(
        self,
        *,
        url: str,
        authenticator: KalshiWebSocketAuthenticator,
        markets: Mapping[str, Market],
        inbound_queue_capacity: int = 1_000,
        reconnect_max_attempts: int = 5,
        reconnect_initial_backoff_seconds: float = 0.5,
        reconnect_max_backoff_seconds: float = 8.0,
        open_timeout_seconds: float = 10.0,
        close_timeout_seconds: float = 5.0,
        opener: ConnectionOpener = _open_connection,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        local_clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        connection_id_factory: Callable[[], str] | None = None,
        start_event_index: int | None = None,
        event_index_allocator: EventIndexAllocator | None = None,
    ) -> None:
        if not url.startswith(("wss://", "ws://")):
            raise ValueError("WebSocket URL must use ws or wss")
        self.tickers = _validated_tickers(tuple(markets))
        if set(self.tickers) != set(markets):
            raise ValueError("market metadata keys must be unique subscribed tickers")
        if inbound_queue_capacity < 1 or reconnect_max_attempts < 0:
            raise ValueError("queue capacity must be positive and reconnect attempts nonnegative")
        numeric_bounds = (
            reconnect_initial_backoff_seconds,
            reconnect_max_backoff_seconds,
            open_timeout_seconds,
            close_timeout_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in numeric_bounds):
            raise ValueError("WebSocket timeout and backoff values must be finite and positive")
        if reconnect_initial_backoff_seconds > reconnect_max_backoff_seconds:
            raise ValueError("initial reconnect backoff cannot exceed the maximum")
        if event_index_allocator is not None and start_event_index is not None:
            raise ValueError("event_index_allocator and start_event_index cannot both be supplied")

        self.url = url
        self.authenticator = authenticator
        self.markets = dict(markets)
        self.options = ConnectionOptions(
            open_timeout_seconds=open_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            inbound_queue_capacity=inbound_queue_capacity,
        )
        self.reconnect_max_attempts = reconnect_max_attempts
        self.reconnect_initial_backoff_seconds = reconnect_initial_backoff_seconds
        self.reconnect_max_backoff_seconds = reconnect_max_backoff_seconds
        self._opener = opener
        self._sleeper = sleeper
        self._local_clock = local_clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._connection_id_factory = connection_id_factory or (lambda: uuid.uuid4().hex)
        self._commands = CommandSequencer()
        self._state = WebSocketStateMachine(self.tickers)
        self._connection: WebSocketConnection | None = None
        self._snapshot_ids: dict[str, str] = {}
        self._event_indices = event_index_allocator or EventIndexAllocator(
            0 if start_event_index is None else start_event_index
        )
        self._pending_subscriptions: dict[int, str] = {}
        self._pending_snapshot_requests: dict[int, tuple[int, tuple[str, ...]]] = {}

    @property
    def state(self) -> WebSocketStateMachine:
        return self._state

    @property
    def event_index_allocator(self) -> EventIndexAllocator:
        """Return the allocator shared with other producers in this recorded run."""

        return self._event_indices

    async def request_snapshots(self, tickers: Sequence[str]) -> None:
        """Request replacement snapshots on the active order-book subscription."""

        connection = self._connection
        sid = self._state.orderbook_sid
        if connection is None or sid is None:
            raise WebSocketProtocolError("cannot request a snapshot without an active subscription")
        selected = _validated_tickers(tickers)
        self._state.require_resync(selected)
        command_id = self._commands.next()
        self._pending_snapshot_requests[command_id] = (sid, selected)
        try:
            await self._send(
                connection,
                snapshot_request_command(command_id, sid=sid, tickers=selected),
            )
        except BaseException:
            self._pending_snapshot_requests.pop(command_id, None)
            raise

    def update_market_metadata(self, market: Market) -> None:
        """Replace one subscribed market only after a bounded authoritative refresh."""

        if market.ticker not in self.markets:
            raise WebSocketProtocolError("cannot update metadata for an unsubscribed market")
        self.markets[market.ticker] = market

    async def iter_items(self, *, duration_seconds: float) -> AsyncIterator[RecordedSessionItem]:
        """Connect, subscribe, and yield normalized items until a fixed deadline."""

        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("duration_seconds must be finite and positive")
        deadline = self._monotonic() + duration_seconds
        attempt = 0
        backoff = self.reconnect_initial_backoff_seconds
        last_error: BaseException | None = None

        total_attempts = self.reconnect_max_attempts + 1
        while self._monotonic() < deadline and attempt < total_attempts:
            attempt += 1
            connection_id = self._connection_id_factory()
            self._state.connecting(connection_id)
            self._snapshot_ids.clear()
            self._pending_subscriptions.clear()
            self._pending_snapshot_requests.clear()
            attempt_error: BaseException | None = None
            try:
                headers = self.authenticator.handshake_headers()
                async with self._opener(self.url, headers, self.options) as connection:
                    self._connection = connection
                    self._state.authenticated()
                    await self._send_initial_subscriptions(connection)
                    while True:
                        remaining = deadline - self._monotonic()
                        if remaining <= 0:
                            self._state.close()
                            return
                        try:
                            raw = await asyncio.wait_for(connection.recv(), timeout=remaining)
                        except TimeoutError:
                            self._state.close()
                            return
                        item = self._decode(raw, connection_id=connection_id)
                        if item is not None:
                            yield item
            except (OSError, WebSocketException, WebSocketProtocolError) as exc:
                attempt_error = exc
                last_error = exc
            finally:
                self._connection = None

            if attempt_error is None:
                raise RuntimeError("WebSocket attempt ended without a result or error")
            self._state.disconnected()
            interrupted_at = self._received_at()
            yield ConnectionInterruptedEvent(
                event_index=self._event_indices.allocate(),
                local_received_ts=interrupted_at,
                connection_id=connection_id,
                tickers=self.tickers,
                reason=f"{type(attempt_error).__name__}; books invalidated before reconnect",
            )
            if self._monotonic() >= deadline:
                self._state.close()
                return
            if attempt >= total_attempts:
                break
            delay = min(backoff, max(0.0, deadline - self._monotonic()))
            await self._sleeper(delay)
            backoff = min(backoff * 2, self.reconnect_max_backoff_seconds)

        raise WebSocketReconnectError(
            f"WebSocket connection exhausted {attempt} bounded attempt(s)"
        ) from last_error

    async def _send_initial_subscriptions(self, connection: WebSocketConnection) -> None:
        await self._send_subscription(connection, ORDERBOOK_CHANNEL)
        for channel in LIFECYCLE_CHANNELS:
            await self._send_subscription(connection, channel)

    async def _send_subscription(
        self,
        connection: WebSocketConnection,
        channel: str,
    ) -> None:
        command_id = self._commands.next()
        self._pending_subscriptions[command_id] = channel
        command = (
            orderbook_subscription_command(command_id, self.tickers)
            if channel == ORDERBOOK_CHANNEL
            else lifecycle_subscription_command(command_id, channel)
        )
        try:
            await self._send(connection, command)
        except BaseException:
            self._pending_subscriptions.pop(command_id, None)
            raise

    @staticmethod
    async def _send(connection: WebSocketConnection, command: Mapping[str, Any]) -> None:
        await connection.send(json.dumps(command, separators=(",", ":"), sort_keys=True))

    def _decode(self, raw: str | bytes, *, connection_id: str) -> RecordedSessionItem | None:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebSocketProtocolError("WebSocket message is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
            raise WebSocketProtocolError("WebSocket message must be a JSON object")
        message = cast(dict[str, Any], payload)
        message_type = message.get("type")
        try:
            if message_type == "subscribed":
                subscribed = WebSocketSubscribedWire.model_validate(message)
                self._correlate_subscription_ack(
                    command_id=subscribed.id,
                    channel=subscribed.msg.channel,
                )
                self._state.subscribed(channel=subscribed.msg.channel, sid=subscribed.msg.sid)
                if subscribed.msg.channel == ORDERBOOK_CHANNEL:
                    subscribed_at = self._received_at()
                    return SubscriptionStartedEvent(
                        event_index=self._event_indices.allocate(),
                        local_received_ts=subscribed_at,
                        connection_id=connection_id,
                        sid=subscribed.msg.sid,
                        tickers=self.tickers,
                    )
                return None
            if message_type == "ok":
                ok = WebSocketOkWire.model_validate(message)
                self._correlate_snapshot_ack(command_id=ok.id, sid=ok.sid)
                return None
            if message_type == "error":
                error = WebSocketErrorWire.model_validate(message)
                raise WebSocketProtocolError(
                    f"Kalshi rejected WebSocket command with code {error.msg.code}"
                )
            if message_type == "orderbook_snapshot":
                return self._decode_snapshot(message, connection_id=connection_id)
            if message_type == "orderbook_delta":
                return self._decode_delta(message, connection_id=connection_id)
            if message_type in LIFECYCLE_CHANNELS:
                return self._decode_lifecycle(message, connection_id=connection_id)
            if message_type == "event_lifecycle":
                event_lifecycle = WebSocketEventLifecycleWire.model_validate(message)
                self._state.require_event_lifecycle_sid(event_lifecycle.sid)
                return None
            if message_type == "event_fee_update":
                return self._decode_event_fee_update(message, connection_id=connection_id)
        except (ValidationError, NormalizationError) as exc:
            raise WebSocketProtocolError("WebSocket message failed strict normalization") from exc
        raise WebSocketProtocolError("WebSocket message has an unsupported type")

    def _correlate_subscription_ack(self, *, command_id: int | None, channel: str) -> None:
        if command_id is None or command_id == 0:
            matching = tuple(
                pending_id
                for pending_id, pending_channel in self._pending_subscriptions.items()
                if pending_channel == channel
            )
            if len(matching) != 1:
                raise WebSocketProtocolError(
                    "subscription acknowledgement cannot be correlated to one command"
                )
            command_id = matching[0]
        expected = self._pending_subscriptions.get(command_id)
        if expected is None:
            raise WebSocketProtocolError("subscription acknowledgement has an unknown command ID")
        if expected != channel:
            raise WebSocketProtocolError("subscription acknowledgement channel does not match")
        del self._pending_subscriptions[command_id]

    def _correlate_snapshot_ack(self, *, command_id: int | None, sid: int | None) -> None:
        if command_id is None or command_id == 0:
            if len(self._pending_snapshot_requests) != 1:
                raise WebSocketProtocolError(
                    "snapshot acknowledgement cannot be correlated to one command"
                )
            command_id = next(iter(self._pending_snapshot_requests))
        pending = self._pending_snapshot_requests.get(command_id)
        if pending is None:
            raise WebSocketProtocolError("snapshot acknowledgement has an unknown command ID")
        expected_sid, _ = pending
        if sid is not None and sid != expected_sid:
            raise WebSocketProtocolError("snapshot acknowledgement has the wrong subscription ID")
        del self._pending_snapshot_requests[command_id]

    def _decode_snapshot(
        self,
        payload: Mapping[str, Any],
        *,
        connection_id: str,
    ) -> MarketDataEvent:
        ticker = _message_ticker(payload)
        market = self._market(ticker)
        event = normalize_websocket_snapshot(
            payload,
            event_index=0,
            connection_id=connection_id,
            local_received_ts=self._received_at(),
            market=market,
        )
        self._validate_orderbook_sid(event.sid)
        event = event.model_copy(
            update={"event_index": self._event_indices.allocate()},
        )
        self._snapshot_ids[ticker] = event.snapshot_id
        self._state.snapshot_received(ticker=ticker, sid=event.sid)
        return event

    def _decode_delta(
        self,
        payload: Mapping[str, Any],
        *,
        connection_id: str,
    ) -> MarketDataEvent:
        ticker = _message_ticker(payload)
        market = self._market(ticker)
        snapshot_id = self._snapshot_ids.get(ticker)
        if snapshot_id is None:
            raise WebSocketProtocolError("delta arrived before a trusted snapshot")
        event = normalize_websocket_delta(
            payload,
            event_index=0,
            connection_id=connection_id,
            local_received_ts=self._received_at(),
            market=market,
            snapshot_id=snapshot_id,
        )
        self._validate_orderbook_sid(event.sid)
        event = event.model_copy(
            update={"event_index": self._event_indices.allocate()},
        )
        return event

    def _decode_lifecycle(
        self,
        payload: Mapping[str, Any],
        *,
        connection_id: str,
    ) -> MarketRefreshStartedEvent | None:
        wire = WebSocketLifecycleWire.model_validate(payload)
        self._state.require_lifecycle_sid(channel=wire.type, sid=wire.sid)
        ticker = wire.msg.market_ticker
        if ticker not in self.markets or wire.msg.event_type not in METADATA_REFRESH_EVENTS:
            return None
        self._state.require_resync((ticker,))
        refresh_started_at = self._received_at()
        return MarketRefreshStartedEvent(
            event_index=self._event_indices.allocate(),
            local_received_ts=refresh_started_at,
            ticker=ticker,
            reason=wire.msg.event_type,
            connection_id=connection_id,
            sid=wire.sid,
        )

    def _decode_event_fee_update(
        self,
        payload: Mapping[str, Any],
        *,
        connection_id: str,
    ) -> FeeRefreshStartedEvent | None:
        wire = WebSocketEventFeeUpdateWire.model_validate(payload)
        self._state.require_lifecycle_sid(channel="market_lifecycle_v2", sid=wire.sid)
        affected_tickers = tuple(
            sorted(
                ticker
                for ticker, market in self.markets.items()
                if market.event_ticker == wire.msg.event_ticker
            )
        )
        if not affected_tickers:
            return None
        refresh_started_at = self._received_at()
        return FeeRefreshStartedEvent(
            event_index=self._event_indices.allocate(),
            local_received_ts=refresh_started_at,
            event_ticker=wire.msg.event_ticker,
            affected_tickers=affected_tickers,
            connection_id=connection_id,
            sid=wire.sid,
        )

    def _received_at(self) -> datetime:
        received_at = self._local_clock()
        if received_at.tzinfo is None or received_at.utcoffset() is None:
            raise WebSocketProtocolError("local receive clock returned a naive timestamp")
        return received_at

    def _market(self, ticker: str) -> Market:
        try:
            return self.markets[ticker]
        except KeyError as exc:
            raise WebSocketProtocolError("message belongs to an unsubscribed market") from exc

    def _validate_orderbook_sid(self, sid: int) -> None:
        if self._state.orderbook_sid is None or sid != self._state.orderbook_sid:
            raise WebSocketProtocolError("message belongs to an unknown order-book subscription")


class MarketDataCollector:
    """Apply one normalized stream to shared state and bounded Parquet persistence."""

    def __init__(
        self,
        *,
        session: MarketDataSession,
        state: EngineState,
        writer: ParquetEventWriter,
        writer_batch_size: int,
        writer_flush_interval_seconds: float,
        metadata_refresher: MetadataRefresher | None,
        event_fee_refresher: EventFeeRefresher | RecordedEventFeeRefresher | None = None,
        event_index_allocator: EventIndexAllocator | None = None,
        monotonic: Callable[[], float] | None = None,
        local_clock: Callable[[], datetime] | None = None,
    ) -> None:
        if writer_batch_size < 1:
            raise ValueError("writer_batch_size must be positive")
        if not math.isfinite(writer_flush_interval_seconds) or writer_flush_interval_seconds <= 0:
            raise ValueError("writer flush interval must be finite and positive")
        self.session = session
        self.state = state
        self.writer = writer
        self.writer_batch_size = writer_batch_size
        self.writer_flush_interval_seconds = writer_flush_interval_seconds
        self.metadata_refresher = metadata_refresher
        self.event_fee_refresher = event_fee_refresher
        session_allocator = getattr(session, "event_index_allocator", None)
        if (
            event_index_allocator is not None
            and session_allocator is not None
            and event_index_allocator is not session_allocator
        ):
            raise ValueError("collector and session must share one event-index allocator")
        self.event_index_allocator = event_index_allocator or session_allocator
        self._monotonic = monotonic or time.monotonic
        self._local_clock = local_clock or (lambda: datetime.now(UTC))

    async def run(self, *, duration_seconds: float) -> CollectorResult:
        """Collect for a bounded duration and return only after durable flush/close."""

        paths: list[Path] = []
        events_written = 0
        resync_requests = 0
        metadata_refreshes = 0
        event_fee_refreshes = 0
        last_flush = self._monotonic()
        active_connections: set[str] = set()
        try:
            async for item in self.session.iter_items(duration_seconds=duration_seconds):
                if isinstance(item, SubscriptionStarted):
                    self.state.register_subscription(
                        connection_id=item.connection_id,
                        sid=item.sid,
                        tickers=item.tickers,
                    )
                    active_connections.add(item.connection_id)
                    continue
                if isinstance(item, ConnectionInterrupted):
                    self.state.disconnect_connection(item.connection_id)
                    active_connections.discard(item.connection_id)
                    continue
                if isinstance(item, SubscriptionStartedEvent):
                    self._append_session_control(item)
                    self.state.register_subscription(
                        connection_id=item.connection_id,
                        sid=item.sid,
                        tickers=item.tickers,
                    )
                    active_connections.add(item.connection_id)
                elif isinstance(item, ConnectionInterruptedEvent):
                    self._append_session_control(item)
                    self.state.disconnect_connection(item.connection_id)
                    active_connections.discard(item.connection_id)
                elif isinstance(item, MarketRefreshStartedEvent):
                    self._append_session_control(item)
                    await self._refresh_recorded_metadata(item)
                    metadata_refreshes += 1
                    resync_requests += 1
                elif isinstance(item, FeeRefreshStartedEvent):
                    self._append_session_control(item)
                    await self._refresh_recorded_event_fee(item)
                    event_fee_refreshes += 1
                if isinstance(item, MetadataRefresh):
                    await self._refresh_metadata(item)
                    metadata_refreshes += 1
                    resync_requests += 1
                    continue
                if isinstance(item, EventFeeRefresh):
                    await self._refresh_event_fee(item)
                    event_fee_refreshes += 1
                    continue
                if isinstance(
                    item,
                    (
                        SubscriptionStartedEvent,
                        ConnectionInterruptedEvent,
                        MarketRefreshStartedEvent,
                        FeeRefreshStartedEvent,
                    ),
                ):
                    pass
                elif isinstance(item, (OrderBookSnapshotEvent, OrderBookDeltaEvent)):
                    self._append_session_book(item)
                    events_written += 1
                    update = self.state.apply_event(item)
                    if update.action is StateAction.RESYNC_REQUIRED:
                        await self.session.request_snapshots(update.affected_tickers)
                        resync_requests += 1
                else:
                    raise WebSocketProtocolError(
                        f"unsupported collector session item {type(item).__name__}"
                    )
                now = self._monotonic()
                if (
                    self.writer.pending_count >= self.writer_batch_size
                    or now - last_flush >= self.writer_flush_interval_seconds
                ):
                    paths.extend(self.writer.flush())
                    last_flush = now
        except BaseException:
            if self.writer.pending_count:
                paths.extend(self.writer.flush())
            raise
        finally:
            for connection_id in tuple(active_connections):
                self.state.disconnect_connection(connection_id)
        paths.extend(self.writer.close())
        return CollectorResult(
            events_written=events_written,
            files_written=tuple(paths),
            resync_requests=resync_requests,
            metadata_refreshes=metadata_refreshes,
            event_fee_refreshes=event_fee_refreshes,
        )

    async def _refresh_metadata(self, item: MetadataRefresh) -> None:
        self.state.require_market_resync(
            item.ticker,
            reason=f"Kalshi lifecycle update: {item.event_type}",
        )
        if self.metadata_refresher is None:
            raise WebSocketProtocolError(
                "market metadata changed but no bounded metadata refresher is configured"
            )
        refreshed = await self.metadata_refresher(item.ticker)
        if refreshed.ticker != item.ticker:
            raise WebSocketProtocolError("metadata refresh returned the wrong market")
        self.session.update_market_metadata(refreshed)
        await self.session.request_snapshots((item.ticker,))

    async def _refresh_event_fee(self, item: EventFeeRefresh) -> None:
        if self.event_fee_refresher is None:
            raise WebSocketProtocolError(
                "event fee metadata changed but no bounded fee refresher is configured"
            )
        refresher = cast(EventFeeRefresher, self.event_fee_refresher)
        await refresher(item)

    def _recorded_allocator(self) -> EventIndexAllocator:
        allocator = self.event_index_allocator
        if not isinstance(allocator, EventIndexAllocator):
            raise WebSocketProtocolError(
                "indexed session controls require the collector to share the session "
                "event-index allocator"
            )
        return allocator

    def _append_session_control(
        self,
        item: (
            SubscriptionStartedEvent
            | ConnectionInterruptedEvent
            | MarketRefreshStartedEvent
            | FeeRefreshStartedEvent
        ),
    ) -> None:
        allocator = self._recorded_allocator()
        if allocator.next_event_index != item.event_index + 1:
            raise WebSocketProtocolError(
                "recorded session control was not allocated from the shared event-index sequence"
            )
        self.writer.append(item)
        if allocator.next_event_index != self.writer.next_event_index:
            raise WebSocketProtocolError(
                "recorded session and writer event-index sequences diverged"
            )

    def _append_session_book(self, item: MarketDataEvent) -> None:
        allocator = self.event_index_allocator
        if (
            isinstance(allocator, EventIndexAllocator)
            and allocator.next_event_index != item.event_index + 1
        ):
            raise WebSocketProtocolError(
                "recorded book event was not allocated from the shared event-index sequence"
            )
        self.writer.append(item)
        if (
            isinstance(allocator, EventIndexAllocator)
            and allocator.next_event_index != self.writer.next_event_index
        ):
            raise WebSocketProtocolError(
                "recorded session and writer event-index sequences diverged"
            )

    def _append_applied_control(
        self,
        item: MarketRefreshAppliedEvent | FeeRefreshAppliedEvent,
    ) -> None:
        allocator = self._recorded_allocator()
        if item.event_index != allocator.next_event_index - 1:
            raise WebSocketProtocolError(
                "applied refresh was not allocated from the shared event-index sequence"
            )
        self.writer.append(item)
        if allocator.next_event_index != self.writer.next_event_index:
            raise WebSocketProtocolError(
                "applied refresh and writer event-index sequences diverged"
            )

    def _control_time(self, lower_bound: datetime) -> datetime:
        received_at = self._local_clock()
        if received_at.tzinfo is None or received_at.utcoffset() is None:
            raise WebSocketProtocolError("collector control clock returned a naive timestamp")
        return max(received_at, lower_bound)

    async def _refresh_recorded_metadata(self, item: MarketRefreshStartedEvent) -> None:
        self.state.require_market_resync(
            item.ticker,
            reason=f"Kalshi lifecycle update: {item.reason}",
        )
        if self.metadata_refresher is None:
            raise WebSocketProtocolError(
                "market metadata changed but no bounded metadata refresher is configured"
            )
        refreshed = await self.metadata_refresher(item.ticker)
        if refreshed.ticker != item.ticker:
            raise WebSocketProtocolError("metadata refresh returned the wrong market")
        allocator = self._recorded_allocator()
        applied = MarketRefreshAppliedEvent(
            event_index=allocator.allocate(),
            local_received_ts=self._control_time(item.local_received_ts),
            refresh_started_event_index=item.event_index,
            market=refreshed,
        )
        self._append_applied_control(applied)
        self.session.update_market_metadata(refreshed)
        await self.session.request_snapshots((item.ticker,))

    async def _refresh_recorded_event_fee(self, item: FeeRefreshStartedEvent) -> None:
        if self.event_fee_refresher is None:
            raise WebSocketProtocolError(
                "event fee metadata changed but no bounded fee refresher is configured"
            )
        refresher = cast(RecordedEventFeeRefresher, self.event_fee_refresher)
        refreshed = await refresher(item)
        if not isinstance(refreshed, Event) or refreshed.ticker != item.event_ticker:
            raise WebSocketProtocolError("event fee refresh returned the wrong event")
        if not set(item.affected_tickers).issubset(refreshed.market_tickers):
            raise WebSocketProtocolError(
                "event fee refresh omitted canonical affected-market membership"
            )
        allocator = self._recorded_allocator()
        applied = FeeRefreshAppliedEvent(
            event_index=allocator.allocate(),
            local_received_ts=self._control_time(item.local_received_ts),
            refresh_started_event_index=item.event_index,
            event=refreshed,
        )
        self._append_applied_control(applied)


def _message_ticker(payload: Mapping[str, Any]) -> str:
    msg = payload.get("msg")
    if not isinstance(msg, Mapping):
        raise WebSocketProtocolError("WebSocket message body is missing")
    ticker = msg.get("market_ticker")
    if not isinstance(ticker, str) or not ticker:
        raise WebSocketProtocolError("WebSocket market ticker is missing")
    return ticker


__all__ = [
    "ConnectionOpener",
    "ConnectionOptions",
    "ConnectionInterrupted",
    "CollectorResult",
    "EventFeeRefresh",
    "EventFeeRefresher",
    "KalshiWebSocketError",
    "KalshiWebSocketSession",
    "LIFECYCLE_CHANNELS",
    "LegacySessionItem",
    "METADATA_REFRESH_EVENTS",
    "MarketDataCollector",
    "MarketDataSession",
    "MetadataRefresher",
    "MetadataRefresh",
    "ORDERBOOK_CHANNEL",
    "RecordedSessionItem",
    "RecordedEventFeeRefresher",
    "SessionItem",
    "SessionPhase",
    "SubscriptionStarted",
    "WebSocketConnection",
    "WebSocketProtocolError",
    "WebSocketReconnectError",
    "WebSocketStateMachine",
    "lifecycle_subscription_command",
    "orderbook_subscription_command",
    "snapshot_request_command",
]
