"""Fail-closed sequence tracking and reconstructed subscription book state.

Kalshi's documented sequence scope is not precise enough to embed a global exchange invariant.
The default policy is therefore deliberately isolated in :class:`SequenceTracker`: synthetic
and offline operation tracks continuity per ``(connection_id, sid)``. A credential-gated live
smoke may validate that assumption later without changing book-state business logic.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from arbiter.logic.worlds import WorldCache, generate_component_worlds
from arbiter.models.event import Event
from arbiter.models.market import Market
from arbiter.models.opportunity import MarketResearchContext
from arbiter.models.orderbook import BookStatus, OrderBook, OrderBookUpdateError
from arbiter.models.relation import LogicalComponent, Relation
from arbiter.models.series import Series
from arbiter.models.world import WorldGenerationResult
from arbiter.relations.graph import RelationGraph
from arbiter.replay.events import (
    MarketDataEvent,
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
)

SequenceKey = tuple[str, int]


class SequenceDisposition(StrEnum):
    """Result of comparing one event with its isolated sequence scope."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class SequenceObservation:
    """One sequence decision with a stable diagnostic reason."""

    disposition: SequenceDisposition
    reason: str


@dataclass(slots=True)
class _SequenceState:
    last_sequence: int | None
    fingerprints: OrderedDict[int, str]
    uncertain: bool = False


def market_data_fingerprint(event: MarketDataEvent) -> str:
    """Hash exchange-relevant event content, excluding local receipt identity.

    ``event_index`` and ``local_received_ts`` necessarily differ when an identical wire message
    is delivered twice. Excluding only those local fields makes exact delivery duplicates
    idempotent while treating any exchange-field difference at the same sequence as a conflict.
    """

    canonical = json.dumps(
        event.model_dump(
            mode="json",
            exclude={"event_index", "local_received_ts"},
        ),
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


class SequenceTracker:
    """Track continuity per ``(connection_id, sid)`` behind one replaceable policy boundary."""

    scope_description = "per (connection_id, sid)"

    def __init__(self, *, fingerprint_history: int = 4096) -> None:
        if (
            isinstance(fingerprint_history, bool)
            or not isinstance(fingerprint_history, int)
            or fingerprint_history < 1
        ):
            raise ValueError("fingerprint_history must be positive")
        self._fingerprint_history = fingerprint_history
        self._states: dict[SequenceKey, _SequenceState] = {}

    @staticmethod
    def key_for(event: MarketDataEvent) -> SequenceKey:
        return event.connection_id, event.sid

    def observe(self, event: MarketDataEvent) -> SequenceObservation:
        """Accept continuity, ignore exact duplicates, and latch every ambiguity uncertain."""

        key = self.key_for(event)
        state = self._states.get(key)
        if state is None:
            self._set_baseline(event)
            return SequenceObservation(SequenceDisposition.ACCEPTED, "initial sequence baseline")
        if state.uncertain:
            return SequenceObservation(
                SequenceDisposition.UNCERTAIN,
                "sequence scope awaits an explicit snapshot reset",
            )

        fingerprint = market_data_fingerprint(event)
        previous = state.fingerprints.get(event.sequence)
        if previous is not None:
            if previous == fingerprint:
                return SequenceObservation(
                    SequenceDisposition.DUPLICATE,
                    "exact sequence duplicate",
                )
            state.uncertain = True
            return SequenceObservation(
                SequenceDisposition.UNCERTAIN,
                "conflicting payloads share one sequence",
            )

        assert state.last_sequence is not None
        expected = state.last_sequence + 1
        if event.sequence != expected:
            state.uncertain = True
            reason = (
                "sequence gap"
                if event.sequence > expected
                else "out-of-order sequence outside duplicate history"
            )
            return SequenceObservation(SequenceDisposition.UNCERTAIN, reason)

        self._remember(state, event.sequence, fingerprint)
        state.last_sequence = event.sequence
        return SequenceObservation(SequenceDisposition.ACCEPTED, "continuous sequence")

    def reset(self, snapshot: OrderBookSnapshotEvent) -> None:
        """Establish a new baseline after the caller has accepted a replacement snapshot."""

        self._set_baseline(snapshot)

    def invalidate(self, connection_id: str, sid: int) -> None:
        """Latch one scope uncertain until an explicit snapshot reset."""

        key = (connection_id, sid)
        state = self._states.get(key)
        if state is None:
            self._states[key] = _SequenceState(
                last_sequence=None,
                fingerprints=OrderedDict(),
                uncertain=True,
            )
        else:
            state.uncertain = True

    def drop(self, connection_id: str, sid: int) -> None:
        """Forget a closed connection/subscription scope."""

        self._states.pop((connection_id, sid), None)

    def _remember(self, state: _SequenceState, sequence: int, fingerprint: str) -> None:
        state.fingerprints[sequence] = fingerprint
        while len(state.fingerprints) > self._fingerprint_history:
            state.fingerprints.popitem(last=False)

    def _set_baseline(self, event: MarketDataEvent) -> None:
        fingerprint = market_data_fingerprint(event)
        self._states[self.key_for(event)] = _SequenceState(
            last_sequence=event.sequence,
            fingerprints=OrderedDict(((event.sequence, fingerprint),)),
        )


class StateAction(StrEnum):
    """Observable outcome of a registry or event-state operation."""

    SUBSCRIPTION_REGISTERED = "subscription_registered"
    SNAPSHOT_STAGED = "snapshot_staged"
    SNAPSHOT_APPLIED = "snapshot_applied"
    DELTA_APPLIED = "delta_applied"
    DUPLICATE_IGNORED = "duplicate_ignored"
    EVENT_IGNORED = "event_ignored"
    RESYNC_REQUIRED = "resync_required"
    RESYNC_COMPLETE = "resync_complete"
    DISCONNECTED = "disconnected"
    BOOK_STALE = "book_stale"
    MARKET_METADATA_REPLACED = "market_metadata_replaced"
    FEE_METADATA_UNCERTAIN = "fee_metadata_uncertain"
    FEE_METADATA_REPLACED = "fee_metadata_replaced"
    RUN_STARTED = "run_started"
    RUN_ENDED = "run_ended"


@dataclass(frozen=True, slots=True)
class StateUpdate:
    """A deterministic state transition suitable for structured logging and tests."""

    action: StateAction
    affected_tickers: tuple[str, ...]
    reason: str


class SubscriptionStateError(ValueError):
    """Raised when an event cannot be mapped to one registered subscription safely."""


class MetadataStateError(ValueError):
    """Raised when metadata catalogs cannot support a fail-closed scanner decision."""


@dataclass(frozen=True, slots=True)
class FeeMetadataUpdate:
    """One explicit event-fee certainty transition and its trusted-component impact."""

    event_ticker: str
    affected_tickers: tuple[str, ...]
    affected_component_ids: tuple[str, ...]
    is_certain: bool
    reason: str


@dataclass(slots=True)
class _SubscriptionState:
    members: frozenset[str]
    pending_snapshots: set[str]
    candidate_books: dict[str, OrderBook] = field(default_factory=dict)
    candidate_snapshot_ids: dict[str, str] = field(default_factory=dict)
    full_resync: bool = True
    recovery_started: bool = False


def _unique_catalog[CatalogItem](
    items: Iterable[CatalogItem],
    *,
    key: Callable[[CatalogItem], str],
    kind: str,
) -> dict[str, CatalogItem]:
    """Materialize one caller iterable exactly once and reject ambiguous identifiers."""

    catalog: dict[str, CatalogItem] = {}
    for item in items:
        identifier = key(item)
        if identifier in catalog:
            raise MetadataStateError(f"duplicate {kind} identifier: {identifier}")
        catalog[identifier] = item
    return catalog


def _validate_catalog_ancestry(
    *,
    markets: Mapping[str, Market],
    events: Mapping[str, Event],
    series: Mapping[str, Series],
    relations: Mapping[str, Relation],
) -> None:
    """Validate one complete metadata snapshot before exposing or mutating it."""

    children_by_event = {event_ticker: set[str]() for event_ticker in events}
    for market in markets.values():
        event = events.get(market.event_ticker)
        if event is None:
            raise MetadataStateError(
                f"market {market.ticker} references unknown event {market.event_ticker}"
            )
        children_by_event[event.ticker].add(market.ticker)
        if market.series_ticker is not None and market.series_ticker not in series:
            raise MetadataStateError(
                f"market {market.ticker} references unknown series {market.series_ticker}"
            )
        if (
            market.series_ticker is not None
            and event.series_ticker is not None
            and market.series_ticker != event.series_ticker
        ):
            raise MetadataStateError(
                f"market {market.ticker} and event {event.ticker} disagree on series ancestry"
            )

    for event in events.values():
        if event.series_ticker is not None and event.series_ticker not in series:
            raise MetadataStateError(
                f"event {event.ticker} references unknown series {event.series_ticker}"
            )
        declared = event.market_tickers
        if any(not ticker.strip() for ticker in declared):
            raise MetadataStateError(f"event {event.ticker} contains a blank market ticker")
        if len(set(declared)) != len(declared):
            raise MetadataStateError(f"event {event.ticker} contains duplicate market tickers")
        actual = children_by_event[event.ticker]
        if set(declared) != actual:
            raise MetadataStateError(
                f"event {event.ticker} market membership does not match loaded market ancestry"
            )
        for change in event.fee_changes:
            if change.event_ticker != event.ticker:
                raise MetadataStateError(
                    f"event {event.ticker} contains a fee change for another event"
                )
            if change.series_ticker not in series:
                raise MetadataStateError(
                    f"event {event.ticker} fee change references unknown series "
                    f"{change.series_ticker}"
                )
            parent_series = {
                candidate
                for candidate in (
                    event.series_ticker,
                    *(markets[ticker].series_ticker for ticker in actual),
                )
                if candidate is not None
            }
            if len(parent_series) != 1 or change.series_ticker not in parent_series:
                raise MetadataStateError(
                    f"event {event.ticker} fee change disagrees with series ancestry"
                )

    known_markets = set(markets)
    for relation in relations.values():
        unknown = set(relation.market_tickers) - known_markets
        if unknown:
            raise MetadataStateError(
                f"relation {relation.relation_id} references unknown markets: "
                f"{', '.join(sorted(unknown))}"
            )


class EngineState:
    """Own immutable scanner metadata plus fail-closed live order-book state."""

    def __init__(
        self,
        *,
        sequence_tracker: SequenceTracker | None = None,
        markets: Iterable[Market] = (),
        events: Iterable[Event] = (),
        series: Iterable[Series] = (),
        relations: Iterable[Relation] = (),
        max_component_markets: int = 12,
    ) -> None:
        if (
            isinstance(max_component_markets, bool)
            or not isinstance(max_component_markets, int)
            or max_component_markets < 1
        ):
            raise MetadataStateError("max_component_markets must be a positive integer")

        self._markets = _unique_catalog(markets, key=lambda item: item.ticker, kind="market")
        self._events = _unique_catalog(events, key=lambda item: item.ticker, kind="event")
        self._series = _unique_catalog(series, key=lambda item: item.ticker, kind="series")
        self._relations = _unique_catalog(
            relations,
            key=lambda item: item.relation_id,
            kind="relation",
        )
        _validate_catalog_ancestry(
            markets=self._markets,
            events=self._events,
            series=self._series,
            relations=self._relations,
        )

        self._graph = RelationGraph(self._relations.values())
        world_cache = WorldCache()
        self._world_sets = {
            component.component_id: generate_component_worlds(
                component,
                max_markets=max_component_markets,
                cache=world_cache,
            )
            for component in self._graph.components
        }
        self._subscription_tickers = tuple(
            sorted(
                {
                    ticker
                    for component in self._graph.components
                    for ticker in component.market_tickers
                }
            )
        )
        self._event_fee_uncertainties: dict[str, str] = {}
        self.sequence_tracker = sequence_tracker or SequenceTracker()
        self._subscriptions: dict[SequenceKey, _SubscriptionState] = {}
        self._ticker_subscriptions: dict[str, SequenceKey] = {}
        self._books: dict[str, OrderBook] = {}
        self._snapshot_ids: dict[str, str] = {}

    @property
    def markets(self) -> Mapping[str, Market]:
        """Return normalized markets keyed by unique ticker without permitting mutation."""

        return MappingProxyType(self._markets)

    @property
    def events(self) -> Mapping[str, Event]:
        """Return complete normalized events keyed by unique ticker."""

        return MappingProxyType(self._events)

    @property
    def series(self) -> Mapping[str, Series]:
        """Return normalized series keyed by unique ticker."""

        return MappingProxyType(self._series)

    @property
    def relations(self) -> Mapping[str, Relation]:
        """Return every sourced relation; only verified entries participate in ``graph``."""

        return MappingProxyType(self._relations)

    @property
    def graph(self) -> RelationGraph:
        """Return the verified-only logical connectivity index."""

        return self._graph

    @property
    def world_sets(self) -> Mapping[str, WorldGenerationResult]:
        """Return bounded world-generation outcomes keyed by trusted component ID."""

        return MappingProxyType(self._world_sets)

    @property
    def subscription_tickers(self) -> tuple[str, ...]:
        """Return the deterministic union of markets in verified relation components."""

        return self._subscription_tickers

    @property
    def books(self) -> Mapping[str, OrderBook]:
        """Expose an immutable view; individual OrderBook values are frozen models."""

        return MappingProxyType(self._books)

    @property
    def orderbooks(self) -> Mapping[str, OrderBook]:
        """Alias ``books`` using the architecture specification's scanner terminology."""

        return MappingProxyType(self._books)

    @property
    def event_fee_uncertainties(self) -> Mapping[str, str]:
        """Return event fee refresh latches that prohibit net-ready fee use."""

        return MappingProxyType(self._event_fee_uncertainties)

    def components_for_markets(self, tickers: Iterable[str]) -> tuple[LogicalComponent, ...]:
        """Return each affected trusted component once in deterministic ID order."""

        if isinstance(tickers, (str, bytes)):
            raise MetadataStateError("component lookup requires an iterable of market tickers")
        values = tuple(tickers)
        if any(not isinstance(ticker, str) or not ticker.strip() for ticker in values):
            raise MetadataStateError("component lookup tickers must be nonblank strings")
        return tuple(self._graph.components_for_markets(set(values)))

    def market_research_contexts(
        self,
        tickers: Iterable[str],
    ) -> tuple[MarketResearchContext, ...]:
        """Snapshot stable research dimensions from the current normalized catalog."""

        if isinstance(tickers, (str, bytes)):
            raise MetadataStateError("research context lookup requires market tickers")
        values = tuple(tickers)
        if any(not isinstance(ticker, str) or not ticker.strip() for ticker in values):
            raise MetadataStateError("research context tickers must be nonblank strings")
        canonical_tickers = tuple(sorted(set(values)))
        unknown = tuple(ticker for ticker in canonical_tickers if ticker not in self._markets)
        if unknown:
            raise MetadataStateError(
                f"research context references unknown markets: {', '.join(unknown)}"
            )

        contexts: list[MarketResearchContext] = []
        for ticker in canonical_tickers:
            market = self._markets[ticker]
            event = self._events[market.event_ticker]
            series_ticker = market.series_ticker or event.series_ticker
            parent_series = None if series_ticker is None else self._series[series_ticker]
            event_category = (
                event.category if event.category is not None and event.category.strip() else None
            )
            series_category = (
                parent_series.category
                if parent_series is not None
                and parent_series.category is not None
                and parent_series.category.strip()
                else None
            )
            settlement_at = next(
                (
                    value
                    for value in (
                        market.settlement_ts,
                        market.expected_expiration_time,
                        market.expiration_time,
                        market.latest_expiration_time,
                        market.close_time,
                    )
                    if value is not None
                ),
                None,
            )
            contexts.append(
                MarketResearchContext(
                    ticker=ticker,
                    event_ticker=market.event_ticker,
                    category=event_category or series_category,
                    settlement_at=settlement_at,
                )
            )
        return tuple(contexts)

    def world_for_component(
        self,
        component: LogicalComponent | str,
    ) -> WorldGenerationResult:
        """Return the cached bounded world outcome for one trusted component."""

        component_id = component if isinstance(component, str) else component.component_id
        try:
            return self._world_sets[component_id]
        except KeyError:
            raise MetadataStateError("unknown trusted component") from None

    def replace_market(self, market: Market) -> Market:
        """Atomically replace lifecycle metadata without permitting ancestry changes."""

        current = self._markets.get(market.ticker)
        if current is None:
            raise MetadataStateError("cannot replace an unknown market")
        if (
            market.event_ticker != current.event_ticker
            or market.series_ticker != current.series_ticker
        ):
            raise MetadataStateError("market replacement cannot change metadata ancestry")
        candidate = dict(self._markets)
        candidate[market.ticker] = market
        _validate_catalog_ancestry(
            markets=candidate,
            events=self._events,
            series=self._series,
            relations=self._relations,
        )
        self._markets[market.ticker] = market
        return market

    def mark_event_fee_uncertain(
        self,
        event_ticker: str,
        *,
        reason: str = "event fee metadata refresh pending",
    ) -> FeeMetadataUpdate:
        """Latch one known event non-net-ready until a complete replacement succeeds."""

        if event_ticker not in self._events:
            raise MetadataStateError("cannot invalidate fee metadata for an unknown event")
        if not isinstance(reason, str) or not reason.strip():
            raise MetadataStateError("event fee uncertainty reason cannot be blank")
        self._event_fee_uncertainties[event_ticker] = reason
        return self._fee_metadata_update(
            event_ticker=event_ticker,
            is_certain=False,
            reason=reason,
        )

    def replace_event(self, event: Event) -> FeeMetadataUpdate:
        """Replace one complete event and clear its fee uncertainty only after validation."""

        if event.ticker not in self._events:
            raise MetadataStateError("cannot replace an unknown event")
        candidate = dict(self._events)
        candidate[event.ticker] = event
        _validate_catalog_ancestry(
            markets=self._markets,
            events=candidate,
            series=self._series,
            relations=self._relations,
        )
        self._events[event.ticker] = event
        self._event_fee_uncertainties.pop(event.ticker, None)
        return self._fee_metadata_update(
            event_ticker=event.ticker,
            is_certain=True,
            reason="complete event fee metadata replaced",
        )

    def is_fee_ready_for_market(self, ticker: str) -> bool:
        """Whether complete, currently certain event/series inputs exist for net fees."""

        try:
            self.fee_metadata_for_market(ticker)
        except MetadataStateError:
            return False
        return True

    def fee_metadata_for_market(self, ticker: str) -> tuple[Event, Series]:
        """Return fee inputs only when event metadata is complete and not refresh-pending."""

        market = self._markets.get(ticker)
        if market is None:
            raise MetadataStateError("unknown market")
        event = self._events[market.event_ticker]
        if event.ticker in self._event_fee_uncertainties:
            raise MetadataStateError("event fee metadata is uncertain")
        series_ticker = market.series_ticker or event.series_ticker
        if series_ticker is None:
            raise MetadataStateError("market has no series fee metadata")
        try:
            parent_series = self._series[series_ticker]
        except KeyError:
            raise MetadataStateError("market series fee metadata is missing") from None
        return event, parent_series

    def _fee_metadata_update(
        self,
        *,
        event_ticker: str,
        is_certain: bool,
        reason: str,
    ) -> FeeMetadataUpdate:
        affected = tuple(
            sorted(
                market.ticker
                for market in self._markets.values()
                if market.event_ticker == event_ticker
            )
        )
        component_ids = tuple(
            component.component_id for component in self.components_for_markets(affected)
        )
        return FeeMetadataUpdate(
            event_ticker=event_ticker,
            affected_tickers=affected,
            affected_component_ids=component_ids,
            is_certain=is_certain,
            reason=reason,
        )

    def get_book(self, ticker: str) -> OrderBook | None:
        return self._books.get(ticker)

    def is_solve_ready(self, ticker: str) -> bool:
        book = self._books.get(ticker)
        return book is not None and book.status is BookStatus.FRESH

    def register_subscription(
        self,
        *,
        connection_id: str,
        sid: int,
        tickers: Iterable[str],
    ) -> StateUpdate:
        """Register the complete membership before accepting any order-book event."""

        if not isinstance(connection_id, str) or not connection_id or not connection_id.strip():
            raise SubscriptionStateError("connection_id cannot be blank")
        if isinstance(sid, bool) or not isinstance(sid, int) or sid < 1:
            raise SubscriptionStateError("sid must be positive")
        if isinstance(tickers, (str, bytes)):
            raise SubscriptionStateError("subscription tickers must be an iterable of identifiers")
        ticker_values = tuple(tickers)
        if not ticker_values or any(
            not isinstance(ticker, str) or not ticker or not ticker.strip()
            for ticker in ticker_values
        ):
            raise SubscriptionStateError("subscription tickers must be nonblank")
        members = frozenset(ticker_values)
        if len(members) != len(ticker_values):
            raise SubscriptionStateError("subscription tickers must be unique")

        key = (connection_id, sid)
        existing = self._subscriptions.get(key)
        if existing is not None:
            if existing.members != members:
                raise SubscriptionStateError("a registered subscription cannot change membership")
            return StateUpdate(
                StateAction.SUBSCRIPTION_REGISTERED,
                tuple(sorted(members)),
                "subscription was already registered",
            )
        overlaps = tuple(
            sorted(ticker for ticker in members if ticker in self._ticker_subscriptions)
        )
        if overlaps:
            raise SubscriptionStateError(
                f"active order-book subscription membership overlaps: {', '.join(overlaps)}"
            )

        self._subscriptions[key] = _SubscriptionState(
            members=members,
            pending_snapshots=set(members),
        )
        for ticker in members:
            self._ticker_subscriptions[ticker] = key
            self._mark_existing_book_resync(ticker, "awaiting initial subscription snapshot")
        self.sequence_tracker.invalidate(connection_id, sid)
        return StateUpdate(
            StateAction.SUBSCRIPTION_REGISTERED,
            tuple(sorted(members)),
            "awaiting initial snapshots for every subscription member",
        )

    def subscription_members(self, *, connection_id: str, sid: int) -> frozenset[str]:
        return self._subscription(connection_id, sid).members

    def connection_members(self, connection_id: str) -> tuple[str, ...]:
        """Return the canonical union currently registered to one connection."""

        return tuple(
            sorted(
                ticker
                for (registered_connection, _), subscription in self._subscriptions.items()
                if registered_connection == connection_id
                for ticker in subscription.members
            )
        )

    def pending_snapshots(self, *, connection_id: str, sid: int) -> frozenset[str]:
        return frozenset(self._subscription(connection_id, sid).pending_snapshots)

    def apply_event(self, event: MarketDataEvent) -> StateUpdate:
        """Apply one normalized event or return a fail-closed ignored/resync transition."""

        key = (event.connection_id, event.sid)
        subscription = self._subscriptions.get(key)
        if subscription is None:
            raise SubscriptionStateError("event belongs to an unknown subscription")
        if event.ticker not in subscription.members:
            return self._start_full_resync(
                key,
                subscription,
                "event ticker is not a registered subscription member",
            )

        if subscription.full_resync:
            return self._apply_during_full_resync(key, subscription, event)

        observation = self.sequence_tracker.observe(event)
        if observation.disposition is SequenceDisposition.DUPLICATE:
            return StateUpdate(
                StateAction.DUPLICATE_IGNORED,
                (event.ticker,),
                observation.reason,
            )
        if observation.disposition is SequenceDisposition.UNCERTAIN:
            return self._start_full_resync(key, subscription, observation.reason)
        if isinstance(event, OrderBookSnapshotEvent):
            return self._apply_snapshot(subscription, event)
        return self._apply_delta(subscription, event)

    def require_market_resync(self, ticker: str, *, reason: str) -> StateUpdate:
        """Invalidate one market after a local rule/depth/metadata uncertainty."""

        key = self._ticker_subscriptions.get(ticker)
        if key is None:
            raise SubscriptionStateError("ticker has no active order-book subscription")
        return self.require_resync(
            connection_id=key[0],
            sid=key[1],
            tickers=(ticker,),
            reason=reason,
        )

    def require_resync(
        self,
        *,
        connection_id: str,
        sid: int,
        tickers: Iterable[str],
        reason: str,
    ) -> StateUpdate:
        """Invalidate selected members, escalating an all-member request subscription-wide."""

        if not isinstance(reason, str) or not reason or not reason.strip():
            raise SubscriptionStateError("resync reason cannot be blank")
        key = (connection_id, sid)
        subscription = self._subscription(connection_id, sid)
        if isinstance(tickers, (str, bytes)):
            raise SubscriptionStateError("resync tickers must be an iterable of identifiers")
        requested_values = tuple(tickers)
        requested = frozenset(requested_values)
        if not requested or len(requested) != len(requested_values):
            raise SubscriptionStateError("resync tickers must be nonempty and unique")
        unknown = requested - subscription.members
        if unknown:
            raise SubscriptionStateError("resync request contains a nonmember ticker")
        if requested == subscription.members:
            return self._start_full_resync(key, subscription, reason)
        for ticker in requested:
            subscription.pending_snapshots.add(ticker)
            subscription.candidate_books.pop(ticker, None)
            subscription.candidate_snapshot_ids.pop(ticker, None)
            self._mark_existing_book_resync(ticker, reason)
        return StateUpdate(
            StateAction.RESYNC_REQUIRED,
            tuple(sorted(requested)),
            reason,
        )

    def disconnect_connection(self, connection_id: str) -> StateUpdate:
        """Invalidate and unregister every book owned by a disconnected connection."""

        keys = tuple(key for key in self._subscriptions if key[0] == connection_id)
        affected: set[str] = set()
        for key in keys:
            subscription = self._subscriptions.pop(key)
            affected.update(subscription.members)
            self.sequence_tracker.drop(*key)
            for ticker in subscription.members:
                self._ticker_subscriptions.pop(ticker, None)
                self._snapshot_ids.pop(ticker, None)
                self._mark_existing_book_resync(ticker, "WebSocket connection disconnected")
        return StateUpdate(
            StateAction.DISCONNECTED,
            tuple(sorted(affected)),
            "connection books invalidated",
        )

    def _subscription(self, connection_id: str, sid: int) -> _SubscriptionState:
        try:
            return self._subscriptions[(connection_id, sid)]
        except KeyError:
            raise SubscriptionStateError("unknown subscription") from None

    def _apply_during_full_resync(
        self,
        key: SequenceKey,
        subscription: _SubscriptionState,
        event: MarketDataEvent,
    ) -> StateUpdate:
        if not subscription.recovery_started:
            if not isinstance(event, OrderBookSnapshotEvent):
                return StateUpdate(
                    StateAction.EVENT_IGNORED,
                    (event.ticker,),
                    "replacement snapshot required before deltas",
                )
            self.sequence_tracker.reset(event)
            subscription.recovery_started = True
            return self._stage_snapshot(subscription, event)

        observation = self.sequence_tracker.observe(event)
        if observation.disposition is SequenceDisposition.DUPLICATE:
            return StateUpdate(
                StateAction.DUPLICATE_IGNORED,
                (event.ticker,),
                observation.reason,
            )
        if observation.disposition is SequenceDisposition.UNCERTAIN:
            return self._start_full_resync(key, subscription, observation.reason)
        if isinstance(event, OrderBookSnapshotEvent):
            return self._stage_snapshot(subscription, event)
        candidate = subscription.candidate_books.get(event.ticker)
        if candidate is None:
            return StateUpdate(
                StateAction.EVENT_IGNORED,
                (event.ticker,),
                "market still awaits its replacement snapshot",
            )
        if subscription.candidate_snapshot_ids.get(event.ticker) != event.snapshot_id:
            return self._mark_market_resync(
                subscription,
                event.ticker,
                "delta snapshot identity does not match the replacement snapshot",
            )
        try:
            updated = self._delta_book(candidate, event)
        except OrderBookUpdateError as exc:
            return self._mark_market_resync(subscription, event.ticker, str(exc))
        subscription.candidate_books[event.ticker] = updated
        self._books[event.ticker] = updated.with_status(
            BookStatus.RESYNC_REQUIRED,
            reason="subscription-wide resynchronization remains incomplete",
        )
        return StateUpdate(
            StateAction.DELTA_APPLIED,
            (event.ticker,),
            "delta retained behind subscription-wide resync barrier",
        )

    def _start_full_resync(
        self,
        key: SequenceKey,
        subscription: _SubscriptionState,
        reason: str,
    ) -> StateUpdate:
        subscription.full_resync = True
        subscription.recovery_started = False
        subscription.pending_snapshots = set(subscription.members)
        subscription.candidate_books.clear()
        subscription.candidate_snapshot_ids.clear()
        self.sequence_tracker.invalidate(*key)
        for ticker in subscription.members:
            self._mark_existing_book_resync(ticker, reason)
        return StateUpdate(
            StateAction.RESYNC_REQUIRED,
            tuple(sorted(subscription.members)),
            reason,
        )

    def _stage_snapshot(
        self,
        subscription: _SubscriptionState,
        event: OrderBookSnapshotEvent,
    ) -> StateUpdate:
        candidate = self._snapshot_book(event)
        subscription.candidate_books[event.ticker] = candidate
        subscription.candidate_snapshot_ids[event.ticker] = event.snapshot_id
        subscription.pending_snapshots.discard(event.ticker)
        if subscription.pending_snapshots:
            self._books[event.ticker] = candidate.with_status(
                BookStatus.RESYNC_REQUIRED,
                reason="subscription-wide resynchronization remains incomplete",
            )
            return StateUpdate(
                StateAction.SNAPSHOT_STAGED,
                (event.ticker,),
                "snapshot staged until every subscription member is replaced",
            )

        if set(subscription.candidate_books) != set(subscription.members):
            raise RuntimeError("complete resync is missing a candidate book")
        for ticker, book in subscription.candidate_books.items():
            self._books[ticker] = book
            self._snapshot_ids[ticker] = subscription.candidate_snapshot_ids[ticker]
        subscription.candidate_books.clear()
        subscription.candidate_snapshot_ids.clear()
        subscription.full_resync = False
        subscription.recovery_started = False
        return StateUpdate(
            StateAction.RESYNC_COMPLETE,
            tuple(sorted(subscription.members)),
            "all subscription snapshots received; books promoted atomically",
        )

    def _apply_snapshot(
        self,
        subscription: _SubscriptionState,
        event: OrderBookSnapshotEvent,
    ) -> StateUpdate:
        was_pending = event.ticker in subscription.pending_snapshots
        self._books[event.ticker] = self._snapshot_book(event)
        self._snapshot_ids[event.ticker] = event.snapshot_id
        subscription.pending_snapshots.discard(event.ticker)
        return StateUpdate(
            StateAction.RESYNC_COMPLETE if was_pending else StateAction.SNAPSHOT_APPLIED,
            (event.ticker,),
            "market snapshot replaced the complete local book",
        )

    def _apply_delta(
        self,
        subscription: _SubscriptionState,
        event: OrderBookDeltaEvent,
    ) -> StateUpdate:
        if event.ticker in subscription.pending_snapshots:
            return StateUpdate(
                StateAction.EVENT_IGNORED,
                (event.ticker,),
                "market awaits a replacement snapshot",
            )
        book = self._books.get(event.ticker)
        if book is None:
            return self._mark_market_resync(subscription, event.ticker, "local book is missing")
        if self._snapshot_ids.get(event.ticker) != event.snapshot_id:
            return self._mark_market_resync(
                subscription,
                event.ticker,
                "delta snapshot identity does not match the current book",
            )
        try:
            self._books[event.ticker] = self._delta_book(book, event)
        except OrderBookUpdateError as exc:
            return self._mark_market_resync(subscription, event.ticker, str(exc))
        return StateUpdate(
            StateAction.DELTA_APPLIED,
            (event.ticker,),
            "market delta applied",
        )

    def _mark_market_resync(
        self,
        subscription: _SubscriptionState,
        ticker: str,
        reason: str,
    ) -> StateUpdate:
        subscription.pending_snapshots.add(ticker)
        subscription.candidate_books.pop(ticker, None)
        subscription.candidate_snapshot_ids.pop(ticker, None)
        self._mark_existing_book_resync(ticker, reason)
        return StateUpdate(StateAction.RESYNC_REQUIRED, (ticker,), reason)

    def _mark_existing_book_resync(self, ticker: str, reason: str) -> None:
        book = self._books.get(ticker)
        if book is not None:
            self._books[ticker] = book.with_status(BookStatus.RESYNC_REQUIRED, reason=reason)

    @staticmethod
    def _snapshot_book(event: OrderBookSnapshotEvent) -> OrderBook:
        return OrderBook(
            ticker=event.ticker,
            yes_bids=event.yes_bids,
            no_bids=event.no_bids,
            sequence=event.sequence,
            exchange_timestamp=event.exchange_ts,
            local_timestamp=event.local_received_ts,
            status=BookStatus.FRESH,
            raw=None,
        )

    @staticmethod
    def _delta_book(book: OrderBook, event: OrderBookDeltaEvent) -> OrderBook:
        return book.apply_delta(
            side=event.side,
            price=event.price,
            quantity_delta=event.quantity_delta,
            sequence=event.sequence,
            local_timestamp=event.local_received_ts,
            exchange_timestamp=event.exchange_ts,
        )
