"""Isolated sequence policy and fail-closed subscription book-state tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from arbiter.engine.state import (
    EngineState,
    SequenceDisposition,
    SequenceTracker,
    StateAction,
    SubscriptionStateError,
    market_data_fingerprint,
)
from arbiter.models.orderbook import BookStatus, PriceLevel
from arbiter.replay.events import OrderBookDeltaEvent, OrderBookSnapshotEvent

NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)


def _snapshot(
    ticker: str,
    sequence: int,
    *,
    sid: int = 7,
    connection_id: str = "connection-1",
    event_index: int | None = None,
    quantity: str = "1",
) -> OrderBookSnapshotEvent:
    return OrderBookSnapshotEvent(
        event_index=sequence if event_index is None else event_index,
        local_received_ts=NOW + timedelta(milliseconds=sequence),
        exchange_ts=NOW,
        ticker=ticker,
        sequence=sequence,
        sid=sid,
        connection_id=connection_id,
        snapshot_id=f"{connection_id}:{sid}:snapshot",
        yes_bids=(PriceLevel(price=Decimal("0.50"), quantity=Decimal(quantity)),),
        no_bids=(PriceLevel(price=Decimal("0.40"), quantity=Decimal("2")),),
    )


def _delta(
    ticker: str,
    sequence: int,
    *,
    quantity_delta: str = "1",
    sid: int = 7,
    connection_id: str = "connection-1",
    event_index: int | None = None,
    local_received_ts: datetime | None = None,
) -> OrderBookDeltaEvent:
    return OrderBookDeltaEvent(
        event_index=sequence if event_index is None else event_index,
        local_received_ts=(
            NOW + timedelta(milliseconds=sequence)
            if local_received_ts is None
            else local_received_ts
        ),
        exchange_ts=NOW,
        ticker=ticker,
        sequence=sequence,
        sid=sid,
        connection_id=connection_id,
        snapshot_id=f"{connection_id}:{sid}:snapshot",
        side="yes",
        price=Decimal("0.50"),
        quantity_delta=Decimal(quantity_delta),
    )


def _registered_state(*tickers: str) -> EngineState:
    state = EngineState()
    state.register_subscription(connection_id="connection-1", sid=7, tickers=tickers)
    return state


def _synchronized_state() -> EngineState:
    state = _registered_state("A", "B")
    assert state.apply_event(_snapshot("A", 1)).action is StateAction.SNAPSHOT_STAGED
    assert state.apply_event(_snapshot("B", 2)).action is StateAction.RESYNC_COMPLETE
    return state


def test_sequence_scope_is_isolated_per_connection_and_subscription() -> None:
    tracker = SequenceTracker()

    assert tracker.observe(_snapshot("A", 10)).disposition is SequenceDisposition.ACCEPTED
    assert tracker.observe(_snapshot("B", 10, sid=8)).disposition is SequenceDisposition.ACCEPTED
    assert (
        tracker.observe(_snapshot("C", 10, connection_id="connection-2")).disposition
        is SequenceDisposition.ACCEPTED
    )
    assert tracker.observe(_delta("A", 11)).disposition is SequenceDisposition.ACCEPTED
    assert tracker.scope_description == "per (connection_id, sid)"


def test_exact_duplicate_ignores_local_receipt_identity_but_conflict_latches_uncertain() -> None:
    tracker = SequenceTracker()
    original = _delta("A", 20, event_index=20)
    duplicate = original.model_copy(
        update={
            "event_index": 99,
            "local_received_ts": NOW + timedelta(seconds=10),
        }
    )
    conflict = original.model_copy(update={"quantity_delta": Decimal("2")})

    assert market_data_fingerprint(original) == market_data_fingerprint(duplicate)
    assert tracker.observe(original).disposition is SequenceDisposition.ACCEPTED
    assert tracker.observe(duplicate).disposition is SequenceDisposition.DUPLICATE
    assert tracker.observe(conflict).disposition is SequenceDisposition.UNCERTAIN
    assert tracker.observe(_delta("A", 21)).disposition is SequenceDisposition.UNCERTAIN


@pytest.mark.parametrize("sequence", [9, 12])
def test_gap_or_out_of_order_sequence_fails_closed(sequence: int) -> None:
    tracker = SequenceTracker(fingerprint_history=1)
    tracker.observe(_snapshot("A", 10))

    observation = tracker.observe(_delta("A", sequence))

    assert observation.disposition is SequenceDisposition.UNCERTAIN
    assert "gap" in observation.reason or "out-of-order" in observation.reason


def test_initial_snapshots_are_staged_and_promoted_only_when_all_members_arrive() -> None:
    state = _registered_state("A", "B")

    first = state.apply_event(_snapshot("A", 1))
    between = state.apply_event(_delta("A", 2, quantity_delta="1"))

    assert first.action is StateAction.SNAPSHOT_STAGED
    assert between.action is StateAction.DELTA_APPLIED
    assert state.get_book("A") is not None
    assert state.get_book("A").status is BookStatus.RESYNC_REQUIRED  # type: ignore[union-attr]
    assert not state.is_solve_ready("A")
    assert state.get_book("B") is None
    assert state.pending_snapshots(connection_id="connection-1", sid=7) == {"B"}

    completed = state.apply_event(_snapshot("B", 3))

    assert completed.action is StateAction.RESYNC_COMPLETE
    assert state.get_book("A") is not None
    assert state.get_book("A").yes_bids[0].quantity == 2  # type: ignore[union-attr]
    assert all(state.is_solve_ready(ticker) for ticker in ("A", "B"))


def test_sequence_gap_invalidates_every_member_and_ignores_triggering_delta() -> None:
    state = _synchronized_state()
    before = state.get_book("A")
    assert before is not None

    update = state.apply_event(_delta("A", 4, quantity_delta="5"))

    assert update.action is StateAction.RESYNC_REQUIRED
    assert update.affected_tickers == ("A", "B")
    assert state.get_book("A") is not None
    assert state.get_book("A").yes_bids == before.yes_bids  # type: ignore[union-attr]
    assert all(not state.is_solve_ready(ticker) for ticker in ("A", "B"))
    assert state.pending_snapshots(connection_id="connection-1", sid=7) == {"A", "B"}

    assert state.apply_event(_delta("A", 5)).action is StateAction.EVENT_IGNORED
    assert state.apply_event(_snapshot("A", 10)).action is StateAction.SNAPSHOT_STAGED
    assert not state.is_solve_ready("A")
    assert state.apply_event(_snapshot("B", 11)).action is StateAction.RESYNC_COMPLETE
    assert all(state.is_solve_ready(ticker) for ticker in ("A", "B"))


def test_exact_duplicate_delta_is_idempotent_but_conflict_resyncs_all_members() -> None:
    state = _synchronized_state()
    applied = _delta("A", 3)
    duplicate = applied.model_copy(
        update={"event_index": 30, "local_received_ts": NOW + timedelta(seconds=2)}
    )

    assert state.apply_event(applied).action is StateAction.DELTA_APPLIED
    assert state.apply_event(duplicate).action is StateAction.DUPLICATE_IGNORED
    assert state.get_book("A") is not None
    assert state.get_book("A").yes_bids[0].quantity == 2  # type: ignore[union-attr]

    conflict = applied.model_copy(update={"quantity_delta": Decimal("2")})
    update = state.apply_event(conflict)

    assert update.action is StateAction.RESYNC_REQUIRED
    assert update.affected_tickers == ("A", "B")
    assert all(not state.is_solve_ready(ticker) for ticker in ("A", "B"))


def test_negative_depth_is_market_local_and_replacement_snapshot_recovers_it() -> None:
    state = _synchronized_state()

    update = state.apply_event(_delta("A", 3, quantity_delta="-2"))

    assert update.action is StateAction.RESYNC_REQUIRED
    assert update.affected_tickers == ("A",)
    assert not state.is_solve_ready("A")
    assert state.is_solve_ready("B")
    assert state.get_book("A") is not None
    assert state.get_book("A").yes_bids[0].quantity == 1  # type: ignore[union-attr]

    recovered = state.apply_event(_snapshot("A", 4, quantity="3"))

    assert recovered.action is StateAction.RESYNC_COMPLETE
    assert state.is_solve_ready("A")
    assert state.get_book("A") is not None
    assert state.get_book("A").yes_bids[0].quantity == 3  # type: ignore[union-attr]


def test_delta_from_a_different_snapshot_is_ignored_and_marks_only_that_market() -> None:
    state = _synchronized_state()
    wrong_snapshot = _delta("A", 3).model_copy(update={"snapshot_id": "old-snapshot"})

    update = state.apply_event(wrong_snapshot)

    assert update.action is StateAction.RESYNC_REQUIRED
    assert update.affected_tickers == ("A",)
    assert not state.is_solve_ready("A")
    assert state.is_solve_ready("B")
    assert state.get_book("A") is not None
    assert state.get_book("A").yes_bids[0].quantity == 1  # type: ignore[union-attr]


def test_explicit_resync_supports_market_local_and_subscription_wide_invalidation() -> None:
    state = _synchronized_state()

    local = state.require_resync(
        connection_id="connection-1",
        sid=7,
        tickers=("A",),
        reason="market metadata changed",
    )

    assert local.action is StateAction.RESYNC_REQUIRED
    assert local.affected_tickers == ("A",)
    assert not state.is_solve_ready("A") and state.is_solve_ready("B")
    assert state.apply_event(_snapshot("A", 3)).action is StateAction.RESYNC_COMPLETE

    global_update = state.require_resync(
        connection_id="connection-1",
        sid=7,
        tickers=("B", "A"),
        reason="malformed subscription message",
    )

    assert global_update.affected_tickers == ("A", "B")
    assert all(not state.is_solve_ready(ticker) for ticker in ("A", "B"))


def test_missing_initial_book_ignores_delta_until_snapshot() -> None:
    state = _registered_state("A")

    ignored = state.apply_event(_delta("A", 1))

    assert ignored.action is StateAction.EVENT_IGNORED
    assert state.get_book("A") is None
    assert state.pending_snapshots(connection_id="connection-1", sid=7) == {"A"}


def test_unexpected_ticker_causes_subscription_wide_resync() -> None:
    state = _synchronized_state()

    update = state.apply_event(_delta("UNREGISTERED", 3))

    assert update.action is StateAction.RESYNC_REQUIRED
    assert update.affected_tickers == ("A", "B")
    assert all(not state.is_solve_ready(ticker) for ticker in ("A", "B"))


def test_disconnect_invalidates_and_unregisters_every_member() -> None:
    state = _synchronized_state()

    update = state.disconnect_connection("connection-1")

    assert update.action is StateAction.DISCONNECTED
    assert update.affected_tickers == ("A", "B")
    assert all(not state.is_solve_ready(ticker) for ticker in ("A", "B"))
    with pytest.raises(SubscriptionStateError, match="unknown subscription"):
        state.subscription_members(connection_id="connection-1", sid=7)
    with pytest.raises(SubscriptionStateError, match="unknown subscription"):
        state.apply_event(_snapshot("A", 3))


def test_registry_rejects_unknown_membership_changes_and_active_overlap() -> None:
    state = _registered_state("A", "B")

    with pytest.raises(SubscriptionStateError, match="cannot change membership"):
        state.register_subscription(connection_id="connection-1", sid=7, tickers=("A",))
    with pytest.raises(SubscriptionStateError, match="overlaps"):
        state.register_subscription(connection_id="connection-1", sid=8, tickers=("B", "C"))
    with pytest.raises(SubscriptionStateError, match="unknown subscription"):
        state.apply_event(_snapshot("Z", 1, sid=99))
