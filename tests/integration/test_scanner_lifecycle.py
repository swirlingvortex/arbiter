"""Event-time scheduler and deterministic opportunity episode integration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from arbiter.engine.scanner import (
    ArbitrageEngine,
    ComponentScanner,
    ComponentScanResult,
    OpportunityLifecycle,
    ScanStatus,
)
from arbiter.engine.state import EngineState, StateAction
from arbiter.models.event import Event
from arbiter.models.market import Market, PriceRange
from arbiter.models.opportunity import (
    OpportunityObservation,
    OpportunityTransition,
    opportunity_observation_id,
)
from arbiter.models.orderbook import PriceLevel
from arbiter.models.relation import Relation, RelationType
from arbiter.models.series import Series
from arbiter.replay.events import (
    BookStaleEvent,
    ConnectionInterruptedEvent,
    FeeRefreshAppliedEvent,
    FeeRefreshStartedEvent,
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
    RunEndedEvent,
    RunInputPayload,
    RunStartedEvent,
)
from arbiter.solver.fees import ZeroFeeModel

NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)


class _ObservationStore:
    def __init__(self) -> None:
        self.observations: list[OpportunityObservation] = []
        self.fail = False
        self.fail_component_id: str | None = None

    def persist_transition(self, observation: OpportunityObservation) -> None:
        if self.fail or observation.component_id == self.fail_component_id:
            raise RuntimeError("synthetic persistent storage failure")
        self.observations.append(observation)


def _state() -> EngineState:
    price_range = (
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
            price_ranges=price_range,
            raw={},
        )
        for ticker in ("A", "M")
    )
    event = Event(
        ticker="EVENT",
        series_ticker="SERIES",
        title="Event",
        market_tickers=("A", "M"),
        raw={},
    )
    series = Series(
        ticker="SERIES",
        title="Series",
        fee_type="quadratic",
        fee_multiplier=Decimal("1"),
        raw={},
    )
    relation = Relation(
        relation_id="m-implies-a",
        market_tickers=("M", "A"),
        relation_type=RelationType.IMPLIES,
        source="manual",
        verified=True,
        rationale="fixture",
        created_at=NOW,
        antecedent="M",
        consequent="A",
    )
    state = EngineState(
        markets=markets,
        events=(event,),
        series=(series,),
        relations=(relation,),
    )
    state.register_subscription(
        connection_id="connection-1",
        sid=7,
        tickers=("A", "M"),
    )
    return state


def _snapshot(
    ticker: str,
    *,
    event_index: int,
    sequence: int,
    yes_bid: str,
    no_bid: str,
    received_at: datetime = NOW,
) -> OrderBookSnapshotEvent:
    return OrderBookSnapshotEvent(
        event_index=event_index,
        local_received_ts=received_at,
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
    return OrderBookDeltaEvent(
        event_index=event_index,
        local_received_ts=received_at,
        exchange_ts=received_at,
        ticker=ticker,
        sequence=sequence,
        sid=7,
        connection_id="connection-1",
        snapshot_id=f"snapshot:{ticker}",
        side=side,  # type: ignore[arg-type]
        price=Decimal(price),
        quantity_delta=Decimal(quantity_delta),
    )


def _engine(
    state: EngineState,
    store: _ObservationStore,
    *,
    run_id: str = "run-1",
    debounce: timedelta = timedelta(milliseconds=25),
) -> tuple[ArbitrageEngine, OpportunityLifecycle]:
    scanner = ComponentScanner(
        state,
        stale_after=timedelta(seconds=5),
        fee_model=ZeroFeeModel(),
        minimum_net_profit=Decimal("0"),
        minimum_net_edge_bps=Decimal("0"),
    )
    lifecycle = OpportunityLifecycle(run_id=run_id, store=store)
    return (
        ArbitrageEngine(
            state=state,
            scanner=scanner,
            lifecycle=lifecycle,
            solve_debounce=debounce,
        ),
        lifecycle,
    )


def _open_canonical(engine: ArbitrageEngine) -> None:
    engine.process_event(
        _snapshot(
            "A",
            event_index=0,
            sequence=1,
            yes_bid="0.60",
            no_bid="0.38",
        )
    )
    engine.process_event(
        _snapshot(
            "M",
            event_index=1,
            sequence=2,
            yes_bid="0.70",
            no_bid="0.28",
        )
    )
    engine.advance_time(NOW + timedelta(milliseconds=25))


def test_synthetic_stream_opens_updates_and_closes_one_event_time_episode() -> None:
    state = _state()
    store = _ObservationStore()
    engine, lifecycle = _engine(state, store)
    _open_canonical(engine)

    opened = store.observations[-1]
    opportunity_id = opened.opportunity_id
    assert opened.transition is OpportunityTransition.OPEN
    assert opened.observed_at == NOW + timedelta(milliseconds=25)
    assert opened.event_index == 1
    assert opportunity_id is not None
    assert opened.opportunity is not None
    assert opened.evidence["capacity"] == opened.opportunity.capital_required

    update_time = NOW + timedelta(seconds=1)
    engine.process_event(
        _delta(
            "A",
            event_index=2,
            sequence=3,
            side="no",
            price="0.40",
            quantity_delta="5.00",
            received_at=update_time,
        )
    )
    engine.advance_time(update_time + timedelta(milliseconds=25))

    updated = store.observations[-1]
    assert updated.transition is OpportunityTransition.UPDATED
    assert updated.opportunity_id == opportunity_id
    assert updated.portfolio_signature != opened.portfolio_signature
    assert updated.opportunity is not None
    assert updated.evidence["capacity"] == updated.opportunity.capital_required

    close_time = NOW + timedelta(seconds=2)
    engine.process_event(
        _delta(
            "M",
            event_index=3,
            sequence=4,
            side="yes",
            price="0.70",
            quantity_delta="-5.00",
            received_at=close_time,
        )
    )
    engine.advance_time(close_time + timedelta(milliseconds=25))

    closed = store.observations[-1]
    assert [item.transition for item in store.observations] == [
        OpportunityTransition.OPEN,
        OpportunityTransition.UPDATED,
        OpportunityTransition.CLOSED,
    ]
    assert closed.opportunity_id == opportunity_id
    assert closed.observed_at == close_time + timedelta(milliseconds=25)
    assert closed.close_reason == "opportunity no longer present"
    assert "capacity" not in closed.evidence
    assert lifecycle.active_episodes == {}
    assert len(lifecycle.completed_episodes) == 1
    episode = lifecycle.completed_episodes[0]
    assert episode.duration == timedelta(seconds=2)
    assert episode.opened_event_index == 1
    assert episode.closed_event_index == 3


def test_same_timestamp_events_precede_zero_debounce_timer() -> None:
    state = _state()
    store = _ObservationStore()
    engine, _ = _engine(state, store, debounce=timedelta(0))

    first = engine.process_event(
        _snapshot(
            "A",
            event_index=0,
            sequence=1,
            yes_bid="0.60",
            no_bid="0.38",
        )
    )
    second = engine.process_event(
        _snapshot(
            "M",
            event_index=1,
            sequence=2,
            yes_bid="0.70",
            no_bid="0.28",
        )
    )

    assert first.scans_before_event == ()
    assert second.scans_before_event == ()
    assert store.observations == []
    decisions = engine.advance_time(NOW)
    assert len(decisions) == 1
    assert decisions[0].observation.transition is OpportunityTransition.OPEN
    assert decisions[0].observation.event_index == 1


def test_advance_before_preserves_same_timestamp_event_precedence() -> None:
    state = _state()
    store = _ObservationStore()
    engine, _ = _engine(state, store)
    engine.process_event(
        _snapshot(
            "A",
            event_index=0,
            sequence=1,
            yes_bid="0.60",
            no_bid="0.38",
        )
    )
    engine.process_event(
        _snapshot(
            "M",
            event_index=1,
            sequence=2,
            yes_bid="0.70",
            no_bid="0.28",
        )
    )
    due_at = NOW + timedelta(milliseconds=25)

    assert engine.advance_before(due_at) == ()
    assert store.observations == []
    decisions = engine.advance_before(due_at + timedelta(microseconds=1))

    assert len(decisions) == 1
    assert decisions[0].observation.transition is OpportunityTransition.OPEN
    assert decisions[0].observation.observed_at == due_at


def test_recorded_disconnect_closes_an_open_episode_at_the_control_index() -> None:
    state = _state()
    store = _ObservationStore()
    engine, lifecycle = _engine(state, store)
    _open_canonical(engine)
    interrupted_at = NOW + timedelta(seconds=1)

    processed = engine.process_record(
        ConnectionInterruptedEvent(
            event_index=2,
            local_received_ts=interrupted_at,
            connection_id="connection-1",
            tickers=("A", "M"),
            reason="socket closed",
        )
    )

    assert processed.state_update.action is StateAction.DISCONNECTED
    assert processed.scheduled_components == ()
    assert len(processed.scans_after_record) == 1
    closed = processed.scans_after_record[0].observation
    assert closed.transition is OpportunityTransition.CLOSED
    assert closed.event_index == 2
    assert closed.observed_at == interrupted_at
    assert lifecycle.active_episodes == {}
    assert lifecycle.completed_episodes[0].closed_event_index == 2


def test_recorded_control_membership_corruption_fails_before_state_mutation() -> None:
    state = _state()
    store = _ObservationStore()
    engine, _ = _engine(state, store)
    _open_canonical(engine)

    with pytest.raises(ValueError, match="disconnect membership"):
        engine.process_record(
            ConnectionInterruptedEvent(
                event_index=2,
                local_received_ts=NOW + timedelta(seconds=1),
                connection_id="connection-1",
                tickers=("A", "M", "UNKNOWN"),
                reason="corrupt fixture",
            )
        )

    assert state.connection_members("connection-1") == ("A", "M")
    assert state.is_solve_ready("A")
    assert state.is_solve_ready("M")

    with pytest.raises(ValueError, match="unknown subscription ticker"):
        engine.process_record(
            BookStaleEvent(
                event_index=2,
                local_received_ts=NOW + timedelta(seconds=1),
                ticker="UNKNOWN",
                source_event_index=0,
                source_connection_id="connection-1",
                source_sid=7,
                source_snapshot_id="snapshot:UNKNOWN",
                source_sequence=1,
            )
        )


def test_recorded_stale_timer_closes_only_when_its_source_is_current() -> None:
    state = _state()
    store = _ObservationStore()
    engine, lifecycle = _engine(state, store)
    _open_canonical(engine)
    stale_at = NOW + timedelta(seconds=5, microseconds=1)

    processed = engine.process_record(
        BookStaleEvent(
            event_index=2,
            local_received_ts=stale_at,
            ticker="A",
            source_event_index=0,
            source_connection_id="connection-1",
            source_sid=7,
            source_snapshot_id="snapshot:A",
            source_sequence=1,
        )
    )

    assert processed.state_update.action is StateAction.BOOK_STALE
    assert processed.scans_after_record[0].observation.transition is OpportunityTransition.CLOSED
    assert lifecycle.completed_episodes[0].closed_event_index == 2

    state = _state()
    store = _ObservationStore()
    engine, lifecycle = _engine(state, store)
    _open_canonical(engine)
    engine.process_event(
        _delta(
            "A",
            event_index=2,
            sequence=3,
            side="yes",
            price="0.60",
            quantity_delta="1.00",
            received_at=NOW + timedelta(seconds=4),
        )
    )
    superseded = engine.process_record(
        BookStaleEvent(
            event_index=3,
            local_received_ts=stale_at,
            ticker="A",
            source_event_index=0,
            source_connection_id="connection-1",
            source_sid=7,
            source_snapshot_id="snapshot:A",
            source_sequence=1,
        )
    )

    assert superseded.state_update.action is StateAction.EVENT_IGNORED
    assert superseded.scans_after_record == ()
    assert len(lifecycle.active_episodes) == 1


def test_fee_refresh_controls_apply_in_order_and_rescan_current_books() -> None:
    state = _state()
    store = _ObservationStore()
    engine, _ = _engine(state, store)
    _open_canonical(engine)
    refresh_at = NOW + timedelta(seconds=1)

    started = engine.process_record(
        FeeRefreshStartedEvent(
            event_index=2,
            local_received_ts=refresh_at,
            event_ticker="EVENT",
            affected_tickers=("A", "M"),
            connection_id="connection-1",
            sid=8,
        )
    )
    assert started.state_update.action is StateAction.FEE_METADATA_UNCERTAIN
    assert started.scans_after_record[0].observation.opportunity is not None
    assert started.scans_after_record[0].observation.opportunity.reason == "fee_metadata_uncertain"

    applied = engine.process_record(
        FeeRefreshAppliedEvent(
            event_index=3,
            local_received_ts=refresh_at,
            refresh_started_event_index=2,
            event=state.events["EVENT"],
        )
    )
    assert applied.state_update.action is StateAction.FEE_METADATA_REPLACED
    assert len(applied.scheduled_components) == 1
    rescans = engine.advance_time(refresh_at + timedelta(milliseconds=25))
    assert rescans[0].observation.opportunity is not None
    assert rescans[0].observation.opportunity.reason is None


def test_engine_rejects_a_noncontiguous_record_index() -> None:
    state = _state()
    engine, _ = _engine(state, _ObservationStore())
    engine.process_event(
        _snapshot(
            "A",
            event_index=4,
            sequence=1,
            yes_bid="0.60",
            no_bid="0.38",
        )
    )

    with pytest.raises(ValueError, match="contiguous"):
        engine.process_event(
            _snapshot(
                "M",
                event_index=6,
                sequence=2,
                yes_bid="0.70",
                no_bid="0.28",
            )
        )


def test_recorded_run_identity_can_differ_from_replay_persistence_identity() -> None:
    state = _state()
    store = _ObservationStore()
    scanner = ComponentScanner(
        state,
        stale_after=timedelta(seconds=5),
        fee_model=ZeroFeeModel(),
        minimum_net_profit=Decimal("0"),
        minimum_net_edge_bps=Decimal("0"),
    )
    engine = ArbitrageEngine(
        state=state,
        scanner=scanner,
        lifecycle=OpportunityLifecycle(run_id="replay-run", store=store),
        solve_debounce=timedelta(milliseconds=25),
        recorded_run_id="source-run",
    )
    inputs = RunInputPayload.build(
        config={"engine": {"solve_debounce_ms": 25}},
        markets=state.markets.values(),
        events=state.events.values(),
        series=state.series.values(),
        relations=state.relations.values(),
        fee_policy={"version": "fixture"},
    )

    started = engine.process_record(
        RunStartedEvent.build(
            event_index=40,
            local_received_ts=NOW,
            run_id="source-run",
            inputs=inputs,
        )
    )
    ended = engine.process_record(
        RunEndedEvent(
            event_index=41,
            local_received_ts=NOW,
            run_id="source-run",
            status="succeeded",
        )
    )

    assert started.state_update.action is StateAction.RUN_STARTED
    assert ended.state_update.action is StateAction.RUN_ENDED


def test_run_start_rejects_catalog_drift_before_advancing_record_order() -> None:
    state = _state()
    store = _ObservationStore()
    engine, _ = _engine(state, store)
    inputs = RunInputPayload.build(
        config={},
        markets=tuple(state.markets.values())[:-1],
        events=state.events.values(),
        series=state.series.values(),
        relations=state.relations.values(),
        fee_policy={},
    )

    with pytest.raises(ValueError, match="do not match"):
        engine.process_record(
            RunStartedEvent.build(
                event_index=0,
                local_received_ts=NOW,
                run_id="run-1",
                inputs=inputs,
            )
        )

    assert engine.last_event_index is None


def test_storage_failure_leaves_lifecycle_uncommitted_and_timer_retryable() -> None:
    state = _state()
    store = _ObservationStore()
    engine, lifecycle = _engine(state, store)
    engine.process_event(
        _snapshot(
            "A",
            event_index=0,
            sequence=1,
            yes_bid="0.60",
            no_bid="0.38",
        )
    )
    engine.process_event(
        _snapshot(
            "M",
            event_index=1,
            sequence=2,
            yes_bid="0.70",
            no_bid="0.28",
        )
    )
    due_at = engine.next_due_at
    assert due_at is not None
    store.fail = True

    try:
        engine.advance_time(due_at)
    except RuntimeError as exc:
        assert "persistent storage failure" in str(exc)
    else:
        raise AssertionError("storage failure should stop the engine")

    assert lifecycle.active_episodes == {}
    assert lifecycle.observations == ()
    assert engine.next_due_at == due_at
    store.fail = False
    engine.advance_time(due_at)
    assert store.observations[0].transition is OpportunityTransition.OPEN


def test_flush_evaluates_pending_scan_without_inventing_a_close() -> None:
    state = _state()
    store = _ObservationStore()
    engine, lifecycle = _engine(state, store, run_id="repeatable-run")
    engine.process_event(
        _snapshot(
            "A",
            event_index=0,
            sequence=1,
            yes_bid="0.60",
            no_bid="0.38",
        )
    )
    engine.process_event(
        _snapshot(
            "M",
            event_index=1,
            sequence=2,
            yes_bid="0.70",
            no_bid="0.28",
        )
    )

    decisions = engine.flush()

    assert len(decisions) == 1
    assert decisions[0].observation.transition is OpportunityTransition.OPEN
    assert len(lifecycle.active_episodes) == 1
    assert lifecycle.completed_episodes == ()


def test_right_censor_all_is_ordered_and_persisted_before_each_state_move() -> None:
    state = _state()
    store = _ObservationStore()
    engine, lifecycle = _engine(state, store)
    _open_canonical(engine)
    first_active = next(iter(lifecycle.active_episodes.values()))
    second_component_id = "component:synthetic-second"
    lifecycle.handle_scan(
        ComponentScanResult(
            component_id=second_component_id,
            market_tickers=first_active.market_tickers,
            relation_types=first_active.relation_types,
            relation_sources=first_active.relation_sources,
            market_contexts=first_active.market_contexts,
            as_of=NOW + timedelta(milliseconds=50),
            trigger_event_index=2,
            status=ScanStatus.OPPORTUNITY,
            opportunity=first_active.opportunity,
            solver_status="optimal",
            reason=None,
            solve_duration_ms=Decimal("0.25"),
            num_states=2,
            num_instruments=len(first_active.portfolio_legs),
            num_legs=len(first_active.portfolio_legs),
            reference_violation=True,
            one_contract_gross_survived=True,
            one_contract_fee_survived=True,
            depth_executable=True,
        )
    )
    component_ids = sorted(lifecycle.active_episodes)
    assert len(component_ids) == 2
    observations_before = lifecycle.observations

    # The run-ended control must be later than every economic event. Candidate
    # validation occurs before persistence, so this leaves both episodes untouched.
    with pytest.raises(ValueError, match="strictly increasing event indices"):
        lifecycle.right_censor_all(NOW + timedelta(seconds=1), 2, "run_ended")
    assert sorted(lifecycle.active_episodes) == component_ids
    assert lifecycle.censored_episodes == ()
    assert lifecycle.observations == observations_before

    # Fail the second canonical component. The first successful write moves only
    # its own episode; the failed episode remains active and can be retried safely.
    store.fail_component_id = component_ids[1]
    with pytest.raises(RuntimeError, match="persistent storage failure"):
        lifecycle.right_censor_all(NOW + timedelta(seconds=1), 3, "run_ended")
    assert tuple(episode.component_id for episode in lifecycle.censored_episodes) == (
        component_ids[0],
    )
    assert tuple(lifecycle.active_episodes) == (component_ids[1],)

    store.fail_component_id = None
    retried = lifecycle.right_censor_all(NOW + timedelta(seconds=1), 3, "run_ended")

    assert tuple(item.component_id for item in retried) == (component_ids[1],)
    assert lifecycle.active_episodes == {}
    assert tuple(episode.component_id for episode in lifecycle.censored_episodes) == tuple(
        component_ids
    )
    assert lifecycle.completed_episodes == ()
    for episode in lifecycle.censored_episodes:
        assert episode.transition is OpportunityTransition.RIGHT_CENSORED
        assert episode.closed_at is None
        assert episode.closed_event_index is None
        assert episode.close_reason is None
        assert episode.censored_at == NOW + timedelta(seconds=1)
        assert episode.censored_event_index == 3
        assert episode.censor_reason == "run_ended"
    censor_observations = lifecycle.observations[-2:]
    assert tuple(item.component_id for item in censor_observations) == tuple(component_ids)
    assert all(
        item.transition is OpportunityTransition.RIGHT_CENSORED for item in censor_observations
    )
    assert tuple(item.observation_id for item in censor_observations) == tuple(
        opportunity_observation_id(
            run_id="run-1",
            component_id=item.component_id,
            event_index=3,
            transition=OpportunityTransition.RIGHT_CENSORED,
            opportunity_id=item.opportunity_id,
        )
        for item in censor_observations
    )
    assert lifecycle.right_censor_all(NOW + timedelta(seconds=1), 3, "run_ended") == ()
