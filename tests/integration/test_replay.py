"""Deterministic replay, timer ordering, and EOF-censor integration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.config import (
    CollectorSettings,
    EngineSettings,
    FeeSettings,
    OrderBookSettings,
    PaperExecutionSettings,
)
from arbiter.engine.paper_execution import PaperExecutionStatus
from arbiter.models.event import Event
from arbiter.models.market import Market, PriceRange
from arbiter.models.opportunity import OpportunityTransition
from arbiter.models.orderbook import BookStatus, PriceLevel
from arbiter.models.relation import Relation, RelationType
from arbiter.models.series import Series
from arbiter.replay.engine import (
    InMemoryReplayStore,
    ReplayEngine,
    ReplayValidationError,
    recorded_event_stream_hash,
    replay_to_repository,
    split_recorded_runs,
)
from arbiter.replay.events import (
    BookStaleEvent,
    ConnectionInterruptedEvent,
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
    RecordedEvent,
    RunEndedEvent,
    RunInputPayload,
    RunStartedEvent,
    SubscriptionStartedEvent,
)
from arbiter.solver.fees import KALSHI_FEE_POLICY_VERSION
from arbiter.storage.duckdb import DuckDBRepository, RunManifestRecord, RunStreamEvidence

NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)
RECORDED_RUN_ID = "recorded-run"


def _inputs(*, latency_ms: int = 100) -> RunInputPayload:
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
        fee_multiplier=Decimal("0"),
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
    config = {
        "engine": EngineSettings(
            max_component_markets=12,
            min_net_profit_dollars=Decimal("0.01"),
            min_net_edge_bps=Decimal("1"),
            solve_debounce_ms=25,
        ).model_dump(mode="json"),
        "orderbook": OrderBookSettings(
            default_depth=100,
            stale_after_ms=2_000,
            resync_on_sequence_gap=True,
        ).model_dump(mode="json"),
        "fees": FeeSettings(account_precision=Decimal("0.0001")).model_dump(mode="json"),
        "paper_execution": PaperExecutionSettings(
            enabled=True,
            latency_ms=latency_ms,
            allow_partial_fill=False,
        ).model_dump(mode="json"),
        "collector": CollectorSettings().model_dump(mode="json"),
        "kalshi_environment": "demo",
    }
    return RunInputPayload.build(
        config=config,
        markets=markets,
        events=(event,),
        series=(series,),
        relations=(relation,),
        fee_policy={"policy_version": KALSHI_FEE_POLICY_VERSION},
    )


def _base_records(*, latency_ms: int = 100) -> list[RecordedEvent]:
    inputs = _inputs(latency_ms=latency_ms)
    return [
        RunStartedEvent.build(
            event_index=0,
            local_received_ts=NOW,
            run_id=RECORDED_RUN_ID,
            inputs=inputs,
        ),
        SubscriptionStartedEvent(
            event_index=1,
            local_received_ts=NOW,
            connection_id="connection-1",
            sid=7,
            tickers=("A", "M"),
        ),
        _snapshot(
            "A",
            event_index=2,
            sequence=1,
            yes_bid="0.60",
            no_bid="0.38",
            received_at=NOW,
        ),
        _snapshot(
            "M",
            event_index=3,
            sequence=2,
            yes_bid="0.70",
            no_bid="0.28",
            received_at=NOW,
        ),
    ]


def _complete(
    records: list[RecordedEvent],
    *,
    end_at: datetime,
) -> tuple[RecordedEvent, ...]:
    records.append(
        RunEndedEvent(
            event_index=len(records),
            local_received_ts=end_at,
            run_id=RECORDED_RUN_ID,
            status="succeeded",
        )
    )
    return tuple(records)


def _snapshot(
    ticker: str,
    *,
    event_index: int,
    sequence: int,
    yes_bid: str,
    no_bid: str,
    received_at: datetime,
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
    *,
    event_index: int,
    received_at: datetime,
    ticker: str = "M",
    sequence: int = 3,
    side: str = "yes",
    price: str = "0.70",
    quantity_delta: str = "-5.00",
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


def _replay(
    records: tuple[RecordedEvent, ...],
    *,
    replay_run_id: str,
    speed: float | str = "max",
    sleeps: list[float] | None = None,
):
    store = InMemoryReplayStore()
    result = ReplayEngine(
        records,
        replay_run_id=replay_run_id,
        store=store,
        speed=speed,  # type: ignore[arg-type]
        sleeper=(lambda seconds: sleeps.append(seconds)) if sleeps is not None else lambda _: None,
    ).run()
    assert tuple(store.observations) == result.observations
    assert tuple(store.paper_executions) == result.paper_executions
    return result


def test_repeated_replay_is_economically_equal_and_stream_hash_is_framed() -> None:
    records = _complete(_base_records(), end_at=NOW + timedelta(seconds=1))

    first = _replay(records, replay_run_id="replay-a")
    second = _replay(tuple(reversed(records)), replay_run_id="replay-b")

    assert first.economic_observations == second.economic_observations
    assert first.event_stream_hash == second.event_stream_hash
    assert first.event_stream_hash == recorded_event_stream_hash(records)
    assert first.replay_run_id != first.recorded_run_id
    assert [item.transition for item in first.observations] == [
        OpportunityTransition.OPEN,
        OpportunityTransition.RIGHT_CENSORED,
    ]
    assert len(first.paper_executions) == 1
    assert first.paper_executions[0].result.status is PaperExecutionStatus.SURVIVED


def test_speed_changes_sleeping_only_and_max_never_sleeps() -> None:
    records = _complete(_base_records(), end_at=NOW + timedelta(seconds=1))
    max_sleeps: list[float] = []
    numeric_sleeps: list[float] = []

    maximum = _replay(
        records,
        replay_run_id="replay-max",
        speed="max",
        sleeps=max_sleeps,
    )
    numeric = _replay(
        records,
        replay_run_id="replay-100x",
        speed=100.0,
        sleeps=numeric_sleeps,
    )

    assert max_sleeps == []
    assert numeric_sleeps == [pytest.approx(0.01)]
    assert maximum.economic_observations == numeric.economic_observations
    assert [item.result.status for item in maximum.paper_executions] == [
        item.result.status for item in numeric.paper_executions
    ]


def test_same_time_book_delta_precedes_paper_deadline_and_failure_is_all_or_none() -> None:
    records = _base_records(latency_ms=100)
    deadline = NOW + timedelta(milliseconds=125)
    records.append(_delta(event_index=4, received_at=deadline))
    completed = _complete(records, end_at=NOW + timedelta(milliseconds=200))

    result = _replay(completed, replay_run_id="replay-disappearing-leg")

    assert len(result.paper_executions) == 1
    paper = result.paper_executions[0].result
    assert paper.simulated_execution_at == deadline
    assert paper.status is PaperExecutionStatus.FAILED
    assert paper.failure_reason == "M:no:insufficient_liquidity_at_or_better"
    assert all(leg.fill_status == "not_filled" for leg in paper.legs)
    assert all(not leg.actual_prices for leg in paper.legs)
    assert [item.transition for item in result.observations] == [
        OpportunityTransition.OPEN,
        OpportunityTransition.CLOSED,
    ]


@pytest.mark.parametrize("control_kind", ["disconnect", "stale"])
def test_recorded_disconnect_and_stale_controls_close_active_evidence(
    control_kind: str,
) -> None:
    records = _base_records()
    if control_kind == "disconnect":
        control_at = NOW + timedelta(milliseconds=100)
        records.append(
            ConnectionInterruptedEvent(
                event_index=4,
                local_received_ts=control_at,
                connection_id="connection-1",
                tickers=("A", "M"),
                reason="socket_closed",
            )
        )
        end_at = NOW + timedelta(milliseconds=200)
    else:
        control_at = NOW + timedelta(milliseconds=2_001)
        records.append(
            BookStaleEvent(
                event_index=4,
                local_received_ts=control_at,
                ticker="A",
                source_event_index=2,
                source_connection_id="connection-1",
                source_sid=7,
                source_snapshot_id="snapshot:A",
                source_sequence=1,
            )
        )
        end_at = NOW + timedelta(milliseconds=2_100)

    result = _replay(
        _complete(records, end_at=end_at),
        replay_run_id=f"replay-{control_kind}",
    )

    assert [item.transition for item in result.observations] == [
        OpportunityTransition.OPEN,
        OpportunityTransition.CLOSED,
    ]
    if control_kind == "disconnect":
        assert result.final_state.orderbooks["A"].status is BookStatus.RESYNC_REQUIRED
        assert result.final_state.orderbooks["M"].status is BookStatus.RESYNC_REQUIRED


def test_eof_marks_future_paper_attempt_insufficient_and_right_censors_episode() -> None:
    records = _complete(
        _base_records(latency_ms=500),
        end_at=NOW + timedelta(milliseconds=100),
    )

    result = _replay(records, replay_run_id="replay-eof")

    assert [item.result.status for item in result.paper_executions] == [
        PaperExecutionStatus.INSUFFICIENT_FUTURE_DATA
    ]
    paper = result.paper_executions[0]
    assert paper.resolved_at == records[-1].local_received_ts
    assert paper.attempted_at is None
    assert [item.transition for item in result.observations] == [
        OpportunityTransition.OPEN,
        OpportunityTransition.RIGHT_CENSORED,
    ]
    assert result.completed_episodes == ()
    assert len(result.censored_episodes) == 1
    assert result.censored_episodes[0].closed_at is None
    assert result.censored_episodes[0].censored_at == records[-1].local_received_ts
    assert result.censored_episodes[0].censor_reason == "run_succeeded"


def test_eof_refuses_to_extrapolate_a_trailing_debounce_scan() -> None:
    records = _base_records(latency_ms=500)
    end_at = NOW + timedelta(milliseconds=100)
    records.append(
        _delta(
            event_index=4,
            received_at=end_at,
            ticker="A",
            sequence=3,
            side="no",
            price="0.40",
            quantity_delta="5.00",
        )
    )
    completed = _complete(records, end_at=end_at)

    with pytest.raises(ReplayValidationError, match="trailing solve debounce"):
        _replay(completed, replay_run_id="replay-truncated-debounce")


def test_corrupted_index_time_and_run_boundaries_are_rejected_before_writes() -> None:
    canonical = _complete(_base_records(), end_at=NOW + timedelta(seconds=1))
    corrupted: list[tuple[tuple[RecordedEvent, ...], str]] = []

    gap = list(canonical)
    gap[2] = gap[2].model_copy(update={"event_index": 99})
    corrupted.append((tuple(gap), "not contiguous"))

    backwards = list(canonical)
    backwards[3] = backwards[3].model_copy(
        update={"local_received_ts": NOW - timedelta(microseconds=1)}
    )
    corrupted.append((tuple(backwards), "moves backwards"))

    no_start = tuple(canonical[1:])
    corrupted.append((no_start, "RunStartedEvent"))

    wrong_end = list(canonical)
    wrong_end[-1] = wrong_end[-1].model_copy(update={"run_id": "another-run"})
    corrupted.append((tuple(wrong_end), "do not match"))

    for index, (records, message) in enumerate(corrupted):
        store = InMemoryReplayStore()
        with pytest.raises(ReplayValidationError, match=message):
            ReplayEngine(
                records,
                replay_run_id=f"replay-corrupt-{index}",
                store=store,
            )
        assert store.observations == []
        assert store.paper_executions == []


def test_replay_id_must_be_distinct_and_speed_must_be_valid() -> None:
    records = _complete(_base_records(), end_at=NOW + timedelta(seconds=1))

    with pytest.raises(ReplayValidationError, match="must differ"):
        ReplayEngine(
            records,
            replay_run_id=RECORDED_RUN_ID,
            store=InMemoryReplayStore(),
        )
    with pytest.raises(ReplayValidationError, match="finite and positive"):
        ReplayEngine(
            records,
            replay_run_id="replay-invalid-speed",
            store=InMemoryReplayStore(),
            speed=0,
        )


def test_repository_replay_persists_manifest_observations_and_paper(
    tmp_path: Path,
) -> None:
    records = _complete(_base_records(), end_at=NOW + timedelta(seconds=1))
    replay_started_at = NOW + timedelta(days=1)

    with DuckDBRepository(tmp_path / "replay.duckdb") as repository:
        result = replay_to_repository(
            records,
            replay_run_id="replay-durable",
            repository=repository,
            speed="max",
            sleeper=lambda _: None,
            clock=lambda: replay_started_at,
        )
        manifest = repository.get_run_manifest("replay-durable")

        assert manifest is not None
        assert manifest.run_type == "replay"
        assert manifest.status == "succeeded"
        assert manifest.source_run_id == RECORDED_RUN_ID
        assert manifest.recording_id == RECORDED_RUN_ID
        assert manifest.first_event_index == records[0].event_index
        assert manifest.last_event_index == records[-1].event_index
        assert manifest.event_count == len(records)
        assert manifest.event_stream_hash == recorded_event_stream_hash(records)
        assert manifest.input_payload == records[0].inputs.model_dump(mode="json")
        assert manifest.started_at == replay_started_at
        assert manifest.ended_at == replay_started_at
        assert repository.table_count("opportunity_observations") == len(result.observations)
        assert repository.table_count("paper_executions") == len(result.paper_executions)


def test_repository_replay_rejects_conflicting_source_manifest_before_writes(
    tmp_path: Path,
) -> None:
    records = _complete(_base_records(), end_at=NOW + timedelta(seconds=1))
    started = records[0]
    assert isinstance(started, RunStartedEvent)
    source_manifest = RunManifestRecord(
        run_id=RECORDED_RUN_ID,
        run_type="live",
        started_at=NOW,
        manifest_version=2,
        recording_format_version=2,
        event_schema_version=2,
        recording_id=RECORDED_RUN_ID,
        first_event_index=records[0].event_index,
        last_event_index=records[-1].event_index,
        event_count=len(records),
        event_stream_hash="0" * 64,
        input_payload=started.inputs.model_dump(mode="json"),
        config_hash=started.config_hash,
        metadata_hash=started.metadata_hash,
        relations_hash=started.relations_hash,
        fee_policy_hash=started.fee_policy_hash,
    )
    source_stream = RunStreamEvidence(
        first_event_index=records[0].event_index,
        last_event_index=records[-1].event_index,
        event_count=len(records),
        event_stream_hash="0" * 64,
    )

    with DuckDBRepository(tmp_path / "conflict.duckdb") as repository:
        repository.start_run(source_manifest)
        repository.finalize_run(
            RECORDED_RUN_ID,
            status="succeeded",
            ended_at=NOW + timedelta(seconds=1),
            stream=source_stream,
        )
        with pytest.raises(ReplayValidationError, match="source manifest"):
            replay_to_repository(
                records,
                replay_run_id="replay-conflict",
                repository=repository,
                sleeper=lambda _: None,
                clock=lambda: NOW + timedelta(days=1),
            )

        assert repository.get_run_manifest("replay-conflict") is None


def test_repository_replay_finalizes_failed_manifest(tmp_path: Path) -> None:
    records = _base_records(latency_ms=500)
    end_at = NOW + timedelta(milliseconds=100)
    records.append(
        _delta(
            event_index=4,
            received_at=end_at,
            ticker="A",
            sequence=3,
            side="no",
            price="0.40",
            quantity_delta="5.00",
        )
    )
    completed = _complete(records, end_at=end_at)

    with DuckDBRepository(tmp_path / "failed.duckdb") as repository:
        with pytest.raises(ReplayValidationError, match="trailing solve debounce"):
            replay_to_repository(
                completed,
                replay_run_id="replay-failed",
                repository=repository,
                sleeper=lambda _: None,
                clock=lambda: NOW + timedelta(days=1),
            )
        manifest = repository.get_run_manifest("replay-failed")

    assert manifest is not None
    assert manifest.status == "failed"
    assert manifest.error == "ReplayValidationError: replay terminated"


def test_split_recorded_runs_is_strict_for_files_and_can_skip_raw_date_records() -> None:
    raw = _snapshot(
        "A",
        event_index=0,
        sequence=1,
        yes_bid="0.60",
        no_bid="0.38",
        received_at=NOW,
    )
    source = _complete(_base_records(), end_at=NOW + timedelta(seconds=1))
    bounded = tuple(
        record.model_copy(update={"event_index": record.event_index + 1}) for record in source
    )
    mixed = (raw, *bounded)

    with pytest.raises(ReplayValidationError, match="outside a self-contained"):
        split_recorded_runs(mixed)

    assert split_recorded_runs(mixed, allow_unbounded_records=True) == (bounded,)


def test_split_recorded_runs_rejects_unterminated_and_duplicate_run_ids() -> None:
    complete = _complete(_base_records(), end_at=NOW + timedelta(seconds=1))
    with pytest.raises(ReplayValidationError, match="before its run-end"):
        split_recorded_runs(complete[:-1])

    second = tuple(
        record.model_copy(
            update={
                "event_index": record.event_index + len(complete),
                "local_received_ts": record.local_received_ts + timedelta(days=1),
            }
        )
        for record in complete
    )
    with pytest.raises(ReplayValidationError, match="repeats a source run ID"):
        split_recorded_runs((*complete, *second))
