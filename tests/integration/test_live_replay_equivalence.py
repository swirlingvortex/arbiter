"""Economic equivalence between synthetic live processing and recorded replay."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from arbiter.engine.scanner import ArbitrageEngine, ComponentScanner, OpportunityLifecycle
from arbiter.engine.state import EngineState
from arbiter.models.opportunity import OpportunityTransition
from arbiter.replay.engine import (
    InMemoryReplayStore,
    ReplayConfiguration,
    ReplayEngine,
    economic_observation_projection,
    recorded_event_stream_hash,
)
from arbiter.replay.events import RecordedEvent, RunEndedEvent, RunStartedEvent
from arbiter.storage.parquet import read_recorded_events

FIXTURE = Path("tests/fixtures/orderbooks/canonical_lifecycle.parquet")


def _synthetic_live_observations(records: tuple[RecordedEvent, ...]):
    started = records[0]
    assert isinstance(started, RunStartedEvent)
    configuration = ReplayConfiguration.model_validate(started.inputs.config)
    state = EngineState(
        markets=started.inputs.markets,
        events=started.inputs.events,
        series=started.inputs.series,
        relations=started.inputs.relations,
        max_component_markets=configuration.engine.max_component_markets,
    )
    store = InMemoryReplayStore()
    lifecycle = OpportunityLifecycle(run_id="synthetic-live", store=store)
    engine = ArbitrageEngine(
        state=state,
        scanner=ComponentScanner(
            state,
            stale_after=timedelta(milliseconds=configuration.orderbook.stale_after_ms),
            account_precision=configuration.fees.account_precision,
            minimum_net_profit=configuration.engine.min_net_profit_dollars,
            minimum_net_edge_bps=configuration.engine.min_net_edge_bps,
        ),
        lifecycle=lifecycle,
        solve_debounce=timedelta(milliseconds=configuration.engine.solve_debounce_ms),
        recorded_run_id=started.run_id,
    )

    for record in records:
        engine.process_record(record)
        if isinstance(record, RunEndedEvent):
            lifecycle.right_censor_all(
                observed_at=record.local_received_ts,
                event_index=record.event_index,
                reason=record.reason or f"run_{record.status}",
            )

    assert engine.next_due_at is None
    assert tuple(store.observations) == lifecycle.observations
    return lifecycle.observations


def test_canonical_recording_has_identical_live_and_replay_decisions() -> None:
    records = read_recorded_events(FIXTURE)
    live_observations = _synthetic_live_observations(records)

    first = ReplayEngine(
        records,
        replay_run_id="fixture-replay-one",
        store=InMemoryReplayStore(),
        speed="max",
        sleeper=lambda _: None,
    ).run()
    second = ReplayEngine(
        tuple(reversed(records)),
        replay_run_id="fixture-replay-two",
        store=InMemoryReplayStore(),
        speed="max",
        sleeper=lambda _: None,
    ).run()

    expected = economic_observation_projection(live_observations)
    assert first.economic_observations == expected
    assert second.economic_observations == expected
    assert first.event_stream_hash == recorded_event_stream_hash(records)
    assert second.event_stream_hash == first.event_stream_hash

    assert [item.transition for item in first.observations] == [
        OpportunityTransition.OPEN,
        OpportunityTransition.CLOSED,
    ]
    opened = first.observations[0].opportunity
    assert opened is not None
    assert opened.capital_required == Decimal("0.92")
    assert opened.gross_profit == Decimal("0.08")
    assert opened.net_profit == Decimal("0.08")

    assert len(first.paper_executions) == 1
    paper = first.paper_executions[0].result
    assert paper.status == "failed"
    assert paper.simulated_execution_at == records[4].local_received_ts
    assert all(leg.fill_status == "not_filled" for leg in paper.legs)
    assert all(not leg.actual_prices for leg in paper.legs)
