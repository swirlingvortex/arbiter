#!/usr/bin/env python3
"""Generate the deterministic schema-v2 lifecycle fixture used by replay tests."""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from arbiter.config import (
    CollectorSettings,
    EngineSettings,
    FeeSettings,
    KalshiEnvironment,
    OrderBookSettings,
    PaperExecutionSettings,
)
from arbiter.engine.paper_execution import PaperExecutionStatus
from arbiter.models.event import Event
from arbiter.models.market import Market, PriceRange
from arbiter.models.opportunity import OpportunityTransition
from arbiter.models.orderbook import PriceLevel
from arbiter.models.relation import Relation, RelationType
from arbiter.models.series import Series
from arbiter.replay.engine import InMemoryReplayStore, ReplayEngine
from arbiter.replay.events import (
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
    RecordedEvent,
    RunEndedEvent,
    RunInputPayload,
    RunStartedEvent,
    SubscriptionStartedEvent,
)
from arbiter.solver.fees import KALSHI_FEE_POLICY_VERSION
from arbiter.storage.parquet import ParquetEventWriter, read_recorded_events

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPOSITORY_ROOT / "tests/fixtures/orderbooks/canonical_lifecycle.parquet"
FIXTURE_START = datetime(2026, 9, 3, 12, tzinfo=UTC)
RECORDED_RUN_ID = "fixture-canonical-lifecycle-v1"
CONNECTION_ID = "fixture-connection-1"
SUBSCRIPTION_ID = 1


def _run_inputs() -> RunInputPayload:
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
            event_ticker="FIXTURE-EVENT",
            series_ticker="FIXTURE-SERIES",
            title=f"Canonical implication market {ticker}",
            status="active",
            settlement_ts=FIXTURE_START + timedelta(hours=2),
            price_ranges=price_ranges,
            raw={},
        )
        for ticker in ("A", "M")
    )
    event = Event(
        ticker="FIXTURE-EVENT",
        series_ticker="FIXTURE-SERIES",
        title="Canonical implication fixture",
        category="Sports",
        market_tickers=("A", "M"),
        raw={},
    )
    series = Series(
        ticker="FIXTURE-SERIES",
        title="Canonical implication fixture",
        category="Sports",
        fee_type="quadratic",
        fee_multiplier=Decimal("0"),
        raw={},
    )
    relation = Relation(
        relation_id="fixture-m-implies-a",
        market_tickers=("M", "A"),
        relation_type=RelationType.IMPLIES,
        source="manual",
        verified=True,
        rationale="Fixture axiom: M implies A.",
        created_at=FIXTURE_START,
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
            latency_ms=100,
            allow_partial_fill=False,
        ).model_dump(mode="json"),
        "collector": CollectorSettings().model_dump(mode="json"),
        "kalshi_environment": KalshiEnvironment.DEMO.value,
    }
    fee_policy = {
        "policy_version": KALSHI_FEE_POLICY_VERSION,
        "events": [
            {
                "ticker": event.ticker,
                "fee_type_override": None,
                "fee_multiplier_override": None,
                "fee_changes": [],
            }
        ],
        "series": [
            {
                "ticker": series.ticker,
                "fee_type": series.fee_type,
                "fee_multiplier": str(series.fee_multiplier),
            }
        ],
    }
    return RunInputPayload.build(
        config=config,
        markets=markets,
        events=(event,),
        series=(series,),
        relations=(relation,),
        fee_policy=fee_policy,
    )


def _records() -> tuple[RecordedEvent, ...]:
    inputs = _run_inputs()
    deadline = FIXTURE_START + timedelta(milliseconds=125)
    return (
        RunStartedEvent.build(
            event_index=0,
            local_received_ts=FIXTURE_START,
            run_id=RECORDED_RUN_ID,
            inputs=inputs,
        ),
        SubscriptionStartedEvent(
            event_index=1,
            local_received_ts=FIXTURE_START,
            connection_id=CONNECTION_ID,
            sid=SUBSCRIPTION_ID,
            tickers=("A", "M"),
        ),
        OrderBookSnapshotEvent(
            event_index=2,
            local_received_ts=FIXTURE_START,
            exchange_ts=FIXTURE_START,
            ticker="A",
            sequence=1,
            sid=SUBSCRIPTION_ID,
            connection_id=CONNECTION_ID,
            snapshot_id="fixture-snapshot-A",
            yes_bids=(PriceLevel(price=Decimal("0.60"), quantity=Decimal("1.00")),),
            no_bids=(PriceLevel(price=Decimal("0.38"), quantity=Decimal("1.00")),),
        ),
        OrderBookSnapshotEvent(
            event_index=3,
            local_received_ts=FIXTURE_START,
            exchange_ts=FIXTURE_START,
            ticker="M",
            sequence=2,
            sid=SUBSCRIPTION_ID,
            connection_id=CONNECTION_ID,
            snapshot_id="fixture-snapshot-M",
            yes_bids=(PriceLevel(price=Decimal("0.70"), quantity=Decimal("1.00")),),
            no_bids=(PriceLevel(price=Decimal("0.28"), quantity=Decimal("1.00")),),
        ),
        OrderBookDeltaEvent(
            event_index=4,
            local_received_ts=deadline,
            exchange_ts=deadline,
            ticker="M",
            sequence=3,
            sid=SUBSCRIPTION_ID,
            connection_id=CONNECTION_ID,
            snapshot_id="fixture-snapshot-M",
            side="yes",
            price=Decimal("0.70"),
            quantity_delta=Decimal("-1.00"),
        ),
        RunEndedEvent(
            event_index=5,
            local_received_ts=FIXTURE_START + timedelta(milliseconds=200),
            run_id=RECORDED_RUN_ID,
            status="succeeded",
        ),
    )


def _assert_canonical_replay(records: tuple[RecordedEvent, ...]) -> None:
    store = InMemoryReplayStore()
    result = ReplayEngine(
        records,
        replay_run_id="fixture-generation-smoke",
        store=store,
        speed="max",
    ).run()

    if tuple(item.transition for item in result.observations) != (
        OpportunityTransition.OPEN,
        OpportunityTransition.CLOSED,
    ):
        raise RuntimeError("canonical fixture did not produce OPEN then CLOSED")
    opened, closed = result.observations
    opportunity = opened.opportunity
    if opportunity is None:
        raise RuntimeError("canonical OPEN observation has no opportunity")
    if (
        opened.observed_at != FIXTURE_START + timedelta(milliseconds=25)
        or opportunity.capital_required != Decimal("0.92")
        or opportunity.gross_profit != Decimal("0.08")
        or closed.observed_at != FIXTURE_START + timedelta(milliseconds=150)
    ):
        raise RuntimeError("canonical fixture economics or event-time deadlines changed")
    if opened.evidence.get("reference_violation") is not True:
        raise RuntimeError("canonical fixture did not exercise the midpoint logical funnel")
    if len(result.paper_executions) != 1:
        raise RuntimeError("canonical fixture did not produce exactly one paper attempt")
    paper = result.paper_executions[0].result
    if (
        paper.status is not PaperExecutionStatus.FAILED
        or paper.simulated_execution_at != FIXTURE_START + timedelta(milliseconds=125)
        or paper.failure_reason != "M:no:insufficient_liquidity_at_or_better"
        or any(leg.fill_status != "not_filled" for leg in paper.legs)
    ):
        raise RuntimeError("canonical fixture did not fail paper execution all-or-none")


def generate_fixture() -> Path:
    """Regenerate, replace, strictly read, and smoke-replay the canonical fixture."""

    expected = _records()
    with TemporaryDirectory(prefix="arbiter-canonical-lifecycle-") as temporary:
        dataset = Path(temporary) / "orderbooks"
        writer = ParquetEventWriter(dataset, max_queue_size=len(expected))
        writer.extend(expected)
        generated = writer.close()
        if len(generated) != 1:
            raise RuntimeError(f"expected one generated Parquet part, received {len(generated)}")

        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        staged = FIXTURE_PATH.with_suffix(".parquet.tmp")
        try:
            shutil.copyfile(generated[0], staged)
            os.replace(staged, FIXTURE_PATH)
        finally:
            staged.unlink(missing_ok=True)

    restored = read_recorded_events(FIXTURE_PATH)
    if restored != expected:
        raise RuntimeError("strict Parquet round trip changed canonical fixture records")
    _assert_canonical_replay(restored)
    return FIXTURE_PATH


def main() -> None:
    fixture = generate_fixture()
    payload = fixture.read_bytes()
    print(f"wrote {fixture.relative_to(REPOSITORY_ROOT)}")
    print(f"records={len(read_recorded_events(fixture))}")
    print(f"bytes={len(payload)}")
    print(f"sha256={sha256(payload).hexdigest()}")


if __name__ == "__main__":
    main()
