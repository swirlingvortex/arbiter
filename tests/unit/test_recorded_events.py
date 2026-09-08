from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from arbiter.models.event import Event
from arbiter.models.market import Market, PriceRange
from arbiter.models.orderbook import PriceLevel
from arbiter.models.relation import Relation, RelationType
from arbiter.models.series import Series
from arbiter.replay.events import (
    RECORDED_CONTROL_SCHEMA_VERSION,
    BookStaleEvent,
    ConnectionInterruptedEvent,
    EventIndexAllocator,
    FeeRefreshAppliedEvent,
    FeeRefreshStartedEvent,
    MarketRefreshAppliedEvent,
    MarketRefreshStartedEvent,
    OrderBookSnapshotEvent,
    RunEndedEvent,
    RunInputPayload,
    RunStartedEvent,
    SubscriptionStartedEvent,
    dump_control_event_json,
    dump_market_data_event_json,
    dump_recorded_event_json,
    load_control_event_json,
    load_market_data_event_json,
    load_recorded_event_json,
)

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def _market(ticker: str = "KX-A") -> Market:
    return Market(
        ticker=ticker,
        event_ticker="KX-EVENT",
        series_ticker="KX-SERIES",
        title=f"Market {ticker}",
        status="active",
        price_ranges=(
            PriceRange(start=Decimal("0.01"), end=Decimal("0.99"), step=Decimal("0.01")),
        ),
        raw={"ticker": ticker},
    )


def _event(*tickers: str) -> Event:
    return Event(
        ticker="KX-EVENT",
        series_ticker="KX-SERIES",
        title="Event",
        market_tickers=tuple(tickers),
        raw={"event_ticker": "KX-EVENT"},
    )


def _series() -> Series:
    return Series(
        ticker="KX-SERIES",
        title="Series",
        fee_type="quadratic",
        fee_multiplier=Decimal("0.07"),
        raw={"ticker": "KX-SERIES"},
    )


def _relation() -> Relation:
    return Relation(
        relation_id="relation-1",
        market_tickers=("KX-A", "KX-B"),
        relation_type=RelationType.EXACTLY_ONE,
        source="manual",
        verified=True,
        rationale="fixture relation",
        created_at=NOW,
    )


def _inputs() -> RunInputPayload:
    return RunInputPayload.build(
        config={"engine": {"debounce_ms": 25}, "environment": "demo"},
        markets=(_market("KX-B"), _market("KX-A")),
        events=(_event("KX-A", "KX-B"),),
        series=(_series(),),
        relations=(_relation(),),
        fee_policy={"policy_version": "kalshi-v1", "multiplier": "0.07"},
    )


def _snapshot() -> OrderBookSnapshotEvent:
    return OrderBookSnapshotEvent(
        event_index=1,
        local_received_ts=NOW,
        exchange_ts=NOW,
        ticker="KX-A",
        sequence=1,
        sid=11,
        connection_id="connection-1",
        snapshot_id="connection-1:11:1",
        yes_bids=(PriceLevel(price=Decimal("0.6"), quantity=Decimal("2")),),
        no_bids=(PriceLevel(price=Decimal("0.39"), quantity=Decimal("3")),),
    )


def _controls() -> tuple[
    RunStartedEvent,
    SubscriptionStartedEvent,
    ConnectionInterruptedEvent,
    BookStaleEvent,
    MarketRefreshStartedEvent,
    MarketRefreshAppliedEvent,
    FeeRefreshStartedEvent,
    FeeRefreshAppliedEvent,
    RunEndedEvent,
]:
    return (
        RunStartedEvent.build(
            event_index=0,
            local_received_ts=NOW,
            run_id="run-1",
            inputs=_inputs(),
        ),
        SubscriptionStartedEvent(
            event_index=2,
            local_received_ts=NOW,
            connection_id="connection-1",
            sid=11,
            tickers=("KX-A", "KX-B"),
        ),
        ConnectionInterruptedEvent(
            event_index=3,
            local_received_ts=NOW,
            connection_id="connection-1",
            tickers=("KX-A", "KX-B"),
            reason="socket_closed",
        ),
        BookStaleEvent(
            event_index=4,
            local_received_ts=NOW,
            ticker="KX-A",
            source_event_index=1,
            source_connection_id="connection-1",
            source_sid=11,
            source_snapshot_id="connection-1:11:1",
            source_sequence=1,
        ),
        MarketRefreshStartedEvent(
            event_index=5,
            local_received_ts=NOW,
            ticker="KX-A",
            reason="metadata_updated",
            connection_id="connection-1",
            sid=21,
        ),
        MarketRefreshAppliedEvent(
            event_index=6,
            local_received_ts=NOW,
            refresh_started_event_index=5,
            market=_market(),
        ),
        FeeRefreshStartedEvent(
            event_index=7,
            local_received_ts=NOW,
            event_ticker="KX-EVENT",
            affected_tickers=("KX-A", "KX-B"),
            connection_id="connection-1",
            sid=22,
        ),
        FeeRefreshAppliedEvent(
            event_index=8,
            local_received_ts=NOW,
            refresh_started_event_index=7,
            event=_event("KX-A", "KX-B"),
        ),
        RunEndedEvent(
            event_index=9,
            local_received_ts=NOW,
            run_id="run-1",
            status="succeeded",
        ),
    )


def test_event_index_allocator_is_strict_contiguous_and_shared() -> None:
    allocator = EventIndexAllocator(start_event_index=41)

    with ThreadPoolExecutor(max_workers=8) as executor:
        allocated = tuple(executor.map(lambda _: allocator.allocate(), range(100)))

    assert sorted(allocated) == list(range(41, 141))
    assert allocator.next_event_index == 141


@pytest.mark.parametrize("value", [-1, True, 1.0, "1"])
def test_event_index_allocator_rejects_invalid_start(value: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        EventIndexAllocator(value)  # type: ignore[arg-type]


def test_run_input_builder_canonicalizes_membership_and_hashes() -> None:
    inputs = _inputs()
    started = RunStartedEvent.build(
        event_index=0,
        local_received_ts=NOW,
        run_id="run-1",
        inputs=inputs,
    )

    assert tuple(market.ticker for market in inputs.markets) == ("KX-A", "KX-B")
    assert inputs.config_json == ('{"engine":{"debounce_ms":25},"environment":"demo"}')
    assert started.schema_version == RECORDED_CONTROL_SCHEMA_VERSION
    assert started.model_dump()["config_hash"] == inputs.computed_hashes()["config_hash"]


def test_run_start_rejects_tampered_or_malformed_hash() -> None:
    started = RunStartedEvent.build(
        event_index=0,
        local_received_ts=NOW,
        run_id="run-1",
        inputs=_inputs(),
    )
    payload = started.model_dump()
    payload["metadata_hash"] = "0" * 64

    with pytest.raises(ValidationError, match="does not match"):
        RunStartedEvent.model_validate(payload)

    payload["metadata_hash"] = "not-a-sha256"
    with pytest.raises(ValidationError):
        RunStartedEvent.model_validate(payload)


@pytest.mark.parametrize(
    "config",
    [
        {"api_key_id": "do-not-record"},
        {"auth": {"private_key_path": "/tmp/key.pem"}},
        {"auth": "-----BEGIN PRIVATE KEY-----payload"},
    ],
)
def test_run_inputs_reject_secrets_and_key_paths(config: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="cannot contain"):
        RunInputPayload.build(
            config=config,
            markets=(_market("KX-A"), _market("KX-B")),
            events=(_event("KX-A", "KX-B"),),
            series=(_series(),),
            relations=(_relation(),),
            fee_policy={"version": 1},
        )


def test_run_inputs_reject_noncanonical_json_and_unsorted_direct_membership() -> None:
    values = _inputs().model_dump()
    values["config_json"] = '{"z": 1, "a": 2}'
    with pytest.raises(ValidationError, match="canonical JSON"):
        RunInputPayload.model_validate(values)

    values = _inputs().model_dump()
    values["markets"] = tuple(reversed(values["markets"]))
    with pytest.raises(ValidationError, match="canonically sorted"):
        RunInputPayload.model_validate(values)


@pytest.mark.parametrize("control", _controls())
def test_every_control_has_deterministic_strict_json_round_trip(control: object) -> None:
    first = dump_control_event_json(control)  # type: ignore[arg-type]
    restored = load_control_event_json(first)

    assert restored == control
    assert dump_control_event_json(restored) == first
    assert load_recorded_event_json(first) == control
    assert dump_recorded_event_json(restored) == first


def test_recorded_union_preserves_schema_v1_market_data_json_behavior() -> None:
    snapshot = _snapshot()
    original = dump_market_data_event_json(snapshot)

    assert dump_recorded_event_json(snapshot) == original
    assert '"price":"0.6000"' in original
    assert '"quantity":"2.00"' in original
    assert load_recorded_event_json(original) == snapshot
    assert load_market_data_event_json(original) == snapshot


def test_recorded_book_json_rejects_values_beyond_persisted_fixed_scale() -> None:
    snapshot = _snapshot().model_copy(
        update={"yes_bids": (PriceLevel(price=Decimal("0.62001"), quantity=Decimal("2.00")),)}
    )

    with pytest.raises(ValueError, match="persisted 4-decimal scale"):
        dump_recorded_event_json(snapshot)


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_index", "2"),
        ("sid", "11"),
        ("schema_version", 1),
        ("local_received_ts", datetime(2026, 9, 3, 12, 0)),
    ],
)
def test_controls_reject_coercion_wrong_schema_and_naive_time(field: str, value: object) -> None:
    payload = _controls()[1].model_dump()
    payload[field] = value

    with pytest.raises(ValidationError):
        SubscriptionStartedEvent.model_validate(payload)


@pytest.mark.parametrize(
    "event",
    [
        SubscriptionStartedEvent,
        ConnectionInterruptedEvent,
        FeeRefreshStartedEvent,
    ],
)
def test_membership_controls_reject_unsorted_or_duplicate_tickers(event: type[object]) -> None:
    payload = _controls()[1 if event is SubscriptionStartedEvent else 2].model_dump()
    if event is FeeRefreshStartedEvent:
        payload = _controls()[6].model_dump()
        payload["affected_tickers"] = ("KX-B", "KX-A")
    else:
        payload["tickers"] = ("KX-B", "KX-A")

    with pytest.raises(ValidationError, match="canonically sorted"):
        event.model_validate(payload)  # type: ignore[attr-defined]


def test_stale_and_refresh_applied_records_require_an_earlier_source() -> None:
    stale = _controls()[3].model_dump()
    stale["source_event_index"] = stale["event_index"]
    with pytest.raises(ValidationError, match="must precede"):
        BookStaleEvent.model_validate(stale)

    refresh = _controls()[5].model_dump()
    refresh["refresh_started_event_index"] = refresh["event_index"]
    with pytest.raises(ValidationError, match="must precede"):
        MarketRefreshAppliedEvent.model_validate(refresh)


def test_applied_refreshes_carry_complete_normalized_models() -> None:
    market_refresh = load_control_event_json(dump_control_event_json(_controls()[5]))
    fee_refresh = load_control_event_json(dump_control_event_json(_controls()[7]))

    assert isinstance(market_refresh, MarketRefreshAppliedEvent)
    assert market_refresh.market == _market()
    assert isinstance(fee_refresh, FeeRefreshAppliedEvent)
    assert fee_refresh.event == _event("KX-A", "KX-B")


def test_control_models_are_frozen_and_forbid_unknown_fields() -> None:
    subscription = _controls()[1]
    with pytest.raises(ValidationError, match="frozen"):
        subscription.sid = 12

    payload = subscription.model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        SubscriptionStartedEvent.model_validate(payload)
