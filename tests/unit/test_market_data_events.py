from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import TypeAdapter, ValidationError

from arbiter.models.orderbook import PriceLevel
from arbiter.replay.events import (
    MarketDataEvent,
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
    dump_market_data_event_json,
    load_market_data_event_json,
)

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def _base_fields() -> dict[str, object]:
    return {
        "event_index": 0,
        "local_received_ts": NOW,
        "exchange_ts": NOW,
        "ticker": "KX-TEST",
        "sequence": 1,
        "sid": 7,
        "connection_id": "connection-1",
        "snapshot_id": "connection-1:7:1",
    }


def _snapshot(**updates: object) -> OrderBookSnapshotEvent:
    values = {
        **_base_fields(),
        "yes_bids": (
            PriceLevel(price=Decimal("0.6200"), quantity=Decimal("2.50")),
            PriceLevel(price=Decimal("0.6100"), quantity=Decimal("1.00")),
        ),
        "no_bids": (PriceLevel(price=Decimal("0.3700"), quantity=Decimal("3.00")),),
        **updates,
    }
    return OrderBookSnapshotEvent.model_validate(values)


def _delta(**updates: object) -> OrderBookDeltaEvent:
    values = {
        **_base_fields(),
        "event_index": 1,
        "sequence": 2,
        "side": "no",
        "price": Decimal("0.3700"),
        "quantity_delta": Decimal("-0.25"),
        **updates,
    }
    return OrderBookDeltaEvent.model_validate(values)


def test_snapshot_retains_exact_internal_bid_levels() -> None:
    event = _snapshot()

    assert event.schema_version == 1
    assert event.event_type == "snapshot"
    assert event.price_convention == "yes_price"
    assert event.yes_bids[0].price == Decimal("0.6200")
    assert event.yes_bids[0].quantity == Decimal("2.50")
    assert event.no_bids[0].price == Decimal("0.3700")


def test_delta_retains_exact_signed_change() -> None:
    event = _delta()

    assert event.event_type == "delta"
    assert event.side == "no"
    assert event.price == Decimal("0.3700")
    assert event.quantity_delta == Decimal("-0.25")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_index", -1),
        ("sequence", 0),
        ("sid", 0),
        ("ticker", ""),
        ("connection_id", "   "),
        ("snapshot_id", ""),
        ("price_convention", "no_price"),
        ("schema_version", 2),
    ],
)
def test_shared_fields_fail_closed(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _snapshot(**{field: value})


def test_shared_fields_are_strict_about_scalar_types() -> None:
    with pytest.raises(ValidationError):
        _snapshot(event_index="0")


@pytest.mark.parametrize("field", ["local_received_ts", "exchange_ts"])
def test_timestamps_must_be_timezone_aware(field: str) -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _snapshot(**{field: datetime(2026, 9, 3, 12, 0)})


@pytest.mark.parametrize(
    "updates",
    [
        {
            "yes_bids": (
                PriceLevel(price=Decimal("0.60"), quantity=Decimal("1")),
                PriceLevel(price=Decimal("0.61"), quantity=Decimal("1")),
            )
        },
        {
            "no_bids": (
                PriceLevel(price=Decimal("0.40"), quantity=Decimal("1")),
                PriceLevel(price=Decimal("0.40"), quantity=Decimal("2")),
            )
        },
    ],
)
def test_snapshot_rejects_unsorted_or_duplicate_bid_prices(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _snapshot(**updates)


@pytest.mark.parametrize("quantity_delta", [Decimal("0"), Decimal("NaN")])
def test_delta_rejects_zero_or_nonfinite_quantity_change(quantity_delta: Decimal) -> None:
    with pytest.raises(ValidationError):
        _delta(quantity_delta=quantity_delta)


@pytest.mark.parametrize("price", [Decimal("-0.0001"), Decimal("1.0001"), Decimal("NaN")])
def test_delta_rejects_invalid_price(price: Decimal) -> None:
    with pytest.raises(ValidationError):
        _delta(price=price)


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _delta(unexpected="value")


def test_discriminator_selects_each_event_type() -> None:
    adapter: TypeAdapter[MarketDataEvent] = TypeAdapter(MarketDataEvent)

    snapshot = adapter.validate_python(_snapshot().model_dump())
    delta = adapter.validate_python(_delta().model_dump())

    assert isinstance(snapshot, OrderBookSnapshotEvent)
    assert isinstance(delta, OrderBookDeltaEvent)


def test_unknown_discriminator_fails_closed() -> None:
    payload = _snapshot().model_dump()
    payload["event_type"] = "trade"

    with pytest.raises(ValidationError):
        TypeAdapter(MarketDataEvent).validate_python(payload)


@pytest.mark.parametrize("event", [_snapshot(), _delta()])
def test_json_round_trip_is_deterministic_and_exact(event: MarketDataEvent) -> None:
    first = dump_market_data_event_json(event)
    restored = load_market_data_event_json(first)

    assert restored == event
    assert dump_market_data_event_json(restored) == first
