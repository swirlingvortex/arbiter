"""Strict normalized records shared by live processing and deterministic replay.

Schema version 1 remains the original normalized order-book wire contract. Schema version 2
adds indexed controls that make state invalidations and authoritative metadata replacements
part of the replayable total order.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal, localcontext
from hashlib import sha256
from json import JSONDecodeError
from threading import Lock
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from arbiter.models.event import Event
from arbiter.models.market import Market
from arbiter.models.orderbook import PriceLevel
from arbiter.models.relation import Relation
from arbiter.models.series import Series

MARKET_DATA_SCHEMA_VERSION: Literal[1] = 1
RECORDED_CONTROL_SCHEMA_VERSION: Literal[2] = 2
PriceConvention = Literal["yes_price"]
BookSide = Literal["yes", "no"]
RunEndStatus = Literal["succeeded", "failed", "cancelled"]


class _OrderBookEventBase(BaseModel):
    """Fields that identify one normalized order-book observation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = MARKET_DATA_SCHEMA_VERSION
    event_index: int = Field(ge=0)
    local_received_ts: datetime
    exchange_ts: datetime | None = None
    ticker: str = Field(min_length=1)
    sequence: int = Field(gt=0)
    sid: int = Field(gt=0)
    connection_id: str = Field(min_length=1)
    price_convention: PriceConvention = "yes_price"
    snapshot_id: str = Field(min_length=1)

    @field_validator("local_received_ts", "exchange_ts")
    @classmethod
    def require_aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("market-data event timestamps must be timezone-aware")
        return value

    @field_validator("ticker", "connection_id", "snapshot_id")
    @classmethod
    def require_nonblank_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("market-data event identifiers cannot be blank")
        return value


class OrderBookSnapshotEvent(_OrderBookEventBase):
    """A complete normalized internal YES/NO bid-book replacement."""

    event_type: Literal["snapshot"] = "snapshot"
    yes_bids: tuple[PriceLevel, ...] = ()
    no_bids: tuple[PriceLevel, ...] = ()

    @model_validator(mode="after")
    def validate_bid_ordering(self) -> OrderBookSnapshotEvent:
        for label, levels in (("YES", self.yes_bids), ("NO", self.no_bids)):
            prices = tuple(level.price for level in levels)
            if prices != tuple(sorted(prices, reverse=True)):
                raise ValueError(f"snapshot {label} bids must be sorted by descending price")
            if len(prices) != len(set(prices)):
                raise ValueError(f"snapshot {label} bid prices must be unique")
        return self


class OrderBookDeltaEvent(_OrderBookEventBase):
    """One normalized signed change to an internal YES or NO bid level."""

    event_type: Literal["delta"] = "delta"
    side: BookSide
    price: Decimal = Field(ge=0, le=1)
    quantity_delta: Decimal

    @field_validator("price")
    @classmethod
    def require_finite_price(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("delta price must be finite")
        return value

    @field_validator("quantity_delta")
    @classmethod
    def require_signed_nonzero_delta(cls, value: Decimal) -> Decimal:
        if not value.is_finite() or value == 0:
            raise ValueError("quantity_delta must be finite and nonzero")
        return value


MarketDataEvent = Annotated[
    OrderBookSnapshotEvent | OrderBookDeltaEvent,
    Field(discriminator="event_type"),
]

_MARKET_DATA_EVENT_ADAPTER: TypeAdapter[MarketDataEvent] = TypeAdapter(MarketDataEvent)


def _fixed_decimal(value: Decimal, *, places: int, label: str) -> str:
    """Render exactly at the fixed scale used by the immutable Arrow schema."""

    quantum = Decimal(1).scaleb(-places)
    with localcontext() as context:
        context.prec = max(50, len(value.as_tuple().digits) + places)
        fixed = value.quantize(quantum)
    if fixed != value:
        raise ValueError(f"{label} exceeds the persisted {places}-decimal scale")
    return format(fixed, "f")


def _market_data_event_payload(event: MarketDataEvent) -> dict[str, object]:
    """Return JSON data whose Decimal spelling survives the Parquet round trip."""

    payload: dict[str, object] = event.model_dump(mode="json")
    if isinstance(event, OrderBookSnapshotEvent):
        for field_name, levels in (("yes_bids", event.yes_bids), ("no_bids", event.no_bids)):
            payload[field_name] = [
                {
                    "price": _fixed_decimal(level.price, places=4, label="book price"),
                    "quantity": _fixed_decimal(
                        level.quantity,
                        places=2,
                        label="book quantity",
                    ),
                }
                for level in levels
            ]
    else:
        payload["price"] = _fixed_decimal(event.price, places=4, label="delta price")
        payload["quantity_delta"] = _fixed_decimal(
            event.quantity_delta,
            places=2,
            label="delta quantity",
        )
    return payload


def dump_market_data_event_json(event: MarketDataEvent) -> str:
    """Serialize an event with stable key order and exact JSON-safe scalar strings."""

    return json.dumps(
        _market_data_event_payload(event),
        separators=(",", ":"),
        sort_keys=True,
    )


def load_market_data_event_json(payload: str | bytes | bytearray) -> MarketDataEvent:
    """Parse strict JSON through the discriminated normalized-event union."""

    return _MARKET_DATA_EVENT_ADAPTER.validate_json(payload)


_SENSITIVE_INPUT_KEYS = frozenset(
    {
        "api_key",
        "api_key_id",
        "credential",
        "credentials",
        "password",
        "private_key",
        "private_key_path",
        "secret",
        "token",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value using Arbiter's canonical JSON encoding."""

    return sha256(_canonical_json(value).encode()).hexdigest()


def _reject_sensitive_payload(value: object, *, path: str) -> None:
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if key in _SENSITIVE_INPUT_KEYS or "private_key" in key:
                raise ValueError(
                    f"recorded run inputs cannot contain sensitive key {path}.{raw_key}"
                )
            _reject_sensitive_payload(child, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_payload(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and "-----BEGIN" in value and "PRIVATE KEY-----" in value:
        raise ValueError(f"recorded run inputs cannot contain private key material at {path}")


def _require_canonical_object_json(value: str, *, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} cannot be blank")
    try:
        decoded = json.loads(value)
    except JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must encode a JSON object")
    canonical = _canonical_json(decoded)
    if value != canonical:
        raise ValueError(f"{label} must use canonical JSON encoding")
    _reject_sensitive_payload(decoded, path=label)
    return value


def _require_sorted_unique(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{label} cannot be empty")
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} cannot contain blank identifiers")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{label} must be unique and canonically sorted")
    return values


class EventIndexAllocator:
    """Allocate the shared contiguous index sequence for all live records."""

    __slots__ = ("_lock", "_next_event_index")

    def __init__(self, start_event_index: int = 0) -> None:
        if (
            isinstance(start_event_index, bool)
            or not isinstance(start_event_index, int)
            or start_event_index < 0
        ):
            raise ValueError("start_event_index must be a nonnegative integer")
        self._next_event_index = start_event_index
        self._lock = Lock()

    @property
    def next_event_index(self) -> int:
        """Return the next index without consuming it."""

        with self._lock:
            return self._next_event_index

    def allocate(self) -> int:
        """Consume and return exactly one index."""

        with self._lock:
            allocated = self._next_event_index
            self._next_event_index += 1
            return allocated


class RunInputPayload(BaseModel):
    """Canonical, non-secret inputs needed to reconstruct one historical run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_json: str = Field(min_length=2)
    markets: tuple[Market, ...]
    events: tuple[Event, ...]
    series: tuple[Series, ...]
    relations: tuple[Relation, ...]
    fee_policy_json: str = Field(min_length=2)

    @field_validator("config_json")
    @classmethod
    def validate_config_json(cls, value: str) -> str:
        return _require_canonical_object_json(value, label="configuration JSON")

    @field_validator("fee_policy_json")
    @classmethod
    def validate_fee_policy_json(cls, value: str) -> str:
        return _require_canonical_object_json(value, label="fee-policy JSON")

    @model_validator(mode="after")
    def validate_canonical_order(self) -> RunInputPayload:
        keyed_inputs = (
            ("markets", tuple(item.ticker for item in self.markets)),
            ("events", tuple(item.ticker for item in self.events)),
            ("series", tuple(item.ticker for item in self.series)),
            ("relations", tuple(item.relation_id for item in self.relations)),
        )
        for label, identifiers in keyed_inputs:
            if identifiers != tuple(sorted(set(identifiers))):
                raise ValueError(f"run input {label} must be unique and canonically sorted")
        for event in self.events:
            if event.market_tickers != tuple(sorted(set(event.market_tickers))):
                raise ValueError("run input event membership must be unique and canonically sorted")
        return self

    @classmethod
    def build(
        cls,
        *,
        config: Mapping[str, object],
        markets: Sequence[Market],
        events: Sequence[Event],
        series: Sequence[Series],
        relations: Sequence[Relation],
        fee_policy: Mapping[str, object],
    ) -> RunInputPayload:
        """Canonicalize caller-owned collections into an immutable payload."""

        return cls(
            config_json=_canonical_json(dict(config)),
            markets=tuple(sorted(markets, key=lambda item: item.ticker)),
            events=tuple(sorted(events, key=lambda item: item.ticker)),
            series=tuple(sorted(series, key=lambda item: item.ticker)),
            relations=tuple(sorted(relations, key=lambda item: item.relation_id)),
            fee_policy_json=_canonical_json(dict(fee_policy)),
        )

    @property
    def config(self) -> dict[str, object]:
        """Return a fresh decoded configuration object."""

        return cast(dict[str, object], json.loads(self.config_json))

    @property
    def fee_policy(self) -> dict[str, object]:
        """Return a fresh decoded fee-policy object."""

        return cast(dict[str, object], json.loads(self.fee_policy_json))

    def computed_hashes(self) -> dict[str, str]:
        """Return hashes over the exact canonical payload used by live composition."""

        metadata = {
            "markets": [item.model_dump(mode="json") for item in self.markets],
            "events": [item.model_dump(mode="json") for item in self.events],
            "series": [item.model_dump(mode="json") for item in self.series],
        }
        return {
            "config_hash": sha256(self.config_json.encode()).hexdigest(),
            "metadata_hash": canonical_sha256(metadata),
            "relations_hash": canonical_sha256(
                [item.model_dump(mode="json") for item in self.relations]
            ),
            "fee_policy_hash": sha256(self.fee_policy_json.encode()).hexdigest(),
        }


class _ControlEventBase(BaseModel):
    """Shared total-order fields for schema-v2 controls."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[2] = RECORDED_CONTROL_SCHEMA_VERSION
    event_index: int = Field(ge=0)
    local_received_ts: datetime

    @field_validator("local_received_ts")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recorded control timestamps must be timezone-aware")
        return value


class RunStartedEvent(_ControlEventBase):
    """Immutable run identity and every non-secret deterministic input."""

    event_type: Literal["run_started"] = "run_started"
    run_id: str = Field(min_length=1)
    inputs: RunInputPayload
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    relations_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fee_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("run_id")
    @classmethod
    def require_nonblank_run_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("run ID cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_input_hashes(self) -> RunStartedEvent:
        expected = self.inputs.computed_hashes()
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"run-start {field_name} does not match its embedded inputs")
        return self

    @classmethod
    def build(
        cls,
        *,
        event_index: int,
        local_received_ts: datetime,
        run_id: str,
        inputs: RunInputPayload,
    ) -> RunStartedEvent:
        """Construct a run-start record with hashes derived from its immutable inputs."""

        hashes = inputs.computed_hashes()
        return cls(
            event_index=event_index,
            local_received_ts=local_received_ts,
            run_id=run_id,
            inputs=inputs,
            config_hash=hashes["config_hash"],
            metadata_hash=hashes["metadata_hash"],
            relations_hash=hashes["relations_hash"],
            fee_policy_hash=hashes["fee_policy_hash"],
        )


class SubscriptionStartedEvent(_ControlEventBase):
    event_type: Literal["subscription_started"] = "subscription_started"
    connection_id: str = Field(min_length=1)
    sid: int = Field(gt=0)
    tickers: tuple[str, ...]

    @field_validator("connection_id")
    @classmethod
    def require_nonblank_connection_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("connection ID cannot be blank")
        return value

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_sorted_unique(value, label="subscription tickers")


class ConnectionInterruptedEvent(_ControlEventBase):
    event_type: Literal["connection_interrupted"] = "connection_interrupted"
    connection_id: str = Field(min_length=1)
    tickers: tuple[str, ...]
    reason: str = Field(min_length=1)

    @field_validator("connection_id", "reason")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("connection interruption identifiers cannot be blank")
        return value

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_sorted_unique(value, label="interrupted tickers")


class BookStaleEvent(_ControlEventBase):
    event_type: Literal["book_stale"] = "book_stale"
    ticker: str = Field(min_length=1)
    source_event_index: int = Field(ge=0)
    source_connection_id: str = Field(min_length=1)
    source_sid: int = Field(gt=0)
    source_snapshot_id: str = Field(min_length=1)
    source_sequence: int = Field(gt=0)

    @field_validator("ticker", "source_connection_id", "source_snapshot_id")
    @classmethod
    def require_nonblank_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("stale-book identities cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_source_precedes_control(self) -> BookStaleEvent:
        if self.source_event_index >= self.event_index:
            raise ValueError("stale-book source event must precede its control")
        return self


class MarketRefreshStartedEvent(_ControlEventBase):
    event_type: Literal["market_refresh_started"] = "market_refresh_started"
    ticker: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    connection_id: str = Field(min_length=1)
    sid: int = Field(gt=0)

    @field_validator("ticker", "reason", "connection_id")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("market-refresh identifiers cannot be blank")
        return value


class MarketRefreshAppliedEvent(_ControlEventBase):
    event_type: Literal["market_refresh_applied"] = "market_refresh_applied"
    refresh_started_event_index: int = Field(ge=0)
    market: Market

    @model_validator(mode="after")
    def validate_start_precedes_applied(self) -> MarketRefreshAppliedEvent:
        if self.refresh_started_event_index >= self.event_index:
            raise ValueError("market refresh start must precede its applied record")
        return self


class FeeRefreshStartedEvent(_ControlEventBase):
    event_type: Literal["fee_refresh_started"] = "fee_refresh_started"
    event_ticker: str = Field(min_length=1)
    affected_tickers: tuple[str, ...]
    connection_id: str = Field(min_length=1)
    sid: int = Field(gt=0)

    @field_validator("event_ticker", "connection_id")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("fee-refresh identifiers cannot be blank")
        return value

    @field_validator("affected_tickers")
    @classmethod
    def validate_affected_tickers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _require_sorted_unique(value, label="fee-refresh affected tickers")


class FeeRefreshAppliedEvent(_ControlEventBase):
    event_type: Literal["fee_refresh_applied"] = "fee_refresh_applied"
    refresh_started_event_index: int = Field(ge=0)
    event: Event

    @model_validator(mode="after")
    def validate_start_precedes_applied(self) -> FeeRefreshAppliedEvent:
        if self.refresh_started_event_index >= self.event_index:
            raise ValueError("fee refresh start must precede its applied record")
        if self.event.market_tickers != tuple(sorted(set(self.event.market_tickers))):
            raise ValueError("applied event membership must be unique and canonically sorted")
        return self


class RunEndedEvent(_ControlEventBase):
    event_type: Literal["run_ended"] = "run_ended"
    run_id: str = Field(min_length=1)
    status: RunEndStatus
    reason: str | None = Field(default=None, min_length=1)

    @field_validator("run_id")
    @classmethod
    def require_nonblank_run_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("run ID cannot be blank")
        return value

    @field_validator("reason")
    @classmethod
    def require_nonblank_reason(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("run-end reason cannot be blank")
        return value


ControlEvent = Annotated[
    RunStartedEvent
    | SubscriptionStartedEvent
    | ConnectionInterruptedEvent
    | BookStaleEvent
    | MarketRefreshStartedEvent
    | MarketRefreshAppliedEvent
    | FeeRefreshStartedEvent
    | FeeRefreshAppliedEvent
    | RunEndedEvent,
    Field(discriminator="event_type"),
]

RecordedEvent = Annotated[
    OrderBookSnapshotEvent
    | OrderBookDeltaEvent
    | RunStartedEvent
    | SubscriptionStartedEvent
    | ConnectionInterruptedEvent
    | BookStaleEvent
    | MarketRefreshStartedEvent
    | MarketRefreshAppliedEvent
    | FeeRefreshStartedEvent
    | FeeRefreshAppliedEvent
    | RunEndedEvent,
    Field(discriminator="event_type"),
]

_CONTROL_EVENT_ADAPTER: TypeAdapter[ControlEvent] = TypeAdapter(ControlEvent)
_RECORDED_EVENT_ADAPTER: TypeAdapter[RecordedEvent] = TypeAdapter(RecordedEvent)


def dump_control_event_json(event: ControlEvent) -> str:
    """Serialize a v2 control with stable key order and exact JSON-safe values."""

    return _canonical_json(event.model_dump(mode="json"))


def load_control_event_json(payload: str | bytes | bytearray) -> ControlEvent:
    """Parse strict JSON through the schema-v2 control union."""

    return _CONTROL_EVENT_ADAPTER.validate_json(payload)


def dump_recorded_event_json(event: RecordedEvent) -> str:
    """Serialize any v1/v2 record deterministically."""

    payload = (
        _market_data_event_payload(event)
        if isinstance(event, (OrderBookSnapshotEvent, OrderBookDeltaEvent))
        else event.model_dump(mode="json")
    )
    return _canonical_json(payload)


def load_recorded_event_json(payload: str | bytes | bytearray) -> RecordedEvent:
    """Parse strict JSON through the complete recorded-stream union."""

    return _RECORDED_EVENT_ADAPTER.validate_json(payload)
