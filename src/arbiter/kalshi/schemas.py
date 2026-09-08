"""Tolerant current Kalshi REST wire schemas.

Wire models intentionally allow unknown fields so additive exchange changes do not break
normalization. Domain models remain strict.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

JsonNumber = str | int | float


class WireModel(BaseModel):
    """Base for additive external payloads."""

    model_config = ConfigDict(extra="allow")


class PriceRangeWire(WireModel):
    start: JsonNumber
    end: JsonNumber
    step: JsonNumber


class MarketWire(WireModel):
    ticker: str
    event_ticker: str
    market_type: str | None = None
    title: str = ""
    subtitle: str | None = None
    yes_sub_title: str | None = None
    no_sub_title: str | None = None
    status: str = "initialized"
    created_time: datetime | None = None
    updated_time: datetime | None = None
    open_time: datetime | None = None
    close_time: datetime | None = None
    latest_expiration_time: datetime | None = None
    expiration_time: datetime | None = None
    expected_expiration_time: datetime | None = None
    settlement_ts: datetime | None = None
    occurrence_datetime: datetime | None = None
    strike_type: str | None = None
    floor_strike: JsonNumber | None = None
    cap_strike: JsonNumber | None = None
    functional_strike: str | None = None
    custom_strike: dict[str, Any] | None = None
    rules_primary: str | None = None
    rules_secondary: str | None = None
    early_close_condition: str | None = None
    price_level_structure: str | None = None
    price_ranges: list[PriceRangeWire] = Field(default_factory=list)
    result: str | None = None


class EventWire(WireModel):
    event_ticker: str
    series_ticker: str | None = None
    title: str = ""
    sub_title: str | None = None
    category: str | None = None
    mutually_exclusive: bool | None = None
    available_on_brokers: bool | None = None
    markets: list[MarketWire] = Field(default_factory=list)
    last_updated_ts: datetime | None = None
    fee_type_override: str | None = Field(
        default=None,
        validation_alias=AliasChoices("fee_type_override", "fee_type"),
    )
    fee_multiplier_override: JsonNumber | None = Field(
        default=None,
        validation_alias=AliasChoices("fee_multiplier_override", "fee_multiplier"),
    )


class EventFeeChangeWire(WireModel):
    id: str
    event_ticker: str
    series_ticker: str
    scheduled_ts: datetime
    fee_type_override: str | None = Field(
        validation_alias=AliasChoices("fee_type_override", "fee_type"),
    )
    fee_multiplier_override: JsonNumber | None = Field(
        validation_alias=AliasChoices("fee_multiplier_override", "fee_multiplier"),
    )


class SettlementSourceWire(WireModel):
    name: str = ""
    url: str = ""


class SeriesWire(WireModel):
    ticker: str
    frequency: str | None = None
    title: str = ""
    category: str | None = None
    tags: list[str] | None = None
    settlement_sources: list[SettlementSourceWire] | None = None
    contract_url: str | None = None
    contract_terms_url: str | None = None
    fee_type: str | None = None
    fee_multiplier: JsonNumber | None = None
    last_updated_ts: datetime | None = None


class MarketsResponse(WireModel):
    markets: list[MarketWire] = Field(default_factory=list)
    cursor: str = ""


class MarketResponse(WireModel):
    market: MarketWire


class EventsResponse(WireModel):
    events: list[EventWire] = Field(default_factory=list)
    cursor: str = ""


class EventResponse(WireModel):
    event: EventWire
    markets: list[MarketWire] = Field(default_factory=list)


class EventFeeChangesResponse(WireModel):
    event_fee_changes: list[EventFeeChangeWire] = Field(default_factory=list)
    cursor: str = ""


class SeriesResponse(WireModel):
    series: SeriesWire


class OrderBookFixedPointWire(WireModel):
    yes_dollars: list[tuple[str, str]] = Field(default_factory=list)
    no_dollars: list[tuple[str, str]] = Field(default_factory=list)


class OrderBookResponse(WireModel):
    orderbook_fp: OrderBookFixedPointWire


class WebSocketOrderBookSnapshotBodyWire(WireModel):
    """Current fixed-point snapshot body when ``use_yes_price`` is enabled."""

    market_ticker: str
    market_id: str
    yes_dollars_fp: list[tuple[str, str]] | None = None
    no_dollars_fp: list[tuple[str, str]] | None = None


class WebSocketOrderBookSnapshotWire(WireModel):
    type: Literal["orderbook_snapshot"]
    sid: int = Field(gt=0)
    seq: int = Field(gt=0)
    msg: WebSocketOrderBookSnapshotBodyWire


class WebSocketOrderBookDeltaBodyWire(WireModel):
    """Current signed fixed-point delta body on the unified YES-price scale."""

    market_ticker: str
    market_id: str
    price_dollars: str
    delta_fp: str
    side: Literal["yes", "no"]
    ts: datetime | None = None
    ts_ms: int | None = Field(default=None, ge=0)


class WebSocketOrderBookDeltaWire(WireModel):
    type: Literal["orderbook_delta"]
    sid: int = Field(gt=0)
    seq: int = Field(gt=0)
    msg: WebSocketOrderBookDeltaBodyWire


class WebSocketSubscribedBodyWire(WireModel):
    channel: str
    sid: int = Field(gt=0)


class WebSocketSubscribedWire(WireModel):
    type: Literal["subscribed"]
    id: int | None = Field(default=None, ge=0)
    msg: WebSocketSubscribedBodyWire


class WebSocketErrorBodyWire(WireModel):
    code: int
    msg: str


class WebSocketErrorWire(WireModel):
    type: Literal["error"]
    id: int | None = None
    msg: WebSocketErrorBodyWire


class WebSocketOkWire(WireModel):
    type: Literal["ok"]
    id: int | None = Field(default=None, ge=0)
    sid: int | None = Field(default=None, gt=0)
    seq: int | None = Field(default=None, gt=0)
    msg: Any = None


class WebSocketLifecycleBodyWire(WireModel):
    """Fields relevant to deciding whether subscribed market metadata became unsafe."""

    event_type: str
    market_ticker: str
    price_level_structure: str | None = None
    price_ranges: list[PriceRangeWire] | None = None


class WebSocketLifecycleWire(WireModel):
    type: Literal["market_lifecycle_v2", "multivariate_market_lifecycle"]
    sid: int = Field(gt=0)
    seq: int | None = Field(default=None, gt=0)
    msg: WebSocketLifecycleBodyWire


class WebSocketEventLifecycleBodyWire(WireModel):
    """Documented event-creation notification shared by both lifecycle channels."""

    event_ticker: str = Field(min_length=1)
    exchange_index: int
    title: str
    subtitle: str
    collateral_return_type: Literal["MECNET", "DIRECNET", ""]
    series_ticker: str = Field(min_length=1)


class WebSocketEventLifecycleWire(WireModel):
    type: Literal["event_lifecycle"]
    sid: int = Field(gt=0)
    seq: int | None = Field(default=None, gt=0)
    msg: WebSocketEventLifecycleBodyWire


class WebSocketEventFeeUpdateBodyWire(WireModel):
    """Complete event-level fee override, or a paired null override clear."""

    event_ticker: str = Field(min_length=1)
    fee_type_override: str | None
    fee_multiplier_override: Decimal | None

    @model_validator(mode="after")
    def validate_override_pair(self) -> WebSocketEventFeeUpdateBodyWire:
        if (self.fee_type_override is None) != (self.fee_multiplier_override is None):
            raise ValueError("event fee override values must both be present or both be null")
        if self.fee_type_override is not None and not self.fee_type_override.strip():
            raise ValueError("event fee type override cannot be blank")
        if self.fee_multiplier_override is not None and (
            not self.fee_multiplier_override.is_finite()
            or self.fee_multiplier_override < Decimal("0")
        ):
            raise ValueError("event fee multiplier override must be finite and nonnegative")
        return self


class WebSocketEventFeeUpdateWire(WireModel):
    type: Literal["event_fee_update"]
    sid: int = Field(gt=0)
    seq: int | None = Field(default=None, gt=0)
    msg: WebSocketEventFeeUpdateBodyWire
