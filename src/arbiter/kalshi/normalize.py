"""Strict normalization from tolerant Kalshi payloads to Arbiter domain models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from arbiter.kalshi.schemas import (
    EventFeeChangeWire,
    EventWire,
    MarketWire,
    OrderBookResponse,
    PriceRangeWire,
    SeriesWire,
    WebSocketOrderBookDeltaWire,
    WebSocketOrderBookSnapshotWire,
)
from arbiter.models.event import Event, EventFeeChange
from arbiter.models.market import Market, PriceRange
from arbiter.models.orderbook import OrderBook, PriceLevel
from arbiter.models.series import Series, SettlementSource
from arbiter.replay.events import OrderBookDeltaEvent, OrderBookSnapshotEvent


class NormalizationError(ValueError):
    """Raised when an exchange payload cannot be made safe for domain use."""


def _decimal(value: str | int | float, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise NormalizationError(f"{field} must be a decimal value")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise NormalizationError(f"{field} is not a valid decimal") from exc
    if not parsed.is_finite():
        raise NormalizationError(f"{field} must be finite")
    return parsed


def _fixed(value: str, *, field: str, decimal_places: int) -> Decimal:
    if not isinstance(value, str):
        raise NormalizationError(f"{field} must be a fixed-point string")
    parsed = _decimal(value, field=field)
    exponent = parsed.as_tuple().exponent
    if not isinstance(exponent, int):
        raise NormalizationError(f"{field} must be finite")
    if exponent < -decimal_places:
        raise NormalizationError(f"{field} exceeds {decimal_places} decimal places")
    return parsed


def dollars(value: str, *, field: str = "price_dollars") -> Decimal:
    """Parse a Kalshi ``*_dollars`` field without a binary-float round trip."""

    parsed = _fixed(value, field=field, decimal_places=4)
    if not Decimal("0") <= parsed <= Decimal("1"):
        raise NormalizationError(f"{field} must be between 0 and 1")
    return parsed


def fixed_quantity(value: str, *, field: str = "quantity_fp") -> Decimal:
    """Parse a nonnegative Kalshi ``*_fp`` contract quantity at 0.01 granularity."""

    parsed = _fixed(value, field=field, decimal_places=2)
    if parsed < 0:
        raise NormalizationError(f"{field} cannot be negative")
    return parsed


def signed_fixed_quantity(value: str, *, field: str = "delta_fp") -> Decimal:
    """Parse a signed, nonzero fixed-point quantity at Kalshi's 0.01 granularity."""

    parsed = _fixed(value, field=field, decimal_places=2)
    if parsed == 0:
        raise NormalizationError(f"{field} must be nonzero")
    return parsed


def _optional_decimal(value: str | int | float | None, *, field: str) -> Decimal | None:
    return None if value is None else _decimal(value, field=field)


def _raw(payload: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(payload))


def normalize_price_range(wire: PriceRangeWire) -> PriceRange:
    """Normalize one market tick interval."""

    try:
        return PriceRange(
            start=_decimal(wire.start, field="price_ranges.start"),
            end=_decimal(wire.end, field="price_ranges.end"),
            step=_decimal(wire.step, field="price_ranges.step"),
        )
    except ValidationError as exc:
        raise NormalizationError(f"invalid price range: {exc}") from exc


def normalize_market(
    payload: Mapping[str, Any],
    *,
    series_ticker: str | None = None,
) -> Market:
    """Validate and preserve one raw market payload, enriching its event series."""

    try:
        wire = MarketWire.model_validate(payload)
        effective_expiration = (
            wire.latest_expiration_time or wire.expected_expiration_time or wire.expiration_time
        )
        return Market(
            ticker=wire.ticker,
            event_ticker=wire.event_ticker,
            series_ticker=series_ticker,
            market_type=wire.market_type,
            title=wire.title or wire.ticker,
            subtitle=wire.subtitle,
            yes_sub_title=wire.yes_sub_title,
            no_sub_title=wire.no_sub_title,
            status=wire.status,
            created_time=wire.created_time,
            updated_time=wire.updated_time,
            open_time=wire.open_time,
            close_time=wire.close_time,
            expiration_time=effective_expiration,
            latest_expiration_time=wire.latest_expiration_time,
            expected_expiration_time=wire.expected_expiration_time,
            settlement_ts=wire.settlement_ts,
            occurrence_datetime=wire.occurrence_datetime,
            strike_type=wire.strike_type,
            floor_strike=_optional_decimal(wire.floor_strike, field="floor_strike"),
            cap_strike=_optional_decimal(wire.cap_strike, field="cap_strike"),
            functional_strike=wire.functional_strike,
            custom_strike=wire.custom_strike,
            rules_primary=wire.rules_primary,
            rules_secondary=wire.rules_secondary,
            early_close_condition=wire.early_close_condition,
            price_level_structure=wire.price_level_structure,
            price_ranges=tuple(normalize_price_range(item) for item in wire.price_ranges),
            result=wire.result,
            raw=_raw(payload),
        )
    except (ValidationError, NormalizationError) as exc:
        if isinstance(exc, NormalizationError):
            raise
        raise NormalizationError(f"invalid market payload: {exc}") from exc


def normalize_event(
    payload: Mapping[str, Any],
    *,
    top_level_markets: Sequence[Mapping[str, Any]] = (),
) -> Event:
    """Normalize an event and collect market tickers from either response shape."""

    try:
        wire = EventWire.model_validate(payload)
        market_tickers = {market.ticker for market in wire.markets}
        market_tickers.update(
            MarketWire.model_validate(market).ticker for market in top_level_markets
        )
        return Event(
            ticker=wire.event_ticker,
            series_ticker=wire.series_ticker,
            title=wire.title or wire.event_ticker,
            subtitle=wire.sub_title,
            category=wire.category,
            mutually_exclusive=wire.mutually_exclusive,
            available_on_brokers=wire.available_on_brokers,
            market_tickers=tuple(sorted(market_tickers)),
            last_updated_ts=wire.last_updated_ts,
            fee_type_override=wire.fee_type_override,
            fee_multiplier_override=_optional_decimal(
                wire.fee_multiplier_override,
                field="event.fee_multiplier",
            ),
            raw=_raw(payload),
        )
    except ValidationError as exc:
        raise NormalizationError(f"invalid event payload: {exc}") from exc


def normalize_event_fee_change(payload: Mapping[str, Any]) -> EventFeeChange:
    """Normalize one scheduled event override while preserving its source record."""

    try:
        wire = EventFeeChangeWire.model_validate(payload)
        return EventFeeChange(
            change_id=wire.id,
            event_ticker=wire.event_ticker,
            series_ticker=wire.series_ticker,
            scheduled_ts=wire.scheduled_ts,
            fee_type_override=wire.fee_type_override,
            fee_multiplier_override=_optional_decimal(
                wire.fee_multiplier_override,
                field="event_fee_change.fee_multiplier",
            ),
            raw=_raw(payload),
        )
    except (ValidationError, NormalizationError) as exc:
        if isinstance(exc, NormalizationError):
            raise
        raise NormalizationError(f"invalid event fee change payload: {exc}") from exc


def normalize_series(payload: Mapping[str, Any]) -> Series:
    """Normalize series fee/settlement metadata and preserve unknown raw fields."""

    try:
        wire = SeriesWire.model_validate(payload)
        return Series(
            ticker=wire.ticker,
            title=wire.title or wire.ticker,
            frequency=wire.frequency,
            category=wire.category,
            tags=tuple(wire.tags or ()),
            fee_type=wire.fee_type,
            fee_multiplier=_optional_decimal(wire.fee_multiplier, field="fee_multiplier"),
            settlement_sources=tuple(
                SettlementSource(
                    name=source.name,
                    url=source.url,
                    **(source.model_extra or {}),
                )
                for source in (wire.settlement_sources or ())
            ),
            contract_url=wire.contract_url,
            contract_terms_url=wire.contract_terms_url,
            last_updated_ts=wire.last_updated_ts,
            raw=_raw(payload),
        )
    except ValidationError as exc:
        raise NormalizationError(f"invalid series payload: {exc}") from exc


def _normalize_levels(
    levels: Sequence[tuple[str, str]],
    *,
    side: str,
    market: Market,
) -> tuple[PriceLevel, ...]:
    aggregated: dict[Decimal, Decimal] = {}
    for price_text, quantity_text in levels:
        price = dollars(price_text, field=f"{side}.price_dollars")
        quantity = fixed_quantity(quantity_text, field=f"{side}.quantity_fp")
        if not market.accepts_price(price):
            raise NormalizationError(
                f"{side} bid price {price} is not valid for market {market.ticker} price ranges"
            )
        if quantity == 0:
            continue
        aggregated[price] = aggregated.get(price, Decimal("0")) + quantity
    return tuple(
        PriceLevel(price=price, quantity=aggregated[price])
        for price in sorted(aggregated, reverse=True)
    )


def normalize_orderbook(
    ticker: str,
    payload: Mapping[str, Any],
    *,
    market: Market,
    local_timestamp: datetime,
    sequence: int | None = None,
    exchange_timestamp: datetime | None = None,
) -> OrderBook:
    """Normalize one fixed-point REST snapshot and derive no nonexistent liquidity."""

    if ticker != market.ticker:
        raise NormalizationError("order-book ticker does not match market metadata")
    try:
        wire = OrderBookResponse.model_validate(payload)
    except ValidationError as exc:
        raise NormalizationError(f"invalid order-book payload: {exc}") from exc
    if not market.price_ranges and (wire.orderbook_fp.yes_dollars or wire.orderbook_fp.no_dollars):
        raise NormalizationError("cannot validate a nonempty order book without price ranges")
    return OrderBook(
        ticker=ticker,
        yes_bids=_normalize_levels(
            wire.orderbook_fp.yes_dollars,
            side="yes",
            market=market,
        ),
        no_bids=_normalize_levels(
            wire.orderbook_fp.no_dollars,
            side="no",
            market=market,
        ),
        sequence=sequence,
        exchange_timestamp=exchange_timestamp,
        local_timestamp=local_timestamp,
        raw=_raw(payload),
    )


def _require_aware(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise NormalizationError(f"{field} must be timezone-aware")
    return value


def _websocket_exchange_timestamp(
    *,
    timestamp: datetime | None,
    timestamp_ms: int | None,
) -> datetime | None:
    try:
        parsed_ms = (
            datetime.fromtimestamp(timestamp_ms / 1_000, tz=UTC)
            if timestamp_ms is not None
            else None
        )
    except (OSError, OverflowError, ValueError) as exc:
        raise NormalizationError("msg.ts_ms is outside the supported timestamp range") from exc
    if timestamp is None:
        return parsed_ms
    aware = _require_aware(timestamp, field="msg.ts")
    if parsed_ms is not None and aware.astimezone(UTC) != parsed_ms:
        raise NormalizationError("msg.ts and msg.ts_ms identify different timestamps")
    return aware


def _normalize_websocket_levels(
    levels: Sequence[tuple[str, str]],
    *,
    side: str,
    market: Market,
) -> tuple[PriceLevel, ...]:
    aggregated: dict[Decimal, Decimal] = {}
    for price_text, quantity_text in levels:
        wire_yes_price = dollars(price_text, field=f"{side}.price_dollars")
        if not market.accepts_price(wire_yes_price):
            raise NormalizationError(
                f"{side} wire YES price {wire_yes_price} is not valid for market "
                f"{market.ticker} price ranges"
            )
        quantity = fixed_quantity(quantity_text, field=f"{side}.quantity_fp")
        if quantity == 0:
            continue
        internal_price = wire_yes_price if side == "yes" else Decimal("1") - wire_yes_price
        aggregated[internal_price] = aggregated.get(internal_price, Decimal("0")) + quantity
    return tuple(
        PriceLevel(price=price, quantity=aggregated[price])
        for price in sorted(aggregated, reverse=True)
    )


def normalize_websocket_snapshot(
    payload: Mapping[str, Any],
    *,
    event_index: int,
    connection_id: str,
    local_received_ts: datetime,
    market: Market,
) -> OrderBookSnapshotEvent:
    """Normalize a current ``use_yes_price`` snapshot into Arbiter's internal book convention."""

    try:
        wire = WebSocketOrderBookSnapshotWire.model_validate(payload)
        if wire.msg.market_ticker != market.ticker:
            raise NormalizationError("WebSocket snapshot ticker does not match market metadata")
        yes_levels = wire.msg.yes_dollars_fp or []
        no_levels = wire.msg.no_dollars_fp or []
        if not market.price_ranges and (yes_levels or no_levels):
            raise NormalizationError("cannot validate a nonempty snapshot without price ranges")
        snapshot_id = f"{connection_id}:{wire.sid}:{wire.msg.market_ticker}:{wire.seq}"
        return OrderBookSnapshotEvent(
            event_index=event_index,
            local_received_ts=_require_aware(
                local_received_ts,
                field="local_received_ts",
            ),
            exchange_ts=None,
            ticker=wire.msg.market_ticker,
            sequence=wire.seq,
            sid=wire.sid,
            connection_id=connection_id,
            price_convention="yes_price",
            snapshot_id=snapshot_id,
            yes_bids=_normalize_websocket_levels(
                yes_levels,
                side="yes",
                market=market,
            ),
            no_bids=_normalize_websocket_levels(
                no_levels,
                side="no",
                market=market,
            ),
        )
    except ValidationError as exc:
        raise NormalizationError(f"invalid WebSocket order-book snapshot: {exc}") from exc


def normalize_websocket_delta(
    payload: Mapping[str, Any],
    *,
    event_index: int,
    connection_id: str,
    local_received_ts: datetime,
    market: Market,
    snapshot_id: str,
) -> OrderBookDeltaEvent:
    """Normalize a signed ``use_yes_price`` delta and complement NO exactly once."""

    try:
        wire = WebSocketOrderBookDeltaWire.model_validate(payload)
        if wire.msg.market_ticker != market.ticker:
            raise NormalizationError("WebSocket delta ticker does not match market metadata")
        wire_yes_price = dollars(wire.msg.price_dollars)
        if not market.price_ranges:
            raise NormalizationError("cannot validate a delta without price ranges")
        if not market.accepts_price(wire_yes_price):
            raise NormalizationError(
                f"delta wire YES price {wire_yes_price} is not valid for market "
                f"{market.ticker} price ranges"
            )
        return OrderBookDeltaEvent(
            event_index=event_index,
            local_received_ts=_require_aware(
                local_received_ts,
                field="local_received_ts",
            ),
            exchange_ts=_websocket_exchange_timestamp(
                timestamp=wire.msg.ts,
                timestamp_ms=wire.msg.ts_ms,
            ),
            ticker=wire.msg.market_ticker,
            sequence=wire.seq,
            sid=wire.sid,
            connection_id=connection_id,
            price_convention="yes_price",
            snapshot_id=snapshot_id,
            side=wire.msg.side,
            price=(wire_yes_price if wire.msg.side == "yes" else Decimal("1") - wire_yes_price),
            quantity_delta=signed_fixed_quantity(wire.msg.delta_fp),
        )
    except ValidationError as exc:
        raise NormalizationError(f"invalid WebSocket order-book delta: {exc}") from exc
