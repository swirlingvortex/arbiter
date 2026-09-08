"""Bounded, injectable client for Kalshi public REST metadata and books."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from arbiter.kalshi.normalize import (
    NormalizationError,
    normalize_event,
    normalize_event_fee_change,
    normalize_market,
    normalize_orderbook,
    normalize_series,
)
from arbiter.kalshi.schemas import (
    EventFeeChangesResponse,
    EventResponse,
    EventsResponse,
    MarketResponse,
    MarketsResponse,
    OrderBookResponse,
    SeriesResponse,
)
from arbiter.models.event import Event, EventFeeChange
from arbiter.models.market import Market
from arbiter.models.orderbook import OrderBook
from arbiter.models.series import Series


class KalshiRestError(RuntimeError):
    """Base error for finite public REST operations."""


class KalshiHttpError(KalshiRestError):
    """A nonretryable or exhausted HTTP response."""

    def __init__(self, status_code: int, path: str, detail: str) -> None:
        super().__init__(f"Kalshi GET {path} returned HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.path = path


class KalshiTransportError(KalshiRestError):
    """A safe GET exhausted bounded transport retries."""


class KalshiPayloadError(KalshiRestError):
    """A successful HTTP response did not match the expected JSON contract."""


class KalshiPaginationError(KalshiRestError):
    """Cursor pagination failed closed rather than looping forever."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class KalshiRestClient:
    """Current public Trade API v2 client with deterministic injection points."""

    _RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout_seconds: float = 10.0,
        max_attempts: int = 3,
        backoff_seconds: float = 0.1,
        max_backoff_seconds: float = 2.0,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if backoff_seconds < 0 or max_backoff_seconds < 0:
            raise ValueError("backoff settings must be nonnegative")
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._sleeper = sleeper
        self._clock = clock
        self._event_cache: dict[tuple[str, bool], Event] = {}
        self._market_cache: dict[str, Market] = {}
        self._series_cache: dict[str, Series] = {}

    def __enter__(self) -> KalshiRestClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _delay(self, attempt: int, response: httpx.Response | None = None) -> float:
        delay = min(self._backoff_seconds * (2 ** (attempt - 1)), self._max_backoff_seconds)
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                with suppress(ValueError):
                    delay = min(max(float(retry_after), 0.0), self._max_backoff_seconds)
        return float(delay)

    def _get_json(
        self,
        path: str,
        *,
        params: Mapping[str, str | int | bool] | None = None,
    ) -> dict[str, Any]:
        last_transport_error: httpx.TransportError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.get(f"{self.base_url}{path}", params=params)
            except httpx.TransportError as exc:
                last_transport_error = exc
                if attempt == self._max_attempts:
                    break
                self._sleeper(self._delay(attempt))
                continue

            if response.status_code in self._RETRYABLE_STATUSES:
                if attempt < self._max_attempts:
                    self._sleeper(self._delay(attempt, response))
                    continue
                raise KalshiHttpError(response.status_code, path, _response_detail(response))
            if response.status_code >= 400:
                raise KalshiHttpError(response.status_code, path, _response_detail(response))
            try:
                payload = response.json()
            except ValueError as exc:
                raise KalshiPayloadError(f"Kalshi GET {path} returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise KalshiPayloadError(f"Kalshi GET {path} JSON root must be an object")
            return cast(dict[str, Any], payload)

        detail = (
            "unknown transport failure"
            if last_transport_error is None
            else str(last_transport_error)
        )
        raise KalshiTransportError(
            f"Kalshi GET {path} exhausted {self._max_attempts} transport attempts: {detail}"
        ) from last_transport_error

    def _page_payloads(
        self,
        path: str,
        *,
        item_key: str,
        params: dict[str, str | int | bool],
        max_pages: int | None,
        response_type: (
            type[MarketsResponse] | type[EventsResponse] | type[EventFeeChangesResponse]
        ),
    ) -> Iterator[dict[str, Any]]:
        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages must be positive when supplied")
        cursor: str | None = None
        seen_cursors: set[str] = set()
        page_number = 0
        while max_pages is None or page_number < max_pages:
            request_params = dict(params)
            if cursor is not None:
                request_params["cursor"] = cursor
            payload = self._get_json(path, params=request_params)
            try:
                response_type.model_validate(payload)
            except ValidationError as exc:
                raise KalshiPayloadError(f"invalid paginated {item_key} payload: {exc}") from exc
            raw_items = payload.get(item_key, [])
            if not isinstance(raw_items, list) or any(
                not isinstance(item, dict) for item in raw_items
            ):
                raise KalshiPayloadError(f"{item_key} must be an array of objects")
            for item in raw_items:
                yield cast(dict[str, Any], item)
            page_number += 1
            raw_cursor = payload.get("cursor", "")
            if not isinstance(raw_cursor, str):
                raise KalshiPayloadError("pagination cursor must be a string")
            next_cursor = raw_cursor.strip()
            if not next_cursor:
                return
            if next_cursor in seen_cursors:
                raise KalshiPaginationError(
                    f"Kalshi {path} repeated cursor {next_cursor!r} on page {page_number}"
                )
            seen_cursors.add(next_cursor)
            cursor = next_cursor

    def iter_markets(
        self,
        *,
        status: str | None = None,
        max_pages: int | None = None,
        enrich_series: bool = True,
    ) -> Iterator[Market]:
        """Iterate normalized markets and optionally resolve event-to-series ancestry."""

        params: dict[str, str | int | bool] = {"limit": 200}
        if status is not None:
            params["status"] = status
        for payload in self._page_payloads(
            "/markets",
            item_key="markets",
            params=params,
            max_pages=max_pages,
            response_type=MarketsResponse,
        ):
            event_ticker = str(payload.get("event_ticker", ""))
            series_ticker = None
            if enrich_series and event_ticker:
                series_ticker = self.get_event(
                    event_ticker,
                    with_nested_markets=False,
                ).series_ticker
            try:
                market = normalize_market(payload, series_ticker=series_ticker)
            except NormalizationError as exc:
                raise KalshiPayloadError(f"unsafe market payload: {exc}") from exc
            self._market_cache[market.ticker] = market
            yield market

    def iter_events(
        self,
        *,
        status: str | None = None,
        with_nested_markets: bool = False,
        max_pages: int | None = None,
    ) -> Iterator[Event]:
        """Iterate cursor-paginated normalized events."""

        params: dict[str, str | int | bool] = {
            "limit": 200,
            "with_nested_markets": with_nested_markets,
        }
        if status is not None:
            params["status"] = status
        for payload in self._page_payloads(
            "/events",
            item_key="events",
            params=params,
            max_pages=max_pages,
            response_type=EventsResponse,
        ):
            try:
                event = normalize_event(payload)
            except NormalizationError as exc:
                raise KalshiPayloadError(f"unsafe event payload: {exc}") from exc
            self._event_cache[(event.ticker, with_nested_markets)] = event
            yield event

    def iter_event_fee_changes(
        self,
        *,
        event_ticker: str | None = None,
        max_pages: int | None = None,
    ) -> Iterator[EventFeeChange]:
        """Iterate the public event fee schedule once, with bounded cursor handling."""

        params: dict[str, str | int | bool] = {"limit": 1000}
        if event_ticker is not None:
            if not event_ticker.strip():
                raise ValueError("event_ticker cannot be empty")
            params["event_ticker"] = event_ticker
        for payload in self._page_payloads(
            "/events/fee_changes",
            item_key="event_fee_changes",
            params=params,
            max_pages=max_pages,
            response_type=EventFeeChangesResponse,
        ):
            try:
                yield normalize_event_fee_change(payload)
            except NormalizationError as exc:
                raise KalshiPayloadError(f"unsafe event fee change payload: {exc}") from exc

    def get_event(
        self,
        event_ticker: str,
        *,
        with_nested_markets: bool = True,
        refresh: bool = False,
    ) -> Event:
        """Fetch one event, optionally bypassing cache after a lifecycle update."""

        key = (event_ticker, with_nested_markets)
        cached = None if refresh else self._event_cache.get(key)
        if cached is not None:
            return cached
        safe_ticker = quote(event_ticker, safe="")
        payload = self._get_json(
            f"/events/{safe_ticker}",
            params={"with_nested_markets": with_nested_markets},
        )
        try:
            parsed = EventResponse.model_validate(payload)
        except ValidationError as exc:
            raise KalshiPayloadError(f"invalid event response: {exc}") from exc
        event_payload = payload.get("event")
        if not isinstance(event_payload, dict):
            raise KalshiPayloadError("event response is missing its event object")
        top_level = payload.get("markets", [])
        if not isinstance(top_level, list):
            raise KalshiPayloadError("event response markets must be an array")
        try:
            event = normalize_event(event_payload, top_level_markets=top_level)
        except NormalizationError as exc:
            raise KalshiPayloadError(f"unsafe event response: {exc}") from exc
        self._event_cache[key] = event
        # Keep the parsed validation result live so schema drift cannot be optimized away.
        del parsed
        return event

    def get_market(
        self,
        ticker: str,
        *,
        enrich_series: bool = True,
        refresh: bool = False,
    ) -> Market:
        """Fetch one market, optionally bypassing cache after a lifecycle update."""

        cached = None if refresh else self._market_cache.get(ticker)
        if cached is not None and (cached.series_ticker is not None or not enrich_series):
            return cached
        safe_ticker = quote(ticker, safe="")
        payload = self._get_json(f"/markets/{safe_ticker}")
        try:
            MarketResponse.model_validate(payload)
        except ValidationError as exc:
            raise KalshiPayloadError(f"invalid market response: {exc}") from exc
        market_payload = payload.get("market")
        if not isinstance(market_payload, dict):
            raise KalshiPayloadError("market response is missing its market object")
        event_ticker = str(market_payload.get("event_ticker", ""))
        series_ticker = None
        if enrich_series and event_ticker:
            series_ticker = self.get_event(
                event_ticker,
                with_nested_markets=False,
            ).series_ticker
        try:
            market = normalize_market(market_payload, series_ticker=series_ticker)
        except NormalizationError as exc:
            raise KalshiPayloadError(f"unsafe market response: {exc}") from exc
        self._market_cache[ticker] = market
        return market

    def get_series(self, series_ticker: str) -> Series:
        """Fetch current fee and settlement-source metadata for one series."""

        cached = self._series_cache.get(series_ticker)
        if cached is not None:
            return cached
        safe_ticker = quote(series_ticker, safe="")
        payload = self._get_json(f"/series/{safe_ticker}")
        try:
            SeriesResponse.model_validate(payload)
        except ValidationError as exc:
            raise KalshiPayloadError(f"invalid series response: {exc}") from exc
        series_payload = payload.get("series")
        if not isinstance(series_payload, dict):
            raise KalshiPayloadError("series response is missing its series object")
        try:
            series = normalize_series(series_payload)
        except NormalizationError as exc:
            raise KalshiPayloadError(f"unsafe series response: {exc}") from exc
        self._series_cache[series_ticker] = series
        return series

    def get_orderbook(
        self,
        ticker: str,
        *,
        depth: int = 100,
        market: Market | None = None,
    ) -> OrderBook:
        """Fetch a fixed-point REST snapshot with no invented sequence or exchange time."""

        if not 0 <= depth <= 100:
            raise ValueError("order-book depth must be between 0 and 100")
        metadata = self.get_market(ticker) if market is None else market
        safe_ticker = quote(ticker, safe="")
        payload = self._get_json(
            f"/markets/{safe_ticker}/orderbook",
            params={"depth": depth},
        )
        try:
            OrderBookResponse.model_validate(payload)
        except ValidationError as exc:
            raise KalshiPayloadError(f"invalid order-book response: {exc}") from exc
        try:
            return normalize_orderbook(
                ticker,
                payload,
                market=metadata,
                local_timestamp=self._clock(),
                sequence=None,
                exchange_timestamp=None,
            )
        except NormalizationError as exc:
            raise KalshiPayloadError(f"unsafe order-book response: {exc}") from exc

    def healthcheck(self) -> None:
        """Perform one bounded public request suitable for an opt-in doctor check."""

        payload = self._get_json("/markets", params={"limit": 1})
        try:
            MarketsResponse.model_validate(payload)
        except ValidationError as exc:
            raise KalshiPayloadError(f"invalid healthcheck response: {exc}") from exc


def _response_detail(response: httpx.Response) -> str:
    text = response.text.strip().replace("\n", " ")
    return text[:300] or "empty response body"
