"""Offline mocked integration tests for bounded Kalshi public REST behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from arbiter.kalshi.client import (
    KalshiHttpError,
    KalshiPaginationError,
    KalshiPayloadError,
    KalshiRestClient,
    KalshiTransportError,
)
from arbiter.kalshi.normalize import normalize_market


def _fixture(project_root: Path, name: str) -> dict[str, Any]:
    value = json.loads((project_root / "tests/fixtures/kalshi" / name).read_text())
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _client(
    handler: httpx.MockTransport,
    *,
    sleeper: object | None = None,
    max_attempts: int = 3,
) -> KalshiRestClient:
    kwargs: dict[str, object] = {}
    if sleeper is not None:
        kwargs["sleeper"] = sleeper
    return KalshiRestClient(
        "https://example.invalid/trade-api/v2",
        client=httpx.Client(transport=handler),
        max_attempts=max_attempts,
        **kwargs,  # type: ignore[arg-type]
    )


def test_market_pagination_terminates_and_enriches_series_from_event(
    project_root: Path,
) -> None:
    market = _fixture(project_root, "market.json")
    second_market = {**market, "ticker": "KXARBITER-26SEP03-T110"}
    event_response = _fixture(project_root, "event.json")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/markets"):
            cursor = request.url.params.get("cursor")
            if cursor is None:
                return httpx.Response(200, json={"markets": [market], "cursor": "page-2"})
            assert cursor == "page-2"
            return httpx.Response(200, json={"markets": [second_market], "cursor": ""})
        if request.url.path.endswith("/events/KXARBITER-26SEP03"):
            return httpx.Response(200, json=event_response)
        raise AssertionError(f"unexpected request: {request.url}")

    with _client(httpx.MockTransport(handler)) as client:
        markets = tuple(client.iter_markets(status="open"))

    assert [item.ticker for item in markets] == [
        "KXARBITER-26SEP03-T100",
        "KXARBITER-26SEP03-T110",
    ]
    assert all(item.series_ticker == "KXARBITER" for item in markets)
    market_requests = [request for request in requests if request.url.path.endswith("/markets")]
    event_requests = [request for request in requests if "/events/" in request.url.path]
    assert market_requests[0].url.params["status"] == "open"
    assert market_requests[0].url.params["limit"] == "200"
    assert len(event_requests) == 1


def test_market_refresh_bypasses_cache_for_lifecycle_metadata(project_root: Path) -> None:
    payload = _fixture(project_root, "market.json")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        title = "Initial title" if calls == 1 else "Refreshed title"
        return httpx.Response(200, json={"market": {**payload, "title": title}})

    with _client(httpx.MockTransport(handler)) as client:
        initial = client.get_market(payload["ticker"], enrich_series=False)
        cached = client.get_market(payload["ticker"], enrich_series=False)
        refreshed = client.get_market(payload["ticker"], enrich_series=False, refresh=True)

    assert initial.title == cached.title == "Initial title"
    assert refreshed.title == "Refreshed title"
    assert calls == 2


def test_event_refresh_bypasses_cache_for_fee_metadata(project_root: Path) -> None:
    response = _fixture(project_root, "event.json")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        event = cast(dict[str, Any], response["event"])
        title = "Initial event" if calls == 1 else "Refreshed event"
        return httpx.Response(200, json={"event": {**event, "title": title}})

    with _client(httpx.MockTransport(handler)) as client:
        initial = client.get_event("KXARBITER-26SEP03", with_nested_markets=False)
        cached = client.get_event("KXARBITER-26SEP03", with_nested_markets=False)
        refreshed = client.get_event(
            "KXARBITER-26SEP03",
            with_nested_markets=False,
            refresh=True,
        )

    assert initial.title == cached.title == "Initial event"
    assert refreshed.title == "Refreshed event"
    assert calls == 2


def test_repeated_cursor_fails_explicitly_without_third_request(project_root: Path) -> None:
    market = _fixture(project_root, "market.json")
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(200, json={"markets": [market], "cursor": "repeat"})

    with (
        _client(httpx.MockTransport(handler)) as client,
        pytest.raises(KalshiPaginationError, match="repeated cursor"),
    ):
        tuple(client.iter_markets(enrich_series=False))

    assert count == 2


def test_event_fee_changes_use_global_cursor_endpoint_and_optional_filter(
    project_root: Path,
) -> None:
    fixture = _fixture(project_root, "event_fee_changes.json")
    changes = cast(list[dict[str, Any]], fixture["event_fee_changes"])
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(
                200,
                json={"event_fee_changes": [changes[0]], "cursor": "next"},
            )
        assert cursor == "next"
        return httpx.Response(
            200,
            json={"event_fee_changes": [changes[1]], "cursor": ""},
        )

    with _client(httpx.MockTransport(handler)) as client:
        parsed = tuple(client.iter_event_fee_changes(event_ticker="KXARBITER-26SEP06"))

    assert [change.change_id for change in parsed] == [
        "fee-change-future",
        "fee-change-clear",
    ]
    assert all(request.url.path.endswith("/events/fee_changes") for request in requests)
    assert requests[0].url.params["limit"] == "1000"
    assert requests[0].url.params["event_ticker"] == "KXARBITER-26SEP06"


def test_event_fee_changes_fail_closed_on_unsafe_item() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "event_fee_changes": [
                    {
                        "id": "bad",
                        "event_ticker": "EVENT",
                        "series_ticker": "SERIES",
                        "scheduled_ts": "2026-09-03T12:00:00Z",
                        "fee_type": "quadratic",
                        "fee_multiplier": None,
                    }
                ],
                "cursor": "",
            },
        )

    with (
        _client(httpx.MockTransport(handler)) as client,
        pytest.raises(KalshiPayloadError, match="unsafe event fee change"),
    ):
        tuple(client.iter_event_fee_changes())


def test_max_pages_bounds_pagination_even_when_cursor_continues(project_root: Path) -> None:
    market = _fixture(project_root, "market.json")
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(200, json={"markets": [market], "cursor": f"next-{count}"})

    with _client(httpx.MockTransport(handler)) as client:
        markets = tuple(client.iter_markets(max_pages=1, enrich_series=False))

    assert len(markets) == 1
    assert count == 1


def test_safe_get_retries_transport_and_retryable_statuses_boundedly() -> None:
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary", request=request)
        if calls == 2:
            return httpx.Response(429, headers={"Retry-After": "0.25"}, text="limited")
        return httpx.Response(200, json={"markets": [], "cursor": ""})

    with _client(httpx.MockTransport(handler), sleeper=delays.append) as client:
        client.healthcheck()

    assert calls == 3
    assert delays == [0.1, 0.25]


def test_transport_retry_exhaustion_is_not_reported_as_empty_data() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadError("still broken", request=request)

    with (
        _client(
            httpx.MockTransport(handler),
            sleeper=lambda _: None,
            max_attempts=2,
        ) as client,
        pytest.raises(KalshiTransportError, match="exhausted 2"),
    ):
        client.healthcheck()

    assert calls == 2


def test_nonretryable_4xx_fails_after_one_attempt() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, json={"error": "missing"})

    with (
        _client(httpx.MockTransport(handler), sleeper=lambda _: None) as client,
        pytest.raises(KalshiHttpError) as raised,
    ):
        client.healthcheck()

    assert raised.value.status_code == 404
    assert calls == 1


def test_event_series_and_orderbook_endpoints_parse_current_wrappers(project_root: Path) -> None:
    event_response = _fixture(project_root, "event.json")
    series_response = _fixture(project_root, "series.json")
    orderbook_response = _fixture(project_root, "orderbook.json")
    market = normalize_market(
        _fixture(project_root, "market.json"),
        series_ticker="KXARBITER",
    )
    received_at = datetime(2026, 9, 3, 15, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        if "/events/" in request.url.path:
            return httpx.Response(200, json=event_response)
        if "/series/" in request.url.path:
            return httpx.Response(200, json=series_response)
        if request.url.path.endswith("/orderbook"):
            return httpx.Response(200, json=orderbook_response)
        raise AssertionError(f"unexpected request: {request.url}")

    client = KalshiRestClient(
        "https://example.invalid/trade-api/v2",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: received_at,
    )
    event = client.get_event("KXARBITER-26SEP03", with_nested_markets=False)
    series = client.get_series("KXARBITER")
    book = client.get_orderbook(market.ticker, market=market, depth=10)

    assert event.market_tickers == ("KXARBITER-26SEP03-T100",)
    assert series.fee_type == "quadratic"
    assert book.local_timestamp == received_at
    assert book.sequence is None and book.exchange_timestamp is None
    assert book.best_yes_bid() is not None
