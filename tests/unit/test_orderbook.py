"""REST bid normalization, complementary asks, and freshness tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from arbiter.kalshi.normalize import NormalizationError, normalize_market, normalize_orderbook
from arbiter.models.orderbook import BookStatus, OrderBook, PriceLevel


def _json(project_root: Path, name: str) -> dict[str, Any]:
    value = json.loads((project_root / "tests/fixtures/kalshi" / name).read_text())
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _market(project_root: Path) -> object:
    return normalize_market(_json(project_root, "market.json"), series_ticker="KXARBITER")


def test_orderbook_sorts_aggregates_drops_zero_and_derives_exact_asks(
    project_root: Path,
) -> None:
    market = _market(project_root)
    received_at = datetime(2026, 9, 3, 12, tzinfo=UTC)

    book = normalize_orderbook(
        "KXARBITER-26SEP03-T100",
        _json(project_root, "orderbook.json"),
        market=market,  # type: ignore[arg-type]
        local_timestamp=received_at,
    )

    assert [(level.price, level.quantity) for level in book.yes_bids] == [
        (Decimal("0.6000"), Decimal("1.00")),
        (Decimal("0.5500"), Decimal("3.00")),
    ]
    assert [(level.price, level.quantity) for level in book.no_bids] == [
        (Decimal("0.4000"), Decimal("3.50")),
        (Decimal("0.3500"), Decimal("4.00")),
    ]
    assert [(level.price, level.quantity) for level in book.yes_asks] == [
        (Decimal("0.6000"), Decimal("3.50")),
        (Decimal("0.6500"), Decimal("4.00")),
    ]
    assert book.best_no_ask() == PriceLevel(price=Decimal("0.4000"), quantity=Decimal("1.00"))
    assert book.sequence is None
    assert book.exchange_timestamp is None


def test_one_sided_and_empty_books_invent_no_liquidity(project_root: Path) -> None:
    market = _market(project_root)
    now = datetime(2026, 9, 3, tzinfo=UTC)
    one_sided = normalize_orderbook(
        "KXARBITER-26SEP03-T100",
        {"orderbook_fp": {"yes_dollars": [["0.5000", "1.00"]]}},
        market=market,  # type: ignore[arg-type]
        local_timestamp=now,
    )
    empty = normalize_orderbook(
        "KXARBITER-26SEP03-T100",
        {"orderbook_fp": {}},
        market=market,  # type: ignore[arg-type]
        local_timestamp=now,
    )

    assert one_sided.yes_asks == ()
    assert one_sided.no_asks == (PriceLevel(price=Decimal("0.5000"), quantity=Decimal("1.00")),)
    assert empty.yes_bids == empty.no_bids == empty.yes_asks == empty.no_asks == ()


@pytest.mark.parametrize(
    ("level", "message"),
    [
        (["0.5550", "1.00"], "not valid"),
        (["0.5500", "-1.00"], "negative"),
        (["0.55000", "1.00"], "4 decimal places"),
        (["0.5500", "1.001"], "2 decimal places"),
    ],
)
def test_invalid_price_or_quantity_fails_closed(
    project_root: Path,
    level: list[str],
    message: str,
) -> None:
    with pytest.raises(NormalizationError, match=message):
        normalize_orderbook(
            "KXARBITER-26SEP03-T100",
            {"orderbook_fp": {"yes_dollars": [level]}},
            market=_market(project_root),  # type: ignore[arg-type]
            local_timestamp=datetime(2026, 9, 3, tzinfo=UTC),
        )


def test_nonempty_book_without_price_ranges_fails_closed() -> None:
    market = normalize_market(
        {"ticker": "M", "event_ticker": "E", "title": "M", "status": "active"}
    )

    with pytest.raises(NormalizationError, match="without price ranges"):
        normalize_orderbook(
            "M",
            {"orderbook_fp": {"yes_dollars": [["0.5000", "1.00"]]}},
            market=market,
            local_timestamp=datetime(2026, 9, 3, tzinfo=UTC),
        )


def test_freshness_uses_explicit_as_of_and_fails_on_future_or_status() -> None:
    timestamp = datetime(2026, 9, 3, 12, tzinfo=UTC)
    fresh = OrderBook(ticker="M", local_timestamp=timestamp)
    stale_status = OrderBook(
        ticker="M",
        local_timestamp=timestamp,
        status=BookStatus.STALE,
        status_reason="test",
    )

    assert fresh.is_fresh(as_of=timestamp + timedelta(seconds=2), stale_after=timedelta(seconds=2))
    assert not fresh.is_fresh(
        as_of=timestamp + timedelta(microseconds=2_000_001),
        stale_after=timedelta(seconds=2),
    )
    assert not fresh.is_fresh(
        as_of=timestamp - timedelta(microseconds=1),
        stale_after=timedelta(seconds=2),
    )
    assert not stale_status.is_fresh(
        as_of=timestamp,
        stale_after=timedelta(seconds=2),
    )


def test_orderbook_domain_rejects_unsorted_or_duplicate_bids() -> None:
    now = datetime(2026, 9, 3, tzinfo=UTC)
    levels = (
        PriceLevel(price=Decimal("0.40"), quantity=Decimal("1")),
        PriceLevel(price=Decimal("0.50"), quantity=Decimal("1")),
    )
    with pytest.raises(ValidationError, match="descending"):
        OrderBook(ticker="M", yes_bids=levels, local_timestamp=now)


def test_orderbook_requires_timezone_aware_clocks() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        OrderBook(ticker="M", local_timestamp=datetime(2026, 9, 3))

    book = OrderBook(ticker="M", local_timestamp=datetime(2026, 9, 3, tzinfo=UTC))
    with pytest.raises(ValueError, match="timezone-aware"):
        book.is_fresh(as_of=datetime(2026, 9, 3), stale_after=timedelta(seconds=1))
