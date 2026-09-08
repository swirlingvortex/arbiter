"""Executable complementary-level construction and book rejection tests."""

from datetime import timedelta
from decimal import Decimal

import pytest
from tests.support.executable import NOW, book, implication_component_and_worlds

from arbiter.models.orderbook import BookStatus
from arbiter.solver.instruments import build_executable_instruments


def test_builds_one_canonically_ordered_instrument_per_real_bid_level() -> None:
    component, _ = implication_component_and_worlds()
    books = {
        "A": book(
            "A",
            yes=(("0.50", "4.00"),),
            no=(("0.40", "1.25"), ("0.35", "2.50")),
        ),
        "M": book("M", yes=(("0.70", "3.00"),), no=(("0.20", "1.00"),)),
    }

    result = build_executable_instruments(
        component,
        books,
        as_of=NOW,
        stale_after=timedelta(seconds=2),
    )

    assert result.status == "ok"
    assert [
        (
            instrument.ticker,
            instrument.side,
            instrument.price,
            instrument.max_quantity,
            instrument.source_side,
            instrument.source_price,
        )
        for instrument in result.instruments
    ] == [
        ("A", "yes", Decimal("0.60"), Decimal("1.25"), "no_bid", Decimal("0.40")),
        ("A", "yes", Decimal("0.65"), Decimal("2.50"), "no_bid", Decimal("0.35")),
        ("A", "no", Decimal("0.50"), Decimal("4.00"), "yes_bid", Decimal("0.50")),
        ("M", "yes", Decimal("0.80"), Decimal("1.00"), "no_bid", Decimal("0.20")),
        ("M", "no", Decimal("0.30"), Decimal("3.00"), "yes_bid", Decimal("0.70")),
    ]


def test_one_sided_book_creates_only_the_complementary_instrument_that_exists() -> None:
    component, _ = implication_component_and_worlds()
    result = build_executable_instruments(
        component,
        {
            "A": book("A", no=(("0.40", "1.00"),)),
            "M": book("M", yes=(("0.70", "1.00"),)),
        },
        as_of=NOW,
        stale_after=timedelta(seconds=2),
    )

    assert result.status == "ok"
    assert [(item.ticker, item.side) for item in result.instruments] == [
        ("A", "yes"),
        ("M", "no"),
    ]


def test_missing_book_fails_closed() -> None:
    component, _ = implication_component_and_worlds()

    result = build_executable_instruments(
        component,
        {"A": book("A", no=(("0.40", "1.00"),))},
        as_of=NOW,
        stale_after=timedelta(seconds=2),
    )

    assert result.status == "missing_book"
    assert result.market_ticker == "M"


@pytest.mark.parametrize(
    ("replacement", "expected_status", "message"),
    [
        (book("A"), "empty_book", "no positive"),
        (
            book("A", no=(("0.40", "1.00"),), local_timestamp=NOW - timedelta(seconds=3)),
            "stale_book",
            "age exceeds",
        ),
        (
            book(
                "A",
                no=(("0.40", "1.00"),),
                status=BookStatus.STALE,
                status_reason="explicit fixture stale",
            ),
            "stale_book",
            "explicitly stale",
        ),
        (
            book(
                "A",
                no=(("0.40", "1.00"),),
                status=BookStatus.RESYNC_REQUIRED,
                status_reason="sequence gap",
            ),
            "invalid_book",
            "resynchronization",
        ),
        (
            book("A", no=(("0.40", "1.00"),), local_timestamp=NOW + timedelta(microseconds=1)),
            "invalid_book",
            "later than",
        ),
    ],
)
def test_empty_stale_resync_and_future_books_fail_closed(
    replacement: object,
    expected_status: str,
    message: str,
) -> None:
    component, _ = implication_component_and_worlds()
    result = build_executable_instruments(
        component,
        {
            "A": replacement,  # type: ignore[dict-item]
            "M": book("M", yes=(("0.70", "1.00"),)),
        },
        as_of=NOW,
        stale_after=timedelta(seconds=2),
    )

    assert result.status == expected_status
    assert message in (result.reason or "")


def test_book_key_and_payload_ticker_mismatch_is_invalid() -> None:
    component, _ = implication_component_and_worlds()

    result = build_executable_instruments(
        component,
        {
            "A": book("WRONG", no=(("0.40", "1.00"),)),
            "M": book("M", yes=(("0.70", "1.00"),)),
        },
        as_of=NOW,
        stale_after=timedelta(seconds=2),
    )

    assert result.status == "invalid_book"
    assert "WRONG" in (result.reason or "")
