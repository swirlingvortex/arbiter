"""Exact market-local order-book delta behavior."""

from datetime import timedelta
from decimal import Decimal

import pytest
from tests.support.executable import NOW, book

from arbiter.models.orderbook import BookStatus, OrderBookUpdateError


def test_delta_add_reduce_and_remove_are_pure_and_sorted() -> None:
    original = book("A", yes=(("0.70", "2"), ("0.50", "1")))

    added = original.apply_delta(
        side="yes",
        price=Decimal("0.60"),
        quantity_delta=Decimal("3.25"),
        sequence=2,
        local_timestamp=NOW + timedelta(milliseconds=1),
        exchange_timestamp=NOW,
    )
    reduced = added.apply_delta(
        side="yes",
        price=Decimal("0.70"),
        quantity_delta=Decimal("-0.50"),
        sequence=3,
        local_timestamp=NOW + timedelta(milliseconds=2),
        exchange_timestamp=NOW + timedelta(milliseconds=1),
    )
    removed = reduced.apply_delta(
        side="yes",
        price=Decimal("0.50"),
        quantity_delta=Decimal("-1"),
        sequence=4,
        local_timestamp=NOW + timedelta(milliseconds=3),
        exchange_timestamp=None,
    )

    assert [(level.price, level.quantity) for level in original.yes_bids] == [
        (Decimal("0.70"), Decimal("2")),
        (Decimal("0.50"), Decimal("1")),
    ]
    assert [(level.price, level.quantity) for level in added.yes_bids] == [
        (Decimal("0.70"), Decimal("2")),
        (Decimal("0.60"), Decimal("3.25")),
        (Decimal("0.50"), Decimal("1")),
    ]
    assert [(level.price, level.quantity) for level in removed.yes_bids] == [
        (Decimal("0.70"), Decimal("1.50")),
        (Decimal("0.60"), Decimal("3.25")),
    ]
    assert removed.sequence == 4
    assert removed.local_timestamp == NOW + timedelta(milliseconds=3)


def test_no_side_delta_does_not_change_yes_side() -> None:
    original = book("A", yes=(("0.70", "2"),), no=(("0.20", "1"),))

    updated = original.apply_delta(
        side="no",
        price=Decimal("0.30"),
        quantity_delta=Decimal("2"),
        sequence=8,
        local_timestamp=NOW,
        exchange_timestamp=NOW,
    )

    assert updated.yes_bids == original.yes_bids
    assert [(level.price, level.quantity) for level in updated.no_bids] == [
        (Decimal("0.30"), Decimal("2")),
        (Decimal("0.20"), Decimal("1")),
    ]


def test_negative_result_and_missing_negative_level_fail_closed() -> None:
    original = book("A", yes=(("0.70", "1"),))

    for price, quantity_delta in (("0.70", "-1.01"), ("0.60", "-0.01")):
        with pytest.raises(OrderBookUpdateError, match="negative displayed depth"):
            original.apply_delta(
                side="yes",
                price=Decimal(price),
                quantity_delta=Decimal(quantity_delta),
                sequence=2,
                local_timestamp=NOW,
                exchange_timestamp=NOW,
            )


def test_nonfresh_book_rejects_deltas_and_validated_status_transitions() -> None:
    original = book("A", yes=(("0.70", "1"),))
    uncertain = original.with_status(BookStatus.RESYNC_REQUIRED, reason="sequence gap")

    with pytest.raises(OrderBookUpdateError, match="non-fresh"):
        uncertain.apply_delta(
            side="yes",
            price=Decimal("0.70"),
            quantity_delta=Decimal("1"),
            sequence=2,
            local_timestamp=NOW,
            exchange_timestamp=NOW,
        )
    with pytest.raises(ValueError, match="requires a status reason"):
        original.with_status(BookStatus.STALE)


@pytest.mark.parametrize(
    ("price", "quantity_delta", "sequence", "local_timestamp", "exchange_timestamp"),
    [
        (Decimal("NaN"), Decimal("1"), 2, NOW, NOW),
        (Decimal("1.01"), Decimal("1"), 2, NOW, NOW),
        (Decimal("0.5"), Decimal("NaN"), 2, NOW, NOW),
        (Decimal("0.5"), Decimal("0"), 2, NOW, NOW),
        (Decimal("0.5"), Decimal("1"), 0, NOW, NOW),
        (Decimal("0.5"), Decimal("1"), 2, NOW.replace(tzinfo=None), NOW),
        (Decimal("0.5"), Decimal("1"), 2, NOW, NOW.replace(tzinfo=None)),
    ],
)
def test_invalid_delta_fields_are_rejected(
    price: Decimal,
    quantity_delta: Decimal,
    sequence: int,
    local_timestamp: object,
    exchange_timestamp: object,
) -> None:
    with pytest.raises(OrderBookUpdateError):
        book("A", yes=(("0.50", "1"),)).apply_delta(
            side="yes",
            price=price,
            quantity_delta=quantity_delta,
            sequence=sequence,
            local_timestamp=local_timestamp,  # type: ignore[arg-type]
            exchange_timestamp=exchange_timestamp,  # type: ignore[arg-type]
        )
