"""Versioned, lossless market-data Parquet persistence tests."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arbiter.models.orderbook import PriceLevel
from arbiter.replay.events import OrderBookDeltaEvent, OrderBookSnapshotEvent
from arbiter.storage.parquet import (
    MARKET_DATA_ARROW_SCHEMA,
    ParquetBackpressureError,
    ParquetEventWriter,
    ParquetStorageError,
    next_market_data_event_index,
    read_market_data_events,
)


def _snapshot(
    event_index: int = 0,
    *,
    local_received_ts: datetime = datetime(2026, 9, 3, 23, 59, 59, tzinfo=UTC),
) -> OrderBookSnapshotEvent:
    return OrderBookSnapshotEvent(
        event_index=event_index,
        local_received_ts=local_received_ts,
        exchange_ts=None,
        ticker="KX-PARQUET",
        sequence=1,
        sid=7,
        connection_id="connection-1",
        snapshot_id="connection-1:7:1",
        yes_bids=(
            PriceLevel(price=Decimal("0.6200"), quantity=Decimal("2.50")),
            PriceLevel(price=Decimal("0.6100"), quantity=Decimal("1.00")),
        ),
        no_bids=(PriceLevel(price=Decimal("0.3700"), quantity=Decimal("3.00")),),
    )


def _delta(
    event_index: int = 1,
    *,
    local_received_ts: datetime = datetime(2026, 9, 4, 0, 0, 1, tzinfo=UTC),
) -> OrderBookDeltaEvent:
    return OrderBookDeltaEvent(
        event_index=event_index,
        local_received_ts=local_received_ts,
        exchange_ts=datetime(2026, 9, 4, 0, 0, 0, 123000, tzinfo=UTC),
        ticker="KX-PARQUET",
        sequence=2,
        sid=7,
        connection_id="connection-1",
        snapshot_id="connection-1:7:1",
        side="no",
        price=Decimal("0.3700"),
        quantity_delta=Decimal("-0.25"),
    )


def test_round_trip_preserves_events_and_partitions_by_utc_date(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    expected = (_snapshot(), _delta())
    writer = ParquetEventWriter(root, max_queue_size=2)
    writer.extend(expected)

    paths = writer.flush()

    assert {path.parent.name for path in paths} == {
        "date=2026-09-03",
        "date=2026-09-04",
    }
    assert writer.pending_count == 0
    assert read_market_data_events(root) == expected
    assert not tuple(root.rglob("*.tmp"))


def test_files_use_exact_schema_and_zstd_compression(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    writer = ParquetEventWriter(root)
    writer.append(_snapshot())
    (path,) = writer.flush()

    parquet_file = pq.ParquetFile(path)

    assert parquet_file.schema_arrow.equals(MARKET_DATA_ARROW_SCHEMA, check_metadata=True)
    metadata = parquet_file.metadata
    assert metadata is not None
    assert {
        metadata.row_group(row_group).column(column).compression
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.row_group(row_group).num_columns)
    } == {"ZSTD"}


def test_raw_rows_keep_snapshot_levels_separate_from_scalar_delta(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    writer = ParquetEventWriter(root)
    writer.extend(
        (_snapshot(), _delta(local_received_ts=datetime(2026, 9, 3, 23, 59, 59, tzinfo=UTC)))
    )
    (path,) = writer.flush()

    rows = pq.ParquetFile(path).read().to_pylist()

    assert rows[0]["yes_bids"] == [
        {"price": Decimal("0.6200"), "quantity": Decimal("2.50")},
        {"price": Decimal("0.6100"), "quantity": Decimal("1.00")},
    ]
    assert rows[0]["delta_side"] is None
    assert rows[1]["yes_bids"] is None
    assert rows[1]["delta_side"] == "no"
    assert rows[1]["delta_price"] == Decimal("0.3700")
    assert rows[1]["quantity_delta"] == Decimal("-0.25")


def test_queue_overflow_fails_visibly_without_dropping_accepted_events(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orderbooks"
    writer = ParquetEventWriter(root, max_queue_size=2)
    writer.extend((_snapshot(), _delta()))

    with pytest.raises(ParquetBackpressureError, match="2-event limit"):
        writer.append(
            _delta(
                2,
                local_received_ts=datetime(2026, 9, 4, 0, 0, 2, tzinfo=UTC),
            )
        )

    assert writer.pending_count == 2
    writer.flush()
    assert read_market_data_events(root) == (_snapshot(), _delta())


def test_failed_flush_retains_queue_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "orderbooks"
    writer = ParquetEventWriter(root)
    writer.append(_snapshot())

    def fail_write(*args: object, **kwargs: object) -> None:
        path = Path(args[1])
        path.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr("arbiter.storage.parquet.pq.write_table", fail_write)

    with pytest.raises(ParquetStorageError, match="disk full"):
        writer.flush()

    assert writer.pending_count == 1
    assert not tuple(root.rglob("*.tmp"))
    assert not tuple(root.rglob("*.parquet"))


def test_writer_rejects_duplicate_or_noncontiguous_indices(tmp_path: Path) -> None:
    writer = ParquetEventWriter(tmp_path / "orderbooks")
    writer.append(_snapshot())

    with pytest.raises(ParquetStorageError, match="duplicate event_index 0"):
        writer.append(_snapshot())
    with pytest.raises(ParquetStorageError, match="expected 1, received 2"):
        writer.append(_delta(2))


def test_reader_rejects_duplicate_indices_across_files(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    first = ParquetEventWriter(root)
    first.append(_snapshot())
    (path,) = first.close()
    duplicate = path.with_name("duplicate.parquet")
    pq.write_table(pq.ParquetFile(path).read(), duplicate, compression="zstd")

    with pytest.raises(ParquetStorageError, match="duplicate event_index"):
        read_market_data_events(root)
    with pytest.raises(ParquetStorageError, match="duplicate event_index"):
        ParquetEventWriter(root)


def test_reader_rejects_noncontiguous_mixed_stream(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    first = ParquetEventWriter(root)
    first.append(_snapshot())
    first.close()
    second = ParquetEventWriter(root)
    second.append(_delta())
    (path,) = second.close()
    table = pq.ParquetFile(path).read()
    noncontiguous = table.set_column(
        table.schema.get_field_index("event_index"),
        MARKET_DATA_ARROW_SCHEMA.field("event_index"),
        pa.array([2], type=pa.int64()),
    )
    pq.write_table(noncontiguous, path, compression="zstd")

    with pytest.raises(ParquetStorageError, match="mixed or incomplete"):
        read_market_data_events(root)


def test_reader_rejects_incompatible_arrow_schema(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks" / "date=2026-09-03"
    root.mkdir(parents=True)
    pq.write_table(pa.table({"event_index": [0]}), root / "broken.parquet")

    with pytest.raises(ParquetStorageError, match="incompatible"):
        read_market_data_events(root.parent)


def test_reader_rejects_mismatched_row_schema_version(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    writer = ParquetEventWriter(root)
    writer.append(_snapshot())
    (path,) = writer.flush()
    table = pq.ParquetFile(path).read()
    incompatible = table.set_column(
        table.schema.get_field_index("schema_version"),
        MARKET_DATA_ARROW_SCHEMA.field("schema_version"),
        pa.array([2], type=pa.int16()),
    )
    pq.write_table(incompatible, path, compression="zstd")

    with pytest.raises(ParquetStorageError, match="invalid market-data event"):
        read_market_data_events(root)


def test_reader_orders_physical_files_by_authoritative_event_index(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    earlier = ParquetEventWriter(root)
    earlier.append(_snapshot())
    (earlier_path,) = earlier.close()
    later = ParquetEventWriter(root)
    later.append(_delta(local_received_ts=datetime(2026, 9, 3, 23, 59, 59, tzinfo=UTC)))
    (later_path,) = later.close()
    earlier_path.rename(earlier_path.with_name("z-physically-last.parquet"))
    later_path.rename(later_path.with_name("a-physically-first.parquet"))

    restored = read_market_data_events(root)

    assert [event.event_index for event in restored] == [0, 1]


def test_sequential_writers_bootstrap_one_contiguous_readable_stream(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"

    assert next_market_data_event_index(root) == 0
    first = ParquetEventWriter(root)
    assert first.next_event_index == 0
    first.append(_snapshot())
    first.close()

    assert next_market_data_event_index(root) == 1
    second = ParquetEventWriter(root)
    assert second.next_event_index == 1
    second.append(_delta())
    second.close()

    assert next_market_data_event_index(root) == 2
    assert read_market_data_events(root) == (_snapshot(), _delta())


def test_dataset_writer_lease_rejects_contention_without_blocking(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    first = ParquetEventWriter(root)

    with pytest.raises(ParquetStorageError, match="already has an active writer"):
        ParquetEventWriter(root)

    first.close()
    resumed = ParquetEventWriter(root)
    assert resumed.next_event_index == 0
    resumed.close()


def test_empty_dataset_and_closed_writer_behavior(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    root.mkdir()
    assert read_market_data_events(root) == ()
    writer = ParquetEventWriter(root)
    writer.append(_snapshot())
    writer.close()

    with pytest.raises(ParquetStorageError, match="closed"):
        writer.append(_delta())
