"""Schema-v2 recorded-stream Parquet compatibility and corruption tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arbiter.models.orderbook import PriceLevel
from arbiter.replay.events import (
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
    SubscriptionStartedEvent,
    dump_recorded_event_json,
)
from arbiter.storage.parquet import (
    MARKET_DATA_ARROW_SCHEMA,
    RECORDED_EVENTS_ARROW_SCHEMA,
    ParquetEventWriter,
    ParquetStorageError,
    next_market_data_event_index,
    read_market_data_events,
    read_recorded_events,
)

NOW = datetime(2026, 9, 3, 23, 59, 59, tzinfo=UTC)


def _subscription(
    event_index: int = 0,
    *,
    received_at: datetime = NOW,
) -> SubscriptionStartedEvent:
    return SubscriptionStartedEvent(
        event_index=event_index,
        local_received_ts=received_at,
        connection_id="connection-1",
        sid=7,
        tickers=("KX-RECORDED",),
    )


def _snapshot(
    event_index: int,
    *,
    received_at: datetime,
) -> OrderBookSnapshotEvent:
    return OrderBookSnapshotEvent(
        event_index=event_index,
        local_received_ts=received_at,
        exchange_ts=None,
        ticker="KX-RECORDED",
        sequence=1,
        sid=7,
        connection_id="connection-1",
        snapshot_id="connection-1:7:1",
        yes_bids=(PriceLevel(price=Decimal("0.6200"), quantity=Decimal("2.50")),),
        no_bids=(PriceLevel(price=Decimal("0.3700"), quantity=Decimal("3.00")),),
    )


def _delta(
    event_index: int,
    *,
    received_at: datetime,
) -> OrderBookDeltaEvent:
    return OrderBookDeltaEvent(
        event_index=event_index,
        local_received_ts=received_at,
        exchange_ts=received_at,
        ticker="KX-RECORDED",
        sequence=2,
        sid=7,
        connection_id="connection-1",
        snapshot_id="connection-1:7:1",
        side="yes",
        price=Decimal("0.6200"),
        quantity_delta=Decimal("-0.50"),
    )


def test_v2_round_trip_preserves_mixed_total_order_and_canonical_control(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orderbooks"
    expected = (
        _subscription(),
        _snapshot(1, received_at=NOW + timedelta(seconds=1)),
        _delta(2, received_at=NOW + timedelta(seconds=2)),
    )
    with ParquetEventWriter(root) as writer:
        writer.extend(expected)

    files = tuple(root.rglob("*.parquet"))
    assert {path.parent.name for path in files} == {"date=2026-09-03", "date=2026-09-04"}
    assert all(
        pq.ParquetFile(path).schema_arrow.equals(RECORDED_EVENTS_ARROW_SCHEMA, check_metadata=True)
        for path in files
    )
    assert read_recorded_events(root) == expected
    assert read_market_data_events(root) == expected[1:]
    assert next_market_data_event_index(root) == 3

    control_rows = [
        row
        for path in files
        for row in pq.ParquetFile(path).read().to_pylist()
        if row["event_type"] == "subscription_started"
    ]
    assert len(control_rows) == 1
    assert control_rows[0]["control_payload_json"] == dump_recorded_event_json(expected[0])
    assert all(control_rows[0][column] is None for column in ("ticker", "sequence", "sid"))


def test_v2_dataset_keeps_v2_container_for_later_book_only_batches(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    with ParquetEventWriter(root) as writer:
        writer.extend(
            (
                _subscription(),
                _snapshot(1, received_at=NOW + timedelta(seconds=1)),
            )
        )
    with ParquetEventWriter(root) as writer:
        writer.append(_delta(2, received_at=NOW + timedelta(seconds=2)))

    assert read_recorded_events(root) == (
        _subscription(),
        _snapshot(1, received_at=NOW + timedelta(seconds=1)),
        _delta(2, received_at=NOW + timedelta(seconds=2)),
    )
    assert all(
        pq.ParquetFile(path).schema_arrow.equals(RECORDED_EVENTS_ARROW_SCHEMA, check_metadata=True)
        for path in root.rglob("*.parquet")
    )


def test_legacy_v1_dataset_rejects_controls_without_mutating_stream(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    with ParquetEventWriter(root) as writer:
        writer.append(_snapshot(0, received_at=NOW))

    writer = ParquetEventWriter(root)
    control = _subscription(1, received_at=NOW + timedelta(seconds=1))
    with pytest.raises(ParquetStorageError, match="schema-v2 control.*legacy schema-v1"):
        writer.append(control)
    assert writer.pending_count == 0
    assert writer.next_event_index == 1
    writer.close()

    assert read_recorded_events(root) == (_snapshot(0, received_at=NOW),)
    (path,) = tuple(root.rglob("*.parquet"))
    assert pq.ParquetFile(path).schema_arrow.equals(MARKET_DATA_ARROW_SCHEMA, check_metadata=True)


def test_writer_rejects_decreasing_event_time_before_queueing(tmp_path: Path) -> None:
    writer = ParquetEventWriter(tmp_path / "orderbooks")
    writer.append(_snapshot(0, received_at=NOW))

    with pytest.raises(ParquetStorageError, match="nondecreasing"):
        writer.append(_delta(1, received_at=NOW - timedelta(microseconds=1)))

    assert writer.pending_count == 1
    assert writer.next_event_index == 1
    writer.close()


def test_reader_rejects_decreasing_event_time_in_authoritative_index_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "orderbooks"
    with ParquetEventWriter(root) as writer:
        writer.extend(
            (
                _snapshot(0, received_at=NOW - timedelta(minutes=1)),
                _delta(1, received_at=NOW),
            )
        )
    (path,) = tuple(root.rglob("*.parquet"))
    table = pq.ParquetFile(path).read()
    timestamp_type = MARKET_DATA_ARROW_SCHEMA.field("local_received_ts").type
    corrupt = table.set_column(
        table.schema.get_field_index("local_received_ts"),
        MARKET_DATA_ARROW_SCHEMA.field("local_received_ts"),
        pa.array((NOW, NOW - timedelta(microseconds=1)), type=timestamp_type),
    )
    pq.write_table(corrupt, path, compression="zstd")

    with pytest.raises(ParquetStorageError, match="nondecreasing"):
        read_recorded_events(root)


def test_market_data_compatibility_reader_validates_control_payloads(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    with ParquetEventWriter(root) as writer:
        writer.extend(
            (
                _subscription(),
                _snapshot(1, received_at=NOW + timedelta(seconds=1)),
            )
        )
    control_path = next(
        path
        for path in root.rglob("*.parquet")
        if any(
            row["event_type"] == "subscription_started"
            for row in pq.ParquetFile(path).read().to_pylist()
        )
    )
    table = pq.ParquetFile(control_path).read()
    rows = table.to_pylist()
    control_index = next(
        index for index, row in enumerate(rows) if row["event_type"] == "subscription_started"
    )
    payloads = [row["control_payload_json"] for row in rows]
    payload = payloads[control_index]
    assert isinstance(payload, str)
    payloads[control_index] = json.dumps(json.loads(payload), sort_keys=True)
    corrupt = table.set_column(
        table.schema.get_field_index("control_payload_json"),
        RECORDED_EVENTS_ARROW_SCHEMA.field("control_payload_json"),
        pa.array(payloads, type=pa.string()),
    )
    pq.write_table(corrupt, control_path, compression="zstd")

    with pytest.raises(ParquetStorageError, match="not canonical JSON"):
        read_market_data_events(root)


def test_reader_rejects_mixed_v1_and_v2_container_files(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    with ParquetEventWriter(root) as writer:
        writer.append(_snapshot(0, received_at=NOW))

    v2_root = tmp_path / "v2"
    with ParquetEventWriter(v2_root) as writer:
        writer.append(_subscription())
    (v2_path,) = tuple(v2_root.rglob("*.parquet"))
    destination = next(root.glob("date=*")) / "foreign-v2.parquet"
    pq.write_table(pq.ParquetFile(v2_path).read(), destination, compression="zstd")

    with pytest.raises(ParquetStorageError, match="mixed schema-v1 and schema-v2"):
        read_recorded_events(root)


def test_reader_rejects_unsupported_v2_container_metadata(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    with ParquetEventWriter(root) as writer:
        writer.append(_subscription())
    (path,) = tuple(root.rglob("*.parquet"))
    table = pq.ParquetFile(path).read()
    metadata = dict(table.schema.metadata or {})
    metadata[b"arbiter_schema_version"] = b"99"
    pq.write_table(table.replace_schema_metadata(metadata), path, compression="zstd")

    with pytest.raises(ParquetStorageError, match="unsupported Parquet schema version"):
        read_recorded_events(root)


def test_reader_rejects_v2_file_in_wrong_utc_partition(tmp_path: Path) -> None:
    root = tmp_path / "orderbooks"
    with ParquetEventWriter(root) as writer:
        writer.append(_subscription())
    (path,) = tuple(root.rglob("*.parquet"))
    wrong_partition = root / "date=2026-09-02"
    wrong_partition.mkdir()
    path.replace(wrong_partition / path.name)

    with pytest.raises(ParquetStorageError, match="UTC date belongs"):
        read_recorded_events(root)
