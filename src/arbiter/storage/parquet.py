"""Lossless, versioned Parquet storage for the authoritative recorded stream."""

from __future__ import annotations

import fcntl
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import suppress
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import BinaryIO, TypeGuard
from uuid import uuid4

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import TypeAdapter, ValidationError

from arbiter.models.orderbook import PriceLevel
from arbiter.replay.events import (
    MARKET_DATA_SCHEMA_VERSION,
    MarketDataEvent,
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
    RecordedEvent,
    dump_recorded_event_json,
    load_recorded_event_json,
)

_SCHEMA_NAME = "arbiter.market_data_events"
_RECORDED_SCHEMA_NAME = "arbiter.recorded_events"
_RECORDED_CONTAINER_SCHEMA_VERSION = 2
_PRICE_TYPE = pa.decimal128(5, 4)
_QUANTITY_TYPE = pa.decimal128(38, 2)
_LEVEL_TYPE = pa.struct(
    (
        pa.field("price", _PRICE_TYPE, nullable=False),
        pa.field("quantity", _QUANTITY_TYPE, nullable=False),
    )
)

MARKET_DATA_ARROW_SCHEMA = pa.schema(
    (
        pa.field("schema_version", pa.int16(), nullable=False),
        pa.field("event_index", pa.int64(), nullable=False),
        pa.field("local_received_ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("exchange_ts", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("sequence", pa.int64(), nullable=False),
        pa.field("sid", pa.int64(), nullable=False),
        pa.field("connection_id", pa.string(), nullable=False),
        pa.field("price_convention", pa.string(), nullable=False),
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field(
            "yes_bids",
            pa.list_(pa.field("element", _LEVEL_TYPE, nullable=True)),
            nullable=True,
        ),
        pa.field(
            "no_bids",
            pa.list_(pa.field("element", _LEVEL_TYPE, nullable=True)),
            nullable=True,
        ),
        pa.field("delta_side", pa.string(), nullable=True),
        pa.field("delta_price", _PRICE_TYPE, nullable=True),
        pa.field("quantity_delta", _QUANTITY_TYPE, nullable=True),
    ),
    metadata={
        b"arbiter_schema": _SCHEMA_NAME.encode("ascii"),
        b"arbiter_schema_version": str(MARKET_DATA_SCHEMA_VERSION).encode("ascii"),
    },
)

# Schema v1 above is an immutable wire contract. The schema-v2 container retains the
# same book representation, makes book-only fields nullable for controls, and stores
# each control's complete canonical JSON payload. Book records inside this container
# intentionally retain their event-level schema_version=1.
RECORDED_EVENTS_ARROW_SCHEMA = pa.schema(
    (
        pa.field("schema_version", pa.int16(), nullable=False),
        pa.field("event_index", pa.int64(), nullable=False),
        pa.field("local_received_ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("exchange_ts", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("ticker", pa.string(), nullable=True),
        pa.field("sequence", pa.int64(), nullable=True),
        pa.field("sid", pa.int64(), nullable=True),
        pa.field("connection_id", pa.string(), nullable=True),
        pa.field("price_convention", pa.string(), nullable=True),
        pa.field("snapshot_id", pa.string(), nullable=True),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field(
            "yes_bids",
            pa.list_(pa.field("element", _LEVEL_TYPE, nullable=True)),
            nullable=True,
        ),
        pa.field(
            "no_bids",
            pa.list_(pa.field("element", _LEVEL_TYPE, nullable=True)),
            nullable=True,
        ),
        pa.field("delta_side", pa.string(), nullable=True),
        pa.field("delta_price", _PRICE_TYPE, nullable=True),
        pa.field("quantity_delta", _QUANTITY_TYPE, nullable=True),
        pa.field("control_payload_json", pa.string(), nullable=True),
    ),
    metadata={
        b"arbiter_schema": _RECORDED_SCHEMA_NAME.encode("ascii"),
        b"arbiter_schema_version": str(_RECORDED_CONTAINER_SCHEMA_VERSION).encode("ascii"),
    },
)

_EVENT_ADAPTER: TypeAdapter[MarketDataEvent] = TypeAdapter(MarketDataEvent)


class ParquetStorageError(RuntimeError):
    """Raised when recorded events cannot be stored or reconstructed safely."""


class ParquetBackpressureError(ParquetStorageError):
    """Raised instead of silently dropping an event when the writer queue is full."""


_WRITER_LEASE_NAME = ".arbiter-writer.lock"
_MAX_EVENT_INDEX = 2**63 - 1
_BOOK_EVENT_TYPES = frozenset(("snapshot", "delta"))
_BOOK_ONLY_COLUMNS = (
    "exchange_ts",
    "ticker",
    "sequence",
    "sid",
    "connection_id",
    "price_convention",
    "snapshot_id",
    "yes_bids",
    "no_bids",
    "delta_side",
    "delta_price",
    "quantity_delta",
)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ParquetStorageError("recorded-event timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _level_row(level: PriceLevel) -> dict[str, Decimal]:
    return {"price": level.price, "quantity": level.quantity}


def _event_row(event: MarketDataEvent) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": event.schema_version,
        "event_index": event.event_index,
        "local_received_ts": _utc(event.local_received_ts),
        "exchange_ts": None if event.exchange_ts is None else _utc(event.exchange_ts),
        "ticker": event.ticker,
        "sequence": event.sequence,
        "sid": event.sid,
        "connection_id": event.connection_id,
        "price_convention": event.price_convention,
        "snapshot_id": event.snapshot_id,
        "event_type": event.event_type,
        "yes_bids": None,
        "no_bids": None,
        "delta_side": None,
        "delta_price": None,
        "quantity_delta": None,
    }
    if isinstance(event, OrderBookSnapshotEvent):
        row["yes_bids"] = [_level_row(level) for level in event.yes_bids]
        row["no_bids"] = [_level_row(level) for level in event.no_bids]
    else:
        row["delta_side"] = event.side
        row["delta_price"] = event.price
        row["quantity_delta"] = event.quantity_delta
    return row


def _is_market_data_event(event: RecordedEvent) -> TypeGuard[MarketDataEvent]:
    return isinstance(event, (OrderBookSnapshotEvent, OrderBookDeltaEvent))


def _recorded_event_row(event: RecordedEvent) -> dict[str, object]:
    if _is_market_data_event(event):
        return {**_event_row(event), "control_payload_json": None}
    return {
        "schema_version": event.schema_version,
        "event_index": event.event_index,
        "local_received_ts": _utc(event.local_received_ts),
        "exchange_ts": None,
        "ticker": None,
        "sequence": None,
        "sid": None,
        "connection_id": None,
        "price_convention": None,
        "snapshot_id": None,
        "event_type": event.event_type,
        "yes_bids": None,
        "no_bids": None,
        "delta_side": None,
        "delta_price": None,
        "quantity_delta": None,
        "control_payload_json": dump_recorded_event_json(event),
    }


def _table(events: Sequence[RecordedEvent], *, container_version: int) -> pa.Table:
    if container_version == MARKET_DATA_SCHEMA_VERSION:
        if not all(_is_market_data_event(event) for event in events):
            raise ParquetStorageError(
                "schema-v2 controls cannot be written to a legacy schema-v1 dataset"
            )
        rows = [_event_row(event) for event in events if _is_market_data_event(event)]
        schema = MARKET_DATA_ARROW_SCHEMA
    elif container_version == _RECORDED_CONTAINER_SCHEMA_VERSION:
        rows = [_recorded_event_row(event) for event in events]
        schema = RECORDED_EVENTS_ARROW_SCHEMA
    else:  # pragma: no cover - internal callers constrain this value
        raise ParquetStorageError(
            f"unsupported recorded-event container version {container_version}"
        )
    try:
        return pa.Table.from_pylist(rows, schema=schema)
    except (pa.ArrowException, ValueError, TypeError) as exc:
        raise ParquetStorageError(
            f"event values are incompatible with the Parquet schema: {exc}"
        ) from exc


def _require_replay_safe_order(events: Sequence[RecordedEvent]) -> None:
    if not events:
        return
    ordered = sorted(events, key=lambda event: event.event_index)
    indices = [event.event_index for event in ordered]
    if len(indices) != len(set(indices)):
        raise ParquetStorageError("duplicate event_index values are not replay-safe")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ParquetStorageError(
            "non-contiguous event_index values indicate a mixed or incomplete stream"
        )
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if _utc(current.local_received_ts) < _utc(previous.local_received_ts):
            raise ParquetStorageError(
                "local_received_ts must be nondecreasing in authoritative event_index order"
            )


def _partition_date(event: RecordedEvent) -> date:
    return _utc(event.local_received_ts).date()


def _write_batch(
    root: Path,
    events: Sequence[RecordedEvent],
    *,
    container_version: int,
) -> tuple[Path, ...]:
    """Write one contiguous batch, atomically replacing each completed partition file."""

    if not events:
        return ()
    _require_replay_safe_order(events)
    by_date: dict[date, list[RecordedEvent]] = defaultdict(list)
    for event in sorted(events, key=lambda item: item.event_index):
        by_date[_partition_date(event)].append(event)

    batch_id = uuid4().hex
    staged: list[tuple[Path, Path]] = []
    completed: list[Path] = []
    try:
        for partition_date, partition_events in sorted(by_date.items()):
            partition = root / f"date={partition_date.isoformat()}"
            partition.mkdir(parents=True, exist_ok=True)
            first = partition_events[0].event_index
            last = partition_events[-1].event_index
            name = f"part-{first:020d}-{last:020d}-{batch_id}.parquet"
            final_path = partition / name
            temporary_path = partition / f".{name}.tmp"
            staged.append((temporary_path, final_path))
            pq.write_table(
                _table(partition_events, container_version=container_version),
                temporary_path,
                compression="zstd",
                version="2.6",
                write_statistics=True,
            )

        for temporary_path, final_path in staged:
            os.replace(temporary_path, final_path)
            completed.append(final_path)
    except Exception as exc:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)
        for final_path in completed:
            final_path.unlink(missing_ok=True)
        if isinstance(exc, ParquetStorageError):
            raise
        raise ParquetStorageError(f"could not flush recorded-event Parquet batch: {exc}") from exc
    return tuple(completed)


class ParquetEventWriter:
    """Small bounded buffer whose overflow is an explicit collector failure.

    ``root`` is the order-book dataset directory, normally
    ``data/parquet/orderbooks``. Each successful flush creates one file per UTC date.
    Queue contents remain available for retry if writing fails. A first book-only flush
    locks a new dataset to legacy v1; a first flush containing any control locks it to
    the mixed-record v2 container. Container versions are never mixed within a dataset.
    """

    def __init__(self, root: str | Path, *, max_queue_size: int = 10_000) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self.root = Path(root)
        self.max_queue_size = max_queue_size
        self._pending: list[RecordedEvent] = []
        self._closed = False
        self._lease_handle: BinaryIO | None = None
        self._acquire_lease()
        try:
            existing, self._container_version = _read_recorded_dataset(self.root)
            self._next_event_index = 0 if not existing else existing[-1].event_index + 1
            self._last_local_received_ts = (
                None if not existing else _utc(existing[-1].local_received_ts)
            )
            if self._next_event_index > _MAX_EVENT_INDEX:
                raise ParquetStorageError("market-data event_index exhausted signed 64-bit storage")
        except BaseException:
            self._release_lease()
            raise

    @property
    def next_event_index(self) -> int:
        """Return the only event index that this leased writer will accept next."""

        return self._next_event_index

    @property
    def pending_count(self) -> int:
        """Return the number of accepted events not yet durably flushed."""

        return len(self._pending)

    def append(self, event: RecordedEvent) -> None:
        """Queue one event, rejecting overflow and broken event-index ordering."""

        if self._closed:
            raise ParquetStorageError("cannot append to a closed Parquet writer")
        if len(self._pending) >= self.max_queue_size:
            raise ParquetBackpressureError(
                f"market-data Parquet queue reached its {self.max_queue_size}-event limit"
            )
        if event.event_index == self._next_event_index - 1:
            raise ParquetStorageError(f"duplicate event_index {event.event_index}")
        if event.event_index != self._next_event_index:
            raise ParquetStorageError(
                "event_index must continue the leased dataset contiguously; "
                f"expected {self._next_event_index}, received {event.event_index}"
            )
        if self._container_version == MARKET_DATA_SCHEMA_VERSION and not _is_market_data_event(
            event
        ):
            raise ParquetStorageError(
                "cannot append a schema-v2 control to a legacy schema-v1 dataset"
            )
        received_at = _utc(event.local_received_ts)
        if self._last_local_received_ts is not None and received_at < self._last_local_received_ts:
            raise ParquetStorageError(
                "local_received_ts must be nondecreasing in authoritative event_index order"
            )
        self._pending.append(event)
        self._next_event_index += 1
        self._last_local_received_ts = received_at

    enqueue = append

    def extend(self, events: Iterable[RecordedEvent]) -> None:
        """Queue events in authoritative replay order."""

        for event in events:
            self.append(event)

    def flush(self) -> tuple[Path, ...]:
        """Persist pending events; retain the queue if any write step fails."""

        if self._closed:
            raise ParquetStorageError("cannot flush a closed Parquet writer")
        if not self._pending:
            return ()
        container_version = self._container_version
        if container_version is None:
            container_version = (
                MARKET_DATA_SCHEMA_VERSION
                if all(_is_market_data_event(event) for event in self._pending)
                else _RECORDED_CONTAINER_SCHEMA_VERSION
            )
        paths = _write_batch(
            self.root,
            self._pending,
            container_version=container_version,
        )
        self._pending.clear()
        self._container_version = container_version
        return paths

    def close(self) -> tuple[Path, ...]:
        """Flush and close the writer."""

        if self._closed:
            return ()
        paths = self.flush()
        self._closed = True
        self._release_lease()
        return paths

    def __enter__(self) -> ParquetEventWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self._closed = True
            self._release_lease()

    def __del__(self) -> None:
        """Release an abandoned lease without attempting an implicit data flush."""

        handle = getattr(self, "_lease_handle", None)
        self._lease_handle = None
        if handle is not None:
            with suppress(OSError):
                handle.close()

    def _acquire_lease(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.root / _WRITER_LEASE_NAME, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError as exc:
            raise ParquetStorageError(
                f"could not prepare market-data dataset lease at {self.root}: {exc}"
            ) from exc

        handle = os.fdopen(descriptor, "rb+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise ParquetStorageError(
                f"market-data dataset already has an active writer: {self.root}"
            ) from exc
        except OSError as exc:
            handle.close()
            raise ParquetStorageError(
                f"could not acquire market-data dataset lease at {self.root}: {exc}"
            ) from exc
        self._lease_handle = handle

    def _release_lease(self) -> None:
        handle = self._lease_handle
        self._lease_handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        if path.suffix != ".parquet":
            raise ParquetStorageError(f"not a Parquet file: {path}")
        return (path,)
    if not path.exists():
        raise ParquetStorageError(f"Parquet path does not exist: {path}")
    if not path.is_dir():
        raise ParquetStorageError(f"Parquet path is neither a file nor directory: {path}")
    return tuple(sorted(path.rglob("*.parquet")))


def _partition_name(path: Path) -> str | None:
    return next((part for part in reversed(path.parts) if part.startswith("date=")), None)


def _validate_file_schema(path: Path, parquet_file: pq.ParquetFile) -> int:
    actual = parquet_file.schema_arrow
    metadata = actual.metadata or {}
    schema_name = metadata.get(b"arbiter_schema")
    schema_version = metadata.get(b"arbiter_schema_version")
    if schema_name == _SCHEMA_NAME.encode("ascii") and schema_version == b"1":
        if not actual.equals(MARKET_DATA_ARROW_SCHEMA, check_metadata=True):
            raise ParquetStorageError(f"incompatible market-data Parquet schema in {path}")
        return MARKET_DATA_SCHEMA_VERSION
    if schema_name == _RECORDED_SCHEMA_NAME.encode("ascii") and schema_version == str(
        _RECORDED_CONTAINER_SCHEMA_VERSION
    ).encode("ascii"):
        if not actual.equals(RECORDED_EVENTS_ARROW_SCHEMA, check_metadata=True):
            raise ParquetStorageError(f"incompatible recorded-event Parquet schema in {path}")
        return _RECORDED_CONTAINER_SCHEMA_VERSION
    if schema_name in {
        _SCHEMA_NAME.encode("ascii"),
        _RECORDED_SCHEMA_NAME.encode("ascii"),
    }:
        raise ParquetStorageError(
            f"unsupported Parquet schema version {schema_version!r} in {path}"
        )
    raise ParquetStorageError(f"incompatible Arbiter Parquet schema metadata in {path}")


def _event_from_row(row: dict[str, object], *, path: Path) -> MarketDataEvent:
    event_type = row.get("event_type")
    common = {
        "schema_version": row.get("schema_version"),
        "event_index": row.get("event_index"),
        "local_received_ts": row.get("local_received_ts"),
        "exchange_ts": row.get("exchange_ts"),
        "ticker": row.get("ticker"),
        "sequence": row.get("sequence"),
        "sid": row.get("sid"),
        "connection_id": row.get("connection_id"),
        "price_convention": row.get("price_convention"),
        "snapshot_id": row.get("snapshot_id"),
        "event_type": event_type,
    }
    if event_type == "snapshot":
        if any(
            row.get(field) is not None for field in ("delta_side", "delta_price", "quantity_delta")
        ):
            raise ParquetStorageError(f"snapshot row contains delta fields in {path}")
        yes_bids = row.get("yes_bids")
        no_bids = row.get("no_bids")
        if not isinstance(yes_bids, list) or not isinstance(no_bids, list):
            raise ParquetStorageError(f"snapshot row is missing nested bid levels in {path}")
        payload = {
            **common,
            "yes_bids": tuple(yes_bids),
            "no_bids": tuple(no_bids),
        }
    elif event_type == "delta":
        if row.get("yes_bids") is not None or row.get("no_bids") is not None:
            raise ParquetStorageError(f"delta row contains snapshot levels in {path}")
        if any(row.get(field) is None for field in ("delta_side", "delta_price", "quantity_delta")):
            raise ParquetStorageError(f"delta row is missing scalar delta fields in {path}")
        payload = {
            **common,
            "side": row["delta_side"],
            "price": row["delta_price"],
            "quantity_delta": row["quantity_delta"],
        }
    else:
        raise ParquetStorageError(f"unknown market-data event type {event_type!r} in {path}")
    try:
        return _EVENT_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise ParquetStorageError(f"invalid market-data event in {path}: {exc}") from exc


def _recorded_event_from_row(row: dict[str, object], *, path: Path) -> RecordedEvent:
    event_type = row.get("event_type")
    payload = row.get("control_payload_json")
    if event_type in _BOOK_EVENT_TYPES:
        if payload is not None:
            raise ParquetStorageError(f"book row contains a control payload in {path}")
        return _event_from_row(row, path=path)

    if any(row.get(column) is not None for column in _BOOK_ONLY_COLUMNS):
        raise ParquetStorageError(f"control row contains book-only fields in {path}")
    if not isinstance(payload, str) or not payload:
        raise ParquetStorageError(f"control row is missing its canonical payload in {path}")
    try:
        event = load_recorded_event_json(payload)
    except (ValidationError, ValueError, TypeError) as exc:
        raise ParquetStorageError(f"invalid recorded control in {path}: {exc}") from exc
    if _is_market_data_event(event):
        raise ParquetStorageError(f"control payload reconstructs a book event in {path}")
    if dump_recorded_event_json(event) != payload:
        raise ParquetStorageError(f"control payload is not canonical JSON in {path}")

    outer_timestamp = row.get("local_received_ts")
    if not isinstance(outer_timestamp, datetime):
        raise ParquetStorageError(f"control row has an invalid local timestamp in {path}")
    if (
        row.get("schema_version") != event.schema_version
        or row.get("event_index") != event.event_index
        or _utc(outer_timestamp) != _utc(event.local_received_ts)
        or event_type != event.event_type
    ):
        raise ParquetStorageError(f"control payload does not match its row envelope in {path}")
    return event


def _read_recorded_dataset(
    path: Path,
) -> tuple[tuple[RecordedEvent, ...], int | None]:
    """Read one file/dataset and return its records plus its container version."""

    restored: list[RecordedEvent] = []
    container_version: int | None = None
    for file_path in _files(path):
        try:
            parquet_file = pq.ParquetFile(file_path)
            file_version = _validate_file_schema(file_path, parquet_file)
            if container_version is None:
                container_version = file_version
            elif file_version != container_version:
                raise ParquetStorageError(
                    "mixed schema-v1 and schema-v2 Parquet files are not a replay-safe dataset"
                )
            rows = parquet_file.read().to_pylist()
        except ParquetStorageError:
            raise
        except (pa.ArrowException, OSError, ValueError) as exc:
            raise ParquetStorageError(
                f"could not read recorded-event Parquet file {file_path}: {exc}"
            ) from exc
        partition_name = _partition_name(file_path)
        for raw_row in rows:
            event = (
                _event_from_row(raw_row, path=file_path)
                if file_version == MARKET_DATA_SCHEMA_VERSION
                else _recorded_event_from_row(raw_row, path=file_path)
            )
            if partition_name is not None:
                expected = f"date={_partition_date(event).isoformat()}"
                if partition_name != expected:
                    raise ParquetStorageError(
                        f"event UTC date belongs in {expected}, not {partition_name}"
                    )
            restored.append(event)

    restored.sort(key=lambda event: event.event_index)
    _require_replay_safe_order(restored)
    return tuple(restored), container_version


def read_recorded_events(path: str | Path) -> tuple[RecordedEvent, ...]:
    """Read and strictly validate one authoritative v1 or v2 recorded stream."""

    events, _ = _read_recorded_dataset(Path(path))
    return events


def read_market_data_events(path: str | Path) -> tuple[MarketDataEvent, ...]:
    """Validate the complete stream, then return its order-book records only."""

    return tuple(event for event in read_recorded_events(path) if _is_market_data_event(event))


def next_market_data_event_index(path: str | Path) -> int:
    """Validate an existing dataset and return its next contiguous replay index.

    A missing or empty dataset starts at zero. Existing rows are reconstructed through
    the full strict reader, so schema, partition, row, duplicate, timestamp, and gap failures
    stop collection before a new file can be mixed into an invalid stream.
    """

    dataset = Path(path)
    if not dataset.exists():
        return 0
    events = read_recorded_events(dataset)
    if not events:
        return 0
    last_index = events[-1].event_index
    if last_index >= _MAX_EVENT_INDEX:
        raise ParquetStorageError("market-data event_index exhausted signed 64-bit storage")
    return last_index + 1
