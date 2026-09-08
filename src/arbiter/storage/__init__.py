"""Versioned DuckDB persistence for metadata and research records."""

from arbiter.storage.duckdb import (
    DuckDBRepository,
    PaperExecutionRecord,
    RunManifestRecord,
    RunStreamEvidence,
    StorageError,
    paper_execution_attempt_id,
)
from arbiter.storage.parquet import (
    MARKET_DATA_ARROW_SCHEMA,
    RECORDED_EVENTS_ARROW_SCHEMA,
    ParquetBackpressureError,
    ParquetEventWriter,
    ParquetStorageError,
    next_market_data_event_index,
    read_market_data_events,
    read_recorded_events,
)

__all__ = [
    "DuckDBRepository",
    "MARKET_DATA_ARROW_SCHEMA",
    "RECORDED_EVENTS_ARROW_SCHEMA",
    "ParquetBackpressureError",
    "ParquetEventWriter",
    "ParquetStorageError",
    "PaperExecutionRecord",
    "RunManifestRecord",
    "RunStreamEvidence",
    "StorageError",
    "next_market_data_event_index",
    "paper_execution_attempt_id",
    "read_market_data_events",
    "read_recorded_events",
]
