"""Transactional DuckDB repository for normalized exchange metadata."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import duckdb
from pydantic import ValidationError

from arbiter.engine.paper_execution import (
    PaperExecutionRequest,
    PaperExecutionResult,
    PaperExecutionStatus,
)
from arbiter.kalshi.normalize import normalize_event, normalize_market, normalize_series
from arbiter.models.event import Event, EventFeeChange
from arbiter.models.market import Market
from arbiter.models.opportunity import (
    EffectiveFeePolicy,
    OpportunityObservation,
    OpportunityTransition,
)
from arbiter.models.relation import Relation, RelationType
from arbiter.models.series import Series, SettlementSource
from arbiter.relations.semantic import (
    SemanticEmbedding,
    SemanticMarketDocument,
    SemanticRelationProposal,
    SemanticReviewState,
    SemanticSuggestion,
    build_semantic_document,
    parse_classifier_output,
    relation_from_semantic_suggestion,
    semantic_suggestion_payload_hash,
    semantic_verified_relation_id,
)
from arbiter.relations.validator import validate_relations
from arbiter.storage.migrations import MIGRATIONS


class StorageError(RuntimeError):
    """Raised when a transactional persistence operation fails."""


class _Connection(Protocol):
    def execute(self, query: str, parameters: Sequence[object] | None = None) -> _Connection: ...

    def executemany(self, query: str, parameters: Iterable[Sequence[object]]) -> _Connection: ...

    def fetchone(self) -> tuple[Any, ...] | None: ...

    def fetchall(self) -> list[tuple[Any, ...]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MarketSummary:
    """Small projection used by the CLI rather than leaking database tuples."""

    ticker: str
    event_ticker: str
    series_ticker: str | None
    title: str
    status: str
    close_time: datetime | None


def market_observation_id(*, run_id: str, market_ticker: str, opened_event_index: int) -> str:
    """Return the retry-safe identity of one market observation window."""

    _nonblank(run_id, label="run ID")
    _nonblank(market_ticker, label="market ticker")
    if opened_event_index < 0:
        raise ValueError("opened event index cannot be negative")
    payload = json.dumps(
        [run_id, market_ticker, opened_event_index],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"market-window:{sha256(payload.encode()).hexdigest()}"


def paper_execution_attempt_id(
    *,
    run_id: str,
    opportunity_id: str,
    source_observation_id: str,
    source_event_index: int,
) -> str:
    """Return the retry-safe identity of one paper attempt source decision."""

    _nonblank(run_id, label="run ID")
    _nonblank(opportunity_id, label="opportunity ID")
    _nonblank(source_observation_id, label="source observation ID")
    if source_event_index < 0:
        raise ValueError("source event index cannot be negative")
    payload = json.dumps(
        [run_id, opportunity_id, source_observation_id, source_event_index],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"paper-attempt:{sha256(payload.encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class RunManifestRecord:
    """Immutable inputs that identify one live or replay scanner run."""

    run_id: str
    run_type: str
    started_at: datetime
    schema_version: int = field(default_factory=lambda: MIGRATIONS[-1].version)
    manifest_version: int = 1
    recording_format_version: int = 1
    event_schema_version: int = 1
    recording_id: str | None = None
    source_run_id: str | None = None
    first_event_index: int | None = None
    last_event_index: int | None = None
    event_count: int = 0
    event_stream_hash: str | None = None
    input_payload: dict[str, object] = field(default_factory=dict)
    config_hash: str | None = None
    metadata_hash: str | None = None
    relations_hash: str | None = None
    fee_policy_hash: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonblank(self.run_id, label="run ID")
        _nonblank(self.run_type, label="run type")
        _require_aware(self.started_at, label="run start")
        for version_label, version_value in (
            ("schema version", self.schema_version),
            ("manifest version", self.manifest_version),
            ("recording format version", self.recording_format_version),
            ("event schema version", self.event_schema_version),
        ):
            if (
                isinstance(version_value, bool)
                or not isinstance(version_value, int)
                or version_value < 1
            ):
                raise ValueError(f"{version_label} must be a positive integer")
        for identifier_label, identifier_value in (
            ("recording ID", self.recording_id),
            ("source run ID", self.source_run_id),
        ):
            if identifier_value is not None:
                _nonblank(identifier_value, label=identifier_label)
        _validate_stream_fields(
            first_event_index=self.first_event_index,
            last_event_index=self.last_event_index,
            event_count=self.event_count,
            event_stream_hash=self.event_stream_hash,
            allow_pending=True,
        )
        for hash_label, hash_value in (
            ("config hash", self.config_hash),
            ("metadata hash", self.metadata_hash),
            ("relations hash", self.relations_hash),
            ("fee-policy hash", self.fee_policy_hash),
        ):
            if hash_value is not None:
                _nonblank(hash_value, label=hash_label)
        _reject_sensitive_manifest_input(self.input_payload, path="input_payload")
        if self.manifest_version >= 2:
            if self.recording_id is None:
                raise ValueError("manifest version 2 requires a recording ID")
            for required_hash_label, required_hash_value in (
                ("config hash", self.config_hash),
                ("metadata hash", self.metadata_hash),
                ("relations hash", self.relations_hash),
                ("fee-policy hash", self.fee_policy_hash),
            ):
                if required_hash_value is None or not _is_sha256(required_hash_value):
                    raise ValueError(
                        f"manifest version 2 requires a lowercase SHA-256 {required_hash_label}"
                    )
            _validate_input_payload_hashes(self)


@dataclass(frozen=True, slots=True)
class RunStreamEvidence:
    """Final contiguous recorded-stream bounds and content identity."""

    first_event_index: int
    last_event_index: int
    event_count: int
    event_stream_hash: str

    def __post_init__(self) -> None:
        _validate_stream_fields(
            first_event_index=self.first_event_index,
            last_event_index=self.last_event_index,
            event_count=self.event_count,
            event_stream_hash=self.event_stream_hash,
            allow_pending=False,
        )


@dataclass(frozen=True, slots=True)
class StoredRunManifest:
    """Query projection for a durable scanner run manifest."""

    run_id: str
    run_type: str
    started_at: datetime
    ended_at: datetime | None
    status: str
    schema_version: int
    manifest_version: int
    recording_format_version: int
    event_schema_version: int
    recording_id: str | None
    source_run_id: str | None
    first_event_index: int | None
    last_event_index: int | None
    event_count: int
    event_stream_hash: str | None
    input_payload: dict[str, object]
    config_hash: str | None
    metadata_hash: str | None
    relations_hash: str | None
    fee_policy_hash: str | None
    metadata: dict[str, object]
    error: str | None


@dataclass(frozen=True, slots=True)
class PaperExecutionRecord:
    """One source-identified, retry-safe paper result and its resolution time."""

    attempt_id: str
    run_id: str
    source_observation_id: str
    source_event_index: int
    request: PaperExecutionRequest
    result: PaperExecutionResult
    resolved_at: datetime
    evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_attempt_id = paper_execution_attempt_id(
            run_id=self.run_id,
            opportunity_id=self.request.opportunity_id,
            source_observation_id=self.source_observation_id,
            source_event_index=self.source_event_index,
        )
        if self.attempt_id != expected_attempt_id:
            raise ValueError("paper attempt ID does not match its source identity")
        _require_aware(self.resolved_at, label="paper resolution time")
        if self.result.opportunity_id != self.request.opportunity_id:
            raise ValueError("paper request and result opportunity IDs differ")
        if self.result.detected_at != self.request.detected_at:
            raise ValueError("paper request and result detection times differ")
        if self.result.simulated_execution_at != self.request.execute_at:
            raise ValueError("paper request and result scheduled times differ")
        if self.result.latency_ms != self.request.latency_ms:
            raise ValueError("paper request and result latencies differ")
        if self.resolved_at < self.request.detected_at:
            raise ValueError("paper result cannot resolve before detection")
        if self.result.status is PaperExecutionStatus.INSUFFICIENT_FUTURE_DATA:
            if self.resolved_at >= self.request.execute_at:
                raise ValueError("insufficient-future-data resolution must precede its deadline")
        elif self.resolved_at < self.request.execute_at:
            raise ValueError("an attempted paper execution cannot resolve before its deadline")
        _reject_sensitive_manifest_input(self.evidence, path="paper_evidence")

    @property
    def attempted_at(self) -> datetime | None:
        """Return the simulated attempt time, absent when coverage ends first."""

        if self.result.status is PaperExecutionStatus.INSUFFICIENT_FUTURE_DATA:
            return None
        return self.result.simulated_execution_at


@dataclass(frozen=True, slots=True)
class MarketObservationRecord:
    """One idempotent fresh/stale/resync window for a subscribed market."""

    observation_id: str
    run_id: str
    market_ticker: str
    opened_at: datetime
    updated_at: datetime
    opened_event_index: int
    last_event_index: int
    status: str
    closed_at: datetime | None = None
    closed_event_index: int | None = None
    start_sequence: int | None = None
    end_sequence: int | None = None
    connection_id: str | None = None
    stale_reason: str | None = None
    resync_reason: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("observation ID", self.observation_id),
            ("run ID", self.run_id),
            ("market ticker", self.market_ticker),
            ("observation status", self.status),
        ):
            _nonblank(value, label=label)
        if self.observation_id != market_observation_id(
            run_id=self.run_id,
            market_ticker=self.market_ticker,
            opened_event_index=self.opened_event_index,
        ):
            raise ValueError("market observation ID does not match its identity fields")
        _require_aware(self.opened_at, label="observation open time")
        _require_aware(self.updated_at, label="observation update time")
        if self.closed_at is not None:
            _require_aware(self.closed_at, label="observation close time")
        if self.opened_event_index < 0 or self.last_event_index < self.opened_event_index:
            raise ValueError("market observation event indexes are inconsistent")
        if self.updated_at < self.opened_at:
            raise ValueError("market observation update cannot precede its open")
        if (self.closed_at is None) != (self.closed_event_index is None):
            raise ValueError("market observation close time and event index must be paired")
        if self.closed_at is not None:
            assert self.closed_event_index is not None
            if self.closed_at < self.updated_at or self.closed_event_index < self.last_event_index:
                raise ValueError("market observation close cannot precede its last update")
        for label, index_value in (
            ("opened event index", self.opened_event_index),
            ("last event index", self.last_event_index),
            ("closed event index", self.closed_event_index),
            ("start sequence", self.start_sequence),
            ("end sequence", self.end_sequence),
        ):
            if index_value is not None and index_value < 0:
                raise ValueError(f"{label} cannot be negative")
        if (
            self.start_sequence is not None
            and self.end_sequence is not None
            and self.end_sequence < self.start_sequence
        ):
            raise ValueError("market observation end sequence cannot precede its start")


@dataclass(frozen=True, slots=True)
class EventFeeState:
    """Persisted current event override and its complete scheduled change set."""

    event_ticker: str
    fee_type_override: str | None
    fee_multiplier_override: Decimal | None
    fee_changes: tuple[EventFeeChange, ...]
    updated_at: datetime


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return _require_aware(value, label="JSON datetime").isoformat()
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("JSON decimal must be finite")
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )


_OBSERVATION_PAYLOAD_HASH_KEY = "_observation_payload_sha256"


def _observation_payload_hash(observation: OpportunityObservation) -> str:
    payload = _json(observation.model_dump(mode="json"))
    return sha256(payload.encode()).hexdigest()


def _paper_execution_payload_hash(record: PaperExecutionRecord) -> str:
    payload = _json(
        {
            "attempt_id": record.attempt_id,
            "evidence": record.evidence,
            "request": record.request.model_dump(mode="json"),
            "resolved_at": record.resolved_at,
            "result": record.result.model_dump(mode="json"),
            "run_id": record.run_id,
            "source_event_index": record.source_event_index,
            "source_observation_id": record.source_observation_id,
        }
    )
    return sha256(payload.encode()).hexdigest()


def _require_aware(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _nonblank(value: str, *, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} cannot be blank")
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_stream_fields(
    *,
    first_event_index: int | None,
    last_event_index: int | None,
    event_count: int,
    event_stream_hash: str | None,
    allow_pending: bool,
) -> None:
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 0:
        raise ValueError("event count must be a nonnegative integer")
    supplied = (
        first_event_index is not None,
        last_event_index is not None,
        event_stream_hash is not None,
    )
    if event_count == 0:
        if any(supplied):
            raise ValueError("an empty or pending stream cannot contain bounds or a hash")
        if not allow_pending:
            raise ValueError("final stream evidence cannot be empty")
        return
    if not all(supplied):
        raise ValueError("a nonempty stream requires both bounds and a SHA-256 hash")
    assert first_event_index is not None
    assert last_event_index is not None
    assert event_stream_hash is not None
    for label, value in (
        ("first event index", first_event_index),
        ("last event index", last_event_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
    if last_event_index < first_event_index:
        raise ValueError("last event index cannot precede first event index")
    if event_count != last_event_index - first_event_index + 1:
        raise ValueError("event count must equal the contiguous stream bounds")
    if not _is_sha256(event_stream_hash):
        raise ValueError("event stream hash must be a lowercase SHA-256 digest")


def _reject_sensitive_manifest_input(value: object, *, path: str) -> None:
    sensitive_keys = {
        "api_key",
        "api_key_id",
        "credential",
        "credentials",
        "password",
        "private_key",
        "private_key_path",
        "secret",
        "token",
    }
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold()
            if key in sensitive_keys or "private_key" in key:
                raise ValueError(
                    f"persistence payload cannot contain sensitive key {path}.{raw_key}"
                )
            _reject_sensitive_manifest_input(child, path=f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_sensitive_manifest_input(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and "-----BEGIN" in value and "PRIVATE KEY-----" in value:
        raise ValueError(f"persistence payload cannot contain private key material at {path}")


def _canonical_object_json(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be canonical JSON text")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(parsed, dict) or _json(parsed) != value:
        raise ValueError(f"{label} must be a canonical JSON object")
    return value


def _validate_input_payload_hashes(record: RunManifestRecord) -> None:
    payload = record.input_payload
    required_keys = {
        "config_json",
        "markets",
        "events",
        "series",
        "relations",
        "fee_policy_json",
    }
    if set(payload) != required_keys:
        raise ValueError("manifest version 2 requires one complete canonical run input payload")
    config_json = _canonical_object_json(payload["config_json"], label="configuration JSON")
    fee_policy_json = _canonical_object_json(
        payload["fee_policy_json"],
        label="fee-policy JSON",
    )
    metadata = {
        "markets": payload["markets"],
        "events": payload["events"],
        "series": payload["series"],
    }
    expected = {
        "config_hash": sha256(config_json.encode()).hexdigest(),
        "metadata_hash": sha256(_json(metadata).encode()).hexdigest(),
        "relations_hash": sha256(_json(payload["relations"]).encode()).hexdigest(),
        "fee_policy_hash": sha256(fee_policy_json.encode()).hexdigest(),
    }
    for field_name, expected_value in expected.items():
        if getattr(record, field_name) != expected_value:
            raise ValueError(f"{field_name} does not match the immutable input payload")


def _semantic_embedding_payload_hash(embedding: SemanticEmbedding) -> str:
    payload = _json(
        {
            "embedding_id": embedding.embedding_id,
            "market_ticker": embedding.market_ticker,
            "canonical_text": embedding.canonical_text,
            "canonical_text_hash": embedding.canonical_text_hash,
            "rules_hash": embedding.rules_hash,
            "timing_hash": embedding.timing_hash,
            "provider": embedding.provider,
            "model": embedding.model,
            "vector": list(embedding.vector),
        }
    )
    return sha256(payload.encode()).hexdigest()


def _semantic_evidence_pair_hash(
    *,
    market_a_ticker: str,
    market_a_hash: str,
    market_b_ticker: str,
    market_b_hash: str,
) -> str:
    payload = _json(
        [
            [market_a_ticker, market_a_hash],
            [market_b_ticker, market_b_hash],
        ]
    )
    return sha256(payload.encode()).hexdigest()


def _semantic_document_matches_suggestion(
    suggestion: SemanticSuggestion,
    document: SemanticMarketDocument,
) -> bool:
    if document.ticker == suggestion.market_a_ticker:
        return (
            document.canonical_text_hash == suggestion.market_a_text_hash
            and document.rules_hash == suggestion.market_a_rules_hash
            and document.timing_hash == suggestion.market_a_timing_hash
        )
    if document.ticker == suggestion.market_b_ticker:
        return (
            document.canonical_text_hash == suggestion.market_b_text_hash
            and document.rules_hash == suggestion.market_b_rules_hash
            and document.timing_hash == suggestion.market_b_timing_hash
        )
    raise ValueError("semantic document does not belong to the suggestion")


_SEMANTIC_EMBEDDING_COLUMNS = (
    "embedding_id",
    "market_ticker",
    "canonical_text",
    "canonical_text_hash",
    "rules_hash",
    "timing_hash",
    "provider",
    "model",
    "dimensions",
    "vector_json",
    "payload_hash",
    "created_at",
    "updated_at",
)

_SEMANTIC_SUGGESTION_COLUMNS = (
    "suggestion_id",
    "market_a_ticker",
    "market_b_ticker",
    "market_a_text_hash",
    "market_b_text_hash",
    "market_a_rules_hash",
    "market_b_rules_hash",
    "market_a_timing_hash",
    "market_b_timing_hash",
    "market_a_title",
    "market_b_title",
    "market_a_rules_text",
    "market_b_rules_text",
    "market_a_timing_text",
    "market_b_timing_text",
    "embedding_provider",
    "embedding_model",
    "cosine_similarity",
    "classifier_provider",
    "classifier_model",
    "prompt_version",
    "prompt",
    "raw_response",
    "parsed_proposal_json",
    "relation",
    "confidence",
    "rationale",
    "requires_review",
    "review_state",
    "reviewed_at",
    "approved_relation_id",
    "stale_reason",
    "payload_hash",
    "created_at",
    "updated_at",
)


def _semantic_embedding_row(embedding: SemanticEmbedding) -> tuple[object, ...]:
    return (
        embedding.embedding_id,
        embedding.market_ticker,
        embedding.canonical_text,
        embedding.canonical_text_hash,
        embedding.rules_hash,
        embedding.timing_hash,
        embedding.provider,
        embedding.model,
        len(embedding.vector),
        _json(list(embedding.vector)),
        _semantic_embedding_payload_hash(embedding),
        _require_aware(embedding.created_at, label="semantic embedding creation time"),
        _require_aware(embedding.updated_at, label="semantic embedding update time"),
    )


def _semantic_embedding_from_row(row: Sequence[object]) -> SemanticEmbedding:
    raw_vector = _json_value(row[9])
    if not isinstance(raw_vector, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_vector
    ):
        raise StorageError("stored semantic embedding vector is invalid")
    dimensions = int(cast(int, row[8]))
    if dimensions != len(raw_vector):
        raise StorageError("stored semantic embedding dimensions do not match its vector")
    try:
        embedding = SemanticEmbedding(
            embedding_id=str(row[0]),
            market_ticker=str(row[1]),
            canonical_text=str(row[2]),
            canonical_text_hash=str(row[3]),
            rules_hash=str(row[4]),
            timing_hash=str(row[5]),
            provider=str(row[6]),
            model=str(row[7]),
            vector=tuple(float(value) for value in raw_vector),
            created_at=cast(datetime, row[11]),
            updated_at=cast(datetime, row[12]),
        )
    except ValidationError as exc:
        raise StorageError("stored semantic embedding is invalid") from exc
    if _semantic_embedding_payload_hash(embedding) != str(row[10]):
        raise StorageError("stored semantic embedding payload hash is invalid")
    return embedding


def _semantic_suggestion_row(suggestion: SemanticSuggestion) -> tuple[object, ...]:
    parsed = parse_classifier_output(suggestion.raw_response)
    if (
        parsed.relation is not suggestion.relation
        or parsed.confidence != suggestion.confidence
        or parsed.rationale != suggestion.rationale
        or parsed.requires_review is not suggestion.requires_review
    ):
        raise ValueError("semantic raw response conflicts with its parsed proposal")
    parsed_proposal = {
        "relation": parsed.relation.value,
        "confidence": parsed.confidence,
        "rationale": parsed.rationale,
        "requires_review": parsed.requires_review,
    }
    return (
        suggestion.suggestion_id,
        suggestion.market_a_ticker,
        suggestion.market_b_ticker,
        suggestion.market_a_text_hash,
        suggestion.market_b_text_hash,
        suggestion.market_a_rules_hash,
        suggestion.market_b_rules_hash,
        suggestion.market_a_timing_hash,
        suggestion.market_b_timing_hash,
        suggestion.market_a_title,
        suggestion.market_b_title,
        suggestion.market_a_rules_text,
        suggestion.market_b_rules_text,
        suggestion.market_a_timing_text,
        suggestion.market_b_timing_text,
        suggestion.embedding_provider,
        suggestion.embedding_model,
        suggestion.cosine_similarity,
        suggestion.classifier_provider,
        suggestion.classifier_model,
        suggestion.prompt_version,
        suggestion.prompt,
        suggestion.raw_response,
        _json(parsed_proposal),
        suggestion.relation.value,
        suggestion.confidence,
        suggestion.rationale,
        suggestion.requires_review,
        suggestion.review_state.value,
        (
            None
            if suggestion.reviewed_at is None
            else _require_aware(suggestion.reviewed_at, label="semantic review time")
        ),
        suggestion.approved_relation_id,
        suggestion.stale_reason,
        semantic_suggestion_payload_hash(suggestion),
        _require_aware(suggestion.created_at, label="semantic suggestion creation time"),
        _require_aware(suggestion.updated_at, label="semantic suggestion update time"),
    )


def _semantic_suggestion_from_row(row: Sequence[object]) -> SemanticSuggestion:
    raw_parsed = _json_mapping(row[23])
    requires_review = bool(row[27])
    if not requires_review:
        raise StorageError("stored semantic suggestion bypasses mandatory review")
    expected_parsed = {
        "relation": str(row[24]),
        "confidence": float(cast(float, row[25])),
        "rationale": str(row[26]),
        "requires_review": True,
    }
    if raw_parsed != expected_parsed:
        raise StorageError("stored semantic parsed proposal conflicts with its columns")
    try:
        suggestion = SemanticSuggestion(
            suggestion_id=str(row[0]),
            market_a_ticker=str(row[1]),
            market_b_ticker=str(row[2]),
            market_a_text_hash=str(row[3]),
            market_b_text_hash=str(row[4]),
            market_a_rules_hash=str(row[5]),
            market_b_rules_hash=str(row[6]),
            market_a_timing_hash=str(row[7]),
            market_b_timing_hash=str(row[8]),
            market_a_title=str(row[9]),
            market_b_title=str(row[10]),
            market_a_rules_text=str(row[11]),
            market_b_rules_text=str(row[12]),
            market_a_timing_text=str(row[13]),
            market_b_timing_text=str(row[14]),
            embedding_provider=str(row[15]),
            embedding_model=str(row[16]),
            cosine_similarity=float(cast(float, row[17])),
            classifier_provider=str(row[18]),
            classifier_model=str(row[19]),
            prompt_version=str(row[20]),
            prompt=str(row[21]),
            raw_response=str(row[22]),
            relation=SemanticRelationProposal(str(row[24])),
            confidence=float(cast(float, row[25])),
            rationale=str(row[26]),
            requires_review=True,
            review_state=SemanticReviewState(str(row[28])),
            reviewed_at=None if row[29] is None else cast(datetime, row[29]),
            approved_relation_id=None if row[30] is None else str(row[30]),
            stale_reason=None if row[31] is None else str(row[31]),
            created_at=cast(datetime, row[33]),
            updated_at=cast(datetime, row[34]),
        )
    except (ValidationError, ValueError) as exc:
        raise StorageError("stored semantic suggestion is invalid") from exc
    if semantic_suggestion_payload_hash(suggestion) != str(row[32]):
        raise StorageError("stored semantic suggestion payload hash is invalid")
    return suggestion


_MARKET_COLUMNS = (
    "ticker",
    "event_ticker",
    "series_ticker",
    "market_type",
    "title",
    "subtitle",
    "yes_sub_title",
    "no_sub_title",
    "status",
    "created_time",
    "market_updated_time",
    "open_time",
    "close_time",
    "expiration_time",
    "latest_expiration_time",
    "expected_expiration_time",
    "settlement_ts",
    "occurrence_datetime",
    "strike_type",
    "floor_strike",
    "cap_strike",
    "functional_strike",
    "custom_strike_json",
    "rules_primary",
    "rules_secondary",
    "early_close_condition",
    "price_level_structure",
    "price_ranges_json",
    "result",
    "raw_json",
    "updated_at",
)

_EVENT_COLUMNS = (
    "ticker",
    "series_ticker",
    "title",
    "subtitle",
    "category",
    "mutually_exclusive",
    "available_on_brokers",
    "market_tickers_json",
    "last_updated_ts",
    "fee_type_override",
    "fee_multiplier_override",
    "fee_changes_json",
    "raw_json",
    "updated_at",
)

_SERIES_COLUMNS = (
    "ticker",
    "title",
    "frequency",
    "category",
    "tags_json",
    "fee_type",
    "fee_multiplier",
    "settlement_sources_json",
    "contract_url",
    "contract_terms_url",
    "last_updated_ts",
    "raw_json",
    "updated_at",
)


class DuckDBRepository:
    """Apply migrations and atomically upsert one complete metadata sync."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
        write_max_attempts: int = 3,
        write_initial_backoff_seconds: float = 0.05,
        write_max_backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if write_max_attempts < 1:
            raise ValueError("storage write attempts must be positive")
        if (
            not math.isfinite(write_initial_backoff_seconds)
            or write_initial_backoff_seconds < 0
            or not math.isfinite(write_max_backoff_seconds)
            or write_max_backoff_seconds < write_initial_backoff_seconds
        ):
            raise ValueError("storage write backoff bounds are invalid")
        self.path = str(path)
        try:
            self.connection = cast(_Connection, duckdb.connect(self.path))
        except Exception as exc:
            raise StorageError(f"could not open DuckDB database {self.path}: {exc}") from exc
        self._clock = clock
        self._write_max_attempts = write_max_attempts
        self._write_initial_backoff_seconds = write_initial_backoff_seconds
        self._write_max_backoff_seconds = write_max_backoff_seconds
        self._sleep = sleep

    def __enter__(self) -> DuckDBRepository:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _write_transaction(self, operation: str, action: Callable[[], None]) -> None:
        """Run one idempotent transaction with finite transient-only retries."""

        delay = self._write_initial_backoff_seconds
        for attempt in range(1, self._write_max_attempts + 1):
            try:
                self.connection.execute("BEGIN TRANSACTION")
                action()
                self.connection.execute("COMMIT")
                return
            except (duckdb.OperationalError, duckdb.TransactionException) as exc:
                with suppress(Exception):
                    self.connection.execute("ROLLBACK")
                if attempt == self._write_max_attempts:
                    raise StorageError(
                        f"{operation} failed after {attempt} transient attempt(s): {exc}"
                    ) from exc
                self._sleep(delay)
                delay = min(delay * 2, self._write_max_backoff_seconds)
            except Exception as exc:
                with suppress(Exception):
                    self.connection.execute("ROLLBACK")
                raise StorageError(f"{operation} failed without retry: {exc}") from exc

    def _require_running_run(self, run_id: str) -> None:
        row = self.connection.execute(
            "SELECT status FROM run_manifests WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError("persistence requires an existing run manifest")
        if str(row[0]) != "running":
            raise ValueError("persistence requires a running run manifest")

    def migrate(self) -> None:
        """Apply each pending append-only migration in its own transaction."""

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        applied = {
            int(row[0])
            for row in self.connection.execute("SELECT version FROM schema_migrations").fetchall()
        }
        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            try:
                self.connection.execute("BEGIN TRANSACTION")
                self.connection.execute(migration.sql)
                self.connection.execute(
                    "INSERT INTO schema_migrations VALUES (?, ?, ?)",
                    (migration.version, migration.name, self._clock()),
                )
                self.connection.execute("COMMIT")
            except Exception as exc:
                with suppress(Exception):
                    self.connection.execute("ROLLBACK")
                raise StorageError(
                    f"migration {migration.version} ({migration.name}) failed: {exc}"
                ) from exc

    def _upsert(
        self,
        table: str,
        columns: tuple[str, ...],
        rows: Sequence[Sequence[object]],
    ) -> None:
        if not rows:
            return
        placeholders = ", ".join("?" for _ in columns)
        assignments = ", ".join(
            f"{column} = excluded.{column}" for column in columns if column != "ticker"
        )
        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT (ticker) DO UPDATE SET {assignments}"
        )
        self.connection.executemany(sql, rows)

    def sync_metadata(
        self,
        markets: Sequence[Market],
        events: Sequence[Event],
        series: Sequence[Series],
        *,
        run_id: str | None = None,
    ) -> str:
        """Upsert one sync atomically while retaining a durable failure record."""

        self.migrate()
        semantic_tickers = {
            str(ticker)
            for row in self.connection.execute(
                """
                SELECT market_a_ticker, market_b_ticker
                FROM semantic_suggestions
                WHERE review_state IN ('pending', 'approved', 'uncertain')
                """
            ).fetchall()
            for ticker in row
        }
        identifier = str(uuid4()) if run_id is None else run_id
        started_at = self._clock()
        self.connection.execute(
            """
            INSERT INTO metadata_sync_runs
                (run_id, started_at, status, market_count, event_count, series_count)
            VALUES (?, ?, 'running', 0, 0, 0)
            """,
            (identifier, started_at),
        )
        try:
            semantic_documents: dict[str, SemanticMarketDocument] = {}
            if semantic_tickers:
                markets_by_ticker = {market.ticker: market for market in markets}
                events_by_ticker = {event.ticker: event for event in events}
                series_by_ticker = {item.ticker: item for item in series}
                for ticker in sorted(semantic_tickers & set(markets_by_ticker)):
                    market = markets_by_ticker[ticker]
                    event = events_by_ticker.get(market.event_ticker)
                    if event is None:
                        raise ValueError(
                            f"semantic market {ticker} is missing event metadata "
                            f"{market.event_ticker}"
                        )
                    series_ticker = market.series_ticker or event.series_ticker
                    parent_series = (
                        None if series_ticker is None else series_by_ticker.get(series_ticker)
                    )
                    if series_ticker is not None and parent_series is None:
                        raise ValueError(
                            f"semantic market {ticker} is missing series metadata {series_ticker}"
                        )
                    semantic_documents[ticker] = build_semantic_document(
                        market,
                        event=event,
                        series=parent_series,
                    )
            self.connection.execute("BEGIN TRANSACTION")
            updated_at = self._clock()
            self._upsert(
                "markets",
                _MARKET_COLUMNS,
                tuple(_market_row(market, updated_at) for market in markets),
            )
            self._upsert(
                "events",
                _EVENT_COLUMNS,
                tuple(_event_row(event, updated_at) for event in events),
            )
            self._upsert(
                "series",
                _SERIES_COLUMNS,
                tuple(_series_row(item, updated_at) for item in series),
            )
            self._reconcile_semantic_staleness_in_transaction(
                semantic_documents,
                updated_at=updated_at,
                expected_tickers=semantic_tickers,
            )
            self.connection.execute(
                """
                UPDATE metadata_sync_runs
                SET completed_at = ?, status = 'succeeded', market_count = ?,
                    event_count = ?, series_count = ?
                WHERE run_id = ?
                """,
                (self._clock(), len(markets), len(events), len(series), identifier),
            )
            self.connection.execute("COMMIT")
        except Exception as exc:
            with suppress(Exception):
                self.connection.execute("ROLLBACK")
            with suppress(Exception):
                self.connection.execute(
                    """
                    UPDATE metadata_sync_runs
                    SET completed_at = ?, status = 'failed', error = ?
                    WHERE run_id = ?
                    """,
                    (self._clock(), str(exc)[:2000], identifier),
                )
            raise StorageError(f"metadata sync {identifier} failed: {exc}") from exc
        return identifier

    def list_markets(self, *, limit: int = 100) -> tuple[MarketSummary, ...]:
        """Return a deterministic human-facing market summary."""

        if limit < 1:
            raise ValueError("market list limit must be positive")
        self.migrate()
        rows = self.connection.execute(
            """
            SELECT ticker, event_ticker, series_ticker, title, status, close_time
            FROM markets
            ORDER BY ticker
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return tuple(
            MarketSummary(
                ticker=str(row[0]),
                event_ticker=str(row[1]),
                series_ticker=None if row[2] is None else str(row[2]),
                title=str(row[3]),
                status=str(row[4]),
                close_time=cast(datetime | None, row[5]),
            )
            for row in rows
        )

    def table_count(self, table: str) -> int:
        """Return a count for one known metadata table, primarily for diagnostics/tests."""

        if table not in {
            "markets",
            "events",
            "series",
            "relations",
            "opportunities",
            "opportunity_observations",
            "opportunity_observation_legs",
            "portfolio_legs",
            "paper_executions",
            "paper_execution_legs",
            "metadata_sync_runs",
            "run_manifests",
            "market_observation_windows",
            "semantic_embeddings",
            "semantic_suggestions",
        }:
            raise ValueError("unsupported table")
        row = self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()
        if row is None:
            raise StorageError(f"could not count table {table}")
        return int(row[0])

    def known_market_tickers(self) -> set[str]:
        """Return every locally synchronized market identifier."""

        self.migrate()
        return {
            str(row[0]) for row in self.connection.execute("SELECT ticker FROM markets").fetchall()
        }

    def load_markets(self) -> tuple[Market, ...]:
        """Rebuild strict market models from preserved raw payloads and ancestry."""

        self.migrate()
        rows = self.connection.execute(
            "SELECT series_ticker, raw_json FROM markets ORDER BY ticker"
        ).fetchall()
        return tuple(
            normalize_market(
                _json_mapping(row[1]),
                series_ticker=None if row[0] is None else str(row[0]),
            )
            for row in rows
        )

    def load_events(self) -> tuple[Event, ...]:
        """Rebuild events while restoring the persisted current market membership."""

        self.migrate()
        rows = self.connection.execute(
            """
            SELECT market_tickers_json, fee_changes_json, raw_json,
                   fee_type_override, fee_multiplier_override
            FROM events
            ORDER BY ticker
            """
        ).fetchall()
        events: list[Event] = []
        for (
            market_tickers_json,
            fee_changes_json,
            raw_json,
            fee_type_override,
            fee_multiplier_override,
        ) in rows:
            raw_tickers = _json_value(market_tickers_json)
            if not isinstance(raw_tickers, list) or any(
                not isinstance(ticker, str) for ticker in raw_tickers
            ):
                raise StorageError("stored event market_tickers_json is invalid")
            raw_fee_changes = _json_value(fee_changes_json)
            if not isinstance(raw_fee_changes, list) or any(
                not isinstance(change, dict) for change in raw_fee_changes
            ):
                raise StorageError("stored event fee_changes_json is invalid")
            try:
                fee_changes = tuple(
                    EventFeeChange.model_validate(change) for change in raw_fee_changes
                )
            except ValidationError as exc:
                raise StorageError("stored event fee_changes_json is invalid") from exc
            event = normalize_event(_json_mapping(raw_json))
            events.append(
                event.model_copy(
                    update={
                        "market_tickers": tuple(raw_tickers),
                        "fee_type_override": (
                            None if fee_type_override is None else str(fee_type_override)
                        ),
                        "fee_multiplier_override": cast(Decimal | None, fee_multiplier_override),
                        "fee_changes": fee_changes,
                    }
                )
            )
        return tuple(events)

    def load_series(self) -> tuple[Series, ...]:
        """Rebuild strict series models from preserved raw payloads."""

        self.migrate()
        return tuple(
            normalize_series(_json_mapping(row[0]))
            for row in self.connection.execute(
                "SELECT raw_json FROM series ORDER BY ticker"
            ).fetchall()
        )

    def upsert_semantic_embeddings(self, embeddings: Sequence[SemanticEmbedding]) -> None:
        """Cache immutable semantic vectors, rejecting conflicting deterministic IDs."""

        self.migrate()
        if not embeddings:
            return
        if len({embedding.embedding_id for embedding in embeddings}) != len(embeddings):
            raise ValueError("semantic embedding batch contains duplicate identities")
        rows = tuple(_semantic_embedding_row(embedding) for embedding in embeddings)

        def write() -> None:
            for row in rows:
                existing = self.connection.execute(
                    """
                    SELECT embedding_id, payload_hash
                    FROM semantic_embeddings
                    WHERE embedding_id = ? OR (
                        market_ticker = ? AND canonical_text_hash = ?
                        AND provider = ? AND model = ?
                    )
                    """,
                    (row[0], row[1], row[3], row[6], row[7]),
                ).fetchall()
                if existing:
                    if len(existing) != 1 or (str(existing[0][0]), str(existing[0][1])) != (
                        str(row[0]),
                        str(row[10]),
                    ):
                        raise ValueError(
                            "semantic embedding identity conflicts with a different payload"
                        )
                    self.connection.execute(
                        """
                        UPDATE semantic_embeddings
                        SET updated_at = greatest(updated_at, ?)
                        WHERE embedding_id = ?
                        """,
                        (row[12], row[0]),
                    )
                    continue
                placeholders = ", ".join("?" for _ in _SEMANTIC_EMBEDDING_COLUMNS)
                self.connection.execute(
                    f"INSERT INTO semantic_embeddings "
                    f"({', '.join(_SEMANTIC_EMBEDDING_COLUMNS)}) VALUES ({placeholders})",
                    row,
                )

        self._write_transaction("upsert semantic embeddings", write)

    def get_semantic_embedding(
        self,
        *,
        market_ticker: str,
        canonical_text_hash: str,
        provider: str,
        model: str,
    ) -> SemanticEmbedding | None:
        """Return one exact cache hit or ``None`` without falling back across models."""

        self.migrate()
        row = self.connection.execute(
            f"SELECT {', '.join(_SEMANTIC_EMBEDDING_COLUMNS)} "
            "FROM semantic_embeddings "
            "WHERE market_ticker = ? AND canonical_text_hash = ? "
            "AND provider = ? AND model = ?",
            (market_ticker, canonical_text_hash, provider, model),
        ).fetchone()
        return None if row is None else _semantic_embedding_from_row(row)

    def list_semantic_embeddings(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> tuple[SemanticEmbedding, ...]:
        """Load cached embeddings in stable market/identity order."""

        self.migrate()
        if (provider is None) != (model is None):
            raise ValueError("semantic embedding provider and model filters must be paired")
        query = f"SELECT {', '.join(_SEMANTIC_EMBEDDING_COLUMNS)} FROM semantic_embeddings"
        parameters: tuple[object, ...] = ()
        if provider is not None and model is not None:
            query += " WHERE provider = ? AND model = ?"
            parameters = (provider, model)
        query += " ORDER BY market_ticker, embedding_id"
        return tuple(
            _semantic_embedding_from_row(row)
            for row in self.connection.execute(query, parameters).fetchall()
        )

    def upsert_semantic_suggestions(self, suggestions: Sequence[SemanticSuggestion]) -> None:
        """Persist the current proposal per pair without preserving obsolete trust."""

        self.migrate()
        if not suggestions:
            return
        if len({suggestion.suggestion_id for suggestion in suggestions}) != len(suggestions):
            raise ValueError("semantic suggestion batch contains duplicate market pairs")
        for suggestion in suggestions:
            if (
                suggestion.review_state is not SemanticReviewState.PENDING
                or suggestion.reviewed_at is not None
                or suggestion.approved_relation_id is not None
                or suggestion.stale_reason is not None
            ):
                raise ValueError("new semantic suggestions must be unreviewed and pending")
        rows = tuple(_semantic_suggestion_row(suggestion) for suggestion in suggestions)

        def write() -> None:
            for suggestion, row in zip(suggestions, rows, strict=True):
                existing_row = self.connection.execute(
                    f"SELECT {', '.join(_SEMANTIC_SUGGESTION_COLUMNS)} "
                    "FROM semantic_suggestions WHERE suggestion_id = ?",
                    (suggestion.suggestion_id,),
                ).fetchone()
                if existing_row is None:
                    placeholders = ", ".join("?" for _ in _SEMANTIC_SUGGESTION_COLUMNS)
                    self.connection.execute(
                        f"INSERT INTO semantic_suggestions "
                        f"({', '.join(_SEMANTIC_SUGGESTION_COLUMNS)}) "
                        f"VALUES ({placeholders})",
                        row,
                    )
                    continue

                existing = _semantic_suggestion_from_row(existing_row)
                same_payload = str(existing_row[32]) == str(row[32])
                if same_payload and existing.review_state is not SemanticReviewState.STALE:
                    continue
                self._deverify_semantic_suggestion(existing, updated_at=suggestion.updated_at)
                assignments = ", ".join(
                    f"{column} = ?" for column in _SEMANTIC_SUGGESTION_COLUMNS[1:]
                )
                self.connection.execute(
                    f"UPDATE semantic_suggestions SET {assignments} WHERE suggestion_id = ?",
                    (*row[1:], row[0]),
                )

        self._write_transaction("upsert semantic suggestions", write)

    def get_semantic_suggestion(self, suggestion_id: str) -> SemanticSuggestion | None:
        """Load one semantic proposal with its current review state."""

        self.migrate()
        row = self.connection.execute(
            f"SELECT {', '.join(_SEMANTIC_SUGGESTION_COLUMNS)} "
            "FROM semantic_suggestions WHERE suggestion_id = ?",
            (suggestion_id,),
        ).fetchone()
        return None if row is None else _semantic_suggestion_from_row(row)

    def list_semantic_suggestions(
        self,
        *,
        states: set[SemanticReviewState] | None = None,
    ) -> tuple[SemanticSuggestion, ...]:
        """Return a deterministic semantic review queue, optionally filtered by state."""

        self.migrate()
        query = f"SELECT {', '.join(_SEMANTIC_SUGGESTION_COLUMNS)} FROM semantic_suggestions"
        parameters: tuple[object, ...] = ()
        if states is not None:
            if not states:
                return ()
            state_values = tuple(sorted(state.value for state in states))
            query += f" WHERE review_state IN ({', '.join('?' for _ in state_values)})"
            parameters = state_values
        query += " ORDER BY created_at, suggestion_id"
        return tuple(
            _semantic_suggestion_from_row(row)
            for row in self.connection.execute(query, parameters).fetchall()
        )

    def reconcile_semantic_staleness(
        self,
        documents: Mapping[str, SemanticMarketDocument],
        *,
        updated_at: datetime | None = None,
        complete_catalog: bool = True,
    ) -> tuple[str, ...]:
        """Atomically stale proposals whose market evidence changed or disappeared."""

        self.migrate()
        timestamp = _require_aware(
            self._clock() if updated_at is None else updated_at,
            label="semantic staleness time",
        )
        expected_tickers = set(documents)
        if complete_catalog:
            expected_tickers.update(
                str(ticker)
                for row in self.connection.execute(
                    """
                    SELECT market_a_ticker, market_b_ticker
                    FROM semantic_suggestions
                    WHERE review_state IN ('pending', 'approved', 'uncertain')
                    """
                ).fetchall()
                for ticker in row
            )
        stale_ids: list[str] = []

        def write() -> None:
            stale_ids.extend(
                self._reconcile_semantic_staleness_in_transaction(
                    documents,
                    updated_at=timestamp,
                    expected_tickers=expected_tickers,
                )
            )

        self._write_transaction("reconcile semantic suggestion staleness", write)
        return tuple(stale_ids)

    def review_semantic_suggestion(
        self,
        suggestion_id: str,
        *,
        action: SemanticReviewState,
        documents: Mapping[str, SemanticMarketDocument] | None = None,
        reviewed_at: datetime | None = None,
        max_component_markets: int = 12,
    ) -> SemanticSuggestion:
        """Review against authoritative persisted evidence and project trust atomically."""

        if action not in {
            SemanticReviewState.APPROVED,
            SemanticReviewState.REJECTED,
            SemanticReviewState.UNCERTAIN,
        }:
            raise ValueError("semantic review action must be approve, reject, or uncertain")
        if (
            isinstance(max_component_markets, bool)
            or not isinstance(max_component_markets, int)
            or max_component_markets < 1
        ):
            raise ValueError("semantic review component limit must be a positive integer")
        self.migrate()
        timestamp = _require_aware(
            self._clock() if reviewed_at is None else reviewed_at,
            label="semantic review time",
        )
        result: list[SemanticSuggestion] = []

        def write() -> None:
            self._audit_semantic_relations_in_transaction(updated_at=timestamp)
            row = self.connection.execute(
                f"SELECT {', '.join(_SEMANTIC_SUGGESTION_COLUMNS)} "
                "FROM semantic_suggestions WHERE suggestion_id = ?",
                (suggestion_id,),
            ).fetchone()
            if row is None:
                raise ValueError("semantic suggestion does not exist")
            suggestion = _semantic_suggestion_from_row(row)
            try:
                current_documents = self._load_persisted_semantic_documents(
                    (suggestion.market_a_ticker, suggestion.market_b_ticker)
                )
            except (StorageError, ValidationError, ValueError) as exc:
                self._mark_semantic_suggestion_stale(
                    suggestion,
                    reason=f"current semantic metadata is unavailable: {exc}",
                    updated_at=timestamp,
                )
            else:
                evidence_changed = any(
                    not _semantic_document_matches_suggestion(
                        suggestion,
                        current_documents[ticker],
                    )
                    for ticker in (suggestion.market_a_ticker, suggestion.market_b_ticker)
                )
                if evidence_changed:
                    self._mark_semantic_suggestion_stale(
                        suggestion,
                        reason="persisted semantic evidence changed before review",
                        updated_at=timestamp,
                    )
                elif documents is not None and any(
                    documents.get(ticker) != current_documents[ticker]
                    for ticker in (suggestion.market_a_ticker, suggestion.market_b_ticker)
                ):
                    raise ValueError(
                        "caller semantic documents do not match authoritative persistence"
                    )
                elif (
                    suggestion.review_state is SemanticReviewState.STALE
                    and action is SemanticReviewState.APPROVED
                ) or suggestion.review_state is action:
                    pass
                elif action is SemanticReviewState.APPROVED:
                    approval_candidate = SemanticSuggestion.model_validate(
                        suggestion.model_dump()
                        | {
                            "review_state": SemanticReviewState.APPROVED,
                            "reviewed_at": timestamp,
                            "approved_relation_id": semantic_verified_relation_id(suggestion),
                            "stale_reason": None,
                            "updated_at": timestamp,
                        }
                    )
                    relation = relation_from_semantic_suggestion(
                        approval_candidate,
                        created_at=timestamp,
                    )
                    relation_set = {
                        item.relation_id: item for item in self._load_relations_in_transaction()
                    }
                    relation_set[relation.relation_id] = relation
                    validation = validate_relations(
                        tuple(relation_set.values()),
                        known_market_tickers={
                            str(item[0])
                            for item in self.connection.execute(
                                "SELECT ticker FROM markets"
                            ).fetchall()
                        },
                        max_component_markets=max_component_markets,
                    )
                    if not validation.valid:
                        details = "; ".join(
                            f"{issue.code}:{issue.item_id}" for issue in validation.issues
                        )
                        raise ValueError(
                            "semantic approval would invalidate the trusted relation set: "
                            + details
                        )
                    self._persist_approved_semantic_relation(
                        approval_candidate,
                        relation=relation,
                        updated_at=timestamp,
                    )
                    self.connection.execute(
                        """
                        UPDATE semantic_suggestions
                        SET review_state = 'approved', reviewed_at = ?,
                            approved_relation_id = ?, stale_reason = NULL, updated_at = ?
                        WHERE suggestion_id = ?
                        """,
                        (
                            timestamp,
                            relation.relation_id,
                            timestamp,
                            suggestion.suggestion_id,
                        ),
                    )
                else:
                    self._deverify_semantic_suggestion(suggestion, updated_at=timestamp)
                    self.connection.execute(
                        """
                        UPDATE semantic_suggestions
                        SET review_state = ?, reviewed_at = ?, approved_relation_id = NULL,
                            stale_reason = NULL, updated_at = ?
                        WHERE suggestion_id = ?
                        """,
                        (action.value, timestamp, timestamp, suggestion.suggestion_id),
                    )
            stored = self.connection.execute(
                f"SELECT {', '.join(_SEMANTIC_SUGGESTION_COLUMNS)} "
                "FROM semantic_suggestions WHERE suggestion_id = ?",
                (suggestion_id,),
            ).fetchone()
            if stored is None:
                raise ValueError("semantic suggestion disappeared during review")
            result.append(_semantic_suggestion_from_row(stored))

        self._write_transaction(f"review semantic suggestion {suggestion_id}", write)
        if not result:
            raise StorageError("semantic review completed without a result")
        return result[0]

    def _load_persisted_semantic_documents(
        self,
        tickers: tuple[str, ...],
    ) -> dict[str, SemanticMarketDocument]:
        documents: dict[str, SemanticMarketDocument] = {}
        for ticker in tickers:
            market_row = self.connection.execute(
                """
                SELECT ticker, event_ticker, series_ticker, title, subtitle,
                       yes_sub_title, no_sub_title, status, open_time, close_time,
                       expiration_time, latest_expiration_time,
                       expected_expiration_time, settlement_ts, occurrence_datetime,
                       rules_primary, rules_secondary, early_close_condition, raw_json
                FROM markets
                WHERE ticker = ?
                """,
                (ticker,),
            ).fetchone()
            if market_row is None:
                raise ValueError(f"semantic market metadata is missing for {ticker}")
            market = Market(
                ticker=str(market_row[0]),
                event_ticker=str(market_row[1]),
                series_ticker=None if market_row[2] is None else str(market_row[2]),
                title=str(market_row[3]),
                subtitle=None if market_row[4] is None else str(market_row[4]),
                yes_sub_title=None if market_row[5] is None else str(market_row[5]),
                no_sub_title=None if market_row[6] is None else str(market_row[6]),
                status=str(market_row[7]),
                open_time=cast(datetime | None, market_row[8]),
                close_time=cast(datetime | None, market_row[9]),
                expiration_time=cast(datetime | None, market_row[10]),
                latest_expiration_time=cast(datetime | None, market_row[11]),
                expected_expiration_time=cast(datetime | None, market_row[12]),
                settlement_ts=cast(datetime | None, market_row[13]),
                occurrence_datetime=cast(datetime | None, market_row[14]),
                rules_primary=None if market_row[15] is None else str(market_row[15]),
                rules_secondary=None if market_row[16] is None else str(market_row[16]),
                early_close_condition=(None if market_row[17] is None else str(market_row[17])),
                raw=_json_mapping(market_row[18]),
            )

            event_row = self.connection.execute(
                """
                SELECT ticker, series_ticker, title, subtitle, category, raw_json
                FROM events
                WHERE ticker = ?
                """,
                (market.event_ticker,),
            ).fetchone()
            if event_row is None:
                raise ValueError(f"semantic event metadata is missing for {market.event_ticker}")
            event = Event(
                ticker=str(event_row[0]),
                series_ticker=None if event_row[1] is None else str(event_row[1]),
                title=str(event_row[2]),
                subtitle=None if event_row[3] is None else str(event_row[3]),
                category=None if event_row[4] is None else str(event_row[4]),
                raw=_json_mapping(event_row[5]),
            )

            series_ticker = market.series_ticker or event.series_ticker
            parent_series: Series | None = None
            if series_ticker is not None:
                series_row = self.connection.execute(
                    """
                    SELECT ticker, title, category, tags_json, settlement_sources_json,
                           contract_url, contract_terms_url, raw_json
                    FROM series
                    WHERE ticker = ?
                    """,
                    (series_ticker,),
                ).fetchone()
                if series_row is None:
                    raise ValueError(f"semantic series metadata is missing for {series_ticker}")
                raw_tags = _json_value(series_row[3])
                if not isinstance(raw_tags, list) or any(
                    not isinstance(tag, str) for tag in raw_tags
                ):
                    raise ValueError("semantic series tags are invalid")
                raw_sources = _json_value(series_row[4])
                if not isinstance(raw_sources, list) or any(
                    not isinstance(source, dict) for source in raw_sources
                ):
                    raise ValueError("semantic settlement sources are invalid")
                parent_series = Series(
                    ticker=str(series_row[0]),
                    title=str(series_row[1]),
                    category=None if series_row[2] is None else str(series_row[2]),
                    tags=tuple(raw_tags),
                    settlement_sources=tuple(
                        SettlementSource.model_validate(source) for source in raw_sources
                    ),
                    contract_url=None if series_row[5] is None else str(series_row[5]),
                    contract_terms_url=(None if series_row[6] is None else str(series_row[6])),
                    raw=_json_mapping(series_row[7]),
                )
            documents[ticker] = build_semantic_document(
                market,
                event=event,
                series=parent_series,
            )
        return documents

    def _audit_semantic_relations_in_transaction(
        self,
        *,
        updated_at: datetime,
    ) -> tuple[str, ...]:
        """Revoke every semantic relation lacking one exact current approved provenance."""

        revoked: list[str] = []
        rows = self.connection.execute(
            """
            SELECT relation_id, relation_type, market_tickers_json, antecedent,
                   consequent, source, confidence, verified, rationale, created_at,
                   semantic_suggestion_id, semantic_rules_hash, semantic_timing_hash
            FROM relations
            WHERE source = 'semantic_verified' AND verified = true
            ORDER BY relation_id
            """
        ).fetchall()
        for row in rows:
            relation_id = str(row[0])
            provenance_id = None if row[10] is None else str(row[10])
            suggestion_rows = self.connection.execute(
                f"SELECT {', '.join(_SEMANTIC_SUGGESTION_COLUMNS)} "
                "FROM semantic_suggestions "
                "WHERE suggestion_id = ? OR approved_relation_id = ? "
                "ORDER BY suggestion_id",
                (provenance_id, relation_id),
            ).fetchall()
            candidates: list[SemanticSuggestion] = []
            corrupt_suggestion = False
            for suggestion_row in suggestion_rows:
                try:
                    candidates.append(_semantic_suggestion_from_row(suggestion_row))
                except StorageError:
                    corrupt_suggestion = True
            approved = tuple(
                suggestion
                for suggestion in candidates
                if suggestion.review_state is SemanticReviewState.APPROVED
                and suggestion.approved_relation_id == relation_id
            )

            reason: str | None = None
            relation: Relation | None = None
            raw_tickers = _json_value(row[2])
            if not isinstance(raw_tickers, list) or any(
                not isinstance(ticker, str) for ticker in raw_tickers
            ):
                reason = "stored semantic relation market membership is invalid"
            else:
                try:
                    relation = Relation(
                        relation_id=relation_id,
                        relation_type=RelationType(str(row[1])),
                        market_tickers=tuple(raw_tickers),
                        antecedent=None if row[3] is None else str(row[3]),
                        consequent=None if row[4] is None else str(row[4]),
                        source=str(row[5]),
                        confidence=None if row[6] is None else float(cast(float, row[6])),
                        verified=bool(row[7]),
                        rationale=str(row[8]),
                        created_at=cast(datetime, row[9]),
                    )
                except (ValidationError, ValueError) as exc:
                    reason = f"stored semantic relation shape is invalid: {exc}"

            if reason is None and corrupt_suggestion:
                reason = "semantic approval evidence is corrupt"
            if reason is None and len(approved) != 1:
                reason = "semantic relation requires exactly one approved suggestion"
            backing = approved[0] if len(approved) == 1 else None
            if reason is None and backing is not None and backing.suggestion_id != provenance_id:
                reason = "semantic relation suggestion provenance is inconsistent"

            if reason is None and backing is not None and relation is not None:
                expected = relation_from_semantic_suggestion(
                    backing,
                    created_at=relation.created_at,
                )
                identity = (
                    relation.relation_id,
                    relation.relation_type,
                    relation.market_tickers,
                    relation.antecedent,
                    relation.consequent,
                    relation.source,
                    relation.confidence,
                    relation.rationale,
                )
                expected_identity = (
                    expected.relation_id,
                    expected.relation_type,
                    expected.market_tickers,
                    expected.antecedent,
                    expected.consequent,
                    expected.source,
                    expected.confidence,
                    expected.rationale,
                )
                if identity != expected_identity or relation.created_at != backing.reviewed_at:
                    reason = "semantic relation does not match its approved proposal"

            if reason is None and backing is not None:
                expected_rules_hash = _semantic_evidence_pair_hash(
                    market_a_ticker=backing.market_a_ticker,
                    market_a_hash=backing.market_a_rules_hash,
                    market_b_ticker=backing.market_b_ticker,
                    market_b_hash=backing.market_b_rules_hash,
                )
                expected_timing_hash = _semantic_evidence_pair_hash(
                    market_a_ticker=backing.market_a_ticker,
                    market_a_hash=backing.market_a_timing_hash,
                    market_b_ticker=backing.market_b_ticker,
                    market_b_hash=backing.market_b_timing_hash,
                )
                if (row[11], row[12]) != (expected_rules_hash, expected_timing_hash):
                    reason = "semantic relation evidence hashes are inconsistent"

            if reason is None and backing is not None:
                try:
                    documents = self._load_persisted_semantic_documents(
                        (backing.market_a_ticker, backing.market_b_ticker)
                    )
                except (StorageError, ValidationError, ValueError) as exc:
                    reason = f"current semantic metadata is unavailable: {exc}"
                else:
                    if any(
                        not _semantic_document_matches_suggestion(backing, documents[ticker])
                        for ticker in (backing.market_a_ticker, backing.market_b_ticker)
                    ):
                        reason = "current semantic evidence no longer matches approval"

            if reason is None:
                continue
            self.connection.execute(
                """
                UPDATE relations
                SET verified = false, updated_at = ?
                WHERE relation_id = ?
                """,
                (updated_at, relation_id),
            )
            for suggestion in approved:
                self.connection.execute(
                    """
                    UPDATE semantic_suggestions
                    SET review_state = 'stale', stale_reason = ?, updated_at = ?
                    WHERE suggestion_id = ?
                    """,
                    (reason[:2000], updated_at, suggestion.suggestion_id),
                )
            revoked.append(relation_id)
        return tuple(revoked)

    def _reconcile_semantic_staleness_in_transaction(
        self,
        documents: Mapping[str, SemanticMarketDocument],
        *,
        updated_at: datetime,
        expected_tickers: set[str],
    ) -> tuple[str, ...]:
        if not expected_tickers:
            return ()
        stale_ids: list[str] = []
        rows = self.connection.execute(
            f"SELECT {', '.join(_SEMANTIC_SUGGESTION_COLUMNS)} "
            "FROM semantic_suggestions "
            "WHERE review_state IN ('pending', 'approved', 'uncertain') "
            "ORDER BY suggestion_id"
        ).fetchall()
        for row in rows:
            suggestion = _semantic_suggestion_from_row(row)
            changed = tuple(
                ticker
                for ticker in (
                    suggestion.market_a_ticker,
                    suggestion.market_b_ticker,
                )
                if ticker in expected_tickers
                and (
                    (document := documents.get(ticker)) is None
                    or not _semantic_document_matches_suggestion(suggestion, document)
                )
            )
            if not changed:
                continue
            reason = "semantic evidence changed for " + ", ".join(changed)
            self._mark_semantic_suggestion_stale(
                suggestion,
                reason=reason,
                updated_at=updated_at,
            )
            stale_ids.append(suggestion.suggestion_id)
        return tuple(stale_ids)

    def _deverify_semantic_suggestion(
        self,
        suggestion: SemanticSuggestion,
        *,
        updated_at: datetime,
    ) -> None:
        if suggestion.approved_relation_id is None:
            return
        self.connection.execute(
            """
            UPDATE relations
            SET verified = false, updated_at = ?
            WHERE relation_id = ? AND source = 'semantic_verified'
              AND semantic_suggestion_id = ?
            """,
            (updated_at, suggestion.approved_relation_id, suggestion.suggestion_id),
        )

    def _mark_semantic_suggestion_stale(
        self,
        suggestion: SemanticSuggestion,
        *,
        reason: str,
        updated_at: datetime,
    ) -> None:
        self._deverify_semantic_suggestion(suggestion, updated_at=updated_at)
        self.connection.execute(
            """
            UPDATE semantic_suggestions
            SET review_state = 'stale', stale_reason = ?, updated_at = ?
            WHERE suggestion_id = ?
            """,
            (reason, updated_at, suggestion.suggestion_id),
        )

    def _persist_approved_semantic_relation(
        self,
        suggestion: SemanticSuggestion,
        *,
        relation: Relation,
        updated_at: datetime,
    ) -> None:
        if relation.source != "semantic_verified" or not relation.verified:
            raise ValueError("semantic approval helper returned an untrusted relation")
        existing = self.connection.execute(
            """
            SELECT relation_type, market_tickers_json, antecedent, consequent, source
            FROM relations
            WHERE relation_id = ?
            """,
            (relation.relation_id,),
        ).fetchone()
        expected_identity = (
            relation.relation_type.value,
            list(relation.market_tickers),
            relation.antecedent,
            relation.consequent,
            "semantic_verified",
        )
        if existing is not None:
            raw_tickers = _json_value(existing[1])
            persisted_identity = (
                str(existing[0]),
                raw_tickers,
                None if existing[2] is None else str(existing[2]),
                None if existing[3] is None else str(existing[3]),
                str(existing[4]),
            )
            if persisted_identity != expected_identity:
                raise ValueError("semantic relation ID conflicts with persisted semantics")

        rules_hash = _semantic_evidence_pair_hash(
            market_a_ticker=suggestion.market_a_ticker,
            market_a_hash=suggestion.market_a_rules_hash,
            market_b_ticker=suggestion.market_b_ticker,
            market_b_hash=suggestion.market_b_rules_hash,
        )
        timing_hash = _semantic_evidence_pair_hash(
            market_a_ticker=suggestion.market_a_ticker,
            market_a_hash=suggestion.market_a_timing_hash,
            market_b_ticker=suggestion.market_b_ticker,
            market_b_hash=suggestion.market_b_timing_hash,
        )
        self.connection.execute(
            """
            INSERT INTO relations (
                relation_id, relation_type, market_tickers_json, antecedent,
                consequent, source, confidence, verified, rationale, created_at,
                updated_at, semantic_suggestion_id, semantic_rules_hash,
                semantic_timing_hash
            ) VALUES (?, ?, ?, ?, ?, 'semantic_verified', ?, true, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (relation_id) DO UPDATE SET
                relation_type = excluded.relation_type,
                market_tickers_json = excluded.market_tickers_json,
                antecedent = excluded.antecedent,
                consequent = excluded.consequent,
                source = excluded.source,
                confidence = excluded.confidence,
                verified = true,
                rationale = excluded.rationale,
                updated_at = excluded.updated_at,
                semantic_suggestion_id = excluded.semantic_suggestion_id,
                semantic_rules_hash = excluded.semantic_rules_hash,
                semantic_timing_hash = excluded.semantic_timing_hash
            """,
            (
                relation.relation_id,
                relation.relation_type.value,
                _json(relation.market_tickers),
                relation.antecedent,
                relation.consequent,
                relation.confidence,
                relation.rationale,
                _require_aware(relation.created_at, label="semantic relation creation time"),
                updated_at,
                suggestion.suggestion_id,
                rules_hash,
                timing_hash,
            ),
        )

    def upsert_relations(self, relations: Sequence[Relation]) -> None:
        """Idempotently persist relations while preserving their original creation time."""

        self.migrate()
        if not relations:
            return
        if any(relation.source == "semantic_verified" for relation in relations):
            raise ValueError(
                "semantic_verified relations may only be created by semantic review approval"
            )
        updated_at = self._clock()
        rows = tuple(
            (
                relation.relation_id,
                relation.relation_type.value,
                _json(relation.market_tickers),
                relation.antecedent,
                relation.consequent,
                relation.source,
                relation.confidence,
                relation.verified,
                relation.rationale,
                relation.created_at,
                updated_at,
            )
            for relation in relations
        )
        try:
            self.connection.execute("BEGIN TRANSACTION")
            self.connection.executemany(
                """
                INSERT INTO relations (
                    relation_id, relation_type, market_tickers_json, antecedent,
                    consequent, source, confidence, verified, rationale, created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (relation_id) DO UPDATE SET
                    relation_type = excluded.relation_type,
                    market_tickers_json = excluded.market_tickers_json,
                    antecedent = excluded.antecedent,
                    consequent = excluded.consequent,
                    source = excluded.source,
                    confidence = excluded.confidence,
                    verified = excluded.verified,
                    rationale = excluded.rationale,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
            self.connection.execute("COMMIT")
        except Exception as exc:
            with suppress(Exception):
                self.connection.execute("ROLLBACK")
            raise StorageError(f"relation upsert failed: {exc}") from exc

    def list_relations(self) -> tuple[Relation, ...]:
        """Load relations after atomically revoking unsupported semantic trust."""

        self.migrate()
        timestamp = _require_aware(self._clock(), label="semantic trust audit time")
        result: list[Relation] = []

        def write() -> None:
            self._audit_semantic_relations_in_transaction(updated_at=timestamp)
            result.extend(self._load_relations_in_transaction())

        self._write_transaction("audit and load relations", write)
        return tuple(result)

    def _load_relations_in_transaction(self) -> tuple[Relation, ...]:
        """Load relations without migration writes while another transaction is active."""

        rows = self.connection.execute(
            """
            SELECT relation_id, relation_type, market_tickers_json, antecedent,
                   consequent, source, confidence, verified, rationale, created_at
            FROM relations
            ORDER BY relation_id
            """
        ).fetchall()
        relations: list[Relation] = []
        for row in rows:
            raw_tickers = _json_value(row[2])
            if not isinstance(raw_tickers, list) or any(
                not isinstance(ticker, str) for ticker in raw_tickers
            ):
                raise StorageError("stored relation market_tickers_json is invalid")
            relations.append(
                Relation(
                    relation_id=str(row[0]),
                    relation_type=RelationType(str(row[1])),
                    market_tickers=tuple(raw_tickers),
                    antecedent=None if row[3] is None else str(row[3]),
                    consequent=None if row[4] is None else str(row[4]),
                    source=str(row[5]),
                    confidence=None if row[6] is None else float(row[6]),
                    verified=bool(row[7]),
                    rationale=str(row[8]),
                    created_at=cast(datetime, row[9]),
                )
            )
        return tuple(relations)

    def start_run(self, record: RunManifestRecord) -> None:
        """Persist a run before scanning, tolerating an identical retry."""

        self.migrate()
        metadata_json = _json(record.metadata)
        input_payload_json = _json(record.input_payload)

        def write() -> None:
            existing = self.connection.execute(
                """
                SELECT run_type, started_at, schema_version, manifest_version,
                       recording_format_version, event_schema_version, recording_id,
                       source_run_id, first_event_index, last_event_index, event_count,
                       event_stream_hash, input_payload_json, config_hash, metadata_hash,
                       relations_hash, fee_policy_hash, metadata_json, status
                FROM run_manifests
                WHERE run_id = ?
                """,
                (record.run_id,),
            ).fetchone()
            if existing is not None:
                persisted = (
                    str(existing[0]),
                    _require_aware(cast(datetime, existing[1]), label="stored run start"),
                    int(existing[2]),
                    int(existing[3]),
                    int(existing[4]),
                    int(existing[5]),
                    None if existing[6] is None else str(existing[6]),
                    None if existing[7] is None else str(existing[7]),
                    None if existing[8] is None else int(existing[8]),
                    None if existing[9] is None else int(existing[9]),
                    int(existing[10]),
                    None if existing[11] is None else str(existing[11]),
                    _json(_json_mapping(existing[12])),
                    None if existing[13] is None else str(existing[13]),
                    None if existing[14] is None else str(existing[14]),
                    None if existing[15] is None else str(existing[15]),
                    None if existing[16] is None else str(existing[16]),
                    _json(_json_mapping(existing[17])),
                )
                expected = (
                    record.run_type,
                    _require_aware(record.started_at, label="run start"),
                    record.schema_version,
                    record.manifest_version,
                    record.recording_format_version,
                    record.event_schema_version,
                    record.recording_id,
                    record.source_run_id,
                    record.first_event_index,
                    record.last_event_index,
                    record.event_count,
                    record.event_stream_hash,
                    input_payload_json,
                    record.config_hash,
                    record.metadata_hash,
                    record.relations_hash,
                    record.fee_policy_hash,
                    metadata_json,
                )
                if persisted != expected:
                    raise ValueError("run ID already exists with a different manifest")
                if str(existing[18]) != "running":
                    raise ValueError("a finalized run ID cannot be started again")
                return
            self.connection.execute(
                """
                INSERT INTO run_manifests (
                    run_id, run_type, started_at, status, schema_version,
                    manifest_version, recording_format_version, event_schema_version,
                    recording_id, source_run_id, first_event_index, last_event_index,
                    event_count, event_stream_hash, input_payload_json, config_hash,
                    metadata_hash, relations_hash, fee_policy_hash, metadata_json
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.run_type,
                    _require_aware(record.started_at, label="run start"),
                    record.schema_version,
                    record.manifest_version,
                    record.recording_format_version,
                    record.event_schema_version,
                    record.recording_id,
                    record.source_run_id,
                    record.first_event_index,
                    record.last_event_index,
                    record.event_count,
                    record.event_stream_hash,
                    input_payload_json,
                    record.config_hash,
                    record.metadata_hash,
                    record.relations_hash,
                    record.fee_policy_hash,
                    metadata_json,
                ),
            )

        self._write_transaction(f"start run {record.run_id}", write)

    def finalize_run(
        self,
        run_id: str,
        *,
        status: str,
        ended_at: datetime | None = None,
        error: str | None = None,
        stream: RunStreamEvidence | None = None,
    ) -> None:
        """Atomically mark one existing run terminal without resetting its manifest."""

        _nonblank(run_id, label="run ID")
        _nonblank(status, label="run status")
        if status == "running":
            raise ValueError("final run status cannot be running")
        if error is not None:
            _nonblank(error, label="run error")
        terminal_at = _require_aware(
            self._clock() if ended_at is None else ended_at,
            label="run end",
        )
        self.migrate()

        def write() -> None:
            existing = self.connection.execute(
                """
                SELECT started_at, ended_at, status, error, manifest_version,
                       first_event_index, last_event_index, event_count, event_stream_hash
                FROM run_manifests
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if existing is None:
                raise ValueError("cannot finalize an unknown run")
            started_at = _require_aware(cast(datetime, existing[0]), label="stored run start")
            if terminal_at < started_at:
                raise ValueError("run end cannot precede its start")
            persisted_stream = (
                None if existing[5] is None else int(existing[5]),
                None if existing[6] is None else int(existing[6]),
                int(existing[7]),
                None if existing[8] is None else str(existing[8]),
            )
            supplied_stream = (
                persisted_stream
                if stream is None
                else (
                    stream.first_event_index,
                    stream.last_event_index,
                    stream.event_count,
                    stream.event_stream_hash,
                )
            )
            if persisted_stream[2] > 0 and supplied_stream != persisted_stream:
                raise ValueError("final stream evidence conflicts with the run manifest")
            if status == "succeeded" and int(existing[4]) >= 2 and supplied_stream[2] == 0:
                raise ValueError("a successful manifest version 2 run requires stream evidence")
            if existing[1] is not None:
                persisted = (
                    _require_aware(cast(datetime, existing[1]), label="stored run end"),
                    str(existing[2]),
                    None if existing[3] is None else str(existing[3]),
                )
                if persisted != (terminal_at, status, error):
                    raise ValueError("run is already finalized with a different result")
                if stream is not None and persisted_stream != supplied_stream:
                    raise ValueError("run is already finalized with different stream evidence")
                return
            self.connection.execute(
                """
                UPDATE run_manifests
                SET ended_at = ?, status = ?, error = ?, first_event_index = ?,
                    last_event_index = ?, event_count = ?, event_stream_hash = ?
                WHERE run_id = ?
                """,
                (
                    terminal_at,
                    status,
                    error,
                    supplied_stream[0],
                    supplied_stream[1],
                    supplied_stream[2],
                    supplied_stream[3],
                    run_id,
                ),
            )

        self._write_transaction(f"finalize run {run_id}", write)

    def get_run_manifest(self, run_id: str) -> StoredRunManifest | None:
        """Load one run manifest without exposing untyped database tuples."""

        _nonblank(run_id, label="run ID")
        self.migrate()
        row = self.connection.execute(
            """
            SELECT run_id, run_type, started_at, ended_at, status, schema_version,
                   manifest_version, recording_format_version, event_schema_version,
                   recording_id, source_run_id, first_event_index, last_event_index,
                   event_count, event_stream_hash, input_payload_json, config_hash,
                   metadata_hash, relations_hash, fee_policy_hash, metadata_json, error
            FROM run_manifests
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return StoredRunManifest(
            run_id=str(row[0]),
            run_type=str(row[1]),
            started_at=_require_aware(cast(datetime, row[2]), label="stored run start"),
            ended_at=(
                None
                if row[3] is None
                else _require_aware(cast(datetime, row[3]), label="stored run end")
            ),
            status=str(row[4]),
            schema_version=int(row[5]),
            manifest_version=int(row[6]),
            recording_format_version=int(row[7]),
            event_schema_version=int(row[8]),
            recording_id=None if row[9] is None else str(row[9]),
            source_run_id=None if row[10] is None else str(row[10]),
            first_event_index=None if row[11] is None else int(row[11]),
            last_event_index=None if row[12] is None else int(row[12]),
            event_count=int(row[13]),
            event_stream_hash=None if row[14] is None else str(row[14]),
            input_payload=_json_mapping(row[15]),
            config_hash=None if row[16] is None else str(row[16]),
            metadata_hash=None if row[17] is None else str(row[17]),
            relations_hash=None if row[18] is None else str(row[18]),
            fee_policy_hash=None if row[19] is None else str(row[19]),
            metadata=_json_mapping(row[20]),
            error=None if row[21] is None else str(row[21]),
        )

    def record_market_observation(self, record: MarketObservationRecord) -> None:
        """Insert or extend one market observation window in a retried transaction."""

        self.migrate()
        row = (
            record.observation_id,
            record.run_id,
            record.market_ticker,
            _require_aware(record.opened_at, label="observation open time"),
            _require_aware(record.updated_at, label="observation update time"),
            (
                None
                if record.closed_at is None
                else _require_aware(record.closed_at, label="observation close time")
            ),
            record.opened_event_index,
            record.last_event_index,
            record.closed_event_index,
            record.status,
            record.start_sequence,
            record.end_sequence,
            record.connection_id,
            record.stale_reason,
            record.resync_reason,
            _json(record.metadata),
        )

        def write() -> None:
            self._require_running_run(record.run_id)
            existing = self.connection.execute(
                """
                SELECT run_id, market_ticker, opened_at, opened_event_index,
                       updated_at, last_event_index, closed_at, closed_event_index
                FROM market_observation_windows
                WHERE observation_id = ?
                """,
                (record.observation_id,),
            ).fetchone()
            if existing is not None:
                identity = (
                    str(existing[0]),
                    str(existing[1]),
                    _require_aware(cast(datetime, existing[2]), label="stored window open"),
                    int(existing[3]),
                )
                expected = (
                    record.run_id,
                    record.market_ticker,
                    _require_aware(record.opened_at, label="observation open time"),
                    record.opened_event_index,
                )
                if identity != expected:
                    raise ValueError("market observation ID collides with another window")
                persisted_updated_at = _require_aware(
                    cast(datetime, existing[4]), label="stored window update"
                )
                if record.updated_at < persisted_updated_at or record.last_event_index < int(
                    existing[5]
                ):
                    raise ValueError("market observation update would regress persisted state")
                if existing[6] is not None:
                    persisted_close = (
                        _require_aware(cast(datetime, existing[6]), label="stored window close"),
                        int(existing[7]),
                    )
                    incoming_close = (record.closed_at, record.closed_event_index)
                    if persisted_close != incoming_close:
                        raise ValueError("closed market observation cannot be changed")
            self.connection.execute(
                """
                INSERT INTO market_observation_windows (
                    observation_id, run_id, market_ticker, opened_at, updated_at,
                    closed_at, opened_event_index, last_event_index, closed_event_index,
                    status, start_sequence, end_sequence, connection_id, stale_reason,
                    resync_reason, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (observation_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    closed_at = excluded.closed_at,
                    last_event_index = excluded.last_event_index,
                    closed_event_index = excluded.closed_event_index,
                    status = excluded.status,
                    end_sequence = excluded.end_sequence,
                    connection_id = excluded.connection_id,
                    stale_reason = excluded.stale_reason,
                    resync_reason = excluded.resync_reason,
                    metadata_json = excluded.metadata_json
                """,
                row,
            )

        self._write_transaction(f"record market observation {record.observation_id}", write)

    def upsert_event_fee_state(self, event: Event) -> None:
        """Atomically persist a refreshed current override and complete fee schedule."""

        self.migrate()
        updated_at = _require_aware(self._clock(), label="event fee update time")

        def write() -> None:
            self._upsert("events", _EVENT_COLUMNS, (_event_row(event, updated_at),))

        self._write_transaction(f"upsert event fee state {event.ticker}", write)

    def upsert_market_state(self, market: Market) -> None:
        """Atomically persist one complete lifecycle-refreshed market record."""

        self.migrate()
        updated_at = _require_aware(self._clock(), label="market state update time")

        def write() -> None:
            self._upsert("markets", _MARKET_COLUMNS, (_market_row(market, updated_at),))

        self._write_transaction(f"upsert market state {market.ticker}", write)

    def get_event_fee_state(self, event_ticker: str) -> EventFeeState | None:
        """Return the exact current override and scheduled fee changes for one event."""

        _nonblank(event_ticker, label="event ticker")
        self.migrate()
        row = self.connection.execute(
            """
            SELECT fee_type_override, fee_multiplier_override, fee_changes_json, updated_at
            FROM events
            WHERE ticker = ?
            """,
            (event_ticker,),
        ).fetchone()
        if row is None:
            return None
        raw_changes = _json_value(row[2])
        if not isinstance(raw_changes, list) or any(
            not isinstance(change, dict) for change in raw_changes
        ):
            raise StorageError("stored event fee_changes_json is invalid")
        try:
            changes = tuple(EventFeeChange.model_validate(change) for change in raw_changes)
        except ValidationError as exc:
            raise StorageError("stored event fee_changes_json is invalid") from exc
        return EventFeeState(
            event_ticker=event_ticker,
            fee_type_override=None if row[0] is None else str(row[0]),
            fee_multiplier_override=cast(Decimal | None, row[1]),
            fee_changes=changes,
            updated_at=_require_aware(cast(datetime, row[3]), label="stored event fee update"),
        )

    def persist_transition(self, observation: OpportunityObservation) -> None:
        """Persist one lifecycle decision and any summary/legs as one atomic unit."""

        self.migrate()

        def write() -> None:
            self._require_running_run(observation.run_id)
            if not self._upsert_opportunity_observation(observation):
                return
            if observation.transition is OpportunityTransition.NOT_PRESENT:
                return
            if observation.transition is OpportunityTransition.CLOSED:
                self._close_opportunity(observation)
                return
            if observation.transition is OpportunityTransition.RIGHT_CENSORED:
                self._right_censor_opportunity(observation)
                return
            self._upsert_open_opportunity(observation)

        self._write_transaction(
            f"persist opportunity transition {observation.observation_id}",
            write,
        )

    def persist_paper_execution(self, record: PaperExecutionRecord) -> None:
        """Persist one source-identified paper attempt and update its summary atomically."""

        self.migrate()
        payload_hash = _paper_execution_payload_hash(record)

        def write() -> None:
            self._require_running_run(record.run_id)
            source = self.connection.execute(
                """
                SELECT run_id, opportunity_id, event_index, stage
                FROM opportunity_observations
                WHERE observation_id = ?
                """,
                (record.source_observation_id,),
            ).fetchone()
            if source is None:
                raise ValueError("paper execution requires its persisted source observation")
            expected_source = (
                record.run_id,
                record.request.opportunity_id,
                record.source_event_index,
            )
            persisted_source = (
                str(source[0]),
                None if source[1] is None else str(source[1]),
                int(source[2]),
            )
            if persisted_source != expected_source:
                raise ValueError("paper execution source identity conflicts with persistence")
            if str(source[3]) != "stage_2":
                raise ValueError("paper execution source must be a Stage 2 observation")

            existing = self.connection.execute(
                "SELECT payload_hash FROM paper_executions WHERE attempt_id = ?",
                (record.attempt_id,),
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != payload_hash:
                    raise ValueError("paper attempt ID conflicts with a different payload")
                return

            result = record.result
            opportunity = record.request.opportunity
            self.connection.execute(
                """
                INSERT INTO paper_executions (
                    attempt_id, run_id, opportunity_id, source_observation_id,
                    source_event_index, detected_at, scheduled_at, attempted_at,
                    resolved_at, latency_ms, status, failure_reason,
                    minimum_terminal_payout, expected_profit, simulated_locked_profit,
                    expected_cost, actual_cost, expected_fees, actual_fees,
                    expected_fee_policy_version, expected_fee_policies_json,
                    expected_fee_quotes_json, execution_fee_policy_version,
                    execution_fee_policies_json, execution_fee_quotes_json,
                    evidence_json, payload_hash
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    record.attempt_id,
                    record.run_id,
                    record.request.opportunity_id,
                    record.source_observation_id,
                    record.source_event_index,
                    _require_aware(result.detected_at, label="paper detection time"),
                    _require_aware(
                        result.simulated_execution_at,
                        label="paper scheduled time",
                    ),
                    (
                        None
                        if record.attempted_at is None
                        else _require_aware(record.attempted_at, label="paper attempt time")
                    ),
                    _require_aware(record.resolved_at, label="paper resolution time"),
                    result.latency_ms,
                    result.status.value,
                    result.failure_reason,
                    result.minimum_terminal_payout,
                    result.expected_profit,
                    result.simulated_locked_profit,
                    result.expected_cost,
                    result.actual_cost,
                    result.expected_fees,
                    result.actual_fees,
                    _fee_policy_version(opportunity.fee_policies),
                    _json(
                        [policy.model_dump(mode="python") for policy in opportunity.fee_policies]
                    ),
                    _json([quote.model_dump(mode="python") for quote in opportunity.fee_quotes]),
                    _fee_policy_version(result.execution_fee_policies),
                    _json(
                        [
                            policy.model_dump(mode="python")
                            for policy in result.execution_fee_policies
                        ]
                    ),
                    _json(
                        [quote.model_dump(mode="python") for quote in result.execution_fee_quotes]
                    ),
                    _json(record.evidence),
                    payload_hash,
                ),
            )
            self.connection.executemany(
                """
                INSERT INTO paper_execution_legs (
                    attempt_id, leg_index, ticker, side, quantity,
                    expected_prices_json, actual_prices_json, expected_average_price,
                    actual_average_price, expected_cost, actual_cost, fill_status,
                    failure_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        record.attempt_id,
                        leg_index,
                        leg.ticker,
                        leg.side,
                        leg.quantity,
                        _json([fill.model_dump(mode="python") for fill in leg.expected_prices]),
                        _json([fill.model_dump(mode="python") for fill in leg.actual_prices]),
                        leg.expected_average_price,
                        leg.actual_average_price,
                        leg.expected_cost,
                        leg.actual_cost,
                        leg.fill_status,
                        leg.failure_reason,
                    )
                    for leg_index, leg in enumerate(result.legs)
                ),
            )

            summary = self.connection.execute(
                """
                SELECT paper_execution_attempt_id, paper_execution_source_event_index
                FROM opportunities
                WHERE opportunity_id = ? AND run_id = ?
                """,
                (record.request.opportunity_id, record.run_id),
            ).fetchone()
            if summary is None:
                raise ValueError("paper execution requires its persisted opportunity summary")
            latest_economic = self.connection.execute(
                """
                SELECT max(event_index)
                FROM opportunity_observations
                WHERE opportunity_id = ? AND transition IN ('open', 'updated')
                """,
                (record.request.opportunity_id,),
            ).fetchone()
            if (
                latest_economic is None
                or latest_economic[0] is None
                or int(latest_economic[0]) > record.source_event_index
            ):
                return
            summary_attempt_id = None if summary[0] is None else str(summary[0])
            summary_event_index = None if summary[1] is None else int(summary[1])
            if summary_event_index is not None and summary_event_index > record.source_event_index:
                return
            if (
                summary_event_index == record.source_event_index
                and summary_attempt_id != record.attempt_id
            ):
                raise ValueError("paper summary event index belongs to a different attempt")

            if result.status is PaperExecutionStatus.SURVIVED:
                self.connection.execute(
                    """
                    UPDATE opportunities
                    SET stage = 'stage_3', paper_execution_status = 'survived',
                        paper_execution_reason = NULL, paper_execution_attempt_id = ?,
                        paper_execution_source_event_index = ?
                    WHERE opportunity_id = ?
                    """,
                    (
                        record.attempt_id,
                        record.source_event_index,
                        record.request.opportunity_id,
                    ),
                )
            else:
                self.connection.execute(
                    """
                    UPDATE opportunities
                    SET stage = CASE WHEN stage = 'stage_3' THEN 'stage_2' ELSE stage END,
                        paper_execution_status = 'failed', paper_execution_reason = ?,
                        paper_execution_attempt_id = ?,
                        paper_execution_source_event_index = ?
                    WHERE opportunity_id = ?
                    """,
                    (
                        result.failure_reason,
                        record.attempt_id,
                        record.source_event_index,
                        record.request.opportunity_id,
                    ),
                )

        self._write_transaction(f"persist paper execution {record.attempt_id}", write)

    def _upsert_opportunity_observation(self, observation: OpportunityObservation) -> bool:
        opportunity = observation.opportunity
        capacity = None if opportunity is None else opportunity.capital_required
        evidence_capacity = _optional_decimal(
            observation.evidence.get("capacity"),
            label="capacity",
        )
        if evidence_capacity is not None and evidence_capacity != capacity:
            raise ValueError("capacity evidence must equal capital required at optimum")
        if _OBSERVATION_PAYLOAD_HASH_KEY in observation.metadata:
            raise ValueError(
                f"observation metadata key {_OBSERVATION_PAYLOAD_HASH_KEY!r} is reserved"
            )
        payload_hash = _observation_payload_hash(observation)
        existing = self.connection.execute(
            "SELECT metadata_json FROM opportunity_observations WHERE observation_id = ?",
            (observation.observation_id,),
        ).fetchone()
        if existing is not None:
            stored_metadata = _json_mapping(existing[0])
            if stored_metadata.get(_OBSERVATION_PAYLOAD_HASH_KEY) != payload_hash:
                raise ValueError("opportunity observation ID conflicts with a different payload")
            return False
        stored_metadata = dict(observation.metadata)
        stored_metadata[_OBSERVATION_PAYLOAD_HASH_KEY] = payload_hash
        self.connection.execute(
            """
            INSERT INTO opportunity_observations (
                observation_id, opportunity_id, run_id, component_id, observed_at,
                event_index, transition, stage, solver_status, reason, solve_duration_ms,
                num_states, num_instruments, num_legs, capital_required, gross_profit,
                fees, net_profit, gross_edge, net_edge, capacity, fee_policy_version,
                fee_policies_json, market_tickers_json, relation_types_json,
                relation_sources_json, market_contexts_json, evidence_json, metadata_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                observation.observation_id,
                observation.opportunity_id,
                observation.run_id,
                observation.component_id,
                _require_aware(observation.observed_at, label="opportunity observation time"),
                observation.event_index,
                observation.transition.value,
                None if opportunity is None else opportunity.stage.value,
                observation.solver_status,
                observation.close_reason or observation.censor_reason or observation.solver_reason,
                float(observation.solve_duration_ms),
                observation.num_states,
                observation.num_instruments,
                observation.num_legs,
                None if opportunity is None else opportunity.capital_required,
                None if opportunity is None else opportunity.gross_profit,
                None if opportunity is None else opportunity.fees,
                None if opportunity is None else opportunity.net_profit,
                None if opportunity is None else opportunity.gross_edge,
                None if opportunity is None else opportunity.net_edge,
                capacity,
                None if opportunity is None else _fee_policy_version(opportunity.fee_policies),
                (
                    "[]"
                    if opportunity is None
                    else _json(
                        [policy.model_dump(mode="python") for policy in opportunity.fee_policies]
                    )
                ),
                _json(observation.market_tickers),
                _json([relation_type.value for relation_type in observation.relation_types]),
                _json(observation.relation_sources),
                _json([context.model_dump(mode="json") for context in observation.market_contexts]),
                _json(observation.evidence),
                _json(stored_metadata),
            ),
        )
        if observation.portfolio_legs:
            assert observation.opportunity_id is not None
            self.connection.executemany(
                """
                INSERT INTO opportunity_observation_legs (
                    observation_id, leg_index, opportunity_id, ticker, side, price,
                    quantity, source_side, source_price, fee
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        observation.observation_id,
                        leg_index,
                        observation.opportunity_id,
                        leg.ticker,
                        leg.side,
                        leg.price,
                        leg.quantity,
                        leg.source_side,
                        leg.source_price,
                        leg.fee,
                    )
                    for leg_index, leg in enumerate(observation.portfolio_legs)
                ),
            )
        return True

    def _upsert_open_opportunity(self, observation: OpportunityObservation) -> None:
        opportunity = observation.opportunity
        opportunity_id = observation.opportunity_id
        if opportunity is None or opportunity_id is None:
            raise ValueError("open/update transition requires opportunity evidence")
        existing = self.connection.execute(
            """
            SELECT run_id, component_id, opened_at, closed_at, updated_at, censored_at
            FROM opportunities
            WHERE opportunity_id = ?
            """,
            (opportunity_id,),
        ).fetchone()
        if observation.transition is OpportunityTransition.UPDATED and existing is None:
            raise ValueError("cannot update an opportunity before it opens")
        opened_at = observation.observed_at
        if existing is not None:
            if observation.transition is OpportunityTransition.OPEN:
                raise ValueError("cannot open an opportunity that already exists")
            if (
                str(existing[0]) != observation.run_id
                or str(existing[1]) != observation.component_id
            ):
                raise ValueError("opportunity ID collides with another run or component")
            if existing[3] is not None:
                raise ValueError("cannot update a closed opportunity")
            if existing[5] is not None:
                raise ValueError("cannot update a right-censored opportunity")
            opened_at = cast(datetime, existing[2])
            persisted_update = _require_aware(
                cast(datetime, existing[4]),
                label="stored opportunity update",
            )
            if observation.observed_at < persisted_update:
                raise ValueError("opportunity update time cannot regress")
        fee_policy_version = _fee_policy_version(opportunity.fee_policies)
        fee_policies_json = _json(
            [policy.model_dump(mode="python") for policy in opportunity.fee_policies]
        )
        summary_metadata = _json(
            {
                "evidence": observation.evidence,
                "fee_quotes": [quote.model_dump(mode="python") for quote in opportunity.fee_quotes],
                "gross_state_profits": opportunity.gross_state_profits,
                "metadata": observation.metadata,
                "net_state_profits": opportunity.net_state_profits,
                "portfolio_signature": observation.portfolio_signature,
                "solver_reason": observation.solver_reason,
                "solver_status": observation.solver_status,
            }
        )
        self.connection.execute(
            """
            INSERT INTO opportunities (
                opportunity_id, component_id, detected_at, ended_at, stage,
                relation_types_json, num_markets, num_legs, capital_required,
                gross_profit, fees, net_profit, gross_edge, net_edge,
                fee_policy_version, fee_policies_json, paper_execution_status,
                paper_execution_reason, metadata_json, run_id, opened_at, closed_at,
                updated_at
            ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT (opportunity_id) DO UPDATE SET
                stage = excluded.stage,
                relation_types_json = excluded.relation_types_json,
                num_markets = excluded.num_markets,
                num_legs = excluded.num_legs,
                capital_required = excluded.capital_required,
                gross_profit = excluded.gross_profit,
                fees = excluded.fees,
                net_profit = excluded.net_profit,
                gross_edge = excluded.gross_edge,
                net_edge = excluded.net_edge,
                fee_policy_version = excluded.fee_policy_version,
                fee_policies_json = excluded.fee_policies_json,
                paper_execution_status = excluded.paper_execution_status,
                paper_execution_reason = excluded.paper_execution_reason,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                opportunity_id,
                observation.component_id,
                _require_aware(opened_at, label="opportunity open time"),
                opportunity.stage.value,
                _json([item.value for item in observation.relation_types]),
                observation.num_markets,
                observation.num_legs,
                opportunity.capital_required,
                opportunity.gross_profit,
                opportunity.fees,
                opportunity.net_profit,
                opportunity.gross_edge,
                opportunity.net_edge,
                fee_policy_version,
                fee_policies_json,
                opportunity.paper_execution_status,
                None,
                summary_metadata,
                observation.run_id,
                _require_aware(opened_at, label="opportunity open time"),
                _require_aware(observation.observed_at, label="opportunity update time"),
            ),
        )
        self.connection.execute(
            "DELETE FROM portfolio_legs WHERE opportunity_id = ?",
            (opportunity_id,),
        )
        if observation.portfolio_legs:
            self.connection.executemany(
                """
                INSERT INTO portfolio_legs (
                    opportunity_id, ticker, side, price, quantity, source_side,
                    source_price, fee
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        opportunity_id,
                        leg.ticker,
                        leg.side,
                        leg.price,
                        leg.quantity,
                        leg.source_side,
                        leg.source_price,
                        leg.fee,
                    )
                    for leg in observation.portfolio_legs
                ),
            )

    def _close_opportunity(self, observation: OpportunityObservation) -> None:
        opportunity_id = observation.opportunity_id
        if opportunity_id is None:
            raise ValueError("closed transition requires an opportunity ID")
        existing = self.connection.execute(
            """
            SELECT run_id, component_id, opened_at, closed_at, updated_at, censored_at
            FROM opportunities
            WHERE opportunity_id = ?
            """,
            (opportunity_id,),
        ).fetchone()
        if existing is None:
            raise ValueError("cannot close an opportunity before it opens")
        if str(existing[0]) != observation.run_id or str(existing[1]) != observation.component_id:
            raise ValueError("opportunity ID collides with another run or component")
        if existing[5] is not None:
            raise ValueError("cannot close a right-censored opportunity")
        closed_at = _require_aware(observation.observed_at, label="opportunity close time")
        opened_at = _require_aware(cast(datetime, existing[2]), label="stored opportunity open")
        if closed_at < opened_at:
            raise ValueError("opportunity close cannot precede its open")
        persisted_update = _require_aware(
            cast(datetime, existing[4]),
            label="stored opportunity update",
        )
        if closed_at < persisted_update:
            raise ValueError("opportunity close cannot precede its latest update")
        if existing[3] is not None:
            persisted_close = _require_aware(
                cast(datetime, existing[3]), label="stored opportunity close"
            )
            if persisted_close != closed_at:
                raise ValueError("opportunity is already closed at a different time")
            return
        self.connection.execute(
            """
            UPDATE opportunities
            SET ended_at = ?, closed_at = ?, updated_at = ?
            WHERE opportunity_id = ?
            """,
            (
                closed_at,
                closed_at,
                closed_at,
                opportunity_id,
            ),
        )

    def _right_censor_opportunity(self, observation: OpportunityObservation) -> None:
        opportunity_id = observation.opportunity_id
        if opportunity_id is None or observation.opportunity is None:
            raise ValueError("right-censored transition requires opportunity evidence")
        existing = self.connection.execute(
            """
            SELECT run_id, component_id, opened_at, closed_at, updated_at,
                   censored_at, censored_event_index, censor_reason
            FROM opportunities
            WHERE opportunity_id = ?
            """,
            (opportunity_id,),
        ).fetchone()
        if existing is None:
            raise ValueError("cannot right-censor an opportunity before it opens")
        if str(existing[0]) != observation.run_id or str(existing[1]) != observation.component_id:
            raise ValueError("opportunity ID collides with another run or component")
        if existing[3] is not None:
            raise ValueError("cannot right-censor a closed opportunity")
        censored_at = _require_aware(
            observation.observed_at,
            label="opportunity censor time",
        )
        opened_at = _require_aware(cast(datetime, existing[2]), label="stored opportunity open")
        persisted_update = _require_aware(
            cast(datetime, existing[4]),
            label="stored opportunity update",
        )
        if censored_at < opened_at or censored_at < persisted_update:
            raise ValueError("opportunity censor cannot precede its persisted episode state")
        if existing[5] is not None:
            persisted = (
                _require_aware(cast(datetime, existing[5]), label="stored opportunity censor"),
                int(existing[6]),
                str(existing[7]),
            )
            incoming = (
                censored_at,
                observation.event_index,
                observation.censor_reason,
            )
            if persisted != incoming:
                raise ValueError("opportunity is already right-censored differently")
            return
        self.connection.execute(
            """
            UPDATE opportunities
            SET censored_at = ?, censored_event_index = ?, censor_reason = ?,
                updated_at = ?
            WHERE opportunity_id = ?
            """,
            (
                censored_at,
                observation.event_index,
                observation.censor_reason,
                censored_at,
                opportunity_id,
            ),
        )


def _market_row(market: Market, updated_at: datetime) -> tuple[object, ...]:
    return (
        market.ticker,
        market.event_ticker,
        market.series_ticker,
        market.market_type,
        market.title,
        market.subtitle,
        market.yes_sub_title,
        market.no_sub_title,
        market.status,
        market.created_time,
        market.updated_time,
        market.open_time,
        market.close_time,
        market.expiration_time,
        market.latest_expiration_time,
        market.expected_expiration_time,
        market.settlement_ts,
        market.occurrence_datetime,
        market.strike_type,
        market.floor_strike,
        market.cap_strike,
        market.functional_strike,
        None if market.custom_strike is None else _json(market.custom_strike),
        market.rules_primary,
        market.rules_secondary,
        market.early_close_condition,
        market.price_level_structure,
        _json([item.model_dump(mode="json") for item in market.price_ranges]),
        market.result,
        _json(market.raw),
        updated_at,
    )


def _event_row(event: Event, updated_at: datetime) -> tuple[object, ...]:
    return (
        event.ticker,
        event.series_ticker,
        event.title,
        event.subtitle,
        event.category,
        event.mutually_exclusive,
        event.available_on_brokers,
        _json(event.market_tickers),
        event.last_updated_ts,
        event.fee_type_override,
        event.fee_multiplier_override,
        _json([change.model_dump(mode="json") for change in event.fee_changes]),
        _json(event.raw),
        updated_at,
    )


def _series_row(series: Series, updated_at: datetime) -> tuple[object, ...]:
    return (
        series.ticker,
        series.title,
        series.frequency,
        series.category,
        _json(series.tags),
        series.fee_type,
        series.fee_multiplier,
        _json([item.model_dump(mode="json") for item in series.settlement_sources]),
        series.contract_url,
        series.contract_terms_url,
        series.last_updated_ts,
        _json(series.raw),
        updated_at,
    )


def _json_value(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise StorageError("stored JSON is invalid") from exc
    return value


def _json_mapping(value: object) -> dict[str, Any]:
    parsed = _json_value(value)
    if not isinstance(parsed, dict):
        raise StorageError("stored raw_json is not an object")
    return cast(dict[str, Any], parsed)


def _optional_decimal(value: object, *, label: str) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{label} must be a finite Decimal")
    return value


def _fee_policy_version(policies: Sequence[EffectiveFeePolicy]) -> str | None:
    versions = sorted({policy.policy_version for policy in policies})
    if not versions:
        return None
    if len(versions) == 1:
        return versions[0]
    payload = _json(versions)
    return f"fee-policy-set:{sha256(payload.encode()).hexdigest()}"
