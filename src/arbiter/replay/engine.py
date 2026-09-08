"""Deterministic event-time replay over one self-contained recorded run.

Replay rebuilds every decision-bearing input from the first ``RunStartedEvent`` and
feeds every persisted record through :meth:`ArbitrageEngine.process_record`.  Wall
clock speed is deliberately isolated from event time: changing ``speed`` can only
change calls to the injected sleeper, never scanner, freshness, debounce, or paper
execution timestamps.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from itertools import groupby
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from arbiter.config import (
    CollectorSettings,
    EngineSettings,
    FeeSettings,
    KalshiEnvironment,
    OrderBookSettings,
    PaperExecutionSettings,
)
from arbiter.engine.paper_execution import (
    PaperExecutionRequest,
    PaperExecutionResult,
    PaperExecutor,
)
from arbiter.engine.scanner import (
    ArbitrageEngine,
    ComponentScanner,
    EngineScanDecision,
    OpportunityLifecycle,
)
from arbiter.engine.state import EngineState, MetadataStateError
from arbiter.models.opportunity import (
    EffectiveFeePolicy,
    OpportunityEpisode,
    OpportunityObservation,
    OpportunityStage,
    OpportunityTransition,
)
from arbiter.replay.events import (
    RecordedEvent,
    RunEndedEvent,
    RunInputPayload,
    RunStartedEvent,
    dump_recorded_event_json,
)
from arbiter.solver.fees import (
    KALSHI_FEE_POLICY_VERSION,
    FeePolicyResolutionError,
    resolve_fee_policy,
)
from arbiter.storage.duckdb import (
    PaperExecutionRecord,
    RunManifestRecord,
    RunStreamEvidence,
    StoredRunManifest,
    paper_execution_attempt_id,
)

ReplaySpeed = float | Literal["max"]
Sleeper = Callable[[float], None]


class ReplayValidationError(ValueError):
    """Raised before or during replay when a recording is not self-contained."""


class ReplayConfiguration(BaseModel):
    """Decision-bearing settings embedded in a schema-v2 run-start record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine: EngineSettings
    orderbook: OrderBookSettings
    fees: FeeSettings
    paper_execution: PaperExecutionSettings
    collector: CollectorSettings
    kalshi_environment: KalshiEnvironment


class ReplayStore(Protocol):
    """Narrow durable boundary shared by lifecycle and paper evidence."""

    def persist_transition(self, observation: OpportunityObservation) -> None: ...

    def persist_paper_execution(self, record: PaperExecutionRecord) -> None: ...


class ReplayRepository(ReplayStore, Protocol):
    """Durable replay boundary, including its own immutable run manifest."""

    def start_run(self, record: RunManifestRecord) -> None: ...

    def finalize_run(
        self,
        run_id: str,
        *,
        status: str,
        ended_at: datetime | None = None,
        error: str | None = None,
        stream: RunStreamEvidence | None = None,
    ) -> None: ...

    def get_run_manifest(self, run_id: str) -> StoredRunManifest | None: ...


class InMemoryReplayStore:
    """Small deterministic store for fixtures and callers that do not need DuckDB."""

    def __init__(self) -> None:
        self.observations: list[OpportunityObservation] = []
        self.paper_executions: list[PaperExecutionRecord] = []

    def persist_transition(self, observation: OpportunityObservation) -> None:
        self.observations.append(observation)

    def persist_paper_execution(self, record: PaperExecutionRecord) -> None:
        self.paper_executions.append(record)


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Complete in-process evidence returned by one successful replay."""

    replay_run_id: str
    recorded_run_id: str
    first_event_index: int
    last_event_index: int
    event_count: int
    event_stream_hash: str
    recorded_through: datetime
    configuration: ReplayConfiguration
    observations: tuple[OpportunityObservation, ...]
    paper_executions: tuple[PaperExecutionRecord, ...]
    completed_episodes: tuple[OpportunityEpisode, ...]
    censored_episodes: tuple[OpportunityEpisode, ...]
    final_state: EngineState

    @property
    def economic_observations(self) -> tuple[dict[str, object], ...]:
        """Return the cross-run comparable observation projection."""

        return economic_observation_projection(self.observations)


@dataclass(frozen=True, slots=True)
class _ValidatedRecording:
    records: tuple[RecordedEvent, ...]
    started: RunStartedEvent
    ended: RunEndedEvent
    configuration: ReplayConfiguration
    stream_hash: str


@dataclass(frozen=True, slots=True)
class _PendingPaperExecution:
    observation: OpportunityObservation
    request: PaperExecutionRequest

    @property
    def order_key(self) -> tuple[datetime, int, str, str]:
        return (
            self.request.execute_at,
            self.observation.event_index,
            self.observation.component_id,
            self.observation.observation_id,
        )


def recorded_event_stream_hash(records: Sequence[RecordedEvent]) -> str:
    """Hash records with the exact unambiguous framing used by live collection."""

    digest = sha256()
    for record in records:
        payload = dump_recorded_event_json(record).encode()
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def economic_observation_projection(
    observations: Sequence[OpportunityObservation],
) -> tuple[dict[str, object], ...]:
    """Project deterministic economics while excluding run identity and solve timing."""

    projected: list[dict[str, object]] = []
    for observation in observations:
        payload = observation.model_dump(
            mode="json",
            exclude={
                "observation_id",
                "opportunity_id",
                "run_id",
                "solve_duration_ms",
            },
        )
        projected.append(cast(dict[str, object], payload))
    return tuple(projected)


class ReplayEngine:
    """Validate and replay one schema-v2 run through the production engine path."""

    def __init__(
        self,
        records: Sequence[RecordedEvent],
        *,
        replay_run_id: str,
        store: ReplayStore,
        speed: ReplaySpeed = "max",
        sleeper: Sleeper = time.sleep,
    ) -> None:
        if not replay_run_id or not replay_run_id.strip():
            raise ReplayValidationError("replay_run_id cannot be blank")
        self._recording = _validate_recording(records)
        if replay_run_id == self._recording.started.run_id:
            raise ReplayValidationError("replay_run_id must differ from the source recorded run ID")
        self.replay_run_id = replay_run_id
        self.store = store
        self.speed = _validate_speed(speed)
        self.sleeper = sleeper
        self._has_run = False
        self._pending_paper: list[_PendingPaperExecution] = []
        self._paper_records: list[PaperExecutionRecord] = []

        inputs = self._recording.started.inputs
        config = self._recording.configuration
        try:
            self.state = EngineState(
                markets=inputs.markets,
                events=inputs.events,
                series=inputs.series,
                relations=inputs.relations,
                max_component_markets=config.engine.max_component_markets,
            )
        except (MetadataStateError, ValueError) as exc:
            raise ReplayValidationError(f"invalid recorded engine inputs: {exc}") from exc
        scanner = ComponentScanner(
            self.state,
            stale_after=timedelta(milliseconds=config.orderbook.stale_after_ms),
            account_precision=config.fees.account_precision,
            minimum_net_profit=config.engine.min_net_profit_dollars,
            minimum_net_edge_bps=config.engine.min_net_edge_bps,
        )
        self.lifecycle = OpportunityLifecycle(run_id=replay_run_id, store=store)
        self.engine = ArbitrageEngine(
            state=self.state,
            scanner=scanner,
            lifecycle=self.lifecycle,
            solve_debounce=timedelta(milliseconds=config.engine.solve_debounce_ms),
            recorded_run_id=self._recording.started.run_id,
        )
        self.paper_executor = PaperExecutor(
            latency_ms=config.paper_execution.latency_ms,
            stale_after=timedelta(milliseconds=config.orderbook.stale_after_ms),
            account_precision=config.fees.account_precision,
            allow_partial_fill=config.paper_execution.allow_partial_fill,
        )

    def run(self) -> ReplayResult:
        """Run once, preserving recorded ordering and all event-time boundaries."""

        if self._has_run:
            raise RuntimeError("a ReplayEngine instance can run only once")
        self._has_run = True

        records = self._recording.records
        previous_record_time = records[0].local_received_ts
        for received_at, group_iter in groupby(
            records,
            key=lambda record: record.local_received_ts,
        ):
            self._sleep_between(previous_record_time, received_at)
            self._drain_timers_before(received_at)
            for record in group_iter:
                try:
                    processed = self.engine.process_record(record)
                except (ValueError, MetadataStateError) as exc:
                    raise ReplayValidationError(
                        f"record {record.event_index} ({record.event_type}) is invalid: {exc}"
                    ) from exc
                self._schedule_paper(processed.scans_before_record)
                self._schedule_paper(processed.scans_after_record)

            # Every record sharing this timestamp precedes timers due exactly now.
            self._drain_timers_through(received_at)
            previous_record_time = received_at

        coverage = self._recording.ended.local_received_ts
        if self.engine.next_due_at is not None:
            due_at = self.engine.next_due_at
            raise ReplayValidationError(
                "recording ended before trailing solve debounce completed; "
                f"next scan is due at {due_at.isoformat()} after coverage "
                f"{coverage.isoformat()}"
            )

        self._censor_pending_paper(recorded_through=coverage)
        censor_reason = self._recording.ended.reason or f"run_{self._recording.ended.status}"
        self.lifecycle.right_censor_all(
            observed_at=coverage,
            event_index=self._recording.ended.event_index,
            reason=censor_reason,
        )
        return ReplayResult(
            replay_run_id=self.replay_run_id,
            recorded_run_id=self._recording.started.run_id,
            first_event_index=records[0].event_index,
            last_event_index=records[-1].event_index,
            event_count=len(records),
            event_stream_hash=self._recording.stream_hash,
            recorded_through=coverage,
            configuration=self._recording.configuration,
            observations=self.lifecycle.observations,
            paper_executions=tuple(self._paper_records),
            completed_episodes=self.lifecycle.completed_episodes,
            censored_episodes=self.lifecycle.censored_episodes,
            final_state=self.state,
        )

    def _sleep_between(self, previous: datetime, current: datetime) -> None:
        if self.speed == "max":
            return
        elapsed = (current - previous).total_seconds()
        if elapsed > 0:
            self.sleeper(elapsed / self.speed)

    def _drain_timers_before(self, boundary: datetime) -> None:
        while True:
            scan_due = self.engine.next_due_at
            paper_due = self._next_paper_due_at
            due = min(
                (
                    candidate
                    for candidate in (scan_due, paper_due)
                    if candidate is not None and candidate < boundary
                ),
                default=None,
            )
            if due is None:
                return
            if scan_due is not None and scan_due == due:
                self._schedule_paper(self.engine.advance_time(due))
            self._execute_paper_through(due)

    def _drain_timers_through(self, boundary: datetime) -> None:
        scan_due = self.engine.next_due_at
        if scan_due is not None and scan_due <= boundary:
            self._schedule_paper(self.engine.advance_time(boundary))
        self._execute_paper_through(boundary)

    @property
    def _next_paper_due_at(self) -> datetime | None:
        if not self._pending_paper:
            return None
        return min(item.request.execute_at for item in self._pending_paper)

    def _schedule_paper(self, decisions: Sequence[EngineScanDecision]) -> None:
        if not self._recording.configuration.paper_execution.enabled:
            return
        for decision in decisions:
            observation = decision.observation
            opportunity = observation.opportunity
            if (
                observation.transition
                not in {OpportunityTransition.OPEN, OpportunityTransition.UPDATED}
                or opportunity is None
                or opportunity.stage is not OpportunityStage.NET_EXECUTABLE
                or observation.opportunity_id is None
            ):
                continue
            request = self.paper_executor.schedule(
                opportunity_id=observation.opportunity_id,
                opportunity=opportunity,
                detected_at=observation.observed_at,
            )
            self._pending_paper.append(
                _PendingPaperExecution(observation=observation, request=request)
            )

    def _execute_paper_through(self, boundary: datetime) -> None:
        due = sorted(
            (pending for pending in self._pending_paper if pending.request.execute_at <= boundary),
            key=lambda pending: pending.order_key,
        )
        if not due:
            return
        due_ids = {pending.observation.observation_id for pending in due}
        self._pending_paper = [
            pending
            for pending in self._pending_paper
            if pending.observation.observation_id not in due_ids
        ]
        for pending in due:
            result = self.paper_executor.execute(
                pending.request,
                orderbooks=self.state.orderbooks,
                fee_policies=self._current_fee_policies(pending.request),
            )
            self._persist_paper(
                pending,
                result=result,
                resolved_at=pending.request.execute_at,
            )

    def _current_fee_policies(
        self,
        request: PaperExecutionRequest,
    ) -> tuple[EffectiveFeePolicy, ...]:
        tickers = tuple(
            sorted({allocation.instrument.ticker for allocation in request.opportunity.quantities})
        )
        policies: list[EffectiveFeePolicy] = []
        try:
            for ticker in tickers:
                if not self.state.is_fee_ready_for_market(ticker):
                    return ()
                event, series = self.state.fee_metadata_for_market(ticker)
                policies.append(
                    resolve_fee_policy(
                        event=event,
                        series=series,
                        as_of=request.execute_at,
                        market_ticker=ticker,
                    )
                )
        except (MetadataStateError, FeePolicyResolutionError):
            return ()
        return tuple(policies)

    def _censor_pending_paper(self, *, recorded_through: datetime) -> None:
        pending_items = sorted(self._pending_paper, key=lambda pending: pending.order_key)
        self._pending_paper.clear()
        for pending in pending_items:
            if pending.request.detected_at > recorded_through:
                raise ReplayValidationError(
                    "a paper attempt was detected beyond recorded coverage; "
                    "RunEnded must follow the final debounce deadline"
                )
            result = self.paper_executor.insufficient_future_data(
                pending.request,
                recorded_through=recorded_through,
            )
            self._persist_paper(
                pending,
                result=result,
                resolved_at=recorded_through,
            )

    def _persist_paper(
        self,
        pending: _PendingPaperExecution,
        *,
        result: PaperExecutionResult,
        resolved_at: datetime,
    ) -> None:
        observation = pending.observation
        assert observation.opportunity_id is not None
        attempt_id = paper_execution_attempt_id(
            run_id=self.replay_run_id,
            opportunity_id=observation.opportunity_id,
            source_observation_id=observation.observation_id,
            source_event_index=observation.event_index,
        )
        record = PaperExecutionRecord(
            attempt_id=attempt_id,
            run_id=self.replay_run_id,
            source_observation_id=observation.observation_id,
            source_event_index=observation.event_index,
            request=pending.request,
            result=result,
            resolved_at=resolved_at,
            evidence={
                "recorded_run_id": self._recording.started.run_id,
                "event_stream_hash": self._recording.stream_hash,
            },
        )
        self.store.persist_paper_execution(record)
        self._paper_records.append(record)


def replay_records(
    records: Sequence[RecordedEvent],
    *,
    replay_run_id: str,
    store: ReplayStore,
    speed: ReplaySpeed = "max",
    sleeper: Sleeper = time.sleep,
) -> ReplayResult:
    """Convenience entry point for callers that do not need to retain engine state."""

    return ReplayEngine(
        records,
        replay_run_id=replay_run_id,
        store=store,
        speed=speed,
        sleeper=sleeper,
    ).run()


def split_recorded_runs(
    records: Sequence[RecordedEvent],
    *,
    allow_unbounded_records: bool = False,
) -> tuple[tuple[RecordedEvent, ...], ...]:
    """Extract complete run envelopes from one globally ordered recorded dataset.

    Date selection may deliberately ignore raw collector records outside a run envelope.
    Explicit file replay is strict by default so a midstream batch or raw-only recording
    fails clearly rather than silently inventing missing inputs.
    """

    if not records:
        raise ReplayValidationError("recording is empty")
    ordered = tuple(sorted(records, key=lambda record: record.event_index))
    _validate_order(ordered)
    runs: list[tuple[RecordedEvent, ...]] = []
    current: list[RecordedEvent] | None = None
    source_run_ids: set[str] = set()
    for record in ordered:
        if isinstance(record, RunStartedEvent):
            if current is not None:
                raise ReplayValidationError("recording contains a nested run-start boundary")
            current = [record]
            continue
        if current is None:
            if isinstance(record, RunEndedEvent):
                raise ReplayValidationError("recording contains a run-end without a run-start")
            if not allow_unbounded_records:
                raise ReplayValidationError(
                    "recording contains records outside a self-contained run boundary"
                )
            continue
        current.append(record)
        if isinstance(record, RunEndedEvent):
            run = tuple(current)
            validated = _validate_recording(run)
            source_run_id = validated.started.run_id
            if source_run_id in source_run_ids:
                raise ReplayValidationError("recording repeats a source run ID")
            source_run_ids.add(source_run_id)
            runs.append(run)
            current = None

    if current is not None:
        raise ReplayValidationError("recording ends before its run-end boundary")
    if not runs:
        raise ReplayValidationError("recording contains no self-contained replay run")
    return tuple(runs)


def build_replay_manifest(
    records: Sequence[RecordedEvent],
    *,
    replay_run_id: str,
    started_at: datetime,
    speed: ReplaySpeed = "max",
) -> RunManifestRecord:
    """Build one replay manifest from the recording's immutable run envelope."""

    recording = _validate_recording(records)
    normalized_speed = _validate_speed(speed)
    _validate_replay_run_id(replay_run_id, source_run_id=recording.started.run_id)
    return _manifest_from_recording(
        recording,
        replay_run_id=replay_run_id,
        started_at=started_at,
        speed=normalized_speed,
    )


def replay_to_repository(
    records: Sequence[RecordedEvent],
    *,
    replay_run_id: str,
    repository: ReplayRepository,
    speed: ReplaySpeed = "max",
    sleeper: Sleeper = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ReplayResult:
    """Validate, persist, run, and finalize one replay as an atomic run lifecycle."""

    recording = _validate_recording(records)
    normalized_speed = _validate_speed(speed)
    _validate_replay_run_id(replay_run_id, source_run_id=recording.started.run_id)
    _validate_source_manifest(repository.get_run_manifest(recording.started.run_id), recording)

    started_at = clock()
    manifest = _manifest_from_recording(
        recording,
        replay_run_id=replay_run_id,
        started_at=started_at,
        speed=normalized_speed,
    )
    engine = ReplayEngine(
        recording.records,
        replay_run_id=replay_run_id,
        store=repository,
        speed=normalized_speed,
        sleeper=sleeper,
    )
    repository.start_run(manifest)
    evidence = _stream_evidence(recording)
    try:
        result = engine.run()
    except BaseException as exc:
        repository.finalize_run(
            replay_run_id,
            status="failed",
            ended_at=max(started_at, clock()),
            error=f"{type(exc).__name__}: replay terminated",
            stream=evidence,
        )
        raise
    repository.finalize_run(
        replay_run_id,
        status="succeeded",
        ended_at=max(started_at, clock()),
        stream=evidence,
    )
    return result


def _validate_speed(speed: ReplaySpeed) -> ReplaySpeed:
    if speed == "max":
        return speed
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise ReplayValidationError("replay speed must be a positive number or 'max'")
    value = float(speed)
    if not math.isfinite(value) or value <= 0:
        raise ReplayValidationError("numeric replay speed must be finite and positive")
    return value


def _validate_replay_run_id(replay_run_id: str, *, source_run_id: str) -> None:
    if not replay_run_id or not replay_run_id.strip():
        raise ReplayValidationError("replay_run_id cannot be blank")
    if replay_run_id == source_run_id:
        raise ReplayValidationError("replay_run_id must differ from the source recorded run ID")


def _stream_evidence(recording: _ValidatedRecording) -> RunStreamEvidence:
    records = recording.records
    return RunStreamEvidence(
        first_event_index=records[0].event_index,
        last_event_index=records[-1].event_index,
        event_count=len(records),
        event_stream_hash=recording.stream_hash,
    )


def _manifest_from_recording(
    recording: _ValidatedRecording,
    *,
    replay_run_id: str,
    started_at: datetime,
    speed: ReplaySpeed,
) -> RunManifestRecord:
    evidence = _stream_evidence(recording)
    started = recording.started
    return RunManifestRecord(
        run_id=replay_run_id,
        run_type="replay",
        started_at=started_at,
        manifest_version=2,
        recording_format_version=2,
        event_schema_version=2,
        recording_id=started.run_id,
        source_run_id=started.run_id,
        first_event_index=evidence.first_event_index,
        last_event_index=evidence.last_event_index,
        event_count=evidence.event_count,
        event_stream_hash=evidence.event_stream_hash,
        input_payload=started.inputs.model_dump(mode="json"),
        config_hash=started.config_hash,
        metadata_hash=started.metadata_hash,
        relations_hash=started.relations_hash,
        fee_policy_hash=started.fee_policy_hash,
        metadata={
            "source_started_at": started.local_received_ts,
            "source_ended_at": recording.ended.local_received_ts,
            "source_status": recording.ended.status,
            "replay_speed": speed,
        },
    )


def _validate_source_manifest(
    source: StoredRunManifest | None,
    recording: _ValidatedRecording,
) -> None:
    """Cross-check local source provenance when the originating manifest is available."""

    if source is None:
        return
    started = recording.started
    evidence = _stream_evidence(recording)
    expected = (
        2,
        2,
        started.run_id,
        evidence.first_event_index,
        evidence.last_event_index,
        evidence.event_count,
        evidence.event_stream_hash,
        started.inputs.model_dump(mode="json"),
        started.config_hash,
        started.metadata_hash,
        started.relations_hash,
        started.fee_policy_hash,
    )
    actual = (
        source.manifest_version,
        source.recording_format_version,
        source.recording_id,
        source.first_event_index,
        source.last_event_index,
        source.event_count,
        source.event_stream_hash,
        source.input_payload,
        source.config_hash,
        source.metadata_hash,
        source.relations_hash,
        source.fee_policy_hash,
    )
    if actual != expected:
        raise ReplayValidationError(
            "recorded stream conflicts with its locally persisted source manifest"
        )


def _validate_recording(records: Sequence[RecordedEvent]) -> _ValidatedRecording:
    if not records:
        raise ReplayValidationError("recording is empty")
    ordered = tuple(sorted(records, key=lambda record: record.event_index))
    _validate_order(ordered)

    starts = tuple(record for record in ordered if isinstance(record, RunStartedEvent))
    ends = tuple(record for record in ordered if isinstance(record, RunEndedEvent))
    if len(starts) != 1 or not isinstance(ordered[0], RunStartedEvent):
        raise ReplayValidationError(
            "recording must contain exactly one RunStartedEvent as its first record"
        )
    if len(ends) != 1 or not isinstance(ordered[-1], RunEndedEvent):
        raise ReplayValidationError(
            "recording must contain exactly one RunEndedEvent as its final record"
        )
    started = starts[0]
    ended = ends[0]
    if ended.run_id != started.run_id:
        raise ReplayValidationError("run-start and run-end IDs do not match")

    expected_hashes = started.inputs.computed_hashes()
    for field_name, expected_hash in expected_hashes.items():
        if getattr(started, field_name) != expected_hash:
            raise ReplayValidationError(
                f"run-start {field_name} does not match its embedded input payload"
            )
    configuration = _validate_configuration(started.inputs)
    return _ValidatedRecording(
        records=ordered,
        started=started,
        ended=ended,
        configuration=configuration,
        stream_hash=recorded_event_stream_hash(ordered),
    )


def _validate_order(ordered: Sequence[RecordedEvent]) -> None:
    """Validate the authoritative global index/time order without assuming run bounds."""

    indices = tuple(record.event_index for record in ordered)
    if len(indices) != len(set(indices)):
        raise ReplayValidationError("recording contains duplicate event_index values")
    expected = tuple(range(indices[0], indices[-1] + 1))
    if indices != expected:
        raise ReplayValidationError("recording event_index values are not contiguous")
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.local_received_ts < previous.local_received_ts:
            raise ReplayValidationError(
                "recording local_received_ts moves backwards in event_index order"
            )


def _validate_configuration(inputs: RunInputPayload) -> ReplayConfiguration:
    try:
        configuration = ReplayConfiguration.model_validate(inputs.config)
    except ValidationError as exc:
        raise ReplayValidationError(
            "run-start configuration cannot reconstruct replay settings"
        ) from exc
    fee_policy: Mapping[str, object] = inputs.fee_policy
    if fee_policy.get("policy_version") != KALSHI_FEE_POLICY_VERSION:
        raise ReplayValidationError("run-start fee-policy version is unsupported")
    return configuration
