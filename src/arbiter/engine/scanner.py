"""Pure affected-component scans over trusted worlds and current executable books."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from arbiter.engine.state import EngineState, StateAction, StateUpdate
from arbiter.models.opportunity import (
    EffectiveFeePolicy,
    FeeValidationStatus,
    MarketResearchContext,
    Opportunity,
    OpportunityEpisode,
    OpportunityObservation,
    OpportunityStage,
    OpportunityTransition,
    PortfolioLegSnapshot,
    fresh_yes_midpoint,
    opportunity_episode_id,
    opportunity_observation_id,
    portfolio_signature,
)
from arbiter.models.portfolio import Instrument, SolverResult
from arbiter.models.relation import LogicalComponent, RelationType
from arbiter.models.world import WorldSet
from arbiter.replay.events import (
    BookStaleEvent,
    ConnectionInterruptedEvent,
    FeeRefreshAppliedEvent,
    FeeRefreshStartedEvent,
    MarketDataEvent,
    MarketRefreshAppliedEvent,
    MarketRefreshStartedEvent,
    OrderBookDeltaEvent,
    OrderBookSnapshotEvent,
    RecordedEvent,
    RunEndedEvent,
    RunStartedEvent,
    SubscriptionStartedEvent,
)
from arbiter.solver.fees import (
    DIRECT_ACCOUNT_PRECISION,
    FeeModel,
    FeePolicyResolutionError,
    resolve_fee_policy,
    validate_solver_fees,
)
from arbiter.solver.instruments import build_executable_instruments
from arbiter.solver.lp import DEFAULT_NUMERIC_TOLERANCE, solve_worst_case_profit


class ScanStatus(StrEnum):
    """Whether one deterministic component evaluation found a signal."""

    OPPORTUNITY = "opportunity"
    NOT_PRESENT = "not_present"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class ComponentScanResult:
    """Complete diagnostics for one component evaluation at one event time."""

    component_id: str
    market_tickers: tuple[str, ...]
    relation_types: tuple[RelationType, ...]
    relation_sources: tuple[str, ...]
    market_contexts: tuple[MarketResearchContext, ...]
    as_of: datetime
    trigger_event_index: int
    status: ScanStatus
    opportunity: Opportunity | None
    solver_status: str
    reason: str | None
    solve_duration_ms: Decimal
    num_states: int
    num_instruments: int
    num_legs: int
    reference_violation: bool | None = None
    one_contract_gross_survived: bool | None = None
    one_contract_fee_survived: bool | None = None
    depth_executable: bool | None = None

    def __post_init__(self) -> None:
        if not self.component_id or not self.component_id.strip():
            raise ValueError("component_id cannot be blank")
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("component scan time must be timezone-aware")
        if self.trigger_event_index < 0:
            raise ValueError("trigger_event_index must be nonnegative")
        if self.solve_duration_ms < 0 or not self.solve_duration_ms.is_finite():
            raise ValueError("solve_duration_ms must be finite and nonnegative")
        if min(self.num_states, self.num_instruments, self.num_legs) < 0:
            raise ValueError("component scan counts must be nonnegative")
        if self.status is ScanStatus.OPPORTUNITY and self.opportunity is None:
            raise ValueError("opportunity scan status requires opportunity evidence")
        if self.status is not ScanStatus.OPPORTUNITY and self.opportunity is not None:
            raise ValueError("non-opportunity scan status cannot contain an opportunity")
        if self.opportunity is not None and self.num_legs != len(self.opportunity.quantities):
            raise ValueError("num_legs must match opportunity allocations")
        if (
            not self.relation_sources
            or self.relation_sources != tuple(sorted(set(self.relation_sources)))
            or any(not source.strip() for source in self.relation_sources)
        ):
            raise ValueError("scan relation sources must be nonempty and canonically sorted")
        if tuple(context.ticker for context in self.market_contexts) != self.market_tickers:
            raise ValueError("scan market contexts must match component markets")
        diagnostics = (
            self.one_contract_gross_survived,
            self.one_contract_fee_survived,
            self.depth_executable,
        )
        if self.status is ScanStatus.SKIPPED:
            if self.reference_violation is not None or any(
                value is not None for value in diagnostics
            ):
                raise ValueError("skipped scans cannot claim completed funnel diagnostics")
        elif any(value is None for value in diagnostics):
            raise ValueError("completed scans require executable funnel diagnostics")
        if self.one_contract_fee_survived and not self.one_contract_gross_survived:
            raise ValueError("fee survival requires one-contract gross survival")
        if self.depth_executable and (
            self.opportunity is None or self.opportunity.stage is OpportunityStage.LOGICAL
        ):
            raise ValueError("depth execution requires an executable opportunity stage")


class ComponentScanner:
    """Evaluate one trusted component without owning scheduling or persistence."""

    def __init__(
        self,
        state: EngineState,
        *,
        stale_after: timedelta,
        fee_model: FeeModel | None = None,
        account_precision: Decimal = DIRECT_ACCOUNT_PRECISION,
        minimum_net_profit: Decimal = Decimal("0.01"),
        minimum_net_edge_bps: Decimal = Decimal("1"),
        minimum_gross_profit: Decimal = Decimal("0"),
        numeric_tolerance: Decimal = DEFAULT_NUMERIC_TOLERANCE,
        quantity_quantum: Decimal = Decimal("0.01"),
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        if stale_after < timedelta(0):
            raise ValueError("stale_after must be nonnegative")
        self.state = state
        self.stale_after = stale_after
        self.fee_model = fee_model
        self.account_precision = account_precision
        self.minimum_net_profit = minimum_net_profit
        self.minimum_net_edge_bps = minimum_net_edge_bps
        self.minimum_gross_profit = minimum_gross_profit
        self.numeric_tolerance = numeric_tolerance
        self.quantity_quantum = quantity_quantum
        self._monotonic = monotonic

    def scan_component(
        self,
        component: LogicalComponent,
        *,
        as_of: datetime,
        trigger_event_index: int,
    ) -> ComponentScanResult:
        """Build real instruments, solve once, and classify all evidence fail-closed."""

        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("scan as_of must be timezone-aware")
        if trigger_event_index < 0:
            raise ValueError("trigger_event_index must be nonnegative")
        started = self._monotonic()
        world_result = self.state.world_for_component(component)
        relation_types = tuple(
            sorted(
                {relation.relation_type for relation in component.relations},
                key=lambda item: item.value,
            )
        )
        if world_result.status != "ok":
            return self._result(
                component,
                relation_types=relation_types,
                as_of=as_of,
                trigger_event_index=trigger_event_index,
                started=started,
                status=ScanStatus.SKIPPED,
                opportunity=None,
                solver_status=world_result.status,
                reason=world_result.reason,
                num_states=0,
                num_instruments=0,
            )
        assert world_result.world_set is not None
        world_set = world_result.world_set
        non_scannable = tuple(
            ticker
            for ticker in component.market_tickers
            if self.state.markets[ticker].status.casefold() not in {"active", "open"}
        )
        if non_scannable:
            ticker = non_scannable[0]
            return self._result(
                component,
                relation_types=relation_types,
                as_of=as_of,
                trigger_event_index=trigger_event_index,
                started=started,
                status=ScanStatus.SKIPPED,
                opportunity=None,
                solver_status="market_not_open",
                reason=f"{ticker}: market status is not open",
                num_states=int(world_set.states.shape[0]),
                num_instruments=0,
            )
        built = build_executable_instruments(
            component,
            self.state.orderbooks,
            as_of=as_of,
            stale_after=self.stale_after,
        )
        if built.status != "ok":
            detail = built.reason or built.status
            return self._result(
                component,
                relation_types=relation_types,
                as_of=as_of,
                trigger_event_index=trigger_event_index,
                started=started,
                status=ScanStatus.SKIPPED,
                opportunity=None,
                solver_status=built.status,
                reason=(
                    f"{built.market_ticker}: {detail}"
                    if built.market_ticker is not None
                    else detail
                ),
                num_states=int(world_set.states.shape[0]),
                num_instruments=0,
            )

        reference_violation = self._reference_violation(
            component,
            as_of=as_of,
            world_set=world_set,
        )
        probe_result = solve_worst_case_profit(
            world_set,
            _one_contract_probe_instruments(built.instruments),
            minimum_guaranteed_profit=Decimal("0"),
            numeric_tolerance=self.numeric_tolerance,
            quantity_quantum=self.quantity_quantum,
        )
        probe_fee_survived = False
        if probe_result.is_arbitrage:
            probe_opportunity = self._fee_validate(probe_result, as_of=as_of)
            probe_fee_survived = probe_opportunity.stage in {
                OpportunityStage.NET_EXECUTABLE,
                OpportunityStage.PAPER_SURVIVED,
            }

        result = solve_worst_case_profit(
            world_set,
            built.instruments,
            minimum_guaranteed_profit=self.minimum_gross_profit,
            numeric_tolerance=self.numeric_tolerance,
            quantity_quantum=self.quantity_quantum,
        )
        opportunity: Opportunity | None
        status: ScanStatus
        reason: str | None
        if result.is_arbitrage:
            opportunity = self._fee_validate(result, as_of=as_of)
            status = ScanStatus.OPPORTUNITY
            reason = opportunity.reason
        else:
            opportunity = (
                Opportunity(
                    stage=OpportunityStage.LOGICAL,
                    reason="reference_price_incoherence",
                )
                if reference_violation
                else None
            )
            status = ScanStatus.OPPORTUNITY if opportunity is not None else ScanStatus.NOT_PRESENT
            reason = opportunity.reason if opportunity is not None else _solver_reason(result)
        return self._result(
            component,
            relation_types=relation_types,
            as_of=as_of,
            trigger_event_index=trigger_event_index,
            started=started,
            status=status,
            opportunity=opportunity,
            solver_status=result.status,
            reason=reason,
            num_states=int(world_set.states.shape[0]),
            num_instruments=len(built.instruments),
            reference_violation=reference_violation,
            one_contract_gross_survived=probe_result.is_arbitrage,
            one_contract_fee_survived=probe_fee_survived,
            depth_executable=result.is_arbitrage,
        )

    def _fee_validate(self, result: SolverResult, *, as_of: datetime) -> Opportunity:
        used_tickers = tuple(
            sorted({allocation.instrument.ticker for allocation in result.quantities})
        )
        if any(not self.state.is_fee_ready_for_market(ticker) for ticker in used_tickers):
            return _unsupported_fee_opportunity(result, reason="fee_metadata_uncertain")
        policies: dict[str, EffectiveFeePolicy] = {}
        try:
            for ticker in used_tickers:
                event, series = self.state.fee_metadata_for_market(ticker)
                policies[ticker] = resolve_fee_policy(
                    event=event,
                    series=series,
                    as_of=as_of,
                    market_ticker=ticker,
                )
        except (KeyError, FeePolicyResolutionError):
            return _unsupported_fee_opportunity(result, reason="unsupported_fee_model")
        return validate_solver_fees(
            result,
            policies_by_ticker=policies,
            fee_model=self.fee_model,
            account_precision=self.account_precision,
            minimum_net_profit=self.minimum_net_profit,
            minimum_net_edge_bps=self.minimum_net_edge_bps,
        )

    def _reference_violation(
        self,
        component: LogicalComponent,
        *,
        as_of: datetime,
        world_set: WorldSet,
    ) -> bool | None:
        reference: list[Instrument] = []
        for ticker in component.market_tickers:
            book = self.state.orderbooks.get(ticker)
            if book is None:
                return None
            midpoint = fresh_yes_midpoint(book, as_of=as_of, stale_after=self.stale_after)
            if midpoint is None:
                return None
            reference.extend(
                (
                    Instrument(
                        ticker=ticker,
                        side="yes",
                        price=midpoint,
                        max_quantity=Decimal("1.00"),
                        source_side="no_bid",
                        source_price=Decimal("1") - midpoint,
                    ),
                    Instrument(
                        ticker=ticker,
                        side="no",
                        price=Decimal("1") - midpoint,
                        max_quantity=Decimal("1.00"),
                        source_side="yes_bid",
                        source_price=midpoint,
                    ),
                )
            )
        reference_result = solve_worst_case_profit(
            world_set,
            reference,
            minimum_guaranteed_profit=Decimal("0"),
            numeric_tolerance=self.numeric_tolerance,
            quantity_quantum=self.quantity_quantum,
        )
        return reference_result.is_arbitrage

    def _result(
        self,
        component: LogicalComponent,
        *,
        relation_types: tuple[RelationType, ...],
        as_of: datetime,
        trigger_event_index: int,
        started: float,
        status: ScanStatus,
        opportunity: Opportunity | None,
        solver_status: str,
        reason: str | None,
        num_states: int,
        num_instruments: int,
        reference_violation: bool | None = None,
        one_contract_gross_survived: bool | None = None,
        one_contract_fee_survived: bool | None = None,
        depth_executable: bool | None = None,
    ) -> ComponentScanResult:
        ended = self._monotonic()
        elapsed_ms = (Decimal(str(ended)) - Decimal(str(started))) * Decimal("1000")
        if elapsed_ms < 0:
            raise ValueError("monotonic scanner clock moved backwards")
        return ComponentScanResult(
            component_id=component.component_id,
            market_tickers=component.market_tickers,
            relation_types=relation_types,
            relation_sources=tuple(sorted({relation.source for relation in component.relations})),
            market_contexts=self.state.market_research_contexts(component.market_tickers),
            as_of=as_of,
            trigger_event_index=trigger_event_index,
            status=status,
            opportunity=opportunity,
            solver_status=solver_status,
            reason=reason,
            solve_duration_ms=elapsed_ms,
            num_states=num_states,
            num_instruments=num_instruments,
            num_legs=0 if opportunity is None else len(opportunity.quantities),
            reference_violation=reference_violation,
            one_contract_gross_survived=one_contract_gross_survived,
            one_contract_fee_survived=one_contract_fee_survived,
            depth_executable=depth_executable,
        )


class OpportunityTransitionStore(Protocol):
    """Atomic persistence boundary required before lifecycle state may advance."""

    def persist_transition(self, observation: OpportunityObservation) -> None: ...


class OpportunityLifecycle:
    """Persist then commit one deterministic active episode per component."""

    def __init__(
        self,
        *,
        run_id: str,
        store: OpportunityTransitionStore,
    ) -> None:
        if not run_id or not run_id.strip():
            raise ValueError("run_id cannot be blank")
        self.run_id = run_id
        self.store = store
        self._active: dict[str, OpportunityEpisode] = {}
        self._completed: list[OpportunityEpisode] = []
        self._censored: list[OpportunityEpisode] = []
        self._observations: list[OpportunityObservation] = []

    @property
    def active_episodes(self) -> Mapping[str, OpportunityEpisode]:
        return MappingProxyType(self._active)

    @property
    def completed_episodes(self) -> tuple[OpportunityEpisode, ...]:
        return tuple(self._completed)

    @property
    def censored_episodes(self) -> tuple[OpportunityEpisode, ...]:
        return tuple(self._censored)

    @property
    def observations(self) -> tuple[OpportunityObservation, ...]:
        return tuple(self._observations)

    def handle_scan(self, result: ComponentScanResult) -> OpportunityObservation:
        """Persist one scan decision atomically before changing active lifecycle state."""

        active = self._active.get(result.component_id)
        if result.status is ScanStatus.OPPORTUNITY:
            assert result.opportunity is not None
            transition = (
                OpportunityTransition.OPEN if active is None else OpportunityTransition.UPDATED
            )
            opportunity_id = (
                opportunity_episode_id(
                    run_id=self.run_id,
                    component_id=result.component_id,
                    opened_event_index=result.trigger_event_index,
                )
                if active is None
                else active.opportunity_id
            )
            legs = _portfolio_leg_snapshots(result.opportunity)
            signature = portfolio_signature(legs)
            observation = self._observation(
                result,
                transition=transition,
                opportunity_id=opportunity_id,
                opportunity=result.opportunity,
                portfolio_legs=legs,
                portfolio_signature_value=signature,
                close_reason=None,
            )
            candidate_episode = (
                OpportunityEpisode.from_open_observation(observation)
                if active is None
                else active.apply_observation(observation)
            )
        elif active is not None:
            close_reason = result.reason or (
                "component scan skipped"
                if result.status is ScanStatus.SKIPPED
                else "opportunity no longer present"
            )
            observation = self._observation(
                result,
                transition=OpportunityTransition.CLOSED,
                opportunity_id=active.opportunity_id,
                opportunity=None,
                portfolio_legs=(),
                portfolio_signature_value=None,
                close_reason=close_reason,
            )
            candidate_episode = active.apply_observation(observation)
        else:
            observation = self._observation(
                result,
                transition=OpportunityTransition.NOT_PRESENT,
                opportunity_id=None,
                opportunity=None,
                portfolio_legs=(),
                portfolio_signature_value=None,
                close_reason=None,
            )
            candidate_episode = None

        # This call must either commit the complete summary/observation/legs transition or
        # raise. Only a successful durable write is allowed to mutate lifecycle memory.
        self.store.persist_transition(observation)
        if observation.transition is OpportunityTransition.CLOSED:
            assert candidate_episode is not None
            self._active.pop(result.component_id, None)
            self._completed.append(candidate_episode)
        elif observation.transition in {
            OpportunityTransition.OPEN,
            OpportunityTransition.UPDATED,
        }:
            assert candidate_episode is not None
            self._active[result.component_id] = candidate_episode
        self._observations.append(observation)
        return observation

    def right_censor_all(
        self,
        observed_at: datetime,
        event_index: int,
        reason: str,
    ) -> tuple[OpportunityObservation, ...]:
        """Right-censor every active episode at a later ordered control event.

        Candidate transitions are validated in canonical component order before any write.
        Each transition is then persisted before its corresponding in-memory episode moves
        from the active collection, so a storage failure leaves that episode retryable.
        """

        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("right-censor time must be timezone-aware")
        if event_index < 0:
            raise ValueError("right-censor event_index cannot be negative")
        if not reason or not reason.strip():
            raise ValueError("right-censor reason cannot be blank")

        candidates: list[tuple[str, OpportunityObservation, OpportunityEpisode]] = []
        for component_id in sorted(self._active):
            active = self._active[component_id]
            observation = OpportunityObservation(
                observation_id=opportunity_observation_id(
                    run_id=self.run_id,
                    component_id=component_id,
                    event_index=event_index,
                    transition=OpportunityTransition.RIGHT_CENSORED,
                    opportunity_id=active.opportunity_id,
                ),
                opportunity_id=active.opportunity_id,
                run_id=self.run_id,
                component_id=component_id,
                observed_at=observed_at,
                event_index=event_index,
                transition=OpportunityTransition.RIGHT_CENSORED,
                market_tickers=active.market_tickers,
                relation_types=active.relation_types,
                relation_sources=active.relation_sources,
                market_contexts=active.market_contexts,
                opportunity=active.opportunity,
                solver_status="right_censored",
                solver_reason=None,
                solve_duration_ms=Decimal(0),
                num_states=0,
                num_instruments=len(active.portfolio_legs),
                num_legs=len(active.portfolio_legs),
                portfolio_legs=active.portfolio_legs,
                portfolio_signature=active.portfolio_signature,
                close_reason=None,
                censor_reason=reason,
                evidence={"right_censored": True},
            )
            candidates.append((component_id, observation, active.apply_observation(observation)))

        persisted: list[OpportunityObservation] = []
        for component_id, observation, candidate_episode in candidates:
            self.store.persist_transition(observation)
            current = self._active.get(component_id)
            if current is None or current.opportunity_id != candidate_episode.opportunity_id:
                raise RuntimeError("active episode changed during synchronous right-censor")
            del self._active[component_id]
            self._censored.append(candidate_episode)
            self._observations.append(observation)
            persisted.append(observation)
        return tuple(persisted)

    def _observation(
        self,
        result: ComponentScanResult,
        *,
        transition: OpportunityTransition,
        opportunity_id: str | None,
        opportunity: Opportunity | None,
        portfolio_legs: tuple[PortfolioLegSnapshot, ...],
        portfolio_signature_value: str | None,
        close_reason: str | None,
    ) -> OpportunityObservation:
        evidence: dict[str, object] = {"scan_status": result.status.value}
        if result.status is not ScanStatus.SKIPPED:
            evidence["midpoint_available"] = result.reference_violation is not None
        for key, value in (
            ("reference_violation", result.reference_violation),
            ("one_contract_gross_survived", result.one_contract_gross_survived),
            ("one_contract_fee_survived", result.one_contract_fee_survived),
            ("depth_executable", result.depth_executable),
        ):
            if value is not None:
                evidence[key] = value
        if opportunity is not None and opportunity.capital_required is not None:
            # The product specification defines executable capacity as capital
            # required by the actual depth-constrained optimum.
            evidence["capacity"] = opportunity.capital_required
        return OpportunityObservation(
            observation_id=opportunity_observation_id(
                run_id=self.run_id,
                component_id=result.component_id,
                event_index=result.trigger_event_index,
                transition=transition,
                opportunity_id=opportunity_id,
            ),
            opportunity_id=opportunity_id,
            run_id=self.run_id,
            component_id=result.component_id,
            observed_at=result.as_of,
            event_index=result.trigger_event_index,
            transition=transition,
            market_tickers=result.market_tickers,
            relation_types=result.relation_types,
            relation_sources=result.relation_sources,
            market_contexts=result.market_contexts,
            opportunity=opportunity,
            solver_status=result.solver_status,
            solver_reason=result.reason,
            solve_duration_ms=result.solve_duration_ms,
            num_states=result.num_states,
            num_instruments=result.num_instruments,
            num_legs=len(portfolio_legs),
            portfolio_legs=portfolio_legs,
            portfolio_signature=portfolio_signature_value,
            close_reason=close_reason,
            evidence=evidence,
        )


@dataclass(frozen=True, slots=True)
class EngineScanDecision:
    """One scheduled scan and its successfully persisted lifecycle transition."""

    result: ComponentScanResult
    observation: OpportunityObservation


@dataclass(frozen=True, slots=True)
class EngineProcessResult:
    """Book transition plus timers completed strictly before the triggering event."""

    state_update: StateUpdate
    scans_before_event: tuple[EngineScanDecision, ...]
    scheduled_components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EngineRecordResult:
    """One fully ordered book/control transition through the shared engine path."""

    state_update: StateUpdate
    scans_before_record: tuple[EngineScanDecision, ...]
    scans_after_record: tuple[EngineScanDecision, ...]
    scheduled_components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PendingScan:
    component: LogicalComponent
    due_at: datetime
    trigger_event_index: int


class ArbitrageEngine:
    """Shared event-time scheduler used identically by live processing and replay."""

    def __init__(
        self,
        *,
        state: EngineState,
        scanner: ComponentScanner,
        lifecycle: OpportunityLifecycle,
        solve_debounce: timedelta,
        recorded_run_id: str | None = None,
    ) -> None:
        if solve_debounce < timedelta(0):
            raise ValueError("solve_debounce must be nonnegative")
        if scanner.state is not state:
            raise ValueError("scanner and engine must share the same EngineState")
        self.state = state
        self.scanner = scanner
        self.lifecycle = lifecycle
        self.solve_debounce = solve_debounce
        self.recorded_run_id = lifecycle.run_id if recorded_run_id is None else recorded_run_id
        if not self.recorded_run_id.strip():
            raise ValueError("recorded_run_id cannot be blank")
        self._pending: dict[str, _PendingScan] = {}
        self._last_event_index: int | None = None
        self._last_event_time: datetime | None = None
        self._clock_time: datetime | None = None
        self._last_book_sources: dict[str, MarketDataEvent] = {}
        self._pending_market_refreshes: dict[str, int] = {}
        self._pending_fee_refreshes: dict[str, int] = {}
        self._run_started = False
        self._run_ended = False

    @property
    def last_event_index(self) -> int | None:
        return self._last_event_index

    @property
    def next_due_at(self) -> datetime | None:
        if not self._pending:
            return None
        return min(item.due_at for item in self._pending.values())

    def process_event(self, event: MarketDataEvent) -> EngineProcessResult:
        """Fire earlier timers, apply one event once, then trail affected components."""

        processed = self.process_record(event)
        return EngineProcessResult(
            state_update=processed.state_update,
            scans_before_event=processed.scans_before_record,
            scheduled_components=processed.scheduled_components,
        )

    def process_record(self, record: RecordedEvent) -> EngineRecordResult:
        """Apply one book/control record in its authoritative event-time total order."""

        if self._run_ended:
            raise ValueError("engine cannot process records after run end")
        if self._last_event_index is not None and record.event_index != self._last_event_index + 1:
            raise ValueError("engine event_index must be contiguous and increase strictly")
        if self._last_event_time is not None and record.local_received_ts < self._last_event_time:
            raise ValueError("engine event time cannot move backwards")
        scans_before = self._advance_time(record.local_received_ts, inclusive=False)
        scans_after: tuple[EngineScanDecision, ...] = ()
        scheduled: tuple[str, ...] = ()

        if isinstance(record, (OrderBookSnapshotEvent, OrderBookDeltaEvent)):
            state_update = self.state.apply_event(record)
            self._track_book_source(record, state_update)
            if state_update.action not in {
                StateAction.DUPLICATE_IGNORED,
                StateAction.EVENT_IGNORED,
            }:
                scheduled = self._schedule_components(
                    state_update.affected_tickers,
                    as_of=record.local_received_ts,
                    trigger_event_index=record.event_index,
                )
        elif isinstance(record, RunStartedEvent):
            if self._last_event_index is not None or self._run_started:
                raise ValueError("run-start must be the first engine record")
            if record.run_id != self.recorded_run_id:
                raise ValueError("run-start ID must match the recorded run ID")
            self._validate_run_inputs(record)
            self._run_started = True
            state_update = StateUpdate(StateAction.RUN_STARTED, (), "run inputs validated")
        elif isinstance(record, SubscriptionStartedEvent):
            if record.tickers != self.state.subscription_tickers:
                raise ValueError("subscription membership does not match trusted components")
            state_update = self.state.register_subscription(
                connection_id=record.connection_id,
                sid=record.sid,
                tickers=record.tickers,
            )
        elif isinstance(record, ConnectionInterruptedEvent):
            registered_tickers = self.state.connection_members(record.connection_id)
            expected_tickers = registered_tickers or self.state.subscription_tickers
            if record.tickers != expected_tickers:
                raise ValueError("disconnect membership contradicts recorded subscription state")
            state_update = self.state.disconnect_connection(record.connection_id)
            self._clear_book_sources(state_update.affected_tickers)
            scans_after = self._scan_components_now(
                state_update.affected_tickers,
                as_of=record.local_received_ts,
                trigger_event_index=record.event_index,
            )
        elif isinstance(record, BookStaleEvent):
            if record.ticker not in self.state.subscription_tickers:
                raise ValueError("stale control references an unknown subscription ticker")
            if self._stale_source_is_current(record):
                state_update = StateUpdate(
                    StateAction.BOOK_STALE,
                    (record.ticker,),
                    "book exceeded the configured freshness window",
                )
                scans_after = self._scan_components_now(
                    state_update.affected_tickers,
                    as_of=record.local_received_ts,
                    trigger_event_index=record.event_index,
                )
            else:
                state_update = StateUpdate(
                    StateAction.EVENT_IGNORED,
                    (record.ticker,),
                    "stale timer source was superseded",
                )
        elif isinstance(record, MarketRefreshStartedEvent):
            if record.ticker in self._pending_market_refreshes:
                raise ValueError("market already has a pending metadata refresh")
            state_update = self.state.require_market_resync(
                record.ticker,
                reason=f"Kalshi lifecycle update: {record.reason}",
            )
            self._pending_market_refreshes[record.ticker] = record.event_index
            self._clear_book_sources(state_update.affected_tickers)
            scans_after = self._scan_components_now(
                state_update.affected_tickers,
                as_of=record.local_received_ts,
                trigger_event_index=record.event_index,
            )
        elif isinstance(record, MarketRefreshAppliedEvent):
            ticker = record.market.ticker
            expected_start = self._pending_market_refreshes.get(ticker)
            if expected_start != record.refresh_started_event_index:
                raise ValueError("market refresh does not match its pending start")
            self.state.replace_market(record.market)
            del self._pending_market_refreshes[ticker]
            state_update = StateUpdate(
                StateAction.MARKET_METADATA_REPLACED,
                (ticker,),
                "complete market metadata replaced",
            )
            scheduled = self._schedule_components(
                (ticker,),
                as_of=record.local_received_ts,
                trigger_event_index=record.event_index,
            )
        elif isinstance(record, FeeRefreshStartedEvent):
            if record.event_ticker in self._pending_fee_refreshes:
                raise ValueError("event already has a pending fee refresh")
            expected_tickers = tuple(
                sorted(
                    market.ticker
                    for market in self.state.markets.values()
                    if market.event_ticker == record.event_ticker
                )
            )
            if expected_tickers != record.affected_tickers:
                raise ValueError("fee-refresh membership does not match metadata ancestry")
            update = self.state.mark_event_fee_uncertain(record.event_ticker)
            self._pending_fee_refreshes[record.event_ticker] = record.event_index
            state_update = StateUpdate(
                StateAction.FEE_METADATA_UNCERTAIN,
                update.affected_tickers,
                update.reason,
            )
            scans_after = self._scan_components_now(
                state_update.affected_tickers,
                as_of=record.local_received_ts,
                trigger_event_index=record.event_index,
            )
        elif isinstance(record, FeeRefreshAppliedEvent):
            event_ticker = record.event.ticker
            expected_start = self._pending_fee_refreshes.get(event_ticker)
            if expected_start != record.refresh_started_event_index:
                raise ValueError("fee refresh does not match its pending start")
            update = self.state.replace_event(record.event)
            del self._pending_fee_refreshes[event_ticker]
            state_update = StateUpdate(
                StateAction.FEE_METADATA_REPLACED,
                update.affected_tickers,
                update.reason,
            )
            scheduled = self._schedule_components(
                state_update.affected_tickers,
                as_of=record.local_received_ts,
                trigger_event_index=record.event_index,
            )
        elif isinstance(record, RunEndedEvent):
            if record.run_id != self.recorded_run_id:
                raise ValueError("run-end ID must match the recorded run ID")
            self._run_ended = True
            state_update = StateUpdate(StateAction.RUN_ENDED, (), f"run {record.status}")
        else:  # pragma: no cover - the discriminated union makes this unreachable
            raise TypeError(f"unsupported recorded event: {type(record).__name__}")

        self._last_event_index = record.event_index
        self._last_event_time = record.local_received_ts
        self._clock_time = record.local_received_ts
        return EngineRecordResult(
            state_update=state_update,
            scans_before_record=scans_before,
            scans_after_record=scans_after,
            scheduled_components=scheduled,
        )

    def _validate_run_inputs(self, record: RunStartedEvent) -> None:
        inputs = record.inputs
        catalogs = (
            (inputs.markets, self.state.markets),
            (inputs.events, self.state.events),
            (inputs.series, self.state.series),
            (inputs.relations, self.state.relations),
        )
        for expected, actual in catalogs:
            if tuple(expected) != tuple(actual[key] for key in sorted(actual)):
                raise ValueError("run-start inputs do not match engine state")

    def _schedule_components(
        self,
        tickers: tuple[str, ...],
        *,
        as_of: datetime,
        trigger_event_index: int,
    ) -> tuple[str, ...]:
        scheduled: list[str] = []
        due_at = as_of + self.solve_debounce
        for component in self.state.components_for_markets(tickers):
            self._pending[component.component_id] = _PendingScan(
                component=component,
                due_at=due_at,
                trigger_event_index=trigger_event_index,
            )
            scheduled.append(component.component_id)
        return tuple(scheduled)

    def _scan_components_now(
        self,
        tickers: tuple[str, ...],
        *,
        as_of: datetime,
        trigger_event_index: int,
    ) -> tuple[EngineScanDecision, ...]:
        decisions: list[EngineScanDecision] = []
        for component in self.state.components_for_markets(tickers):
            result = self.scanner.scan_component(
                component,
                as_of=as_of,
                trigger_event_index=trigger_event_index,
            )
            observation = self.lifecycle.handle_scan(result)
            self._pending.pop(component.component_id, None)
            decisions.append(EngineScanDecision(result=result, observation=observation))
        return tuple(decisions)

    def _track_book_source(self, event: MarketDataEvent, update: StateUpdate) -> None:
        if update.action in {
            StateAction.SNAPSHOT_STAGED,
            StateAction.SNAPSHOT_APPLIED,
            StateAction.DELTA_APPLIED,
            StateAction.RESYNC_COMPLETE,
        }:
            self._last_book_sources[event.ticker] = event
        elif update.action is StateAction.RESYNC_REQUIRED:
            self._clear_book_sources(update.affected_tickers)

    def _clear_book_sources(self, tickers: tuple[str, ...]) -> None:
        for ticker in tickers:
            self._last_book_sources.pop(ticker, None)

    def _stale_source_is_current(self, record: BookStaleEvent) -> bool:
        source = self._last_book_sources.get(record.ticker)
        return source is not None and (
            source.event_index == record.source_event_index
            and source.connection_id == record.source_connection_id
            and source.sid == record.source_sid
            and source.snapshot_id == record.source_snapshot_id
            and source.sequence == record.source_sequence
        )

    def advance_time(self, as_of: datetime) -> tuple[EngineScanDecision, ...]:
        """Fire every timer due at or before an explicit aware event-time clock value."""

        return self._advance_time(as_of, inclusive=True)

    def advance_before(self, as_of: datetime) -> tuple[EngineScanDecision, ...]:
        """Fire timers strictly before a following event or control at ``as_of``."""

        return self._advance_time(as_of, inclusive=False)

    def flush(self) -> tuple[EngineScanDecision, ...]:
        """Evaluate all trailing debounce timers without closing surviving episodes."""

        decisions: list[EngineScanDecision] = []
        while self._pending:
            assert self.next_due_at is not None
            decisions.extend(self.advance_time(self.next_due_at))
        return tuple(decisions)

    def _advance_time(
        self,
        as_of: datetime,
        *,
        inclusive: bool,
    ) -> tuple[EngineScanDecision, ...]:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("engine clock must be timezone-aware")
        if self._clock_time is not None and as_of < self._clock_time:
            raise ValueError("engine clock cannot move backwards")
        decisions: list[EngineScanDecision] = []
        while True:
            due = tuple(
                item
                for item in self._pending.values()
                if item.due_at < as_of or (inclusive and item.due_at == as_of)
            )
            if not due:
                break
            pending = min(due, key=lambda item: (item.due_at, item.component.component_id))
            result = self.scanner.scan_component(
                pending.component,
                as_of=pending.due_at,
                trigger_event_index=pending.trigger_event_index,
            )
            observation = self.lifecycle.handle_scan(result)
            current = self._pending.get(pending.component.component_id)
            if current != pending:
                raise RuntimeError("component timer changed during a synchronous scan")
            del self._pending[pending.component.component_id]
            decisions.append(EngineScanDecision(result=result, observation=observation))
            self._clock_time = pending.due_at
        self._clock_time = as_of
        return tuple(decisions)


def _one_contract_probe_instruments(
    instruments: tuple[Instrument, ...],
) -> tuple[Instrument, ...]:
    """Cap each synthetic `(ticker, side)` order at one contract across price levels."""

    remaining: dict[tuple[str, str], Decimal] = {}
    probe: list[Instrument] = []
    for instrument in instruments:
        key = (instrument.ticker, instrument.side)
        available = remaining.setdefault(key, Decimal("1"))
        quantity = min(instrument.max_quantity, available)
        if quantity <= 0:
            continue
        probe.append(instrument.model_copy(update={"max_quantity": quantity}))
        remaining[key] = available - quantity
    return tuple(probe)


def _unsupported_fee_opportunity(result: SolverResult, *, reason: str) -> Opportunity:
    return Opportunity(
        stage=OpportunityStage.GROSS_EXECUTABLE,
        quantities=result.quantities,
        capital_required=result.capital_required,
        gross_profit=result.guaranteed_gross_profit,
        gross_edge=result.gross_edge,
        gross_state_profits=result.state_profits,
        fee_status=FeeValidationStatus.UNSUPPORTED,
        reason=reason,
    )


def _portfolio_leg_snapshots(
    opportunity: Opportunity,
) -> tuple[PortfolioLegSnapshot, ...]:
    quoted_fees: list[tuple[str, str, Decimal, Decimal, Decimal]] = []
    for quote in opportunity.fee_quotes:
        for fill_quote in quote.fills:
            fill = fill_quote.fill
            quoted_fees.append(
                (
                    fill.ticker,
                    fill.side,
                    fill.price,
                    fill.quantity,
                    fill_quote.net_fee,
                )
            )

    legs: list[PortfolioLegSnapshot] = []
    for allocation in opportunity.quantities:
        instrument = allocation.instrument
        fee: Decimal | None = None
        for index, quoted in enumerate(quoted_fees):
            ticker, side, price, quantity, candidate_fee = quoted
            if (
                ticker == instrument.ticker
                and side == instrument.side
                and price == instrument.price
                and quantity == allocation.quantity
            ):
                fee = candidate_fee
                quoted_fees.pop(index)
                break
        legs.append(PortfolioLegSnapshot.from_allocation(allocation, fee=fee))
    if quoted_fees:
        raise ValueError("fee quotes contain fills that do not match opportunity allocations")
    return tuple(legs)


def _solver_reason(result: SolverResult) -> str | None:
    reason = result.diagnostics.get("reason")
    return str(reason) if reason is not None else None


__all__ = [
    "ArbitrageEngine",
    "ComponentScanResult",
    "ComponentScanner",
    "EngineProcessResult",
    "EngineScanDecision",
    "OpportunityLifecycle",
    "OpportunityTransitionStore",
    "ScanStatus",
]
