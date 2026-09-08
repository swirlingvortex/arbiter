"""Deterministic opportunity-level analytics for research reporting.

The reducer in this module deliberately accepts normalized, immutable rows
rather than querying DuckDB itself.  Reporting adapters may therefore obtain
the rows from DuckDB, pandas, or fixture data without changing the statistical
semantics implemented here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from enum import StrEnum
from typing import Literal

type ObservationTransition = Literal["not_present", "open", "updated", "closed", "right_censored"]
type OpportunityStage = Literal["stage_0", "stage_1", "stage_2", "stage_3"]
type PaperStatus = Literal["survived", "failed", "insufficient_future_data"]
type TerminalStatus = Literal["open", "closed", "right_censored"]

_ACTIVE_TRANSITIONS = frozenset({"open", "updated"})
_TERMINAL_TRANSITIONS = frozenset({"closed", "right_censored"})
_STAGE_RANK: dict[OpportunityStage, int] = {
    "stage_0": 0,
    "stage_1": 1,
    "stage_2": 2,
    "stage_3": 3,
}

MARKET_HOUR_DENOMINATOR_METHOD = (
    "sum of unioned fresh observation intervals per market in each deduplicated "
    "recording cohort; source-live windows are preferred, and an unclosed window "
    "ends at its last recorded update"
)
FUNNEL_DENOMINATOR_METHOD = (
    "each downstream diagnostic uses unique episodes with prior-step evidence on the same "
    "observation; unavailable midpoints and insufficient-future-data paper attempts are not "
    "failures"
)
HYPOTHETICAL_PROFIT_NOTE = (
    "Sum of one peak net guarantee per episode; episodes can overlap and are not "
    "independent realized profits."
)


class SettlementBucket(StrEnum):
    """Fixed, exhaustive time-to-settlement buckets used by reports."""

    LESS_THAN_ONE_HOUR = "<1h"
    ONE_TO_SIX_HOURS = "1–6h"
    SIX_TO_TWENTY_FOUR_HOURS = "6–24h"
    ONE_TO_SEVEN_DAYS = "1–7d"
    MORE_THAN_SEVEN_DAYS = ">7d"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RunCohortRow:
    """Minimal run-manifest projection needed to prevent replay double counting."""

    run_id: str
    run_type: str
    started_at: datetime
    ended_at: datetime | None = None
    status: str = "succeeded"
    recording_id: str | None = None
    source_run_id: str | None = None

    def __post_init__(self) -> None:
        _nonblank(self.run_id, "run_id")
        _nonblank(self.run_type, "run_type")
        _nonblank(self.status, "status")
        _aware(self.started_at, "started_at")
        if self.ended_at is not None:
            _aware(self.ended_at, "ended_at")
            if self.ended_at < self.started_at:
                raise ValueError("run ended_at cannot precede started_at")
        for label, value in (
            ("recording_id", self.recording_id),
            ("source_run_id", self.source_run_id),
        ):
            if value is not None:
                _nonblank(value, label)
        if (
            self.recording_id is not None
            and self.source_run_id is not None
            and self.recording_id != self.source_run_id
        ):
            raise ValueError("recording_id and source_run_id must identify the same cohort")

    @property
    def cohort_id(self) -> str:
        """Return the immutable source-recording identity for this run."""

        return self.source_run_id or self.recording_id or self.run_id


@dataclass(frozen=True, slots=True)
class SelectedRunCohort:
    """One economic projection and one event-time coverage source per recording."""

    cohort_id: str
    projection_run_id: str
    coverage_run_id: str
    member_run_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonblank(self.cohort_id, "cohort_id")
        _nonblank(self.projection_run_id, "projection_run_id")
        _nonblank(self.coverage_run_id, "coverage_run_id")
        if self.member_run_ids != tuple(sorted(set(self.member_run_ids))):
            raise ValueError("member_run_ids must be unique and canonically sorted")
        if self.projection_run_id not in self.member_run_ids:
            raise ValueError("projection run must belong to its cohort")
        if self.coverage_run_id not in self.member_run_ids:
            raise ValueError("coverage run must belong to its cohort")


@dataclass(frozen=True, slots=True)
class RunCohortSelection:
    """Deterministic successful-run selection for all recording cohorts."""

    cohorts: tuple[SelectedRunCohort, ...]
    omitted_run_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.cohorts != tuple(sorted(self.cohorts, key=lambda item: item.cohort_id)):
            raise ValueError("selected cohorts must be canonically sorted")
        cohort_ids = tuple(item.cohort_id for item in self.cohorts)
        if len(cohort_ids) != len(set(cohort_ids)):
            raise ValueError("selected cohort identities must be unique")
        if self.omitted_run_ids != tuple(sorted(set(self.omitted_run_ids))):
            raise ValueError("omitted_run_ids must be unique and canonically sorted")

    @property
    def projection_run_ids(self) -> tuple[str, ...]:
        """Runs whose deterministic opportunity and paper results are measured."""

        return tuple(item.projection_run_id for item in self.cohorts)

    @property
    def coverage_run_ids(self) -> tuple[str, ...]:
        """Runs whose source-event-time market coverage forms the denominator."""

        return tuple(item.coverage_run_id for item in self.cohorts)

    @property
    def excluded_projection_run_ids(self) -> tuple[str, ...]:
        """Cohort members intentionally excluded from economic numerators."""

        selected = set(self.projection_run_ids)
        return tuple(
            sorted(
                run_id
                for cohort in self.cohorts
                for run_id in cohort.member_run_ids
                if run_id not in selected
            )
        )


@dataclass(frozen=True, slots=True)
class MarketContextMetricRow:
    """Historical category and settlement context for one required market."""

    ticker: str
    event_ticker: str
    category: str | None = None
    settlement_at: datetime | None = None

    def __post_init__(self) -> None:
        _nonblank(self.ticker, "ticker")
        _nonblank(self.event_ticker, "event_ticker")
        if self.category is not None:
            _nonblank(self.category, "category")
        if self.settlement_at is not None:
            _aware(self.settlement_at, "settlement_at")


@dataclass(frozen=True, slots=True)
class ObservationMetricRow:
    """Normalized opportunity-observation row consumed by the episode reducer."""

    observation_id: str
    opportunity_id: str | None
    run_id: str
    component_id: str
    observed_at: datetime
    event_index: int
    transition: ObservationTransition
    stage: OpportunityStage | None = None
    gross_edge: Decimal | None = None
    net_edge: Decimal | None = None
    capacity: Decimal | None = None
    net_profit: Decimal | None = None
    market_tickers: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    relation_sources: tuple[str, ...] = ()
    market_contexts: tuple[MarketContextMetricRow, ...] = ()
    midpoint_evaluable: bool | None = None
    reference_violation: bool | None = None
    one_contract_gross_survived: bool | None = None
    one_contract_fee_survived: bool | None = None
    depth_executable: bool | None = None

    def __post_init__(self) -> None:
        _nonblank(self.observation_id, "observation_id")
        _nonblank(self.run_id, "run_id")
        _nonblank(self.component_id, "component_id")
        if self.opportunity_id is not None:
            _nonblank(self.opportunity_id, "opportunity_id")
        if self.event_index < 0:
            raise ValueError("event_index cannot be negative")
        _aware(self.observed_at, "observed_at")
        if self.transition not in {
            "not_present",
            "open",
            "updated",
            "closed",
            "right_censored",
        }:
            raise ValueError("unsupported opportunity transition")
        if self.transition in _ACTIVE_TRANSITIONS:
            if self.opportunity_id is None or self.stage is None:
                raise ValueError("OPEN/UPDATED rows require opportunity and stage evidence")
        elif self.transition in _TERMINAL_TRANSITIONS and self.opportunity_id is None:
            raise ValueError("terminal rows require an opportunity identity")
        if self.stage is not None and self.stage not in _STAGE_RANK:
            raise ValueError("unsupported opportunity stage")
        for label, value in (
            ("gross_edge", self.gross_edge),
            ("net_edge", self.net_edge),
            ("capacity", self.capacity),
            ("net_profit", self.net_profit),
        ):
            _finite(value, label)
        if self.capacity is not None and self.capacity < 0:
            raise ValueError("capacity cannot be negative")
        _labels(self.market_tickers, "market_tickers")
        if self.market_tickers != tuple(sorted(set(self.market_tickers))):
            raise ValueError("market_tickers must be unique and canonically sorted")
        _labels(self.relation_types, "relation_types")
        _labels(self.relation_sources, "relation_sources")
        context_tickers = tuple(context.ticker for context in self.market_contexts)
        if len(context_tickers) != len(set(context_tickers)):
            raise ValueError("market_contexts must contain at most one row per ticker")
        if any(ticker not in set(self.market_tickers) for ticker in context_tickers):
            raise ValueError("market_contexts cannot reference markets outside market_tickers")
        if self.midpoint_evaluable is True and self.reference_violation is None:
            raise ValueError("an evaluable midpoint requires a logical-violation result")
        if self.midpoint_evaluable is not True and self.reference_violation is not None:
            raise ValueError("a midpoint result requires explicit midpoint evaluability")
        if self.one_contract_fee_survived is True and self.one_contract_gross_survived is not True:
            raise ValueError("modeled-fee survival requires one-contract spread survival")


@dataclass(frozen=True, slots=True)
class PaperExecutionMetricRow:
    """Normalized terminal result for one latency-aware paper attempt."""

    attempt_id: str
    run_id: str
    opportunity_id: str
    status: PaperStatus
    simulated_locked_profit: Decimal | None = None

    def __post_init__(self) -> None:
        _nonblank(self.attempt_id, "attempt_id")
        _nonblank(self.run_id, "run_id")
        _nonblank(self.opportunity_id, "opportunity_id")
        if self.status not in {"survived", "failed", "insufficient_future_data"}:
            raise ValueError("unsupported paper-execution status")
        _finite(self.simulated_locked_profit, "simulated_locked_profit")
        if self.status == "survived":
            if self.simulated_locked_profit is None or self.simulated_locked_profit <= 0:
                raise ValueError("a surviving paper attempt requires positive locked profit")
        elif self.simulated_locked_profit is not None:
            raise ValueError("a non-surviving paper attempt cannot claim locked profit")


@dataclass(frozen=True, slots=True)
class MarketObservationWindowRow:
    """One fresh market-observation interval used for market-hour exposure."""

    observation_id: str
    run_id: str
    market_ticker: str
    opened_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None

    def __post_init__(self) -> None:
        _nonblank(self.observation_id, "observation_id")
        _nonblank(self.run_id, "run_id")
        _nonblank(self.market_ticker, "market_ticker")
        _aware(self.opened_at, "opened_at")
        _aware(self.updated_at, "updated_at")
        if self.updated_at < self.opened_at:
            raise ValueError("window updated_at cannot precede opened_at")
        if self.closed_at is not None:
            _aware(self.closed_at, "closed_at")
            if self.closed_at < self.updated_at:
                raise ValueError("window closed_at cannot precede updated_at")

    @property
    def covered_until(self) -> datetime:
        """Use a conservative lower bound when an interval lacks terminal evidence."""

        return self.closed_at or self.updated_at


@dataclass(frozen=True, slots=True)
class EpisodeFact:
    """Exactly one analytics fact per OPEN-led opportunity episode."""

    fact_id: str
    cohort_id: str
    run_id: str
    opportunity_id: str
    component_id: str
    opened_at: datetime
    terminal_status: TerminalStatus
    ended_at: datetime | None
    duration_seconds: Decimal | None
    relation_types: tuple[str, ...]
    relation_sources: tuple[str, ...]
    categories: tuple[str, ...]
    settlement_bucket: SettlementBucket
    peak_stage: OpportunityStage
    peak_gross_edge: Decimal | None
    peak_net_edge: Decimal | None
    peak_capacity: Decimal | None
    peak_net_guarantee: Decimal | None
    midpoint_evaluable: bool | None
    reference_violation: bool | None
    one_contract_gross_survived: bool | None
    one_contract_fee_survived: bool | None
    depth_executable: bool | None
    executable_without_midpoint: bool
    paper_status: PaperStatus | None
    paper_locked_profit: Decimal | None

    @property
    def is_right_censored(self) -> bool:
        """Whether duration is only a lower bound rather than a completed lifetime."""

        return self.terminal_status == "right_censored"


@dataclass(frozen=True, slots=True)
class DecimalQuantiles:
    """Decimal-safe deterministic distribution summary."""

    count: int
    median: Decimal | None
    p90: Decimal | None
    maximum: Decimal | None


@dataclass(frozen=True, slots=True)
class DiagnosticFunnel:
    """Counts with explicit conditional denominators for every evidence stage."""

    unique_episode_count: int
    midpoint_evaluable_count: int
    midpoint_unavailable_count: int
    midpoint_unknown_count: int
    logical_violation_count: int
    spread_evaluable_count: int
    gross_executable_count: int
    fee_evaluable_count: int
    fee_adjusted_executable_count: int
    depth_evaluable_count: int
    depth_executable_count: int
    net_executable_count: int
    executable_without_midpoint_count: int
    paper_eligible_count: int
    paper_evaluable_count: int
    paper_surviving_count: int
    paper_failed_count: int
    paper_unevaluable_count: int
    paper_not_evaluated_count: int
    incomplete_open_duration_count: int
    denominator_method: str = FUNNEL_DENOMINATOR_METHOD


@dataclass(frozen=True, slots=True)
class MarketHourRate:
    """A count divided by an auditable market-time exposure denominator."""

    metric: str
    count: int
    market_hours: Decimal
    rate_per_market_hour: Decimal | None
    denominator_method: str = MARKET_HOUR_DENOMINATOR_METHOD


@dataclass(frozen=True, slots=True)
class BreakdownRow:
    """One episode count and share for a named reporting dimension."""

    label: str
    episode_count: int
    share_of_episodes: Decimal | None
    rate_per_market_hour: Decimal | None


@dataclass(frozen=True, slots=True)
class Breakdown:
    """Canonical breakdown with its additivity semantics made explicit."""

    dimension: str
    rows: tuple[BreakdownRow, ...]
    additive: bool
    method: str
    rate_denominator_method: str = MARKET_HOUR_DENOMINATOR_METHOD


@dataclass(frozen=True, slots=True)
class AnalyticsSummary:
    """Complete pure-metric projection consumed by report renderers."""

    episode_count: int
    funnel: DiagnosticFunnel
    market_hour_rates: tuple[MarketHourRate, ...]
    gross_edge: DecimalQuantiles
    net_edge: DecimalQuantiles
    closed_duration_seconds: DecimalQuantiles
    right_censored_lower_bound_seconds: DecimalQuantiles
    maximum_capital: DecimalQuantiles
    guaranteed_dollars: DecimalQuantiles
    paper_locked_profit: DecimalQuantiles
    total_hypothetical_guaranteed_dollars: Decimal
    hypothetical_profit_note: str
    relation_type_breakdown: Breakdown
    relation_source_breakdown: Breakdown
    category_breakdown: Breakdown
    settlement_breakdown: Breakdown


@dataclass(frozen=True, slots=True)
class AnalyticsResult:
    """Cohort decision, episode facts, and aggregate metrics in one return value."""

    selection: RunCohortSelection
    episodes: tuple[EpisodeFact, ...]
    summary: AnalyticsSummary


def select_run_cohorts(runs: Sequence[RunCohortRow]) -> RunCohortSelection:
    """Select one reproducible projection and source coverage run per recording.

    A successful replay is the preferred normalized economic/paper projection;
    repeated replays tie-break by stable ``run_id``.  A successful non-replay
    source supplies the event-time exposure denominator whenever available.
    Cohorts without any successful run are omitted explicitly.
    """

    by_id: dict[str, RunCohortRow] = {}
    for run in runs:
        existing = by_id.get(run.run_id)
        if existing is not None and existing != run:
            raise ValueError(f"run_id {run.run_id!r} has conflicting rows")
        by_id[run.run_id] = run

    grouped: dict[str, list[RunCohortRow]] = defaultdict(list)
    for run in by_id.values():
        grouped[run.cohort_id].append(run)

    selected: list[SelectedRunCohort] = []
    omitted: list[str] = []
    for cohort_id, members in sorted(grouped.items()):
        successful = [item for item in members if item.status == "succeeded"]
        omitted.extend(item.run_id for item in members if item.status != "succeeded")
        if not successful:
            continue
        replays = sorted(
            (item for item in successful if item.run_type == "replay"),
            key=lambda item: item.run_id,
        )
        sources = sorted(
            (item for item in successful if item.run_type != "replay"),
            key=lambda item: (item.run_id != cohort_id, item.run_id),
        )
        projection = replays[0] if replays else (sources[0] if sources else successful[0])
        coverage = sources[0] if sources else projection
        selected.append(
            SelectedRunCohort(
                cohort_id=cohort_id,
                projection_run_id=projection.run_id,
                coverage_run_id=coverage.run_id,
                member_run_ids=tuple(sorted(item.run_id for item in members)),
            )
        )
    return RunCohortSelection(
        cohorts=tuple(selected),
        omitted_run_ids=tuple(sorted(omitted)),
    )


def build_episode_facts(
    observations: Sequence[ObservationMetricRow],
    paper_executions: Sequence[PaperExecutionMetricRow],
    selection: RunCohortSelection,
) -> tuple[EpisodeFact, ...]:
    """Reduce OPEN/UPDATED observations to one auditable peak fact per episode."""

    projection_to_cohort = {item.projection_run_id: item.cohort_id for item in selection.cohorts}
    selected_observations = _deduplicate_observations(
        item for item in observations if item.run_id in projection_to_cohort
    )
    selected_paper = _deduplicate_paper(
        item for item in paper_executions if item.run_id in projection_to_cohort
    )

    active: dict[tuple[str, str], list[ObservationMetricRow]] = defaultdict(list)
    terminals: dict[tuple[str, str], ObservationMetricRow] = {}
    for row in selected_observations:
        if row.opportunity_id is None:
            continue
        key = (row.run_id, row.opportunity_id)
        if row.transition in _ACTIVE_TRANSITIONS:
            active[key].append(row)
        elif row.transition in _TERMINAL_TRANSITIONS:
            existing = terminals.get(key)
            if existing is not None:
                raise ValueError(f"episode {row.opportunity_id!r} has multiple terminal rows")
            terminals[key] = row

    paper_by_episode: dict[tuple[str, str], list[PaperExecutionMetricRow]] = defaultdict(list)
    for paper_row in selected_paper:
        paper_key = (paper_row.run_id, paper_row.opportunity_id)
        paper_by_episode[paper_key].append(paper_row)

    facts: list[EpisodeFact] = []
    for key, rows in sorted(active.items()):
        run_id, opportunity_id = key
        ordered = sorted(
            rows,
            key=lambda item: (item.event_index, item.observed_at, item.observation_id),
        )
        opening_rows = [item for item in ordered if item.transition == "open"]
        if len(opening_rows) != 1 or ordered[0].transition != "open":
            raise ValueError(f"episode {opportunity_id!r} must have exactly one leading OPEN row")
        opening = opening_rows[0]
        if any(item.component_id != opening.component_id for item in ordered):
            raise ValueError(f"episode {opportunity_id!r} changes component identity")
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if (
                current.event_index <= previous.event_index
                or current.observed_at < previous.observed_at
            ):
                raise ValueError(f"episode {opportunity_id!r} observations are not monotonic")

        terminal = terminals.pop(key, None)
        terminal_status: TerminalStatus = "open"
        ended_at: datetime | None = None
        duration_seconds: Decimal | None = None
        if terminal is not None:
            if terminal.event_index <= ordered[-1].event_index:
                raise ValueError(f"episode {opportunity_id!r} terminal event is not later")
            if terminal.observed_at < ordered[-1].observed_at:
                raise ValueError(f"episode {opportunity_id!r} terminal time regresses")
            terminal_status = "closed" if terminal.transition == "closed" else "right_censored"
            ended_at = terminal.observed_at
            duration_seconds = _timedelta_seconds(ended_at - opening.observed_at)

        paper_status, paper_locked_profit = _aggregate_paper_attempts(paper_by_episode.pop(key, []))
        peak_stage = max(
            (item.stage for item in ordered if item.stage is not None),
            key=lambda stage: _STAGE_RANK[stage],
        )
        if paper_status is not None and _STAGE_RANK[peak_stage] < _STAGE_RANK["stage_2"]:
            raise ValueError("paper evidence requires a net-executable episode")
        midpoint_evaluable = _combine_diagnostic(item.midpoint_evaluable for item in ordered)
        midpoint_violation_rows = tuple(
            item
            for item in ordered
            if item.midpoint_evaluable is True and item.reference_violation is True
        )
        reference_violation = _combine_diagnostic(
            item.reference_violation for item in ordered if item.midpoint_evaluable is True
        )
        gross_surviving_rows = tuple(
            item for item in midpoint_violation_rows if item.one_contract_gross_survived is True
        )
        fee_surviving_rows = tuple(
            item for item in gross_surviving_rows if item.one_contract_fee_survived is True
        )

        facts.append(
            EpisodeFact(
                fact_id=f"{projection_to_cohort[run_id]}:{opening.component_id}:{opening.event_index}",
                cohort_id=projection_to_cohort[run_id],
                run_id=run_id,
                opportunity_id=opportunity_id,
                component_id=opening.component_id,
                opened_at=opening.observed_at,
                terminal_status=terminal_status,
                ended_at=ended_at,
                duration_seconds=duration_seconds,
                relation_types=tuple(sorted(set(opening.relation_types))),
                relation_sources=tuple(sorted(set(opening.relation_sources))),
                categories=tuple(
                    sorted(
                        {
                            context.category
                            for context in opening.market_contexts
                            if context.category is not None
                        }
                    )
                ),
                settlement_bucket=settlement_bucket_from_contexts(
                    observed_at=opening.observed_at,
                    market_tickers=opening.market_tickers,
                    market_contexts=opening.market_contexts,
                ),
                peak_stage=peak_stage,
                peak_gross_edge=_maximum(item.gross_edge for item in ordered),
                peak_net_edge=_maximum(item.net_edge for item in ordered),
                peak_capacity=_maximum(item.capacity for item in ordered),
                peak_net_guarantee=_maximum(
                    item.net_profit
                    for item in ordered
                    if item.stage is not None and _STAGE_RANK[item.stage] >= 2
                ),
                midpoint_evaluable=midpoint_evaluable,
                reference_violation=reference_violation,
                one_contract_gross_survived=_combine_diagnostic(
                    item.one_contract_gross_survived for item in midpoint_violation_rows
                ),
                one_contract_fee_survived=_combine_diagnostic(
                    item.one_contract_fee_survived for item in gross_surviving_rows
                ),
                depth_executable=_combine_diagnostic(
                    item.depth_executable for item in fee_surviving_rows
                ),
                executable_without_midpoint=any(
                    item.midpoint_evaluable is not True
                    and item.stage is not None
                    and _STAGE_RANK[item.stage] >= 1
                    for item in ordered
                ),
                paper_status=paper_status,
                paper_locked_profit=paper_locked_profit,
            )
        )

    if terminals:
        unknown = next(iter(sorted(terminals)))[1]
        raise ValueError(f"terminal row references unknown episode {unknown!r}")
    if paper_by_episode:
        unknown = next(iter(sorted(paper_by_episode)))[1]
        raise ValueError(f"paper row references unknown episode {unknown!r}")
    return tuple(sorted(facts, key=lambda item: item.fact_id))


def decimal_quantiles(values: Iterable[Decimal]) -> DecimalQuantiles:
    """Return type-7 median/p90/max without converting exact decimals to floats."""

    materialized = tuple(values)
    for value in materialized:
        _finite(value, "quantile value")
    ordered = sorted(materialized)
    if not ordered:
        return DecimalQuantiles(count=0, median=None, p90=None, maximum=None)
    return DecimalQuantiles(
        count=len(ordered),
        median=_decimal_percentile(ordered, Decimal("0.5")),
        p90=_decimal_percentile(ordered, Decimal("0.9")),
        maximum=ordered[-1],
    )


def settlement_bucket(
    *,
    observed_at: datetime,
    settlement_at: datetime | None,
) -> SettlementBucket:
    """Classify event-time distance with exact, gap-free boundary semantics."""

    _aware(observed_at, "observed_at")
    if settlement_at is None:
        return SettlementBucket.UNKNOWN
    _aware(settlement_at, "settlement_at")
    delta = settlement_at - observed_at
    if delta < timedelta(0):
        return SettlementBucket.UNKNOWN
    if delta < timedelta(hours=1):
        return SettlementBucket.LESS_THAN_ONE_HOUR
    if delta < timedelta(hours=6):
        return SettlementBucket.ONE_TO_SIX_HOURS
    if delta < timedelta(hours=24):
        return SettlementBucket.SIX_TO_TWENTY_FOUR_HOURS
    if delta <= timedelta(days=7):
        return SettlementBucket.ONE_TO_SEVEN_DAYS
    return SettlementBucket.MORE_THAN_SEVEN_DAYS


def settlement_bucket_from_contexts(
    *,
    observed_at: datetime,
    market_tickers: Sequence[str],
    market_contexts: Sequence[MarketContextMetricRow],
) -> SettlementBucket:
    """Bucket the latest required settlement only when every market is known.

    Capital may remain committed until every leg resolves.  Missing context,
    missing timing, duplicate context, an unexpected market, or any already-past
    required timestamp therefore makes the episode timing unknown.
    """

    _aware(observed_at, "observed_at")
    required = set(market_tickers)
    if not required or len(required) != len(market_tickers):
        return SettlementBucket.UNKNOWN
    contexts: dict[str, MarketContextMetricRow] = {}
    for context in market_contexts:
        if context.ticker in contexts or context.ticker not in required:
            return SettlementBucket.UNKNOWN
        contexts[context.ticker] = context
    if set(contexts) != required:
        return SettlementBucket.UNKNOWN
    settlements = tuple(context.settlement_at for context in contexts.values())
    if any(value is None or value < observed_at for value in settlements):
        return SettlementBucket.UNKNOWN
    known_settlements = tuple(value for value in settlements if value is not None)
    return settlement_bucket(
        observed_at=observed_at,
        settlement_at=max(known_settlements),
    )


def compute_market_hours(
    windows: Sequence[MarketObservationWindowRow],
    selection: RunCohortSelection,
) -> Decimal:
    """Sum unioned source-event-time fresh intervals across markets and cohorts."""

    coverage_to_cohort = {item.coverage_run_id: item.cohort_id for item in selection.cohorts}
    unique: dict[str, MarketObservationWindowRow] = {}
    for row in windows:
        if row.run_id not in coverage_to_cohort:
            continue
        existing = unique.get(row.observation_id)
        if existing is not None and existing != row:
            raise ValueError(f"observation window {row.observation_id!r} conflicts")
        unique[row.observation_id] = row

    intervals: dict[tuple[str, str], list[tuple[datetime, datetime]]] = defaultdict(list)
    for row in unique.values():
        intervals[(coverage_to_cohort[row.run_id], row.market_ticker)].append(
            (row.opened_at, row.covered_until)
        )

    seconds = Decimal("0")
    for grouped in intervals.values():
        ordered = sorted(grouped)
        current_start, current_end = ordered[0]
        for start, end in ordered[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                seconds += _timedelta_seconds(current_end - current_start)
                current_start, current_end = start, end
        seconds += _timedelta_seconds(current_end - current_start)
    with localcontext() as context:
        context.prec = 50
        return seconds / Decimal("3600")


def market_hour_rate(
    metric: str,
    count: int,
    market_hours: Decimal,
) -> MarketHourRate:
    """Build an explicit rate, leaving a zero-exposure denominator undefined."""

    _nonblank(metric, "metric")
    if count < 0:
        raise ValueError("count cannot be negative")
    _finite(market_hours, "market_hours")
    if market_hours < 0:
        raise ValueError("market_hours cannot be negative")
    rate: Decimal | None = None
    if market_hours > 0:
        with localcontext() as context:
            context.prec = 50
            rate = Decimal(count) / market_hours
    return MarketHourRate(
        metric=metric,
        count=count,
        market_hours=market_hours,
        rate_per_market_hour=rate,
    )


def summarize_analytics(
    episodes: Sequence[EpisodeFact],
    *,
    market_hours: Decimal,
) -> AnalyticsSummary:
    """Aggregate episode facts while retaining censoring and denominator honesty."""

    unique = {item.fact_id: item for item in episodes}
    if len(unique) != len(episodes):
        raise ValueError("episode facts must have unique fact_id values")
    ordered = tuple(unique[key] for key in sorted(unique))
    episode_count = len(ordered)

    midpoint_evaluable = tuple(item for item in ordered if item.midpoint_evaluable is True)
    logical_violations = tuple(
        item for item in midpoint_evaluable if item.reference_violation is True
    )
    spread_evaluable = tuple(
        item for item in logical_violations if item.one_contract_gross_survived is not None
    )
    gross_executable = tuple(
        item for item in spread_evaluable if item.one_contract_gross_survived is True
    )
    fee_evaluable = tuple(
        item for item in gross_executable if item.one_contract_fee_survived is not None
    )
    fee_executable = tuple(item for item in fee_evaluable if item.one_contract_fee_survived is True)
    depth_evaluable = tuple(item for item in fee_executable if item.depth_executable is not None)
    depth_executable = tuple(item for item in depth_evaluable if item.depth_executable is True)
    net_executable = tuple(item for item in ordered if _STAGE_RANK[item.peak_stage] >= 2)
    executable_without_midpoint = tuple(
        item for item in ordered if item.executable_without_midpoint
    )
    paper_eligible = net_executable
    paper_evaluable = tuple(
        item for item in paper_eligible if item.paper_status in {"survived", "failed"}
    )
    paper_unevaluable = tuple(
        item for item in paper_eligible if item.paper_status == "insufficient_future_data"
    )
    funnel = DiagnosticFunnel(
        unique_episode_count=episode_count,
        midpoint_evaluable_count=len(midpoint_evaluable),
        midpoint_unavailable_count=sum(item.midpoint_evaluable is False for item in ordered),
        midpoint_unknown_count=sum(item.midpoint_evaluable is None for item in ordered),
        logical_violation_count=len(logical_violations),
        spread_evaluable_count=len(spread_evaluable),
        gross_executable_count=len(gross_executable),
        fee_evaluable_count=len(fee_evaluable),
        fee_adjusted_executable_count=len(fee_executable),
        depth_evaluable_count=len(depth_evaluable),
        depth_executable_count=len(depth_executable),
        net_executable_count=len(net_executable),
        executable_without_midpoint_count=len(executable_without_midpoint),
        paper_eligible_count=len(paper_eligible),
        paper_evaluable_count=len(paper_evaluable),
        paper_surviving_count=sum(item.paper_status == "survived" for item in paper_evaluable),
        paper_failed_count=sum(item.paper_status == "failed" for item in paper_evaluable),
        paper_unevaluable_count=len(paper_unevaluable),
        paper_not_evaluated_count=sum(item.paper_status is None for item in paper_eligible),
        incomplete_open_duration_count=sum(item.terminal_status == "open" for item in ordered),
    )

    closed_durations = (
        item.duration_seconds
        for item in ordered
        if item.terminal_status == "closed" and item.duration_seconds is not None
    )
    censored_durations = (
        item.duration_seconds
        for item in ordered
        if item.terminal_status == "right_censored" and item.duration_seconds is not None
    )
    guarantees = tuple(
        item.peak_net_guarantee for item in ordered if item.peak_net_guarantee is not None
    )
    return AnalyticsSummary(
        episode_count=episode_count,
        funnel=funnel,
        market_hour_rates=tuple(
            market_hour_rate(metric, count, market_hours)
            for metric, count in (
                ("unique_episodes", episode_count),
                ("logical_violations", len(logical_violations)),
                ("gross_executable", len(gross_executable)),
                ("fee_adjusted_executable", len(fee_executable)),
                ("depth_executable", len(depth_executable)),
                ("net_executable", len(net_executable)),
                ("executable_without_midpoint", len(executable_without_midpoint)),
                ("paper_surviving", funnel.paper_surviving_count),
            )
        ),
        gross_edge=decimal_quantiles(
            item.peak_gross_edge for item in ordered if item.peak_gross_edge is not None
        ),
        net_edge=decimal_quantiles(
            item.peak_net_edge for item in ordered if item.peak_net_edge is not None
        ),
        closed_duration_seconds=decimal_quantiles(closed_durations),
        right_censored_lower_bound_seconds=decimal_quantiles(censored_durations),
        maximum_capital=decimal_quantiles(
            item.peak_capacity for item in ordered if item.peak_capacity is not None
        ),
        guaranteed_dollars=decimal_quantiles(guarantees),
        paper_locked_profit=decimal_quantiles(
            item.paper_locked_profit for item in ordered if item.paper_locked_profit is not None
        ),
        total_hypothetical_guaranteed_dollars=sum(guarantees, Decimal("0")),
        hypothetical_profit_note=HYPOTHETICAL_PROFIT_NOTE,
        relation_type_breakdown=_multi_label_breakdown(
            ordered,
            dimension="relation_type",
            labels=tuple(item.relation_types for item in ordered),
            market_hours=market_hours,
        ),
        relation_source_breakdown=_multi_label_breakdown(
            ordered,
            dimension="relation_source",
            labels=tuple(item.relation_sources for item in ordered),
            market_hours=market_hours,
        ),
        category_breakdown=_multi_label_breakdown(
            ordered,
            dimension="category",
            labels=tuple(item.categories for item in ordered),
            market_hours=market_hours,
        ),
        settlement_breakdown=_settlement_breakdown(ordered, market_hours=market_hours),
    )


def compute_analytics(
    *,
    runs: Sequence[RunCohortRow],
    observations: Sequence[ObservationMetricRow],
    paper_executions: Sequence[PaperExecutionMetricRow] = (),
    market_windows: Sequence[MarketObservationWindowRow] = (),
) -> AnalyticsResult:
    """Apply cohort selection, episode reduction, exposure, and summary in order."""

    selection = select_run_cohorts(runs)
    episodes = build_episode_facts(observations, paper_executions, selection)
    exposure = compute_market_hours(market_windows, selection)
    return AnalyticsResult(
        selection=selection,
        episodes=episodes,
        summary=summarize_analytics(episodes, market_hours=exposure),
    )


def _multi_label_breakdown(
    episodes: Sequence[EpisodeFact],
    *,
    dimension: str,
    labels: Sequence[tuple[str, ...]],
    market_hours: Decimal,
) -> Breakdown:
    counts: dict[str, int] = defaultdict(int)
    for episode_labels in labels:
        normalized = set(episode_labels) or {"unknown"}
        for label in normalized:
            counts[label] += 1
    return Breakdown(
        dimension=dimension,
        rows=tuple(
            BreakdownRow(
                label=label,
                episode_count=count,
                share_of_episodes=_share(count, len(episodes)),
                rate_per_market_hour=_rate_value(count, market_hours),
            )
            for label, count in sorted(counts.items())
        ),
        additive=False,
        method="multi-label; one episode contributes once to every distinct label",
    )


def _settlement_breakdown(
    episodes: Sequence[EpisodeFact],
    *,
    market_hours: Decimal,
) -> Breakdown:
    counts = {bucket: 0 for bucket in SettlementBucket}
    for episode in episodes:
        counts[episode.settlement_bucket] += 1
    return Breakdown(
        dimension="time_to_settlement",
        rows=tuple(
            BreakdownRow(
                label=bucket.value,
                episode_count=counts[bucket],
                share_of_episodes=_share(counts[bucket], len(episodes)),
                rate_per_market_hour=_rate_value(counts[bucket], market_hours),
            )
            for bucket in SettlementBucket
        ),
        additive=True,
        method="single fixed bucket per episode, measured at OPEN event time",
    )


def _share(count: int, total: int) -> Decimal | None:
    if total == 0:
        return None
    with localcontext() as context:
        context.prec = 50
        return Decimal(count) / Decimal(total)


def _rate_value(count: int, market_hours: Decimal) -> Decimal | None:
    if market_hours == 0:
        return None
    with localcontext() as context:
        context.prec = 50
        return Decimal(count) / market_hours


def _decimal_percentile(ordered: Sequence[Decimal], quantile: Decimal) -> Decimal:
    if not Decimal("0") <= quantile <= Decimal("1"):
        raise ValueError("quantile must be between zero and one")
    if len(ordered) == 1:
        return ordered[0]
    with localcontext() as context:
        context.prec = 50
        rank = Decimal(len(ordered) - 1) * quantile
        lower = int(rank)
        fraction = rank - Decimal(lower)
        if fraction == 0:
            return ordered[lower]
        return ordered[lower] + (ordered[lower + 1] - ordered[lower]) * fraction


def _maximum(values: Iterable[Decimal | None]) -> Decimal | None:
    present = tuple(value for value in values if value is not None)
    return max(present, default=None)


def _combine_diagnostic(values: Iterable[bool | None]) -> bool | None:
    present = tuple(value for value in values if value is not None)
    if any(present):
        return True
    if present:
        return False
    return None


def _aggregate_paper_attempts(
    attempts: Sequence[PaperExecutionMetricRow],
) -> tuple[PaperStatus | None, Decimal | None]:
    survived = tuple(item for item in attempts if item.status == "survived")
    if survived:
        locked_profits = tuple(
            item.simulated_locked_profit
            for item in survived
            if item.simulated_locked_profit is not None
        )
        return "survived", max(locked_profits)
    if any(item.status == "failed" for item in attempts):
        return "failed", None
    if attempts:
        return "insufficient_future_data", None
    return None, None


def _deduplicate_observations(
    observations: Iterable[ObservationMetricRow],
) -> tuple[ObservationMetricRow, ...]:
    unique: dict[str, ObservationMetricRow] = {}
    for row in observations:
        existing = unique.get(row.observation_id)
        if existing is not None and existing != row:
            raise ValueError(f"observation_id {row.observation_id!r} has conflicting rows")
        unique[row.observation_id] = row
    return tuple(unique[key] for key in sorted(unique))


def _deduplicate_paper(
    rows: Iterable[PaperExecutionMetricRow],
) -> tuple[PaperExecutionMetricRow, ...]:
    unique: dict[str, PaperExecutionMetricRow] = {}
    for row in rows:
        existing = unique.get(row.attempt_id)
        if existing is not None and existing != row:
            raise ValueError(f"attempt_id {row.attempt_id!r} has conflicting rows")
        unique[row.attempt_id] = row
    return tuple(unique[key] for key in sorted(unique))


def _timedelta_seconds(value: timedelta) -> Decimal:
    return Decimal(value.days * 86_400 + value.seconds) + Decimal(value.microseconds) / Decimal(
        "1000000"
    )


def _finite(value: Decimal | None, label: str) -> None:
    if value is not None and not value.is_finite():
        raise ValueError(f"{label} must be finite")


def _aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


def _nonblank(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} cannot be blank")


def _labels(values: Sequence[str], label: str) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} cannot contain blank labels")


__all__ = [
    "AnalyticsResult",
    "AnalyticsSummary",
    "Breakdown",
    "BreakdownRow",
    "DecimalQuantiles",
    "DiagnosticFunnel",
    "EpisodeFact",
    "FUNNEL_DENOMINATOR_METHOD",
    "HYPOTHETICAL_PROFIT_NOTE",
    "MARKET_HOUR_DENOMINATOR_METHOD",
    "MarketContextMetricRow",
    "MarketHourRate",
    "MarketObservationWindowRow",
    "ObservationMetricRow",
    "PaperExecutionMetricRow",
    "RunCohortRow",
    "RunCohortSelection",
    "SelectedRunCohort",
    "SettlementBucket",
    "build_episode_facts",
    "compute_analytics",
    "compute_market_hours",
    "decimal_quantiles",
    "market_hour_rate",
    "select_run_cohorts",
    "settlement_bucket",
    "settlement_bucket_from_contexts",
    "summarize_analytics",
]
