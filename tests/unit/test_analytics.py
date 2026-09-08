"""Unit tests for deterministic episode-level research analytics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from arbiter.analytics import (
    MarketContextMetricRow,
    MarketObservationWindowRow,
    ObservationMetricRow,
    PaperExecutionMetricRow,
    RunCohortRow,
    RunCohortSelection,
    SettlementBucket,
    build_episode_facts,
    compute_market_hours,
    decimal_quantiles,
    market_hour_rate,
    select_run_cohorts,
    settlement_bucket,
    settlement_bucket_from_contexts,
    summarize_analytics,
)

NOW = datetime(2026, 9, 5, 12, tzinfo=UTC)


def _context(
    ticker: str,
    *,
    category: str | None = "Sports",
    settlement_at: datetime | None = None,
) -> MarketContextMetricRow:
    return MarketContextMetricRow(
        ticker=ticker,
        event_ticker=f"E-{ticker}",
        category=category,
        settlement_at=settlement_at or NOW + timedelta(hours=2),
    )


def _run(
    run_id: str,
    run_type: str,
    *,
    recording_id: str,
    status: str = "succeeded",
    source_run_id: str | None = None,
) -> RunCohortRow:
    return RunCohortRow(
        run_id=run_id,
        run_type=run_type,
        started_at=NOW,
        ended_at=NOW + timedelta(hours=1),
        status=status,
        recording_id=recording_id,
        source_run_id=source_run_id,
    )


def _open(
    opportunity_id: str,
    event_index: int,
    *,
    run_id: str = "replay-a",
    component_id: str | None = None,
    observed_at: datetime = NOW,
    stage: str = "stage_0",
    midpoint_evaluable: bool | None = True,
    reference_violation: bool | None = True,
    gross_survived: bool | None = False,
    fee_survived: bool | None = False,
    depth_executable: bool | None = False,
    gross_edge: Decimal | None = None,
    net_edge: Decimal | None = None,
    capacity: Decimal | None = None,
    net_profit: Decimal | None = None,
    market_tickers: tuple[str, ...] = ("A",),
    relation_types: tuple[str, ...] = ("implies",),
    relation_sources: tuple[str, ...] = ("manual",),
    market_contexts: tuple[MarketContextMetricRow, ...] | None = None,
) -> ObservationMetricRow:
    if market_contexts is None:
        market_contexts = tuple(_context(ticker) for ticker in market_tickers)
    return ObservationMetricRow(
        observation_id=f"observation-{run_id}-{opportunity_id}-{event_index}",
        opportunity_id=opportunity_id,
        run_id=run_id,
        component_id=component_id or f"component-{opportunity_id}",
        observed_at=observed_at,
        event_index=event_index,
        transition="open",
        stage=stage,  # type: ignore[arg-type]
        gross_edge=gross_edge,
        net_edge=net_edge,
        capacity=capacity,
        net_profit=net_profit,
        market_tickers=market_tickers,
        relation_types=relation_types,
        relation_sources=relation_sources,
        market_contexts=market_contexts,
        midpoint_evaluable=midpoint_evaluable,
        reference_violation=reference_violation,
        one_contract_gross_survived=gross_survived,
        one_contract_fee_survived=fee_survived,
        depth_executable=depth_executable,
    )


def _terminal(
    opportunity_id: str,
    event_index: int,
    transition: str,
    *,
    observed_at: datetime,
    run_id: str = "replay-a",
    component_id: str | None = None,
) -> ObservationMetricRow:
    return ObservationMetricRow(
        observation_id=f"observation-{run_id}-{opportunity_id}-{event_index}",
        opportunity_id=opportunity_id,
        run_id=run_id,
        component_id=component_id or f"component-{opportunity_id}",
        observed_at=observed_at,
        event_index=event_index,
        transition=transition,  # type: ignore[arg-type]
    )


def _selection() -> RunCohortSelection:
    return select_run_cohorts(
        (_run("replay-a", "replay", recording_id="source", source_run_id="source"),)
    )


def test_cohort_selection_prefers_one_successful_replay_and_source_coverage() -> None:
    runs = (
        _run("replay-z", "replay", recording_id="source", source_run_id="source"),
        _run("source", "live", recording_id="source"),
        _run("replay-a", "replay", recording_id="source", source_run_id="source"),
        _run(
            "replay-0-failed",
            "replay",
            recording_id="source",
            source_run_id="source",
            status="failed",
        ),
        _run("orphan-failed", "live", recording_id="orphan-failed", status="failed"),
    )

    selected = select_run_cohorts(runs)

    assert selected.projection_run_ids == ("replay-a",)
    assert selected.coverage_run_ids == ("source",)
    assert selected.excluded_projection_run_ids == (
        "replay-0-failed",
        "replay-z",
        "source",
    )
    assert selected.omitted_run_ids == ("orphan-failed", "replay-0-failed")

    observations = tuple(
        _open(f"opportunity-{run_id}", 1, run_id=run_id)
        for run_id in ("source", "replay-a", "replay-z")
    )
    facts = build_episode_facts(observations, (), selected)
    assert [item.run_id for item in facts] == ["replay-a"]


def test_episode_reducer_uses_one_fact_open_context_peaks_and_multiple_paper_attempts() -> None:
    opening_contexts = (
        _context("A", category="Sports", settlement_at=NOW + timedelta(hours=2)),
        _context("B", category="Politics", settlement_at=NOW + timedelta(hours=8)),
    )
    opening = _open(
        "opportunity",
        1,
        component_id="component",
        stage="stage_1",
        gross_survived=True,
        gross_edge=Decimal("0.10"),
        capacity=Decimal("10"),
        market_tickers=("A", "B"),
        relation_types=("implies", "equivalent", "implies"),
        relation_sources=("semantic_verified", "manual", "manual"),
        market_contexts=opening_contexts,
    )
    update = ObservationMetricRow(
        observation_id="observation-update",
        opportunity_id="opportunity",
        run_id="replay-a",
        component_id="component",
        observed_at=NOW + timedelta(seconds=10),
        event_index=2,
        transition="updated",
        stage="stage_2",
        gross_edge=Decimal("0.30"),
        net_edge=Decimal("0.20"),
        capacity=Decimal("25"),
        net_profit=Decimal("5"),
        market_tickers=("A", "B"),
        relation_types=("mutually_exclusive",),
        relation_sources=("exchange_declared",),
        market_contexts=(
            _context("A", category="Finance", settlement_at=NOW + timedelta(days=10)),
            _context("B", category="Finance", settlement_at=NOW + timedelta(days=10)),
        ),
        midpoint_evaluable=True,
        reference_violation=True,
        one_contract_gross_survived=True,
        one_contract_fee_survived=True,
        depth_executable=True,
    )
    closed = _terminal(
        "opportunity",
        3,
        "closed",
        observed_at=NOW + timedelta(seconds=120),
        component_id="component",
    )
    paper = (
        PaperExecutionMetricRow("attempt-1", "replay-a", "opportunity", "failed"),
        PaperExecutionMetricRow("attempt-2", "replay-a", "opportunity", "insufficient_future_data"),
        PaperExecutionMetricRow(
            "attempt-3",
            "replay-a",
            "opportunity",
            "survived",
            simulated_locked_profit=Decimal("4"),
        ),
        PaperExecutionMetricRow(
            "attempt-4",
            "replay-a",
            "opportunity",
            "survived",
            simulated_locked_profit=Decimal("4.5"),
        ),
    )

    facts = build_episode_facts((opening, update, closed, update), paper, _selection())

    assert len(facts) == 1
    fact = facts[0]
    assert fact.peak_gross_edge == Decimal("0.30")
    assert fact.peak_net_edge == Decimal("0.20")
    assert fact.peak_capacity == Decimal("25")
    assert fact.peak_net_guarantee == Decimal("5")
    assert fact.duration_seconds == Decimal("120")
    assert fact.terminal_status == "closed"
    assert fact.relation_types == ("equivalent", "implies")
    assert fact.relation_sources == ("manual", "semantic_verified")
    assert fact.categories == ("Politics", "Sports")
    assert fact.settlement_bucket is SettlementBucket.SIX_TO_TWENTY_FOUR_HOURS
    assert fact.paper_status == "survived"
    assert fact.paper_locked_profit == Decimal("4.5")

    summary = summarize_analytics(facts, market_hours=Decimal("2"))
    assert summary.episode_count == 1
    assert summary.total_hypothetical_guaranteed_dollars == Decimal("5")
    assert summary.guaranteed_dollars.count == 1
    assert summary.relation_type_breakdown.additive is False
    assert sum(row.episode_count for row in summary.relation_type_breakdown.rows) == 2
    assert summary.relation_source_breakdown.additive is False
    assert sum(row.episode_count for row in summary.relation_source_breakdown.rows) == 2
    assert summary.category_breakdown.additive is False
    assert sum(row.episode_count for row in summary.category_breakdown.rows) == 2
    assert "not independent realized profits" in summary.hypothetical_profit_note


def test_funnel_is_conditional_and_keeps_no_midpoint_and_paper_censoring_separate() -> None:
    observations = (
        _open("logical-only", 1),
        _open(
            "full-funnel",
            2,
            stage="stage_2",
            gross_survived=True,
            fee_survived=True,
            depth_executable=True,
            gross_edge=Decimal("0.1"),
            net_edge=Decimal("0.08"),
            capacity=Decimal("10"),
            net_profit=Decimal("0.8"),
        ),
        _open(
            "no-midpoint",
            3,
            stage="stage_2",
            midpoint_evaluable=False,
            reference_violation=None,
            gross_survived=True,
            fee_survived=True,
            depth_executable=True,
            gross_edge=Decimal("0.2"),
            net_edge=Decimal("0.1"),
            capacity=Decimal("20"),
            net_profit=Decimal("2"),
        ),
        _open(
            "unknown-midpoint",
            4,
            stage="stage_2",
            midpoint_evaluable=None,
            reference_violation=None,
            gross_survived=True,
            fee_survived=True,
            depth_executable=True,
            gross_edge=Decimal("0.3"),
            net_edge=Decimal("0.2"),
            capacity=Decimal("30"),
            net_profit=Decimal("6"),
        ),
    )
    paper = (
        PaperExecutionMetricRow(
            "attempt-insufficient",
            "replay-a",
            "full-funnel",
            "insufficient_future_data",
        ),
        PaperExecutionMetricRow(
            "attempt-survived",
            "replay-a",
            "no-midpoint",
            "survived",
            simulated_locked_profit=Decimal("1"),
        ),
        PaperExecutionMetricRow("attempt-failed", "replay-a", "unknown-midpoint", "failed"),
    )
    facts = build_episode_facts(observations, paper, _selection())

    summary = summarize_analytics(facts, market_hours=Decimal("2"))
    funnel = summary.funnel
    assert funnel.unique_episode_count == 4
    assert funnel.midpoint_evaluable_count == 2
    assert funnel.midpoint_unavailable_count == 1
    assert funnel.midpoint_unknown_count == 1
    assert funnel.logical_violation_count == 2
    assert funnel.spread_evaluable_count == 2
    assert funnel.gross_executable_count == 1
    assert funnel.fee_evaluable_count == 1
    assert funnel.fee_adjusted_executable_count == 1
    assert funnel.depth_evaluable_count == 1
    assert funnel.depth_executable_count == 1
    assert funnel.net_executable_count == 3
    assert funnel.executable_without_midpoint_count == 2
    assert funnel.paper_eligible_count == 3
    assert funnel.paper_evaluable_count == 2
    assert funnel.paper_surviving_count == 1
    assert funnel.paper_failed_count == 1
    assert funnel.paper_unevaluable_count == 1
    assert funnel.paper_not_evaluated_count == 0
    assert funnel.incomplete_open_duration_count == 4

    rates = {item.metric: item.rate_per_market_hour for item in summary.market_hour_rates}
    assert rates == {
        "unique_episodes": Decimal("2"),
        "logical_violations": Decimal("1"),
        "gross_executable": Decimal("0.5"),
        "fee_adjusted_executable": Decimal("0.5"),
        "depth_executable": Decimal("0.5"),
        "net_executable": Decimal("1.5"),
        "executable_without_midpoint": Decimal("1"),
        "paper_surviving": Decimal("0.5"),
    }


def test_funnel_does_not_join_incompatible_evidence_from_different_updates() -> None:
    opening = _open("changing-reference", 1)
    later_execution = ObservationMetricRow(
        observation_id="observation-later-execution",
        opportunity_id="changing-reference",
        run_id="replay-a",
        component_id="component-changing-reference",
        observed_at=NOW + timedelta(seconds=1),
        event_index=2,
        transition="updated",
        stage="stage_2",
        gross_edge=Decimal("0.2"),
        net_edge=Decimal("0.1"),
        capacity=Decimal("20"),
        net_profit=Decimal("2"),
        market_tickers=("A",),
        relation_types=("implies",),
        relation_sources=("manual",),
        market_contexts=(_context("A"),),
        midpoint_evaluable=False,
        reference_violation=None,
        one_contract_gross_survived=True,
        one_contract_fee_survived=True,
        depth_executable=True,
    )

    facts = build_episode_facts((opening, later_execution), (), _selection())
    summary = summarize_analytics(facts, market_hours=Decimal("1"))

    assert summary.funnel.logical_violation_count == 1
    assert summary.funnel.spread_evaluable_count == 1
    assert summary.funnel.gross_executable_count == 0
    assert summary.funnel.fee_evaluable_count == 0
    assert summary.funnel.depth_evaluable_count == 0
    assert summary.funnel.executable_without_midpoint_count == 1


def test_closed_and_right_censored_durations_are_never_mixed() -> None:
    observations = (
        _open("closed", 1),
        _terminal("closed", 10, "closed", observed_at=NOW + timedelta(seconds=10)),
        _open("censored", 2),
        _terminal(
            "censored",
            11,
            "right_censored",
            observed_at=NOW + timedelta(seconds=20),
        ),
        _open("still-open", 3),
    )

    facts = build_episode_facts(observations, (), _selection())
    summary = summarize_analytics(facts, market_hours=Decimal("1"))

    assert summary.closed_duration_seconds.count == 1
    assert summary.closed_duration_seconds.median == Decimal("10")
    assert summary.right_censored_lower_bound_seconds.count == 1
    assert summary.right_censored_lower_bound_seconds.median == Decimal("20")
    assert summary.funnel.incomplete_open_duration_count == 1


def test_decimal_quantiles_are_exact_deterministic_and_validate_before_sorting() -> None:
    with localcontext() as context:
        context.prec = 4
        result = decimal_quantiles((Decimal("3"), Decimal("0"), Decimal("2"), Decimal("1")))

    assert result.count == 4
    assert result.median == Decimal("1.5")
    assert result.p90 == Decimal("2.7")
    assert result.maximum == Decimal("3")
    assert decimal_quantiles(()).median is None
    with pytest.raises(ValueError, match="finite"):
        decimal_quantiles((Decimal("NaN"), Decimal("1")))
    with pytest.raises(ValueError, match="finite"):
        decimal_quantiles((Decimal("Infinity"), Decimal("1")))


def test_market_hours_union_source_windows_and_ignore_repeated_replay_windows() -> None:
    selection = select_run_cohorts(
        (
            _run("source", "live", recording_id="source"),
            _run("replay-a", "replay", recording_id="source", source_run_id="source"),
        )
    )
    source_a_first = MarketObservationWindowRow(
        "source-a-first",
        "source",
        "A",
        NOW,
        NOW + timedelta(minutes=30),
        NOW + timedelta(hours=1),
    )
    windows = (
        source_a_first,
        source_a_first,
        MarketObservationWindowRow(
            "source-a-overlap",
            "source",
            "A",
            NOW + timedelta(minutes=30),
            NOW + timedelta(hours=1, minutes=30),
            NOW + timedelta(hours=2),
        ),
        MarketObservationWindowRow(
            "source-b",
            "source",
            "B",
            NOW,
            NOW + timedelta(minutes=30),
            NOW + timedelta(hours=1),
        ),
        MarketObservationWindowRow(
            "replay-ignored",
            "replay-a",
            "A",
            NOW,
            NOW + timedelta(hours=5),
            NOW + timedelta(hours=10),
        ),
    )

    hours = compute_market_hours(windows, selection)
    rate = market_hour_rate("logical_violations", 6, hours)

    assert hours == Decimal("3")
    assert rate.rate_per_market_hour == Decimal("2")
    assert "unioned fresh observation intervals" in rate.denominator_method
    assert market_hour_rate("empty", 0, Decimal("0")).rate_per_market_hour is None


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (None, SettlementBucket.UNKNOWN),
        (timedelta(microseconds=-1), SettlementBucket.UNKNOWN),
        (timedelta(0), SettlementBucket.LESS_THAN_ONE_HOUR),
        (timedelta(hours=1) - timedelta(microseconds=1), SettlementBucket.LESS_THAN_ONE_HOUR),
        (timedelta(hours=1), SettlementBucket.ONE_TO_SIX_HOURS),
        (timedelta(hours=6) - timedelta(microseconds=1), SettlementBucket.ONE_TO_SIX_HOURS),
        (timedelta(hours=6), SettlementBucket.SIX_TO_TWENTY_FOUR_HOURS),
        (
            timedelta(hours=24) - timedelta(microseconds=1),
            SettlementBucket.SIX_TO_TWENTY_FOUR_HOURS,
        ),
        (timedelta(hours=24), SettlementBucket.ONE_TO_SEVEN_DAYS),
        (timedelta(days=7), SettlementBucket.ONE_TO_SEVEN_DAYS),
        (
            timedelta(days=7) + timedelta(microseconds=1),
            SettlementBucket.MORE_THAN_SEVEN_DAYS,
        ),
    ],
)
def test_settlement_bucket_exact_boundaries(
    delta: timedelta | None,
    expected: SettlementBucket,
) -> None:
    settlement = None if delta is None else NOW + delta
    assert settlement_bucket(observed_at=NOW, settlement_at=settlement) is expected


def test_context_bucket_requires_every_market_and_uses_latest_required_settlement() -> None:
    contexts = (
        _context("A", settlement_at=NOW + timedelta(minutes=30)),
        _context("B", settlement_at=NOW + timedelta(hours=6)),
    )

    assert (
        settlement_bucket_from_contexts(
            observed_at=NOW,
            market_tickers=("A", "B"),
            market_contexts=contexts,
        )
        is SettlementBucket.SIX_TO_TWENTY_FOUR_HOURS
    )
    assert (
        settlement_bucket_from_contexts(
            observed_at=NOW,
            market_tickers=("A", "B"),
            market_contexts=contexts[:1],
        )
        is SettlementBucket.UNKNOWN
    )
    assert (
        settlement_bucket_from_contexts(
            observed_at=NOW,
            market_tickers=("A", "B"),
            market_contexts=(
                contexts[0],
                MarketContextMetricRow("B", "E-B", settlement_at=None),
            ),
        )
        is SettlementBucket.UNKNOWN
    )
    assert (
        settlement_bucket_from_contexts(
            observed_at=NOW,
            market_tickers=("A", "B"),
            market_contexts=(contexts[0], _context("B", settlement_at=NOW - timedelta(seconds=1))),
        )
        is SettlementBucket.UNKNOWN
    )
    assert (
        settlement_bucket_from_contexts(
            observed_at=NOW,
            market_tickers=("A", "B"),
            market_contexts=(contexts[0], contexts[0]),
        )
        is SettlementBucket.UNKNOWN
    )


def test_midpoint_result_requires_explicit_evaluability() -> None:
    with pytest.raises(ValueError, match="explicit midpoint evaluability"):
        _open(
            "invalid",
            1,
            midpoint_evaluable=False,
            reference_violation=False,
        )
    with pytest.raises(ValueError, match="requires a logical-violation result"):
        _open(
            "invalid",
            1,
            midpoint_evaluable=True,
            reference_violation=None,
        )
