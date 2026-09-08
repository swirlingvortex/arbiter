"""Trusted-component routing and fail-closed scan integration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from arbiter.engine.scanner import (
    ArbitrageEngine,
    ComponentScanner,
    ComponentScanResult,
    OpportunityLifecycle,
    ScanStatus,
)
from arbiter.engine.state import EngineState
from arbiter.models.event import Event
from arbiter.models.market import Market, PriceRange
from arbiter.models.opportunity import (
    FeeValidationStatus,
    MarketResearchContext,
    OpportunityObservation,
    OpportunityStage,
)
from arbiter.models.orderbook import PriceLevel
from arbiter.models.relation import LogicalComponent, Relation, RelationType
from arbiter.models.series import Series
from arbiter.replay.events import OrderBookDeltaEvent, OrderBookSnapshotEvent
from arbiter.solver.fees import ZeroFeeModel

NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)


def _market(ticker: str, event_ticker: str, series_ticker: str | None) -> Market:
    return Market(
        ticker=ticker,
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        title=ticker,
        status="active",
        price_ranges=(
            PriceRange(
                start=Decimal("0.0000"),
                end=Decimal("1.0000"),
                step=Decimal("0.0100"),
            ),
        ),
        raw={},
    )


def _event(
    ticker: str,
    series_ticker: str,
    markets: tuple[str, ...],
    *,
    category: str | None = None,
) -> Event:
    return Event(
        ticker=ticker,
        series_ticker=series_ticker,
        title=ticker,
        category=category,
        market_tickers=tuple(sorted(markets)),
        raw={},
    )


def _series(ticker: str, *, category: str | None = None) -> Series:
    return Series(
        ticker=ticker,
        title=ticker,
        category=category,
        fee_type="quadratic",
        fee_multiplier=Decimal("1"),
        raw={},
    )


def _relation(
    relation_id: str,
    antecedent: str,
    consequent: str,
    *,
    verified: bool = True,
) -> Relation:
    return Relation(
        relation_id=relation_id,
        market_tickers=(antecedent, consequent),
        relation_type=RelationType.IMPLIES,
        source="manual",
        verified=verified,
        rationale="fixture implication",
        created_at=NOW,
        antecedent=antecedent,
        consequent=consequent,
    )


def _snapshot(
    ticker: str,
    event_index: int,
    sequence: int,
    *,
    yes_bid: str,
    no_bid: str,
    members: tuple[str, ...],
) -> OrderBookSnapshotEvent:
    del members
    return OrderBookSnapshotEvent(
        event_index=event_index,
        local_received_ts=NOW,
        ticker=ticker,
        sequence=sequence,
        sid=7,
        connection_id="connection-1",
        snapshot_id=f"snapshot:{ticker}:{sequence}",
        yes_bids=(PriceLevel(price=Decimal(yes_bid), quantity=Decimal("5.00")),),
        no_bids=(PriceLevel(price=Decimal(no_bid), quantity=Decimal("5.00")),),
    )


def _state(
    *,
    second_component: bool = False,
    include_unverified: bool = False,
    max_component_markets: int = 12,
    event_series_fallback: bool = False,
    event_category: str | None = None,
    series_category: str | None = None,
) -> EngineState:
    groups: list[tuple[str, str, str, str]] = [("A", "M", "EVENT-1", "SERIES-1")]
    relations = [_relation("m-implies-a", "M", "A")]
    if second_component:
        groups.append(("X", "Y", "EVENT-2", "SERIES-2"))
        relations.append(_relation("x-implies-y", "X", "Y"))
    if include_unverified:
        groups.append(("U", "V", "EVENT-3", "SERIES-3"))
        relations.append(_relation("u-implies-v", "U", "V", verified=False))

    markets = tuple(
        _market(
            ticker,
            event_ticker,
            None if event_series_fallback and ticker == "A" else series_ticker,
        )
        for first, second, event_ticker, series_ticker in groups
        for ticker in (first, second)
    ) + (_market("Z", "EVENT-Z", "SERIES-Z"),)
    events = tuple(
        _event(
            event_ticker,
            series_ticker,
            (first, second),
            category=event_category,
        )
        for first, second, event_ticker, series_ticker in groups
    ) + (_event("EVENT-Z", "SERIES-Z", ("Z",), category=event_category),)
    series = tuple(_series(group[3], category=series_category) for group in groups) + (
        _series("SERIES-Z", category=series_category),
    )
    return EngineState(
        markets=markets,
        events=events,
        series=series,
        relations=tuple(relations),
        max_component_markets=max_component_markets,
    )


def _load_books(
    state: EngineState,
    quotes: dict[str, tuple[str, str]],
) -> None:
    members = tuple(sorted(quotes))
    state.register_subscription(connection_id="connection-1", sid=7, tickers=members)
    for index, ticker in enumerate(members):
        yes_bid, no_bid = quotes[ticker]
        state.apply_event(
            _snapshot(
                ticker,
                index,
                index + 1,
                yes_bid=yes_bid,
                no_bid=no_bid,
                members=members,
            )
        )


def _scanner(state: EngineState) -> ComponentScanner:
    clock = iter((1.0, 1.002))
    return ComponentScanner(
        state,
        stale_after=timedelta(seconds=2),
        fee_model=ZeroFeeModel(),
        minimum_net_profit=Decimal("0"),
        minimum_net_edge_bps=Decimal("0"),
        monotonic=lambda: next(clock),
    )


def test_state_subscribes_only_to_verified_component_markets_and_caches_worlds() -> None:
    state = _state(second_component=True, include_unverified=True)

    assert state.subscription_tickers == ("A", "M", "X", "Y")
    assert len(state.graph.components) == 2
    assert set(state.world_sets) == {component.component_id for component in state.graph.components}
    first = state.graph.components[0]
    assert state.world_for_component(first) is state.world_for_component(first.component_id)
    assert state.components_for_markets(("A", "Z")) == tuple(
        component for component in state.graph.components if "A" in component.market_tickers
    )


def test_canonical_component_produces_depth_and_fee_verified_opportunity() -> None:
    state = _state()
    _load_books(state, {"A": ("0.60", "0.38"), "M": ("0.70", "0.28")})
    component = state.graph.components[0]

    result = _scanner(state).scan_component(
        component,
        as_of=NOW,
        trigger_event_index=1,
    )

    assert result.status is ScanStatus.OPPORTUNITY
    assert result.opportunity is not None
    assert result.opportunity.stage is OpportunityStage.NET_EXECUTABLE
    assert result.opportunity.fee_status is FeeValidationStatus.APPLIED
    assert result.opportunity.fees == Decimal("0")
    assert result.opportunity.gross_profit == Decimal("0.40")
    assert result.num_states == 3
    assert result.num_instruments == 4
    assert result.num_legs == 2
    assert result.relation_sources == ("manual",)
    assert result.solve_duration_ms == Decimal("2.0")
    assert result.reference_violation is True
    assert result.one_contract_gross_survived is True
    assert result.one_contract_fee_survived is True
    assert result.depth_executable is True


def test_missing_or_stale_book_skips_before_solver_and_cannot_claim_opportunity() -> None:
    state = _state()
    component = state.graph.components[0]

    missing = _scanner(state).scan_component(
        component,
        as_of=NOW,
        trigger_event_index=0,
    )

    assert missing.status is ScanStatus.SKIPPED
    assert missing.solver_status == "missing_book"
    assert missing.num_instruments == 0
    assert missing.opportunity is None

    _load_books(state, {"A": ("0.60", "0.38"), "M": ("0.70", "0.28")})
    stale = _scanner(state).scan_component(
        component,
        as_of=NOW + timedelta(seconds=3),
        trigger_event_index=2,
    )
    assert stale.status is ScanStatus.SKIPPED
    assert stale.solver_status == "stale_book"
    assert stale.opportunity is None


def test_non_open_market_skips_before_solver_even_with_fresh_books() -> None:
    state = _state()
    _load_books(state, {"A": ("0.60", "0.38"), "M": ("0.70", "0.28")})
    state.replace_market(state.markets["A"].model_copy(update={"status": "closed"}))

    result = _scanner(state).scan_component(
        state.graph.components[0],
        as_of=NOW,
        trigger_event_index=2,
    )

    assert result.status is ScanStatus.SKIPPED
    assert result.solver_status == "market_not_open"
    assert result.opportunity is None
    assert result.num_instruments == 0


def test_fee_uncertainty_preserves_gross_evidence_but_blocks_net_stage() -> None:
    state = _state()
    _load_books(state, {"A": ("0.60", "0.38"), "M": ("0.70", "0.28")})
    state.mark_event_fee_uncertain("EVENT-1", reason="fee update pending")

    result = _scanner(state).scan_component(
        state.graph.components[0],
        as_of=NOW,
        trigger_event_index=1,
    )

    assert result.opportunity is not None
    assert result.opportunity.stage is OpportunityStage.GROSS_EXECUTABLE
    assert result.opportunity.fee_status is FeeValidationStatus.UNSUPPORTED
    assert result.opportunity.reason == "fee_metadata_uncertain"
    assert result.one_contract_gross_survived is True
    assert result.one_contract_fee_survived is False


def test_event_series_ancestry_is_sufficient_for_fee_validation() -> None:
    state = _state(event_series_fallback=True)
    _load_books(state, {"A": ("0.60", "0.38"), "M": ("0.70", "0.28")})

    result = _scanner(state).scan_component(
        state.graph.components[0],
        as_of=NOW,
        trigger_event_index=1,
    )

    assert result.opportunity is not None
    assert result.opportunity.stage is OpportunityStage.NET_EXECUTABLE
    assert result.opportunity.fee_status is FeeValidationStatus.APPLIED


def test_scan_snapshots_category_fallback_and_settlement_precedence() -> None:
    state = _state(event_category=None, series_category="Series fallback")
    settlement_fields = (
        "settlement_ts",
        "expected_expiration_time",
        "expiration_time",
        "latest_expiration_time",
        "close_time",
    )
    settlement_values = tuple(
        NOW + timedelta(hours=offset) for offset in range(1, len(settlement_fields) + 1)
    )
    for chosen_index, chosen_field in enumerate(settlement_fields):
        updates: dict[str, datetime | None] = {field: None for field in settlement_fields}
        updates.update(
            {
                field: settlement_values[index]
                for index, field in enumerate(settlement_fields)
                if index >= chosen_index
            }
        )
        state.replace_market(state.markets["A"].model_copy(update=updates))
        context = state.market_research_contexts(("A",))[0]
        assert context.settlement_at == updates[chosen_field]
        assert context.category == "Series fallback"

    event = state.events["EVENT-1"].model_copy(update={"category": "Event primary"})
    state.replace_event(event)
    state.replace_market(
        state.markets["A"].model_copy(
            update={
                "settlement_ts": settlement_values[0],
                "expected_expiration_time": settlement_values[1],
            }
        )
    )
    state.replace_market(
        state.markets["M"].model_copy(update={"close_time": settlement_values[-1]})
    )
    _load_books(state, {"A": ("0.60", "0.38"), "M": ("0.70", "0.28")})

    result = _scanner(state).scan_component(
        state.graph.components[0],
        as_of=NOW,
        trigger_event_index=1,
    )

    assert result.market_contexts == (
        MarketResearchContext(
            ticker="A",
            event_ticker="EVENT-1",
            category="Event primary",
            settlement_at=settlement_values[0],
        ),
        MarketResearchContext(
            ticker="M",
            event_ticker="EVENT-1",
            category="Event primary",
            settlement_at=settlement_values[-1],
        ),
    )
    assert result.relation_sources == ("manual",)


def test_reference_midpoints_can_emit_stage_zero_without_executable_claim() -> None:
    state = _state()
    # Midpoints are A=.60 and M=.70 (incoherent for M=>A), while the real
    # executable A-YES + M-NO asks total 1.08 and provide no gross arbitrage.
    _load_books(state, {"A": ("0.51", "0.31"), "M": ("0.61", "0.21")})

    result = _scanner(state).scan_component(
        state.graph.components[0],
        as_of=NOW,
        trigger_event_index=1,
    )

    assert result.status is ScanStatus.OPPORTUNITY
    assert result.opportunity is not None
    assert result.opportunity.stage is OpportunityStage.LOGICAL
    assert result.opportunity.quantities == ()
    assert result.reason == "reference_price_incoherence"
    assert result.reference_violation is True
    assert result.one_contract_gross_survived is False
    assert result.one_contract_fee_survived is False
    assert result.depth_executable is False


def test_executable_signal_without_complete_midpoints_is_reported_separately() -> None:
    state = _state()
    state.register_subscription(connection_id="connection-1", sid=7, tickers=("A", "M"))
    state.apply_event(
        OrderBookSnapshotEvent(
            event_index=0,
            local_received_ts=NOW,
            ticker="A",
            sequence=1,
            sid=7,
            connection_id="connection-1",
            snapshot_id="snapshot:A:1",
            yes_bids=(),
            no_bids=(PriceLevel(price=Decimal("0.38"), quantity=Decimal("5.00")),),
        )
    )
    state.apply_event(
        _snapshot(
            "M",
            1,
            2,
            yes_bid="0.70",
            no_bid="0.28",
            members=("A", "M"),
        )
    )

    result = _scanner(state).scan_component(
        state.graph.components[0],
        as_of=NOW,
        trigger_event_index=2,
    )

    assert result.status is ScanStatus.OPPORTUNITY
    assert result.reference_violation is None
    assert result.one_contract_gross_survived is True
    assert result.one_contract_fee_survived is True
    assert result.depth_executable is True

    class Store:
        def persist_transition(self, observation: OpportunityObservation) -> None:
            pass

    observation = OpportunityLifecycle(run_id="run", store=Store()).handle_scan(result)
    assert observation.evidence["midpoint_available"] is False
    assert "reference_violation" not in observation.evidence


def test_oversized_component_is_explicitly_skipped() -> None:
    state = _state(max_component_markets=1)

    result = _scanner(state).scan_component(
        state.graph.components[0],
        as_of=NOW,
        trigger_event_index=0,
    )

    assert result.status is ScanStatus.SKIPPED
    assert result.solver_status == "oversized"
    assert "configured maximum" in (result.reason or "")


def test_engine_debounces_only_the_component_affected_by_rapid_updates() -> None:
    state = _state(second_component=True)
    _load_books(
        state,
        {
            "A": ("0.60", "0.38"),
            "M": ("0.70", "0.28"),
            "X": ("0.49", "0.49"),
            "Y": ("0.49", "0.49"),
        },
    )

    class RecordingScanner(ComponentScanner):
        def __init__(self) -> None:
            super().__init__(
                state,
                stale_after=timedelta(seconds=5),
                fee_model=ZeroFeeModel(),
                minimum_net_profit=Decimal("0"),
                minimum_net_edge_bps=Decimal("0"),
            )
            self.calls: list[str] = []

        def scan_component(
            self,
            component: LogicalComponent,
            *,
            as_of: datetime,
            trigger_event_index: int,
        ) -> ComponentScanResult:
            self.calls.append(component.component_id)
            return super().scan_component(
                component,
                as_of=as_of,
                trigger_event_index=trigger_event_index,
            )

    class Store:
        def __init__(self) -> None:
            self.observations: list[OpportunityObservation] = []

        def persist_transition(self, observation: OpportunityObservation) -> None:
            self.observations.append(observation)

    scanner = RecordingScanner()
    engine = ArbitrageEngine(
        state=state,
        scanner=scanner,
        lifecycle=OpportunityLifecycle(run_id="routing-run", store=Store()),
        solve_debounce=timedelta(milliseconds=25),
    )
    changed_at = NOW + timedelta(seconds=1)
    first = OrderBookDeltaEvent(
        event_index=4,
        local_received_ts=changed_at,
        exchange_ts=changed_at,
        ticker="A",
        sequence=5,
        sid=7,
        connection_id="connection-1",
        snapshot_id="snapshot:A:1",
        side="no",
        price=Decimal("0.38"),
        quantity_delta=Decimal("1.00"),
    )
    second = first.model_copy(
        update={
            "event_index": 5,
            "sequence": 6,
            "local_received_ts": changed_at + timedelta(milliseconds=10),
            "exchange_ts": changed_at + timedelta(milliseconds=10),
        }
    )

    engine.process_event(first)
    engine.process_event(second)
    engine.advance_time(changed_at + timedelta(milliseconds=34))
    assert scanner.calls == []
    engine.advance_time(changed_at + timedelta(milliseconds=35))

    affected = state.components_for_markets(("A",))[0]
    unrelated = state.components_for_markets(("X",))[0]
    assert scanner.calls == [affected.component_id]
    assert unrelated.component_id not in scanner.calls
