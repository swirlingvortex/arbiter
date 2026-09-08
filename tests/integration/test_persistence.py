"""Transactional scanner-run and opportunity-lifecycle persistence tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest
from tests.support.executable import book

from arbiter.engine.paper_execution import PaperExecutor
from arbiter.models.event import Event, EventFeeChange
from arbiter.models.opportunity import (
    EffectiveFeePolicy,
    FeeValidationStatus,
    MarketResearchContext,
    Opportunity,
    OpportunityObservation,
    OpportunityStage,
    OpportunityTransition,
    PortfolioLegSnapshot,
    opportunity_episode_id,
    opportunity_observation_id,
    portfolio_signature,
)
from arbiter.models.portfolio import Instrument, InstrumentAllocation
from arbiter.models.relation import RelationType
from arbiter.solver.fees import KALSHI_FEE_POLICY_VERSION, ZeroFeeModel
from arbiter.storage.duckdb import (
    DuckDBRepository,
    MarketObservationRecord,
    PaperExecutionRecord,
    RunManifestRecord,
    RunStreamEvidence,
    StorageError,
    market_observation_id,
    paper_execution_attempt_id,
)
from arbiter.storage.migrations import MIGRATIONS

START = datetime(2026, 9, 3, 12, tzinfo=UTC)

RESEARCH_CONTEXTS = (
    MarketResearchContext(
        ticker="MARKET-A",
        event_ticker="EVENT-A",
        category="Politics",
        settlement_at=START + timedelta(days=1),
    ),
    MarketResearchContext(
        ticker="MARKET-B",
        event_ticker="EVENT-A",
        category="Politics",
        settlement_at=START + timedelta(days=1),
    ),
)


def _start_test_run(repository: DuckDBRepository) -> None:
    repository.start_run(
        RunManifestRecord(
            run_id="run-1",
            run_type="test",
            started_at=START,
        )
    )


def _gross_observation(
    *,
    transition: OpportunityTransition,
    event_index: int,
    price: Decimal,
    opportunity_id: str,
) -> OpportunityObservation:
    instrument = Instrument(
        ticker="MARKET-A",
        side="yes",
        price=price,
        max_quantity=Decimal("5"),
        source_side="no_bid",
        source_price=Decimal("1") - price,
    )
    allocation = InstrumentAllocation(
        instrument=instrument,
        quantity=Decimal("1"),
        cost=price,
    )
    opportunity = Opportunity(
        stage=OpportunityStage.GROSS_EXECUTABLE,
        quantities=(allocation,),
        capital_required=price,
        gross_profit=Decimal("1") - price,
        gross_edge=(Decimal("1") - price) / price,
        gross_state_profits=(Decimal("1") - price,),
    )
    leg = PortfolioLegSnapshot.from_allocation(allocation)
    observed_at = START + timedelta(seconds=event_index - 10)
    return OpportunityObservation(
        observation_id=opportunity_observation_id(
            run_id="run-1",
            component_id="component-1",
            event_index=event_index,
            transition=transition,
            opportunity_id=opportunity_id,
        ),
        opportunity_id=opportunity_id,
        run_id="run-1",
        component_id="component-1",
        observed_at=observed_at,
        event_index=event_index,
        transition=transition,
        market_tickers=("MARKET-A", "MARKET-B"),
        relation_types=(RelationType.IMPLIES,),
        relation_sources=("manual",),
        opportunity=opportunity,
        solver_status="optimal",
        solve_duration_ms=Decimal("1.25"),
        num_states=3,
        num_instruments=4,
        num_legs=1,
        market_contexts=RESEARCH_CONTEXTS,
        portfolio_legs=(leg,),
        portfolio_signature=portfolio_signature((leg,)),
        evidence={"capacity": opportunity.capital_required, "residual": Decimal("0.0000")},
        metadata={"source": "synthetic"},
    )


def _net_observation(
    *,
    event_index: int = 10,
    opportunity_id: str,
    transition: OpportunityTransition = OpportunityTransition.OPEN,
    price: Decimal = Decimal("0.40"),
) -> OpportunityObservation:
    instrument = Instrument(
        ticker="MARKET-A",
        side="yes",
        price=price,
        max_quantity=Decimal("1"),
        source_side="no_bid",
        source_price=Decimal("1") - price,
    )
    allocation = InstrumentAllocation(
        instrument=instrument,
        quantity=Decimal("1"),
        cost=price,
    )
    fee_policy = EffectiveFeePolicy(
        market_ticker="MARKET-A",
        event_ticker="EVENT-A",
        series_ticker="SERIES-A",
        fee_type="quadratic",
        fee_multiplier=Decimal("0"),
        source="series_default",
        policy_version=KALSHI_FEE_POLICY_VERSION,
    )
    opportunity = Opportunity(
        stage=OpportunityStage.NET_EXECUTABLE,
        quantities=(allocation,),
        capital_required=price,
        gross_profit=Decimal("1") - price,
        gross_edge=(Decimal("1") - price) / price,
        gross_state_profits=(Decimal("1") - price,),
        fee_status=FeeValidationStatus.APPLIED,
        fees=Decimal("0"),
        net_profit=Decimal("1") - price,
        net_edge=(Decimal("1") - price) / price,
        net_state_profits=(Decimal("1") - price,),
        fee_policies=(fee_policy,),
    )
    leg = PortfolioLegSnapshot.from_allocation(allocation, fee=Decimal("0"))
    return OpportunityObservation(
        observation_id=opportunity_observation_id(
            run_id="run-1",
            component_id="component-1",
            event_index=event_index,
            transition=transition,
            opportunity_id=opportunity_id,
        ),
        opportunity_id=opportunity_id,
        run_id="run-1",
        component_id="component-1",
        observed_at=START + timedelta(seconds=event_index - 10),
        event_index=event_index,
        transition=transition,
        market_tickers=("MARKET-A", "MARKET-B"),
        relation_types=(RelationType.IMPLIES,),
        relation_sources=("manual",),
        opportunity=opportunity,
        solver_status="optimal",
        solve_duration_ms=Decimal("1"),
        num_states=1,
        num_instruments=1,
        num_legs=1,
        market_contexts=RESEARCH_CONTEXTS,
        portfolio_legs=(leg,),
        portfolio_signature=portfolio_signature((leg,)),
        evidence={"capacity": opportunity.capital_required},
    )


def _close_observation(opportunity_id: str) -> OpportunityObservation:
    transition = OpportunityTransition.CLOSED
    return OpportunityObservation(
        observation_id=opportunity_observation_id(
            run_id="run-1",
            component_id="component-1",
            event_index=12,
            transition=transition,
            opportunity_id=opportunity_id,
        ),
        opportunity_id=opportunity_id,
        run_id="run-1",
        component_id="component-1",
        observed_at=START + timedelta(seconds=2),
        event_index=12,
        transition=transition,
        market_tickers=("MARKET-A", "MARKET-B"),
        relation_types=(RelationType.IMPLIES,),
        relation_sources=("manual",),
        solver_status="not_profitable",
        solver_reason="minimum state profit is nonpositive",
        solve_duration_ms=Decimal("0.75"),
        num_states=3,
        num_instruments=4,
        num_legs=0,
        market_contexts=RESEARCH_CONTEXTS,
        close_reason="edge_disappeared",
    )


def _version_two_manifest(**updates: object) -> RunManifestRecord:
    input_payload: dict[str, object] = {
        "config_json": "{}",
        "markets": [],
        "events": [],
        "series": [],
        "relations": [],
        "fee_policy_json": "{}",
    }

    def digest(value: object) -> str:
        return sha256(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()

    values: dict[str, object] = {
        "run_id": "run-v2",
        "run_type": "live",
        "started_at": START,
        "manifest_version": 2,
        "recording_format_version": 2,
        "event_schema_version": 2,
        "recording_id": "recording-1",
        "input_payload": input_payload,
        "config_hash": sha256(b"{}").hexdigest(),
        "metadata_hash": digest({"markets": [], "events": [], "series": []}),
        "relations_hash": digest([]),
        "fee_policy_hash": sha256(b"{}").hexdigest(),
    }
    values.update(updates)
    return RunManifestRecord(**values)  # type: ignore[arg-type]


def test_append_only_migrations_include_research_observation_context(tmp_path: Path) -> None:
    assert [migration.version for migration in MIGRATIONS] == [1, 2, 3, 4, 5, 6, 7]
    assert MIGRATIONS[-4].name == "scanner_runs_and_observations"
    assert MIGRATIONS[-3].name == "replay_paper_evidence"
    assert MIGRATIONS[-2].name == "semantic_discovery_review"
    assert MIGRATIONS[-1].name == "research_observation_context"

    with DuckDBRepository(tmp_path / "scanner.duckdb") as repository:
        repository.migrate()
        tables = {str(row[0]) for row in repository.connection.execute("SHOW TABLES").fetchall()}
        opportunity_columns = {
            str(row[1])
            for row in repository.connection.execute(
                "PRAGMA table_info('opportunities')"
            ).fetchall()
        }
        manifest_columns = {
            str(row[1])
            for row in repository.connection.execute(
                "PRAGMA table_info('run_manifests')"
            ).fetchall()
        }
        observation_columns = {
            str(row[1])
            for row in repository.connection.execute(
                "PRAGMA table_info('opportunity_observations')"
            ).fetchall()
        }

    assert {
        "run_manifests",
        "market_observation_windows",
        "opportunity_observations",
        "opportunity_observation_legs",
        "paper_executions",
        "paper_execution_legs",
    } <= tables
    assert {
        "run_id",
        "opened_at",
        "closed_at",
        "updated_at",
        "censored_at",
        "censored_event_index",
        "censor_reason",
    } <= opportunity_columns
    assert {
        "manifest_version",
        "recording_format_version",
        "event_schema_version",
        "recording_id",
        "source_run_id",
        "first_event_index",
        "last_event_index",
        "event_count",
        "event_stream_hash",
        "input_payload_json",
    } <= manifest_columns
    assert {
        "market_tickers_json",
        "relation_types_json",
        "relation_sources_json",
        "market_contexts_json",
    } <= observation_columns


def test_migration_seven_preserves_unknown_legacy_research_context_as_null(
    tmp_path: Path,
) -> None:
    with DuckDBRepository(tmp_path / "legacy-observation.duckdb") as repository:
        repository.connection.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name VARCHAR NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        for migration in MIGRATIONS[:-1]:
            repository.connection.execute(migration.sql)
            repository.connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?)",
                (migration.version, migration.name, START),
            )
        repository.connection.execute(
            """
            INSERT INTO opportunity_observations (
                observation_id, run_id, component_id, observed_at, event_index,
                transition, solve_duration_ms, num_states, num_instruments, num_legs
            ) VALUES ('legacy-observation', 'legacy-run', 'legacy-component', ?, 1,
                      'not_present', 0, 0, 0, 0)
            """,
            (START,),
        )

        repository.migrate()

        snapshots = repository.connection.execute(
            """
            SELECT market_tickers_json, relation_types_json, relation_sources_json,
                   market_contexts_json
            FROM opportunity_observations
            WHERE observation_id = 'legacy-observation'
            """
        ).fetchone()

    assert snapshots == (None, None, None, None)


def test_run_manifest_and_market_window_are_idempotent_and_queryable(tmp_path: Path) -> None:
    manifest = RunManifestRecord(
        run_id="run-1",
        run_type="live",
        started_at=START,
        config_hash="config-hash",
        metadata_hash="metadata-hash",
        relations_hash="relations-hash",
        fee_policy_hash="fee-hash",
        metadata={"mode": "fixture", "threshold": Decimal("0.0100")},
    )
    observation_id = market_observation_id(
        run_id="run-1",
        market_ticker="MARKET-A",
        opened_event_index=10,
    )
    opened = MarketObservationRecord(
        observation_id=observation_id,
        run_id="run-1",
        market_ticker="MARKET-A",
        opened_at=START,
        updated_at=START,
        opened_event_index=10,
        last_event_index=10,
        status="fresh",
        start_sequence=50,
        end_sequence=50,
        connection_id="connection-1",
    )
    closed = MarketObservationRecord(
        observation_id=observation_id,
        run_id="run-1",
        market_ticker="MARKET-A",
        opened_at=START,
        updated_at=START + timedelta(seconds=1),
        opened_event_index=10,
        last_event_index=11,
        status="stale",
        closed_at=START + timedelta(seconds=1),
        closed_event_index=11,
        start_sequence=50,
        end_sequence=51,
        connection_id="connection-1",
        stale_reason="sequence gap",
        resync_reason="snapshot requested",
    )

    with DuckDBRepository(tmp_path / "runs.duckdb") as repository:
        repository.start_run(manifest)
        repository.start_run(manifest)
        repository.record_market_observation(opened)
        repository.record_market_observation(closed)
        repository.finalize_run(
            "run-1",
            status="succeeded",
            ended_at=START + timedelta(seconds=2),
        )
        with pytest.raises(StorageError, match="finalized run ID"):
            repository.start_run(manifest)
        stored = repository.get_run_manifest("run-1")
        window = repository.connection.execute(
            """
            SELECT status, opened_at, closed_at, opened_event_index,
                   closed_event_index, stale_reason, resync_reason
            FROM market_observation_windows
            WHERE observation_id = ?
            """,
            (observation_id,),
        ).fetchone()

        assert repository.table_count("run_manifests") == 1
        assert repository.table_count("market_observation_windows") == 1

    assert stored is not None
    assert stored.status == "succeeded"
    assert stored.ended_at == START + timedelta(seconds=2)
    assert stored.metadata == {"mode": "fixture", "threshold": "0.0100"}
    assert window == (
        "stale",
        START,
        START + timedelta(seconds=1),
        10,
        11,
        "sequence gap",
        "snapshot requested",
    )


def test_version_two_manifest_persists_inputs_provenance_and_final_stream(
    tmp_path: Path,
) -> None:
    manifest = _version_two_manifest(source_run_id="source-live-run")
    stream = RunStreamEvidence(
        first_event_index=7,
        last_event_index=9,
        event_count=3,
        event_stream_hash="a" * 64,
    )

    with DuckDBRepository(tmp_path / "manifest-v2.duckdb") as repository:
        repository.start_run(manifest)
        repository.start_run(manifest)
        repository.finalize_run(
            manifest.run_id,
            status="succeeded",
            ended_at=START + timedelta(seconds=1),
            stream=stream,
        )
        repository.finalize_run(
            manifest.run_id,
            status="succeeded",
            ended_at=START + timedelta(seconds=1),
            stream=stream,
        )
        stored = repository.get_run_manifest(manifest.run_id)

    assert stored is not None
    assert stored.manifest_version == 2
    assert stored.recording_format_version == 2
    assert stored.event_schema_version == 2
    assert stored.recording_id == "recording-1"
    assert stored.source_run_id == "source-live-run"
    assert stored.first_event_index == 7
    assert stored.last_event_index == 9
    assert stored.event_count == 3
    assert stored.event_stream_hash == "a" * 64
    assert stored.input_payload == manifest.input_payload


def test_version_two_manifest_rejects_hash_mismatch_and_invalid_stream_bounds() -> None:
    with pytest.raises(ValueError, match="config_hash does not match"):
        _version_two_manifest(config_hash="0" * 64)
    with pytest.raises(ValueError, match="contiguous stream bounds"):
        RunStreamEvidence(
            first_event_index=7,
            last_event_index=9,
            event_count=2,
            event_stream_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        RunStreamEvidence(
            first_event_index=7,
            last_event_index=9,
            event_count=3,
            event_stream_hash="not-a-hash",
        )


def test_opportunity_open_update_close_persists_atomically_and_exactly(
    tmp_path: Path,
) -> None:
    opportunity_id = opportunity_episode_id(
        run_id="run-1",
        component_id="component-1",
        opened_event_index=10,
    )
    opened = _gross_observation(
        transition=OpportunityTransition.OPEN,
        event_index=10,
        price=Decimal("0.40"),
        opportunity_id=opportunity_id,
    )
    updated = _gross_observation(
        transition=OpportunityTransition.UPDATED,
        event_index=11,
        price=Decimal("0.25"),
        opportunity_id=opportunity_id,
    )
    closed = _close_observation(opportunity_id)

    with DuckDBRepository(tmp_path / "opportunities.duckdb") as repository:
        _start_test_run(repository)
        repository.persist_transition(opened)
        repository.persist_transition(opened)
        conflicting = opened.model_copy(update={"solve_duration_ms": Decimal("9.99")})
        with pytest.raises(StorageError, match="conflicts with a different payload"):
            repository.persist_transition(conflicting)
        repository.persist_transition(updated)
        repository.persist_transition(closed)
        summary = repository.connection.execute(
            """
            SELECT run_id, component_id, detected_at, ended_at, opened_at, closed_at,
                   updated_at, stage, num_markets, num_legs, capital_required,
                   gross_profit, gross_edge, metadata_json, paper_execution_reason
            FROM opportunities
            WHERE opportunity_id = ?
            """,
            (opportunity_id,),
        ).fetchone()
        leg = repository.connection.execute(
            """
            SELECT ticker, side, price, quantity, source_side, source_price, fee
            FROM portfolio_legs
            WHERE opportunity_id = ?
            """,
            (opportunity_id,),
        ).fetchone()
        observations = repository.connection.execute(
            """
            SELECT transition, event_index, capacity, evidence_json,
                   market_tickers_json, relation_types_json, relation_sources_json,
                   market_contexts_json
            FROM opportunity_observations
            ORDER BY event_index
            """
        ).fetchall()

        assert repository.table_count("opportunities") == 1
        assert repository.table_count("portfolio_legs") == 1
        assert repository.table_count("opportunity_observations") == 3

    assert summary is not None
    assert summary[:10] == (
        "run-1",
        "component-1",
        START,
        START + timedelta(seconds=2),
        START,
        START + timedelta(seconds=2),
        START + timedelta(seconds=2),
        "stage_1",
        2,
        1,
    )
    assert summary[10:13] == (
        Decimal("0.25"),
        Decimal("0.75"),
        Decimal("3"),
    )
    assert json.loads(str(summary[13]))["portfolio_signature"] == updated.portfolio_signature
    assert summary[14] is None
    assert leg == (
        "MARKET-A",
        "yes",
        Decimal("0.25"),
        Decimal("1"),
        "no_bid",
        Decimal("0.75"),
        None,
    )
    assert [(row[0], row[1]) for row in observations] == [
        ("open", 10),
        ("updated", 11),
        ("closed", 12),
    ]
    assert observations[0][2] == Decimal("0.40")
    assert observations[1][2] == Decimal("0.25")
    assert json.loads(str(observations[0][3]))["residual"] == "0.0000"
    assert json.loads(str(observations[0][4])) == ["MARKET-A", "MARKET-B"]
    assert json.loads(str(observations[0][5])) == ["implies"]
    assert json.loads(str(observations[0][6])) == ["manual"]
    assert json.loads(str(observations[0][7])) == [
        context.model_dump(mode="json") for context in RESEARCH_CONTEXTS
    ]


def test_right_censor_is_distinct_and_observation_legs_remain_immutable(
    tmp_path: Path,
) -> None:
    opportunity_id = opportunity_episode_id(
        run_id="run-1",
        component_id="component-1",
        opened_event_index=10,
    )
    opened = _gross_observation(
        transition=OpportunityTransition.OPEN,
        event_index=10,
        price=Decimal("0.40"),
        opportunity_id=opportunity_id,
    )
    updated = _gross_observation(
        transition=OpportunityTransition.UPDATED,
        event_index=11,
        price=Decimal("0.25"),
        opportunity_id=opportunity_id,
    )
    censored_transition = OpportunityTransition.RIGHT_CENSORED
    censored = OpportunityObservation(
        observation_id=opportunity_observation_id(
            run_id="run-1",
            component_id="component-1",
            event_index=12,
            transition=censored_transition,
            opportunity_id=opportunity_id,
        ),
        opportunity_id=opportunity_id,
        run_id="run-1",
        component_id="component-1",
        observed_at=START + timedelta(seconds=2),
        event_index=12,
        transition=censored_transition,
        market_tickers=updated.market_tickers,
        relation_types=updated.relation_types,
        relation_sources=updated.relation_sources,
        market_contexts=updated.market_contexts,
        opportunity=updated.opportunity,
        solver_status="right_censored",
        solver_reason="recorded coverage ended",
        solve_duration_ms=Decimal("0"),
        num_states=updated.num_states,
        num_instruments=updated.num_instruments,
        num_legs=updated.num_legs,
        portfolio_legs=updated.portfolio_legs,
        portfolio_signature=updated.portfolio_signature,
        censor_reason="end_of_recorded_stream",
    )

    with DuckDBRepository(tmp_path / "censored.duckdb") as repository:
        _start_test_run(repository)
        repository.persist_transition(opened)
        repository.persist_transition(updated)
        repository.persist_transition(censored)
        repository.persist_transition(censored)
        summary = repository.connection.execute(
            """
            SELECT ended_at, closed_at, censored_at, censored_event_index,
                   censor_reason, updated_at
            FROM opportunities
            WHERE opportunity_id = ?
            """,
            (opportunity_id,),
        ).fetchone()
        leg_rows = repository.connection.execute(
            """
            SELECT observation_id, leg_index, price
            FROM opportunity_observation_legs
            ORDER BY observation_id, leg_index
            """
        ).fetchall()
        with pytest.raises(StorageError, match="right-censored opportunity"):
            repository.persist_transition(_close_observation(opportunity_id))

        assert repository.table_count("opportunity_observations") == 3
        assert repository.table_count("opportunity_observation_legs") == 3

    assert summary == (
        None,
        None,
        START + timedelta(seconds=2),
        12,
        "end_of_recorded_stream",
        START + timedelta(seconds=2),
    )
    assert sorted(row[2] for row in leg_rows) == [
        Decimal("0.25"),
        Decimal("0.25"),
        Decimal("0.40"),
    ]


def _paper_record(
    source: OpportunityObservation,
    *,
    available: bool = True,
    evidence: dict[str, object] | None = None,
) -> PaperExecutionRecord:
    assert source.opportunity_id is not None
    assert source.opportunity is not None
    executor = PaperExecutor(
        latency_ms=100,
        stale_after=timedelta(seconds=2),
        fee_model=ZeroFeeModel(),
    )
    request = executor.schedule(
        opportunity_id=source.opportunity_id,
        opportunity=source.opportunity,
        detected_at=source.observed_at,
    )
    execution_policy = source.opportunity.fee_policies[0].model_copy(
        update={"fee_multiplier": Decimal("2")}
    )
    result = executor.execute(
        request,
        orderbooks={
            "MARKET-A": book(
                "MARKET-A",
                no=(("0.60", "1"),) if available else (),
                local_timestamp=request.execute_at,
            )
        },
        fee_policies=(execution_policy,),
    )
    return PaperExecutionRecord(
        attempt_id=paper_execution_attempt_id(
            run_id=source.run_id,
            opportunity_id=source.opportunity_id,
            source_observation_id=source.observation_id,
            source_event_index=source.event_index,
        ),
        run_id=source.run_id,
        source_observation_id=source.observation_id,
        source_event_index=source.event_index,
        request=request,
        result=result,
        resolved_at=request.execute_at,
        evidence={} if evidence is None else evidence,
    )


def test_paper_execution_persists_exact_evidence_and_promotes_survivor(
    tmp_path: Path,
) -> None:
    opportunity_id = opportunity_episode_id(
        run_id="run-1",
        component_id="component-1",
        opened_event_index=10,
    )
    source = _net_observation(opportunity_id=opportunity_id)
    record = _paper_record(source, evidence={"mode": "fixture"})

    with DuckDBRepository(tmp_path / "paper-survived.duckdb") as repository:
        _start_test_run(repository)
        repository.persist_transition(source)
        repository.persist_paper_execution(record)
        repository.persist_paper_execution(record)
        summary = repository.connection.execute(
            """
            SELECT stage, paper_execution_status, paper_execution_reason,
                   paper_execution_attempt_id, paper_execution_source_event_index,
                   ended_at, closed_at
            FROM opportunities
            WHERE opportunity_id = ?
            """,
            (opportunity_id,),
        ).fetchone()
        attempt = repository.connection.execute(
            """
            SELECT source_observation_id, source_event_index, scheduled_at,
                   attempted_at, status, expected_profit, simulated_locked_profit,
                   expected_fee_policies_json, execution_fee_policies_json,
                   execution_fee_quotes_json, evidence_json, payload_hash
            FROM paper_executions
            WHERE attempt_id = ?
            """,
            (record.attempt_id,),
        ).fetchone()
        leg = repository.connection.execute(
            """
            SELECT leg_index, ticker, side, expected_prices_json,
                   actual_prices_json, fill_status
            FROM paper_execution_legs
            WHERE attempt_id = ?
            """,
            (record.attempt_id,),
        ).fetchone()
        conflicting = _paper_record(source, evidence={"mode": "different"})
        with pytest.raises(StorageError, match="different payload"):
            repository.persist_paper_execution(conflicting)

        assert repository.table_count("paper_executions") == 1
        assert repository.table_count("paper_execution_legs") == 1

    assert summary == (
        "stage_3",
        "survived",
        None,
        record.attempt_id,
        10,
        None,
        None,
    )
    assert attempt is not None
    assert attempt[:7] == (
        source.observation_id,
        10,
        START + timedelta(milliseconds=100),
        START + timedelta(milliseconds=100),
        "survived",
        Decimal("0.60"),
        Decimal("0.60"),
    )
    assert json.loads(str(attempt[7]))[0]["fee_multiplier"] == "0"
    assert json.loads(str(attempt[8]))[0]["fee_multiplier"] == "2"
    assert len(json.loads(str(attempt[9]))) == 1
    assert json.loads(str(attempt[10])) == {"mode": "fixture"}
    assert len(str(attempt[11])) == 64
    assert leg is not None
    assert leg[:3] == (0, "MARKET-A", "yes")
    assert json.loads(str(leg[3])) == [{"cost": "0.40", "price": "0.40", "quantity": "1"}]
    assert json.loads(str(leg[4])) == [{"cost": "0.40", "price": "0.40", "quantity": "1"}]
    assert leg[5] == "filled"


def test_failed_paper_execution_updates_status_without_fabricating_close(
    tmp_path: Path,
) -> None:
    opportunity_id = opportunity_episode_id(
        run_id="run-1",
        component_id="component-1",
        opened_event_index=10,
    )
    source = _net_observation(opportunity_id=opportunity_id)
    record = _paper_record(source, available=False)

    with DuckDBRepository(tmp_path / "paper-failed.duckdb") as repository:
        _start_test_run(repository)
        repository.persist_transition(source)
        repository.persist_paper_execution(record)
        summary = repository.connection.execute(
            """
            SELECT stage, paper_execution_status, paper_execution_reason,
                   ended_at, closed_at, censored_at
            FROM opportunities
            WHERE opportunity_id = ?
            """,
            (opportunity_id,),
        ).fetchone()

    assert summary == (
        "stage_2",
        "failed",
        "MARKET-A:yes:insufficient_liquidity_at_or_better",
        None,
        None,
        None,
    )


def test_older_latency_result_is_retained_but_does_not_overwrite_newer_summary(
    tmp_path: Path,
) -> None:
    opportunity_id = opportunity_episode_id(
        run_id="run-1",
        component_id="component-1",
        opened_event_index=10,
    )
    opened = _net_observation(opportunity_id=opportunity_id)
    updated = _net_observation(
        opportunity_id=opportunity_id,
        event_index=11,
        transition=OpportunityTransition.UPDATED,
        price=Decimal("0.30"),
    )
    old_result = _paper_record(opened)

    with DuckDBRepository(tmp_path / "paper-race.duckdb") as repository:
        _start_test_run(repository)
        repository.persist_transition(opened)
        repository.persist_transition(updated)
        repository.persist_paper_execution(old_result)
        summary = repository.connection.execute(
            """
            SELECT stage, capital_required, paper_execution_status,
                   paper_execution_attempt_id
            FROM opportunities
            WHERE opportunity_id = ?
            """,
            (opportunity_id,),
        ).fetchone()

        assert repository.table_count("paper_executions") == 1
        assert repository.table_count("paper_execution_legs") == 1

    assert summary == ("stage_2", Decimal("0.30"), "not_evaluated", None)


def test_not_present_diagnostic_does_not_create_summary(tmp_path: Path) -> None:
    transition = OpportunityTransition.NOT_PRESENT
    observation = OpportunityObservation(
        observation_id=opportunity_observation_id(
            run_id="run-1",
            component_id="component-1",
            event_index=9,
            transition=transition,
            opportunity_id=None,
        ),
        run_id="run-1",
        component_id="component-1",
        observed_at=START,
        event_index=9,
        transition=transition,
        market_tickers=("MARKET-A", "MARKET-B"),
        relation_types=(RelationType.IMPLIES,),
        solver_status="stale_book",
        solver_reason="MARKET-A is stale",
        solve_duration_ms=Decimal("0"),
        num_states=0,
        num_instruments=0,
        num_legs=0,
        evidence={"stale_reason": "sequence gap", "resync_reason": "snapshot requested"},
    )

    with DuckDBRepository(tmp_path / "not-present.duckdb") as repository:
        _start_test_run(repository)
        repository.persist_transition(observation)
        assert repository.table_count("opportunity_observations") == 1
        assert repository.table_count("opportunities") == 0
        assert repository.table_count("portfolio_legs") == 0


def test_event_fee_state_upsert_and_query_preserves_current_and_scheduled_state(
    tmp_path: Path,
) -> None:
    change = EventFeeChange(
        change_id="change-1",
        event_ticker="EVENT-A",
        series_ticker="SERIES-A",
        scheduled_ts=START + timedelta(days=1),
        fee_type_override="quadratic",
        fee_multiplier_override=Decimal("1.25"),
        raw={"fee_multiplier": "1.25"},
    )
    event = Event(
        ticker="EVENT-A",
        series_ticker="SERIES-A",
        title="Event A",
        fee_type_override="quadratic",
        fee_multiplier_override=Decimal("1.50"),
        fee_changes=(change,),
        raw={"event_ticker": "EVENT-A", "title": "Event A"},
    )

    with DuckDBRepository(
        tmp_path / "fees.duckdb",
        clock=lambda: START,
    ) as repository:
        repository.upsert_event_fee_state(event)
        state = repository.get_event_fee_state("EVENT-A")
        restored = repository.load_events()[0]

    assert state is not None
    assert state.fee_type_override == "quadratic"
    assert state.fee_multiplier_override == Decimal("1.50")
    assert state.fee_changes == (change,)
    assert state.updated_at == START
    assert restored.fee_type_override == "quadratic"
    assert restored.fee_multiplier_override == Decimal("1.50")
    assert restored.fee_changes == (change,)


def test_persistence_inputs_reject_naive_times_and_conflicting_run_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        RunManifestRecord(
            run_id="run-1",
            run_type="live",
            started_at=datetime(2026, 9, 3, 12),
        )

    with DuckDBRepository(tmp_path / "identity.duckdb") as repository:
        repository.start_run(RunManifestRecord("run-1", "live", START))
        with pytest.raises(StorageError, match="different manifest"):
            repository.start_run(RunManifestRecord("run-1", "replay", START))
