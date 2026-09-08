"""Bounded DuckDB retry and rollback behavior for scanner persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from arbiter.models.opportunity import (
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
from arbiter.storage.duckdb import DuckDBRepository, RunManifestRecord, StorageError

OBSERVED_AT = datetime(2026, 9, 3, 12, tzinfo=UTC)


def _start_test_run(repository: DuckDBRepository) -> None:
    repository.start_run(
        RunManifestRecord(
            run_id="run-1",
            run_type="test",
            started_at=OBSERVED_AT,
        )
    )


def _open_observation() -> OpportunityObservation:
    instrument = Instrument(
        ticker="MARKET-A",
        side="yes",
        price=Decimal("0.25"),
        max_quantity=Decimal("2"),
        source_side="no_bid",
        source_price=Decimal("0.75"),
    )
    allocation = InstrumentAllocation(
        instrument=instrument,
        quantity=Decimal("1"),
        cost=Decimal("0.25"),
    )
    opportunity = Opportunity(
        stage=OpportunityStage.GROSS_EXECUTABLE,
        quantities=(allocation,),
        capital_required=Decimal("0.25"),
        gross_profit=Decimal("0.75"),
        gross_edge=Decimal("3"),
        gross_state_profits=(Decimal("0.75"),),
    )
    leg = PortfolioLegSnapshot.from_allocation(allocation)
    opportunity_id = opportunity_episode_id(
        run_id="run-1",
        component_id="component-1",
        opened_event_index=10,
    )
    transition = OpportunityTransition.OPEN
    return OpportunityObservation(
        observation_id=opportunity_observation_id(
            run_id="run-1",
            component_id="component-1",
            event_index=10,
            transition=transition,
            opportunity_id=opportunity_id,
        ),
        opportunity_id=opportunity_id,
        run_id="run-1",
        component_id="component-1",
        observed_at=OBSERVED_AT,
        event_index=10,
        transition=transition,
        market_tickers=("MARKET-A", "MARKET-B"),
        relation_types=(RelationType.IMPLIES,),
        opportunity=opportunity,
        solver_status="optimal",
        solve_duration_ms=Decimal("1"),
        num_states=3,
        num_instruments=2,
        num_legs=1,
        portfolio_legs=(leg,),
        portfolio_signature=portfolio_signature((leg,)),
    )


class _InjectedFailureRepository(DuckDBRepository):
    def __init__(
        self,
        path: Path,
        *,
        failures: int,
        exception_type: type[Exception],
        **kwargs: object,
    ) -> None:
        super().__init__(path, **kwargs)  # type: ignore[arg-type]
        self.failures = failures
        self.exception_type = exception_type
        self.transition_attempts = 0

    def _upsert_open_opportunity(self, observation: OpportunityObservation) -> None:
        super()._upsert_open_opportunity(observation)
        self.transition_attempts += 1
        if self.transition_attempts <= self.failures:
            raise self.exception_type("synthetic DuckDB failure after partial writes")


def test_transient_transaction_failures_retry_with_bounded_backoff(tmp_path: Path) -> None:
    sleeps: list[float] = []
    with _InjectedFailureRepository(
        tmp_path / "transient.duckdb",
        failures=2,
        exception_type=duckdb.OperationalError,
        write_max_attempts=3,
        write_initial_backoff_seconds=0.1,
        write_max_backoff_seconds=0.15,
        sleep=sleeps.append,
    ) as repository:
        _start_test_run(repository)
        repository.persist_transition(_open_observation())

        assert repository.transition_attempts == 3
        assert sleeps == [0.1, 0.15]
        assert repository.table_count("opportunity_observations") == 1
        assert repository.table_count("opportunities") == 1
        assert repository.table_count("portfolio_legs") == 1


def test_persistent_transient_failure_stops_and_leaves_no_partial_transition(
    tmp_path: Path,
) -> None:
    with _InjectedFailureRepository(
        tmp_path / "persistent.duckdb",
        failures=3,
        exception_type=duckdb.TransactionException,
        write_max_attempts=3,
        write_initial_backoff_seconds=0,
        write_max_backoff_seconds=0,
        sleep=lambda _: None,
    ) as repository:
        _start_test_run(repository)
        with pytest.raises(StorageError, match="after 3 transient attempt"):
            repository.persist_transition(_open_observation())

        assert repository.transition_attempts == 3
        assert repository.table_count("opportunity_observations") == 0
        assert repository.table_count("opportunities") == 0
        assert repository.table_count("portfolio_legs") == 0


def test_constraint_failure_is_not_retried_and_rolls_back(tmp_path: Path) -> None:
    sleeps: list[float] = []
    with _InjectedFailureRepository(
        tmp_path / "constraint.duckdb",
        failures=1,
        exception_type=duckdb.ConstraintException,
        write_max_attempts=5,
        sleep=sleeps.append,
    ) as repository:
        _start_test_run(repository)
        with pytest.raises(StorageError, match="failed without retry"):
            repository.persist_transition(_open_observation())

        assert repository.transition_attempts == 1
        assert sleeps == []
        assert repository.table_count("opportunity_observations") == 0
        assert repository.table_count("opportunities") == 0
        assert repository.table_count("portfolio_legs") == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"write_max_attempts": 0}, "attempts must be positive"),
        ({"write_initial_backoff_seconds": -1}, "backoff bounds are invalid"),
        (
            {"write_initial_backoff_seconds": 2, "write_max_backoff_seconds": 1},
            "backoff bounds are invalid",
        ),
    ],
)
def test_invalid_retry_configuration_fails_before_database_open(
    tmp_path: Path,
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        DuckDBRepository(tmp_path / "invalid.duckdb", **kwargs)  # type: ignore[arg-type]
    assert not (tmp_path / "invalid.duckdb").exists()
