"""Deterministic opportunity episode, observation, and leg-snapshot invariants."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from arbiter.models.opportunity import (
    Opportunity,
    OpportunityEpisode,
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

NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)
RUN_ID = "run-fixture"
COMPONENT_ID = "component:fixture"
OPEN_EVENT_INDEX = 10


def _allocation(*, quantity: str = "1.00", price: str = "0.50") -> InstrumentAllocation:
    quantity_decimal = Decimal(quantity)
    price_decimal = Decimal(price)
    return InstrumentAllocation(
        instrument=Instrument(
            ticker="M",
            side="yes",
            price=price_decimal,
            max_quantity=Decimal("10"),
            source_side="no_bid",
            source_price=Decimal("1") - price_decimal,
        ),
        quantity=quantity_decimal,
        cost=price_decimal * quantity_decimal,
    )


def _opportunity(*, quantity: str = "1.00") -> Opportunity:
    allocation = _allocation(quantity=quantity)
    capital = allocation.cost
    profit = Decimal("0.10")
    return Opportunity(
        stage=OpportunityStage.GROSS_EXECUTABLE,
        quantities=(allocation,),
        capital_required=capital,
        gross_profit=profit,
        gross_edge=profit / capital,
        gross_state_profits=(profit,),
    )


def _observation(
    transition: OpportunityTransition,
    *,
    event_index: int,
    observed_at: datetime,
    opportunity: Opportunity | None = None,
    opportunity_id: str | None = None,
    close_reason: str | None = None,
    censor_reason: str | None = None,
) -> OpportunityObservation:
    if opportunity_id is None and transition is not OpportunityTransition.NOT_PRESENT:
        opportunity_id = opportunity_episode_id(
            run_id=RUN_ID,
            component_id=COMPONENT_ID,
            opened_event_index=OPEN_EVENT_INDEX,
        )
    legs = (
        tuple(PortfolioLegSnapshot.from_allocation(item) for item in opportunity.quantities)
        if opportunity is not None
        else ()
    )
    signature = portfolio_signature(legs) if opportunity is not None else None
    return OpportunityObservation(
        observation_id=opportunity_observation_id(
            run_id=RUN_ID,
            component_id=COMPONENT_ID,
            event_index=event_index,
            transition=transition,
            opportunity_id=opportunity_id,
        ),
        opportunity_id=opportunity_id,
        run_id=RUN_ID,
        component_id=COMPONENT_ID,
        observed_at=observed_at,
        event_index=event_index,
        transition=transition,
        market_tickers=("A", "M"),
        relation_types=(RelationType.IMPLIES,),
        opportunity=opportunity,
        solver_status="optimal" if opportunity is not None else "no_arbitrage",
        solver_reason=None if opportunity is not None else "no positive verified guarantee",
        solve_duration_ms=Decimal("1.25"),
        num_states=3,
        num_instruments=len(legs),
        num_legs=len(legs),
        portfolio_legs=legs,
        portfolio_signature=signature,
        close_reason=close_reason,
        censor_reason=censor_reason,
        evidence={"min_profit": "0.10"},
        metadata={"fixture": True},
    )


def test_episode_and_observation_ids_are_deterministic_and_scoped() -> None:
    opportunity_id = opportunity_episode_id(
        run_id=RUN_ID,
        component_id=COMPONENT_ID,
        opened_event_index=OPEN_EVENT_INDEX,
    )
    assert opportunity_id == opportunity_episode_id(
        run_id=RUN_ID,
        component_id=COMPONENT_ID,
        opened_event_index=OPEN_EVENT_INDEX,
    )
    assert opportunity_id != opportunity_episode_id(
        run_id=RUN_ID,
        component_id=COMPONENT_ID,
        opened_event_index=OPEN_EVENT_INDEX + 1,
    )

    observation_id = opportunity_observation_id(
        run_id=RUN_ID,
        component_id=COMPONENT_ID,
        event_index=OPEN_EVENT_INDEX,
        transition=OpportunityTransition.OPEN,
        opportunity_id=opportunity_id,
    )
    assert observation_id.startswith("observation:")
    assert opportunity_id.startswith("opportunity:")


def test_portfolio_snapshot_is_exact_and_signature_is_order_independent() -> None:
    first = PortfolioLegSnapshot.from_allocation(_allocation(price="0.50"), fee=Decimal("0"))
    second = PortfolioLegSnapshot(
        ticker="A",
        side="no",
        price=Decimal("0.30"),
        quantity=Decimal("2.00"),
        source_side="yes_bid",
        source_price=Decimal("0.70"),
        fee=None,
    )
    equivalent_first = PortfolioLegSnapshot(
        ticker="M",
        side="yes",
        price=Decimal("0.500"),
        quantity=Decimal("1"),
        source_side="no_bid",
        source_price=Decimal("0.5"),
        fee=Decimal("0.000"),
    )

    assert first.ticker == "M"
    assert first.quantity == Decimal("1")
    assert portfolio_signature((first, second)) == portfolio_signature((second, equivalent_first))
    assert (
        first.signature
        != PortfolioLegSnapshot.from_allocation(_allocation(), fee=Decimal("0.01")).signature
    )


def test_portfolio_snapshot_rejects_untrusted_source_and_is_frozen() -> None:
    leg = PortfolioLegSnapshot.from_allocation(_allocation())

    with pytest.raises(ValidationError, match="originate from no_bid"):
        PortfolioLegSnapshot(
            ticker="M",
            side="yes",
            price=Decimal("0.5"),
            quantity=Decimal("1"),
            source_side="yes_bid",
            source_price=Decimal("0.5"),
        )
    with pytest.raises(ValidationError, match="frozen"):
        leg.quantity = Decimal("2")  # type: ignore[misc]


def test_observation_shapes_distinguish_absent_active_and_closed() -> None:
    not_present = _observation(
        OpportunityTransition.NOT_PRESENT,
        event_index=9,
        observed_at=NOW - timedelta(seconds=1),
    )
    opened = _observation(
        OpportunityTransition.OPEN,
        event_index=OPEN_EVENT_INDEX,
        observed_at=NOW,
        opportunity=_opportunity(),
    )
    closed = _observation(
        OpportunityTransition.CLOSED,
        event_index=12,
        observed_at=NOW + timedelta(seconds=2),
        close_reason="verified net profit no longer positive",
    )
    censored = _observation(
        OpportunityTransition.RIGHT_CENSORED,
        event_index=13,
        observed_at=NOW + timedelta(seconds=3),
        opportunity=_opportunity(),
        censor_reason="run_ended",
    )

    assert not_present.opportunity_id is None
    assert not_present.num_markets == 2
    assert opened.portfolio_signature == portfolio_signature(opened.portfolio_legs)
    assert closed.close_reason == "verified net profit no longer positive"
    assert censored.censor_reason == "run_ended"
    assert censored.opportunity == opened.opportunity


@pytest.mark.parametrize(
    ("transition", "changes", "error"),
    [
        (
            OpportunityTransition.OPEN,
            {"observed_at": datetime(2026, 9, 3, 12)},
            "timezone-aware",
        ),
        (
            OpportunityTransition.OPEN,
            {"observation_id": "observation:wrong"},
            "deterministic identity",
        ),
        (
            OpportunityTransition.OPEN,
            {"portfolio_signature": "wrong"},
            "does not match",
        ),
        (
            OpportunityTransition.CLOSED,
            {"close_reason": None},
            "close reason",
        ),
        (
            OpportunityTransition.RIGHT_CENSORED,
            {"censor_reason": None},
            "censor reason",
        ),
    ],
)
def test_observation_rejects_invalid_identity_time_signature_and_close_shape(
    transition: OpportunityTransition,
    changes: dict[str, object],
    error: str,
) -> None:
    base = _observation(
        transition,
        event_index=OPEN_EVENT_INDEX if transition is OpportunityTransition.OPEN else 12,
        observed_at=NOW,
        opportunity=(
            _opportunity()
            if transition in {OpportunityTransition.OPEN, OpportunityTransition.RIGHT_CENSORED}
            else None
        ),
        close_reason="closed" if transition is OpportunityTransition.CLOSED else None,
        censor_reason=("run_ended" if transition is OpportunityTransition.RIGHT_CENSORED else None),
    )
    values = base.model_dump()
    values.update(changes)

    with pytest.raises(ValidationError, match=error):
        OpportunityObservation.model_validate(values)


def test_episode_opens_updates_and_closes_with_exact_event_time_duration() -> None:
    opened = _observation(
        OpportunityTransition.OPEN,
        event_index=OPEN_EVENT_INDEX,
        observed_at=NOW,
        opportunity=_opportunity(),
    )
    episode = OpportunityEpisode.from_open_observation(opened)
    updated = _observation(
        OpportunityTransition.UPDATED,
        event_index=11,
        observed_at=NOW + timedelta(milliseconds=750),
        opportunity=_opportunity(quantity="2.00"),
    )
    episode = episode.apply_observation(updated)
    closed = _observation(
        OpportunityTransition.CLOSED,
        event_index=12,
        observed_at=NOW + timedelta(seconds=2),
        close_reason="stale book",
    )
    episode = episode.apply_observation(closed)

    assert episode.transition is OpportunityTransition.CLOSED
    assert episode.observation_count == 3
    assert episode.duration == timedelta(seconds=2)
    assert episode.closed_at == closed.observed_at
    assert episode.closed_event_index == 12
    assert episode.opportunity.quantities[0].quantity == Decimal("2")
    assert episode.close_reason == "stale book"


def test_episode_right_censor_preserves_last_economic_state_without_closing() -> None:
    opened = _observation(
        OpportunityTransition.OPEN,
        event_index=OPEN_EVENT_INDEX,
        observed_at=NOW,
        opportunity=_opportunity(),
    )
    updated = _observation(
        OpportunityTransition.UPDATED,
        event_index=11,
        observed_at=NOW + timedelta(seconds=1),
        opportunity=_opportunity(quantity="2.00"),
    )
    active = OpportunityEpisode.from_open_observation(opened).apply_observation(updated)
    censored_observation = _observation(
        OpportunityTransition.RIGHT_CENSORED,
        event_index=12,
        observed_at=NOW + timedelta(seconds=3),
        opportunity=updated.opportunity,
        censor_reason="run_ended",
    )

    censored = active.apply_observation(censored_observation)

    assert censored.transition is OpportunityTransition.RIGHT_CENSORED
    assert censored.opportunity == active.opportunity
    assert censored.portfolio_legs == active.portfolio_legs
    assert censored.portfolio_signature == active.portfolio_signature
    assert censored.closed_at is None
    assert censored.closed_event_index is None
    assert censored.close_reason is None
    assert censored.censored_at == NOW + timedelta(seconds=3)
    assert censored.censored_event_index == 12
    assert censored.censor_reason == "run_ended"
    assert censored.duration == timedelta(seconds=3)

    with pytest.raises(ValueError, match="terminal opportunity episode"):
        censored.apply_observation(
            _observation(
                OpportunityTransition.UPDATED,
                event_index=13,
                observed_at=NOW + timedelta(seconds=4),
                opportunity=_opportunity(),
            )
        )


def test_episode_rejects_identity_index_and_event_time_regressions() -> None:
    opened = _observation(
        OpportunityTransition.OPEN,
        event_index=OPEN_EVENT_INDEX,
        observed_at=NOW,
        opportunity=_opportunity(),
    )
    episode = OpportunityEpisode.from_open_observation(opened)
    same_index = _observation(
        OpportunityTransition.UPDATED,
        event_index=OPEN_EVENT_INDEX,
        observed_at=NOW + timedelta(seconds=1),
        opportunity=_opportunity(),
    )
    backwards_time = _observation(
        OpportunityTransition.UPDATED,
        event_index=OPEN_EVENT_INDEX + 1,
        observed_at=NOW - timedelta(microseconds=1),
        opportunity=_opportunity(),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        episode.apply_observation(same_index)
    with pytest.raises(ValueError, match="cannot move backwards"):
        episode.apply_observation(backwards_time)


def test_closed_episode_cannot_reopen_or_update() -> None:
    episode = OpportunityEpisode.from_open_observation(
        _observation(
            OpportunityTransition.OPEN,
            event_index=OPEN_EVENT_INDEX,
            observed_at=NOW,
            opportunity=_opportunity(),
        )
    ).apply_observation(
        _observation(
            OpportunityTransition.CLOSED,
            event_index=OPEN_EVENT_INDEX + 1,
            observed_at=NOW + timedelta(seconds=1),
            close_reason="no arbitrage",
        )
    )

    with pytest.raises(ValueError, match="cannot transition again"):
        episode.apply_observation(
            _observation(
                OpportunityTransition.UPDATED,
                event_index=OPEN_EVENT_INDEX + 2,
                observed_at=NOW + timedelta(seconds=2),
                opportunity=_opportunity(),
            )
        )


def test_lifecycle_models_reject_unknown_fields() -> None:
    observation = _observation(
        OpportunityTransition.OPEN,
        event_index=OPEN_EVENT_INDEX,
        observed_at=NOW,
        opportunity=_opportunity(),
    )

    with pytest.raises(ValidationError, match="Extra inputs"):
        OpportunityObservation.model_validate({**observation.model_dump(), "surprise": True})


def test_observation_rejects_legs_from_a_different_portfolio() -> None:
    observation = _observation(
        OpportunityTransition.OPEN,
        event_index=OPEN_EVENT_INDEX,
        observed_at=NOW,
        opportunity=_opportunity(),
    )
    wrong_leg = observation.portfolio_legs[0].model_copy(update={"quantity": Decimal("0.5")})

    with pytest.raises(ValidationError, match="must match the opportunity allocations"):
        OpportunityObservation.model_validate(
            {
                **observation.model_dump(),
                "portfolio_legs": (wrong_leg,),
                "portfolio_signature": portfolio_signature((wrong_leg,)),
            }
        )
