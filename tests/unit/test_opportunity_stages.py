"""Evidence-stage taxonomy and conservative Stage 0 midpoint tests."""

from datetime import timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError
from tests.support.executable import NOW, book, implication_component_and_worlds
from tests.support.fees import policy

from arbiter.models.opportunity import (
    FeeValidationStatus,
    Opportunity,
    OpportunityStage,
    classify_opportunity_stage,
    fresh_yes_midpoint,
)
from arbiter.models.orderbook import BookStatus
from arbiter.solver.lp import solve_worst_case_profit


def _gross_result():
    _, worlds = implication_component_and_worlds()
    from tests.unit.test_solver_numerics import _instruments

    return solve_worst_case_profit(worlds, _instruments())


def test_stage_zero_midpoint_requires_a_fresh_two_sided_yes_market() -> None:
    two_sided = book("M", yes=(("0.70", "1"),), no=(("0.20", "1"),))

    assert fresh_yes_midpoint(
        two_sided,
        as_of=NOW,
        stale_after=timedelta(seconds=2),
    ) == Decimal("0.75")
    assert (
        fresh_yes_midpoint(
            book("M", yes=(("0.70", "1"),)),
            as_of=NOW,
            stale_after=timedelta(seconds=2),
        )
        is None
    )


@pytest.mark.parametrize(
    "candidate",
    [
        book(
            "M",
            yes=(("0.70", "1"),),
            no=(("0.20", "1"),),
            status=BookStatus.STALE,
            status_reason="fixture stale",
        ),
        book(
            "M",
            yes=(("0.70", "1"),),
            no=(("0.20", "1"),),
            status=BookStatus.RESYNC_REQUIRED,
            status_reason="gap",
        ),
        book(
            "M",
            yes=(("0.70", "1"),),
            no=(("0.20", "1"),),
            local_timestamp=NOW + timedelta(microseconds=1),
        ),
    ],
)
def test_nonfresh_books_have_no_stage_zero_reference(candidate: object) -> None:
    assert (
        fresh_yes_midpoint(
            candidate,  # type: ignore[arg-type]
            as_of=NOW,
            stale_after=timedelta(seconds=2),
        )
        is None
    )


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ({"logical_violation": False}, None),
        ({"logical_violation": True}, OpportunityStage.LOGICAL),
        (
            {"logical_violation": True, "gross_executable": True},
            OpportunityStage.GROSS_EXECUTABLE,
        ),
        (
            {
                "logical_violation": True,
                "gross_executable": True,
                "net_executable": True,
            },
            OpportunityStage.NET_EXECUTABLE,
        ),
        (
            {
                "logical_violation": True,
                "gross_executable": True,
                "net_executable": True,
                "paper_survives": True,
            },
            OpportunityStage.PAPER_SURVIVED,
        ),
    ],
)
def test_highest_evidenced_stage_wins(
    evidence: dict[str, bool],
    expected: OpportunityStage | None,
) -> None:
    assert classify_opportunity_stage(**evidence) is expected


@pytest.mark.parametrize(
    "evidence",
    [
        {"logical_violation": False, "gross_executable": True},
        {"logical_violation": True, "net_executable": True},
        {
            "logical_violation": True,
            "gross_executable": True,
            "paper_survives": True,
        },
    ],
)
def test_stage_prerequisites_cannot_be_skipped(evidence: dict[str, bool]) -> None:
    with pytest.raises(ValueError):
        classify_opportunity_stage(**evidence)


def test_stage_zero_is_informational_and_contains_no_executable_claim() -> None:
    opportunity = Opportunity(stage=OpportunityStage.LOGICAL)

    assert opportunity.capital_required is None
    assert opportunity.quantities == ()
    assert opportunity.fee_status is FeeValidationStatus.NOT_EVALUATED


@pytest.mark.parametrize(
    "update",
    [
        {"fee_status": FeeValidationStatus.APPLIED},
        {"net_edge": Decimal("0")},
        {"paper_execution_status": "failed"},
        {"fee_policies": (policy("A"),)},
    ],
)
def test_stage_zero_rejects_executable_evidence(update: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Stage 0"):
        Opportunity(stage=OpportunityStage.LOGICAL, **update)


def test_stage_one_preserves_the_exact_gross_solver_evidence() -> None:
    result = _gross_result()
    assert result.status == "optimal"

    opportunity = Opportunity(
        stage=OpportunityStage.GROSS_EXECUTABLE,
        quantities=result.quantities,
        capital_required=result.capital_required,
        gross_profit=result.guaranteed_gross_profit,
        gross_edge=result.gross_edge,
        gross_state_profits=result.state_profits,
    )

    assert opportunity.gross_profit == Decimal("0.08")
    assert opportunity.net_profit is None


def test_fee_status_and_net_economics_must_agree() -> None:
    result = _gross_result()
    gross = {
        "stage": OpportunityStage.GROSS_EXECUTABLE,
        "quantities": result.quantities,
        "capital_required": result.capital_required,
        "gross_profit": result.guaranteed_gross_profit,
        "gross_edge": result.gross_edge,
        "gross_state_profits": result.state_profits,
    }

    with pytest.raises(ValidationError, match="complete net economics"):
        Opportunity(**gross, fee_status=FeeValidationStatus.APPLIED)
    with pytest.raises(ValidationError, match="applied fee evidence"):
        Opportunity(
            **gross,
            fees=Decimal("0"),
            net_profit=result.guaranteed_gross_profit,
            net_edge=result.gross_edge,
            net_state_profits=result.state_profits,
        )


def test_stage_three_requires_explicit_paper_survival() -> None:
    result = _gross_result()

    with pytest.raises(ValidationError, match="paper execution survival"):
        Opportunity(
            stage=OpportunityStage.PAPER_SURVIVED,
            quantities=result.quantities,
            capital_required=result.capital_required,
            gross_profit=result.guaranteed_gross_profit,
            gross_edge=result.gross_edge,
            gross_state_profits=result.state_profits,
            fee_status=FeeValidationStatus.APPLIED,
            fees=Decimal("0"),
            net_profit=result.guaranteed_gross_profit,
            net_edge=result.gross_edge,
            net_state_profits=result.state_profits,
            fee_policies=(policy("A"), policy("M")),
        )


def test_fee_policy_snapshot_is_json_serializable_without_losing_decimal_or_version() -> None:
    snapshot = policy("A", multiplier=Decimal("0.5"))
    restored = type(snapshot).model_validate_json(snapshot.model_dump_json())

    assert restored == snapshot
    assert restored.market_ticker == "A"
    assert restored.fee_multiplier == Decimal("0.5")
