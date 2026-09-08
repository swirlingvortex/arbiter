"""Verified fee formulas, metadata policy resolution, and fail-closed cases."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError
from tests.support.fees import POLICY_TIME, event, fee_change, series

from arbiter.models.event import EventFeeChange
from arbiter.solver.fees import (
    KALSHI_FEE_POLICY_VERSION,
    FeePolicyResolutionError,
    KalshiFeeModel,
    UnsupportedFeeModel,
    ZeroFeeModel,
    resolve_fee_policy,
)


def _official_fixture(project_root: Path) -> dict[str, Any]:
    loaded = json.loads((project_root / "tests/fixtures/kalshi/fees_2026-07-07.json").read_text())
    return cast(dict[str, Any], loaded)


def test_official_fixture_records_dated_primary_sources(project_root: Path) -> None:
    fixture = _official_fixture(project_root)

    assert fixture["schedule_url"] == "https://kalshi.com/docs/kalshi-fee-schedule.pdf"
    assert fixture["schedule_effective_date"] == "2026-07-07"
    assert fixture["rounding_url"] == ("https://docs.kalshi.com/getting_started/fee_rounding")
    assert fixture["captured_date"] == "2026-09-03"
    assert fixture["policy_version"] == KALSHI_FEE_POLICY_VERSION


@pytest.mark.parametrize("side", ["yes", "no"])
def test_zero_fee_model_is_explicit_and_symmetric(side: str) -> None:
    assert (
        ZeroFeeModel().fee_for_fill(
            series=series(),
            side=side,
            price=Decimal("0.37"),
            quantity=Decimal("2.50"),
        )
        == 0
    )


def test_official_general_taker_table_matches_raw_formula(project_root: Path) -> None:
    fixture = _official_fixture(project_root)
    model = KalshiFeeModel()

    for example in fixture["official_taker_examples"]:
        fee = model.fee_for_fill(
            series=series(multiplier=Decimal(example["multiplier"])),
            side="yes",
            price=Decimal(example["price"]),
            quantity=Decimal(example["quantity"]),
        )
        assert fee == Decimal(example["raw_fee"])


@pytest.mark.parametrize(
    "fee_type",
    [
        "quadratic",
        "quadratic_with_maker_fees",
        "quadratic_with_combo_maker_fees",
    ],
)
def test_all_current_quadratic_variants_share_the_taker_formula(fee_type: str) -> None:
    model = KalshiFeeModel()

    assert model.fee_for_fill(
        series=series(fee_type=fee_type),
        side="yes",
        price=Decimal("0.62"),
        quantity=Decimal("1"),
    ) == Decimal("0.016492")
    assert model.fee_for_fill(
        series=series(fee_type=fee_type),
        side="no",
        price=Decimal("0.38"),
        quantity=Decimal("1"),
    ) == Decimal("0.016492")


@pytest.mark.parametrize(
    ("fee_type", "expected"),
    [
        ("quadratic", Decimal("0")),
        ("quadratic_with_maker_fees", Decimal("0.437500")),
        ("quadratic_with_combo_maker_fees", Decimal("0.875000")),
    ],
)
def test_current_maker_variants_match_the_official_schedule(
    fee_type: str,
    expected: Decimal,
) -> None:
    assert (
        KalshiFeeModel().fee_for_fill(
            series=series(fee_type=fee_type),
            side="yes",
            price=Decimal("0.50"),
            quantity=Decimal("100"),
            liquidity_role="maker",
        )
        == expected
    )


def test_multiplier_and_verified_zero_waiver_are_honored() -> None:
    model = KalshiFeeModel()
    assert model.fee_for_fill(
        series=series(multiplier=Decimal("1.25")),
        side="yes",
        price=Decimal("0.50"),
        quantity=Decimal("100"),
    ) == Decimal("2.187500")
    assert (
        model.fee_for_fill(
            series=series(multiplier=Decimal("0")),
            side="yes",
            price=Decimal("0.50"),
            quantity=Decimal("100"),
        )
        == 0
    )


def test_missing_multiplier_uses_conservative_standard_not_a_waiver() -> None:
    model = KalshiFeeModel()

    assert model.fee_for_fill(
        series=series(multiplier=None),
        side="yes",
        price=Decimal("0.50"),
        quantity=Decimal("100"),
    ) == Decimal("1.75")
    assert model.fee_for_fill(
        series=series(fee_type=None, multiplier=None),
        side="yes",
        price=Decimal("0.50"),
        quantity=Decimal("100"),
    ) == Decimal("1.75")


def test_multiplier_without_fee_type_fails_closed_in_policy_resolution() -> None:
    with pytest.raises(FeePolicyResolutionError, match="without a fee type"):
        resolve_fee_policy(
            event=event(),
            series=series(fee_type=None, multiplier=Decimal("2")),
            as_of=POLICY_TIME,
        )


@pytest.mark.parametrize("fee_type", ["flat", "future_unknown_type"])
def test_unverified_fee_types_fail_closed(fee_type: str) -> None:
    with pytest.raises(UnsupportedFeeModel, match="unsupported"):
        KalshiFeeModel().fee_for_fill(
            series=series(fee_type=fee_type),
            side="yes",
            price=Decimal("0.50"),
            quantity=Decimal("1"),
        )


def test_event_override_activates_exactly_at_scheduled_time_and_clear_falls_back() -> None:
    scheduled = POLICY_TIME + timedelta(days=1)
    cleared = POLICY_TIME + timedelta(days=2)
    market_event = event(
        changes=(
            fee_change(
                "override",
                scheduled,
                fee_type="quadratic_with_maker_fees",
                multiplier=Decimal("1"),
            ),
            fee_change("clear", cleared, fee_type=None, multiplier=None),
        )
    )
    parent = series(
        fee_type="quadratic_with_maker_fees",
        multiplier=Decimal("0.5"),
    )

    before = resolve_fee_policy(
        event=market_event,
        series=parent,
        as_of=scheduled - timedelta(microseconds=1),
        market_ticker="A",
    )
    active = resolve_fee_policy(
        event=market_event,
        series=parent,
        as_of=scheduled,
        market_ticker="A",
    )
    after_clear = resolve_fee_policy(
        event=market_event,
        series=parent,
        as_of=cleared,
        market_ticker="A",
    )

    assert before.source == "series_default" and before.fee_multiplier == Decimal("0.5")
    assert before.next_scheduled_at == scheduled and before.next_change_id == "override"
    assert active.source == "event_override" and active.fee_multiplier == Decimal("1")
    assert active.effective_at == scheduled and active.change_id == "override"
    assert after_clear.source == "series_default"
    assert after_clear.fee_multiplier == Decimal("0.5")
    assert after_clear.effective_at == cleared and after_clear.change_id == "clear"


def test_current_event_override_wins_when_no_scheduled_change_has_applied() -> None:
    resolved = resolve_fee_policy(
        event=event(
            fee_type_override="quadratic",
            fee_multiplier_override=Decimal("0"),
        ),
        series=series(multiplier=Decimal("1")),
        as_of=POLICY_TIME,
    )

    assert resolved.source == "event_override"
    assert resolved.fee_multiplier == 0


def test_conflicting_equal_time_changes_and_naive_clock_fail_closed() -> None:
    one = fee_change(
        "one",
        POLICY_TIME,
        fee_type="quadratic",
        multiplier=Decimal("1"),
    )
    two = fee_change(
        "two",
        POLICY_TIME,
        fee_type="quadratic",
        multiplier=Decimal("0.5"),
    )
    with pytest.raises(FeePolicyResolutionError, match="conflicting"):
        resolve_fee_policy(
            event=event(changes=(one, two)),
            series=series(),
            as_of=POLICY_TIME,
        )
    with pytest.raises(FeePolicyResolutionError, match="timezone-aware"):
        resolve_fee_policy(
            event=event(),
            series=series(),
            as_of=datetime(2026, 9, 3),
        )


def test_partial_null_and_naive_change_are_invalid_domain_data() -> None:
    with pytest.raises(ValidationError):
        EventFeeChange(
            change_id="partial",
            event_ticker="KXARBITER-EVENT",
            series_ticker="KXARBITER",
            fee_type_override="quadratic",
            fee_multiplier_override=None,
            scheduled_ts=POLICY_TIME,
            raw={},
        )
    with pytest.raises(ValidationError):
        EventFeeChange(
            change_id="naive",
            event_ticker="KXARBITER-EVENT",
            series_ticker="KXARBITER",
            fee_type_override=None,
            fee_multiplier_override=None,
            scheduled_ts=datetime(2026, 9, 3),
            raw={},
        )


@pytest.mark.parametrize(
    ("side", "price", "quantity", "role"),
    [
        ("bad", Decimal("0.5"), Decimal("1"), "taker"),
        ("yes", Decimal("-0.01"), Decimal("1"), "taker"),
        ("yes", Decimal("1.01"), Decimal("1"), "taker"),
        ("yes", Decimal("0.5"), Decimal("-1"), "taker"),
        ("yes", Decimal("0.5"), Decimal("1"), "unknown"),
    ],
)
def test_fee_inputs_are_validated(
    side: str,
    price: Decimal,
    quantity: Decimal,
    role: str,
) -> None:
    with pytest.raises(ValueError):
        KalshiFeeModel().fee_for_fill(
            series=series(),
            side=side,
            price=price,
            quantity=quantity,
            liquidity_role=role,
        )


def test_zero_quantity_and_boundary_prices_have_zero_model_fee() -> None:
    model = KalshiFeeModel()
    assert (
        model.fee_for_fill(series=series(), side="yes", price=Decimal("0"), quantity=Decimal("1"))
        == 0
    )
    assert (
        model.fee_for_fill(series=series(), side="yes", price=Decimal("1"), quantity=Decimal("1"))
        == 0
    )
    assert (
        model.fee_for_fill(series=series(), side="yes", price=Decimal("0.5"), quantity=Decimal("0"))
        == 0
    )
