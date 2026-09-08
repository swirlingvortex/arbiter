"""Exact six-decimal fee rounding and per-order accumulator tests."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from tests.support.fees import series

from arbiter.models.opportunity import FeeFill
from arbiter.models.series import Series
from arbiter.solver.fees import (
    DIRECT_ACCOUNT_PRECISION,
    NON_DIRECT_ACCOUNT_PRECISION,
    FeeRoundingLedger,
    KalshiFeeModel,
    ZeroFeeModel,
    ceil_model_fee,
    quote_order,
)


def _fill(
    price: str,
    *,
    quantity: str = "1",
    order_id: str = "A:yes",
    ticker: str = "A",
    side: str = "yes",
) -> FeeFill:
    return FeeFill(
        order_id=order_id,
        ticker=ticker,
        side=side,
        price=Decimal(price),
        quantity=Decimal(quantity),
    )


def _fixture(project_root: Path) -> dict[str, Any]:
    value = json.loads((project_root / "tests/fixtures/kalshi/fees_2026-07-07.json").read_text())
    return cast(dict[str, Any], value)


def test_official_fcm_rounding_example_matches_every_component(project_root: Path) -> None:
    expected = _fixture(project_root)["official_fcm_rounding_example"]
    ledger = FeeRoundingLedger(order_id="A:yes", account_precision=Decimal("0.01"))

    quote = quote_order(
        (_fill(expected["price"], quantity=expected["quantity"]),),
        ledger,
        series=series(),
        fee_model=KalshiFeeModel(),
    )
    item = quote.fills[0]

    assert item.model_fee == Decimal(expected["model_fee"])
    assert item.trade_fee == Decimal(expected["trade_fee"])
    assert item.aligned_change_before_rebate == Decimal(expected["aligned_change"])
    assert item.rounding_fee == Decimal(expected["rounding_fee"])
    assert item.rebate == 0
    assert item.net_fee == Decimal(expected["net_fee"])
    assert ledger.accumulated_rounding == Decimal("0.001361")


def test_same_official_fill_uses_direct_member_precision_without_truncation() -> None:
    ledger = FeeRoundingLedger(order_id="A:yes")

    item = quote_order(
        (_fill("0.055"),),
        ledger,
        series=series(),
        fee_model=KalshiFeeModel(),
    ).fills[0]

    assert item.trade_fee == Decimal("0.003639")
    assert item.aligned_change_before_rebate == Decimal("-0.058700")
    assert item.rounding_fee == Decimal("0.000061")
    assert item.net_fee == Decimal("0.003700")
    assert item.balance_change == Decimal("-0.058700")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.00363825", "0.003639"),
        ("0.003639000", "0.003639"),
        ("0", "0.000000"),
    ],
)
def test_model_fee_ceiling_is_exactly_six_decimal_dollars(
    raw: str,
    expected: str,
) -> None:
    assert ceil_model_fee(Decimal(raw)) == Decimal(expected)


def test_real_fee_accumulator_spans_calls_for_one_order() -> None:
    ledger = FeeRoundingLedger(
        order_id="A:yes",
        account_precision=NON_DIRECT_ACCOUNT_PRECISION,
    )
    model = KalshiFeeModel()
    quotes = [
        quote_order((_fill("0.10"),), ledger, series=series(), fee_model=model),
        quote_order((_fill("0.10"),), ledger, series=series(), fee_model=model),
        quote_order((_fill("0.40"),), ledger, series=series(), fee_model=model),
    ]

    assert [quote.fills[0].rounding_fee for quote in quotes] == [
        Decimal("0.003700"),
        Decimal("0.003700"),
        Decimal("0.003200"),
    ]
    assert [quote.fills[0].rebate for quote in quotes] == [
        Decimal("0"),
        Decimal("0"),
        Decimal("0.010000"),
    ]
    assert [quote.net_fee for quote in quotes] == [
        Decimal("0.010000"),
        Decimal("0.010000"),
        Decimal("0.010000"),
    ]
    assert ledger.accumulated_rounding == Decimal("0.000600")


def test_new_ledger_per_fill_overcharges_relative_to_one_order_accumulator() -> None:
    fresh_total = Decimal("0")
    for index, price in enumerate(("0.10", "0.10", "0.40")):
        order_id = f"A:yes:{index}"
        fresh_total += quote_order(
            (_fill(price, order_id=order_id),),
            FeeRoundingLedger(
                order_id=order_id,
                account_precision=NON_DIRECT_ACCOUNT_PRECISION,
            ),
            series=series(),
            fee_model=KalshiFeeModel(),
        ).net_fee

    assert fresh_total == Decimal("0.040000")


def test_distinct_portfolio_legs_require_distinct_ledgers() -> None:
    yes_ledger = FeeRoundingLedger(order_id="A:yes")
    no_ledger = FeeRoundingLedger(order_id="A:no")

    quote_order((_fill("0.055"),), yes_ledger, series=series(), fee_model=KalshiFeeModel())
    quote_order(
        (_fill("0.055", order_id="A:no", side="no"),),
        no_ledger,
        series=series(),
        fee_model=KalshiFeeModel(),
    )

    assert yes_ledger.accumulated_rounding == Decimal("0.000061")
    assert no_ledger.accumulated_rounding == Decimal("0.000061")


def test_rebate_is_capped_when_current_fill_cannot_cover_one_grid_increment() -> None:
    ledger = FeeRoundingLedger(
        order_id="A:yes",
        account_precision=NON_DIRECT_ACCOUNT_PRECISION,
        accumulated_rounding=Decimal("0.01"),
    )

    item = quote_order(
        (_fill("0.01"),),
        ledger,
        series=series(),
        fee_model=ZeroFeeModel(),
    ).fills[0]

    assert item.rounding_fee == 0
    assert item.rebate == 0
    assert item.net_fee == 0
    assert ledger.accumulated_rounding == Decimal("0.01")


def test_zero_model_can_still_require_balance_alignment() -> None:
    item = quote_order(
        (_fill("0.5555", quantity="0.01"),),
        FeeRoundingLedger(order_id="A:yes"),
        series=series(),
        fee_model=ZeroFeeModel(),
    ).fills[0]

    assert item.signed_revenue == Decimal("-0.005555")
    assert item.aligned_change_before_rebate == Decimal("-0.0056")
    assert item.rounding_fee == item.net_fee == Decimal("0.000045")


@pytest.mark.parametrize("precision", ["0", "-0.01", "0.001", "0.1"])
def test_only_documented_account_precisions_are_accepted(precision: str) -> None:
    with pytest.raises(ValueError, match="account precision"):
        FeeRoundingLedger(order_id="A:yes", account_precision=Decimal(precision))


def test_negative_or_nonfinite_model_fee_is_rejected() -> None:
    with pytest.raises(ValueError):
        ceil_model_fee(Decimal("-0.01"))
    with pytest.raises(ValueError):
        ceil_model_fee(Decimal("NaN"))


def test_quote_is_atomic_when_the_model_rejects_a_later_fill() -> None:
    class RejectSecondModel:
        calls = 0

        def fee_for_fill(
            self,
            *,
            series: Series,
            side: str,
            price: Decimal,
            quantity: Decimal,
            liquidity_role: str = "taker",
        ) -> Decimal:
            del series, side, price, quantity, liquidity_role
            self.calls += 1
            if self.calls == 2:
                raise ValueError("fixture rejection")
            return Decimal("0")

    ledger = FeeRoundingLedger(order_id="A:yes")
    with pytest.raises(ValueError, match="fixture rejection"):
        quote_order(
            (_fill("0.5555", quantity="0.01"), _fill("0.50")),
            ledger,
            series=series(),
            fee_model=RejectSecondModel(),
        )

    assert ledger.accumulated_rounding == 0


def test_precision_constants_match_current_documentation() -> None:
    assert Decimal("0.0001") == DIRECT_ACCOUNT_PRECISION
    assert Decimal("0.01") == NON_DIRECT_ACCOUNT_PRECISION
