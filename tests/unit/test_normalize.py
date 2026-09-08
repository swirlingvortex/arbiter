"""Current fixed-point payload normalization tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from arbiter.kalshi.normalize import (
    NormalizationError,
    dollars,
    fixed_quantity,
    normalize_event,
    normalize_event_fee_change,
    normalize_market,
    normalize_series,
)


def _payload(project_root: Path, name: str, wrapper: str | None = None) -> dict[str, Any]:
    path = project_root / "tests/fixtures/kalshi" / name
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    if wrapper is None:
        return cast(dict[str, Any], value)
    nested = value[wrapper]
    assert isinstance(nested, dict)
    return cast(dict[str, Any], nested)


def test_market_normalization_preserves_raw_unknown_fields_and_decimal_strike(
    project_root: Path,
) -> None:
    payload = _payload(project_root, "market.json")

    market = normalize_market(payload, series_ticker="KXARBITER")

    assert market.series_ticker == "KXARBITER"
    assert market.status == "active"
    assert market.floor_strike == Decimal("100.5")
    assert market.expiration_time == market.latest_expiration_time
    assert market.expected_expiration_time != market.expiration_time
    assert market.price_ranges[0].step == Decimal("0.0100")
    assert market.accepts_price(Decimal("0.5500"))
    assert not market.accepts_price(Decimal("0.5550"))
    assert market.raw == payload
    assert market.raw["future_additive_field"] == {"preserve": True}
    assert market.raw["fractional_trading_enabled"] is True


def test_missing_optional_market_fields_and_additive_fields_are_tolerated() -> None:
    payload = {
        "ticker": "MINIMAL",
        "event_ticker": "EVENT",
        "status": "initialized",
        "unknown": 42,
    }

    market = normalize_market(payload)

    assert market.title == "MINIMAL"
    assert market.open_time is None
    assert market.price_ranges == ()
    assert market.raw["unknown"] == 42


def test_event_supports_legacy_top_level_markets(project_root: Path) -> None:
    response = _payload(project_root, "event.json")
    event_payload = cast(dict[str, Any], response["event"])
    markets = cast(list[dict[str, Any]], response["markets"])

    event = normalize_event(event_payload, top_level_markets=markets)

    assert event.ticker == "KXARBITER-26SEP03"
    assert event.series_ticker == "KXARBITER"
    assert event.market_tickers == ("KXARBITER-26SEP03-T100",)
    assert event.raw["future_event_field"] == "preserved"


def test_event_current_fee_override_is_decimal_and_strictly_paired() -> None:
    event = normalize_event(
        {
            "event_ticker": "EVENT",
            "title": "Event",
            "fee_type": "quadratic",
            "fee_multiplier": "1.125",
        }
    )

    assert event.fee_type_override == "quadratic"
    assert event.fee_multiplier_override == Decimal("1.125")
    with pytest.raises(NormalizationError, match="both be present or both be null"):
        normalize_event(
            {
                "event_ticker": "EVENT",
                "title": "Event",
                "fee_type": "quadratic",
            }
        )


def test_event_fee_change_normalizes_decimal_timestamp_and_raw(project_root: Path) -> None:
    response = _payload(project_root, "event_fee_changes.json")
    payload = cast(list[dict[str, Any]], response["event_fee_changes"])[0]

    change = normalize_event_fee_change(payload)

    assert change.change_id == "fee-change-future"
    assert change.scheduled_ts == datetime(2026, 9, 6, 1, 40, tzinfo=UTC)
    assert change.fee_type_override == "quadratic_with_maker_fees"
    assert change.fee_multiplier_override == Decimal("1")
    assert change.raw["future_field"] == "preserved"


@pytest.mark.parametrize(
    ("fee_type", "fee_multiplier"),
    [("quadratic", None), (None, "1.0")],
)
def test_event_fee_change_rejects_partial_override(
    fee_type: str | None,
    fee_multiplier: str | None,
) -> None:
    with pytest.raises(NormalizationError, match="both be present or both be null"):
        normalize_event_fee_change(
            {
                "id": "change",
                "event_ticker": "EVENT",
                "series_ticker": "SERIES",
                "scheduled_ts": "2026-09-03T12:00:00Z",
                "fee_type": fee_type,
                "fee_multiplier": fee_multiplier,
            }
        )


def test_event_fee_change_rejects_naive_time_and_invalid_multiplier() -> None:
    base: dict[str, Any] = {
        "id": "change",
        "event_ticker": "EVENT",
        "series_ticker": "SERIES",
        "scheduled_ts": "2026-09-03T12:00:00",
        "fee_type": "quadratic",
        "fee_multiplier": "1.0",
    }
    with pytest.raises(NormalizationError, match="timezone-aware"):
        normalize_event_fee_change(base)
    with pytest.raises(NormalizationError, match="finite"):
        normalize_event_fee_change(
            {**base, "scheduled_ts": "2026-09-03T12:00:00Z", "fee_multiplier": "-1"}
        )


def test_event_fee_change_requires_explicit_override_or_clear_fields() -> None:
    with pytest.raises(NormalizationError, match="Field required"):
        normalize_event_fee_change(
            {
                "id": "change",
                "event_ticker": "EVENT",
                "series_ticker": "SERIES",
                "scheduled_ts": "2026-09-03T12:00:00Z",
            }
        )


def test_series_normalization_preserves_fee_and_source_precision(project_root: Path) -> None:
    payload = _payload(project_root, "series.json", "series")

    series = normalize_series(payload)

    assert series.fee_type == "quadratic"
    assert series.fee_multiplier == Decimal("1.25")
    assert series.settlement_sources[0].name == "Fixture Authority"
    assert series.settlement_sources[0].model_extra == {"future_source_field": "preserved"}
    assert series.raw["future_series_field"] is True


def test_live_compatible_null_series_collections_normalize_to_empty_tuples() -> None:
    series = normalize_series(
        {
            "ticker": "SERIES",
            "title": "Series",
            "tags": None,
            "settlement_sources": None,
        }
    )

    assert series.tags == ()
    assert series.settlement_sources == ()


@pytest.mark.parametrize("fee_multiplier", ["-0.1", "Infinity", "NaN"])
def test_series_fee_multiplier_must_be_finite_and_nonnegative(
    fee_multiplier: str,
) -> None:
    with pytest.raises(NormalizationError, match="finite"):
        normalize_series(
            {
                "ticker": "SERIES",
                "title": "Series",
                "fee_type": "quadratic",
                "fee_multiplier": fee_multiplier,
            }
        )


@pytest.mark.parametrize("value", [0.5, 50, Decimal("0.5")])
def test_fixed_point_parsers_require_wire_strings(value: object) -> None:
    with pytest.raises(NormalizationError, match="fixed-point string"):
        dollars(value)  # type: ignore[arg-type]


def test_fixed_point_precision_and_bounds_fail_closed() -> None:
    assert dollars("0.5600") == Decimal("0.5600")
    assert fixed_quantity("10.00") == Decimal("10.00")
    with pytest.raises(NormalizationError, match="4 decimal places"):
        dollars("0.12345")
    with pytest.raises(NormalizationError, match="between 0 and 1"):
        dollars("1.0100")
    with pytest.raises(NormalizationError, match="2 decimal places"):
        fixed_quantity("1.001")
    with pytest.raises(NormalizationError, match="negative"):
        fixed_quantity("-0.01")
