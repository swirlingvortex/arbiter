"""Engine metadata catalogs, trusted components, worlds, and fee-certainty tests."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from tests.support.relations import make_relation

from arbiter.engine.state import EngineState, MetadataStateError
from arbiter.models.event import Event, EventFeeChange
from arbiter.models.market import Market
from arbiter.models.relation import RelationType
from arbiter.models.series import Series

NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)


def _market(
    ticker: str,
    *,
    event_ticker: str = "EVENT",
    series_ticker: str | None = "SERIES",
) -> Market:
    return Market(
        ticker=ticker,
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        title=ticker,
        status="active",
        raw={},
    )


def _event(
    *market_tickers: str,
    ticker: str = "EVENT",
    series_ticker: str | None = "SERIES",
    fee_changes: tuple[EventFeeChange, ...] = (),
) -> Event:
    return Event(
        ticker=ticker,
        series_ticker=series_ticker,
        title=ticker,
        market_tickers=market_tickers,
        fee_changes=fee_changes,
        raw={},
    )


def _series(ticker: str = "SERIES") -> Series:
    return Series(
        ticker=ticker,
        title=ticker,
        fee_type="quadratic",
        fee_multiplier=Decimal("1"),
        raw={},
    )


def _catalog_state(*, max_component_markets: int = 12) -> EngineState:
    markets = tuple(_market(ticker) for ticker in ("A", "B", "C", "D"))
    relations = (
        make_relation(RelationType.IMPLIES, ("A", "B"), relation_id="trusted"),
        make_relation(
            RelationType.EQUIVALENT,
            ("C", "D"),
            relation_id="unverified",
            verified=False,
        ),
    )
    return EngineState(
        markets=markets,
        events=(_event("A", "B", "C", "D"),),
        series=(_series(),),
        relations=relations,
        max_component_markets=max_component_markets,
    )


def test_catalog_views_are_immutable_and_only_verified_components_are_subscribed() -> None:
    state = _catalog_state()

    assert tuple(state.markets) == ("A", "B", "C", "D")
    assert tuple(state.events) == ("EVENT",)
    assert tuple(state.series) == ("SERIES",)
    assert set(state.relations) == {"trusted", "unverified"}
    assert len(state.graph.components) == 1
    component = state.graph.components[0]
    assert component.market_tickers == ("A", "B")
    assert state.subscription_tickers == ("A", "B")
    assert state.components_for_markets(("C", "A", "B")) == (component,)
    assert state.components_for_markets(("C", "D")) == ()

    with pytest.raises(TypeError):
        state.markets["X"] = _market("X")  # type: ignore[index]
    with pytest.raises(TypeError):
        state.world_sets[component.component_id] = state.world_for_component(component)  # type: ignore[index]


def test_world_results_are_precomputed_bounded_and_available_by_component_or_id() -> None:
    state = _catalog_state(max_component_markets=1)
    component = state.graph.components[0]

    result = state.world_for_component(component)

    assert result is state.world_for_component(component.component_id)
    assert result is state.world_sets[component.component_id]
    assert result.status == "oversized"
    assert "2 markets" in (result.reason or "")
    with pytest.raises(MetadataStateError, match="unknown trusted component"):
        state.world_for_component("missing")


def test_orderbooks_alias_preserves_the_existing_book_mapping_contract() -> None:
    state = _catalog_state()

    assert state.orderbooks == state.books == {}
    with pytest.raises(TypeError):
        state.orderbooks["A"] = object()  # type: ignore[assignment]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"markets": (_market("A"), _market("A"))}, "duplicate market identifier"),
        ({"events": (_event(ticker="EMPTY"), _event(ticker="EMPTY"))}, "duplicate event"),
        ({"series": (_series(), _series())}, "duplicate series"),
        (
            {
                "relations": (
                    make_relation(RelationType.IMPLIES, ("A", "B"), relation_id="same"),
                    make_relation(RelationType.IMPLIES, ("A", "B"), relation_id="same"),
                )
            },
            "duplicate relation",
        ),
    ],
)
def test_catalog_identifiers_must_be_unique(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(MetadataStateError, match=message):
        EngineState(**kwargs)  # type: ignore[arg-type]


def test_catalog_rejects_unknown_or_inconsistent_ancestry_and_membership() -> None:
    with pytest.raises(MetadataStateError, match="unknown event"):
        EngineState(markets=(_market("A"),))

    with pytest.raises(MetadataStateError, match="unknown series"):
        EngineState(
            markets=(_market("A"),),
            events=(_event("A"),),
        )

    with pytest.raises(MetadataStateError, match="disagree on series ancestry"):
        EngineState(
            markets=(_market("A", series_ticker="OTHER"),),
            events=(_event("A"),),
            series=(_series(), _series("OTHER")),
        )

    with pytest.raises(MetadataStateError, match="membership"):
        EngineState(
            markets=(_market("A"), _market("B")),
            events=(_event("A"),),
            series=(_series(),),
        )

    with pytest.raises(MetadataStateError, match="unknown markets: B"):
        EngineState(
            markets=(_market("A"),),
            events=(_event("A"),),
            series=(_series(),),
            relations=(make_relation(RelationType.IMPLIES, ("A", "B")),),
        )


def test_market_replacement_is_atomic_and_cannot_change_ancestry() -> None:
    state = _catalog_state()
    replacement = state.markets["A"].model_copy(update={"status": "closed"})

    assert state.replace_market(replacement) is replacement
    assert state.markets["A"].status == "closed"

    invalid = replacement.model_copy(update={"event_ticker": "OTHER"})
    with pytest.raises(MetadataStateError, match="cannot change metadata ancestry"):
        state.replace_market(invalid)
    assert state.markets["A"] is replacement
    with pytest.raises(MetadataStateError, match="unknown market"):
        state.replace_market(_market("UNKNOWN"))


def test_event_fee_uncertainty_blocks_fee_inputs_until_complete_event_replacement() -> None:
    state = _catalog_state()
    component = state.graph.components[0]
    original_event, parent_series = state.fee_metadata_for_market("A")
    assert original_event.ticker == "EVENT"
    assert parent_series.ticker == "SERIES"

    uncertain = state.mark_event_fee_uncertain("EVENT", reason="exchange fee update")

    assert uncertain.affected_tickers == ("A", "B", "C", "D")
    assert uncertain.affected_component_ids == (component.component_id,)
    assert not uncertain.is_certain
    assert state.event_fee_uncertainties == {"EVENT": "exchange fee update"}
    assert not state.is_fee_ready_for_market("A")
    with pytest.raises(MetadataStateError, match="fee metadata is uncertain"):
        state.fee_metadata_for_market("A")

    incomplete = state.events["EVENT"].model_copy(update={"market_tickers": ("A", "B")})
    with pytest.raises(MetadataStateError, match="membership"):
        state.replace_event(incomplete)
    assert not state.is_fee_ready_for_market("A")
    assert "EVENT" in state.event_fee_uncertainties

    refreshed = state.events["EVENT"].model_copy(
        update={
            "fee_type_override": "quadratic",
            "fee_multiplier_override": Decimal("0.5"),
        }
    )
    completed = state.replace_event(refreshed)

    assert completed.is_certain
    assert completed.affected_tickers == ("A", "B", "C", "D")
    assert state.event_fee_uncertainties == {}
    assert state.is_fee_ready_for_market("A")
    assert state.fee_metadata_for_market("A")[0] is refreshed


def test_event_fee_change_ancestry_is_validated_before_catalog_or_replacement_use() -> None:
    wrong_change = EventFeeChange(
        change_id="change",
        event_ticker="OTHER",
        series_ticker="SERIES",
        scheduled_ts=NOW,
        fee_type_override="quadratic",
        fee_multiplier_override=Decimal("1"),
        raw={},
    )
    with pytest.raises(MetadataStateError, match="fee change for another event"):
        EngineState(
            markets=(_market("A"),),
            events=(_event("A", fee_changes=(wrong_change,)),),
            series=(_series(),),
        )


def test_fee_readiness_is_false_for_unknown_or_seriesless_markets() -> None:
    market = _market("A", series_ticker=None)
    state = EngineState(
        markets=(market,),
        events=(_event("A", series_ticker=None),),
    )

    assert not state.is_fee_ready_for_market("A")
    assert not state.is_fee_ready_for_market("UNKNOWN")
    with pytest.raises(MetadataStateError, match="no series"):
        state.fee_metadata_for_market("A")
    with pytest.raises(MetadataStateError, match="unknown event"):
        state.mark_event_fee_uncertain("UNKNOWN")
