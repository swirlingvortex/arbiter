"""Canonical semantic documents, compatibility filters, cache, and cosine retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from arbiter.models.event import Event
from arbiter.models.market import Market
from arbiter.models.series import Series
from arbiter.relations.semantic import (
    SemanticEmbedding,
    SemanticMarketDocument,
    build_semantic_document,
    embed_semantic_documents,
    retrieve_semantic_candidates,
    semantic_documents_compatible,
    semantic_embedding_id,
)

NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)


def _document(
    ticker: str,
    *,
    title: str | None = None,
    event_ticker: str = "EVENT",
    series_ticker: str = "SERIES",
    category: str = "Sports",
    close_time: datetime = NOW,
    rules: str | None = None,
) -> SemanticMarketDocument:
    market_title = title or f"Will Team {ticker} win?"
    market = Market(
        ticker=ticker,
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        title=market_title,
        subtitle="Match winner",
        yes_sub_title="Team wins",
        no_sub_title="Team does not win",
        status="open",
        open_time=close_time - timedelta(days=1),
        close_time=close_time,
        expected_expiration_time=close_time,
        settlement_ts=close_time,
        rules_primary=rules or f"Resolves YES if Team {ticker} wins.",
        rules_secondary="Official league statistics are final.",
        raw={},
    )
    event = Event(
        ticker=event_ticker,
        series_ticker=series_ticker,
        title=f"Team {ticker} match",
        category=category,
        raw={},
    )
    series = Series(
        ticker=series_ticker,
        title="League matches",
        category=category,
        tags=("league", "match"),
        contract_terms_url="https://example.invalid/terms",
        raw={},
    )
    return build_semantic_document(market, event=event, series=series)


def test_canonical_document_is_stable_and_hashes_rules_and_timing_separately() -> None:
    original = _document("A", rules="Resolves YES if  Team A wins.\r\nOfficial result applies.")
    equivalent_whitespace = _document(
        "A",
        rules="  Resolves YES if  Team A wins.\nOfficial result applies.  ",
    )
    changed_title = _document(
        "A",
        title="Does Team A win the match?",
        rules="Resolves YES if  Team A wins.\nOfficial result applies.",
    )
    changed_rules = _document("A", rules="Resolves YES if Team A wins after overtime.")
    changed_time = _document("A", close_time=NOW + timedelta(days=1))

    assert original == equivalent_whitespace
    assert "rules_primary" in original.rules_text
    assert "Team A wins" in original.rules_text
    assert original.canonical_text_hash != changed_title.canonical_text_hash
    assert original.rules_hash == changed_title.rules_hash
    assert original.timing_hash == changed_title.timing_hash
    assert original.rules_hash != changed_rules.rules_hash
    assert original.timing_hash != changed_time.timing_hash
    assert original.canonical_text_hash != changed_rules.canonical_text_hash
    assert original.canonical_text_hash != changed_time.canonical_text_hash


def test_document_builder_rejects_inconsistent_ancestry_and_naive_times() -> None:
    market = Market(
        ticker="A",
        event_ticker="EVENT",
        series_ticker="SERIES",
        title="A",
        status="open",
        close_time=NOW.replace(tzinfo=None),
        raw={},
    )
    event = Event(
        ticker="EVENT",
        series_ticker="SERIES",
        title="Event",
        raw={},
    )
    series = Series(ticker="SERIES", title="Series", raw={})

    with pytest.raises(ValueError, match="timezone-aware"):
        build_semantic_document(market, event=event, series=series)
    with pytest.raises(ValueError, match="event identity"):
        build_semantic_document(
            market.model_copy(update={"close_time": NOW}),
            event=event.model_copy(update={"ticker": "OTHER"}),
            series=series,
        )


def test_compatibility_is_conservative_about_category_entity_and_timing() -> None:
    same_series_a = _document("A", title="Will Argentina win?")
    same_series_b = _document("B")
    different_category = _document("C", category="Politics")
    different_day = _document("D", close_time=NOW + timedelta(days=2))
    category_only_related = _document(
        "E",
        title="Will Argentina qualify?",
        event_ticker="OTHER-EVENT",
        series_ticker="OTHER-SERIES",
    )
    category_only_unrelated = _document(
        "F",
        title="Will Player Z qualify?",
        event_ticker="THIRD-EVENT",
        series_ticker="THIRD-SERIES",
    )

    assert semantic_documents_compatible(same_series_a, same_series_b)
    assert not semantic_documents_compatible(same_series_a, same_series_a)
    assert not semantic_documents_compatible(same_series_a, different_category)
    assert not semantic_documents_compatible(same_series_a, different_day)
    assert semantic_documents_compatible(same_series_a, category_only_related)
    assert not semantic_documents_compatible(same_series_a, category_only_unrelated)


def test_cosine_top_k_is_finite_deduplicated_and_deterministic_on_ties() -> None:
    documents = tuple(_document(ticker) for ticker in ("C", "A", "B"))
    embeddings = {
        "A": (1.0, 0.0),
        "B": (1.0, 1.0),
        "C": (1.0, -1.0),
    }

    first = retrieve_semantic_candidates(documents, embeddings, top_k=1)
    second = retrieve_semantic_candidates(tuple(reversed(documents)), embeddings, top_k=1)

    assert first == second
    assert [(item.market_a_ticker, item.market_b_ticker) for item in first] == [
        ("A", "B"),
        ("A", "C"),
    ]
    assert all(-1 <= item.cosine_similarity <= 1 for item in first)


@pytest.mark.parametrize(
    ("embeddings", "message"),
    [
        ({"A": (1.0, 0.0)}, "missing semantic embedding"),
        ({"A": (1.0, 0.0), "B": (1.0,)}, "common dimension"),
        ({"A": (0.0, 0.0), "B": (1.0, 0.0)}, "nonzero"),
        ({"A": (float("nan"), 0.0), "B": (1.0, 0.0)}, "finite"),
    ],
)
def test_retrieval_rejects_missing_or_nonnumeric_geometry(
    embeddings: dict[str, tuple[float, ...]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        retrieve_semantic_candidates((_document("A"), _document("B")), embeddings, top_k=1)


class _FakeEmbeddingProvider:
    provider = "fake-local"
    model = "fake-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(texts))
        return tuple((float(index + 1), 1.0) for index, _ in enumerate(texts))


def test_embedding_cache_reuses_exact_text_model_and_provider_identity() -> None:
    market_a = _document("A")
    market_b = _document("B")
    provider = _FakeEmbeddingProvider()
    cached = SemanticEmbedding.create(
        market_ticker="A",
        canonical_text=market_a.canonical_text,
        canonical_text_hash=market_a.canonical_text_hash,
        rules_hash=market_a.rules_hash,
        timing_hash=market_a.timing_hash,
        provider=provider.provider,
        model=provider.model,
        vector=(9.0, 1.0),
        created_at=NOW,
    )

    resolved = embed_semantic_documents(
        (market_b, market_a),
        provider,
        cached_embeddings=(cached,),
        clock=lambda: NOW,
    )

    assert provider.calls == [(market_b.canonical_text,)]
    assert [item.market_ticker for item in resolved] == ["A", "B"]
    assert resolved[0] is cached
    assert resolved[1].embedding_id == semantic_embedding_id(
        "B",
        market_b.canonical_text_hash,
        provider.provider,
        provider.model,
    )
    assert resolved[1].canonical_text == market_b.canonical_text
    assert resolved[1].rules_hash == market_b.rules_hash
    assert resolved[1].timing_hash == market_b.timing_hash


def test_embedding_provider_wrong_row_count_fails_closed() -> None:
    provider = _FakeEmbeddingProvider()
    provider.embed = lambda texts: ()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="wrong row count"):
        embed_semantic_documents((_document("A"),), provider, clock=lambda: NOW)
