"""Strict semantic-classifier parsing, audit models, and lazy-adapter tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from arbiter.models.event import Event
from arbiter.models.market import Market
from arbiter.models.relation import RelationType
from arbiter.models.series import Series
from arbiter.relations import semantic
from arbiter.relations.semantic import (
    SEMANTIC_PROMPT_VERSION,
    OpenAICompatibleSemanticClassifier,
    SemanticParseError,
    SemanticProviderUnavailable,
    SemanticRelationProposal,
    SemanticReviewState,
    SemanticSuggestion,
    SentenceTransformerEmbeddingProvider,
    build_classifier_prompt,
    build_semantic_document,
    parse_classifier_output,
    relation_from_semantic_suggestion,
    semantic_suggestion_id,
    semantic_suggestion_payload_hash,
    semantic_verified_relation_id,
)

NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)


def _document(ticker: str, title: str) -> semantic.SemanticMarketDocument:
    market = Market(
        ticker=ticker,
        event_ticker="EVENT",
        series_ticker="SERIES",
        title=title,
        yes_sub_title="Yes subtitle",
        no_sub_title="No subtitle",
        status="open",
        close_time=NOW,
        settlement_ts=NOW,
        rules_primary=f"{title} resolves YES under the stated condition.",
        rules_secondary="The official source is final.",
        raw={},
    )
    event = Event(
        ticker="EVENT",
        series_ticker="SERIES",
        title="Shared event",
        category="Sports",
        raw={},
    )
    series = Series(ticker="SERIES", title="Shared series", category="Sports", raw={})
    return build_semantic_document(market, event=event, series=series)


def _suggestion(
    proposal: SemanticRelationProposal,
    *,
    review_state: SemanticReviewState = SemanticReviewState.PENDING,
) -> SemanticSuggestion:
    market_a = _document("A", "Alpha")
    market_b = _document("B", "Beta")
    pending = SemanticSuggestion(
        suggestion_id=semantic_suggestion_id("A", "B"),
        market_a_ticker="A",
        market_b_ticker="B",
        market_a_text_hash=market_a.canonical_text_hash,
        market_b_text_hash=market_b.canonical_text_hash,
        market_a_rules_hash=market_a.rules_hash,
        market_b_rules_hash=market_b.rules_hash,
        market_a_timing_hash=market_a.timing_hash,
        market_b_timing_hash=market_b.timing_hash,
        market_a_title=market_a.title,
        market_b_title=market_b.title,
        market_a_rules_text=market_a.rules_text,
        market_b_rules_text=market_b.rules_text,
        market_a_timing_text=market_a.timing_text,
        market_b_timing_text=market_b.timing_text,
        embedding_provider="fake-local",
        embedding_model="fixture-embedding-v1",
        cosine_similarity=0.75,
        classifier_provider="fake",
        classifier_model="fixture-classifier-v1",
        prompt_version=SEMANTIC_PROMPT_VERSION,
        prompt="strict fixture prompt",
        raw_response=json.dumps(
            {
                "relation": proposal.value,
                "confidence": 0.99,
                "rationale": "Exact rules establish the proposed relation.",
                "requires_review": True,
            }
        ),
        relation=proposal,
        confidence=0.99,
        rationale="Exact rules establish the proposed relation.",
        created_at=NOW,
        updated_at=NOW,
    )
    if review_state is SemanticReviewState.PENDING:
        return pending
    return SemanticSuggestion(
        **{
            **pending.model_dump(),
            "review_state": review_state,
            "reviewed_at": NOW,
            "approved_relation_id": (
                semantic_verified_relation_id(pending)
                if review_state is SemanticReviewState.APPROVED
                else None
            ),
            "stale_reason": (
                "rules changed" if review_state is SemanticReviewState.STALE else None
            ),
        }
    )


def test_parser_accepts_only_the_documented_classifier_shape() -> None:
    raw = json.dumps(
        {
            "relation": "A_IMPLIES_B",
            "confidence": 0.94,
            "rationale": "Every A-YES resolution also resolves B YES.",
            "requires_review": True,
        },
        separators=(",", ":"),
    )

    parsed = parse_classifier_output(raw)

    assert parsed.relation is SemanticRelationProposal.A_IMPLIES_B
    assert parsed.confidence == 0.94
    assert parsed.requires_review is True


@pytest.mark.parametrize(
    "raw",
    [
        "",
        " {}",
        "{}\n",
        "not-json",
        "[]",
        "```json\n{}\n```",
        ('{"relation":"NONE","confidence":0.1,"rationale":"x","requires_review":true,"extra":1}'),
        ('{"relation":"POSSIBLY","confidence":0.1,"rationale":"x","requires_review":true}'),
        ('{"relation":"NONE","confidence":"0.1","rationale":"x","requires_review":true}'),
        ('{"relation":"NONE","confidence":NaN,"rationale":"x","requires_review":true}'),
        ('{"relation":"NONE","confidence":0.1,"rationale":"  ","requires_review":true}'),
        ('{"relation":"NONE","confidence":0.1,"rationale":"x","requires_review":false}'),
        ('{"relation":"NONE","confidence":0.1,"rationale":"x","requires_review":1}'),
        (
            '{"relation":"NONE","relation":"EQUIVALENT","confidence":0.1,'
            '"rationale":"x","requires_review":true}'
        ),
    ],
)
def test_parser_rejects_malformed_wrapped_coerced_or_drifted_output(raw: str) -> None:
    with pytest.raises(SemanticParseError):
        parse_classifier_output(raw)


def test_prompt_contains_exact_evidence_and_fail_closed_instructions() -> None:
    market_a = _document("A", "Lionel Messi scores")
    market_b = _document("B", "Argentina scores")

    prompt = build_classifier_prompt(market_a, market_b)

    assert "EVERY valid resolution state" in prompt
    assert "Probability, correlation" in prompt
    assert "rules-consistent counterexample" in prompt
    assert "requires_review must be true" in prompt
    assert "Lionel Messi scores" in prompt
    assert "Argentina scores" in prompt
    assert market_a.rules_hash in prompt
    assert market_b.rules_hash in prompt


@pytest.mark.parametrize(
    ("proposal", "relation_type", "antecedent", "consequent"),
    [
        (SemanticRelationProposal.A_IMPLIES_B, RelationType.IMPLIES, "A", "B"),
        (SemanticRelationProposal.B_IMPLIES_A, RelationType.IMPLIES, "B", "A"),
        (SemanticRelationProposal.EQUIVALENT, RelationType.EQUIVALENT, None, None),
        (
            SemanticRelationProposal.MUTUALLY_EXCLUSIVE,
            RelationType.MUTUALLY_EXCLUSIVE,
            None,
            None,
        ),
    ],
)
def test_actionable_proposal_maps_to_one_verified_semantic_relation(
    proposal: SemanticRelationProposal,
    relation_type: RelationType,
    antecedent: str | None,
    consequent: str | None,
) -> None:
    relation = relation_from_semantic_suggestion(
        _suggestion(proposal, review_state=SemanticReviewState.APPROVED),
        created_at=NOW,
    )

    assert relation.relation_type is relation_type
    assert relation.antecedent == antecedent
    assert relation.consequent == consequent
    assert relation.source == "semantic_verified"
    assert relation.verified is True
    assert relation.confidence == 0.99


@pytest.mark.parametrize(
    "review_state",
    [
        SemanticReviewState.PENDING,
        SemanticReviewState.REJECTED,
        SemanticReviewState.UNCERTAIN,
        SemanticReviewState.STALE,
    ],
)
def test_only_approved_suggestions_can_project_a_trusted_relation(
    review_state: SemanticReviewState,
) -> None:
    suggestion = _suggestion(
        SemanticRelationProposal.EQUIVALENT,
        review_state=review_state,
    )

    with pytest.raises(ValueError, match="only an approved semantic suggestion"):
        relation_from_semantic_suggestion(suggestion, created_at=NOW)


def test_review_model_rejects_automatic_or_incomplete_approval() -> None:
    values = _suggestion(SemanticRelationProposal.EQUIVALENT).model_dump()

    with pytest.raises(ValidationError, match="require review evidence"):
        SemanticSuggestion(
            **{
                **values,
                "review_state": SemanticReviewState.APPROVED,
                "reviewed_at": None,
                "approved_relation_id": None,
            }
        )
    with pytest.raises(ValidationError, match="cannot be approved"):
        SemanticSuggestion(
            **{
                **_suggestion(SemanticRelationProposal.NONE).model_dump(),
                "review_state": SemanticReviewState.APPROVED,
                "reviewed_at": NOW,
                "approved_relation_id": "relation-id",
            }
        )


@pytest.mark.parametrize(
    "field",
    [
        "market_a_rules_text",
        "market_b_rules_text",
        "market_a_timing_text",
        "market_b_timing_text",
    ],
)
def test_displayed_rule_and_timing_evidence_is_bound_to_its_hash(field: str) -> None:
    values = _suggestion(SemanticRelationProposal.EQUIVALENT).model_dump()

    with pytest.raises(ValidationError, match="hash does not match its displayed evidence"):
        SemanticSuggestion(**{**values, field: f"{values[field]} tampered"})


def test_projection_revalidates_copies_that_bypassed_pydantic_validation() -> None:
    approved = _suggestion(
        SemanticRelationProposal.EQUIVALENT,
        review_state=SemanticReviewState.APPROVED,
    )
    tampered = approved.model_copy(
        update={"market_a_rules_text": f"{approved.market_a_rules_text} tampered"}
    )

    with pytest.raises(ValidationError, match="hash does not match its displayed evidence"):
        relation_from_semantic_suggestion(tampered, created_at=NOW)


def test_projection_requires_the_approved_relation_id_to_match_the_proposal() -> None:
    approved = _suggestion(
        SemanticRelationProposal.A_IMPLIES_B,
        review_state=SemanticReviewState.APPROVED,
    )
    mismatched = SemanticSuggestion(
        **{**approved.model_dump(), "approved_relation_id": "semantic_verified:wrong"}
    )

    with pytest.raises(ValueError, match="relation ID does not match"):
        relation_from_semantic_suggestion(mismatched, created_at=NOW)


def test_suggestion_payload_hash_ignores_only_review_bookkeeping() -> None:
    pending = _suggestion(SemanticRelationProposal.EQUIVALENT)
    approved_values = pending.model_dump()
    approved = SemanticSuggestion(
        **{
            **approved_values,
            "review_state": SemanticReviewState.APPROVED,
            "reviewed_at": NOW,
            "approved_relation_id": "relation-id",
        }
    )

    assert semantic_suggestion_payload_hash(pending) == semantic_suggestion_payload_hash(approved)
    changed = pending.model_copy(update={"confidence": 0.5})
    assert semantic_suggestion_payload_hash(pending) != semantic_suggestion_payload_hash(changed)


def test_optional_adapters_do_not_import_providers_until_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def unavailable(name: str) -> None:
        imported.append(name)
        raise ImportError(name)

    monkeypatch.setattr(semantic.importlib, "import_module", unavailable)

    embedder = SentenceTransformerEmbeddingProvider("fixture-model")
    classifier = OpenAICompatibleSemanticClassifier(
        provider="openai-compatible",
        model="fixture-model",
        api_key="secret",
        base_url="https://example.invalid/v1",
    )

    assert imported == []
    with pytest.raises(SemanticProviderUnavailable):
        embedder.embed(("text",))
    with pytest.raises(SemanticProviderUnavailable):
        classifier.classify("prompt")
    assert imported == ["sentence_transformers", "openai"]
