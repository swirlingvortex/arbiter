"""Semantic cache, review trust boundary, and staleness persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from arbiter.models.event import Event
from arbiter.models.market import Market
from arbiter.models.relation import Relation, RelationType
from arbiter.models.series import Series
from arbiter.relations.semantic import (
    SEMANTIC_PROMPT_VERSION,
    SemanticEmbedding,
    SemanticMarketDocument,
    SemanticRelationProposal,
    SemanticReviewState,
    SemanticSuggestion,
    build_semantic_document,
    relation_from_semantic_suggestion,
    semantic_suggestion_id,
)
from arbiter.storage.duckdb import DuckDBRepository, StorageError
from arbiter.storage.migrations import MIGRATIONS

NOW = datetime(2026, 9, 3, 12, tzinfo=UTC)


def _catalog(
    *,
    rules_a: str = "Alpha resolving YES necessarily makes Beta resolve YES.",
    include_a: bool = True,
) -> tuple[tuple[Market, ...], tuple[Event, ...], tuple[Series, ...]]:
    tickers = ("A", "B") if include_a else ("B",)
    markets = tuple(
        Market(
            ticker=ticker,
            event_ticker="EVENT",
            series_ticker="SERIES",
            title="Alpha" if ticker == "A" else "Beta",
            status="open",
            close_time=NOW + timedelta(days=1),
            settlement_ts=NOW + timedelta(days=1),
            rules_primary=(rules_a if ticker == "A" else "Beta uses the official result."),
            raw={},
        )
        for ticker in tickers
    )
    event = Event(
        ticker="EVENT",
        series_ticker="SERIES",
        title="Shared event",
        category="Sports",
        market_tickers=tickers,
        raw={},
    )
    series = Series(
        ticker="SERIES",
        title="Shared series",
        category="Sports",
        contract_terms_url="https://example.invalid/terms",
        raw={},
    )
    return markets, (event,), (series,)


def _documents(
    catalog: tuple[tuple[Market, ...], tuple[Event, ...], tuple[Series, ...]],
) -> dict[str, SemanticMarketDocument]:
    markets, events, series = catalog
    return {
        market.ticker: build_semantic_document(market, event=events[0], series=series[0])
        for market in markets
    }


def _suggestion(
    documents: dict[str, SemanticMarketDocument],
    *,
    proposal: SemanticRelationProposal = SemanticRelationProposal.A_IMPLIES_B,
    created_at: datetime = NOW,
) -> SemanticSuggestion:
    market_a = documents["A"]
    market_b = documents["B"]
    response = {
        "relation": proposal.value,
        "confidence": 0.95,
        "rationale": "The exact resolution rules establish this relation.",
        "requires_review": True,
    }
    return SemanticSuggestion(
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
        embedding_model="embedding-v1",
        cosine_similarity=0.8,
        classifier_provider="fake",
        classifier_model="classifier-v1",
        prompt_version=SEMANTIC_PROMPT_VERSION,
        prompt="Strict fixture prompt with exact rules.",
        raw_response=json.dumps(response, separators=(",", ":")),
        relation=proposal,
        confidence=0.95,
        rationale="The exact resolution rules establish this relation.",
        created_at=created_at,
        updated_at=created_at,
    )


def _seed(
    repository: DuckDBRepository,
) -> tuple[
    tuple[tuple[Market, ...], tuple[Event, ...], tuple[Series, ...]],
    dict[str, SemanticMarketDocument],
    SemanticSuggestion,
]:
    catalog = _catalog()
    documents = _documents(catalog)
    suggestion = _suggestion(documents)
    repository.sync_metadata(*catalog, run_id="metadata-1")
    repository.upsert_semantic_suggestions((suggestion,))
    return catalog, documents, suggestion


def test_migration_six_adds_separate_semantic_tables_and_provenance(tmp_path: Path) -> None:
    semantic_migration = next(migration for migration in MIGRATIONS if migration.version == 6)
    assert semantic_migration.name == "semantic_discovery_review"

    with DuckDBRepository(tmp_path / "semantic-schema.duckdb") as repository:
        repository.migrate()
        tables = {str(row[0]) for row in repository.connection.execute("SHOW TABLES").fetchall()}
        relation_columns = {
            str(row[1])
            for row in repository.connection.execute("PRAGMA table_info('relations')").fetchall()
        }

        assert repository.table_count("semantic_embeddings") == 0
        assert repository.table_count("semantic_suggestions") == 0

    assert {"semantic_embeddings", "semantic_suggestions"} <= tables
    assert {
        "semantic_suggestion_id",
        "semantic_rules_hash",
        "semantic_timing_hash",
    } <= relation_columns


def test_embedding_cache_round_trips_exact_evidence_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    document = _documents(_catalog())["A"]
    embedding = SemanticEmbedding.create(
        market_ticker=document.ticker,
        canonical_text=document.canonical_text,
        canonical_text_hash=document.canonical_text_hash,
        rules_hash=document.rules_hash,
        timing_hash=document.timing_hash,
        provider="fake-local",
        model="embedding-v1",
        vector=(1.0, 2.0),
        created_at=NOW,
    )

    with DuckDBRepository(tmp_path / "embedding.duckdb") as repository:
        repository.upsert_semantic_embeddings((embedding,))
        repository.upsert_semantic_embeddings((embedding,))
        restored = repository.get_semantic_embedding(
            market_ticker="A",
            canonical_text_hash=document.canonical_text_hash,
            provider="fake-local",
            model="embedding-v1",
        )

        assert restored == embedding
        assert repository.list_semantic_embeddings() == (embedding,)
        assert repository.table_count("semantic_embeddings") == 1

        conflicting = embedding.model_copy(update={"vector": (2.0, 1.0)})
        with pytest.raises(StorageError, match="different payload"):
            repository.upsert_semantic_embeddings((conflicting,))
        assert repository.list_semantic_embeddings() == (embedding,)

        repository.connection.execute(
            "UPDATE semantic_embeddings SET payload_hash = ? WHERE embedding_id = ?",
            ("0" * 64, embedding.embedding_id),
        )
        with pytest.raises(StorageError, match="payload hash is invalid"):
            repository.get_semantic_embedding(
                market_ticker="A",
                canonical_text_hash=document.canonical_text_hash,
                provider="fake-local",
                model="embedding-v1",
            )


def test_approval_is_atomic_idempotent_and_generic_upsert_cannot_bypass_review(
    tmp_path: Path,
) -> None:
    with DuckDBRepository(tmp_path / "approve.duckdb") as repository:
        _, documents, suggestion = _seed(repository)
        with pytest.raises(ValueError, match="approved semantic suggestion"):
            relation_from_semantic_suggestion(suggestion, created_at=NOW)

        reviewed = repository.review_semantic_suggestion(
            suggestion.suggestion_id,
            action=SemanticReviewState.APPROVED,
            documents=documents,
            reviewed_at=NOW + timedelta(seconds=1),
        )
        repeated = repository.review_semantic_suggestion(
            suggestion.suggestion_id,
            action=SemanticReviewState.APPROVED,
            documents=documents,
            reviewed_at=NOW + timedelta(seconds=2),
        )
        relations = repository.list_relations()
        provenance = repository.connection.execute(
            """
            SELECT semantic_suggestion_id, semantic_rules_hash, semantic_timing_hash
            FROM relations
            """
        ).fetchone()

        assert reviewed.review_state is SemanticReviewState.APPROVED
        assert repeated == reviewed
        assert len(relations) == 1
        assert relations[0].verified is True
        assert relations[0].source == "semantic_verified"
        assert reviewed.approved_relation_id == relations[0].relation_id
        assert provenance is not None
        assert provenance[0] == suggestion.suggestion_id
        assert all(isinstance(value, str) and len(value) == 64 for value in provenance[1:])

        unreviewed_relation = relation_from_semantic_suggestion(reviewed, created_at=NOW)
        with pytest.raises(ValueError, match="only be created by semantic review"):
            repository.upsert_relations((unreviewed_relation,))
        assert repository.table_count("relations") == 1

        rejected = repository.review_semantic_suggestion(
            suggestion.suggestion_id,
            action=SemanticReviewState.REJECTED,
            documents=documents,
            reviewed_at=NOW + timedelta(seconds=3),
        )
        assert rejected.review_state is SemanticReviewState.REJECTED
        assert rejected.approved_relation_id is None
        assert repository.list_relations()[0].verified is False


def test_direct_sql_orphan_semantic_relation_is_deverified_before_load(
    tmp_path: Path,
) -> None:
    with DuckDBRepository(tmp_path / "orphan-relation.duckdb") as repository:
        repository.sync_metadata(*_catalog(), run_id="metadata-1")
        repository.connection.execute(
            """
            INSERT INTO relations (
                relation_id, relation_type, market_tickers_json, antecedent,
                consequent, source, confidence, verified, rationale, created_at,
                updated_at, semantic_suggestion_id, semantic_rules_hash,
                semantic_timing_hash
            ) VALUES (?, 'implies', '["A","B"]', 'A', 'B',
                      'semantic_verified', 0.99, true, ?, ?, ?, NULL, NULL, NULL)
            """,
            ("semantic:direct-sql-orphan", "Unreviewed direct SQL", NOW, NOW),
        )

        relations = repository.list_relations()
        persisted = repository.connection.execute(
            "SELECT verified FROM relations WHERE relation_id = ?",
            ("semantic:direct-sql-orphan",),
        ).fetchone()

        assert len(relations) == 1
        assert relations[0].verified is False
        assert persisted == (False,)


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("semantic_suggestion_id", "semantic:wrong-provenance"),
        ("semantic_rules_hash", "0" * 64),
        ("rationale", "Tampered semantic rationale"),
    ],
)
def test_semantic_relation_audit_revokes_inconsistent_approval_projection(
    tmp_path: Path,
    column: str,
    replacement: str,
) -> None:
    with DuckDBRepository(tmp_path / f"tampered-{column}.duckdb") as repository:
        _, documents, suggestion = _seed(repository)
        reviewed = repository.review_semantic_suggestion(
            suggestion.suggestion_id,
            action=SemanticReviewState.APPROVED,
            documents=documents,
            reviewed_at=NOW + timedelta(seconds=1),
        )
        assert reviewed.approved_relation_id is not None
        repository.connection.execute(
            f"UPDATE relations SET {column} = ? WHERE relation_id = ?",  # noqa: S608
            (replacement, reviewed.approved_relation_id),
        )

        relation = repository.list_relations()[0]
        stale = repository.get_semantic_suggestion(suggestion.suggestion_id)

        assert relation.verified is False
        assert stale is not None
        assert stale.review_state is SemanticReviewState.STALE
        assert stale.stale_reason is not None


def test_semantic_relation_audit_revokes_when_current_db_evidence_changed(
    tmp_path: Path,
) -> None:
    with DuckDBRepository(tmp_path / "audit-current-evidence.duckdb") as repository:
        _, documents, suggestion = _seed(repository)
        repository.review_semantic_suggestion(
            suggestion.suggestion_id,
            action=SemanticReviewState.APPROVED,
            documents=documents,
            reviewed_at=NOW + timedelta(seconds=1),
        )
        repository.connection.execute(
            "UPDATE markets SET rules_primary = ? WHERE ticker = 'A'",
            ("Rules changed without the metadata sync path.",),
        )

        relation = repository.list_relations()[0]
        stale = repository.get_semantic_suggestion(suggestion.suggestion_id)

        assert relation.verified is False
        assert stale is not None
        assert stale.review_state is SemanticReviewState.STALE
        assert stale.stale_reason == "current semantic evidence no longer matches approval"


def test_suggestion_storage_rejects_raw_and_parsed_proposal_disagreement(
    tmp_path: Path,
) -> None:
    documents = _documents(_catalog())
    suggestion = _suggestion(documents)
    conflicting_raw = suggestion.model_copy(
        update={
            "raw_response": json.dumps(
                {
                    "relation": "B_IMPLIES_A",
                    "confidence": suggestion.confidence,
                    "rationale": suggestion.rationale,
                    "requires_review": True,
                },
                separators=(",", ":"),
            )
        }
    )
    with DuckDBRepository(tmp_path / "raw-conflict.duckdb") as repository:
        with pytest.raises(ValueError, match="raw response conflicts"):
            repository.upsert_semantic_suggestions((conflicting_raw,))
        assert repository.table_count("semantic_suggestions") == 0


@pytest.mark.parametrize(
    "action",
    [SemanticReviewState.REJECTED, SemanticReviewState.UNCERTAIN],
)
def test_non_approval_review_states_never_insert_relations(
    tmp_path: Path,
    action: SemanticReviewState,
) -> None:
    with DuckDBRepository(tmp_path / f"{action.value}.duckdb") as repository:
        _, documents, suggestion = _seed(repository)
        reviewed = repository.review_semantic_suggestion(
            suggestion.suggestion_id,
            action=action,
            documents=documents,
            reviewed_at=NOW + timedelta(seconds=1),
        )

        assert reviewed.review_state is action
        assert reviewed.reviewed_at == NOW + timedelta(seconds=1)
        assert reviewed.approved_relation_id is None
        assert repository.table_count("relations") == 0


@pytest.mark.parametrize("missing_market", [False, True])
def test_metadata_sync_automatically_stales_and_deverifies_changed_or_missing_evidence(
    tmp_path: Path,
    missing_market: bool,
) -> None:
    with DuckDBRepository(tmp_path / f"stale-{missing_market}.duckdb") as repository:
        _, documents, suggestion = _seed(repository)
        repository.review_semantic_suggestion(
            suggestion.suggestion_id,
            action=SemanticReviewState.APPROVED,
            documents=documents,
            reviewed_at=NOW + timedelta(seconds=1),
        )
        changed_catalog = (
            _catalog(include_a=False)
            if missing_market
            else _catalog(rules_a="Alpha now uses materially different resolution rules.")
        )

        repository.sync_metadata(*changed_catalog, run_id="metadata-2")

        stale = repository.get_semantic_suggestion(suggestion.suggestion_id)
        relations = repository.list_relations()
        assert stale is not None
        assert stale.review_state is SemanticReviewState.STALE
        assert stale.stale_reason is not None and "A" in stale.stale_reason
        assert len(relations) == 1 and relations[0].verified is False


def test_new_evidence_replaces_a_stale_pair_as_pending_without_restoring_trust(
    tmp_path: Path,
) -> None:
    with DuckDBRepository(tmp_path / "replace-stale.duckdb") as repository:
        _, documents, original = _seed(repository)
        repository.review_semantic_suggestion(
            original.suggestion_id,
            action=SemanticReviewState.APPROVED,
            documents=documents,
            reviewed_at=NOW + timedelta(seconds=1),
        )
        changed_catalog = _catalog(rules_a="Changed exact resolution rules.")
        changed_documents = _documents(changed_catalog)
        repository.sync_metadata(*changed_catalog, run_id="metadata-2")

        replacement = _suggestion(
            changed_documents,
            created_at=NOW + timedelta(seconds=2),
        )
        repository.upsert_semantic_suggestions((replacement,))

        restored = repository.get_semantic_suggestion(original.suggestion_id)
        assert restored == replacement
        assert repository.list_relations()[0].verified is False


class _FailingStalenessSyncRepository(DuckDBRepository):
    fail_after_staleness = False

    def _reconcile_semantic_staleness_in_transaction(
        self,
        documents: Mapping[str, SemanticMarketDocument],
        *,
        updated_at: datetime,
        expected_tickers: set[str],
    ) -> tuple[str, ...]:
        result = super()._reconcile_semantic_staleness_in_transaction(
            documents,
            updated_at=updated_at,
            expected_tickers=expected_tickers,
        )
        if self.fail_after_staleness:
            raise RuntimeError("synthetic failure after semantic staleness update")
        return result


def test_metadata_and_semantic_staleness_share_one_transaction(tmp_path: Path) -> None:
    with _FailingStalenessSyncRepository(tmp_path / "sync-rollback.duckdb") as repository:
        _, documents, suggestion = _seed(repository)
        repository.review_semantic_suggestion(
            suggestion.suggestion_id,
            action=SemanticReviewState.APPROVED,
            documents=documents,
            reviewed_at=NOW + timedelta(seconds=1),
        )
        repository.fail_after_staleness = True

        with pytest.raises(StorageError, match="synthetic failure after semantic staleness"):
            repository.sync_metadata(
                *_catalog(rules_a="Changed rules that must roll back."),
                run_id="metadata-failed",
            )

        restored = repository.get_semantic_suggestion(suggestion.suggestion_id)
        relation = repository.list_relations()[0]
        run = repository.connection.execute(
            "SELECT status FROM metadata_sync_runs WHERE run_id = 'metadata-failed'"
        ).fetchone()
        assert restored is not None
        assert restored.review_state is SemanticReviewState.APPROVED
        assert relation.verified is True
        assert run == ("failed",)


def test_review_rechecks_database_evidence_inside_approval_transaction(tmp_path: Path) -> None:
    with DuckDBRepository(tmp_path / "review-race.duckdb") as repository:
        _, documents, suggestion = _seed(repository)
        repository.review_semantic_suggestion(
            suggestion.suggestion_id,
            action=SemanticReviewState.APPROVED,
            documents=documents,
            reviewed_at=NOW + timedelta(seconds=1),
        )
        repository.connection.execute(
            "UPDATE markets SET rules_primary = ? WHERE ticker = 'A'",
            ("Rules changed after the caller loaded its snapshot.",),
        )

        stale = repository.review_semantic_suggestion(
            suggestion.suggestion_id,
            action=SemanticReviewState.APPROVED,
            documents=documents,
            reviewed_at=NOW + timedelta(seconds=2),
        )

        assert stale.review_state is SemanticReviewState.STALE
        assert "persisted semantic evidence changed" in (stale.stale_reason or "")
        assert repository.list_relations()[0].verified is False


def test_approval_validates_the_complete_trusted_relation_set(tmp_path: Path) -> None:
    with DuckDBRepository(tmp_path / "invalid-component.duckdb") as repository:
        _, documents, _ = _seed(repository)
        suggestion = _suggestion(
            documents,
            proposal=SemanticRelationProposal.EQUIVALENT,
        )
        repository.upsert_semantic_suggestions((suggestion,))
        repository.upsert_relations(
            (
                Relation(
                    relation_id="manual:exactly-one",
                    market_tickers=("A", "B"),
                    relation_type=RelationType.EXACTLY_ONE,
                    source="manual",
                    verified=True,
                    rationale="Exactly one market must resolve YES.",
                    created_at=NOW,
                ),
            )
        )

        with pytest.raises(StorageError, match="impossible_component"):
            repository.review_semantic_suggestion(
                suggestion.suggestion_id,
                action=SemanticReviewState.APPROVED,
                documents=documents,
                reviewed_at=NOW + timedelta(seconds=1),
            )

        restored = repository.get_semantic_suggestion(suggestion.suggestion_id)
        assert restored is not None
        assert restored.review_state is SemanticReviewState.PENDING
        assert repository.table_count("relations") == 1


class _FailingApprovalRepository(DuckDBRepository):
    def _persist_approved_semantic_relation(
        self,
        suggestion: SemanticSuggestion,
        *,
        relation: Relation,
        updated_at: datetime,
    ) -> None:
        super()._persist_approved_semantic_relation(
            suggestion,
            relation=relation,
            updated_at=updated_at,
        )
        raise RuntimeError("synthetic failure after semantic relation insert")


def test_approval_failure_rolls_back_relation_and_review_state(tmp_path: Path) -> None:
    with _FailingApprovalRepository(tmp_path / "rollback.duckdb") as repository:
        _, documents, suggestion = _seed(repository)
        with pytest.raises(StorageError, match="synthetic failure"):
            repository.review_semantic_suggestion(
                suggestion.suggestion_id,
                action=SemanticReviewState.APPROVED,
                documents=documents,
                reviewed_at=NOW + timedelta(seconds=1),
            )

        restored = repository.get_semantic_suggestion(suggestion.suggestion_id)
        assert restored is not None
        assert restored.review_state is SemanticReviewState.PENDING
        assert repository.table_count("relations") == 0
