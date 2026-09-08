"""Semantic CLI composition, disabled isolation, and fake-provider review flow."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import pytest
from typer.testing import CliRunner

from arbiter import cli
from arbiter.cli import app
from arbiter.kalshi.normalize import normalize_event, normalize_market, normalize_series
from arbiter.relations.semantic import (
    SemanticProviderUnavailable,
    SemanticReviewState,
)
from arbiter.storage.duckdb import DuckDBRepository

runner = CliRunner()


def _config(
    tmp_path: Path,
    *,
    semantic_enabled: bool = False,
    provider: str = "disabled",
) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "data"
    db_path = data_dir / "arbiter.duckdb"
    config = tmp_path / "config.yaml"
    config.write_text(
        "relations:\n"
        f"  semantic_enabled: {str(semantic_enabled).lower()}\n"
        "  nearest_neighbors: 1\n"
        "  discover_exchange: false\n"
        "  discover_thresholds: false\n"
        "storage:\n"
        f"  data_dir: {data_dir}\n"
        f"  db_path: {db_path}\n"
        "semantic:\n"
        f"  provider: {provider}\n"
        "  model: fixture-classifier\n"
        "  embedding_model: fixture-embedding\n"
        "  prompt_version: semantic-relations-v1\n",
        encoding="utf-8",
    )
    env_file = tmp_path / "missing.env"
    return config, db_path, env_file


def _seed_metadata(db_path: Path) -> None:
    close_time = "2026-09-04T12:00:00Z"
    market_payloads = (
        {
            "ticker": "A",
            "event_ticker": "EVENT",
            "title": "Alpha outcome",
            "yes_sub_title": "Alpha happens",
            "no_sub_title": "Alpha does not happen",
            "status": "open",
            "open_time": "2026-09-03T12:00:00Z",
            "close_time": close_time,
            "settlement_ts": close_time,
            "rules_primary": "ALPHA_EXACT_RULE_TOKEN implies the shared result.",
            "rules_secondary": "The official source is final.",
        },
        {
            "ticker": "B",
            "event_ticker": "EVENT",
            "title": "Beta outcome",
            "yes_sub_title": "Beta happens",
            "no_sub_title": "Beta does not happen",
            "status": "open",
            "open_time": "2026-09-03T12:00:00Z",
            "close_time": close_time,
            "settlement_ts": close_time,
            "rules_primary": "BETA_EXACT_RULE_TOKEN uses the shared result.",
            "rules_secondary": "The official source is final.",
        },
    )
    markets = tuple(
        normalize_market(payload, series_ticker="SERIES") for payload in market_payloads
    )
    event = normalize_event(
        {
            "event_ticker": "EVENT",
            "series_ticker": "SERIES",
            "title": "Shared event",
            "category": "Sports",
            "markets": market_payloads,
        }
    )
    series = normalize_series(
        {
            "ticker": "SERIES",
            "title": "Shared series",
            "category": "Sports",
            "contract_terms_url": "https://example.invalid/terms",
        }
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with DuckDBRepository(db_path) as repository:
        repository.sync_metadata(markets, (event,), (series,), run_id="seed")


def _manual_file(tmp_path: Path) -> Path:
    manual = tmp_path / "relations.yaml"
    manual.write_text(
        "relations:\n"
        "  - type: implies\n"
        "    antecedent: A\n"
        "    consequent: B\n"
        "    rationale: Alpha always entails Beta.\n",
        encoding="utf-8",
    )
    return manual


def _base_args(config: Path, env_file: Path) -> list[str]:
    return ["--config", str(config), "--env-file", str(env_file)]


def test_default_disabled_discovery_constructs_and_calls_no_semantic_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, db_path, env_file = _config(tmp_path)
    _seed_metadata(db_path)
    manual = _manual_file(tmp_path)
    calls: list[str] = []

    def forbidden_classifier(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        calls.append("classifier")
        raise AssertionError("disabled mode constructed a classifier")

    def forbidden_embedder(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        calls.append("embedder")
        raise AssertionError("disabled mode constructed an embedder")

    monkeypatch.setattr(cli, "_new_semantic_classifier", forbidden_classifier)
    monkeypatch.setattr(cli, "_new_semantic_embedding_provider", forbidden_embedder)

    result = runner.invoke(
        app,
        [
            "relations",
            "discover",
            "--manual",
            str(manual),
            *_base_args(config, env_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == []
    assert "Semantic discovery" not in result.output
    with DuckDBRepository(db_path) as repository:
        assert len(repository.list_relations()) == 1
        assert repository.table_count("semantic_embeddings") == 0
        assert repository.table_count("semantic_suggestions") == 0


def test_explicit_unavailable_provider_is_nonfatal_after_deterministic_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, db_path, env_file = _config(tmp_path, provider="fixture-provider")
    _seed_metadata(db_path)
    manual = _manual_file(tmp_path)

    def unavailable(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise SemanticProviderUnavailable("fixture provider unavailable")

    monkeypatch.setattr(cli, "_new_semantic_classifier", unavailable)

    result = runner.invoke(
        app,
        [
            "relations",
            "discover",
            "--manual",
            str(manual),
            "--semantic",
            *_base_args(config, env_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "upserted 1 trusted relations" in result.output
    assert "Semantic discovery skipped" in result.output
    assert "fixture provider unavailable" in result.output
    with DuckDBRepository(db_path) as repository:
        relations = repository.list_relations()
        assert len(relations) == 1
        assert relations[0].source == "manual"
        assert repository.table_count("semantic_suggestions") == 0


class _FakeEmbedder:
    provider = "fake-local"
    model = "fixture-embedding"

    def __init__(self, calls: list[tuple[str, ...]]) -> None:
        self.calls = calls

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(texts))
        return tuple((1.0, float(index + 1)) for index, _ in enumerate(texts))


class _FakeClassifier:
    provider = "fake-provider"
    model = "fixture-classifier"

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def classify(self, prompt: str) -> str:
        self.calls.append(prompt)
        return json.dumps(
            {
                "relation": "MUTUALLY_EXCLUSIVE",
                "confidence": 1.0,
                "rationale": "The exact fixture rules establish mutual exclusion.",
                "requires_review": True,
            },
            separators=(",", ":"),
        )


def test_configured_fake_discovery_uses_cache_and_review_requires_explicit_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, db_path, env_file = _config(
        tmp_path,
        semantic_enabled=True,
        provider="fake-provider",
    )
    _seed_metadata(db_path)
    embedding_calls: list[tuple[str, ...]] = []
    classifier_calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "_new_semantic_embedding_provider",
        lambda settings: _FakeEmbedder(embedding_calls),
    )
    monkeypatch.setattr(
        cli,
        "_new_semantic_classifier",
        lambda settings: _FakeClassifier(classifier_calls),
    )

    args = ["relations", "discover", "--limit", "1", *_base_args(config, env_file)]
    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "persisted 1 pending suggestion" in first.output
    assert len(embedding_calls) == 1
    assert len(embedding_calls[0]) == 2
    assert len(classifier_calls) == 2
    with DuckDBRepository(db_path) as repository:
        suggestions = repository.list_semantic_suggestions()
        assert len(suggestions) == 1
        suggestion = suggestions[0]
        assert suggestion.review_state is SemanticReviewState.PENDING
        assert repository.table_count("semantic_embeddings") == 2
        assert repository.table_count("relations") == 0

    listed = runner.invoke(
        app,
        ["relations", "review", *_base_args(config, env_file)],
        env={"COLUMNS": "300"},
    )
    missing_action = runner.invoke(
        app,
        [
            "relations",
            "review",
            "--suggestion-id",
            suggestion.suggestion_id,
            *_base_args(config, env_file),
        ],
    )
    approved = runner.invoke(
        app,
        [
            "relations",
            "review",
            "--suggestion-id",
            suggestion.suggestion_id,
            "--action",
            "approve",
            *_base_args(config, env_file),
        ],
        env={"COLUMNS": "300"},
    )

    assert listed.exit_code == 0, listed.output
    assert "ALPHA_EXACT_RULE_TOKEN" in listed.output
    assert "BETA_EXACT_RULE_TOKEN" in listed.output
    assert "MUTUALLY_EXCLUSIVE" in listed.output
    assert missing_action.exit_code == 2
    assert "must be provided together" in missing_action.output
    assert approved.exit_code == 0, approved.output
    assert "is now approved" in approved.output
    with DuckDBRepository(db_path) as repository:
        relations = repository.list_relations()
        assert len(relations) == 1
        assert relations[0].source == "semantic_verified"
        assert relations[0].verified is True
