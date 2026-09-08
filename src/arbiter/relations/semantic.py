"""Conservative semantic-candidate retrieval and strictly parsed proposals.

Optional providers are imported only inside their call paths. Importing this module,
building canonical documents, and running with semantic discovery disabled therefore
cannot import or initialize ``sentence-transformers`` or an OpenAI-compatible client.
"""

from __future__ import annotations

import importlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Protocol, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from arbiter.models.event import Event
from arbiter.models.market import Market
from arbiter.models.relation import Relation, RelationType
from arbiter.models.series import Series
from arbiter.relations.deterministic import canonical_relation_id

SEMANTIC_PROMPT_VERSION = "semantic-relations-v1"
SEMANTIC_VERIFIED_SOURCE = "semantic_verified"


class SemanticError(RuntimeError):
    """Base class for fail-closed semantic discovery errors."""


class SemanticParseError(SemanticError):
    """Raised when classifier output is not one exact supported JSON object."""


class SemanticProviderUnavailable(SemanticError):
    """Raised when an explicitly requested optional provider cannot be loaded."""


class SemanticProviderError(SemanticError):
    """Raised when an optional provider returns unusable data or fails."""


class SemanticRelationProposal(StrEnum):
    """Only pairwise proposals allowed at the semantic trust boundary."""

    A_IMPLIES_B = "A_IMPLIES_B"
    B_IMPLIES_A = "B_IMPLIES_A"
    EQUIVALENT = "EQUIVALENT"
    MUTUALLY_EXCLUSIVE = "MUTUALLY_EXCLUSIVE"
    NONE = "NONE"
    UNCERTAIN = "UNCERTAIN"

    @property
    def actionable(self) -> bool:
        """Whether a human could turn this proposal into a supported relation."""

        return self not in {self.NONE, self.UNCERTAIN}


class SemanticReviewState(StrEnum):
    """Durable states for a semantic suggestion's mandatory human review."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    STALE = "stale"


class SemanticClassifierOutput(BaseModel):
    """The exact classifier response schema; no implicit authorization is encoded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation: SemanticRelationProposal
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    rationale: str = Field(min_length=1)
    requires_review: Literal[True]

    @field_validator("confidence", mode="before")
    @classmethod
    def confidence_must_be_json_number(cls, value: object) -> object:
        """Reject booleans and numeric strings instead of relying on coercion."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("confidence must be a JSON number")
        return value

    @field_validator("rationale")
    @classmethod
    def rationale_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rationale cannot be blank")
        return value

    @field_validator("requires_review", mode="before")
    @classmethod
    def review_flag_must_be_json_boolean(cls, value: object) -> object:
        """Reject truthy numbers or strings at the classifier boundary."""

        if not isinstance(value, bool):
            raise ValueError("requires_review must be a JSON boolean")
        return value


class SemanticMarketDocument(BaseModel):
    """Canonical, hash-addressed market evidence sent to semantic providers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str = Field(min_length=1)
    event_ticker: str = Field(min_length=1)
    series_ticker: str | None = None
    event_category: str | None = None
    series_category: str | None = None
    title: str = Field(min_length=1)
    event_title: str = Field(min_length=1)
    series_title: str | None = None
    rules_text: str = Field(min_length=1)
    timing_text: str = Field(min_length=1)
    canonical_text: str = Field(min_length=1)
    canonical_text_hash: str
    rules_hash: str
    timing_hash: str

    @model_validator(mode="after")
    def hashes_match_evidence(self) -> SemanticMarketDocument:
        expected = {
            "canonical text": (self.canonical_text_hash, _text_hash(self.canonical_text)),
            "rules": (self.rules_hash, _text_hash(self.rules_text)),
            "timing": (self.timing_hash, _text_hash(self.timing_text)),
        }
        for label, (actual, digest) in expected.items():
            if actual != digest:
                raise ValueError(f"{label} hash does not match its evidence")
        return self


class SemanticEmbedding(BaseModel):
    """One cacheable local embedding tied to exact canonical market text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding_id: str = Field(min_length=1)
    market_ticker: str = Field(min_length=1)
    canonical_text: str = Field(min_length=1)
    canonical_text_hash: str
    rules_hash: str
    timing_hash: str
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    vector: tuple[float, ...] = Field(min_length=1)
    created_at: datetime
    updated_at: datetime

    @field_validator("vector", mode="before")
    @classmethod
    def vector_must_be_numeric_sequence(cls, value: object) -> object:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError("embedding vector must be a numeric sequence")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise ValueError("embedding vector must contain only numbers")
        return value

    @model_validator(mode="after")
    def validate_embedding(self) -> SemanticEmbedding:
        if self.canonical_text_hash != _text_hash(self.canonical_text):
            raise ValueError("canonical text hash does not match embedding evidence")
        _require_hash(self.rules_hash, label="rules hash")
        _require_hash(self.timing_hash, label="timing hash")
        expected_id = semantic_embedding_id(
            self.market_ticker,
            self.canonical_text_hash,
            self.provider,
            self.model,
        )
        if self.embedding_id != expected_id:
            raise ValueError("embedding ID does not match its cache identity")
        vector = np.asarray(self.vector, dtype=np.float64)
        if not np.isfinite(vector).all() or float(np.linalg.norm(vector)) == 0:
            raise ValueError("embedding vector must be finite and nonzero")
        created_at = _aware_utc(self.created_at, label="embedding creation time")
        updated_at = _aware_utc(self.updated_at, label="embedding update time")
        if updated_at < created_at:
            raise ValueError("embedding update cannot precede creation")
        return self

    @classmethod
    def create(
        cls,
        *,
        market_ticker: str,
        canonical_text: str,
        canonical_text_hash: str,
        rules_hash: str,
        timing_hash: str,
        provider: str,
        model: str,
        vector: Sequence[float],
        created_at: datetime,
        updated_at: datetime | None = None,
    ) -> SemanticEmbedding:
        """Construct an embedding with its deterministic cache identity."""

        return cls(
            embedding_id=semantic_embedding_id(
                market_ticker,
                canonical_text_hash,
                provider,
                model,
            ),
            market_ticker=market_ticker,
            canonical_text=canonical_text,
            canonical_text_hash=canonical_text_hash,
            rules_hash=rules_hash,
            timing_hash=timing_hash,
            provider=provider,
            model=model,
            vector=tuple(vector),
            created_at=created_at,
            updated_at=created_at if updated_at is None else updated_at,
        )


class SemanticCandidate(BaseModel):
    """One canonical market pair selected for semantic classification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    market_a_ticker: str = Field(min_length=1)
    market_b_ticker: str = Field(min_length=1)
    cosine_similarity: float = Field(ge=-1, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def pair_is_canonical(self) -> SemanticCandidate:
        if self.market_a_ticker >= self.market_b_ticker:
            raise ValueError("semantic candidate tickers must be unique and sorted")
        return self


class SemanticSuggestion(BaseModel):
    """Fully auditable classifier evidence and its independent review state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suggestion_id: str = Field(min_length=1)
    market_a_ticker: str = Field(min_length=1)
    market_b_ticker: str = Field(min_length=1)
    market_a_text_hash: str
    market_b_text_hash: str
    market_a_rules_hash: str
    market_b_rules_hash: str
    market_a_timing_hash: str
    market_b_timing_hash: str
    market_a_title: str = Field(min_length=1)
    market_b_title: str = Field(min_length=1)
    market_a_rules_text: str = Field(min_length=1)
    market_b_rules_text: str = Field(min_length=1)
    market_a_timing_text: str = Field(min_length=1)
    market_b_timing_text: str = Field(min_length=1)
    embedding_provider: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    cosine_similarity: float = Field(ge=-1, le=1, allow_inf_nan=False)
    classifier_provider: str = Field(min_length=1)
    classifier_model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    raw_response: str = Field(min_length=1)
    relation: SemanticRelationProposal
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    rationale: str = Field(min_length=1)
    requires_review: Literal[True] = True
    review_state: SemanticReviewState = SemanticReviewState.PENDING
    created_at: datetime
    updated_at: datetime
    reviewed_at: datetime | None = None
    approved_relation_id: str | None = None
    stale_reason: str | None = None

    @model_validator(mode="after")
    def validate_suggestion(self) -> SemanticSuggestion:
        if self.market_a_ticker >= self.market_b_ticker:
            raise ValueError("semantic suggestion tickers must be unique and sorted")
        if self.suggestion_id != semantic_suggestion_id(
            self.market_a_ticker,
            self.market_b_ticker,
        ):
            raise ValueError("suggestion ID does not match its canonical market pair")
        for label, digest in (
            ("market A text hash", self.market_a_text_hash),
            ("market B text hash", self.market_b_text_hash),
            ("market A rules hash", self.market_a_rules_hash),
            ("market B rules hash", self.market_b_rules_hash),
            ("market A timing hash", self.market_a_timing_hash),
            ("market B timing hash", self.market_b_timing_hash),
        ):
            _require_hash(digest, label=label)
        for label, digest, evidence in (
            ("market A rules", self.market_a_rules_hash, self.market_a_rules_text),
            ("market B rules", self.market_b_rules_hash, self.market_b_rules_text),
            ("market A timing", self.market_a_timing_hash, self.market_a_timing_text),
            ("market B timing", self.market_b_timing_hash, self.market_b_timing_text),
        ):
            if digest != _text_hash(evidence):
                raise ValueError(f"{label} hash does not match its displayed evidence")
        if not self.rationale.strip():
            raise ValueError("suggestion rationale cannot be blank")
        created_at = _aware_utc(self.created_at, label="suggestion creation time")
        updated_at = _aware_utc(self.updated_at, label="suggestion update time")
        if updated_at < created_at:
            raise ValueError("suggestion update cannot precede creation")
        if self.reviewed_at is not None:
            _aware_utc(self.reviewed_at, label="suggestion review time")

        if self.review_state is SemanticReviewState.PENDING:
            if any(
                item is not None
                for item in (self.reviewed_at, self.approved_relation_id, self.stale_reason)
            ):
                raise ValueError("pending suggestions cannot contain review results")
        elif self.review_state is SemanticReviewState.APPROVED:
            if not self.relation.actionable:
                raise ValueError("NONE or UNCERTAIN cannot be approved")
            if self.reviewed_at is None or not self.approved_relation_id:
                raise ValueError("approved suggestions require review evidence and a relation ID")
            if self.stale_reason is not None:
                raise ValueError("an approved suggestion cannot be stale")
        elif self.review_state in {
            SemanticReviewState.REJECTED,
            SemanticReviewState.UNCERTAIN,
        }:
            if self.reviewed_at is None or self.approved_relation_id is not None:
                raise ValueError("non-approved reviews require a review time and no relation ID")
            if self.stale_reason is not None:
                raise ValueError("a completed non-approved review cannot have a stale reason")
        elif not self.stale_reason or not self.stale_reason.strip():
            raise ValueError("stale suggestions require an explicit reason")
        return self


class EmbeddingProvider(Protocol):
    """Minimal injectable interface used by offline deterministic tests."""

    provider: str
    model: str

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one finite vector for every input text."""


class SemanticClassifier(Protocol):
    """Minimal OpenAI-compatible or fake classifier interface."""

    provider: str
    model: str

    def classify(self, prompt: str) -> str:
        """Return the raw JSON response for durable parsing and storage."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _text_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _require_hash(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized or None


def _embedding_text(value: str | None) -> str:
    normalized = _optional_text(value)
    return "" if normalized is None else " ".join(normalized.split())


def _timestamp_text(value: datetime | None, *, label: str) -> str | None:
    if value is None:
        return None
    return _aware_utc(value, label=label).isoformat()


def _labeled_lines(items: Sequence[tuple[str, object]]) -> str:
    return "\n".join(f"{label}: {_canonical_json(value)}" for label, value in items)


def build_semantic_document(
    market: Market,
    *,
    event: Event,
    series: Series | None,
) -> SemanticMarketDocument:
    """Build exact display evidence and stable hashes from normalized metadata."""

    if market.event_ticker != event.ticker:
        raise ValueError("semantic market/event identity mismatch")
    expected_series = market.series_ticker or event.series_ticker
    if (
        market.series_ticker is not None
        and event.series_ticker is not None
        and market.series_ticker != event.series_ticker
    ):
        raise ValueError("semantic market/event series identity mismatch")
    if series is not None and expected_series != series.ticker:
        raise ValueError("semantic series identity mismatch")
    if series is None and expected_series is not None:
        raise ValueError("semantic document requires referenced series metadata")

    settlement_sources = []
    if series is not None:
        settlement_sources = [
            {"name": _optional_text(source.name), "url": _optional_text(source.url)}
            for source in sorted(
                series.settlement_sources,
                key=lambda source: (source.name.casefold(), source.url.casefold()),
            )
        ]
    rules_payload = {
        "contract_terms_url": None if series is None else _optional_text(series.contract_terms_url),
        "contract_url": None if series is None else _optional_text(series.contract_url),
        "early_close_condition": _optional_text(market.early_close_condition),
        "rules_primary": _optional_text(market.rules_primary),
        "rules_secondary": _optional_text(market.rules_secondary),
        "settlement_sources": settlement_sources,
    }
    timing_payload = {
        "close_time": _timestamp_text(market.close_time, label="market close time"),
        "expected_expiration_time": _timestamp_text(
            market.expected_expiration_time,
            label="market expected expiration time",
        ),
        "expiration_time": _timestamp_text(
            market.expiration_time,
            label="market expiration time",
        ),
        "latest_expiration_time": _timestamp_text(
            market.latest_expiration_time,
            label="market latest expiration time",
        ),
        "occurrence_datetime": _timestamp_text(
            market.occurrence_datetime,
            label="market occurrence time",
        ),
        "open_time": _timestamp_text(market.open_time, label="market open time"),
        "settlement_ts": _timestamp_text(market.settlement_ts, label="market settlement time"),
    }
    rules_text = _canonical_json(rules_payload)
    timing_text = _canonical_json(timing_payload)
    canonical_text = _labeled_lines(
        (
            ("market_title", _embedding_text(market.title)),
            ("market_subtitle", _embedding_text(market.subtitle)),
            ("yes_subtitle", _embedding_text(market.yes_sub_title)),
            ("no_subtitle", _embedding_text(market.no_sub_title)),
            ("event_title", _embedding_text(event.title)),
            ("event_subtitle", _embedding_text(event.subtitle)),
            ("series_title", "" if series is None else _embedding_text(series.title)),
            ("event_category", _embedding_text(event.category)),
            ("series_category", "" if series is None else _embedding_text(series.category)),
            (
                "series_tags",
                [] if series is None else sorted(_embedding_text(tag) for tag in series.tags),
            ),
            ("rules", json.loads(rules_text)),
            ("timing", json.loads(timing_text)),
        )
    )
    return SemanticMarketDocument(
        ticker=market.ticker,
        event_ticker=event.ticker,
        series_ticker=None if series is None else series.ticker,
        event_category=_optional_text(event.category),
        series_category=None if series is None else _optional_text(series.category),
        title=market.title,
        event_title=event.title,
        series_title=None if series is None else series.title,
        rules_text=rules_text,
        timing_text=timing_text,
        canonical_text=canonical_text,
        canonical_text_hash=_text_hash(canonical_text),
        rules_hash=_text_hash(rules_text),
        timing_hash=_text_hash(timing_text),
    )


_TOKEN = re.compile(r"[a-z0-9]+")
_GENERIC_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "be",
        "before",
        "by",
        "contract",
        "event",
        "for",
        "happen",
        "in",
        "is",
        "league",
        "match",
        "matches",
        "market",
        "no",
        "of",
        "on",
        "or",
        "score",
        "team",
        "the",
        "to",
        "will",
        "win",
        "yes",
    }
)


def _categories(document: SemanticMarketDocument) -> set[str]:
    return {
        normalized.casefold()
        for value in (document.event_category, document.series_category)
        if (normalized := _optional_text(value)) is not None
    }


def _entity_tokens(document: SemanticMarketDocument) -> set[str]:
    source = " ".join(
        value
        for value in (document.title, document.event_title, document.series_title)
        if value is not None
    ).casefold()
    return {token for token in _TOKEN.findall(source) if token not in _GENERIC_TOKENS}


def _timing_payload(document: SemanticMarketDocument) -> dict[str, str | None]:
    try:
        raw = json.loads(document.timing_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - model construction guards hashes
        raise ValueError("semantic timing evidence is invalid JSON") from exc
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or (value is not None and not isinstance(value, str))
        for key, value in raw.items()
    ):
        raise ValueError("semantic timing evidence has an invalid shape")
    return cast(dict[str, str | None], raw)


def _parsed_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    return _aware_utc(parsed, label="canonical semantic time")


def _effective_end(document: SemanticMarketDocument) -> datetime | None:
    timing = _timing_payload(document)
    for key in (
        "settlement_ts",
        "expected_expiration_time",
        "expiration_time",
        "latest_expiration_time",
        "close_time",
    ):
        if (result := _parsed_time(timing.get(key))) is not None:
            return result
    return None


def semantic_documents_compatible(
    market_a: SemanticMarketDocument,
    market_b: SemanticMarketDocument,
) -> bool:
    """Apply conservative hard filters before a pair reaches a classifier."""

    if market_a.ticker == market_b.ticker:
        return False
    categories_a = _categories(market_a)
    categories_b = _categories(market_b)
    if categories_a and categories_b and categories_a.isdisjoint(categories_b):
        return False

    same_event = market_a.event_ticker == market_b.event_ticker
    same_series = (
        market_a.series_ticker is not None and market_a.series_ticker == market_b.series_ticker
    )
    related_category = bool(categories_a & categories_b)
    if not (same_event or same_series or related_category):
        return False
    if not (same_event or same_series) and not (
        _entity_tokens(market_a) & _entity_tokens(market_b)
    ):
        return False

    end_a = _effective_end(market_a)
    end_b = _effective_end(market_b)
    if end_a is not None and end_b is not None and end_a.date() != end_b.date():
        return False
    timing_a = _timing_payload(market_a)
    timing_b = _timing_payload(market_b)
    open_a = _parsed_time(timing_a.get("open_time"))
    open_b = _parsed_time(timing_b.get("open_time"))
    if end_a is not None and open_b is not None and end_a < open_b:
        return False
    return not (end_b is not None and open_a is not None and end_b < open_a)


def _numeric_vector(value: Sequence[float], *, label: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a numeric vector")
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric vector") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{label} must be one nonempty vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError(f"{label} must be nonzero")
    return vector


def retrieve_semantic_candidates(
    documents: Sequence[SemanticMarketDocument],
    embeddings: Mapping[str, Sequence[float]],
    *,
    top_k: int,
) -> tuple[SemanticCandidate, ...]:
    """Return the compatible union of each market's deterministic cosine top-k."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("semantic top_k must be a positive integer")
    by_ticker = {document.ticker: document for document in documents}
    if len(by_ticker) != len(documents):
        raise ValueError("semantic documents must have unique tickers")

    vectors: dict[str, np.ndarray] = {}
    dimension: int | None = None
    for ticker in sorted(by_ticker):
        raw = embeddings.get(ticker)
        if raw is None:
            raise ValueError(f"missing semantic embedding for {ticker}")
        vector = _numeric_vector(raw, label=f"embedding for {ticker}")
        if dimension is None:
            dimension = int(vector.size)
        elif vector.size != dimension:
            raise ValueError("semantic embeddings must have one common dimension")
        vectors[ticker] = vector / np.linalg.norm(vector)

    pair_scores: dict[tuple[str, str], float] = {}
    tickers = sorted(by_ticker)
    for ticker in tickers:
        scored: list[tuple[float, str]] = []
        for other in tickers:
            if other == ticker or not semantic_documents_compatible(
                by_ticker[ticker],
                by_ticker[other],
            ):
                continue
            similarity = float(np.dot(vectors[ticker], vectors[other]))
            similarity = max(-1.0, min(1.0, similarity))
            scored.append((similarity, other))
        for similarity, other in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]:
            pair = tuple(sorted((ticker, other)))
            pair_scores[cast(tuple[str, str], pair)] = similarity

    return tuple(
        SemanticCandidate(
            market_a_ticker=market_a,
            market_b_ticker=market_b,
            cosine_similarity=similarity,
        )
        for (market_a, market_b), similarity in sorted(
            pair_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


def semantic_embedding_id(
    market_ticker: str,
    canonical_text_hash: str,
    provider: str,
    model: str,
) -> str:
    """Return the stable identity of one exact provider/model/text cache entry."""

    for label, value in (
        ("market ticker", market_ticker),
        ("embedding provider", provider),
        ("embedding model", model),
    ):
        if not value.strip():
            raise ValueError(f"{label} cannot be blank")
    _require_hash(canonical_text_hash, label="canonical text hash")
    payload = _canonical_json([market_ticker, canonical_text_hash, provider, model])
    return f"semantic-embedding:{_text_hash(payload)}"


def semantic_suggestion_id(market_a_ticker: str, market_b_ticker: str) -> str:
    """Return one stable suggestion identity per unordered market pair."""

    members = tuple(sorted((market_a_ticker, market_b_ticker)))
    if any(not member.strip() for member in members) or members[0] == members[1]:
        raise ValueError("semantic suggestion requires two distinct nonblank tickers")
    return f"semantic-suggestion:{_text_hash(_canonical_json(members))}"


def embed_semantic_documents(
    documents: Sequence[SemanticMarketDocument],
    provider: EmbeddingProvider,
    *,
    cached_embeddings: Iterable[SemanticEmbedding] = (),
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[SemanticEmbedding, ...]:
    """Reuse exact cache hits and invoke the provider only for changed/missing text."""

    by_ticker = {document.ticker: document for document in documents}
    if len(by_ticker) != len(documents):
        raise ValueError("semantic documents must have unique tickers")
    cache = {
        (
            embedding.market_ticker,
            embedding.canonical_text_hash,
            embedding.provider,
            embedding.model,
        ): embedding
        for embedding in cached_embeddings
    }
    result: dict[str, SemanticEmbedding] = {}
    missing: list[SemanticMarketDocument] = []
    for ticker in sorted(by_ticker):
        document = by_ticker[ticker]
        cached = cache.get(
            (ticker, document.canonical_text_hash, provider.provider, provider.model)
        )
        if cached is None:
            missing.append(document)
        else:
            result[ticker] = cached

    if missing:
        raw_vectors = provider.embed([document.canonical_text for document in missing])
        if len(raw_vectors) != len(missing):
            raise SemanticProviderError("embedding provider returned the wrong row count")
        now = _aware_utc(clock(), label="embedding cache time")
        for document, raw_vector in zip(missing, raw_vectors, strict=True):
            vector = _numeric_vector(raw_vector, label=f"embedding for {document.ticker}")
            result[document.ticker] = SemanticEmbedding.create(
                market_ticker=document.ticker,
                canonical_text=document.canonical_text,
                canonical_text_hash=document.canonical_text_hash,
                rules_hash=document.rules_hash,
                timing_hash=document.timing_hash,
                provider=provider.provider,
                model=provider.model,
                vector=tuple(float(value) for value in vector),
                created_at=now,
            )
    return tuple(result[ticker] for ticker in sorted(result))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticParseError(f"semantic classifier JSON repeats key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SemanticParseError(f"semantic classifier JSON contains invalid number {value}")


def parse_classifier_output(raw_response: str) -> SemanticClassifierOutput:
    """Parse one unwrapped strict JSON object and reject all schema drift."""

    if not raw_response.strip():
        raise SemanticParseError("semantic classifier response is blank")
    if raw_response != raw_response.strip():
        raise SemanticParseError("semantic classifier response must be bare JSON without padding")
    try:
        payload = json.loads(
            raw_response,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except SemanticParseError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SemanticParseError("semantic classifier response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise SemanticParseError("semantic classifier response must be a JSON object")
    try:
        return SemanticClassifierOutput.model_validate(payload)
    except ValidationError as exc:
        raise SemanticParseError(f"semantic classifier schema is invalid: {exc}") from exc


def build_classifier_prompt(
    market_a: SemanticMarketDocument,
    market_b: SemanticMarketDocument,
    *,
    prompt_version: str = SEMANTIC_PROMPT_VERSION,
) -> str:
    """Build the versioned rule-grounded classifier prompt."""

    if market_a.ticker >= market_b.ticker:
        raise ValueError("classifier prompt markets must be in canonical ticker order")
    if not prompt_version.strip():
        raise ValueError("semantic prompt version cannot be blank")
    evidence = {
        "market_a": {
            "canonical_text": market_a.canonical_text,
            "rules_hash": market_a.rules_hash,
            "ticker": market_a.ticker,
        },
        "market_b": {
            "canonical_text": market_b.canonical_text,
            "rules_hash": market_b.rules_hash,
            "ticker": market_b.ticker,
        },
        "prompt_version": prompt_version,
    }
    return (
        "Classify a logical relationship between two prediction-market contracts.\n"
        "A proposed relation must hold in EVERY valid resolution state under the exact "
        "rules. Probability, correlation, likelihood, and common outcomes are irrelevant. "
        "If any plausible rules-consistent counterexample exists, return NONE or UNCERTAIN.\n"
        "Return exactly one bare JSON object with no markdown and exactly these keys: "
        "relation, confidence, rationale, requires_review. relation must be one of "
        "A_IMPLIES_B, B_IMPLIES_A, EQUIVALENT, MUTUALLY_EXCLUSIVE, NONE, UNCERTAIN; "
        "confidence must be a number from 0 through 1; requires_review must be true.\n"
        f"Evidence: {_canonical_json(evidence)}"
    )


def semantic_suggestion_payload_hash(suggestion: SemanticSuggestion) -> str:
    """Hash immutable provider/evidence content, excluding mutable review bookkeeping."""

    validated = SemanticSuggestion.model_validate(suggestion.model_dump())
    payload = validated.model_dump(
        mode="json",
        exclude={
            "approved_relation_id",
            "created_at",
            "review_state",
            "reviewed_at",
            "stale_reason",
            "updated_at",
        },
    )
    return _text_hash(_canonical_json(payload))


def _semantic_relation_shape(
    suggestion: SemanticSuggestion,
) -> tuple[RelationType, dict[str, str]]:
    proposal = suggestion.relation
    market_a = suggestion.market_a_ticker
    market_b = suggestion.market_b_ticker
    if proposal is SemanticRelationProposal.A_IMPLIES_B:
        return RelationType.IMPLIES, {"antecedent": market_a, "consequent": market_b}
    if proposal is SemanticRelationProposal.B_IMPLIES_A:
        return RelationType.IMPLIES, {"antecedent": market_b, "consequent": market_a}
    if proposal is SemanticRelationProposal.EQUIVALENT:
        return RelationType.EQUIVALENT, {}
    if proposal is SemanticRelationProposal.MUTUALLY_EXCLUSIVE:
        return RelationType.MUTUALLY_EXCLUSIVE, {}
    raise ValueError("NONE or UNCERTAIN cannot become a trusted relation")


def semantic_verified_relation_id(suggestion: SemanticSuggestion) -> str:
    """Compute an actionable proposal's relation ID without granting it trust."""

    suggestion = SemanticSuggestion.model_validate(suggestion.model_dump())
    relation_type, directional = _semantic_relation_shape(suggestion)
    return canonical_relation_id(
        SEMANTIC_VERIFIED_SOURCE,
        relation_type,
        (suggestion.market_a_ticker, suggestion.market_b_ticker),
        antecedent=directional.get("antecedent"),
        consequent=directional.get("consequent"),
    )


def relation_from_semantic_suggestion(
    suggestion: SemanticSuggestion,
    *,
    created_at: datetime,
) -> Relation:
    """Build the only trusted-relation shape an approval transaction may insert."""

    _aware_utc(created_at, label="semantic relation creation time")
    suggestion = SemanticSuggestion.model_validate(suggestion.model_dump())
    if suggestion.review_state is not SemanticReviewState.APPROVED:
        raise ValueError("only an approved semantic suggestion can become a trusted relation")
    market_a = suggestion.market_a_ticker
    market_b = suggestion.market_b_ticker
    relation_type, directional = _semantic_relation_shape(suggestion)
    relation_id = semantic_verified_relation_id(suggestion)
    if suggestion.approved_relation_id != relation_id:
        raise ValueError("approved suggestion relation ID does not match its proposal")
    members = (market_a, market_b)
    return Relation(
        relation_id=relation_id,
        market_tickers=members,
        relation_type=relation_type,
        source=SEMANTIC_VERIFIED_SOURCE,
        confidence=suggestion.confidence,
        verified=True,
        rationale=suggestion.rationale,
        created_at=created_at,
        **directional,
    )


class _SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object: ...


class SentenceTransformerEmbeddingProvider:
    """Lazily loaded local sentence-transformers adapter."""

    provider = "sentence-transformers"

    def __init__(self, model: str) -> None:
        if not model.strip():
            raise ValueError("embedding model cannot be blank")
        self.model = model
        self._encoder: _SentenceEncoder | None = None

    def _load(self) -> _SentenceEncoder:
        if self._encoder is not None:
            return self._encoder
        try:
            module = importlib.import_module("sentence_transformers")
            factory = cast(Callable[[str], object], vars(module)["SentenceTransformer"])
            self._encoder = cast(_SentenceEncoder, factory(self.model))
        except (ImportError, KeyError) as exc:
            raise SemanticProviderUnavailable(
                "sentence-transformers is unavailable; install Arbiter's semantic extra"
            ) from exc
        return self._encoder

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return ()
        try:
            raw = self._load().encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            matrix = np.asarray(raw, dtype=np.float64)
        except SemanticProviderUnavailable:
            raise
        except Exception as exc:
            raise SemanticProviderError("sentence-transformers embedding failed") from exc
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise SemanticProviderError("sentence-transformers returned an invalid matrix")
        return tuple(tuple(float(value) for value in row) for row in matrix)


class _Completions(Protocol):
    def create(self, **kwargs: object) -> object: ...


class _Chat(Protocol):
    completions: _Completions


class _OpenAIClient(Protocol):
    chat: _Chat


class _CompletionMessage(Protocol):
    content: str | None


class _CompletionChoice(Protocol):
    message: _CompletionMessage


class _CompletionResponse(Protocol):
    choices: Sequence[_CompletionChoice]


class OpenAICompatibleSemanticClassifier:
    """Lazily loaded JSON-mode chat-completions classifier adapter."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        for label, value in (
            ("semantic provider", provider),
            ("semantic model", model),
            ("semantic API key", api_key),
        ):
            if not value.strip():
                raise ValueError(f"{label} cannot be blank")
        if base_url is not None and not base_url.strip():
            raise ValueError("semantic base URL cannot be blank")
        self.provider = provider
        self.model = model
        self._api_key = api_key
        self._base_url = base_url

    def classify(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("semantic classifier prompt cannot be blank")
        try:
            module = importlib.import_module("openai")
            factory = cast(Callable[..., object], vars(module)["OpenAI"])
        except (ImportError, KeyError) as exc:
            raise SemanticProviderUnavailable(
                "OpenAI-compatible client is unavailable; install Arbiter's semantic extra"
            ) from exc
        client_kwargs: dict[str, object] = {"api_key": self._api_key}
        if self._base_url is not None:
            client_kwargs["base_url"] = self._base_url
        try:
            client = cast(_OpenAIClient, factory(**client_kwargs))
            response = cast(
                _CompletionResponse,
                client.chat.completions.create(
                    model=self.model,
                    messages=(
                        {
                            "role": "system",
                            "content": "Return only the requested strict JSON object.",
                        },
                        {"role": "user", "content": prompt},
                    ),
                    temperature=0,
                    response_format={"type": "json_object"},
                ),
            )
            choices = response.choices
            if not isinstance(choices, Sequence) or not choices:
                raise TypeError("response contains no choices")
            content = choices[0].message.content
            if not isinstance(content, str) or not content:
                raise TypeError("response contains no text content")
            return content
        except Exception as exc:
            raise SemanticProviderError(
                f"semantic classifier request failed for provider {self.provider}"
            ) from exc
