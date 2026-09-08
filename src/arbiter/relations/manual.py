"""Strict manual relation YAML parsing and trusted relation construction."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from arbiter.models.relation import Relation, RelationType
from arbiter.relations.deterministic import canonical_relation_id


class ManualRelationError(ValueError):
    """Raised when a manual relation file is malformed or references unknown markets."""


class _ManualBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_id: str | None = Field(default=None, min_length=1)
    rationale: str = Field(min_length=1)


class ManualImplication(_ManualBase):
    type: Literal["implies"]
    antecedent: str = Field(min_length=1)
    consequent: str = Field(min_length=1)


class ManualEquivalence(_ManualBase):
    type: Literal["equivalent"]
    markets: tuple[str, str]


class ManualMutualExclusion(_ManualBase):
    type: Literal["mutually_exclusive"]
    markets: tuple[str, ...] = Field(min_length=2)


class ManualExactlyOne(_ManualBase):
    type: Literal["exactly_one"]
    markets: tuple[str, ...] = Field(min_length=2)


ManualRelationSpec = Annotated[
    ManualImplication | ManualEquivalence | ManualMutualExclusion | ManualExactlyOne,
    Field(discriminator="type"),
]


class ManualRelationFile(BaseModel):
    """Only the documented top-level YAML shape is accepted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relations: tuple[ManualRelationSpec, ...]


_FILE_ADAPTER = TypeAdapter(ManualRelationFile)


def load_manual_relations(
    path: Path,
    *,
    known_market_tickers: set[str],
    created_at: datetime | None = None,
) -> tuple[Relation, ...]:
    """Parse, validate, and mark documented manual relations as trusted."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManualRelationError(f"could not read manual relation file {path}") from exc
    except yaml.YAMLError as exc:
        raise ManualRelationError(f"invalid YAML in manual relation file {path}: {exc}") from exc
    try:
        document = _FILE_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise ManualRelationError(f"invalid manual relation schema: {exc}") from exc

    timestamp = datetime.now(UTC) if created_at is None else created_at
    relations: list[Relation] = []
    for spec in document.relations:
        members: tuple[str, ...]
        if isinstance(spec, ManualImplication):
            relation_type = RelationType.IMPLIES
            members = (spec.antecedent, spec.consequent)
            directional = {
                "antecedent": spec.antecedent,
                "consequent": spec.consequent,
            }
        else:
            relation_type = RelationType(spec.type)
            members = tuple(spec.markets)
            directional = {}
        missing = sorted(set(members) - known_market_tickers)
        if missing:
            raise ManualRelationError(
                f"manual relation references unknown markets: {', '.join(missing)}"
            )
        relation_id = spec.relation_id or canonical_relation_id(
            "manual",
            relation_type,
            members,
            antecedent=directional.get("antecedent"),
            consequent=directional.get("consequent"),
        )
        try:
            relations.append(
                Relation(
                    relation_id=relation_id,
                    market_tickers=members,
                    relation_type=relation_type,
                    source="manual",
                    confidence=None,
                    verified=True,
                    rationale=spec.rationale,
                    created_at=timestamp,
                    **directional,
                )
            )
        except ValidationError as exc:
            raise ManualRelationError(f"invalid manual relation {relation_id}: {exc}") from exc
    return tuple(sorted(relations, key=lambda relation: relation.relation_id))
