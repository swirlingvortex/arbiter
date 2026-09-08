"""Central fail-closed trust policy for relations used by the solver."""

from __future__ import annotations

from typing import Final

from arbiter.models.relation import Relation

# These are production provenance labels, not quality scores. In particular, model
# confidence never promotes a semantic proposal into this set; only the explicit
# review transaction may create ``semantic_verified`` relations.
TRUSTED_RELATION_SOURCES: Final[frozenset[str]] = frozenset(
    {
        "exchange_declared",
        "manual",
        "semantic_verified",
        "threshold_deterministic",
    }
)


def is_allowed_relation_source(source: str) -> bool:
    """Return whether an exact persisted provenance label may constrain worlds."""

    return source in TRUSTED_RELATION_SOURCES


def is_trusted_relation(relation: Relation) -> bool:
    """Require independent verification and approved provenance, regardless of confidence."""

    return relation.verified and is_allowed_relation_source(relation.source)
