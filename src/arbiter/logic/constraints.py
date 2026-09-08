"""Pure logical constraints over binary market assignments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LogicalConstraint(Protocol):
    """Contract shared by constraints accepted by the world enumerator."""

    @property
    def tickers(self) -> tuple[str, ...]:
        """Return every ticker referenced by the constraint."""

    def is_satisfied(self, assignment: dict[str, int]) -> bool:
        """Return whether a complete binary assignment satisfies the constraint."""


def _outcome(assignment: dict[str, int], ticker: str) -> int:
    try:
        value = assignment[ticker]
    except KeyError as exc:
        raise KeyError(f"assignment is missing ticker {ticker!r}") from exc
    if value not in (0, 1):
        raise ValueError(f"assignment for {ticker!r} must be binary")
    return value


@dataclass(frozen=True, slots=True)
class ImplicationConstraint:
    """Forbid the sole invalid implication state: antecedent YES, consequent NO."""

    antecedent: str
    consequent: str

    def __post_init__(self) -> None:
        if not self.antecedent or not self.consequent:
            raise ValueError("implication endpoints must be nonempty")
        if self.antecedent == self.consequent:
            raise ValueError("implication endpoints must differ")

    @property
    def tickers(self) -> tuple[str, str]:
        """Return the ordered directional endpoints."""

        return (self.antecedent, self.consequent)

    def is_satisfied(self, assignment: dict[str, int]) -> bool:
        """Evaluate ``antecedent <= consequent`` for a binary assignment."""

        return not _outcome(assignment, self.antecedent) or bool(
            _outcome(assignment, self.consequent)
        )


@dataclass(frozen=True, slots=True)
class EquivalenceConstraint:
    """Require two markets to settle to the same binary outcome."""

    left: str
    right: str

    def __post_init__(self) -> None:
        if not self.left or not self.right:
            raise ValueError("equivalence endpoints must be nonempty")
        if self.left == self.right:
            raise ValueError("equivalence endpoints must differ")

    @property
    def tickers(self) -> tuple[str, str]:
        return (self.left, self.right)

    def is_satisfied(self, assignment: dict[str, int]) -> bool:
        return _outcome(assignment, self.left) == _outcome(assignment, self.right)


@dataclass(frozen=True, slots=True)
class MutualExclusionConstraint:
    """Allow at most one YES settlement among a group of markets."""

    members: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_group(self.members, "mutual-exclusion")

    @property
    def tickers(self) -> tuple[str, ...]:
        return self.members

    def is_satisfied(self, assignment: dict[str, int]) -> bool:
        return sum(_outcome(assignment, ticker) for ticker in self.members) <= 1


@dataclass(frozen=True, slots=True)
class ExactlyOneConstraint:
    """Require exactly one YES settlement among a group of markets."""

    members: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_group(self.members, "exactly-one")

    @property
    def tickers(self) -> tuple[str, ...]:
        return self.members

    def is_satisfied(self, assignment: dict[str, int]) -> bool:
        return sum(_outcome(assignment, ticker) for ticker in self.members) == 1


def _validate_group(members: tuple[str, ...], label: str) -> None:
    if len(members) < 2:
        raise ValueError(f"{label} constraints require at least two markets")
    if any(not ticker for ticker in members):
        raise ValueError(f"{label} constraint tickers must be nonempty")
    if len(set(members)) != len(members):
        raise ValueError(f"{label} constraint tickers must be unique")
