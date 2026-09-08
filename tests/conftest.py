"""Shared pytest fixtures for Arbiter."""

from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the repository root independent of the test working directory."""

    return Path(__file__).resolve().parents[1]
