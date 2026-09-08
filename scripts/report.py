#!/usr/bin/env python3
"""Generate Arbiter's read-only research report directly."""

import typer

from arbiter.cli import report

if __name__ == "__main__":
    typer.run(report)
