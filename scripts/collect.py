#!/usr/bin/env python3
"""Run Arbiter's authenticated, research-only market-data collector directly."""

import typer

from arbiter.cli import collect

if __name__ == "__main__":
    typer.run(collect)
