#!/usr/bin/env python3
"""Replay Arbiter recorded runs through the deterministic scanner directly."""

import typer

from arbiter.cli import replay

if __name__ == "__main__":
    typer.run(replay)
