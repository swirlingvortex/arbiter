#!/usr/bin/env python3
"""Run Arbiter's authenticated, research-only live scanner directly."""

import typer

from arbiter.cli import scan

if __name__ == "__main__":
    typer.run(scan)
