"""Offline CLI integration tests for the authenticated live-scanner boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from arbiter.cli import app
from arbiter.engine.live import LiveScannerError
from arbiter.storage.duckdb import StorageError

runner = CliRunner()


def test_scan_help_documents_bounded_live_command() -> None:
    result = runner.invoke(app, ["scan", "--help"])

    assert result.exit_code == 0
    assert "--duration-seconds" in result.stdout
    assert "--output-dir" in result.stdout
    assert "never trade" in result.stdout


def test_scan_requires_credentials_without_attempting_network(
    project_root: Path,
    monkeypatch: Any,
) -> None:
    called = False

    async def unexpected_runner(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("live runner must not be called")

    monkeypatch.setattr("arbiter.engine.live.run_live_scan", unexpected_runner)
    result = runner.invoke(
        app,
        [
            "scan",
            "--config",
            str(project_root / "config/default.yaml"),
            "--env-file",
            str(project_root / ".missing-test-env"),
        ],
        env={"KALSHI_API_KEY_ID": "", "KALSHI_PRIVATE_KEY_PATH": ""},
    )

    assert result.exit_code == 1
    assert "Not configured" in result.stdout
    assert "Traceback" not in result.stdout
    assert not called


def test_scan_rejects_nonfinite_duration_before_loading_configuration() -> None:
    result = runner.invoke(
        app,
        ["scan", "--duration-seconds", "nan", "--config", "does-not-exist.yaml"],
    )

    assert result.exit_code == 2
    assert "must be finite and positive" in result.stdout
    assert "Configuration failed" not in result.stdout


def test_scan_wires_settings_duration_destination_and_result_output(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    key_path = tmp_path / "private.pem"
    key_path.write_text("test-only-placeholder", encoding="utf-8")
    key_path.chmod(0o600)
    output_dir = tmp_path / "events"
    observed: dict[str, object] = {}

    async def fake_run(
        settings: object,
        *,
        duration_seconds: float,
        output_dir: Path,
    ) -> object:
        observed.update(
            settings=settings,
            duration_seconds=duration_seconds,
            output_dir=output_dir,
        )
        return SimpleNamespace(
            run_id="run-test-123",
            events_written=7,
            scans_completed=3,
            opportunity_observations=2,
            files_written=(output_dir / "part.parquet",),
            resync_requests=1,
            metadata_refreshes=4,
            event_fee_refreshes=1,
        )

    def fake_configure_logging() -> None:
        observed["logging_configured"] = True

    monkeypatch.setattr("arbiter.cli.configure_logging", fake_configure_logging)
    monkeypatch.setattr("arbiter.engine.live.run_live_scan", fake_run)
    result = runner.invoke(
        app,
        [
            "scan",
            "--duration-seconds",
            "2.5",
            "--output-dir",
            str(output_dir),
            "--config",
            str(project_root / "config/default.yaml"),
            "--env-file",
            str(project_root / ".missing-test-env"),
        ],
        env={
            "KALSHI_API_KEY_ID": "never-print-key-id",
            "KALSHI_PRIVATE_KEY_PATH": str(key_path),
        },
    )

    assert result.exit_code == 0
    output = " ".join(result.stdout.split())
    assert observed["duration_seconds"] == 2.5
    assert observed["output_dir"] == output_dir
    assert observed["logging_configured"] is True
    assert "run-test-123" in output
    assert "7 ordered event(s)" in output
    assert "3 component scan(s)" in output
    assert "2 opportunity observation(s)" in output
    assert "1 Parquet file(s)" in output
    assert "never-print-key-id" not in output
    assert str(key_path) not in output


@pytest.mark.parametrize("error_type", [LiveScannerError, StorageError])
def test_scan_maps_live_runner_error_without_exposing_credentials(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: Any,
    error_type: type[Exception],
) -> None:
    key_path = tmp_path / "private.pem"
    key_path.write_text("test-only-placeholder", encoding="utf-8")
    key_path.chmod(0o600)

    async def failed_run(*args: object, **kwargs: object) -> object:
        raise error_type(
            "persistent observation write failed after bounded retries: "
            f"never-print-key-id {key_path}"
        )

    monkeypatch.setattr("arbiter.engine.live.run_live_scan", failed_run)
    result = runner.invoke(
        app,
        [
            "scan",
            "--config",
            str(project_root / "config/default.yaml"),
            "--env-file",
            str(project_root / ".missing-test-env"),
        ],
        env={
            "KALSHI_API_KEY_ID": "never-print-key-id",
            "KALSHI_PRIVATE_KEY_PATH": str(key_path),
        },
    )

    assert result.exit_code == 1
    assert "Live scan failed" in result.stdout
    assert "persistent observation write failed after bounded retries" in result.stdout
    assert "Traceback" not in result.stdout
    assert "never-print-key-id" not in result.stdout
    assert str(key_path) not in result.stdout
    assert "[REDACTED]" in result.stdout
