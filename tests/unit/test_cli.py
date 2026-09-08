"""Command-line smoke tests for the Arbiter shell, doctor, and offline demo."""

import tomllib
from importlib.metadata import version
from pathlib import Path

from typer.testing import CliRunner

from arbiter import __version__
from arbiter.cli import CheckStatus, _db_parent_check, app

runner = CliRunner()


def test_release_version_is_consistent(project_root: Path) -> None:
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text())

    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert metadata["project"]["version"] == version("arbiter") == __version__ == "1.0.0"
    assert result.stdout.strip() == __version__


def test_database_doctor_probe_is_transactional_and_noncreating(tmp_path: Path) -> None:
    database_path = tmp_path / "arbiter.duckdb"

    result = _db_parent_check(database_path)

    assert result.status is CheckStatus.PASS
    assert not database_path.exists()


def test_database_doctor_rejects_an_unopenable_existing_database(tmp_path: Path) -> None:
    database_path = tmp_path / "arbiter.duckdb"
    original = b"not a DuckDB database"
    database_path.write_bytes(original)

    result = _db_parent_check(database_path)

    assert result.status is CheckStatus.FAIL
    assert database_path.read_bytes() == original


def test_top_level_help_exposes_safe_milestone_zero_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "demo" in result.stdout
    assert "collect" in result.stdout
    assert "replay" in result.stdout
    assert "report" in result.stdout
    assert "markets" in result.stdout
    assert "relations" in result.stdout


def test_replay_help_exposes_file_date_and_speed_sources() -> None:
    result = runner.invoke(app, ["replay", "--help"])

    assert result.exit_code == 0
    assert "--file" in result.stdout
    assert "--date" in result.stdout
    assert "--speed" in result.stdout


def test_replay_requires_exactly_one_source_before_loading_config() -> None:
    missing = runner.invoke(app, ["replay", "--config", "does-not-exist.yaml"])
    both = runner.invoke(
        app,
        [
            "replay",
            "--file",
            "recording.parquet",
            "--date",
            "2026-09-03",
            "--config",
            "does-not-exist.yaml",
        ],
    )

    assert missing.exit_code == 2
    assert both.exit_code == 2
    assert "exactly one of --file or --date" in missing.stdout
    assert "exactly one of --file or --date" in both.stdout


def test_replay_rejects_nonpositive_and_nonfinite_speed_before_loading_config() -> None:
    for invalid in ("0", "-1", "nan", "inf", "fast"):
        result = runner.invoke(
            app,
            [
                "replay",
                "--file",
                "recording.parquet",
                "--speed",
                invalid,
                "--config",
                "does-not-exist.yaml",
            ],
        )

        assert result.exit_code == 2
        assert "finite positive number" in result.stdout


def test_doctor_passes_without_optional_credentials(
    project_root: Path,
    monkeypatch: object,
) -> None:
    # CliRunner does not isolate the current directory, so explicitly enter the repo root.
    monkeypatch.chdir(project_root)  # type: ignore[attr-defined]
    result = runner.invoke(
        app,
        [
            "doctor",
            "--config",
            str(project_root / "config/default.yaml"),
            "--env-file",
            str(project_root / ".missing-test-env"),
        ],
        env={
            "KALSHI_API_KEY_ID": "",
            "KALSHI_PRIVATE_KEY_PATH": "",
            "SEMANTIC_API_KEY": "",
        },
    )

    assert result.exit_code == 0
    assert "Python runtime" in result.stdout
    assert "Kalshi WebSocket endpoint" in result.stdout
    assert "external-api-ws.kalshi.com" in result.stdout
    assert "Collector bounds" in result.stdout
    assert "Not configured" in result.stdout
    assert "FAIL" not in result.stdout


def test_doctor_can_require_websocket_credentials(
    project_root: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.chdir(project_root)  # type: ignore[attr-defined]
    result = runner.invoke(
        app,
        [
            "doctor",
            "--require-auth",
            "--config",
            str(project_root / "config/default.yaml"),
            "--env-file",
            str(project_root / ".missing-test-env"),
        ],
        env={"KALSHI_API_KEY_ID": "", "KALSHI_PRIVATE_KEY_PATH": ""},
    )

    assert result.exit_code == 1
    assert "Required but not configured" in result.stdout
    assert "FAIL" in result.stdout


def test_doctor_accepts_owner_only_key_without_printing_secrets(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.chdir(project_root)  # type: ignore[attr-defined]
    key_path = tmp_path / "private-auth-material.pem"
    key_path.write_text("never-print-this-private-material", encoding="utf-8")
    key_path.chmod(0o600)

    result = runner.invoke(
        app,
        [
            "doctor",
            "--config",
            str(project_root / "config/default.yaml"),
            "--env-file",
            str(project_root / ".missing-test-env"),
        ],
        env={
            "KALSHI_API_KEY_ID": "never-print-this-key-id",
            "KALSHI_PRIVATE_KEY_PATH": str(key_path),
        },
    )

    assert result.exit_code == 0
    assert "owner-only key" in result.stdout
    assert "never-print-this-key-id" not in result.stdout
    assert "never-print-this-private-material" not in result.stdout
    assert str(key_path) not in result.stdout


def test_doctor_rejects_broad_private_key_permissions_without_printing_path(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    monkeypatch.chdir(project_root)  # type: ignore[attr-defined]
    key_path = tmp_path / "broad-private-key.pem"
    key_path.write_text("private-material", encoding="utf-8")
    key_path.chmod(0o644)

    result = runner.invoke(
        app,
        [
            "doctor",
            "--config",
            str(project_root / "config/default.yaml"),
            "--env-file",
            str(project_root / ".missing-test-env"),
        ],
        env={
            "KALSHI_API_KEY_ID": "private-id",
            "KALSHI_PRIVATE_KEY_PATH": str(key_path),
        },
    )

    assert result.exit_code == 1
    assert "group/other access" in result.stdout
    assert "private-id" not in result.stdout
    assert str(key_path) not in result.stdout


def test_doctor_reports_invalid_config_without_traceback(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("engine: [not, a, mapping]\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["doctor", "--config", str(config_path), "--env-file", str(tmp_path / "none")],
    )

    assert result.exit_code == 1
    assert "Configuration failed" in result.stdout
    assert "Traceback" not in result.stdout


def test_demo_prints_canonical_economics_without_network() -> None:
    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 0
    assert "Cost: 0.92" in result.stdout
    assert "Worst-case payout: 1.00" in result.stdout
    assert "Gross guaranteed profit: 0.08" in result.stdout
