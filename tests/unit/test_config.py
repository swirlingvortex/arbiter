"""Tests for typed, safely redacted application configuration."""

from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.config import ConfigLoadError, KalshiEnvironment, load_settings


def test_default_config_loads_with_expected_types(project_root: Path) -> None:
    settings = load_settings(
        project_root / "config/default.yaml",
        env_file=None,
        environ={},
    )

    assert settings.engine.max_component_markets == 12
    assert settings.engine.min_net_profit_dollars == Decimal("0.01")
    assert settings.orderbook.resync_on_sequence_gap is True
    assert settings.fees.account_precision == Decimal("0.0001")
    assert settings.paper_execution.allow_partial_fill is False
    assert settings.relations.semantic_requires_manual_verification is True
    assert settings.relations.manual_path is None
    assert settings.relations.discover_exchange is True
    assert settings.relations.discover_thresholds is True
    assert settings.storage.data_dir == Path("data")
    assert settings.storage.write_max_attempts == 3
    assert settings.storage.write_initial_backoff_seconds == 0.05
    assert settings.storage.write_max_backoff_seconds == 1.0
    assert settings.collector.inbound_queue_capacity == 1_000
    assert settings.collector.writer_queue_capacity == 1_000
    assert settings.collector.reconnect_max_attempts == 5
    assert settings.collector.reconnect_initial_backoff_seconds == 0.5
    assert settings.collector.reconnect_max_backoff_seconds == 8.0
    assert settings.collector.open_timeout_seconds == 10.0
    assert settings.collector.close_timeout_seconds == 5.0
    assert settings.collector.writer_batch_size == 1_000
    assert settings.collector.writer_flush_interval_seconds == 1.0
    assert settings.kalshi.environment is KalshiEnvironment.PRODUCTION
    assert settings.kalshi.rest_base_url == "https://external-api.kalshi.com/trade-api/v2"
    assert settings.kalshi.websocket_url == ("wss://external-api-ws.kalshi.com/trade-api/ws/v2")
    assert settings.kalshi.websocket_path == "/trade-api/ws/v2"
    assert settings.semantic.provider == "disabled"
    assert settings.semantic.embedding_model == "all-MiniLM-L6-v2"
    assert settings.semantic.prompt_version == "semantic-relations-v1"


def test_environment_values_override_only_environment_specific_settings(
    project_root: Path,
) -> None:
    settings = load_settings(
        project_root / "config/default.yaml",
        env_file=None,
        environ={
            "KALSHI_ENV": "demo",
            "KALSHI_API_KEY_ID": "private-id",
            "KALSHI_PRIVATE_KEY_PATH": "/tmp/private.pem",
            "ARBITER_DATA_DIR": "/tmp/arbiter-data",
            "ARBITER_DB_PATH": "/tmp/arbiter-data/test.duckdb",
            "SEMANTIC_PROVIDER": "example-provider",
            "SEMANTIC_MODEL": "example-model",
            "SEMANTIC_BASE_URL": "https://semantic.example/v1",
            "SEMANTIC_API_KEY": "private-semantic-key",
        },
    )

    assert settings.kalshi.environment is KalshiEnvironment.DEMO
    assert settings.kalshi.websocket_url == ("wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2")
    assert settings.storage.data_dir == Path("/tmp/arbiter-data")
    assert settings.semantic.enabled is True
    assert settings.semantic.base_url == "https://semantic.example/v1"
    summary = repr(settings.safe_environment_summary())
    assert "private-id" not in summary
    assert "private-semantic-key" not in summary
    assert "/tmp/private.pem" not in summary
    assert "api_key_configured': True" in summary
    assert "auth_complete': True" in summary
    assert "external-api-ws.demo.kalshi.co" in summary


def test_blank_secret_values_are_treated_as_unconfigured(project_root: Path) -> None:
    settings = load_settings(
        project_root / "config/default.yaml",
        env_file=None,
        environ={"KALSHI_API_KEY_ID": "", "SEMANTIC_API_KEY": "  "},
    )

    assert settings.kalshi.api_key_id is None
    assert settings.semantic.api_key is None


@pytest.mark.parametrize(
    "unsafe_setting",
    [
        "relations:\n  semantic_requires_manual_verification: false\n",
        "orderbook:\n  resync_on_sequence_gap: false\n",
        "storage:\n  parquet_partition_by_date: false\n",
        "storage:\n  write_max_attempts: 0\n",
        ("storage:\n  write_initial_backoff_seconds: 2\n  write_max_backoff_seconds: 1\n"),
    ],
)
def test_invalid_or_unsafe_config_fails_closed(
    tmp_path: Path,
    unsafe_setting: str,
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(unsafe_setting, encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="validation failed"):
        load_settings(config_path, env_file=None, environ={})


@pytest.mark.parametrize(
    "collector_yaml",
    [
        "inbound_queue_capacity: 0",
        "writer_queue_capacity: 0",
        "reconnect_max_attempts: 101",
        "writer_batch_size: 0",
        "writer_flush_interval_seconds: .inf",
        ("reconnect_initial_backoff_seconds: 9\n  reconnect_max_backoff_seconds: 8"),
    ],
)
def test_invalid_collector_bounds_fail_closed(
    tmp_path: Path,
    collector_yaml: str,
) -> None:
    config_path = tmp_path / "invalid-collector.yaml"
    config_path.write_text(f"collector:\n  {collector_yaml}\n", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="validation failed"):
        load_settings(config_path, env_file=None, environ={})
