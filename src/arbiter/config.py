"""Typed application configuration loaded from YAML and environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
DEFAULT_ENV_FILE = Path(".env")


class ConfigLoadError(RuntimeError):
    """Raised when configuration cannot be read or validated safely."""


class FrozenSettings(BaseModel):
    """Base class that rejects misspelled settings and prevents mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EngineSettings(FrozenSettings):
    """Logical-component and solver scheduling settings."""

    max_component_markets: int = Field(default=12, ge=1, le=24)
    min_net_profit_dollars: Decimal = Field(default=Decimal("0.01"), ge=0)
    min_net_edge_bps: Decimal = Field(default=Decimal("1"), ge=0)
    solve_debounce_ms: int = Field(default=25, ge=0)


class OrderBookSettings(FrozenSettings):
    """Order-book snapshot and freshness settings."""

    default_depth: int = Field(default=100, ge=1)
    stale_after_ms: int = Field(default=2_000, ge=1)
    resync_on_sequence_gap: Literal[True] = True


class FeeSettings(FrozenSettings):
    """Exchange fee-account rounding behavior for research classification."""

    account_precision: Decimal = Decimal("0.0001")

    @field_validator("account_precision")
    @classmethod
    def validate_account_precision(cls, value: Decimal) -> Decimal:
        if value not in {Decimal("0.0001"), Decimal("0.01")}:
            raise ValueError("account_precision must be 0.0001 (direct) or 0.01 (non-direct)")
        return value


class PaperExecutionSettings(FrozenSettings):
    """Paper-only execution simulation settings."""

    enabled: bool = True
    latency_ms: int = Field(default=100, ge=0)
    allow_partial_fill: bool = False


class RelationSettings(FrozenSettings):
    """Deterministic and semantic relationship discovery settings."""

    semantic_enabled: bool = False
    semantic_requires_manual_verification: Literal[True] = True
    nearest_neighbors: int = Field(default=10, ge=1)
    manual_path: Path | None = None
    discover_exchange: bool = True
    discover_thresholds: bool = True


class StorageSettings(FrozenSettings):
    """Persistence locations and batching settings."""

    flush_interval_seconds: int = Field(default=5, ge=1)
    write_max_attempts: int = Field(default=3, ge=1, le=100)
    write_initial_backoff_seconds: float = Field(default=0.05, ge=0, le=60)
    write_max_backoff_seconds: float = Field(default=1.0, ge=0, le=300)
    parquet_partition_by_date: Literal[True] = True
    data_dir: Path = Path("./data")
    db_path: Path = Path("./data/arbiter.duckdb")

    @model_validator(mode="after")
    def validate_write_backoff_window(self) -> StorageSettings:
        if self.write_initial_backoff_seconds > self.write_max_backoff_seconds:
            raise ValueError("initial storage backoff cannot exceed maximum storage backoff")
        return self


class CollectorSettings(FrozenSettings):
    """Finite live-collection resource, timeout, and retry bounds."""

    inbound_queue_capacity: int = Field(default=1_000, ge=1, le=1_000_000)
    writer_queue_capacity: int = Field(default=1_000, ge=1, le=1_000_000)
    reconnect_max_attempts: int = Field(default=5, ge=0, le=100)
    reconnect_initial_backoff_seconds: float = Field(
        default=0.5,
        gt=0,
        le=60,
        allow_inf_nan=False,
    )
    reconnect_max_backoff_seconds: float = Field(
        default=8.0,
        gt=0,
        le=300,
        allow_inf_nan=False,
    )
    open_timeout_seconds: float = Field(default=10.0, gt=0, le=120, allow_inf_nan=False)
    close_timeout_seconds: float = Field(default=5.0, gt=0, le=120, allow_inf_nan=False)
    writer_batch_size: int = Field(default=1_000, ge=1, le=100_000)
    writer_flush_interval_seconds: float = Field(
        default=1.0,
        gt=0,
        le=300,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_backoff_window(self) -> CollectorSettings:
        if self.reconnect_initial_backoff_seconds > self.reconnect_max_backoff_seconds:
            raise ValueError("initial reconnect backoff cannot exceed maximum reconnect backoff")
        return self


class KalshiEnvironment(StrEnum):
    """Supported Kalshi deployment environments."""

    PRODUCTION = "production"
    DEMO = "demo"


_REST_BASE_URLS: dict[KalshiEnvironment, str] = {
    KalshiEnvironment.PRODUCTION: "https://external-api.kalshi.com/trade-api/v2",
    KalshiEnvironment.DEMO: "https://external-api.demo.kalshi.co/trade-api/v2",
}

_WEBSOCKET_URLS: dict[KalshiEnvironment, str] = {
    KalshiEnvironment.PRODUCTION: "wss://external-api-ws.kalshi.com/trade-api/ws/v2",
    KalshiEnvironment.DEMO: "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2",
}
KALSHI_WEBSOCKET_PATH = "/trade-api/ws/v2"


class KalshiSettings(FrozenSettings):
    """Environment selection and optional credentials for exchange access."""

    environment: KalshiEnvironment = KalshiEnvironment.PRODUCTION
    api_key_id: SecretStr | None = None
    private_key_path: Path | None = None

    @property
    def rest_base_url(self) -> str:
        """Return the documented REST endpoint for the selected environment."""

        return _REST_BASE_URLS[self.environment]

    @property
    def websocket_url(self) -> str:
        """Return the documented WebSocket endpoint for the selected environment."""

        return _WEBSOCKET_URLS[self.environment]

    @property
    def websocket_path(self) -> str:
        """Return the path signed for the authenticated WebSocket handshake."""

        return KALSHI_WEBSOCKET_PATH


class SemanticProviderSettings(FrozenSettings):
    """Optional semantic-provider configuration; disabled by default."""

    provider: str = Field(default="disabled", min_length=1)
    model: str | None = Field(default=None, min_length=1)
    base_url: str | None = Field(default=None, min_length=1)
    embedding_model: str = Field(default="all-MiniLM-L6-v2", min_length=1)
    prompt_version: str = Field(default="semantic-relations-v1", min_length=1)
    api_key: SecretStr | None = None

    @field_validator(
        "provider",
        "model",
        "base_url",
        "embedding_model",
        "prompt_version",
    )
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("semantic provider settings cannot be blank")
        return value

    @property
    def enabled(self) -> bool:
        """Whether an external semantic provider was selected."""

        return self.provider.casefold() != "disabled"


class ArbiterSettings(FrozenSettings):
    """Complete validated application configuration."""

    engine: EngineSettings = Field(default_factory=EngineSettings)
    orderbook: OrderBookSettings = Field(default_factory=OrderBookSettings)
    fees: FeeSettings = Field(default_factory=FeeSettings)
    paper_execution: PaperExecutionSettings = Field(default_factory=PaperExecutionSettings)
    relations: RelationSettings = Field(default_factory=RelationSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    collector: CollectorSettings = Field(default_factory=CollectorSettings)
    kalshi: KalshiSettings = Field(default_factory=KalshiSettings)
    semantic: SemanticProviderSettings = Field(default_factory=SemanticProviderSettings)

    def safe_environment_summary(self) -> dict[str, object]:
        """Return environment diagnostics without exposing secret values."""

        return {
            "kalshi_environment": self.kalshi.environment.value,
            "kalshi_rest_base_url": self.kalshi.rest_base_url,
            "kalshi_websocket_url": self.kalshi.websocket_url,
            "kalshi_websocket_path": self.kalshi.websocket_path,
            "kalshi_api_key_configured": self.kalshi.api_key_id is not None,
            "kalshi_private_key_configured": self.kalshi.private_key_path is not None,
            "kalshi_auth_complete": (
                self.kalshi.api_key_id is not None and self.kalshi.private_key_path is not None
            ),
            "collector_inbound_queue_capacity": self.collector.inbound_queue_capacity,
            "collector_writer_queue_capacity": self.collector.writer_queue_capacity,
            "collector_reconnect_max_attempts": self.collector.reconnect_max_attempts,
            "semantic_provider": self.semantic.provider,
            "semantic_model": self.semantic.model,
            "semantic_embedding_model": self.semantic.embedding_model,
            "semantic_prompt_version": self.semantic.prompt_version,
            "semantic_api_key_configured": self.semantic.api_key is not None,
            "fee_account_precision": str(self.fees.account_precision),
            "data_dir": str(self.storage.data_dir),
            "db_path": str(self.storage.db_path),
        }


def _read_yaml(config_path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigLoadError(f"Could not read configuration file: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Invalid YAML in configuration file: {config_path}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigLoadError("The configuration root must be a YAML mapping.")
    return cast(dict[str, Any], loaded)


def _environment_values(
    *, env_file: Path | None, environ: Mapping[str, str] | None
) -> dict[str, str]:
    values: dict[str, str] = {}
    if env_file is not None and env_file.is_file():
        values.update(
            {key: value for key, value in dotenv_values(env_file).items() if value is not None}
        )
    values.update(dict(os.environ if environ is None else environ))
    return values


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    existing = raw.setdefault(name, {})
    if not isinstance(existing, dict):
        raise ConfigLoadError(f"Configuration section '{name}' must be a mapping.")
    return cast(dict[str, Any], existing)


def _nonempty(values: Mapping[str, str], key: str) -> str | None:
    value = values.get(key)
    if value is None or not value.strip():
        return None
    return value.strip()


def load_settings(
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    env_file: Path | None = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> ArbiterSettings:
    """Load normal settings from YAML and approved environment-specific values.

    The optional ``environ`` argument makes precedence explicit and keeps tests isolated
    from the caller's process environment. Values in the environment override ``.env``.
    """

    raw = deepcopy(_read_yaml(config_path))
    values = _environment_values(env_file=env_file, environ=environ)

    storage = _section(raw, "storage")
    data_dir = _nonempty(values, "ARBITER_DATA_DIR")
    db_path = _nonempty(values, "ARBITER_DB_PATH")
    if data_dir is not None:
        storage["data_dir"] = data_dir
    if db_path is not None:
        storage["db_path"] = db_path

    kalshi = _section(raw, "kalshi")
    environment = _nonempty(values, "KALSHI_ENV")
    api_key_id = _nonempty(values, "KALSHI_API_KEY_ID")
    private_key_path = _nonempty(values, "KALSHI_PRIVATE_KEY_PATH")
    if environment is not None:
        kalshi["environment"] = environment
    if api_key_id is not None:
        kalshi["api_key_id"] = api_key_id
    if private_key_path is not None:
        kalshi["private_key_path"] = private_key_path

    semantic = _section(raw, "semantic")
    provider = _nonempty(values, "SEMANTIC_PROVIDER")
    model = _nonempty(values, "SEMANTIC_MODEL")
    base_url = _nonempty(values, "SEMANTIC_BASE_URL")
    semantic_api_key = _nonempty(values, "SEMANTIC_API_KEY")
    if provider is not None:
        semantic["provider"] = provider
    if model is not None:
        semantic["model"] = model
    if base_url is not None:
        semantic["base_url"] = base_url
    if semantic_api_key is not None:
        semantic["api_key"] = semantic_api_key

    try:
        return ArbiterSettings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigLoadError(f"Configuration validation failed: {exc}") from exc
