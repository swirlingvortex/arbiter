"""Small structured-logging helpers with defensive secret redaction."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

_SENSITIVE_KEY_PARTS = ("api_key", "private_key", "password", "secret", "token")


def _is_sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def redact(value: object, *, key: str | None = None) -> object:
    """Recursively replace values attached to secret-like field names."""

    if key is not None and _is_sensitive_key(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(item_key): redact(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    """Render one compact JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "structured_fields", None)
        if isinstance(fields, Mapping):
            payload["fields"] = redact(fields)
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(level: str = "INFO") -> None:
    """Configure the process root logger for concise structured output."""

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def log_event(logger: logging.Logger, level: int, message: str, **fields: object) -> None:
    """Log an event with structured, redacted contextual fields."""

    logger.log(level, message, extra={"structured_fields": redact(fields)})
