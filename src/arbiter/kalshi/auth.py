"""Secret-safe RSA-PSS signing for the authenticated Kalshi WebSocket handshake."""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from pathlib import Path

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

WEBSOCKET_SIGNING_PATH = "/trade-api/ws/v2"

_ACCESS_KEY_HEADER = "KALSHI-ACCESS-KEY"
_ACCESS_TIMESTAMP_HEADER = "KALSHI-ACCESS-TIMESTAMP"
_ACCESS_SIGNATURE_HEADER = "KALSHI-ACCESS-SIGNATURE"


class KalshiAuthError(ValueError):
    """Raised when credentials or signing inputs cannot be used safely."""


def _current_timestamp_ms() -> int:
    return time.time_ns() // 1_000_000


def _load_unencrypted_rsa_key(path: Path) -> rsa.RSAPrivateKey:
    try:
        encoded_key = path.read_bytes()
    except FileNotFoundError:
        raise KalshiAuthError("private key file does not exist") from None
    except OSError:
        raise KalshiAuthError("private key file could not be read") from None

    try:
        private_key = serialization.load_pem_private_key(encoded_key, password=None)
    except (TypeError, ValueError, UnsupportedAlgorithm):
        raise KalshiAuthError("private key must be an unencrypted PEM RSA private key") from None
    finally:
        del encoded_key

    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise KalshiAuthError("private key must be an unencrypted PEM RSA private key")
    return private_key


def _validated_timestamp_ms(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise KalshiAuthError("timestamp_ms must be a nonnegative integer")
    return value


def _signing_message(*, timestamp_ms: int, method: str, path: str) -> bytes:
    if not isinstance(method, str) or not method or method.strip() != method:
        raise KalshiAuthError("HTTP method must be a nonempty value without surrounding whitespace")
    if not isinstance(path, str) or not path.startswith("/"):
        raise KalshiAuthError("request path must be an absolute path")
    path_without_query = path.partition("?")[0]
    if not path_without_query or "#" in path_without_query:
        raise KalshiAuthError("request path is invalid")
    return f"{timestamp_ms}{method.upper()}{path_without_query}".encode()


class KalshiWebSocketAuthenticator:
    """Load one RSA key and construct exact authentication headers on demand.

    The object deliberately exposes neither its key material nor credential identifier in its
    representation. It signs only caller-supplied request metadata and does not perform I/O
    beyond loading the explicitly supplied private-key path during construction.
    """

    __slots__ = ("_api_key_id", "_clock_ms", "_private_key")

    def __init__(
        self,
        *,
        api_key_id: str,
        private_key_path: Path,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(api_key_id, str) or not api_key_id or api_key_id.strip() != api_key_id:
            raise KalshiAuthError(
                "API key ID must be a nonempty value without surrounding whitespace"
            )
        if not isinstance(private_key_path, Path):
            raise TypeError("private_key_path must be an explicit pathlib.Path")
        self._api_key_id = api_key_id
        self._private_key = _load_unencrypted_rsa_key(private_key_path)
        self._clock_ms = _current_timestamp_ms if clock_ms is None else clock_ms

    def __repr__(self) -> str:
        return "KalshiWebSocketAuthenticator(api_key_id=<redacted>, private_key=<redacted>)"

    def sign_request(self, *, timestamp_ms: int, method: str, path: str) -> str:
        """Return the base64 RSA-PSS signature for one canonical request message."""

        validated_timestamp = _validated_timestamp_ms(timestamp_ms)
        message = _signing_message(
            timestamp_ms=validated_timestamp,
            method=method,
            path=path,
        )
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("ascii")

    def handshake_headers(
        self,
        *,
        method: str = "GET",
        path: str = WEBSOCKET_SIGNING_PATH,
        timestamp_ms: int | None = None,
    ) -> dict[str, str]:
        """Return the three exact headers required for a Kalshi WebSocket handshake."""

        effective_timestamp = self._clock_ms() if timestamp_ms is None else timestamp_ms
        validated_timestamp = _validated_timestamp_ms(effective_timestamp)
        return {
            _ACCESS_KEY_HEADER: self._api_key_id,
            _ACCESS_TIMESTAMP_HEADER: str(validated_timestamp),
            _ACCESS_SIGNATURE_HEADER: self.sign_request(
                timestamp_ms=validated_timestamp,
                method=method,
                path=path,
            ),
        }
