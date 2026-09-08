"""Current Kalshi RSA-PSS WebSocket handshake authentication tests."""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa

from arbiter.kalshi.auth import (
    WEBSOCKET_SIGNING_PATH,
    KalshiAuthError,
    KalshiWebSocketAuthenticator,
)

TIMESTAMP_MS = 1_788_436_800_123
API_KEY_ID = "test-key-id"


def _write_rsa_key(
    path: Path,
    *,
    encryption: serialization.KeySerializationEncryption | None = None,
) -> rsa.RSAPrivateKey:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    encryption_algorithm = serialization.NoEncryption() if encryption is None else encryption
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption_algorithm,
        )
    )
    return key


def _authenticator(path: Path) -> tuple[KalshiWebSocketAuthenticator, rsa.RSAPrivateKey]:
    key = _write_rsa_key(path)
    return (
        KalshiWebSocketAuthenticator(
            api_key_id=API_KEY_ID,
            private_key_path=path,
            clock_ms=lambda: TIMESTAMP_MS,
        ),
        key,
    )


def test_handshake_headers_have_exact_names_and_verifiable_signature(tmp_path: Path) -> None:
    authenticator, key = _authenticator(tmp_path / "kalshi.pem")

    headers = authenticator.handshake_headers()

    assert headers.keys() == {
        "KALSHI-ACCESS-KEY",
        "KALSHI-ACCESS-TIMESTAMP",
        "KALSHI-ACCESS-SIGNATURE",
    }
    assert headers["KALSHI-ACCESS-KEY"] == API_KEY_ID
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == str(TIMESTAMP_MS)
    key.public_key().verify(
        base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"], validate=True),
        f"{TIMESTAMP_MS}GET{WEBSOCKET_SIGNING_PATH}".encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )


def test_signing_uppercases_method_and_removes_query_string(tmp_path: Path) -> None:
    authenticator, key = _authenticator(tmp_path / "kalshi.pem")

    signature = base64.b64decode(
        authenticator.sign_request(
            timestamp_ms=TIMESTAMP_MS,
            method="get",
            path="/trade-api/ws/v2?ignored=yes&also=ignored",
        ),
        validate=True,
    )

    key.public_key().verify(
        signature,
        f"{TIMESTAMP_MS}GET{WEBSOCKET_SIGNING_PATH}".encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )


def test_explicit_timestamp_overrides_injected_clock(tmp_path: Path) -> None:
    authenticator, _ = _authenticator(tmp_path / "kalshi.pem")

    headers = authenticator.handshake_headers(timestamp_ms=42)

    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "42"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("", WEBSOCKET_SIGNING_PATH),
        (" GET", WEBSOCKET_SIGNING_PATH),
        ("GET", "trade-api/ws/v2"),
        ("GET", "/bad#fragment"),
    ],
)
def test_invalid_request_metadata_fails_closed(tmp_path: Path, method: str, path: str) -> None:
    authenticator, _ = _authenticator(tmp_path / "kalshi.pem")

    with pytest.raises(KalshiAuthError):
        authenticator.sign_request(timestamp_ms=TIMESTAMP_MS, method=method, path=path)


@pytest.mark.parametrize("timestamp_ms", [-1, True])
def test_invalid_timestamp_fails_closed(tmp_path: Path, timestamp_ms: int) -> None:
    authenticator, _ = _authenticator(tmp_path / "kalshi.pem")

    with pytest.raises(KalshiAuthError, match="timestamp_ms"):
        authenticator.handshake_headers(timestamp_ms=timestamp_ms)


def test_private_key_path_must_be_an_explicit_path() -> None:
    with pytest.raises(TypeError, match="pathlib.Path"):
        KalshiWebSocketAuthenticator(
            api_key_id=API_KEY_ID,
            private_key_path="/tmp/key.pem",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("api_key_id", ["", "  ", " key-id", "key-id "])
def test_invalid_api_key_id_fails_before_key_io(api_key_id: str, tmp_path: Path) -> None:
    with pytest.raises(KalshiAuthError, match="API key ID"):
        KalshiWebSocketAuthenticator(
            api_key_id=api_key_id,
            private_key_path=tmp_path / "absent.pem",
        )


def test_missing_and_unreadable_key_paths_fail_without_disclosure(tmp_path: Path) -> None:
    missing = tmp_path / "missing-secret-name.pem"
    with pytest.raises(KalshiAuthError, match="does not exist") as missing_error:
        KalshiWebSocketAuthenticator(api_key_id=API_KEY_ID, private_key_path=missing)
    assert str(missing) not in str(missing_error.value)

    unreadable = tmp_path / "unreadable-secret-name.pem"
    unreadable.mkdir()
    with pytest.raises(KalshiAuthError, match="could not be read") as unreadable_error:
        KalshiWebSocketAuthenticator(api_key_id=API_KEY_ID, private_key_path=unreadable)
    assert str(unreadable) not in str(unreadable_error.value)


def test_invalid_encrypted_and_non_rsa_pem_keys_fail_without_content_leak(tmp_path: Path) -> None:
    cases: list[tuple[Path, bytes]] = []

    invalid = tmp_path / "invalid.pem"
    invalid_content = b"-----BEGIN PRIVATE KEY-----\nSENSITIVE-MARKER\n-----END PRIVATE KEY-----\n"
    invalid.write_bytes(invalid_content)
    cases.append((invalid, invalid_content))

    encrypted = tmp_path / "encrypted.pem"
    password = b"SENSITIVE-PASSWORD"
    _write_rsa_key(encrypted, encryption=serialization.BestAvailableEncryption(password))
    cases.append((encrypted, password))

    non_rsa = tmp_path / "ec.pem"
    ec_key = ec.generate_private_key(ec.SECP256R1())
    non_rsa.write_bytes(
        ec_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cases.append((non_rsa, b"EC PRIVATE KEY"))

    for path, secret_marker in cases:
        with pytest.raises(KalshiAuthError, match="unencrypted PEM RSA") as error:
            KalshiWebSocketAuthenticator(api_key_id=API_KEY_ID, private_key_path=path)
        rendered = str(error.value).encode()
        assert secret_marker not in rendered
        assert str(path).encode() not in rendered


def test_authenticator_repr_and_logging_do_not_expose_credentials(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    key_path = tmp_path / "SENSITIVE-KEY-PATH.pem"
    authenticator, _ = _authenticator(key_path)

    with caplog.at_level(logging.DEBUG):
        rendered = repr(authenticator)

    assert API_KEY_ID not in rendered
    assert str(key_path) not in rendered
    assert "BEGIN PRIVATE KEY" not in rendered
    assert caplog.records == []
