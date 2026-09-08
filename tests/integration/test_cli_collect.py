"""Offline CLI integration tests for the authenticated collector boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from arbiter.cli import _run_market_data_collection, app
from arbiter.config import load_settings
from arbiter.kalshi.websocket import CollectorResult
from arbiter.models.event import Event
from arbiter.models.market import Market, PriceRange
from arbiter.replay.events import FeeRefreshStartedEvent

runner = CliRunner()


def test_collect_requires_credentials_without_attempting_network(project_root: Path) -> None:
    result = runner.invoke(
        app,
        [
            "collect",
            "--market-ticker",
            "KX-TEST",
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


def test_collect_validates_unique_nonblank_tickers(project_root: Path) -> None:
    duplicate = runner.invoke(
        app,
        [
            "collect",
            "--market-ticker",
            "KX-TEST",
            "--market-ticker",
            "KX-TEST",
            "--config",
            str(project_root / "config/default.yaml"),
        ],
    )
    missing = runner.invoke(app, ["collect"])

    assert duplicate.exit_code == 2
    assert "must be unique" in duplicate.stdout
    assert missing.exit_code == 2
    assert "at least one" in missing.stdout


def test_collect_wires_validated_settings_without_revealing_credentials(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    key_path = tmp_path / "private.pem"
    key_path.write_text("test-only-placeholder", encoding="utf-8")
    key_path.chmod(0o600)
    output_dir = tmp_path / "events"
    observed: dict[str, object] = {}

    async def fake_run(**kwargs: object) -> CollectorResult:
        observed.update(kwargs)
        return CollectorResult(
            events_written=4,
            files_written=(output_dir / "part.parquet",),
            resync_requests=1,
            metadata_refreshes=0,
        )

    monkeypatch.setattr("arbiter.cli._run_market_data_collection", fake_run)
    result = runner.invoke(
        app,
        [
            "collect",
            "--market-ticker",
            "B",
            "--market-ticker",
            "A",
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
    assert observed["tickers"] == ("A", "B")
    assert observed["duration_seconds"] == 2.5
    assert observed["output_dir"] == output_dir
    assert "Recorded 4" in result.stdout
    assert "1 Parquet" in result.stdout
    assert "never-print-key-id" not in result.stdout
    assert str(key_path) not in result.stdout


@pytest.mark.asyncio
async def test_collection_composition_shares_allocator_and_returns_complete_fee_event(
    project_root: Path,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    key_path = tmp_path / "private.pem"
    key_path.write_text("test-only-placeholder", encoding="utf-8")
    settings = load_settings(
        project_root / "config/default.yaml",
        env_file=None,
        environ={
            "KALSHI_API_KEY_ID": "test-key",
            "KALSHI_PRIVATE_KEY_PATH": str(key_path),
        },
    )
    market = Market(
        ticker="A",
        event_ticker="EVENT",
        title="A",
        status="active",
        price_ranges=(
            PriceRange(
                start=Decimal("0.0000"),
                end=Decimal("1.0000"),
                step=Decimal("0.0100"),
            ),
        ),
        raw={},
    )
    event = Event(
        ticker="EVENT",
        title="Event",
        market_tickers=("A", "B"),
        fee_type_override="quadratic",
        fee_multiplier_override=Decimal("1.25"),
        raw={},
    )
    observed: dict[str, object] = {}

    class FakeRestClient:
        def __init__(self, base_url: str) -> None:
            observed["rest_base_url"] = base_url

        def __enter__(self) -> FakeRestClient:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get_market(self, ticker: str, *, refresh: bool = False) -> Market:
            del refresh
            assert ticker == "A"
            return market

        def get_event(
            self,
            event_ticker: str,
            *,
            with_nested_markets: bool,
            refresh: bool,
        ) -> Event:
            observed["event_request"] = (event_ticker, with_nested_markets, refresh)
            return event

        def iter_event_fee_changes(self, *, event_ticker: str) -> tuple[()]:
            assert event_ticker == "EVENT"
            return ()

    class FakeWriter:
        next_event_index = 41

        def __init__(self, *args: object, **kwargs: object) -> None:
            observed["writer_args"] = (args, kwargs)

        def __enter__(self) -> FakeWriter:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeSession:
        def __init__(self, **kwargs: object) -> None:
            observed["session_kwargs"] = kwargs

    class FakeCollector:
        def __init__(self, **kwargs: object) -> None:
            observed["collector_kwargs"] = kwargs

        async def run(self, *, duration_seconds: float) -> CollectorResult:
            observed["duration_seconds"] = duration_seconds
            kwargs = observed["collector_kwargs"]
            assert isinstance(kwargs, dict)
            refresher = kwargs["event_fee_refresher"]
            allocator = kwargs["event_index_allocator"]
            refreshed = await refresher(
                FeeRefreshStartedEvent(
                    event_index=allocator.allocate(),
                    local_received_ts=datetime(2026, 9, 3, tzinfo=UTC),
                    event_ticker="EVENT",
                    affected_tickers=("A",),
                    connection_id="connection-1",
                    sid=8,
                )
            )
            observed["refreshed_event"] = refreshed
            return CollectorResult(0, (), 0, 0)

    monkeypatch.setattr("arbiter.cli.KalshiWebSocketAuthenticator", lambda **kwargs: kwargs)
    monkeypatch.setattr("arbiter.cli.KalshiRestClient", FakeRestClient)
    monkeypatch.setattr("arbiter.cli.ParquetEventWriter", FakeWriter)
    monkeypatch.setattr("arbiter.cli.KalshiWebSocketSession", FakeSession)
    monkeypatch.setattr("arbiter.cli.MarketDataCollector", FakeCollector)

    result = await _run_market_data_collection(
        settings=settings,
        tickers=("A",),
        duration_seconds=2.5,
        output_dir=tmp_path / "events",
    )

    session_kwargs = observed["session_kwargs"]
    collector_kwargs = observed["collector_kwargs"]
    assert isinstance(session_kwargs, dict)
    assert isinstance(collector_kwargs, dict)
    assert session_kwargs["event_index_allocator"] is collector_kwargs["event_index_allocator"]
    assert "start_event_index" not in session_kwargs
    assert observed["event_request"] == ("EVENT", True, True)
    assert observed["refreshed_event"] == event
    assert result.events_written == 0
