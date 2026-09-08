# Research fixture

`arbiter.duckdb` is generated from the tracked
`../orderbooks/canonical_lifecycle.parquet` recording. It contains current catalog rows copied
from the recording's immutable `RunStartedEvent`, one deterministic successful replay, OPEN and
CLOSED observation snapshots, and one failed all-or-none paper execution.

Regenerate it from the repository root with:

```bash
.venv/bin/python scripts/generate_research_fixture.py
```

The generator builds and validates a temporary database before atomically replacing the fixture.
Legacy rows are not synthesized; migration-7 snapshot columns remain nullable to represent
historical context that was never captured.
