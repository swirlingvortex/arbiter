# Canonical lifecycle recording

`canonical_lifecycle.parquet` is a deterministic schema-v2 recorded run for the verified
implication `M => A`. At event time zero, displayed depth derives a YES-A ask of `$0.62`
and a NO-M ask of `$0.30`, so one contract of each costs `$0.92`, pays at least `$1.00`,
and has `$0.08` guaranteed gross profit under the fixture's explicit zero fee multiplier.

The engine opens the opportunity after its 25 ms debounce. At the paper executor's 125 ms
deadline, a same-time delta has already removed the M YES bid, so the attempt fails
all-or-none. The subsequent 25 ms debounce closes the opportunity at 150 ms, before the
run ends at 200 ms.

Regenerate and validate the tracked binary from the repository root:

```bash
.venv/bin/python scripts/generate_canonical_lifecycle_fixture.py
```

The generator uses Arbiter's public normalized models and `ParquetEventWriter`, then
strictly reads and smoke-replays the resulting fixed-name fixture before succeeding.
