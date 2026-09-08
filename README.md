# Arbiter

Arbiter is a Python 3.12 research and paper-trading system for finding structural
inconsistencies among logically related prediction-market contracts and testing whether
they survive executable prices, displayed liquidity, fees, and latency-aware paper execution.
It implements a transparent pipeline of trusted logical relations, feasible settlement worlds,
state-contingent payoffs, and linear optimization.

## Status

The offline engine through Milestone 11 is implemented. Strict manual/exchange/threshold
relations feed feasible worlds and a depth-constrained, independently verified LP; public Kalshi
REST metadata and order books
are normalized and persisted; quantities are snapped to the documented grid; and dated current
Kalshi fee policies, exact rounding, and Stage 0-3 opportunity evidence are modeled. The
authenticated WebSocket collector now reconstructs books fail-closed and records normalized,
replayable market-data events in date-partitioned Parquet. The bounded live scanner subscribes
only to verified relation components, incrementally solves affected components, and persists
run manifests, fresh-book observation windows, and deterministic opportunity episodes. Exact
Schema-v2 scan recordings include indexed run, subscription, disconnect, staleness, metadata,
fee, and terminal controls. Deterministic replay reconstructs its inputs from that immutable
stream, follows the same engine path, and simulates latency-aware all-or-none paper execution.
Semantic discovery is optional and proposal-only: an explicit review transaction is required to
create a trusted `semantic_verified` relation, and changed source evidence revokes that projection.
Milestone 11 adds unique-episode research analytics and fixture-derived Markdown, CSV, and Parquet
artifacts. Release verification evidence lives in
[`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md); the architecture, mathematics, schemas, and
research methodology are documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/MATH.md`](docs/MATH.md), [`docs/DATA_MODEL.md`](docs/DATA_MODEL.md), and
[`docs/RESEARCH.md`](docs/RESEARCH.md).

## Development setup

Python 3.12 is required. From the repository root:

```bash
make install
make verify
.venv/bin/arbiter --help
.venv/bin/arbiter doctor
.venv/bin/arbiter demo
.venv/bin/arbiter report \
  --db tests/fixtures/research/arbiter.duckdb \
  --output-dir build/fixture-report
```

Copy `.env.example` to `.env` only when local credentials or environment-specific paths
are needed. `.env` and private-key files are ignored by Git. Normal algorithm settings
belong in `config/default.yaml`, not environment variables.

The offline demo, fixture report, and test suite need no credentials. Missing Kalshi credentials
block only credential-dependent live REST/WebSocket smoke tests, not offline development,
replay, or reporting.

## Record live order books

Set `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY_PATH` in `.env`, select production or demo with
`KALSHI_ENV`, and keep the private-key file readable only by its owner. The
`arbiter doctor --require-auth` command requires credential presence and checks key-file
permissions without printing the key ID, key path, or key contents. Then run a bounded
collection:

```bash
.venv/bin/arbiter collect \
  --market-ticker <OPEN_TICKER> \
  --duration-seconds 30
```

Repeat `--market-ticker` to record several related markets. By default, files are written under
`data/parquet/orderbooks/date=YYYY-MM-DD/`; use `--output-dir` to select another dataset root.
The equivalent direct wrapper is `.venv/bin/python scripts/collect.py --market-ticker ...`.

Collection is read-only with respect to Kalshi: it fetches market metadata, subscribes to market
data and lifecycle channels, and writes local Parquet files. It has no order-entry path and never
submits real or paper orders. Missing credentials fail this command before network access but do
not affect `make verify`, the offline demo, or synthetic collector tests.

The subscription requests Kalshi's unified YES-price convention. Arbiter stores YES bids at the
wire price and converts a no-side wire YES price `p` to the internal NO-leg bid price `1 - p`
exactly once. A sequence gap, conflicting duplicate, malformed/negative-depth update, connection
loss, or relevant lifecycle metadata change makes the affected books ineligible for solving
until replacement snapshots arrive. Lifecycle changes trigger a bounded REST metadata refresh
before resynchronization. Queue overflow and storage failure are visible command failures; events
are never silently discarded.

Each Parquet dataset root is one ordered stream. `event_index`, not exchange or local
timestamps, is its authoritative total order. A later collection into the same compatible root
resumes at the next index; duplicate, skipped, or incompatible records fail closed. Do not run
concurrent collectors against one dataset root; a nonblocking writer lease rejects the second
process visibly. See `docs/DATA_MODEL.md` for the versioned schema and append boundary.

## Scan trusted live components

First synchronize open metadata and create or load verified relations. Review relation output
before scanning; unverified semantic proposals are never solver constraints.

```bash
.venv/bin/arbiter markets sync
.venv/bin/arbiter relations discover
.venv/bin/arbiter relations validate
.venv/bin/arbiter scan --duration-seconds 30
```

`arbiter scan` derives its subscription set from active/open markets participating in verified
relations; it does not accept an arbitrary ticker list. It writes the normalized book records
plus the indexed control and immutable input records required for self-contained decision replay,
and records scanner state in the configured DuckDB database. A sequence gap,
connection interruption, metadata ambiguity, non-open lifecycle refresh, persistent storage
failure, or exhausted network retry fails closed. A non-open lifecycle refresh ends the bounded
run before requesting another snapshot, so closed/paused/settled markets cannot re-enter the
solver through stale subscription state. The equivalent direct wrapper is
`.venv/bin/python scripts/scan.py --duration-seconds 30`.

Opportunity identity is deterministic within a run and component. Each observation records the
solver funnel, actual optimum capacity (`capital_required`), fees and edges, while market
observation windows separately record where reconstructed books were fresh, stale, disconnected,
or awaiting resynchronization. Run manifests hash the effective non-secret configuration,
metadata, relations, and fee policy inputs. Credential values and private-key paths are neither
stored in manifests nor emitted by the scan command.

## Replay and paper execution

Replay a complete schema-v2 scan recording without network access:

```bash
.venv/bin/arbiter replay \
  --file tests/fixtures/orderbooks/canonical_lifecycle.parquet \
  --speed max
.venv/bin/arbiter replay --date 2026-09-03 --speed 100
```

Exactly one of `--file` or `--date` is required. A file may be one Parquet part or a dataset
directory, but it must contain complete `RunStartedEvent` through `RunEndedEvent` boundaries.
Date selection reads the complete configured order-book root and chooses runs by their UTC start
date, so a run that crosses midnight is not truncated at a partition boundary. `--speed max`
does not sleep; a positive numeric speed changes wall-clock pacing only. Event time, ordering,
freshness, debounce, opportunity duration, and paper deadlines remain unchanged. The equivalent
wrapper is `.venv/bin/python scripts/replay.py --file ... --speed max`.

Every replay receives a distinct DuckDB run manifest linked to its source run and exact
length-framed stream hash. The first run-start record supplies all decision-bearing non-secret
configuration, metadata, relations, and fee-policy inputs; replay never substitutes today's
mutable metadata. Invalid gaps, backward times, mismatched manifests, midstream files, or raw-only
collector streams fail closed. Raw `arbiter collect` output remains exactly reconstructible as
book/control data, while self-contained strategy replay requires the input/run envelope written
by `arbiter scan`.

The tracked canonical fixture opens the generic `M => A` portfolio at a `$0.92` cost and `$0.08`
guaranteed profit. A same-time depth removal at the configured 100 ms deadline makes its paper
attempt fail all-or-none, and the following debounce closes the opportunity. Regenerate and
validate it with `.venv/bin/python scripts/generate_canonical_lifecycle_fixture.py`.

## Generate the research report

Generate the reproducible fixture report from the repository root:

```bash
.venv/bin/arbiter report \
  --db tests/fixtures/research/arbiter.duckdb \
  --output-dir build/fixture-report
```

The output directory contains `report.md`, `summary.csv`, `funnel.csv`, `breakdowns.csv`, and
`episodes.parquet`. The report counts unique opportunity episodes, deduplicates replay runs that
represent the same recording, uses event-time market-hours, and keeps right-censored episodes and
paper attempts with insufficient future coverage out of ordinary failure denominators.

The tracked [fixture report](docs/fixture-report/report.md) is a reviewable example of those
artifacts. Its synthetic implication has a `$0.92` entry cost, a `$1.00` minimum settlement
payout, and `$0.08` guaranteed gross profit before its latency-aware paper attempt fails because
one leg disappears. This is a pipeline and arithmetic check, not evidence about how often
opportunities occur on Kalshi or how much money could be realized.

[`notebooks/research.ipynb`](notebooks/research.ipynb) reads the generated tables without
reimplementing their production aggregation logic. See
[`docs/RESEARCH.md`](docs/RESEARCH.md) before interpreting a report.

## Limitations

- Arbiter buys displayed liquidity in simulation; it does not model queue position, hidden
  liquidity, market impact, funding constraints, or guaranteed multi-leg fills.
- Fees are resolved from the run's dated metadata, but unsupported or uncertain policies fail
  closed rather than being guessed.
- Exhaustive feasible-world enumeration is intentionally bounded to small logical components.
- Recorded runs are observational samples. Right-censored episodes are lower-bound durations,
  and summed hypothetical guarantees may overlap in time, capital, or liquidity and are not
  independent realized profit.
- Semantic models may propose relations, but confidence is never trust. Only reviewed,
  allowlisted relations can enter the solver.
- Arbiter v1 is intra-venue. It does not implement cross-platform or real-money execution.

## Safety boundary

Arbiter is a research and paper-execution project. It contains no real-money order-submission
path. API credentials are used only for authenticated market-data access, private keys are never
stored in run inputs or logs, and known-stale or ambiguous books cannot produce an executable
claim. Paper execution is an all-or-none simulation of a fundamentally non-atomic multi-leg
process; a simulated survivor is not a promise of a real-world fill or profit.
