# Arbiter Data Model

Arbiter separates tolerant exchange wire schemas, strict normalized domain models, and
versioned analytical storage. Unknown additive Kalshi fields are retained in each metadata
record's `raw` object while misspelled internal fields remain validation errors. This document
covers the metadata, relation-review, opportunity, recorded-event, replay, paper-execution, and
research schemas implemented through Milestone 11.

## Fixed-point boundary

Current `*_dollars` price values are parsed directly from strings into `Decimal` with at most
four decimal places. Current `*_fp` quantities are likewise parsed directly with two-decimal
contract granularity. Arbiter never routes these fields through binary float. Every nonempty
order-book price must match at least one `Market.price_ranges` interval and step.

Market payload status is stored as returned (for example, `active`). This is intentionally
separate from REST query filters such as `status=open`. `series_ticker` is enriched from the
market's parent event rather than guessed from its ticker or title.

## Domain records

- `Market` contains identity, event/series ancestry, titles, lifecycle times, structured
  strike data, full resolution rules, occurrence/expiration timing, price ranges, and raw JSON.
- `Event` contains series ancestry, category, exchange-declared mutual exclusion, current
  market tickers, update time, the current paired fee override when present, a deterministically
  ordered tuple of scheduled `EventFeeChange` records, and raw JSON. Each change retains its
  exchange ID, event/series tickers, aware `scheduled_ts`, paired type/multiplier override, and
  original payload. Both override values may be null only together, which means clear.
- `Series` contains category/frequency, fee defaults, canonical settlement sources, contract
  links, update time, and raw JSON.
- `OrderBook` stores only exchange-provided YES and NO bids. YES asks are complements of real
  NO bids, and NO asks are complements of real YES bids with exactly the same quantity.
  REST snapshots have no fabricated exchange timestamp or sequence.
- `BookStatus` is `fresh`, `stale`, or `resync_required`. Freshness takes an explicit `as_of`
  time and duration so deterministic replay never consults wall-clock time.
- `EffectiveFeePolicy` is the immutable policy snapshot used for one market: event and Series
  identity, type, multiplier, source, effective change/time, next scheduled change/time, and
  the dated policy version.
- `FeeFill`, `FeeFillQuote`, and `FeeQuote` preserve exact per-level input, raw/trade/rounding
  fees, rebates, signed balance changes, and accumulator state. A mutable
  `FeeRoundingLedger` is passed explicitly and belongs to one synthetic order; it is not stored
  as hidden global state.
- `Opportunity` stores Stage 0-3 evidence plus exact allocations, gross economics, fee status,
  optional supported net economics, policy snapshots, quotes, reason, and paper status.
  Validation prevents a stage from claiming evidence that its prerequisites do not support.

## DuckDB metadata tables

`schema_migrations` records each append-only schema version. `metadata_sync_runs` records the
start, completion, status, counts, and any bounded error for every attempted sync.

`markets`, `events`, and `series` use ticker primary keys and idempotent upserts. Timestamps
use `TIMESTAMPTZ`; strikes and fee multipliers use wide `DECIMAL`; structured collections and
original payloads use JSON columns. A sync inserts its run marker first, then performs every
metadata upsert and the success update in one transaction. On failure, that transaction rolls
back and only the run marker is updated to `failed`, so partial metadata cannot appear valid.

Migration 3 adds `events.fee_changes_json`. Metadata synchronization fetches the cursor-paged
event schedule once, attaches only matching changes sorted by `(scheduled_ts, change_id)`, and
stores the typed schedule separately from the original Event payload. Loading events validates
and reconstructs those typed changes. The current Event override remains preserved in raw JSON
and is normalized again on load; parent defaults remain in the typed `series` columns.

Migration 3 also provisions these lifecycle tables for the scanner milestones:

### `opportunities`

```text
opportunity_id                 VARCHAR PRIMARY KEY
component_id                   VARCHAR NOT NULL
detected_at                    TIMESTAMPTZ NOT NULL
ended_at                       TIMESTAMPTZ
stage                          VARCHAR NOT NULL
relation_types_json            JSON NOT NULL
num_markets                    INTEGER NOT NULL
num_legs                       INTEGER NOT NULL
capital_required               DECIMAL(38,18)
gross_profit                   DECIMAL(38,18)
fees                           DECIMAL(38,18)
net_profit                     DECIMAL(38,18)
gross_edge                     DECIMAL(38,18)
net_edge                       DECIMAL(38,18)
fee_policy_version             VARCHAR
fee_policies_json              JSON NOT NULL
paper_execution_status         VARCHAR NOT NULL
paper_execution_reason         VARCHAR
metadata_json                  JSON NOT NULL
```

`fee_policies_json` holds the effective per-market policy snapshots, including their source
and effective/scheduled times. `fee_policy_version` makes the external rule assumptions used
for a classification queryable without decoding that JSON. Stage 0 or unsupported candidates
may have null numeric fee/net fields; they must never be silently filled with zero.

### `portfolio_legs`

```text
opportunity_id                 VARCHAR NOT NULL
ticker                         VARCHAR NOT NULL
side                           VARCHAR NOT NULL
price                          DECIMAL(38,18) NOT NULL
quantity                       DECIMAL(38,18) NOT NULL
source_side                    VARCHAR NOT NULL
source_price                   DECIMAL(38,18) NOT NULL
fee                            DECIMAL(38,18)
```

Migration 4 makes these summaries run-aware by adding nullable `run_id`, `opened_at`,
`closed_at`, and `updated_at` columns. Scanner-created rows populate all four. The original
columns remain compatible with records created before the scanner lifecycle existed.

Portfolio legs are replaced transactionally with each OPEN/UPDATED summary. CLOSED observations
remove the current legs but do not overwrite the last opportunity economics with an absence.
`paper_execution_status` and `paper_execution_reason` summarize the most recent applicable
Milestone 9 paper decision. Scanner and lifecycle reasons are not written into those fields.

## Scanner runs and observations

Migration 4 adds complete, queryable live-run provenance. Every observation write requires its
referenced manifest to remain in `running` state. A finalized run ID cannot be reopened.

### `run_manifests`

```text
run_id                           VARCHAR PRIMARY KEY
run_type                         VARCHAR NOT NULL
started_at                       TIMESTAMPTZ NOT NULL
ended_at                         TIMESTAMPTZ
status                           VARCHAR NOT NULL
schema_version                   INTEGER NOT NULL
manifest_version                 INTEGER NOT NULL
recording_format_version         INTEGER NOT NULL
event_schema_version             INTEGER NOT NULL
recording_id                     VARCHAR
source_run_id                    VARCHAR
first_event_index                BIGINT
last_event_index                 BIGINT
event_count                      BIGINT NOT NULL
event_stream_hash                VARCHAR
input_payload_json               JSON NOT NULL
config_hash                      VARCHAR
metadata_hash                    VARCHAR
relations_hash                   VARCHAR
fee_policy_hash                  VARCHAR
metadata_json                    JSON NOT NULL
error                            VARCHAR
```

Live manifests hash canonical, non-secret configuration, the exact active metadata snapshot,
verified relations, and effective fee inputs. Metadata records the environment, subscribed
tickers, and starting raw `event_index`; credentials and private-key paths are excluded.
Migration 5 adds manifest version 2. A successful schema-v2 live or replay run also records the
recording identity, inclusive contiguous event-index bounds, event count, a length-framed SHA-256
hash of the exact canonical recorded events, and the complete canonical run input payload. Replay
manifests use a distinct `run_id`, set `source_run_id` and `recording_id` to the originating scan
run, and copy the four input hashes. When the source manifest is available locally, replay checks
all of this evidence before opening its own manifest or writing observations.

### `market_observation_windows`

```text
observation_id                   VARCHAR PRIMARY KEY
run_id                           VARCHAR NOT NULL
market_ticker                    VARCHAR NOT NULL
opened_at, updated_at            TIMESTAMPTZ NOT NULL
closed_at                        TIMESTAMPTZ
opened_event_index               BIGINT NOT NULL
last_event_index                 BIGINT NOT NULL
closed_event_index               BIGINT
status                           VARCHAR NOT NULL
start_sequence, end_sequence     BIGINT
connection_id                    VARCHAR
stale_reason, resync_reason      VARCHAR
metadata_json                    JSON NOT NULL
```

The deterministic identifier is derived from run, market, and opening event index. A row is
updated only forward in event time and becomes immutable once closed. Windows close on age,
sequence uncertainty, lifecycle resynchronization, disconnect, failure, or bounded run end.
Freshness is inclusive at exactly `stale_after`; the first representable later instant is stale.

### `opportunity_observations`

```text
observation_id                   VARCHAR PRIMARY KEY
opportunity_id                   VARCHAR
run_id, component_id             VARCHAR NOT NULL
observed_at                      TIMESTAMPTZ NOT NULL
event_index                      BIGINT NOT NULL
transition                       VARCHAR NOT NULL
stage, solver_status, reason     VARCHAR
solve_duration_ms                DOUBLE NOT NULL
num_states, num_instruments,
num_legs                         INTEGER NOT NULL
capital_required, gross_profit,
fees, net_profit, gross_edge,
net_edge, capacity               DECIMAL(38,18)
fee_policy_version               VARCHAR
fee_policies_json                JSON NOT NULL
market_tickers_json              JSON
relation_types_json              JSON
relation_sources_json            JSON
market_contexts_json             JSON
evidence_json, metadata_json     JSON NOT NULL
```

Observation and episode IDs are deterministic hashes of their identity fields. The lifecycle
permits one active optimal episode per component and persists `NOT_PRESENT`, `OPEN`, `UPDATED`,
`CLOSED`, and terminal `RIGHT_CENSORED` decisions before mutating memory. Each OPEN/UPDATED
observation has an exact immutable leg snapshot in `opportunity_observation_legs` matching its
Opportunity allocations. `capacity` is not a separate
heuristic: for Stage 1+ it is exactly the actual depth-constrained optimum's
`capital_required`; Stage 0 and absent/closed observations leave it NULL. Evidence also records
the reference-price, one-contract gross, one-contract fee, and actual-depth funnel results.

Migration 7 appends the four nullable research snapshot columns shown above. New observations
write `market_tickers_json`, `relation_types_json`, and `relation_sources_json` in canonical order,
plus one `market_contexts_json` entry per market containing the scan-time event, category, and
settlement proxy. These values are immutable observation evidence: reporting must not reconstruct
them from today's mutable `markets`, `events`, `series`, or `relations` tables.

The columns deliberately remain nullable for rows created before Migration 7. A legacy `NULL`
means that the historical dimension was unavailable; it is not an empty component, an uncategorized
market, or permission to backfill from current metadata. Reports preserve that distinction as
unknown/unavailable and exclude such rows from dimension-specific denominators where appropriate.
This append-only rule prevents later metadata edits or relation reviews from changing a historical
result.

Migration 5 gives opportunity summaries `censored_at`, `censored_event_index`, and
`censor_reason`. A right-censored episode was still open when recorded coverage ended; it is not
misreported as an economically observed close. A later replay run uses its own deterministic
run-scoped observation and opportunity identifiers, while the cross-run economic projection
compares event time, source event index, transition, economics, legs, and reasons.

All scanner transition writes are one DuckDB transaction, use bounded retries only for classified
transient DuckDB failures, and stop the run after retry exhaustion. Reusing an observation ID with
different content fails rather than silently updating history.

Migration 4 also stores the normalized current event fee override in typed
`events.fee_type_override` and `events.fee_multiplier_override` columns. A live fee notification
latches the event uncertain until the complete refreshed schedule validates and persists.

High-volume order-book events are intentionally not stored in these tables. They use the
versioned, date-partitioned zstd Parquet stream described below.

## Parquet recorded events

Live WebSocket input is normalized into the strict discriminated event union that replay consumes.
The book records are:

- `OrderBookSnapshotEvent` is a complete replacement book containing descending, unique nested
  YES- and NO-bid levels.
- `OrderBookDeltaEvent` is one signed quantity change at one normalized YES- or NO-bid price.

Book event schema version `1` is immutable. Pydantic rejects unknown normalized fields, invalid
identifiers, nonpositive exchange sequence/subscription IDs, naive timestamps, non-finite
decimals, and zero-valued deltas. A schema-v2 recorded container adds canonical controls without
changing the schema-v1 book representation. Its Arrow metadata names
`arbiter.recorded_events` version `2`; book rows retain event-level schema version `1`, while
control rows use event-level schema version `2`.

The schema-v2 control union contains `run_started`, `subscription_started`,
`connection_interrupted`, `book_stale`, paired market/fee refresh start and applied records, and
`run_ended`. `run_started` embeds every non-secret decision input: effective configuration,
exact Markets/Events/Series, verified relations, and effective fee-policy payload, together with
their canonical hashes. Applied refresh controls carry the authoritative replacement Market or
Event, so replay never queries current exchange metadata.

The Arrow container uses these columns:

| Column | Arrow representation | Meaning |
| --- | --- | --- |
| `schema_version` | `int16` | Exact event-level schema version: `1` for books, `2` for controls. |
| `event_index` | `int64` | Authoritative contiguous total order for this recording root. |
| `local_received_ts` | UTC timestamp (microseconds) | Aware local receipt time; also selects the UTC date partition. |
| `exchange_ts` | nullable UTC timestamp (microseconds) | Exchange time when supplied; snapshots may not have one. |
| `ticker` | nullable string | Subscribed market ticker for a book record. |
| `sequence` | nullable `int64` | Positive exchange sequence value for a book record. |
| `sid` | nullable `int64` | Positive WebSocket subscription ID for a book record. |
| `connection_id` | nullable string | Physical connection identity for a book record. |
| `price_convention` | nullable string | Book source convention; version 1 requires `yes_price`. |
| `snapshot_id` | nullable string | Snapshot lineage for a book record and its following deltas. |
| `event_type` | string | Strict union discriminator for a book or control record. |
| `yes_bids`, `no_bids` | nullable lists of `{price, quantity}` | Full snapshot levels; prices are `decimal128(5,4)` and quantities are `decimal128(38,2)`. |
| `delta_side` | nullable string | Internal side (`yes` or `no`) for a delta. |
| `delta_price` | nullable `decimal128(5,4)` | Internal bid price for a delta. |
| `quantity_delta` | nullable `decimal128(38,2)` | Signed fixed-point depth change for a delta. |
| `control_payload_json` | nullable string | Complete canonical JSON for a control; null for books. |

Snapshot rows require both nested level lists and prohibit scalar delta fields. Delta rows
require all three scalar delta fields and prohibit snapshot lists. Control rows require all
book-only fields to be null and carry their complete canonical JSON. The reader validates this
shape, exact Arrow schema metadata, strict event model, and the relationship between each row's
`local_received_ts` and its partition before returning data. Legacy schema-v1 book-only datasets
remain readable, but controls cannot be appended to them.

### Price normalization

The collector subscribes with `use_yes_price=true` and preserves that source convention in every
row so replay never guesses what a wire price meant. A YES-side wire price is an internal YES bid.
For a no-side update, Kalshi's wire value is still a YES-scale price, so Arbiter stores the
internal NO-leg bid at `1 - p`, with the same quantity, exactly once. All values cross the
exchange boundary as decimal strings and remain fixed-point `Decimal` values. As with REST
books, asks are derived only as complements of real opposing bids; the event stream never
invents liquidity.

### Partitioning, ordering, and append boundary

Files are written as:

```text
data/parquet/orderbooks/date=YYYY-MM-DD/part-<first-index>-<last-index>-<id>.parquet
```

The partition date is derived from `local_received_ts` after conversion to UTC. Completed files
use zstd compression and are atomically moved into place. Failed writes leave queued events
available to report/retry and do not leave a completed partial file.

`event_index` is the only authoritative replay ordering. Neither `sequence` nor either timestamp
is used to break ties or reorder observations. Within a recording root, indices must be unique
and contiguous. A collector resuming an existing compatible root begins at its next index; the
reader rejects duplicate or missing indices, mismatched partition dates, mixed schema versions,
and incompatible Arrow schemas. One root therefore represents one sequential recording and must
have only one active writer. A nonblocking filesystem lease rejects concurrent writers rather
than risking overlapping indices. Use a different root for an independent recording.

The exchange sequence policy is deliberately separate. Offline tests currently track continuity
per `(connection_id, sid)` behind `SequenceTracker`; the credential-gated live smoke must verify
the actual exchange scope. A duplicate with identical exchange content is idempotent. A sequence
gap, out-of-order event, conflicting duplicate, or ambiguous connection transition marks the
subscription's books stale/resynchronization-required until replacement snapshots establish a
new baseline. Known-stale books are never solver inputs.

### Replay identity and validation

Canonical stream hashing is over each record's fixed-scale canonical JSON, framed with its byte
length before hashing so record boundaries cannot collide. Book prices are rendered at four
decimal places and quantities at two, exactly matching Parquet persistence; values outside that
representable precision fail before recording. Controls retain their strict canonical model JSON.

Replay first sorts by `event_index`, then rejects gaps, duplicates, decreasing event time,
midstream or nested run envelopes, mismatched run IDs, corrupt input hashes, refresh pairs that do
not match, and trusted subscription-membership contradictions. Records at the same event time are
processed in event-index order before timers due at that instant. A schema-v2 scan stream is
self-contained for strategy replay. A raw collector stream remains reconstructible book/control
evidence but intentionally lacks the run-start decision-input envelope required by `arbiter
replay`.

## Paper execution evidence

Migration 5 adds `paper_executions` and `paper_execution_legs`. An attempt ID is a deterministic
hash of replay run, opportunity, source observation, and source event index; persistence rejects
the same identity with different content. Each attempt links to a persisted Stage 2 observation
and stores detection, scheduled, attempted, and resolution times; configured latency; status and
failure; expected and actual costs, fees, prices, policies, and per-leg fills; the minimum terminal
payout; simulated locked profit; canonical evidence; and a payload hash.

Paper execution is all-or-none. Any unavailable leg, worse price, insufficient depth, unsupported
current fee policy, or nonpositive current locked profit fails the complete attempt without a
partial fill. If recorded coverage ends before the deadline, status is
`insufficient_future_data` and `attempted_at` is null. Only `survived` promotes the current
opportunity summary to Stage 3; newer economic observations always take precedence over an older
paper result.

### Collector failure and lifecycle behavior

The collector uses finite WebSocket and writer queues, finite reconnect attempts with bounded
backoff, finite connection timeouts, batch thresholds, and a periodic flush interval. Exceeding
the writer capacity raises an explicit backpressure error rather than dropping an event. Pending
accepted events are flushed on orderly shutdown and, when storage remains available, before an
upstream failure is propagated. Books from a completed or interrupted connection are invalidated
before the collector returns or reconnects.

Arbiter also subscribes to the documented market lifecycle channels. A change that can affect
price or quantity validation marks the relevant book stale, performs a bounded authoritative REST
metadata refresh, updates the normalizer, and requests a replacement snapshot. Failure to refresh
or resynchronize is visible and leaves the book unusable rather than continuing with obsolete
rules. Event-creation notifications are validated and ignored for a fixed-ticker recording.
Relevant event-fee notifications carry their authoritative set/clear values through an explicit
control signal and trigger a bounded event-specific schedule refresh; they do not request an
unrelated order-book snapshot and are never mistaken for order-book observations.

## Trusted relations

Migration 2 adds `relations`, keyed by deterministic or supplied `relation_id`. It stores the
relation type, exact market ordering, explicit implication endpoints, source, confidence,
verification state, rationale, and original `created_at`. Idempotent upserts may refresh
metadata but deliberately do not replace `created_at`.

Only a verified relation whose exact source is `exchange_declared`, `threshold_deterministic`,
`manual`, or `semantic_verified` may enter the logical graph. Confidence is stored as evidence,
not treated as authority. Ambiguous threshold candidates are returned as skip diagnostics.
Duplicate mathematical semantics, missing markets, impossible components, untrusted source
labels, and components larger than the configured bound fail validation before newly discovered
relations are inserted.

Migration 6 adds `semantic_suggestion_id`, `semantic_rules_hash`, and `semantic_timing_hash` to
approved semantic relation rows and creates two isolated evidence tables:

- `semantic_embeddings` stores canonical market text and its rules/timing hashes, provider and
  model identity, dimensions, vector JSON, and a payload hash. Its natural cache key includes the
  market, canonical-text hash, provider, and model.
- `semantic_suggestions` stores the ordered market pair, exact displayed title/rules/timing
  evidence and hashes, retrieval similarity, classifier and prompt identity, raw response, strict
  parsed proposal, confidence, rationale, and review state. Review state is one of `pending`,
  `approved`, `rejected`, `uncertain`, or `stale`.

Semantic approval rechecks the suggestion against authoritative persisted market evidence and
creates its `semantic_verified` relation in the same transaction. Rejecting or marking a
suggestion uncertain creates no relation. If either market's rules or timing evidence later
changes or disappears, staleness reconciliation marks the suggestion stale and de-verifies the
projected relation. Direct relation upserts cannot forge a `semantic_verified` row.

## Research artifact schema

`arbiter report` derives deterministic analytical products without mutating source observations:

- `report.md` is the human-readable methodology, cohort, funnel, summary, breakdown, and
  limitation report;
- `summary.csv`, `funnel.csv`, and `breakdowns.csv` are machine-readable aggregate tables;
- `episodes.parquet` contains exactly one fact per selected opportunity episode.

The episode artifact exposes these stable fields:

```text
fact_id
cohort_id
run_id
opportunity_id
component_id
opened_at
terminal_status
ended_at
duration_seconds
relation_types
relation_sources
categories
settlement_bucket
peak_stage
peak_gross_edge
peak_net_edge
peak_capacity
peak_net_guarantee
midpoint_available
reference_violation
one_contract_gross_survived
one_contract_fee_survived
depth_executable
paper_status
paper_locked_profit
```

`cohort_id` is the source recording identity used to deduplicate its live and replay projections.
`terminal_status` distinguishes an observed close from right-censoring; a censored
`duration_seconds` is only a lower bound. Diagnostic booleans remain nullable when the relevant
stage could not be evaluated. Relation and category collections may be multi-label, while
`settlement_bucket` is one mutually exclusive fixed bucket. Decimal-valued economics remain exact
through the reducer and are serialized without silently replacing missing values with zero.
