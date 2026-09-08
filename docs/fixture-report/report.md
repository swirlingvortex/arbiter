# Arbiter research report

> This report is descriptive research evidence, not realized P&L or a claim of atomic execution. Multi-leg fills are non-atomic.

## Scope and cohort

- Report schema: `1`
- Selected source-recording cohorts: `1`
- Unique OPEN-led episodes: `1`
- Excluded duplicate projections: `0`
- Omitted unsuccessful runs: `0`
- Market-hour method: `recording_coverage_fallback`

Historical category, settlement, relation-type, and relation-source dimensions come only from append-only observation snapshots. Legacy NULL snapshots remain unknown; current mutable metadata is never used to rewrite them.

## Diagnostic funnel

| stage | input_count | evaluable_count | surviving_count | rate_per_market_hour | note |
| --- | --- | --- | --- | --- | --- |
| midpoint_logical_violation | 1 | 1 | 1 | 9000.0000000000000000000000000000000000000000000001 | midpoint unavailable=0; legacy availability unknown=0 |
| one_contract_spread_survival | 1 | 1 | 1 | 9000.0000000000000000000000000000000000000000000001 | fixed one-contract probe; executable signals without a midpoint reference=0 |
| modeled_fee_survival | 1 | 1 | 1 | 9000.0000000000000000000000000000000000000000000001 | run-specific fee policy and account rounding |
| actual_depth_survival | 1 | 1 | 1 | 9000.0000000000000000000000000000000000000000000001 | all net-executable episodes=1 |
| paper_execution_survival | 1 | 1 | 0 | 0 | failed=1; insufficient future data=0; not evaluated=0 |

Each stage uses its displayed evaluable denominator. Missing midpoint evidence and `insufficient_future_data` are unavailable/unevaluable, not failures.

## Episode-level summaries

| metric | count | median | p90 | maximum | value | note |
| --- | --- | --- | --- | --- | --- | --- |
| unique_episodes | 1 |  |  |  | 1 | one OPEN-led episode after recording/replay cohort deduplication |
| market_hours |  |  |  |  | 0.00011111111111111111111111111111111111111111111111111 | recording_coverage_fallback |
| gross_edge | 1 | 0.086956521739130435 | 0.086956521739130435 | 0.086956521739130435 |  | Decimal type-7 interpolation; missing values are excluded |
| net_edge | 1 | 0.086956521739130435 | 0.086956521739130435 | 0.086956521739130435 |  | Decimal type-7 interpolation; missing values are excluded |
| closed_duration_seconds | 1 | 0.125 | 0.125 | 0.125 |  | Decimal type-7 interpolation; missing values are excluded |
| right_censored_lower_bound_seconds | 0 |  |  |  |  | Decimal type-7 interpolation; missing values are excluded |
| maximum_capital | 1 | 0.920000000000000000 | 0.920000000000000000 | 0.920000000000000000 |  | Decimal type-7 interpolation; missing values are excluded |
| guaranteed_dollars | 1 | 0.080000000000000000 | 0.080000000000000000 | 0.080000000000000000 |  | Decimal type-7 interpolation; missing values are excluded |
| paper_locked_profit | 0 |  |  |  |  | Decimal type-7 interpolation; missing values are excluded |
| total_hypothetical_guaranteed_dollars | 1 |  |  |  | 0.080000000000000000 | Sum of one peak net guarantee per episode; episodes can overlap and are not independent realized profits. |

**Overlap caveat:** Sum of one peak net guarantee per episode; episodes can overlap and are not independent realized profits.

Closed durations and right-censored lower bounds are summarized separately. Open episodes without a terminal boundary: `0`.

## Breakdowns

| dimension | label | episode_count | share_of_episodes | rate_per_market_hour | additive | method | rate_denominator_method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| relation_type | implies | 1 | 1 | 9000.0000000000000000000000000000000000000000000001 | False | multi-label; one episode contributes once to every distinct label | recording_coverage_fallback; deduplicated by source recording; event-time intervals are unioned within each market |
| relation_source | manual | 1 | 1 | 9000.0000000000000000000000000000000000000000000001 | False | multi-label; one episode contributes once to every distinct label | recording_coverage_fallback; deduplicated by source recording; event-time intervals are unioned within each market |
| category | Sports | 1 | 1 | 9000.0000000000000000000000000000000000000000000001 | False | multi-label; one episode contributes once to every distinct label | recording_coverage_fallback; deduplicated by source recording; event-time intervals are unioned within each market |
| time_to_settlement | <1h | 0 | 0 | 0 | True | single fixed bucket per episode, measured at OPEN event time | recording_coverage_fallback; deduplicated by source recording; event-time intervals are unioned within each market |
| time_to_settlement | 1–6h | 1 | 1 | 9000.0000000000000000000000000000000000000000000001 | True | single fixed bucket per episode, measured at OPEN event time | recording_coverage_fallback; deduplicated by source recording; event-time intervals are unioned within each market |
| time_to_settlement | 6–24h | 0 | 0 | 0 | True | single fixed bucket per episode, measured at OPEN event time | recording_coverage_fallback; deduplicated by source recording; event-time intervals are unioned within each market |
| time_to_settlement | 1–7d | 0 | 0 | 0 | True | single fixed bucket per episode, measured at OPEN event time | recording_coverage_fallback; deduplicated by source recording; event-time intervals are unioned within each market |
| time_to_settlement | >7d | 0 | 0 | 0 | True | single fixed bucket per episode, measured at OPEN event time | recording_coverage_fallback; deduplicated by source recording; event-time intervals are unioned within each market |
| time_to_settlement | unknown | 0 | 0 | 0 | True | single fixed bucket per episode, measured at OPEN event time | recording_coverage_fallback; deduplicated by source recording; event-time intervals are unioned within each market |

Relation-type, relation-source, and category tables are multi-label and therefore non-additive. Settlement buckets are single-label and additive. `unknown` means the required observation-time metadata was missing, incomplete, contradictory, or past.

## Interpretation boundary

The report counts recorded, deduplicated research episodes. It does not establish a venue-wide frequency without a declared collection cohort, and it does not model queue priority, hidden liquidity, market impact, custody, or guaranteed real-world fills.
