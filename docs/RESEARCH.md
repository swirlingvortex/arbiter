# Arbiter Research Methodology

Arbiter studies whether logical inconsistencies among related Kalshi contracts remain after
executable spreads, fees, finite depth, and simulated non-atomic execution. The analysis is
descriptive and reproducible. It does not forecast outcomes, estimate a causal effect, or claim
that a paper-surviving portfolio could have been filled atomically in a real account.

The tracked example is synthetic fixture evidence. It verifies the data path and calculations;
it is not a sample of Kalshi market behavior. Substantive frequency, profitability, category, or
timing claims require a declared collection window and successfully recorded live data.

## Analysis unit and canonical cohort

The primary unit is one opportunity **episode**, not one solver update. An episode begins with
`OPEN`, may have any number of `UPDATED` observations, and ends with an observed `CLOSED` or a
`RIGHT_CENSORED` boundary. Each episode contributes at most once to a funnel count. Update-heavy
markets therefore do not receive more statistical weight merely because they changed more often.

A single recording may have a live run and several deterministic replay runs with different
run-scoped IDs. Counting all of them would duplicate the same economics. Reports group runs by
their immutable recording identity and choose one decision-bearing run deterministically:

1. prefer a successful replay when it supplies the latency-aware paper evidence;
2. otherwise use the successful source live run;
3. never add a second replay of the same recording to aggregate counts or market-hours;
4. list included and excluded run IDs so the cohort decision is auditable.

The run manifest's embedded configuration, metadata, relations, and fee-policy snapshot is the
authoritative historical input. The report must not substitute current mutable metadata for a
past run. Failed or structurally incomplete runs are reported as exclusions rather than silently
mixed into the default cohort. A sensitivity analysis may study them separately if that choice is
declared before inspecting results.

## Diagnostic funnel

The funnel is evaluated in this fixed order:

1. **Midpoint logical violation.** Fresh two-sided YES midpoints violate at least one trusted
   relation. This is informational Stage 0, not guaranteed arbitrage.
2. **One-contract spread survival.** A diagnostic portfolio using executable asks for at most one
   contract has positive gross worst-state profit.
3. **One-contract modeled-fee survival.** The same diagnostic remains positive after the effective
   fee model and exact rounding.
4. **Actual depth and capacity.** The depth-constrained, quantity-snapped, independently verified
   optimum has positive gross profit; Stage 2 additionally requires net profit and edge above the
   configured thresholds.
5. **Paper-execution survival.** At least one scheduled all-or-none attempt for the episode fills
   every leg within its detection-time price budget and retains positive locked profit.

A midpoint is eligible only when every required market has a fresh bid and ask. Midpoint-unavailable
episodes are reported separately, not silently changed into `False` and not inserted into a
midpoint-conditioned denominator. An executable signal can exist when a two-sided midpoint is
unavailable; it is labeled `executable_without_midpoint_reference` rather than fabricated as a
midpoint survivor. This keeps the locked Stage-0 definition and the executable evidence honest.

The funnel table contains distinct episode counts, the applicable denominator at each step, and
rates per market-hour. Fee survival implies gross probe survival, and paper survival requires a
Stage-2 source observation. Any persisted contradiction is a data-quality failure, not something
the report repairs after the fact.

## Market-hours

The exposure denominator is event-time market-hours, never replay wall-clock runtime. Its primary
definition is the sum of fresh source-run observation windows across subscribed markets:

```text
fresh market-hours = sum(closed_at - opened_at for each fresh market window) / 1 hour
rate                = unique episode count / fresh market-hours
```

Intervals for different markets are intentionally additive: observing two markets for one hour
is two market-hours. Replays contribute no additional exposure. If a fixture or imported replay
has no source observation-window rows, the report may use
`source event-time coverage × subscribed market count`; that fallback must be labeled
`recording_coverage_fallback`, never presented as measured fresh-book exposure. Zero exposure
makes a rate undefined rather than infinite or zero.

## Episode-level measurements

For every eligible episode, the report derives:

- gross edge: `gross_profit / capital_required`;
- net edge: `net_profit / capital_required`;
- maximum executable capital: the largest actual optimum `capital_required` observed in the
  episode;
- guaranteed dollars: the positive fee-adjusted worst-state profit, summarized once per episode;
- event-time duration from `opened_at` to an observed close or censor boundary;
- relation type, relation provenance, category, and time-to-settlement dimensions from the
  historical run input and observation evidence.

Median, p90, and maximum edge and duration are calculated across episode-level values, not raw
updates. Median maximum capital and median guaranteed dollars use one derived value per episode.
Missing economics remain missing; they are not coerced to zero. Reports state their quantile
interpolation convention and the number of observations behind every summary.

The sum of per-episode guarantees is labeled **total hypothetical, overlapping, non-independent
guaranteed dollars**. Opportunities can overlap in time, reuse the same displayed liquidity, need
the same capital, or be alternative solutions to the same information event. Their guarantees
cannot be added and described as realized P&L. The sum is a descriptive scale statistic only.

## Right-censoring and paper coverage

`CLOSED` means a later economic observation showed that the opportunity was no longer present.
`RIGHT_CENSORED` means only that recorded coverage ended while it was still present. Closed
episode durations enter ordinary duration quantiles. A censored duration is a lower bound and is
reported with the censor count and observed-at-risk time; no synthetic close time is imputed.

For paper execution, `insufficient_future_data` is also censoring. It is neither survival nor
failure because the recording ended before the scheduled deadline. Paper survival rates report
the evaluable denominator and the insufficient-coverage count. If formal survival curves are
added later, they must preserve these censor indicators rather than treating them as ordinary
failures.

## Breakdowns and settlement buckets

Relation-type and category tables count an episode once for each label it actually contains.
Because a component may have multiple relation types or categories, exploded breakdown counts
are not necessarily additive to the overall total; the report states this beside the tables.
Semantic contribution uses only approved `semantic_verified` relations and is compared with the
deterministic/manual cohort without treating classifier confidence as trust.

For a multi-market portfolio, time to settlement is measured from episode open to the latest
required market settlement proxy, because capital may remain committed until every leg resolves.
The per-market timestamp precedence is `settlement_ts`, `expected_expiration_time`,
`expiration_time`, `latest_expiration_time`, then `close_time`. If any required market lacks a
usable future timestamp, the episode is `unknown` rather than assigned a guessed bucket.

The fixed buckets are:

| Bucket | Event-time interval |
| --- | --- |
| `<1h` | at least 0 and less than 1 hour |
| `1–6h` | at least 1 hour and less than 6 hours |
| `6–24h` | at least 6 hours and less than 24 hours |
| `1–7d` | at least 24 hours and at most 7 days |
| `>7d` | greater than 7 days |
| `unknown` | missing, contradictory, or already-past required timing |

Bucket tables include episode counts and available edge, duration, capacity, and guaranteed-dollar
summaries. They describe association, not causation.

## Data-quality and analysis rules

- Use only normalized records that passed the production trust, book-state, fee, and independent
  portfolio checks.
- Preserve the configured thresholds and account precision from each run manifest.
- Deduplicate retry-safe observation, attempt, and episode identities before aggregation.
- Keep `NOT_PRESENT`, skipped/stale scans, unsupported fee policies, paper failures, insufficient
  coverage, and censoring visible in the diagnostic counts.
- Never infer a missing fee, category, settlement time, close, or fill as zero.
- State the collection interval, Kalshi environment, relation-source mix, and exclusions.
- Fix funnel definitions, buckets, and primary summaries before inspecting category results.
- Show the full funnel even when the result is null or negative; do not select only profitable
  categories or time windows after seeing the data.
- Treat the fixture as a pipeline check and collected observations as an observational sample,
  not proof of a durable trading strategy.

These rules reduce survivorship, repeated-measures, threshold-shopping, and selection-bias risks.
They do not remove exchange latency, queue position, hidden liquidity, market-impact, or
non-atomic execution risk.

## Authoritative research questions

The report and any narrative analysis are organized around the following ten questions.

1. **How often do logically related prediction-market prices violate coherence?**
   Count unique midpoint-eligible Stage-0 episodes, show the unavailable-reference count, and
   divide by fresh market-hours.
2. **How often do apparent violations survive executable bid/ask spreads?**
   Compare the midpoint cohort with the unique episodes passing the fixed one-contract gross
   probe; separately label executable signals lacking a midpoint reference.
3. **How often do they survive fees?**
   Compare gross-probe and fee-probe counts under the exact run-specific policy, keeping
   unsupported fee metadata visible.
4. **How much executable depth is available?**
   Report the distribution of maximum actual depth-constrained capital and guaranteed dollars,
   including zero eligible observations without inventing a value.
5. **Which relation types produce the most opportunities?**
   Break unique episodes and market-hour rates down by relation type, with multi-label
   non-additivity disclosed.
6. **How long do opportunities persist?**
   Report closed-duration median, p90, and maximum alongside right-censor counts and lower-bound
   exposure.
7. **Do opportunities cluster near settlement or after rapid information events?**
   Use the fixed settlement buckets for the first part. The current schema does not by itself
   establish causal news timing, so any rapid-information analysis must use a declared,
   independently defined proxy and remain exploratory.
8. **Does opportunity magnitude trade off against executable capacity?**
   Compare episode-level net edge and maximum capacity, report sample size, and avoid interpreting
   a fixture correlation or a small observational sample as a stable law.
9. **How often would a latency-aware paper executor still capture all legs?**
   Divide unique paper-surviving episodes by evaluable Stage-2 episodes, with failed and
   insufficient-future-data outcomes shown separately.
10. **How much does semantic relationship discovery add beyond deterministic relationships?**
    Compare approved semantic-relation components with deterministic/manual components using the
    same funnel and exposure definitions. Pending, rejected, uncertain, or stale proposals never
    enter this comparison as trusted constraints.

The synthetic fixture can demonstrate that these computations run and that known economics flow
through the stages. It cannot answer “how often,” establish category rankings, or quantify a
semantic lift in the live venue.

## Reproduction

From the repository root, generate the fixture report and then inspect the notebook:

```bash
.venv/bin/arbiter report \
  --db tests/fixtures/research/arbiter.duckdb \
  --output-dir build/fixture-report
```

The report directory contains `report.md`, `summary.csv`, `funnel.csv`, `breakdowns.csv`, and
`episodes.parquet`. The tracked [`research.ipynb`](../notebooks/research.ipynb) reads those
artifacts rather than reimplementing production metrics. Generated build output remains
untracked; the tracked [fixture report](fixture-report/report.md) makes the expected result
reviewable.

The fixture contains one generic implication episode. Its displayed portfolio costs `$0.92`, has
a minimum terminal payout of `$1.00`, and therefore has `$0.08` guaranteed gross profit under its
explicit zero fee multiplier. A same-time book change removes one leg before simulated execution,
so the all-or-none paper attempt fails and the episode later closes. Those deliberately constructed
numbers validate the reporting path; they are not an estimate of a live-market rate, semantic
increment, fill probability, or realizable return.

## Related work and future work

Cross-platform Kalshi–Polymarket arbitrage is a natural future extension. Arbiter v1 does not
implement it and remains an intra-venue structural/combinatorial system. A future v2 may
generalize the venue and instrument layers, add reviewed equivalent-contract matching, implement
venue-specific fees and execution models, and reuse the existing feasible-world, payoff-matrix,
and worst-case-profit LP core. That extension would need new settlement-equivalence, custody,
latency, funding, and partial-execution validation; none is assumed by the v1 results.
