# Arbiter v1.0 Final Adversarial Audit

Audit date: 2026-09-08

Scope: the completed Milestones 0–11 implementation and its offline release surface

Release target: `v1.0.0`

## Verdict

**READY FOR v1.0.0.** The audited implementation passes every required offline release gate,
including all 622 tests, and no critical or high-severity correctness, security, or
reproducibility defect remains known.

An initial release commit containing this exact audited tree is ready to receive the `v1.0.0`
tag. This audit intentionally creates neither a tag nor a remote. Missing Kalshi credentials leave
the bounded authenticated live checks outstanding, as documented below; the authoritative
specification makes those checks non-blocking for offline v1 acceptance.

## Recovery boundary

The interrupted audit had completed substantial mathematical, fee, order-book, runtime, and test
review, but had not created this document. Its two coherent correctness fixes were preserved and
reverified. At recovery time the repository had no initial commit, so `git diff` could not
describe the tree or prior audit changes. Recovery therefore used direct file inventory and source
inspection, `git status`, the implementation checkpoint, focused regressions, and a fresh
complete gate. No sibling repository was inspected or modified.

## Issues found, ranked by severity

No critical or high-severity issue was found.

| ID | Severity | Issue | Resolution |
|---|---|---|---|
| A-01 | Medium | `WorldSet` converted input to `int8` before validating it. Fractional values such as `0.9`, `1.9`, and `-0.1` could therefore be silently changed into apparently valid binary states. Production enumeration already emits integers, but the public model boundary was unsafe for malformed or alternate callers. | Validate dimensionality and exact membership in `{0, 1}` on the raw array before making the owned `int8` copy. |
| A-02 | Medium | Exact Decimal post-solve verification allowed recomputed capital to exceed the configured limit by the generic numerical tolerance. A portfolio with exact cost `0.92` could be accepted against a limit of `0.919999999`. | Any exact recomputed capital overage now fails closed as `post_verification_failed`; tolerance remains confined to the floating-point optimization boundary. |
| A-03 | Low | `docs/MATH.md` and the `OpportunityStage` docstring said every stage required all preceding stages. That contradicted the approved M11 behavior: Stage 0 needs a two-sided midpoint and is independent, so a valid executable Stage 1/2 observation may exist without Stage-0 evidence. | Documented Stage 0 as an independent reference diagnostic and Stages 1–3 as the cumulative executable chain. Missing midpoint evidence remains explicitly unavailable, never fabricated as false. |
| A-04 | Low | Package and source metadata still reported `0.1.0` despite the completed `v1.0.0` release target. | Updated source/package versions to `1.0.0`, reinstalled the editable distribution, and added a source/metadata/CLI consistency test. |
| A-05 | Low | `docs/DATA_MODEL.md` described three Migration-7 context columns and omitted `relation_sources_json`; the migration and implementation correctly contain four nullable historical snapshot columns. | Corrected the schema table and migration narrative. |
| A-06 | Low | The README status sentence stopped at Milestone 10 although its following text described completed M11 analytics. | Updated the status to Milestone 11. |

## Fixes and regression coverage

Files changed by this audit:

- `src/arbiter/models/world.py`
- `src/arbiter/solver/lp.py`
- `src/arbiter/models/opportunity.py`
- `src/arbiter/__init__.py`
- `tests/unit/test_worlds.py`
- `tests/unit/test_solver_numerics.py`
- `tests/unit/test_cli.py`
- `pyproject.toml`
- `README.md`
- `docs/MATH.md`
- `docs/DATA_MODEL.md`
- `docs/FINAL_AUDIT.md`

New or strengthened regressions:

- `test_world_set_rejects_values_that_would_become_binary_only_after_integer_cast` covers three
  adversarial fractional inputs that previously survived lossy conversion.
- `test_exact_recomputed_capital_cannot_exceed_limit_within_numeric_tolerance` proves that a
  Decimal-repriced portfolio may never exceed the configured capital bound.
- `test_release_version_is_consistent` requires `pyproject.toml`, installed distribution
  metadata, `arbiter.__version__`, and `arbiter version` to agree on `1.0.0`.
- The existing historical-report regressions were retained and rerun: mutating current market,
  event, series, and relation records cannot change any prior report artifact, and pre-M7 `NULL`
  snapshots remain unknown rather than being backfilled from present data.
- The existing semantic regressions were retained and rerun: confidence never grants trust,
  malformed/tampered evidence is rejected, and stale or orphaned semantic projections are
  transactionally deverified before loading.

The focused post-recovery selection for the newly changed math and CLI boundaries passed 35 tests
with warnings treated as errors. The complete gate subsequently collected and passed 622 tests.

## Adversarial review findings

### Mathematics, numerical boundaries, depth, and fees

- Constraint compilation for implication, equivalence, mutual exclusion, and exactly-one agrees
  with the feasible-world definitions. Components are isolated and capped before exponential
  enumeration. Payoff columns preserve explicit ticker/world ordering and correctly map YES to
  `x` and NO to `1-x`.
- The two-phase LP maximizes the minimum state profit and then minimizes capital at the retained
  guarantee. Every reported portfolio is independently repriced with Decimal outside SciPy after
  quantity normalization. Depth, price priority, state profit, minimum profit, and the now-exact
  capital bound are checked before acceptance.
- Quantity flooring to the documented `0.01` contract grid is conservative and is not claimed to
  find the globally optimal discrete portfolio. Continuous quantities remain diagnostic only.
- Executable instruments are created only from displayed complementary bid levels and inherit
  their exact quantities. A missing, empty, stale, future-dated, resyncing, or inconsistent book
  fails the complete component closed; no opportunity can exceed displayed depth.
- Fee resolution is deterministic at injected event time: an effective event override wins over
  the parent series, clear events fall back to the series, and conflicting, partial, unknown, or
  unsupported metadata cannot reach Stage 2. Taker fills use one rounding ledger per synthetic
  `(ticker, side)` order, with the model fee ceiled to six decimals and balance alignment at the
  configured account precision. Fees remain a post-LP validation step, so Arbiter correctly does
  not claim global fee-optimality.
- Current boundary assumptions were rechecked against Kalshi's official
  [fixed-point representation](https://docs.kalshi.com/getting_started/fixed_point_migration),
  [order-direction convention](https://docs.kalshi.com/getting_started/order_direction),
  [fee-rounding rules](https://docs.kalshi.com/getting_started/fee_rounding), and
  [fee schedule](https://kalshi.com/docs/kalshi-fee-schedule.pdf). The implemented fixed-point
  parsing, exactly-once NO-price complement, quadratic prediction-contract fee formula, and
  six-decimal rounding agree with those sources as reviewed on the audit date.

### Market-data state, live/replay determinism, and paper execution

- Snapshot and delta normalization is exact Decimal, with zero-level removal and negative-depth
  rejection. Sequence handling is isolated per synthetic `(connection_id, sid)` policy; duplicate
  fingerprints are idempotent, while conflicting duplicates, gaps, out-of-order records, and
  ambiguous state latch affected books non-solvable until complete replacement snapshots are
  atomically promoted.
- Live and replay consume the same recorded-event/control union through the same engine path.
  `event_index` is authoritative, event time is nondecreasing, same-time records precede due
  timers, replay speed affects sleeping only, corrupt streams fail closed, and EOF uses explicit
  insufficient-future-data/right-censor outcomes instead of invented events.
- The paper executor fixes the detected portfolio, walks actual later displayed depth within each
  detection-time price tranche, re-resolves execution-time fees, and succeeds only if every leg
  fills and exact locked profit remains positive. It submits no order and models neither atomicity
  nor actual queue position; documentation does not overstate it as real execution.
- Authentication follows the documented RSA-PSS/SHA-256 handshake and query-free signed path in
  Kalshi's [authenticated-request](https://docs.kalshi.com/getting_started/quick_start_authenticated_requests)
  and [WebSocket](https://docs.kalshi.com/getting_started/quick_start_websockets) guidance. Current
  `get_snapshot` recovery usage was also checked against the official
  [changelog](https://docs.kalshi.com/changelog).

### Historical integrity, semantic trust, and persistence

- Migration 7 adds four nullable observation-time dimensions without a backfill. New
  observations snapshot tickers, relation types, relation sources, and per-market context.
  Reporting reads those snapshots and immutable run/observation/paper/window evidence; it does
  not join mutable current metadata to reconstruct history.
- Legacy rows with absent snapshots remain explicitly unavailable. The system does not invent
  historical categories, settlement times, or relation provenance. Observation payload hashes
  reject conflicting retry writes. Application writes are append-only/idempotent, although a
  local DuckDB owner with direct SQL access can still alter the database; this is listed as a
  limitation rather than misrepresented as cryptographic immutability.
- Migrations 1–7 are append-only and fresh/upgrade paths are tested. Run inputs, manifests,
  stream hashes, event bounds, opportunity observations, immutable legs, paper attempts, and
  censor evidence cross-validate before persistence or replay.
- Trusted graph input requires both `verified=True` and an exact source allowlist. Semantic model
  confidence is never authorization. Approval reconstructs current evidence inside one
  transaction; changed rules/timing, stale evidence, projection mismatch, and orphaned direct-SQL
  rows revoke trust before a relation reaches the solver.

### Security, tests, and release-facing claims

- No private key, `.env`, PEM, key, or P8 file was found outside ignored/generated areas, and no
  private-key header was found in production/documentation files. `.gitignore` covers `.env`, key
  formats, DuckDB, raw/Parquet data, generated reports, caches, and build artifacts while retaining
  `.env.example`.
- Authentication errors, doctor output, run inputs, storage payloads, and structured logs exclude
  credential values and private-key contents/paths. The Kalshi client exposes read-only market
  data/subscription operations; no HTTP order submission or real-money execution path exists.
- Core correctness does not depend only on mocks: property tests independently compile/check
  worlds, reprice solver output with Decimal, enforce depth/capital, and compare tiny grid-aligned
  cases to exhaustive portfolio enumeration. Integration tests use real temporary DuckDB and
  Parquet boundaries. Credentialed exchange behavior is deliberately not simulated as a passed
  live check.
- README, architecture, math, data-model, and research claims match implemented commands and
  tests. The tracked research result is clearly fixture-derived, hypothetical profits are labeled
  overlapping/non-independent, and Polymarket/multi-venue support remains documentation-only
  future work.

## Exact final verification evidence

| Command | Result |
|---|---|
| `make install` | PASS; rebuilt and installed editable `arbiter==1.0.0` under Python 3.12.12. |
| Focused changed-boundary test command with `-W error` | PASS; 35 tests passed in 2.78s. |
| `make verify` | PASS; Ruff lint passed, Ruff confirmed 119 Python files formatted, strict mypy reported no issues in 54 source files, and **622 tests passed in 15.91s** on the final pre-commit rerun. |
| `arbiter demo` | PASS; cost `0.92`, state payouts `1.00, 2.00, 1.00`, worst-case payout `1.00`, and guaranteed gross profit `0.08`. |
| `arbiter doctor` | PASS; Python/configuration/endpoints/bounds/data paths passed; WebSocket authentication was informationally unconfigured. |
| `arbiter doctor --require-auth` | Expected credential check failure, exit 1; both required authentication settings are absent and no secret/path was printed. |
| Research report regeneration | PASS; generated one unique episode using `recording_coverage_fallback`. |
| Comparison of `report.md`, `summary.csv`, `funnel.csv`, `breakdowns.csv`, and `episodes.parquet` | PASS; all five regenerated files matched `docs/fixture-report` byte for byte. |
| `arbiter version` plus installed/source metadata check | PASS; CLI, distribution, source, and `pyproject.toml` all report `1.0.0`. |
| Secret-file/header and ignore-policy checks | PASS; zero private credential files or production private-key headers found; representative sensitive/generated paths are ignored. |
| `git diff --check` | PASS. The pre-commit audit also used Ruff/format checks, direct file inspection, and a staged-tree whitespace check because the recovered repository initially had no tracked files. |

Tracked fixture-report SHA-256 values at the accepted boundary:

```text
report.md        2b0ce33a84f4b83946e2280154a5e0b1f0ee32a5d9a5268f814bbcef90cc9cfc
summary.csv      ee37f148488673884b28c5ac941182ca7da3416c774959b85fe3bbb4ff9039be
funnel.csv       96bf4a34307552e6aebbe04c9ebff6e1a9f789c5d3fea0c21e500abe0e5ece97
breakdowns.csv   7169f20e9a4b6f619d17d36dc3582a7a7f2619363ac9cf420dd967ba14879341
episodes.parquet 95d58ad65557659fa1a4b7956d0ff261dfd848d557ff623f50cf9fe238d97b4c
```

## Remaining known limitations

- Paper execution is an all-or-none counterfactual over observed displayed books, not evidence of
  a real fill. It omits exchange queue position, intervening messages not present in a recording,
  market impact, network uncertainty beyond configured latency, and atomic multi-leg execution.
- Quantity flooring is conservative rather than a global discrete optimization. Fees are applied
  after the gross LP, so the selected gross optimum is not claimed to be the global net-fee
  optimum.
- Unknown/unsupported fee types fail closed at Stage 1. Account balance precision defaults to the
  documented direct-member setting; an FCM/non-direct user must configure the applicable grid.
- Feasible-world enumeration intentionally rejects components above the configured 12-market
  limit. No SAT/large-component fallback is part of v1.
- Historical append-only behavior is enforced by application identity/hash checks, not by an
  immutable database service. Legacy pre-M7 dimensions remain unavailable by design.
- The research artifact contains a single deterministic synthetic fixture episode. It validates
  methodology and reproducibility but supports no claim about real-market prevalence or returns.
- The semantic provider smoke is unperformed because the optional provider is disabled; fake
  providers verify the parser, retrieval, persistence, review, and trust boundary offline.
- No remote or release tag is configured. Publication and tag creation remain deliberate operator
  actions after review of the initial release commit.

## Credential-dependent Kalshi checks still outstanding

The following were **not run and are not represented as passing**:

- a bounded authenticated production WebSocket connection and 30-second `arbiter collect` smoke;
- a bounded authenticated `arbiter scan` smoke over verified live components;
- empirical confirmation of Kalshi's actual sequence-number scope beyond the isolated synthetic
  `(connection_id, sid)` policy;
- live observation of reconnect/resynchronization behavior against the production service.

Their absence does not invalidate the completed offline criteria. Until they can be run, sequence
ambiguity and any gap continue to fail closed, and the release should be described as a research
and paper-trading system with credentialed live integration not yet smoke-verified.
