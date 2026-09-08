# Arbiter Mathematics

This document explains the mathematical and execution engine implemented through Milestone 11. It deliberately
uses a small implication example, but every step is generic: no solver branch knows anything
about Messi, Argentina, sports, or Kalshi.

## Outcomes and logical constraints

Each binary market has an outcome variable `x`:

- `x = 1` when its YES contract settles at $1;
- `x = 0` when its NO contract settles at $1.

For the example, let `M` mean “Messi scores” and `A` mean “Argentina scores.” The trusted
logical relation is `M => A`. A binary implication is equivalent to the inequality
`x_M <= x_A`. It rules out only `M=1, A=0`.

Arbiter enumerates candidate bit patterns in a declared ticker-column order and retains only
those satisfying every compiled constraint. In `(M, A)` order, the feasible worlds are:

| World | M | A |
|---|---:|---:|
| S1 | 0 | 0 |
| S2 | 0 | 1 |
| S3 | 1 | 1 |

The impossible `10` row never enters the optimization.

## Instruments and payoff matrix

An instrument is one bounded opportunity to buy YES or NO at an executable price. A YES
instrument pays its market outcome `x`; a NO instrument pays `1-x`. For this example:

- buy Argentina YES for `$0.62`;
- buy Messi NO for `$0.30`.

With worlds as rows and instruments `(Argentina YES, Messi NO)` as columns, the terminal
payoff matrix is

```text
    Argentina YES   Messi NO
S1        0             1
S2        1             1
S3        1             0
```

For quantity vector `q`, matrix multiplication `Aq` gives the terminal payout in every
world. The purchase cost is `c q`, where `c = (0.62, 0.30)`.

## Worst-case-profit linear program

Arbiter introduces one more variable, `g`, for the profit guaranteed across all feasible
worlds. It solves

```text
maximize    g
subject to  A[s] q - c q >= g       for every feasible world s
            0 <= q[j] <= depth[j]   for every instrument j
            c q <= capital_limit    when a limit is configured
```

These expressions are linear. SciPy minimizes, so Arbiter negates the `g` objective and
rewrites each world inequality as `(c - A[s])q + g <= 0`.

There can be several portfolios with effectively the same best guarantee. Arbiter therefore
uses two phases:

1. maximize `g`;
2. keep `g` within a tiny numerical epsilon of that optimum and minimize purchase cost.

This stable tie-break prevents an unnecessarily large portfolio from being reported.

## Independent verification

SciPy receives floating-point arrays because that is its numerical interface. A tiny floating
residual is not accepted as money. After solving, Arbiter converts quantities with
`Decimal(str(value))`, normalizes only boundary-sized numerical noise, and independently
recomputes outside SciPy:

```text
cost       = sum(price[j] * quantity[j])
payout[s]  = sum(payoff[s,j] * quantity[j])
profit[s]  = payout[s] - cost
guarantee  = min(profit[s])
```

It rejects nonfinite values, depth or capital violations, and guarantees at or below the
configured tolerance.

For one unit of each example instrument, the cost is `$0.92`, state payouts are `$1.00`,
`$2.00`, and `$1.00`, and the worst-case gross profit is therefore `$0.08`.

## Displayed depth and executable quantity

Each real complementary bid level becomes a separate bounded LP variable. For example, a NO
bid for `3.50` contracts at `$0.38` creates a YES-buy instrument at `$0.62` with maximum
quantity `3.50`; it does not create unlimited liquidity. Missing, empty, stale, future-dated,
or resynchronizing books make the whole connected component ineligible for executable solving.

The current fixed-point exchange documentation supports a `0.01`-contract quantity grid.
After the continuous LP returns, Arbiter normalizes only tiny solver noise and floors every
quantity toward zero to that grid. It then recomputes cost, payouts, state profits, depth, and
capital using `Decimal`. The continuous answer remains a diagnostic; only the snapped and
reverified answer is executable. Flooring is conservative, but it does not prove that the
result is the globally best portfolio on the discrete grid.

## Current Kalshi prediction-contract fees

The implemented production policy is versioned
`kalshi-prediction-fees-2026-07-07+rounding-2026-09-03`. For one fill, let:

- `M` be the series/event fee multiplier;
- `C` be the contract quantity;
- `P` be the purchased contract price in dollars, between zero and one.

For displayed-liquidity purchases, Arbiter uses the current quadratic taker formula:

```text
raw taker fee = M * 0.07 * C * P * (1 - P)
```

In plain English, the fee is largest near a 50-cent contract and smaller near either payout
boundary. YES and NO purchases use the same formula at their own purchased price. The current
maker variants are modeled for completeness: ordinary `quadratic` has no maker fee,
`quadratic_with_maker_fees` uses `0.0175` in place of `0.07`, and
`quadratic_with_combo_maker_fees` uses `0.035`. The scanner always buys displayed liquidity,
so every proposed entry fill is classified as taker.

Fees are a post-LP validation layer. Arbiter first finds and exactly verifies the
gross-optimal portfolio, then applies fees to that portfolio. It therefore does **not** claim
that the gross-optimal portfolio is also the globally fee-optimal portfolio.

### Exact two-stage rounding

Kalshi first rounds each raw model fee upward to six decimal dollars:

```text
trade_fee = ceil(raw_fee to $0.000001)
```

A buy has signed revenue `R = -(P * C)`. Let `d` be the account balance grid: `$0.0001` for a
direct member, or `$0.01` for a non-direct/FCM account. Arbiter then computes:

```text
unaligned balance change = R - trade_fee
aligned balance change   = floor(unaligned balance change to grid d)
rounding fee             = unaligned - aligned
```

Because buyer revenue is negative, flooring means moving to the next no-greater grid value.
The positive difference is the rounding fee. For the documented non-direct example at
`P=0.055`, `C=1`, and `M=1`, the raw fee is `$0.00363825`, the trade fee is `$0.003639`, the
aligned balance change is `-$0.060000`, and the resulting rounding fee is `$0.001361`; the
fill's net fee is `$0.005000`.

Rounding residuals accumulate across all fills of exactly one exchange order. In Arbiter,
all consumed price levels for one `(market ticker, purchased side)` portfolio leg form one
synthetic taker order and share one explicit `FeeRoundingLedger`. Different legs never share a
ledger. When accumulated residual reaches a whole account-grid increment, that increment may
be rebated. A rebate is bounded so the current fill's net fee can never become negative:

```text
net fill fee = trade_fee + rounding_fee - rebate >= 0
```

This explicit ledger is the one intentional extension to the specification's otherwise pure
per-fill fee protocol; current exchange rounding cannot be represented correctly by a
stateless function.

### Effective policy and fail-closed behavior

Fee policy is resolved at an injected event/replay time, never by consulting the wall clock.
The latest matching event fee change whose `scheduled_ts` is at or before that time overrides
the parent Series policy. A scheduled pair of null override values clears the event override
and restores the Series default. Future changes are recorded but do not apply early.

Missing fee type and multiplier together are treated conservatively as the documented
standard quadratic policy with multiplier one; a missing multiplier is never interpreted as
a waiver. A multiplier without a fee type, partial Event override, conflicting changes at the
same timestamp, ticker mismatch, `flat`, perpetual, or another insufficiently documented fee
type fails closed. The gross candidate may remain Stage 1 with reason
`unsupported_fee_model`, but it cannot claim a numeric net guarantee.

For supported policies, total entry fees are subtracted from every feasible-state gross
profit:

```text
net_profit[s] = gross_profit[s] - total_entry_fees
net guarantee = min(net_profit[s])
net edge      = net guarantee / capital_required
```

The same deterministic entry fee appears in every state because these fees are incurred when
the portfolio is purchased, before settlement is known.

## Opportunity evidence stages

- **Stage 0 — logical price inconsistency:** a fresh two-sided YES midpoint violates a
  coherence rule. This is informational and is not called guaranteed arbitrage.
- **Stage 1 — gross snapshot-executable:** snapped displayed asks and depth produce a strictly
  positive independently verified worst-state gross profit.
- **Stage 2 — net snapshot-executable:** supported fees and exact rounding leave both net
  guaranteed profit and net edge strictly above their configured thresholds.
- **Stage 3 — paper-execution surviving:** the later non-atomic latency simulation finds every
  required leg still fillable at or better than its budgeted price.

Stage 0 is an independent reference-price diagnostic. Stages 1 through 3 form the executable
evidence chain: Stage 2 requires Stage 1, and Stage 3 requires Stage 2. An executable Stage 1 or
Stage 2 signal can exist when a two-sided midpoint is unavailable; analytics label that condition
explicitly rather than fabricating Stage-0 evidence. No stage implies that real multi-leg execution
is atomic.

## Deterministic replay and paper execution

The recorded `event_index` is the authoritative total order. Replay advances a logical event
clock using recorded `local_received_ts`; playback speed changes only wall-clock sleeping. At one
timestamp, every recorded event is applied in index order before a debounce or paper timer due at
that same instant. Freshness, opportunity durations, and execution deadlines therefore have the
same meaning at `max` speed and at any positive numeric speed.

An OPEN or economically UPDATED Stage 2 observation schedules a paper attempt at

```text
execution time = detection time + configured latency
```

The requested portfolio is fixed at detection: the paper executor does not solve a different
portfolio with hindsight. At the deadline it rebuilds executable fills from the then-current
fresh books. For each leg, quantities consume price levels no worse than the detection-time
budget, including cumulative tranches when several levels belong to one leg. The attempt then
re-resolves the effective fee policy at execution time and recomputes exact fees and rounding.

Success requires every required quantity to fill and the resulting portfolio to retain a
strictly positive locked profit:

```text
locked profit = minimum feasible-world terminal payout
                - actual acquisition cost
                - actual entry fees
```

The simulation is deliberately all-or-none. If one leg lacks depth, exceeds its price budget,
has stale or resynchronizing data, uses an unsupported fee policy, or leaves nonpositive locked
profit, all legs are recorded as an unsuccessful paper attempt; no partial-fill portfolio is
claimed. If the recording ends before the deadline, the distinct result is insufficient future
data rather than a guessed fill.

Opportunity episodes still open at a terminal `RunEndedEvent` are right-censored. Censoring says
only that evidence coverage ended; it is not an observed CLOSED transition. Replay reconstructs
that distinction and compares economic projections independently of run-specific IDs and measured
solver latency.
