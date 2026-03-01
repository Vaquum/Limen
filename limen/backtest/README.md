# `limen.backtest`

> Evaluate prediction sequences by simulating trades and computing financial performance metrics.

## Responsibilities

Translates binary model predictions into trade outcomes and summarises the result as a set of performance statistics.

Does **not** own feature engineering, model training, or live execution logic.

## Key concepts

- **BacktestSequential** – stateful bar-by-bar simulator; opens and closes a full position each bar a `1` is predicted, using a live `Account` object to track equity
- **backtest_snapshot** – stateless, vectorised evaluator; operates on a pre-aligned DataFrame and returns a one-row results table; used by the `Log` system to batch-evaluate all permutations at once
- **equity curve** – running account balance recorded after every closed trade (sequential) or compounded bar-return series (snapshot)
- **round-trip cost** – fee + slippage applied once per consecutive run of `1`-predictions in snapshot mode

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `BacktestSequential` | `backtest_sequential.py` | Instantiate and call `.run()` to simulate a single prediction sequence bar-by-bar |
| `backtest_snapshot()` | `backtest_snapshot.py` | Called by `Log.experiment_backtest_results()` to evaluate all permutations at once |

## Dependencies

- **Internal:** `limen.trading.Account` (used by `BacktestSequential` to track positions)
- **External:** `numpy`, `pandas`

## Quick orientation

```text
backtest/
├── backtest_sequential.py   # Stateful bar-by-bar simulator with Account
└── backtest_snapshot.py     # Vectorised, stateless single-pass evaluator
```

## Gotchas / things to know

- `BacktestSequential` is long-only; it buys at `open` and sells at `close` of the same bar
- `backtest_snapshot` uses a hold-while-1 logic: a position continues across consecutive `1` bars and only pays round-trip cost at the exit bar
- All percentage outputs in `backtest_snapshot` are in `%` units (not fractions); Sharpe is per-bar (unitless)
- `BacktestSequential` rounds all results to 2 decimal places; `backtest_snapshot` keeps 1–3 decimal places depending on the column
