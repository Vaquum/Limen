# `limen.backtest`

> Evaluate aligned predictions as trading outcomes through either a vectorized snapshot or a sequential simulation.

## Canonical docs

- [Backtest](../../docs/Backtest.md)
- [Benchmark](../../docs/Benchmark.md)
- [Log](../../docs/Log.md)

## What this package owns

Owns Limen's two backtest modes and the assumptions they encode.
Does **not** own signal generation, experiment logging, or raw portfolio bookkeeping beyond what it delegates to `limen.trading`.

## Key entry points

| Entry point | Use it when | Notes |
|-------------|-------------|-------|
| `backtest_snapshot` | You want the fast, vectorized evaluation used across many rounds | Import from `limen.backtest.backtest_snapshot` |
| `BacktestSequential` | You want bar-by-bar account-state simulation | Import from `limen.backtest.backtest_sequential` |

## Adjacent modules

- `limen.log` uses `backtest_snapshot()` to summarize experiment permutations.
- `limen.trading` provides the `Account` ledger used by the sequential path.
- `limen.experiment` and `limen.sfd` sit upstream by producing the predictions that backtests consume.

## Quick orientation

```text
backtest/
├── backtest_snapshot.py     # Vectorized snapshot evaluator
└── backtest_sequential.py   # Stateful account-based simulation
```

## Things to know

- The package root currently does not re-export the backtest helpers, so import from the module paths directly.
- `backtest_snapshot()` is the common analysis path for experiment sweeps because it is simple and fast.
- `BacktestSequential` is the right tool when you need a ledger and bar-by-bar account transitions.
- Snapshot outputs are reported in percent units, while fee and slippage inputs are specified in basis points.

## Read next

- [Backtest](../../docs/Backtest.md)
- [Benchmark](../../docs/Benchmark.md)
- [Log](../../docs/Log.md)
