# `limen.backtest`

> Evaluate aligned predictions as trading outcomes through a vectorized snapshot.

## Canonical docs

- [Backtest](../../docs/Backtest.md)
- [Benchmark](../../docs/Benchmark.md)
- [Log](../../docs/Log.md)

## What this package owns

Owns Limen's snapshot backtest and the assumptions it encodes.
Does **not** own signal generation, experiment logging, or portfolio bookkeeping.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `backtest_snapshot` | Vectorized evaluation across rounds | Import from `limen.backtest.backtest_snapshot` |
| `long_flat_strategy` | Default execution model, or a template for a new strategy | Returns an `ExecutionResult`; import from `limen.backtest.long_flat_strategy` |

## Adjacent modules

- `limen.log` uses `backtest_snapshot()` to summarize experiment permutations.
- `limen.experiment` and `limen.sfd` sit upstream by producing the predictions that backtests consume.

## Quick orientation

```text
backtest/
├── backtest_snapshot.py     # Vectorized snapshot evaluator (price validation + ledger)
└── long_flat_strategy.py    # Default long-only execution model
```

## Things to know

- The package root currently does not re-export the backtest helpers, so import from the module paths directly.
- `backtest_snapshot()` is the standard analysis path for vectorized experiment-sweep evaluation.
- Snapshot outputs are reported in percent units, while fee and slippage inputs are specified in basis points.

## Read next

- [Backtest](../../docs/Backtest.md)
- [Benchmark](../../docs/Benchmark.md)
- [Log](../../docs/Log.md)
