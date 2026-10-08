# `limen.backtest`

> Evaluate aligned predictions as trading outcomes through a vectorized snapshot.

## Canonical docs

- [Backtest](../../docs/Backtest.md)
- [Benchmark](../../docs/Benchmark.md)
- [Log](../../docs/Log.md)

## What this package owns

Owns Limen's research execution, inventory/economics ledger and snapshot assumptions.
Signal generation and operational account management belong to their respective callers.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `backtest_snapshot` | Vectorized evaluation across rounds | `from limen.backtest import backtest_snapshot` |
| `long_flat_strategy` | Default execution model, or a template for a new strategy | Returns an `ExecutionResult`; `from limen.backtest import long_flat_strategy` |
| `trade_execution` | Signed exposure with actual fill costs and persistent episodes | Accepts `TradeInputs` and `TradePolicy` with a `ProductSpec` |

The package root exports `backtest_snapshot`, `long_flat_strategy`, `ExecutionResult`, `trade_execution`, `TradeInputs`, `TradePolicy` and `ProductSpec`.

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

- Configure tunable fixed TP/SL on the manifest; [Backtest](../../docs/Backtest.md#tunable-take-profit-and-stop-loss) defines exits, OHLC validation, and replay requirements.
- `backtest_snapshot()` is the standard analysis path for vectorized experiment-sweep evaluation.
- Snapshot return and cost outputs are basis-point scaled (`*_bps` columns), matching the basis-point fee and slippage inputs.

## Read next

- [Backtest](../../docs/Backtest.md)
- [Benchmark](../../docs/Benchmark.md)
- [Log](../../docs/Log.md)
