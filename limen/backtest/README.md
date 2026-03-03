# `limen.backtest`

> Evaluate trading strategies against historical predictions using either a sequential simulation or a vectorised snapshot approach.

## Responsibilities

Owns the two backtesting modes used to measure the real-world value of model predictions.
Does **not** own data fetching, feature engineering, or model training — it only consumes pre-aligned prediction arrays / DataFrames.

## Key concepts

- **BacktestSequential** – bar-by-bar simulation that tracks an `Account` object, applies per-trade fees, and accumulates an equity curve.
- **backtest_snapshot** – fully vectorised, DataFrame-in / DataFrame-out function that computes performance statistics in a single pass without maintaining state.
- **Round-trip cost** – `backtest_snapshot` charges one round-trip fee per consecutive run of `prediction == 1`, applied on the exit bar.
- **Equity curve** – `BacktestSequential` records USDT value after every trade; used to compute max drawdown and Sharpe.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `BacktestSequential` | `backtest_sequential.py` | Instantiate once per experiment round, then call `.run()` with aligned arrays |
| `backtest_snapshot()` | `backtest_snapshot.py` | Called by `limen.log` to produce per-permutation backtest stats from a predictions DataFrame |

## Dependencies

- **Internal:** `limen.trading` — `BacktestSequential` delegates buy/sell/short/cover ledger tracking to `Account`
- **External:** `numpy`, `pandas`

## Quick orientation
```text
backtest/
├── backtest_sequential.py   # Stateful, bar-by-bar simulation using Account
└── backtest_snapshot.py     # Stateless, vectorised snapshot evaluator
```

## Gotchas / things to know

- `BacktestSequential` is long-only; a `prediction == 1` triggers a full buy-at-open / sell-at-close within the same bar.
- `backtest_snapshot` supports a `trades_count_mode` param: `'runs'` counts entry events, `'bars'` counts individual bars-in-market.
- All percentage outputs from `backtest_snapshot` are in `%` units, not fractions.
- Fee and slippage in `backtest_snapshot` are in basis points (default: 5 bps each, 20 bps round-trip total).
