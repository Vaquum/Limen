# Backtest

Backtest is Limen's trading-economics layer. It answers the question: if a decoder's predictions were traded as a simple long-only signal, what would the return profile look like after costs?

## Current Surface

The main backtest outputs are:

- `uel.experiment_backtest_results`
- `uel._log.experiment_backtest_results()`
- `limen.backtest.backtest_snapshot`
- `limen.BacktestSequential`

## Snapshot Backtest

`backtest_snapshot()` is the default evaluation path used by `Log.experiment_backtest_results()`. It operates on the output of `permutation_prediction_performance()` and computes one result row per experiment round.

Current behavior:

- long-only
- `prediction == 1` means "in market"
- no signal shift is applied
- one round-trip cost is charged per consecutive `1` run
- outputs are reported in percent units

The summary includes:

- trade win rate
- trade expectancy
- max drawdown
- gross and net total return
- mean win and loss
- bars total
- Sharpe per bar
- bars in market
- trade count
- round-trip cost in bps

## Sequential Backtest

`BacktestSequential` is the stateful alternative. It simulates trades bar by bar using an `Account` object and is useful when you want an explicit ledger-style execution trace instead of the vectorized snapshot summary.

## Where It Fits

Backtest comes after benchmark-style prediction analysis. In Limen, a typical evaluation stack is:

1. experiment log
2. confusion and benchmark analytics
3. backtest results

That separation is important because a model can look good on prediction metrics while still producing weak trading economics once position logic and costs are applied.
