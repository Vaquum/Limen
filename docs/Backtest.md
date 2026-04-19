# Backtest

Backtest is Limen's trading-economics layer. It takes prediction outputs and asks the next question after benchmark:

if we traded this signal as a simple long-only strategy, what would the return profile look like after costs?

Limen currently exposes two backtest surfaces:

- a vectorized snapshot backtest used throughout the `Log` layer
- a stateful sequential backtest for ledger-style simulation

## Where Backtest Lives

The most common backtest outputs are:

- `uel.experiment_backtest_results`
- `uel._log.experiment_backtest_results()`
- `limen.backtest.backtest_snapshot`
- `limen.BacktestSequential`

## Snapshot Backtest

`backtest_snapshot()` is the default backtest path used by `Log.experiment_backtest_results()`.

It consumes the per-round table returned by:

```python
uel._log.permutation_prediction_performance(round_id=0)
```

and returns one summary row.

### Current assumptions

The current snapshot backtest is intentionally simple and opinionated:

- long-only
- `prediction == 1` means "in market"
- by default, execution starts on the next bar (`execution_lag_bars=1`)
- `execution_lag_bars=0` reproduces the legacy same-row execution
- trailing rows that cannot execute after the lag are dropped from the tradable window
- entry-bar return is based on `price_change / open`
- continuation-bar return is based on `close_t / close_{t-1} - 1`
- one round-trip cost is charged per consecutive `1` run
- outputs are reported in percent units

This default is meant to align snapshot backtests with the common Limen pattern where a finished bar provides the features for the next decision. If you explicitly want same-row execution, set `execution_lag_bars=0`.

This makes snapshot backtests fast and comparable across rounds, but it also means they are not trying to be a full execution simulator.

### Output columns

Snapshot backtests produce:

- `trade_win_rate_pct`
- `trade_expectancy_pct`
- `max_drawdown_pct`
- `total_return_gross_pct`
- `total_return_net_pct`
- `trade_return_mean_win_pct`
- `trade_return_mean_loss_pct`
- `bar_win_rate_pct`
- `bar_expectancy_pct`
- `bar_return_mean_win_pct`
- `bar_return_mean_loss_pct`
- `tp_mean_return_pct`
- `fp_mean_return_pct`
- `tn_mean_return_pct`
- `fn_mean_return_pct`
- `mean_kelly_pct`
- `bars_total`
- `sharpe_per_bar`
- `bars_in_market_pct`
- `bars_in_market_count`
- `trades_count`
- `trade_runs_count`
- `cost_round_trip_bps`
- `execution_lag_bars`

By default, `trade_*` fields are computed per consecutive `1`-run, which matches the economic idea of one held trade. The `bar_*` fields are always the in-market bar statistics. This is a behavioral change from the earlier bar-level `trade_*` defaults, so old experiment backtest tables are not directly comparable under the same column names. If you need the legacy bar-level `trade_*` behavior, call `backtest_snapshot(..., trades_count_mode='bars')`.

When `actuals` are present in the input table, or when `actual_col` points to an equivalent label column, snapshot also reports `tp_mean_return_pct`, `fp_mean_return_pct`, `tn_mean_return_pct`, and `fn_mean_return_pct`. These are the mean aligned one-bar returns for each confusion bucket after applying `execution_lag_bars`, before run-level holding logic and transaction costs. If your labels already describe the future execution bar, this alignment is correct. If your labels are coincident with the feature bar instead, the confusion-bucket return columns will refer to a different bar than the label.

For inline SFD backtests, Limen now passes the already-aligned test labels straight through to snapshot. Binary SFDs use `y_test` as-is, while regression SFDs pass an explicit directional `actuals` series derived from `y_test > 0`.

For experiment/log backtests, Limen applies the same directional convention to regression rounds before calling snapshot, so post-run `backtest_*` columns stay aligned with the inline reference-architecture path.

`mean_kelly_pct` is estimated from the active return distribution used by the snapshot mode: per held trade by default, or per in-market bar when `trades_count_mode='bars'`. It is the full-Kelly fraction, keeps breakeven observations in the empirical sample, and remains `NaN` when the sample does not contain both winners and losers.

### Typical use

```python
backtest = uel.experiment_backtest_results
```

or for one round:

```python
from limen.backtest.backtest_snapshot import backtest_snapshot

perf = uel._log.permutation_prediction_performance(round_id=0)
round0_backtest = backtest_snapshot(perf)
```

Use the experiment-wide table to compare many rounds. Use the single-round snapshot when you want to study a specific permutation.

## Sequential Backtest

`BacktestSequential` is the more stateful alternative. It simulates trades bar by bar through a trading `Account` object and returns a small ledger-style metrics summary.

```python
from limen import BacktestSequential

backtest = BacktestSequential(start_usdt=30_000)
results = backtest.run(
    actual=perf['actuals'],
    prediction=perf['predictions'],
    price_change=perf['price_change'],
    open_prices=perf['open'],
    close_prices=perf['close'],
)
```

This path is useful when you want an explicit sequence of account updates rather than the vectorized snapshot summary.

### Current sequential outputs

`BacktestSequential.run()` returns:

- `PnL`
- `win_rate`
- `max_drawdown`
- `expected_value`
- `sharpe_ratio`
- `net_long_volume`
- `net_short_volume`
- `net_trade_volume`

## Sequential Ledger Semantics

`BacktestSequential` delegates position bookkeeping to `limen.trading.Account`.

`Account` supports these actions:

- `hold`
- `buy`
- `sell`
- `short`
- `cover`

and exposes:

- `long_position`
- `short_position`
- `net_position`

That said, the current `BacktestSequential.run()` implementation is still a long-only evaluator. It uses:

- `buy`
- `sell`
- `hold`

and does not currently open short or cover actions during the backtest loop.

On a live local sequential run in this repo:

- `net_short_volume` remained `0`
- the action history began `hold, buy, sell, buy, sell, ...`

So the right mental model today is:

- `Account` is capable of both long and short bookkeeping
- `BacktestSequential.run()` currently exercises only the long side

## Backtest Versus Benchmark

Benchmark and backtest should be read together, not treated as substitutes.

- benchmark asks whether the signal contains predictive structure
- backtest asks whether that structure survives a specific trading interpretation

Examples of why the layers diverge:

- a signal can have decent precision but still spend too much time in market
- a signal can separate TP and FP weakly yet still avoid the worst losses
- a signal can score well statistically but lose most of its edge once costs are charged

That is why Limen keeps the layers separate in both the API and the docs.

## What Snapshot Backtest Does Not Try To Do

The snapshot backtest is not:

- a venue-aware execution simulator
- a portfolio allocator
- a short-selling engine
- a latency-aware order model

Those concerns belong downstream from Limen or in more specialized evaluation layers.

## Read Next

- Continue to [Trainer](Trainer.md) if you want to promote strong experiment rounds into reusable trained sensors.
- Continue to [Log](Log.md) for the broader post-run workflow that produces the backtest inputs.
- Continue to [Benchmark](Benchmark.md) if you want the prediction-quality layer that should usually be inspected before the trading-economics layer.
