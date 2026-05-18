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
- direct snapshot predictions must already be binary `0/1`
- invalid or missing direct snapshot predictions raise instead of being coerced
- `prediction == 1` means "in market"; `prediction == 0` means flat
- completed-bar pipelines execute prediction row `t` on the immediate next execution row by default (`execution_lag_bars=1`)
- `execution_lag_bars=0` gives same-row execution of tradable rows, not the old raw-row denominator behavior
- price columns must be numeric; missing price rows are treated as non-tradable gaps
- entry-bar return is based on `price_change / open`
- `price_change` must match `close - open` when all three fields are present
- continuation-bar return is based on `close_t / close_{t-1} - 1`
- fee and slippage costs are applied multiplicatively on entry and exit fills
- `trade_*` metrics are computed from compounded consecutive `1` runs
- output metrics are quantiles over their declared substrate
- return and ratio outputs are basis-point scaled

This makes snapshot backtests fast and comparable across rounds, but it also means they are not trying to be a full execution simulator.

### Output columns

Snapshot backtests produce:

- `edge_per_signal_bps_p5`, `edge_per_signal_bps_p50`, `edge_per_signal_bps_p95`
- `trade_pnl_net_bps_p5`, `trade_pnl_net_bps_p50`, `trade_pnl_net_bps_p95`
- `cost_drag_bps_p5`, `cost_drag_bps_p50`, `cost_drag_bps_p95`
- `rolling_return_net_bps_p5`, `rolling_return_net_bps_p50`, `rolling_return_net_bps_p95`
- `return_on_exposure_p5`, `return_on_exposure_p50`, `return_on_exposure_p95`
- `drawdown_depth_bps_p5`, `drawdown_depth_bps_p50`, `drawdown_depth_bps_p95`
- `drawdown_duration_days_p5`, `drawdown_duration_days_p50`, `drawdown_duration_days_p95`
- `cvar_95_return_bps`

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
