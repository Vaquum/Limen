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
- every column is computed per bar over all bars in the window (a flat bar counts as a real `0`)
- distribution columns carry p5/p50/p95; the rest are single intensive (run-length-invariant) scalars
- return and cost outputs are basis-point scaled

The fee and slippage rates default to `5.0` bps each per fill (a ~20 bps round trip) and are configured on the manifest, not on the model: `manifest.set_backtest_config(fee_bps=..., slip_bps=...)`. Each value is a fixed number or a search-param name — pass a param name to sweep cost across the search (for example `set_backtest_config(fee_bps='fee')` with `fee` in `params()`), to match a venue's costs, make cost a search dimension, or stress-test how sensitive an edge is to the cost assumption. Omitting the config keeps the 5 + 5 default, so existing experiments are unchanged.

In a YAML/CLI manifest the same configuration is a `backtest:` block under `sfd.manifest`, sibling to `target:` and `scaler:`. Each cost is a fixed number or a `"{param}"` reference into `sfd.params`, mirroring how indicator and target params are swept:

```yaml
sfd:
  manifest:
    backtest:
      fee_bps: "{fee}"   # swept across the search
      slip_bps: 5.0      # fixed
  params:
    fee: [1.0, 5.0, 10.0]
```

The block is optional and applies to both `ml` and `rule_based` manifests; omitting it — or leaving it empty — keeps the 5 + 5 default. `limen validate` rejects an unknown key, a negative or non-finite cost, and a `"{param}"` reference missing from `sfd.params`.

This makes snapshot backtests fast and comparable across rounds, but it also means they are not trying to be a full execution simulator.

### Output columns

Snapshot backtests produce 21 columns over one population — every bar in the window.

Per-bar distributions (`p5` / `p50` / `p95`):

- `edge_bps_*` — gross per-bar return
- `pnl_bps_*` — net per-bar return
- `cost_bps_*` — per-bar cost (gross minus net)
- `drawdown_bps_*` — net equity against its running peak (≤ 0)

Intensive scalars:

- `win_rate` — share of bars with positive net return
- `pnl_per_bar_bps` — mean net return per bar
- `avg_win_bps`, `avg_loss_bps` — mean of the positive / negative bars (NaN when there are none)
- `cvar_95_pnl_bps` — mean of the worst 5% of per-bar net returns (NaN below 20 bars)
- `trades_per_bar` — entries per bar (turnover)
- `in_market_per_bar` — share of bars holding a position
- `inventory_per_bar` — mean fraction of capital deployed per bar
- `cost_per_bar_bps` — mean cost per bar

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
