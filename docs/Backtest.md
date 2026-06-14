# Backtest

Backtest is Limen's trading-economics layer. It takes prediction outputs and asks the next question after benchmark:

The layer maps a binary signal into long-only returns after fees and slippage.

Limen exposes a vectorized snapshot backtest, used throughout the `Log` layer.

## Where backtest lives

Backtest outputs are exposed through:

- `uel.experiment_backtest_results`
- `uel._log.experiment_backtest_results()`
- `limen.backtest.backtest_snapshot`

## Snapshot Backtest

`backtest_snapshot()` is the default backtest path used by `Log.experiment_backtest_results()`.

It consumes the per-round table returned by:

```python
uel._log.permutation_prediction_performance(round_id=0)
```

and returns one summary row.

### Current assumptions

The current snapshot backtest has a fixed long-flat contract:

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
- position size is a tunable fraction of capital (`notional_rate`, default `1.0` = all-in) that scales the per-bar `edge`, `pnl`, and `cost` and sets the average deployed notional
- every column is computed per bar over all bars in the window (a flat bar counts as a real `0`)
- distribution columns carry p5/p50/p95; the rest are single intensive (run-length-invariant) scalars
- return and cost outputs are basis-point scaled

The fee and slippage rates default to `5.0` bps each per fill (`20.0` bps for one entry and one exit) and are configured on the manifest, not on the model: `manifest.set_backtest_config(fee_bps=5.0, slip_bps=5.0)`. Each value is a fixed number or a search-param name — pass a param name to sweep cost across the search (for example `set_backtest_config(fee_bps='fee')` with `fee` in `params()`), to match a venue's costs, make cost a search dimension, or stress-test how sensitive an edge is to the cost assumption. Omitting the config keeps the 5 + 5 default, so existing experiments are unchanged.

`set_backtest_config` also takes `notional_rate` — the fraction of capital deployed while in position, in `(0, 1]` (default `1.0`, all-in; a 10% book is `notional_rate=0.1`). It scales `edge`, `pnl`, and `cost` together so the ledger reflects the account at that bet size (`drawdown` also moves with bet size, but path-dependently — it compounds against a running peak, so it is not a linear multiple of the full-size drawdown), leaves `wins_per_bar` and `trades_per_bar` untouched, and turns `inventory_per_bar` into the average deployed notional. Like the costs it is a fixed number or a search-param name, so one search can sweep bet size.

In a YAML/CLI manifest the same configuration is a `backtest:` block under `sfd.manifest`, sibling to `target:` and `scaler:`. Each value is a fixed number or a `"{param}"` reference into `sfd.params`, mirroring how indicator and target params are swept:

```yaml
sfd:
  manifest:
    backtest:
      fee_bps: "{fee}"        # swept across the search
      slip_bps: 5.0           # fixed
      notional_rate: "{size}" # swept bet size, in (0, 1]
  params:
    fee: [1.0, 5.0, 10.0]
    size: [0.1, 0.5, 1.0]
```

The block is optional and applies to both `ml` and `rule_based` manifests; omitting it — or leaving it empty — keeps the defaults (5 + 5 bps, all-in). `limen validate` rejects an unknown key, a negative or non-finite cost, a `notional_rate` outside `(0, 1]`, and a `"{param}"` reference missing from `sfd.params`.

This makes snapshot backtests fast and comparable across rounds, but it also means they are not trying to be a full execution simulator.

### Output columns

Snapshot backtests produce 20 columns over one population — every bar in the window.

Per-bar distributions (`p5` / `p50` / `p95`):

- `edge_bps_*` — gross per-bar return
- `pnl_bps_*` — net per-bar return
- `cost_bps_*` — per-bar cost (gross minus net)
- `drawdown_bps_*` — net equity against its running peak (≤ 0)

Intensive scalars:

- `wins_per_bar` — share of all bars with positive net return (a flat bar is not a win, so it cannot exceed the in-market share — `inventory_per_bar` under the all-in model)
- `pnl_per_bar_bps` — mean net return per bar
- `avg_win_bps`, `avg_loss_bps` — mean of the positive / negative bars (NaN when there are none)
- `cvar_95_pnl_bps` — mean of the worst 5% of per-bar net returns (NaN below 20 bars)
- `trades_per_bar` — entries per bar (turnover)
- `inventory_per_bar` — mean position held per bar; the average deployed notional (`notional_rate` × the share of bars in market)
- `cost_per_bar_bps` — mean cost per bar

### Pluggable strategy

The execution model — how a `0/1` signal becomes per-bar returns — is a swappable `strategy`. `backtest_snapshot()` validates the price columns, calls the strategy, and builds the ledger from what it returns; it does not itself know how positions or costs are formed.

The default is `long_flat_strategy` (the long-only, hold-while-1 model described above). A strategy is any callable with this shape:

```python
from limen.backtest.long_flat_strategy import ExecutionResult

def my_strategy(predictions, open_px, close_px, price_change, *,
                execution_lag_bars, fee_bps, slip_bps) -> ExecutionResult:
    position = predictions.astype(float).shift(execution_lag_bars, fill_value=0.0)
    gross_return = position * price_change / open_px

    entry_mask = (position == 1.0) & (position.shift(1, fill_value=0.0) == 0.0)
    exit_mask = (position == 1.0) & (position.shift(-1, fill_value=0.0) == 0.0)
    fee = fee_bps / 10_000.0
    slip = slip_bps / 10_000.0
    cost_mult = position.copy()
    cost_mult[:] = 1.0
    cost_mult.loc[entry_mask] *= (1.0 - fee) / (1.0 + slip)
    cost_mult.loc[exit_mask] *= (1.0 - fee) * (1.0 - slip)

    net_return = ((1.0 + gross_return) * cost_mult) - 1.0
    return ExecutionResult(pos=position, gross=gross_return, net=net_return)
```

`ExecutionResult` carries three per-bar series over every bar in the window: `pos` (position held — `0`/`1` here, a deployed fraction once sizing exists), `gross` (per-bar return before costs), and `net` (per-bar return after costs). Every ledger column flows from that triple. Pass a strategy through the `strategy` argument:

```python
round0_backtest = backtest_snapshot(perf, strategy=my_strategy)
```

The strategy owns its own signal contract — `long_flat_strategy` requires binary `0/1` and validates it — and applies its own costs, since only it knows where entries and exits fall.

`backtest_snapshot` forwards only the fill-shaping kwargs (`execution_lag_bars`, `fee_bps`, `slip_bps`) to the strategy. `notional_rate` is *not* a strategy concern — it is a uniform scale on the returned triple, so `backtest_snapshot` applies it after the strategy returns, leaving the strategy contract stable as that knob (and others like it) is added.

### Usage

```python
backtest = uel.experiment_backtest_results
```

or for one round:

```python
from limen.backtest.backtest_snapshot import backtest_snapshot

perf = uel._log.permutation_prediction_performance(round_id=0)
round0_backtest = backtest_snapshot(perf)
```

Use the experiment-wide table to compare rounds. Use the single-round snapshot for one permutation.

## Backtest versus benchmark

Benchmark and backtest should be read together, not treated as substitutes.

- benchmark asks whether the signal contains predictive structure
- backtest asks whether that structure survives a specific trading interpretation

Examples of why the layers diverge:

- a signal can have high precision but excessive market exposure
- a signal can have low TP/FP separation yet still avoid the largest losses
- a signal can score high statistically but lose edge after costs

That is why Limen keeps the layers separate in both the API and the docs.

## What snapshot backtest does not try to do

The snapshot backtest is not:

- a venue-aware execution simulator
- a portfolio allocator
- a short-selling engine
- a latency-aware order model

Those concerns belong downstream from Limen or in more specialized evaluation layers.

## Read next

- Continue to [Trainer](Trainer.md) for promotion of selected experiment rounds into reusable trained sensors.
- Continue to [Log](Log.md) for the broader post-run workflow that produces the backtest inputs.
- Continue to [Benchmark](Benchmark.md) for the prediction-quality layer that precedes trading-economics inspection.
