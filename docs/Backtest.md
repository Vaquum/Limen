# Backtest


Backtest is Limen's trading-economics ledger. It converts binary prediction output into long-flat per-bar returns after declared fill costs, then reports one row of intensive metrics per evaluated round.

The layer answers one question: did the predictive structure retain economic value under the declared execution contract?

## Entry points

Table 1. Backtest is exposed through three paths.

| path | use |
|---|---|
| `uel.experiment_backtest_results` | Experiment-wide table with one row per round. |
| `uel._log.experiment_backtest_results()` | Log-layer method that builds the experiment-wide table. |
| `limen.backtest.backtest_snapshot` | Package function for one per-round prediction table. |

## Snapshot contract

`backtest_snapshot()` is the ledger path used by `Log.experiment_backtest_results()`. It consumes the per-round table returned by `permutation_prediction_performance()` and returns one summary row.

```python
uel._log.permutation_prediction_performance(round_id=0)
```

Table 2. The default strategy is a fixed long-flat contract.

| dimension | rule |
|---|---|
| Signal | Direct snapshot predictions must be binary `0` or `1`; invalid and missing values raise. |
| Position | `prediction == 1` means in market; `prediction == 0` means flat. The default path is long-only. |
| Execution lag | Completed-bar pipelines execute prediction row `t` on the next execution row by default with `execution_lag_bars=1`. |
| Same-row execution | `execution_lag_bars=0` executes on the same tradable row; it does not restore the old raw-row denominator behavior. |
| Price inputs | `open`, `close`, and `price_change` must be numeric. Missing price rows are non-tradable gaps. |
| Price identity | `price_change` must equal `close - open` when all three fields are present. |
| Entry return | Entry-bar gross return is `price_change / open`. |
| Continuation return | Continuation-bar gross return is `close_t / close_{t-1} - 1`. |
| Fill cost | Fee and slippage are applied multiplicatively on entry and exit fills. |
| Position size | `notional_rate` is a deployed-capital fraction in `(0, 1]`. It scales per-bar `edge`, `pnl`, and `cost`. |
| Population | Every bar in the window is counted. A flat bar contributes a real `0`. |
| Units | Return and cost outputs are basis-point scaled. |

This contract makes each round comparable because every output column is computed over the same population: all bars in the evaluation window.

## Economic inputs

Fees and slippage default to `5.0` bps each per fill. A one-entry, one-exit path applies four `5.0` bps adjustments: entry fee, entry slippage, exit fee, and exit slippage.

Configure the economic inputs on the manifest, not on the model:

```python
manifest.set_backtest_config(fee_bps=5.0, slip_bps=5.0, notional_rate=1.0)
```

Each value is either a fixed number or a search-parameter name. Pass a parameter name when cost or position size belongs in the search space.

```python
manifest.set_backtest_config(fee_bps='fee', slip_bps=5.0, notional_rate='size')
```

A YAML/CLI manifest carries the same configuration under `sfd.manifest.backtest`, sibling to `target` and `scaler`.

```yaml
sfd:
  manifest:
    backtest:
      fee_bps: "{fee}"
      slip_bps: 5.0
      notional_rate: "{size}"
  params:
    fee: [1.0, 5.0, 10.0]
    size: [0.1, 0.5, 1.0]
```

The block is optional. When omitted or empty, the defaults remain `fee_bps=5.0`, `slip_bps=5.0`, and `notional_rate=1.0`. `limen validate` rejects unknown keys, negative costs, non-finite costs, `notional_rate` outside `(0, 1]`, and `"{param}"` references missing from `sfd.params`.

## Output ledger

Snapshot backtests produce 20 columns over one population: every bar in the window.

Table 3. Distribution columns report `p5`, `p50`, and `p95`.

| prefix | columns | meaning |
|---|---|---|
| `edge_bps` | `edge_bps_p5`, `edge_bps_p50`, `edge_bps_p95` | Gross per-bar return. |
| `pnl_bps` | `pnl_bps_p5`, `pnl_bps_p50`, `pnl_bps_p95` | Net per-bar return. |
| `cost_bps` | `cost_bps_p5`, `cost_bps_p50`, `cost_bps_p95` | Per-bar gross return minus net return. |
| `drawdown_bps` | `drawdown_bps_p5`, `drawdown_bps_p50`, `drawdown_bps_p95` | Net equity against its running peak. Values are less than or equal to `0`. |

Table 4. Scalar columns are intensive metrics.

| column | meaning |
|---|---|
| `wins_per_bar` | Share of all bars with positive net return. A flat bar is not a win. |
| `pnl_per_bar_bps` | Mean net return per bar. |
| `avg_win_bps` | Mean positive-bar net return; `NaN` when no positive bar exists. |
| `avg_loss_bps` | Mean negative-bar net return; `NaN` when no negative bar exists. |
| `cvar_95_pnl_bps` | Mean of the worst `5%` of per-bar net returns; `NaN` below 20 bars. |
| `trades_per_bar` | Entry count divided by total bar count. |
| `inventory_per_bar` | Mean deployed notional; `notional_rate` multiplied by the share of bars in market. |
| `cost_per_bar_bps` | Mean per-bar gross return minus net return. |

## Strategy boundary

The execution model is swappable. `backtest_snapshot()` validates price columns, calls a strategy, and builds the ledger from the returned per-bar series. The shipped strategy is `long_flat_strategy` in `limen.backtest.long_flat_strategy`.

A strategy receives `predictions`, `open_px`, `close_px`, `price_change`, `execution_lag_bars`, `fee_bps`, and `slip_bps`. It returns `ExecutionResult(pos, gross, net)`, where each field is a per-bar series over the full window.

```python
from limen.backtest.backtest_snapshot import backtest_snapshot
from limen.backtest.long_flat_strategy import long_flat_strategy

round0_backtest = backtest_snapshot(perf, strategy=long_flat_strategy)
```

The strategy owns its signal contract and fill mechanics. `backtest_snapshot()` applies `notional_rate` after the strategy returns, so position sizing remains a ledger-level scale rather than a strategy argument.

## Usage

Use the experiment-wide table to compare rounds.

```python
backtest = uel.experiment_backtest_results
```

Use the package function to inspect one permutation.

```python
from limen.backtest.backtest_snapshot import backtest_snapshot

perf = uel._log.permutation_prediction_performance(round_id=0)
round0_backtest = backtest_snapshot(perf)
```

## Benchmark boundary

Benchmark and backtest answer different questions.

Table 5. The layers are separate because statistical structure and trading economics can fail independently.

| layer | question | input frame |
|---|---|---|
| Benchmark | Does the signal contain predictive structure? | Predictions and realized labels. |
| Backtest | Does that structure survive the declared trading interpretation? | Binary signal, price columns, costs, lag, and notional. |

Limen keeps the layers separate in the API and the docs because benchmark quality is not a substitute for economic inspection.

## Non-goals

Snapshot backtest is not an execution simulator.

Table 6. These concerns sit outside the snapshot contract.

| concern | status |
|---|---|
| Venue-aware execution | Out of scope. |
| Portfolio allocation | Out of scope. |
| Short selling | Out of scope. |
| Latency-aware order modeling | Out of scope. |

## Read next

- Continue to [Trainer](Trainer.md) for promotion of selected experiment rounds into reusable trained sensors.
- Continue to [Log](Log.md) for the post-run workflow that produces backtest inputs.
- Continue to [Benchmark](Benchmark.md) for the prediction-quality layer that precedes trading-economics inspection.
