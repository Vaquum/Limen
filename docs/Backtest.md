# Backtest

Backtest is Limen's trading-economics ledger. It converts binary prediction output into long-flat per-bar returns after declared fill costs, then reports one row of intensive metrics per evaluated round.

The layer evaluates entry signals together with declared costs, sizing, and optional fixed take-profit/stop-loss exits.

## Prerequisites

- aligned binary predictions and OHLC/price-change columns
- explicit fee, slippage, notional, and execution-lag assumptions
- retained round artifacts for post-run experiment-wide analysis

## Risk boundary

Backtest output is research evidence, not investment advice, trading advice, execution simulation, regulatory approval, or a promise of future performance. Past performance is not predictive, digital-asset trading can result in total loss of capital, and snapshot backtests do not model venue queues, latency, borrow, liquidation, funding, portfolio constraints, or live order execution.

## Entry points

Table 1. Backtest is exposed through three paths.

| path | use |
|---|---|
| `uel.experiment_backtest_results` | Experiment-wide table with one row per round. |
| `uel._log.experiment_backtest_results()` | Log-layer method that builds the experiment-wide table. |
| `limen.backtest.backtest_snapshot.backtest_snapshot` | Module function for one per-round prediction table. |

## Snapshot contract

`backtest_snapshot()` returns one summary row as a metrics dict keyed by the ledger columns. Shipped reference architectures use the same execution path inline and during manifest-backed post-run replay. With configured TP/SL, replay reconstructs private OHLC inputs; the public prediction-performance table keeps its existing columns.

```python
uel._log.permutation_prediction_performance(round_id=0)
```

Table 2. The default strategy is a fixed long-flat contract.

| dimension | rule |
|---|---|
| Signal | Direct snapshot predictions must be binary `0` or `1`; invalid and missing values raise. |
| Position | `prediction == 1` means in market; `prediction == 0` means flat. The default path is long-only. |
| Execution lag | Completed-bar pipelines execute prediction row `t` on the next execution row by default with `execution_lag_bars=1`. |
| Same-row execution | With both barriers disabled, `execution_lag_bars=0` executes on the same tradable row and prices the entry at the previous close, before the signal exists; it is a research diagnostic, not a deployable configuration. |
| Price inputs | `open`, `close`, and `price_change` must be numeric. Missing price rows are non-tradable gaps. `open` and `price_change` are validated and gate tradability but do not price returns. |
| Price identity | `price_change` must equal `close - open` when all three fields are present. |
| Entry return | Entry-bar gross return is `close_t / close_{t-1} - 1`: the fill is the close of the bar before the execution row (the signal bar's close under the default `execution_lag_bars=1`), and an execution row whose prior close is missing or zero is non-tradable. |
| Continuation return | Continuation-bar gross return is `close_t / close_{t-1} - 1`. |
| Fill cost | The entry fee is `fee_bps` of the entry notional, paid from cash at entry; the exit fee is `fee_bps` of the exit proceeds; slippage adjusts the fill prices (`close × (1 + slip)` at entry, `close × (1 − slip)` at exit). Within a trade, equity is the position minus the entry fee (on the exit bar, the proceeds after the exit fee minus the entry fee), so per-bar net returns are equity returns and the entry fee does not compound with the position. |
| Position size | `notional_rate` is a deployed-capital fraction in `(0, 1]`. It scales per-bar `edge`, `pnl`, and `cost`. |
| Population | Every bar in the window is counted. A flat bar contributes a real `0`. |
| Units | Return and cost outputs are basis-point scaled. |

This contract makes each round comparable because every output column is computed over the same population: all bars in the evaluation window.

## Economic inputs

Fees and slippage default to `5.0` bps each per fill. A one-entry, one-exit path pays `fee_bps` of the entry notional from cash at entry, `fee_bps` of the exit proceeds at exit, and slippage on both fill prices, so a trade with gross return `g` ends with equity `(1 + g)(1 − fee)(1 − slip) / (1 + slip) − fee` per unit of entry notional, a net return of `g − fee × (2 + g)` at zero slippage, the deployed arithmetic. Because equity within a trade is the position minus the entry fee, `cost_bps` on a held bar that is neither the entry nor the exit is the small gap between the position's return and the equity return: slightly negative on an up bar, slightly positive on a down bar.

Configure the economic inputs on the manifest, not on the model:

```python
manifest.set_backtest_config(fee_bps=5.0, slip_bps=5.0, notional_rate=1.0)
```

Each economic value is either a fixed number or a search-parameter name. Pass a parameter name when cost or position size belongs in the search space.

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

## Tunable take-profit and stop-loss

`take_profit_bps` and `stop_loss_bps` are separate manifest backtest fields. Each accepts a fixed number, a Python search-parameter name (plain or single-braced), or `None`. `None` disables that barrier independently. TP must be finite and greater than zero; SL must be finite and strictly between zero and 10,000 bps. Zero and booleans are invalid. Missing references raise a field-naming `ValueError`.

`set_backtest_config()` replaces the entire configuration. Preserve declared costs and sizing when adding barriers:

```python
costs = manifest.backtest_config
manifest.set_backtest_config(
    fee_bps=costs.fee_bps,
    slip_bps=costs.slip_bps,
    notional_rate=costs.notional_rate,
    take_profit_bps='tp',
    stop_loss_bps='sl',
)
```

In the SFD's `params()`, add the two candidate lists alongside the entry-model parameters. YAML uses braced references:

```yaml
sfd:
  manifest:
    backtest:
      fee_bps: 10.0
      slip_bps: 5.0
      notional_rate: 0.5
      take_profit_bps: "{tp}"
      stop_loss_bps: "{sl}"
  params:
    tp: [null, 50.0, 100.0]
    sl: [null, 25.0, 50.0]
```

`limen validate` checks every referenced candidate, including null values. `manifest.resolve_backtest_config(round_params)` returns all five resolved fields, including disabled `None` barriers; an absent configuration returns `{}`. Each round overwrites both barrier fields, so disabled candidates cannot inherit an earlier round's exits.

### Exit contract

Table 3. Barriers use gross raw prices, before fill costs.

| Dimension | Rule |
|---|---|
| Entry reference | On entry execution row `t`, raw `E = close[t-1]`. TP is `E × (1 + take_profit_bps / 10000)`; SL is `E × (1 − stop_loss_bps / 10000)`. Levels stay fixed for the trade. |
| Opening gap | Check open first. An open at/below SL exits at that adverse open. An open at/above TP exits at TP, capping favorable gaps. |
| Intrabar touch | Otherwise check `low <= SL`, then `high >= TP`. If both touch, SL wins. Comparisons use exact float64 values, without a touch tolerance. |
| Precedence | Check barriers on every held bar, including the last episode/window bar, before ordinary close liquidation. |
| Re-entry | A barrier exit suppresses the remainder of that lagged `1` episode. A subsequent lagged `0` re-arms the strategy; the next lagged `1` can enter. Earliest re-entry is two rows after the exit. |
| Window start | Each split/window starts flat and armed. An episode active at its first execution row enters using the preceding close within that window; earlier stops do not carry across boundaries. |
| Lag | Enabled barriers require a positive integer lag; bool and fractional values raise. Lag at/above window length is valid and all-flat. |
| Costs and sizing | The existing fill engine applies entry/exit fees and slippage once. TP's actual sell fill is `TP × (1 − slip)`. `notional_rate` scales the ledger without changing levels or exit decisions. |

On an exit row with raw exit price `X`, gross return is `X / close[t-1] − 1`. Within a trade, marked equity is `close / (E × (1 + slip)) − fee`; exit equity is `(X / E) × (1 − fee) × (1 − slip) / (1 + slip) − fee`. Per-bar net return is current equity divided by previous equity minus one; previous equity is one on entry. Window-end liquidation follows the same accounting.

### Price and provenance requirements

Enabled snapshots require finite, positive, equal-length one-dimensional `open`, `high`, `low`, and `close`, consistent OHLC bounds and `price_change`, and binary predictions across the entire window, including flat and lag-excluded rows. `high_col` and `low_col` can map other names. With both barriers disabled, high/low are unnecessary and legacy non-tradable gaps remain supported.

Manifests with a barrier literal or reference record source identities after pre-split selection and bar formation, before features/targets remove rows. Manifests with neither barrier skip this capture. Validated retained spans use compact ranges, so the witness adds constant storage per round. Any configured barrier, including a reference currently resolving to null, requires this evidence before model training. Interior deletions within the retained evaluation span raise; leading warmup/CCO rows and trailing target rows are outside that span. ML test-source timestamp duplicates raise before the unique-first price join, even when a transform removes one duplicate member. Rule-based splits accept positional duplicates when order and identity remain clear. Private identities never enter features, targets, scalers, PCA, predicates, or public outputs.

Direct snapshots are positional and accept time-free arrays. The caller owns completeness, order, and alignment. Manifest-backed replay verifies the full source fingerprint and retained identities before reconstructing OHLC; missing evidence, reordered sources, or censored rows fail explicitly. The evidence is retained in memory, so configured file-only/resumed replay is unsupported. Trainer reconstructs fresh evidence when rebuilding a round.

### Supported evaluation paths

Shipped `ReferenceModel` architectures and built-in rule-based entry strategies apply manifest exits. Custom architectures must integrate the backtest configuration themselves; constructors with `**kwargs` may still receive TP/SL search keys. This stage adds fixed barriers; rule-based SFD exit logic is deferred. Sensor continues to emit entry predictions and does not execute these research exits.

Each UEL candidate repeats preparation, training, and prediction. Predictions match across an exit-only sweep only when entry parameters, inputs, preparation, and model behavior are deterministic. A frozen-prediction grid is deferred.

## Output ledger

Snapshot backtests produce 20 columns over one population: every bar in the window.

Table 4. Distribution columns report `p5`, `p50`, and `p95`.

| prefix | columns | meaning |
|---|---|---|
| `edge_bps` | `edge_bps_p5`, `edge_bps_p50`, `edge_bps_p95` | Gross per-bar return. |
| `pnl_bps` | `pnl_bps_p5`, `pnl_bps_p50`, `pnl_bps_p95` | Net per-bar return. |
| `cost_bps` | `cost_bps_p5`, `cost_bps_p50`, `cost_bps_p95` | Per-bar gross return minus net return. |
| `drawdown_bps` | `drawdown_bps_p5`, `drawdown_bps_p50`, `drawdown_bps_p95` | Net equity against its running peak. Values are less than or equal to `0`. |

Table 5. Scalar columns are intensive metrics.

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

### Rule-based mean PnL per trade

`RuleBasedStrategy` adds `pnl_per_trade_bps_{split}` and its aligned `num_executed_trades_{split}` denominator outside the generic snapshot contract; `backtest_snapshot()` itself remains exactly 20 columns. An executed trade is one contiguous segment where the lagged strategy position is above zero. Its return compounds the segment's net per-bar returns after fee, slippage, and `notional_rate`; the metric is the arithmetic mean across executed trades, in basis points. It is `NaN` and the denominator is zero when no trade executes. The older `num_trades_{split}` remains a pre-execution signal-entry count.

For the bundled dollar-bar crash-reversal sweep, `fee_bps=10.0` and `slip_bps=5.0` mean 15 bps on each entry or exit fill. The mean therefore measures the surviving net edge per completed position path, not a gross signal return.

## Strategy boundary

With both barriers disabled, the execution model is swappable. Enabled barriers require the default long-flat strategy and reject a custom callback. `backtest_snapshot()` validates price columns, calls a strategy, and builds the ledger from the returned per-bar arrays. The shipped strategy is `long_flat_strategy` in `limen.backtest.long_flat_strategy`.

A strategy receives `predictions`, `open_px`, `close_px`, `price_change`, `execution_lag_bars`, `fee_bps`, and `slip_bps`. It returns `ExecutionResult(pos, gross, net)`, where each field is a finite numeric per-bar array over the full window, aligned positionally to the input length. `backtest_snapshot()` rejects malformed custom strategy outputs before computing the ledger.

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

Use the module function to inspect one permutation.

```python
from limen.backtest.backtest_snapshot import backtest_snapshot

perf = uel._log.permutation_prediction_performance(round_id=0)
round0_backtest = backtest_snapshot(perf)
```

## Benchmark boundary

Benchmark and backtest answer different questions.

Table 6. The layers are separate because statistical structure and trading economics can fail independently.

| layer | question | input frame |
|---|---|---|
| Benchmark | Does the signal contain predictive structure? | Predictions and realized labels. |
| Backtest | Does that structure survive the declared trading interpretation? | Binary signal, price columns, costs, lag, and notional. |

Limen keeps the layers separate in the API and the docs because benchmark quality is not a substitute for economic inspection.

## Non-goals

Snapshot backtest is not an execution simulator.

Table 7. These concerns sit outside the snapshot contract.

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
