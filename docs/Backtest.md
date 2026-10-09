# Backtest

Backtest evaluates declared trading economics. The legacy snapshot reports binary long-flat per-bar returns. Configured event execution supports signed exposure, elapsed exits and funding with separate intent, fill and episode ledgers.

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

## Signed exposure, elapsed exits and funding

Event execution is selected by explicit product metadata, signed output, elapsed exits, timers, an execution source, funding, or changed event-only sizing/timing options. Unconfigured binary snapshots retain their existing behavior. A model declares `prediction_mode='target_exposure'` explicitly; a custom native architecture returns `_prediction_mode` with `_preds`. Probabilities are not implicitly position sizes.

```python
from limen.experiment.manifest_core import FundingConfig, ProductConfig

manifest.set_backtest_config(
    prediction_mode='target_exposure',
    product=ProductConfig('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT',
                          quantity_step=0.00001, min_notional=5.0),
    initial_equity=10000.0,
    notional_rate='allocation',
    max_holding_seconds='holding_seconds',
    take_profit_bps='tp',
    stop_loss_bps='sl',
    funding=FundingConfig(preset='binance_btcusdt',
                          params={'rate': '{funding_rate}'}),
)
```

The product names and quantity rules are declarations supplied by the caller. Cash spot uses `kind='cash_spot'` and supports long/flat; linear quote-settled perpetuals support long/flat/short. Perpetual funding does not model spot borrowing. All accounting uses the declared quote currency, with no currency conversion, inverse contracts, liquidation or order-book model.

The YAML equivalents are nested mappings under `sfd.manifest.backtest`:

```yaml
backtest:
  prediction_mode: target_exposure
  product:
    kind: linear_perpetual
    instrument: BTCUSDT
    base_currency: BTC
    quote_currency: USDT
    quantity_step: 0.00001
    min_notional: 5.0
  initial_equity: 10000.0
  max_holding_seconds: "{holding_seconds}"
  funding:
    preset: binance_btcusdt
    params:
      rate: "{funding_rate}"
```

Each referenced parameter must exist in `sfd.params`. Native numeric fields also accept bare parameter names. Nested adapter/source parameters resolve explicit `{parameter}` references. Unknown keys, nonfinite values, incompatible products and missing required data fail.

### Signal-change sizing

A finite signal `z` lies in `[-1,1]`: positive is long, zero is flat, negative is short. `abs(z) <= flat_threshold` is canonical flat. Requested exposure is `notional_rate * z`, bounded by `max_exposure <= 1`. Only a changed canonical signal creates a sizing intent. `signal_change_bps` defaults to zero; explicit flat and reversal always count. An unchanged signal holds quantity through price, equity and funding changes.

A changed signal sizes against current marked equity after recognized funding, reserving the fee and adverse slippage on its actual quantity delta. Quantity rounds toward zero to `quantity_step`; subminimum openings/resizes record no fill. Full closure permits residual dust. Cash spot cannot borrow. Perpetual fills must meet `initial_margin_fraction`; nonpositive equity or a breached `maintenance_margin_fraction` fails explicitly.

Resizes retain episode ID, first actual fill time and original barrier anchor. Accounting entry basis may change. Full closure ends the episode; reversal closes and costs the old position before sizing the opposite entry. A forced barrier/time exit suppresses re-entry until an explicit flat signal, including through opposite signals.

### Recorded prices and clocks

`max_holding_seconds` measures elapsed UTC time from first fill. Expiry is requested at the first declared signal or timer event reaching that deadline. `timer_interval_seconds` adds checks on the UTC epoch grid with `timer_phase_utc_seconds`; timers make no predictions. Requests remain pending until a recorded admissible price arrives. End marking leaves open episodes open and charges no invented close fee.

Execution observations carry `row_id, start_ns, end_ns, open_available_at_ns, available_at_ns, open, high, low, close`; times are integer UTC nanoseconds. Points have equal start/end and one recorded price. OHLC opens are usable at known starts, close/extrema only at availability. Held barriers process SL before TP, with adverse gaps and interval uncertainty. Whole-interval extrema cannot be applied to inventory entered/resized inside that interval.

Regular raw OHLC may supply a verified source interval (`kline_size` in seconds, including `HistoricalData.get_spot_klines`, legacy `klines_size`, or constant `base_interval`). A non-null `kline_size` takes precedence; absent or null `kline_size` retains `klines_size`. Irregular sources must supply actual interval/availability metadata. Limen does not derive irregular duration from bar count or interpolate prices. Optional `execution_data_source=DataSourceConfig(method, params)` supplies finer recorded observations once per round; HistoricalData execution methods must be bound Python methods (`HistoricalData().get_spot_klines`), because YAML method binding remains separate work in #859. Its coverage must span the true split; `max_price_gap_seconds` bounds both interval width and gaps. Second-resolution execution therefore requires recorded data meeting that bound.

Equal-time order is funding on preboundary inventory, held price barriers, elapsed expiry, newly available signals, pending exits, then surviving changed sizing. `execution_lag_seconds` delays signal intents; an earlier open cannot fill a later signal.

### Funding modes

`FundingConfig` chooses an importable versioned adapter, optional preset, tunable JSON `params` and optional `data_source`. Native adapter parameters include `mechanism`, `rate`, `rate_unit` (`decimal` or `bps`), `rate_basis_seconds`, `currency`, `valuation`, `approximation`, settlement interval/UTC phase, and continuous cash-settlement interval/UTC phase. Defaults expand before explicit overrides. Rates already represent payments or quoted accrual; no premium formula or second period conversion is applied.

| Preset | Default scenario rate | Mechanics |
|---|---|---|
| `binance_btcusdt` | `0.000028` per eight hours | Discrete eight-hour payments; mark valuation in history mode. |
| `hyperliquid_btc` | `0.0000035` per hour | Discrete hourly payments; oracle valuation in history mode. |
| `deribit_btc_usdc` | `0.000028` per eight hours | Continuous eight-hour quoted basis; daily 08:00 UTC cash transfer. |

No-history presets use a shared BTC scenario assumption with an explicit execution-price proxy. The baseline is dated `2026-10-08`, with the reference window `2025-10-08`–`2026-10-08`; it is not a measured mean for each preset's venue. Rates, periods and settlement phases remain tunable. The resolved preset version and assumption metadata survive export. These constants do not reproduce historical funding variation; use a supplied historical source for that.

Historical funding is optional and authoritative when supplied. An explicit constant rate conflicts with it; preset constants are removed. Discrete history requires unique `event_id`, `settlement_at`, `rate_decimal`, `valuation_price`, plus recorded `schedule_start`, `schedule_end`, `settlement_interval_seconds` and `settlement_phase_utc_seconds`. Schedule intervals are half-open UTC nanoseconds and must cover every required payment, including historical schedule changes. Optional `settlement_slot_ns` preserves actual payment timestamp jitter while identifying its declared schedule slot.

Continuous history requires unique `event_id`, `start`, `end`, `rate_decimal`, `valuation_price`. Declare `history_interpretation='quoted'` with per-row `rate_basis_seconds`, or `'integrated'` for an interval's realized payment. Support intervals must cover the window without gaps/overlap. Integrated history cannot establish exact intrainterval resizing; provide finer evidence or select sampled integration. Datetime timestamps convert to UTC nanoseconds.

Positive funding debits long inventory and credits shorts. Discrete cashflow is `-quantity * valuation_price * rate_decimal`. Continuous accrual integrates that amount over elapsed seconds divided by its quoted basis, partitioned at inventory/rate/valuation events. Recognition changes equity before sizing; cash transfer does not charge it twice. Funding continues until actual closure. Exact history requires recorded valuation; scenario/sampled mode may explicitly use a causal execution-price proxy.

Future settled funding remains private cost evidence. A funding feature requires a separate causal source.

### Ledger and replay

`limen.backtest.trade_execution` accepts resolved `TradePolicy` and bound `TradeInputs`. Its `TradeLedger` separates states, intents, fills, episodes and funding. Returns and drawdown use actual marked equity; costs, funding credits/debits, gross/net PnL, completed/open episodes and mean absolute exposure are separate. Exposure samples use model availability, so shorts cannot cancel longs in an average.

Each round freezes canonical `trade_contract`, SHA-256 digest, source identities/checksums, policy/adapter/preset versions, calibration and event/episode ledger in JSONL. In-memory Log replays unchanged signed predictions and verifies reconstructed bindings before accounting. Configured file-only/resumed Log replay fails explicitly: a contract cannot supply missing prices. Deployment parity remains subject to the linked Praxis/Nexus requests.
