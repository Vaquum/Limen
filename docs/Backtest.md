# Account for the trades a prediction requests

Backtest evaluates declared trading economics. The legacy snapshot reports binary long-flat per-bar returns. Configured event execution supports signed exposure, elapsed exits and funding with separate intent, fill and episode ledgers.

The layer evaluates entry signals together with declared costs, sizing, and optional fixed take-profit/stop-loss exits.

For reproducible comparisons with other engines using identical recorded prices
and strategy signals, see [Backtest engine benchmarks](Backtest-Benchmarks.md).

The [Python project](examples/python-research/README.md) binds recorded bars,
signal availability, capital and a trading policy directly. The
[Compiled project](examples/compiled-research/README.md) declares those choices
in its Manifest and retains their native ledgers. A prediction becomes a trading
result only under that declared policy.

## Follow the two execution cases

The Python function and Compiled YAML below execute ordinary cash-spot behavior
and a separate linear-perpetual case using the same recorded funding history.
The tests independently reconcile fill clocks, quantities, fees, funding and
final equity. No funding rate or fill price is invented for the comparison.
The Python function uses its complete project's imports and helpers. The Compiled YAML names its complete source and execution policy.

### Python Research

<!-- example: backtest-python-execution source=docs/examples/python-research/scenarios.py region=execution scenario=execution interface=python role=executed -->
```python
def execution(data_dir, output):
    bars = recorded(data_dir, 'spot_1h_20240101_20241231.parquet', 168)
    interval = 3600 * 1_000_000_000
    observations = bars.with_row_index('row_id').with_columns(pl.col('row_id').cast(pl.String), pl.col('datetime').dt.epoch('ns').alias('start_ns')).with_columns((pl.col('start_ns') + interval).alias('end_ns'), pl.col('start_ns').alias('open_available_at_ns'), (pl.col('start_ns') + interval).alias('available_at_ns')).select('row_id', 'start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns', 'open', 'high', 'low', 'close')
    start, end = observations['start_ns'][0], observations['end_ns'][-1]
    signals = observations.select('row_id', 'available_at_ns').with_columns(pl.lit(None, dtype=pl.Float64).alias('target'))
    source = source_binding(observations, INPUTS['spot_1h_20240101_20241231.parquet'], start, end, interval, 'recorded hourly OHLC; high/low order unknown')
    inputs = TradeInputs(10000, start, end, signals, observations, None, (source,))
    inputs = with_predictions(inputs, (bars['close'] > bars['open']).cast(pl.Float64).to_numpy())
    policy = TradePolicy(ProductSpec('cash_spot', 'BTCUSDT', 'BTC', 'USDT', .00001, 5), prediction_mode='binary', fee_bps=7, slip_bps=3, execution_lag_seconds=3600, max_holding_seconds=7200)
    ledger = trade_execution(inputs, policy)
    for name in ('states', 'intents', 'fills', 'episodes', 'funding'):
        getattr(ledger, name).write_parquet(output / (name + '.parquet'))
    history = funding_history(data_dir)
    funding = FundingPolicy('limen.backtest.funding_adapter.funding_adapter', 'limen-funding-v1', {}, None, None, 'discrete', 28800.0, 'mark', 'USDT', 'recorded_exact')
    events = prepare_funding(funding, history, start, end)
    binding = source_binding(events, HASHES[FUNDING], start, end, 1_000_000, 'recorded realized rates and mark prices; explicit eight-hour slots')
    funded_inputs = replace(inputs, funding_events=events, sources=(*inputs.sources, binding))
    funded_policy = replace(policy, product=ProductSpec('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0), funding=funding)
    funded = trade_execution(funded_inputs, funded_policy)
    for name in ('states', 'intents', 'fills', 'episodes', 'funding'):
        getattr(funded, name).write_parquet(output / ('funded-' + name + '.parquet'))
    context = {'observations': observations.to_dicts(), 'partition': [start, end], 'contract': export_trade_contract(policy, inputs)}
    funded_context = context | {'contract': export_trade_contract(funded_policy, funded_inputs)}
    return {'metrics': dict(ledger.metrics), 'contract_digest': ledger.contract_digest, 'fills': ledger.fills.to_dicts(), 'ledger_excerpt': ledger.fills.head(8).to_dicts(), 'episodes': ledger.episodes.to_dicts(), 'context': context, 'initial_equity': 10000, 'fee_bps': 7, 'slip_bps': 3, 'source_interval_seconds': 3600, 'funded': {'metrics': dict(funded.metrics), 'contract_digest': funded.contract_digest, 'payments': funded.funding.to_dicts(), 'fills': funded.fills.to_dicts(), 'context': funded_context, 'recorded_events': history.height, 'input_sha256': HASHES[FUNDING]}}
```

### Compiled Research

<!-- example: backtest-compiled-execution source=docs/examples/compiled-research/execution.yaml region=declaration scenario=execution interface=compiled role=executed -->
```yaml
schema_version: '1.0'
metadata:
  name: book_compiled_execution
  mode: development
sfd:
  manifest:
    type: rule_based
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_1h_20240101_20241231.parquet
    split_dates:
      train_start: '2024-01-02'
      train_end: '2024-01-03'
      val_start: '2024-01-04'
      val_end: '2024-01-05'
      test_start: '2024-01-05'
      test_end: '2024-01-06'
    backtest:
      product:
        kind: cash_spot
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 1e-05
        min_notional: 5
      source_interval_seconds: 3600
      fee_bps: 10
      slip_bps: 5
      take_profit_bps_column: volatility_take_profit_bps
      stop_loss_bps_column: volatility_stop_loss_bps
      barrier_features:
        method: limen.features.volatility_barriers.volatility_barriers
        params:
          span: '{risk_span}'
          regime_span: 24
          min_periods: 3
    reference_architecture: 
      limen.sfd.reference_architecture.rule_based.rule_based
    strategy:
      conditions:
      - id: entry
        name: Recorded close above open
        type: relative
        column: close
        operator: '>'
        other_column: open
      entry: entry
  params:
    risk_span:
    - 6
    - 12
uel:
  output_path: compiled-execution
  n_permutations: 2
  search_strategy:
    type: grid
  output_format: csv
  checkpoint_interval: 1
  prep_each_round: true
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
```

## Prerequisites

- aligned binary predictions and OHLC/price-change columns
- explicit fee, slippage, notional, and execution-lag assumptions
- captured metric columns for aggregate results; retained row evidence only when constructing explicit snapshot inputs

## Risk boundary

Backtest output is research evidence, not investment advice, trading advice, execution simulation, regulatory approval, or a promise of future performance. Past performance is not predictive, digital-asset trading can result in total loss of capital, and snapshot backtests do not model venue queues, latency, borrow, liquidation, funding, portfolio constraints, or live order execution.

## Entry points

Table 1. Backtest is exposed through three paths.

| path | use |
|---|---|
| `uel.experiment_backtest_results` | Experiment-wide table with one row per round. |
| `uel._log.experiment_backtest_results()` | Read captured results; see the [Log contract](Log.md#backtest-surface) for CSV and fold behavior. |
| `limen.backtest.backtest_snapshot.backtest_snapshot` | Module function for one per-round prediction table. |

## Snapshot contract

`backtest_snapshot()` returns one summary row as a metrics dict keyed by the ledger columns. Shipped reference architectures compute these metrics during the round. Aggregate `Log` reads preserve those captured results without replay; [Log](Log.md#backtest-surface) defines that contract. Explicit prediction-performance tables keep their existing columns.

<!-- reference: backtest-log-performance source=limen/log/log.py region=signature:Log.permutation_prediction_performance role=signature -->
```python
@_holdout.replay_guard
def permutation_prediction_performance(self, round_id: int) -> pl.DataFrame:
    ...
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

### Internal legacy snapshot normalization

`_prepare_snapshot_backtest_input` in `limen.log._experiment_backtest_results` is an internal helper for explicit legacy snapshot replay; its repository callers are tests. The default `prediction_kind='binary'` accepts `0`/`1` predictions. Other supported numeric values require the explicit keyword `prediction_kind='directional'`: positive scores become `1`, zero and negative scores become `0`, even when every score lies within `(0, 1)`. Threshold probabilities upstream; the helper neither infers their meaning from their range nor chooses a threshold.

Existing nonnumeric and multiclass exclusions apply in both modes: three or more distinct nonnegative exact-integer predictions or actuals remain unsupported. Binary detection ignores existing nulls and passes through the copied values; on the nonbinary directional path, a null comparison becomes flat. Downstream direct-snapshot validation remains authoritative and rejects missing predictions.

Signed `trade_contract` replay bypasses this helper and preserves exposure. Aggregate `Log` readers consume captured metrics without invoking it.

## Economic inputs

Fees and slippage default to `5.0` bps each per fill. A one-entry, one-exit path pays `fee_bps` of the entry notional from cash at entry, `fee_bps` of the exit proceeds at exit, and slippage on both fill prices, so a trade with gross return `g` ends with equity `(1 + g)(1 − fee)(1 − slip) / (1 + slip) − fee` per unit of entry notional, a net return of `g − fee × (2 + g)` at zero slippage, the deployed arithmetic. Because equity within a trade is the position minus the entry fee, `cost_bps` on a held bar that is neither the entry nor the exit is the small gap between the position's return and the equity return: slightly negative on an up bar, slightly positive on a down bar.

The Manifest owns native economic inputs. Its current setter is:

<!-- reference: backtest-manifest-policy source=limen/experiment/manifest_core.py region=signature:Manifest.set_backtest_config role=signature -->
```python
def set_backtest_config(self, fee_bps: float | str=5.0, slip_bps: float | str=5.0, notional_rate: float | str=1.0, *, take_profit_bps: float | str | None=None, stop_loss_bps: float | str | None=None, prediction_mode: Literal['binary', 'target_exposure']='binary', product: ProductConfig | None=None, initial_equity: float | str=10000.0, max_exposure: float | str=1.0, signal_change_bps: float | str=0.0, flat_threshold: float | str=0.0, max_holding_seconds: float | str | None=None, timer_interval_seconds: float | str | None=None, timer_phase_utc_seconds: float | str=0.0, execution_lag_seconds: float | str=0.0, max_price_gap_seconds: float | str | None=None, execution_data_source: DataSourceConfig | None=None, funding: FundingConfig | None=None, take_profit_bps_column: str | None=None, stop_loss_bps_column: str | None=None, barrier_features: DataSourceConfig | None=None, source_interval_seconds: float | None=None, source_timestamp_role: Literal['start', 'end']='start') -> 'Manifest':
    ...
```

Each economic value is either a fixed number or a search-parameter name. Pass a parameter name when cost or position size belongs in the search space.

Use a plain parameter name such as `fee` or `size` in the native builder. The candidate values belong to the SFD's parameter domain.

A YAML/CLI manifest carries the same configuration under `sfd.manifest.backtest`, sibling to `target` and `scaler`.

| YAML field | Fixed value or search reference |
|---|---|
| `sfd.manifest.backtest.fee_bps` | A finite nonnegative number, or `"{fee}"` with numeric candidates in `sfd.params.fee`. |
| `sfd.manifest.backtest.slip_bps` | A finite nonnegative number, or an admitted parameter reference. |
| `sfd.manifest.backtest.notional_rate` | A finite value in `(0, 1]`, or `"{size}"` with compatible candidates in `sfd.params.size`. |

The block is optional. When omitted or empty, the defaults remain `fee_bps=5.0`, `slip_bps=5.0`, and `notional_rate=1.0`. `limen validate` rejects unknown keys, negative costs, non-finite costs, `notional_rate` outside `(0, 1]`, and `"{param}"` references missing from `sfd.params`.

## Tunable take-profit and stop-loss

`take_profit_bps` and `stop_loss_bps` are separate manifest backtest fields. Each accepts a fixed number, a Python search-parameter name (plain or single-braced), or `None`. `None` disables that barrier independently. TP must be finite and greater than zero; SL must be finite and strictly between zero and 10,000 bps. Zero and booleans are invalid. Missing references raise a field-naming `ValueError`.

`set_backtest_config()` replaces the entire configuration. Preserve declared costs and sizing when adding barriers:

Read the existing `manifest.backtest_config` first and pass its `fee_bps`, `slip_bps` and `notional_rate` alongside the new barrier arguments in the source-bound setter above.

In the SFD's `params()`, declare the barrier candidate lists alongside the entry-model parameters. YAML uses braced references:

| Parameter | Permitted candidates |
|---|---|
| `take_profit_bps` | `null` to disable, or finite positive distances. |
| `stop_loss_bps` | `null` to disable, or finite distances strictly between `0` and `10000`. |

Declare the candidates in `sfd.params` and reference their names with braces from `sfd.manifest.backtest`.

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

By default each UEL candidate repeats preparation, training, and prediction. Eligible deterministic snapshot grids can opt into `uel.factorize: true` to reuse signal work across independent economics-only axes. The existing snapshot executor still evaluates every economic combination separately; unsupported/ambiguous cases reject, and default runs remain unchanged.

## Output ledger

Snapshot backtests produce 20 columns over one population: every bar in the window.

### Numerical precision

All package measurements retain their computed floating-point precision through calculation, recording, serialization, filtering and ranking. Round only the final display text. This includes ledger values, classification and regression metrics, confusion rates and returns, rule statistics, historical market aggregates, weighted features, cohort percentages and execution timings. A small nonzero measurement must remain distinguishable from zero; nearby scores must remain distinguishable for selection. Full precision means retaining the computed value, not recovering precision absent from source data or a model backend.

| Surface | Precision rule |
|---|---|
| Numeric API values, CSV and JSON artifacts | Preserve computed values; no decimal rounding or formatting before storage or ranking. `Log` uses round-trip CSV float parsing. |
| Feature weights | Calculate with Float64; do not force configurable weights into Float32. |
| Fractional differentiation search | Evaluate the declared decimal start/step grid, including its endpoint, without a fixed decimal cutoff. |
| Sampling filter | Compare the declared fraction against the full 64-bit deterministic hash range; no 1/10,000 buckets. Membership can differ from earlier releases. New checkpoint descriptors record `hash_rule: blake2b-64-v1`; restoring an unversioned or unsupported sampling filter fails. Resume those experiments with their original package version, or start a new experiment. |
| Cohort correlation index | Retain `100 * n_rows / total_rows`; adjacent small cohorts keep distinct percentages. |
| CLI profile text | Format seconds to three decimals for measured per-permutation time; duration estimates use one decimal and counts use two decimals with their unit suffix. |
| Diagnostic text | Format reported thresholds and reasons for readability; decisions use the original values. |

Remaining numerical rounding expresses discrete contracts: execution, funding and outcome clocks convert seconds to integer nanoseconds with Python's nearest/ties-to-even `round`; focus-reducer integer domains use nearest integer candidates and outward floor/ceil bounds; execution quantities round toward zero to the declared lot step; CVaR selects an integer tail count; bar formation assigns integer time/notional buckets. Classification labels and row/window counts are discrete by definition. These operations do not justify rounding continuous measurements. `tests/test_rounding_precision.py` checks recorded-data calculations and the remaining explicit rounding inventory.

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

With both barriers disabled, the execution model is swappable. Enabled barriers require the default long-flat strategy and reject a custom callback. `backtest_snapshot()` validates price columns, calls a strategy, and builds the ledger from the returned per-bar arrays. The default strategy is `long_flat_strategy` in `limen.backtest.long_flat_strategy`.

A strategy receives `predictions`, `open_px`, `close_px`, `price_change`, `execution_lag_bars`, `fee_bps`, and `slip_bps`. It returns `ExecutionResult(pos, gross, net)`, where each field is a finite numeric per-bar array over the full window, aligned positionally to the input length. `backtest_snapshot()` rejects malformed custom strategy outputs before computing the ledger.

<!-- reference: backtest-snapshot source=limen/backtest/backtest_snapshot.py region=signature:backtest_snapshot role=signature -->
```python
def backtest_snapshot(columns: Mapping[str, Any], *, pred_col: str='predictions', open_col: str='open', close_col: str='close', price_change_col: str='price_change', strategy: Callable[..., ExecutionResult]=long_flat_strategy, execution_lag_bars: int=1, fee_bps: float=5.0, slip_bps: float=5.0, notional_rate: float=1.0, take_profit_bps: float | None=None, stop_loss_bps: float | None=None, high_col: str='high', low_col: str='low', exit_col: str | None=None) -> dict[str, float]:
    ...
```

The strategy owns its signal contract and fill mechanics. `backtest_snapshot()` applies `notional_rate` after the strategy returns, so position sizing remains a ledger-level scale rather than a strategy argument.

### Independent binary exits

Pass `exit_col` naming an aligned, nonmissing boolean column to opt into entry/hold/exit snapshots. Entry predictions remain binary. A false entry after opening leaves inventory held; a true exit closes it. After a closure or pending-entry cancellation, a strictly later false entry rearms, and a subsequent true entry can open again. An exit while flat vetoes that decision without consuming initial arming. Observations during a pending closure cannot queue re-entry or earn rearm credit.

Explicit exits require the default long-flat strategy, complete finite positive prices and a positive integer lag. Exit decision row `e` with lag `L` closes at `close[e + L - 1]`, on the final held return row; barriers in that interval take priority at the existing snapshot fill price. Missing or ambiguous evidence raises. Omitted `exit_col` preserves the existing snapshot contract and its 20 ledger fields.

### Explicit short/flat snapshots

Import `short_flat_strategy` from `limen.backtest.short_flat_strategy`, then call `backtest_snapshot(perf, strategy=short_flat_strategy)` with retained prediction and price rows.

<!-- reference: backtest-short-flat source=limen/backtest/short_flat_strategy.py region=signature:short_flat_strategy role=signature -->
```python
def short_flat_strategy(predictions: npt.ArrayLike, open_px: npt.ArrayLike, close_px: npt.ArrayLike, price_change: npt.ArrayLike, *, execution_lag_bars: int=1, fee_bps: float=5.0, slip_bps: float=5.0) -> ExecutionResult:
    ...
```

Here binary `1` activates a short and `0` is flat. `pos` remains a non-negative exposure magnitude, preserving the 20-column ledger and its trade/inventory counts. After the declared lag, entries use the execution-row open: gross return is `-(close - open) / open`. Continuations use `-(close_t / close_previous - 1)`. Missing price rows flatten the position; a later active valid row starts a new trade at its open. Zero lag can trade the first row and is a same-row research diagnostic.

Let `E = (1 - fee) * (1 - slip)` and `X = (1 - fee) / (1 + slip)`, where fee/slip are basis points divided by 10000. Entry debits `1 - E` from the gross return. Exit debits `(1 - gross) * (1 - X)`, multiplied by `E` when entry and exit share a bar. These bases are positive underlying fill notionals, so fees worsen losses even when the short loses more than its starting capital. A one-bar trade pays both; at flat prices, 50 bps each of fee and slippage produces 198.3 bps of round-trip drag. Continuation bars pay no fill costs.

This is a constant-notional `-1×` return approximation. It does not model fixed-quantity short-sale accounting, borrowing, margin, liquidation or funding. It is explicit-call only; Manifest/Log execution and default long fills keep their contracts. Snapshot TP/SL cannot be combined with this strategy.

## Usage

Use the experiment-wide table to compare rounds.

`uel.experiment_backtest_results` exposes the captured aggregate table after applicable post-processing.

With retained live row evidence, use the module function to evaluate explicit snapshot inputs. This is separate from reading captured aggregate results.

<!-- reference: backtest-long-flat source=limen/backtest/long_flat_strategy.py region=signature:long_flat_strategy role=signature -->
```python
def long_flat_strategy(predictions: Any, open_px: Any, close_px: Any, price_change: Any, *, execution_lag_bars: int=1, fee_bps: float=5.0, slip_bps: float=5.0) -> ExecutionResult:
    ...
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
| Short selling | Explicit constant-notional short/flat snapshots only; borrowing and margin mechanics are out of scope. |
| Latency-aware order modeling | Out of scope. |

## Signed exposure, elapsed exits and funding

Event execution is selected by explicit product metadata, signed output, elapsed exits, timers, an execution source, funding, or changed event-only sizing/timing options. Unconfigured binary snapshots retain their existing behavior. A model declares `prediction_mode='target_exposure'` explicitly; a custom native architecture returns `_prediction_mode` with `_preds`. Probabilities are not implicitly position sizes.

<!-- reference: backtest-productspec source=limen/backtest/trade_contract.py region=signature:ProductSpec role=signature -->
```python
@dataclass(frozen=True)
class ProductSpec:
    kind: Literal['cash_spot', 'linear_perpetual']
    instrument: str
    base_currency: str
    quote_currency: str
    quantity_step: float
    min_notional: float
    initial_margin_fraction: float = 1.0
    maintenance_margin_fraction: float = 0.0
```

<!-- reference: backtest-tradeinputs source=limen/backtest/trade_contract.py region=signature:TradeInputs role=signature -->
```python
@dataclass(frozen=True)
class TradeInputs:
    initial_equity: float
    partition_start_ns: int
    partition_end_ns: int
    signals: pl.DataFrame
    observations: pl.DataFrame
    funding_events: pl.DataFrame | None
    sources: tuple[SourceBinding, ...]
```

<!-- reference: backtest-tradepolicy source=limen/backtest/trade_contract.py region=signature:TradePolicy role=signature -->
```python
@dataclass(frozen=True)
class TradePolicy:
    product: ProductSpec
    prediction_mode: PredictionMode = 'target_exposure'
    rule_version: str = RULE_VERSION
    fee_bps: float = 5.0
    slip_bps: float = 5.0
    notional_rate: float = 1.0
    max_exposure: float = 1.0
    signal_change_bps: float = 0.0
    flat_threshold: float = 0.0
    take_profit_bps: float | None = None
    stop_loss_bps: float | None = None
    max_holding_seconds: float | None = None
    timer_interval_seconds: float | None = None
    timer_phase_utc_seconds: float = 0.0
    execution_lag_seconds: float = 0.0
    max_price_gap_seconds: float | None = None
    funding: FundingPolicy | None = None
    take_profit_bps_column: str | None = None
    stop_loss_bps_column: str | None = None
    barrier_spec: Mapping[str, JsonValue] | None = None
    source_interval_seconds: float | None = None
    source_timestamp_role: Literal['start', 'end'] | None = None
```

<!-- reference: backtest-fundingpolicy source=limen/backtest/trade_contract.py region=signature:FundingPolicy role=signature -->
```python
@dataclass(frozen=True)
class FundingPolicy:
    adapter_ref: str
    adapter_version: str
    params: Mapping[str, JsonValue]
    preset_id: str | None
    preset_version: str | None
    mechanism: Literal['discrete', 'continuous']
    rate_basis_seconds: float
    valuation: Literal['mark', 'oracle', 'execution_proxy']
    currency: str
    approximation: Literal['scenario', 'sampled', 'recorded_exact']
    calibration: Mapping[str, JsonValue] | None = None
```

The product names and quantity rules are declarations supplied by the caller. Cash spot uses `kind='cash_spot'` and supports long/flat; linear quote-settled perpetuals support long/flat/short. Perpetual funding does not model spot borrowing. All accounting uses the declared quote currency, with no currency conversion, inverse contracts, liquidation or order-book model.

The Compiled YAML reads the bundled `inputs/recorded-funding.parquet` directly. The bundle pins that normalization of the recorded funding events; no research driver prepares or rewrites it:

<!-- example: backtest-compiled-funding source=docs/examples/compiled-research/funded.yaml region=__file__ scenario=execution interface=compiled role=executed -->
```yaml
# region declaration
schema_version: '1.0'
metadata:
  name: book_compiled_recorded_funding
  mode: development
sfd:
  manifest:
    type: rule_based
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_1h_20240101_20241231.parquet
    split_dates:
      train_start: '2024-01-02'
      train_end: '2024-01-03'
      val_start: '2024-01-04'
      val_end: '2024-01-05'
      test_start: '2024-01-05'
      test_end: '2024-01-06'
    backtest:
      product:
        kind: linear_perpetual
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 1e-09
        min_notional: 0
      source_interval_seconds: 3600
      fee_bps: 10
      slip_bps: 5
      take_profit_bps_column: volatility_take_profit_bps
      stop_loss_bps_column: volatility_stop_loss_bps
      barrier_features:
        method: limen.features.volatility_barriers.volatility_barriers
        params:
          span: '{risk_span}'
          regime_span: 24
          min_periods: 3
      funding:
        preset: binance_btcusdt
        data_source:
          method: limen.data.HistoricalData.get_any_file
          params:
            file_path_or_url: inputs/recorded-funding.parquet
    reference_architecture: 
      limen.sfd.reference_architecture.rule_based.rule_based
    strategy:
      conditions:
      - id: entry
        name: Recorded close above open
        type: relative
        column: close
        operator: '>'
        other_column: open
      entry: entry
  params:
    risk_span:
    - 6
    - 12
uel:
  output_path: compiled-funded
  n_permutations: 2
  search_strategy:
    type: grid
  output_format: csv
  checkpoint_interval: 1
  prep_each_round: true
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
# endregion declaration
```

Each referenced parameter must exist in `sfd.params`. Native numeric fields also accept bare parameter names. Nested adapter/source parameters resolve explicit `{parameter}` references. Unknown keys, nonfinite values, incompatible products and missing required data fail.

### Signal-change sizing

A finite signal `z` lies in `[-1,1]`: positive is long, zero is flat, negative is short. `abs(z) <= flat_threshold` is canonical flat. Requested exposure is `notional_rate * z`, bounded by `max_exposure <= 1`. Only a changed canonical signal creates a sizing intent. `signal_change_bps` defaults to zero; explicit flat and reversal always count. An unchanged signal holds quantity through price, equity and funding changes.

A changed signal sizes against current marked equity after recognized funding, reserving the fee and adverse slippage on its actual quantity delta. Quantity rounds toward zero to `quantity_step`; subminimum openings/resizes record no fill. Full closure permits residual dust. Cash spot cannot borrow. Perpetual fills must meet `initial_margin_fraction`; nonpositive equity or a breached `maintenance_margin_fraction` fails explicitly.

Resizes retain episode ID, first actual fill time and original barrier anchor. Accounting entry basis may change. Full closure ends the episode; reversal closes and costs the old position before sizing the opposite entry. A forced barrier/time exit suppresses re-entry until an explicit flat signal, including through opposite signals.

### Recorded prices and clocks

`max_holding_seconds` measures elapsed UTC time from first fill. Expiry is requested at the first declared signal or timer event reaching that deadline. `timer_interval_seconds` adds checks on the UTC epoch grid with `timer_phase_utc_seconds`; timers make no predictions. Requests remain pending until a recorded admissible price arrives. End marking leaves open episodes open and charges no invented close fee.

`max_holding_seconds`, `timer_interval_seconds` and `max_price_gap_seconds` accept `None` to disable the control in event execution, including `null` candidates selected through a `{param}` search reference; a reference still requires explicit product metadata, and a null price-gap bound retains recorded interval/availability requirements.

Execution observations carry `row_id, start_ns, end_ns, open_available_at_ns, available_at_ns, open, high, low, close`; times are integer UTC nanoseconds. Points have equal start/end and one recorded price. OHLC opens are usable at known starts, close/extrema only at availability. Held barriers process SL before TP, with adverse gaps and interval uncertainty. Whole-interval extrema cannot be applied to inventory entered/resized inside that interval.

Regular raw OHLC may supply a verified source interval (`kline_size` in seconds, including `HistoricalData.get_spot_klines`, legacy `klines_size`, or constant `base_interval`). A non-null `kline_size` takes precedence; absent or null `kline_size` retains `klines_size`. Irregular sources must supply actual interval/availability metadata. Limen does not derive irregular duration from bar count or interpolate prices. Optional `execution_data_source=DataSourceConfig(method, params)` supplies finer recorded observations; HistoricalData execution sources accept bound or unbound Python methods and YAML method paths. Source parameters resolve for each round unless the fixed-source behavior below applies. Declare any source date bounds explicitly. Its coverage must span the true split; `max_price_gap_seconds` bounds both interval width and gaps. Second-resolution execution therefore requires recorded data meeting that bound.

Formed model bars become available at the maximum declared availability of their complete native constituents. Research execution and Sensor decisions use the same derivation. Native clocks and constituent ranges are retained separately and bound into new execution contracts, including non-max constituent clocks. Derived decisions must remain ordered within the original partition; overtaking and arrivals outside it are refused before target fitting. Event execution admits the standard volume, trade and liquidity formations with recoverable contiguous membership. Custom formation is refused on this path because its native membership is not declared. Independent execution observations retain their own clocks.

When all declared execution and funding source configurations are fixed, serial `UniversalExperimentLoop.run()` captures those sources lazily on first use within an invocation. Rounds and walk-forward folds consume the same captured observations. If a source configuration depends on round or context parameters, existing per-round loading remains. Instance-bound loaders, callable instances and custom parameter, preparation or model overrides retain existing loading. Direct `Manifest.prepare_data()` calls retain their existing loading; another UEL invocation captures fresh observations. Preparation, model training and `factorize` controls retain their existing behavior. Parallel mode keeps its existing source-freezing path.

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

Each round freezes canonical `trade_contract`, SHA-256 digest, source identities/checksums, policy/adapter/preset versions, calibration and event/episode ledger in JSONL. Aggregate reads use the captured results; recomputing configured execution separately requires original bound prices and provenance. A saved contract cannot supply missing prices or recreate private source bindings. See [Log](Log.md#backtest-surface) for aggregate and retained-row boundaries. Deployment parity remains subject to the linked Praxis/Nexus requests.

Execution optimizations retain the complete ledger and original event ordering.
Equal-input numerical comparisons require absolute error at most `1e-10` with
zero relative tolerance; schemas, identities, clocks and missing-value masks are
unchanged. See [computation-reuse comparisons](Backtest-Benchmarks.md#backtest-computation-reuse)
for the recorded-data measurement contract and command.

## Causal entry barriers (research only)

Use `backtest.take_profit_bps_column` and/or `stop_loss_bps_column` instead of the corresponding scalar. Declare an explicit product and a private `barrier_features` producer. It receives admitted, unsplit recorded datetime/OHLC history through the test end; primary features and targets never receive its sidecars. Generic regular sources can declare literal `source_interval_seconds` (for example 3600).

<!-- example: backtest-compiled-barriers source=docs/examples/compiled-research/execution.yaml region=__file__ scenario=execution interface=compiled role=executed -->
```yaml
# region declaration
schema_version: '1.0'
metadata:
  name: book_compiled_execution
  mode: development
sfd:
  manifest:
    type: rule_based
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params:
        file_path_or_url: inputs/spot_1h_20240101_20241231.parquet
    split_dates:
      train_start: '2024-01-02'
      train_end: '2024-01-03'
      val_start: '2024-01-04'
      val_end: '2024-01-05'
      test_start: '2024-01-05'
      test_end: '2024-01-06'
    backtest:
      product:
        kind: cash_spot
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 1e-05
        min_notional: 5
      source_interval_seconds: 3600
      fee_bps: 10
      slip_bps: 5
      take_profit_bps_column: volatility_take_profit_bps
      stop_loss_bps_column: volatility_stop_loss_bps
      barrier_features:
        method: limen.features.volatility_barriers.volatility_barriers
        params:
          span: '{risk_span}'
          regime_span: 24
          min_periods: 3
    reference_architecture: 
      limen.sfd.reference_architecture.rule_based.rule_based
    strategy:
      conditions:
      - id: entry
        name: Recorded close above open
        type: relative
        column: close
        operator: '>'
        other_column: open
      entry: entry
  params:
    risk_span:
    - 6
    - 12
uel:
  output_path: compiled-execution
  n_permutations: 2
  search_strategy:
    type: grid
  output_format: csv
  checkpoint_interval: 1
  prep_each_round: true
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
# endregion declaration
```

The definition retains its complete product and economics declaration alongside the barrier producer. Numeric producer parameters accept `{param}` search references; producer identity and column names are literals. Scalar and column declarations for the same side conflict. Resampled/ambiguous bars are unsupported.

The latest surviving pending intent owns decision-time distances; the first actual fill freezes distances around its fill price. Adds and reductions retain that anchor and the elapsed deadline. Reversals use the reversing decision's distances for the new episode. Barrier/timeout ordering, adverse gap pricing and forced-exit latching retain their existing rules. This does not add a liquidity partial-fill model.

Hypothetical `TradeOutcomeTarget` candidates inside warmup are unavailable with reason `barrier_warmup`; this does not simulate a primary entry.

A producer declares `__barrier_ready__`: false only for a leading warmup prefix with null distances. An entry in that prefix makes the entire sampled candidate unavailable (`StrictModeError`); later readiness cannot manufacture a new entry. Interior missing, nonfinite or nonpositive distances are errors. Distances must be below 10000 bps. Original distance-bearing context is required for Log replay; file-only omitted context cannot reconstruct it. Trainer/Sensor promotion, resume and native workers >1 reject these research configurations.

## Learned allocation around a primary

Keep the original binary primary and add `allocation.method: limen.sfd.reference_architecture.meta_allocation.meta_allocation` to its manifest. The primary architecture must return its original fitted `ReferenceModel`; native rule strategies require no fit. `allocation.params` accepts folds, internal_validation_fraction, min_train_samples, features, include_primary_proba, probability_factory, sizing_factory, probability_params, sizing_params, probability_threshold, return_scale, max_size, risk_budget_fraction and seed. Numeric and feature-list candidates use ordinary `{parameter}` references.

Use event `prediction_mode: target_exposure`, `notional_rate: 1`, zero execution lag and declared regular source intervals. This is a complete research-batch contract; Trainer/Sensor promotion, resume and native parallel workers reject. Reconstruct original targets, preprocessing and calibration within each chronological fitting prefix; never reuse outer fitted state. The bound prediction batch supplies target-free inputs to the original binary fitted model. No live allocation or independent validation is implied.

Chronological prefix fits produce primary decisions on feature-valid held-out training rows. Only primary opportunities with actual fills and closes inside their opening block supervise the allocator. Logistic regression learns positive net return probability; Ridge learns signed net return on entry notional, including losing trades. Both use training-only standardization. Completed and unavailable opportunity counts expose fill/completion selection bias.

At each primary rising edge, freeze exposure as `1[p >= threshold] * min(max_size, max_exposure, max(0, predicted_net_return / return_scale), risk_cap)`. With a risk budget, `risk_cap = risk_budget_fraction / ((stop_bps + 2*fee_bps + 2*slip_bps)/10000)`; otherwise it is unbounded. This allowance cannot guarantee realized loss under gaps. Hold exposure unchanged until primary flat; forced-exit latching remains owned by the execution engine. Flat/zero-return opportunities cannot become fabricated profitable labels.

Named meta features select the original prepared primary surface; optional actual primary probability is appended independently. Detailed training rows use `trade_ledger.allocation_training`, removed by existing trade-ledger recording opt-outs. Optional model outputs distinguish original primary, take probability, signed return estimate and composed exposure. Sweep recording does not promise those arrays. See [Reference Architecture](Reference-Architecture.md) for complete-batch prediction.

Primary keyword resolution is unchanged. A caller-owned architecture accepting `**params` receives the original round parameters even when a producer or allocator references the same key; its adapter owns any filtering before constructing an estimator.

For complete fixed-duration bars stamped at their closing time, declare `backtest.source_timestamp_role: end` and `source_interval_seconds`. Model datetimes and split membership stay unchanged; execution observes each original interval from `datetime - interval` through `datetime`. Ordered integer-multiple gaps remain gaps, with no candles or prices filled. Declare the role from source provenance, not inferred performance. The default `start` retains existing strict regular-grid clocks. Declared crash-reversal bar intervals also accept aligned gaps without reading the next recorded timestamp.

## Recorded execution price costs

`limen.backtest.observed_costs.CostCalibration` audits local normalized fill
records and saves immutable evidence. Recorded cross-account fees are
descriptive; they do not establish another account's fee tier. The bundled
Chainticks sample has three dates, inferred minute snapshot times and strongly
concentrated sell support. Its report exposes those limits explicitly.

The Python project inspects the licensed recorded sample without network access. Its calibration constructor is:

<!-- reference: backtest-cost-calibration source=limen/backtest/observed_costs.py region=signature:CostCalibration.from_history role=signature -->
```python
@classmethod
def from_history(cls, path: str | Path, *, cutoff_ns: int) -> CostCalibration:
    ...
```

Reading a pinned artifact preserves its recorded estimates. Changed content or
an incompatible expected digest fails; writing different content to an existing
artifact path fails. Dataset attribution and reproduction are in
`tests/fixtures/observed_costs/license.json` and `reproduce.py`.

### Frozen candidates in custom Python research

`CostCandidate` freezes caller-owned `TradeInputs`, `TradePolicy`, conditions and
benchmark rows. `CostCalibration.compare()` executes fresh original and
calibrated ledgers from those same signals. It changes only `slip_bps`; candidate
fees, funding, sizing, barriers and lag remain their declared software scenarios.
It neither fits a model nor changes UEL's search or selection rules.

The following Python-project function uses recorded midpoint prices and a caller
model. Its exposure decisions and capital are hypothetical software controls.
The point encoding repeats each recorded midpoint in OHLC; it supplies no
intraminute extrema. Both runs are retrospective stress: the August 4 archive
availability follows the August 1 evaluation, and historical delivery is unknown.

<!-- recorded-cost-python:start -->
<!-- example: python-observed-cost-model source=docs/examples/python-research/cost_research.py region=observed_cost_model scenario=observed_costs interface=python role=executed -->
```python
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from limen.backtest.observed_costs import CostCalibration, CostCandidate, CostConditions
from limen.backtest.trade_contract import (
    PredictionMode, ProductSpec, TradeInputs, TradePolicy, source_binding,
)
from limen.experiment import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy


class CallerModel:
    def fit(self, training):
        self.mean = training["price"].mean()
        return self

    def predict(self, evaluation, mode, scale):
        exposure = [1.0 if price >= self.mean else
                    (0.0 if mode == "binary" else -scale)
                    for price in evaluation["price"]]
        exposure[-1] = 0.0
        return pl.Series("target", exposure, dtype=pl.Float64)


def run_cost_experiment(calibration_path, specification_path, benchmark_path,
                        output, *, mode: PredictionMode="binary", hour=1):
    expected = json.loads(Path(specification_path).read_text())["calibration_digest"]
    calibration = CostCalibration.read(calibration_path, expected_digest=expected)
    benchmark = pl.DataFrame(json.loads(Path(benchmark_path).read_text()), schema={
        "row_id": pl.String, "phase": pl.String, "time_ns": pl.Int64,
        "replay_available_at_ns": pl.Int64, "price": pl.Float64,
        "reference_id": pl.String, "observed_receipt_ns": pl.Int64,
    })
    first = benchmark["time_ns"][0] + (hour - 1) * 3_600_000_000_000
    recorded = benchmark.filter(pl.col("time_ns").is_between(
        first, first + 420_000_000_000))
    output = Path(output)
    output.mkdir()
    comparisons = []
    space = {"scale": [0.5]}

    def prep(frame, round_params):
        return {"training": frame.head(3), "evaluation": frame.tail(5)}

    def model(data, round_params):
        evaluation = data["evaluation"]
        fitted = CallerModel().fit(data["training"])
        predictions = fitted.predict(evaluation, mode, round_params["scale"])
        signals = evaluation.select(
            "row_id", pl.col("time_ns").alias("available_at_ns")
        ).with_columns(predictions)
        observations = evaluation.select(
            "row_id", pl.col("time_ns").alias("start_ns"),
            pl.col("time_ns").alias("end_ns"),
            pl.col("replay_available_at_ns").alias("open_available_at_ns"),
            pl.col("replay_available_at_ns").alias("available_at_ns"),
            *(pl.col("price").alias(name) for name in ("open", "high", "low", "close")),
        )
        start, end = evaluation["time_ns"][0], evaluation["time_ns"][-1]
        binding = source_binding(observations, "recorded_midpoints", start, end,
                                 1, "retrospective_assumed_exchange_label")
        inputs = TradeInputs(10_000.0, start, end, signals, observations, None, (binding,))
        policy = TradePolicy(
            ProductSpec("linear_perpetual", "BTC", "BTC", "USDC", 0.00001, 1.0),
            prediction_mode=mode, fee_bps=5.0, slip_bps=5.0,
        )
        conditions: CostConditions = {
            "venue": "hyperliquid", "environment": "mainnet",
            "product_kind": "linear_perpetual", "instrument": "BTC",
            "base_currency": "BTC", "quote_currency": "USDC",
            "reference_id": benchmark["reference_id"][0],
            "reference_convention": "minute_midpoint_label",
            "clock": "assumed_exchange_label",
        }
        candidate = CostCandidate(inputs, policy, f"caller:experiment-{hour}",
                                  conditions, benchmark)
        comparison = calibration.compare({"caller": candidate})
        comparison.write(output / "comparison.json")
        comparisons.append(comparison)
        return {
            "original_return": comparison.ledger("caller", "original").metrics["total_return"],
            "calibrated_return": comparison.ledger("caller", "calibrated").metrics["total_return"],
        }

    sfd = SimpleNamespace(params=lambda: space, prep=prep, model=model)
    experiment = UniversalExperimentLoop(
        data=recorded, sfd=sfd, search_strategy=GridStrategy(ParamDomain(space)),
    )
    experiment.run(str(output / "experiment"), n_permutations=1, workers=1,
                   post_processing=False, progress_bar=False)
    return experiment, comparisons[0]
```
<!-- recorded-cost-python:end -->

Create the calibration once, pin its digest separately, then reuse that artifact
for the later hour. No estimated rate is copied into the caller's policy:

<!-- example: backtest-python-observed-costs source=docs/examples/python-research/scenarios.py region=observed_costs scenario=observed_costs interface=python role=executed -->
```python
def observed_costs(data_dir, output):
    fixture = data_dir / 'observed_costs'
    expected = {'history.zip': 'df4d7776172df459ec297cbcfe8e72a4748a51ab626e3c990deec83415102093',
                'benchmark.json': 'b71011697713417b21302e670abf78c0bff739ec9135c7012db02fa3b2d6ccf8'}
    for name, digest in expected.items():
        if hashlib.sha256((fixture / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Recorded execution-cost input differs: ' + name)
    with ZipFile(fixture / 'history.zip') as archive:
        history = output / 'history.json'
        history.write_bytes(archive.read('admitted.json'))
    calibration = CostCalibration.from_history(history, cutoff_ns=1785810791000000000)
    artifact, specification = output / 'calibration.json', output / 'research-specification.json'
    calibration.write(artifact)
    specification.write_text(json.dumps({'calibration_digest': calibration.digest}))
    reports = []
    for mode in ('binary', 'target_exposure'):
        for hour in (1, 2):
            experiment, comparison = run_cost_experiment(artifact, specification, fixture / 'benchmark.json',
                                                       output / f'{mode}-{hour}', mode=mode, hour=hour)
            reports.append({'mode': mode, 'hour': hour, 'rounds': project.completed_log(experiment).height,
                            'comparison': comparison.report()})
    return {'calibration': calibration.report(), 'runs': reports,
            'historical_delivery': 'unknown; retrospective stress only'}
```

Each saved comparison binds the pinned calibration, caller research identity,
ordered inputs, benchmark, policies, executed ledgers and support dispositions.
`report()` and `ledger()` return copies. Missing reference/phase evidence or
incompatible conditions retains the candidate with an unavailable calibrated
scenario; malformed inputs raise before any family member executes. Only the
kernel's declared economic failures become `economic_failure`; other errors
propagate. Comparisons have a write API and no read API.

Ranks are scenario-specific complete-family dense ranks. All candidates must
succeed with common partition, benchmark, currency, capital and effective
fee/slippage settings. Otherwise that rank set is null; rank deltas require both
sets. A calibration does not transfer account fees or establish deployment
survival. This opt-in API supports custom Python research; no automatic
Manifest/YAML option is exposed.

## Read next

- Continue to [Trainer](Trainer.md) for promotion of selected experiment rounds into reusable trained sensors.
- Continue to [Log](Log.md) for the post-run workflow that produces backtest inputs.
- Continue to [Benchmark](Benchmark.md) for the prediction-quality layer that precedes trading-economics inspection.
