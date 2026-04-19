# Log

`Log` is Limen's post-run analysis layer. It sits on top of a finished experiment and turns raw round results into round-level prediction tables, benchmark-style summaries, backtest summaries, and parameter-correlation views.

In most workflows you do not instantiate it yourself. `UniversalExperimentLoop` creates `uel._log` automatically at the end of a successful run and also exposes the most-used derived tables directly on the `uel` object.

## Two Ways To Use `Log`

### UEL-backed `Log`

This is the normal path.

```python
uel = limen.UniversalExperimentLoop(...)
uel.run(...)

log = uel._log
```

This mode has access to:

- the experiment dataframe
- the round parameters
- stored predictions
- alignment metadata
- prep logic needed to reconstruct per-round test windows

That is why the full post-run surface works from a UEL-backed `Log`.

### File-backed `Log`

You can also load a CSV log from disk:

```python
import limen

log = limen.Log(file_path='my_experiment.csv')
```

This path is useful when you mainly want the cleaned experiment log itself, or experiment-log-only analysis such as parameter correlation.

Important limitation:

- file-backed `Log` does not have `data`, `prep`, `preds`, or `_alignment`
- methods that reconstruct per-round predictions or test windows require a UEL-backed `Log`

## The Main Post-Run Workflow

The most common sequence is:

1. inspect one round's prediction table
2. compare rounds with benchmark summaries
3. compare rounds with backtest summaries
4. inspect which parameters move with your target metric

```python
log = uel._log

round0 = log.permutation_prediction_performance(round_id=0)
benchmark = log.experiment_confusion_metrics('price_change')
backtest = log.experiment_backtest_results()
correlation = log.experiment_parameter_correlation('auc', min_n=10)
```

## `permutation_prediction_performance(round_id)`

This is the most concrete place to start. It reconstructs a single round's test-period table and joins:

- model predictions
- actual outcomes
- hit/miss flags
- aligned price data

```python
perf = uel._log.permutation_prediction_performance(round_id=0)
```

The resulting table has these columns:

- `predictions`
- `actuals`
- `hit`
- `miss`
- `open`
- `close`
- `price_change`

On a live local run in this repo, the table for one round contained 218 test rows with exactly that schema.

Use this table when you want to understand a round before jumping to summary statistics. It is also the direct input to Limen's snapshot backtest.

## Benchmark Surfaces

The benchmark layer answers: is the signal making useful directional calls before we translate those calls into trading results?

### `experiment_confusion_metrics(x, disable_progress_bar=False)`

Produces one row per round.

```python
bench = uel._log.experiment_confusion_metrics('price_change')
```

This table combines:

- positive-rate diagnostics
- precision and recall
- TP and FP counts
- mean and median of `x` within TP and FP
- TP-versus-FP separation through Cohen's d and KS

The same summary is exposed directly on UEL as:

```python
uel.experiment_confusion_metrics
```

because UEL computes:

```python
uel._log.experiment_confusion_metrics('price_change')
```

automatically at the end of the run.

### `permutation_confusion_metrics(x, round_id, ...)`

Produces the same style of summary for one specific round.

```python
round0_conf = uel._log.permutation_confusion_metrics(
    x='price_change',
    round_id=0,
)
```

This is the right view when a round looks interesting and you want to inspect its benchmark behavior in isolation.

### Reading the benchmark table

Good questions to ask:

- is `precision_pct` high because the signal is selective, or because it barely predicts positives?
- does `recall_pct` stay useful, or is the model missing most real positives?
- are `tp_x_mean` and `tp_x_median` materially better than `fp_x_mean` and `fp_x_median`?
- is `tp_fp_cohen_d` stable enough to suggest real separation rather than noise?

This is exactly why benchmark and backtest are separate in Limen: a round can look statistically interesting before it proves itself economically.

## Backtest Surface

### `experiment_backtest_results(disable_progress_bar=False)`

Produces one snapshot backtest row per experiment round.

```python
bt = uel._log.experiment_backtest_results()
```

The same table is exposed directly on UEL as:

```python
uel.experiment_backtest_results
```

The current summary columns are:

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

Use this table to compare the trading-economics side of rounds after you have already inspected the benchmark layer.

The snapshot defaults behind this table now execute on the next tradable bar and compute `trade_*` fields per held run. That means historical backtest tables produced before this change are not directly comparable under the same `trade_*` column names.

## Parameter Correlation Surface

### `experiment_parameter_correlation(metric, ...)`

This method looks for robust relationships between experiment parameters and a chosen metric across explicit cohorts.

```python
corr = uel.experiment_parameter_correlation(
    'auc',
    min_n=10,
)
```

The result is a dataframe indexed by:

- `cohort_pct`
- `feature`

with columns:

- `n_rows`
- `corr`
- `corr_med`
- `ci_lo`
- `ci_hi`
- `sign_stability`

This is most useful after a run is large enough to support meaningful cohorts. On tiny runs, the output is still legal but usually too unstable to interpret with confidence.

## Persisting Predictions And Complex Artifacts

If you want `Log` to have enough information for round-level reconstruction:

- store test predictions as `round_results['_preds']`
- keep prep deterministic with respect to `round_params`
- use `prep_each_round=True` when the prep stage depends on round parameters

If your model or prep returns large objects that should not be flattened into the experiment log, put them under:

```python
round_results['extras'] = ...
```

UEL preserves those in `uel.extras`.

## Determinism Matters

`Log` reconstructs test data by replaying the relevant prep path. That means non-deterministic prep logic can break alignment between:

- stored predictions
- reconstructed actuals
- reconstructed prices

For reliable post-run analysis:

- prefer deterministic prep
- if randomness is necessary, make it explicit in `round_params`
- use `prep_each_round=True` when round parameters affect preparation

## `read_from_file(file_path)`

`read_from_file()` is the CSV-cleaning helper behind file-backed `Log`.

It:

- removes duplicated header rows that may appear in streamed CSV logs
- trims whitespace in object columns
- returns a cleaned pandas dataframe

Use it when you need to recover or inspect an experiment log outside a live UEL object.

## Read Next

- Continue to [Benchmark](Benchmark.md) for the prediction-quality layer built on top of `Log`.
- Continue to [Backtest](Backtest.md) for the trading-economics layer built on top of `permutation_prediction_performance()`.
- Continue to [Trainer](Trainer.md) if you want to promote selected experiment rounds into reusable sensors.
