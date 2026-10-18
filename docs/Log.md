# Log

`Log` reads captured experiment metrics and supports explicit row-level analysis when the required live evidence was retained.

For YAML CLI runs, start from the generated result directory and `results.csv`. After a Python/MSQ run, construct `limen.Log(uel_object=uel)` for captured aggregates. Use `post_processing=True` for retained row analysis and the automatic `uel._log`.

In [walk-forward sweeps](Experiment-Manifest.md#walk-forward-sweeps), `results.csv` contains trial aggregates. Use the recorded fold results and per-trial net-return track to inspect the individual test windows behind those means.

## Prerequisites

- a completed `results.csv`, or a successful direct UEL run
- `post_processing=True` when analysis needs retained predictions, scalers, alignment, or the live `uel._log`
- complete captured metric columns for aggregate views; compatible prices and retained evidence for explicit row analysis

## Two ways to use `Log`

### UEL-backed `Log`

Run the [Python/MSQ example](Universal-Experiment-Loop.md#direct-python-run) with `post_processing=True`, then access the live analysis object:

```python
log = uel._log
```

This mode has access to:

- the experiment dataframe
- the round parameters
- stored predictions
- alignment metadata
- prep logic needed to reconstruct per-round test windows

Aggregate views require only the captured experiment rows. Construct `limen.Log(uel_object=uel)` after a run even when `post_processing=False`; retained state is needed only for explicit row analysis.

### File-backed `Log`

A CSV log can also be loaded from disk:

```python
import limen

log = limen.Log(file_path='my_experiment.csv')
```

This path supports the cleaned experiment log, captured aggregate metrics and parameter correlation. It reads only the supplied CSV; it does not discover or load sibling round or fold artifacts.

Important limitation:

- file-backed `Log` does not have `data`, `prep`, `preds`, or `_alignment`
- methods that reconstruct per-round predictions or test windows require a UEL-backed `Log`

## The post-run workflow

Use captured aggregate views to compare completed rounds. Explicit row inspection is a separate path requiring retained evidence.

The four-round Python/MSQ example supports aggregate and retained row inspection. Correlation needs a larger experiment: with the default cohorts and `min_n=10`, at least 11 rows with valid target metrics are needed for a cohort to qualify, and the metric and numeric features must vary. Continue to [parameter correlation](#parameter-correlation-surface) after that larger run.

```python
log = limen.Log(uel_object=uel)

benchmark = log.experiment_confusion_metrics()
backtest = log.experiment_backtest_results()
```

## `permutation_prediction_performance(round_id)`

This method reconstructs a single round's test-period table and joins:

- model predictions
- actual outcomes
- hit/miss flags
- aligned price data

```python
perf = uel._log.permutation_prediction_performance(round_id=0)
```

The resulting table has these columns:

- `datetime` when the reconstructed price frame contains it
- `predictions`
- `actuals`
- `hit`
- `miss`
- `open`
- `close`
- `price_change`

The row count follows the reconstructed test window; it is not a fixed contract. `round_id` is the ordinal in retained successful-round lists, not a CSV hash or `_round_index`.

Row inspection requires the live serial preparation/data, predictions, round parameters and alignment retained through `post_processing=True`; inverse-scaled features also require the fitted scaler. Missing evidence raises explicitly. Parallel runs do not retain these inputs, and walk-forward row replay requires a fold-specific test set and is unsupported by this method.

Use this table for round-level inspection before summary statistics. It is also a direct input to disabled-barrier snapshots. Configured TP/SL replay reconstructs private high/low columns from verified source identities; this public table remains unchanged.

## Benchmark surfaces

The benchmark layer measures directional signal quality before translation into trading results.

### `experiment_confusion_metrics()`

With no `x`, this method reads the captured `confusion_*` columns and removes their prefix. It preserves stored values and units: `precision` and `recall` remain fractions, not percentages. The complete group contains `tp`, `fp`, `tn`, `fn`, `precision`, `recall`, and the four `*_mean_return_pct` fields.

```python
bench = log.experiment_confusion_metrics()
```

This untrimmed view requires no predictions, scaler, preparation or execution. It excludes recorded strict-mode failures and preserves successful occurrence order, including repeated parameter identities, zero values and valid metric NaNs. Missing or partial captured groups raise; there is no replay fallback. Raw captured columns are retained separately from optional categorical encoding used for parameter correlation.

### `experiment_confusion_metrics(x)`

Passing `x` selects the existing row-analysis summary. It requires retained live evidence and produces one row per successful retained round.

```python
bench = uel._log.experiment_confusion_metrics('price_change')
```

This table combines:

- positive-rate diagnostics
- precision and recall
- TP and FP counts
- mean and median of `x` within TP and FP
- TP-versus-FP separation through Cohen's d and KS

The explicit `price_change` summary is exposed directly on UEL with `post_processing=True` as:

```python
uel.experiment_confusion_metrics
```

because UEL computes:

```python
uel._log.experiment_confusion_metrics('price_change')
```

automatically at the end of the run.

### `permutation_confusion_metrics(x, round_id)`

Produces the same style of summary for one specific round.

```python
round0_conf = uel._log.permutation_confusion_metrics(
    x='price_change',
    round_id=0,
)
```

This view isolates benchmark behavior for one selected round.

### Reading the benchmark table

Benchmark-table review should inspect whether high `precision_pct` comes from selectivity or low activity, whether `recall_pct` captures actual positives, whether `tp_x_mean` and `tp_x_median` exceed `fp_x_mean` and `fp_x_median`, and whether `tp_fp_cohen_d` indicates separation rather than noise.

Benchmark and backtest stay separate because statistical signal quality and trading economics can diverge.

When the confusion table includes:

- `tp_mean_return_pct`
- `fp_mean_return_pct`
- `tn_mean_return_pct`
- `fn_mean_return_pct`

those four fields use the same immediate-next-execution-row contract as snapshot backtests for completed-bar pipelines. They are not same-row feature-bar returns.

## Backtest surface

### `experiment_backtest_results()`

Read one captured backtest row per successful single-split round, removing the `backtest_` prefix. This method performs no preparation, prediction or execution and requires no retained row arrays. It works with UEL-backed or CSV-backed `Log` when the complete metric group is available.

```python
bt = uel._log.experiment_backtest_results()
```

With `post_processing=True`, the same table is exposed directly on UEL as:

```python
uel.experiment_backtest_results
```

Snapshot captures retain the 20 bar-based backtest ledger fields in their existing column order. Event captures retain the execution engine's metric order. Recorded strict-mode failures are excluded; successful occurrences preserve their order, including repeats, valid zeros and NaNs. Missing or partial metric groups raise without replay.

For walk-forward runs, a live `Log` retains the existing captured per-fold table with `id` and `fold`. These rows are distinct from the trial averages in `results.csv`. CSV input reads only its captured trial columns; it cannot supply an `id`/`fold` table. Event trial averages use the same metric names as single-split results, so their names alone cannot establish fold evidence.

Per-bar distributions (`p5` / `p50` / `p95`):

- `edge_bps_*` — gross per-bar return
- `pnl_bps_*` — net per-bar return
- `cost_bps_*` — per-bar cost (gross minus net)
- `drawdown_bps_*` — net equity against its running peak

Intensive scalars:

- `wins_per_bar`, `pnl_per_bar_bps`, `avg_win_bps`, `avg_loss_bps`, `cvar_95_pnl_bps`, `trades_per_bar`, `inventory_per_bar`, `cost_per_bar_bps`

The metrics were computed with each round's declared trading policy during execution; aggregate reads preserve those numbers. Source changes after capture cannot change the aggregate result. For explicit snapshot inputs, TP/SL provenance and source requirements, see [Backtest](Backtest.md#tunable-take-profit-and-stop-loss).

## Parameter correlation surface

### `experiment_parameter_correlation(metric)`

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

The helper works over numeric columns that remain after cleaning. It drops constants and all-NaN columns. Framework bookkeeping columns (`id`, `_id`, `_round_index`, `execution_time`, `_warnings`) are excluded from feature correlation by default, so the review stays focused on experiment parameters, diagnostics, and metrics.

with columns:

- `n_rows`
- `corr`
- `corr_med`
- `ci_lo`
- `ci_hi`
- `sign_stability`

The default `min_n=10` skips smaller cohorts. If every requested cohort is smaller than `min_n`, or cleaning leaves no usable numeric cohort, the method raises `ValueError` instead of returning an unstable table.

## Parameter importance

### `experiment_parameter_importance(metric, *, parameters, random_state=0, n_repeats=10)`

Estimate how much each explicitly selected parameter helps a Random Forest predict one recorded metric. This advisory analysis reads captured successful experiment rows from a UEL-backed or CSV-backed `Log`; it does not prepare data, fit the experiment's models, replay execution, write artifacts or alter the search.

For an objective-enabled run with at least 20 distinct recorded `lookback`/`C` configurations:

```python
log = limen.Log(file_path='results.csv')
importance = log.experiment_parameter_importance(
    'val_backtest_total_return',
    parameters=['lookback', 'C'],
    random_state=0,
    n_repeats=10,
)
```

Select parameter columns explicitly; bookkeeping and other metric columns are not inferred as inputs. Parameters must contain finite scalar numbers, booleans or text, with no null or nested values. The chosen metric must be finite and numeric on every successful row. Missing, duplicate or malformed selections raise. Recorded strict-mode failures are excluded. CSV input retains the existing CSV reader's type interpretation; this method does not reconstruct typed round parameters.

`random_state` must be a non-boolean integer in `[0, 2**32 - 1]`; `n_repeats` must be a non-boolean integer of at least 2.

The seeded split reserves 25% of distinct selected configurations for evaluation. Repeated occurrences remain in the analysis, but identical selected parameter tuples always stay on the same side. At least 20 distinct configurations are required; this is a diagnostic floor, not evidence of statistical adequacy.

A 100-tree, single-threaded Random Forest fits the training rows, with categorical one-hot encoding fitted only on that training set. Unseen evaluation categories receive an all-zero encoding; their effects were not learned. Held-out permutation importance shuffles each original parameter column and measures the decrease in R². Categorical levels are therefore permuted together rather than interpreted as separate parameters. A constant held-out metric raises because R² would not provide a meaningful baseline.

The result contains one row per parameter, ordered by descending `importance_mean`, with input order breaking ties:

- `parameter`
- `importance_mean`, `importance_std`
- `baseline_score`: held-out R² before shuffling
- `n_train`, `n_evaluation`: retained occurrence counts in each partition
- `n_configurations`: total distinct selected configurations

`importance_std` is dispersion across repeated shuffles, not a confidence interval or significance measure. Negative importances are retained. A poor baseline score makes the importance estimates uninformative. Correlated parameters can share or obscure importance, and adaptive sampling leaves unexplored regions unmeasured. These estimates describe predictive usefulness within the recorded sample; they establish neither causal effects nor safe pruning decisions. Reproducibility requires the same input values and library versions; CSV parsing precision can change forest tie decisions relative to live rows.

## Analyzing perturbation impact

Round parameter values become columns in `experiment_log` alongside the metrics. When ablation is configured, `_dropped_features` is taken from the actual prepared parameters and serialized as JSON text on every row, including `[]` for zero drops; automatic post-processing preserves this column. A perturbation sweep therefore leaves its own audit trail: `scaler_type`, `feature_groups`, any `use_*` toggle, `feature_drop_count`, `feature_drop_seed`, and the JSON-encoded `_dropped_features` list all appear as columns.

Because `experiment_log` is a pandas DataFrame, the simplest analysis is direct filtering and grouping:

```python
robust_only = log.experiment_log[log.experiment_log['scaler_type'] == 'robust']
by_groups = log.experiment_log.groupby('feature_groups')['auc'].mean()
```

`experiment_parameter_correlation` coerces every column to numeric, so string and list columns — `scaler_type`, `feature_groups`, and the JSON-encoded `_dropped_features` — become NaN and drop out. Only numeric perturbation columns survive; boolean `use_*` flags survive because they coerce to `0`/`1` and group naturally.

To bring a categorical perturbation into the correlation, one-hot it when constructing the `Log`:

```python
log = limen.Log(file_path='my_experiment.csv', cols_to_multilabel=['scaler_type'])
```

That replaces `scaler_type` with `scaler_type_robust`, `scaler_type_logreg`, … `0`/`1` columns (and casts bool columns to int), which then reach `experiment_parameter_correlation`. It cannot expand the JSON-encoded `_dropped_features`; deriving feature importance from ablation needs a per-feature boolean membership column, covered in [Perturbation Strategies](Perturbation-Strategies.md).

## Persisting predictions and complex artifacts

`Log` needs these fields for round-level reconstruction:

- store test predictions as `round_results['_preds']`
- keep prep deterministic with respect to `round_params`
- Python/MSQ prepares data for every candidate; legacy callers must use `prep_each_round=True` when preparation depends on round parameters

Large model or prep objects that should not be flattened into the experiment log belong under:

```python
round_results['extras'] = {'model': fitted_model}
```

UEL preserves those in `uel.extras`.

## Determinism matters

Explicit row analysis reconstructs test data by replaying the relevant prep path. Captured aggregate views do not. Non-deterministic prep logic can break row alignment between:

- stored predictions
- reconstructed actuals
- reconstructed prices

For reliable explicit row analysis:

- prefer deterministic prep
- if randomness is necessary, make it explicit in `round_params`
- retain each round's alignment metadata and predictions with `post_processing=True`

## `read_from_file(file_path)`

`read_from_file()` is the CSV-cleaning helper behind file-backed `Log`.

It:

- removes duplicated header rows that may appear in streamed CSV logs
- trims whitespace in object columns
- returns a cleaned pandas dataframe

Use `read_from_file()` to recover or inspect an experiment log outside a live UEL object.

## Read next

- Continue to [Benchmark](Benchmark.md) for the prediction-quality layer built on top of `Log`.
- Continue to [Backtest](Backtest.md) for the trading-economics layer and explicit snapshot inputs.
- Continue to [Trainer](Trainer.md) for promotion of selected experiment rounds into reusable sensors.

For signed sizing, elapsed exits, recorded execution clocks and funding configuration, see [Backtest](Backtest.md#signed-exposure-elapsed-exits-and-funding).
