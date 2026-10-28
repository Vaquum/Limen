# Read the recorded experiment

`Log` reads captured experiment metrics and supports explicit row-level analysis when the required live evidence was retained.

For YAML CLI runs, start from the generated result directory and `results.csv`. After a Python/MSQ run, construct `limen.Log(uel_object=uel)` for captured aggregates. Use `post_processing=True` for retained row analysis and the automatic `uel._log`.

In [walk-forward sweeps](Experiment-Manifest.md#walk-forward-sweeps), `results.csv` contains trial aggregates. Use the recorded fold results and per-trial net-return track to inspect the individual test windows behind those means.

All table-valued `Log` methods return `polars.DataFrame`. Correlation results contain explicit `cohort_pct` and `feature` columns instead of a pandas MultiIndex. Use Polars filtering and grouping on these columns. Pandas is not required by the package runtime.

## Follow the two projects

Python Research reads the completed loop's Polars frame and sorts its caller-defined validation return. Compiled Research retains the configured native results on disk; `limen assess` ranks their declared validation objective. The Python helper below is not a Compiled prerequisite and does not manufacture native metric groups for arbitrary results.

### Python Research

<!-- example: log-python-completed source=docs/examples/python-research/research.py region=completed_log scenario=selection interface=python role=executed -->
```python
def completed_log(loop: UniversalExperimentLoop) -> pl.DataFrame:
    log = loop.experiment_log
    if log is None:
        raise ValueError("The experiment has no completed result log")
    return log
```

### Compiled Research

<!-- example: log-compiled-completed source=docs/examples/compiled-research/research.yaml region=run scenario=search interface=compiled role=executed -->
```yaml
uel:
  output_path: compiled-research
  n_permutations: 8
  workers: 1
  search_strategy:
    type: grid
    shuffle: true
    seed: 17
  prep_each_round: true
  checkpoint_interval: 1
  feedback_interval: 1
  output_format: parquet
  recording:
    predictions: true
    alignment: true
    trade_ledger: true
    fold_details: true
```

`Log` adds captured-group and row-analysis APIs to those results. The constructor
and method signatures below are source-bound reference; the project proof does
not make a four-candidate population sufficient for every diagnostic. The separate analysis campaign below supplies 30 distinct configurations.

## Prerequisites

- a completed `results.csv`, or a successful direct UEL run
- `post_processing=True` when analysis needs retained predictions, scalers, alignment, or the live `uel._log`
- complete captured metric columns for aggregate views; compatible prices and retained evidence for explicit row analysis

## Two ways to use `Log`

### UEL-backed `Log`

Use `post_processing=True` when the next step requires live serial row evidence. The [Compiled project](examples/compiled-research/README.md) enables it for its selection run; the [Python project](examples/python-research/README.md) keeps caller-owned fitted state and reads completed metrics without it.

The live analysis object is `uel._log` after automatic post-processing.

This mode has access to:

- the experiment dataframe
- the round parameters
- stored predictions
- alignment metadata
- prep logic needed to reconstruct per-round test windows

Aggregate views require only the captured experiment rows. Construct `limen.Log(uel_object=uel)` after a run even when `post_processing=False`; retained state is needed only for explicit row analysis.

### File-backed `Log`

The constructor accepts either a live loop or a CSV path:

<!-- reference: log-constructor source=limen/log/log.py region=signature:Log.__init__ role=signature -->
```python
def __init__(self, uel_object: Any | None=None, file_path: str | None=None, inverse_scaler: Callable[[pl.DataFrame, Any], pl.DataFrame] | None=None, cols_to_multilabel: list[str] | None=None) -> None:
    ...
```

This path supports the cleaned experiment log, captured aggregate metrics and parameter correlation. It reads only the supplied CSV; it does not discover or load sibling round or fold artifacts.

Important limitation:

- file-backed `Log` does not have `data`, `prep`, `preds`, or `_alignment`
- methods that reconstruct per-round predictions or test windows require a UEL-backed `Log`

## The post-run workflow

Use captured aggregate views to compare completed rounds. Explicit row inspection is a separate path requiring retained evidence.

The four-round Python selection reads its caller-defined validation metric. It does not claim native confusion groups, snapshot columns or retained row replay. The Compiled selection retains the native groups and serial evidence. Correlation needs a larger experiment: with the default cohorts and `min_n=10`, at least 11 rows with valid target metrics are needed for a cohort to qualify, and the metric and numeric features must vary. Continue to [parameter correlation](#parameter-correlation-surface) after that larger run.

Read `experiment_confusion_metrics()` for captured classifier metrics and `experiment_backtest_results()` for captured trading metrics. These are methods of the constructed `Log`.

## `permutation_prediction_performance(round_id)`

This method reconstructs a single round's test-period table and joins:

- model predictions
- actual outcomes
- hit/miss flags
- aligned price data

<!-- reference: log-round-performance source=limen/log/log.py region=signature:Log.permutation_prediction_performance role=signature -->
```python
@_holdout.replay_guard
def permutation_prediction_performance(self, round_id: int) -> pl.DataFrame:
    ...
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

Replay calls `prep(data, round_params=retained_round_params)` once, matching UEL execution. One-argument-only preparations are unsupported. Preparation errors propagate unchanged, including during UEL post-processing finalization; already recorded rounds remain available.

Use this table for round-level inspection before summary statistics. It is also a direct input to disabled-barrier snapshots. Configured TP/SL replay reconstructs private high/low columns from verified source identities; this public table remains unchanged.

## Benchmark surfaces

The benchmark layer measures directional signal quality before translation into trading results.

### `experiment_confusion_metrics()`

With no `x`, this method reads the captured `confusion_*` columns and removes their prefix. It preserves stored values and units: `precision` and `recall` remain fractions, not percentages. The complete group contains `tp`, `fp`, `tn`, `fn`, `precision`, `recall`, and the four `*_mean_return_pct` fields.

<!-- reference: log-confusion source=limen/log/_experiment_confusion_metrics.py region=signature:experiment_confusion_metrics role=signature -->
```python
def experiment_confusion_metrics(self: Any, x: str | None=None) -> pl.DataFrame:
    ...
```

This untrimmed view requires no predictions, scaler, preparation or execution. It excludes recorded strict-mode failures and preserves successful occurrence order, including repeated parameter identities, zero values and valid metric NaNs. Missing or partial captured groups raise; there is no replay fallback. Raw captured columns are retained separately from optional categorical encoding used for parameter correlation.

### `experiment_confusion_metrics(x)`

Passing `x` selects the existing row-analysis summary. It requires retained live evidence and produces one row per successful retained round.

Pass `x="price_change"` to request row analysis instead of the captured aggregate group.

This table combines:

- positive-rate diagnostics
- precision and recall
- TP and FP counts
- mean and median of `x` within TP and FP
- TP-versus-FP separation through Cohen's d and KS

With `post_processing=True`, UEL exposes the explicit row summary as `uel.experiment_confusion_metrics`, computed through `uel._log.experiment_confusion_metrics("price_change")` at run conclusion.

Rounds whose effective outputs are not binary classification — a non-binary
`prediction_mode`, fractional predictions, or no classifiable labels — raise
`ConfusionNotApplicableError`. UEL finalization then records
`experiment_confusion_metrics` as `None` and still populates backtests,
parameter correlation, and the convergence report.

### `permutation_confusion_metrics(x, round_id)`

Produces the same style of summary for one specific round.

<!-- reference: log-round-confusion source=limen/log/_permutation_confusion_metrics.py region=signature:permutation_confusion_metrics role=signature -->
```python
def permutation_confusion_metrics(self: Any, x: str, round_id: int, *, pred_col: str='predictions', actual_col: str='actuals', proba_col: str | None=None, threshold: float=0.5, outlier_quantiles: Sequence[float]=(0.01, 0.99), outlier_mode: str='filter', id_cols: dict[str, Any] | None=None) -> pl.DataFrame:
    ...
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

<!-- reference: log-backtest source=limen/log/_experiment_backtest_results.py region=signature:experiment_backtest_results role=signature -->
```python
def experiment_backtest_results(self: Any) -> pl.DataFrame:
    ...
```

With `post_processing=True`, UEL also exposes the table through `uel.experiment_backtest_results`.

Snapshot captures retain the 20 bar-based backtest ledger fields in their existing column order. Event captures retain the execution engine's metric order. Recorded strict-mode failures are excluded; successful occurrences preserve their order, including repeats, valid zeros and NaNs. Missing or partial metric groups raise without replay.

For walk-forward runs, a live `Log` retains the existing captured per-fold table with `id` and `fold`. These rows are distinct from the trial averages in `results.csv`. CSV input reads only its captured trial columns; it cannot supply an `id`/`fold` table. Event trial averages use the same metric names as single-split results, so their names alone cannot establish fold evidence.

Per-bar distributions (`p5` / `p50` / `p95`):

- `edge_bps_*` — gross per-bar return
- `pnl_bps_*` — net per-bar return
- `cost_bps_*` — per-bar cost (gross minus net)
- `drawdown_bps_*` — net equity against its running peak

Intensive scalars:

- `wins_per_bar`, `pnl_per_bar_bps`, `avg_win_bps`, `avg_loss_bps`, `cvar_95_pnl_bps`, `trades_per_bar`, `inventory_per_bar`, `cost_per_bar_bps`

The metrics were computed with each round's declared trading policy during execution; aggregate reads preserve those numbers. Source changes after capture cannot change the aggregate result. Aggregate readers do not invoke the [internal legacy snapshot helper](Backtest.md#internal-legacy-snapshot-normalization). For explicit snapshot inputs, TP/SL provenance and source requirements, see [Backtest](Backtest.md#tunable-take-profit-and-stop-loss).

For retained multiclass or multilabel prediction/price rows, call the module-level `backtest_snapshot()` with an explicit `class_mapping` or `label_index`; [Backtest](Backtest.md#explicit-multiclass-and-multilabel-inputs) defines accepted shapes and mapping. This produces a separate snapshot interpretation. Aggregate reads preserve captured metrics, and a file-only Log without row evidence cannot reconstruct missing predictions or prices.

## Parameter correlation surface

### `experiment_parameter_correlation(metric)`

This method looks for robust relationships between experiment parameters and a chosen metric across explicit cohorts.

<!-- reference: log-parameter-correlation source=limen/log/_experiment_parameter_correlation.py region=signature:experiment_parameter_correlation role=signature -->
```python
def experiment_parameter_correlation(self: Any, metric: str, *, cols_to_drop: list[str] | None=None, sort_key: str | None=None, sort_ascending: bool=False, heads: Sequence[float | int] | None=None, method: Literal['pearson', 'kendall', 'spearman']='spearman', n_boot: int=300, min_n: int=10, random_state: int=0) -> pl.DataFrame:
    ...
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

This method requires at least 20 distinct selected configurations. The two-candidate Compiled selection and four-candidate Python selection do not satisfy that floor. Its current signature is:

<!-- reference: log-parameter-importance source=limen/log/_experiment_parameter_importance.py region=signature:experiment_parameter_importance role=signature -->
```python
def experiment_parameter_importance(self: object, metric: str, *, parameters: Sequence[str], random_state: int=0, n_repeats: int=10) -> pl.DataFrame:
    ...
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

Use the captured parameter columns to compare perturbations:

Filter the Polars `Log.experiment_log` by its `scaler_type` column to compare one scaler. Group its `feature_groups` column and aggregate a captured metric to compare declared groups. Both the loop and `Log` store Polars frames; `Log` clones the captured rows.

`experiment_parameter_correlation` coerces every column to numeric, so string and list columns — `scaler_type`, `feature_groups`, and the JSON-encoded `_dropped_features` — become NaN and drop out. Only numeric perturbation columns survive; boolean `use_*` flags survive because they coerce to `0`/`1` and group naturally.

To bring a categorical perturbation into the correlation, one-hot it when constructing the `Log`:

Pass `cols_to_multilabel=["scaler_type"]` to the `Log` constructor above.

That replaces `scaler_type` with `scaler_type_robust`, `scaler_type_logreg`, … `0`/`1` columns (and casts bool columns to int), which then reach `experiment_parameter_correlation`. It cannot expand the JSON-encoded `_dropped_features`; deriving feature importance from ablation needs a per-feature boolean membership column, covered in [Perturbation Strategies](Perturbation-Strategies.md).

## Persisting predictions and complex artifacts

`Log` needs these fields for round-level reconstruction:

- store test predictions as `round_results['_preds']`
- keep prep deterministic with respect to `round_params`
- Python/MSQ prepares data for every candidate; legacy callers must use `prep_each_round=True` when preparation depends on round parameters

Large model or prep objects that should not be flattened into the experiment log belong under:

`round_results["extras"]` as a mapping; for example, a `model` entry may retain the fitted object.

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
- trims whitespace in string columns
- returns a cleaned Polars dataframe

<!-- reference: log-read-file source=limen/log/_read_from_file.py region=signature:read_from_file role=signature -->
```python
def read_from_file(_self: Any, file_path: str) -> pl.DataFrame:
    ...
```

Use `read_from_file()` to recover or inspect an experiment log outside a live UEL object.

## Read next

- Continue to [Benchmark](Benchmark.md) for the prediction-quality layer built on top of `Log`.
- Continue to [Backtest](Backtest.md) for the trading-economics layer and explicit snapshot inputs.
- Continue to [Trainer](Trainer.md) for promotion of selected experiment rounds into reusable sensors.

For signed sizing, elapsed exits, recorded execution clocks and funding configuration, see [Backtest](Backtest.md#signed-exposure-elapsed-exits-and-funding).

## A population for parameter analysis

Python Research fits ten momentum windows against three Ridge penalties on the same recorded chronological population. Correlation resamples the recorded validation RMSE. Importance holds out complete parameter configurations, fits a surrogate and measures its loss of R² when a parameter is shuffled; it is not a causal feature-importance measure of the research model.

<!-- example: log-python-analysis source=docs/examples/python-research/analysis_cases.py region=analysis_results scenario=search interface=python role=executed -->
```python
def analyze_log(loop, directory, space):
    analysis = Log(uel_object=loop)
    correlation = analysis.experiment_parameter_correlation('validation_rmse', cols_to_drop=[column for column in project.completed_log(loop).columns if column not in {*space, 'validation_rmse'}],
        heads=(1.,), n_boot=50, min_n=10, random_state=42)
    importance = analysis.experiment_parameter_importance('validation_rmse', parameters=['window', 'alpha'], n_repeats=2, random_state=42)
    correlation.write_csv(directory / 'correlation.csv')
    importance.write_csv(directory / 'importance.csv')
    return correlation, importance
```

<!-- result: log-python-importance interface=python scenario=search fields=analysis.importance[].parameter,analysis.importance[].importance_mean,analysis.importance[].baseline_score,analysis.importance[].n_configurations -->

Compiled Research's `analysis.yaml` runs 20 threshold candidates across the recorded walk-forward windows. The CLI writes return tracks and an acceptance report; it does not expose the Python Log methods as CLI subcommands. The report compares trial returns using their recorded timestamps and reports DSR and PBO. Twenty candidates do not create additional independent time windows. With two PBO blocks the probability resolution remains 0.5.

<!-- result: log-compiled-analysis interface=compiled scenario=search fields=analysis.n_trials,analysis.n_bars,analysis.deflated_sharpe_probability,analysis.pbo,analysis.pbo_resolution -->

Configured independent exits retain their static rule definition and evidence digest in each round. `recording.alignment` alone controls the ordered `exit_evidence` rows, independently of entry predictions and execution series. Serial, parallel and fold records carry the same fields. Replay requires original bound prices plus matching saved and freshly prepared rules, identities, availability and exit values; lighter/file-only records cannot supply them. Adding, removing or changing the unresolved exit declaration rejects resume before artifact writes. See [independent exits](Experiment-Manifest.md#independent-exit-rules).
