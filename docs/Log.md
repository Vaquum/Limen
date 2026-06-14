# Log

`Log` is Limen's post-run analysis layer. It sits on top of a finished experiment and turns raw round results into round-level prediction tables, benchmark-style summaries, backtest summaries, and parameter-correlation views.

`UniversalExperimentLoop` creates `uel._log` automatically at the end of a successful run and exposes the primary derived tables directly on the `uel` object.

## Two ways to use `Log`

### UEL-backed `Log`

This is the normal path.

```python
from limen.data import HistoricalData

historical = HistoricalData()
data = historical.get_spot_klines(kline_size=7200, row_count_limit=2000)

uel = limen.UniversalExperimentLoop(data=data, sfd=limen.sfd.logreg_binary)
uel.run(experiment_name='logreg-first', n_permutations=4, prep_each_round=True)

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

A CSV log can also be loaded from disk:

```python
import limen

log = limen.Log(file_path='my_experiment.csv')
```

This path supports the cleaned experiment log itself and experiment-log-only analysis such as parameter correlation.

Important limitation:

- file-backed `Log` does not have `data`, `prep`, `preds`, or `_alignment`
- methods that reconstruct per-round predictions or test windows require a UEL-backed `Log`

## The post-run workflow

The standard sequence is:

1. inspect one round's prediction table
2. compare rounds with benchmark summaries
3. compare rounds with backtest summaries
4. inspect which parameters move with the target metric

```python
log = uel._log

round0 = log.permutation_prediction_performance(round_id=0)
benchmark = log.experiment_confusion_metrics('price_change')
backtest = log.experiment_backtest_results()
correlation = log.experiment_parameter_correlation('auc', min_n=10)
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

- `predictions`
- `actuals`
- `hit`
- `miss`
- `open`
- `close`
- `price_change`

On a live local run in this repo, the table for one round contained 218 test rows with exactly that schema.

Use this table for round-level inspection before summary statistics. It is also the direct input to Limen's snapshot backtest.

## Benchmark surfaces

The benchmark layer measures directional signal quality before translation into trading results.

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

### `experiment_backtest_results(disable_progress_bar=False)`

Produces one snapshot backtest row per experiment round.

```python
bt = uel._log.experiment_backtest_results()
```

The same table is exposed directly on UEL as:

```python
uel.experiment_backtest_results
```

The current summary columns are the 20 bar-based backtest ledger fields — every column is computed per bar over all bars in the window.

Per-bar distributions (`p5` / `p50` / `p95`):

- `edge_bps_*` — gross per-bar return
- `pnl_bps_*` — net per-bar return
- `cost_bps_*` — per-bar cost (gross minus net)
- `drawdown_bps_*` — net equity against its running peak

Intensive scalars:

- `wins_per_bar`, `pnl_per_bar_bps`, `avg_win_bps`, `avg_loss_bps`, `cvar_95_pnl_bps`, `trades_per_bar`, `inventory_per_bar`, `cost_per_bar_bps`

Use this table to compare trading economics after benchmark inspection.

Post-run snapshot backtests currently support:

- binary `0/1` predictions directly
- directional regression scores via sign (`pred > 0` -> long, otherwise flat)

Logged multiclass outputs are not supported on this surface and raise explicitly instead of being silently collapsed.

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

with columns:

- `n_rows`
- `corr`
- `corr_med`
- `ci_lo`
- `ci_hi`
- `sign_stability`

This method requires enough rows to support cohort-level interpretation. Tiny runs produce legal output but unstable estimates.

## Persisting predictions and complex artifacts

`Log` needs these fields for round-level reconstruction:

- store test predictions as `round_results['_preds']`
- keep prep deterministic with respect to `round_params`
- use `prep_each_round=True` when the prep stage depends on round parameters

Large model or prep objects that should not be flattened into the experiment log belong under:

```python
round_results['extras'] = {'model': fitted_model}
```

UEL preserves those in `uel.extras`.

## Determinism matters

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

Use `read_from_file()` to recover or inspect an experiment log outside a live UEL object.

## Read next

- Continue to [Benchmark](Benchmark.md) for the prediction-quality layer built on top of `Log`.
- Continue to [Backtest](Backtest.md) for the trading-economics layer built on top of `permutation_prediction_performance()`.
- Continue to [Trainer](Trainer.md) for promotion of selected experiment rounds into reusable sensors.
