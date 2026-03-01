# `limen.log`

> Aggregate and analyse the results of a completed `UniversalExperimentLoop` run.

## Responsibilities

Provides a `Log` object that wraps the experiment result DataFrame and exposes methods for computing permutation-level prediction performance, confusion metrics, backtest results, and parameter correlations.

Does **not** own raw experiment execution, model training, or backtest simulation logic — it reads results already produced by `limen.experiment` and delegates backtesting to `limen.backtest`.

## Key concepts

- **Log** – main facade; constructed from a UEL object (in-memory) or a CSV file path (offline); exposes all analysis methods as properties or callable methods
- **permutation_prediction_performance** – for a single round, reconstructs test-period rows with all columns and aligns predictions to them
- **experiment_confusion_metrics** – computes confusion/performance statistics across all permutations for a target column
- **experiment_backtest_results** – runs `backtest_snapshot` across all permutations and returns a summary DataFrame
- **experiment_parameter_correlation** – correlation matrix between parameter values and performance metrics
- **_alignment** – per-round dict recording `first_test_datetime`, `last_test_datetime`, and `missing_datetimes` for precise test-window reconstruction

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `Log` | `log.py` | Instantiated automatically by UEL after `.run()`; also constructable from a file path |
| `Log.permutation_confusion_metrics(round_id)` | `_permutation_confusion_metrics.py` | Per-round confusion matrix stats |
| `Log.permutation_prediction_performance(round_id)` | `_permutation_prediction_performance.py` | Per-round aligned prediction DataFrame |
| `Log.experiment_confusion_metrics(target_col)` | `_experiment_confusion_metrics.py` | Aggregate metrics across all rounds |
| `Log.experiment_backtest_results()` | `_experiment_backtest_results.py` | Backtest summary for all rounds |
| `Log.experiment_parameter_correlation` | `_experiment_parameter_correlation.py` | Parameter–metric correlation table |
| `Log.read_from_file(file_path)` | `_read_from_file.py` | Load a saved experiment CSV into a Log |

## Dependencies

- **Internal:** `limen.backtest.backtest_snapshot`, `limen.experiment.Manifest` (optional, for test-bar reconstruction)
- **External:** `polars`, `pandas`, `wrangle`

## Quick orientation

```text
log/
├── log.py                                    # Log class and _get_test_data_with_all_cols
├── _experiment_backtest_results.py           # experiment_backtest_results mixin
├── _experiment_confusion_metrics.py          # experiment_confusion_metrics mixin
├── _experiment_parameter_correlation.py      # experiment_parameter_correlation mixin
├── _permutation_confusion_metrics.py         # permutation_confusion_metrics mixin
├── _permutation_prediction_performance.py    # permutation_prediction_performance mixin
└── _read_from_file.py                        # read_from_file mixin
```

## Gotchas / things to know

- All analysis methods are mixed in from separate `_*.py` files and imported as class attributes in `log.py`; the split is for organisation, not abstraction
- `_get_test_data_with_all_cols` uses `_alignment` to filter the raw dataset to exactly the test window used in a given round — necessary because different rounds may use different bar types or date ranges
- For manifest-driven experiments, the method calls `manifest.compute_test_bars()` to re-form bars before aligning predictions; for legacy SFDs it falls back to `self.data` directly
- `cols_to_multilabel` in the constructor converts string columns that contain comma-joined labels into one-hot columns via `wrangle.col_to_multilabel`
