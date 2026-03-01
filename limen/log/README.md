# `limen.log`

> Analyse and surface experiment results: per-permutation performance, confusion metrics, backtest statistics, and parameter correlations.

## Responsibilities

Owns post-experiment analysis — takes the completed `UniversalExperimentLoop` state and provides structured views of what happened across all permutations.
Does **not** own the experiment runner, model training, or persistence (CSV/SQLite writing is handled by UEL).

## Key concepts

- **Log** – main class; constructed either from a live UEL object or from a saved CSV file; exposes analysis methods as bound methods.
- **permutation_prediction_performance** – reconstructs the aligned test-period predictions and actuals for a single round, returning a tidy DataFrame.
- **permutation_confusion_metrics** – computes precision, recall, TP/FP statistics, and distribution separability (Cohen's d, KS test) for a single round.
- **experiment_confusion_metrics** – aggregates confusion metrics across all permutations into one summary DataFrame.
- **experiment_backtest_results** – runs `backtest_snapshot` over every permutation and returns a combined performance DataFrame.
- **experiment_parameter_correlation** – computes correlation between hyperparameter values and performance metrics across all permutations.
- **Alignment** – `_alignment` stores `first_test_datetime`, `last_test_datetime`, and `missing_datetimes` per round so the log can precisely reconstruct the test window.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `Log` | `log.py` | Created automatically by UEL after `.run()`; also constructable from a CSV file path |
| `permutation_prediction_performance(round_id)` | `_permutation_prediction_performance.py` | Get predictions + actuals + price data for one round |
| `permutation_confusion_metrics(round_id)` | `_permutation_confusion_metrics.py` | Detailed classification stats for one round |
| `experiment_confusion_metrics(target_col)` | `_experiment_confusion_metrics.py` | Aggregated metrics across all rounds |
| `experiment_backtest_results()` | `_experiment_backtest_results.py` | Backtest stats for every permutation |
| `experiment_parameter_correlation` | `_experiment_parameter_correlation.py` | Parameter-vs-metric correlation table |

## Dependencies

- **Internal:** `limen.backtest.backtest_snapshot` (used inside `experiment_backtest_results`)
- **External:** `polars`, `pandas`, `wrangle` (multi-label encoding of string columns)

## Quick orientation
```text
log/
├── log.py                                     # Log class — constructor and _get_test_data_with_all_cols
├── _permutation_prediction_performance.py     # Per-round predictions DataFrame
├── _permutation_confusion_metrics.py          # Per-round classification metrics
├── _experiment_confusion_metrics.py           # Cross-permutation aggregated metrics
├── _experiment_backtest_results.py            # Cross-permutation backtest stats
├── _experiment_parameter_correlation.py       # Param vs. metric correlation
└── _read_from_file.py                         # Load experiment log from CSV
```

## Gotchas / things to know

- `Log` from a file path (`file_path=`) gives access only to `experiment_log`; methods that need `data`, `prep`, or `round_params` (e.g. `permutation_prediction_performance`) require construction from a UEL object.
- `cols_to_multilabel` triggers `wrangle.col_to_multilabel` on string columns; UEL passes these automatically after `.run()`.
- Manifest-aware `Log` instances reconstruct test bars by calling `manifest.compute_test_bars()`, ensuring bar-formation params are applied consistently with training.
- `experiment_confusion_metrics` is also attached directly to the UEL object as `uel.experiment_confusion_metrics` for convenience.
