# `limen.log`

> Turn finished experiment runs into analyzable tables for prediction quality, confusion behavior, backtests, and parameter diagnostics.

## Canonical docs

- [Log](../../docs/Log.md)
- [Benchmark](../../docs/Benchmark.md)
- [Backtest](../../docs/Backtest.md)

## What this package owns

Owns post-run analysis and reconstruction of per-round results.
Does **not** own experiment execution, model training, or artifact persistence during the run itself.

## Key entry points

| Entry point | Use it when | Notes |
|-------------|-------------|-------|
| `Log` | You want the main post-run analysis object | Import from `limen.log.log` |
| `permutation_prediction_performance` | You want aligned predictions, actuals, and price columns for one round | Available as a `Log` method |
| `permutation_confusion_metrics` | You want one-round classification diagnostics | Available as a `Log` method |
| `experiment_confusion_metrics` | You want one table across all permutations | Available as a `Log` method and re-exported helper |
| `experiment_backtest_results` | You want snapshot backtests across all rounds | Uses `limen.backtest` downstream |
| `experiment_parameter_correlation` | You want to inspect which parameters move with a chosen metric | Useful for sweep interpretation |

## Adjacent modules

- `limen.experiment` creates the run state that `Log` analyzes.
- `limen.backtest` powers the backtest layer that `experiment_backtest_results()` calls.
- `limen.cohort` commonly consumes confusion-metrics output downstream.

## Quick orientation

```text
log/
├── log.py                                 # Log class and test-window reconstruction
├── _permutation_prediction_performance.py
├── _permutation_confusion_metrics.py
├── _experiment_confusion_metrics.py
├── _experiment_backtest_results.py
├── _experiment_parameter_correlation.py
└── _read_from_file.py
```

## Things to know

- `Log(file_path=...)` is useful, but it does not expose every method that a live UEL-backed `Log` can support.
- Manifest-aware logs reconstruct the test window with the same bar-formation logic used during the run.
- The package root re-exports helper functions, but the `Log` class itself lives in `limen.log.log`.
- A good mental model is that `Log` is the bridge between raw experiment output and higher-level selection decisions.
- `permutation_confusion_metrics(x='aligned_return_pct')` is the confusion-side path for next-bar TP/FP/TN/FN mean-return analysis.

## Read next

- [Log](../../docs/Log.md)
- [Benchmark](../../docs/Benchmark.md)
- [Backtest](../../docs/Backtest.md)
