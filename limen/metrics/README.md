# `limen.metrics`

> Score classifier and regressor outputs so experiments can compare permutations on consistent metrics.

## Canonical docs

- [Standard Metrics Library](../../docs/Standard-Metrics-Library.md)

## What this package owns

Owns the metric helpers used inside model functions and experiment outputs.
Does **not** own model fitting, prediction generation, experiment logging, or backtesting.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `binary_metrics` | Binary-classification metrics from predictions and probabilities | Import from `limen.metrics.binary_metrics` for the function form |
| `multiclass_metrics` | Macro or weighted metrics for multiclass problems | Import from `limen.metrics.multiclass_metrics` |
| `continuous_metrics` | Regression metrics like MAE, RMSE, and R2 | Import from `limen.metrics.continuous_metrics` |
| `safe_ovr_auc` | OvR AUC without blowing up on missing-class edge cases | Import from `limen.metrics.safe_ovr_auc` |
| `balanced_metric` | Single optimization target for balanced binary prediction quality | Exported directly from the package root |

## Adjacent modules

- `limen.sfd.reference_architecture` is a common caller of these helpers.
- `limen.log` and `limen.backtest` evaluate experiment outcomes after the model phase, but they are downstream from these raw metrics.
- `limen.utils` re-exports a small subset of metrics for convenience on older code paths.

## Quick orientation

```text
metrics/
├── binary_metrics.py
├── multiclass_metrics.py
├── continuous_metrics.py
├── safe_ovr_auc.py
└── balanced_metric.py
```

## Things to know

- The clearest import style is function-level imports from each module, even though the package root also exposes part of the surface.
- `binary_metrics` assumes the standard Limen `data_dict` shape and reads `data['y_test']`.
- `safe_ovr_auc` returns `NaN` rather than raising when the class structure makes AUC undefined. Reindex probability columns before calling it if a fold omits an intermediate class.
- `balanced_metric` is Limen-specific and should not be described as a standard sklearn metric.

## Read next

- [Standard Metrics Library](../../docs/Standard-Metrics-Library.md)
- [Single-File Decoder](../../docs/Single-File-Decoder.md)
- [Log](../../docs/Log.md)
