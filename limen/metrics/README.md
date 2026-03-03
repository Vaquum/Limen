# `limen.metrics`

> Compute classification and regression evaluation metrics from model predictions.

## Responsibilities

Owns the library of scoring functions used inside SFD model functions to quantify prediction quality.
Does **not** own the model itself, data preparation, or results logging — metric functions are pure, stateless computations.

## Key concepts

- **binary_metrics** – standard binary classification scores (recall, precision, FPR, AUC, accuracy) from `sklearn`.
- **multiclass_metrics** – weighted/macro precision, recall, and F1 for multi-class problems.
- **continuous_metrics** – regression metrics (MAE, RMSE, R²) for continuous targets.
- **safe_ovr_auc** – OvR (One-vs-Rest) AUC that gracefully handles edge cases (single class present, missing classes) without raising.
- **balanced_metric** – composite score that penalises imbalanced predictions; used to reward models that predict both classes meaningfully.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `binary_metrics(data, preds, probs)` | `binary_metrics.py` | Inside an SFD model function after sklearn `predict` / `predict_proba` |
| `multiclass_metrics(data, preds, probs)` | `multiclass_metrics.py` | Inside an SFD model function for multi-class classifiers |
| `continuous_metrics(data, preds)` | `continuous_metrics.py` | Inside an SFD model function for regression targets |
| `safe_ovr_auc(y_true, probs)` | `safe_ovr_auc.py` | Drop-in replacement for `roc_auc_score` when class presence is not guaranteed |
| `balanced_metric(data, preds, probs)` | `balanced_metric.py` | Composite scoring function rewarding calibrated, balanced classifiers |

## Dependencies

- **Internal:** none — this is a leaf module
- **External:** `scikit-learn` (sklearn.metrics), `numpy`

## Quick orientation
```text
metrics/
├── binary_metrics.py      # Recall, precision, FPR, AUC, accuracy
├── multiclass_metrics.py  # Weighted/macro P, R, F1 for multi-class
├── continuous_metrics.py  # MAE, RMSE, R² for regression
├── safe_ovr_auc.py        # OvR AUC with edge-case handling
└── balanced_metric.py     # Composite balanced classification score
```

## Gotchas / things to know

- `binary_metrics` expects `data['y_test']` — the standard key used by `split_data_to_prep_output`.
- `safe_ovr_auc` returns `NaN` (not an exception) when only one class is present in `y_true` or the probability matrix has missing columns.
- `balanced_metric` is intended as a single-number optimisation target for hyperparameter search; it is not a standard sklearn metric.
