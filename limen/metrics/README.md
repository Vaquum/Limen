# `limen.metrics`

> Compute classification and regression evaluation metrics from model predictions.

## Responsibilities

Provides small, focused functions that accept a `data` dict (containing `y_test`, `y_val`, etc.) plus prediction arrays and return a flat dict of scalar metric values, ready to merge into the round-results record.

Does **not** own the experiment loop, data preparation, or any model training.

## Key concepts

- **binary_metrics** – precision, recall, FPR, AUC, and accuracy for binary classifiers
- **multiclass_metrics** – per-class and macro-averaged metrics for multi-class classifiers
- **continuous_metrics** – MAE, RMSE, and R² for regression models
- **safe_ovr_auc** – one-vs-rest AUC that handles edge cases (single-class test sets, missing classes) without raising
- **balanced_metric** – composites precision and recall into a single score with configurable weighting

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `binary_metrics()` | `binary_metrics.py` | Called in an SFD's model function after binary classification |
| `multiclass_metrics()` | `multiclass_metrics.py` | Called in an SFD's model function after multi-class classification |
| `continuous_metrics()` | `continuous_metrics.py` | Called in an SFD's model function after regression |
| `safe_ovr_auc()` | `safe_ovr_auc.py` | Drop-in replacement for `roc_auc_score` when test sets may be single-class |
| `balanced_metric()` | `balanced_metric.py` | Combine precision + recall into a single optimisation target |

## Dependencies

- **Internal:** none
- **External:** `scikit-learn` (`roc_auc_score`, `accuracy_score`, `precision_score`, `recall_score`, `confusion_matrix`)

## Quick orientation

```text
metrics/
├── binary_metrics.py       # precision, recall, FPR, AUC, accuracy
├── multiclass_metrics.py   # per-class and macro classification metrics
├── continuous_metrics.py   # MAE, RMSE, R²
├── safe_ovr_auc.py         # edge-case-safe OvR AUC
└── balanced_metric.py      # weighted precision/recall composite
```

## Gotchas / things to know

- All functions return values rounded to 3 decimal places
- `binary_metrics` expects `data['y_test']` to contain ground-truth labels as a 1-D array; `preds` are hard-label predictions and `probs` are class-1 probabilities
- `safe_ovr_auc` silently returns `nan` (or a configurable fallback) when the test set contains only one class — useful inside experiment loops where edge-case rounds should not crash the run
