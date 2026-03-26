# Standard Metrics Library

Limen's metrics layer provides the low-level evaluation helpers used inside reference-architecture model functions.

## Public Surface

The current public metrics exports are:

- `binary_metrics`
- `multiclass_metrics`
- `continuous_metrics`
- `balanced_metric`
- `safe_ovr_auc`

These live under `limen.metrics`.

## `binary_metrics(data, preds, probs)`

Compute core binary-classification metrics from `data['y_test']`, predicted labels, and positive-class probabilities.

### Returns

A `dict` with:

- `recall`
- `precision`
- `fpr`
- `auc`
- `accuracy`

## `multiclass_metrics(data, preds, probs, average='macro')`

Compute multiclass classification metrics from `data['y_test']`, predicted labels, and class probabilities.

### Returns

A `dict` with:

- `precision`
- `recall`
- `auc`
- `accuracy`

## `continuous_metrics(data, preds)`

Compute regression metrics from `data['y_test']` and continuous predictions.

### Returns

A `dict` with:

- `bias`
- `mae`
- `rmse`
- `r2`
- `mape`

## `balanced_metric(y_true, y_pred)`

Compute Limen's balanced binary score for threshold selection and model evaluation when class balance matters.

Use it when you want a single scalar that penalizes degenerate behavior such as always predicting one class.

## `safe_ovr_auc(y_true, probs)`

Compute one-vs-rest AUC while handling edge cases more safely than a direct raw sklearn call.

Use it when class presence is unstable across folds or permutations.

## Usage

Reference architectures typically call these helpers directly:

```python
from limen.metrics import binary_metrics

results = binary_metrics(data, preds, probs)
results['_preds'] = preds
```

These helpers are intentionally small and composable. Higher-level experiment analytics such as confusion summaries and backtests are handled by `limen.log`, not by this module.
