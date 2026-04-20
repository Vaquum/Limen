# Standard Metrics Library

Limen's metrics layer provides the low-level evaluation helpers used inside SFD model functions and reference architectures.

These helpers are intentionally small. They compute the core task metrics and return plain dictionaries. Higher-level experiment analytics such as benchmark summaries and backtests live in [Log](Log.md), not here.

## Public Surface

The current public metrics exports are:

- `binary_metrics`
- `multiclass_metrics`
- `continuous_metrics`
- `balanced_metric`
- `safe_ovr_auc`

## Import Pattern

The safest import style is to import the callable from its submodule:

```python
from limen.metrics.binary_metrics import binary_metrics
from limen.metrics.multiclass_metrics import multiclass_metrics
from limen.metrics.continuous_metrics import continuous_metrics
from limen.metrics.balanced_metric import balanced_metric
from limen.metrics.safe_ovr_auc import safe_ovr_auc
```

Why this matters:

- `balanced_metric` is re-exported as a callable
- `limen.metrics` re-exports `binary_metrics`, `multiclass_metrics`, `continuous_metrics`, and `safe_ovr_auc` as modules

So if you write:

```python
from limen.metrics import binary_metrics
```

you are importing the module, not the function.

This low-level metrics layer also sits underneath [Reference Architecture](Reference-Architecture.md). The class-based models add confusion and backtest fields later; these helpers stay at the smaller task-metric level.

## `binary_metrics(data, preds, probs)`

Computes the standard binary metrics used by Limen's binary reference models.

Expected inputs:

- `data['y_test']`
- predicted labels `preds`
- positive-class probabilities `probs`

Returns a dictionary with:

- `recall`
- `precision`
- `fpr`
- `auc`
- `accuracy`

On a live local `LogRegBinary.evaluate(..., inline_metrics=False)` run in this repo, this task-metric layer was exactly:

- `accuracy`
- `auc`
- `precision`
- `recall`
- `fpr`

Example:

```python
results = binary_metrics(data, preds, probs)
results['_preds'] = preds
```

### Edge cases

`binary_metrics()` assumes the test fold is suitable for binary metrics. In degenerate folds:

- `auc` can fail if only one class is present in `y_test`
- `fpr` can become invalid when there are no negative examples

For stable public experiments, make sure the test target is not degenerate.

## `multiclass_metrics(data, preds, probs, average='macro')`

Computes multiclass classification metrics from:

- `data['y_test']`
- predicted labels
- class probabilities

Returns:

- `precision`
- `recall`
- `auc`
- `accuracy`

This helper uses `safe_ovr_auc()` instead of calling raw multiclass AUC directly.

## `continuous_metrics(data, preds)`

Computes the current regression metrics from:

- `data['y_test']`
- continuous predictions `preds`

Returns:

- `bias`
- `mae`
- `rmse`
- `r2`
- `mape`

`mape` is reported in percent units.

## `balanced_metric(y_true, y_pred)`

`balanced_metric()` is Limen's compact binary score for cases where class balance matters.

Current formula:

```text
precision * sqrt(trade_rate)
```

This rewards accurate positive calls while penalizing degenerate behavior such as never trading.

If there are no positive predictions, it returns `0.0`.

Example:

```python
score = balanced_metric(y_true, y_pred)
```

## `safe_ovr_auc(y_true, proba)`

`safe_ovr_auc()` computes one-vs-rest AUC more defensively than a raw direct multiclass AUC call.

Its purpose is to make multiclass evaluation more stable when not every class is present in every fold.

If no valid class-vs-rest comparisons can be made, it returns `NaN`.

## Where These Helpers Fit

A typical reference-architecture model function looks like:

```python
from limen.metrics.binary_metrics import binary_metrics

def model(data, ...):
    preds = ...
    probs = ...

    results = binary_metrics(data, preds, probs)
    results['_preds'] = preds
    return results
```

That is the level of abstraction these helpers are meant for.

## Read Next

- Continue to [Log](Log.md) for experiment-level analysis built on top of these low-level metrics.
- Continue to [Benchmark](Benchmark.md) for the benchmark layer that uses reconstructed round outputs rather than these raw helper functions directly.
- Continue to [Reference Architecture](Reference-Architecture.md) for the model layer that wraps these task metrics into richer evaluation payloads.
