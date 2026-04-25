# Reference Architecture

The reference architecture is Limen's class-based model layer. It sits underneath the foundational SFDs and underneath `Trainer`.

This is the page to read when you want to understand:

- what a `ReferenceModel` must implement
- how the built-in model classes behave
- what `evaluate(..., inline_metrics=True)` really adds
- how class-based models relate to the simpler function wrappers used in manifests

## Public Surface

The current public reference-architecture exports are:

- `ReferenceModel`
- `LogRegBinary`
- `RandomBinary`
- `XGBoostRegressor`
- `TabPFNBinary` when `tabpfn` is installed

Each model module also exposes a function-style wrapper with the same behavioral surface used by foundational manifests.

## `ReferenceModel`

`ReferenceModel` is the base contract. Every subclass must implement:

```python
train(data, **params)
predict(data)
evaluate(data, inline_metrics=True)
```

### Data expectations

In a live local reference-architecture run in this repo, the prepared `data_dict` included:

- `x_train`, `y_train`
- `x_val`, `y_val`
- `x_test`, `y_test`
- `price_data_for_backtest`
- `_feature_names`
- `_alignment`
- `_scaler`

Not every model needs every key, but this is the standard Limen shape that the class-based models are designed around.

## The Built-In Model Classes

| Class | Task shape | Deterministic | Notes |
|---|---|---|---|
| `LogRegBinary` | binary classification | yes | sklearn logistic regression wrapper |
| `RandomBinary` | binary baseline | no | intentionally stochastic |
| `XGBoostRegressor` | regression | no | requires `xgboost` |
| `TabPFNBinary` | binary classification | no | optional, requires `tabpfn` |

The `deterministic` flag matters because [Trainer](Trainer.md) uses it to choose its validation tolerance.

## Probability Support for Cohort

For Cohort, “probability” always means **P(1)**: the probability that the positive class is `1`.

Architectures that expose valid P(1) may use Cohort's probability-weighted aggregation path. Architectures that do not expose valid P(1) use Cohort's majority-vote fallback path instead.

| Architecture | Returns probabilities P(1) | Cohort mode | Notes |
|---|---:|---|---|
| `LogRegBinary` | yes | probability | `predict()` returns `_probs = predict_proba(... )[:, 1]`, which is directly the class-1 probability P(1). |
| `RandomBinary` | yes | probability | `predict()` returns `_probs`, but they are synthetic confidence values (`0.9` for predicted 1, `0.1` for predicted 0), not model-derived calibrated probabilities. Still usable as P(1)-shaped output if Cohort accepts implementation-defined probability-like outputs. |
| `TabPFNBinary` | yes | probability | `predict()` returns `_probs` as positive-class probability, optionally calibrated with isotonic calibration before thresholding. This is compatible with P(1). |
| `XGBoostRegressor` | no | fallback | `predict()` returns only `_preds` and does not expose `_probs`. Since this is a regressor, any Cohort use would have to fall back unless a separate binary-probability wrapper is introduced. |

## `predict()` Versus `evaluate()`

`predict()` is the small inference surface. For the built-in binary models it returns:

- `_preds`
- `_probs`

`evaluate()` is the richer offline evaluation surface.

### `inline_metrics=False`

With `inline_metrics=False`, `evaluate()` returns the task metrics only.

On a live local `LogRegBinary` evaluation in this repo, that plain result included:

- `accuracy`
- `auc`
- `precision`
- `recall`
- `fpr`

### `inline_metrics=True`

With `inline_metrics=True`, `evaluate()` adds:

- `confusion_*` metrics
- `backtest_*` metrics when `price_data_for_backtest` is present

On that same live local run, `LogRegBinary.evaluate(..., inline_metrics=True)` added keys such as:

- `backtest_total_return_net_pct`
- `backtest_max_drawdown_pct`
- `backtest_sharpe_per_bar`
- `confusion_tp`
- `confusion_fp`
- `confusion_precision`

That is why the reference-architecture layer is the bridge between raw model output and the experiment-level analytics surfaces.

## Example: Class-Based Usage

```python
from limen.sfd.reference_architecture import LogRegBinary

model = LogRegBinary().train(
    data,
    solver='lbfgs',
    penalty='l2',
    C=0.1,
    class_weight=0.55,
    max_iter=60,
)

pred = model.predict({'x_test': data['x_test']})
results = model.evaluate(data, inline_metrics=True)
```

This is the same contract that `Trainer` eventually relies on when it promotes finished experiment rounds into `Sensor` objects.

## Function Wrappers Versus Classes

Most foundational manifests call the function wrapper:

```python
.with_model(logreg_binary)
```

That wrapper typically:

1. instantiates the matching class
2. trains it
3. evaluates it with `inline_metrics=True`

The class is the canonical reusable architecture surface. The function wrapper is the convenient manifest-facing adapter.

## Trainer Relationship

`Trainer` resolves the `ReferenceModel` subclass from the model module used by the original manifest.

That is why the class-based layer matters even if your day-to-day work mostly touches foundational SFDs:

- foundational SFDs package the experiment
- reference architecture owns the model contract
- `Trainer` promotes selected rounds back into trained class-based models

On a live local logreg trainer run in this repo:

- deterministic validation passed with no mismatches
- `Sensor.predict()` returned `_preds` and `_probs`
- the promoted sensor produced predictions for `884` test bars

On a live local `random_binary` trainer run, promotion raised `ReconstructionError` because the stochastic rerun did not reproduce the original logged metrics closely enough.

## Optional Dependencies

- `xgboost_regressor` requires `xgboost`
- `tabpfn_binary` requires `tabpfn`

In a live local smoke pass in this repo:

- `logreg_binary`, `random_binary`, and `xgboost_regressor` all ran
- `tabpfn_binary` was unavailable because `tabpfn` was not installed

## Read Next

- Continue to [Built-In SFDs](Built-In-SFDs.md) to see how the shipped foundational SFDs package these model surfaces.
- Continue to [Trainer](Trainer.md) for the promotion workflow that reconstructs and retrains selected rounds.
- Continue to [Standard Metrics Library](Standard-Metrics-Library.md) for the low-level metric helpers used inside these model classes.
