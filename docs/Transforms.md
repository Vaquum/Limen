# Transforms

Transforms in Limen are lightweight helpers used either during data preparation or immediately after model scoring. They are not the same thing as train-fitted scalers.

Use this page when you need to shape a target, clip or normalize a frame in a stateless way, or calibrate a classifier after fitting. For train-only fitted preprocessing, see [Scalers](Scalers.md).

## Two Transform Families

| Family | What it does | Typical place in the pipeline |
|---|---|---|
| DataFrame transforms | Modify or filter columns in a `pl.DataFrame` | target building, small preprocessing steps, or split-local shaping |
| Post-model helpers | Work on fitted classifiers or probability arrays | calibration and threshold selection after model training |

## DataFrame Transforms

These helpers operate on the frame passed into them. They do not carry learned state across splits. If you call them separately on train, validation, and test, each call uses the statistics of the frame it received.

| Function | Behavior | Notes |
|---|---|---|
| `mad_transform(df, time_col='datetime')` | rescales numeric columns by median absolute deviation | Leaves the time column untouched. |
| `winsorize_transform(df, time_col='datetime')` | clips numeric columns to fixed 1% and 99% quantiles | Good when you want to tame outliers without dropping rows. |
| `quantile_trim_transform(df, time_col='datetime')` | removes rows outside fixed 0.5% and 99.5% bounds across numeric columns | More aggressive than winsorization because rows can disappear. |
| `zscore_transform(df, time_col='datetime')` | standardizes numeric columns to mean zero and unit variance | Stateless per call, unlike a train-fitted scaler. |
| `shift_column_transform(data, shift, column)` | shifts one column in place | Common in target construction. Negative values shift forward in time. |

## Post-Model Helpers

These helpers are used after a model has already been fitted.

| Function | Returns | Notes |
|---|---|---|
| `calibrate_classifier(clf, x_val, y_val, x_sets, method='isotonic')` | a tuple of calibrated positive-class probability arrays | Uses `CalibratedClassifierCV` with `cv='prefit'`. `method` is usually `isotonic` or `sigmoid`. |
| `optimize_binary_threshold(y_val, y_val_proba, ...)` | `(best_threshold, best_score)` | Sweeps thresholds on validation probabilities and optimizes `balanced`, `f1`, `precision`, or `accuracy`. |

## Target-Building Example

```python
from limen.targets import QuantileBinaryTarget

manifest.with_target_label(
    'quantile_flag',
    QuantileBinaryTarget,
    fit_params={'source_column': 'roc_{roc_period}', 'quantile': 'q'},
    transform_params={'shift': 'shift'},
)
```

The important detail is that `QuantileBinaryTarget.__init__` computes the cutoff on the training split only; `transform()` reuses the stored cutoff on validation and test without refitting.

## Calibration Example

```python
from limen.transforms import calibrate_classifier, optimize_binary_threshold

val_proba_cal, test_proba_cal = calibrate_classifier(
    clf,
    x_val=x_val,
    y_val=y_val,
    x_sets=[x_val, x_test],
    method='isotonic',
)

best_threshold, best_score = optimize_binary_threshold(
    y_val=y_val,
    y_val_proba=val_proba_cal,
    metric='balanced',
)
```

## Boundaries

- Use a transform when the operation is lightweight and local to the frame or prediction arrays you already have.
- Use a scaler when the operation must be fitted on train and then reused unchanged on validation and test.
- If you need split-safe learned parameters inside a target, compute them through the manifest target builder rather than hiding the fitting inside the transform itself.

## Read Next

- [Scalers](Scalers.md) for train-fitted preprocessing
- [Features](Features.md) for target and regime helpers that often pair with transforms
- [Experiment Manifest](Experiment-Manifest.md) for where transforms live in the split-first execution order
