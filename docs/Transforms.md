# Transforms

Transforms in Limen are lightweight helpers used during data preparation or target construction. They are not the same thing as train-fitted scalers. For probability calibration and threshold optimization, see `limen.calibration`.

Use this page when you need to shape a target, clip or normalize a frame in a stateless way. For train-only fitted preprocessing, see [Scalers](Scalers.md).

## DataFrame Transforms

These helpers operate on the frame passed into them. They do not carry learned state across splits. If you call them separately on train, validation, and test, each call uses the statistics of the frame it received.

| Function | Behavior | Notes |
|---|---|---|
| `mad_transform(df, time_col='datetime')` | rescales numeric columns by median absolute deviation | Leaves the time column untouched. |
| `winsorize_transform(df, time_col='datetime')` | clips numeric columns to fixed 1% and 99% quantiles | Good when you want to tame outliers without dropping rows. |
| `quantile_trim_transform(df, time_col='datetime')` | removes rows outside fixed 0.5% and 99.5% bounds across numeric columns | More aggressive than winsorization because rows can disappear. |
| `zscore_transform(df, time_col='datetime')` | standardizes numeric columns to mean zero and unit variance | Stateless per call, unlike a train-fitted scaler. |
| `shift_column_transform(data, shift, column)` | shifts one column in place | Common in target construction. Negative values shift forward in time. |

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

## Boundaries

- Use a transform when the operation is lightweight and local to the frame or prediction arrays you already have.
- Use a scaler when the operation must be fitted on train and then reused unchanged on validation and test.
- If you need split-safe learned parameters inside a target, compute them through the manifest target builder rather than hiding the fitting inside the transform itself.
- For probability calibration and threshold selection after model training, use `limen.calibration` and the manifest's `with_calibration()` builder.

## Read Next

- [Scalers](Scalers.md) for train-fitted preprocessing
- [Features](Features.md) for target and regime helpers that often pair with transforms
- [Experiment Manifest](Experiment-Manifest.md) for where transforms live in the split-first execution order
