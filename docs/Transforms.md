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

## Function Reference

### `mad_transform(df, *, time_col='datetime')`

Scales every numeric column except `time_col` as `(value - median) / mad`, where `mad` is the column median absolute deviation inside the frame passed to the function.

Return behavior:

- returns a `pl.DataFrame`
- preserves nonnumeric columns unchanged
- moves nonnumeric/context columns before transformed numeric columns in the returned column order
- uses `1.0` when MAD is zero, so constant numeric columns become `0.0`
- raises `TypeError` for an all-null transformed numeric column because the current implementation converts null medians or MADs to floats

### `winsorize_transform(df, *, time_col='datetime')`

Clips every numeric column except `time_col` to that column's 1st and 99th percentile inside the current frame.

Return behavior:

- returns a `pl.DataFrame` with the same row count
- preserves nonnumeric columns unchanged
- returns the input frame unchanged when there are no numeric columns other than `time_col`
- uses current-frame quantiles, not train-fitted bounds
- raises `TypeError` for an all-null transformed numeric column because the current implementation converts null quantile bounds to floats

### `quantile_trim_transform(df, *, time_col='datetime')`

Filters rows whose numeric values fall outside the 0.5th and 99.5th percentile bounds of any transformed numeric column.

Return behavior:

- returns a `pl.DataFrame` with row count less than or equal to the input row count
- preserves rows where the tested numeric value is null when the column still has computable quantile bounds
- raises `TypeError` for an all-null transformed numeric column because the current implementation converts null quantile bounds to floats
- returns the input frame unchanged when there are no numeric columns other than `time_col`
- applies an AND mask across all transformed numeric columns

### `zscore_transform(df, *, time_col='datetime')`

Scales every numeric column except `time_col` as `(value - mean) / std` using current-frame statistics.

Return behavior:

- returns a `pl.DataFrame`
- preserves nonnumeric columns unchanged
- uses `1.0` when standard deviation is zero, so multi-row constant numeric columns become `0.0`
- raises `TypeError` when Polars returns a null standard deviation, including one-row numeric columns and all-null transformed numeric columns
- returns the input frame unchanged when there are no numeric columns other than `time_col`

### `shift_column_transform(data, shift, column)`

Replaces `column` with `pl.col(column).shift(shift)`.

Return behavior:

- positive `shift` moves previous values later
- negative `shift` moves future values earlier
- nulls are introduced at the shifted boundary
- the target column must exist in `data`

## Manifest Usage

Transforms usually appear in manifest target construction or as callable helpers inside a custom SFD prep path. The important split-safety rule is that stateless transforms do not remember train statistics. If validation and test must use train-fitted parameters, use [Scalers](Scalers.md) or a target class that fits on train in `__init__`.

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
- Use `quantile_trim_transform` only when row removal is acceptable for the downstream split. If preserving row alignment matters, prefer `winsorize_transform`.
- Keep `time_col` explicit when your time column is not named `datetime`; otherwise it will be transformed like any other numeric column.

## Read Next

- [Scalers](Scalers.md) for train-fitted preprocessing
- [Features](Features.md) for target and regime helpers that often pair with transforms
- [Experiment Manifest](Experiment-Manifest.md) for where transforms live in the split-first execution order
