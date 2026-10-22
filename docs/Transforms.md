# Transforms

Transforms in Limen are stateless helpers used during data preparation or target construction. They are not the same thing as train-fitted scalers. For probability calibration and threshold optimization, see `limen.calibration`.

This page covers stateless target shaping, clipping, and frame normalization. For train-only fitted preprocessing, see [Scalers](Scalers.md).

## DataFrame transforms

These helpers operate on the frame passed into them. They do not carry learned state across splits. Separate calls on train, validation, and test use the statistics of the frame passed to that call.

| Function | Behavior | Notes |
|---|---|---|
| `mad_transform(df, time_col='datetime')` | rescales numeric columns by median absolute deviation | Leaves the time column untouched. |
| `winsorize_transform(df, time_col='datetime')` | clips numeric columns to fixed 1% and 99% quantiles | Preserves rows while capping outliers. |
| `quantile_trim_transform(df, time_col='datetime')` | removes rows outside fixed 0.5% and 99.5% bounds across numeric columns | More aggressive than winsorization because rows can disappear. |
| `zscore_transform(df, time_col='datetime')` | standardizes numeric columns to mean zero and unit variance | Stateless per call, unlike a train-fitted scaler. |
| `shift_column_transform(data, shift, column)` | shifts one column in place | Common in target construction. Negative values shift forward in time. |

## Function reference

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

## Manifest usage

Transforms appear in manifest target construction or as callable helpers inside a custom SFD prep path. The important split-safety rule is that stateless transforms do not remember train statistics. If validation and test must use train-fitted parameters, use [Scalers](Scalers.md) or a target class that fits on train in `__init__`.

## Target construction in the projects

The [Python project](examples/python-research/README.md) computes next-bar return separately inside each split. The forward shift constructs a label; it never becomes an inference feature. The final row has no next-bar outcome and is dropped from fitting or validation.

<!-- example: transforms-python-label source=docs/examples/python-research/research.py region=next_return_target scenario=selection interface=python role=executed -->
```python
def prepare_split(bars, round_params):
    return features(bars, round_params["window"]).with_columns(
        (pl.col("close").shift(-1) / pl.col("open").shift(-1) - 1)
        .alias("next_return")
    ).slice(12).drop_nulls(["momentum", "next_return"])
```

The [Compiled project](examples/compiled-research/README.md) asks `QuantileBinaryTarget` to fit a cutoff on training momentum and reuse it on later splits. Its negative shift assigns the next observation's flag to the current training row. This is a different outcome from next-bar return.

<!-- example: transforms-compiled-label source=docs/examples/compiled-research/selection.yaml region=fitted_target scenario=selection interface=compiled role=executed -->
```yaml
    target:
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params:
        source_column: roc_1
        quantile: 0.5
      transform_params:
        shift: -1
```

A stateless shift and a fitted target have different responsibilities. The shift moves observations; it does not decide which data may fit the cutoff. The Manifest's split-first preparation owns that boundary. The [target reference](Targets.md) describes each fitted target and its input requirements.

## Boundaries

- Use a transform when the operation is stateless and local to the frame or prediction arrays.
- Use a scaler when the operation must be fitted on train and then reused unchanged on validation and test.
- Compute split-safe learned parameters through the manifest target builder rather than hiding the fitting inside the transform itself.
- For probability calibration and threshold selection after model training, use `limen.calibration` and the manifest's `with_calibration()` builder.
- Use `quantile_trim_transform` only when row removal is acceptable for the downstream split. If preserving row alignment matters, prefer `winsorize_transform`.
- Keep `time_col` explicit when the time column is not named `datetime`; otherwise it will be transformed like any other numeric column.

## Read next

- [Scalers](Scalers.md) for train-fitted preprocessing
- [Features](Features.md) for target and regime helpers paired with transforms
- [Experiment Manifest](Experiment-Manifest.md) for where transforms live in the split-first execution order

## Callable contracts

These entries give the exact public signatures and the source-owned argument, return and failure contracts. Use the project examples above for the research context; a signature is reference evidence, not a claim that every option ran in the base experiment.

### `limen.transforms.mad_transform`

<!-- reference: api-limen-transforms-mad-transform source=limen/transforms/mad_transform.py region=signature:mad_transform role=signature -->
```python
def mad_transform(df: pl.DataFrame, *, time_col: str='datetime'):
    ...
```

Compute Median Absolute Deviation (MAD) Transform.

#### Args

df (pl.DataFrame): The input DataFrame.

time_col (str): The name of the time column.

#### Returns

pl.DataFrame: The transformed DataFrame.

### `limen.transforms.quantile_trim_transform`

<!-- reference: api-limen-transforms-quantile-trim-transform source=limen/transforms/quantile_trim_transform.py region=signature:quantile_trim_transform role=signature -->
```python
def quantile_trim_transform(df: pl.DataFrame, *, time_col: str='datetime') -> pl.DataFrame:
    ...
```

Compute outlier trimming by removing rows outside fixed quantile bounds across numeric columns.

#### Args

df (pl.DataFrame): Klines dataset with numeric columns to trim

time_col (str): Column name to exclude from numeric transforms

#### Returns

pl.DataFrame: The input data filtered within bounds for all numeric columns

### `limen.transforms.shift_column_transform`

<!-- reference: api-limen-transforms-shift-column-transform source=limen/transforms/shift_column_transform.py region=signature:shift_column_transform role=signature -->
```python
def shift_column_transform(data: pl.DataFrame, shift: int, column: str) -> pl.DataFrame:
    ...
```

Shift a column by a specified number of periods.

#### Args

data (pl.DataFrame): Input DataFrame

shift (int): Number of periods to shift (negative for forward shift)

column (str): Name of column to shift

#### Returns

pl.DataFrame: DataFrame with shifted column

### `limen.transforms.winsorize_transform`

<!-- reference: api-limen-transforms-winsorize-transform source=limen/transforms/winsorize_transform.py region=signature:winsorize_transform role=signature -->
```python
def winsorize_transform(df: pl.DataFrame, *, time_col: str='datetime') -> pl.DataFrame:
    ...
```

Compute winsorization by clipping numeric columns to fixed quantile bounds.

#### Args

df (pl.DataFrame): Klines dataset with numeric columns to clip

time_col (str): Column name to exclude from numeric transforms

#### Returns

pl.DataFrame: The input data with winsorized numeric columns

### `limen.transforms.zscore_transform`

<!-- reference: api-limen-transforms-zscore-transform source=limen/transforms/zscore_transform.py region=signature:zscore_transform role=signature -->
```python
def zscore_transform(df: pl.DataFrame, *, time_col: str='datetime') -> pl.DataFrame:
    ...
```

Compute standard Z-score scaling for numeric columns.

#### Args

df (pl.DataFrame): Klines dataset with numeric columns to scale

time_col (str): Column name to exclude from numeric transforms

#### Returns

pl.DataFrame: The input data with Z-scored numeric columns

