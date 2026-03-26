# Transforms

Transforms in Limen are lightweight helpers used during data preparation or model post-processing. Most operate directly on DataFrames, while a smaller subset helps with classifier calibration and threshold selection.

For stateful scalers that fit on training data, see [Scalers](Scalers.md).

## `limen.transforms`

### `mad_transform`

Compute Median Absolute Deviation (MAD) Transform.

#### Args

| Parameter  | Type           | Description                 |
|------------|----------------|-----------------------------|
| `df`       | `pl.DataFrame` | The input DataFrame         |
| `time_col` | `str`          | The name of the time column |

#### Returns

`pl.DataFrame`: The transformed DataFrame

### `winsorize_transform`

Compute winsorization by clipping numeric columns to fixed quantile bounds.

#### Args

| Parameter  | Type           | Description                                   |
|------------|----------------|-----------------------------------------------|
| `df`       | `pl.DataFrame` | Klines dataset with numeric columns to clip   |
| `time_col` | `str`          | Column name to exclude from numeric transforms |

#### Returns

`pl.DataFrame`: The input data with winsorized numeric columns

### `quantile_trim_transform`

Compute outlier trimming by removing rows outside fixed quantile bounds across numeric columns.

#### Args

| Parameter  | Type           | Description                                   |
|------------|----------------|-----------------------------------------------|
| `df`       | `pl.DataFrame` | Klines dataset with numeric columns to trim   |
| `time_col` | `str`          | Column name to exclude from numeric transforms |

#### Returns

`pl.DataFrame`: The input data filtered within bounds for all numeric columns

### `zscore_transform`

Compute standard Z-score scaling for numeric columns.

#### Args

| Parameter  | Type           | Description                                   |
|------------|----------------|-----------------------------------------------|
| `df`       | `pl.DataFrame` | Klines dataset with numeric columns to scale  |
| `time_col` | `str`          | Column name to exclude from numeric transforms |

#### Returns

`pl.DataFrame`: The input data with Z-scored numeric columns

### `shift_column_transform`

Shift a column by a specified number of periods.

#### Args

| Parameter  | Type           | Description                                   |
|------------|----------------|-----------------------------------------------|
| `data`     | `pl.DataFrame` | Input DataFrame                               |
| `shift`    | `int`          | Number of periods to shift (negative for forward shift) |
| `column`   | `str`          | Name of column to shift                       |

#### Returns

`pl.DataFrame`: DataFrame with shifted column

### `calibrate_classifier`

Apply probability calibration to a fitted classifier and return calibrated probability arrays.

#### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `clf` | `Any` | Fitted classifier with `predict_proba` |
| `x_val` | `np.ndarray` | Validation features used for calibration fitting |
| `y_val` | `np.ndarray` | Validation labels used for calibration fitting |
| `x_sets` | `list` | Feature arrays to score after calibration |
| `method` | `str` | Calibration method, typically `isotonic` or `sigmoid` |

#### Returns

`tuple`: Calibrated positive-class probability arrays for each entry in `x_sets`

### `optimize_binary_threshold`

Sweep validation thresholds and return the best binary decision threshold.

#### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_val` | `np.ndarray` | Validation labels |
| `y_val_proba` | `np.ndarray` | Validation probabilities for the positive class |
| `threshold_min` | `float` | Minimum threshold to test |
| `threshold_max` | `float` | Maximum threshold to test |
| `threshold_step` | `float` | Threshold step size |
| `default_threshold` | `float` | Fallback threshold |
| `metric` | `str` | Metric to optimize, such as `balanced`, `f1`, `precision`, or `accuracy` |

#### Returns

`tuple`: `(best_threshold, best_score)`
