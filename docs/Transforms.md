# Transforms

Transforms are stateless preprocessing operations applied to datasets prior to modeling or exploration. They compute statistics and apply transformations in a single step. They scale, clip, filter, or normalize numeric columns while preserving the core semantics of the data.

For stateful scalers that fit on training data, see [Scalers](Scalers.md).

## `loop.transforms`

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
