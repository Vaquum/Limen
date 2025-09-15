# Transforms

Transforms are preprocessing operations applied to datasets prior to modeling or exploration. They scale, clip, filter, or normalize numeric columns while preserving the core semantics of the data.

## `loop.transforms`

### `LogRegTransform`

LogRegTransform class for scaling and inverse scaling data.

#### Args

| Parameter | Type           | Description       |
|-----------|----------------|-------------------|
| `x_train` | `pl.DataFrame` | The training data |

#### Methods
- `transform(df: pl.DataFrame) -> pl.DataFrame`: Transform the data using the scaling rules
- `logreg_inverse_transform(df: pl.DataFrame, scaler: LogRegTransform) -> pl.DataFrame`: Inverse transform the data using the scaling rules

### `logreg_inverse_transform`

Inverse transform the data using the scaling rules.

#### Args

| Parameter | Type              | Description         |
|-----------|-------------------|---------------------|
| `df`      | `pl.DataFrame`    | The input DataFrame |
| `scaler`  | `LogRegTransform` | The scaler object   |

#### Returns

`pl.DataFrame`: The inverse transformed DataFrame

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

### `LinearTransform`

Linear transformation utility for scaling features.

#### Args

| Parameter | Type                      | Description           |
|-----------|---------------------------|-----------------------|
| `x_train` | `pl.DataFrame`            | Training DataFrame    |
| `rules`   | `dict[str, str] | None` | Regex-to-rule mapping |
| `default` | `str`                     | Fallback scaling rule |

#### Methods
- `transform(df: pl.DataFrame) -> pl.DataFrame`: Apply linear scaling transformation
- `inverse_transform(df: pl.DataFrame, scaler: LinearTransform) -> pl.DataFrame`: Apply inverse scaling transformation
