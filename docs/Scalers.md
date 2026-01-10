# Scalers

Scalers are stateful preprocessing transformations that fit on training data and then transform any dataset using the learned parameters. They are used with the `.set_scaler()` method in manifests.

## Scaler Convention

All scaler classes must follow a standard structure for compatibility with the manifest system and post-processing pipelines.

### Required Structure

**`__init__` method:**
```python
def __init__(self, x_train: pl.DataFrame, **kwargs):
    """
    Fit the scaler on training data.

    Args:
        x_train: Training DataFrame to compute scaling parameters
        **kwargs: Optional configuration parameters
    """
    # Compute and store scaling parameters from x_train
    # Example: self.means = {...}, self.stds = {...}
```

**`transform` method:**
```python
def transform(self, df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply scaling transformation using fitted parameters.

    Args:
        df: DataFrame to transform

    Returns:
        Transformed DataFrame
    """
    # Apply transformation using stored parameters
```

**`inverse_transform` helper (optional):**
```python
def inverse_transform(df: pl.DataFrame, scaler: YourScaler) -> pl.DataFrame:
    """
    Reverse the scaling transformation.

    Args:
        df: Scaled DataFrame to inverse transform
        scaler: Fitted scaler instance with parameters

    Returns:
        DataFrame in original scale
    """
    # Reverse the transformation for post-processing
```

### Why This Convention?

- **Manifest compatibility**: The `.set_scaler()` method expects this interface
- **Stateful operation**: `__init__` fits on training data, `transform` applies to any data
- **Post-processing**: `inverse_transform` enables converting predictions back to original scale
- **Consistency**: All scalers work the same way, making them interchangeable

## `limen.scalers`

### `LogRegScaler`

LogRegScaler class for scaling and inverse scaling data.

#### Args

| Parameter | Type           | Description       |
|-----------|----------------|-------------------|
| `x_train` | `pl.DataFrame` | The training data |

#### Methods
- `transform(df: pl.DataFrame) -> pl.DataFrame`: Transform the data using the scaling rules

#### Helper Functions
- `inverse_transform(df: pl.DataFrame, scaler: LogRegScaler) -> pl.DataFrame`: Inverse transform the data back to original scale

#### Example

```python
from limen.scalers import LogRegScaler
from limen.scalers.logreg_scaler import inverse_transform

# In manifest
manifest.set_scaler(LogRegScaler)

# For post-processing (inverse transform)
original_scale_df = inverse_transform(scaled_df, fitted_scaler)
```

### `LinearScaler`

Linear transformation utility for scaling features using configurable rules.

#### Args

| Parameter | Type                      | Description           |
|-----------|---------------------------|-----------------------|
| `x_train` | `pl.DataFrame`            | Training DataFrame    |
| `rules`   | `dict[str, str] \| None` | Regex-to-rule mapping |
| `default` | `str`                     | Fallback scaling rule |

#### Methods
- `transform(df: pl.DataFrame) -> pl.DataFrame`: Apply linear scaling transformation

#### Helper Functions
- `inverse_transform(df: pl.DataFrame, scaler: LinearScaler) -> pl.DataFrame`: Apply inverse scaling transformation

#### Example

```python
from limen.scalers import LinearScaler
from limen.scalers.linear_scaler import inverse_transform

# In manifest
manifest.set_scaler(LinearScaler)

# For post-processing (inverse transform)
original_scale_df = inverse_transform(scaled_df, fitted_scaler)
```
