# Experiment Manifest

## Introduction

The Experiment Manifest provides a declarative system for configuring Loop experiment pipelines. Instead of manually implementing data preparation and model functions, you define your experiment pipeline through a fluent manifest API that handles data fetching, feature engineering, target preparation, and model configuration.

### Key Benefits

- **Automatic data fetching**: Configure data sources once, UEL fetches automatically based on environment.
- **Reproducible experiments**: Entire pipeline configuration in one place.
- **Universal Split-First architecture**: Data splits before processing, preventing data leakage.
- **Auto-generated SFM functions**: Manifest generates much of the boilerplate code in the SFM.
- **Declarative approach**: Clear, readable pipeline definition.

### Universal Split-First Architecture

The manifest enforces a split-first processing pattern:

1. **Split Phase**: Raw data divided into train/validation/test splits
2. **Bar Formation Phase**: Each split processes bars independently (if configured)
3. **Feature Engineering Phase**: Indicators and features computed per split
4. **Target Transformation Phase**: Targets computed with fitted parameters (train) or applied parameters (val/test)
5. **Scaling Phase**: Data scaled using fitted scalers (train) or applied scalers (val/test)

This architecture ensures no data leakage between splits and maintains reproducible results.

### Philosophy & Trade-offs

The manifest approach is **deliberately opinionated** to enforce best practices in financial ML:

**Strong Opinions (Enforced):**
- **Split-first architecture**: Prevents data leakage (cannot fit on full dataset)
- **Polars LazyFrame**: Enforces lazy evaluation for performance (learning curve)
- **Immutability**: Functions return new data, don't modify input (functional programming style)
- **Training-only fitting**: Parameters fitted only on training data (proper ML methodology)

**Benefits:**
- Prevents critical mistakes (data leakage, improper fitting)
- Reproducible experiments (declarative configuration)
- Consistent codebase (standardized patterns)
- Performance optimizations (lazy evaluation)

**Trade-offs:**
- Learning curve (Polars API, manifest patterns)
- Less flexible (harder for non-standard workflows)

**When to use:** Production SFMs, reproducibility-critical work, collaborative projects

**When to skip:** complex data workflows, external library integration (use Legacy SFMs)

This is appropriate for financial trading systems where **correctness and reproducibility are paramount**. The opinions are deliberate guardrails, not arbitrary restrictions.

## Quick Start

Here's a complete minimal manifest-based SFM:

```python
from loop.manifest import Manifest
from loop.historical_data import HistoricalData
from loop.tests.utils.get_data import get_klines_data_fast
from loop.indicators import roc
from loop.features import quantile_flag, compute_quantile_cutoff
from loop.utils import shift_column
from loop.transforms.linear_transform import LogregTransform
from loop.sfm.model import logreg_binary

def params():
    return {
        'roc_period': [4, 8, 12],
        'q': [0.32, 0.35, 0.37],
        'shift': [-1, -2, -3],
        'alpha': [1.0, 5.0, 10.0],
        'use_calibration': [True, False],
    }

def manifest():
    return (Manifest()
        # Data sources
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(method=get_klines_data_fast)

        # Split configuration
        .set_split_config(7, 1, 2)

        # Feature engineering
        .add_indicator(roc, period='roc_period')

        # Target configuration
        .with_target('quantile_flag')
            .add_fitted_transform(quantile_flag)
                .fit_param('_cutoff', compute_quantile_cutoff,
                          col='roc_{roc_period}', q='q')
                .with_params(col='roc_{roc_period}', cutoff='_cutoff')
            .add_transform(shift_column, shift='shift', column='target_column')
            .done()

        # Scaling and model
        .set_scaler(LogregTransform)
        .with_model(logreg_binary)
    )
```

**Usage with UEL:**

```python
import loop
from loop import sfm

# Data is automatically fetched from manifest-configured sources
uel = loop.UniversalExperimentLoop(single_file_model=sfm.reference.logreg)

uel.run(experiment_name='my_experiment', n_permutations=100)
```

## Data Source Configuration

Configure where UEL fetches data for training and testing.

### `.set_data_source(method, params=None)`

Configure production data source (uses `loop.historical_data.HistoricalData`).

**Args:**

| Parameter | Type       | Description                                      |
|-----------|------------|--------------------------------------------------|
| `method`  | `Callable` | HistoricalData method reference (e.g., `HistoricalData.get_spot_klines`) |
| `params`  | `dict`     | Parameters to pass to the method                |

**Returns:** `Manifest` (self for chaining)

**Available methods:**
- `HistoricalData.get_spot_klines` - Fetch spot market kline data
- `HistoricalData.get_futures_klines` - Fetch futures market kline data
- See `loop.historical_data.HistoricalData` for all available methods

**Example:**

```python
from loop.historical_data import HistoricalData

.set_data_source(
    method=HistoricalData.get_spot_klines,
    params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
)
```

### `.set_test_data_source(method, params=None)`

Configure test data source (uses `loop.tests.utils.get_data`).

**Args:**

| Parameter | Type       | Description                                    |
|-----------|------------|------------------------------------------------|
| `method`  | `Callable` | Test utils function reference (e.g., `get_klines_data_fast`) |
| `params`  | `dict`     | Parameters to pass to the function            |

**Returns:** `Manifest` (self for chaining)

**Available methods:**
- `get_klines_data_fast` - Small dataset for quick tests
- `get_klines_data_large` - Larger dataset for comprehensive tests
- `get_klines_data_small_fast` - Very small dataset for rapid iteration

**Example:**

```python
from loop.tests.utils.get_data import get_klines_data_fast

.set_test_data_source(method=get_klines_data_fast)
```

### Environment-Based Data Fetching

UEL automatically selects the appropriate data source based on the `LOOP_ENV` environment variable:

- `LOOP_ENV='test'` (default): Uses test data source
- `LOOP_ENV='production'`: Uses production data source

This enables seamless switching between test and production environments without code changes.

## Data Pipeline Configuration

### `.set_split_config(train, val, test)`

Configure train/validation/test split ratios.

**Args:**

| Parameter | Type  | Description              |
|-----------|-------|--------------------------|
| `train`   | `int` | Training split ratio     |
| `val`     | `int` | Validation split ratio   |
| `test`    | `int` | Test split ratio         |

**Returns:** `Manifest` (self for chaining)

**Note:** Ratios are relative (e.g., `7, 1, 2` means 7/10 train, 1/10 val, 2/10 test).

**Example:**

```python
.set_split_config(7, 1, 2)  # 70% train, 10% val, 20% test
```

### `.set_pre_split_data_selector(func, **params)`

Configure data selection before splitting (e.g., random sampling for faster experiments).

**Args:**

| Parameter | Type       | Description                        |
|-----------|------------|------------------------------------|
| `func`    | `Callable` | Data selector function             |
| `**params`| `dict`     | Parameter mappings from round_params |

**Returns:** `Manifest` (self for chaining)

**Common use case:**

```python
from loop.utils.random_slice import random_slice

.set_pre_split_data_selector(
    random_slice,
    rows='random_slice_size',
    safe_range_low='random_slice_min_pct',
    safe_range_high='random_slice_max_pct',
    seed='random_seed'
)
```

Then in `params()`:

```python
def params():
    return {
        'random_slice_size': [10000],
        'random_slice_min_pct': [0.25],
        'random_slice_max_pct': [0.75],
        'random_seed': [42],
        # ... other params
    }
```

### `.set_bar_formation(func, **params)`

Configure optional bar formation (time-based, volume-based, etc.).

**Args:**

| Parameter | Type       | Description                     |
|-----------|------------|---------------------------------|
| `func`    | `Callable` | Bar formation function          |
| `**params`| `dict`     | Parameter mappings              |

**Returns:** `Manifest` (self for chaining)

**Note:** Bar formation is optional. If not configured, raw data is used directly.

### `.set_required_bar_columns(columns)`

Set required columns that must be present after bar formation (validation check).

**Args:**

| Parameter | Type         | Description              |
|-----------|--------------|--------------------------|
| `columns` | `List[str]`  | Required column names    |

**Returns:** `Manifest` (self for chaining)

**Example:**

```python
.set_required_bar_columns(['datetime', 'open', 'high', 'low', 'close', 'volume'])
```

## Feature Engineering

### `.add_indicator(func, **params)`

Add technical indicator to the pipeline. Indicators are computational functions that derive technical analysis metrics from price/volume data.

**Args:**

| Parameter | Type       | Description                        |
|-----------|------------|------------------------------------|
| `func`    | `Callable` | Indicator function                 |
| `**params`| `dict`     | Parameter mappings from round_params |

**Returns:** `Manifest` (self for chaining)

**Import from:** `loop.indicators`

**Available indicators (common ones):**

```python
from loop.indicators import (
    roc,              # Rate of Change
    wilder_rsi,       # Relative Strength Index
    ppo,              # Percentage Price Oscillator
    atr,              # Average True Range
    macd,             # MACD
    stochastic_oscillator,  # Stochastic Oscillator
    cci,              # Commodity Channel Index
    bollinger_bands,  # Bollinger Bands
    rolling_volatility,  # Rolling Volatility
    # ... and more in loop/indicators/
)
```

**Parameter mapping:**

Parameters are mapped from `round_params` using string references:

```python
.add_indicator(roc, period='roc_period')
# When round_params = {'roc_period': 12}, this calls: roc(data, period=12)
```

**Example:**

```python
# Single parameter
.add_indicator(wilder_rsi, period='rsi_period')

# Multiple parameters
.add_indicator(ppo,
    fast_period='ppo_fast',
    slow_period='ppo_slow',
    signal_period='ppo_signal'
)
```

### `.add_feature(func, **params)`

Add feature computation to the pipeline. Features are derived metrics that provide additional context or transformations beyond standard indicators.

**Args:**

| Parameter | Type       | Description                        |
|-----------|------------|------------------------------------|
| `func`    | `Callable` | Feature function                   |
| `**params`| `dict`     | Parameter mappings from round_params |

**Returns:** `Manifest` (self for chaining)

**Import from:** `loop.features`

**Available features (common ones):**

```python
from loop.features import (
    volume_regime,          # Volume regime classification
    ichimoku_cloud,         # Ichimoku Cloud
    trend_strength,         # Trend strength metrics
    close_position,         # Close position in range
    gap_high,               # Gap high detection
    price_range_position,   # Price position in range
    range_pct,              # Range percentage
    sma_crossover,          # SMA crossover signals
    vwap,                   # Volume-Weighted Average Price
    ema_breakout,           # EMA breakout detection
    kline_imbalance,        # Kline imbalance metrics
    # ... and more - check loop/features/__init__.py for full list
)
```

**Example:**

```python
# Simple feature (no parameters)
.add_feature(close_position)

# Feature with parameters
.add_feature(volume_regime, lookback='lookback')

# Complex feature
.add_feature(ichimoku_cloud,
    tenkan_period='tenkan_period',
    kijun_period='kijun_period',
    senkou_b_period='senkou_b_period',
    displacement='displacement'
)
```

### Custom Indicators and Features

You can define custom indicator or feature functions. They must follow this contract:

**Function Signature:**

```python
def custom_indicator(data: pl.LazyFrame, param1=default1, param2=default2, ...) -> pl.LazyFrame:
    """
    Custom indicator implementation.

    Args:
        data: Polars LazyFrame with input data
        param1: Parameter with default value
        param2: Another parameter with default value

    Returns:
        Polars LazyFrame with new columns added
    """
    # Implementation using lazy evaluation
    return data.with_columns([
        # ... your computations
    ])
```

**Requirements:**

1. **Input**: Must accept `pl.LazyFrame` as first argument
2. **Output**: Must return `pl.LazyFrame`
3. **Lazy evaluation**: Use LazyFrame operations (not DataFrame) for performance
4. **Immutability**: Return new LazyFrame, don't modify input
5. **Column addition**: Add new columns, don't remove existing ones (use `with_columns`)
6. **Default parameters**: Provide defaults for all parameters

**Example:**

```python
import polars as pl

def custom_momentum(data: pl.LazyFrame, window: int = 20, threshold: float = 0.02) -> pl.LazyFrame:
    """Calculate custom momentum indicator."""
    return data.with_columns([
        ((pl.col('close') - pl.col('close').shift(window)) / pl.col('close').shift(window))
        .alias(f'custom_momentum_{window}'),

        (pl.col(f'custom_momentum_{window}') > threshold)
        .cast(pl.Int8)
        .alias(f'momentum_signal_{window}')
    ])

# Use in manifest
.add_indicator(custom_momentum, window='momentum_window', threshold='momentum_threshold')
```

**In params():**

```python
def params():
    return {
        'momentum_window': [10, 20, 30],
        'momentum_threshold': [0.01, 0.02, 0.03],
        # ...
    }
```

## Target Configuration

Configure target variable transformations with optional fitted parameters.

### Understanding Fitted vs Simple Transforms

**Fitted transforms** learn parameters from training data and apply them to validation/test data. This prevents data leakage.

**Common use case:** Binary classification using a quantile threshold
- **Wrong**: Compute 75th percentile on full dataset. This results in data leakage!
- **Right**: Compute 75th percentile on training data only, apply same threshold to val/test data

**Example workflow:**
1. **Training set**: Compute quantile cutoff (e.g., 75th percentile = 0.032)
2. **Validation set**: Apply same cutoff (0.032) - don't recompute!
3. **Test set**: Apply same cutoff (0.032) - don't recompute!

**When to use each:**
- **`add_fitted_transform`**: Transforms requiring parameters computed from training data (quantiles, normalization bounds, etc.)
- **`add_transform`**: Simple transforms without learning (shifting, clipping, type casting)

### `.with_target(target_column)`

Begin target transformation configuration. Returns a `TargetBuilder` for chained operations.

**Args:**

| Parameter       | Type  | Description          |
|-----------------|-------|----------------------|
| `target_column` | `str` | Target column name   |

**Returns:** `TargetBuilder`

### `TargetBuilder.add_fitted_transform(func)`

Add transformation that learns parameters from training data (e.g., quantile cutoffs, normalization bounds).

**Args:**

| Parameter | Type       | Description           |
|-----------|------------|-----------------------|
| `func`    | `Callable` | Transform function    |

**Returns:** `FittedTransformBuilder`

**Workflow:**

```python
.with_target('my_target')
    .add_fitted_transform(transformation_function)
        .fit_param('param_name', compute_function, arg1='value1', ...)
        .with_params(func_arg1='param1', func_arg2='param2')
    # ... more transforms
    .done()
```

### `FittedTransformBuilder.fit_param(param_name, fit_func, **kwargs)`

Define parameter to be fitted on training data.

**Args:**

| Parameter    | Type       | Description                    |
|--------------|------------|--------------------------------|
| `param_name` | `str`      | Name of fitted parameter (use `_prefix` convention) |
| `fit_func`   | `Callable` | Function to compute parameter from training data |
| `**kwargs`   | `dict`     | Arguments for fit function (supports string interpolation) |

**Returns:** `FittedTransformBuilder` (self for chaining)

**How it works:**

This is **Step 1** of the two-step fitted transform process: defining what to compute FROM training data.

1. **On training split**: `fit_func(train_data, **kwargs)` is called to compute the parameter
2. **Parameter stored**: Result stored with `param_name` (e.g., `_cutoff = 0.032`)
3. **On val/test splits**: Same parameter value used (no recomputation!)

**String interpolation:**

Parameter names in `**kwargs` can reference `round_params` using `{param_name}` syntax:

```python
# In params()
def params():
    return {'roc_period': [4, 8, 12], 'q': [0.32, 0.35]}

# In manifest
.fit_param('_cutoff', compute_quantile_cutoff, col='roc_{roc_period}', q='q')

# When round_params = {'roc_period': 12, 'q': 0.35}:
# - col='roc_{roc_period}' → col='roc_12' (string interpolation!)
# - q='q' → q=0.35 (direct parameter mapping)
# - Calls: compute_quantile_cutoff(train_data, col='roc_12', q=0.35)
# - Returns: 0.032 (stored as _cutoff)
```

**Complete workflow example:**

```python
# Step 1: Define fitted parameter computation
.fit_param('_cutoff', compute_quantile_cutoff, col='roc_{roc_period}', q='q')

# Step 2: Pass to transform function (see .with_params() below)
.with_params(col='roc_{roc_period}', cutoff='_cutoff')

# What happens on each split:
# Training split:
#   1. compute_quantile_cutoff(train_data, col='roc_12', q=0.35) → returns 0.032
#   2. quantile_flag(train_data, col='roc_12', cutoff=0.032)
#
# Val/test splits:
#   1. quantile_flag(val_data, col='roc_12', cutoff=0.032)  # Uses same 0.032!
#   2. quantile_flag(test_data, col='roc_12', cutoff=0.032) # Uses same 0.032!
```

**Naming convention:**

Use `_prefix` for fitted parameter names (e.g., `_cutoff`, `_min`, `_max`) to distinguish them from regular parameters.

### `FittedTransformBuilder.with_params(**params)`

Set parameters for the transform function (may include fitted parameters).

**Args:**

| Parameter | Type   | Description          |
|-----------|--------|----------------------|
| `**params`| `dict` | Parameter mappings for the transform function |

**Returns:** `TargetBuilder`

**How it works:**

This is **Step 2** of the two-step fitted transform process: passing parameters TO the transform function.

- **Fitted parameters**: Reference parameters defined via `.fit_param()` (e.g., `cutoff='_cutoff'`)
- **Regular parameters**: Map directly from `round_params` (e.g., `col='roc_period'`)
- **String interpolation**: Supported for dynamic column names (e.g., `col='roc_{roc_period}'`)

**Parameter resolution order:**

1. Check if parameter value starts with `_` → use fitted parameter (e.g., `'_cutoff'` → 0.032)
2. Check if parameter value is in `round_params` → use that value
3. Use parameter value literally (for constants)

**Example:**

```python
# Step 1: Define what to compute
.fit_param('_cutoff', compute_quantile_cutoff, col='roc_{roc_period}', q='q')

# Step 2: Pass computed parameter to transform
.with_params(col='roc_{roc_period}', cutoff='_cutoff')
#          ↑ string interpolation     ↑ fitted parameter reference

# Resolution when round_params = {'roc_period': 12, 'q': 0.35}:
# - col='roc_{roc_period}' → col='roc_12' (interpolated)
# - cutoff='_cutoff' → cutoff=0.032 (fitted parameter from training data)
# Calls: quantile_flag(data, col='roc_12', cutoff=0.032)
```

### `TargetBuilder.add_transform(func, **params)`

Add simple transformation without parameter fitting.

**Args:**

| Parameter | Type       | Description           |
|-----------|------------|-----------------------|
| `func`    | `Callable` | Transform function    |
| `**params`| `dict`     | Parameter mappings    |

**Returns:** `TargetBuilder` (self for chaining)

**Example:**

```python
from loop.utils import shift_column

.add_transform(shift_column, shift='shift', column='target_column')
```

### `TargetBuilder.done()`

Complete target configuration and return to main manifest.

**Returns:** `Manifest`

### Complete Target Example

```python
from loop.features import quantile_flag, compute_quantile_cutoff
from loop.utils import shift_column

.with_target('quantile_flag')
    # Fitted transform: compute quantile cutoff on training data
    .add_fitted_transform(quantile_flag)
        .fit_param('_cutoff', compute_quantile_cutoff,
                  col='roc_{roc_period}', q='q')
        .with_params(col='roc_{roc_period}', cutoff='_cutoff')
    # Simple transform: shift target for prediction horizon
    .add_transform(shift_column, shift='shift', column='target_column')
    .done()
```

## Scaling

### `.set_scaler(transform_class, param_name='_scaler')`

Set scaler/transform class for data preprocessing. The scaler is fitted on training data and applied to val/test splits.

**Args:**

| Parameter         | Type   | Description                        |
|-------------------|--------|------------------------------------|
| `transform_class` | `type` | Transform class                    |
| `param_name`      | `str`  | Parameter name for fitted scaler (default: '_scaler') |

**Returns:** `Manifest` (self for chaining)

**Available scalers:**

```python
from loop.transforms.linear_transform import LinearTransform    # Min-max scaling
from loop.transforms.logreg_transform import LogRegTransform    # Logistic regression specific
```

**Example:**

```python
.set_scaler(LinearTransform)
```

## Data Dict Extension

### `.add_to_data_dict(func)`

Configure function to add custom entries to data_dict after standard preparation.

**Args:**

| Parameter | Type       | Description                                        |
|-----------|------------|----------------------------------------------------|
| `func`    | `Callable` | Extension function (see signature below)           |

**Returns:** `Manifest` (self for chaining)

**Function Signature:**

```python
def extend_data_dict(data_dict: dict,
                     split_data: dict,
                     round_params: dict,
                     fitted_params: dict) -> dict:
    """
    Extend data_dict with custom entries.

    Args:
        data_dict: Base data dict with x_train, y_train, etc.
        split_data: Full split DataFrames (train_df, val_df, test_df)
        round_params: Current round parameters
        fitted_params: Fitted parameters from training data

    Returns:
        Modified data_dict with additional entries
    """
    # Add custom entries
    data_dict['custom_key'] = custom_value
    return data_dict
```

**Use case:** When your model function needs additional data beyond the standard x_train, y_train, x_val, y_val, x_test, y_test.

**Example:**

```python
def add_metadata(data_dict, split_data, round_params, fitted_params):
    """Add datetime information to data_dict."""
    data_dict['train_dates'] = split_data['train_df']['datetime'].to_list()
    data_dict['test_dates'] = split_data['test_df']['datetime'].to_list()
    return data_dict

.add_to_data_dict(add_metadata)
```

## Model Configuration

### `.with_model(model_function)`

Configure model function for training and evaluation. Parameters are automatically mapped from round_params.

**Args:**

| Parameter        | Type       | Description                              |
|------------------|------------|------------------------------------------|
| `model_function` | `Callable` | Model function (see contract below)      |

**Returns:** `Manifest` (self for chaining)

**Import from:** `loop.sfm.model`

**Available model functions:**

```python
from loop.sfm.model import (
    ridge_binary,              # Ridge classifier with binary metrics
    logreg_binary,             # Logistic regression with binary metrics
    logreg_multiclass,         # Logistic regression with multiclass metrics
    ridge_regression,          # Ridge regression with continuous metrics
    lgb_tradeable_regression,  # LightGBM tradeable regression
    # ... and more in loop/sfm/model/
)
```

**Example:**

```python
.with_model(ridge_binary)
```

### Auto-Parameter Mapping

The manifest automatically maps parameters from `round_params` to the model function:

1. Inspects model function signature
2. For each parameter in model function:
   - If exists in `round_params`: use that value
   - Otherwise: use model function's default value
3. Passes resolved parameters to model function

**Example:**

```python
# Model function signature
def ridge_binary(data, alpha=1.0, use_calibration=True, ...):
    ...

# In params()
def params():
    return {
        'alpha': [1.0, 5.0, 10.0],
        'use_calibration': [True, False],
    }

# Manifest
.with_model(ridge_binary)
# Parameters 'alpha' and 'use_calibration' automatically mapped!
```

### Model Function Contract

Model functions must follow this signature:

```python
def model_name(data: dict, param1=default1, ..., paramN=defaultN) -> dict:
    """
    Execute complete ML pipeline: train, validate, test, compute metrics.

    Args:
        data: Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test
        param1: Model-specific parameter with default value
        ...
        paramN: Additional model parameters with default values

    Returns:
        dict: Results dictionary containing:
            - Scalar metrics (logged to experiment_log)
            - '_preds' (np.ndarray): Test set predictions
            - 'models' (optional): Trained model objects
            - 'extras' (optional): Additional artifacts
    """
    # Train model on train set
    # Validate on val set
    # Predict on test set
    # Compute metrics

    return {
        'accuracy': ...,
        'precision': ...,
        # ... other metrics
        '_preds': predictions_array,
    }
```

**Requirements:**

1. Accept `data` dict as first argument
2. Provide default values for all parameters
3. Return dict with scalar metrics and `_preds`
4. Use appropriate metrics helper (`binary_metrics`, `multiclass_metrics`, `continuous_metrics`)

## Complete Reference Example

Here's a comprehensive manifest-based SFM showing most available features:

```python
from loop.manifest import Manifest
from loop.historical_data import HistoricalData
from loop.tests.utils.get_data import get_klines_data_fast
from loop.indicators import roc, ppo, wilder_rsi, atr, rolling_volatility
from loop.features import (
    ichimoku_cloud, volume_regime,
    close_position, trend_strength, gap_high,
    price_range_position, range_pct, quantile_flag,
    compute_quantile_cutoff
)
from loop.utils import shift_column
from loop.transforms.linear_transform import LinearTransform
from loop.sfm.model import ridge_binary

def params():
    return {
        # Target
        'roc_period': [1, 4, 8],
        'q': [0.32, 0.35, 0.37],
        'shift': [-1, -2, -3],

        # Indicators
        'ppo_fast': [8, 12],
        'ppo_slow': [26, 32],
        'ppo_signal': [9, 12],
        'rsi_period': [8, 14],
        'atr_period': [12, 24],
        'volatility_window': [12, 24],

        # Features
        'tenkan_period': [9, 14],
        'kijun_period': [26, 30],
        'senkou_b_period': [52, 60],
        'displacement': [26, 30],
        'lookback': [50, 100],
        'trend_fast_period': [10, 20],
        'trend_slow_period': [50, 100],
        'price_range_position_period': [50, 100],

        # Model
        'alpha': [1.0, 5.0, 10.0],
        'use_calibration': [True, False],
        'calibration_method': ['sigmoid', 'isotonic'],
        'pred_threshold': [0.45, 0.50, 0.55],
    }

def manifest():
    return (Manifest()
        # Data sources
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(method=get_klines_data_fast)

        # Split configuration
        .set_split_config(6, 2, 2)

        # Required columns validation
        .set_required_bar_columns(['datetime', 'open', 'high', 'low', 'close'])

        # Indicators
        .add_indicator(roc, period='roc_period')
        .add_indicator(ppo, fast_period='ppo_fast', slow_period='ppo_slow',
                      signal_period='ppo_signal')
        .add_indicator(wilder_rsi, period='rsi_period')
        .add_indicator(atr, period='atr_period')
        .add_indicator(rolling_volatility, column='close', window='volatility_window')

        # Features
        .add_feature(ichimoku_cloud, tenkan_period='tenkan_period',
                    kijun_period='kijun_period', senkou_b_period='senkou_b_period',
                    displacement='displacement')
        .add_feature(volume_regime, lookback='lookback')
        .add_feature(close_position)
        .add_feature(trend_strength, fast_period='trend_fast_period',
                    slow_period='trend_slow_period')
        .add_feature(gap_high)
        .add_feature(price_range_position, period='price_range_position_period')
        .add_feature(range_pct)

        # Target configuration
        .with_target('quantile_flag')
            .add_fitted_transform(quantile_flag)
                .fit_param('_cutoff', compute_quantile_cutoff,
                          col='roc_{roc_period}', q='q')
                .with_params(col='roc_{roc_period}', cutoff='_cutoff')
            .add_transform(shift_column, shift='shift', column='target_column')
            .done()

        # Scaler
        .set_scaler(LinearTransform)

        # Model
        .with_model(ridge_binary)
    )
```

## Function Contracts Reference

### Indicator/Feature Function Contract

**Signature:**

```python
def indicator_or_feature(data: pl.LazyFrame,
                         param1=default1,
                         param2=default2,
                         ...) -> pl.LazyFrame:
    """
    Indicator or feature computation.

    Args:
        data: Input LazyFrame with market data
        param1: Parameter with default value
        param2: Another parameter with default value

    Returns:
        LazyFrame with new columns added
    """
    return data.with_columns([
        # ... computations using lazy evaluation
    ])
```

**Requirements:**

1. **Input type**: `pl.LazyFrame`
2. **Output type**: `pl.LazyFrame`
3. **Lazy evaluation**: Use LazyFrame operations (`.with_columns()`, `.select()`, etc.)
4. **Immutability**: Return new LazyFrame, don't modify input
5. **Column addition**: Use `.with_columns()` to add, don't remove existing columns
6. **Default parameters**: All parameters must have default values
7. **Parameter naming**: Use descriptive names that map to round_params

### Transform Function Contract

**Signature:**

```python
def transform_function(data: pl.LazyFrame,
                       param1=default1,
                       ...) -> pl.LazyFrame:
    """
    Transform function for targets or features.

    Args:
        data: Input LazyFrame
        param1: Parameter with default value

    Returns:
        Transformed LazyFrame
    """
    return data.with_columns([...])
```

**Same requirements as indicator/feature functions.**

### Fitted Parameter Computation Contract

**Signature:**

```python
def compute_fitted_param(data: pl.DataFrame,
                         param1=default1,
                         ...) -> Any:
    """
    Compute fitted parameter from training data.

    Args:
        data: Training DataFrame (not LazyFrame!)
        param1: Parameter with default value

    Returns:
        Computed parameter value (scalar, array, etc.)
    """
    # Compute and return parameter
    return computed_value
```

**Note:** Uses `pl.DataFrame` (not LazyFrame) because fitting requires materialized data.

### Model Function Contract

**Signature:**

```python
def model_function(data: dict,
                   param1=default1,
                   ...,
                   paramN=defaultN) -> dict:
    """
    Complete ML pipeline: train, validate, test, compute metrics.

    Args:
        data: Dictionary with:
            - x_train, y_train: Training data
            - x_val, y_val: Validation data
            - x_test, y_test: Test data
        param1: Model parameter with default
        ...
        paramN: Additional model parameters with defaults

    Returns:
        Dictionary with:
            - Scalar metrics (logged to experiment)
            - '_preds': np.ndarray of test predictions
            - 'models' (optional): Trained models
            - 'extras' (optional): Additional artifacts
    """
    # Train model
    # Validate
    # Predict on test
    # Compute metrics using loop.metrics helpers

    from loop.metrics import binary_metrics  # or multiclass_metrics, continuous_metrics

    return binary_metrics(
        y_true=data['y_test'],
        y_pred=predictions,
        y_pred_proba=probabilities,
        models=trained_models
    )
```

**Requirements:**

1. Accept `data` dict as first parameter
2. All other parameters must have defaults
3. Return dict from `loop.metrics` helpers
4. Include `_preds` in return dict for UEL collection

---

**For more information:**
- [Single File Model](Single-File-Model.md) - SFM structure and requirements
- [Universal Experiment Loop](Universal-Experiment-Loop.md) - Running experiments
- Code examples in `loop/sfm/` directory
