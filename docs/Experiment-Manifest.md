# Experiment Manifest

## `loop.manifest`

### `Manifest`

Defines manifest for Loop experiments with Universal Split-First architecture. The manifest provides a declarative configuration system for experiment pipelines that enforces data leakage prevention through split-first processing.

#### Core Principle: Universal Split-First

Raw data is split into train/validation/test sets first, then each split undergoes bar formation and feature engineering independently. This ensures no data leakage between splits and maintains reproducible results.

#### Method Chaining API

The manifest uses a fluent interface for configuration:

```python
from loop.sfm.model import ridge_binary

manifest = (Manifest()
    .set_split_config(8, 1, 2)
    .set_bar_formation(adaptive_bar_formation, bar_type='time')
    .add_indicator(rsi, period=14)
    .add_feature(vwap)
    .with_target('price_direction')
        .add_fitted_transform(quantile_flag)
            .fit_param('_cutoff', compute_quantile_cutoff, col='roc', q='q')
            .with_params(col='roc', cutoff='_cutoff')
        .done()
    .set_scaler(StandardScaler)
    .with_model(ridge_binary)
)
```

### `set_split_config`

Configure train/validation/test split ratios.

#### Args

| Parameter | Type  | Description                    |
|-----------|-------|--------------------------------|
| `train`   | `int` | Training split ratio           |
| `val`     | `int` | Validation split ratio         |
| `test`    | `int` | Test split ratio               |

#### Returns

`Manifest`: Self for method chaining

### `set_bar_formation`

Configure bar formation function and parameters.

#### Args

| Parameter | Type       | Description                           |
|-----------|------------|---------------------------------------|
| `func`    | `Callable` | Bar formation function                |
| `**params`| `dict`     | Parameter mappings for bar formation  |

#### Returns

`Manifest`: Self for method chaining

#### Supported Bar Types

- `'base'`: Original data without bar formation
- `'time'`: Time-based bars (5m, 15m, 30m, 1h)
- `'volume'`: Volume-based bars with configurable thresholds
- `'liquidity'`: Liquidity-based bars with configurable thresholds

### `set_required_bar_columns`

Set required columns that must be present after bar formation.

#### Args

| Parameter | Type         | Description                    |
|-----------|--------------|--------------------------------|
| `columns` | `List[str]`  | Required column names          |

#### Returns

`Manifest`: Self for method chaining

### `add_feature`

Add feature computation to the pipeline.

#### Args

| Parameter | Type       | Description                    |
|-----------|------------|--------------------------------|
| `func`    | `Callable` | Feature computation function   |
| `**params`| `dict`     | Parameter mappings for feature |

#### Returns

`Manifest`: Self for method chaining

### `add_indicator`

Add technical indicator to the pipeline. Functionally equivalent to `add_feature` but provides semantic clarity.

#### Args

| Parameter | Type       | Description                        |
|-----------|------------|------------------------------------|
| `func`    | `Callable` | Indicator computation function     |
| `**params`| `dict`     | Parameter mappings for indicator   |

#### Returns

`Manifest`: Self for method chaining

### `with_target`

Begin target transformation configuration. Returns a `TargetBuilder` for chained target operations.

#### Args

| Parameter       | Type  | Description              |
|-----------------|-------|--------------------------|
| `target_column` | `str` | Target column name       |

#### Returns

`TargetBuilder`: Builder for target transformations

### `set_scaler`

Set the scaler/transform class for data preprocessing.

#### Args

| Parameter      | Type      | Description                          |
|----------------|-----------|--------------------------------------|
| `transform_class` | `type` | Transform class (e.g., StandardScaler) |
| `param_name`   | `str`     | Parameter name for scaler storage    |

#### Returns

`Manifest`: Self for method chaining

### `with_model`

Configure model function for training and evaluation.

#### Args

| Parameter        | Type       | Description                                  |
|------------------|------------|----------------------------------------------|
| `model_function` | `Callable` | Model function that takes (data, **params)   |

#### Returns

`Manifest`: Self for method chaining

#### Note

The model function should accept data dict and return results dict with metrics and predictions. Parameters are auto-mapped from round_params based on model function signature.

#### Example

```python
from loop.sfm.model import logreg_binary, ridge_binary

# Simple usage - parameters auto-detected
.with_model(logreg_binary)

# Parameters in round_params override model function defaults
# Parameters not in round_params use model function defaults
.with_model(ridge_binary)
```

### `prepare_data`

Execute the complete data preparation pipeline with Universal Split-First architecture.

#### Args

| Parameter      | Type           | Description                    |
|----------------|----------------|--------------------------------|
| `raw_data`     | `pl.DataFrame` | Raw input data                 |
| `round_params` | `Dict[str, Any]` | Parameters for current round |

#### Returns

`dict`: Prepared data dictionary with train/validation/test splits

#### Processing Pipeline

1. **Split Phase**: Raw data divided into train/val/test splits
2. **Bar Formation Phase**: Each split processes bars independently
3. **Feature Engineering Phase**: Features computed per split
4. **Target Transformation Phase**: Targets computed with fitted parameters
5. **Scaling Phase**: Data scaled using fitted scalers


### `run_model`

Execute model training and evaluation using configured model function with auto-parameter mapping.

#### Args

| Parameter      | Type           | Description                    |
|----------------|----------------|--------------------------------|
| `data`         | `dict`         | Prepared data dictionary       |
| `round_params` | `Dict[str, Any]` | Parameters for current round |

#### Returns

`dict`: Results including predictions, metrics, and model artifacts

#### Auto-Parameter Mapping

The `run_model` method automatically maps parameters from `round_params` to the model function signature:

1. Inspects model function signature using `inspect.signature()`
2. For each parameter in the model function:
   - If parameter exists in `round_params`, use that value
   - Otherwise, use the model function's default value
3. Passes resolved parameters to model function

This eliminates the need to manually specify parameter mappings in the manifest.

## `TargetBuilder`

Helper class for building target transformations with context.

### `add_fitted_transform`

Add a fitted transformation that learns parameters from training data.

#### Args

| Parameter | Type       | Description                    |
|-----------|------------|--------------------------------|
| `func`    | `Callable` | Transform function             |

#### Returns

`FittedTransformBuilder`: Builder for fitted parameter configuration

### `add_transform`

Add a simple transformation without parameter fitting.

#### Args

| Parameter | Type       | Description                    |
|-----------|------------|--------------------------------|
| `func`    | `Callable` | Transform function             |
| `**params`| `dict`     | Parameter mappings             |

#### Returns

`TargetBuilder`: Self for method chaining

### `done`

Complete target configuration and return to main manifest.

#### Returns

`Manifest`: Parent manifest for continued chaining

## `FittedTransformBuilder`

Helper class for building fitted transforms with parameter fitting.

### `fit_param`

Define a parameter to be fitted on training data.

#### Args

| Parameter     | Type       | Description                           |
|---------------|------------|---------------------------------------|
| `param_name`  | `str`      | Name of fitted parameter              |
| `fit_func`    | `Callable` | Function to compute parameter         |
| `**kwargs`    | `dict`     | Arguments for fit function            |

#### Returns

`FittedTransformBuilder`: Self for method chaining

### `with_params`

Set parameters for the transform function.

#### Args

| Parameter | Type   | Description              |
|-----------|--------|--------------------------|
| `**params`| `dict` | Parameter mappings       |

#### Returns

`FittedTransformBuilder`: Self for method chaining


## Model Functions

Model functions are complete ML pipeline functions stored in `loop/sfm/model/` that encapsulate training, validation, testing, calibration, and metrics computation. Each model function combines an algorithm + metrics + configuration into a reusable unit.

### Available Model Functions

- `logreg_binary`: Logistic regression with binary classification metrics
- `ridge_binary`: Ridge classifier with optional calibration and binary metrics

Import via:
```python
from loop.sfm.model import logreg_binary, ridge_binary
```

### Model Function Contract

All model functions must follow this signature:

```python
def model_name(data: dict, param1=default1, param2=default2, **kwargs) -> dict:
    """
    Execute complete ML pipeline: train on train set, validate on val set,
    predict on test set, compute metrics.

    Args:
        data (dict): Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test
        param1: Model-specific parameter with default value
        param2: Another parameter with default value
        **kwargs: Additional parameters (ignored)

    Returns:
        dict: Dictionary containing:
            - All metrics computed on test set
            - '_preds' (np.ndarray): Predictions on test set
    """
    pass
```

### Parameter Resolution

Parameters are automatically resolved from `round_params`:
- If parameter exists in `round_params`, that value is used
- Otherwise, the model function's default value is used
- This allows single source of truth for parameter defaults

## Integration with SFM

The manifest approach uses only `params()` and `manifest()` functions:

```python
from loop.sfm.model import ridge_binary

def params():

    return {
        'roc_period': [4, 8, 12],
        'alpha': [1.0, 5.0, 10.0],
        'use_calibration': [True, False]
    }

def manifest():
    
    return (Manifest()
        .set_split_config(8, 1, 2)
        .add_indicator(roc, period='roc_period')
        .with_target('quantile_flag')
            .add_fitted_transform(quantile_flag)
                .fit_param('_cutoff', compute_quantile_cutoff,
                          col='roc_{roc_period}', q=0.85)
                .with_params(col='roc_{roc_period}', cutoff='_cutoff')
            .done()
        .set_scaler(StandardScaler)
        .with_model(ridge_binary)
    )
```

## Integration with UEL

UEL automatically detects whether to use manifest or legacy mode:

```python
import loop
import loop.sfm.ridge.ridge_classifier as ridge_classifier

# Create UEL - auto-detects manifest mode
uel = loop.UniversalExperimentLoop(
    data=data,
    single_file_model=ridge_classifier
)

# Run experiment
uel.run(
    experiment_name='experiment',
    n_permutations=1000
)

# Access results (same for both modes)
print(uel.experiment_backtest_results)
```

## Example: Complete SFM with Manifest

```python
from loop.manifest import Manifest
from loop.features import quantile_flag, compute_quantile_cutoff
from loop.indicators import roc
from loop.sfm.model import ridge_binary

def manifest():
    def shift_transform(data, shift, target_column):
        return data.with_columns(
            pl.col(target_column).shift(shift).alias(target_column)
        )

    return (Manifest()
        .set_split_config(8, 1, 2)
        .set_bar_formation(adaptive_bar_formation, bar_type='bar_type')
        .add_indicator(roc, period='roc_period')
        .with_target('quantile_flag')
            .add_fitted_transform(quantile_flag)
                .fit_param('_cutoff', compute_quantile_cutoff, col='roc_{roc_period}', q='q')
                .with_params(col='roc_{roc_period}', cutoff='_cutoff')
            .add_transform(shift_transform, shift='shift', target_column='target_column')
            .done()
        .set_scaler(LogRegTransform)
        .with_model(ridge_binary)
    )

def params():
    return {
        'bar_type': ['base', 'time', 'volume'],
        'roc_period': [1, 4, 12],
        'q': [0.35, 0.41, 0.47],
        'shift': [-1, -2, -3],
        'alpha': [1.0, 5.0, 10.0],
        'use_calibration': [True, False],
        'calibration_method': ['sigmoid', 'isotonic'],
        'calibration_cv': [3, 5],
        'pred_threshold': [0.5, 0.55, 0.6]
    }
```