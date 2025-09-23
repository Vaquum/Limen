# Experiment Manifest

## `loop.manifest`

### `Manifest`

Defines manifest for Loop experiments with Universal Split-First architecture. The manifest provides a declarative configuration system for experiment pipelines that enforces data leakage prevention through split-first processing.

#### Core Principle: Universal Split-First

Raw data is split into train/validation/test sets first, then each split undergoes bar formation and feature engineering independently. This ensures no data leakage between splits and maintains reproducible results.

#### Method Chaining API

The manifest uses a fluent interface for configuration:

```python
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

## Integration with SFM

The manifest integrates with Single-File Models through the `prep` function:

```python
def prep(data, round_params, manifest):
    return manifest.prepare_data(data, round_params)
```

## Integration with UEL

Execute experiments with manifest support:

```python
uel = loop.UniversalExperimentLoop(data=data, single_file_model=sfm)
uel.run(experiment_name='experiment', manifest=manifest(), n_permutations=1000)
```

## Example: Complete SFM with Manifest

```python
from loop.manifest import Manifest
from loop.features import quantile_flag, compute_quantile_cutoff
from loop.indicators import roc

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
    )

def params():
    return {
        'bar_type': ['base', 'time', 'volume'],
        'roc_period': [1, 4, 12],
        'q': [0.35, 0.41, 0.47],
        'shift': [-1, -2, -3],
    }

def prep(data, round_params, manifest):
    return manifest.prepare_data(data, round_params)

def model(data, round_params):
    # Model implementation
    pass
```