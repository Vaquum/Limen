# Single File Decoder

A Single File Decoder (SFD) is the unit of experiment definition in Limen. It is a Python module that packages the parameter space together with either a declarative manifest or fully custom preparation and model functions.

When you pass an SFD into `UniversalExperimentLoop`, Limen knows how to turn that module into an actual parameter sweep.

## Choose The SFD Style

Limen supports two SFD styles.

| Style | Best for | Required functions | Data handling |
|---|---|---|---|
| manifest-driven | most Limen experiments, reproducible shared research, built-in workflows | `params()`, `manifest()` | data can be fetched automatically from the manifest |
| custom functions | non-standard prep logic, external libraries, experimental flows | `params()`, `prep()`, `model()` | you pass `data=` explicitly to `UniversalExperimentLoop` |

In practice, the manifest-driven path should be your default. Reach for the custom path only when the declarative pipeline is too restrictive for the job.

## What Every SFD Must Expose

Every SFD must expose `params()`.

```python
def params():
    return {
        'shift': [-1, -2, -3],
        'roc_period': [4, 8, 12],
        'C': [0.1, 1.0, 5.0],
    }
```

`params()` returns the search space dictionary. Each key is a parameter name and each value must be a list, even when only one value is present.

### `params()` Rules

- the return value must be a dictionary
- every value must be a list
- the individual values are usually scalars such as ints, floats, strings, booleans, or callables
- structured values are possible when a manifest step explicitly expects them, but keep them deterministic and easy to inspect

## Manifest-Driven SFDs

Manifest-driven SFDs expose `manifest()` instead of custom `prep()` and `model()` functions.

```python
from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.experiment import MLManifest
from limen.indicators import roc
from limen.scalers import LogRegScaler
from limen.sfd.reference_architecture import logreg_binary
from limen.targets import QuantileBinaryTarget

def params():
    return {
        'roc_period': [4, 8, 12],
        'q': [0.35, 0.40, 0.45],
        'shift': [-1, -2, -3],
        'C': [0.1, 1.0, 5.0],
        'class_weight': [0.45, 0.65, 0.85],
    }

def manifest() -> Manifest:
    return (
        MLManifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'},
        )
        .set_test_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 7200, 'n_rows': 5000},
        )
        .set_split_config(8, 1, 2)
        .add_indicator(roc, period='roc_period')
        .with_target_label(
            'quantile_flag',
            QuantileBinaryTarget,
            fit_params={'source_column': 'roc_{roc_period}', 'quantile': 'q'},
            transform_params={'shift': 'shift'},
        )
        .set_scaler(LogRegScaler)
        .with_reference_architecture(logreg_binary)
    )
```

This style is how Limen's foundational SFDs are built.

### What You Get From The Manifest Path

- declarative data fetching
- split-first prep with train-only fitting
- automatic prep/model wiring inside `UniversalExperimentLoop`
- better reproducibility and clearer collaboration surface

### Runtime Rules For Manifest-Driven SFDs

- `UniversalExperimentLoop.run(..., prep_each_round=True)` is required
- you cannot override `prep` or `model` in `run()`
- if you do not pass `data=`, Limen fetches data from the manifest

## Custom SFDs

Custom SFDs expose `prep()` and `model()` directly. Use this path when you need full control over data preparation or do not want to express the pipeline through a manifest.

```python
import polars as pl

from limen.data.utils import split_data_to_prep_output, split_sequential
from limen.sfd.reference_architecture import logreg_binary as base_model

def params():
    return {
        'shift': [-1, -2, -3],
        'C': [0.1, 1.0, 5.0],
        'class_weight': [0.45, 0.65, 0.85],
    }

def prep(data: pl.DataFrame, round_params: dict):
    all_datetimes = data['datetime'].to_list()

    prepared = data.with_columns([
        pl.col('close').pct_change().alias('return_1'),
        ((pl.col('close').shift(round_params['shift']) > pl.col('close')).cast(pl.Int8)).alias('target'),
    ]).drop_nulls()

    splits = split_sequential(prepared, (8, 1, 2))
    return split_data_to_prep_output(splits, prepared.columns, all_datetimes)

def model(data: dict, round_params: dict):
    results = base_model(
        data,
        C=round_params['C'],
        class_weight=round_params['class_weight'],
    )
    return results
```

In this style you pass the input dataframe explicitly:

```python
import limen

uel = limen.UniversalExperimentLoop(data=data, sfd=my_custom_sfd)
```

## Function Contracts

### `prep(data, round_params)`

Custom `prep()` receives the experiment dataframe and the round-specific parameters. It must return a `data_dict` that the model function can consume.

Good defaults:

- keep `datetime` in the dataframe until just before `split_data_to_prep_output()`
- capture `all_datetimes = data['datetime'].to_list()` before dropping rows if you want correct alignment metadata
- make the function deterministic with respect to `round_params`

### `model(data, round_params)`

Custom `model()` receives the prepared `data_dict` and the round parameters. It must return a results dictionary, typically produced by one of:

- `limen.metrics.binary_metrics`
- `limen.metrics.multiclass_metrics`
- `limen.metrics.continuous_metrics`

If you want predictions to be available through `uel.preds` and `Log`, include:

```python
round_results['_preds'] = preds
```

If your prep stage fits an object that should be preserved, put it into the data dict under:

```python
data_dict['_scaler'] = fitted_scaler
```

## Foundational Vs Custom

Limen ships foundational SFDs under `limen.sfd.foundational_sfd`. These are reference implementations that show the preferred style for shared Limen workflows.

Custom SFDs are your own experiment modules. They can use the same manifest system, or they can take the custom `prep()` and `model()` path when the workflow demands it.

## Foundational SFDs Versus Reference Architecture

There is one more design split that matters in practice:

| Layer | Owns |
|---|---|
| foundational SFD | `params()` plus the packaged manifest |
| reference architecture | the class-based model contract and the function wrapper used by the manifest |

For example:

- `limen.sfd.foundational_sfd.logreg_binary` packages the experiment
- `limen.sfd.reference_architecture.logreg_binary` owns the model implementation

This is why [Trainer](Trainer.md) can promote a finished experiment round back into a trained `ReferenceModel`.

On a live local smoke pass in this repo:

- `logreg_binary`, `random_binary`, and `xgboost_regressor` all ran
- `tabpfn_binary` was unavailable because `tabpfn` was not installed

Use [Built-In SFDs](Built-In-SFDs.md) for the current shipped catalog and [Reference Architecture](Reference-Architecture.md) for the model layer underneath it.

## Read Next

- Continue to [Built-In SFDs](Built-In-SFDs.md) for the shipped foundational decoder catalog.
- Continue to [Experiment Manifest](Experiment-Manifest.md) for the full declarative pipeline used by most SFDs.
- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) to run an SFD.
- Continue to [Reference Architecture](Reference-Architecture.md) if you are authoring model implementations rather than only packaged decoders.
- Use [Indicators](Indicators.md), [Features](Features.md), [Transforms](Transforms.md), and [Scalers](Scalers.md) as the reference layer while authoring new SFDs.
