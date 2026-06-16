# Single file decoder

A Single File Decoder (SFD) is the Python experiment unit beneath Limen's YAML and CLI path. It packages a parameter space together with either a declarative manifest or fully custom preparation and model functions.

`limen run` compiles YAML into a manifest-backed SFD and passes it to `UniversalExperimentLoop`. Direct SFD authoring is the extension path for custom decoders and model code.

## Choose the SFD style

Limen supports two SFD styles.

| Style | Fit | Required functions | Data handling |
|---|---|---|---|
| manifest-driven | standard Limen experiments, reproducible shared research, built-in workflows | `params()`, `manifest()` | data can be fetched automatically from the manifest |
| custom functions | non-standard prep logic, external libraries, experimental flows | `params()`, `prep()`, `model()` | caller passes `data=` explicitly to `UniversalExperimentLoop` |

The manifest-driven path is the default. Use the custom path only when the declarative pipeline is too restrictive for the job.

## What every SFD must expose

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

### `params()` rules

- the return value must be a dictionary
- every value must be a list
- individual values are scalars such as ints, floats, strings, booleans, or callables
- structured values are possible when a manifest step explicitly expects them; keep them deterministic and inspectable

## Manifest-driven SFDs

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
            params={'kline_size': 7200, 'row_count_limit': 5000},
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

### What the manifest path provides

- declarative data fetching
- split-first prep with train-only fitting
- automatic prep/model wiring inside the CLI-backed UEL engine
- reproducible collaboration surface

### Runtime rules for manifest-driven SFDs

- `limen run` sets `prep_each_round` from YAML; direct `UniversalExperimentLoop.run(prep_each_round=True)` is required
- `prep` and `model` cannot be overridden in `run()`
- when `data=` is omitted, Limen fetches data from the manifest

## Custom SFDs

Custom SFDs expose `prep()` and `model()` directly. Use this path for full control over data preparation or workflows that do not fit the manifest pipeline.

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

This style passes the input dataframe explicitly:

```python
import limen

uel = limen.UniversalExperimentLoop(data=data, sfd=my_custom_sfd)
```

## Function Contracts

### `prep(data, round_params)`

Custom `prep()` receives the experiment dataframe and the round-specific parameters. It must return a `data_dict` that the model function can consume.

Defaults:

- keep `datetime` in the dataframe until just before `split_data_to_prep_output()`
- pass throwaway split lists and column lists to `split_data_to_prep_output()` because it drops `datetime` from the split frames and removes `datetime` from the column list
- capture `all_datetimes = data['datetime'].to_list()` before dropping rows when alignment metadata is required
- make the function deterministic with respect to `round_params`

### `model(data, round_params)`

Custom `model()` receives the prepared `data_dict` and the round parameters. It must return a results dictionary produced by one of:

- `limen.metrics.binary_metrics`
- `limen.metrics.multiclass_metrics`
- `limen.metrics.continuous_metrics`

For predictions available through `uel.preds` and `Log`, include:

```python
round_results['_preds'] = preds
```

Fitted prep-stage objects that should be preserved belong in the data dict under:

```python
data_dict['_scaler'] = fitted_scaler
```

## Foundational versus custom

Limen ships foundational SFDs under `limen.sfd.foundational_sfd`. These are reference implementations that show the preferred style for shared Limen workflows.

Custom SFDs are project experiment modules. They can use the same manifest system or the custom `prep()` and `model()` path.

## Foundational SFDs versus reference architecture

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

## Read next

- Continue to [Command Line Interface](Command-Line-Interface.md) for the normal YAML run path.
- Continue to [Experiment Manifest](Experiment-Manifest.md) for the full declarative pipeline used by standard SFDs.
- Continue to [Built-In SFDs](Built-In-SFDs.md) for the shipped foundational decoder catalog.
- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) for direct Python execution.
- Continue to [Reference Architecture](Reference-Architecture.md) for model implementation authoring beyond packaged decoders.
- Use [Indicators](Indicators.md), [Features](Features.md), [Transforms](Transforms.md), and [Scalers](Scalers.md) as the reference layer while authoring new SFDs.
