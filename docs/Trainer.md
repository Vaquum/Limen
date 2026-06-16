# Trainer

`Trainer` is Limen's promotion layer for finished experiment rounds. It takes a completed artifact-backed experiment directory, reconstructs the manifest and round parameters, validates selected permutations, and retrains them into reusable `Sensor` objects.

Trainer bridges a selected experiment row and a trained model object for downstream inference.

## What Trainer needs

Trainer works only with YAML-based artifact directories, normally created by `limen run`, that include the metadata needed to reconstruct the manifest.

At minimum, the directory must contain:

- `metadata.json`
- `round_data.jsonl`

Every nonblank `round_data.jsonl` line must be a valid JSON object with `round_id` and object-valued `round_params`. Trainer rejects malformed JSONL instead of skipping corrupt lines, so promotion cannot proceed from a partial artifact view.

If `results.csv` is also present, Trainer validates retrained metrics against matching numeric columns in the original experiment log. If it is missing, Trainer skips validation and proceeds directly to creating sensors.

Metric validation is intersection-based: Trainer compares numeric, non-private metrics returned by retraining only when the same key exists in `results.csv`. Parameter columns, private artifact keys, non-numeric values, and metrics absent from the original log are skipped. A requested permutation ID missing from `results.csv` remains a validation failure.

## Prerequisites

- the experiment must have a result directory created by `limen run`
- the experiment must be YAML-based and include `metadata.json["yaml_reference"]`
- the SFD must be a manifest-driven ML architecture (rule-based architectures are not supported)

## Workflow

Start from an existing experiment directory:

```python
import polars as pl
from limen.experiment.trainer import Trainer

# 1) identify permutation IDs from results.csv
results = pl.read_csv('path/to/experiment/results.csv')
top_ids = results.sort('accuracy', descending=True).head(3)['id'].to_list()

# 2) retrain selected permutations into Sensor objects
trainer = Trainer('path/to/experiment')
sensors = trainer.train(top_ids)

# 3) run inference on live klines
sensor = sensors[0]
bar_pred = sensor.predict(raw_klines)
```

In this workflow:

- `Trainer` reconstructs the manifest and round metadata
- `trainer.train(top_ids)` validates and retrains the selected rounds
- each returned `Sensor` wraps one trained `ReferenceModel`

## Why Trainer Exists

Experiment runs normally use train/validation/test splits. Promotion retrains a selected round on all available data before downstream use.

Trainer handles that transition cleanly by:

- reconstructing the original experiment logic from `yaml_reference`
- validating that the pipeline still reproduces the logged round metrics
- wrapping the validated model in a `Sensor` ready for inference

## Training and validation

Trainer reruns:

- `manifest.prepare_data(data, round_params)`
- `manifest.run_model(prepared, round_params)`

with the original round parameters and wraps the resulting model in a `Sensor`. If `results.csv` is present, it also compares the resulting metrics against matching numeric columns in the original experiment log.

Deterministic models require near-exact metric matches. Stochastic models use a scaled tolerance.

If validation fails, Trainer raises `ReconstructionError`.

```python
from limen import ReconstructionError

try:
    sensors = trainer.train(top_ids)
except ReconstructionError as e:
    print(e)
```

This is Limen's guard against pipeline drift.

## Deterministic versus stochastic models

Trainer uses the model class's `deterministic` attribute to choose the validation tolerance.

| Model type | Validation style |
|---|---|
| `deterministic = True` | near-exact metric match |
| `deterministic = False` | scaled tolerance for expected randomness |

This is why promotion is more reliable for deterministic reference models than for intentionally stochastic ones.

## Trainer and the reference-architecture contract

Trainer does not promote arbitrary model objects. It resolves exactly one `ReferenceModel` subclass from the original model module and uses that class for retraining.

That means the promotion stack depends on the [Reference Architecture](Reference-Architecture.md) contract:

- `train(data, **params)`
- `predict(data)`
- `evaluate(data, inline_metrics=True)`
- `deterministic`

## `Trainer(experiment_dir, data=None)`

### Arguments

| Argument | Meaning |
|---|---|
| `experiment_dir` | path to the completed experiment directory |
| `data` | optional dataframe override; if omitted, Trainer fetches data from the reconstructed manifest |

Use `data=` with an exact dataframe override. Otherwise Trainer falls back to `manifest.fetch_data()`.

## `train(permutation_ids)`

```python
sensors = trainer.train(top_ids)
```

`permutation_ids` is a `list[str]` of SHA-256 round IDs from `round_data.jsonl`. These are the `id` values in `results.csv` — use that file to identify which rounds to promote.

This method:

- verifies that the requested permutation IDs exist in `round_data.jsonl`
- validates them against `results.csv` when available
- retrains them on all data
- returns `list[Sensor]`

Raises:

- `ValueError` if a permutation ID is missing
- `ReconstructionError` if validation detects metric drift

## Sensor

A `Sensor` is the promoted form of a trained round.

Each sensor exposes:

- `permutation_id` — SHA-256 round ID from the experiment log; required for cohort binding
- `manifest_id` — SHA-256 content hash of the YAML manifest, carried from `metadata.json` for traceability
- `round_params` — parameter values used for this permutation

### Sensor example

```python
sensor = sensors[0]

print(sensor.permutation_id)   # 'sha256:a3f1c9'
print(sensor.manifest_id)      # 'sha256:7c2e41'
print(sensor.round_params)

bar_pred = sensor.predict(raw_klines)
```

Sensors are also callable. `__call__` aliases `predict_all` and returns `list[BarPrediction]`, one entry per bar:

```python
all_preds = sensor(raw_klines)  # list[BarPrediction]
```

### `predict(raw_klines) -> BarPrediction`

Takes a `pl.DataFrame` of raw klines with the same schema as the manifest data source. Returns a `BarPrediction` for the last bar.

```python
@dataclass
class BarPrediction:
    datetime: Any
    prediction: int | float | None
    probability: float | None
    reason: str | None
```

`reason` values:

| Value | Meaning |
|---|---|
| `None` | valid prediction; use `prediction` and `probability` |
| `'warm-up'` | not enough bars to satisfy indicator lookback |
| `'inside-training-window'` | bar falls within the train/test split window |
| `'null-features'` | mid-stream null feature value (data gap or transform anomaly) |
| `'sensor-error'` | unexpected exception; prediction is not available |

`predict()` never raises for data or model failures — all non-prediction conditions are returned as `BarPrediction` with the corresponding `reason`. The one exception is `NotImplementedError`, which is raised if the manifest has `decoder_lookback > 1` (not yet supported).

### `predict_all(raw_klines) -> list[BarPrediction]`

Returns one `BarPrediction` per bar in the post-bar-formation data. Warm-up bars and inside-window bars have `reason` set; valid bars have `prediction` and `probability` populated.

On unexpected exceptions the method returns a list of `reason='sensor-error'` entries (one per input bar) rather than raising. The one exception is `NotImplementedError`, which is raised if the manifest has `decoder_lookback > 1` (not yet supported).

### What `predict()` expects

Sensor callers pass raw klines, not `x_test`. The Sensor rebuilds the feature matrix internally from the stored manifest, fitted preprocessing params, and round params, then passes `x_test` to the trained reference model. Calibrated models (those trained with `use_calibration: true`) store the fitted calibrator internally during the training evaluation step; subsequent `predict()` calls reuse it without needing `x_val` or `y_val`. The caller never needs to supply validation data at inference time.

## What Trainer reads from disk

Trainer uses:

- `metadata.json` — reads `yaml_reference` to reconstruct the manifest; `manifest_id` is also read and passed to each Sensor
- `round_data.jsonl` — loads `round_params` for each permutation
- `results.csv` — when available, validates retrained metrics against matching numeric columns in the original experiment log

## Scope note

Trainer depends on the artifact-backed YAML run path, which uses UEL with a concrete `SearchStrategy`. Limen ships built-in strategies (`GridStrategy`, `RandomStrategy`) and the `SearchStrategy` abstraction for custom strategies.

The operating model is:

- `limen run` creates the promotion-ready experiment directory
- Trainer turns selected rounds from that directory into sensors

## Read next

- Continue to [Reference Architecture](Reference-Architecture.md) for the class-based model contract that Trainer reconstructs and retrains.
- Continue to [Cohort](Cohort.md) to bind selected sensors into an ensemble inference surface.
- Continue to [Command Line Interface](Command-Line-Interface.md) for the YAML run layer that produces result directories.
