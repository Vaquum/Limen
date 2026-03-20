# Trainer

The Trainer takes a completed experiment directory and retrains selected permutations. It reads `metadata.json` to reconstruct the SFD manifest and parameters, then uses `round_data.jsonl` to look up the parameter values for each requested permutation.

The output of training is a list of [Sensor](#sensor) instances wrapping trained `ReferenceModel` objects callable for live inference.

## Prerequisites

- The experiment must have been run with `experiment_dir` set in [UniversalExperimentLoop](Universal-Experiment-Loop.md). This ensures `metadata.json` and `round_data.jsonl` are available for the Trainer to reconstruct the pipeline.
- The SFD must be **manifest-based** (i.e., it must expose `manifest()` and `params()`). Custom-function SFDs are not supported.
- The `experiment_dir` must be trusted — the SFD module path from `metadata.json` is imported at runtime.

## 2-Pass Training

Training proceeds in two passes:

- **Pass 1 — Validation**: Re-runs `manifest.prepare_data()` and `manifest.run_model()` with the original round parameters and compares metrics against the experiment log. Raises `ReconstructionError` if metrics deviate beyond tolerance. This detects pipeline drift between experiment completion and training.
- **Pass 2 — Retraining**: Retrains on the full dataset using `split_config=(1,0,0)` (all data for training, no validation/test split). The model class is resolved from the model function's module and instantiated directly via `.train()`. The resulting trained model is wrapped in a callable Sensor.

### Tolerance Thresholds

Tolerance is determined by the model's `deterministic` class attribute:

| Model Type | Tolerance | Examples |
|------------|-----------|----------|
| Deterministic (`deterministic = True`) | Exact match (1e-6 float tolerance) | `LogRegBinary`, `XGBoostRegressor` |
| Stochastic (`deterministic = False`) | 1% relative difference | `RandomBinary`, `TabPFNBinary` |

### ReconstructionError

Raised when Pass 1 validation detects metric deviation beyond tolerance. The error message includes the permutation ID and a list of mismatched metrics with their original and new values.

```python
from limen import ReconstructionError

try:
    sensors = trainer.train(permutation_ids=[42])
except ReconstructionError as e:
    print(f"Pipeline drift detected: {e}")
```

## `Trainer`

### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_dir` | `str \| Path` | Path to completed experiment directory |
| `data` | `pl.DataFrame \| None` | Optional data override. If None, fetches from manifest data source |

### `train(permutation_ids)`

Runs 2-pass training for selected permutations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `permutation_ids` | `list[int]` | Round IDs from experiment_log to retrain |

Returns a `list[Sensor]`, one per permutation ID.

Raises `ValueError` if any permutation ID is not found in `round_data.jsonl`.
Raises `ReconstructionError` if Pass 1 metrics deviate beyond tolerance.

### Usage

```python
from limen import Trainer

trainer = Trainer(experiment_dir='path/to/experiment')
sensors = trainer.train(permutation_ids=[42, 87, 103])

for sensor in sensors:
    print(sensor.permutation_id, sensor.round_params)
    prediction = sensor.predict({'x_test': live_features})
```

## Sensor

Wraps a trained `ReferenceModel` instance for live inference. Each Sensor stores the permutation ID, round parameters, experiment metadata, and Pass 1 validation results for full traceability.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `permutation_id` | `int` | Round ID from experiment log |
| `model` | `ReferenceModel` | Trained model instance |
| `round_params` | `dict` | Parameter values used for this permutation |
| `metadata` | `dict` | Experiment metadata from metadata.json |
| `results` | `dict \| None` | Model evaluation results from Pass 1 |

### `predict(data)`

Generate predictions from feature data. Most models only require `x_test` in the data dictionary (no labels needed). `TabPFNBinary` additionally requires `x_val` and `y_val` for threshold tuning.

```python
result = sensor.predict({'x_test': features})
preds = result['_preds']
```

Sensors are also callable — `sensor(data)` is shorthand for `sensor.predict(data)`.

### Traceability

```python
# Identify which permutation and model produced this sensor
print(sensor.permutation_id)           # e.g. 42
print(sensor.round_params)             # parameter values
print(sensor.metadata['sfd_module'])   # SFD module path
print(sensor.model.__class__.__name__) # e.g. 'LogRegBinary'
print(sensor.results)                  # Pass 1 validation metrics
```
