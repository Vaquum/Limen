# Trainer

The Trainer takes a completed experiment directory and retrains selected permutations. It reads `metadata.json` to reconstruct the SFD manifest and parameters, then uses `round_data.jsonl` to look up the parameter values for each requested permutation.

The output of training is a list of [Sensor](#sensor) instances — callable wrappers around trained models ready for live inference.

## Prerequisites

The experiment must have been run with `experiment_dir` set in [UniversalExperimentLoop](Universal-Experiment-Loop.md). This ensures `metadata.json` and `round_data.jsonl` are available for the Trainer to reconstruct the pipeline.

## Pass 1 and Pass 2

Training has two passes:

- **Pass 1** (current): Validation. Re-runs `manifest.prepare_data()` and `manifest.run_model()` with the original round parameters. The returned Sensor instances contain evaluation results but no trained model object. This validates that permutations reproduce correctly.
- **Pass 2** (future): Full-data retraining. Will retrain on the full dataset (no validation/test split) using the model class directly, producing Sensor instances with callable trained models.

## `Trainer`

### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment_dir` | `str \| Path` | Path to completed experiment directory |
| `data` | `pl.DataFrame \| None` | Optional data override. If None, fetches from manifest data source |

### `train(permutation_ids)`

Runs training for selected permutations.

| Parameter | Type | Description |
|-----------|------|-------------|
| `permutation_ids` | `list[int]` | Round IDs from experiment_log to retrain |

Returns a `list[Sensor]`, one per permutation ID.

Raises `ValueError` if any permutation ID is not found in `round_data.jsonl`.

### Usage

```python
from limen import Trainer

trainer = Trainer(experiment_dir='path/to/experiment')
sensors = trainer.train(permutation_ids=[42, 87, 103])

for sensor in sensors:
    print(sensor.round_params)
    print(sensor.results)
```

## Sensor

Wraps the results of a trained permutation. In Pass 1, it stores validation results but has no trained model. In Pass 2 (future), it will wrap a trained `ReferenceModel` instance and be callable for live inference.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model` | `Any \| None` | Trained model instance (None in Pass 1) |
| `round_params` | `dict` | Parameter values used for this permutation |
| `metadata` | `dict` | Experiment metadata from metadata.json |
| `results` | `dict \| None` | Model evaluation results |

### Calling a Sensor

```python
# Pass 1 — access results directly
print(sensor.results)
print(sensor.round_params)

# Pass 2 (future) — callable with trained model
result = sensor(data_dict)
```
