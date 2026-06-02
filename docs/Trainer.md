# Trainer

`Trainer` is Limen's promotion layer for finished experiment rounds. It takes a completed artifact-rich experiment directory, reconstructs the manifest and round parameters, validates selected permutations, and retrains them into reusable `Sensor` objects.

This is the bridge between "a good row in an experiment" and "a trained model object I can carry forward."

## What Trainer Needs

Trainer works only with experiments that were run through the artifact-rich UEL path and wrote an `experiment_dir`.

At minimum, the directory must contain:

- `metadata.json`
- `round_data.jsonl`

If `results.csv` is also present, Trainer validates the retrained metrics against the original experiment log. If it is missing, Trainer skips validation and proceeds directly to creating sensors.

## Prerequisites

- the experiment must have been created with `experiment_dir=...`
- the experiment must be YAML-based (created via `limen run`)
- the SFD must be a manifest-driven ML architecture (rule-based architectures are not supported)

## Typical Workflow

Start from an existing experiment directory:

```python
import limen

trainer = limen.Trainer(
    experiment_dir='path/to/experiment',
    data=my_data,  # optional override
)

sensors = trainer.train([0, 7, 19])
sensor = sensors[0]

result = sensor.predict({'x_test': live_features})
```

In this workflow:

- `Trainer` reconstructs the manifest and round metadata
- `train([ ... ])` validates and retrains the selected rounds
- each returned `Sensor` wraps one trained `ReferenceModel`

## Why Trainer Exists

Experiment runs are usually done on train/validation/test splits. Promotion is different: once a round is selected, you usually want to retrain it on all available data before carrying it downstream.

Trainer handles that transition cleanly by:

- reconstructing the original experiment logic from `yaml_reference`
- validating that the pipeline still reproduces the logged round metrics
- wrapping the validated model in a `Sensor` ready for inference

## Training and Validation

Trainer reruns:

- `manifest.prepare_data(...)`
- `manifest.run_model(...)`

with the original round parameters and wraps the resulting model in a `Sensor`. If `results.csv` is present, it also compares the resulting metrics against the original experiment log.

If the model is deterministic, validation expects an exact match within a very small float tolerance. If the model is stochastic, Trainer uses a looser scaled tolerance.

If validation fails, Trainer raises `ReconstructionError`.

```python
from limen import ReconstructionError

try:
    sensors = trainer.train([42])
except ReconstructionError as e:
    print(e)
```

This is Limen's guard against pipeline drift.

## Deterministic Vs Stochastic Models

Trainer uses the model class's `deterministic` attribute to choose the validation tolerance.

| Model type | Validation style |
|---|---|
| `deterministic = True` | near-exact metric match |
| `deterministic = False` | scaled tolerance for expected randomness |

This is why promotion is more reliable for deterministic reference models than for intentionally stochastic ones.

## Trainer And The Reference-Architecture Contract

Trainer does not promote arbitrary model objects. It resolves exactly one `ReferenceModel` subclass from the original model module and uses that class for retraining.

That means the promotion stack depends on the [Reference Architecture](Reference-Architecture.md) contract:

- `train(data, **params)`
- `predict(data)`
- `evaluate(data, inline_metrics=True)`
- `deterministic`

On a live local `logreg_binary` promotion run in this repo:

- validation completed with no metric mismatches
- `Sensor.predict()` returned `_preds` and `_probs`
- the promoted sensor produced predictions for `884` test bars

On a live local `random_binary` promotion run in this repo, Trainer raised `ReconstructionError` because the stochastic rerun did not reproduce the original logged metrics closely enough.

This is expected behavior, not a special case in the docs.

## `Trainer(experiment_dir, data=None)`

### Arguments

| Argument | Meaning |
|---|---|
| `experiment_dir` | path to the completed experiment directory |
| `data` | optional dataframe override; if omitted, Trainer fetches data from the reconstructed manifest |

Use `data=` when you already have the exact dataframe you want Trainer to use. Otherwise Trainer falls back to `manifest.fetch_data()`.

## `train(permutation_ids)`

```python
sensors = trainer.train([0, 1, 2])
```

This method:

- verifies that the requested permutation ids exist in `round_data.jsonl`
- validates them against `results.csv` when available
- retrains them on all data
- returns `list[Sensor]`

Raises:

- `ValueError` if a permutation id is missing
- `ReconstructionError` if validation detects metric drift

## Sensor

A `Sensor` is the promoted form of a trained round.

Each sensor exposes:

- `permutation_id` — round ID from the experiment log; required for cohort binding
- `round_params` — parameter values used for this permutation

### Example

```python
sensor = sensors[0]

print(sensor.permutation_id)
print(sensor.round_params)

pred = sensor.predict(raw_klines)
```

Sensors are also callable:

```python
pred = sensor(raw_klines)
```

### What `predict()` expects

All reference models need only `x_test` for inference. Calibrated models (those trained with `use_calibration: true`) store the fitted calibrator internally during the training evaluation step; subsequent `predict()` calls reuse it without needing `x_val` or `y_val`. The caller never needs to supply validation data at inference time.

## What Trainer Reads From Disk

Trainer uses:

- `metadata.json` — reads `yaml_reference` to reconstruct the manifest; `sfd_module` may be present in older experiments but is not used by the YAML-only Trainer
- `round_data.jsonl` — loads `round_params` for each permutation
- `results.csv` — when available, validates retrained metrics against the original experiment log

## Scope Note

Trainer depends on the artifact-rich UEL path, which in turn depends on a concrete `SearchStrategy`. Limen ships built-in strategies (`GridStrategy`, `RandomStrategy`) and the `SearchStrategy` abstraction for writing your own.

So the clean mental model is:

- UEL artifact-rich runs create the promotion-ready experiment directory
- Trainer turns selected rounds from that directory into sensors

## Read Next

- Continue to [Reference Architecture](Reference-Architecture.md) for the class-based model contract that Trainer reconstructs and retrains.
- Continue to [Cohort](Cohort.md) if you want to bind selected sensors into an ensemble inference surface.
- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) if you need the run layer that produces `experiment_dir`.
