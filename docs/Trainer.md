# Trainer

`Trainer` is Limen's promotion layer for finished experiment rounds. It takes a completed artifact-rich experiment directory, reconstructs the manifest and round parameters, validates selected permutations, and retrains them into reusable `Sensor` objects.

This is the bridge between "a good row in an experiment" and "a trained model object I can carry forward."

## What Trainer Needs

Trainer works only with experiments that were run through the artifact-rich UEL path and wrote an `experiment_dir`.

At minimum, the directory must contain:

- `metadata.json`
- `round_data.jsonl`

If `results.csv` is also present, Trainer performs Pass 1 validation against the original metrics. If it is missing, Trainer skips validation and proceeds directly to retraining.

## Prerequisites

- the experiment must have been created with `experiment_dir=...`
- the SFD must be manifest-driven
- the experiment directory must be trusted

The trust warning matters because Trainer imports the SFD module path stored in `metadata.json`.

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

- reconstructing the original experiment logic
- validating that the pipeline still reproduces the logged round
- retraining on all data with `split_config=(1,0,0)`

## Two-Pass Training

### Pass 1: Validation

Trainer reruns:

- `manifest.prepare_data(...)`
- `manifest.run_model(...)`

with the original round parameters and compares the resulting metrics against the original experiment log.

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

### Pass 2: Retraining

After validation, Trainer deep-copies the manifest with:

```python
split_config=(1, 0, 0)
```

and retrains the resolved `ReferenceModel` on the full dataset.

That trained model is then wrapped in a `Sensor`.

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

- Pass 1 validation completed with `validation_mismatches == []`
- the promoted `Sensor.results` included task metrics plus `backtest_*` keys
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

Use `data=` when you already have the exact dataframe you want Trainer to use. Otherwise Trainer falls back to `manifest.fetch_data_for_env()`.

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

Each sensor stores:

- `permutation_id`
- `model`
- `round_params`
- `metadata`
- `results`

### Example

```python
sensor = sensors[0]

print(sensor.permutation_id)
print(sensor.round_params)
print(sensor.metadata['sfd_module'])

pred = sensor.predict({'x_test': live_features})
```

Sensors are also callable:

```python
pred = sensor({'x_test': live_features})
```

### What `predict()` expects

Most reference models only need:

- `x_test`

Some models may require more. The requirement comes from the underlying model class, not from the `Sensor` wrapper itself.

### What `results` contains

`Sensor.results` comes from the Pass 1 evaluation result, not from a stripped-down inference-only payload.

In a live local logreg promotion run in this repo, the stored keys included:

- `_preds`
- `accuracy`
- `auc`
- `backtest_total_return_net_pct`
- `backtest_max_drawdown_pct`
- `backtest_sharpe_per_bar`

That is why `Sensor.results` is useful for provenance and review, while `Sensor.predict()` is the smaller live inference surface.

## What Trainer Reads From Disk

Trainer uses:

- `metadata.json` to discover the SFD module and experiment metadata
- `round_data.jsonl` to load `round_params`, stored predictions, and alignment metadata
- `results.csv` when available for Pass 1 metric validation

On a live local artifact-rich run in this repo, `metadata.json` contained:

- `sfd_module`
- `limen_version`
- `created_at`

and `round_data.jsonl` contained entries with:

- `round_id`
- `round_params`
- `preds`
- `alignment`

## Scope Note

Trainer depends on the artifact-rich UEL path, which in turn depends on a concrete `SearchStrategy`. Limen exposes the `SearchStrategy` abstraction, but it does not ship one canonical production strategy in the public package.

So the clean mental model is:

- UEL artifact-rich runs create the promotion-ready experiment directory
- Trainer turns selected rounds from that directory into sensors

## Read Next

- Continue to [Reference Architecture](Reference-Architecture.md) for the class-based model contract that Trainer reconstructs and retrains.
- Continue to [Regime Diversified Opinion Pools](Regime-Diversified-Opinion-Pools.md) if you want to work with diversified pools of selected rounds rather than isolated sensors.
- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) if you need the run layer that produces `experiment_dir`.
