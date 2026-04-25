# Cohort

`Cohort` is Limen’s inference-time ensemble surface for combining multiple trained decoders reconstructed from one completed experiment.

Use it when you want to move from single-round predictions to a controlled multi-member prediction surface while staying inside Limen artefacts (`metadata.json`, `round_data.jsonl`, Trainer-reconstructed sensors).

## What This Page Covers

- what a Cohort is and where it sits in the Limen workflow
- construction and validation rules
- aggregation behavior (probability-weighted vs fallback majority vote)
- output contracts for `predict(...)` and `__call__(...)`
- a real end-to-end example from experiment artefacts

## Prerequisites

Before constructing a Cohort, you need:

1. a completed UEL experiment directory
2. valid experiment artefacts (`metadata.json`, `round_data.jsonl`)
3. selected permutation IDs to include
4. trained members (typically from `Trainer.train(...)`)

If you are new to this flow, read:

- [Universal Experiment Loop](Universal-Experiment-Loop.md)
- [Trainer](Trainer.md)
- [Regime Diversified Opinion Pools](Regime-Diversified-Opinion-Pools.md)

## Where Cohort Fits In The Pipeline

Typical path:

1. run experiment search with UEL
2. identify permutations to promote
3. reconstruct/train those permutations with Trainer
4. create Cohort from experiment source + permutation IDs
5. bind trained members via `set_members(...)`
6. infer with either:
   - `predict(...)` for Sensor-compatible dict output (with optional tuple variants)
   - `__call__(...)` as an alias of `predict(...)`

## Construction

```python
from limen.cohort import Cohort

cohort = Cohort(
    experiment_log_path='experiments/my_exp',
    permutation_ids=[0, 1],
)
```

### Experiment source rules

You must provide exactly one source:

- `experiment_id=...`
- or `experiment_log_path=...`

Providing none or both raises `ValueError`.

### Permutation rules

- omitted `permutation_ids` means “use all available permutations”
- provided list must be non-empty
- IDs must be unique after normalization
- unknown IDs raise `ValueError`

### Architecture consistency rule

All selected permutation IDs must resolve to the same architecture identifier. Mixed-architecture selection raises `ValueError`.

## Binding Members

After construction, bind trained decoder members:

```python
cohort.set_members(sensors)
```

Members are expected to behave like Sensor/model wrappers with a `predict(data_dict)` method returning at least `{'_preds': ...}` and optionally `{'_probs': ...}`.

## Aggregation Modes

Aggregation mode is selected at construction based on architecture capability hints.

### 1) `probability_weighted`

- expects each member to return `_probs` as P(1)
- validates probabilities are finite and in `[0, 1]`
- computes mean P(1) across members
- converts to class via strict threshold `> 0.5`
- tie at exactly `0.5` resolves to class `0`
- for a single-member cohort, preserves the member payload as-is for default
  dict output (drop-in Sensor behavior)

### 2) `majority_vote`

- uses member `_preds`
- computes mean vote across members
- applies strict threshold `> 0.5`
- exact ties resolve to class `0`

## Output Contracts

### `predict(data, ...)`

Sensor-compatible contract:

- `predict(data)` → `dict` with `'_preds'` and (if available) `'_probs'`
- `predict(data, return_probs=True)` → `(y_pred, probs)`
- `predict(data, return_meta=True)` → `(y_pred, meta)`
- `predict(data, return_probs=True, return_meta=True)` → `(y_pred, probs, meta)`

Input contract:

- canonical input is a decoder-style dict (for example `{'x_test': ...}`)
- architectures that require additional context (for example TabPFN variants)
  must be called with a dict that includes required fields such as `x_val`/`y_val`

Where:

- `probs` is a per-decoder probability matrix with shape
  `(n_samples, n_members)` in the same order as `permutation_ids`
- `meta` currently contains:
  - `permutation_ids`
  - `decoder_count`
  - `architecture_id`
  - `aggregation_mode`

Single-member note:

- in probability mode, `predict(data)` returns the member payload unchanged
  (including any extra keys beyond `_preds` / `_probs`)

If `return_probs=True` is requested in fallback mode, Cohort raises `ValueError`.

### `__call__(data_dict)`

Alias of `predict(data_dict)`, preserving Sensor-style decoder dict behavior.

## Real End-To-End Example

```python
from pathlib import Path

from limen.cohort import Cohort
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.experiment.trainer import Trainer
from limen.sfd.foundational_sfd import logreg_binary as sfd_module

experiment_dir = Path('tmp/cohort_docs_example')

# 1) run a real experiment
params = sfd_module.params()
domain = ParamDomain(params)
strategy = GridStrategy(domain)

uel = UniversalExperimentLoop(
    sfd=sfd_module,
    search_strategy=strategy,
    experiment_dir=experiment_dir,
)
uel.run(experiment_name='cohort_docs_example', n_permutations=2)

# 2) reconstruct trained members from selected permutations
trainer = Trainer(experiment_dir)
permutation_ids = [0, 1]
members = trainer.train(permutation_ids)

# 3) prepare per-member inference payloads (schemas may differ by permutation)
payloads_by_pid = {}
for member in members:
    prepared = trainer._manifest.prepare_data(trainer._data, member.round_params)
    payloads_by_pid[member.permutation_id] = {'x_test': prepared['x_test']}

# 4) build and bind cohort
cohort = Cohort(
    experiment_log_path=str(experiment_dir),
    permutation_ids=permutation_ids,
)
cohort.set_members(members)

# 5a) sensor-compatible prediction surface
pred = cohort.predict({'_by_permutation_id': payloads_by_pid})
y_pred = pred['_preds']

# 5b) optional structured returns
y_pred2, probs, meta = cohort.predict(
    {'_by_permutation_id': payloads_by_pid},
    return_probs=True,
    return_meta=True,
)

# 5c) decoder-compatible adapter
decoder_result = cohort({'_by_permutation_id': payloads_by_pid})
```

If all selected members truly share one schema, you may still pass one common
decoder payload (for example `{'x_test': ...}`), but heterogeneous cohorts
should provide per-member payloads as above.

## Failure Cases And Caveats

- missing experiment artefacts (`metadata.json` / `round_data.jsonl`) raise `FileNotFoundError`
- unresolvable or ambiguous experiment ID resolution raises `ValueError`
- no bound members at inference raises `RuntimeError`
- shape mismatch across member outputs raises `ValueError`
- member inference exceptions propagate and fail the whole call
- architecture capability detection is currently hint-based; validate behavior in your target architecture set

## Read Next

- [Trainer](Trainer.md) for reconstruction and promotion flows
- [Regime Diversified Opinion Pools](Regime-Diversified-Opinion-Pools.md) for downstream cohort-style usage
- [Reference Architecture](Reference-Architecture.md) for model output conventions
