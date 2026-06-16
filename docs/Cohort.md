# Cohort

`Cohort` is Limen's inference-time ensemble surface for combining multiple trained sensors reconstructed from one completed experiment.

`Cohort` moves single-round predictions into a controlled multi-member prediction surface while staying inside Limen artefacts (`metadata.json`, `round_data.jsonl`, Trainer-reconstructed sensors).

## What this page covers

- what a Cohort is and where it sits in the Limen workflow
- construction and validation rules
- selector contract and built-in selectors
- aggregation behavior (probability-weighted vs fallback majority vote)
- output contracts for `predict(raw_klines)` and `predict_all(raw_klines)`
- a real end-to-end example from experiment artefacts

## Prerequisites

A Cohort requires:

1. a completed YAML CLI result directory
2. valid experiment artefacts (`metadata.json`, `round_data.jsonl`)
3. selected permutation IDs, or a selector that chooses them
4. trained members from `Trainer.train(permutation_ids)`

Read these pages first:

- [Command Line Interface](Command-Line-Interface.md)
- [Trainer](Trainer.md)

## Where Cohort fits in the pipeline

Pipeline path:

1. run experiment search with `limen run`
2. identify permutations to promote, manually or with a selector
3. reconstruct/train those permutations with Trainer
4. create Cohort from experiment source + permutation IDs or selector
5. bind trained members via `set_members(sensors)`
6. infer with:
   - `predict(raw_klines)` → one `BarPrediction` for the last bar
   - `predict_all(raw_klines)` → one `BarPrediction` per bar
   - `__call__(raw_klines)` as an alias of `predict_all(raw_klines)` → `list[BarPrediction]`

## Construction

```python
import polars as pl
from limen.cohort import Cohort
from limen.experiment.trainer import Trainer

results = pl.read_csv('experiments/my_exp/results.csv')
top_ids = results.sort('accuracy', descending=True).head(3)['id'].to_list()

cohort = Cohort(
    experiment_log_path='experiments/my_exp',
    permutation_ids=top_ids,
)
```

`permutation_ids` is a `list[str]` of SHA-256 round IDs. Use the `id` column in `results.csv` to identify which rounds to include.

### Experiment source rules

Exactly one source is required:

- `experiment_id`
- `experiment_log_path`

Providing none or both raises `ValueError`.

### Permutation rules

- omitted `permutation_ids` means "use the default `all` selector"
- provided list must be non-empty
- IDs must be unique after normalization
- unknown IDs raise `ValueError`
- `permutation_ids` and `selector` are mutually exclusive

### Cohort identity fields

After construction and before `set_members()`:

- `cohort.manifest_id` — SHA-256 hash of the YAML manifest, read from `metadata.json`
- `cohort.cohort_id` — `None` until `set_members()` is called

After `set_members()`:

- `cohort.cohort_id` — `'sha256:<hex>'` derived from `manifest_id` and the sorted set of `permutation_ids`

## Selector contract

Selectors choose Cohort members before any training or inference work starts.

Minimum contract:

```python
def select(context: dict) -> list[str]:
    return sorted(context['available_permutation_ids'])
```

A selector strategy is intentionally just one Python file with one public
function named `select`. Limen's built-ins live under `limen.cohort.sfc`
("single-file cohort"). User strategies can live anywhere as long as the
callable follows the same input/output contract.

The selector receives:

- `experiment_dir`
- `metadata`
- `round_entries`
- `available_permutation_ids`
- `results` when `results.csv` exists

The selector must return permutation IDs only. Cohort still owns ID validation,
architecture consistency, member binding, and prediction aggregation.

### Single-file Cohort selectors

```python
Cohort(experiment_log_path='experiments/my_exp')
```

Uses `all`, preserving the previous omitted-`permutation_ids` behavior.

Built-in strategy files:

- `limen.cohort.sfc.all`
- `limen.cohort.sfc.top_n`
- `limen.cohort.sfc.backtest_pareto`
- `limen.cohort.sfc.diverse_metrics`

Use them by built-in name:

```python
Cohort(
    experiment_log_path='experiments/my_exp',
    selector='top_n',
    selector_params={'column': 'backtest_pnl_per_bar_bps', 'n': 5},
)
```

`top_n` ranks `results.csv` by one numeric column.

Or pass the single-file function directly:

```python
from limen.cohort.sfc.top_n import select as select_top_n

Cohort(
    experiment_log_path='experiments/my_exp',
    selector=select_top_n,
    selector_params={'column': 'backtest_pnl_per_bar_bps', 'n': 5},
)
```

```python
Cohort(
    experiment_log_path='experiments/my_exp',
    selector='backtest_pareto',
    selector_params={'target_count': 10, 'min_signals': 5},
)
```

`backtest_pareto` selects a trading-metric Pareto front from `results.csv`,
using backtest return/risk columns and a signal-count guard.

```python
Cohort(
    experiment_log_path='experiments/my_exp',
    selector='diverse_metrics',
    selector_params={'target_count': 10},
)
```

`diverse_metrics` uses metric-space PCA/KMeans medoids to keep a diverse set of
rounds. It prefers backtest metrics and falls back to confusion metrics when
backtest columns are unavailable.

### Architecture consistency rule

All selected permutation IDs must resolve to the same architecture identifier. Mixed-architecture selection raises `ValueError`.

## Binding members

After construction, bind trained sensor members:

```python
trainer = Trainer('experiments/my_exp')
sensors = trainer.train(top_ids)

cohort.set_members(sensors)
```

Each member must expose `permutation_id` for binding. `set_members()` validates that:

- all `permutation_ids` are covered exactly once
- member `permutation_id` values match the cohort's selected IDs
- all members share the same architecture

After a successful `set_members()` call, `cohort.cohort_id` is populated.

## Aggregation modes

Aggregation mode is selected at construction based on architecture capability hints.
For YAML CLI runs, Cohort reads `metadata.json["yaml_reference"]` to recover the
manifest `reference_architecture`; this keeps `sfd_module: yaml:<name>` bundles
compatible with Trainer-reconstructed sensors.

### 1) `probability_weighted`

- expects each member to return a finite `probability` in `[0, 1]`
- computes mean probability across members
- converts to class via strict threshold `> 0.5`
- tie at exactly `0.5` resolves to class `0`

### 2) `majority_vote`

- uses member `prediction` values
- computes mean vote across members
- applies strict threshold `> 0.5`
- exact ties resolve to class `0`

## Output contracts

### `predict(raw_klines) -> BarPrediction`

Takes a `pl.DataFrame` of raw klines. Returns one aggregated `BarPrediction` for the last bar.

```python
bar_pred = cohort.predict(raw_klines)

if bar_pred.reason is None:
    print(bar_pred.prediction)   # 0 or 1
    print(bar_pred.probability)  # mean probability across members
```

`reason` values mirror the per-sensor contract:

| Value | Meaning |
|---|---|
| `None` | valid aggregated prediction |
| `'warm-up'` | the highest-priority blocking reason across members |
| `'inside-training-window'` | highest-priority blocking reason |
| `'null-features'` | highest-priority blocking reason |
| `'sensor-error'` | all members returned `sensor-error`; no valid prediction available |

When one or more members return `sensor-error` but at least one member remains valid, failed members are excluded from
aggregation and a warning is logged. The cohort still produces a valid prediction from
the remaining members.

### `predict_all(raw_klines) -> list[BarPrediction]`

Returns one aggregated `BarPrediction` per bar. Each bar is aggregated independently
across members.

```python
all_preds = cohort.predict_all(raw_klines)

for bar in all_preds:
    if bar.reason is None:
        print(bar.datetime, bar.prediction, bar.probability)
```

### `__call__(raw_klines)`

Alias of `predict_all(raw_klines)`. Returns `list[BarPrediction]`, one entry per bar. Replay loops use this form for every bar in a window.

## Real end-to-end example

```python
import polars as pl
from limen.cohort import Cohort
from limen.experiment.trainer import Trainer

experiment_dir = 'experiments/my_exp'

# 1) identify top permutations from results
results = pl.read_csv(f'{experiment_dir}/results.csv')
top_ids = results.sort('accuracy', descending=True).head(3)['id'].to_list()

# 2) retrain selected permutations
trainer = Trainer(experiment_dir)
sensors = trainer.train(top_ids)

# 3) build and bind cohort
cohort = Cohort(
    experiment_log_path=experiment_dir,
    permutation_ids=top_ids,
)
cohort.set_members(sensors)

print(cohort.cohort_id.startswith('sha256:'))
print(cohort.manifest_id.startswith('sha256:'))

# 4a) predict last bar
bar_pred = cohort.predict(raw_klines)
if bar_pred.reason is None:
    print(bar_pred.prediction, bar_pred.probability)

# 4b) predict all bars
all_preds = cohort.predict_all(raw_klines)
valid = [p for p in all_preds if p.reason is None]
print(f'{len(valid)} valid bars out of {len(all_preds)}')

# 4c) callable alias, same as predict_all
all_preds2 = cohort(raw_klines)  # list[BarPrediction]
```

## Failure cases and caveats

- missing experiment artefacts (`metadata.json` / `round_data.jsonl`) raise `FileNotFoundError`
- unresolvable or ambiguous experiment ID resolution raises `ValueError`
- no bound members at inference raises `RuntimeError`
- member sensor lists of different lengths raise `ValueError` in `predict_all`
- when a member returns `sensor-error` for a bar, it is excluded from aggregation for that bar; only when all members return `sensor-error` does the cohort return `reason='sensor-error'`
- architecture capability detection is currently hint-based; validate behavior for the target architecture set
- selectors that depend on `results.csv` raise if the required columns are missing

## Read next

- [Trainer](Trainer.md) for reconstruction and promotion flows
- [Reference Architecture](Reference-Architecture.md) for model output conventions
