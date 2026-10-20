# Cohort

`Cohort.from_python(selected_sensors)` combines live selected Sensors in one line.
It preserves the fitted member references and supports mean, binary vote or your
own aggregation. The artifact-backed constructor remains available below.

## Python cohorts

```python
cohort = Cohort.from_python(selected_sensors)
last = cohort.predict(completed_history)
batch = cohort.predict_all(completed_history)  # cohort(history) is equivalent
```

Each member must declare the same explicit non-None `output_id`, for example
`Sensor.from_model(fitted, selection_cutoff=cutoff, output_id='next_bar_return')`.
An ID declares compatible target/units; it does not verify them or convert scores.
See the executable [recorded workflow](Universal-Experiment-Loop.md#direct-python-run)
and [Sensor controls](Trainer.md#python-construction).

| `Cohort.from_python` argument | Default / contract |
|---|---|
| `sensors` | Required nonempty sequence of native Sensors; retained in order |
| `aggregate` | `'mean'`; `'vote'` or `(Sequence[BarPrediction]) -> (prediction, probability)` |
| `prepare` | Shared history; optional `(history, sensor) -> member_history` |
| `map_output` | None; optional `(sensor, bar) -> (prediction, probability)` |
| `mapped_output_id` | Required common mapped input meaning when a mapper is supplied; rejected without one |
| `state_id` | Generated UUID; optional nonblank live identifier |
| `selection_cutoff` | Latest Python member cutoff; supply any additional cohort-selection boundary explicitly |

Mean preserves signed/continuous magnitude and averages probability only when all
probabilities are known. One-member mean preserves its prediction/probability.
Vote requires exact numeric 0/1, returns 1 only above a 0.5 mean, resolves ties to
0 and returns probability None. Nonbinary vote inputs produce sensor-error rows.
Custom aggregation owns its final numeric meaning; the Cohort has no output-ID
property. No selector framework or binding tuple is required.

Every member must return identical strictly increasing unique UTC decision
timestamps and cardinality. There is no truncation, sorting or intersection.
All-empty batches return []; mixed empty/nonempty batches raise ValueError.
Any unavailable member blocks that row before output mapping or aggregation,
with both numeric fields None. Reason priority is inside-training-window,
sensor-error, null-features, warm-up. Selection masking intentionally dominates
values on excluded decisions. Availability is the maximum member availability
when all are known, otherwise None, on blocked and actionable rows alike.
Callbacks receive detached result copies and cannot alter trusted metadata.
Invalid aligned numeric pairs become sensor-error; callback exceptions propagate.

An explicit cutoff can only add masking, never remove a member's guard. Omission
for Python-only cohorts declares no later observations were used to select the
cohort. Compiled members require `map_output`, `mapped_output_id` and an explicit
selection cutoff because they expose no public selection boundary. Declare the
latest actual selection observation, including any compiled test interval used
for selection. The mapper must actually convert compatible units/targets; an ID
alone cannot relabel raw scores. Signed compiled Sensors retain their original
trade contracts, while the Python aggregate carries no compiled proof.
Base-bar members can share the same history directly; resampled/formed members
need `prepare` to produce matching decision bars. A compiled caught failure with
no timestamps raises an alignment ValueError naming its position and identity.

Membership is immutable: `set_members` raises TypeError. Duplicate objects,
Python state IDs or known compiled `(manifest_id, permutation_id)` pairs reject.
The native Python Cohort exposes `origin='python'`, `prediction_mode='python'`,
`state_id`, `selection_cutoff` and `members`; `manifest_id`/`cohort_id` are None.
Compiled-only `member_ids`, `aggregation_mode`, `architecture_id`, `permutation_ids`,
`available_permutation_ids`, `supports_probabilities`, `metadata`, `experiment_id`,
`experiment_dir` and `source_metadata` are absent (AttributeError).
Every result has `trade_contract_digest=None`. Live Python objects have no saved
reconstruction, Trainer replay or protected final-holdout evidence;
`freeze_holdout_selection` rejects both Python Sensors and Python Cohorts.
Each call performs batch alignment/result work plus member/callback cost; retain
required history context when trimming completed prefixes.

| Behavior | Compiled Cohort | Python Cohort |
|---|---|---|
| Numeric aggregation | Probability/majority class threshold | Mean, exact binary vote or custom function |
| Member sensor-error | Exclude failed members when others remain | Block the row; never drop a member |
| Availability | None | Maximum when all known, otherwise None |
| Exceptions | Existing catch-and-degrade policy | Callback errors propagate; unalignable compiled failures raise |
| Identity | Bound artifact lineage | Caller-owned live state, no compiled identity |

## Compiled cohorts

The remaining construction, selector, binding and aggregation sections describe
the artifact-backed path. New runs use compressed round artifacts; existing
JSONL runs remain readable. See the [round storage format](Universal-Experiment-Loop.md#round-storage-format).
`Cohort` reconstructs no models itself: select rounds and bind Sensors returned
by Trainer. Compiled `set_members` rejects Python-origin Sensors with advice to
use `Cohort.from_python`.

## What this page covers

- what a Cohort is and where it sits in the Limen workflow
- construction and validation rules
- selector contract and built-in selectors
- aggregation behavior (probability-weighted vs fallback majority vote)
- output contracts for `predict(raw_klines)` and `predict_all(raw_klines)`
- a real end-to-end example from experiment artifacts

## Prerequisites

A Cohort requires:

1. completed ordinary experiment result directories
2. valid experiment artifacts (`metadata.json`, `round_data.jsonl.gz`)
3. selected permutation IDs, qualified by directory when using several sources
4. members reconstructed separately with each source's `Trainer.train(permutation_ids)`

Read these pages first:

- [Command-Line Interface](Command-Line-Interface.md)
- [Trainer](Trainer.md)

## Where Cohort fits in the pipeline

Pipeline path:

1. run experiment search with `limen run`
2. identify permutations to promote, manually or with a selector
3. reconstruct and replay those permutations with Trainer
4. create Cohort from a single source + permutation IDs/selector, or explicit `(directory, permutation_id)` pairs
5. bind trained members via `set_members(sensors)`
6. infer with:
   - `predict(raw_klines)` → one `BarPrediction` for the last bar
   - `predict_all(raw_klines)` → one `BarPrediction` per bar
   - `__call__(raw_klines)` as an alias of `predict_all(raw_klines)` → `list[BarPrediction]`

## Construction

```python
import polars as pl
from limen.cohort import Cohort
from limen.inference import Trainer

results = pl.read_csv('experiments/my_exp/results.csv')
top_ids = results.sort('accuracy', descending=True).head(3)['id'].to_list()

cohort = Cohort(
    experiment_log_path='experiments/my_exp',
    permutation_ids=top_ids,
)
```

`permutation_ids` is a `list[str]` of SHA-256 round IDs. Use the `id` column in `results.csv` to identify which rounds to include.

### Experiment source rules

For the existing single-source form, exactly one source is required:

- `experiment_id`
- `experiment_log_path`

Providing none or both raises `ValueError`.

### Permutation rules

- omitted `permutation_ids` means "use the default `all` selector"
- provided list must be non-empty
- IDs must be unique after normalization
- unknown IDs raise `ValueError`
- `permutation_ids` and `selector` are mutually exclusive

### Explicit rounds from several experiments

```python
selected_rounds = [(directory_a, round_a), (directory_b, round_b)]
cohort = Cohort(rounds=selected_rounds)
sensors_a = Trainer(directory_a, data=recorded).train([round_a])
sensors_b = Trainer(directory_b, data=recorded).train([round_b])
cohort.set_members(sensors_a + sensors_b)
```

`rounds` must be non-empty and is mutually exclusive with `experiment_id`,
`experiment_log_path`, `permutation_ids`, `selector` and `selector_params`.
Directories are resolved before lookup. Each requested ID must exist in its
named directory, whose metadata must have a known `manifest_id`. Each source
uses its own Trainer, YAML, fitted preprocessing and replay validation;
reconstruction failures remain visible rather than dropping selected members.
Supplying the same recorded DataFrame to each Trainer avoids repeated fetching;
Cohort adds no shared training or data cache.

For ordinary rounds, the canonical member key is `(manifest_id, permutation_id)`. Duplicate keys
reject, including aliases of a directory or repeated runs of one manifest.
Directories sharing a manifest may contribute distinct rounds. The same bare
round ID under different manifests is allowed and counted once per selected
member. Explicitly selecting equivalent models gives that prediction more
weight; Cohort does not detect or deduplicate equivalent models.

### Provenance and identity

For ordinary-only selections, `member_ids` preserves the selected order of qualified pairs. `source_metadata`
maps resolved directory paths to their own metadata. The new form's existing
provenance attributes depend on its number of resolved directories:

| Attribute | One directory | Several directories |
|---|---|---|
| `experiment_dir`, `metadata`, `available_permutation_ids`, `permutation_ids` | describe that directory | `None` |
| `experiment_id` | the existing explicit-path value (`None`) | `None` |
| `manifest_id` | the source manifest ID | the sole distinct manifest ID, otherwise `None` |

Existing single-source constructor attributes keep their meanings.
`cohort_id` remains `None` until complete successful `set_members()` binding.
The hash uses `json.dumps(payload, sort_keys=True)` and SHA-256, prefixed by
`sha256:`. One distinct manifest uses the exact legacy payload, even with the
new constructor or distinct rounds spread across several directories:

```python
payload = {
    'aggregation_mode': cohort.aggregation_mode,
    'architecture_id': cohort.architecture_id,
    'manifest_id': cohort.manifest_id,
    'permutation_ids': sorted(pid for _, pid in cohort.member_ids),
}
```

Several distinct manifests in ordinary-only selections use:

```python
payload = {
    'aggregation_mode': cohort.aggregation_mode,
    'architecture_id': cohort.architecture_id,
    'rounds': [list(pair) for pair in sorted(cohort.member_ids)],
}
```

Paths and selection/binding order do not enter the hash. It identifies selected
artifact lineage, not distinct learned models or code/dependency semantics.
`cohort_id` is a cohort binding identity, not a code-signing attestation.

### Compatibility

The `rounds` form requires one architecture, the same configured source method
and identity-defining parameters (including instrument/file and base cadence),
and the same effective selected-round bar type and applied formation function
and arguments. Only `start_date_limit` and `end_date_limit` are excluded from
source comparison; different split windows, seeds and narrowed parameter
domains can remain compatible. Whole manifests are not compared.

Both constructors check actual participating, non-error timestamps at each
batch position and at the last bar returned by `predict`. Healthy output
lengths must agree. Mismatched timestamps or an actionable prediction
(`reason=None`) without a timestamp reject with the affected member identified;
bars are never truncated, intersected, interpolated or relabelled. Existing
blocked/unavailable outputs may have `datetime=None`, including empty-input
warm-up. Compatible legacy predictions and IDs remain unchanged.

Each Sensor keeps its own splits and guards. Any participating blocked member
still blocks the aggregate bar; different windows can leave no usable
predictions. Target, calibration and cost settings may differ. This does not
certify equivalent target meaning or comparable cross-run research metrics.
The aggregate is inference-only: `trade_contract_digest` and `available_at_ns`
remain `None`. Ordinary persisted rounds and [qualified binary folds](#recorded-walk-forward-members) are supported; signed target-exposure aggregation remains unavailable.

## Recorded walk-forward members

Choose explicit zero-based folds with three-element selections; ordinary selections keep two elements. Reconstruct each selected fold through its own Trainer, then bind the returned Sensors:

```python
from limen import Cohort, Trainer

# A completed campaign retained recording.fold_details: true and at least four folds.
fold_trainer = Trainer(fold_directory, data=recorded_source)
fold_members = [fold_trainer.train([fold_id], fold=k)[0] for k in (1, 3)]
ordinary_member = Trainer(ordinary_directory, data=recorded_source).train([ordinary_id])[0]
cohort = Cohort(rounds=[
    (fold_directory, fold_id, 1),
    (ordinary_directory, ordinary_id),
    (fold_directory, fold_id, 3),
])
cohort.set_members([ordinary_member, *fold_members])
latest = cohort.predict(future_history)
```

Selection and binding perform no fitting or source fetching. Missing files raise `FileNotFoundError`; malformed JSON, versioned witnesses or disagreeing metadata/YAML declarations raise `ReconstructionError` naming the source. Invalid selections and incompatible/bound members raise `ValueError`. Folds require nonnegative literal integers: booleans, floats, nulls and out-of-range indices reject. An unqualified walk-forward round rejects with `supply fold`; an ordinary source cannot take a fold index.

Fold-containing `member_ids` retain caller order and mix `(manifest_id, permutation_id, fold, provenance_digest)` with ordinary `(manifest_id, permutation_id)` pairs. The digest is the Sensor's raw 64-character hexadecimal selected-artifact digest. Same trial, distinct folds are distinct members. Duplicate keys reject across copies and aliases; different evidence for the same trial/fold raises `Cohort evidence conflict`. Binding checks every member key, architecture and native binary declaration before matching fold count and exact selected-artifact digest. A wrong digest cannot bind by matching the trial/fold key alone. Transparent attribute-forwarding wrappers may bind; hiding fold identity cannot convert a fold Sensor into an ordinary member.

All members must share the configured source method and parameters except date bounds, the actual architecture function and the effective bar formation. Fold members additionally share original campaign source digest, row count, timestamp extent, interval and split declaration; serial/parallel recording origin alone does not create incompatibility. Ordinary members need no invented historical fingerprint. Each Sensor retains its own preprocessing and guards, and live output clocks must align.

Every member's fitted native model must declare immutable `inference_output = ('binary', bool_probabilities)` and binary prediction mode. Continuous, signed and undeclared meanings reject. If every member promises probabilities, aggregation is `probability_weighted`; otherwise it is `majority_vote`. Both retain the existing binary threshold 0.5. This explicit declaration replaces architecture-name guessing only for fold-containing cohorts.

Before successful binding, `cohort_id`, `supports_probabilities` and `aggregation_mode` are `None`; after binding the latter two reflect the actual models. `permutation_ids` remains `None`, because `member_ids` carries the fold identities. A single resolved directory supplies `experiment_dir`, `metadata` and `available_permutation_ids`; multiple directories set those attributes to `None`. `experiment_id` is `None`; `manifest_id` is the sole distinct known ID or `None`. `source_metadata` maps every resolved directory to its metadata.

Fold-containing identity uses this payload with Python's default JSON separators, `json.dumps(payload, sort_keys=True)`, and the existing `sha256:` prefix:

```python
payload = {
    'architecture_id': cohort.architecture_id,
    'aggregation_mode': cohort.aggregation_mode,
    'members': sorted_tagged_member_rows,
}
# Rows: ['fold', manifest_id, permutation_id, fold, provenance_digest]
#    or ['ordinary', manifest_id, permutation_id]
# Sort by (manifest_id, permutation_id, kind, fold_or_minus_one, digest_or_empty).
```

Paths, copies and selection/binding order do not change identity; changed selected evidence does. Ordinary-only IDs keep the earlier payloads byte-for-byte. The hash records artifact lineage, not model equivalence or code attestation.

Every fold blocks timestamps earlier than its campaign selection cutoff, including its own training history, intervening folds, gaps and warm-up. The cohort remains blocked if any member is inside its guarded history, including when another member fails. Formed-bar failures preserve the established formed decision clock. Consequently, mixed ordinary/fold cohorts can have no usable historical predictions. See [Trainer](Trainer.md#recorded-walk-forward-folds) for original-source and reconstruction requirements. Python fold composition and frozen-holdout fold assessment remain unavailable.

## Selector contract

Selectors choose members from one experiment before training or inference. Their context and bare-ID return contract are unchanged; they cannot be combined with `rounds`.

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
- `results` when `results.csv` exists, loaded as a polars DataFrame

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
using backtest return/risk columns and a signal-count guard. Rows with missing,
infinite, or otherwise non-finite selector metrics are excluded before the
Pareto front is computed.

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

Each member must expose `permutation_id`. In the `rounds` form, its
`manifest_id` must also match the selected source-qualified key. Binding covers
every selected member exactly once and checks architecture against that
member's own source metadata. Wrong-source, duplicate or incomplete binding
rejects; signed target-exposure Sensors remain unsupported. Only complete
successful binding populates `cohort.cohort_id`.

## Aggregation modes

Aggregation mode is selected at construction based on architecture capability hints.
For YAML CLI runs, Cohort reads each source's `metadata.json["yaml_reference"]`
to recover `reference_architecture`; this keeps `sfd_module: yaml:<name>` bundles
compatible with Trainer-reconstructed Sensors. Weight is equal per selected
member, not per experiment, including unequal source membership.

### 1) `probability_weighted`

- expects each member to return a finite `probability` in `[0, 1]`
- computes mean probability across members
- converts to class via strict threshold `> 0.5`
- tie at exactly `0.5` resolves to class `0`

### 2) `majority_vote`

For ordinary cohorts, members whose architecture id contains `regressor` vote up at
`prediction > 0`; other member predictions use `> 0.5`.

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

## Recorded end-to-end example

Run from the repository root with Limen installed. This creates two small
ordinary sweeps over the committed recorded 15-minute fixture, selects two
members from one source and one from the other, reconstructs each against its
own artifacts, then predicts recorded January 8 bars. Temporary artifacts are
removed when the example exits.

```python
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import yaml

from limen.cli.commands.run import run_experiment
from limen.cohort import Cohort
from limen.data import HistoricalData
from limen.inference import Trainer

fixture = Path('tests/fixtures/spot_15m_20250101_20250531.parquet').resolve()
recorded = HistoricalData().get_any_file(str(fixture))
config = yaml.safe_load("""
schema_version: "1.0"
metadata:
  name: cohort_recorded
  mode: development
sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.HistoricalData.get_any_file
      params: {}
    split_dates:
      train_start: "2025-01-01"
      train_end: "2025-01-04"
      val_start: "2025-01-04"
      val_end: "2025-01-05"
      test_start: "2025-01-05"
      test_end: "2025-01-06"
    indicators:
      - func: limen.indicators.roc
        params: {period: 1, group: momentum}
    target:
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params: {source_column: roc_1, quantile: 0.5}
      transform_params: {shift: -1}
    scaler: {from_params: scaler_type}
    reference_architecture: limen.sfd.reference_architecture.logreg_binary
  params:
    scaler_type: [logreg]
    C: [0.1, 1.0, 10.0]
    random_state: [42]
    max_iter: [100]
uel:
  n_permutations: 3
  search_strategy: {type: grid, seed: 42}
  output_format: csv
""")
config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(fixture)

with TemporaryDirectory() as temporary:
    root = Path(temporary)
    selected = []
    members = []
    for seed, count in [(42, 2), (43, 1)]:
        directory = root / f'run_{seed}'
        config['metadata']['name'] = f'cohort_recorded_{seed}'
        config['uel']['search_strategy']['seed'] = seed
        config['uel']['output_path'] = str(directory)
        yaml_path = root / f'run_{seed}.yaml'
        yaml_path.write_text(yaml.safe_dump(config))
        assert run_experiment(yaml_path, progress_bar=False)
        ids = sorted(pl.read_csv(directory / 'results.csv')['id'].to_list())[:count]
        selected.extend((directory, pid) for pid in ids)
        members.extend(Trainer(directory, data=recorded).train(ids))

    cohort = Cohort(rounds=selected)
    cohort.set_members(members)
    inference = recorded.filter(pl.col('datetime').dt.date() == date(2025, 1, 8))
    last = cohort.predict(inference)
    batch = cohort.predict_all(inference)
    assert cohort(inference) == batch
    print(cohort.cohort_id)
    print(cohort.member_ids)
    print(last.datetime, last.reason, last.prediction, last.probability)
```

The selected list is caller-owned; changing a grid seed does not imply distinct
models or non-overlapping rounds. Selectors may still choose IDs independently
within each directory before assembling that list. For a parent/narrowed
experiment line, use the same per-source reconstruction and binding workflow;
source/bar compatibility and every member's guards still apply.

## Failure cases and caveats

- missing experiment artifacts (`metadata.json` / `round_data.jsonl.gz`) raise `FileNotFoundError`
- unresolvable or ambiguous experiment ID resolution raises `ValueError`
- no bound members at inference raises `RuntimeError`
- healthy member lists of different lengths or participating timestamp mismatches raise `ValueError`
- when a member returns `sensor-error` for a bar, it is excluded from aggregation for that bar; only when all members return `sensor-error` does the cohort return `reason='sensor-error'`
- ordinary-only architecture capability detection remains hint-based; fold-containing cohorts require native binary model declarations
- selectors that depend on `results.csv` raise if the required columns are missing

## Read next

- [Trainer](Trainer.md) for reconstruction and promotion flows
- [Reference Architecture](Reference-Architecture.md) for model output conventions

See [Reserve and assess final observations](Final-Holdout.md) for project research exclusion, frozen fitted-selection assessment, retained scenario evidence and the limits of local exposure history.

Fold-bearing compiled Sensors require [qualified fold selection](#recorded-walk-forward-members). Legacy `set_members`, ordinary round binding and `Cohort.from_python` reject them so a two-part trial identity cannot discard fold provenance or its campaign cutoff.
