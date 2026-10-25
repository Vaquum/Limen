# Combine compatible Sensors

A Cohort combines selected Sensors under an explicit aggregation rule. The [Python project](examples/python-research/README.md) retains its evaluated models, declares their common next-bar return meaning and averages their forecasts. The [Compiled project](examples/compiled-research/README.md) reconstructs recorded fold members and binds their complete artifact identities. Neither a common output name nor a binding hash establishes that the selected research is sound.

## Python cohorts

<!-- example: cohort-python-selected-fits source=docs/examples/python-research/research.py region=experiment scenario=selection interface=python role=executed -->
```python
def run_research(bars, experiment_name):
    research = bars[:1440]
    uel, retained = run_trials(research, experiment_name)
    selected = choose_parameters(uel)
    sensors = selected_sensors(bars, selected, retained)
    cohort = Cohort.from_python(sensors)
    return cohort, uel, selected, retained
```

<!-- reference: cohort-python-api source=limen/cohort/cohort.py region=signature:Cohort.from_python role=signature -->
```python
@staticmethod
def from_python(sensors: Sequence[Sensor], *, aggregate: Aggregate='mean', prepare: CohortPrepare[HistoryT] | None=None, map_output: CohortMap | None=None, mapped_output_id: str | None=None, state_id: str | None=None, selection_cutoff: datetime | None=None) -> '_PythonCohort[HistoryT]':
    ...
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

## Compiled Research through the CLI

Declare validation ranking, selected count and prediction in YAML. `limen assess` reconstructs the selected Sensors, binds their Cohort and writes `assessment.json` and `predictions.parquet`. The complete Compiled project performs this handoff with the CLI:

<!-- example: cohort-compiled-assessment source=docs/examples/compiled-research/run.sh region=selection scenario=selection interface=compiled role=executed -->
```bash
limen validate compiled-research/selection.yaml
limen profile compiled-research/selection.yaml
limen run --dry-run compiled-research/selection.yaml
limen run --no-progress-bar compiled-research/selection.yaml
limen assess compiled-research/selection.yaml --results results/dev/compiled-cli
```

## Optional artifact-backed component API

The construction, selectors and binding APIs below are for Python integration with native artifacts. They are outside the YAML/CLI authoring track. The current YAML `assessment` supports validation metric, direction, selected count, fold selection and prediction source; it does not expose arbitrary multi-source selectors or custom aggregation callbacks. New runs write compressed round records; existing JSONL remains readable. See [round storage](Universal-Experiment-Loop.md#round-storage-format).

## What this page covers

- what a Cohort is and where it sits in the Limen workflow
- construction and validation rules
- selector contract and built-in selectors
- aggregation behavior (probability-weighted vs fallback majority vote)
- output contracts for `predict(raw_klines)` and `predict_all(raw_klines)`
- a real end-to-end example from experiment artifacts

## Prerequisites

Artifact-backed construction requires:

1. completed ordinary experiment result directories
2. valid experiment artifacts (`metadata.json`, `round_data.jsonl.gz`)
3. selected permutation IDs, qualified by directory when using several sources
4. members reconstructed separately with each source's `Trainer.train(permutation_ids)`

Read these pages first:

- [Command-Line Interface](Command-Line-Interface.md)
- [Trainer](Trainer.md)

## Native integration sequence

A component integrator can construct and inspect the native objects directly:

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

<!-- reference: cohort-compiled-constructor source=limen/cohort/cohort.py region=signature:Cohort.__init__ role=signature -->
```python
def __init__(self, *, experiment_id: str | None=None, experiment_log_path: str | None=None, permutation_ids: list[str] | None=None, selector: str | Selector | None=None, selector_params: Any=None, rounds: list[tuple[str | Path, str] | tuple[str | Path, str, int]] | None=None) -> None:
    ...
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

| Selected member | Selection tuple |
|---|---|
| Ordinary recorded round | `(directory, permutation_id)` |
| Qualified recorded fold | `(directory, permutation_id, fold)` |

Reconstruct ordinary selections through the owning directory's `Trainer`; use its explicit `fold=` argument for qualified folds. Bind the complete resulting Sensor set. The recorded fold project below executes this construction rather than assuming models can be recovered from an ID alone.

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

| Identity field | Value |
|---|---|
| `aggregation_mode` | Bound Cohort aggregation mode |
| `architecture_id` | Shared or canonical mixed binary architecture identifier |
| `manifest_id` | Sole distinct manifest identifier |
| `permutation_ids` | Sorted round IDs from selected member pairs |

Several distinct manifests in ordinary-only selections use:

| Identity field | Value |
|---|---|
| `aggregation_mode` | Bound Cohort aggregation mode |
| `architecture_id` | Shared or canonical mixed binary architecture identifier |
| `rounds` | Sorted member pairs, encoded as JSON lists |

Paths and selection/binding order do not enter the hash. It identifies selected
artifact lineage, not distinct learned models or code/dependency semantics.
`cohort_id` is a cohort binding identity, not a code-signing attestation.

### Compatibility

The `rounds` form requires the same configured source method
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

Compiled Research declares zero-based fold indices in `assessment.folds`. `limen assess` reconstructs the selected fold Sensors and binds their Cohort:

<!-- example: cohort-compiled-recorded-folds source=docs/examples/compiled-research/run.sh region=walk_forward scenario=walk_forward interface=compiled role=executed -->
```bash
limen run --no-progress-bar compiled-research/walk_forward.yaml
limen assess compiled-research/walk_forward.yaml --results results/dev/compiled-walk_forward
limen run --no-progress-bar compiled-research/acceptance.yaml
```

For native API integration, three-element selections identify folds; ordinary selections keep two elements. Selection and binding themselves perform no fitting or source fetching; Trainer performs the checked refit used by the CLI. Missing files raise `FileNotFoundError`; malformed JSON, versioned witnesses or disagreeing metadata/YAML declarations raise `ReconstructionError` naming the source. Invalid selections and incompatible/bound members raise `ValueError`. Folds require nonnegative literal integers: booleans, floats, nulls and out-of-range indices reject. An unqualified walk-forward round rejects with `supply fold`; an ordinary source cannot take a fold index.

Fold-containing `member_ids` retain caller order and mix `(manifest_id, permutation_id, fold, provenance_digest)` with ordinary `(manifest_id, permutation_id)` pairs. The digest is the Sensor's raw 64-character hexadecimal selected-artifact digest. Same trial, distinct folds are distinct members. Duplicate keys reject across copies and aliases; different evidence for the same trial/fold raises `Cohort evidence conflict`. Binding checks every member key, architecture and native binary declaration before matching fold count and exact selected-artifact digest. A wrong digest cannot bind by matching the trial/fold key alone. Transparent attribute-forwarding wrappers may bind; hiding fold identity cannot convert a fold Sensor into an ordinary member.

All members must share the configured source method and parameters except date bounds, and the effective bar formation. Different architecture functions may compose when their fitted models declare compatible binary outputs. Fold members additionally share original campaign source digest, row count, timestamp extent, interval and split declaration; serial/parallel recording origin alone does not create incompatibility. Ordinary members need no invented historical fingerprint. Each Sensor retains its own preprocessing and guards, and live output clocks must align.

Every member's fitted native model must declare immutable `inference_output = ('binary', bool_probabilities)` and binary prediction mode. Continuous, signed and undeclared meanings reject. If every member promises probabilities, aggregation is `probability_weighted`; otherwise it is `majority_vote`. Both retain the existing binary threshold 0.5. Fold-containing cohorts require this declaration. Ordinary binding also uses a unanimous native binary declaration when present; otherwise it retains the construction-time architecture heuristic. Rule Sensors therefore use majority voting without fabricated probabilities. Rebinding recalculates the mode before computing cohort identity. An aggregate remains inference-only and carries no execution contract.

Before successful binding, `cohort_id`, `supports_probabilities` and `aggregation_mode` are `None`; after binding the latter two reflect the actual models. `permutation_ids` remains `None`, because `member_ids` carries the fold identities. A single resolved directory supplies `experiment_dir`, `metadata` and `available_permutation_ids`; multiple directories set those attributes to `None`. `experiment_id` is `None`; `manifest_id` is the sole distinct known ID or `None`. `source_metadata` maps every resolved directory to its metadata.

Fold-containing identity uses this payload with Python's default JSON separators, `json.dumps(payload, sort_keys=True)`, and the existing `sha256:` prefix:

| Identity field | Value |
|---|---|
| `architecture_id` | Shared or canonical mixed binary architecture identifier |
| `aggregation_mode` | Mode established by bound native model declarations |
| `members` | Sorted tagged ordinary/fold identity rows |

A fold row is `['fold', manifest_id, permutation_id, fold, provenance_digest]`; an ordinary row is `['ordinary', manifest_id, permutation_id]`. Sort by manifest ID, round ID, kind, fold (or `-1`) and digest (or the empty string).

Paths, copies and selection/binding order do not change identity; changed selected evidence does. Ordinary-only IDs keep the earlier payloads byte-for-byte. The hash records artifact lineage, not model equivalence or code attestation.

Every fold blocks timestamps earlier than its campaign selection cutoff, including its own training history, intervening folds, gaps and warm-up. The cohort remains blocked if any member is inside its guarded history, including when another member fails. Formed-bar failures preserve the established formed decision clock. Consequently, mixed ordinary/fold cohorts can have no usable historical predictions. See [Trainer](Trainer.md#recorded-walk-forward-folds) for original-source and reconstruction requirements. Python fold composition and frozen-holdout fold assessment remain unavailable.

## Selector contract

Selectors choose members from one experiment before training or inference. Their context and bare-ID return contract are unchanged; they cannot be combined with `rounds`.

The public `all` selector shows the current callable boundary. The canonical projects use SHA-256 string IDs; the selector API also preserves normalized legacy integer IDs:

<!-- reference: cohort-selector-contract source=limen/cohort/sfc/all.py region=signature:select role=signature -->
```python
def select(context: dict[str, Any]) -> list[int | str]:
    ...
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

The selector must return permutation IDs only, using the supported string/integer forms. Cohort still owns ID validation,
architecture consistency, member binding, and prediction aggregation.

### Single-file Cohort selectors

<!-- reference: cohort-selector-registry source=limen/cohort/sfc/__init__.py region=definition:BUILTIN_SELECTORS role=definition -->
```python
BUILTIN_SELECTORS: dict[str, Selector] = {'all': select_all, 'top_n': select_top_n, 'backtest_pareto': select_backtest_pareto, 'diverse_metrics': select_diverse_metrics}
```

Omitting `permutation_ids` and `selector` uses `all`, preserving the earlier default behavior.

Built-in strategy files:

- `limen.cohort.sfc.all`
- `limen.cohort.sfc.top_n`
- `limen.cohort.sfc.backtest_pareto`
- `limen.cohort.sfc.diverse_metrics`

Supply a built-in name to `selector`, with its keyword parameters under `selector_params`. The source-bound signatures below document the current controls:

<!-- reference: cohort-selector-top-n source=limen/cohort/sfc/top_n.py region=signature:select role=signature -->
```python
def select(context: dict[str, Any], *, column: str, n: object, ascending: bool=False) -> list[int | str]:
    ...
```

`top_n` ranks `results.csv` by one numeric column.

A callable may be supplied directly too:

The same public `select` function can be supplied as the `selector` callable. Use `selector_params` for its keyword arguments; no new selector language is involved.

<!-- reference: cohort-selector-pareto source=limen/cohort/sfc/backtest_pareto.py region=signature:select role=signature -->
```python
def select(context: dict[str, Any], *, target_count: object=20, min_signals: object=1, metric_cols: list[str] | None=None) -> list[int | str]:
    ...
```

`backtest_pareto` selects a trading-metric Pareto front from `results.csv`,
using backtest return/risk columns and a signal-count guard. Rows with missing,
infinite, or otherwise non-finite selector metrics are excluded before the
Pareto front is computed.

<!-- reference: cohort-selector-diverse source=limen/cohort/sfc/diverse_metrics.py region=signature:select role=signature -->
```python
def select(context: dict[str, Any], *, target_count: object=20, metric_cols: list[str] | None=None, iqr_multiplier: float=3.0, n_components: object=None, n_clusters: object=8, random_state: int=42) -> list[int | str]:
    ...
```

`diverse_metrics` uses metric-space PCA/KMeans medoids to keep a diverse set of
rounds. It prefers backtest metrics and falls back to confusion metrics when
backtest columns are unavailable.

### Architecture consistency rule

Mixed architecture selections remain unbound until every fitted member declares immutable `inference_output = ('binary', bool_probabilities)` and binary prediction mode. Each member must match its own selected source and architecture. If every member promises probabilities, the cohort averages probabilities; otherwise it votes on binary decisions. Every actionable mixed-member decision must be numeric 0 or 1, and every promised probability must be finite and in [0, 1], including in vote mode. Existing timestamp, blocking and sensor-error precedence applies.

A mixed cohort uses a canonical `mixed-binary-v1:` architecture identifier. Its cohort identity also includes the source-qualified assignment of architectures and probability promises. Reduction follows canonical member order; selection and binding order do not change the result. An invalid rebind preserves the previous valid membership and policy. Frozen supported cohorts own and digest the same policy.

## Binding members

After construction, bind trained sensor members:

<!-- reference: cohort-bind-members source=limen/cohort/cohort.py region=signature:Cohort.set_members role=signature -->
```python
def set_members(self, members: list[Any]) -> None:
    ...
```

Each member must expose `permutation_id`. In the `rounds` form, its
`manifest_id` must also match the selected source-qualified key. Binding covers
every selected member exactly once and checks architecture against that
member's own source metadata. Wrong-source, duplicate or incomplete binding
rejects; signed target-exposure Sensors remain unsupported. Only complete
successful binding populates `cohort.cohort_id`.

## Aggregation modes

Homogeneous ordinary cohorts select their initial aggregation mode from architecture capability hints. Mixed cohorts select their mode from every fitted member’s binary declaration at binding.
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

<!-- reference: cohort-predict-api source=limen/cohort/cohort.py region=signature:Cohort.predict role=signature -->
```python
def predict(self, raw_klines: pl.DataFrame) -> BarPrediction:
    ...
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

<!-- reference: cohort-predict-all-api source=limen/cohort/cohort.py region=signature:Cohort.predict_all role=signature -->
```python
def predict_all(self, raw_klines: pl.DataFrame) -> list[BarPrediction]:
    ...
```

### `__call__(raw_klines)`

Alias of `predict_all(raw_klines)`. Returns `list[BarPrediction]`, one entry per bar. Replay loops use this form for every bar in a window.

## Recorded end-to-end example

Use each project's entry point, matching wheel and recorded input bundle. Their model source, preparation helpers and inputs travel together. A chapter excerpt is not a standalone replacement for that project.

The Python function and Compiled CLI commands above are the executable examples. The Python selection proof checks retained object identity, mean aggregation and every causal inference prefix without refitting. The Compiled walk-forward proof checks both reconstructed fold identities, fitting/selection cutoffs, recorded fold membership and objective metrics. The CLI also writes batch fold-Cohort predictions from the declared inference source. This does not establish future performance or a multi-directory selection. Those extension contracts are described by the source-bound API and identity rules above.

An explicit cross-directory selected list remains caller-owned; changing a search seed does not establish distinct fitted models or non-overlapping parameter rounds. Selectors may still choose IDs independently
within each directory before assembling that list. For a parent/narrowed
experiment line, use the same per-source reconstruction and binding workflow;
source/bar compatibility and every member's guards still apply.

## Failure cases and caveats

- missing experiment artifacts (`metadata.json` / `round_data.jsonl.gz`) raise `FileNotFoundError`
- unresolvable or ambiguous experiment ID resolution raises `ValueError`
- no bound members at inference raises `RuntimeError`
- healthy member lists of different lengths or participating timestamp mismatches raise `ValueError`
- when a member returns `sensor-error` for a bar, it is excluded from aggregation for that bar; only when all members return `sensor-error` does the cohort return `reason='sensor-error'`
- homogeneous ordinary cohorts retain architecture capability hints; mixed and fold-containing cohorts require native binary model declarations
- selectors that depend on `results.csv` raise if the required columns are missing

## Read next

- [Trainer](Trainer.md) for reconstruction and promotion flows
- [Reference Architecture](Reference-Architecture.md) for model output conventions

See [Reserve and assess final observations](Final-Holdout.md) for project research exclusion, frozen fitted-selection assessment, retained scenario evidence and the limits of local exposure history.

Fold-bearing compiled Sensors require [qualified fold selection](#recorded-walk-forward-members). Legacy `set_members`, ordinary round binding and `Cohort.from_python` reject them so a two-part trial identity cannot discard fold provenance or its campaign cutoff.

## Exercise Python composition

The calibration variant retains four evaluated classifier Sensors. Its composition case uses their real outputs for mean, binary vote and a caller-defined signed-direction median. The mapper changes numeric values and declares their new meaning. A shared preparation callback supplies each member’s history. Later selection masks an additional decision; it cannot restore an excluded member row. Empty history and incompatible membership/alignment have explicit outcomes. Inference must leave every fitted object unchanged.

<!-- example: cohort-python-composition source=docs/examples/python-research/inference_cases.py region=composition scenario=calibration interface=python role=executed -->
```python
def compose_selected(sensors, bars: pl.DataFrame):
    def record(row):
        value = asdict(row)
        value['datetime'] = row.datetime.isoformat() if row.datetime is not None else None
        return value

    history = bars[1428:1920]
    before = [pickle.dumps(sensor.fitted_state) for sensor in sensors]
    members = [sensor.predict_all(history) for sensor in sensors]
    mean = Cohort.from_python(sensors, aggregate='mean', state_id='calibrated-mean')
    vote = Cohort.from_python(sensors, aggregate='vote', state_id='calibrated-vote')
    def member_history(context: dict[str, pl.DataFrame], member: Sensor) -> pl.DataFrame:
        return context['bars']

    def signed_pair(member: Sensor, row: BarPrediction):
        assert row.prediction is not None  # Cohort blocks unavailable members before mapping.
        return 2 * row.prediction - 1, None

    def signed_median(rows: Sequence[BarPrediction]):
        values = []
        for row in rows:
            assert row.prediction is not None
            values.append(row.prediction)
        return float(median(values)), None

    def mismatched_history(data: pl.DataFrame, member: Sensor) -> pl.DataFrame:
        return data if member is sensors[0] else data.tail(data.height-1)

    signed = Cohort.from_python(sensors, prepare=member_history,
        map_output=signed_pair, mapped_output_id='next_bar_signed_direction',
        aggregate=signed_median, state_id='signed-median')
    later = Cohort.from_python(sensors, selection_cutoff=bars['datetime'][1441], state_id='later-selection')
    result = {'members': [[record(row) for row in tape] for tape in members],
              'mean': [record(row) for row in mean.predict_all(history)],
              'vote': [record(row) for row in vote(history)],
              'signed': [record(row) for row in signed.predict_all({'bars': history})],
              'later': [record(row) for row in later.predict_all(history)],
              'last': record(mean.predict(history)), 'empty': mean.predict_all(history.head(0))}
    if before != [pickle.dumps(sensor.fitted_state) for sensor in sensors]:
        raise ValueError('Inference changed the evaluated fitted state')
    refusals = {}
    for name, call in {'duplicate': lambda: Cohort.from_python([sensors[0], sensors[0]]),
                       'missing_mapping_identity': lambda: Cohort.from_python(sensors, map_output=lambda member, row: (row.prediction, row.probability)),
                       'unaligned_members': lambda: Cohort.from_python(sensors, prepare=mismatched_history).predict_all(history),
                       'reverse_decisions': lambda: mean.predict_all(history.reverse())}.items():
        try:
            call()
        except ValueError as error:
            refusals[name] = str(error)
        else:
            raise ValueError('An invalid composition was accepted: ' + name)
    # Results retain real UTC timestamps and declared availability, serialized by the caller.
    return {'results': result, 'refusals': refusals, 'retained_state_unchanged': True}
```
